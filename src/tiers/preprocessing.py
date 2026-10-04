"""Capture directory validation and normalization.

Accepts the photo and video formats real devices produce and turns them into what the
pipeline reads: JPEG/PNG stills and OpenCV-readable video. Originals are never modified
or overwritten; converted copies are written next to them.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HEIC_EXTS = {".heic", ".heif"}  # iPhone
RAW_EXTS = {".dng"}  # Samsung Pro / RAW mode
TIFF_EXTS = {".tif", ".tiff"}  # some DSLRs
CONVERT_IMAGE_EXTS = HEIC_EXTS | RAW_EXTS | TIFF_EXTS | {".webp"}  # converted to JPEG
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}  # read directly by OpenCV and PIL
ALL_IMAGE_EXTS = IMAGE_EXTS | CONVERT_IMAGE_EXTS
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
LIDAR_EXTS = {".ply", ".obj"}
CONVERTED_VIDEO_SUFFIX = "_h264"  # clip.mov -> clip_h264.mp4 when OpenCV cannot read clip.mov

MIN_PHOTOS_PER_ROOM = 2
MAX_PHOTOS_PER_ROOM = 8
FFMPEG_TIMEOUT = 3600  # seconds


class CaptureValidationError(ValueError):
    """The capture directory does not match what the tier expects."""


def convert_heic_to_jpeg(input_path: str, output_path: str, quality: int = 95) -> str:
    """Convert one HEIC/HEIF file to JPEG with pillow-heif.

    EXIF orientation is applied to the pixels and the EXIF block (focal length etc.)
    is kept for downstream intrinsics estimation. Returns output_path.
    """
    import pillow_heif
    from PIL import Image, ImageOps

    pillow_heif.register_heif_opener()
    with Image.open(input_path) as img:
        img = ImageOps.exif_transpose(img)
        exif = img.getexif().tobytes()
        img = img.convert("RGB")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "JPEG", quality=quality, exif=exif)
    return output_path


def stage_upright_copy(src: str, dst: str) -> None:
    """Write `dst` as src with EXIF orientation baked into the pixels, or symlink if already upright.

    COLMAP ignores EXIF orientation, so a portrait iPhone JPEG would be reconstructed
    sideways, which breaks gravity alignment. The original file is never modified.
    """
    import os

    from PIL import Image, ImageOps

    with Image.open(src) as img:
        if img.getexif().get(274, 1) == 1:
            os.symlink(Path(src).resolve(), dst)
            return
        fixed = ImageOps.exif_transpose(img)
        fixed.save(dst, "JPEG", quality=95, exif=fixed.getexif().tobytes())


def _is_hidden(path: Path, root: Path) -> bool:
    """True for dotfiles and anything inside a dot-directory below root (.DS_Store, ._IMG_1.jpg, .git/)."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part.startswith(".") for part in parts)


def _files_with_ext(root: Path, exts: set[str]) -> list[Path]:
    """Files under root (recursive, extension match is case-insensitive), sorted, ignoring hidden files."""
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts and not _is_hidden(p, root)
    )


def _converted_video_path(video: Path) -> Path:
    return video.with_name(f"{video.stem}{CONVERTED_VIDEO_SUFFIX}.mp4")


def list_videos(capture_dir: str) -> list[Path]:
    """Video files under capture_dir, leaving out originals that have an ffmpeg-converted twin."""
    root = Path(capture_dir)
    return [v for v in _files_with_ext(root, VIDEO_EXTS) if not _converted_video_path(v).exists()
            or v.stem.endswith(CONVERTED_VIDEO_SUFFIX)]


def video_readable(path: Path) -> bool:
    """True if OpenCV can open the file and read one frame."""
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python not installed; cannot check %s", path)
        return False
    cap = cv2.VideoCapture(str(path))
    try:
        ok, _ = cap.read() if cap.isOpened() else (False, None)
        return bool(ok)
    finally:
        cap.release()


def _image_readable(path: Path) -> bool:
    """True if PIL can decode the whole file (catches truncated and corrupt images)."""
    from PIL import Image

    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception as exc:
        logger.warning("cannot read image %s: %s", path, exc)
        return False


def _to_rgb(img):
    """PIL image of any mode (16-bit, float, palette, CMYK, alpha) as 8-bit RGB."""
    import numpy as np
    from PIL import Image

    if img.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        arr = np.asarray(img).astype("float64")
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        return Image.fromarray(arr.astype("uint8")).convert("RGB")
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


def convert_image_to_jpeg(input_path: str, output_path: str, quality: int = 95) -> str:
    """Convert one WEBP/TIFF (or any PIL-readable) image to JPEG; EXIF is kept when present.

    The first page of a multi-page TIFF is used. Returns output_path.
    """
    from PIL import Image, ImageOps

    with Image.open(input_path) as img:
        img.seek(0)
        exif = b""
        try:
            exif = img.getexif().tobytes()
        except Exception:  # EXIF is a convenience; never block the pixels
            pass
        img = _to_rgb(ImageOps.exif_transpose(img))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            img.save(output_path, "JPEG", quality=quality, exif=exif)
        except Exception:
            img.save(output_path, "JPEG", quality=quality)
    return output_path


def convert_dng_to_jpeg(input_path: str, output_path: str, quality: int = 95) -> str:
    """Develop a DNG raw file to JPEG with rawpy. Raises ImportError when rawpy is missing."""
    import rawpy
    from PIL import Image

    with rawpy.imread(input_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_path, "JPEG", quality=quality)
    return output_path


def _convert_image(src: Path, dst: Path) -> None:
    ext = src.suffix.lower()
    if ext in HEIC_EXTS:
        convert_heic_to_jpeg(str(src), str(dst))
    elif ext in RAW_EXTS:
        convert_dng_to_jpeg(str(src), str(dst))
    else:
        convert_image_to_jpeg(str(src), str(dst))


def _convert_video(src: Path) -> Optional[Path]:
    """Re-encode to H.264 MP4 with ffmpeg next to src. Returns the new path, or None."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.warning("OpenCV cannot read %s and ffmpeg is not installed (brew install ffmpeg); skipping", src)
        return None
    dst = _converted_video_path(src)
    if dst == src:
        return None
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("ffmpeg could not convert %s: %s", src, exc)
        dst.unlink(missing_ok=True)
        return None
    if not video_readable(dst):
        logger.warning("ffmpeg output %s is still unreadable", dst)
        return None
    return dst


def normalize_capture_dir(capture_dir: str) -> dict[str, list[str]]:
    """Make every image and video under capture_dir (recursive) consumable by the pipeline.

    - JPEG/PNG are verified readable and used as they are.
    - HEIC/HEIF, WEBP, TIFF and DNG are converted to a JPEG with the same name next to
      the original. Originals are kept, and an existing .jpg of that name is never
      overwritten (cameras that save DNG + JPEG pairs keep their own JPEG). DNG needs
      rawpy; without it the file is reported as failed with a warning.
    - Videos that OpenCV cannot open are re-encoded with ffmpeg (<name>_h264.mp4); without
      ffmpeg they are reported as failed with a warning.

    Never raises on a bad file. Returns
    {"images": readable image paths (converted files appear as their JPEG),
     "videos": readable video paths,
     "converted": source files that were converted (now or on an earlier run),
     "failed": files that could not be read or converted}.
    """
    root = Path(capture_dir)
    images: dict[str, None] = {}  # ordered, de-duplicated
    videos: dict[str, None] = {}
    converted: list[str] = []
    failed: list[str] = []

    for src in _files_with_ext(root, ALL_IMAGE_EXTS):
        if src.suffix.lower() in IMAGE_EXTS:
            if _image_readable(src):
                images[str(src)] = None
            else:
                failed.append(str(src))
            continue
        dst = src.with_suffix(".jpg")
        if dst.exists():  # converted earlier, or the camera's own JPEG twin: keep it
            if _image_readable(dst):
                images[str(dst)] = None
                converted.append(str(src))
            else:
                failed.append(str(src))
            continue
        try:
            _convert_image(src, dst)
            images[str(dst)] = None
            converted.append(str(src))
        except ImportError as exc:
            logger.warning("cannot convert %s (%s not installed); skipping", src, exc.name or exc)
            failed.append(str(src))
        except Exception as exc:  # corrupt file, unsupported variant, ...
            logger.warning("conversion failed for %s: %s", src, exc)
            dst.unlink(missing_ok=True)
            failed.append(str(src))

    for video in _files_with_ext(root, VIDEO_EXTS):
        if video_readable(video):
            videos[str(video)] = None
            continue
        logger.warning("OpenCV cannot read video %s", video)
        fixed = _converted_video_path(video)
        if not (fixed.exists() and video_readable(fixed)):
            fixed = _convert_video(video)
        if fixed is not None:
            videos[str(fixed)] = None
            converted.append(str(video))
        else:
            failed.append(str(video))

    return {"images": list(images), "videos": list(videos), "converted": converted, "failed": failed}


def extract_image_metadata(image_path: str) -> dict:
    """Dimensions and EXIF facts that help COLMAP and the report.

    Returns {"dimensions": (width, height) or None, "focal_length_mm": float or None,
    "camera_model": str or None, "timestamp": str or None}. Missing EXIF gives None
    values; an unreadable file gives all None. Never raises.
    """
    meta: dict = {"dimensions": None, "focal_length_mm": None, "camera_model": None, "timestamp": None}
    try:
        from PIL import Image

        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        with Image.open(image_path) as img:
            meta["dimensions"] = (int(img.width), int(img.height))
            exif = img.getexif()
            try:
                exif_ifd = exif.get_ifd(0x8769)  # Exif sub-IFD: focal length, capture time
            except Exception:
                exif_ifd = {}
            focal = exif_ifd.get(0x920A)  # FocalLength
            if focal is not None:
                try:
                    meta["focal_length_mm"] = float(focal)
                except (TypeError, ValueError):
                    pass
            model = exif.get(0x0110)  # Model
            if model:
                make = exif.get(0x010F)  # Make
                model = str(model).strip().strip("\x00")
                meta["camera_model"] = f"{str(make).strip()} {model}" if make and str(make).strip() not in model else model
            stamp = exif_ifd.get(0x9003) or exif_ifd.get(0x9004) or exif.get(0x0132)  # original / digitized / modified
            if stamp:
                meta["timestamp"] = str(stamp).strip().strip("\x00")
    except Exception as exc:
        logger.warning("cannot read metadata of %s: %s", image_path, exc)
    return meta


def list_room_images(room_dir: str) -> list[str]:
    """Usable images of one room folder (any supported format), in name order.

    A HEIC/WEBP/TIFF/DNG that has a JPEG or PNG twin of the same name (made by
    normalize_capture_dir, or the camera's own JPEG) is represented by the twin only,
    so images are never counted or processed twice.
    """
    root = Path(room_dir)
    images = [p for p in sorted(root.iterdir())
              if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in ALL_IMAGE_EXTS]
    stems_with_direct = {p.with_suffix("").name for p in images if p.suffix.lower() in IMAGE_EXTS}
    return [str(p) for p in images
            if p.suffix.lower() in IMAGE_EXTS or p.with_suffix("").name not in stems_with_direct]


def _has_depth_stream(root: Path) -> bool:
    return any((p.parent / "depth").is_dir() for p in root.rglob("odometry.csv") if not _is_hidden(p, root))


def _summary(root: Path) -> str:
    """'Found X images in Y formats, Z videos' counted over every supported file under root."""
    images = _files_with_ext(root, ALL_IMAGE_EXTS)
    formats = sorted({p.suffix.lower().lstrip(".") for p in images})
    videos = list_videos(str(root))
    detail = f" ({', '.join(formats)})" if formats else ""
    return f"Found {len(images)} images in {len(formats)} formats{detail}, {len(videos)} videos"


def validate_capture_dir(capture_dir: str, tier: str) -> list[str]:
    """Check that capture_dir has the files the tier expects, and print a summary.

    - lidar: at least one .ply or .obj, or a raw depth capture (depth/ + odometry.csv)
    - video: at least one video OpenCV can read (.mp4 .mov .mkv .avi)
    - photo: one or more room sub-folders, each with 2-8 images in any supported
      format (jpg, png, heic, webp, tiff, dng)

    Raises CaptureValidationError listing every problem. Returns a list of
    non-fatal warnings (e.g. more than 8 photos in a room).
    """
    root = Path(capture_dir)
    problems: list[str] = []
    warnings: list[str] = []

    if not root.is_dir():
        raise CaptureValidationError(f"Not a directory: {capture_dir}")

    print(_summary(root), flush=True)

    if tier == "lidar":
        if not _files_with_ext(root, LIDAR_EXTS) and not _has_depth_stream(root):
            problems.append("lidar tier needs at least one .ply or .obj file, "
                            "or a raw depth capture (depth/ folder + odometry.csv)")
    elif tier == "video":
        videos = list_videos(capture_dir)
        if not videos:
            problems.append("video tier needs at least one video file (.mp4, .mov, .mkv or .avi)")
        elif not any(video_readable(v) for v in videos):
            problems.append("no readable video: OpenCV cannot open any video file "
                            "(install ffmpeg so it can be converted, or re-export as H.264 .mp4)")
    elif tier == "photo":
        room_dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        if not room_dirs:
            problems.append("photo tier needs one sub-folder per room (room-1/, room-2/, ...)")
        for room in room_dirs:
            count = len(list_room_images(str(room)))
            if count < MIN_PHOTOS_PER_ROOM:
                problems.append(
                    f"{room.name}: {count} photo(s), need at least {MIN_PHOTOS_PER_ROOM}"
                )
            elif count > MAX_PHOTOS_PER_ROOM:
                warnings.append(
                    f"{room.name}: {count} photos, protocol asks for {MIN_PHOTOS_PER_ROOM}-{MAX_PHOTOS_PER_ROOM}"
                )
    else:
        problems.append(f"unknown tier '{tier}' (expected photo, video or lidar)")

    if problems:
        raise CaptureValidationError("; ".join(problems))
    return warnings
