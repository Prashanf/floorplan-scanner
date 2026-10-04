"""Capture directory validation and normalization (HEIC -> JPEG, video readability)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

HEIC_EXTS = {".heic", ".heif"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".mov"}
LIDAR_EXTS = {".ply", ".obj"}

MIN_PHOTOS_PER_ROOM = 2
MAX_PHOTOS_PER_ROOM = 8


class CaptureValidationError(ValueError):
    """The capture directory does not match what the tier expects."""


@dataclass
class NormalizeReport:
    """What normalize_capture_dir did."""

    converted: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    videos_ok: list[str] = field(default_factory=list)
    videos_unreadable: list[str] = field(default_factory=list)


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


def _files_with_ext(root: Path, exts: set[str]) -> list[Path]:
    """Files under root (recursive) whose extension is in exts, sorted, ignoring dotfiles."""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in exts
    )


def _video_readable(path: Path) -> bool:
    """True if OpenCV can open the file and read one frame."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        ok, _ = cap.read() if cap.isOpened() else (False, None)
        return bool(ok)
    finally:
        cap.release()


def normalize_capture_dir(capture_dir: str) -> NormalizeReport:
    """Make a capture directory consumable by the pipeline.

    - Converts every HEIC/HEIF image to a JPEG next to the original (original kept;
      an existing up-to-date JPEG is not rewritten).
    - Checks every video opens in OpenCV (iPhone .mov/.mp4 with HEVC can fail on
      some OpenCV builds); unreadable files are reported, not fatal.
    """
    root = Path(capture_dir)
    report = NormalizeReport()

    for src in _files_with_ext(root, HEIC_EXTS):
        dst = src.with_suffix(".jpg")
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            report.skipped.append(str(src))
            continue
        try:
            convert_heic_to_jpeg(str(src), str(dst))
            report.converted.append(str(src))
        except Exception as exc:  # corrupt file, missing pillow-heif, ...
            logger.warning("HEIC conversion failed for %s: %s", src, exc)
            report.failed.append(str(src))

    videos = _files_with_ext(root, VIDEO_EXTS)
    if videos:
        try:
            import cv2  # noqa: F401
        except ImportError:
            logger.warning("opencv-python not installed; skipping video readability check")
            return report
        for video in videos:
            if _video_readable(video):
                report.videos_ok.append(str(video))
            else:
                logger.warning("OpenCV cannot read video %s", video)
                report.videos_unreadable.append(str(video))

    return report


def list_room_images(room_dir: str) -> list[str]:
    """Usable images of one room folder, in name order.

    A HEIC that has a converted JPEG twin is represented by the JPEG only, so
    images are never counted or processed twice.
    """
    root = Path(room_dir)
    images = [p for p in sorted(root.iterdir())
              if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS | HEIC_EXTS]
    stems_with_jpeg = {p.with_suffix("").name for p in images if p.suffix.lower() in IMAGE_EXTS}
    return [str(p) for p in images
            if p.suffix.lower() in IMAGE_EXTS or p.with_suffix("").name not in stems_with_jpeg]


def validate_capture_dir(capture_dir: str, tier: str) -> list[str]:
    """Check that capture_dir has the files the tier expects.

    - lidar: at least one .ply or .obj
    - video: at least one .mp4 or .mov
    - photo: one or more room sub-folders, each with 2-8 images (jpg/png/heic)

    Raises CaptureValidationError listing every problem. Returns a list of
    non-fatal warnings (e.g. more than 8 photos in a room).
    """
    root = Path(capture_dir)
    problems: list[str] = []
    warnings: list[str] = []

    if not root.is_dir():
        raise CaptureValidationError(f"Not a directory: {capture_dir}")

    if tier == "lidar":
        if not _files_with_ext(root, LIDAR_EXTS):
            problems.append("lidar tier needs at least one .ply or .obj file")
    elif tier == "video":
        if not _files_with_ext(root, VIDEO_EXTS):
            problems.append("video tier needs at least one .mp4 or .mov file")
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
