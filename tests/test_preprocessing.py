"""Preprocessing: every common photo/video format ends up readable, and bad files never crash."""

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.tiers.preprocessing import (CaptureValidationError, extract_image_metadata,
                                     normalize_capture_dir, validate_capture_dir)


def _picture(size=(96, 64)) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def _write_video(path: Path, fourcc: str) -> bool:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 10.0, (64, 48))
    if not writer.isOpened():
        return False
    rng = np.random.default_rng(1)
    for _ in range(10):
        writer.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    writer.release()
    return path.exists() and path.stat().st_size > 0


def test_jpeg_comes_out_readable_by_opencv(tmp_path):
    """A JPEG from any source passes through normalize_capture_dir and OpenCV reads it."""
    src = tmp_path / "IMG_0001.jpg"
    _picture().save(src, "JPEG")
    result = normalize_capture_dir(str(tmp_path))
    assert result["images"] == [str(src)]
    assert result["failed"] == [] and result["converted"] == []
    img = cv2.imread(result["images"][0])
    assert img is not None and img.shape[:2] == (64, 96)


def test_uppercase_extension_and_nested_folders(tmp_path):
    nested = tmp_path / "room-1" / "deep"
    nested.mkdir(parents=True)
    _picture().save(nested / "IMG_1.JPG", "JPEG")
    _picture().save(nested / "shot.PNG", "PNG")
    result = normalize_capture_dir(str(tmp_path))
    assert sorted(Path(p).name for p in result["images"]) == ["IMG_1.JPG", "shot.PNG"]


@pytest.mark.parametrize("name,fmt,kwargs", [
    ("a.webp", "WEBP", {}),
    ("b.tiff", "TIFF", {}),
    ("c.TIF", "TIFF", {}),
])
def test_other_formats_are_converted_to_readable_jpeg(tmp_path, name, fmt, kwargs):
    src = tmp_path / name
    _picture().save(src, fmt, **kwargs)
    result = normalize_capture_dir(str(tmp_path))
    jpg = src.with_suffix(".jpg")
    assert result["converted"] == [str(src)] and result["failed"] == []
    assert result["images"] == [str(jpg)]
    assert src.exists()  # original kept
    assert cv2.imread(str(jpg)) is not None


def test_heic_is_converted(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    src = tmp_path / "IMG_9.HEIC"
    try:
        _picture().save(src, "HEIF")
    except Exception as exc:
        pytest.skip(f"no HEIF encoder in this build: {exc}")
    result = normalize_capture_dir(str(tmp_path))
    assert result["images"] == [str(src.with_suffix(".jpg"))]
    assert cv2.imread(result["images"][0]) is not None


def test_16bit_tiff_and_rgba_png_survive(tmp_path):
    gray16 = Image.fromarray((np.arange(64 * 96, dtype=np.uint16).reshape(64, 96)))
    gray16.save(tmp_path / "deep.tiff")
    Image.new("RGBA", (40, 40), (10, 200, 30, 128)).save(tmp_path / "alpha.png")
    result = normalize_capture_dir(str(tmp_path))
    assert result["failed"] == []
    assert cv2.imread(str(tmp_path / "deep.jpg")) is not None
    assert cv2.imread(str(tmp_path / "alpha.png")) is not None


def test_existing_jpeg_twin_is_never_overwritten(tmp_path):
    """Cameras that save DNG + JPEG pairs keep their own JPEG."""
    jpg = tmp_path / "IMG_5.jpg"
    _picture().save(jpg, "JPEG")
    before = jpg.read_bytes()
    _picture().save(tmp_path / "IMG_5.webp", "WEBP")
    result = normalize_capture_dir(str(tmp_path))
    assert jpg.read_bytes() == before
    assert result["images"] == [str(jpg)]


def test_dng_without_rawpy_is_skipped_with_warning(tmp_path, monkeypatch, caplog):
    (tmp_path / "IMG_7.dng").write_bytes(b"not really a dng")
    monkeypatch.setitem(sys.modules, "rawpy", None)  # import rawpy -> ImportError
    result = normalize_capture_dir(str(tmp_path))
    assert result["failed"] == [str(tmp_path / "IMG_7.dng")] and result["images"] == []
    assert "rawpy" in caplog.text


def test_corrupt_files_are_reported_not_raised(tmp_path):
    (tmp_path / "bad.jpg").write_bytes(b"\x00garbage")
    (tmp_path / "broken.webp").write_bytes(b"RIFFgarbage")
    (tmp_path / "bad.dng").write_bytes(b"garbage")
    (tmp_path / "bad.mov").write_bytes(b"garbage")
    _picture().save(tmp_path / "good.jpg", "JPEG")
    result = normalize_capture_dir(str(tmp_path))
    assert result["images"] == [str(tmp_path / "good.jpg")]
    assert sorted(Path(p).name for p in result["failed"]) == ["bad.dng", "bad.jpg", "bad.mov", "broken.webp"]
    assert not (tmp_path / "broken.jpg").exists()  # no half-written output left behind


def test_hidden_files_are_ignored(tmp_path):
    (tmp_path / ".DS_Store").write_bytes(b"x")
    (tmp_path / "._IMG_1.jpg").write_bytes(b"resource fork")
    hidden = tmp_path / ".cache"
    hidden.mkdir()
    _picture().save(hidden / "x.jpg", "JPEG")
    assert normalize_capture_dir(str(tmp_path)) == {"images": [], "videos": [], "converted": [], "failed": []}


def test_missing_directory_returns_empty(tmp_path):
    assert normalize_capture_dir(str(tmp_path / "nope"))["images"] == []


@pytest.mark.parametrize("name,fourcc", [("walk.mp4", "mp4v"), ("old.avi", "MJPG"), ("rec.mkv", "mp4v"),
                                         ("IMG.MOV", "mp4v")])
def test_videos_are_found_when_opencv_reads_them(tmp_path, name, fourcc):
    if not _write_video(tmp_path / name, fourcc):
        pytest.skip(f"this OpenCV build cannot write {name}")
    result = normalize_capture_dir(str(tmp_path))
    assert result["videos"] == [str(tmp_path / name)] and result["failed"] == []


def test_unreadable_video_without_ffmpeg_is_reported(tmp_path, monkeypatch, caplog):
    (tmp_path / "clip.mov").write_bytes(b"garbage")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    result = normalize_capture_dir(str(tmp_path))
    assert result["videos"] == [] and result["failed"] == [str(tmp_path / "clip.mov")]
    assert "ffmpeg" in caplog.text


def test_validate_photo_tier_accepts_any_supported_format(tmp_path, capsys):
    for room in ("room-1", "room-2"):
        (tmp_path / room).mkdir()
    _picture().save(tmp_path / "room-1" / "a.webp", "WEBP")
    _picture().save(tmp_path / "room-1" / "b.tiff", "TIFF")
    _picture().save(tmp_path / "room-2" / "c.jpg", "JPEG")
    _picture().save(tmp_path / "room-2" / "d.png", "PNG")
    assert validate_capture_dir(str(tmp_path), "photo") == []
    assert "Found 4 images in 4 formats" in capsys.readouterr().out


def test_validate_photo_tier_needs_two_images_per_room(tmp_path):
    (tmp_path / "room-1").mkdir()
    _picture().save(tmp_path / "room-1" / "a.jpg", "JPEG")
    with pytest.raises(CaptureValidationError, match="room-1: 1 photo"):
        validate_capture_dir(str(tmp_path), "photo")


def test_validate_video_tier_needs_a_readable_video(tmp_path, capsys):
    (tmp_path / "clip.mov").write_bytes(b"garbage")
    with pytest.raises(CaptureValidationError, match="no readable video"):
        validate_capture_dir(str(tmp_path), "video")
    assert "1 videos" in capsys.readouterr().out  # found, though unreadable
    if _write_video(tmp_path / "good.mp4", "mp4v"):
        validate_capture_dir(str(tmp_path), "video")


def test_validate_lidar_tier(tmp_path):
    with pytest.raises(CaptureValidationError):
        validate_capture_dir(str(tmp_path), "lidar")
    (tmp_path / "scan.PLY").write_bytes(b"ply")
    assert validate_capture_dir(str(tmp_path), "lidar") == []


def test_extract_image_metadata_reads_exif(tmp_path):
    exif = Image.Exif()
    exif[0x010F] = "Samsung"
    exif[0x0110] = "SM-S918B"
    ifd = exif.get_ifd(0x8769)
    ifd[0x920A] = 6.3
    ifd[0x9003] = "2026:10:04 09:30:00"
    path = tmp_path / "exif.jpg"
    _picture((120, 80)).save(path, "JPEG", exif=exif)
    meta = extract_image_metadata(str(path))
    assert meta["dimensions"] == (120, 80)
    assert meta["focal_length_mm"] == pytest.approx(6.3, abs=0.01)
    assert meta["camera_model"] == "Samsung SM-S918B"
    assert meta["timestamp"] == "2026:10:04 09:30:00"


def test_extract_image_metadata_without_exif_or_file(tmp_path):
    path = tmp_path / "plain.png"
    _picture().save(path, "PNG")
    assert extract_image_metadata(str(path)) == {
        "dimensions": (96, 64), "focal_length_mm": None, "camera_model": None, "timestamp": None}
    assert extract_image_metadata(str(tmp_path / "missing.jpg"))["dimensions"] is None
