"""Bad input must give a valid report (maybe with no rooms), never a crash."""

import json

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from run import main
from src.models import PropertyReport
from src.pipeline import run_pipeline
from src.tiers.preprocessing import count_portrait_images
from src.tiers.single_image import estimate_room_from_images, find_vanishing_points


def _noise_image(path, size=(160, 120), seed=0):
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8))


def _assert_valid_empty_report(out_dir):
    data = json.loads((out_dir / "report.json").read_text())
    report = PropertyReport.model_validate(data)
    assert report.room_count == 0 and report.rooms == []
    assert report.damage_regions == [] and report.concealed_damage_flags == [] and report.scope_line_items == []
    assert report.total_floor_area.value == 0.0
    first = report.warnings[0]
    assert first.startswith("COLMAP reconstruction failed: insufficient feature matches between images. "
                            "Capture may have too little visual overlap.") or first.startswith("No usable room")
    assert (out_dir / "floor_plan.png").stat().st_size > 1000
    return report


def test_unmatchable_photos_give_an_empty_report(tmp_path):
    room = tmp_path / "photos" / "room-1"
    room.mkdir(parents=True)
    for i in range(3):
        _noise_image(room / f"IMG_{i}.jpg", seed=i)  # no two images share anything
    report = run_pipeline(str(tmp_path / "photos"), "photo", str(tmp_path / "out"), render=True)
    assert report.room_count == 0
    _assert_valid_empty_report(tmp_path / "out")


def test_cli_exits_zero_on_unmatchable_photos(tmp_path):
    room = tmp_path / "photos" / "room-1"
    room.mkdir(parents=True)
    for i in range(3):
        _noise_image(room / f"IMG_{i}.jpg", seed=i)
    result = CliRunner().invoke(main, [str(tmp_path / "photos"), "--tier", "photo", "--output-dir", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert "no rooms could be reconstructed" in result.output
    _assert_valid_empty_report(tmp_path / "out")


def test_unusable_video_gives_an_empty_report(tmp_path):
    cap = tmp_path / "video"
    cap.mkdir()
    writer = cv2.VideoWriter(str(cap / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    rng = np.random.default_rng(1)
    for _ in range(40):
        writer.write((rng.random((120, 160, 3)) * 255).astype(np.uint8))
    writer.release()
    run_pipeline(str(cap), "video", str(tmp_path / "out"), render=True)
    _assert_valid_empty_report(tmp_path / "out")


def test_junk_point_cloud_gives_an_empty_report(tmp_path):
    from create_test_ply import write_ply

    cap = tmp_path / "lidar"
    cap.mkdir()
    write_ply(cap / "scan.ply", np.random.default_rng(0).random((3000, 3)) * [0.5, 0.5, 0.1])  # a pancake, no room
    run_pipeline(str(cap), "lidar", str(tmp_path / "out"), render=True)
    _assert_valid_empty_report(tmp_path / "out")


def _interior_image(path, size=(640, 480)):
    """Line drawing of a room corner: lines to a central vanishing point plus true verticals/horizontals."""
    w, h = size
    img = np.full((h, w, 3), 235, np.uint8)
    vp = np.array([w / 2, h / 2])
    for corner in ([40, 40], [w - 40, 40], [40, h - 40], [w - 40, h - 40]):
        for t in np.linspace(0, 1, 6):  # several parallel-in-3D lines per corner direction
            start = np.array(corner) + t * 30
            cv2.line(img, tuple(start.astype(int)), tuple(vp.astype(int)), (40, 40, 40), 2)
    for x in (40, w - 40, w // 2 - 120, w // 2 + 120):
        cv2.line(img, (x, 60), (x, h - 60), (30, 30, 30), 2)
    for y in (60, h // 2 - 100, h // 2 + 100, h - 60):
        cv2.line(img, (60, y), (w - 60, y), (30, 30, 30), 2)
    cv2.imwrite(str(path), img)


def test_vanishing_points_of_an_interior_drawing(tmp_path):
    _interior_image(tmp_path / "room.png")
    vp = find_vanishing_points(cv2.imread(str(tmp_path / "room.png")))
    assert len(vp.points) >= 2 and vp.orthogonal and vp.coverage > 0.35


def test_single_image_estimate_is_rough_and_wide(tmp_path):
    _interior_image(tmp_path / "room.png")
    room = estimate_room_from_images([str(tmp_path / "room.png")], "room-1")
    assert room is not None and room.metadata["rough_estimate"] and room.metadata["dimensions_are_priors"]


def test_noise_is_not_turned_into_a_room(tmp_path):
    _noise_image(tmp_path / "noise.png", size=(640, 480))
    assert estimate_room_from_images([str(tmp_path / "noise.png")], "room-1") is None
    assert estimate_room_from_images([str(tmp_path / "missing.png")], "room-1") is None


def test_fallback_room_reaches_the_report_with_wide_intervals(tmp_path):
    """Two identical drawings cannot be reconstructed (no baseline), so the single-image fallback runs."""
    room = tmp_path / "photos" / "room-1"
    room.mkdir(parents=True)
    for i in range(2):
        _interior_image(room / f"IMG_{i}.png")
    report = run_pipeline(str(tmp_path / "photos"), "photo", str(tmp_path / "out"), render=True)
    assert report.room_count == 1 and report.rooms[0].rough_estimate
    area = report.rooms[0].floor_area
    assert area.confidence_low < 0.35 * area.value and area.confidence_high > 2.0 * area.value
    wall = report.rooms[0].walls[0].length
    assert wall.confidence_low == pytest.approx(wall.value * 0.5, rel=0.01)
    assert any("single-image estimate" in w for w in report.warnings)
    assert PropertyReport.model_validate(json.loads((tmp_path / "out" / "report.json").read_text()))


def test_portrait_images_are_counted_after_exif(tmp_path):
    from PIL import Image

    Image.new("RGB", (60, 100)).save(tmp_path / "portrait.jpg")
    Image.new("RGB", (100, 60)).save(tmp_path / "landscape.jpg")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotated 90 degrees: a 100x60 file displays as 60x100
    Image.new("RGB", (100, 60)).save(tmp_path / "rotated.jpg", exif=exif)
    paths = [str(tmp_path / n) for n in ("portrait.jpg", "landscape.jpg", "rotated.jpg")]
    assert count_portrait_images(paths) == 2


def test_motion_keyframes_follow_the_motion(tmp_path):
    from src.tiers.video import extract_keyframes

    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((240, 960)) * 255).astype(np.uint8), (0, 0), 1.5)

    def video(name, speed):
        path = tmp_path / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
        for i in range(150):
            x = int(i * speed)
            writer.write(cv2.cvtColor(base[:, x:x + 320], cv2.COLOR_GRAY2BGR))
        writer.release()
        return path

    moving = extract_keyframes(str(video("pan.mp4", 4)), str(tmp_path / "a"), mode="motion")
    still = extract_keyframes(str(video("still.mp4", 0)), str(tmp_path / "b"), mode="motion")
    assert len(moving) >= 8  # a new keyframe at least every 10 frames while the view changes
    assert len(still) <= 4  # a static view only gets the periodic keyframe
    fixed = extract_keyframes(str(tmp_path / "pan.mp4"), str(tmp_path / "c"), step=5)
    assert len(fixed) >= 25
