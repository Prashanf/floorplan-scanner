"""Photo and video tiers through real COLMAP on rendered rooms. Slow: `pytest -m slow`."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow
pytest.importorskip("cv2")
pytest.importorskip("pycolmap")

from create_test_photos import build_room, make_video, photo_poses, look_at, render, _save_jpeg  # noqa: E402
from create_test_ply import layout  # noqa: E402


@pytest.fixture(scope="module")
def room1_photos(tmp_path_factory):
    root = tmp_path_factory.mktemp("photos")
    rng = np.random.default_rng(0)
    space = layout()[0]
    planes = build_room(space, rng)
    (root / "room-1").mkdir()
    for i, (eye, target) in enumerate(photo_poses(space, 8, rng), 1):
        _save_jpeg(root / "room-1" / f"IMG_{i:04d}.jpg", render(planes, eye, look_at(eye, target)))
    return root


def test_photo_tier_single_room(room1_photos):
    from src.tiers.photo import process_photos

    ir = process_photos(str(room1_photos))
    assert ir.tier == "photo" and len(ir.rooms) == 1
    room = ir.rooms[0]
    assert room.tier == "photo" and room.metadata["registered_images"] >= 6
    assert len(room.camera_poses) == room.metadata["registered_images"]
    assert all(p.image_path.startswith(str(room1_photos)) for p in room.camera_poses)
    z = room.point_cloud.points[:, 2]
    assert np.ptp(np.percentile(z, [1, 99])) == pytest.approx(2.5, rel=0.2)


def test_photo_tier_is_repeatable(room1_photos):
    from src.tiers.photo import process_photos

    a = process_photos(str(room1_photos)).rooms[0]
    b = process_photos(str(room1_photos)).rooms[0]
    assert a.metadata["scale_factor"] == pytest.approx(b.metadata["scale_factor"], rel=1e-3)
    assert len(a.point_cloud) == len(b.point_cloud)


def test_photo_pipeline_end_to_end(room1_photos, tmp_path):
    from src.pipeline import run_pipeline

    report = run_pipeline(str(room1_photos), "photo", str(tmp_path), render=True)
    assert report.capture_tier == "photo" and report.room_count == 1
    assert report.rooms[0].floor_area.value == pytest.approx(12.0, rel=0.25)
    assert (tmp_path / "floor_plan.png").exists()


def test_video_tier_and_pipeline(tmp_path):
    from src.pipeline import run_pipeline

    clip = make_video(tmp_path / "clip", seed=0, seconds=6)
    report = run_pipeline(str(clip.parent), "video", str(tmp_path / "out"), render=False)
    assert report.capture_tier == "video" and report.room_count >= 1
    room = report.rooms[0]
    assert room.floor_area.value == pytest.approx(12.0, rel=0.15)
    assert room.ceiling_height.value == pytest.approx(2.5, abs=0.3)
