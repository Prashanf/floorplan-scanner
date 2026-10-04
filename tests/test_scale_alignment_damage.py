"""Scale recovery chain, wall alignment of SfM clouds, and damage false-positive filters."""

import cv2
import numpy as np
import pytest

from src.damage.detection import DamageDetection, detect_damage, drop_persistent
from src.geometry.floor_area import compute_floor_area
from src.geometry.wall_fitting import dominant_wall_angle, fit_walls
from src.pipeline import _cap_room_detections
from src.tiers import colmap_utils as cu
from test_sfm_helpers import sfm_like_scene


def _area(cloud):
    _, polygon = fit_walls(cloud, inlier_threshold=cu.SFM_WALL_THRESHOLD)
    return compute_floor_area(polygon)


@pytest.mark.parametrize("units_per_meter", [0.05, 0.4, 2.7, 25.0])
def test_scale_recovery_gives_a_realistic_room_from_any_arbitrary_scale(units_per_meter):
    pts, poses, _ = sfm_like_scene(unit_per_meter=units_per_meter)
    cloud, _, meta = cu.make_metric_point_cloud(pts, poses)
    vertical = np.ptp(np.percentile(cloud.points[:, 2], [1, 99]))
    assert vertical == pytest.approx(2.5, abs=0.4)  # the vertical extent is what pins the scale
    assert _area(cloud) == pytest.approx(12.0, rel=0.25)  # the synthetic room is 4 x 3 m
    assert meta["scale_method"] in {"door", "ceiling-height-prior", "floor-ceiling", "longest-wall-prior"}
    assert not meta["scale_unreliable"]


def test_tiny_reconstruction_falls_through_to_the_longest_wall_prior(monkeypatch):
    """If the ceiling prior still leaves a room under 4 m2, the longest wall is scaled to 3.5 m."""
    pts, poses, _ = sfm_like_scene(unit_per_meter=2.7)
    areas = iter([3.0, 3.0, 3.0, 9.0, 9.0])  # door step, ceiling step, then after the wall step
    monkeypatch.setattr(cu, "_outline_area", lambda cloud: next(areas))
    monkeypatch.setattr(cu, "_longest_wall", lambda cloud: 2.0)
    _, _, meta = cu.make_metric_point_cloud(pts, poses)
    assert meta["scale_method"] == "longest-wall-prior"
    assert "3.5" in meta["scale_detail"]


def test_implausible_final_area_is_flagged(monkeypatch):
    pts, poses, _ = sfm_like_scene()
    monkeypatch.setattr(cu, "_outline_area", lambda cloud: 0.8)
    monkeypatch.setattr(cu, "_longest_wall", lambda cloud: 0.0)
    _, _, meta = cu.make_metric_point_cloud(pts, poses)
    assert meta["scale_unreliable"] is True


def test_plan_is_axis_aligned_whatever_the_reconstruction_heading():
    for seed in (0, 1, 2):  # sfm_like_scene rotates the room by a different random rotation each time
        pts, poses, _ = sfm_like_scene(seed=seed)
        cloud, new_poses, meta = cu.make_metric_point_cloud(pts, poses)
        assert meta["aligned_to_walls"]
        angle = dominant_wall_angle(cloud, cu.SFM_WALL_THRESHOLD)
        assert min(angle % (np.pi / 2), np.pi / 2 - angle % (np.pi / 2)) < np.deg2rad(4)
        segments, _ = fit_walls(cloud, inlier_threshold=cu.SFM_WALL_THRESHOLD, reference_angle=0.0)
        for s in segments:
            dx, dy = abs(s.end[0] - s.start[0]), abs(s.end[1] - s.start[1])
            assert min(dx, dy) < 0.06 * max(dx, dy)  # every wall is horizontal or vertical on the plan
        # the poses were rotated with the cloud: cameras still look into the room from the same side
        centers = np.array([-p.rotation.T @ p.translation for p in new_poses])
        assert np.all(np.abs(centers[:, 2] - centers[:, 2].mean()) < 0.1)


# ---------------------------------------------------------------- damage

def _det(box, cls="crack", conf=0.5):
    return DamageDetection(image_path="x.jpg", bbox=box, damage_class=cls, confidence=conf, pixel_area=100,
                           image_size=(1000, 800))


def test_fixed_position_detection_over_many_frames_is_dropped():
    frames = [[_det((100, 100, 300, 160))] for _ in range(5)]  # same place in 5 consecutive frames
    assert drop_persistent(frames, limit=3, iou=0.5) == [[], [], [], [], []]


def test_short_runs_and_moving_detections_survive():
    short = [[_det((100, 100, 300, 160))] for _ in range(3)] + [[]]  # 3 frames is allowed
    assert [len(f) for f in drop_persistent(short, limit=3, iou=0.5)] == [1, 1, 1, 0]
    moving = [[_det((100 + 80 * i, 100, 300 + 80 * i, 160))] for i in range(6)]  # moves across the frame
    assert sum(map(len, drop_persistent(moving, limit=3, iou=0.5))) == 6
    other_class = [[_det((100, 100, 300, 160), "crack")] if i % 2 == 0 else [_det((100, 100, 300, 160), "mold")]
                   for i in range(6)]  # alternating classes never form a run
    assert sum(map(len, drop_persistent(other_class, limit=3, iou=0.5))) == 6


def _blob_image(path, blobs, size=(1280, 960)):
    img = np.full((size[1], size[0], 3), 235, np.uint8)
    for x, y, w, h in blobs:
        img[y:y + h, x:x + w] = (110, 150, 185)  # BGR light brown: inside the water-stain HSV range
    cv2.imwrite(str(path), img)


def test_at_most_five_detections_per_image(tmp_path):
    blobs = [(40 + 200 * (i % 6), 40 + 300 * (i // 6), 100, 100) for i in range(12)]  # twelve 100 x 100 stains
    _blob_image(tmp_path / "many.png", blobs)
    found = detect_damage([str(tmp_path / "many.png")])
    assert 1 <= len(found) <= 5
    assert any(d.damage_class == "water_stain" for d in found)


def test_detections_covering_most_of_the_image_are_ignored(tmp_path):
    _blob_image(tmp_path / "wall.png", [(0, 0, 1280, 500)])  # a brown wall: 52% of the frame
    assert not any(d.damage_class == "water_stain" for d in detect_damage([str(tmp_path / "wall.png")]))


def test_small_blobs_below_the_new_minimum_area_are_ignored(tmp_path):
    _blob_image(tmp_path / "small.png", [(100, 100, 40, 40), (400, 400, 30, 30)])  # 1600 px and 900 px: grout-sized
    assert not any(d.damage_class == "water_stain" for d in detect_damage([str(tmp_path / "small.png")]))


class _Projected:
    def __init__(self, room, conf):
        self.room_id = room
        self.damage_detection = _det((0, 0, 10, 10), conf=conf)


def test_a_room_with_more_than_twenty_detections_keeps_the_top_ten():
    items = [_Projected("room-1", i / 100) for i in range(25)] + [_Projected("room-2", 0.5) for _ in range(3)]
    warnings: list[str] = []
    kept = _cap_room_detections(items, warnings)
    room1 = [d for d in kept if d.room_id == "room-1"]
    assert len(room1) == 10 and min(d.damage_detection.confidence for d in room1) == pytest.approx(0.15)
    assert len([d for d in kept if d.room_id == "room-2"]) == 3
    assert warnings and "high false positive rate" in warnings[0]
