"""Room merging (video tier), the rectangle fallback (not LiDAR), and the guard that leaves the LiDAR tier alone."""

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from create_test_ply import make_apartment, write_ply
from src.geometry.openings import OpeningDetection
from src.geometry.rectangle_fit import fit_rectangle
from src.geometry.wall_fitting import WallSegment
from src.room_ir import CameraPose, PointCloud, PropertyIR, RoomIR
from src.stitching import room_merge
from src.stitching.multi_room import stitch_rooms
from test_synthetic_room import make_room_cloud


def box_room(room_id, x0, y0, x1, y1, openings=(), ceiling=2.5, tier="video"):
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    walls = [WallSegment(start=pts[i], end=pts[(i + 1) % 4], length=math.dist(pts[i], pts[(i + 1) % 4]),
                         direction=math.atan2(pts[(i + 1) % 4][1] - pts[i][1], pts[(i + 1) % 4][0] - pts[i][0]),
                         inlier_count=50) for i in range(4)]
    cloud = np.column_stack([np.random.default_rng(0).uniform([x0, y0], [x1, y1], (200, 2)), np.full(200, 1.0)])
    return RoomIR(room_id=room_id, point_cloud=PointCloud(points=cloud), tier=tier, wall_segments=walls,
                  floor_polygon=pts, ceiling_height=ceiling, images=[f"{room_id}.jpg"], openings=list(openings))


def op(wall, pos, width=0.9, kind="door"):
    return OpeningDetection(wall_index=wall, type=kind, position_along_wall=pos, width=width, height=2.0, confidence=1.0)


def stitched(rooms, tier="video"):
    return stitch_rooms(PropertyIR(rooms=rooms, tier=tier, capture_dir="x"), drift_correction=False)


# ---------------------------------------------------------------- merging

def test_a_hallway_cut_in_two_becomes_one_room():
    # two 2 x 3 m fragments stacked along y, sharing the full 2 m width, no opening between them
    a = box_room("room-1", 0, 0, 2, 3, openings=[op(1, 1.05)], ceiling=2.5)           # a door on its east wall
    b = box_room("room-2", 0, 3, 2, 6, openings=[op(0, 0.55)], ceiling=2.7)           # an "opening" on the removed interface
    ir = stitched([a, b])
    assert [r.room_id for r in ir.rooms] == ["room-1"]
    merged = ir.rooms[0]
    assert merged.metadata["merged_from"] == ["room-1", "room-2"]
    assert Polygon(merged.floor_polygon).area == pytest.approx(12.0, abs=0.05)
    assert len(merged.wall_segments) == 4 and sorted(round(w.length, 1) for w in merged.wall_segments) == [2.0, 2.0, 6.0, 6.0]
    assert merged.ceiling_height == pytest.approx(2.6, abs=0.01)                      # area-weighted
    assert len(merged.openings) == 1 and merged.openings[0].width == 0.9              # the interface opening is gone
    assert merged.images == ["room-1.jpg", "room-2.jpg"] and len(merged.point_cloud) == 400
    assert ir.room_transforms == {"room-1": (0.0, 0.0, 0.0)} and ir.adjacencies == []
    assert any("room-2 merged into room-1" in w for w in ir.warnings)


def test_three_fragments_merge_step_by_step():
    rooms = [box_room(f"room-{i + 1}", 0, 3 * i, 2, 3 * i + 3) for i in range(3)]
    ir = stitched(rooms)
    assert len(ir.rooms) == 1 and Polygon(ir.rooms[0].floor_polygon).area == pytest.approx(18.0, abs=0.1)


def test_rooms_separated_by_a_doorway_stay_separate():
    a = box_room("room-1", 0, 0, 3, 3, openings=[op(1, 1.05)])         # door on the east wall ...
    b = box_room("room-2", 3, 0, 6, 3, openings=[op(3, 1.05)])         # ... and the matching one on the west wall
    ir = stitched([a, b])
    assert [r.room_id for r in ir.rooms] == ["room-1", "room-2"]
    assert ir.adjacencies[0]["shared_opening_id"] == "room-1/door-0"


def test_a_short_shared_boundary_does_not_merge():
    a = box_room("room-1", 0, 0, 4, 3)
    b = box_room("room-2", 3, 3, 6, 6)                                  # touches room-1 along 1 m of its 4 m / 3 m walls
    ir = stitched([a, b])
    assert len(ir.rooms) == 2


def test_lidar_and_photo_tiers_are_not_merged():
    for tier in ("lidar", "photo"):
        rooms = [box_room("room-1", 0, 0, 2, 3, tier=tier), box_room("room-2", 0, 3, 2, 6, tier=tier)]
        ir = stitched(rooms, tier=tier)
        assert len(ir.rooms) == 2 and not ir.warnings


def test_merged_room_keeps_poses_consistent_with_the_cloud():
    tf = (2.0, 1.0, 0.4)
    rng = np.random.default_rng(1)
    rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    rot *= np.sign(np.linalg.det(rot))
    pose = CameraPose("a.jpg", rot, rng.normal(size=3), np.eye(3))
    moved = room_merge._transform_poses([pose], tf)[0]
    x = rng.normal(size=3)
    x_global = room_merge._rotation(tf[2]) @ x + np.array([tf[0], tf[1], 0.0])
    assert moved.rotation @ x_global + moved.translation == pytest.approx(pose.rotation @ x + pose.translation)


# ---------------------------------------------------------------- rectangle fallback

def test_rectangle_replaces_a_noisy_box():
    cloud = make_room_cloud(seed=2)
    found = fit_rectangle(cloud, n_walls=9)
    assert found is not None
    walls, polygon = found
    assert len(walls) == 4 and sorted(round(w.length, 1) for w in walls) == [3.0, 3.0, 4.0, 4.0]
    assert Polygon(polygon).area == pytest.approx(12.0, abs=0.3)
    assert Polygon(polygon).exterior.is_ccw
    assert all(w.inlier_count > 100 for w in walls)


def test_rectangle_is_only_tried_above_six_walls_and_rejects_non_rectangular_rooms():
    box = make_room_cloud(seed=3)
    assert fit_rectangle(box, n_walls=6) is None and fit_rectangle(box, n_walls=4) is None
    a, b = make_room_cloud(seed=4), make_room_cloud(seed=5, offset=(2.0, 3.0, 0.0))      # a staircase shape
    stair = PointCloud(points=np.vstack([a.points, b.points]))
    assert fit_rectangle(stair, n_walls=9) is None                                       # the bounding box explains < 70 %


def test_rectangle_handles_a_rotated_room():
    cloud = make_room_cloud(seed=6, yaw_deg=33.0)
    walls, _ = fit_rectangle(cloud, n_walls=8)
    assert sorted(round(w.length, 1) for w in walls) == [3.0, 3.0, 4.0, 4.0]


# ---------------------------------------------------------------- the LiDAR tier is untouched

def test_lidar_pipeline_never_calls_the_rectangle_fallback_or_the_merge(tmp_path, monkeypatch):
    from src import pipeline

    def boom(*a, **k):
        raise AssertionError("must not run for the LiDAR tier")

    monkeypatch.setattr(pipeline, "fit_rectangle", boom)
    monkeypatch.setattr(room_merge, "merge_oversegmented_rooms", boom)
    points = make_apartment(seed=1, density=300.0)
    write_ply(str(tmp_path / "apartment.ply"), points)
    report = pipeline.run_pipeline(str(tmp_path), "lidar", str(tmp_path / "out"), render=False)
    assert report.room_count >= 1
