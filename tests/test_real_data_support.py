"""Support for real scans: raw depth logs, partial rooms (no ceiling, L-shapes), wall-aligned clouds."""

import csv

import cv2
import numpy as np
import pytest

from src.geometry.ceiling import is_ceiling_observed
from src.geometry.room_outline import fit_outline
from src.geometry.wall_fitting import fit_walls
from src.room_ir import PointCloud
from src.tiers.depth_stream import find_depth_streams, load_depth_stream


def _write_stream(root, poses, depth_mm=2000, size=(64, 48), intrinsics=(400.0, 400.0, 320.0, 240.0)):
    """A depth log: constant depth, full confidence; poses = [(x, y, z, qx, qy, qz, qw)]."""
    (root / "depth").mkdir(parents=True)
    (root / "confidence").mkdir()
    header = ["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy",
              "distortion_center_x", "distortion_center_y"]
    with open(root / "odometry.csv", "w", newline="") as f:
        f.write(", ".join(header) + "\n")  # real logs pad the header with spaces
        writer = csv.writer(f)
        for i, pose in enumerate(poses):
            writer.writerow([float(i), f"{i:06d}", *pose, *intrinsics, "", ""])
            cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), np.full(size[::-1], depth_mm, np.uint16))
            cv2.imwrite(str(root / "confidence" / f"{i:06d}.png"), np.full(size[::-1], 2, np.uint8))
    np.savetxt(root / "camera_matrix.csv", [[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]],
                                             [0, 0, 1]], delimiter=",")


def test_depth_stream_backprojects_onto_the_wall(tmp_path):
    """Identity pose, camera looks along +z: every point must sit on the plane z = depth."""
    _write_stream(tmp_path / "scan", [(0, 0, 0, 0, 0, 0, 1)] * 3)
    assert find_depth_streams(str(tmp_path)) == [tmp_path / "scan"]
    points = load_depth_stream(str(tmp_path / "scan"), voxel_size=0.01)
    assert len(points) > 100
    assert np.allclose(points[:, 2], 2.0, atol=1e-6)
    # the 64 px wide depth map spans 640 px of RGB: half-width = 320 px * 2 m / 400 px = 1.6 m
    assert points[:, 0].max() == pytest.approx(1.6, abs=0.1) and points[:, 0].min() == pytest.approx(-1.6, abs=0.1)


def test_depth_stream_applies_translation(tmp_path):
    _write_stream(tmp_path / "scan", [(1.0, 2.0, 3.0, 0, 0, 0, 1)])
    points = load_depth_stream(str(tmp_path / "scan"))
    assert np.allclose(points[:, 2], 5.0, atol=1e-6)
    assert np.allclose(points[:, 1].mean(), 2.0, atol=0.2)


def test_depth_stream_ignores_low_confidence_and_far_returns(tmp_path):
    _write_stream(tmp_path / "scan", [(0, 0, 0, 0, 0, 0, 1)], depth_mm=9000)  # beyond 5 m
    with pytest.raises(ValueError, match="no usable depth"):
        load_depth_stream(str(tmp_path / "scan"))


def _room_cloud(polygon, height=2.6, ceiling=True, density=900, seed=0):
    """Floor, walls (with a 0.9 m door gap skipped) and optionally ceiling for a rectilinear polygon."""
    rng = np.random.default_rng(seed)
    from matplotlib.path import Path as MplPath

    poly = np.array(polygon, float)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    xy = rng.uniform(lo, hi, (int(density * np.prod(hi - lo)), 2))
    xy = xy[MplPath(poly).contains_points(xy)]
    chunks = [np.column_stack([xy, np.zeros(len(xy))])]
    if ceiling:
        chunks.append(np.column_stack([xy, np.full(len(xy), height)]))
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n = int(density * np.linalg.norm(b - a) * height * 0.5)
        t = rng.uniform(0, 1, n)
        chunks.append(np.column_stack([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), rng.uniform(0, height, n)]))
    points = np.vstack(chunks)
    return PointCloud(points=points + rng.normal(0, 0.004, points.shape))


L_SHAPE = [(0, 0), (5, 0), (5, 2), (2.5, 2), (2.5, 4), (0, 4)]


def test_outline_recovers_an_l_shaped_room():
    cloud = _room_cloud(L_SHAPE)
    segments, polygon = fit_outline(cloud.points, floor_z=0.0)
    assert len(segments) == 6
    from shapely.geometry import Polygon

    assert Polygon(polygon).area == pytest.approx(Polygon(L_SHAPE).area, rel=0.03)
    assert sorted(round(s.length, 1) for s in segments) == [2.0, 2.0, 2.5, 2.5, 4.0, 5.0]


def test_fit_walls_prefers_the_outline_for_an_l_shape():
    segments, polygon = fit_walls(_room_cloud(L_SHAPE), reference_angle=0.0)
    assert len(segments) == 6


def test_fit_walls_keeps_precise_lines_for_a_clean_box():
    segments, _ = fit_walls(_room_cloud([(0, 0), (4, 0), (4, 3), (0, 3)]), reference_angle=0.0)
    assert sorted(round(s.length, 2) for s in segments) == pytest.approx([3.0, 3.0, 4.0, 4.0], abs=0.03)


def test_missing_ceiling_is_detected():
    assert is_ceiling_observed(_room_cloud([(0, 0), (4, 0), (4, 3), (0, 3)], ceiling=True))
    assert not is_ceiling_observed(_room_cloud([(0, 0), (4, 0), (4, 3), (0, 3)], height=1.6, ceiling=False))
    assert not is_ceiling_observed(_room_cloud([(0, 0), (4, 0), (4, 3), (0, 3)], height=2.6, ceiling=False))


def test_unobserved_ceiling_gets_an_upward_interval(tmp_path):
    """A lower bound must not claim a symmetric +/- 1.5 cm interval."""
    from src.calibration.confidence import calibrate_measurements
    from src.models import Measurement, PropertyReport, Room, Wall
    from datetime import datetime, timezone

    m = Measurement(value=1.8, confidence_low=1.8, confidence_high=1.8, unit="m")
    wall = Wall(id="r/wall-0", start=(0, 0), end=(1, 0), length=m, height=m, surface_id="r/wall-0")
    room = Room(id="r", name="R", walls=[wall], openings=[], ceiling_height=m, ceiling_observed=False,
                floor_area=Measurement(value=4.0, confidence_low=4.0, confidence_high=4.0, unit="m2"),
                floor_polygon=[(0, 0), (2, 0), (2, 2), (0, 2)])
    report = PropertyReport(capture_id="x", capture_tier="lidar", capture_timestamp=datetime.now(timezone.utc),
                            device="d", rooms=[room], adjacencies=[], damage_regions=[], concealed_damage_flags=[],
                            scope_line_items=[], total_floor_area=room.floor_area, room_count=1,
                            processing_time_seconds=0.0, pipeline_version="t")
    ceiling = calibrate_measurements(report, "lidar").rooms[0].ceiling_height
    assert ceiling.confidence_low == 1.8 and ceiling.confidence_high > 2.5
