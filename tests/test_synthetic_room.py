"""Geometry pipeline on a synthetic 4 m x 3 m room (ceiling 2.5 m, 0.86 m door centered on the east wall)."""

import numpy as np
import pytest

from src.geometry.ceiling import detect_ceiling_height
from src.geometry.floor_area import compute_floor_area
from src.geometry.openings import detect_openings
from src.geometry.wall_fitting import fit_walls
from src.room_ir import PointCloud

LENGTH_X, LENGTH_Y, HEIGHT = 4.0, 3.0, 2.5
DOOR_WIDTH, DOOR_HEIGHT = 0.86, 2.1
N_POINTS = 50_000
NOISE = 0.005


def make_room_cloud(seed: int = 0, yaw_deg: float = 0.0, offset=(0.0, 0.0, 0.0)) -> PointCloud:
    """Uniform samples on floor, ceiling and four walls; the east wall (x = 4) has the door cut out."""
    rng = np.random.default_rng(seed)
    # (area, sampler) per surface; points are allocated proportionally to area.
    surfaces = [
        (LENGTH_X * LENGTH_Y, lambda r: (r[0] * LENGTH_X, r[1] * LENGTH_Y, 0.0 * r[0])),  # floor
        (LENGTH_X * LENGTH_Y, lambda r: (r[0] * LENGTH_X, r[1] * LENGTH_Y, HEIGHT + 0.0 * r[0])),  # ceiling
        (LENGTH_X * HEIGHT, lambda r: (r[0] * LENGTH_X, 0.0 * r[0], r[1] * HEIGHT)),  # south
        (LENGTH_X * HEIGHT, lambda r: (r[0] * LENGTH_X, LENGTH_Y + 0.0 * r[0], r[1] * HEIGHT)),  # north
        (LENGTH_Y * HEIGHT, lambda r: (0.0 * r[0], r[0] * LENGTH_Y, r[1] * HEIGHT)),  # west
        (LENGTH_Y * HEIGHT, lambda r: (LENGTH_X + 0.0 * r[0], r[0] * LENGTH_Y, r[1] * HEIGHT)),  # east
    ]
    total_area = sum(area for area, _ in surfaces)
    chunks = []
    for i, (area, sampler) in enumerate(surfaces):
        n = int(N_POINTS * area / total_area)
        x, y, z = sampler(rng.random((2, n)))
        pts = np.column_stack([x, y, z])
        if i == 5:  # east wall: remove the door opening, floor to 2.1 m, centered on y = 1.5
            in_door = (np.abs(pts[:, 1] - LENGTH_Y / 2) < DOOR_WIDTH / 2) & (pts[:, 2] < DOOR_HEIGHT)
            pts = pts[~in_door]
        chunks.append(pts)
    points = np.vstack(chunks)
    points += rng.normal(0.0, NOISE, points.shape)

    yaw = np.deg2rad(yaw_deg)
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return PointCloud(points=points @ rot.T + np.array(offset))


@pytest.fixture(scope="module", params=[
    pytest.param(dict(yaw_deg=0.0, offset=(0.0, 0.0, 0.0)), id="axis-aligned"),
    pytest.param(dict(yaw_deg=33.0, offset=(5.0, -2.0, 0.3)), id="rotated-and-shifted"),
])
def result(request):
    cloud = make_room_cloud(**request.param)
    walls, polygon = fit_walls(cloud)
    ceiling, confidence = detect_ceiling_height(cloud)
    openings = detect_openings(cloud, walls)
    return cloud, walls, polygon, ceiling, confidence, openings


def test_four_walls_with_correct_lengths(result):
    _, walls, polygon, *_ = result
    assert len(walls) == 4
    assert len(polygon) == 4
    lengths = sorted(w.length for w in walls)
    expected = sorted([LENGTH_X, LENGTH_X, LENGTH_Y, LENGTH_Y])
    assert lengths == pytest.approx(expected, abs=0.05)


def test_walls_are_closed_and_counterclockwise(result):
    _, walls, polygon, *_ = result
    for i, wall in enumerate(walls):
        assert wall.start == pytest.approx(walls[i - 1].end)
    x, y = np.array(polygon).T
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0


def test_ceiling_height(result):
    *_, ceiling, confidence, _ = result
    assert ceiling == pytest.approx(HEIGHT, abs=0.05)
    assert 0.0 < confidence <= 1.0


def test_one_door_detected(result):
    _, walls, _, _, _, openings = result
    assert len(openings) == 1
    door = openings[0]
    assert door.type == "door"
    assert door.width == pytest.approx(DOOR_WIDTH, abs=0.1)
    assert door.height == pytest.approx(DOOR_HEIGHT, abs=0.1)
    # the door is on a 3 m wall, centered
    wall = walls[door.wall_index]
    assert wall.length == pytest.approx(LENGTH_Y, abs=0.05)
    assert door.position_along_wall + door.width / 2 == pytest.approx(LENGTH_Y / 2, abs=0.1)
    assert 0.5 <= door.confidence <= 1.0


def test_floor_area(result):
    _, _, polygon, *_ = result
    assert compute_floor_area(polygon) == pytest.approx(LENGTH_X * LENGTH_Y, abs=0.5)


def test_repeatable():
    """Same input twice gives the same plan (RANSAC is seeded)."""
    cloud = make_room_cloud(seed=3)
    a, _ = fit_walls(cloud)
    b, _ = fit_walls(cloud)
    assert [w.length for w in a] == [w.length for w in b]


def test_floor_area_degenerate_polygon():
    assert compute_floor_area([]) == 0.0
    assert compute_floor_area([(0, 0), (1, 1)]) == 0.0
