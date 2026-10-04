"""Segmentation, pipeline, JSON and renderer on the synthetic 3-space apartment
(Room 1 | Hallway | Room 2, walls 0.12 m thick, 0.86 m doors)."""

import json

import numpy as np
import pytest

from create_test_ply import DOOR_WIDTH, ground_truth, make_apartment, write_ply
from src.models import PropertyReport
from src.room_ir import PointCloud
from src.tiers.room_segmentation import segment_rooms

TRUTH = {room["room_id"]: room for room in ground_truth()["rooms"]}


@pytest.fixture(scope="module")
def apartment_points() -> np.ndarray:
    points = make_apartment(seed=1, density=1200.0)
    keys = np.floor(points / 0.02).astype(int)  # same effect as the 2 cm voxel downsample
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def test_segmentation_splits_three_rooms(apartment_points):
    rooms = segment_rooms(PointCloud(points=apartment_points))
    assert len(rooms) == 3
    centers_x = [float(np.median(r.points[:, 0])) for r in rooms]
    assert centers_x == sorted(centers_x)  # left to right: room, hallway, room


def test_segmented_rooms_have_exactly_four_walls(apartment_points):
    from src.geometry.wall_fitting import fit_walls

    for room in segment_rooms(PointCloud(points=apartment_points)):
        walls, _ = fit_walls(room)
        assert len(walls) == 4


def test_single_box_is_one_room():
    from test_synthetic_room import make_room_cloud

    assert len(segment_rooms(make_room_cloud())) == 1


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory):
    pytest.importorskip("open3d", exc_type=ImportError)
    from src.pipeline import run_pipeline

    capture = tmp_path_factory.mktemp("capture")
    write_ply(capture / "apartment.ply", make_apartment(seed=2, density=1200.0))
    out = tmp_path_factory.mktemp("out")
    report = run_pipeline(str(capture), "lidar", str(out), drift_correction=True, render=True)
    return report, out


def test_pipeline_matches_ground_truth(pipeline_result):
    report, _ = pipeline_result
    assert report.room_count == 3
    for room, truth in zip(report.rooms, ground_truth()["rooms"]):
        assert room.floor_area.value == pytest.approx(truth["floor_area_m2"], abs=0.3)
        assert room.ceiling_height.value == pytest.approx(truth["ceiling_height_m"], abs=0.03)
        assert len(room.walls) == 4
        assert len(room.openings) == len(truth["openings"])
        for opening in room.openings:
            assert opening.type == "door"
            assert opening.width.value == pytest.approx(DOOR_WIDTH, abs=0.05)
    assert report.total_floor_area.value == pytest.approx(ground_truth()["total_floor_area_m2"], abs=0.6)


def test_pipeline_outputs_valid_json_and_plan(pipeline_result):
    report, out = pipeline_result
    data = json.loads((out / "report.json").read_text())
    PropertyReport.model_validate(data)
    for key in ("damage_regions", "concealed_damage_flags", "scope_line_items", "adjacencies"):
        assert key in data and isinstance(data[key], list)
    assert (out / "floor_plan.png").stat().st_size > 10_000
    for room in report.rooms:  # intervals are tier-scaled and contain the value
        assert room.ceiling_height.confidence_low < room.ceiling_height.value < room.ceiling_height.confidence_high
