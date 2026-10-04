"""Drift ablation: with accumulated pose drift injected, correction must pull the stitched
footprint back toward ground truth, and --no-drift-correction must leave it drifted."""

import numpy as np
import pytest

from create_test_ply import ground_truth, make_apartment, write_ply
from src.pipeline import run_pipeline

DRIFT = (0.0, 0.15)  # metres per space walked
TRUTH_ORIGIN = {r["room_id"]: np.array(r["origin_xy"]) for r in ground_truth()["rooms"]}


def _origin_errors(report) -> list[float]:
    """Distance from each room's (south-west) polygon corner to its true position."""
    errors = []
    for room in report.rooms:
        corner = np.array(room.floor_polygon).min(axis=0)
        errors.append(float(np.linalg.norm(corner - TRUTH_ORIGIN[room.id])))
    return errors


@pytest.fixture(scope="module")
def drifted_capture(tmp_path_factory):
    capture = tmp_path_factory.mktemp("drifted")
    write_ply(capture / "apartment.ply", make_apartment(seed=1, density=800.0, drift=DRIFT))
    return capture


def test_drift_correction_reduces_footprint_error(drifted_capture, tmp_path):
    on = run_pipeline(str(drifted_capture), "lidar", str(tmp_path / "on"), drift_correction=True, render=False)
    off = run_pipeline(str(drifted_capture), "lidar", str(tmp_path / "off"), drift_correction=False, render=False)
    err_on, err_off = max(_origin_errors(on)), max(_origin_errors(off))
    assert err_off > 0.25  # uncorrected plan shows the injected 2 x 0.15 m drift
    assert err_on < err_off * 0.5
    assert err_on < 0.15


def test_no_drift_input_is_left_alone(tmp_path):
    """A wall-thickness gap between rooms is not drift: correction must not collapse it."""
    capture = tmp_path / "clean"
    capture.mkdir()
    write_ply(capture / "apartment.ply", make_apartment(seed=1, density=800.0))
    report = run_pipeline(str(capture), "lidar", str(tmp_path / "out"), drift_correction=True, render=False)
    assert max(_origin_errors(report)) < 0.05
