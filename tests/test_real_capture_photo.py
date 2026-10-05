"""Test suite for real photo capture pipeline execution.

Tests that real photo capture folders (room1, room2, room3, room4)
are ingested, processed through the photo tier pipeline, and produce:
1. report.json conforming to PropertyReport schema
2. floor_plan.png (rendered architectural drawing / status plan)
"""

from pathlib import Path
import json
import pytest

from src.models import PropertyReport
from src.tiers.preprocessing import validate_capture_dir
from src.pipeline import run_pipeline


@pytest.fixture(scope="module")
def real_capture_dir():
    """Path to the extracted real capture root directory."""
    p = Path(__file__).resolve().parents[1] / "benchmark" / "captures" / "photos" / "real_capture"
    if not p.is_dir() or not (p / "room1").is_dir():
        pytest.skip(f"Real capture directory not found at {p}")
    return p


def test_real_capture_directory_structure(real_capture_dir):
    """Verify that the 4 real room folders exist and pass ingestion validation."""
    expected_rooms = ["room1", "room2", "room3", "room4"]
    for r in expected_rooms:
        room_path = real_capture_dir / r
        assert room_path.is_dir(), f"Expected room folder {r} in {real_capture_dir}"
        images = list(room_path.glob("*.jpg")) + list(room_path.glob("*.png")) + list(room_path.glob("*.jpeg"))
        assert len(images) >= 2, f"Room {r} should have at least 2 images (found {len(images)})"

    # Ingestion validation
    warnings = validate_capture_dir(str(real_capture_dir), "photo")
    assert isinstance(warnings, list)


def test_real_capture_pipeline_generates_artifacts(real_capture_dir, tmp_path):
    """Run pipeline end-to-end and assert floor_plan.png and report.json are generated."""
    out_dir = tmp_path / "real_photo_output"
    report = run_pipeline(
        capture_dir=str(real_capture_dir),
        tier="photo",
        output_dir=str(out_dir),
        render=True,
    )

    # 1. Output files must exist
    plan_path = out_dir / "floor_plan.png"
    report_path = out_dir / "report.json"
    assert plan_path.is_file(), f"floor_plan.png missing at {plan_path}"
    assert report_path.is_file(), f"report.json missing at {report_path}"
    assert plan_path.stat().st_size > 500, "floor_plan.png should not be empty"

    # 2. Valid PropertyReport schema compliance
    raw_json = json.loads(report_path.read_text())
    validated = PropertyReport.model_validate(raw_json)
    assert validated.capture_tier == "photo"
    assert validated.processing_time_seconds > 0
    assert isinstance(validated.rooms, list)
    assert isinstance(validated.warnings, list)
