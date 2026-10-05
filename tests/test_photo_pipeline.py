"""End-to-end pipeline tests for the photo tier.

Tests that a folder-by-folder multi-room photo capture runs through the pipeline,
produces floor_plan.png and report.json, and satisfies all PropertyReport schema requirements.
"""

from pathlib import Path
import json
import pytest

from src.pipeline import run_pipeline
from src.models import PropertyReport


def test_photo_tier_pipeline_produces_plan_and_report(tmp_path):
    """Test photo tier pipeline end-to-end on test_photos (or generated synthetic room)."""
    # Prefer existing test_photos directory if populated
    test_photos_dir = Path(__file__).resolve().parents[1] / "test_photos"
    if not (test_photos_dir / "room-1").is_dir():
        pytest.skip("test_photos/room-1 not found; run tests/create_test_photos.py first")

    out_dir = tmp_path / "photo_out"
    report = run_pipeline(
        capture_dir=str(test_photos_dir),
        tier="photo",
        output_dir=str(out_dir),
        render=True,
    )

    # 1. Output file artifacts
    plan_file = out_dir / "floor_plan.png"
    report_file = out_dir / "report.json"
    assert plan_file.is_file(), f"Expected floor_plan.png at {plan_file}"
    assert report_file.is_file(), f"Expected report.json at {report_file}"
    assert plan_file.stat().st_size > 1000, "floor_plan.png should not be empty"

    # 2. Schema compliance & deserialization
    data = json.loads(report_file.read_text())
    validated = PropertyReport.model_validate(data)
    assert validated.capture_tier == "photo"
    assert validated.room_count >= 1

    # 3. Floor area & room measurements
    assert validated.total_floor_area.value > 0
    assert validated.total_floor_area.unit == "m2"
    assert validated.total_floor_area.confidence_low <= validated.total_floor_area.confidence_high

    for room in validated.rooms:
        assert len(room.walls) >= 3, f"Room {room.id} should have at least 3 walls"
        assert room.floor_area.value > 0
        assert room.ceiling_height.value > 0
        for wall in room.walls:
            assert wall.length.value > 0
            assert wall.surface_id.startswith(room.id)

    # 4. Damage and scope arrays
    assert isinstance(validated.damage_regions, list)
    assert isinstance(validated.scope_line_items, list)
    assert isinstance(validated.concealed_damage_flags, list)
