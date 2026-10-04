"""The output contract: sample report instantiates, exports valid JSON, schema is documented."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel, ValidationError

from src import models
from src.models import (
    Adjacency,
    ConcealedDamageFlag,
    DamageRegion,
    Measurement,
    Opening,
    PropertyReport,
    Room,
    ScopeLineItem,
    Wall,
)


def m(value: float, half_width: float = 0.01, unit: str = "m") -> Measurement:
    return Measurement(value=value, confidence_low=value - half_width,
                       confidence_high=value + half_width, unit=unit)


def make_room(room_id: str, x0: float) -> Room:
    corners = [(x0, 0.0), (x0 + 4.0, 0.0), (x0 + 4.0, 3.0), (x0, 3.0)]
    walls = [
        Wall(id=f"{room_id}/wall-{i}", start=corners[i], end=corners[(i + 1) % 4],
             length=m(4.0 if i % 2 == 0 else 3.0), height=m(2.5), surface_id=f"{room_id}/wall-{i}")
        for i in range(4)
    ]
    door = Opening(id=f"{room_id}/door-0", type="door", wall_id=f"{room_id}/wall-1",
                   position_along_wall=m(1.07), width=m(0.86, 0.02), height=m(2.1, 0.02),
                   confidence=0.9)
    return Room(id=room_id, name=room_id.title(), walls=walls, openings=[door],
                ceiling_height=m(2.5), floor_area=m(12.0, 0.1, "m2"), floor_polygon=corners)


def make_report() -> PropertyReport:
    damage = DamageRegion(
        id="damage-0", surface_id="room-1/wall-2", damage_class="water_stain",
        extent_width=m(0.3, 0.05), extent_height=m(0.2, 0.05), area=m(0.06, 0.02, "m2"),
        location_on_surface=(1.2, 0.1), confidence=0.7, source_image="room-1/IMG_0001.jpg")
    flag = ConcealedDamageFlag(
        id="flag-0", surface_id="room-1/wall-2", rule_id="RULE_WATER_01",
        rule_description="Water damage near floor junction", evidence=["damage-0"],
        confidence=0.53, recommended_action="Investigate possible concealed water damage.")
    item = ScopeLineItem(
        id="scope-0", surface_id="room-1/wall-2", damage_region_id="damage-0",
        description="Investigate moisture source. Remediate, dry, and repaint. Area: 0.06 m²",
        estimated_area=m(0.06, 0.02, "m2"), priority="high")
    return PropertyReport(
        capture_id="test-001", capture_tier="lidar",
        capture_timestamp=datetime(2026, 10, 4, 11, 30, tzinfo=timezone.utc), device="iPhone 15 Pro",
        rooms=[make_room("room-1", 0.0), make_room("room-2", 4.0)],
        adjacencies=[Adjacency(room_a_id="room-1", room_b_id="room-2", shared_opening_id="room-1/door-0")],
        damage_regions=[damage], concealed_damage_flags=[flag], scope_line_items=[item],
        total_floor_area=m(24.0, 0.2, "m2"), room_count=2, processing_time_seconds=12.3,
        pipeline_version="0.1.0")


def test_report_exports_valid_json() -> None:
    report = make_report()
    payload = report.model_dump_json(indent=2)
    data = json.loads(payload)
    assert data["room_count"] == 2
    assert data["rooms"][0]["floor_polygon"][0] == [0.0, 0.0]
    assert PropertyReport.model_validate_json(payload) == report


def test_empty_lists_are_present_not_missing() -> None:
    report = make_report().model_copy(update={
        "damage_regions": [], "concealed_damage_flags": [], "scope_line_items": []})
    data = report.model_dump(mode="json")
    assert data["damage_regions"] == []
    assert data["concealed_damage_flags"] == []
    assert data["scope_line_items"] == []


def test_json_schema_lists_contract_fields() -> None:
    schema = PropertyReport.generate_json_schema()
    json.dumps(schema)  # serializable
    for key in ["capture_id", "capture_tier", "rooms", "adjacencies", "damage_regions",
                "concealed_damage_flags", "scope_line_items", "total_floor_area",
                "room_count", "processing_time_seconds", "pipeline_version"]:
        assert key in schema["properties"]


def test_every_field_has_description_and_examples() -> None:
    classes = [c for c in vars(models).values()
               if isinstance(c, type) and issubclass(c, BaseModel) and c is not BaseModel
               and c.__module__ == models.__name__]
    assert len(classes) == 9
    for cls in classes:
        for name, field in cls.model_fields.items():
            assert field.description, f"{cls.__name__}.{name} has no description"
            if field.annotation is not Measurement and not _is_model_field(field):
                assert field.examples, f"{cls.__name__}.{name} has no examples"


def _is_model_field(field) -> bool:
    """Fields typed as another model (or list of one) take examples from that model."""
    import typing
    ann = field.annotation
    args = typing.get_args(ann) or (ann,)
    return any(isinstance(a, type) and issubclass(a, BaseModel) for a in args)


def test_inverted_interval_rejected() -> None:
    with pytest.raises(ValidationError):
        Measurement(value=1.0, confidence_low=2.0, confidence_high=0.5)


def test_invalid_damage_class_rejected() -> None:
    with pytest.raises(ValidationError):
        DamageRegion(id="d", surface_id="s", damage_class="fire", extent_width=m(1), extent_height=m(1),
                     area=m(1, unit="m2"), location_on_surface=(0, 0), confidence=0.5, source_image="x")
