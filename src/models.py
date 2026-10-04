"""Output contract: Pydantic v2 models for the property report JSON.

Every field carries a description and examples; they flow into the published
JSON Schema (``schema/output_schema.json``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src import __version__

CaptureTier = Literal["photo", "video", "lidar"]
DamageClass = Literal["crack", "water_stain", "mold", "hole", "peeling_paint"]


class Measurement(BaseModel):
    """A scalar measurement with a confidence interval."""

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"value": 4.02, "confidence_low": 4.0, "confidence_high": 4.04, "unit": "m"}
    ]})

    value: float = Field(
        ..., description="Best estimate of the measured quantity, in `unit`.", examples=[4.02]
    )
    confidence_low: float = Field(
        ..., description="Lower bound of the confidence interval, in `unit`.", examples=[4.0]
    )
    confidence_high: float = Field(
        ..., description="Upper bound of the confidence interval, in `unit`.", examples=[4.04]
    )
    unit: str = Field(
        "m", description="Unit of value and interval bounds: 'm' for lengths, 'm2' for areas.",
        examples=["m", "m2"],
    )

    @model_validator(mode="after")
    def _interval_ordered(self) -> "Measurement":
        if self.confidence_low > self.confidence_high:
            raise ValueError("confidence_low must be <= confidence_high")
        return self


class Wall(BaseModel):
    """One wall segment of a room, in the property's global 2D frame."""

    id: str = Field(..., description="Unique wall id within the report.", examples=["room-1/wall-0"])
    start: tuple[float, float] = Field(
        ..., description="Wall start point (x, y) in meters, global frame.", examples=[[0.0, 0.0]]
    )
    end: tuple[float, float] = Field(
        ..., description="Wall end point (x, y) in meters, global frame.", examples=[[4.0, 0.0]]
    )
    length: Measurement = Field(..., description="Wall length along the floor line.")
    height: Measurement = Field(..., description="Wall height, floor to ceiling.")
    surface_id: str = Field(
        ..., description="Surface id that damage regions and scope items refer to.",
        examples=["room-1/wall-0"],
    )


class Opening(BaseModel):
    """A door or window in a wall."""

    id: str = Field(..., description="Unique opening id within the report.", examples=["room-1/door-0"])
    type: Literal["door", "window"] = Field(
        ..., description="Opening type.", examples=["door", "window"]
    )
    wall_id: str = Field(..., description="Id of the wall containing the opening.", examples=["room-1/wall-1"])
    position_along_wall: Measurement = Field(
        ..., description="Distance from the wall start to the opening's near edge."
    )
    width: Measurement = Field(..., description="Opening width along the wall.")
    height: Measurement = Field(..., description="Opening height (vertical extent).")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence in [0, 1].", examples=[0.9]
    )


class Room(BaseModel):
    """A single room with walls, openings, ceiling height and floor area."""

    id: str = Field(..., description="Unique room id.", examples=["room-1"])
    name: str = Field(..., description="Human-readable room name.", examples=["Room 1"])
    walls: list[Wall] = Field(..., description="Walls ordered counterclockwise around the room.")
    openings: list[Opening] = Field(..., description="Doors and windows in the room's walls.")
    ceiling_height: Measurement = Field(..., description="Floor-to-ceiling height.")
    floor_area: Measurement = Field(..., description="Floor area, unit 'm2'.")
    floor_polygon: list[tuple[float, float]] = Field(
        ..., description="Closed floor outline, counterclockwise (x, y) in meters, global frame.",
        examples=[[[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]],
    )


class Adjacency(BaseModel):
    """Two rooms connected through a shared opening."""

    room_a_id: str = Field(..., description="Id of the first room.", examples=["room-1"])
    room_b_id: str = Field(..., description="Id of the second room.", examples=["room-2"])
    shared_opening_id: str = Field(
        ..., description="Id of the opening that connects the two rooms.", examples=["room-1/door-0"]
    )


class DamageRegion(BaseModel):
    """A damaged region on a surface, with class and metric extent."""

    id: str = Field(..., description="Unique damage region id.", examples=["damage-0"])
    surface_id: str = Field(..., description="Surface the damage lies on.", examples=["room-1/wall-2"])
    damage_class: DamageClass = Field(
        ..., description="Damage class.", examples=["crack", "water_stain"]
    )
    extent_width: Measurement = Field(..., description="Horizontal extent on the surface.")
    extent_height: Measurement = Field(..., description="Vertical extent on the surface.")
    area: Measurement = Field(..., description="Damaged area on the surface, unit 'm2'.")
    location_on_surface: tuple[float, float] = Field(
        ..., description="Region center (u, v) in meters: along the surface from its start, and height above floor.",
        examples=[[1.2, 0.3]],
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence in [0, 1].", examples=[0.7]
    )
    source_image: str = Field(
        ..., description="Path of the image the detection came from.", examples=["room-1/IMG_0001.jpg"]
    )


class ConcealedDamageFlag(BaseModel):
    """A rule-based flag for damage that is likely hidden behind a surface."""

    id: str = Field(..., description="Unique flag id.", examples=["flag-0"])
    surface_id: str = Field(..., description="Surface the flag applies to.", examples=["room-1/wall-2"])
    rule_id: str = Field(..., description="Id of the rule that fired.", examples=["RULE_WATER_01"])
    rule_description: str = Field(
        ..., description="Human-readable description of the rule.",
        examples=["Water damage near floor junction"],
    )
    evidence: list[str] = Field(
        ..., description="Ids of the damage regions that triggered the rule.", examples=[["damage-0"]]
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Flag confidence in [0, 1].", examples=[0.6]
    )
    recommended_action: str = Field(
        ..., description="Action recommended to the adjuster.",
        examples=["Investigate possible concealed water damage behind surface. Check for plumbing leaks."],
    )


class ScopeLineItem(BaseModel):
    """A repair scope line item keyed to a surface."""

    id: str = Field(..., description="Unique scope line item id.", examples=["scope-0"])
    surface_id: str = Field(..., description="Surface the work applies to.", examples=["room-1/wall-2"])
    damage_region_id: str = Field(
        ..., description="Damage region (or flag) this item addresses.", examples=["damage-0"]
    )
    description: str = Field(
        ..., description="Work description.", examples=["Patch, fill, and repaint. Estimated area: 0.05 m²"]
    )
    estimated_area: Measurement = Field(..., description="Estimated work area, unit 'm2'.")
    priority: Literal["high", "medium", "low"] = Field(
        ..., description="Repair priority.", examples=["high"]
    )


class PropertyReport(BaseModel):
    """Full per-capture report: plan, damage, concealed flags and scope."""

    capture_id: str = Field(..., description="Unique capture id.", examples=["capture-2026-10-04-001"])
    capture_tier: CaptureTier = Field(
        ..., description="Input tier the report was produced from.", examples=["lidar"]
    )
    capture_timestamp: datetime = Field(
        ..., description="Capture time, ISO 8601.", examples=["2026-10-04T11:30:00Z"]
    )
    device: str = Field(..., description="Capture device.", examples=["iPhone 15 Pro"])
    rooms: list[Room] = Field(..., description="Rooms with dimensioned plans.")
    adjacencies: list[Adjacency] = Field(..., description="Room connectivity.")
    damage_regions: list[DamageRegion] = Field(..., description="Detected damage regions.")
    concealed_damage_flags: list[ConcealedDamageFlag] = Field(
        ..., description="Concealed-damage flags, each with the rule that fired."
    )
    scope_line_items: list[ScopeLineItem] = Field(..., description="Scope line items keyed to surfaces.")
    total_floor_area: Measurement = Field(
        ..., description="Whole-property floor area (sum over rooms), unit 'm2'."
    )
    room_count: int = Field(..., ge=0, description="Number of rooms in the report.", examples=[3])
    processing_time_seconds: float = Field(
        0.0, ge=0.0, description="Wall-clock pipeline time in seconds.", examples=[42.5]
    )
    pipeline_version: str = Field(
        __version__, description="Version of the pipeline that produced the report.", examples=["0.1.0"]
    )

    @classmethod
    def generate_json_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema for the full report (published output contract)."""
        return cls.model_json_schema()
