"""Confidence-interval calibration per tier."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src import config as cfg
from src.models import (
    DamageRegion,
    Measurement,
    Opening,
    PropertyReport,
    Room,
    ScopeLineItem,
    Wall,
)

log = logging.getLogger("floorplan.calibration")

BASE_UNCERTAINTY = cfg.BASE_UNCERTAINTY

TIER_MULTIPLIER = cfg.CONFIDENCE_MULTIPLIER

UNOBSERVED_CEILING_EXTRA = cfg.UNOBSERVED_CEILING_EXTRA

CALIBRATION_FILE = Path(__file__).resolve().parents[2] / "calibration_data.json"


def _interval(value: float, base: float, multiplier: float) -> tuple[float, float]:
    half = base * multiplier
    return (value - half, value + half)


def _recalibrate(m: Measurement, base: float, multiplier: float) -> Measurement:
    low, high = _interval(m.value, base, multiplier)
    return m.model_copy(update={"confidence_low": low, "confidence_high": high})


def _load_fitted_params(path: Path) -> dict | None:
    """Load isotonic-regression calibration params if the file exists."""
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("calibration_data.json unreadable, falling back to defaults: %s", exc)
        return None


def _fitted_interval(value: float, kind: str, tier: str, params: dict) -> tuple[float, float]:
    """Use fitted half-width from calibration_data.json.

    Expected format: {"lidar": {"wall_length": <half_width>, ...}, ...}
    """
    half = params.get(tier, {}).get(kind)
    if half is None:
        half = BASE_UNCERTAINTY[kind] * TIER_MULTIPLIER[tier]
    return (value - half, value + half)


def calibrate_measurements(report: PropertyReport, tier: str) -> PropertyReport:
    """Set confidence_low/high on every Measurement from per-type base
    uncertainty times a per-tier multiplier (or fitted parameters from
    calibration_data.json when present). Returns the updated report.
    """
    fitted = _load_fitted_params(CALIBRATION_FILE)
    mult = TIER_MULTIPLIER.get(tier, TIER_MULTIPLIER["photo"])

    def cal(m: Measurement, kind: str) -> Measurement:
        if fitted:
            low, high = _fitted_interval(m.value, kind, tier, fitted)
        else:
            low, high = _interval(m.value, BASE_UNCERTAINTY[kind], mult)
        return m.model_copy(update={"confidence_low": low, "confidence_high": high})

    new_rooms: list[Room] = []
    for room in report.rooms:
        new_walls = [
            w.model_copy(update={
                "length": cal(w.length, "wall_length"),
                "height": cal(w.height, "ceiling_height"),
            })
            for w in room.walls
        ]
        new_openings = [
            o.model_copy(update={
                "width": cal(o.width, "opening_width"),
                "height": cal(o.height, "opening_width"),
                "position_along_wall": cal(o.position_along_wall, "opening_width"),
            })
            for o in room.openings
        ]
        if room.ceiling_observed:
            ceiling = cal(room.ceiling_height, "ceiling_height")
        else:  # a lower bound: the interval opens upward instead of being symmetric
            ceiling = room.ceiling_height.model_copy(update={
                "confidence_low": room.ceiling_height.value,
                "confidence_high": room.ceiling_height.value + UNOBSERVED_CEILING_EXTRA})
        new_walls = [w.model_copy(update={"height": ceiling}) for w in new_walls]
        new_rooms.append(room.model_copy(update={
            "walls": new_walls,
            "openings": new_openings,
            "ceiling_height": ceiling,
            "floor_area": cal(room.floor_area, "floor_area"),
        }))

    new_damage = [
        d.model_copy(update={
            "extent_width": cal(d.extent_width, "damage_extent"),
            "extent_height": cal(d.extent_height, "damage_extent"),
            "area": cal(d.area, "damage_extent"),
        })
        for d in report.damage_regions
    ]

    new_scope = [
        s.model_copy(update={
            "estimated_area": cal(s.estimated_area, "damage_extent"),
        })
        for s in report.scope_line_items
    ]

    return report.model_copy(update={
        "rooms": new_rooms,
        "damage_regions": new_damage,
        "scope_line_items": new_scope,
        "total_floor_area": cal(report.total_floor_area, "floor_area"),
    })
