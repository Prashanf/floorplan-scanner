"""Floor area from the floor polygon."""

from __future__ import annotations

from shapely.geometry import Polygon


def compute_floor_area(polygon: list[tuple[float, float]]) -> float:
    """Return the area in m2 of a simple 2D polygon (Shapely Polygon.area)."""
    if len(polygon) < 3:
        return 0.0
    shape = Polygon(polygon)
    if not shape.is_valid:
        shape = shape.buffer(0)  # repair self-touching outlines instead of returning nonsense
    return float(shape.area)
