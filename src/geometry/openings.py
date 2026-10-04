"""Door and window detection as gaps in wall point density."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.geometry.wall_fitting import WallSegment
from src.room_ir import PointCloud


@dataclass
class OpeningDetection:
    """A detected opening on one wall (meters)."""

    wall_index: int
    type: Literal["door", "window"]
    position_along_wall: float
    width: float
    height: float
    confidence: float


def detect_openings(point_cloud: PointCloud, wall_segments: list[WallSegment]) -> list[OpeningDetection]:
    """Detect doors and windows on each wall.

    Per wall: take points within 0.15 m of the wall plane, project to (u along
    wall, v vertical), bin along u (0.05 m), find gaps of fewer than 3 points
    per bin wider than 0.4 m. A gap starting within 0.2 m of the floor is a
    door; one starting above 0.7 m is a window.
    """
    raise NotImplementedError("Not yet implemented")
