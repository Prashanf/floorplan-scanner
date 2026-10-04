"""Wall detection: floor/ceiling planes, wall-height slice, RANSAC lines, orthogonal snapping."""

from __future__ import annotations

from dataclasses import dataclass

from src.room_ir import PointCloud


@dataclass
class WallSegment:
    """A fitted wall line in the XY plane (meters)."""

    start: tuple[float, float]
    end: tuple[float, float]
    length: float
    direction: float  # radians
    inlier_count: int


def fit_walls(point_cloud: PointCloud) -> tuple[list[WallSegment], list[tuple[float, float]]]:
    """Fit wall segments and a closed counterclockwise floor polygon.

    Finds floor and ceiling, slices points 0.8-1.5 m above the floor, projects
    to 2D, fits lines with iterative RANSAC (0.03 m threshold, min 20 inliers),
    snaps lines to 90 degrees of the dominant direction, intersects adjacent
    lines for vertices, and returns (wall segments, polygon vertices).
    """
    raise NotImplementedError("Not yet implemented")
