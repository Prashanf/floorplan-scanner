"""Door and window detection as gaps in wall point density."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage

from src.geometry.ceiling import find_floor_and_ceiling
from src.geometry.wall_fitting import WallSegment
from src.room_ir import PointCloud

PLANE_DISTANCE = 0.15  # meters: points this close to the wall plane belong to the wall
BIN_WIDTH = 0.05  # meters along the wall
MIN_OPENING_WIDTH = 0.4
MIN_OPENING_HEIGHT = 0.4  # a column is open if it has an empty vertical run at least this tall
DOOR_MAX_START = 0.2  # gap starting within this of the floor is a door
WINDOW_MIN_START = 0.7  # gap starting above this is a window
SIMILAR_GAP = 0.15  # columns in one opening agree on gap start/end within this


@dataclass
class OpeningDetection:
    """A detected opening on one wall (meters)."""

    wall_index: int
    type: Literal["door", "window"]
    position_along_wall: float  # distance from wall start to the opening's near edge
    width: float
    height: float
    confidence: float


def _largest_vertical_gap(v: np.ndarray, top: float) -> tuple[float, float]:
    """Largest empty interval in [0, top] among sorted heights v. Returns (start, end)."""
    edges = np.concatenate(([0.0], np.clip(v, 0.0, top), [top]))
    gaps = np.diff(edges)
    k = int(np.argmax(gaps))
    return float(edges[k]), float(edges[k + 1])


def _detect_on_wall(
    wall_index: int, wall: WallSegment, points: np.ndarray, floor_z: float, height: float,
    min_gap_height: float,
) -> list[OpeningDetection]:
    start = np.array(wall.start)
    tangent = (np.array(wall.end) - start) / wall.length
    normal = np.array([-tangent[1], tangent[0]])

    rel = points[:, :2] - start
    u = rel @ tangent
    d = rel @ normal
    v = points[:, 2] - floor_z
    near = (np.abs(d) < PLANE_DISTANCE) & (u >= 0.0) & (u <= wall.length) & (v >= -0.05) & (v <= height + 0.05)
    u, v = u[near], v[near]

    n_bins = int(np.ceil(wall.length / BIN_WIDTH))
    if n_bins == 0:
        return []
    bin_index = np.minimum((u / BIN_WIDTH).astype(int), n_bins - 1)
    order = np.argsort(bin_index, kind="stable")
    bin_index, u, v = bin_index[order], u[order], v[order]
    bounds = np.searchsorted(bin_index, np.arange(n_bins + 1))

    # Per column: the biggest empty vertical run. An opening is a stretch of columns where it is tall.
    gap_start = np.zeros(n_bins)
    gap_end = np.zeros(n_bins)
    for b in range(n_bins):
        heights = np.sort(v[bounds[b]:bounds[b + 1]])
        gap_start[b], gap_end[b] = _largest_vertical_gap(heights, height)
    is_open = (gap_end - gap_start) >= min_gap_height

    closed = ndimage.binary_closing(is_open, structure=np.ones(3, dtype=bool))  # bridge 1-2 bin holes
    labels, n_runs = ndimage.label(closed)

    detections: list[OpeningDetection] = []
    for run in range(1, n_runs + 1):
        cols = np.flatnonzero(labels == run)
        if (cols[-1] - cols[0] + 1) * BIN_WIDTH < MIN_OPENING_WIDTH:
            continue
        open_cols = cols[is_open[cols]]
        if open_cols.size == 0:
            continue
        g0, g1 = float(np.median(gap_start[open_cols])), float(np.median(gap_end[open_cols]))
        consistent = np.mean((np.abs(gap_start[open_cols] - g0) < SIMILAR_GAP)
                             & (np.abs(gap_end[open_cols] - g1) < SIMILAR_GAP))

        # Refine the edges with the points in the middle of the gap's height band.
        lo_u, hi_u = cols[0] * BIN_WIDTH, (cols[-1] + 1) * BIN_WIDTH
        band = (v > g0 + 0.2 * (g1 - g0)) & (v < g1 - 0.2 * (g1 - g0))
        mid = (lo_u + hi_u) / 2
        left_pts, right_pts = u[band & (u < mid)], u[band & (u >= mid)]
        left = float(left_pts.max()) if left_pts.size else 0.0
        right = float(right_pts.min()) if right_pts.size else wall.length
        if abs((right - left) - (hi_u - lo_u)) > 2 * BIN_WIDTH:  # edge points were noise; keep bin extent
            left, right = lo_u, hi_u

        kind: Literal["door", "window"] = "door" if g0 <= DOOR_MAX_START else "window"
        confidence = 0.5 + 0.5 * float(consistent)
        if kind == "window" and g0 < WINDOW_MIN_START:
            confidence *= 0.8  # gap starts between door and window range: ambiguous
        detections.append(OpeningDetection(
            wall_index=wall_index, type=kind, position_along_wall=left, width=right - left,
            height=g1 - (0.0 if kind == "door" else g0), confidence=round(confidence, 3)))
    return detections


def detect_openings(
    point_cloud: PointCloud, wall_segments: list[WallSegment], *, min_gap_height: float = MIN_OPENING_HEIGHT
) -> list[OpeningDetection]:
    """Detect doors and windows on each wall.

    Per wall: take points within 0.15 m of the wall plane, project to (u along
    wall, v vertical) and split into 0.05 m columns along u. A column is open
    when its largest empty vertical run is at least 0.4 m (this tolerates floor
    strip points and the lintel above a door, which a plain point-count test
    would count as wall). Runs of open columns wider than 0.4 m are openings:
    a gap starting within 0.2 m of the floor is a door, otherwise a window
    (starts between 0.2 and 0.7 m get reduced confidence). Sparse clouds (SfM) leave
    random empty stretches, so callers raise min_gap_height (e.g. 1.5 m, door-sized only).
    """
    levels = find_floor_and_ceiling(point_cloud.points[:, 2])
    detections: list[OpeningDetection] = []
    for index, wall in enumerate(wall_segments):
        detections.extend(_detect_on_wall(
            index, wall, point_cloud.points, levels.floor_z, levels.height, min_gap_height))
    return detections
