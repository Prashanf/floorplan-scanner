"""Rectangle fallback for rooms whose wall fit is noisy (many short wall segments).

Noisy SfM reconstructions can give 8 to 18 wall segments for what is a box-shaped room. When the minimum rotated
rectangle of the floor-level points explains most of the wall points, the room is replaced by that rectangle.
Not used for the LiDAR tier, whose fits are clean.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
from shapely.geometry import MultiPoint
from shapely.geometry.polygon import orient

from src import config as cfg
from src.geometry.ceiling import find_floor_and_ceiling
from src.geometry.wall_fitting import WallSegment
from src.room_ir import PointCloud

log = logging.getLogger("floorplan.geometry")

FLOOR_BAND = 0.15  # metres above the floor level counted as floor points
TRIM = 0.02  # share of the farthest points (stray SfM triangulations) left out of the rectangle


def _edge_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance of each 2D point to the segment a-b."""
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def fit_rectangle(
    point_cloud: PointCloud, n_walls: int, *, max_walls: int = cfg.RECT_MAX_WALLS,
    tolerance: float = cfg.RECT_EDGE_TOLERANCE, min_explained: float = cfg.RECT_MIN_EXPLAINED,
) -> Optional[tuple[list[WallSegment], list[tuple[float, float]]]]:
    """(wall segments, floor polygon) of a bounding rectangle, or None when it should not replace the fit.

    Applies only when the room has more than `max_walls` wall segments (`n_walls`). The rectangle is the minimum
    rotated rectangle of the floor-level points plus the wall-band points (the 2 % farthest from the centre are
    dropped); it is used when at least `min_explained` of the wall-band points lie within `tolerance` metres of
    one of its edges. The polygon is counter-clockwise like the line fit's.
    """
    if n_walls <= max_walls:
        return None
    z = point_cloud.points[:, 2]
    try:
        floor_z = find_floor_and_ceiling(z).floor_z
    except ValueError:
        return None
    lo, hi = cfg.WALL_HEIGHT_BAND
    wall_pts = point_cloud.points[(z >= floor_z + lo) & (z <= floor_z + hi), :2]
    floor_pts = point_cloud.points[np.abs(z - floor_z) <= FLOOR_BAND, :2]
    pts = np.vstack([wall_pts, floor_pts])
    if len(wall_pts) < 20 or len(pts) < 20:
        return None
    centre = np.median(pts, axis=0)
    keep = np.linalg.norm(pts - centre, axis=1) <= np.quantile(np.linalg.norm(pts - centre, axis=1), 1 - TRIM)
    rect = MultiPoint([tuple(p) for p in pts[keep]]).minimum_rotated_rectangle
    if rect.geom_type != "Polygon" or rect.area < 1e-3:
        return None
    corners = np.array(orient(rect, 1.0).exterior.coords)[:-1]  # counter-clockwise, 4 corners
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    dist = np.stack([_edge_distance(wall_pts, a, b) for a, b in edges])  # 4 x N
    nearest = dist.min(axis=0)
    explained = float((nearest <= tolerance).mean())
    if explained < min_explained:
        log.info("rectangle explains %.0f%% of the wall points (< %.0f%%); keeping the %d-wall fit",
                 100 * explained, 100 * min_explained, n_walls)
        return None
    owner = dist.argmin(axis=0)
    segments = []
    for i, (a, b) in enumerate(edges):
        length = float(np.linalg.norm(b - a))
        segments.append(WallSegment(
            start=(float(a[0]), float(a[1])), end=(float(b[0]), float(b[1])), length=length,
            direction=math.atan2(b[1] - a[1], b[0] - a[0]),
            inlier_count=int(((owner == i) & (nearest <= tolerance)).sum())))
    log.info("replaced a %d-wall fit by a %.2f x %.2f m rectangle (%.0f%% of wall points explained)",
             n_walls, segments[0].length, segments[1].length, 100 * explained)
    return segments, [(float(x), float(y)) for x, y in corners]
