"""Room outline from an occupancy grid: the fallback for real scans.

Line fitting (wall_fitting.py) assumes every wall of a room is seen as one long line. Real
rooms have furniture, doorways, alcoves and unscanned stretches, so their walls come out as
a dozen fragments. Here the room is the area the scanner actually saw as floor, closed up
over small holes, and its outline is simplified to a rectilinear polygon (axis-aligned
walls, as the LiDAR tier rotates walls onto the axes). Each wall is then moved to where the
wall points really are, so lengths are precise rather than grid-quantized.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from src.geometry.wall_fitting import WallSegment
from src import config as cfg

CELL = cfg.OUTLINE_CELL
FLOOR_BAND = 0.12  # points this close to the floor level mark observed floor
WALL_BAND = cfg.WALL_HEIGHT_BAND
CLOSE_RADIUS = cfg.OUTLINE_CLOSE_RADIUS
SIMPLIFY = cfg.OUTLINE_SIMPLIFY
MIN_EDGE = cfg.OUTLINE_MIN_EDGE
REFINE_BAND = 0.12  # wall points within this of an edge pull it to their median
MIN_REFINE_POINTS = 30
MIN_AREA = 0.5  # m2


def _runs(vertices: np.ndarray) -> tuple[list[str], list[list[float]]]:
    """Group consecutive polygon edges by orientation; each run is one axis-aligned wall.

    Returns orientations ('H' or 'V') and, per run, [offset, lo, hi] where offset is the
    run's y (H) or x (V), weighted by edge length, and lo/hi its extent along the wall.
    """
    n = len(vertices)
    delta = np.roll(vertices, -1, axis=0) - vertices
    orient = ["H" if abs(d[0]) >= abs(d[1]) else "V" for d in delta]
    start = next((i for i in range(n) if orient[i] != orient[i - 1]), None)
    if start is None:
        return [], []
    kinds: list[str] = []
    runs: list[list[float]] = []
    weights: list[float] = []
    for k in range(n):
        i = (start + k) % n
        a, b = vertices[i], vertices[(i + 1) % n]
        horizontal = orient[i] == "H"
        offset = (a[1] + b[1]) / 2 if horizontal else (a[0] + b[0]) / 2
        lo, hi = sorted((a[0], b[0]) if horizontal else (a[1], b[1]))
        length = max(hi - lo, 1e-6)
        if kinds and kinds[-1] == orient[i]:
            run, w = runs[-1], weights[-1]
            run[0] = (run[0] * w + offset * length) / (w + length)
            run[1], run[2] = min(run[1], lo), max(run[2], hi)
            weights[-1] = w + length
        else:
            kinds.append(orient[i])
            runs.append([offset, lo, hi])
            weights.append(length)
    return kinds, runs


def _vertices(kinds: list[str], runs: list[list[float]]) -> np.ndarray:
    """Corner points where consecutive runs (alternating H and V) cross."""
    n = len(runs)
    corners = []
    for i in range(n):
        a, b = runs[i], runs[(i + 1) % n]
        if kinds[i] == "H":
            corners.append((b[0], a[0]))  # x from the V run, y from the H run
        else:
            corners.append((a[0], b[0]))
    return np.array(corners)


def _merge_short(kinds: list[str], runs: list[list[float]]) -> tuple[list[str], list[list[float]]]:
    """Remove runs shorter than MIN_EDGE (staircase noise); their neighbours become one run."""
    while len(runs) > 4:
        corners = _vertices(kinds, runs)
        lengths = [float(np.hypot(*(corners[i] - corners[i - 1]))) for i in range(len(runs))]
        k = int(np.argmin(lengths))
        if lengths[k] >= MIN_EDGE:
            break
        n = len(runs)
        before, after = (k - 1) % n, (k + 1) % n
        merged = [(runs[before][0] + runs[after][0]) / 2,
                  min(runs[before][1], runs[after][1]), max(runs[before][2], runs[after][2])]
        # rebuild cyclically: the merged run, then the runs after `after` up to just before `before`
        order = [(after + 1 + j) % n for j in range(n - 3)]
        kinds = [kinds[before]] + [kinds[i] for i in order]
        runs = [merged] + [runs[i] for i in order]
    return kinds, runs


def fit_outline(points: np.ndarray, floor_z: float, angle: float = 0.0
                ) -> tuple[list[WallSegment], list[tuple[float, float]]]:
    """Rectilinear room outline from an N x 3 cloud with a known floor level.

    `angle` is the direction (radians) of the room's walls; the outline is built in a frame
    rotated by -angle so the walls are axis-aligned there, then rotated back.
    Raises ValueError when no usable room area can be found.
    """
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = points[:, :2] @ rotation  # row vectors: same as rotating each point by -angle
    height = points[:, 2] - floor_z
    floor_pts = xy[np.abs(height) <= FLOOR_BAND]
    wall_pts = xy[(height >= WALL_BAND[0]) & (height <= WALL_BAND[1])]
    if len(floor_pts) < 50 and len(wall_pts) < 50:
        raise ValueError("not enough floor or wall points for a room outline")

    seen = np.vstack([floor_pts, wall_pts]) if len(wall_pts) else floor_pts
    radius = int(round(CLOSE_RADIUS / CELL))
    pad = radius + 3  # closing grows then shrinks the mask; the border must be out of its reach
    origin = seen.min(axis=0) - pad * CELL
    shape = tuple(int(n) for n in np.ceil((seen.max(axis=0) - origin) / CELL) + pad + 1)

    def cells(p: np.ndarray) -> np.ndarray:
        grid = np.zeros(shape, dtype=bool)
        idx = np.floor((p - origin) / CELL).astype(int)
        grid[idx[:, 0], idx[:, 1]] = True
        return grid

    mask = cells(floor_pts) | cells(wall_pts)
    mask = ndimage.binary_closing(mask, structure=ndimage.generate_binary_structure(2, 1), iterations=radius)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=1)
    labels, count = ndimage.label(mask)
    if count == 0:
        raise ValueError("no room area found")
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    mask = labels == (int(np.argmax(sizes)) + 1)
    if mask.sum() * CELL * CELL < MIN_AREA:
        raise ValueError("room area too small")

    # cv2 wants image[row, col] = grid[ix, iy] transposed
    contours, _ = cv2.findContours(mask.T.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, SIMPLIFY / CELL, True)[:, 0, :].astype(float)
    if len(approx) < 4:
        raise ValueError("outline has fewer than 4 corners")
    vertices = origin + (approx + 0.5) * CELL

    kinds, runs = _runs(vertices)
    if len(runs) < 4:
        raise ValueError("outline is not rectilinear enough")
    if len(runs) % 2:  # a closed axis-aligned outline alternates H and V, so the count is even
        raise ValueError("outline has inconsistent orientations")
    kinds, runs = _merge_short(kinds, runs)

    # Move each wall to where the wall points really are (sub-cell precision).
    if len(wall_pts) >= MIN_REFINE_POINTS:
        for kind, run in zip(kinds, runs):
            offset, lo, hi = run
            along, across = (wall_pts[:, 0], wall_pts[:, 1]) if kind == "H" else (wall_pts[:, 1], wall_pts[:, 0])
            near = (np.abs(across - offset) <= REFINE_BAND) & (along >= lo) & (along <= hi)
            if near.sum() >= MIN_REFINE_POINTS:
                run[0] = float(np.median(across[near]))

    corners = _vertices(kinds, runs)
    x, y = corners.T
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if area < 0:  # counterclockwise
        corners = corners[::-1]
        area = -area
    if area < MIN_AREA:
        raise ValueError("outline area too small")
    corners = corners @ rotation.T  # back to the input frame

    n = len(corners)
    segments = []
    for i in range(n):
        start, end = corners[i], corners[(i + 1) % n]
        delta = end - start
        length = float(np.hypot(*delta))
        if length < 1e-6:
            continue
        segments.append(WallSegment(
            start=(float(start[0]), float(start[1])), end=(float(end[0]), float(end[1])), length=length,
            direction=float(np.arctan2(delta[1], delta[0])), inlier_count=int(len(wall_pts))))
    polygon = [(float(c[0]), float(c[1])) for c in corners]
    return segments, polygon
