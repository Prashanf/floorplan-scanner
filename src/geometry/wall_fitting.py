"""Wall detection: floor/ceiling planes, wall-height slice, RANSAC lines, orthogonal snapping."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.geometry.ceiling import find_floor_and_ceiling
from src.room_ir import PointCloud
from src import config as cfg

WALL_BAND = cfg.WALL_HEIGHT_BAND
INLIER_THRESHOLD = cfg.RANSAC_INLIER_THRESHOLD
MIN_INLIERS = cfg.RANSAC_MIN_INLIERS
RANSAC_ITERATIONS = cfg.RANSAC_ITERATIONS
RANSAC_SAMPLE = 5000  # hypotheses are scored on at most this many points
MAX_LINES = cfg.MAX_WALL_LINES
SNAP_TOLERANCE = np.deg2rad(cfg.SNAP_ANGLE_TOLERANCE_DEG)  # lines further than this from the grid stay unsnapped
MERGE_OFFSET = cfg.WALL_MERGE_OFFSET
MIN_WALL_LENGTH = cfg.MIN_WALL_LENGTH
COVERAGE_BIN = 0.1  # meters along a line
MIN_BIN_POINTS = 1
MIN_COVERAGE = cfg.MIN_WALL_COVERAGE
SEED = 0  # fixed: same room in, same plan out


@dataclass
class WallSegment:
    """A fitted wall line in the XY plane (meters)."""

    start: tuple[float, float]
    end: tuple[float, float]
    length: float
    direction: float  # radians
    inlier_count: int


@dataclass
class _Line:
    """A 2D line through `center` with direction `angle`, and the points supporting it."""

    center: np.ndarray
    angle: float
    points: np.ndarray = field(repr=False)

    @property
    def direction(self) -> np.ndarray:
        return np.array([np.cos(self.angle), np.sin(self.angle)])

    @property
    def normal(self) -> np.ndarray:
        d = self.direction
        return np.array([-d[1], d[0]])

    def extent(self) -> tuple[np.ndarray, np.ndarray]:
        """End points of the supporting points projected on the line."""
        t = (self.points - self.center) @ self.direction
        return self.center + t.min() * self.direction, self.center + t.max() * self.direction


def _total_least_squares(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = points.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((points - center).T))
    major = vecs[:, -1]
    return center, float(np.arctan2(major[1], major[0]) % np.pi)


def _ransac_line(points: np.ndarray, rng: np.random.Generator, threshold: float) -> _Line | None:
    """Best line by consensus, refit by total least squares on its inliers."""
    n = len(points)
    sample = points if n <= RANSAC_SAMPLE else points[rng.choice(n, RANSAC_SAMPLE, replace=False)]
    pairs = rng.integers(0, len(sample), size=(RANSAC_ITERATIONS, 2))
    p, q = sample[pairs[:, 0]], sample[pairs[:, 1]]
    d = q - p
    norm = np.hypot(d[:, 0], d[:, 1])
    ok = norm > 1e-6
    if not ok.any():
        return None
    p, d, norm = p[ok], d[ok], norm[ok]
    nx, ny = -d[:, 1] / norm, d[:, 0] / norm
    dist = np.abs((sample[None, :, 0] - p[:, None, 0]) * nx[:, None]
                  + (sample[None, :, 1] - p[:, None, 1]) * ny[:, None])
    best = int(np.argmax((dist < threshold).sum(axis=1)))
    center, angle = p[best], float(np.arctan2(d[best, 1], d[best, 0]) % np.pi)

    inliers = points
    for _ in range(2):  # refit, then re-collect inliers around the refit line
        normal = np.array([-np.sin(angle), np.cos(angle)])
        inliers = points[np.abs((points - center) @ normal) < threshold]
        if len(inliers) < 2:
            return None
        center, angle = _total_least_squares(inliers)
    return _Line(center=center, angle=angle, points=inliers)


def _coverage(line: _Line) -> float:
    """Share of 0.1 m bins along the line's extent that contain inliers.

    A real wall (even with a door in it) is mostly covered; stray points scattered
    along a line, such as a few jamb-edge points, are not.
    """
    t = (line.points - line.center) @ line.direction
    extent = float(t.max() - t.min())
    if extent < 1e-6:
        return 0.0
    n_bins = max(1, int(np.ceil(extent / COVERAGE_BIN)))
    counts = np.bincount(np.minimum(((t - t.min()) / COVERAGE_BIN).astype(int), n_bins - 1), minlength=n_bins)
    return float((counts >= MIN_BIN_POINTS).sum() / n_bins)


def _fit_lines(xy: np.ndarray, threshold: float = INLIER_THRESHOLD) -> list[_Line]:
    """Iterative RANSAC: fit a line, remove its inliers, repeat until support runs out."""
    rng = np.random.default_rng(SEED)
    lines: list[_Line] = []
    remaining = xy
    while len(lines) < MAX_LINES and len(remaining) >= MIN_INLIERS:
        line = _ransac_line(remaining, rng, threshold)
        if line is None or len(line.points) < MIN_INLIERS:
            break
        if _coverage(line) >= MIN_COVERAGE:
            lines.append(line)
        normal = line.normal
        remaining = remaining[np.abs((remaining - line.center) @ normal) >= threshold]
    return lines


def _angle_residual(angle: float, reference: float) -> float:
    """Signed distance of `angle` from the nearest multiple of 90 degrees off `reference`."""
    return float((angle - reference + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def _snap_and_merge(lines: list[_Line], merge_offset: float = MERGE_OFFSET,
                    reference_angle: float | None = None) -> list[_Line]:
    """Snap lines to a 90 degree grid; merge duplicates of one wall.

    The grid follows the longest-supported line unless reference_angle (radians) is given,
    e.g. 0 for a cloud that was already rotated onto the axes.
    """
    dominant = max(lines, key=lambda ln: len(ln.points)).angle if reference_angle is None else reference_angle
    snapped: list[_Line] = []
    for ln in lines:
        residual = _angle_residual(ln.angle, dominant)
        if abs(residual) <= SNAP_TOLERANCE:
            ln = _Line(center=ln.center, angle=(ln.angle - residual) % np.pi, points=ln.points)
        snapped.append(ln)

    merged: list[_Line] = []
    for ln in sorted(snapped, key=lambda x: -len(x.points)):
        for other in merged:
            diff = abs(ln.angle - other.angle) % np.pi
            parallel = min(diff, np.pi - diff) < 1e-6  # only snapped lines match exactly
            if parallel and abs((ln.center - other.center) @ other.normal) < merge_offset:
                pts = np.vstack([other.points, ln.points])
                other.points = pts
                other.center = pts.mean(axis=0)
                break
        else:
            merged.append(ln)
    return [ln for ln in merged if np.linalg.norm(np.subtract(*ln.extent())) >= MIN_WALL_LENGTH]


def _intersect(a: _Line, b: _Line) -> np.ndarray | None:
    da, db = a.direction, b.direction
    det = da[0] * db[1] - da[1] * db[0]
    if abs(det) < 1e-6:
        return None
    t = ((b.center[0] - a.center[0]) * db[1] - (b.center[1] - a.center[1]) * db[0]) / det
    return a.center + t * da


def _corner(a: _Line, b: _Line) -> np.ndarray:
    """Corner between consecutive walls; falls back to the closest end points if parallel."""
    point = _intersect(a, b)
    if point is not None:
        return point
    ends_a, ends_b = a.extent(), b.extent()
    pa, pb = min(((x, y) for x in ends_a for y in ends_b), key=lambda xy: np.linalg.norm(xy[0] - xy[1]))
    return (pa + pb) / 2


MAX_LINE_WALLS = cfg.MAX_LINE_WALLS
MAX_AREA_DISAGREEMENT = cfg.MAX_AREA_DISAGREEMENT


def fit_walls(
    point_cloud: PointCloud, *, inlier_threshold: float = INLIER_THRESHOLD, method: str = "auto",
    reference_angle: float | None = None,
) -> tuple[list[WallSegment], list[tuple[float, float]]]:
    """Fit wall segments and a closed counterclockwise floor polygon.

    method="lines" fits RANSAC lines (precise, but needs every wall seen as one long line),
    "outline" traces the scanned floor area (robust on real rooms, see room_outline.py), and
    "auto" (default) uses the line fit when it gives a clean box that agrees with the outline
    and the outline otherwise. Raises ValueError when neither works.
    reference_angle (radians) fixes the wall grid, e.g. 0.0 when the cloud is already
    axis-aligned; by default the grid follows the best-supported wall.
    """
    from shapely.geometry import Polygon

    from src.geometry.room_outline import fit_outline

    if method == "lines":
        return _fit_walls_lines(point_cloud, inlier_threshold=inlier_threshold, reference_angle=reference_angle)
    floor_z = find_floor_and_ceiling(point_cloud.points[:, 2]).floor_z
    try:
        lines = _fit_walls_lines(point_cloud, inlier_threshold=inlier_threshold, reference_angle=reference_angle)
    except ValueError:
        lines = None
    if reference_angle is not None:
        angle = reference_angle
    else:
        angle = max(lines[0], key=lambda seg: seg.inlier_count).direction if lines else 0.0
    try:
        outline = fit_outline(point_cloud.points, floor_z, angle)
    except ValueError:
        outline = None
    if method == "outline":
        if outline is None:
            raise ValueError("no room outline found")
        return outline
    if lines is None:
        if outline is None:
            raise ValueError("neither wall lines nor a room outline found")
        return outline
    if outline is None:
        return lines
    line_poly = Polygon(lines[1])
    outline_area = Polygon(outline[1]).area
    clean = (len(lines[0]) <= MAX_LINE_WALLS and line_poly.is_valid and outline_area > 0
             and abs(line_poly.area - outline_area) / outline_area <= MAX_AREA_DISAGREEMENT)
    return lines if clean else outline


def _fit_walls_lines(
    point_cloud: PointCloud, *, inlier_threshold: float = INLIER_THRESHOLD,
    reference_angle: float | None = None,
) -> tuple[list[WallSegment], list[tuple[float, float]]]:
    """Fit wall segments and a closed counterclockwise floor polygon from RANSAC lines.

    Finds floor and ceiling, slices points 0.8-1.5 m above the floor, projects
    to 2D, fits lines with iterative RANSAC (0.03 m threshold, min 20 inliers),
    snaps lines to 90 degrees of the dominant direction, intersects adjacent
    lines for vertices, and returns (wall segments, polygon vertices). Sparse SfM clouds
    are noisier: pass a larger inlier_threshold.
    """
    pts = point_cloud.points
    levels = find_floor_and_ceiling(pts[:, 2])
    band = (pts[:, 2] >= levels.floor_z + WALL_BAND[0]) & (pts[:, 2] <= levels.floor_z + WALL_BAND[1])
    xy = pts[band, :2]
    if len(xy) < MIN_INLIERS:
        raise ValueError(f"only {len(xy)} points in the {WALL_BAND[0]}-{WALL_BAND[1]} m wall band")

    # noisier clouds (larger inlier threshold) also see one wall as parallel lines further apart
    lines = _snap_and_merge(_fit_lines(xy, inlier_threshold), max(MERGE_OFFSET, 1.5 * inlier_threshold),
                            reference_angle)
    if len(lines) < 3:
        raise ValueError(f"found {len(lines)} wall line(s); need at least 3 to close a room")

    # Order walls counterclockwise by the angle of their midpoint around the room center.
    mids = [(ln.extent()[0] + ln.extent()[1]) / 2 for ln in lines]
    center = np.mean(mids, axis=0)
    order = np.argsort([np.arctan2(m[1] - center[1], m[0] - center[0]) for m in mids])
    lines = [lines[i] for i in order]

    corners = [_corner(lines[i], lines[(i + 1) % len(lines)]) for i in range(len(lines))]
    x, y = np.array(corners).T
    if 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:  # clockwise: flip
        lines, corners = lines[::-1], corners[::-1]
        corners = corners[1:] + corners[:1]

    # Wall i runs from the corner before it to the corner after it.
    segments: list[WallSegment] = []
    for i, ln in enumerate(lines):
        start, end = corners[i - 1], corners[i]
        delta = end - start
        segments.append(WallSegment(
            start=(float(start[0]), float(start[1])),
            end=(float(end[0]), float(end[1])),
            length=float(np.hypot(*delta)),
            direction=float(np.arctan2(delta[1], delta[0])),
            inlier_count=int(len(ln.points)),
        ))
    polygon = [(float(c[0]), float(c[1])) for c in corners]
    return segments, polygon
