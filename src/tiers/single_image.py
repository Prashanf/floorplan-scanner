"""Last-resort room estimate from a single photo (photo tier, when SfM finds no model).

This is NOT a measurement. A single photo without depth or a second view carries no metric
scale, so the room's dimensions here are typical-room priors (see DEFAULT_*), reported with
+/-50 % intervals and flagged `rough_estimate` in the report. What the image does contribute
is a check: vanishing points found from its line segments must describe an orthogonal,
box-like interior. An image that does not (a close-up of a fixture, a blur, a blank wall)
gives no room at all, because inventing a room from it would be confident garbage.

Method: Canny + probabilistic Hough line segments; RANSAC over segment pairs finds up to
three vanishing points in homogeneous coordinates (parallel lines meet at infinity, which the
homogeneous form handles); the box test needs two or more vanishing points that explain
enough of the total line length and whose back-projected directions are roughly orthogonal
for a focal length prior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.room_ir import PointCloud, RoomIR

logger = logging.getLogger(__name__)

DEFAULT_WIDTH = 3.6  # metres, typical bedroom or living room side
DEFAULT_LENGTH = 3.0
DEFAULT_CEILING = 2.5
ROUGH_RELATIVE_ERROR = 0.5  # +/-50 % on every dimension of a rough estimate

FOCAL_FACTOR = 1.2  # focal length prior = this x the longer image side (COLMAP's default guess)
MIN_SEGMENT_FRACTION = 0.06  # segments shorter than this share of the image diagonal are ignored
INLIER_ANGLE = np.deg2rad(4.0)  # a segment supports a vanishing point if it points at it within this
MIN_COVERAGE = 0.35  # share of total segment length the vanishing points must explain
ORTHOGONAL_TOLERANCE = np.deg2rad(20.0)
RANSAC_ITERATIONS = 400
POINT_DENSITY = 1200.0  # points per m2 of the stand-in box surface
MIN_SEGMENTS = 25  # with fewer segments any two vanishing points "explain" them
MAX_EDGE_DENSITY = 0.12  # share of edge pixels; interiors sit around 0.02-0.08, noise and foliage far above


@dataclass
class VanishingPoints:
    """Result of the line analysis of one image."""

    points: list[np.ndarray]  # homogeneous (x, y, w), strongest first
    coverage: float  # share of total segment length explained by them
    orthogonal: bool  # the back-projected directions of the first two are roughly perpendicular
    n_segments: int


def _segments(gray: np.ndarray) -> np.ndarray:
    """Line segments (x1, y1, x2, y2) of the image, long ones only; none for a heavily textured image."""
    h, w = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    if float(np.count_nonzero(edges)) / edges.size > MAX_EDGE_DENSITY:
        return np.zeros((0, 4))  # edges everywhere: random lines would "agree" on any vanishing point
    min_length = MIN_SEGMENT_FRACTION * np.hypot(h, w)
    found = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=int(min_length), maxLineGap=8)
    return np.zeros((0, 4)) if found is None else np.asarray(found, dtype=float).reshape(-1, 4)  # shape differs by OpenCV version


def _support(vp: np.ndarray, mid: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Boolean mask: segments (midpoints `mid`, unit `direction`) pointing at the homogeneous point `vp`."""
    toward = np.column_stack([vp[0] - vp[2] * mid[:, 0], vp[1] - vp[2] * mid[:, 1]])
    norm = np.linalg.norm(toward, axis=1)
    toward = toward / np.maximum(norm[:, None], 1e-9)
    cos = np.abs(np.sum(toward * direction, axis=1))  # a segment may point toward or away from the point
    return (cos > np.cos(INLIER_ANGLE)) & (norm > 1e-9)


def find_vanishing_points(image: np.ndarray, seed: int = 0) -> VanishingPoints:
    """Up to three vanishing points of a BGR or gray image (RANSAC over its line segments)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    segs = _segments(gray)
    if len(segs) < MIN_SEGMENTS:
        return VanishingPoints([], 0.0, False, len(segs))
    p1, p2 = segs[:, :2], segs[:, 2:]
    mid = (p1 + p2) / 2
    delta = p2 - p1
    length = np.linalg.norm(delta, axis=1)
    direction = delta / length[:, None]
    lines = np.cross(np.column_stack([p1, np.ones(len(segs))]), np.column_stack([p2, np.ones(len(segs))]))
    total = float(length.sum())

    rng = np.random.default_rng(seed)
    remaining = np.arange(len(segs))
    found: list[np.ndarray] = []
    explained = 0.0
    for _ in range(3):
        if len(remaining) < 4:
            break
        best_vp, best_score, best_mask = None, 0.0, None
        for _ in range(RANSAC_ITERATIONS):
            i, j = rng.choice(remaining, 2, replace=False)
            vp = np.cross(lines[i], lines[j])
            if np.linalg.norm(vp) < 1e-9:
                continue
            vp = vp / np.linalg.norm(vp)
            mask = _support(vp, mid[remaining], direction[remaining])
            score = float(length[remaining][mask].sum())
            if score > best_score:
                best_vp, best_score, best_mask = vp, score, mask
        if best_vp is None or best_score < 0.08 * total:
            break
        found.append(best_vp)
        explained += best_score
        remaining = remaining[~best_mask]

    orthogonal = False
    if len(found) >= 2:
        f = FOCAL_FACTOR * max(h, w)
        k_inv = np.linalg.inv(np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]))
        d1, d2 = k_inv @ found[0], k_inv @ found[1]  # homogeneous point -> direction in camera space
        cos = abs(float(d1 @ d2)) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-12)
        orthogonal = cos < np.sin(ORTHOGONAL_TOLERANCE)
    return VanishingPoints(found, explained / total if total else 0.0, orthogonal, len(segs))


def looks_like_a_room(vp: VanishingPoints) -> bool:
    """The image shows a box-like interior: 2+ orthogonal vanishing points explaining enough line length."""
    return len(vp.points) >= 2 and vp.orthogonal and vp.coverage >= MIN_COVERAGE


def _sharpest(paths: list[str]) -> tuple[str, np.ndarray] | None:
    best = None
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        sharp = float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        if best is None or sharp > best[0]:
            best = (sharp, path, image)
    return None if best is None else (best[1], best[2])


def _box_cloud(width: float, length: float, height: float, seed: int = 0) -> PointCloud:
    """Noise-free surface samples of an axis-aligned box room: floor, ceiling and four walls."""
    rng = np.random.default_rng(seed)

    def plane(area: float, fixed_axis: int, value: float, extent: tuple[float, float]):
        n = max(50, int(POINT_DENSITY * area))
        pts = np.empty((n, 3))
        pts[:, fixed_axis] = value
        for axis, hi in zip([a for a in range(3) if a != fixed_axis], extent):
            pts[:, axis] = rng.uniform(0.0, hi, n)
        return pts

    chunks = [
        plane(width * length, 2, 0.0, (width, length)),  # floor
        plane(width * length, 2, height, (width, length)),  # ceiling
        plane(width * height, 1, 0.0, (width, height)),  # wall y = 0
        plane(width * height, 1, length, (width, height)),  # wall y = length
        plane(length * height, 0, 0.0, (length, height)),  # wall x = 0
        plane(length * height, 0, width, (length, height)),  # wall x = width
    ]
    return PointCloud(points=np.vstack(chunks))


def estimate_room_from_images(image_paths: list[str], room_id: str) -> RoomIR | None:
    """A rough RoomIR for a room whose reconstruction failed, or None when no image looks like a room.

    The sharpest image is analysed. The dimensions are priors (3.6 x 3.0 x 2.5 m) and the room is
    flagged in `metadata["rough_estimate"]`; the pipeline gives it +/-50 % intervals and a warning.
    """
    best = _sharpest(image_paths)
    if best is None:
        return None
    path, image = best
    vp = find_vanishing_points(image)
    logger.info("%s: single-image analysis of %s: %d segments, %d vanishing points, coverage %.2f, "
                "orthogonal=%s", room_id, Path(path).name, vp.n_segments, len(vp.points), vp.coverage,
                vp.orthogonal)
    if not looks_like_a_room(vp):
        return None
    cloud = _box_cloud(DEFAULT_WIDTH, DEFAULT_LENGTH, DEFAULT_CEILING)
    room = RoomIR(room_id=room_id, point_cloud=cloud, images=list(image_paths), tier="photo",
                  point_density=len(cloud) / (DEFAULT_WIDTH * DEFAULT_LENGTH * DEFAULT_CEILING))
    room.metadata.update(rough_estimate=True, fallback="single-image", analysed_image=path,
                         vanishing_points=len(vp.points), line_coverage=round(vp.coverage, 3),
                         dimensions_are_priors=True)
    return room
