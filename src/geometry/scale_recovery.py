"""Metric scale from the floor tile grid.

SfM clouds have an arbitrary scale. Floor tiles are square and come in a few standard sizes
(0.30, 0.45, 0.60, 0.80 m), so a visible tile grid fixes the scale without assuming a ceiling
height or a door width.

Method, per image:
1. Floor plane: the floor level of the Z-up cloud.
2. Rectify: every floor-plane point around the camera is projected into the image and sampled, which
   gives a top-down floor image whose pixel size is known in cloud units. (This is the exact form of
   `metric = pixel * distance / focal_length`, which only holds for a camera looking straight down.)
   Only the floor inside the cloud's footprint, and not too close to the horizon, is used.
3. Grid period: Canny + HoughLinesP, keep the two perpendicular line families, take the repeating
   spacing of each. If Hough finds no grid, the 2D autocorrelation of the floor texture is used.
4. Snap the period (cloud units x a rough prior scale) to the nearest standard tile size.
5. scale = standard tile size / period in cloud units.

Images are combined by `recover_scale_from_tiles`: the most common tile size wins, the median scale
is used, and disagreement lowers the confidence. Never raises: no usable floor gives None.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from src import config as cfg
from src.geometry.ceiling import find_floor_and_ceiling
from src.room_ir import CameraPose, PointCloud

logger = logging.getLogger(__name__)

MIN_TILE_PERIOD_M = 0.2  # grids finer than this are floor texture, not tiles
MAX_TILE_PERIOD_M = 1.2


@dataclass
class TileGridResult:
    """A scale from one image, or the combined result of several."""

    scale_factor: float  # standard tile size / tile period in cloud units: metres per cloud unit
    confidence: float  # 0 to 1
    tile_size_m: float  # the standard size assumed
    period_units: float  # tile period in cloud units
    computed_size_m: float  # period x prior scale, before snapping to a standard size
    method: str  # "hough" or "autocorrelation"
    n_images: int = 1  # images that agree on this result


# --------------------------------------------------------------------------- floor rectification

def _floor_level(points: np.ndarray) -> float:
    """Floor height of a Z-up cloud: the dense bin in the lower part of the Z range."""
    z = points[:, 2]
    low, high = np.percentile(z, [1, 99])
    extent = float(high - low)
    if extent <= 0:
        raise ValueError("no vertical extent")
    try:
        levels = find_floor_and_ceiling(z[(z >= low) & (z <= high)], bin_size=extent / 60)
        if levels.floor_peak_count >= 0.03 * len(z):
            return float(levels.floor_z)
    except ValueError:
        pass
    return float(low)


def _rectify_floor(gray: np.ndarray, pose: CameraPose, intrinsics: np.ndarray, floor_z: float,
                   footprint: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Top-down view of the floor: (image, valid mask, cloud units per pixel), or None.

    `pose` is world-to-camera (x_cam = R x_world + t) in the same Z-up frame as the cloud.
    `footprint` holds the cloud's XY points; floor outside their convex hull is not used.
    """
    height, width = gray.shape
    rot, trans = np.asarray(pose.rotation, float), np.asarray(pose.translation, float)
    center = -rot.T @ trans
    if center[2] - floor_z <= 1e-9:  # camera at or below the floor level: no usable ray hits
        return None
    sin_min = np.sin(np.deg2rad(cfg.TILE_MIN_DEPRESSION_DEG))

    # Where does the image see floor? Cast a coarse grid of rays.
    us, vs = np.meshgrid(np.arange(0, width, 16), np.arange(0, height, 16))
    uv1 = np.stack([us.ravel(), vs.ravel(), np.ones(us.size)])
    rays = (rot.T @ np.linalg.inv(intrinsics) @ uv1).T
    norm = np.linalg.norm(rays, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (floor_z - center[2]) / rays[:, 2]
    ok = (rays[:, 2] < 0) & (lam > 0) & (-rays[:, 2] / norm >= sin_min)
    if ok.sum() < 20:
        return None
    hits = center + lam[ok, None] * rays[ok]
    dist = lam[ok] * norm[ok]
    lo, hi = np.percentile(hits[:, :2], 2, axis=0), np.percentile(hits[:, :2], 98, axis=0)
    res = float(np.median(dist) / intrinsics[0, 0])  # floor length covered by one pixel at that distance
    span = hi - lo
    if not np.all(span > 0) or res <= 0:
        return None
    res = max(res, float(span.max()) / cfg.TILE_ORTHO_MAX)
    nx, ny = int(np.ceil(span[0] / res)), int(np.ceil(span[1] / res))
    if min(nx, ny) < 120:
        return None

    gx = lo[0] + res * (np.arange(nx) + 0.5)
    gy = lo[1] + res * (np.arange(ny) + 0.5)
    ground = np.stack([np.tile(gx, ny), np.repeat(gy, nx), np.full(nx * ny, floor_z)], axis=1)
    cam = ground @ rot.T + trans
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (intrinsics[0, 0] * cam[:, 0] + intrinsics[0, 1] * cam[:, 1]) / z + intrinsics[0, 2]
        v = intrinsics[1, 1] * cam[:, 1] / z + intrinsics[1, 2]
    depression = (center[2] - floor_z) / np.linalg.norm(ground - center, axis=1)
    valid = (z > 1e-9) & (u >= 2) & (u <= width - 3) & (v >= 2) & (v <= height - 3) & (depression >= sin_min)
    ortho = cv2.remap(gray, u.reshape(ny, nx).astype(np.float32), v.reshape(ny, nx).astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = valid.reshape(ny, nx).astype(np.uint8)

    hull = cv2.convexHull(footprint.astype(np.float32)).reshape(-1, 2)
    hull = hull.mean(axis=0) + 1.1 * (hull - hull.mean(axis=0))
    inside = np.zeros((ny, nx), np.uint8)
    cv2.fillConvexPoly(inside, np.round((hull - lo) / res).astype(np.int32), 1)
    mask &= inside
    if mask.sum() < 40000:  # too little floor in view to see a repeating pattern
        return None
    return ortho, mask, res


# --------------------------------------------------------------------------- grid period

@dataclass
class _Grid:
    period_px: float
    quality: float  # 0 to 1
    method: str


def _cluster(values: np.ndarray, weights: np.ndarray, tol: float) -> np.ndarray:
    """Merge values closer than `tol` into weighted means (sorted)."""
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    groups, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] - values[i - 1] > tol:
            w = weights[start:i]
            groups.append(float((values[start:i] * w).sum() / w.sum()))
            start = i
    return np.array(groups)


def _fit_period(positions: np.ndarray, min_period: float, max_period: float) -> Optional[tuple[float, float]]:
    """Repeating spacing of sorted line positions: (period, share of gaps that are whole multiples of it).

    Missed lines give gaps of 2 or 3 periods; a sub-multiple of the true period fits as well as the
    period itself, so the largest period that fits (nearly) as many gaps as the best one is taken.
    """
    gaps = np.diff(positions)
    if len(gaps) < 2:
        return None
    candidates = sorted({g / k for g in gaps for k in (1, 2, 3) if min_period <= g / k <= max_period})
    if not candidates:
        return None
    scores = []
    for p in candidates:
        ratio = gaps / p
        scores.append(int(((np.abs(ratio - np.round(ratio)) < 0.12) & (np.round(ratio) >= 1)).sum()))
    best = max(scores)
    if best < 2:
        return None
    period = max(p for p, s in zip(candidates, scores) if s >= 0.85 * best)
    ratio = gaps / period
    fit = (np.abs(ratio - np.round(ratio)) < 0.12) & (np.round(ratio) >= 1)
    if fit.sum() < 2:
        return None
    return float(np.mean(gaps[fit] / np.round(ratio[fit]))), float(fit.mean())


def _grid_from_hough(ortho: np.ndarray, mask: np.ndarray, m_per_px: float) -> Optional[_Grid]:
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(cv2.GaussianBlur(ortho, (5, 5), 0))
    inner = cv2.erode(mask, np.ones((9, 9), np.uint8))
    edges = cv2.Canny(gray, 40, 120) * inner
    h, w = edges.shape
    min_len = int(np.clip(0.45 / m_per_px, 25, 0.4 * min(h, w)))
    gap = int(np.clip(0.15 / m_per_px, 10, 60))
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, max(30, min_len // 2), minLineLength=min_len, maxLineGap=gap)
    if segs is None:
        return None
    segs = segs.reshape(-1, 4).astype(float)  # HoughLinesP's array shape differs between OpenCV builds
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    length = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx) % np.pi
    if len(segs) < 2 * cfg.TILE_MIN_LINES:
        return None

    # Dominant grid orientation: lines fall into two perpendicular families, so fold angles mod 90 degrees.
    fold = theta % (np.pi / 2)
    hist = np.bincount(np.minimum((fold / (np.pi / 2) * 90).astype(int), 89), weights=length, minlength=90)
    smooth = np.convolve(np.r_[hist[-2:], hist, hist[:2]], np.ones(5) / 5, mode="valid")
    peak = (np.argmax(smooth) + 0.5) * (np.pi / 2) / 90
    near = np.abs(np.angle(np.exp(4j * (fold - peak)))) / 4 < np.deg2rad(6)
    phi = peak + np.angle(np.sum(length[near] * np.exp(4j * (fold[near] - peak)))) / 4

    tol = np.deg2rad(6)
    mid = np.stack([(segs[:, 0] + segs[:, 2]) / 2, (segs[:, 1] + segs[:, 3]) / 2], axis=1)
    periods, consistency, line_counts = [], [], []
    for alpha in (phi, phi + np.pi / 2):
        diff = np.abs(np.angle(np.exp(2j * (theta - alpha)))) / 2
        member = diff < tol
        if member.sum() < cfg.TILE_MIN_LINES:
            return None
        normal = np.array([-np.sin(alpha), np.cos(alpha)])
        lines = _cluster(mid[member] @ normal, length[member], max(4.0, 0.04 / m_per_px))
        if len(lines) < cfg.TILE_MIN_LINES:
            return None
        fit = _fit_period(lines, MIN_TILE_PERIOD_M / m_per_px, MAX_TILE_PERIOD_M / m_per_px)
        if fit is None:
            return None
        periods.append(fit[0])
        consistency.append(fit[1])
        line_counts.append(len(lines))

    mean_period = float(np.mean(periods))
    disagreement = abs(periods[0] - periods[1]) / mean_period
    if disagreement > cfg.TILE_SQUARE_TOLERANCE:  # not square: planks, a brick bond, or noise
        return None
    quality = (np.sqrt(min(1.0, sum(line_counts) / 10) * float(np.mean(consistency)))
               * (1 - 0.5 * disagreement / cfg.TILE_SQUARE_TOLERANCE))
    return _Grid(mean_period, float(quality), "hough")


def _grid_from_autocorrelation(ortho: np.ndarray, mask: np.ndarray, m_per_px: float) -> Optional[_Grid]:
    """Tile period from the first lattice peaks of the masked 2D autocorrelation of the floor texture."""
    img = ortho.astype(np.float32)
    weight = mask.astype(np.float32)
    background = cv2.GaussianBlur(img * weight, (0, 0), 0.1 * min(img.shape)) / np.maximum(
        cv2.GaussianBlur(weight, (0, 0), 0.1 * min(img.shape)), 1e-3)
    x = (img - background) * weight  # illumination gradient removed
    shape = [int(s * 2) for s in x.shape]
    spectrum = np.fft.rfft2(x, shape)
    ac = np.fft.irfft2(np.abs(spectrum) ** 2, shape)
    overlap = np.fft.irfft2(np.abs(np.fft.rfft2(weight, shape)) ** 2, shape)
    ac = np.fft.fftshift(ac / np.maximum(overlap, 0.3 * overlap.max()))
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    ac /= ac[cy, cx]

    min_px = MIN_TILE_PERIOD_M / m_per_px
    radius = int(max(3, min_px / 2))
    size = 2 * radius + 1
    peaks = (ac == cv2.dilate(ac, np.ones((size, size), np.uint8))) & (ac > 0.15)
    ys, xs = np.nonzero(peaks)
    vec = np.stack([xs - cx, ys - cy], axis=1).astype(float)
    dist = np.hypot(vec[:, 0], vec[:, 1])
    keep = (dist >= min_px) & (dist <= MAX_TILE_PERIOD_M / m_per_px)
    vec, dist, vals = vec[keep], dist[keep], ac[ys[keep], xs[keep]]
    if len(vec) < 2:
        return None
    first = int(np.argmin(dist))  # the nearest strong peak is one lattice vector
    for j in np.argsort(dist):
        cosang = abs(vec[first] @ vec[j]) / (dist[first] * dist[j])
        if cosang < np.sin(np.deg2rad(12)) and abs(dist[j] - dist[first]) / dist[first] < cfg.TILE_SQUARE_TOLERANCE:
            period = float((dist[first] + dist[j]) / 2)
            return _Grid(period, float(min(1.0, min(vals[first], vals[j]) / 0.5)), "autocorrelation")
    return None


# --------------------------------------------------------------------------- snapping and public API

def snap_tile_size(size_m: float) -> Optional[float]:
    """Standard tile size whose band holds `size_m` (0.25-0.35 -> 0.30, 0.40-0.50 -> 0.45, 0.50-0.65 -> 0.60,
    0.70-0.90 -> 0.80), or None when it falls outside every band."""
    hits = [t for t, (lo, hi) in cfg.TILE_BANDS.items() if lo <= size_m <= hi]
    if not hits:
        return None
    return min(hits, key=lambda t: abs(np.log(size_m / t)))


def _load_gray(image: Union[str, Path, np.ndarray], intrinsics: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Grayscale image (long edge at most TILE_IMAGE_LONG_EDGE) and the intrinsics matching it."""
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image), cv2.IMREAD_COLOR)  # applies EXIF orientation, like staging does
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    k = np.asarray(intrinsics, float).copy()
    scale = cfg.TILE_IMAGE_LONG_EDGE / max(gray.shape)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        k[:2, :] *= scale
    return gray, k


def detect_tile_grid(image: Union[str, Path, np.ndarray], camera_pose: CameraPose, point_cloud: PointCloud,
                     prior_scale: Optional[float] = None) -> Optional[TileGridResult]:
    """Scale (metres per cloud unit) from a repeating floor tile pattern in one image, or None.

    `point_cloud` is the Z-up reconstruction in arbitrary units and `camera_pose` is in that frame
    (world-to-camera, intrinsics of the full-size image). `prior_scale` is a rough metres-per-unit
    (default: the cloud's vertical extent taken as a 2.5 m ceiling); it only decides which standard tile
    size the measured period is closest to, the scale itself comes from the tile size.
    Returns a TileGridResult whose `scale_factor` is the answer; None when the floor is not visible,
    has no square grid, or the grid size matches no standard tile.
    """
    try:
        points = point_cloud.points
        loaded = _load_gray(image, camera_pose.intrinsics)
        if loaded is None or len(points) < 10:
            return None
        gray, k = loaded
        if prior_scale is None:
            low, high = np.percentile(points[:, 2], [1, 99])
            prior_scale = cfg.TYPICAL_CEILING / float(high - low)
        rectified = _rectify_floor(gray, camera_pose, k, _floor_level(points), points[:, :2])
        if rectified is None:
            return None
        ortho, mask, res = rectified
        if float(ortho[mask > 0].std()) < 3.0:  # a blank floor has no grid
            return None
        m_per_px = res * prior_scale
        grid = _grid_from_hough(ortho, mask, m_per_px) or _grid_from_autocorrelation(ortho, mask, m_per_px)
        if grid is None:
            return None
        period_units = grid.period_px * res
        computed = period_units * prior_scale
        tile = snap_tile_size(computed)
        if tile is None:
            logger.info("tile grid of %.2f m matches no standard tile size", computed)
            return None
        closeness = 0.7 + 0.3 * max(0.0, 1 - abs(np.log(computed / tile)) / np.log(1.25))
        return TileGridResult(scale_factor=float(tile / period_units), confidence=float(grid.quality * closeness),
                              tile_size_m=tile, period_units=float(period_units), computed_size_m=float(computed),
                              method=grid.method)
    except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
        logger.debug("tile detection failed: %s", exc)
        return None


def recover_scale_from_tiles(poses: list[CameraPose], point_cloud: PointCloud,
                             prior_scale: Optional[float] = None,
                             max_images: int = cfg.TILE_MAX_IMAGES) -> Optional[TileGridResult]:
    """Combine `detect_tile_grid` over up to `max_images` registered images that exist on disk.

    The most common tile size wins; the scale is the median of its images. Confidence is the mean
    image confidence, lowered when images disagree, when other images found a different tile size
    or none, and when only one image found a grid.
    """
    usable = [p for p in poses if p.image_path and Path(p.image_path).is_file()]
    if not usable:
        return None
    if len(usable) > max_images:
        usable = [usable[i] for i in np.linspace(0, len(usable) - 1, max_images).round().astype(int)]
    results = [r for r in (detect_tile_grid(p.image_path, p, point_cloud, prior_scale) for p in usable) if r]
    if not results:
        return None
    sizes = [r.tile_size_m for r in results]
    tile = max(set(sizes), key=lambda s: (sizes.count(s), sum(r.confidence for r in results if r.tile_size_m == s)))
    group = [r for r in results if r.tile_size_m == tile]
    factors = np.array([r.scale_factor for r in group])
    factor = float(np.median(factors))
    spread = float(np.std(factors) / factor)
    agreement = (len(group) / len(results)) * (1 - 0.5 * min(spread / 0.2, 1.0))
    confidence = float(np.mean([r.confidence for r in group])) * agreement
    if len(group) == 1 and len(usable) > 1:
        confidence *= 0.85  # one image of several saw the grid
    best = max(group, key=lambda r: r.confidence)
    return TileGridResult(scale_factor=factor, confidence=float(min(confidence, 0.99)), tile_size_m=tile,
                          period_units=tile / factor, computed_size_m=float(np.median([r.computed_size_m for r in group])),
                          method=best.method, n_images=len(group))
