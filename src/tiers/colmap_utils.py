"""Structure-from-motion helpers shared by the photo and video tiers.

Runs COLMAP (the `colmap` binary if installed, otherwise the `pycolmap` Python
package), reads the sparse model, and turns the arbitrary-scale, arbitrary-orientation
reconstruction into a metric, Z-up point cloud:

1. align_to_gravity: rotate so the vertical axis is Z (camera up vectors, refined by
   horizontal and vertical planes).
2. approximate_scale: a rough meters-per-unit from the vertical extent, so the
   meter-based geometry thresholds mean something.
3. recover_scale: refine with a standard door width (0.86 m) once openings are found.

Camera poses are COLMAP world-to-camera: x_cam = R @ x_world + t, camera axes x right,
y down, z forward.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.geometry.ceiling import find_floor_and_ceiling
from src.room_ir import CameraPose, PointCloud

logger = logging.getLogger(__name__)

DOOR_WIDTH = 0.86  # meters; the scale reference
TYPICAL_CEILING = 2.5
TYPICAL_LONGEST_WALL = 4.0
FEATURE_HEIGHT = 2.2  # typical vertical extent of reconstructed features on walls
DOOR_WIDTH_RANGE = (0.6, 1.2)  # detected "doors" outside this (in approximate meters) are ignored
DOOR_MIN_GAP_HEIGHT = 1.5  # sparse SfM clouds: only door-tall empty columns count as openings
DOOR_CORRECTION_RANGE = (0.8, 1.25)  # the prior is good to ~15%; a bigger door-based correction is a false door
MAX_SCALE_CORRECTION = (0.6, 1.6)
SFM_WALL_THRESHOLD = 0.06  # meters; SfM points scatter more around a wall than LiDAR
MIN_POINTS = 30
MIN_IMAGES = 3
MAX_REPROJECTION_ERROR = 1.5  # pixels
SEED = 0  # fixed so repeated runs agree
COLMAP_TIMEOUT = 3600  # seconds per COLMAP command

# number of focal-length parameters at the start of each COLMAP camera model's parameter list
_FOCAL_PARAMS = {
    "SIMPLE_PINHOLE": 1, "SIMPLE_RADIAL": 1, "RADIAL": 1, "SIMPLE_RADIAL_FISHEYE": 1,
    "RADIAL_FISHEYE": 1, "PINHOLE": 2, "OPENCV": 2, "OPENCV_FISHEYE": 2, "FULL_OPENCV": 2,
    "FOV": 2, "THIN_PRISM_FISHEYE": 2,
}


class ColmapError(RuntimeError):
    """COLMAP is missing, failed, or produced an unusable reconstruction."""


@dataclass
class ScaleEstimate:
    """A multiplicative correction for the cloud it was computed on."""

    factor: float
    method: str  # "door" | "longest-wall" | "none"
    detail: str = ""


# --------------------------------------------------------------------------- running COLMAP

def colmap_backend() -> str:
    """"cli" if the colmap binary is on PATH, else "pycolmap". FLOORPLAN_COLMAP_BACKEND overrides."""
    forced = os.environ.get("FLOORPLAN_COLMAP_BACKEND", "").lower()
    if forced in {"cli", "pycolmap"}:
        return forced
    if shutil.which("colmap"):
        return "cli"
    try:
        import pycolmap  # noqa: F401
    except ImportError:
        raise ColmapError("COLMAP not found: `brew install colmap` or `pip install pycolmap`") from None
    return "pycolmap"


def _run(cmd: list[str]) -> None:
    logger.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=COLMAP_TIMEOUT)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        raise ColmapError(f"`{' '.join(cmd[:2])}` failed (exit {proc.returncode}): {' | '.join(tail)}")


def _reconstruct_cli(image_dir: Path, db: Path, sparse: Path, matcher: str) -> None:
    _run(["colmap", "feature_extractor", "--database_path", str(db), "--image_path", str(image_dir),
          "--ImageReader.single_camera", "1"])
    _run(["colmap", f"{matcher}_matcher", "--database_path", str(db)])
    _run(["colmap", "mapper", "--database_path", str(db), "--image_path", str(image_dir),
          "--output_path", str(sparse)])


def _reconstruct_pycolmap(image_dir: Path, db: Path, sparse: Path, matcher: str) -> None:
    """pycolmap with fixed seeds and a single mapper thread, so the same images give the same model."""
    import pycolmap

    pycolmap.set_random_seed(SEED)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = 1
    pycolmap.extract_features(str(db), str(image_dir), camera_mode=pycolmap.CameraMode.SINGLE,
                              extraction_options=extraction)
    verification = pycolmap.TwoViewGeometryOptions()
    verification.ransac.random_seed = SEED
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = 1
    if matcher == "sequential":
        pycolmap.match_sequential(str(db), matching_options=matching, verification_options=verification)
    else:
        pycolmap.match_exhaustive(str(db), matching_options=matching, verification_options=verification)
    options = pycolmap.IncrementalPipelineOptions()
    options.random_seed = SEED
    options.num_threads = 1
    pycolmap.incremental_mapping(str(db), str(image_dir), str(sparse), options=options)


def _to_text_model(model_dir: Path, backend: str) -> Path:
    """Convert a binary COLMAP model to cameras.txt / images.txt / points3D.txt next to it."""
    if (model_dir / "images.txt").exists():
        return model_dir
    text_dir = model_dir.with_name(model_dir.name + "_txt")
    text_dir.mkdir(exist_ok=True)
    if backend == "cli":
        _run(["colmap", "model_converter", "--input_path", str(model_dir),
              "--output_path", str(text_dir), "--output_type", "TXT"])
    else:
        import pycolmap

        pycolmap.Reconstruction(str(model_dir)).write_text(str(text_dir))
    return text_dir


def run_colmap_reconstruction(
    image_dir: str, workspace_dir: str, matcher: str = "exhaustive"
) -> tuple[np.ndarray, list[CameraPose]]:
    """Reconstruct a folder of images and return (points N x 3, camera poses).

    Work files (database.db, sparse/) go to workspace_dir. matcher is "exhaustive"
    (photos) or "sequential" (video frames, which only overlap with neighbors). When
    COLMAP builds several disconnected models the one with the most images is used.
    The coordinates are COLMAP's own: arbitrary scale and orientation.
    """
    image_dir, workspace = Path(image_dir), Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    db, sparse = workspace / "database.db", workspace / "sparse"
    if db.exists():
        db.unlink()
    shutil.rmtree(sparse, ignore_errors=True)
    sparse.mkdir()

    backend = colmap_backend()
    logger.info("COLMAP backend: %s, %d images", backend, len(list(image_dir.iterdir())))
    try:
        (_reconstruct_cli if backend == "cli" else _reconstruct_pycolmap)(image_dir, db, sparse, matcher)
    except ColmapError:
        raise
    except Exception as exc:  # pycolmap raises RuntimeError and friends
        raise ColmapError(f"COLMAP reconstruction failed: {exc}") from exc

    models = sorted(p for p in sparse.iterdir() if p.is_dir() and p.name.isdigit())
    if not models:
        raise ColmapError("COLMAP produced no model (too few matching features between images)")
    best: tuple[np.ndarray, list[CameraPose]] | None = None
    for model in models:
        points, poses = parse_colmap_sparse(str(_to_text_model(model, backend)))
        if best is None or len(poses) > len(best[1]):
            best = (points, poses)
    points, poses = best
    if len(poses) < MIN_IMAGES or len(points) < MIN_POINTS:
        raise ColmapError(f"reconstruction too small: {len(poses)} images, {len(points)} points")
    logger.info("reconstructed %d images, %d points", len(poses), len(points))
    return points, poses


# --------------------------------------------------------------------------- parsing

def _quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _intrinsics(model: str, params: list[float]) -> np.ndarray:
    n_focal = _FOCAL_PARAMS.get(model, 1)
    fx = params[0]
    fy = params[1] if n_focal == 2 else params[0]
    cx, cy = params[n_focal], params[n_focal + 1]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _data_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def parse_colmap_sparse(sparse_dir: str) -> tuple[np.ndarray, list[CameraPose]]:
    """Read cameras.txt, images.txt and points3D.txt from a COLMAP text model.

    Returns (points N x 3, poses). Points with a large reprojection error or seen in
    fewer than 2 images are dropped. CameraPose.image_path is the image name as COLMAP
    stored it (relative to the image folder).
    """
    root = Path(sparse_dir)
    cameras: dict[int, np.ndarray] = {}
    for line in _data_lines(root / "cameras.txt"):
        parts = line.split()
        cameras[int(parts[0])] = _intrinsics(parts[1], [float(v) for v in parts[4:]])

    poses: list[CameraPose] = []
    image_lines = _data_lines(root / "images.txt")
    for line in image_lines[0::2]:  # every second line is the 2D points of the previous image
        parts = line.split()
        qw, qx, qy, qz, tx, ty, tz = (float(v) for v in parts[1:8])
        poses.append(CameraPose(
            image_path=parts[9], rotation=_quaternion_to_rotation(qw, qx, qy, qz),
            translation=np.array([tx, ty, tz]), intrinsics=cameras[int(parts[8])]))

    points = []
    for line in _data_lines(root / "points3D.txt"):
        parts = line.split()
        error, track_len = float(parts[7]), (len(parts) - 8) // 2
        if error <= MAX_REPROJECTION_ERROR and track_len >= 2:
            points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(points, dtype=float).reshape(-1, 3), poses


# --------------------------------------------------------------------------- orientation

def _ransac_plane(points: np.ndarray, rng: np.random.Generator, threshold: float, iterations: int = 300):
    """Best plane (unit normal, offset, inlier mask), or None."""
    if len(points) < 10:
        return None
    idx = rng.integers(0, len(points), size=(iterations, 3))
    a, b, c = points[idx[:, 0]], points[idx[:, 1]], points[idx[:, 2]]
    normals = np.cross(b - a, c - a)
    norm = np.linalg.norm(normals, axis=1)
    ok = norm > 1e-12
    if not ok.any():
        return None
    normals, a = normals[ok] / norm[ok, None], a[ok]
    offsets = np.einsum("ij,ij->i", normals, a)
    counts = (np.abs(points @ normals.T - offsets) < threshold).sum(axis=0)
    best = int(np.argmax(counts))
    inliers = np.abs(points @ normals[best] - offsets[best]) < threshold
    mean = points[inliers].mean(axis=0)
    normal = np.linalg.svd(points[inliers] - mean, full_matrices=False)[2][-1]  # least-squares refit
    offset = float(normal @ mean)
    return normal, offset, np.abs(points @ normal - offset) < threshold


def _camera_up(poses: list[CameraPose]) -> np.ndarray:
    ups = np.array([p.rotation.T @ np.array([0.0, -1.0, 0.0]) for p in poses])
    up = ups.mean(axis=0)
    return up / np.linalg.norm(up)


def align_to_gravity(points: np.ndarray, poses: list[CameraPose], seed: int = 0) -> np.ndarray:
    """Rotation A (3 x 3) such that A @ x has the vertical axis on +Z.

    Starts from the mean camera up vector (images are upright and roughly level), then
    refines it from the scene: the up direction should be perpendicular to wall planes
    and parallel to floor/ceiling normals.
    """
    up = _camera_up(poses)
    rng = np.random.default_rng(seed)
    center = np.median(points, axis=0)
    scale = np.percentile(np.linalg.norm(points - center, axis=1), 90)
    threshold = 0.02 * scale

    remaining, matrix, used = points, np.zeros((3, 3)), 0
    for _ in range(6):
        plane = _ransac_plane(remaining, rng, threshold)
        if plane is None:
            break
        normal, _, inliers = plane
        if inliers.sum() < max(10, 0.06 * len(points)):
            break
        along_up = abs(normal @ up)
        weight = float(inliers.sum())
        if along_up > np.cos(np.deg2rad(20)):  # floor / ceiling
            matrix -= weight * np.outer(normal, normal)
            used += 1
        elif along_up < np.sin(np.deg2rad(20)):  # wall
            matrix += weight * np.outer(normal, normal)
            used += 1
        remaining = remaining[~inliers]
    if used:
        vals, vecs = np.linalg.eigh(matrix)
        refined = vecs[:, 0]
        refined *= np.sign(refined @ up) or 1.0
        if np.degrees(np.arccos(np.clip(refined @ up, -1, 1))) < 25:  # sanity: stay near the camera estimate
            up = refined

    z = up / np.linalg.norm(up)
    x = np.cross([0.0, 1.0, 0.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    return np.vstack([x, np.cross(z, x), z])


def approximate_scale(points_z_up: np.ndarray) -> tuple[float, str]:
    """Rough meters-per-unit for a Z-up cloud of unknown scale. Returns (factor, method).

    If the Z histogram shows a floor and a ceiling, they are assumed 2.5 m apart.
    Otherwise the vertical extent of the points is assumed to be about 2.1 m.
    """
    z = points_z_up[:, 2]
    low, high = np.percentile(z, [1, 99])
    extent = float(high - low)
    if extent <= 0:
        raise ColmapError("degenerate reconstruction: no vertical extent")
    try:
        levels = find_floor_and_ceiling(z[(z >= low) & (z <= high)], bin_size=extent / 60)
        if min(levels.floor_peak_count, levels.ceiling_peak_count) >= 0.05 * len(z) and levels.height > 0.5 * extent:
            return TYPICAL_CEILING / levels.height, "floor-ceiling"
    except ValueError:
        pass
    return FEATURE_HEIGHT / extent, "vertical-extent"


def recover_scale(
    point_cloud: PointCloud, openings: list, wall_segments: Optional[list] = None
) -> ScaleEstimate:
    """Correction factor from openings (and walls) detected on `point_cloud`.

    The widest door is taken to be 0.86 m wide: factor = 0.86 / width. With no usable
    door, the longest wall is taken to be 4 m. Implausible results (door width outside
    0.5-1.5 m, factor outside 0.6-1.6) are ignored and give factor 1.0.
    """
    lo, hi = MAX_SCALE_CORRECTION
    doors = [o for o in openings if o.type == "door" and DOOR_WIDTH_RANGE[0] <= o.width <= DOOR_WIDTH_RANGE[1]]
    if doors:
        widest = max(doors, key=lambda o: o.width)
        factor = DOOR_WIDTH / widest.width
        if lo <= factor <= hi:
            return ScaleEstimate(factor, "door", f"widest door {widest.width:.2f} m -> {DOOR_WIDTH} m")
    if wall_segments:
        longest = max(w.length for w in wall_segments)
        factor = TYPICAL_LONGEST_WALL / longest
        if lo <= factor <= hi:
            return ScaleEstimate(factor, "longest-wall", f"longest wall {longest:.2f} m -> {TYPICAL_LONGEST_WALL} m")
    return ScaleEstimate(1.0, "none", "no usable door or wall")


# --------------------------------------------------------------------------- full pipeline

def make_metric_point_cloud(
    points: np.ndarray, poses: list[CameraPose]
) -> tuple[PointCloud, list[CameraPose], dict]:
    """COLMAP output -> Z-up point cloud in (approximately, then door-refined) meters.

    Returns (cloud, poses in the same frame, metadata with scale method and factor).
    Geometry is run once on the approximately scaled cloud to find doors for the refinement.
    """
    from src.geometry.openings import detect_openings
    from src.geometry.wall_fitting import fit_walls

    points = _drop_far_outliers(points)
    rotation = align_to_gravity(points, poses)
    aligned = _trim_vertical_outliers(points @ rotation.T)
    factor0, method0 = approximate_scale(aligned)
    meters = aligned * factor0
    cloud = PointCloud(points=meters)

    meta = {"scale_prior": method0, "scale_prior_factor": float(factor0), "scale_method": method0,
            "scale_correction": 1.0}
    correction = 1.0
    try:
        walls, _ = fit_walls(cloud, inlier_threshold=SFM_WALL_THRESHOLD)
        openings = detect_openings(cloud, walls, min_gap_height=DOOR_MIN_GAP_HEIGHT)
        estimate = recover_scale(cloud, openings, walls)
        # The longest-wall fallback (4 m) is a weaker prior than the vertical one, so it is not applied.
        if estimate.method == "door" and DOOR_CORRECTION_RANGE[0] <= estimate.factor <= DOOR_CORRECTION_RANGE[1]:
            correction, meta["scale_method"], meta["scale_detail"] = estimate.factor, "door", estimate.detail
        else:
            logger.info("no usable door for scale (%s); keeping the %s prior", estimate.method, method0)
    except ValueError as exc:  # too sparse to fit walls here; the caller will hit the same wall later
        logger.warning("scale refinement skipped: %s", exc)
    meta["scale_correction"] = float(correction)
    total = factor0 * correction
    meta["scale_factor"] = float(total)
    logger.info("scale: prior %s x%.3f, refinement %s x%.3f", method0, factor0, meta["scale_method"], correction)

    cloud = PointCloud(points=aligned * total)
    new_poses = [CameraPose(image_path=p.image_path, rotation=p.rotation @ rotation.T,
                            translation=p.translation * total, intrinsics=p.intrinsics) for p in poses]
    return cloud, new_poses, meta


def _drop_far_outliers(points: np.ndarray, factor: float = 4.0) -> np.ndarray:
    """Remove points much farther from the median than most (stray SfM triangulations)."""
    center = np.median(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)
    return points[dist <= factor * np.median(dist)]


def _trim_vertical_outliers(aligned: np.ndarray, margin: float = 0.15) -> np.ndarray:
    """Drop points far above/below the bulk of a Z-up cloud (stray SfM triangulations).

    Keeps Z within the 0.5-99.5 percentile range widened by `margin` of its height; a
    handful of far points would otherwise stretch the Z histogram and hide floor and ceiling.
    """
    low, high = np.percentile(aligned[:, 2], [0.5, 99.5])
    pad = margin * (high - low)
    return aligned[(aligned[:, 2] >= low - pad) & (aligned[:, 2] <= high + pad)]
