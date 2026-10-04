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
from src import config as cfg

logger = logging.getLogger(__name__)

DOOR_WIDTH = cfg.DEFAULT_DOOR_WIDTH
TYPICAL_CEILING = cfg.TYPICAL_CEILING
TYPICAL_LONGEST_WALL = 4.0
FEATURE_HEIGHT = 2.2  # typical vertical extent of reconstructed features on walls
DOOR_WIDTH_RANGE = cfg.DOOR_WIDTH_RANGE
DOOR_MIN_GAP_HEIGHT = 1.5  # sparse SfM clouds: only door-tall empty columns count as openings
DOOR_CORRECTION_RANGE = cfg.DOOR_CORRECTION_RANGE
MAX_SCALE_CORRECTION = (0.6, 1.6)
SFM_WALL_THRESHOLD = cfg.SFM_WALL_INLIER_THRESHOLD
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


def _camera_param_string(prior: tuple[float, float, float], model: str = "SIMPLE_RADIAL") -> str:
    """COLMAP camera parameters for a focal length and principal point prior, for the given camera model."""
    f, cx, cy = prior
    if model == "SIMPLE_PINHOLE":
        values = [f, cx, cy]
    elif model == "OPENCV":
        values = [f, f, cx, cy, 0, 0, 0, 0]
    else:  # SIMPLE_RADIAL
        values = [f, cx, cy, 0]
    return ",".join(f"{v:.3f}" for v in values)


COLMAP_MAX_NUM_IMAGES = 2147483647  # constant COLMAP uses to encode an image pair as one id
SEQUENTIAL_ABOVE = 15  # more images than this: sequential matching (neighbours overlap), not exhaustive

# Auto-tuning cascade: configurations are tried in order and the first that gives a model wins, so a
# difficult capture (low texture, little overlap, close-ups) needs no manual parameter changes.
# Keys follow COLMAP's names; "extra" holds Mapper.* options. Beyond the basic keys, "peak_threshold"
# (lower finds features on blank walls), "guided_matching" and "force_best_pair" (start the mapper from
# the best-matched image pair) belong to the harder configurations.
COLMAP_CONFIGS: list[dict] = [
    {
        "name": "strict",
        "max_num_features": 8192,
        "max_image_size": 3200,
        "camera_model": "SIMPLE_RADIAL",
        "single_camera": True,
        "matcher": "exhaustive",
        "min_num_matches": 15,
        "min_model_size": 3,
    },
    {
        "name": "relaxed",
        "max_num_features": 16384,
        "max_image_size": 4000,
        "camera_model": "SIMPLE_PINHOLE",
        "single_camera": True,
        "matcher": "exhaustive",
        "min_num_matches": 5,
        "min_model_size": 2,
        "peak_threshold": 0.002,
        "guided_matching": True,
    },
    {
        "name": "aggressive",
        "max_num_features": 32768,
        "max_image_size": 4000,
        "camera_model": "SIMPLE_PINHOLE",
        "single_camera": False,  # per-image camera parameters
        "matcher": "exhaustive",
        "min_num_matches": 3,
        "min_model_size": 2,
        "peak_threshold": 0.002,
        "guided_matching": True,
        "extra": {
            "Mapper.init_min_num_inliers": 10,
            "Mapper.multiple_models": True,
            "Mapper.min_num_matches": 3,
        },
    },
    {
        "name": "desperate",
        "max_num_features": 32768,
        "max_image_size": 4000,
        "camera_model": "OPENCV",
        "single_camera": False,
        "matcher": "exhaustive",
        "min_num_matches": 2,
        "min_model_size": 2,
        "peak_threshold": 0.001,
        "guided_matching": True,
        "force_best_pair": True,
        "extra": {
            "Mapper.init_min_num_inliers": 5,
            "Mapper.multiple_models": True,
            "Mapper.min_num_matches": 2,
            "Mapper.init_min_tri_angle": 1.0,
            "Mapper.abs_pose_min_num_inliers": 5,
        },
    },
]
MIN_RESULT_POINTS = 10  # a configuration "succeeds" with more than this many points


@dataclass
class ColmapResult:
    """What a successful COLMAP configuration returns."""

    points: np.ndarray
    poses: list[CameraPose]
    config_name: str

    @property
    def num_points(self) -> int:
        return len(self.points)

    @property
    def num_registered(self) -> int:
        return len(self.poses)


def _effective_matcher(config: dict, requested: str, n_images: int) -> str:
    """Sequential for video-style captures or many images, else the configuration's matcher."""
    if requested == "sequential" or n_images > SEQUENTIAL_ABOVE:
        return "sequential"
    return config.get("matcher", "exhaustive")


def _cli_option(key: str, value) -> list[str]:
    return [f"--{key}", "1" if value is True else "0" if value is False else str(value)]


def _reconstruct_cli(image_dir: Path, db: Path, sparse: Path, matcher: str,
                     camera_prior: tuple[float, float, float] | None, config: dict) -> None:
    extract = ["colmap", "feature_extractor", "--database_path", str(db), "--image_path", str(image_dir),
               *_cli_option("ImageReader.single_camera", config["single_camera"]),
               *_cli_option("SiftExtraction.max_num_features", config["max_num_features"]),
               *_cli_option("SiftExtraction.max_image_size", config["max_image_size"]),
               *_cli_option("ImageReader.camera_model", config["camera_model"])]
    if "peak_threshold" in config:
        extract += _cli_option("SiftExtraction.peak_threshold", config["peak_threshold"])
    if camera_prior is not None:
        extract += _cli_option("ImageReader.camera_params", _camera_param_string(camera_prior, config["camera_model"]))
    _run(extract)
    match = ["colmap", f"{matcher}_matcher", "--database_path", str(db),
             *_cli_option("TwoViewGeometry.min_num_inliers", config["min_num_matches"])]
    if config.get("guided_matching"):
        match += _cli_option("SiftMatching.guided_matching", True)
    _run(match)
    mapper = ["colmap", "mapper", "--database_path", str(db), "--image_path", str(image_dir),
              "--output_path", str(sparse), *_cli_option("Mapper.min_model_size", config["min_model_size"]),
              *_cli_option("Mapper.min_num_matches", config["min_num_matches"])]
    for key, value in config.get("extra", {}).items():
        mapper += _cli_option(key, value)
    _run(mapper)


def _device_and_threads(pycolmap) -> tuple[object, int, bool]:
    """(Device, feature/matching threads, uses_gpu) from FLOORPLAN_COLMAP_DEVICE and _THREADS.

    FLOORPLAN_COLMAP_DEVICE = auto (default: CUDA when this pycolmap build has it) | cpu | cuda.
    FLOORPLAN_COLMAP_THREADS (default 1) sets CPU threads for feature extraction and matching;
    1 keeps repeated runs identical, more is faster but may change features slightly.
    The mapper always runs on one thread.
    """
    choice = os.environ.get("FLOORPLAN_COLMAP_DEVICE", "auto").lower()
    cuda = bool(getattr(pycolmap, "has_cuda", False))
    if choice == "cpu" or (choice == "auto" and not cuda):
        device, gpu = pycolmap.Device.cpu, False
    else:
        device, gpu = pycolmap.Device.cuda if choice == "cuda" else pycolmap.Device.auto, cuda
    try:
        threads = max(1, int(os.environ.get("FLOORPLAN_COLMAP_THREADS", "1")))
    except ValueError:
        threads = 1
    return device, threads, gpu


def _best_pair(db: Path) -> tuple[int, int] | None:
    """The image pair with the most verified inlier matches in a COLMAP database (None if there is none)."""
    import pycolmap

    try:
        database = pycolmap.Database.open(str(db))
        try:
            pair_ids, counts = database.read_two_view_geometry_num_inliers()
            keypoints = {img.image_id: database.num_keypoints_for_image(img.image_id)
                         for img in database.read_all_images()}
        finally:
            database.close()
        if len(counts):
            best = max(range(len(counts)), key=lambda i: counts[i])
            if counts[best] > 0:
                id1, id2 = divmod(int(pair_ids[best]), COLMAP_MAX_NUM_IMAGES)  # pair_id = max * id1 + id2
                return id1, id2
        if len(keypoints) >= 2:  # nothing verified: fall back to the two images with the most features
            id1, id2 = sorted(keypoints, key=keypoints.get, reverse=True)[:2]
            return int(id1), int(id2)
    except Exception as exc:  # API differences between pycolmap versions
        logger.warning("could not choose an initial image pair: %s", exc)
    return None


def _apply_extra(options, extra: dict) -> None:
    """Set 'Mapper.xxx' options on the pycolmap pipeline options (mapper sub-options, else pipeline level)."""
    for key, value in extra.items():
        name = key.split(".", 1)[-1]
        target = options.mapper if hasattr(options.mapper, name) else options
        if hasattr(target, name):
            setattr(target, name, value)
        else:
            logger.debug("pycolmap has no option %s; ignored", key)


def _reconstruct_pycolmap(image_dir: Path, db: Path, sparse: Path, matcher: str,
                          camera_prior: tuple[float, float, float] | None, config: dict) -> None:
    """pycolmap with fixed seeds and a single mapper thread, so the same images give the same model."""
    import pycolmap

    pycolmap.set_random_seed(SEED)
    device, threads, gpu = _device_and_threads(pycolmap)
    logger.info("COLMAP device: %s (%d feature/matching thread(s))", "GPU" if gpu else "CPU", threads)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = threads
    extraction.max_image_size = config["max_image_size"]
    extraction.sift.max_num_features = config["max_num_features"]
    if "peak_threshold" in config:
        extraction.sift.peak_threshold = config["peak_threshold"]
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = config["camera_model"]
    if camera_prior is not None:  # a known focal length makes registration far more reliable
        reader.camera_params = _camera_param_string(camera_prior, config["camera_model"])
    mode = pycolmap.CameraMode.SINGLE if config["single_camera"] else pycolmap.CameraMode.PER_IMAGE
    pycolmap.extract_features(str(db), str(image_dir), camera_mode=mode, reader_options=reader,
                              extraction_options=extraction, device=device)
    verification = pycolmap.TwoViewGeometryOptions()
    verification.ransac.random_seed = SEED
    verification.min_num_inliers = config["min_num_matches"]
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = threads
    matching.guided_matching = bool(config.get("guided_matching"))
    if matcher == "sequential":
        pycolmap.match_sequential(str(db), matching_options=matching, verification_options=verification,
                                  device=device)
    else:
        pycolmap.match_exhaustive(str(db), matching_options=matching, verification_options=verification,
                                  device=device)
    options = pycolmap.IncrementalPipelineOptions()
    options.random_seed = SEED
    options.num_threads = 1
    options.min_num_matches = config["min_num_matches"]
    options.min_model_size = config["min_model_size"]
    _apply_extra(options, config.get("extra", {}))
    if config.get("force_best_pair"):
        pair = _best_pair(db)
        if pair is not None:
            options.init_image_id1, options.init_image_id2 = pair
            options.mapper.init_max_forward_motion = 0.99
            logger.info("starting from image pair %s", pair)
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


def _best_model(sparse: Path, backend: str, min_images: int, min_points: int) -> tuple[np.ndarray, list[CameraPose]] | None:
    """(points, poses) of the registered model with the most images, or None if none is big enough."""
    best: tuple[np.ndarray, list[CameraPose]] | None = None
    for model in sorted(p for p in sparse.iterdir() if p.is_dir() and p.name.isdigit()):
        candidate = parse_colmap_sparse(str(_to_text_model(model, backend)))
        if best is None or len(candidate[1]) > len(best[1]):
            best = candidate
    if best is None or len(best[1]) < min_images or len(best[0]) < min_points:
        return None
    return best


def try_colmap(image_dir: Path, workspace: Path, config: dict, matcher: str = "exhaustive",
               camera_prior: tuple[float, float, float] | None = None) -> ColmapResult | None:
    """Run COLMAP once with one configuration; None when it fails or the model is too small."""
    db, sparse = workspace / "database.db", workspace / "sparse"
    if db.exists():
        db.unlink()
    shutil.rmtree(sparse, ignore_errors=True)
    sparse.mkdir(parents=True)
    backend = colmap_backend()
    n_images = len(list(image_dir.iterdir()))
    matcher = _effective_matcher(config, matcher, n_images)
    try:
        (_reconstruct_cli if backend == "cli" else _reconstruct_pycolmap)(
            image_dir, db, sparse, matcher, camera_prior, config)
        model = _best_model(sparse, backend, config["min_model_size"], MIN_RESULT_POINTS + 1)
    except ColmapError:
        raise
    except Exception as exc:  # pycolmap raises RuntimeError and friends
        logger.warning("COLMAP config '%s' raised: %s", config["name"], exc)
        return None
    if model is None:
        return None
    return ColmapResult(points=model[0], poses=model[1], config_name=config["name"])


def run_colmap_with_fallback(image_dir: str, workspace_dir: str, matcher: str = "exhaustive",
                             camera_prior: tuple[float, float, float] | None = None) -> ColmapResult | None:
    """Try each configuration of COLMAP_CONFIGS in order; return the first that succeeds, else None."""
    image_dir, workspace = Path(image_dir), Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    for config in COLMAP_CONFIGS:
        logger.info("COLMAP attempt: %s", config["name"])
        result = try_colmap(image_dir, workspace, config, matcher, camera_prior)
        if result is not None and result.num_points > MIN_RESULT_POINTS:
            logger.info("COLMAP succeeded with config: %s, %d points, %d images registered",
                        config["name"], result.num_points, result.num_registered)
            return result
        logger.warning("COLMAP config '%s' failed, trying next", config["name"])
    logger.error("All COLMAP configs failed")
    return None


def run_colmap_reconstruction(
    image_dir: str, workspace_dir: str, matcher: str = "exhaustive",
    camera_prior: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, list[CameraPose]]:
    """Reconstruct a folder of images and return (points N x 3, camera poses).

    Runs the auto-tuning cascade (run_colmap_with_fallback). Work files (database.db, sparse/) go to
    workspace_dir. matcher is "exhaustive" (photos) or "sequential" (video frames, which only overlap with
    neighbors; also used above SEQUENTIAL_ABOVE images). When COLMAP builds several disconnected models the
    one with the most images is used. The coordinates are COLMAP's own: arbitrary scale and orientation.
    camera_prior = (focal_px, cx, cy) in the images' pixels, when the intrinsics are known (for example from a
    phone's calibration log); COLMAP then starts from it. Raises ColmapError when every configuration fails.
    """
    result = run_colmap_with_fallback(image_dir, workspace_dir, matcher, camera_prior)
    if result is None:
        raise ColmapError("COLMAP produced no usable model (too few matching features between images); "
                          "all configurations failed: " + ", ".join(c["name"] for c in COLMAP_CONFIGS))
    logger.info("reconstructed %d images, %d points", result.num_registered, result.num_points)
    return result.points, result.poses


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
