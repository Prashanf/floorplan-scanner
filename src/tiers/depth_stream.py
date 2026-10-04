"""Raw iPhone LiDAR logs (depth PNG + confidence + ARKit odometry + intrinsics) -> one point cloud.

Layout of one capture folder (as written by Stray Scanner style logging apps):

    depth/000000.png ...   16-bit PNG, millimetres, usually 256 x 192
    confidence/000000.png  0 (low), 1 (medium), 2 (high) per depth pixel
    odometry.csv           timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, ...
    camera_matrix.csv      3 x 3 RGB intrinsics (fallback when odometry has no fx..cy)
    rgb.mp4, imu.csv       used elsewhere (video tier) or not at all

Each depth pixel is back-projected with the intrinsics scaled to the depth resolution and
moved to the ARKit world frame with the frame's pose. The world frame is gravity aligned
with +Y up, which is what the LiDAR tier's up-axis handling expects. Poses are camera to
world with the OpenCV camera convention (x right, y down, z forward); this was checked on
real captures by the sharpness of the floor plane.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from src import config as cfg

logger = logging.getLogger(__name__)

MIN_DEPTH = cfg.DEPTH_MIN
MAX_DEPTH = cfg.DEPTH_MAX
MIN_CONFIDENCE = cfg.DEPTH_MIN_CONFIDENCE
TARGET_FRAMES = cfg.DEPTH_TARGET_FRAMES
VOXEL_SIZE = cfg.VOXEL_SIZE
CHUNK_FRAMES = 40  # frames merged and downsampled together to bound memory
DEPTH_UNIT = 1000.0  # millimetres per metre


def find_depth_streams(capture_dir: str) -> list[Path]:
    """Folders under capture_dir that hold a depth stream (depth/ + odometry.csv)."""
    root = Path(capture_dir)
    if not root.is_dir():
        return []
    return sorted(p.parent for p in root.rglob("odometry.csv")
                  if (p.parent / "depth").is_dir() and not any(part.startswith(".") for part in p.relative_to(root).parts))


def _read_odometry(path: Path) -> dict[str, np.ndarray]:
    """Columns of odometry.csv by name (the header has spaces after the commas)."""
    with open(path, newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = [h.strip() for h in next(reader)]
        rows = [row for row in reader if row]
    columns = {}
    for i, name in enumerate(header):
        if name == "frame":
            columns[name] = [row[i].strip() for row in rows]
        else:
            columns[name] = np.array([float(row[i]) if row[i].strip() else np.nan for row in rows])
    return columns


def _fallback_intrinsics(stream_dir: Path) -> tuple[float, float, float, float]:
    k = np.loadtxt(stream_dir / "camera_matrix.csv", delimiter=",")
    return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])


def _rgb_width(stream_dir: Path, cx: float) -> float:
    """Pixel width the intrinsics refer to (the RGB frame): the video's, else twice the principal point."""
    import cv2

    video = stream_dir / "rgb.mp4"
    if video.is_file():
        cap = cv2.VideoCapture(str(video))
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        finally:
            cap.release()
        if width > 0:
            return float(width)
    return 2.0 * cx


def load_depth_stream(stream_dir: str, target_frames: int = TARGET_FRAMES, min_confidence: int = MIN_CONFIDENCE,
                      min_depth: float = MIN_DEPTH, max_depth: float = MAX_DEPTH,
                      voxel_size: float = VOXEL_SIZE) -> np.ndarray:
    """Back-project a depth stream to an (N, 3) float array of world points, +Y up, metres.

    About `target_frames` evenly spaced frames are used. Pixels with confidence below
    `min_confidence` or depth outside [min_depth, max_depth] are dropped, and the result
    is voxel-downsampled to `voxel_size`. Raises FileNotFoundError / ValueError when the
    stream is incomplete or yields no points.
    """
    import cv2
    import open3d as o3d
    from scipy.spatial.transform import Rotation

    root = Path(stream_dir)
    odometry = _read_odometry(root / "odometry.csv")
    frames = odometry["frame"]
    if not frames:
        raise ValueError(f"{root / 'odometry.csv'} has no rows")
    positions = np.column_stack([odometry["x"], odometry["y"], odometry["z"]])
    rotations = Rotation.from_quat(np.column_stack([odometry["qx"], odometry["qy"], odometry["qz"], odometry["qw"]]))
    if all(k in odometry for k in ("fx", "fy", "cx", "cy")):
        intrinsics = np.column_stack([odometry["fx"], odometry["fy"], odometry["cx"], odometry["cy"]])
    else:
        intrinsics = np.tile(_fallback_intrinsics(root), (len(frames), 1))
    rgb_width = _rgb_width(root, float(np.nanmedian(intrinsics[:, 2])))

    step = max(1, len(frames) // max(1, target_frames))
    chosen = range(0, len(frames), step)
    merged = o3d.geometry.PointCloud()
    pending: list[np.ndarray] = []
    used = 0
    grid = None

    def flush() -> None:
        nonlocal merged, pending
        if not pending:
            return
        chunk = o3d.geometry.PointCloud()
        chunk.points = o3d.utility.Vector3dVector(np.vstack(pending))
        merged += chunk.voxel_down_sample(voxel_size)
        merged = merged.voxel_down_sample(voxel_size)
        pending = []

    for n, i in enumerate(chosen):
        depth_path = root / "depth" / f"{frames[i]}.png"
        conf_path = root / "confidence" / f"{frames[i]}.png"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None or not np.all(np.isfinite(positions[i])) or not np.all(np.isfinite(intrinsics[i])):
            continue
        depth = depth.astype(np.float64) / DEPTH_UNIT
        conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
        keep = (depth >= min_depth) & (depth <= max_depth)
        if conf is not None and conf.shape == depth.shape:
            keep &= conf >= min_confidence
        if not keep.any():
            continue
        h, w = depth.shape
        if grid is None or grid[0].shape != depth.shape:
            grid = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        scale = w / rgb_width  # intrinsics are for the RGB frame
        fx, fy, cx, cy = intrinsics[i] * scale
        z = depth[keep]
        camera = np.column_stack([(grid[0][keep] - cx) * z / fx, (grid[1][keep] - cy) * z / fy, z])
        pending.append(rotations[i].apply(camera) + positions[i])
        used += 1
        if (n + 1) % CHUNK_FRAMES == 0:
            flush()
    flush()
    if not merged.has_points():
        raise ValueError(f"no usable depth points in {root}")
    logger.info("%s: %d of %d frames -> %d points", root.name, used, len(frames), len(merged.points))
    return np.asarray(merged.points)
