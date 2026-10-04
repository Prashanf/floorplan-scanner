"""LiDAR tier front-end: iPhone point clouds / meshes (3D Scanner App export) or raw depth
logs (depth + odometry + intrinsics, see depth_stream.py) -> PropertyIR."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from src.room_ir import PointCloud, PropertyIR, RoomIR
from src.tiers.depth_stream import find_depth_streams, load_depth_stream
from src.tiers.room_segmentation import segment_rooms
from src import config as cfg

logger = logging.getLogger(__name__)

VOXEL_SIZE = cfg.VOXEL_SIZE
NORMAL_RADIUS = cfg.NORMAL_SEARCH_RADIUS
NORMAL_MAX_NN = cfg.NORMAL_MAX_NN
MESH_SAMPLE_DENSITY = 2000.0  # points per m2 when an .obj mesh is turned into a cloud
LIDAR_EXTS = {".ply", ".obj"}
ALIGN_TO_WALLS = cfg.ALIGN_TO_WALLS


def find_lidar_files(capture_dir: str) -> list[Path]:
    """All .ply and .obj files under capture_dir, sorted."""
    return sorted(p for p in Path(capture_dir).rglob("*")
                  if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in LIDAR_EXTS)


def _load(path: Path):
    """Load one file as an Open3D point cloud. Meshes are sampled uniformly by area."""
    import open3d as o3d

    if path.suffix.lower() == ".ply":
        cloud = o3d.io.read_point_cloud(str(path))
        if cloud.has_points():
            return cloud
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.has_triangles():
        n = max(1000, int(mesh.get_surface_area() * MESH_SAMPLE_DENSITY))
        sampled = mesh.sample_points_uniformly(number_of_points=n)
        return sampled
    cloud = o3d.geometry.PointCloud()
    cloud.points = mesh.vertices  # mesh without faces: use its vertices
    if mesh.has_vertex_colors():
        cloud.colors = mesh.vertex_colors
    return cloud


def _up_axis(points: np.ndarray) -> int:
    """Index (0, 1, 2) of the vertical axis.

    ARKit-based apps may export Y-up. A building is wider than it is tall, so if one
    axis has clearly the smallest extent (under 0.9 of the Z extent) it is taken as
    vertical; otherwise Z. Override with FLOORPLAN_UP_AXIS=x|y|z.
    """
    forced = os.environ.get("FLOORPLAN_UP_AXIS", "").lower()
    if forced in {"x", "y", "z"}:
        return "xyz".index(forced)
    lo, hi = np.percentile(points, [1, 99], axis=0)
    extent = hi - lo
    up = int(np.argmin(extent))
    return 2 if up == 2 or extent[up] > 0.9 * extent[2] else up


def _to_z_up(array: np.ndarray, up: int) -> np.ndarray:
    """Rotate (proper rotation, no mirroring) so axis `up` becomes Z."""
    if up == 2:
        return array
    if up == 1:  # Y-up: rotate +90 degrees about X, (x, y, z) -> (x, -z, y)
        return np.column_stack([array[:, 0], -array[:, 2], array[:, 1]])
    return np.column_stack([array[:, 1], array[:, 2], array[:, 0]])  # X-up: (x, y, z) -> (y, z, x)


def _manhattan_angle(points: np.ndarray, normals: np.ndarray) -> float:
    """Rotation (radians about Z, in (-pi/4, pi/4]) that makes the walls axis-aligned.

    Wall normals are horizontal, and a rectilinear building has them at 0/90/180/270 degrees
    from one base direction, so the base direction is the circular mean of 4 * angle.
    Real scans start at an arbitrary heading; the grid-based room segmentation and the
    wall snapping both work much better once the walls line up with the axes.
    """
    vertical = np.abs(normals[:, 2]) < 0.3
    if vertical.sum() < 100:
        return 0.0
    theta = np.arctan2(normals[vertical, 1], normals[vertical, 0])
    return float(np.arctan2(np.sin(4 * theta).sum(), np.cos(4 * theta).sum()) / 4)


def _rotate_z(array: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the XY part of an N x 3 array by `angle` radians about Z."""
    c, s = np.cos(angle), np.sin(angle)
    out = array.copy()
    out[:, 0] = c * array[:, 0] - s * array[:, 1]
    out[:, 1] = s * array[:, 0] + c * array[:, 1]
    return out


def process_lidar(capture_dir: str) -> PropertyIR:
    """Load .ply/.obj scans from capture_dir and build a PropertyIR.

    Merges multiple files, voxel-downsamples (0.02 m), estimates normals, and
    splits multi-room scans into one RoomIR per room (see room_segmentation).
    Tier is "lidar".
    """
    import open3d as o3d

    files = find_lidar_files(capture_dir)
    streams = find_depth_streams(capture_dir)
    if not files and not streams:
        raise FileNotFoundError(f"no .ply, .obj or raw depth capture (depth/ + odometry.csv) in {capture_dir}")

    merged = o3d.geometry.PointCloud()
    if files:
        for path in files:
            cloud = _load(path)
            logger.info("loaded %s: %d points", path.name, len(cloud.points))
            merged += cloud
        if streams:
            logger.warning("using the point clouds; ignoring raw depth capture %s", streams[0].name)
    else:
        if len(streams) > 1:
            logger.warning("%d depth captures found; each has its own world frame, using %s only",
                           len(streams), streams[0].name)
        merged.points = o3d.utility.Vector3dVector(load_depth_stream(str(streams[0])))
    if not merged.has_points():
        raise ValueError(f"no points found in {[p.name for p in files]}")

    merged = merged.voxel_down_sample(voxel_size=VOXEL_SIZE)
    merged.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN))

    points = np.asarray(merged.points)
    up = _up_axis(points)
    if up != 2:
        logger.warning("treating axis %s as up (rotating it onto Z)", "xyz"[up])
    cloud = PointCloud(
        points=_to_z_up(points, up),
        colors=np.asarray(merged.colors) if merged.has_colors() else None,
        normals=_to_z_up(np.asarray(merged.normals), up) if merged.has_normals() else None,
    )

    if cloud.normals is not None and ALIGN_TO_WALLS:
        angle = _manhattan_angle(cloud.points, cloud.normals)
        cloud = PointCloud(points=_rotate_z(cloud.points, -angle), colors=cloud.colors,
                           normals=_rotate_z(cloud.normals, -angle))
        logger.info("aligned walls to the axes (rotated %.1f deg)", np.degrees(-angle))
    else:
        angle = 0.0

    rooms: list[RoomIR] = []
    for index, room_cloud in enumerate(segment_rooms(cloud), start=1):
        extent = room_cloud.points.max(axis=0) - room_cloud.points.min(axis=0)
        volume = float(np.prod(np.maximum(extent, 1e-6)))
        rooms.append(RoomIR(
            room_id=f"room-{index}", point_cloud=room_cloud, tier="lidar",
            point_density=len(room_cloud) / volume,
            metadata={"plan_rotation_deg": float(np.degrees(-angle))}))
    logger.info("segmented %d room(s)", len(rooms))
    return PropertyIR(rooms=rooms, tier="lidar", capture_dir=str(capture_dir))
