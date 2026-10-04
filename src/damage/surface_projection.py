"""Project image-space damage onto wall surfaces in metric units."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.damage.detection import DamageDetection
from src.geometry.ceiling import find_floor_and_ceiling
from src.room_ir import CameraPose, RoomIR

log = logging.getLogger("floorplan.damage")

FALLBACK_DISTANCE = 1.5  # m, assumed camera-to-wall distance when no pose exists
FALLBACK_FOCAL_FRACTION = 0.75  # focal length as a fraction of the long image side (~26 mm-equivalent lens)
HEIGHT_MARGIN = 0.05  # m, tolerance when a ray hits just below the floor or above the ceiling


@dataclass
class ProjectedDamage:
    """A damage detection placed on a surface, in meters."""

    damage_detection: DamageDetection
    room_id: str
    surface_id: str
    location_on_surface: tuple[float, float]  # (u along surface, v height), meters
    extent_width: float
    extent_height: float
    id: str = ""  # damage-N, assigned in output order
    crack_width_m: Optional[float] = None  # cracks only: line thickness in meters
    used_fallback: bool = False  # True when no camera pose was available


def _key(path: str) -> str:
    return os.path.abspath(path)


def _scaled_intrinsics(k: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """K from SfM refers to the image COLMAP saw; rescale if the source image has another size."""
    width, height = image_size
    if width <= 0 or k[0, 2] <= 0:
        return k
    ratio = width / (2.0 * k[0, 2])
    if abs(ratio - 1.0) <= 0.05:
        return k
    scaled = k.copy()
    scaled[0] *= ratio
    scaled[1] *= ratio
    return scaled


def _ray_wall_hit(
    origin: np.ndarray, direction: np.ndarray, room: RoomIR, floor_z: float, height: float
) -> Optional[tuple[int, float, float, float]]:
    """Nearest wall hit of origin + s * direction: (wall index, s, u along wall, v above floor)."""
    best: Optional[tuple[int, float, float, float]] = None
    d_xy = direction[:2]
    for i, seg in enumerate(room.wall_segments):
        a = np.array(seg.start, dtype=float)
        edge = np.array(seg.end, dtype=float) - a
        # origin_xy + s * d_xy = a + q * edge
        matrix = np.array([[d_xy[0], -edge[0]], [d_xy[1], -edge[1]]])
        if abs(np.linalg.det(matrix)) < 1e-9:
            continue
        s, q = np.linalg.solve(matrix, a - origin[:2])
        if s <= 1e-6 or not 0.0 <= q <= 1.0:
            continue
        v = origin[2] + s * direction[2] - floor_z
        if not -HEIGHT_MARGIN <= v <= height + HEIGHT_MARGIN:
            continue
        if best is None or s < best[1]:
            best = (i, float(s), float(q * seg.length), float(min(max(v, 0.0), height)))
    return best


def _project_with_pose(
    det: DamageDetection, room: RoomIR, pose: CameraPose, floor_z: float, height: float
) -> Optional[ProjectedDamage]:
    k = _scaled_intrinsics(np.asarray(pose.intrinsics, dtype=float), det.image_size)
    rotation = np.asarray(pose.rotation, dtype=float)
    translation = np.asarray(pose.translation, dtype=float)
    center = -rotation.T @ translation  # COLMAP world-to-camera: x_cam = R x_world + t
    x1, y1, x2, y2 = det.bbox
    pixel = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, 1.0])
    direction = rotation.T @ np.linalg.solve(k, pixel)  # camera z = 1, so the ray parameter s is the depth
    hit = _ray_wall_hit(center, direction, room, floor_z, height)
    if hit is None:
        return None
    index, depth, u, v = hit
    fx, fy = k[0, 0], k[1, 1]
    crack_width = None if det.thickness_px is None else det.thickness_px * depth / fx
    return ProjectedDamage(
        damage_detection=det, room_id=room.room_id, surface_id=f"{room.room_id}/wall-{index}",
        location_on_surface=(u, v), extent_width=(x2 - x1) * depth / fx,
        extent_height=(y2 - y1) * depth / fy, crack_width_m=crack_width)


def _project_fallback(det: DamageDetection, room: RoomIR, floor_z: float, height: float) -> ProjectedDamage:
    """No pose: assume the camera faces the room's longest wall from 1.5 m, eye height 1.2 m."""
    index = int(np.argmax([seg.length for seg in room.wall_segments]))
    wall_length = room.wall_segments[index].length
    width_px, height_px = det.image_size if det.image_size[0] > 0 else (det.bbox[2], det.bbox[3])
    focal = FALLBACK_FOCAL_FRACTION * max(width_px, height_px)
    x1, y1, x2, y2 = det.bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    scale = FALLBACK_DISTANCE / focal
    u = float(np.clip(wall_length / 2.0 + (cx - width_px / 2.0) * scale, 0.0, wall_length))
    v = float(np.clip(1.2 - (cy - height_px / 2.0) * scale, 0.0, height))
    crack_width = None if det.thickness_px is None else det.thickness_px * scale
    return ProjectedDamage(
        damage_detection=det, room_id=room.room_id, surface_id=f"{room.room_id}/wall-{index}",
        location_on_surface=(u, v), extent_width=(x2 - x1) * scale, extent_height=(y2 - y1) * scale,
        crack_width_m=crack_width, used_fallback=True)


def project_damage_to_surfaces(
    detections: list[DamageDetection], rooms: list[RoomIR]
) -> list[ProjectedDamage]:
    """Assign each detection to a room wall and convert its pixel size to meters.

    Casts a ray through the bbox center from the image's camera pose, takes the
    nearest wall intersection, and scales pixel size by distance / focal length.
    Without poses, assumes the camera is 1.5 m from the nearest wall. Detections whose
    ray hits no wall (floor, ceiling, outside the fitted room) are dropped.
    Rooms must be in their own frame (walls and poses before stitching transforms).
    """
    owner: dict[str, RoomIR] = {}
    poses: dict[str, CameraPose] = {}
    levels: dict[str, tuple[float, float]] = {}
    for room in rooms:
        if not room.wall_segments:
            continue
        for image in room.images:
            owner.setdefault(_key(image), room)
        for pose in room.camera_poses:
            poses.setdefault(_key(pose.image_path), pose)
        try:
            found = find_floor_and_ceiling(room.point_cloud.points[:, 2])
            levels[room.room_id] = (found.floor_z, room.ceiling_height or found.height)
        except ValueError:
            levels[room.room_id] = (float(room.point_cloud.points[:, 2].min()), room.ceiling_height or 2.5)

    projected: list[ProjectedDamage] = []
    for det in detections:
        key = _key(det.image_path)
        room = owner.get(key)
        if room is None:
            log.debug("no room owns image %s; detection dropped", det.image_path)
            continue
        floor_z, height = levels[room.room_id]
        pose = poses.get(key)
        item = (_project_with_pose(det, room, pose, floor_z, height) if pose is not None
                else _project_fallback(det, room, floor_z, height))
        if item is None:
            log.debug("%s in %s: ray hit no wall; dropped", det.damage_class, det.image_path)
            continue
        item.id = f"damage-{len(projected)}"
        projected.append(item)
    return projected
