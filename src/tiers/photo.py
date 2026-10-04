"""Photo tier front-end: per-room photo folders (2-8 stills each) -> PropertyIR."""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

from src.room_ir import PropertyIR, RoomIR
from src.tiers.colmap_utils import ColmapError, make_metric_point_cloud, run_colmap_reconstruction
from src.tiers.preprocessing import CONVERT_IMAGE_EXTS, list_room_images, stage_upright_copy

logger = logging.getLogger(__name__)


def _natural_key(path: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def _usable_images(room_dir: Path) -> list[str]:
    """JPEG/PNG images of a room (other formats were converted to JPEG during preprocessing)."""
    return [p for p in list_room_images(str(room_dir)) if Path(p).suffix.lower() not in CONVERT_IMAGE_EXTS]


def _reconstruct_room(room_dir: Path, workspace: Path) -> RoomIR:
    images = _usable_images(room_dir)
    if len(images) < 2:
        raise ColmapError(f"only {len(images)} usable image(s)")
    image_dir = workspace / "images"
    image_dir.mkdir(parents=True)
    for i, src in enumerate(images):  # staged under unique names; originals stay untouched
        stage_upright_copy(src, str(image_dir / f"{i:03d}{Path(src).suffix.lower()}"))

    points, poses = run_colmap_reconstruction(str(image_dir), str(workspace), matcher="exhaustive")
    cloud, poses, meta = make_metric_point_cloud(points, poses)
    for pose in poses:  # point back at the user's files
        pose.image_path = images[int(Path(pose.image_path).stem)]
    extent = cloud.points.max(axis=0) - cloud.points.min(axis=0)
    room = RoomIR(
        room_id=room_dir.name, point_cloud=cloud, camera_poses=poses, images=images, tier="photo",
        point_density=len(cloud) / float(max(extent.prod(), 1e-6)))
    room.metadata.update(meta, registered_images=len(poses), input_images=len(images))
    return room


def process_photos(capture_dir: str) -> PropertyIR:
    """Reconstruct each room-N/ folder with COLMAP and build a PropertyIR.

    Per room: run SfM, rotate to Z-up, scale to meters (floor/ceiling prior, then a
    0.86 m door if one is found), and create a RoomIR with tier "photo". Rooms are
    reconstructed independently, so each has its own coordinate frame until stitching.
    A room whose reconstruction fails is skipped with a warning, not fatal.
    """
    root = Path(capture_dir)
    room_dirs = sorted((p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")),
                       key=_natural_key)
    keep = bool(os.environ.get("FLOORPLAN_KEEP_WORKSPACE"))
    work_root = Path(tempfile.mkdtemp(prefix="floorplan-colmap-"))
    rooms: list[RoomIR] = []
    try:
        for room_dir in room_dirs:
            try:
                rooms.append(_reconstruct_room(room_dir, work_root / room_dir.name))
                logger.info("%s: %s", room_dir.name, rooms[-1].metadata)
            except (ColmapError, ValueError) as exc:
                logger.warning("skipping %s: %s", room_dir.name, exc)
                print(f"warning: skipping {room_dir.name}: {exc}", flush=True)
    finally:
        if keep:
            logger.info("COLMAP workspaces kept in %s", work_root)
        else:
            shutil.rmtree(work_root, ignore_errors=True)
    if not rooms:
        raise RuntimeError("no room could be reconstructed; check photo overlap and lighting")
    return PropertyIR(rooms=rooms, tier="photo", capture_dir=str(capture_dir))
