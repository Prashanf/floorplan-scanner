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
from src.tiers.preprocessing import (CONVERT_IMAGE_EXTS, list_room_images, prepare_images_for_colmap,
                                     restore_pose)
from src.tiers.single_image import estimate_room_from_images

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
    # Every time, before COLMAP: EXIF orientation applied, portrait turned to landscape, oversize images
    # downscaled, files renamed 000.jpg, 001.jpg, ... Originals stay untouched.
    image_dir = workspace / "images"
    prepared = prepare_images_for_colmap(images, str(image_dir))
    if len(prepared) < 2:
        raise ColmapError(f"only {len(prepared)} readable image(s)")
    by_name = {p.name: p for p in prepared}

    points, poses = run_colmap_reconstruction(str(image_dir), str(workspace), matcher="exhaustive")
    for pose in poses:  # back to the user's files: original orientation and size, so "up" is up again
        item = by_name[Path(pose.image_path).name]
        pose.rotation, pose.translation, pose.intrinsics = restore_pose(
            pose.rotation, pose.translation, pose.intrinsics, item)
        pose.image_path = item.source
    cloud, poses, meta = make_metric_point_cloud(points, poses)
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
    A room whose reconstruction fails gets a rough single-image estimate when its best image looks
    like a room (flagged, +/-50 %), otherwise it is skipped with a warning. Never raises for bad images:
    with no rooms at all the pipeline writes a report with room_count 0 and the warnings.
    """
    root = Path(capture_dir)
    room_dirs = sorted((p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")),
                       key=_natural_key)
    keep = bool(os.environ.get("FLOORPLAN_KEEP_WORKSPACE"))
    work_root = Path(tempfile.mkdtemp(prefix="floorplan-colmap-"))
    rooms: list[RoomIR] = []
    warnings: list[str] = []
    try:
        for room_dir in room_dirs:
            try:
                rooms.append(_reconstruct_room(room_dir, work_root / room_dir.name))
                logger.info("%s: %s", room_dir.name, rooms[-1].metadata)
                continue
            except Exception as exc:  # COLMAP failures, a model too small to align, unreadable images
                reason = str(exc)
                logger.warning("reconstruction failed for %s: %s", room_dir.name, reason)
                print(f"warning: reconstruction failed for {room_dir.name}: {reason}", flush=True)
            # Last resort: a rough single-image estimate, only if the image looks like a room.
            fallback = estimate_room_from_images(_usable_images(room_dir), room_dir.name)
            if fallback is not None:
                logger.warning("Fallback: single-image room estimate for %s", room_dir.name)
                print(f"warning: Fallback: single-image room estimate for {room_dir.name}", flush=True)
                rooms.append(fallback)
                warnings.append(
                    f"{room_dir.name}: COLMAP reconstruction failed (insufficient feature matches between "
                    "images); a rough single-image estimate with typical-room dimensions was used instead "
                    "(about +/-50% intervals). It is not a measurement.")
            else:
                warnings.append(
                    f"{room_dir.name}: COLMAP reconstruction failed (insufficient feature matches between "
                    "images, capture may have too little visual overlap) and no image looked like a room, "
                    "so the room is missing from the plan.")
    finally:
        if keep:
            logger.info("COLMAP workspaces kept in %s", work_root)
        else:
            shutil.rmtree(work_root, ignore_errors=True)
    return PropertyIR(rooms=rooms, tier="photo", capture_dir=str(capture_dir), warnings=warnings)
