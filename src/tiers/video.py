"""Video tier front-end: one handheld walkthrough clip -> PropertyIR."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from src.room_ir import CameraPose, PointCloud, PropertyIR, RoomIR
from src.tiers.colmap_utils import ColmapError, make_metric_point_cloud, run_colmap_reconstruction
from src.tiers.preprocessing import VIDEO_EXTS
from src.tiers.room_segmentation import segment_rooms

logger = logging.getLogger(__name__)

TARGET_KEYFRAMES = 100  # per video; the 50-150 range keeps COLMAP fast and the baseline wide
MAX_LONG_EDGE = 1600  # pixels; keyframes are downscaled to this
FALLBACK_STEP = 20  # frames, when the container does not report a frame count


def find_videos(capture_dir: str) -> list[Path]:
    return sorted(p for p in Path(capture_dir).rglob("*")
                  if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in VIDEO_EXTS)


def extract_keyframes(video_path: str, out_dir: str, prefix: str = "frame",
                      target: int = TARGET_KEYFRAMES) -> list[str]:
    """Save about `target` sharp, evenly spaced JPEG keyframes of a video; return their paths.

    The video is cut into equal windows and the sharpest frame (variance of the Laplacian)
    of each window is kept, which avoids motion-blurred frames that SfM cannot match.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ColmapError(f"cannot open video {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(total / target)) if total > 0 else FALLBACK_STEP
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    best_frame, best_score = None, -1.0

    def flush() -> None:
        nonlocal best_frame, best_score
        if best_frame is not None:
            h, w = best_frame.shape[:2]
            if max(h, w) > MAX_LONG_EDGE:
                s = MAX_LONG_EDGE / max(h, w)
                best_frame = cv2.resize(best_frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            path = out / f"{prefix}_{len(saved):04d}.jpg"
            cv2.imwrite(str(path), best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append(str(path))
        best_frame, best_score = None, -1.0

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        score = float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        if score > best_score:
            best_frame, best_score = frame, score
        index += 1
        if index % step == 0:
            flush()
    flush()
    cap.release()
    logger.info("%s: %d frames -> %d keyframes (window %d)", Path(video_path).name, index, len(saved), step)
    return saved


def _camera_center(pose: CameraPose) -> np.ndarray:
    return -pose.rotation.T @ pose.translation


def _assign_poses(rooms: list[RoomIR], poses: list[CameraPose], frames: dict[str, str]) -> None:
    """Give each room the keyframes whose camera sits inside (else nearest to) its footprint."""
    boxes = [(r.point_cloud.points[:, :2].min(axis=0), r.point_cloud.points[:, :2].max(axis=0)) for r in rooms]
    for pose in poses:
        xy = _camera_center(pose)[:2]
        inside = [i for i, (lo, hi) in enumerate(boxes) if np.all(xy >= lo) and np.all(xy <= hi)]
        pool = inside or range(len(rooms))
        target = min(pool, key=lambda i: np.linalg.norm(xy - (boxes[i][0] + boxes[i][1]) / 2))
        rooms[target].camera_poses.append(pose)
        rooms[target].images.append(frames[Path(pose.image_path).name])


def process_video(capture_dir: str) -> PropertyIR:
    """Extract keyframes from the walkthrough, reconstruct with COLMAP, split into rooms.

    Keyframes (~100 per video) go through the same COLMAP helpers as the photo tier
    (sequential matching, since neighbors overlap), then the single point cloud is split
    into rooms like the LiDAR tier. Tier is "video".
    """
    videos = find_videos(capture_dir)
    if not videos:
        raise FileNotFoundError(f"no .mp4 or .mov file in {capture_dir}")
    keep = bool(os.environ.get("FLOORPLAN_KEEP_WORKSPACE"))
    work = Path(tempfile.mkdtemp(prefix="floorplan-colmap-"))
    try:
        frame_dir = work / "images"
        for i, video in enumerate(videos):
            extract_keyframes(str(video), str(frame_dir), prefix=f"v{i}")
        frames = {Path(p).name: p for p in map(str, sorted(frame_dir.iterdir()))}
        if len(frames) < 3:
            raise ColmapError(f"only {len(frames)} keyframes extracted; is the video empty?")

        points, poses = run_colmap_reconstruction(str(frame_dir), str(work), matcher="sequential")
        cloud, poses, meta = make_metric_point_cloud(points, poses)

        # Keep the frames after the temp workspace goes away.
        persistent = Path(tempfile.mkdtemp(prefix="floorplan-frames-")) if not keep else frame_dir
        if not keep:
            for name, path in frames.items():
                shutil.copy2(path, persistent / name)
            frames = {name: str(persistent / name) for name in frames}
    finally:
        if keep:
            logger.info("COLMAP workspace kept in %s", work)
        else:
            shutil.rmtree(work, ignore_errors=True)

    rooms: list[RoomIR] = []
    for index, room_cloud in enumerate(segment_rooms(cloud), start=1):
        extent = room_cloud.points.max(axis=0) - room_cloud.points.min(axis=0)
        room = RoomIR(room_id=f"room-{index}", point_cloud=room_cloud, tier="video",
                      point_density=len(room_cloud) / float(max(extent.prod(), 1e-6)))
        room.metadata.update(meta, registered_images=len(poses))
        rooms.append(room)
    _assign_poses(rooms, poses, frames)
    return PropertyIR(rooms=rooms, tier="video", capture_dir=str(capture_dir))
