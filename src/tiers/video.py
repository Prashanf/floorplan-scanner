"""Video tier front-end: one handheld walkthrough clip -> PropertyIR."""

from __future__ import annotations

from src.room_ir import PropertyIR


def process_video(capture_dir: str) -> PropertyIR:
    """Extract keyframes from the walkthrough, reconstruct with COLMAP, split into rooms.

    Keyframes (~50-150) go through the same COLMAP helpers as the photo tier,
    then the single point cloud is segmented into rooms like the LiDAR tier.
    Tier is "video".
    """
    raise NotImplementedError("Not yet implemented")
