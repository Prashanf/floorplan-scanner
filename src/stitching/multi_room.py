"""Whole-property stitching: adjacency graph, room transforms, overlap validation."""

from __future__ import annotations

from src.room_ir import PropertyIR


def stitch_rooms(property_ir: PropertyIR, drift_correction: bool = True) -> PropertyIR:
    """Place all rooms in one frame and fill adjacencies and room_transforms.

    LiDAR/video rooms already share a frame: link rooms whose openings lie
    within 0.5 m, optionally run drift correction, and warn on polygon
    overlaps. Photo-tier rooms are delegated to photo_stitch.stitch_photos.
    """
    raise NotImplementedError("Not yet implemented")
