"""LiDAR tier front-end: iPhone point clouds / meshes (3D Scanner App export) -> PropertyIR."""

from __future__ import annotations

from src.room_ir import PropertyIR


def process_lidar(capture_dir: str) -> PropertyIR:
    """Load .ply/.obj scans from capture_dir and build a PropertyIR.

    Merges multiple files, voxel-downsamples (0.02 m), estimates normals, and
    splits multi-room scans into one RoomIR per DBSCAN cluster of wall-height
    points. Tier is "lidar".
    """
    raise NotImplementedError("Not yet implemented")
