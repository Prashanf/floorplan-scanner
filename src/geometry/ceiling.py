"""Ceiling height from the Z histogram of the point cloud."""

from __future__ import annotations

from src.room_ir import PointCloud


def detect_ceiling_height(point_cloud: PointCloud) -> tuple[float, float]:
    """Return (ceiling_height_m, confidence_proxy).

    Histograms Z (0.02 m bins), takes the tallest bin in the lower 25% as the
    floor and in the upper 25% as the ceiling. Confidence proxy is
    min(floor_peak_count, ceiling_peak_count) / total_points.
    """
    raise NotImplementedError("Not yet implemented")
