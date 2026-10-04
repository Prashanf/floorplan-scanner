"""Ceiling height from the Z histogram of the point cloud."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.room_ir import PointCloud

BIN_SIZE = 0.02  # meters
END_FRACTION = 0.25  # floor lives in the lower 25% of the Z range, ceiling in the upper 25%
MIN_CEILING_HEIGHT = 2.0  # metres; a "ceiling" lower than this is furniture or the top of a partial wall
MIN_CEILING_TO_FLOOR_DENSITY = 0.1  # the ceiling peak must hold at least this share of the floor peak's points


@dataclass
class FloorCeiling:
    """Floor and ceiling levels found from the Z histogram."""

    floor_z: float
    ceiling_z: float
    floor_peak_count: int
    ceiling_peak_count: int
    total_points: int

    @property
    def height(self) -> float:
        return self.ceiling_z - self.floor_z

    @property
    def ceiling_observed(self) -> bool:
        """True when a ceiling plane was actually scanned.

        A scan that never pointed up still has a "tallest bin in the upper 25%": the top of
        the highest wall or a cupboard. A ceiling is a plane, so it must be at least
        MIN_CEILING_HEIGHT above the floor and hold a real share of the floor's point density.
        """
        return (self.height >= MIN_CEILING_HEIGHT
                and self.ceiling_peak_count >= MIN_CEILING_TO_FLOOR_DENSITY * self.floor_peak_count)


def find_floor_and_ceiling(z: np.ndarray, bin_size: float = BIN_SIZE) -> FloorCeiling:
    """Locate floor and ceiling as the tallest histogram bins in the lower/upper 25% of Z.

    The level is the mean Z of points within one bin of the peak center, which
    removes the half-bin quantization of the raw bin center.
    """
    z = np.asarray(z, dtype=float)
    if z.size == 0:
        raise ValueError("empty point cloud")
    z_min, z_max = float(z.min()), float(z.max())
    span = z_max - z_min
    if span < 4 * bin_size:
        raise ValueError(f"point cloud has no vertical extent (Z range {span:.3f} m)")

    n_bins = int(np.ceil(span / bin_size)) + 1
    edges = z_min + bin_size * np.arange(n_bins + 1)
    counts, _ = np.histogram(z, bins=edges)
    centers = edges[:-1] + bin_size / 2

    lower = centers <= z_min + END_FRACTION * span
    upper = centers >= z_max - END_FRACTION * span
    floor_bin = int(np.argmax(np.where(lower, counts, -1)))
    ceiling_bin = int(np.argmax(np.where(upper, counts, -1)))

    def level(bin_index: int) -> float:
        near = z[np.abs(z - centers[bin_index]) <= bin_size]
        return float(near.mean()) if near.size else float(centers[bin_index])

    return FloorCeiling(
        floor_z=level(floor_bin),
        ceiling_z=level(ceiling_bin),
        floor_peak_count=int(counts[floor_bin]),
        ceiling_peak_count=int(counts[ceiling_bin]),
        total_points=int(z.size),
    )


def detect_ceiling_height(point_cloud: PointCloud) -> tuple[float, float]:
    """Return (ceiling_height_m, confidence_proxy).

    Histograms Z (0.02 m bins), takes the tallest bin in the lower 25% as the
    floor and in the upper 25% as the ceiling. Confidence proxy is
    min(floor_peak_count, ceiling_peak_count) / total_points.
    """
    levels = find_floor_and_ceiling(point_cloud.points[:, 2])
    proxy = min(levels.floor_peak_count, levels.ceiling_peak_count) / levels.total_points
    return levels.height, float(proxy)


def is_ceiling_observed(point_cloud: PointCloud) -> bool:
    """Whether the scan contains a ceiling plane (see FloorCeiling.ceiling_observed)."""
    return find_floor_and_ceiling(point_cloud.points[:, 2]).ceiling_observed


def observed_top_height(point_cloud: PointCloud) -> float:
    """Height of the highest well-supported points above the floor: a lower bound on the
    ceiling height when no ceiling was scanned (99.5th percentile, robust to stray points)."""
    levels = find_floor_and_ceiling(point_cloud.points[:, 2])
    return float(np.percentile(point_cloud.points[:, 2], 99.5) - levels.floor_z)
