"""Internal intermediate representation passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PointCloud:
    """3D points in meters. Z is up."""

    points: np.ndarray  # N x 3
    colors: Optional[np.ndarray] = None  # N x 3, floats in [0, 1]
    normals: Optional[np.ndarray] = None  # N x 3

    def __len__(self) -> int:
        return int(self.points.shape[0])


@dataclass
class CameraPose:
    """Camera pose for one image (world-from-camera convention is set by the tier front-end)."""

    image_path: str
    rotation: np.ndarray  # 3 x 3
    translation: np.ndarray  # (3,)
    intrinsics: np.ndarray  # 3 x 3


@dataclass
class RoomIR:
    """One room as it moves through the pipeline; geometry fields are filled stage by stage."""

    room_id: str
    point_cloud: PointCloud
    camera_poses: list[CameraPose] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    tier: str = "lidar"
    wall_segments: Optional[list] = None
    openings: Optional[list] = None
    ceiling_height: Optional[float] = None
    ceiling_observed: bool = True  # False: no ceiling plane was scanned; ceiling_height is a lower bound
    floor_polygon: Optional[list] = None
    point_density: Optional[float] = None
    metadata: dict = field(default_factory=dict)  # e.g. scale method and factor from SfM tiers


@dataclass
class PropertyIR:
    """All rooms of one capture."""

    rooms: list[RoomIR]
    tier: str
    capture_dir: str
    room_transforms: Optional[dict] = None  # room_id -> (dx, dy, rotation)
    adjacencies: Optional[list] = None
    warnings: list = field(default_factory=list)  # human-readable notes from the front-end (skipped rooms, fallbacks)
