"""Project image-space damage onto wall surfaces in metric units."""

from __future__ import annotations

from dataclasses import dataclass

from src.damage.detection import DamageDetection
from src.room_ir import RoomIR


@dataclass
class ProjectedDamage:
    """A damage detection placed on a surface, in meters."""

    damage_detection: DamageDetection
    room_id: str
    surface_id: str
    location_on_surface: tuple[float, float]  # (u along surface, v height), meters
    extent_width: float
    extent_height: float


def project_damage_to_surfaces(
    detections: list[DamageDetection], rooms: list[RoomIR]
) -> list[ProjectedDamage]:
    """Assign each detection to a room wall and convert its pixel size to meters.

    Casts a ray through the bbox center from the image's camera pose, takes the
    nearest wall intersection, and scales pixel size by distance / focal length.
    Without poses, assumes the camera is 1.5 m from the nearest wall.
    """
    raise NotImplementedError("Not yet implemented")
