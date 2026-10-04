"""Classical-CV damage detection (cracks, water stains, mold, peeling paint, holes)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DamageDetection:
    """One detected damage region in image space."""

    image_path: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    damage_class: str  # crack | water_stain | mold | hole | peeling_paint
    confidence: float
    pixel_area: int


def detect_damage(images: list[str]) -> list[DamageDetection]:
    """Detect visible damage in each image with OpenCV heuristics.

    Canny + linearity for cracks, HSV ranges for water stains and mold,
    Laplacian variance for peeling paint, dark circular regions for holes;
    overlapping same-class boxes are merged by non-maximum suppression.
    """
    raise NotImplementedError("Not yet implemented")
