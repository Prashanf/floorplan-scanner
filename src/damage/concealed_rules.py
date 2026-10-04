"""Rule engine for concealed-damage flags."""

from __future__ import annotations

import numpy as np

from src.damage.surface_projection import ProjectedDamage
from src.geometry.ceiling import find_floor_and_ceiling
from src.models import ConcealedDamageFlag
from src.room_ir import RoomIR

JUNCTION_DISTANCE = 0.3  # m, RULE_WATER_01: stain center this close to the floor
ALIGN_TOLERANCE = 0.2  # m, RULE_WATER_02: horizontal spread of an aligned stain group
CRACK_WIDTH_LIMIT = 0.003  # m, RULE_CRACK_01
CRACK_DIAGONAL = (30.0, 60.0)  # degrees from horizontal
CEILING_TOLERANCE = 0.02  # m, RULE_CEILING_01
CEILING_CELL = 0.5  # m, grid cell used to sample ceiling height
CEILING_BAND = 0.15  # m below the ceiling level that counts as ceiling points
CEILING_MIN_POINTS = 30

DESCRIPTIONS = {
    "RULE_WATER_01": "Water damage near floor junction",
    "RULE_WATER_02": "Vertical stain alignment",
    "RULE_CRACK_01": "Structural crack pattern",
    "RULE_MOLD_01": "Mold near moisture source",
    "RULE_CEILING_01": "Ceiling irregularity",
}
ACTIONS = {
    "RULE_WATER_01": "Investigate possible concealed water damage behind surface. Check for plumbing leaks.",
    "RULE_WATER_02": "Multiple aligned stains suggest pipe leak path behind wall.",
    "RULE_CRACK_01": "Crack pattern suggests possible structural movement. Recommend structural assessment.",
    "RULE_MOLD_01": "Mold near exterior wall suggests possible moisture intrusion through envelope.",
    "RULE_CEILING_01": "Ceiling height irregularity suggests possible structural issue above ceiling.",
}


def _area(d: ProjectedDamage) -> float:
    return d.extent_width * d.extent_height


def _wall_index(surface_id: str) -> int:
    return int(surface_id.rsplit("-", 1)[1])


def _rule_water_01(damages: list[ProjectedDamage]) -> list[tuple]:
    out = []
    for d in damages:
        if d.damage_detection.damage_class == "water_stain" and d.location_on_surface[1] <= JUNCTION_DISTANCE:
            out.append(("RULE_WATER_01", d.surface_id, [d.id], min(0.95, 0.5 + 0.3 * (_area(d) / 0.5))))
    return out


def _rule_water_02(damages: list[ProjectedDamage]) -> list[tuple]:
    by_surface: dict[str, list[ProjectedDamage]] = {}
    for d in damages:
        if d.damage_detection.damage_class == "water_stain":
            by_surface.setdefault(d.surface_id, []).append(d)
    out = []
    for surface, stains in sorted(by_surface.items()):
        stains.sort(key=lambda d: d.location_on_surface[0])
        group = [stains[0]]
        for d in stains[1:] + [None]:
            if d is not None and d.location_on_surface[0] - group[0].location_on_surface[0] <= ALIGN_TOLERANCE:
                group.append(d)
                continue
            if len(group) >= 2:
                out.append(("RULE_WATER_02", surface, [g.id for g in group], min(0.9, 0.6 + 0.1 * len(group))))
            group = [d] if d is not None else []
    return out


def _rule_crack_01(damages: list[ProjectedDamage]) -> list[tuple]:
    out = []
    for d in damages:
        det = d.damage_detection
        if det.damage_class != "crack":
            continue
        width = d.crack_width_m or 0.0
        angle = None if det.angle_deg is None else min(det.angle_deg % 180.0, 180.0 - det.angle_deg % 180.0)
        diagonal = angle is not None and CRACK_DIAGONAL[0] <= angle <= CRACK_DIAGONAL[1]
        if width > CRACK_WIDTH_LIMIT or diagonal:
            out.append(("RULE_CRACK_01", d.surface_id, [d.id], min(0.85, 0.4 + 0.3 * (width / 0.01))))
    return out


def _rule_mold_01(damages: list[ProjectedDamage], wall_counts: dict[str, int]) -> list[tuple]:
    """Exterior wall = first or last wall of the room polygon. No plumbing-fixture detector
    exists, so the 'within 1 m of a fixture' branch of the rule never fires."""
    out = []
    for d in damages:
        if d.damage_detection.damage_class != "mold":
            continue
        last = wall_counts.get(d.room_id, 0) - 1
        if _wall_index(d.surface_id) in (0, last):
            out.append(("RULE_MOLD_01", d.surface_id, [d.id], 0.5))
    return out


def _rule_ceiling_01(room: RoomIR) -> list[tuple]:
    """Ceiling height sampled on a 0.5 m grid (densest Z bin per cell); fires when the worst cell
    differs from the median cell by more than 0.02 m."""
    if room.point_cloud is None or len(room.point_cloud) == 0:
        return []
    points = room.point_cloud.points
    try:
        ceiling_z = find_floor_and_ceiling(points[:, 2]).ceiling_z
    except ValueError:
        return []
    band = points[(points[:, 2] >= ceiling_z - CEILING_BAND) & (points[:, 2] <= ceiling_z + CEILING_BAND)]
    if len(band) < CEILING_MIN_POINTS:
        return []
    cells = np.floor(band[:, :2] / CEILING_CELL).astype(int)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    heights = []
    for c in range(inverse.max() + 1):
        z = band[inverse == c, 2]
        if len(z) < CEILING_MIN_POINTS:
            continue
        # Densest 1 cm bin = the ceiling plane; wall points near the top edge spread over the band.
        counts, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.02, 0.01))
        peak = int(np.argmax(counts))
        if counts[peak] < CEILING_MIN_POINTS:
            continue
        near = z[(z >= edges[peak] - 0.01) & (z <= edges[peak + 1] + 0.01)]
        heights.append(float(near.mean()))
    if len(heights) < 2:
        return []
    deviation = float(np.max(np.abs(np.array(heights) - np.median(heights))))
    if deviation <= CEILING_TOLERANCE:
        return []
    return [("RULE_CEILING_01", f"{room.room_id}/ceiling", [], min(0.8, 0.4 + 0.3 * (deviation / 0.05)))]


def check_concealed_damage(
    rooms: list[RoomIR], projected_damages: list[ProjectedDamage]
) -> list[ConcealedDamageFlag]:
    """Evaluate RULE_WATER_01, RULE_WATER_02, RULE_CRACK_01, RULE_MOLD_01 and
    RULE_CEILING_01 and return one ConcealedDamageFlag per rule firing, each
    carrying the rule id, evidence damage ids, confidence and recommended action.
    """
    wall_counts = {r.room_id: len(r.wall_segments or []) for r in rooms}
    firings = (_rule_water_01(projected_damages) + _rule_water_02(projected_damages)
               + _rule_crack_01(projected_damages) + _rule_mold_01(projected_damages, wall_counts))
    for room in rooms:
        firings += _rule_ceiling_01(room)
    return [
        ConcealedDamageFlag(
            id=f"flag-{i}", surface_id=surface, rule_id=rule, rule_description=DESCRIPTIONS[rule],
            evidence=evidence, confidence=float(round(confidence, 3)), recommended_action=ACTIONS[rule])
        for i, (rule, surface, evidence, confidence) in enumerate(firings)
    ]
