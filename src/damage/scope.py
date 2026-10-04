"""Scope line items from damage regions and concealed-damage flags."""

from __future__ import annotations

from src.damage.surface_projection import ProjectedDamage
from src.models import ConcealedDamageFlag, Measurement, ScopeLineItem

SCOPE_TABLE = {
    "crack": ("Patch, fill, and repaint. Estimated area: {area:.2f} m²", "medium"),
    "water_stain": ("Investigate moisture source. Remediate, dry, and repaint. Area: {area:.2f} m²", "high"),
    "mold": ("Professional mold remediation required. Affected area: {area:.2f} m²", "high"),
    "hole": ("Patch drywall, finish, and repaint. Area: {area:.2f} m²", "medium"),
    "peeling_paint": ("Scrape, prime, and repaint. Area: {area:.2f} m²", "low"),
}


def area_measurement(width: float, height: float, extent_error: float) -> Measurement:
    """width * height in m2, with the interval from +/- extent_error on each side."""
    area = width * height
    half = width * extent_error + height * extent_error + extent_error ** 2
    return Measurement(value=float(area), confidence_low=float(max(0.0, area - half)),
                       confidence_high=float(area + half), unit="m2")


def generate_scope(
    damages: list[ProjectedDamage], concealed_flags: list[ConcealedDamageFlag],
    extent_error: float = 0.05,
) -> list[ScopeLineItem]:
    """Map each damage class to a repair description and priority via a lookup
    table, and add a high-priority 'further investigation' item per concealed flag.
    extent_error is the +/- error in meters on each damage dimension (already tier-scaled).
    """
    areas = {d.id: d.extent_width * d.extent_height for d in damages}
    surfaces = {d.id: d for d in damages}
    items: list[ScopeLineItem] = []
    for d in damages:
        template, priority = SCOPE_TABLE[d.damage_detection.damage_class]
        items.append(ScopeLineItem(
            id=f"scope-{len(items)}", surface_id=d.surface_id, damage_region_id=d.id,
            description=template.format(area=areas[d.id]),
            estimated_area=area_measurement(d.extent_width, d.extent_height, extent_error),
            priority=priority))
    for flag in concealed_flags:
        evidence = [surfaces[e] for e in flag.evidence if e in surfaces]
        area = sum(areas[d.id] for d in evidence)
        half = sum(d.extent_width * extent_error + d.extent_height * extent_error + extent_error ** 2
                   for d in evidence)
        items.append(ScopeLineItem(
            id=f"scope-{len(items)}", surface_id=flag.surface_id,
            damage_region_id=flag.evidence[0] if flag.evidence else flag.id,
            description=f"Further investigation recommended: {flag.recommended_action}",
            estimated_area=Measurement(value=float(area), confidence_low=float(max(0.0, area - half)),
                                       confidence_high=float(area + half), unit="m2"),
            priority="high"))
    return items
