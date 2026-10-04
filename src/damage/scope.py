"""Scope line items from damage regions and concealed-damage flags."""

from __future__ import annotations

from src.damage.surface_projection import ProjectedDamage
from src.models import ConcealedDamageFlag, ScopeLineItem


def generate_scope(
    damages: list[ProjectedDamage], concealed_flags: list[ConcealedDamageFlag]
) -> list[ScopeLineItem]:
    """Map each damage class to a repair description and priority via a lookup
    table, and add a high-priority 'further investigation' item per concealed flag.
    """
    raise NotImplementedError("Not yet implemented")
