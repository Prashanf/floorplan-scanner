"""Rule engine for concealed-damage flags."""

from __future__ import annotations

from src.damage.surface_projection import ProjectedDamage
from src.models import ConcealedDamageFlag
from src.room_ir import RoomIR


def check_concealed_damage(
    rooms: list[RoomIR], projected_damages: list[ProjectedDamage]
) -> list[ConcealedDamageFlag]:
    """Evaluate RULE_WATER_01, RULE_WATER_02, RULE_CRACK_01, RULE_MOLD_01 and
    RULE_CEILING_01 and return one ConcealedDamageFlag per rule firing, each
    carrying the rule id, evidence damage ids, confidence and recommended action.
    """
    raise NotImplementedError("Not yet implemented")
