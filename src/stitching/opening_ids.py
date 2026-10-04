"""Opening ids shared by the stitchers and the report builder."""

from __future__ import annotations

from src.room_ir import RoomIR


def opening_id(room: RoomIR, opening) -> str:
    """The id the report gives this opening: <room>/<type>-<index among openings of that type>."""
    same = [o for o in room.openings if o.type == opening.type]
    return f"{room.room_id}/{opening.type}-{same.index(opening)}"
