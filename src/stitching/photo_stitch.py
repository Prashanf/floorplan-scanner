"""Photo-tier stitching: independent rooms joined through matching doors."""

from __future__ import annotations

from src.room_ir import RoomIR


def stitch_photos(rooms: list[RoomIR]) -> tuple[dict, list]:
    """Return (room_transforms, adjacencies) for rooms with no shared frame.

    Rooms arrive in walk order (room-1, room-2, ...). Room N is linked to room
    N+1 through the first pair of doors with similar width (within 15%), placed
    greedily so the door walls coincide. Overlaps are checked with Shapely and
    resolved by nudging.
    """
    raise NotImplementedError("Not yet implemented")
