"""Translation-only drift correction from shared-wall discrepancies."""

from __future__ import annotations

from src.room_ir import RoomIR


def correct_drift(rooms: list[RoomIR], adjacencies: list) -> dict[str, tuple[float, float, float]]:
    """Return corrected room_id -> (dx, dy, rotation) transforms.

    Finds shared wall pairs between adjacent rooms (near-parallel, within
    0.3 m, similar length), and minimizes squared midpoint/length
    discrepancies over per-room XY translations with scipy.optimize.minimize.
    The first room stays fixed; rotation is returned as 0.0.
    """
    raise NotImplementedError("Not yet implemented")
