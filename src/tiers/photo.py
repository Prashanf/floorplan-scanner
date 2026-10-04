"""Photo tier front-end: per-room photo folders (2-8 stills each) -> PropertyIR."""

from __future__ import annotations

from src.room_ir import PropertyIR


def process_photos(capture_dir: str) -> PropertyIR:
    """Reconstruct each room-N/ folder with COLMAP and build a PropertyIR.

    Per room: run SfM, parse the sparse model, recover metric scale from a
    door-width prior, and create a RoomIR with tier "photo". A room whose
    reconstruction fails is skipped with a warning, not fatal.
    """
    raise NotImplementedError("Not yet implemented")
