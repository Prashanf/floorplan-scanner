"""Translation-only drift correction from shared-wall discrepancies."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

from src.room_ir import RoomIR

log = logging.getLogger("floorplan.stitch")

PARALLEL_ANGLE = np.deg2rad(10.0)
PROXIMITY = 0.3  # metres: how close midpoints must be to count as "shared"
LENGTH_RATIO = 0.20  # lengths must agree within this fraction
MAX_CORRECTION = 0.5  # metres per room: drift between rooms of one scan is small; more means a bad match
REGULARIZATION = 0.05  # pulls offsets toward zero so under-constrained rooms do not wander
WALL_THICKNESS_TOL = 0.15  # metres: two scans of one wall sit a wall thickness apart, not drift


def _wall_midpoint(seg) -> np.ndarray:
    return np.array([(seg.start[0] + seg.end[0]) / 2,
                     (seg.start[1] + seg.end[1]) / 2])


def _wall_angle(seg) -> float:
    return float(seg.direction)


def _find_shared_walls(room_a: RoomIR, room_b: RoomIR) -> list[tuple]:
    """Return pairs (seg_a, seg_b) of candidate shared walls."""
    pairs = []
    for sa in room_a.wall_segments:
        for sb in room_b.wall_segments:
            real_diff = abs((_wall_angle(sa) - _wall_angle(sb)) % np.pi)
            if real_diff > PARALLEL_ANGLE and abs(real_diff - np.pi) > PARALLEL_ANGLE:
                continue
            dist = float(np.linalg.norm(_wall_midpoint(sa) - _wall_midpoint(sb)))
            if dist > PROXIMITY:
                continue
            if sa.length == 0 or sb.length == 0:
                continue
            ratio = abs(sa.length - sb.length) / max(sa.length, sb.length)
            if ratio > LENGTH_RATIO:
                continue
            pairs.append((sa, sb))
    return pairs


def correct_drift(
    rooms: list[RoomIR],
    adjacencies: list,
) -> dict[str, tuple[float, float, float]]:
    """Return corrected room_id -> (dx, dy, 0.0) transforms.

    Finds shared wall pairs between adjacent rooms and minimises squared
    midpoint discrepancies (a wall-thickness gap across the wall is free)
    over per-room XY translations.  The first room
    stays fixed at the origin; rotation is always 0.
    """
    if len(rooms) <= 1:
        return {rooms[0].room_id: (0.0, 0.0, 0.0)} if rooms else {}

    adj_set: set[tuple[str, str]] = set()
    for a in adjacencies:
        if isinstance(a, dict):
            adj_set.add((a["room_a_id"], a["room_b_id"]))
        else:
            adj_set.add((a.room_a_id, a.room_b_id))

    room_idx = {r.room_id: i for i, r in enumerate(rooms)}
    shared: list[tuple[int, int, list[tuple]]] = []
    for ra in rooms:
        for rb in rooms:
            if ra.room_id >= rb.room_id:
                continue
            key = (ra.room_id, rb.room_id)
            rev = (rb.room_id, ra.room_id)
            if key not in adj_set and rev not in adj_set:
                continue
            pairs = _find_shared_walls(ra, rb)
            if pairs:
                shared.append((room_idx[ra.room_id], room_idx[rb.room_id], pairs))

    if not shared:
        log.info("no shared walls found; skipping drift correction")
        return {r.room_id: (0.0, 0.0, 0.0) for r in rooms}

    n = len(rooms)

    def cost(x: np.ndarray) -> float:
        offsets = x.reshape(n - 1, 2)
        total = 0.0
        for ia, ib, pairs in shared:
            da = np.zeros(2) if ia == 0 else offsets[ia - 1]
            db = np.zeros(2) if ib == 0 else offsets[ib - 1]
            for sa, sb in pairs:
                ma = _wall_midpoint(sa) + da
                mb = _wall_midpoint(sb) + db
                gap = ma - mb
                tangent = np.array([np.cos(sa.direction), np.sin(sa.direction)])
                along = float(gap @ tangent)
                across = abs(float(gap @ np.array([-tangent[1], tangent[0]])))
                # Along the wall any offset is drift; across it, up to a wall thickness is real.
                total += along ** 2 + max(0.0, across - WALL_THICKNESS_TOL) ** 2
                total += (sa.length - sb.length) ** 2
        return total + REGULARIZATION * float(np.sum(offsets ** 2))

    x0 = np.zeros(2 * (n - 1))
    result = minimize(cost, x0, method="L-BFGS-B", bounds=[(-MAX_CORRECTION, MAX_CORRECTION)] * len(x0))
    offsets = result.x.reshape(n - 1, 2)

    transforms: dict[str, tuple[float, float, float]] = {}
    for i, r in enumerate(rooms):
        if i == 0:
            transforms[r.room_id] = (0.0, 0.0, 0.0)
        else:
            dx, dy = offsets[i - 1]
            transforms[r.room_id] = (float(dx), float(dy), 0.0)

    log.info("drift correction converged: cost %.6f -> %.6f", cost(x0), result.fun)
    return transforms
