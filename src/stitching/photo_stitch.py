"""Photo-tier stitching: independent rooms joined through matching doors."""

from __future__ import annotations

import logging
import math

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from src.room_ir import RoomIR
from src.stitching.opening_ids import opening_id

log = logging.getLogger("floorplan.stitch")

WIDTH_TOLERANCE = 0.15  # 15 % match on door width
NUDGE_STEP = 0.1  # metres per nudge attempt
MAX_NUDGE = 10  # attempts


def _door_openings(room: RoomIR) -> list:
    """Return opening detections classified as doors."""
    if not room.openings:
        return []
    return [o for o in room.openings if o.type == "door"]


def _wall_normal(seg) -> np.ndarray:
    """Outward-pointing unit normal (90 deg CCW from wall direction)."""
    d = seg.direction
    return np.array([-math.sin(d), math.cos(d)])


def _door_position_global(room: RoomIR, opening, transform: tuple[float, float, float]) -> np.ndarray:
    """World-frame position of a door's centre along its wall."""
    seg = room.wall_segments[opening.wall_index]
    dx, dy, rot = transform
    t = (opening.position_along_wall + opening.width / 2) / seg.length if seg.length > 0 else 0.5
    px = seg.start[0] + t * (seg.end[0] - seg.start[0])
    py = seg.start[1] + t * (seg.end[1] - seg.start[1])
    c, s = math.cos(rot), math.sin(rot)
    return np.array([c * px - s * py + dx, s * px + c * py + dy])


def _room_polygon(room: RoomIR, transform: tuple[float, float, float]) -> ShapelyPolygon:
    dx, dy, rot = transform
    c, s = math.cos(rot), math.sin(rot)
    pts = [(c * x - s * y + dx, s * x + c * y + dy) for x, y in room.floor_polygon]
    if len(pts) < 3:
        return ShapelyPolygon()
    poly = ShapelyPolygon(pts)
    return poly if poly.is_valid else poly.buffer(0)


def _match_doors(doors_a, doors_b) -> tuple | None:
    """First pair of doors whose widths agree within WIDTH_TOLERANCE."""
    for da in doors_a:
        for db in doors_b:
            if da.width == 0 or db.width == 0:
                continue
            ratio = abs(da.width - db.width) / max(da.width, db.width)
            if ratio <= WIDTH_TOLERANCE:
                return da, db
    return None


def _compute_placement(
    room_prev: RoomIR, door_prev, tf_prev: tuple[float, float, float],
    room_next: RoomIR, door_next,
) -> tuple[float, float, float]:
    """Compute transform for room_next so its door_next aligns with door_prev."""
    pos_prev = _door_position_global(room_prev, door_prev, tf_prev)
    wall_prev = room_prev.wall_segments[door_prev.wall_index]
    normal_prev = _wall_normal(wall_prev)

    wall_next = room_next.wall_segments[door_next.wall_index]
    t_next = (door_next.position_along_wall + door_next.width / 2) / wall_next.length if wall_next.length > 0 else 0.5
    local_door = np.array([
        wall_next.start[0] + t_next * (wall_next.end[0] - wall_next.start[0]),
        wall_next.start[1] + t_next * (wall_next.end[1] - wall_next.start[1]),
    ])

    target_angle = math.atan2(normal_prev[1], normal_prev[0])
    next_wall_angle = wall_next.direction
    next_normal_angle = next_wall_angle + math.pi / 2
    rot = target_angle - next_normal_angle + math.pi

    c, s = math.cos(rot), math.sin(rot)
    rotated_door = np.array([c * local_door[0] - s * local_door[1],
                             s * local_door[0] + c * local_door[1]])
    offset = pos_prev + normal_prev * 0.01 - rotated_door
    return (float(offset[0]), float(offset[1]), float(rot))


def _nudge_if_overlap(
    rooms: list[RoomIR],
    transforms: dict[str, tuple[float, float, float]],
    idx: int,
) -> None:
    """Move room at idx away from existing rooms if polygons overlap."""
    rid = rooms[idx].room_id
    for attempt in range(MAX_NUDGE):
        poly = _room_polygon(rooms[idx], transforms[rid])
        if poly.is_empty:
            break
        overlaps = False
        for j in range(len(rooms)):
            if j == idx:
                continue
            other = _room_polygon(rooms[j], transforms[rooms[j].room_id])
            if other.is_empty:
                continue
            inter = poly.intersection(other)
            if inter.area > 0.01:
                overlaps = True
                cx, cy = inter.centroid.x, inter.centroid.y
                px, py = poly.centroid.x, poly.centroid.y
                dx = px - cx
                dy = py - cy
                norm = math.hypot(dx, dy) or 1.0
                tx, ty, tr = transforms[rid]
                transforms[rid] = (tx + NUDGE_STEP * dx / norm,
                                   ty + NUDGE_STEP * dy / norm,
                                   tr)
                break
        if not overlaps:
            break
    else:
        log.warning("could not fully resolve overlap for %s after %d nudges", rid, MAX_NUDGE)


def stitch_photos(rooms: list[RoomIR]) -> tuple[dict[str, tuple[float, float, float]], list[dict]]:
    """Return (room_transforms, adjacencies) for rooms with no shared frame.

    Rooms arrive in walk order.  Room N is linked to room N+1 through the
    first pair of doors with similar width, placed greedily so the door
    walls coincide.
    """
    if not rooms:
        return {}, []

    transforms: dict[str, tuple[float, float, float]] = {rooms[0].room_id: (0.0, 0.0, 0.0)}
    adjacencies: list[dict] = []

    for i in range(1, len(rooms)):
        prev = rooms[i - 1]
        curr = rooms[i]
        doors_prev = _door_openings(prev)
        doors_curr = _door_openings(curr)

        match = _match_doors(doors_prev, doors_curr)
        if match is not None:
            dp, dc = match
            tf = _compute_placement(prev, dp, transforms[prev.room_id], curr, dc)
            transforms[curr.room_id] = tf
            adjacencies.append({
                "room_a_id": prev.room_id,
                "room_b_id": curr.room_id,
                "shared_opening_id": opening_id(prev, dp),
            })
        else:
            log.warning("no matching door between %s and %s; placing side by side",
                        prev.room_id, curr.room_id)
            prev_poly = _room_polygon(prev, transforms[prev.room_id])
            if prev_poly.is_empty:
                dx_offset = 5.0 * i
            else:
                dx_offset = prev_poly.bounds[2] + 1.0
            transforms[curr.room_id] = (dx_offset, 0.0, 0.0)

        _nudge_if_overlap(rooms, transforms, i)

    for rid in transforms:
        poly = _room_polygon(
            next(r for r in rooms if r.room_id == rid), transforms[rid]
        )
        for rid2 in transforms:
            if rid >= rid2:
                continue
            other = _room_polygon(
                next(r for r in rooms if r.room_id == rid2), transforms[rid2]
            )
            if not poly.is_empty and not other.is_empty:
                inter = poly.intersection(other)
                if inter.area > 0.01:
                    log.warning("rooms %s and %s overlap by %.3f m2", rid, rid2, inter.area)

    return transforms, adjacencies
