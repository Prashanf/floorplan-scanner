"""Photo-tier stitching: independent rooms joined through a shared doorway camera or matching doors.

Every room pair is a candidate edge of an adjacency graph. Walk order (room N to N+1) is the primary
assumption and is tried first; other pairs are tried as well (a hall can join rooms 1 and 4). An edge comes from
1. a shared doorway photo that was registered in both rooms' reconstructions (exact relative pose), and/or
2. the best door pair: widths within DOOR_WIDTH_MATCH, smallest width difference, every door used once.
Rooms are then placed by breadth-first search from the first room over those edges.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from src import config as cfg
from src.room_ir import RoomIR
from src.stitching.opening_ids import opening_id

log = logging.getLogger("floorplan.stitch")

WIDTH_TOLERANCE = cfg.DOOR_WIDTH_MATCH  # share of the larger width
NUDGE_STEP = 0.1  # metres per nudge attempt
MAX_NUDGE = 10  # attempts

Transform = tuple[float, float, float]  # (dx, dy, rotation): p_global = Rz(rotation) p + (dx, dy)


# --------------------------------------------------------------------------- geometry helpers

def _door_openings(room: RoomIR) -> list:
    """Return opening detections classified as doors."""
    if not room.openings:
        return []
    return [o for o in room.openings if o.type == "door"]


def _wall_normal(seg) -> np.ndarray:
    """Unit normal at 90 deg CCW from the wall direction: into the room for a counter-clockwise polygon."""
    d = seg.direction
    return np.array([-math.sin(d), math.cos(d)])


def _door_position_global(room: RoomIR, opening, transform: Transform) -> np.ndarray:
    """World-frame position of a door's centre along its wall."""
    seg = room.wall_segments[opening.wall_index]
    dx, dy, rot = transform
    t = (opening.position_along_wall + opening.width / 2) / seg.length if seg.length > 0 else 0.5
    px = seg.start[0] + t * (seg.end[0] - seg.start[0])
    py = seg.start[1] + t * (seg.end[1] - seg.start[1])
    c, s = math.cos(rot), math.sin(rot)
    return np.array([c * px - s * py + dx, s * px + c * py + dy])


def _room_polygon(room: RoomIR, transform: Transform) -> ShapelyPolygon:
    dx, dy, rot = transform
    c, s = math.cos(rot), math.sin(rot)
    pts = [(c * x - s * y + dx, s * x + c * y + dy) for x, y in room.floor_polygon]
    if len(pts) < 3:
        return ShapelyPolygon()
    poly = ShapelyPolygon(pts)
    return poly if poly.is_valid else poly.buffer(0)


def compose(outer: Transform, inner: Transform) -> Transform:
    """Transform applying `inner` first, then `outer`."""
    c, s = math.cos(outer[2]), math.sin(outer[2])
    return (c * inner[0] - s * inner[1] + outer[0], s * inner[0] + c * inner[1] + outer[1], outer[2] + inner[2])


def invert(tf: Transform) -> Transform:
    c, s = math.cos(-tf[2]), math.sin(-tf[2])
    return (-(c * tf[0] - s * tf[1]), -(s * tf[0] + c * tf[1]), -tf[2])


# --------------------------------------------------------------------------- door matching

def find_best_door_match(room_a_openings, room_b_openings):
    """Best matching door pair between two rooms, or None.

    Only doors count. A pair matches when the widths differ by less than WIDTH_TOLERANCE of the larger one;
    among matching pairs the smallest width difference wins.
    """
    best_pair, best_score = None, float("inf")
    for a in room_a_openings or []:
        if a.type != "door":
            continue
        for b in room_b_openings or []:
            if b.type != "door":
                continue
            width_diff = abs(a.width - b.width)
            if a.width > 0 and b.width > 0 and width_diff < WIDTH_TOLERANCE * max(a.width, b.width):
                if width_diff < best_score:
                    best_score, best_pair = width_diff, (a, b)
    return best_pair


def _compute_placement(
    room_prev: RoomIR, door_prev, tf_prev: Transform,
    room_next: RoomIR, door_next,
) -> Transform:
    """Transform for room_next so its door_next sits on room_prev's door_prev, the rooms on opposite sides."""
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
    # normal_prev points into room_prev (the polygons are counter-clockwise); the new room starts 1 cm outside it
    offset = pos_prev - normal_prev * 0.01 - rotated_door
    return (float(offset[0]), float(offset[1]), float(rot))


# --------------------------------------------------------------------------- shared doorway camera

def _pose_of(room: RoomIR, image_path: str):
    return next((p for p in room.camera_poses if p.image_path == image_path), None)


def pose_relative_transform(room_a: RoomIR, room_b: RoomIR) -> Optional[Transform]:
    """Transform of room b's frame into room a's frame from the doorway photo both rooms registered.

    Room a's doorway photo was also reconstructed with room b (`borrowed_photo`). Both reconstructions are Z-up
    and metric, so the same camera in the two frames gives the yaw between them and, by putting the two camera
    centres on top of each other, the shift. None when the photo is missing in either room, or the two "up"
    directions disagree by more than DOORWAY_MAX_TILT_DEG (a bad registration).
    """
    shared = room_a.metadata.get("doorway_photo")
    if not shared or shared != room_b.metadata.get("borrowed_photo"):
        return None
    pa, pb = _pose_of(room_a, shared), _pose_of(room_b, shared)
    if pa is None or pb is None:
        return None
    ra, ta = np.asarray(pa.rotation, float), np.asarray(pa.translation, float)
    rb, tb = np.asarray(pb.rotation, float), np.asarray(pb.translation, float)
    m = ra.T @ rb  # b-frame directions in the a frame
    if m[2, 2] < math.cos(math.radians(cfg.DOORWAY_MAX_TILT_DEG)):
        log.warning("doorway photo of %s: the two rooms' up directions differ by %.0f deg; not used",
                    room_a.room_id, math.degrees(math.acos(max(-1.0, min(1.0, m[2, 2])))))
        return None
    yaw = math.atan2(m[1, 0], m[0, 0])
    centre_a, centre_b = -ra.T @ ta, -rb.T @ tb
    c, s = math.cos(yaw), math.sin(yaw)
    shift = centre_a[:2] - np.array([c * centre_b[0] - s * centre_b[1], s * centre_b[0] + c * centre_b[1]])
    return (float(shift[0]), float(shift[1]), float(yaw))


# --------------------------------------------------------------------------- graph and placement

@dataclass
class _Edge:
    a: int
    b: int
    door_a: object = None
    door_b: object = None
    pose_tf: Optional[Transform] = None  # transform of room b into room a's frame


def _build_edges(rooms: list[RoomIR]) -> list[_Edge]:
    """Edges for every room pair that can be joined; walk-order pairs first, then the rest by distance in the order."""
    used: set[int] = set()
    n = len(rooms)

    def free_doors(room: RoomIR) -> list:
        return [o for o in _door_openings(room) if id(o) not in used]

    pairs = [(i, i + 1) for i in range(n - 1)] + [(i, i + gap) for gap in range(2, n) for i in range(n - gap)]
    edges: list[_Edge] = []
    for i, j in pairs:
        pose = pose_relative_transform(rooms[i], rooms[j]) if j == i + 1 else None
        match = find_best_door_match(free_doors(rooms[i]), free_doors(rooms[j]))
        if match:
            used.update((id(match[0]), id(match[1])))
        if match or pose:
            edges.append(_Edge(i, j, match[0] if match else None, match[1] if match else None, pose))
    return edges


def _place_via_edge(rooms, transforms, edge: _Edge, placed: int, new: int) -> Transform:
    tf_placed = transforms[rooms[placed].room_id]
    if edge.pose_tf is not None:
        return compose(tf_placed, edge.pose_tf if placed == edge.a else invert(edge.pose_tf))
    door_placed, door_new = (edge.door_a, edge.door_b) if placed == edge.a else (edge.door_b, edge.door_a)
    return _compute_placement(rooms[placed], door_placed, tf_placed, rooms[new], door_new)


def _nudge_if_overlap(rooms: list[RoomIR], transforms: dict[str, Transform], idx: int) -> None:
    """Move room at idx away from the already placed rooms it overlaps."""
    rid = rooms[idx].room_id
    for _ in range(MAX_NUDGE):
        poly = _room_polygon(rooms[idx], transforms[rid])
        if poly.is_empty:
            break
        overlaps = False
        for j, other_room in enumerate(rooms):
            if j == idx or other_room.room_id not in transforms:  # rooms not placed yet have no transform
                continue
            other = _room_polygon(other_room, transforms[other_room.room_id])
            if other.is_empty:
                continue
            inter = poly.intersection(other)
            if inter.area > 0.01:
                overlaps = True
                dx, dy = poly.centroid.x - inter.centroid.x, poly.centroid.y - inter.centroid.y
                norm = math.hypot(dx, dy) or 1.0
                tx, ty, tr = transforms[rid]
                transforms[rid] = (tx + NUDGE_STEP * dx / norm, ty + NUDGE_STEP * dy / norm, tr)
                break
        if not overlaps:
            return
    log.warning("could not fully resolve overlap for %s after %d nudges", rid, MAX_NUDGE)


def stitch_photos(rooms: list[RoomIR]) -> tuple[dict[str, Transform], list[dict]]:
    """Return (room_transforms, adjacencies) for rooms with no shared frame.

    Rooms arrive in walk order. All pairs are tried for an edge (see module docstring); rooms are placed by BFS
    from the first room, over pose edges and walk-order edges first. A room no edge reaches is put to the right
    of everything placed so far. Overlaps are nudged apart and warned about.
    """
    if not rooms:
        return {}, []
    edges = _build_edges(rooms)
    neighbours: dict[int, list[_Edge]] = {i: [] for i in range(len(rooms))}
    for e in edges:
        neighbours[e.a].append(e)
        neighbours[e.b].append(e)

    transforms: dict[str, Transform] = {rooms[0].room_id: (0.0, 0.0, 0.0)}
    placed, queue = {0}, deque([0])
    while len(placed) < len(rooms):
        while queue:
            u = queue.popleft()
            for e in sorted(neighbours[u], key=lambda e: (e.pose_tf is None, abs(e.a - e.b))):
                v = e.b if e.a == u else e.a
                if v in placed:
                    continue
                transforms[rooms[v].room_id] = _place_via_edge(rooms, transforms, e, u, v)
                _nudge_if_overlap(rooms, transforms, v)
                placed.add(v)
                queue.append(v)
        if len(placed) < len(rooms):  # nothing links the rest to what is placed: the next room goes to the right
            v = min(set(range(len(rooms))) - placed)
            right = max((_room_polygon(rooms[k], transforms[rooms[k].room_id]).bounds[2] for k in placed
                         if not _room_polygon(rooms[k], transforms[rooms[k].room_id]).is_empty), default=0.0)
            log.warning("no door or doorway photo links %s to the placed rooms; placing it side by side",
                        rooms[v].room_id)
            transforms[rooms[v].room_id] = (right + 1.0, 0.0, 0.0)
            placed.add(v)
            queue.append(v)

    adjacencies = [{
        "room_a_id": rooms[e.a].room_id, "room_b_id": rooms[e.b].room_id,
        "shared_opening_id": opening_id(rooms[e.a], e.door_a) if e.door_a is not None else None,
    } for e in sorted(edges, key=lambda e: (e.a, e.b))]

    ids = [r.room_id for r in rooms]
    for i, rid in enumerate(ids):
        pa = _room_polygon(rooms[i], transforms[rid])
        for j in range(i + 1, len(ids)):
            pb = _room_polygon(rooms[j], transforms[ids[j]])
            if not pa.is_empty and not pb.is_empty and pa.intersection(pb).area > 0.01:
                log.warning("rooms %s and %s overlap by %.3f m2", rid, ids[j], pa.intersection(pb).area)
    log.info("photo stitch: %d edge(s): %s", len(edges),
             ", ".join(f"{rooms[e.a].room_id}-{rooms[e.b].room_id}{' (doorway photo)' if e.pose_tf else ''}"
                       for e in edges) or "none")
    return transforms, adjacencies
