"""Merge rooms that room segmentation split in two (typically a hallway cut into fragments).

Two adjacent rooms are one room when they share a boundary at least MERGE_SHARED_FRACTION (80 %) as long as the
facing wall of the shorter room. Pairs joined by a detected opening are separate rooms (a real wall with a door).
Their polygons are united (Shapely `unary_union`), and walls, openings, area, ceiling height, point cloud and
camera poses are rebuilt for the merged room, which is expressed in the global frame with an identity transform.
Used for the video tier; photo rooms are one folder each, and the LiDAR tier is left as it is.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from src import config as cfg
from src.geometry.wall_fitting import WallSegment
from src.geometry.openings import OpeningDetection
from src.room_ir import CameraPose, PointCloud, PropertyIR, RoomIR

log = logging.getLogger("floorplan.stitch")

CONTACT_GAP = 0.2  # metres: walls this close count as touching
MIN_CONTACT_LENGTH = 0.6  # metres of shared boundary needed at all
SIMPLIFY = 0.08  # metres: tolerance when simplifying the united outline
OPENING_ON_WALL = 0.3  # metres: an opening this close to a merged wall stays on it; others were on the removed interface


def _rotation(rot: float) -> np.ndarray:
    c, s = math.cos(rot), math.sin(rot)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _place(pt, tf) -> tuple[float, float]:
    c, s = math.cos(tf[2]), math.sin(tf[2])
    return (c * pt[0] - s * pt[1] + tf[0], s * pt[0] + c * pt[1] + tf[1])


def _global_polygon(room: RoomIR, tf) -> Polygon:
    poly = Polygon([_place(p, tf) for p in room.floor_polygon])
    return poly if poly.is_valid else poly.buffer(0)


def _contact_length(pa: Polygon, pb: Polygon) -> float:
    a, b = pa.buffer(CONTACT_GAP / 2), pb.buffer(CONTACT_GAP / 2)
    return float(a.intersection(b).area / CONTACT_GAP) if a.intersects(b) else 0.0


def _facing_wall_length(room: RoomIR, tf, other: Polygon) -> float:
    """Length of the longest wall of `room` lying along `other` (its longest wall when none touches it)."""
    lengths = []
    for seg in room.wall_segments:
        mid = Point(*_place(((seg.start[0] + seg.end[0]) / 2, (seg.start[1] + seg.end[1]) / 2), tf))
        if other.distance(mid) <= CONTACT_GAP + 0.05:
            lengths.append(seg.length)
    return max(lengths) if lengths else max(seg.length for seg in room.wall_segments)


def should_merge(ra: RoomIR, tfa, rb: RoomIR, tfb) -> bool:
    pa, pb = _global_polygon(ra, tfa), _global_polygon(rb, tfb)
    shared = _contact_length(pa, pb)
    if shared < MIN_CONTACT_LENGTH:
        return False
    shorter = min(_facing_wall_length(ra, tfa, pb), _facing_wall_length(rb, tfb, pa))
    return shared >= cfg.MERGE_SHARED_FRACTION * shorter


def _transform_poses(poses: list[CameraPose], tf) -> list[CameraPose]:
    """Poses of a room frame x in the global frame x' = M x + d: R' = R M^T, t' = t - R' d."""
    m, d = _rotation(tf[2]), np.array([tf[0], tf[1], 0.0])
    out = []
    for p in poses:
        r = np.asarray(p.rotation, float) @ m.T
        out.append(CameraPose(image_path=p.image_path, rotation=r, translation=np.asarray(p.translation, float) - r @ d,
                              intrinsics=p.intrinsics))
    return out


def _walls_from_outline(points: list[tuple[float, float]], source_walls: list) -> list[WallSegment]:
    walls = []
    for i, a in enumerate(points):
        b = points[(i + 1) % len(points)]
        length = math.dist(a, b)
        if length < 1e-6:
            continue
        line = LineString([a, b])
        support = sum(s.inlier_count for s, mid in source_walls if line.distance(Point(*mid)) <= 0.2)
        walls.append(WallSegment(start=(float(a[0]), float(a[1])), end=(float(b[0]), float(b[1])), length=float(length),
                                 direction=math.atan2(b[1] - a[1], b[0] - a[0]), inlier_count=int(support)))
    return walls


def _opening_centre(room: RoomIR, op, tf) -> tuple[float, float]:
    seg = room.wall_segments[op.wall_index]
    t = (op.position_along_wall + op.width / 2) / seg.length if seg.length > 0 else 0.5
    return _place((seg.start[0] + t * (seg.end[0] - seg.start[0]), seg.start[1] + t * (seg.end[1] - seg.start[1])), tf)


def merge_pair(ir: PropertyIR, ra: RoomIR, rb: RoomIR) -> bool:
    """Replace ra and rb by one merged room (named like ra). False when their outlines do not form one polygon."""
    from src.stitching.opening_ids import opening_id

    tfs = ir.room_transforms or {}
    tfa, tfb = tfs.get(ra.room_id, (0.0, 0.0, 0.0)), tfs.get(rb.room_id, (0.0, 0.0, 0.0))
    pa, pb = _global_polygon(ra, tfa), _global_polygon(rb, tfb)
    union = unary_union([pa.buffer(CONTACT_GAP / 2), pb.buffer(CONTACT_GAP / 2)]).buffer(-CONTACT_GAP / 2)
    if union.geom_type != "Polygon" or union.is_empty:
        return False
    outline = orient(union.simplify(SIMPLIFY, preserve_topology=True), 1.0)
    points = [(float(x), float(y)) for x, y in outline.exterior.coords][:-1]
    if len(points) < 3:
        return False

    source_walls = [(s, _place(((s.start[0] + s.end[0]) / 2, (s.start[1] + s.end[1]) / 2), tf))
                    for room, tf in ((ra, tfa), (rb, tfb)) for s in room.wall_segments]
    walls = _walls_from_outline(points, source_walls)

    # openings: re-attach to the nearest merged wall; those on the removed interface disappear
    openings, id_map = [], {}
    for room, tf in ((ra, tfa), (rb, tfb)):
        for op in room.openings or []:
            cx, cy = _opening_centre(room, op, tf)
            dists = [LineString([w.start, w.end]).distance(Point(cx, cy)) for w in walls]
            k = int(np.argmin(dists))
            if dists[k] > OPENING_ON_WALL:
                continue
            w = walls[k]
            along = ((cx - w.start[0]) * (w.end[0] - w.start[0]) + (cy - w.start[1]) * (w.end[1] - w.start[1])) / w.length
            moved = OpeningDetection(wall_index=k, type=op.type, position_along_wall=max(0.0, along - op.width / 2),
                                     width=op.width, height=op.height, confidence=op.confidence)
            id_map[opening_id(room, op)] = moved
            openings.append(moved)

    cloud = np.vstack([np.asarray(room.point_cloud.points, float) @ _rotation(tf[2]).T + np.array([tf[0], tf[1], 0.0])
                       for room, tf in ((ra, tfa), (rb, tfb))])
    area_a, area_b = pa.area, pb.area
    all_observed = ra.ceiling_observed and rb.ceiling_observed
    heights = [h for h in (ra.ceiling_height, rb.ceiling_height) if h is not None]
    ceiling = (max(heights) if not all_observed else
               (ra.ceiling_height * area_a + rb.ceiling_height * area_b) / (area_a + area_b)) if heights else None
    extent = cloud.max(axis=0) - cloud.min(axis=0)
    merged = RoomIR(
        room_id=ra.room_id, point_cloud=PointCloud(points=cloud),
        camera_poses=_transform_poses(ra.camera_poses, tfa) + _transform_poses(rb.camera_poses, tfb),
        images=list(dict.fromkeys(list(ra.images) + list(rb.images))), tier=ra.tier, wall_segments=walls,
        openings=openings, ceiling_height=ceiling, ceiling_observed=all_observed, floor_polygon=points,
        point_density=len(cloud) / float(max(extent.prod(), 1e-6)),
        metadata={**ra.metadata, "merged_from": [ra.room_id, rb.room_id]})

    # opening ids in the adjacency list follow the openings to their new place in the merged room
    new_ids = {old: opening_id(merged, op) for old, op in id_map.items()}
    ir.rooms = [merged if r is ra else r for r in ir.rooms if r is not rb]
    ir.room_transforms = {**{k: v for k, v in tfs.items() if k not in (ra.room_id, rb.room_id)},
                          merged.room_id: (0.0, 0.0, 0.0)}
    links: dict[tuple[str, str], dict] = {}
    for adj in ir.adjacencies or []:
        a_id = ra.room_id if adj["room_a_id"] == rb.room_id else adj["room_a_id"]
        b_id = ra.room_id if adj["room_b_id"] == rb.room_id else adj["room_b_id"]
        if a_id == b_id:
            continue
        key = tuple(sorted((a_id, b_id)))
        opening = adj.get("shared_opening_id")
        opening = new_ids.get(opening) if opening in new_ids else (
            None if opening and opening.split("/")[0] in (ra.room_id, rb.room_id) else opening)
        old = links.get(key)
        if old is None or (old["shared_opening_id"] is None and opening is not None):
            links[key] = {"room_a_id": key[0], "room_b_id": key[1], "shared_opening_id": opening}
    ir.adjacencies = list(links.values())
    return True


def merge_oversegmented_rooms(ir: PropertyIR) -> list[str]:
    """Merge rooms that share a wall-length boundary and no opening, until none is left. Returns notes for the report."""
    notes: list[str] = []
    rejected: set[frozenset] = set()
    while True:
        tfs = ir.room_transforms or {}
        by_id = {r.room_id: r for r in ir.rooms}
        best, best_len = None, 0.0
        for adj in ir.adjacencies or []:
            if adj.get("shared_opening_id") is not None:  # a doorway: two rooms
                continue
            ra, rb = by_id.get(adj["room_a_id"]), by_id.get(adj["room_b_id"])
            if ra is None or rb is None or frozenset((ra.room_id, rb.room_id)) in rejected:
                continue
            if any(r.metadata.get("rough_estimate") or not r.floor_polygon or not r.wall_segments for r in (ra, rb)):
                continue
            tfa, tfb = tfs.get(ra.room_id, (0.0, 0.0, 0.0)), tfs.get(rb.room_id, (0.0, 0.0, 0.0))
            if should_merge(ra, tfa, rb, tfb):
                shared = _contact_length(_global_polygon(ra, tfa), _global_polygon(rb, tfb))
                if shared > best_len:
                    best, best_len = (ra, rb), shared
        if best is None:
            return notes
        ra, rb = sorted(best, key=lambda r: ir.rooms.index(r))
        if merge_pair(ir, ra, rb):
            note = f"{rb.room_id} merged into {ra.room_id}: they share {best_len:.1f} m of boundary and no opening, " \
                   "so they were probably one room split by the room segmentation."
            log.info(note)
            notes.append(note)
        else:
            rejected.add(frozenset((ra.room_id, rb.room_id)))


def merge_small_rooms(ir: PropertyIR, min_area: float) -> list[str]:
    """Merge every room under min_area m2 into the neighbour it shares the longest boundary with (at least
    MIN_CONTACT_LENGTH). A room that touches nothing stays. Returns notes for the report."""
    notes: list[str] = []
    rejected: set[frozenset] = set()
    while True:
        tfs = ir.room_transforms or {}
        usable = [r for r in ir.rooms if r.floor_polygon and r.wall_segments and not r.metadata.get("rough_estimate")]
        polys = {r.room_id: _global_polygon(r, tfs.get(r.room_id, (0.0, 0.0, 0.0))) for r in usable}
        best = None  # (area of the small room, small room, neighbour, contact length)
        for small in sorted(usable, key=lambda r: polys[r.room_id].area):
            area = polys[small.room_id].area
            if area >= min_area:
                break
            contacts = [(_contact_length(polys[small.room_id], polys[o.room_id]), o) for o in usable
                        if o is not small and frozenset((small.room_id, o.room_id)) not in rejected]
            contacts = [c for c in contacts if c[0] >= MIN_CONTACT_LENGTH]
            if contacts:
                length, other = max(contacts, key=lambda c: c[0])
                best = (small, other, length, area)
                break
        if best is None:
            return notes
        small, other, length, area = best
        ra, rb = sorted((small, other), key=lambda r: ir.rooms.index(r))
        if merge_pair(ir, ra, rb):
            note = (f"{small.room_id} ({area:.1f} m2) merged into {other.room_id}: it was under {min_area:.1f} m2 and "
                    f"shared {length:.1f} m of boundary, so it was probably a fragment of the room segmentation.")
            log.info(note)
            notes.append(note)
        else:
            rejected.add(frozenset((small.room_id, other.room_id)))

