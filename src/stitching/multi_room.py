"""Whole-property stitching: adjacency graph, room transforms, overlap validation."""

from __future__ import annotations

import logging
import math

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from src.room_ir import PropertyIR, RoomIR
from src.stitching.drift_correction import correct_drift
from src.stitching.photo_stitch import stitch_photos

log = logging.getLogger("floorplan.stitch")

OPENING_PROXIMITY = 0.5  # metres: openings closer than this link two rooms


def _opening_global_pos(room: RoomIR, opening) -> np.ndarray:
    """World-frame XY position of the opening centre along its wall."""
    seg = room.wall_segments[opening.wall_index]
    t = (opening.position_along_wall + opening.width / 2) / seg.length if seg.length > 0 else 0.5
    return np.array([
        seg.start[0] + t * (seg.end[0] - seg.start[0]),
        seg.start[1] + t * (seg.end[1] - seg.start[1]),
    ])


def _build_adjacency_graph(rooms: list[RoomIR]) -> list[dict]:
    """Link rooms whose openings are within OPENING_PROXIMITY metres."""
    adjacencies: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for i, ra in enumerate(rooms):
        if not ra.openings:
            continue
        for j, rb in enumerate(rooms):
            if j <= i or not rb.openings:
                continue
            key = (ra.room_id, rb.room_id)
            if key in seen:
                continue
            for oa in ra.openings:
                for ob in rb.openings:
                    pa = _opening_global_pos(ra, oa)
                    pb = _opening_global_pos(rb, ob)
                    if float(np.linalg.norm(pa - pb)) < OPENING_PROXIMITY:
                        adjacencies.append({
                            "room_a_id": ra.room_id,
                            "room_b_id": rb.room_id,
                            "shared_opening_id": f"{ra.room_id}/{oa.type}-{oa.wall_index}",
                        })
                        seen.add(key)
                        break
                if key in seen:
                    break
    return adjacencies


def _validate_no_overlap(rooms: list[RoomIR], transforms: dict) -> None:
    """Log warnings for overlapping room polygons."""
    polys: list[tuple[str, ShapelyPolygon]] = []
    for r in rooms:
        if not r.floor_polygon or len(r.floor_polygon) < 3:
            continue
        tf = transforms.get(r.room_id, (0.0, 0.0, 0.0))
        dx, dy, rot = tf
        c, s = math.cos(rot), math.sin(rot)
        pts = [(c * x - s * y + dx, s * x + c * y + dy) for x, y in r.floor_polygon]
        polys.append((r.room_id, ShapelyPolygon(pts)))

    for i, (id_a, pa) in enumerate(polys):
        for j in range(i + 1, len(polys)):
            id_b, pb = polys[j]
            if pa.is_empty or pb.is_empty:
                continue
            inter = pa.intersection(pb)
            if inter.area > 0.01:
                log.warning("rooms %s and %s overlap by %.3f m2", id_a, id_b, inter.area)


def stitch_rooms(property_ir: PropertyIR, drift_correction: bool = True) -> PropertyIR:
    """Place all rooms in one frame and fill adjacencies and room_transforms.

    LiDAR/video rooms already share a frame: link rooms whose openings lie
    within 0.5 m, optionally run drift correction, and warn on polygon
    overlaps.  Photo-tier rooms are delegated to photo_stitch.stitch_photos.
    """
    rooms = property_ir.rooms

    if not rooms:
        property_ir.room_transforms = {}
        property_ir.adjacencies = []
        return property_ir

    if property_ir.tier == "photo":
        transforms, adjacencies = stitch_photos(rooms)
        property_ir.room_transforms = transforms
        property_ir.adjacencies = adjacencies
        log.info("photo stitch: %d room(s), %d adjacencies", len(rooms), len(adjacencies))
        return property_ir

    adjacencies = _build_adjacency_graph(rooms)
    log.info("adjacency graph: %d link(s) among %d room(s)", len(adjacencies), len(rooms))

    if drift_correction and len(rooms) > 1 and adjacencies:
        transforms = correct_drift(rooms, adjacencies)
        log.info("drift correction applied")
    else:
        transforms = {r.room_id: (0.0, 0.0, 0.0) for r in rooms}
        if not drift_correction:
            log.info("drift correction disabled (ablation mode)")

    _validate_no_overlap(rooms, transforms)

    property_ir.room_transforms = transforms
    property_ir.adjacencies = adjacencies
    return property_ir
