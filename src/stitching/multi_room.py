"""Whole-property stitching: adjacency graph, room transforms, overlap validation."""

from __future__ import annotations

import logging
import math

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from src.room_ir import PropertyIR, RoomIR
from src.stitching.drift_correction import correct_drift
from src.stitching.opening_ids import opening_id
from src.stitching.photo_stitch import stitch_photos

log = logging.getLogger("floorplan.stitch")

OPENING_PROXIMITY = 0.5  # metres: openings closer than this link two rooms
CONTACT_GAP = 0.2  # metres: walls this close count as touching (a wall is 0.1 to 0.3 m thick)
MIN_CONTACT_LENGTH = 0.6  # metres of shared boundary needed to call two rooms adjacent


def _opening_global_pos(room: RoomIR, opening) -> np.ndarray:
    """World-frame XY position of the opening centre along its wall."""
    seg = room.wall_segments[opening.wall_index]
    t = (opening.position_along_wall + opening.width / 2) / seg.length if seg.length > 0 else 0.5
    return np.array([
        seg.start[0] + t * (seg.end[0] - seg.start[0]),
        seg.start[1] + t * (seg.end[1] - seg.start[1]),
    ])


def _contact_length(ra: RoomIR, rb: RoomIR) -> float:
    """Approximate length (m) of boundary that two room outlines share within CONTACT_GAP."""
    if not ra.floor_polygon or not rb.floor_polygon:
        return 0.0
    pa = _valid(ShapelyPolygon(ra.floor_polygon)).buffer(CONTACT_GAP / 2)
    pb = _valid(ShapelyPolygon(rb.floor_polygon)).buffer(CONTACT_GAP / 2)
    return float(pa.intersection(pb).area / CONTACT_GAP) if pa.intersects(pb) else 0.0


def _build_adjacency_graph(rooms: list[RoomIR]) -> list[dict]:
    """Link rooms joined by an opening (openings within OPENING_PROXIMITY metres of each other),
    or, when partial scans missed the doorway, rooms whose outlines share a boundary of
    MIN_CONTACT_LENGTH or more (shared_opening_id is then None)."""
    adjacencies: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for i, ra in enumerate(rooms):
        for j, rb in enumerate(rooms):
            if j <= i:
                continue
            key = (ra.room_id, rb.room_id)
            if key in seen:
                continue
            for oa in (ra.openings or []):
                for ob in (rb.openings or []):
                    pa = _opening_global_pos(ra, oa)
                    pb = _opening_global_pos(rb, ob)
                    if float(np.linalg.norm(pa - pb)) < OPENING_PROXIMITY:
                        adjacencies.append({
                            "room_a_id": ra.room_id,
                            "room_b_id": rb.room_id,
                            "shared_opening_id": opening_id(ra, oa),
                        })
                        seen.add(key)
                        break
                if key in seen:
                    break
            if key not in seen and _contact_length(ra, rb) >= MIN_CONTACT_LENGTH:
                adjacencies.append({"room_a_id": ra.room_id, "room_b_id": rb.room_id,
                                    "shared_opening_id": None})
                seen.add(key)
    return adjacencies


def _valid(poly: ShapelyPolygon) -> ShapelyPolygon:
    """Repair a self-intersecting outline (common with partial real scans) so Shapely ops cannot raise."""
    return poly if poly.is_valid else poly.buffer(0)


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
        polys.append((r.room_id, _valid(ShapelyPolygon(pts))))

    for i, (id_a, pa) in enumerate(polys):
        for j in range(i + 1, len(polys)):
            id_b, pb = polys[j]
            if pa.is_empty or pb.is_empty:
                continue
            inter = pa.intersection(pb)
            if inter.area > 0.01:
                log.warning("rooms %s and %s overlap by %.3f m2", id_a, id_b, inter.area)


def stitch_rooms(
    property_ir: PropertyIR,
    drift_correction: bool = True,
    photo_stitcher: str = "classical",
    vlm_adjacency: dict | None = None,
    vlm_options: dict | None = None,
) -> PropertyIR:
    """Place all rooms in one frame and fill adjacencies and room_transforms.

    LiDAR/video rooms already share a frame: link rooms whose openings lie
    within 0.5 m, optionally run drift correction, and warn on polygon
    overlaps. Photo-tier rooms are delegated to photo_stitch.stitch_photos,
    optionally guided by VLM spatial reasoning (photo_stitcher='vlm').
    """
    rooms = property_ir.rooms

    if not rooms:
        property_ir.room_transforms = {}
        property_ir.adjacencies = []
        return property_ir

    if property_ir.tier == "photo":
        if str(photo_stitcher).lower() == "vlm" and vlm_adjacency is None:
            from src.stitching.vlm_adjacency import infer_room_adjacency_vlm
            opts = dict(vlm_options or {})
            room_imgs = {r.room_id: r.images for r in rooms if r.images}
            if len(room_imgs) >= 2:
                log.info("Running VLM spatial reasoning across %d rooms...", len(room_imgs))
                try:
                    vlm_adjacency = infer_room_adjacency_vlm(room_imgs, **opts)
                except Exception as exc:
                    log.warning("VLM adjacency inference failed: %s; falling back to classical stitching", exc)
                    property_ir.warnings.append(
                        f"VLM spatial reasoning failed ({exc}); fell back to classical door matching."
                    )
                    vlm_adjacency = None
        transforms, adjacencies = stitch_photos(rooms, vlm_adjacency=vlm_adjacency)
        property_ir.room_transforms = transforms
        property_ir.adjacencies = adjacencies
        log.info("photo stitch (%s): %d room(s), %d adjacencies", photo_stitcher, len(rooms), len(adjacencies))
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

    property_ir.room_transforms = transforms
    property_ir.adjacencies = adjacencies
    if property_ir.tier == "video":  # segmentation can cut one room (a hallway) into fragments; LiDAR is left alone
        from src.stitching.room_merge import merge_oversegmented_rooms
        property_ir.warnings.extend(merge_oversegmented_rooms(property_ir))
    _validate_no_overlap(property_ir.rooms, property_ir.room_transforms)
    return property_ir
