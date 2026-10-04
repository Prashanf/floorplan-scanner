"""End-to-end orchestration of one capture."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import __version__
from src.calibration.confidence import calibrate_measurements
from src.damage.concealed_rules import check_concealed_damage
from src.damage.detection import detect_damage
from src.damage.scope import area_measurement, generate_scope
from src.damage.surface_projection import project_damage_to_surfaces
from src.geometry.ceiling import detect_ceiling_height
from src.geometry.floor_area import compute_floor_area
from src.geometry.openings import detect_openings
from src.geometry.wall_fitting import fit_walls
from src.models import (Adjacency, DamageRegion, Measurement, Opening, PropertyReport, Room, Wall)
from src.output.json_writer import write_output
from src.output.renderer import render_floor_plan
from src.room_ir import PropertyIR, RoomIR
from src.stitching.multi_room import stitch_rooms
from src.tiers.lidar import process_lidar
from src.tiers.photo import process_photos
from src.tiers.preprocessing import normalize_capture_dir, validate_capture_dir
from src.tiers.video import process_video

log = logging.getLogger("floorplan")

# Basic calibration (replaced by src/calibration/confidence.py once benchmark data exists).
TIER_MULTIPLIER = {"lidar": 1.0, "video": 2.5, "photo": 5.0}
BASE_ERROR = {"wall": 0.01, "ceiling": 0.01, "opening": 0.02, "damage": 0.05}  # meters

# RANSAC wall-line inlier distance per tier: SfM clouds are noisier than LiDAR.
WALL_INLIER_THRESHOLD = {"lidar": 0.03, "video": 0.06, "photo": 0.06}

MIN_ROOM_AREA = 1.0  # m2; a smaller polygon is a failed fit, not a room

FRONT_ENDS = {"lidar": process_lidar, "photo": process_photos, "video": process_video}


def _say(message: str) -> None:
    print(message, flush=True)


def _measurement(value: float, base_error: float, tier: str, unit: str = "m",
                 half_width: float | None = None) -> Measurement:
    """value +/- (base_error * tier multiplier), or +/- half_width when given."""
    half = base_error * TIER_MULTIPLIER[tier] if half_width is None else half_width
    return Measurement(value=float(value), confidence_low=float(value) - half,
                       confidence_high=float(value) + half, unit=unit)


def _place(point: tuple[float, float], transform: tuple[float, float, float]) -> tuple[float, float]:
    """Apply a (dx, dy, rotation_radians) room transform: rotate about the origin, then translate."""
    dx, dy, rot = transform
    c, s = np.cos(rot), np.sin(rot)
    return (float(c * point[0] - s * point[1] + dx), float(s * point[0] + c * point[1] + dy))


def _build_room(room: RoomIR, tier: str, transform: tuple[float, float, float]) -> Room:
    """RoomIR (after geometry) -> output Room in the global frame."""
    walls: list[Wall] = []
    height = _measurement(room.ceiling_height, BASE_ERROR["ceiling"], tier)
    for i, seg in enumerate(room.wall_segments):
        wall_id = f"{room.room_id}/wall-{i}"
        walls.append(Wall(
            id=wall_id, start=_place(seg.start, transform), end=_place(seg.end, transform),
            length=_measurement(seg.length, BASE_ERROR["wall"], tier), height=height,
            surface_id=wall_id))

    counters: dict[str, int] = {}
    openings: list[Opening] = []
    for det in room.openings:
        k = counters.get(det.type, 0)
        counters[det.type] = k + 1
        openings.append(Opening(
            id=f"{room.room_id}/{det.type}-{k}", type=det.type, wall_id=f"{room.room_id}/wall-{det.wall_index}",
            position_along_wall=_measurement(det.position_along_wall, BASE_ERROR["opening"], tier),
            width=_measurement(det.width, BASE_ERROR["opening"], tier),
            height=_measurement(det.height, BASE_ERROR["opening"], tier),
            confidence=min(1.0, max(0.0, det.confidence))))

    polygon = [_place(p, transform) for p in room.floor_polygon]
    perimeter = sum(seg.length for seg in room.wall_segments)
    area = compute_floor_area(room.floor_polygon)
    # First-order area error from per-wall length error e: dA ~ e * perimeter / 2.
    area_half = BASE_ERROR["wall"] * TIER_MULTIPLIER[tier] * perimeter / 2
    return Room(
        id=room.room_id, name=room.room_id.replace("-", " ").title(), walls=walls, openings=openings,
        ceiling_height=height, floor_area=_measurement(area, 0.0, tier, unit="m2", half_width=area_half),
        floor_polygon=polygon)


def _build_damage_region(d, tier: str) -> DamageRegion:
    """ProjectedDamage -> output DamageRegion; the extent interval widens with the tier."""
    err = BASE_ERROR["damage"] * TIER_MULTIPLIER[tier]
    return DamageRegion(
        id=d.id, surface_id=d.surface_id, damage_class=d.damage_detection.damage_class,
        extent_width=_measurement(d.extent_width, 0.0, tier, half_width=err),
        extent_height=_measurement(d.extent_height, 0.0, tier, half_width=err),
        area=area_measurement(d.extent_width, d.extent_height, err),
        location_on_surface=(float(d.location_on_surface[0]), float(d.location_on_surface[1])),
        confidence=float(min(1.0, max(0.0, d.damage_detection.confidence))),
        source_image=d.damage_detection.image_path)


def run_pipeline(
    capture_dir: str,
    tier: str,
    output_dir: str,
    drift_correction: bool = True,
    render: bool = True,
) -> PropertyReport:
    """Run preprocess -> tier front-end -> geometry -> stitch -> damage -> scope ->
    calibrate -> write JSON -> render, and return the PropertyReport.
    drift_correction=False is the ablation switch (--no-drift-correction).
    """
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    _say(f"Pipeline {__version__} | tier={tier} | drift_correction={drift_correction}")
    log.info("drift_correction=%s", drift_correction)

    def timed(name: str, start: float) -> None:
        timings[name] = time.perf_counter() - start

    # 1. preprocess
    t = time.perf_counter()
    for warning in validate_capture_dir(capture_dir, tier):
        log.warning(warning)
    normalized = normalize_capture_dir(capture_dir)
    if normalized.converted:
        _say(f"Converted {len(normalized.converted)} HEIC file(s) to JPEG")
    for bad in normalized.failed + normalized.videos_unreadable:
        log.warning("could not process %s", bad)
    timed("preprocess", t)

    # 2. tier front-end
    t = time.perf_counter()
    try:
        property_ir: PropertyIR = FRONT_ENDS[tier](capture_dir)
    except NotImplementedError:
        raise NotImplementedError(f"Step {tier} front-end not yet implemented") from None
    timed(f"{tier} front-end", t)
    _say(f"{len(property_ir.rooms)} room(s) found")

    # 3. geometry: walls, ceiling, openings, floor polygon per room
    t = time.perf_counter()
    solved: list[RoomIR] = []
    for room in property_ir.rooms:
        try:
            room.wall_segments, room.floor_polygon = fit_walls(
                room.point_cloud, inlier_threshold=WALL_INLIER_THRESHOLD[tier])
            room.ceiling_height, _ = detect_ceiling_height(room.point_cloud)
            room.openings = detect_openings(room.point_cloud, room.wall_segments)
            area = compute_floor_area(room.floor_polygon)
            if len(room.wall_segments) < 4 or area < MIN_ROOM_AREA:
                raise ValueError(f"degenerate fit ({len(room.wall_segments)} walls, {area:.2f} m2)")
            solved.append(room)
        except ValueError as exc:
            log.warning("geometry failed for %s, room dropped: %s", room.room_id, exc)
            _say(f"warning: geometry failed for {room.room_id}: {exc}")
    if not solved:
        raise RuntimeError("geometry failed for every room; check the capture")
    property_ir.rooms = solved
    timed("geometry", t)

    # 4. stitching
    t = time.perf_counter()
    property_ir = stitch_rooms(property_ir, drift_correction=drift_correction)
    _say(f"Stitched {len(property_ir.rooms)} room(s), "
         f"{len(property_ir.adjacencies or [])} adjacency link(s), "
         f"drift_correction={drift_correction}")
    timed("stitch", t)

    # 5-7. damage detection, concealed-damage rules, scope. Rooms are still in their own
    # frames here, so surface ids and (u, v) positions match the walls written to the report.
    t = time.perf_counter()
    images = list(dict.fromkeys(img for r in property_ir.rooms for img in r.images))
    extent_error = BASE_ERROR["damage"] * TIER_MULTIPLIER[tier]
    if images:
        detections = detect_damage(images)
        projected = project_damage_to_surfaces(detections, property_ir.rooms)
    else:
        detections, projected = [], []
    flags = check_concealed_damage(property_ir.rooms, projected)
    scope_items = generate_scope(projected, flags, extent_error=extent_error)
    damage_regions = [_build_damage_region(d, tier) for d in projected]
    _say(f"Damage: {len(detections)} detection(s), {len(projected)} placed on walls, "
         f"{len(flags)} concealed flag(s), {len(scope_items)} scope item(s)")
    timed("damage + scope", t)

    # 8-9. calibration (per-tier multiplier) happens while building the report
    t = time.perf_counter()
    transforms = property_ir.room_transforms or {}
    rooms = [_build_room(r, tier, transforms.get(r.room_id, (0.0, 0.0, 0.0))) for r in property_ir.rooms]
    adjacencies = [a if isinstance(a, Adjacency) else Adjacency.model_validate(a)
                   for a in (property_ir.adjacencies or [])]
    total_half = sum((r.floor_area.confidence_high - r.floor_area.value) for r in rooms)
    report = PropertyReport(
        capture_id=f"{Path(capture_dir).resolve().name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
        capture_tier=tier,
        capture_timestamp=datetime.now(timezone.utc),
        device="unknown",
        rooms=rooms,
        adjacencies=adjacencies,
        damage_regions=damage_regions,
        concealed_damage_flags=flags,
        scope_line_items=scope_items,
        total_floor_area=_measurement(sum(r.floor_area.value for r in rooms), 0.0, tier, unit="m2",
                                      half_width=total_half),
        room_count=len(rooms),
        processing_time_seconds=0.0,
        pipeline_version=__version__,
    )
    report = calibrate_measurements(report, tier)
    timed("calibrate + build report", t)

    # 10. outputs. Processing time is final before the JSON is written.
    report.processing_time_seconds = round(time.perf_counter() - t_start, 3)
    t = time.perf_counter()
    json_path = write_output(report, output_dir)
    timed("write JSON", t)
    _say(f"Wrote {json_path}")
    if render:
        t = time.perf_counter()
        plan_path = render_floor_plan(report, output_dir)
        timed("render", t)
        _say(f"Wrote {plan_path}")

    # 11. timing summary
    total = time.perf_counter() - t_start
    _say("Timing: " + ", ".join(f"{name} {secs:.2f}s" for name, secs in timings.items())
         + f" | total {total:.2f}s")
    return report
