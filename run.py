"""CLI: one command per capture.

    python run.py <capture_dir> --tier photo|video|lidar [--output-dir ./output/]
                  [--no-drift-correction] [--render/--no-render] [--verbose]

Steps (unimplemented ones print "Step X not yet implemented" and the run continues):
validate -> preprocess -> tier front-end -> geometry -> stitch -> damage -> scope
-> calibrate -> write JSON -> render -> summary.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import click

from src import __version__
from src.models import Measurement, PropertyReport
from src.room_ir import PointCloud, PropertyIR, RoomIR
from src.tiers.preprocessing import (
    CaptureValidationError,
    normalize_capture_dir,
    validate_capture_dir,
)

log = logging.getLogger("floorplan")

DONE, PENDING = "done", "not yet implemented"


class Steps:
    """Runs pipeline steps, records status, and tolerates unimplemented stubs."""

    def __init__(self) -> None:
        self.status: dict[str, str] = {}
        self.timings: dict[str, float] = {}

    def run(self, name: str, fn, *args, **kwargs):
        """Call fn. NotImplementedError marks the step pending and returns None."""
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except NotImplementedError:
            click.echo(f"Step {name} not yet implemented")
            self.status[name] = PENDING
            return None
        finally:
            self.timings[name] = time.perf_counter() - start
        self.status[name] = DONE
        log.info("Step %s done in %.2fs", name, self.timings[name])
        return result


def _zero(unit: str = "m") -> Measurement:
    return Measurement(value=0.0, confidence_low=0.0, confidence_high=0.0, unit=unit)


def _empty_report(tier: str, capture_dir: str, rooms: int, elapsed: float) -> PropertyReport:
    """Report with no geometry yet; lists are present but empty so the JSON shape is stable."""
    return PropertyReport(
        capture_id=f"{Path(capture_dir).resolve().name}-{uuid.uuid4().hex[:8]}",
        capture_tier=tier,
        capture_timestamp=datetime.now(timezone.utc),
        device="unknown",
        rooms=[],
        adjacencies=[],
        damage_regions=[],
        concealed_damage_flags=[],
        scope_line_items=[],
        total_floor_area=_zero("m2"),
        room_count=rooms,
        processing_time_seconds=elapsed,
        pipeline_version=__version__,
    )


def _placeholder_ir(tier: str, capture_dir: str) -> PropertyIR:
    """Empty one-room IR so downstream stubs still get a typed input while front-ends are pending."""
    import numpy as np

    room = RoomIR(room_id="room-1", point_cloud=PointCloud(points=np.empty((0, 3))), tier=tier)
    return PropertyIR(rooms=[room], tier=tier, capture_dir=capture_dir)


@click.command()
@click.argument("capture_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--tier", required=True, type=click.Choice(["photo", "video", "lidar"]),
              help="Input tier of the capture.")
@click.option("--output-dir", default="./output/", show_default=True,
              type=click.Path(file_okay=False, path_type=Path), help="Where report.json and floor_plan.png go.")
@click.option("--no-drift-correction", "no_drift_correction", is_flag=True,
              help="Ablation: skip drift correction when stitching rooms.")
@click.option("--render/--no-render", default=True, show_default=True, help="Render the floor plan PNG.")
@click.option("--verbose", is_flag=True, help="Debug logging.")
def main(capture_dir: Path, tier: str, output_dir: Path, no_drift_correction: bool,
         render: bool, verbose: bool) -> None:
    """Process CAPTURE_DIR into a dimensioned, stitched property report."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    drift_correction = not no_drift_correction
    click.echo(f"floorplan-scanner {__version__} | tier={tier} | drift_correction={drift_correction}")
    t0 = time.perf_counter()
    steps = Steps()

    # 1. validate
    try:
        for warning in validate_capture_dir(str(capture_dir), tier):
            click.echo(f"warning: {warning}", err=True)
    except CaptureValidationError as exc:
        raise click.ClickException(f"Invalid capture for tier '{tier}': {exc}")
    steps.status["validate"] = DONE

    # 2. preprocess
    norm = steps.run("preprocess", normalize_capture_dir, str(capture_dir))
    if norm is not None:
        if norm.converted:
            click.echo(f"Converted {len(norm.converted)} HEIC file(s) to JPEG")
        for bad in norm.failed + norm.videos_unreadable:
            click.echo(f"warning: could not process {bad}", err=True)

    # 3. tier front-end
    from src.tiers.lidar import process_lidar
    from src.tiers.photo import process_photos
    from src.tiers.video import process_video

    front_end = {"lidar": process_lidar, "photo": process_photos, "video": process_video}[tier]
    property_ir = steps.run(f"{tier} front-end", front_end, str(capture_dir))
    if property_ir is None:
        property_ir = _placeholder_ir(tier, str(capture_dir))

    # 4. geometry (per room)
    from src.geometry.ceiling import detect_ceiling_height
    from src.geometry.floor_area import compute_floor_area
    from src.geometry.openings import detect_openings
    from src.geometry.wall_fitting import fit_walls

    for room in property_ir.rooms:
        def _geometry(room: RoomIR = room) -> None:
            room.wall_segments, room.floor_polygon = fit_walls(room.point_cloud)
            room.ceiling_height, _ = detect_ceiling_height(room.point_cloud)
            room.openings = detect_openings(room.point_cloud, room.wall_segments)
            compute_floor_area(room.floor_polygon)

        if steps.run("geometry", _geometry) is None and steps.status["geometry"] == PENDING:
            break

    # 5. stitch
    from src.stitching.multi_room import stitch_rooms

    log.info("drift_correction=%s", drift_correction)
    stitched = steps.run("stitch", stitch_rooms, property_ir, drift_correction=drift_correction)
    if stitched is not None:
        property_ir = stitched

    # 6. damage -> 7. scope
    from src.damage.concealed_rules import check_concealed_damage
    from src.damage.detection import detect_damage
    from src.damage.scope import generate_scope
    from src.damage.surface_projection import project_damage_to_surfaces

    images = [img for room in property_ir.rooms for img in room.images]
    detections = steps.run("damage detection", detect_damage, images) or []
    projected = steps.run("surface projection", project_damage_to_surfaces,
                          detections, property_ir.rooms) or []
    flags = steps.run("concealed damage", check_concealed_damage, property_ir.rooms, projected) or []
    steps.run("scope", generate_scope, projected, flags)

    # 8. calibrate -> 9. write JSON -> 10. render
    from src.calibration.confidence import calibrate_measurements
    from src.output.json_writer import write_output
    from src.output.renderer import render_floor_plan

    report = _empty_report(tier, str(capture_dir), len(property_ir.rooms), time.perf_counter() - t0)
    calibrated = steps.run("calibrate", calibrate_measurements, report, tier)
    if calibrated is not None:
        report = calibrated

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = steps.run("write JSON", write_output, report, str(output_dir))
    plan_path = steps.run("render", render_floor_plan, report, str(output_dir)) if render else None

    # 11. summary
    total = time.perf_counter() - t0
    click.echo("\n--- Summary ---")
    click.echo(f"capture: {capture_dir}  tier: {tier}  rooms: {len(property_ir.rooms)}")
    click.echo(f"report: {json_path or 'not written'}")
    click.echo(f"plan:   {plan_path or ('skipped (--no-render)' if not render else 'not rendered')}")
    pending = [name for name, state in steps.status.items() if state == PENDING]
    click.echo(f"steps implemented: {len(steps.status) - len(pending)}/{len(steps.status)}"
               + (f" | pending: {', '.join(pending)}" if pending else ""))
    click.echo(f"time: {total:.2f}s")


if __name__ == "__main__":
    sys.exit(main())
