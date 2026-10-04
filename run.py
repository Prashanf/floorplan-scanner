"""CLI: one command per capture.

    python run.py <capture_dir> --tier photo|video|lidar [--output-dir ./output/]
                  [--no-drift-correction] [--render/--no-render] [--verbose]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from src.pipeline import run_pipeline
from src.tiers.preprocessing import CaptureValidationError


@click.command()
@click.argument("capture_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--tier", required=True, type=click.Choice(["photo", "video", "lidar"]),
              help="Input tier of the capture.")
@click.option("--output-dir", default="./output/", show_default=True,
              type=click.Path(file_okay=False, path_type=Path),
              help="Where report.json and floor_plan.png go.")
@click.option("--no-drift-correction", "no_drift_correction", is_flag=True,
              help="Ablation: skip drift correction when stitching rooms.")
@click.option("--render/--no-render", default=True, show_default=True, help="Render the floor plan PNG.")
@click.option("--verbose", is_flag=True, help="Debug logging.")
def main(capture_dir: Path, tier: str, output_dir: Path, no_drift_correction: bool,
         render: bool, verbose: bool) -> None:
    """Process CAPTURE_DIR into a dimensioned, stitched property report."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        report = run_pipeline(str(capture_dir), tier, str(output_dir),
                              drift_correction=not no_drift_correction, render=render)
    except CaptureValidationError as exc:
        raise click.ClickException(f"Invalid capture for tier '{tier}': {exc}")
    except NotImplementedError as exc:
        raise click.ClickException(str(exc))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:  # reconstruction or geometry gave nothing usable
        if verbose:
            raise
        raise click.ClickException(f"{tier} tier could not process this capture: {exc} (run with --verbose for details)")

    click.echo("\n--- Summary ---")
    click.echo(f"tier: {report.capture_tier} | rooms: {report.room_count} | "
               f"total floor area: {report.total_floor_area.value:.2f} m² "
               f"[{report.total_floor_area.confidence_low:.2f}, {report.total_floor_area.confidence_high:.2f}]")
    for room in report.rooms:
        click.echo(f"  {room.id}: {room.floor_area.value:.2f} m², ceiling {room.ceiling_height.value:.2f} m, "
                   f"{len(room.walls)} walls, {len(room.openings)} opening(s)")
    if report.room_count == 0:
        click.echo("no rooms could be reconstructed from this capture (see warnings in the report)")
    for line in report.warnings:
        click.echo(f"  warning: {line}")
    click.echo(f"report: {output_dir / 'report.json'}")
    if render:
        click.echo(f"plan:   {output_dir / 'floor_plan.png'}")


if __name__ == "__main__":
    sys.exit(main())
