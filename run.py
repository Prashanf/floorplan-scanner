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
              type=click.Path(path_type=Path),
              help="Where report.json and floor_plan.png go (directory or path to target file).")
@click.option("--damage-model", "--damage-detector", "damage_detector", default="opencv",
              type=click.Choice(["opencv", "mobilesam", "heuristic", "model", "nanoowl"], case_sensitive=False),
              show_default=True,
              help="Damage detection method: 'opencv' (OpenCV rules, default; 'heuristic' is the same) or 'mobilesam' (OWLv2 boxes + MobileSAM masks; see requirements-damage-model.txt).")
@click.option("--damage-threshold", type=float, default=None,
              help="mobilesam only: OWL score above which a box is kept (default from src/config.py).")
@click.option("--owl-model", default=None,
              help="mobilesam only: Hugging Face OWL model, e.g. google/owlvit-base-patch32 for v1 (default from src/config.py).")
@click.option("--ceiling-height", type=click.FloatRange(min=1.0, max=10.0), default=None,
              help="Known ceiling height in meters for scale recovery (photo and video). Skips auto-detection.")
@click.option("--photo-stitcher", type=click.Choice(["classical", "vlm"], case_sensitive=False),
              default="classical", show_default=True,
              help="Photo tier stitching method: 'classical' (CV door matching/camera poses) or 'vlm' (Qwen2-VL spatial reasoning + metric snapping).")
@click.option("--vlm-model", default="qwen2-vl-2b", show_default=True,
              help="VLM model variant: 'qwen2-vl-2b' (fast), 'qwen2-vl-7b' (high accuracy), or Hugging Face repo ID.")
@click.option("--vlm-adjacency", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Optional path to precomputed VLM adjacency JSON file.")
@click.option("--vlm-device", type=click.Choice(["auto", "cuda", "mps", "cpu"]), default="auto", show_default=True,
              help="Device for VLM inference.")
@click.option("--vlm-4bit", is_flag=True,
              help="Use 4-bit quantization (NF4) for 7B VLM on CUDA.")
@click.option("--no-drift-correction", "no_drift_correction", is_flag=True,
              help="Ablation: skip drift correction when stitching rooms.")
@click.option("--render/--no-render", default=True, show_default=True, help="Render the floor plan PNG.")
@click.option("--verbose", is_flag=True, help="Debug logging.")
def main(capture_dir: Path, tier: str, output_dir: Path, damage_detector: str, damage_threshold: float | None,
         owl_model: str | None, ceiling_height: float | None, photo_stitcher: str, vlm_model: str,
         vlm_adjacency: Path | None, vlm_device: str, vlm_4bit: bool,
         no_drift_correction: bool, render: bool, verbose: bool) -> None:
    """Process CAPTURE_DIR into a dimensioned, stitched property report."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    # If the user specified a filename (e.g. output/floor_plan.png), use its parent directory
    actual_output_dir = output_dir.parent if output_dir.suffix.lower() in (".png", ".json", ".jpg") else output_dir
    damage_options = {}
    if damage_threshold is not None:
        damage_options["confidence_threshold"] = damage_threshold
    if owl_model:
        damage_options["owl_model_name"] = owl_model
    if damage_options and damage_detector.lower() in ("opencv", "heuristic"):
        raise click.ClickException("--damage-threshold and --owl-model need --damage-model mobilesam")
    vlm_options = {
        "model_name": vlm_model,
        "device": vlm_device,
        "load_in_4bit": vlm_4bit,
    }
    try:
        report = run_pipeline(str(capture_dir), tier, str(actual_output_dir),
                              drift_correction=not no_drift_correction, render=render,
                              damage_detector=damage_detector.lower(), damage_options=damage_options,
                              ceiling_height=ceiling_height,
                              photo_stitcher=photo_stitcher.lower(),
                              vlm_model=vlm_model,
                              vlm_adjacency=str(vlm_adjacency) if vlm_adjacency else None,
                              vlm_options=vlm_options)
    except CaptureValidationError as exc:
        raise click.ClickException(f"Invalid capture for tier '{tier}': {exc}")
    except (NotImplementedError, ImportError) as exc:
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
