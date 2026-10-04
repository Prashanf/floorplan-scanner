"""End-to-end orchestration of one capture."""

from __future__ import annotations

from src.models import PropertyReport


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
    raise NotImplementedError("Not yet implemented")
