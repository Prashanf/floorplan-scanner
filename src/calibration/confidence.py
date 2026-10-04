"""Confidence-interval calibration per tier."""

from __future__ import annotations

from src.models import PropertyReport


def calibrate_measurements(report: PropertyReport, tier: str) -> PropertyReport:
    """Set confidence_low/high on every Measurement from per-type base
    uncertainty times a per-tier multiplier (or fitted parameters from
    calibration_data.json when present). Returns the updated report.
    """
    raise NotImplementedError("Not yet implemented")
