"""Write the report JSON and the published schema."""

from __future__ import annotations

import json
from pathlib import Path

from src.models import PropertyReport

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "output_schema.json"


def write_output(report: PropertyReport, output_dir: str) -> str:
    """Write report.json (indent=2) to output_dir, refresh schema/output_schema.json,
    and return the report path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n")

    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(json.dumps(PropertyReport.model_json_schema(), indent=2) + "\n")
    return str(report_path)
