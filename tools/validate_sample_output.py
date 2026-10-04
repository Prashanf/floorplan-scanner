"""Validate pipeline output on the sample data.

Checks every output/sample_*/ folder: the report loads through the pydantic schema, all
required fields exist (empty lists are fine, missing ones are not), every measurement has a
confidence interval that contains its value, and a floor plan PNG was written.

    python tools/validate_sample_output.py [output_dir]

Exit status is 1 when any folder fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import PropertyReport  # noqa: E402

REQUIRED = ["capture_id", "capture_tier", "rooms", "adjacencies", "damage_regions",
            "concealed_damage_flags", "scope_line_items", "total_floor_area", "room_count"]
ROOM_FIELDS = ["id", "name", "walls", "openings", "ceiling_height", "floor_area"]


def _interval_problems(label: str, m: dict) -> list[str]:
    if not all(k in m for k in ("value", "confidence_low", "confidence_high")):
        return [f"FAIL: {label} has no confidence interval"]
    if not m["confidence_low"] <= m["value"] <= m["confidence_high"]:
        return [f"FAIL: {label} interval [{m['confidence_low']:.3f}, {m['confidence_high']:.3f}] "
                f"does not contain {m['value']:.3f}"]
    return []


def validate_output(output_dir: Path) -> list[str]:
    """Return a list of 'FAIL: ...', 'WARN: ...' lines; ['PASS: ...'] when clean."""
    issues: list[str] = []
    json_path = output_dir / "report.json"
    if not json_path.exists():
        return [f"FAIL: no report.json in {output_dir}"]
    report = json.loads(json_path.read_text())

    for name in REQUIRED:
        if name not in report:
            issues.append(f"FAIL: missing field '{name}'")
    try:
        PropertyReport.model_validate(report)
    except Exception as exc:  # pydantic ValidationError
        issues.append(f"FAIL: schema validation: {str(exc).splitlines()[0]}")

    issues += _interval_problems("total_floor_area", report.get("total_floor_area", {}))
    for i, room in enumerate(report.get("rooms", [])):
        for name in ROOM_FIELDS:
            if name not in room:
                issues.append(f"FAIL: room {i} missing '{name}'")
        issues += _interval_problems(f"room {room.get('id', i)} ceiling_height", room.get("ceiling_height", {}))
        issues += _interval_problems(f"room {room.get('id', i)} floor_area", room.get("floor_area", {}))
        for j, wall in enumerate(room.get("walls", [])):
            issues += _interval_problems(f"room {room.get('id', i)} wall {j} length", wall.get("length", {}))
        for j, opening in enumerate(room.get("openings", [])):
            issues += _interval_problems(f"room {room.get('id', i)} opening {j} width", opening.get("width", {}))
    if report.get("room_count") != len(report.get("rooms", [])):
        issues.append("FAIL: room_count does not match the number of rooms")

    if not (output_dir / "floor_plan.png").exists():
        issues.append("WARN: no floor_plan.png")
    if not any(i.startswith("FAIL") for i in issues):
        issues.append(
            f"PASS: {report['room_count']} rooms, {report['total_floor_area']['value']:.1f} m2, "
            f"{len(report['adjacencies'])} adjacencies, {len(report['damage_regions'])} damage regions, "
            f"{len(report['concealed_damage_flags'])} concealed flags, {len(report['scope_line_items'])} scope items")
    return issues


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")
    folders = sorted(p for p in root.glob("sample_*") if p.is_dir())
    if not folders:
        print(f"no sample_* folders in {root}")
        return 1
    failed = False
    for folder in folders:
        print(f"\n=== {folder.name} ===")
        issues = validate_output(folder)
        for line in issues:
            print(f"  {line}")
        failed |= any(line.startswith("FAIL") for line in issues)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
