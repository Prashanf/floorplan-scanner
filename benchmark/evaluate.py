"""Benchmark gate evaluation against tape/laser ground truth.

Usage:
    python benchmark/evaluate.py --results-dir benchmark/results/ \\
        --ground-truth-dir benchmark/ground_truth/

Loads pipeline JSON outputs and ground-truth YAML files, matches by room id,
computes per-dimension errors, scores gates, and writes a Markdown report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def _load_ground_truths(gt_dir: str) -> dict[str, dict]:
    """Load all YAML ground-truth files, keyed by room_id."""
    rooms: dict[str, dict] = {}
    for p in sorted(Path(gt_dir).glob("*.yaml")):
        with open(p) as f:
            data = yaml.safe_load(f)
        if data is None:
            continue
        if "rooms" in data:
            for r in data["rooms"]:
                rooms[r["room_id"]] = r
        elif "room_id" in data:
            rooms[data["room_id"]] = data
    return rooms


def _load_results(results_dir: str) -> list[dict]:
    """Load all pipeline JSON reports."""
    reports: list[dict] = []
    for p in sorted(Path(results_dir).glob("*.json")):
        if p.name == "gates_report.json":
            continue
        with open(p) as f:
            reports.append(json.load(f))
    return reports


def _match_walls(pred_walls: list[dict], gt_walls: list[dict]) -> list[tuple[dict, dict]]:
    """Match predicted walls to ground-truth walls by length proximity (greedy)."""
    gt_remaining = list(gt_walls)
    pairs: list[tuple[dict, dict]] = []
    for pw in pred_walls:
        plen = pw["length"]["value"] if isinstance(pw["length"], dict) else pw["length"]
        best_idx, best_err = -1, float("inf")
        for i, gw in enumerate(gt_remaining):
            glen = gw["length_m"]
            err = abs(plen - glen)
            if err < best_err:
                best_err = err
                best_idx = i
        if best_idx >= 0:
            pairs.append((pw, gt_remaining.pop(best_idx)))
    return pairs


def _match_openings(pred_openings: list[dict], gt_openings: list[dict]) -> list[tuple[dict, dict]]:
    """Match predicted openings to ground-truth openings by type and width proximity."""
    gt_remaining = list(gt_openings)
    pairs: list[tuple[dict, dict]] = []
    for po in pred_openings:
        ptype = po.get("type", "")
        pwidth = po["width"]["value"] if isinstance(po["width"], dict) else po["width"]
        best_idx, best_err = -1, float("inf")
        for i, go in enumerate(gt_remaining):
            if go.get("type", "") != ptype:
                continue
            gwidth = go["width_m"]
            err = abs(pwidth - gwidth)
            if err < best_err:
                best_err = err
                best_idx = i
        if best_idx >= 0:
            pairs.append((po, gt_remaining.pop(best_idx)))
    return pairs


def _interval_contains(pred: dict, gt_value: float) -> bool:
    """Check if a Measurement's 90% CI contains the ground-truth value."""
    if isinstance(pred, dict) and "confidence_low" in pred:
        return pred["confidence_low"] <= gt_value <= pred["confidence_high"]
    return True


def evaluate(results_dir: str, ground_truth_dir: str) -> str:
    """Run full gate evaluation. Returns Markdown report string."""
    gt_rooms = _load_ground_truths(ground_truth_dir)
    reports = _load_results(results_dir)

    if not gt_rooms:
        return "# Gate Report\n\nNo ground-truth files found.\n"
    if not reports:
        return "# Gate Report\n\nNo result files found.\n"

    wall_errors: list[float] = []
    ceiling_errors: list[float] = []
    opening_errors: list[float] = []
    opening_count = 0
    opening_within_2cm = 0
    floor_errors: list[float] = []
    interval_checks: list[bool] = []
    room_details: list[str] = []

    # Repeatability: group predictions by (room_id, tier).
    repeat_map: dict[str, list[dict]] = defaultdict(list)

    missed_openings = 0
    phantom_openings = 0

    for report in reports:
        tier = report.get("capture_tier", "unknown")
        for pred_room in report.get("rooms", []):
            rid = pred_room["id"]
            gt = gt_rooms.get(rid)
            if gt is None:
                continue

            repeat_map[f"{rid}|{tier}"].append(pred_room)

            # --- Walls ---
            pairs = _match_walls(pred_room.get("walls", []), gt.get("walls", []))
            for pw, gw in pairs:
                plen = pw["length"]["value"] if isinstance(pw["length"], dict) else pw["length"]
                glen = gw["length_m"]
                err = plen - glen
                wall_errors.append(err)
                if isinstance(pw.get("length"), dict):
                    interval_checks.append(_interval_contains(pw["length"], glen))
                room_details.append(
                    f"| {rid} | wall {gw.get('id','')} | {glen:.3f} | {plen:.3f} | {err:+.3f} | {tier} |"
                )

            # --- Ceiling ---
            if "ceiling_height" in pred_room and "ceiling_height_m" in gt:
                pval = pred_room["ceiling_height"]["value"] if isinstance(pred_room["ceiling_height"], dict) else pred_room["ceiling_height"]
                gval = gt["ceiling_height_m"]
                err = pval - gval
                ceiling_errors.append(err)
                if isinstance(pred_room.get("ceiling_height"), dict):
                    interval_checks.append(_interval_contains(pred_room["ceiling_height"], gval))
                room_details.append(
                    f"| {rid} | ceiling | {gval:.3f} | {pval:.3f} | {err:+.3f} | {tier} |"
                )

            # --- Openings ---
            pred_ops = pred_room.get("openings", [])
            gt_ops = gt.get("openings", [])
            pairs_o = _match_openings(pred_ops, gt_ops)
            matched_pred_ids = {id(po) for po, _ in pairs_o}
            matched_gt_ids = {id(go) for _, go in pairs_o}

            missed_openings += len(gt_ops) - len(pairs_o)
            phantom_openings += len(pred_ops) - len(pairs_o)

            for po, go in pairs_o:
                pwidth = po["width"]["value"] if isinstance(po["width"], dict) else po["width"]
                gwidth = go["width_m"]
                err = pwidth - gwidth
                opening_errors.append(err)
                opening_count += 1
                if abs(err) <= 0.02:
                    opening_within_2cm += 1
                if isinstance(po.get("width"), dict):
                    interval_checks.append(_interval_contains(po["width"], gwidth))
                room_details.append(
                    f"| {rid} | {go.get('type','opening')} {go.get('id','')} | {gwidth:.3f} | {pwidth:.3f} | {err:+.3f} | {tier} |"
                )

            # --- Floor area ---
            if "floor_area" in pred_room and "floor_area_m2" in gt:
                pval = pred_room["floor_area"]["value"] if isinstance(pred_room["floor_area"], dict) else pred_room["floor_area"]
                gval = gt["floor_area_m2"]
                err = pval - gval
                floor_errors.append(err)
                if isinstance(pred_room.get("floor_area"), dict):
                    interval_checks.append(_interval_contains(pred_room["floor_area"], gval))

    # --- Gate scoring ---
    total_opening_events = opening_count + missed_openings + phantom_openings
    opening_hit_rate = opening_within_2cm / total_opening_events if total_opening_events else 0.0
    opening_gate = opening_hit_rate >= 0.85

    ceiling_abs = [abs(e) for e in ceiling_errors]
    ceiling_max = max(ceiling_abs) if ceiling_abs else 0.0
    ceiling_gate = ceiling_max <= 0.015

    # Repeatability: for each (room, tier) with 2+ runs, compare per-wall lengths.
    repeat_max_err = 0.0
    repeat_count = 0
    for key, runs in repeat_map.items():
        if len(runs) < 2:
            continue
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                walls_i = {w["id"]: w for w in runs[i].get("walls", [])}
                walls_j = {w["id"]: w for w in runs[j].get("walls", [])}
                for wid in set(walls_i) & set(walls_j):
                    li = walls_i[wid]["length"]["value"] if isinstance(walls_i[wid]["length"], dict) else walls_i[wid]["length"]
                    lj = walls_j[wid]["length"]["value"] if isinstance(walls_j[wid]["length"], dict) else walls_j[wid]["length"]
                    diff = abs(li - lj)
                    pct = diff / max(li, lj, 0.01)
                    repeat_max_err = max(repeat_max_err, diff)
                    repeat_count += 1
    repeat_gate = (repeat_max_err <= 0.01 or repeat_max_err / max(repeat_max_err, 0.01) <= 0.005) if repeat_count > 0 else None

    # Photo stitch footprint: total floor area error %.
    photo_floor_pct = None
    for report in reports:
        if report.get("capture_tier") != "photo":
            continue
        pred_total = report.get("total_floor_area", {})
        pred_val = pred_total["value"] if isinstance(pred_total, dict) else pred_total
        gt_total = sum(gt.get("floor_area_m2", 0) for gt in gt_rooms.values())
        if gt_total > 0:
            photo_floor_pct = abs(pred_val - gt_total) / gt_total * 100
    photo_stitch_gate = photo_floor_pct is not None and photo_floor_pct <= 8.0

    # Calibration score: % of intervals that contain ground truth.
    cal_score = sum(interval_checks) / len(interval_checks) * 100 if interval_checks else 0.0

    # --- Build Markdown ---
    lines: list[str] = []
    lines.append("# Benchmark Gate Report\n")

    lines.append("## Gate Summary\n")
    lines.append("| Gate | Value | Threshold | Pass |")
    lines.append("|------|-------|-----------|------|")

    def yn(b: bool | None) -> str:
        if b is None:
            return "N/A"
        return "PASS" if b else "**FAIL**"

    lines.append(f"| Opening widths (≤2cm on ≥85%) | {opening_hit_rate*100:.1f}% ({opening_within_2cm}/{total_opening_events}) | ≥85% | {yn(opening_gate)} |")
    lines.append(f"| Ceiling height | {ceiling_max*100:.1f} cm max | ≤1.5 cm | {yn(ceiling_gate)} |")
    if repeat_count > 0:
        lines.append(f"| Repeatability | {repeat_max_err*100:.2f} cm max ({repeat_count} pairs) | ≤1 cm or 0.5% | {yn(repeat_gate)} |")
    else:
        lines.append("| Repeatability | no repeat captures | ≤1 cm or 0.5% | N/A |")
    if photo_floor_pct is not None:
        lines.append(f"| Photo stitch footprint | {photo_floor_pct:.1f}% | ≤8% | {yn(photo_stitch_gate)} |")
    else:
        lines.append("| Photo stitch footprint | no photo-tier result | ≤8% | N/A |")
    lines.append(f"| Calibration (90% CI coverage) | {cal_score:.1f}% | ~90% | {'PASS' if 80 <= cal_score <= 100 else '**FAIL**'} |")

    lines.append("")
    lines.append("## Per-dimension errors\n")
    lines.append("| Room | Dimension | Ground Truth (m) | Predicted (m) | Error (m) | Tier |")
    lines.append("|------|-----------|------------------|---------------|-----------|------|")
    lines.extend(room_details)

    lines.append("")
    lines.append("## Error statistics\n")
    if wall_errors:
        lines.append(f"- Wall length: mean {sum(abs(e) for e in wall_errors)/len(wall_errors)*100:.2f} cm, "
                      f"max {max(abs(e) for e in wall_errors)*100:.2f} cm (n={len(wall_errors)})")
    if ceiling_errors:
        lines.append(f"- Ceiling height: mean {sum(abs(e) for e in ceiling_errors)/len(ceiling_errors)*100:.2f} cm, "
                      f"max {max(abs(e) for e in ceiling_errors)*100:.2f} cm (n={len(ceiling_errors)})")
    if opening_errors:
        lines.append(f"- Opening width: mean {sum(abs(e) for e in opening_errors)/len(opening_errors)*100:.2f} cm, "
                      f"max {max(abs(e) for e in opening_errors)*100:.2f} cm (n={len(opening_errors)})")
        lines.append(f"- Missed openings: {missed_openings}, phantom openings: {phantom_openings}")
    if floor_errors:
        lines.append(f"- Floor area: mean {sum(abs(e) for e in floor_errors)/len(floor_errors):.3f} m², "
                      f"max {max(abs(e) for e in floor_errors):.3f} m² (n={len(floor_errors)})")

    md = "\n".join(lines) + "\n"

    out_path = Path(results_dir) / "gates_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(md)
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate benchmark gates")
    parser.add_argument("--results-dir", required=True, help="Directory with pipeline JSON outputs")
    parser.add_argument("--ground-truth-dir", required=True, help="Directory with ground-truth YAML files")
    args = parser.parse_args()
    evaluate(args.results_dir, args.ground_truth_dir)


if __name__ == "__main__":
    main()
