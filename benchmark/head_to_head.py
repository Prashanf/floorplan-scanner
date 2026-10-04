"""Head-to-head comparison against a consumer scanning app export.

Usage:
    python benchmark/head_to_head.py \\
        --ours benchmark/results/room-1.json \\
        --theirs benchmark/competitor/room-1_polycam.json \\
        --ground-truth benchmark/ground_truth/room-1.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _get_rooms(data: dict) -> list[dict]:
    if "rooms" in data:
        return data["rooms"]
    if "room_id" in data or "walls" in data:
        return [data]
    return []


def _room_id(room: dict) -> str:
    return room.get("id", room.get("room_id", "room"))


def _mval(v) -> float | None:
    if isinstance(v, dict):
        return v.get("value")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _wall_length(w: dict) -> float | None:
    return _mval(w.get("length", w.get("length_m")))


def _opening_width(o: dict) -> float | None:
    return _mval(o.get("width", w.get("width_m") if (w := o) else None))


def _ceiling(room: dict) -> float | None:
    return _mval(room.get("ceiling_height", room.get("ceiling_height_m")))


def _floor_area(room: dict) -> float | None:
    return _mval(room.get("floor_area", room.get("floor_area_m2")))


def _match_by_value(pred: list[float], gt: list[float]) -> list[tuple[int, int]]:
    """Greedy match predicted values to GT values by proximity."""
    gt_avail = list(range(len(gt)))
    pairs: list[tuple[int, int]] = []
    for pi, pv in enumerate(pred):
        best_gi, best_err = -1, float("inf")
        for gi in gt_avail:
            err = abs(pv - gt[gi])
            if err < best_err:
                best_err = err
                best_gi = gi
        if best_gi >= 0:
            pairs.append((pi, best_gi))
            gt_avail.remove(best_gi)
    return pairs


def compare(ours_path: str, theirs_path: str, ground_truth_path: str) -> str:
    with open(ours_path) as f:
        ours_data = json.load(f)
    with open(theirs_path) as f:
        theirs_data = json.load(f)
    with open(ground_truth_path) as f:
        gt_data = yaml.safe_load(f)

    our_rooms = {_room_id(r): r for r in _get_rooms(ours_data)}
    their_rooms = {_room_id(r): r for r in _get_rooms(theirs_data)}
    gt_rooms_list = _get_rooms(gt_data)
    gt_rooms = {_room_id(r): r for r in gt_rooms_list}

    lines: list[str] = []
    lines.append("# Head-to-Head Comparison\n")
    lines.append("| Dimension | Ground Truth | Ours | Our Error | Theirs | Their Error | Winner |")
    lines.append("|-----------|-------------|------|-----------|--------|-------------|--------|")

    wins = 0
    total = 0

    def row(dim: str, gt_val: float, our_val: float | None, their_val: float | None):
        nonlocal wins, total
        our_err = abs(our_val - gt_val) if our_val is not None else None
        their_err = abs(their_val - gt_val) if their_val is not None else None

        if our_val is None and their_val is None:
            return

        total += 1
        if our_err is not None and their_err is not None:
            if our_err < their_err - 0.001:
                winner = "Ours"
                wins += 1
            elif their_err < our_err - 0.001:
                winner = "Theirs"
            else:
                winner = "Tie"
                wins += 1
        elif our_err is not None:
            winner = "Ours"
            wins += 1
        else:
            winner = "Theirs"

        o_str = f"{our_val:.3f}" if our_val is not None else "—"
        t_str = f"{their_val:.3f}" if their_val is not None else "—"
        oe_str = f"{our_err:.3f}" if our_err is not None else ""
        te_str = f"{their_err:.3f}" if their_err is not None else ""
        lines.append(f"| {dim} | {gt_val:.3f} | {o_str} | {oe_str} | {t_str} | {te_str} | {winner} |")

    for rid, gt_room in sorted(gt_rooms.items()):
        our_room = our_rooms.get(rid, {})
        their_room = their_rooms.get(rid, {})

        # Walls — match by length proximity
        gt_walls = gt_room.get("walls", [])
        gt_wlens = [w["length_m"] for w in gt_walls]
        gt_wnames = [w.get("id", f"wall-{i}") for i, w in enumerate(gt_walls)]

        our_walls = our_room.get("walls", [])
        our_wlens = [_wall_length(w) for w in our_walls]
        our_wlens_clean = [v for v in our_wlens if v is not None]

        their_walls = their_room.get("walls", [])
        their_wlens = [_wall_length(w) for w in their_walls]
        their_wlens_clean = [v for v in their_wlens if v is not None]

        our_matched: dict[int, float] = {}
        for pi, gi in _match_by_value(our_wlens_clean, gt_wlens):
            our_matched[gi] = our_wlens_clean[pi]

        their_matched: dict[int, float] = {}
        for pi, gi in _match_by_value(their_wlens_clean, gt_wlens):
            their_matched[gi] = their_wlens_clean[pi]

        for gi, gt_len in enumerate(gt_wlens):
            row(f"{rid}/{gt_wnames[gi]}", gt_len, our_matched.get(gi), their_matched.get(gi))

        # Openings — match by type + width proximity
        gt_ops = gt_room.get("openings", [])
        our_ops = our_room.get("openings", [])
        their_ops = their_room.get("openings", [])

        for go in gt_ops:
            gtype = go.get("type", "")
            gwidth = go["width_m"]
            gname = go.get("id", gtype)

            our_val = None
            best_err = float("inf")
            for oo in our_ops:
                otype = oo.get("type", "")
                if otype != gtype:
                    continue
                ow = _opening_width(oo)
                if ow is not None and abs(ow - gwidth) < best_err:
                    best_err = abs(ow - gwidth)
                    our_val = ow

            their_val = None
            best_err = float("inf")
            for to in their_ops:
                ttype = to.get("type", "")
                if ttype != gtype:
                    continue
                tw = _opening_width(to)
                if tw is not None and abs(tw - gwidth) < best_err:
                    best_err = abs(tw - gwidth)
                    their_val = tw

            row(f"{rid}/{gname}", gwidth, our_val, their_val)

        # Ceiling
        gt_ceil = gt_room.get("ceiling_height_m")
        if gt_ceil is not None:
            row(f"{rid}/ceiling", gt_ceil, _ceiling(our_room), _ceiling(their_room))

        # Floor area
        gt_fa = gt_room.get("floor_area_m2")
        if gt_fa is not None:
            row(f"{rid}/floor_area", gt_fa, _floor_area(our_room), _floor_area(their_room))

    lines.append("")
    pct = wins / total * 100 if total else 0
    lines.append(f"**Beat or tied on {wins}/{total} dimensions ({pct:.0f}%)**")
    gate = pct >= 70
    lines.append(f"\nGate (≥70%): {'PASS' if gate else '**FAIL**'}")

    md = "\n".join(lines) + "\n"
    print(md)
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description="Head-to-head vs competitor app")
    parser.add_argument("--ours", required=True, help="Our pipeline JSON output")
    parser.add_argument("--theirs", required=True, help="Competitor app JSON export")
    parser.add_argument("--ground-truth", required=True, help="Ground-truth YAML")
    args = parser.parse_args()
    compare(args.ours, args.theirs, args.ground_truth)


if __name__ == "__main__":
    main()
