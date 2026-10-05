"""Run the heuristic and the MobileSAM+OWL-ViT damage detectors on the same images and compare them.

    python tools/compare_damage_detectors.py <images or folders...> --out output/damage_compare [--labels labels.json]

Writes comparison.json, comparison.md and overlays/<image>.jpg (blue boxes = heuristic, red = model).
labels.json (optional): {"IMG_1.jpg": [{"class": "crack", "bbox": [x1, y1, x2, y2]}], "IMG_2.jpg": []}
where an empty list marks a clean image; with it the tool also reports precision and recall per detector.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.damage.compare import MATCH_IOU, compare_detectors  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
COLORS = {"heuristic": (255, 100, 0), "mobilesam": (0, 0, 255)}  # BGR


def collect_images(paths: list[str]) -> list[str]:
    found: list[str] = []
    for item in paths:
        p = Path(item)
        found += [str(f) for f in sorted(p.rglob("*")) if f.suffix.lower() in IMAGE_SUFFIXES] if p.is_dir() else [str(p)]
    return found


def write_overlays(images: list[str], detections: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(images):
        frame = cv2.imread(path)
        if frame is None:
            continue
        for name, found in detections.items():
            for d in (x for x in found if x.image_path == path):
                x1, y1, x2, y2 = d.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[name], 3)
                cv2.putText(frame, f"{name[0].upper()}:{d.damage_class} {d.confidence:.2f}", (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[name], 2)
        cv2.imwrite(str(out / f"{index:03d}_{Path(path).stem}.jpg"), frame)


def markdown(result: dict) -> str:
    lines = ["# Damage detector comparison", "", "| | heuristic | mobilesam |", "|---|---|---|"]
    s = result["summary"]
    for field in ("detections", "images_with_detections", "mean_confidence", "seconds_per_image"):
        lines.append(f"| {field} | {s['heuristic'][field]} | {s['mobilesam'][field]} |")
    classes = sorted(set(s["heuristic"]["per_class"]) | set(s["mobilesam"]["per_class"]))
    for c in classes:
        lines.append(f"| {c} | {s['heuristic']['per_class'].get(c, 0)} | {s['mobilesam']['per_class'].get(c, 0)} |")
    a = result["agreement"]
    lines += ["", f"Agreement (IoU >= {MATCH_IOU}): {a['same_class']} same class, {a['same_place_other_class']} same place "
              f"but another class, {a['only_first']} only heuristic, {a['only_second']} only mobilesam."]
    if "scores" in result:
        lines += ["", "## Against labels", "", "| detector | precision | recall | F1 | TP | FP | FN |", "|---|---|---|---|---|---|---|"]
        for name, score in result["scores"].items():
            o = score["overall"]
            fmt = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
            lines.append(f"| {name} | {fmt(o['precision'])} | {fmt(o['recall'])} | {fmt(o['f1'])} | {o['tp']} | {o['fp']} | {o['fn']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="image files or folders (searched recursively)")
    parser.add_argument("--out", default="output/damage_compare")
    parser.add_argument("--labels", help="labels.json with hand-drawn boxes (see above)")
    parser.add_argument("--iou", type=float, default=MATCH_IOU)
    parser.add_argument("--threshold", type=float, help="OWL-ViT score threshold (default from src/config.py)")
    args = parser.parse_args()

    images = collect_images(args.images)
    if not images:
        sys.exit("no images found")
    labels = json.loads(Path(args.labels).read_text()) if args.labels else None
    extra = {"confidence_threshold": args.threshold} if args.threshold is not None else {}
    result = compare_detectors(images, labels, args.iou, **extra)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_overlays(images, result["detections"], out / "overlays")
    serial = {k: v for k, v in result.items() if k != "detections"}
    serial["detections"] = {name: [{"image": d.image_path, "class": d.damage_class, "bbox": list(d.bbox),
                                    "confidence": round(d.confidence, 3), "pixel_area": d.pixel_area}
                                   for d in found] for name, found in result["detections"].items()}
    (out / "comparison.json").write_text(json.dumps(serial, indent=2))
    (out / "comparison.md").write_text(markdown(result))
    print(markdown(result))
    print(f"wrote {out}/comparison.json, comparison.md, overlays/")


if __name__ == "__main__":
    main()
