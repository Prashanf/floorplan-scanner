"""Compare damage detectors on the same images: agreement between them and, with labels, precision and recall."""

from __future__ import annotations

import time
from collections import Counter
from typing import Callable, Optional

from src.damage.detection import DamageDetection, _iou, detect_damage

MATCH_IOU = 0.3


def match_boxes(a: list, b: list, iou: float = MATCH_IOU, same_class: bool = True) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by IoU: [(index in a, index in b, iou)] for pairs at or above `iou`.

    Items need `.bbox` and `.damage_class`. With `same_class` only equal classes can match.
    """
    pairs = sorted(((_iou(x.bbox, y.bbox), i, j) for i, x in enumerate(a) for j, y in enumerate(b)
                    if not same_class or x.damage_class == y.damage_class), reverse=True)
    used_a, used_b, matches = set(), set(), []
    for value, i, j in pairs:
        if value < iou:
            break
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matches.append((i, j, value))
    return matches


def agreement(first: list[DamageDetection], second: list[DamageDetection], iou: float = MATCH_IOU) -> dict:
    """How two detectors' outputs on the same images overlap (matching is per image)."""
    both_same = both_other = 0
    only_first = only_second = 0
    for image in sorted({d.image_path for d in first} | {d.image_path for d in second}):
        a = [d for d in first if d.image_path == image]
        b = [d for d in second if d.image_path == image]
        same = match_boxes(a, b, iou, True)
        left_a = [x for k, x in enumerate(a) if k not in {i for i, _, _ in same}]
        left_b = [y for k, y in enumerate(b) if k not in {j for _, j, _ in same}]
        other = match_boxes(left_a, left_b, iou, False)  # same place, different class
        both_same += len(same)
        both_other += len(other)
        only_first += len(left_a) - len(other)
        only_second += len(left_b) - len(other)
    return {"same_class": both_same, "same_place_other_class": both_other,
            "only_first": only_first, "only_second": only_second}


def score_against_labels(detections: list[DamageDetection], labels: dict[str, list[dict]],
                         iou: float = MATCH_IOU) -> dict:
    """Precision, recall and F1 against hand labels; per class and overall.

    `labels` maps an image path (or its file name) to [{"class": ..., "bbox": [x1, y1, x2, y2]}]; an empty
    list means the image is clean. Only images that appear in `labels` are scored. A detection is right when it
    overlaps an unmatched label of its class by at least `iou`.
    """
    def key_of(path: str) -> Optional[str]:
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        return path if path in labels else name if name in labels else None

    tp, fp, fn = Counter(), Counter(), Counter()
    detected_by_key: dict[str, list] = {}
    for d in detections:
        key = key_of(d.image_path)
        if key is not None:
            detected_by_key.setdefault(key, []).append(d)
    for key, truth in labels.items():
        class _Box:  # labels share the .bbox / .damage_class interface of detections
            def __init__(self, item):
                self.bbox, self.damage_class = tuple(item["bbox"]), item["class"]
        truth_boxes = [_Box(t) for t in truth]
        found = detected_by_key.get(key, [])
        matches = match_boxes(found, truth_boxes, iou, True)
        hit_found, hit_truth = {i for i, _, _ in matches}, {j for _, j, _ in matches}
        for i, d in enumerate(found):
            (tp if i in hit_found else fp)[d.damage_class] += 1
        for j, t in enumerate(truth_boxes):
            if j not in hit_truth:
                fn[t.damage_class] += 1

    def prf(t: int, f: int, n: int) -> dict:
        precision = t / (t + f) if t + f else None
        recall = t / (t + n) if t + n else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if t + f + n else None)
        return {"tp": t, "fp": f, "fn": n, "precision": precision, "recall": recall, "f1": f1}

    classes = sorted(set(tp) | set(fp) | set(fn))
    return {"overall": prf(sum(tp.values()), sum(fp.values()), sum(fn.values())),
            "per_class": {c: prf(tp[c], fp[c], fn[c]) for c in classes}}


def summarize(detections: list[DamageDetection], seconds: float, n_images: int) -> dict:
    classes = Counter(d.damage_class for d in detections)
    return {"detections": len(detections), "per_class": dict(classes),
            "mean_confidence": round(sum(d.confidence for d in detections) / len(detections), 3) if detections else None,
            "images_with_detections": len({d.image_path for d in detections}), "images": n_images,
            "seconds": round(seconds, 2), "seconds_per_image": round(seconds / max(n_images, 1), 3)}


def compare_detectors(images: list[str], labels: Optional[dict[str, list[dict]]] = None, iou: float = MATCH_IOU,
                      run: Callable = detect_damage, **model_kwargs) -> dict:
    """Run the heuristic and the model detector on `images`; returns detections and the comparison.

    Result keys: "detections" ({name: [DamageDetection]}), "summary", "agreement" (heuristic = first,
    mobilesam = second) and, when `labels` is given, "scores".
    """
    outputs, summary = {}, {}
    for name in ("heuristic", "mobilesam"):
        started = time.perf_counter()
        outputs[name] = run(images, detector=name, **(model_kwargs if name == "mobilesam" else {}))
        summary[name] = summarize(outputs[name], time.perf_counter() - started, len(images))
    result = {"detections": outputs, "summary": summary,
              "agreement": agreement(outputs["heuristic"], outputs["mobilesam"], iou)}
    if labels is not None:
        result["scores"] = {name: score_against_labels(dets, labels, iou) for name, dets in outputs.items()}
    return result
