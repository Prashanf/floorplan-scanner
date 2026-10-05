"""Model-based damage detection: OWL (v2 by default, v1 selectable) proposes boxes from text prompts, MobileSAM cuts a mask per box.

This is the alternative to the OpenCV rules in `detection.py`; `detect_damage(..., detector="mobilesam")`
dispatches here. Both return the same `DamageDetection` records, so everything downstream is unchanged.

Heavy dependencies (torch, transformers, mobile_sam) are imported only when this detector runs.
Install them with `pip install -r requirements-damage-model.txt` and fetch the MobileSAM weights with
`python tools/download_mobile_sam.py`. Without the weights the detector still runs and uses the
box area instead of the mask area (it says so in the log).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src import config as cfg
from src.damage.detection import (MAX_AREA_FRACTION, MAX_PER_IMAGE, DamageDetection, _clip01, _merge_overlaps,
                                  drop_persistent)

log = logging.getLogger("floorplan.damage")

# OWL models match "a photo of ..." phrases far better than bare noun phrases ("crack on wall" scored
# under 0.05 on clear cracks with OWL-ViT v1).
DEFAULT_PROMPTS: dict[str, list[str]] = {
    "crack": ["a photo of a crack in a wall", "a photo of a crack in plaster"],
    "water_stain": ["a photo of a water stain on a wall or ceiling", "a photo of a moisture mark on a wall"],
    "mold": ["a photo of mold on a wall", "a photo of black mildew spots"],
    "peeling_paint": ["a photo of peeling paint", "a photo of flaking chipped paint on a wall"],
    "hole": ["a photo of a hole in a wall", "a photo of a damaged plaster cavity"],
}
WEIGHTS_ENV = "FLOORPLAN_MOBILE_SAM_WEIGHTS"
_CACHE: dict = {}  # loaded models, kept for the life of the process


def _limit_threads_on_macos() -> None:
    """One torch thread on macOS. torch and pycolmap/open3d each bundle an OpenMP runtime there, and running
    the model after COLMAP in the same process crashes (OMP error #15 or SIGSEGV). Slower on CPU; no effect elsewhere."""
    import sys
    if sys.platform == "darwin" and "threads_limited" not in _CACHE:
        import torch
        torch.set_num_threads(1)
        _CACHE["threads_limited"] = True


def resolve_device(device: str = "auto") -> str:
    """"auto" is cuda when available, else cpu."""
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def find_sam_weights(explicit: Optional[str] = None) -> Optional[str]:
    """First existing MobileSAM checkpoint: the argument, $FLOORPLAN_MOBILE_SAM_WEIGHTS, then the usual places."""
    root = Path(__file__).resolve().parents[2]
    candidates = [explicit, os.environ.get(WEIGHTS_ENV), "mobile_sam.pt", "weights/mobile_sam.pt",
                  root / "weights" / "mobile_sam.pt", "/tmp/mobile_sam.pt"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


def _load_owl(model_name: str, device: str):
    key = ("owl", model_name, device)
    if key not in _CACHE:
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ImportError("the model damage detector needs torch and transformers: "
                              "pip install -r requirements-damage-model.txt") from exc
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device).eval()
        _CACHE[key] = (processor, model)
        log.info("loaded %s on %s", model_name, device)
    return _CACHE[key]


def _load_sam(weights: Optional[str], device: str):
    """SamPredictor, or None when mobile_sam or its weights are missing (box area is used instead)."""
    path = find_sam_weights(weights)
    if path is None:
        log.warning("MobileSAM weights not found (python tools/download_mobile_sam.py); using box areas")
        return None
    key = ("sam", path, device)
    if key not in _CACHE:
        try:
            from mobile_sam import SamPredictor, sam_model_registry
        except ImportError as exc:
            log.warning("mobile_sam is not installed (%s); using box areas", exc)
            return None
        sam = sam_model_registry["vit_t"](checkpoint=path)
        sam.to(device=device).eval()
        _CACHE[key] = SamPredictor(sam)
        log.info("loaded MobileSAM from %s on %s", path, device)
    return _CACHE[key]


def _owl_boxes(image, texts: list[str], processor, model, device: str, threshold: float):
    """[(x1, y1, x2, y2, score, text)] in pixels of `image` (PIL)."""
    import torch

    width, height = image.size
    inputs = processor(text=[texts], images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # OWLv2 pads the image to a square and predicts boxes in that square; v1 resizes without padding.
    side = max(height, width)
    sizes = torch.tensor([[side, side] if model.config.model_type == "owlv2" else [height, width]], device=device)
    if hasattr(processor, "post_process_grounded_object_detection"):
        result = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=sizes, threshold=threshold, text_labels=[texts])[0]
        labels = result["text_labels"]
    else:  # older transformers: integer labels index into `texts`
        post = getattr(processor, "post_process_object_detection", None) or \
            processor.image_processor.post_process_object_detection
        result = post(outputs=outputs, target_sizes=sizes, threshold=threshold)[0]
        labels = [texts[int(i)] for i in result["labels"]]
    found = []
    for box, score, text in zip(result["boxes"], result["scores"], labels):
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        x1, y1 = max(0, min(width - 1, x1)), max(0, min(height - 1, y1))
        x2, y2 = max(x1 + 1, min(width, x2)), max(y1 + 1, min(height, y2))
        found.append((x1, y1, x2, y2, float(score), text))
    return found


def _mask_stats(predictor, box: tuple[int, int, int, int], is_crack: bool):
    """(mask area, crack angle, crack thickness) from a MobileSAM mask for `box`; Nones when it fails."""
    try:
        masks, _, _ = predictor.predict(box=np.array(box), multimask_output=False)
    except Exception as exc:  # a bad box must not lose the other detections
        log.debug("MobileSAM failed for box %s: %s", box, exc)
        return None, None, None
    if masks is None or len(masks) == 0:
        return None, None, None
    mask = masks[0].astype(np.uint8)
    area = int(mask.sum())
    if not is_crack or area == 0:
        return area, None, None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return area, None, None
    (_, _), (rw, rh), angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    return area, float(angle if rw >= rh else angle + 90.0) % 180.0, float(min(rw, rh))


def detect_damage_model(
    images: list[str],
    persistent_frames: int = 0,
    confidence_threshold: float = cfg.MODEL_DAMAGE_THRESHOLD,
    device: str = "auto",
    mobile_sam_weights: Optional[str] = None,
    owl_model_name: str = cfg.MODEL_OWL_NAME,
    custom_prompts: Optional[dict[str, list[str]]] = None,
) -> list[DamageDetection]:
    """OWL-ViT + MobileSAM damage detection; same output contract as the heuristic `detect_damage`.

    Per image: OWL-ViT scores every prompt, boxes above `confidence_threshold` are kept (not more than
    MAX_AREA_FRACTION of the frame), MobileSAM gives each a mask whose area is `pixel_area` (cracks also get
    angle and thickness from the mask), overlapping same-class boxes are merged, and at most MAX_PER_IMAGE
    strongest are kept. Unreadable images are skipped. `persistent_frames` works as in the heuristic detector.
    """
    from PIL import Image

    prompts = custom_prompts or DEFAULT_PROMPTS
    texts = [q for queries in prompts.values() for q in queries]
    to_class = {q: cls for cls, queries in prompts.items() for q in queries}
    _limit_threads_on_macos()
    dev = resolve_device(device)
    processor, owl = _load_owl(owl_model_name, dev)
    sam = _load_sam(mobile_sam_weights, dev)
    log.info("model damage detection on %d image(s), device=%s, masks=%s", len(images), dev, sam is not None)

    per_image: list[list[DamageDetection]] = []
    for path in images:
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as exc:  # unreadable or missing file
            log.warning("could not read %s: %s", path, exc)
            continue
        width, height = pil.size
        boxes = [b for b in _owl_boxes(pil, texts, processor, owl, dev, confidence_threshold)
                 if (b[2] - b[0]) * (b[3] - b[1]) <= MAX_AREA_FRACTION * width * height]
        if sam is not None and boxes:
            sam.set_image(np.array(pil))
        found: list[DamageDetection] = []
        for x1, y1, x2, y2, score, text in boxes:
            damage_class = to_class[text]
            area, angle, thickness = (None, None, None)
            if sam is not None:
                area, angle, thickness = _mask_stats(sam, (x1, y1, x2, y2), damage_class == "crack")
            found.append(DamageDetection(
                image_path=path, bbox=(x1, y1, x2, y2), damage_class=damage_class, confidence=_clip01(score),
                pixel_area=area if area else (x2 - x1) * (y2 - y1), image_size=(width, height),
                angle_deg=angle, thickness_px=thickness))
        per_image.append(sorted(_merge_overlaps(found), key=lambda d: (-d.confidence, d.bbox))[:MAX_PER_IMAGE])

    if persistent_frames > 0:
        per_image = drop_persistent(per_image, persistent_frames, cfg.PERSISTENT_IOU)
    return [d for detections in per_image for d in detections]
