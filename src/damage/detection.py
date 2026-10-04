"""Classical-CV damage detection (cracks, water stains, mold, peeling paint, holes)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from src import config as cfg

log = logging.getLogger("floorplan.damage")

WORK_SIZE = 1280  # long image side used for detection; thresholds below are at this scale
MAX_AREA_FRACTION = cfg.MAX_DETECTION_AREA_FRACTION  # a detection covering this much of the frame is a wall or object, not damage
MAX_PER_IMAGE = cfg.MAX_DETECTIONS_PER_IMAGE

CRACK_MIN_LENGTH = cfg.CRACK_EDGE_MIN_LENGTH
CRACK_MIN_ASPECT = 4.0
CRACK_MIN_CONTRAST = 15.0  # gray levels darker than the surrounding wall
STAIN_MIN_AREA = cfg.STAIN_MIN_AREA
MOLD_MIN_AREA = cfg.MOLD_MIN_AREA
PEEL_MIN_AREA = 800
PEEL_STD_THRESHOLD = 12.0
HOLE_MIN_AREA = 150
HOLE_MIN_CIRCULARITY = 0.7
NMS_IOU = 0.3


@dataclass
class DamageDetection:
    """One detected damage region in image space."""

    image_path: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    damage_class: str  # crack | water_stain | mold | hole | peeling_paint
    confidence: float
    pixel_area: int
    image_size: tuple[int, int] = (0, 0)  # (width, height) of the source image, pixels
    angle_deg: Optional[float] = None  # cracks: principal direction, degrees from image horizontal (0-180)
    thickness_px: Optional[float] = None  # cracks: line thickness in source-image pixels


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _components(mask: np.ndarray, min_area: int, max_area: float):
    """Yield (x, y, w, h, area, label_mask_index) for each connected component in range."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if min_area <= area <= max_area:
            yield int(x), int(y), int(w), int(h), int(area), labels == i


def _clean(mask: np.ndarray, size: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


def _color_blobs(hsv: np.ndarray, lo: tuple, hi: tuple, min_area: int, damage_class: str):
    """Components of an HSV range. Confidence = how much of the bbox the blob fills."""
    mask = _clean(cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)))
    max_area = MAX_AREA_FRACTION * mask.size
    for x, y, w, h, area, _ in _components(mask, min_area, max_area):
        fill = area / float(w * h)
        size_bonus = min(1.0, area / (4.0 * min_area))
        yield (x, y, x + w, y + h), damage_class, _clip01(0.3 + 0.4 * fill + 0.3 * size_bonus), area, None, None


def _detect_stains(hsv: np.ndarray):
    return _color_blobs(hsv, cfg.WATER_STAIN_HSV["lo"], cfg.WATER_STAIN_HSV["hi"], STAIN_MIN_AREA, "water_stain")


def _detect_mold(hsv: np.ndarray):
    return _color_blobs(hsv, cfg.MOLD_HSV["lo"], cfg.MOLD_HSV["hi"], MOLD_MIN_AREA, "mold")


def _detect_cracks(gray: np.ndarray):
    """Long, thin, dark edge runs. Straight object edges are rejected by the contrast test:
    a crack is darker than the wall on both sides, an edge is dark on one side only."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(blurred, 50, 150), np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        length = cv2.arcLength(contour, False) / 2.0  # a closed trace of a thin line is ~2x its length
        if length < CRACK_MIN_LENGTH:
            continue
        (_, _), (rw, rh), angle = cv2.minAreaRect(contour)
        long_side, short_side = max(rw, rh), max(min(rw, rh), 1.0)
        if long_side / short_side < CRACK_MIN_ASPECT:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        pad = 12
        x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, gray.shape[1]), min(y + h + pad, gray.shape[0])
        patch = gray[y0:y1, x0:x1].astype(np.float32)
        line = np.zeros(patch.shape, np.uint8)
        cv2.drawContours(line, [contour - np.array([x0, y0])], -1, 255, -1)
        line = cv2.erode(line, np.ones((3, 3), np.uint8)) > 0
        ring = (cv2.dilate(line.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & ~(
            cv2.dilate(line.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
        if line.sum() < 5 or ring.sum() < 20:
            continue
        contrast = float(patch[ring].mean() - patch[line].mean())
        if contrast < CRACK_MIN_CONTRAST:
            continue
        # direction of the long side, degrees from horizontal in [0, 180)
        direction = angle if rw >= rh else angle + 90.0
        confidence = _clip01(0.3 + 0.3 * min(1.0, contrast / 60.0) + 0.3 * min(1.0, long_side / short_side / 15.0))
        yield (x, y, x + w, y + h), "crack", confidence, int(cv2.contourArea(contour) or w * h), float(direction % 180.0), float(short_side)


def _detect_peeling(gray: np.ndarray, hsv: np.ndarray):
    """Light surface with high local texture: Laplacian energy averaged over a window."""
    lap = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F)
    local_std = np.sqrt(cv2.blur(lap * lap, (15, 15)))
    light = (hsv[..., 2] > 150) & (hsv[..., 1] < 60)
    mask = _clean(((local_std > PEEL_STD_THRESHOLD) & light).astype(np.uint8) * 255, 9)
    max_area = MAX_AREA_FRACTION * mask.size
    for x, y, w, h, area, region in _components(mask, PEEL_MIN_AREA, max_area):
        strength = float(local_std[region].mean()) / PEEL_STD_THRESHOLD
        yield (x, y, x + w, y + h), "peeling_paint", _clip01(0.25 + 0.25 * min(strength, 2.0)), area, None, None


def _detect_holes(hsv: np.ndarray):
    mask = _clean((hsv[..., 2] < 50).astype(np.uint8) * 255, 3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = MAX_AREA_FRACTION * mask.size
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if not HOLE_MIN_AREA <= area <= max_area or perimeter == 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < HOLE_MIN_CIRCULARITY:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        yield (x, y, x + w, y + h), "hole", _clip01(circularity), int(area), None, None


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _merge_overlaps(detections: list[DamageDetection]) -> list[DamageDetection]:
    """Merge same-class boxes with IoU > NMS_IOU into their union (keeps the strongest confidence)."""
    merged: list[DamageDetection] = []
    for det in sorted(detections, key=lambda d: -d.confidence):
        for kept in merged:
            if kept.damage_class == det.damage_class and _iou(kept.bbox, det.bbox) > NMS_IOU:
                kept.bbox = (min(kept.bbox[0], det.bbox[0]), min(kept.bbox[1], det.bbox[1]),
                             max(kept.bbox[2], det.bbox[2]), max(kept.bbox[3], det.bbox[3]))
                kept.pixel_area += det.pixel_area
                break
        else:
            merged.append(det)
    return merged


def _detect_one(image_path: str) -> list[DamageDetection]:
    image = cv2.imread(image_path)  # applies EXIF orientation, matching the SfM upright copies
    if image is None:
        log.warning("could not read image %s", image_path)
        return []
    height, width = image.shape[:2]
    scale = min(1.0, WORK_SIZE / float(max(height, width)))
    work = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))),
                      interpolation=cv2.INTER_AREA) if scale < 1.0 else image
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

    found = [*_detect_cracks(gray), *_detect_stains(hsv), *_detect_mold(hsv),
             *_detect_peeling(gray, hsv), *_detect_holes(hsv)]
    inv = 1.0 / scale
    detections = []
    for bbox, damage_class, confidence, area, angle, thickness in found:
        detections.append(DamageDetection(
            image_path=image_path,
            bbox=tuple(int(round(v * inv)) for v in bbox),
            damage_class=damage_class, confidence=float(confidence),
            pixel_area=int(round(area * inv * inv)), image_size=(width, height),
            angle_deg=angle, thickness_px=None if thickness is None else float(thickness * inv)))
    # a box covering more than MAX_AREA_FRACTION of the image is a wall or large object (cracks and thin
    # regions can have a small pixel area but a huge box)
    detections = [d for d in detections
                  if (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) <= MAX_AREA_FRACTION * width * height]
    merged = sorted(_merge_overlaps(detections), key=lambda d: (-d.confidence, d.bbox))
    return merged[:MAX_PER_IMAGE]  # the strongest few only


def drop_persistent(per_image: list[list[DamageDetection]], limit: int, iou: float) -> list[list[DamageDetection]]:
    """Drop detections that sit at the same image position in more than `limit` consecutive frames.

    A real stain moves across the frame as the camera moves; a box that stays put over many frames is
    a lens mark, an overlay or a fixed pattern, not damage. per_image is in frame order. Caveat: this works
    on image position, so a persistent feature that moves with the camera motion is not caught by it.
    """
    runs: list[list[tuple[int, int]]] = []  # each run: (image index, detection index) pairs
    active: dict[int, list[tuple[int, int]]] = {}  # detection index in the previous image -> its run
    for i, detections in enumerate(per_image):
        nxt: dict[int, list[tuple[int, int]]] = {}
        for k, d in enumerate(detections):
            best, best_iou = None, iou
            for prev_k in active:  # the best still-unclaimed detection of the previous frame
                prev = per_image[i - 1][prev_k]
                overlap = _iou(prev.bbox, d.bbox)
                if prev.damage_class == d.damage_class and overlap >= best_iou:
                    best, best_iou = prev_k, overlap
            if best is not None:
                run = active.pop(best)
                run.append((i, k))
            else:
                run = [(i, k)]
                runs.append(run)
            nxt[k] = run
        active = nxt
    drop = {pair for run in runs if len(run) > limit for pair in run}
    return [[d for k, d in enumerate(detections) if (i, k) not in drop] for i, detections in enumerate(per_image)]


def detect_damage(images: list[str], persistent_frames: int = 0) -> list[DamageDetection]:
    """Detect visible damage in each image with OpenCV heuristics.

    Canny + contrast for cracks, HSV ranges for water stains and mold,
    Laplacian energy for peeling paint, dark circular regions for holes;
    overlapping same-class boxes are merged by non-maximum suppression.
    At most MAX_PER_IMAGE detections (the strongest) are kept per image, and none that cover more than
    MAX_AREA_FRACTION of it. persistent_frames > 0 (video keyframes, in time order) also drops detections that
    stay at the same position in more than that many consecutive frames.
    Output order is deterministic: image order, then descending confidence.
    """
    per_image = [sorted(_detect_one(path), key=lambda d: (-d.confidence, d.bbox)) for path in images]
    if persistent_frames > 0:
        before = sum(map(len, per_image))
        per_image = drop_persistent(per_image, persistent_frames, cfg.PERSISTENT_IOU)
        removed = before - sum(map(len, per_image))
        if removed:
            log.info("dropped %d detection(s) that stay in place over more than %d frames", removed, persistent_frames)
    return [d for detections in per_image for d in detections]
