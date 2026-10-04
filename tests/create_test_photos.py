"""Render synthetic photos and a walkthrough video of the test apartment for the SfM tiers.

Each space of the apartment in create_test_ply.py becomes a closed box with random
feature-rich textures on every surface and flat-colored 0.86 m x 2.1 m doors (closed
doors, so each room is photographed on its own). Cameras are pinhole, 1280 x 960,
about 1.4 m high with small random pitch.

    python tests/create_test_photos.py [--photos-dir test_photos] [--video-dir test_video]
                                       [--photos-per-room 8] [--seed 0]

Writes test_photos/room-1 .. room-3 (one folder per space) and test_video/walkthrough.mp4
(Room 1 only, 8 s at 30 fps).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from create_test_ply import CEILING, DEPTH, DOOR_HEIGHT, DOOR_WIDTH, Space, layout

PPM = 300  # texture pixels per meter
IMAGE_SIZE = (1280, 960)  # width, height
FOCAL = 900.0
EYE_HEIGHT = 1.4
FOCAL_35MM = FOCAL / IMAGE_SIZE[0] * 36.0  # written to EXIF like a real phone photo


@dataclass
class Plane:
    axis: int  # axis the plane is perpendicular to
    value: float
    bounds: tuple[float, float, float, float]  # (b0, b1, c0, c1) along the two other axes
    texture: np.ndarray


def _texture(rng: np.random.Generator, width_m: float, height_m: float, tint: np.ndarray) -> np.ndarray:
    """Multi-scale noise plus colored shapes and speckle: plenty of SIFT features."""
    h, w = int(height_m * PPM), int(width_m * PPM)
    img = np.zeros((h, w, 3), dtype=np.float32)
    for cell, weight in ((6, 0.5), (14, 0.5), (30, 0.4), (70, 0.4), (160, 0.3)):
        noise = rng.random((h // cell + 2, w // cell + 2, 3)).astype(np.float32)
        img += weight * cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    img = 255 * (img - img.min()) / (np.ptp(img) + 1e-6)
    img = 0.55 * img + 0.45 * tint
    for _ in range(int(14 * width_m * height_m)):  # posters, stains, tiles
        color = tuple(float(c) for c in rng.integers(20, 235, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        size = int(rng.integers(int(0.05 * PPM), int(0.4 * PPM)))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + size, y + int(size * rng.uniform(0.4, 1.6))), color, -1)
        else:
            cv2.circle(img, (x, y), size // 2, color, -1)
    img += rng.normal(0, 5, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _paint_door(tex: np.ndarray, center: float, b0: float, c1: float) -> None:
    """Flat brown door panel (no texture, so SfM puts no points on it)."""
    x0 = int((center - DOOR_WIDTH / 2 - b0) * PPM)
    x1 = int((center + DOOR_WIDTH / 2 - b0) * PPM)
    y0 = int((c1 - DOOR_HEIGHT) * PPM)
    cv2.rectangle(tex, (x0 - 6, y0 - 6), (x1 + 6, tex.shape[0] - 1), (35, 45, 70), -1)  # frame (BGR)
    cv2.rectangle(tex, (x0, y0), (x1, tex.shape[0] - 1), (60, 95, 135), -1)


def build_room(space: Space, rng: np.random.Generator) -> list[Plane]:
    """Six textured planes for one space, in global apartment coordinates."""
    x0, x1, y0, y1 = space.x0, space.x1, 0.0, DEPTH
    tints = [np.array(t, dtype=np.float32) for t in ((150, 140, 130), (200, 200, 205), (190, 180, 170),
                                                     (185, 190, 180), (175, 170, 190), (190, 185, 175))]
    planes = [
        Plane(2, 0.0, (x0, x1, y0, y1), _texture(rng, x1 - x0, y1 - y0, tints[0])),  # floor (b=x, c=y)
        Plane(2, CEILING, (x0, x1, y0, y1), _texture(rng, x1 - x0, y1 - y0, tints[1])),
        Plane(1, y0, (x0, x1, 0.0, CEILING), _texture(rng, x1 - x0, CEILING, tints[2])),  # south (b=x, c=z)
        Plane(1, y1, (x0, x1, 0.0, CEILING), _texture(rng, x1 - x0, CEILING, tints[3])),  # north
        Plane(0, x0, (y0, y1, 0.0, CEILING), _texture(rng, y1 - y0, CEILING, tints[4])),  # west (b=y, c=z)
        Plane(0, x1, (y0, y1, 0.0, CEILING), _texture(rng, y1 - y0, CEILING, tints[5])),  # east
    ]
    for side, plane in (("west", planes[4]), ("east", planes[5])):
        if side in space.doors:
            _paint_door(plane.texture, DEPTH / 2, plane.bounds[0], CEILING)
    return planes


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world rotation; columns are the camera's x (right), y (down), z (forward) axes."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.column_stack([right, down, forward])


def render(planes: list[Plane], eye: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Ray-cast the box from `eye`; returns a BGR image."""
    width, height = IMAGE_SIZE
    xs, ys = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    rays_cam = np.stack([(xs - width / 2) / FOCAL, (ys - height / 2) / FOCAL, np.ones_like(xs)], axis=-1)
    rays = rays_cam @ rotation.T
    nearest = np.full((height, width), np.inf)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for plane in planes:
        b, c = [a for a in range(3) if a != plane.axis]
        b0, b1, c0, c1 = plane.bounds
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (plane.value - eye[plane.axis]) / rays[..., plane.axis]
        hit_b, hit_c = eye[b] + t * rays[..., b], eye[c] + t * rays[..., c]
        inside = (t > 1e-6) & (hit_b >= b0) & (hit_b <= b1) & (hit_c >= c0) & (hit_c <= c1) & (t < nearest)
        map_x = np.nan_to_num((hit_b - b0) * PPM).astype(np.float32)
        map_y = np.nan_to_num((c1 - hit_c) * PPM).astype(np.float32)
        sample = cv2.remap(plane.texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        image[inside] = sample[inside]
        nearest[inside] = t[inside]
    return image


def _save_jpeg(path: Path, bgr: np.ndarray) -> None:
    """JPEG with a 35 mm focal length in EXIF, as a phone would write."""
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA405] = round(FOCAL_35MM)
    img.save(path, "JPEG", quality=93, exif=exif)


def photo_poses(space: Space, count: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """(eye, target) pairs going once around the room as the protocol asks: corners and wall
    midpoints in order, each aimed across the room so neighboring photos overlap."""
    if space.width < 2.0:  # hallway: stand at each end and look down its length
        xc = (space.x0 + space.x1) / 2
        poses = []
        for k in range(count):
            end_y, far_y = (0.4, DEPTH) if k % 2 == 0 else (DEPTH - 0.4, 0.0)
            eye = np.array([xc + rng.uniform(-0.2, 0.2), end_y, EYE_HEIGHT]) + rng.normal(0, 0.02, 3) * [1, 1, 0.5]
            target = np.array([xc + rng.uniform(-0.3, 0.3), far_y, 1.3])
            poses.append((eye, target))
        return poses
    inset = min(0.45, (space.width - 0.1) / 2)
    x0, x1, y0, y1 = space.x0 + inset, space.x1 - inset, inset, DEPTH - inset
    xm, ym = (x0 + x1) / 2, DEPTH / 2
    ring = [(x0, y0), (xm, y0), (x1, y0), (x1, ym), (x1, y1), (xm, y1), (x0, y1), (x0, ym)]
    step = len(ring) / count
    chosen = [ring[int(i * step)] for i in range(count)]
    cx, cy = (space.x0 + space.x1) / 2, DEPTH / 2
    poses = []
    for x, y in chosen:
        eye = np.array([x, y, EYE_HEIGHT]) + rng.normal(0, 0.03, 3) * [1, 1, 0.5]
        target = np.array([cx, cy, 1.2]) + rng.normal(0, 0.15, 3)  # a little hand-held aim jitter
        poses.append((eye, target))
    return poses


def make_photos(out_dir: Path, per_room: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for space in layout():
        planes = build_room(space, rng)
        room_dir = out_dir / space.room_id
        room_dir.mkdir(parents=True, exist_ok=True)
        for i, (eye, target) in enumerate(photo_poses(space, per_room, rng), start=1):
            _save_jpeg(room_dir / f"IMG_{i:04d}.jpg", render(planes, eye, look_at(eye, target)))
        print(f"{room_dir}: {per_room} photos")


def make_video(out_dir: Path, seed: int, seconds: float = 8.0, fps: int = 30, size=(960, 720)) -> Path:
    """Walk an ellipse inside Room 1, looking at the wall across the room."""
    global IMAGE_SIZE, FOCAL
    saved = (IMAGE_SIZE, FOCAL)
    IMAGE_SIZE, FOCAL = size, FOCAL * size[0] / saved[0][0]
    try:
        rng = np.random.default_rng(seed)
        space = layout()[0]
        planes = build_room(space, rng)
        cx, cy = (space.x0 + space.x1) / 2, DEPTH / 2
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "walkthrough.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        n = int(seconds * fps)
        for k in range(n):
            angle = 2 * np.pi * k / n
            eye = np.array([cx + 1.3 * np.cos(angle), cy + 0.75 * np.sin(angle), EYE_HEIGHT + 0.02 * np.sin(8 * angle)])
            target = np.array([cx - 1.5 * np.cos(angle), cy - 1.2 * np.sin(angle), 1.3])
            writer.write(render(planes, eye, look_at(eye, target)))
        writer.release()
        print(f"{path}: {n} frames")
        return path
    finally:
        IMAGE_SIZE, FOCAL = saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--photos-dir", default="test_photos", type=Path)
    parser.add_argument("--video-dir", default="test_video", type=Path)
    parser.add_argument("--photos-per-room", default=8, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    make_photos(args.photos_dir, args.photos_per_room, args.seed)
    make_video(args.video_dir, args.seed)


if __name__ == "__main__":
    main()
