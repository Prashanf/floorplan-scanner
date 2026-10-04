"""Turn a raw depth capture (rgb.mp4 + odometry.csv + depth) into per-room photo folders.

The photo tier wants 2 to 8 stills per room, one folder per room, in walk order. A phone
that logs depth also records the RGB video, so stills are cut from it: the LiDAR front-end
finds the rooms, each video frame is assigned to the room its camera stands in, and per
room up to N sharp frames with different viewing directions are saved as JPEG.

    python tools/make_photo_sets.py <capture_dir> <out_dir> [--per-room 8]

<capture_dir> is the folder with rgb.mp4, odometry.csv and depth/. <out_dir> receives
room-1/, room-2/, ... (rooms numbered in the order the camera first enters them).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tiers.depth_stream import _read_odometry  # noqa: E402
from src.tiers.lidar import process_lidar  # noqa: E402

MAX_CAMERA_DISTANCE = 0.6  # metres from a room's scanned floor/walls for a camera to belong to it
MIN_ROOM_AREA = 3.0  # m2 of scanned footprint; smaller pieces are hallway fragments, not rooms
SHARPNESS_SAMPLE = 25  # frames examined per candidate bin


def _upright(frame: np.ndarray, up_in_camera: np.ndarray) -> np.ndarray:
    """Rotate a landscape sensor frame by a multiple of 90 degrees so that world-up points up.

    up_in_camera is the world up vector in camera axes (x right, y down, z forward); the phone
    is usually held in portrait, so its frames arrive on their side.
    """
    x, y = up_in_camera[0], up_in_camera[1]
    if abs(y) >= abs(x):
        return frame if y < 0 else cv2.rotate(frame, cv2.ROTATE_180)
    return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE if x > 0 else cv2.ROTATE_90_CLOCKWISE)


def _longest_stay(frames: np.ndarray, max_gap: int = 60) -> np.ndarray:
    """The longest run of frame indices with no gap above max_gap frames (about 1 s)."""
    if len(frames) == 0:
        return frames
    breaks = np.flatnonzero(np.diff(frames) > max_gap) + 1
    runs = np.split(frames, breaks)
    return max(runs, key=len)


def _sharpness(frame: np.ndarray) -> float:
    small = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def make_photo_sets(capture_dir: Path, out_dir: Path, per_room: int = 8) -> dict[str, int]:
    """Write room-N folders under out_dir; returns {folder name: number of photos}."""
    from scipy.spatial.transform import Rotation

    property_ir = process_lidar(str(capture_dir))
    odo = _read_odometry(capture_dir / "odometry.csv")
    positions = np.column_stack([odo["x"], odo["y"], odo["z"]])
    rotations = Rotation.from_quat(np.column_stack([odo["qx"], odo["qy"], odo["qz"], odo["qw"]]))

    rooms = []
    for room in property_ir.rooms:
        xy = room.point_cloud.points[:, :2]
        if len(np.unique(np.floor(xy / 0.1).astype(int), axis=0)) * 0.01 < MIN_ROOM_AREA:
            continue
        rooms.append((room.room_id, cKDTree(xy[:: max(1, len(xy) // 20000)]),
                      np.radians(room.metadata.get("plan_rotation_deg", 0.0))))
    if not rooms:
        raise SystemExit("no rooms found in the scan")

    def to_plan(vec: np.ndarray, angle: float) -> np.ndarray:  # ARKit world -> aligned plan frame
        x, y = vec[:, 0], -vec[:, 2]
        c, s = np.cos(angle), np.sin(angle)
        return np.column_stack([c * x - s * y, s * x + c * y])

    angle = rooms[0][2]
    cam_xy = to_plan(positions, angle)
    room_of = np.full(len(cam_xy), -1)
    best = np.full(len(cam_xy), np.inf)
    for k, (_, tree, _) in enumerate(rooms):
        dist, _ = tree.query(cam_xy)
        better = (dist < MAX_CAMERA_DISTANCE) & (dist < best)
        room_of[better], best[better] = k, dist[better]

    # Rooms in walk order = order of first frame
    first = {k: int(np.argmax(room_of == k)) for k in range(len(rooms)) if (room_of == k).any()}
    order = sorted(first, key=first.get)

    # Plan: per room, the candidate frames of each slot along its longest stay.
    plan: list[list[np.ndarray]] = []  # per room, per slot: candidate frame indices
    for k in order:
        frames = _longest_stay(np.flatnonzero(room_of == k))
        if len(frames) < 2:
            plan.append([])
            continue
        # Evenly spaced along one continuous stay, so neighbouring photos overlap the way a person's
        # photos of one room do; within each slot keep the sharpest frame.
        edges = np.linspace(0, len(frames), per_room + 1).astype(int)
        slots = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            window = frames[lo:hi]
            if len(window):
                slots.append(window[np.linspace(0, len(window) - 1, min(SHARPNESS_SAMPLE, len(window))).astype(int)])
        plan.append(slots)

    # Pass 1: one sequential read of the video scores every candidate (seeking per frame is far slower).
    wanted = {int(i) for slots in plan for slot in slots for i in slot}
    score: dict[int, float] = {}
    cap = cv2.VideoCapture(str(capture_dir / "rgb.mp4"))
    last = max(wanted, default=-1)
    for index in range(last + 1):
        if index not in wanted:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        score[index] = _sharpness(frame)

    # Pass 2: write the sharpest candidate of each slot.
    written: dict[str, int] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for slots in plan:
        chosen = sorted(int(max((i for i in slot if int(i) in score), key=lambda i: score[int(i)]))
                        for slot in slots if any(int(i) in score for i in slot))
        if len(chosen) < 2:
            continue
        folder = out_dir / f"room-{len(written) + 1}"
        folder.mkdir(parents=True, exist_ok=True)
        for i, index in enumerate(chosen, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = _upright(frame, rotations[index].inv().apply(np.array([0.0, 1.0, 0.0])))
            cv2.imwrite(str(folder / f"IMG_{i:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written[folder.name] = len(chosen)
    cap.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--per-room", type=int, default=8)
    args = parser.parse_args()
    for name, count in make_photo_sets(args.capture_dir, args.out_dir, args.per_room).items():
        print(f"{name}: {count} photos")


if __name__ == "__main__":
    main()
