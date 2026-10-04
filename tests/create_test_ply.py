"""Generate the synthetic 2-room apartment used by the LiDAR end-to-end test.

Layout (plan view, inner room surfaces, meters; walls are 0.12 m thick so every
shared wall is scanned as two parallel surfaces, as in a real scan):

    Room 1 (4.0 x 3.0)  |door|  Hallway (1.2 x 3.0)  |door|  Room 2 (3.5 x 3.0)

Ceiling 2.5 m everywhere, doors 0.86 m wide x 2.1 m tall centered on each
shared wall. Noise sigma 0.005 m. Writes <out-dir>/apartment.ply and
<out-dir>/ground_truth.yaml.

    python tests/create_test_ply.py [--out-dir test_data] [--density 1500] [--seed 0] [--drift DX DY]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

CEILING = 2.5
WALL_THICKNESS = 0.12
DOOR_WIDTH = 0.86
DOOR_HEIGHT = 2.1
NOISE = 0.005
DEPTH = 3.0  # all spaces are 3.0 m in y


@dataclass
class Space:
    name: str
    room_id: str
    x0: float
    width: float  # along x
    doors: tuple[str, ...]  # which walls have a door: "west" and/or "east"

    @property
    def x1(self) -> float:
        return self.x0 + self.width


def layout() -> list[Space]:
    room1 = Space("Room 1", "room-1", 0.0, 4.0, ("east",))
    hall = Space("Hallway", "room-2", room1.x1 + WALL_THICKNESS, 1.2, ("west", "east"))
    room2 = Space("Room 2", "room-3", hall.x1 + WALL_THICKNESS, 3.5, ("west",))
    return [room1, hall, room2]


def _sample_plane(rng, n, fixed_axis, fixed_value, lo, hi):
    """n uniform points on an axis-aligned rectangle; lo/hi are (min, max) per free axis."""
    pts = np.empty((n, 3))
    free = [a for a in range(3) if a != fixed_axis]
    pts[:, fixed_axis] = fixed_value
    for axis, (a, b) in zip(free, (lo, hi)):
        pts[:, axis] = rng.uniform(a, b, n)
    return pts


def space_points(space: Space, rng: np.random.Generator, density: float) -> np.ndarray:
    """Points on floor, ceiling and four walls of one space, with the doors cut out."""
    x0, x1, y0, y1 = space.x0, space.x1, 0.0, DEPTH
    count = lambda area: int(density * area)
    chunks = [
        _sample_plane(rng, count(space.width * DEPTH), 2, 0.0, (x0, x1), (y0, y1)),  # floor
        _sample_plane(rng, count(space.width * DEPTH), 2, CEILING, (x0, x1), (y0, y1)),  # ceiling
        _sample_plane(rng, count(space.width * CEILING), 1, y0, (x0, x1), (0, CEILING)),  # south
        _sample_plane(rng, count(space.width * CEILING), 1, y1, (x0, x1), (0, CEILING)),  # north
    ]
    for side, x in (("west", x0), ("east", x1)):
        wall = _sample_plane(rng, count(DEPTH * CEILING), 0, x, (y0, y1), (0, CEILING))
        if side in space.doors:
            in_door = (np.abs(wall[:, 1] - DEPTH / 2) < DOOR_WIDTH / 2) & (wall[:, 2] < DOOR_HEIGHT)
            wall = wall[~in_door]
        chunks.append(wall)
    return np.vstack(chunks)


def make_apartment(seed: int = 0, density: float = 1500.0,
                   drift: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """All spaces as one N x 3 array of noisy points.

    drift = (dx, dy) metres of pose drift added per space walked: space k is shifted by
    k * drift while the ground truth stays put, so it models accumulated tracking drift.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for k, space in enumerate(layout()):
        pts = space_points(space, rng, density)
        pts[:, 0] += k * drift[0]
        pts[:, 1] += k * drift[1]
        chunks.append(pts)
    points = np.vstack(chunks)
    return points + rng.normal(0.0, NOISE, points.shape)


def write_ply(path: Path, points: np.ndarray) -> None:
    """Binary little-endian PLY, x y z as float32."""
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(points.astype("<f4").tobytes())


def ground_truth() -> dict:
    """Ground truth in the per-room format used by benchmark/evaluate.py, plus adjacency."""
    spaces = layout()
    rooms = []
    for s in spaces:
        walls = [
            {"id": "wall-south", "length_m": s.width},
            {"id": "wall-east", "length_m": DEPTH},
            {"id": "wall-north", "length_m": s.width},
            {"id": "wall-west", "length_m": DEPTH},
        ]
        openings = [
            {"id": f"door-{i + 1}", "type": "door", "wall": f"wall-{side}", "width_m": DOOR_WIDTH,
             "height_m": DOOR_HEIGHT}
            for i, side in enumerate(s.doors)
        ]
        rooms.append({
            "room_id": s.room_id, "name": s.name,
            "origin_xy": [round(s.x0, 3), 0.0],
            "walls": walls, "ceiling_height_m": CEILING,
            "openings": openings, "floor_area_m2": round(s.width * DEPTH, 3),
        })
    return {
        "capture": "synthetic apartment (create_test_ply.py)",
        "wall_thickness_m": WALL_THICKNESS,
        "rooms": rooms,
        "adjacencies": [
            {"rooms": [spaces[0].room_id, spaces[1].room_id], "door_width_m": DOOR_WIDTH},
            {"rooms": [spaces[1].room_id, spaces[2].room_id], "door_width_m": DOOR_WIDTH},
        ],
        "total_floor_area_m2": round(sum(s.width * DEPTH for s in spaces), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", default="test_data", type=Path)
    parser.add_argument("--density", default=1500.0, type=float, help="points per m2 of surface")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--drift", default=(0.0, 0.0), type=float, nargs=2, metavar=("DX", "DY"),
                        help="pose drift in metres added per space walked (ablation input)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    points = make_apartment(args.seed, args.density, tuple(args.drift))
    write_ply(args.out_dir / "apartment.ply", points)
    with open(args.out_dir / "ground_truth.yaml", "w") as f:
        yaml.safe_dump(ground_truth(), f, sort_keys=False)
    print(f"wrote {len(points)} points to {args.out_dir / 'apartment.ply'}")


if __name__ == "__main__":
    main()
