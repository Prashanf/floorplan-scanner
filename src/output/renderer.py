"""Render the stitched floor plan to PNG."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models import Opening, PropertyReport, Room, Wall

WALL_LINEWIDTH = 2
LABEL_OFFSET = 0.2  # meters, label distance from the wall toward the room interior
SCALE_BAR_LENGTH = 1.0


def _centroid(polygon: list[tuple[float, float]]) -> tuple[float, float]:
    """Area centroid of a simple polygon (falls back to the vertex mean when degenerate)."""
    pts = np.array(polygon)
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2
    if abs(area) < 1e-9:
        return float(x.mean()), float(y.mean())
    return float(((x + x1) * cross).sum() / (6 * area)), float(((y + y1) * cross).sum() / (6 * area))


def _wall_pieces(wall: Wall, openings: list[Opening]) -> list[tuple[float, float]]:
    """(start, end) distances along the wall that remain after cutting out its openings."""
    length = float(np.hypot(wall.end[0] - wall.start[0], wall.end[1] - wall.start[1]))
    cuts = sorted((max(0.0, o.position_along_wall.value),
                   min(length, o.position_along_wall.value + o.width.value))
                  for o in openings if o.wall_id == wall.id)
    pieces, cursor = [], 0.0
    for lo, hi in cuts:
        if lo > cursor:
            pieces.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < length:
        pieces.append((cursor, length))
    return pieces


def _draw_room(ax, room: Room) -> None:
    from matplotlib.patches import Polygon as PolygonPatch

    if len(room.floor_polygon) >= 3:
        ax.add_patch(PolygonPatch(room.floor_polygon, closed=True, facecolor="#e6e6e6",
                                  edgecolor="none", zorder=1))
    for wall in room.walls:
        start, end = np.array(wall.start), np.array(wall.end)
        length = float(np.linalg.norm(end - start))
        if length < 1e-6:
            continue
        direction = (end - start) / length
        for lo, hi in _wall_pieces(wall, room.openings):
            a, b = start + direction * lo, start + direction * hi
            ax.plot([a[0], b[0]], [a[1], b[1]], color="black", linewidth=WALL_LINEWIDTH,
                    solid_capstyle="butt", zorder=3)
        # Length label, centered on the wall, nudged inside, rotated to the wall, kept upright.
        inward = np.array([-direction[1], direction[0]])  # left of a counterclockwise wall
        pos = (start + end) / 2 + inward * LABEL_OFFSET
        angle = np.degrees(np.arctan2(direction[1], direction[0]))
        if angle > 90 or angle <= -90:
            angle += 180
        ax.text(pos[0], pos[1], f"{wall.length.value:.2f} m", ha="center", va="center",
                rotation=angle, fontsize=7, color="#222222", zorder=4)
    cx, cy = _centroid(room.floor_polygon) if room.floor_polygon else (0.0, 0.0)
    ax.text(cx, cy, f"{room.name}\n{room.floor_area.value:.1f} m²", ha="center", va="center",
            fontsize=9, fontweight="bold", zorder=5)


def render_floor_plan(report: PropertyReport, output_dir: str) -> str:
    """Draw rooms, walls with lengths, openings, labels and a scale bar with
    matplotlib; save output_dir/floor_plan.png and return its path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    ax.set_facecolor("white")
    for room in report.rooms:
        _draw_room(ax, room)

    points = [p for room in report.rooms for p in room.floor_polygon]
    if points:
        xs, ys = zip(*points)
        margin = 0.8
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
        # Scale bar in the lower-left corner of the plot area.
        x0, y0 = min(xs) - margin + 0.2, min(ys) - margin + 0.25
        ax.plot([x0, x0 + SCALE_BAR_LENGTH], [y0, y0], color="black", linewidth=3, zorder=6)
        ax.text(x0 + SCALE_BAR_LENGTH / 2, y0 + 0.12, f"{SCALE_BAR_LENGTH:g} m", ha="center",
                va="bottom", fontsize=8, zorder=6)
    else:
        ax.text(0.5, 0.5, "No rooms detected", transform=ax.transAxes, ha="center", va="center")
    ax.set_aspect("equal")
    ax.set_title(f"Floor Plan — {report.capture_tier} tier")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, color="#f0f0f0", linewidth=0.5, zorder=0)
    fig.tight_layout()

    path = Path(output_dir) / "floor_plan.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return str(path)
