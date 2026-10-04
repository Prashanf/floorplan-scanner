"""Render the stitched floor plan to PNG."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models import Opening, PropertyReport, Room, Wall

WALL_LINEWIDTH = 2
LABEL_OFFSET = 0.2  # meters, label distance from the wall toward the room interior
SCALE_BAR_LENGTH = 1.0
ROOM_COLORS = ["#dbe9f6", "#fde6c8", "#dff0d8", "#f3dcec", "#fff3b8", "#d9ece9", "#e8e0f5", "#f6dcd2"]
DAMAGE_COLORS = {"crack": "#c62828", "water_stain": "#d84315", "mold": "#8e0000",
                 "hole": "#ad1457", "peeling_paint": "#e53935"}
DAMAGE_LABELS = {"crack": "Crack", "water_stain": "Water stain", "mold": "Mold", "hole": "Hole",
                 "peeling_paint": "Peeling paint"}


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


def _draw_openings(ax, wall: Wall, room: Room, start: np.ndarray, direction: np.ndarray,
                   length: float) -> None:
    """Doors stay open gaps with a thin jamb tick; windows are bridged by a dashed blue line."""
    for o in room.openings:
        if o.wall_id != wall.id:
            continue
        lo = max(0.0, o.position_along_wall.value)
        hi = min(length, o.position_along_wall.value + o.width.value)
        a, b = start + direction * lo, start + direction * hi
        if o.type == "window":
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#1565c0", linewidth=WALL_LINEWIDTH,
                    linestyle=(0, (3, 2)), zorder=3)
        else:
            normal = np.array([-direction[1], direction[0]]) * 0.06
            for end in (a, b):  # jambs mark the door width
                ax.plot([end[0] - normal[0], end[0] + normal[0]], [end[1] - normal[1], end[1] + normal[1]],
                        color="black", linewidth=1, zorder=3)


def _draw_room(ax, room: Room, color: str) -> None:
    from matplotlib.patches import Polygon as PolygonPatch

    if len(room.floor_polygon) >= 3:
        ax.add_patch(PolygonPatch(room.floor_polygon, closed=True, facecolor=color,
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
        _draw_openings(ax, wall, room, start, direction, length)
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


def _damage_xy(report: PropertyReport) -> list[tuple[str, float, float]]:
    """(damage_class, x, y) of each damage region in plan view: its wall point at the u offset."""
    walls = {w.id: w for room in report.rooms for w in room.walls}
    placed = []
    for d in report.damage_regions:
        wall = walls.get(d.surface_id)
        if wall is None:
            continue
        start, end = np.array(wall.start), np.array(wall.end)
        length = float(np.linalg.norm(end - start))
        if length < 1e-6:
            continue
        u = min(max(d.location_on_surface[0], 0.0), length)
        x, y = start + (end - start) / length * u
        placed.append((d.damage_class, float(x), float(y)))
    return placed


def _scale_bar_length(span: float) -> float:
    """A round length (1, 2 or 5 x 10^k m) near a fifth of the plan width."""
    target = max(span / 5, 0.5)
    mag = 10 ** np.floor(np.log10(target))
    return float(min((m * mag for m in (1, 2, 5, 10)), key=lambda v: abs(v - target)))


def render_floor_plan(report: PropertyReport, output_dir: str) -> str:
    """Draw rooms (one colour each), walls with lengths, doors and windows, damage markers,
    labels, legend, north arrow and scale bar with matplotlib; save output_dir/floor_plan.png
    and return its path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    ax.set_facecolor("white")
    for i, room in enumerate(report.rooms):
        _draw_room(ax, room, ROOM_COLORS[i % len(ROOM_COLORS)])

    from matplotlib.lines import Line2D

    damage = _damage_xy(report)
    for damage_class, x, y in damage:
        ax.plot(x, y, marker="X", markersize=11, color=DAMAGE_COLORS.get(damage_class, "#c62828"),
                markeredgecolor="white", markeredgewidth=0.8, linestyle="none", zorder=7)

    points = [p for room in report.rooms for p in room.floor_polygon]
    if points:
        xs, ys = zip(*points)
        margin = 0.8
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
        # Scale bar in the lower-left corner of the plot area.
        bar = _scale_bar_length(max(xs) - min(xs))
        x0, y0 = min(xs) - margin + 0.2, min(ys) - margin + 0.25
        ax.plot([x0, x0 + bar], [y0, y0], color="black", linewidth=3, zorder=6)
        ax.text(x0 + bar / 2, y0 + 0.12, f"{bar:g} m", ha="center", va="bottom", fontsize=8, zorder=6)
        # North arrow in the upper-right corner (plan +y is north).
        nx, ny = max(xs) + margin - 0.35, max(ys) + margin - 0.9
        ax.annotate("", xy=(nx, ny + 0.6), xytext=(nx, ny),
                    arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 1.5}, zorder=6)
        ax.text(nx, ny + 0.68, "N", ha="center", va="bottom", fontsize=10, fontweight="bold", zorder=6)
    else:
        ax.text(0.5, 0.5, "No rooms detected", transform=ax.transAxes, ha="center", va="center")
    handles = [Line2D([0], [0], color="black", linewidth=WALL_LINEWIDTH, label="Wall"),
               Line2D([0], [0], color="#1565c0", linewidth=WALL_LINEWIDTH, linestyle=(0, (3, 2)),
                      label="Window"),
               Line2D([0], [0], color="none", label="Door (open gap)")]
    for damage_class in dict.fromkeys(c for c, _, _ in damage):
        handles.append(Line2D([0], [0], marker="X", linestyle="none", markersize=9,
                              color=DAMAGE_COLORS.get(damage_class, "#c62828"),
                              label=DAMAGE_LABELS.get(damage_class, damage_class)))
    if not damage:
        handles.append(Line2D([0], [0], color="none", label="No damage detected"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7,
              framealpha=0.9, title="Legend", title_fontsize=8)
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
