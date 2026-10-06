"""Split a whole-property point cloud into one point cloud per room (LiDAR and video tiers).

Two stages on the wall-height slice (0.8-1.5 m above the floor, projected to 2D):

1. DBSCAN (eps 0.5 m, min_samples 50) separates disconnected scan parts and drops
   stray points.
2. DBSCAN cannot split rooms that share walls (the points chain together), so each
   cluster is split by free space. Free cells get their distance to the nearest wall;
   a room is a peak of that distance map, and two peaks stay separate rooms when the
   pass between them (a doorway) is lower than the lower peak by at least PERSISTENCE.
   A narrow hallway is a low ridge, but still higher than the doorways around it.

Each point of the full-height cloud is then assigned to rooms. Floor and ceiling points
take the room whose region they lie in. A wall point probes PROBE meters to either side
of itself along its normal and joins every room it finds there: the two faces of a thick
wall each see only their own room, while a wall scanned once sees both neighbors.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import DBSCAN

from src.geometry.ceiling import find_floor_and_ceiling
from src.room_ir import PointCloud
from src import config as cfg

WALL_BAND = cfg.WALL_HEIGHT_BAND
DBSCAN_EPS = cfg.DBSCAN_EPS
DBSCAN_MIN_SAMPLES = cfg.DBSCAN_MIN_SAMPLES
CELL = cfg.SEGMENTATION_CELL
PERSISTENCE = cfg.ROOM_PERSISTENCE
MIN_ROOM_RADIUS = cfg.MIN_ROOM_RADIUS
MIN_FREE_AREA = 0.5  # m2 of enclosed free space below which the walls are treated as leaking
MAX_REACH = 0.4  # points farther than this from every room region are dropped
PROBE = 0.1  # meters; wall points look this far to each side for a room
NORMAL_NEIGHBORS = 30
MIN_ROOM_POINTS = cfg.MIN_ROOM_POINTS


def _subset(cloud: PointCloud, mask: np.ndarray) -> PointCloud:
    return PointCloud(
        points=cloud.points[mask],
        colors=None if cloud.colors is None else cloud.colors[mask],
        normals=None if cloud.normals is None else cloud.normals[mask],
    )


def _free_space(wall_cells: np.ndarray) -> np.ndarray:
    """Free cells enclosed by walls (True inside a room); empty if the walls leak."""
    closed = ndimage.binary_dilation(wall_cells, iterations=1)  # bridge scan holes up to ~5 cm
    padded = np.pad(closed, 1, constant_values=False)
    regions = ndimage.label(~padded)[0]
    outside = regions == regions[0, 0]
    return (~padded & ~outside)[1:-1, 1:-1]


def _hull_free_space(xy: np.ndarray, origin: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Fallback when walls leak: everything inside the convex hull of the points."""
    hull = xy[ConvexHull(xy).vertices]
    gx, gy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    centers = np.column_stack([gx.ravel(), gy.ravel()]) * CELL + origin + CELL / 2
    return MplPath(hull).contains_points(centers).reshape(shape)


def _watershed(distance: np.ndarray, free: np.ndarray, persistence: float = PERSISTENCE,
               min_room_radius: float = MIN_ROOM_RADIUS) -> np.ndarray:
    """Label free cells by room: flood the distance map from its peaks downward.

    A new label starts at every cell with no labeled neighbor. Where fronts of two labels
    meet at height d, the lower peak is absorbed if peak - d < PERSISTENCE.
    """
    labels = np.zeros(distance.shape, dtype=int)
    parent, peak = [0], [0.0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    rows, cols = np.nonzero(free)
    order = np.argsort(-distance[rows, cols], kind="stable")
    n_rows, n_cols = distance.shape
    for k in order:
        i, j = int(rows[k]), int(cols[k])
        d = float(distance[i, j])
        roots = {find(int(labels[a, b]))
                 for a in range(max(i - 1, 0), min(i + 2, n_rows))
                 for b in range(max(j - 1, 0), min(j + 2, n_cols)) if labels[a, b]}
        if not roots:
            parent.append(len(parent))
            peak.append(d)
            labels[i, j] = len(parent) - 1
            continue
        main, *others = sorted(roots, key=lambda r: (-peak[r], r))
        for other in others:
            if peak[other] - d < persistence:
                parent[other] = main
        # Join the label of the highest neighbor (steepest ascent), so low cells along a wall
        # stay with the room they border instead of the room with the biggest peak.
        window = [(a, b) for a in range(max(i - 1, 0), min(i + 2, n_rows))
                  for b in range(max(j - 1, 0), min(j + 2, n_cols)) if labels[a, b]]
        best = max(window, key=lambda ab: distance[ab])
        labels[i, j] = find(int(labels[best]))

    out = np.zeros_like(labels)
    keep = {}
    for r in {find(int(v)) for v in np.unique(labels[labels > 0])}:
        if peak[r] >= min_room_radius:
            keep[r] = len(keep) + 1
    for v in np.unique(labels[labels > 0]):
        out[labels == v] = keep.get(find(int(v)), 0)
    return out


def _split_by_free_space(xy: np.ndarray, persistence: float = PERSISTENCE,
                         min_room_radius: float = MIN_ROOM_RADIUS) -> tuple[np.ndarray, np.ndarray, int]:
    """Room label per grid cell. Returns (labels, grid origin, n_rooms); label 0 is no room."""
    origin = xy.min(axis=0) - 2 * CELL
    shape = tuple(int(n) for n in np.ceil((xy.max(axis=0) - origin) / CELL) + 3)
    idx = np.floor((xy - origin) / CELL).astype(int)
    wall_cells = np.zeros(shape, dtype=bool)
    wall_cells[idx[:, 0], idx[:, 1]] = True

    free = _free_space(wall_cells)
    if free.sum() * CELL * CELL < MIN_FREE_AREA:  # walls leak (scan holes): fall back to the hull
        free = _hull_free_space(xy, origin, shape) & ~ndimage.binary_dilation(wall_cells)

    distance = ndimage.distance_transform_edt(free) * CELL
    labels = _watershed(distance, free, persistence, min_room_radius)
    return labels, origin, int(labels.max())


def _estimate_normals(points: np.ndarray) -> np.ndarray:
    """Unoriented normals from the smallest principal axis of each point's neighborhood."""
    _, neighbors = cKDTree(points).query(points, k=min(NORMAL_NEIGHBORS, len(points)))
    local = points[neighbors] - points[neighbors].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", local, local)
    return np.linalg.eigh(cov)[1][:, :, 0]


def _assign_points(
    labels: np.ndarray, origin: np.ndarray, n_rooms: int, points: np.ndarray, normals: np.ndarray
) -> list[np.ndarray]:
    """Boolean mask over points for each room (see module docstring for the rule)."""
    shape = np.array(labels.shape)

    def label_at(xy: np.ndarray) -> np.ndarray:
        cell = np.floor((xy - origin) / CELL).astype(int)
        inside = np.all((cell >= 0) & (cell < shape), axis=1)
        out = np.zeros(len(xy), dtype=int)
        out[inside] = labels[cell[inside, 0], cell[inside, 1]]
        return out

    xy = points[:, :2]
    vertical = np.abs(normals[:, 2]) < 0.5
    direction = normals[:, :2] / np.maximum(np.linalg.norm(normals[:, :2], axis=1, keepdims=True), 1e-9)
    side_a = label_at(xy + PROBE * direction)
    side_b = label_at(xy - PROBE * direction)
    here = label_at(xy)

    # Nearest room region, for points that find nothing (floor next to a wall, scan noise).
    near_dist, near_idx = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    cell = np.clip(np.floor((xy - origin) / CELL).astype(int), 0, shape - 1)
    nearest = labels[near_idx[0][cell[:, 0], cell[:, 1]], near_idx[1][cell[:, 0], cell[:, 1]]]
    nearest = np.where(near_dist[cell[:, 0], cell[:, 1]] * CELL <= MAX_REACH, nearest, 0)

    masks = []
    for room in range(1, n_rooms + 1):
        wall_vote = (side_a == room) | (side_b == room)
        found = (side_a > 0) | (side_b > 0)
        masks.append(np.where(vertical,
                              wall_vote | (~found & (nearest == room)),
                              (here == room) | ((here == 0) & (nearest == room))))
    return masks


def _cluster_band(xy: np.ndarray) -> np.ndarray:
    """DBSCAN labels (-1 = noise) of the wall-band points.

    Points are first binned to CELL-sized cells and DBSCAN runs on the cell centres weighted
    by their point counts. That gives the same clusters as running on every point, but
    dense real scans (hundreds of thousands of band points) stay fast instead of taking minutes.
    """
    cells, inverse, counts = np.unique(np.floor(xy / CELL).astype(np.int64), axis=0,
                                       return_inverse=True, return_counts=True)
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(
        (cells + 0.5) * CELL, sample_weight=counts)
    return labels[inverse.reshape(-1)]


def segment_rooms(cloud: PointCloud, persistence: float = PERSISTENCE,
                  min_room_radius: float = MIN_ROOM_RADIUS) -> list[PointCloud]:
    """Return one full-height PointCloud per room, ordered left to right (then bottom to top).

    A cloud with no usable wall structure is returned as a single room.
    """
    pts = cloud.points
    levels = find_floor_and_ceiling(pts[:, 2])
    band = (pts[:, 2] >= levels.floor_z + WALL_BAND[0]) & (pts[:, 2] <= levels.floor_z + WALL_BAND[1])
    if band.sum() < DBSCAN_MIN_SAMPLES:
        return [cloud]

    band_idx = np.flatnonzero(band)
    cluster_of_band = _cluster_band(pts[band_idx, :2])
    cluster_ids = [c for c in np.unique(cluster_of_band) if c != -1]
    if not cluster_ids:
        return [cloud]

    rooms: list[PointCloud] = []
    for cid in cluster_ids:
        xy_wall = pts[band_idx[cluster_of_band == cid], :2]
        lo, hi = xy_wall.min(axis=0) - MAX_REACH, xy_wall.max(axis=0) + MAX_REACH
        in_cluster = np.all((pts[:, :2] >= lo) & (pts[:, :2] <= hi), axis=1)
        member = np.flatnonzero(in_cluster)

        labels, origin, n_rooms = _split_by_free_space(xy_wall, persistence, min_room_radius)
        if n_rooms <= 1:
            rooms.append(_subset(cloud, in_cluster))
            continue
        normals = cloud.normals[member] if cloud.normals is not None else _estimate_normals(pts[member])
        for mask in _assign_points(labels, origin, n_rooms, pts[member], normals):
            full = np.zeros(len(pts), dtype=bool)
            full[member[mask]] = True
            rooms.append(_subset(cloud, full))

    rooms = [r for r in rooms if len(r) >= MIN_ROOM_POINTS] or [cloud]
    rooms.sort(key=lambda r: (round(float(np.median(r.points[:, 0])), 1), float(np.median(r.points[:, 1]))))
    return rooms
