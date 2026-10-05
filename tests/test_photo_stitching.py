"""Photo-tier stitching: doorway photos, best door match, adjacency graph with BFS placement, shared-camera transform."""

import math
from datetime import datetime, timedelta

import numpy as np
import pytest
from shapely.geometry import Polygon

from src.geometry.openings import OpeningDetection
from src.geometry.wall_fitting import WallSegment
from src.room_ir import CameraPose, PointCloud, RoomIR
from src.stitching import photo_stitch as ps
from src.tiers import photo


def door(width=0.9, pos=0.5, wall=0, kind="door"):
    return OpeningDetection(wall_index=wall, type=kind, position_along_wall=pos, width=width, height=2.0, confidence=1.0)


def room(room_id, w, h, doors=(), meta=None):
    """Axis-aligned rectangle room, counter-clockwise; doors are (wall index, position, width)."""
    pts = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    walls = [WallSegment(start=pts[i], end=pts[(i + 1) % 4], length=math.dist(pts[i], pts[(i + 1) % 4]),
                         direction=math.atan2(pts[(i + 1) % 4][1] - pts[i][1], pts[(i + 1) % 4][0] - pts[i][0]),
                         inlier_count=100) for i in range(4)]
    r = RoomIR(room_id=room_id, point_cloud=PointCloud(points=np.zeros((1, 3))), tier="photo", wall_segments=walls,
               floor_polygon=pts, openings=[door(width, pos, wall) for wall, pos, width in doors])
    r.metadata.update(meta or {})
    return r


def poly(r, tf):
    return ps._room_polygon(r, tf)


# ---------------------------------------------------------------- doorway photo choice

def test_doorway_photo_is_the_last_image_without_exif(monkeypatch):
    monkeypatch.setattr(photo, "_exif_time", lambda path: None)
    assert photo.pick_doorway_photo(["a1", "a2", "a6"], ["b1", "b2"]) == "a6"
    assert photo.pick_doorway_photo([], ["b1"]) is None
    assert photo.pick_doorway_photo(["a1", "a2"], None) == "a2"  # the last room: no next room


def test_doorway_photo_is_the_one_taken_near_the_next_rooms_first_photo(monkeypatch):
    t0 = datetime(2026, 10, 5, 12, 0, 0)
    times = {"a1": t0, "a5": t0 + timedelta(seconds=100), "a6": t0 + timedelta(seconds=104),
             "b1": t0 + timedelta(seconds=120)}                      # a5 is 20 s from b1, a6 is 16 s
    times["a_last"] = t0 + timedelta(seconds=150)                     # last in the folder, 30 s after b1: edge of the window
    monkeypatch.setattr(photo, "_exif_time", lambda path: times.get(path))
    assert photo.pick_doorway_photo(["a1", "a5", "a6", "a_last"], ["b1"]) == "a6"      # closest within 30 s
    far = {"a1": t0, "a2": t0 + timedelta(seconds=10), "b1": t0 + timedelta(seconds=500)}
    monkeypatch.setattr(photo, "_exif_time", lambda path: far.get(path))
    assert photo.pick_doorway_photo(["a1", "a2"], ["b1"]) == "a2"                      # none close: last image


def test_doorway_plan_borrows_the_previous_rooms_doorway_photo(monkeypatch):
    monkeypatch.setattr(photo, "_exif_time", lambda path: None)
    plan = photo.doorway_plan([["a1", "a2", "a3"], ["b1", "b2"], ["c1", "c2"]])
    assert plan == [("a3", None), ("b2", "a3"), (None, "b2")]


# ---------------------------------------------------------------- door matching

def test_best_door_match_prefers_the_closest_width_and_ignores_windows():
    a = [door(0.80), door(0.90), door(1.5, kind="window")]
    b = [door(1.4, kind="window"), door(0.95), door(0.78)]
    pair = ps.find_best_door_match(a, b)
    assert (pair[0].width, pair[1].width) == (0.80, 0.78)          # |0.80 - 0.78| beats |0.90 - 0.95|
    assert ps.find_best_door_match([door(0.7)], [door(1.0)]) is None   # 0.3 apart: more than 15%
    assert ps.find_best_door_match([door(1.5, kind="window")], [door(1.5, kind="window")]) is None
    assert ps.find_best_door_match([], [door()]) is None


# ---------------------------------------------------------------- placement

def test_two_rooms_are_placed_side_by_side_on_their_door():
    a = room("room-1", 4, 3, doors=[(1, 1.05, 0.9)])                # east wall (x = 4)
    b = room("room-2", 3, 3, doors=[(3, 1.05, 0.9)])                # west wall (x = 0)
    transforms, adjacencies = ps.stitch_photos([a, b])
    pa, pb = poly(a, transforms["room-1"]), poly(b, transforms["room-2"])
    assert pa.intersection(pb).area < 0.01                           # the rooms do not overlap
    assert pb.centroid.x > pa.centroid.x                             # room 2 is beyond the east wall
    da = ps._door_position_global(a, a.openings[0], transforms["room-1"])
    db = ps._door_position_global(b, b.openings[0], transforms["room-2"])
    assert np.linalg.norm(da - db) < 0.05                            # the doors coincide
    assert adjacencies == [{"room_a_id": "room-1", "room_b_id": "room-2", "shared_opening_id": "room-1/door-0"}]


def test_a_central_hall_links_non_consecutive_rooms_and_places_everything():
    # walk order: room-1, room-2, room-3, hall. The hall has one door per room (distinct widths), on its own walls.
    r1 = room("room-1", 3, 3, doors=[(1, 1.05, 0.70)])
    r2 = room("room-2", 3, 3, doors=[(2, 1.05, 0.90)])
    r3 = room("room-3", 3, 3, doors=[(3, 1.05, 1.20)])
    hall = room("hall", 2, 8, doors=[(3, 1.0, 0.70), (1, 3.0, 0.90), (1, 6.0, 1.20)])
    transforms, adjacencies = ps.stitch_photos([r1, r2, r3, hall])
    links = {(a["room_a_id"], a["room_b_id"]) for a in adjacencies}
    assert links == {("room-1", "hall"), ("room-2", "hall"), ("room-3", "hall")}     # not the chain 1-2, 2-3, 3-4
    assert set(transforms) == {"room-1", "room-2", "room-3", "hall"}
    rooms = [r1, r2, r3, hall]
    polys = [poly(r, transforms[r.room_id]) for r in rooms]
    for i in range(4):
        for j in range(i + 1, 4):
            assert polys[i].intersection(polys[j]).area < 0.05
    for adj in adjacencies:                                           # every linked door pair sits on the same point
        ra = next(r for r in rooms if r.room_id == adj["room_a_id"])
        rb = next(r for r in rooms if r.room_id == adj["room_b_id"])
        da = next(o for o in ra.openings if f"{ra.room_id}/door-{ra.openings.index(o)}" == adj["shared_opening_id"])
        best = min(np.linalg.norm(ps._door_position_global(ra, da, transforms[ra.room_id])
                                  - ps._door_position_global(rb, ob, transforms[rb.room_id])) for ob in rb.openings)
        assert best < 0.05


def test_three_or_more_solved_rooms_do_not_crash_and_unlinked_rooms_go_to_the_side():
    # Before: the overlap nudge read the transform of rooms not placed yet (KeyError with 3 rooms).
    rooms = [room(f"room-{i}", 3, 3, doors=[]) for i in range(1, 4)]            # no doors at all
    transforms, adjacencies = ps.stitch_photos(rooms)
    assert adjacencies == [] and len(transforms) == 3
    xs = [poly(r, transforms[r.room_id]).bounds for r in rooms]
    assert xs[1][0] >= xs[0][2] and xs[2][0] >= xs[1][2]                          # side by side, left to right


def test_each_door_is_used_for_one_link_only():
    # room-2 has a single door that fits both neighbours: only the walk-order pair (room-1, room-2) may use it.
    r1 = room("room-1", 3, 3, doors=[(1, 1.0, 0.9)])
    r2 = room("room-2", 3, 3, doors=[(3, 1.0, 0.9)])
    r3 = room("room-3", 3, 3, doors=[(3, 1.0, 0.9)])
    _, adjacencies = ps.stitch_photos([r1, r2, r3])
    assert [(a["room_a_id"], a["room_b_id"]) for a in adjacencies] == [("room-1", "room-2")]


# ---------------------------------------------------------------- shared doorway camera

def _camera(centre, yaw_deg, tilt_deg=0.0):
    """World-to-camera (R, t) for a camera at `centre`; camera x right, y down, z forward, heading `yaw_deg`."""
    yaw, tilt = math.radians(yaw_deg), math.radians(tilt_deg)
    fwd = np.array([math.cos(yaw) * math.cos(tilt), math.sin(yaw) * math.cos(tilt), -math.sin(tilt)])
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    rot = np.vstack([right, np.cross(fwd, right), fwd])
    return rot, -rot @ np.asarray(centre, float)


def test_shared_doorway_camera_gives_the_relative_transform():
    # Room b's frame = room a's frame rotated by 90 deg and shifted: p_a = Rz(90) p_b + (6, 1). One camera sees both.
    true_tf = (6.0, 1.0, math.pi / 2)
    c_a = np.array([5.0, 2.0, 1.4])                                 # doorway camera, room a's frame
    c, s = math.cos(true_tf[2]), math.sin(true_tf[2])
    c_b = np.array([c * 0 + s * 0, 0, 0])                            # placeholder, solved below
    inv = np.linalg.inv(np.array([[c, -s], [s, c]]))
    c_b = np.r_[inv @ (c_a[:2] - np.array(true_tf[:2])), c_a[2]]     # the same point in room b's frame
    ra, ta = _camera(c_a, 20.0)
    rb, tb = _camera(c_b, 20.0 - 90.0)                               # heading seen from room b: yaw_a - 90 deg
    path = "/photos/room-1/IMG_0006.jpg"
    k = np.eye(3)
    a = room("room-1", 4, 3, meta={"doorway_photo": path})
    a.camera_poses = [CameraPose(path, ra, ta, k)]
    b = room("room-2", 3, 3, meta={"borrowed_photo": path})
    b.camera_poses = [CameraPose(path, rb, tb, k)]
    tf = ps.pose_relative_transform(a, b)
    assert tf == pytest.approx(true_tf, abs=1e-6)
    # without the photo in one room, or when it is not the shared one, there is no pose edge
    b.camera_poses = []
    assert ps.pose_relative_transform(a, b) is None
    b.camera_poses = [CameraPose(path, rb, tb, k)]
    b.metadata["borrowed_photo"] = "/photos/other.jpg"
    assert ps.pose_relative_transform(a, b) is None


def test_a_tilted_registration_is_rejected():
    path = "/p/a6.jpg"
    ra, ta = _camera([0, 0, 1.4], 0.0)
    rb, tb = _camera([0, 0, 1.4], 0.0)
    tilt = math.radians(25)
    rb = rb @ np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]])
    a = room("room-1", 4, 3, meta={"doorway_photo": path})
    a.camera_poses = [CameraPose(path, ra, ta, np.eye(3))]
    b = room("room-2", 3, 3, meta={"borrowed_photo": path})
    b.camera_poses = [CameraPose(path, rb, tb, np.eye(3))]
    assert ps.pose_relative_transform(a, b) is None


def test_pose_edge_places_the_next_room_without_any_door():
    true_tf = (6.0, 1.0, 0.0)                                       # room b is room a shifted by (6, 1), no rotation
    path = "/p/a6.jpg"
    c_a = np.array([4.0, 1.5, 1.4])
    ra, ta = _camera(c_a, 0.0)
    rb, tb = _camera(c_a - np.array([6.0, 1.0, 0.0]), 0.0)
    a = room("room-1", 4, 3, meta={"doorway_photo": path})
    a.camera_poses = [CameraPose(path, ra, ta, np.eye(3))]
    b = room("room-2", 3, 3, meta={"borrowed_photo": path})
    b.camera_poses = [CameraPose(path, rb, tb, np.eye(3))]
    transforms, adjacencies = ps.stitch_photos([a, b])
    assert transforms["room-2"] == pytest.approx(true_tf, abs=1e-6)
    assert adjacencies == [{"room_a_id": "room-1", "room_b_id": "room-2", "shared_opening_id": None}]


def test_transform_helpers_compose_and_invert():
    t1, t2 = (1.0, 2.0, 0.7), (-3.0, 0.5, -1.2)
    p = np.array([0.4, -0.9])
    def apply(tf, q):
        c, s = math.cos(tf[2]), math.sin(tf[2])
        return np.array([c * q[0] - s * q[1] + tf[0], s * q[0] + c * q[1] + tf[1]])
    assert apply(ps.compose(t1, t2), p) == pytest.approx(apply(t1, apply(t2, p)))
    assert apply(ps.invert(t1), apply(t1, p)) == pytest.approx(p)


# ---------------------------------------------------------------- wiring into the photo tier

def _room_dir(tmp_path, name, n):
    from PIL import Image
    d = tmp_path / name
    d.mkdir()
    for i in range(1, n + 1):
        Image.new("RGB", (64, 48), (i * 10, 0, 0)).save(d / f"IMG_{i:04d}.jpg")
    return d


def test_the_borrowed_doorway_photo_reaches_the_colmap_input_but_not_the_rooms_own_images(tmp_path, monkeypatch):
    from src.tiers.colmap_utils import ColmapError
    monkeypatch.setattr(photo, "_exif_time", lambda path: None)
    d1, d2 = _room_dir(tmp_path, "room-1", 3), _room_dir(tmp_path, "room-2", 3)
    plan = photo.doorway_plan([photo._usable_images(d1), photo._usable_images(d2)])
    assert plan[0][0].endswith("room-1/IMG_0003.jpg") and plan[1][1] == plan[0][0]
    seen = []

    def stop(images, out_dir):
        seen.append(list(images))
        raise ColmapError("stop here")

    monkeypatch.setattr(photo, "prepare_images_for_colmap", stop)
    with pytest.raises(ColmapError):
        photo._reconstruct_room(d2, tmp_path / "ws", None, plan[1][0], plan[1][1])
    names = [p.split("/")[-2:] for p in seen[0]]
    assert len(seen[0]) == 4 and names[-1] == ["room-1", "IMG_0003.jpg"]             # own 3 + the borrowed one
    with pytest.raises(ColmapError):
        photo._reconstruct_room(d1, tmp_path / "ws1", None, plan[0][0], None)
    assert len(seen[1]) == 3                                                          # the first room borrows nothing
