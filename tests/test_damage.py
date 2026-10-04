"""Damage detection, surface projection, concealed rules and scope (no COLMAP needed)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.damage.concealed_rules import check_concealed_damage
from src.damage.detection import DamageDetection, detect_damage
from src.damage.scope import generate_scope
from src.damage.surface_projection import ProjectedDamage, project_damage_to_surfaces
from src.geometry.wall_fitting import WallSegment
from src.room_ir import CameraPose, PointCloud, RoomIR

W, H = 1280, 960


def _hsv(h: int, s: int, v: int) -> tuple[int, int, int]:
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


def _wall_image(tmp_path, name: str, draw) -> str:
    image = np.full((H, W, 3), 235, np.uint8)
    draw(image)
    path = str(tmp_path / name)
    cv2.imwrite(path, image)
    return path


def _classes(detections) -> set[str]:
    return {d.damage_class for d in detections}


def test_detects_crack_stain_mold_hole(tmp_path):
    crack = _wall_image(tmp_path, "crack.png", lambda im: cv2.line(im, (200, 200), (700, 600), (40, 40, 40), 4))
    stain = _wall_image(tmp_path, "stain.png", lambda im: cv2.ellipse(
        im, (600, 500), (120, 90), 0, 0, 360, _hsv(25, 100, 160), -1))
    mold = _wall_image(tmp_path, "mold.png", lambda im: cv2.circle(im, (500, 400), 60, _hsv(60, 60, 60), -1))
    hole = _wall_image(tmp_path, "hole.png", lambda im: cv2.circle(im, (600, 500), 40, (10, 10, 10), -1))
    clean = _wall_image(tmp_path, "clean.png", lambda im: None)

    assert "crack" in _classes(detect_damage([crack]))
    assert "water_stain" in _classes(detect_damage([stain]))
    assert "mold" in _classes(detect_damage([mold]))
    assert "hole" in _classes(detect_damage([hole]))
    assert detect_damage([clean]) == []


def test_crack_has_direction_and_thickness(tmp_path):
    path = _wall_image(tmp_path, "crack.png", lambda im: cv2.line(im, (200, 200), (700, 600), (40, 40, 40), 4))
    crack = next(d for d in detect_damage([path]) if d.damage_class == "crack")
    assert crack.thickness_px is not None and crack.thickness_px > 0
    assert 25 <= crack.angle_deg % 180 <= 55 or 125 <= crack.angle_deg % 180 <= 155
    assert crack.image_size == (W, H)


def test_detection_is_deterministic(tmp_path):
    path = _wall_image(tmp_path, "stain.png", lambda im: cv2.ellipse(
        im, (600, 500), (120, 90), 0, 0, 360, _hsv(25, 100, 160), -1))
    assert detect_damage([path]) == detect_damage([path])


def _room(tmp_path, with_pose: bool, image: str) -> RoomIR:
    rng = np.random.default_rng(0)
    z = np.concatenate([np.zeros(1500), np.full(1500, 2.5), rng.uniform(0, 2.5, 1000)])  # floor, ceiling, walls
    pts = np.column_stack([rng.uniform(0, 4, 4000), rng.uniform(0, 3, 4000), z])
    corners = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    walls = [WallSegment(start=corners[i], end=corners[(i + 1) % 4],
                         length=float(np.hypot(*np.subtract(corners[(i + 1) % 4], corners[i]))),
                         direction=0.0, inlier_count=100) for i in range(4)]
    room = RoomIR(room_id="room-1", point_cloud=PointCloud(points=pts), images=[image],
                  wall_segments=walls, ceiling_height=2.5, floor_polygon=corners)
    if with_pose:
        # camera at (2, 1.5, 1.2) looking along +x; COLMAP axes: x right, y down, z forward
        rotation = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
        center = np.array([2.0, 1.5, 1.2])
        k = np.array([[800.0, 0, 640.0], [0, 800.0, 480.0], [0, 0, 1.0]])
        room.camera_poses = [CameraPose(image_path=image, rotation=rotation,
                                        translation=-rotation @ center, intrinsics=k)]
    return room


def _det(image: str, cls: str = "water_stain", bbox=(600, 440, 680, 520)) -> DamageDetection:
    return DamageDetection(image_path=image, bbox=bbox, damage_class=cls, confidence=0.8,
                           pixel_area=6400, image_size=(W, H))


def test_projection_with_pose(tmp_path):
    image = str(tmp_path / "a.png")
    room = _room(tmp_path, True, image)
    (item,) = project_damage_to_surfaces([_det(image)], [room])
    assert item.surface_id == "room-1/wall-1"  # east wall, 2 m ahead
    assert item.location_on_surface == pytest.approx((1.5, 1.2), abs=0.02)
    assert item.extent_width == pytest.approx(80 * 2.0 / 800, abs=1e-6)
    assert item.id == "damage-0" and not item.used_fallback


def test_projection_fallback_without_pose(tmp_path):
    image = str(tmp_path / "a.png")
    room = _room(tmp_path, False, image)
    (item,) = project_damage_to_surfaces([_det(image)], [room])
    assert item.used_fallback
    assert item.surface_id in {"room-1/wall-0", "room-1/wall-2"}  # longest walls
    assert item.extent_width == pytest.approx(80 * 1.5 / (0.75 * W), abs=1e-6)


def test_projection_drops_unowned_image(tmp_path):
    room = _room(tmp_path, True, str(tmp_path / "a.png"))
    assert project_damage_to_surfaces([_det(str(tmp_path / "other.png"))], [room]) == []


def _pd(id: str, cls: str, surface: str, u: float, v: float, w: float = 0.3, h: float = 0.3,
        angle=None, crack_width=None) -> ProjectedDamage:
    det = DamageDetection(image_path="x.jpg", bbox=(0, 0, 10, 10), damage_class=cls, confidence=0.8,
                          pixel_area=100, angle_deg=angle)
    return ProjectedDamage(det, "room-1", surface, (u, v), w, h, id=id, crack_width_m=crack_width)


def _flat_room() -> RoomIR:
    corners = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    walls = [WallSegment(corners[i], corners[(i + 1) % 4], 3.0, 0.0, 10) for i in range(4)]
    return RoomIR(room_id="room-1", point_cloud=PointCloud(points=np.zeros((0, 3))),
                  wall_segments=walls, ceiling_height=2.5)


def _rules(damages) -> dict[str, list]:
    flags = check_concealed_damage([_flat_room()], damages)
    out: dict[str, list] = {}
    for f in flags:
        out.setdefault(f.rule_id, []).append(f)
    return out


def test_water_01_near_floor_only():
    fired = _rules([_pd("damage-0", "water_stain", "room-1/wall-1", 1.0, 0.2),
                    _pd("damage-1", "water_stain", "room-1/wall-2", 1.0, 1.5)])
    assert [f.evidence for f in fired["RULE_WATER_01"]] == [["damage-0"]]
    assert fired["RULE_WATER_01"][0].confidence == pytest.approx(0.5 + 0.3 * (0.09 / 0.5))


def test_water_02_aligned_stains():
    fired = _rules([_pd("damage-0", "water_stain", "room-1/wall-1", 1.00, 1.0),
                    _pd("damage-1", "water_stain", "room-1/wall-1", 1.15, 1.8),
                    _pd("damage-2", "water_stain", "room-1/wall-1", 2.50, 1.0)])
    (flag,) = fired["RULE_WATER_02"]
    assert flag.evidence == ["damage-0", "damage-1"]
    assert flag.confidence == pytest.approx(0.8)
    assert "RULE_WATER_02" not in _rules([_pd("damage-0", "water_stain", "room-1/wall-1", 1.0, 1.0)])


def test_crack_01_width_or_diagonal():
    fired = _rules([_pd("damage-0", "crack", "room-1/wall-1", 1, 1, angle=10, crack_width=0.005),
                    _pd("damage-1", "crack", "room-1/wall-1", 2, 1, angle=45, crack_width=0.001),
                    _pd("damage-2", "crack", "room-1/wall-1", 3, 1, angle=5, crack_width=0.001)])
    assert [f.evidence for f in fired["RULE_CRACK_01"]] == [["damage-0"], ["damage-1"]]
    assert fired["RULE_CRACK_01"][0].confidence == pytest.approx(0.4 + 0.3 * 0.5)


def test_mold_01_exterior_walls_only():
    fired = _rules([_pd("damage-0", "mold", "room-1/wall-0", 1, 1),
                    _pd("damage-1", "mold", "room-1/wall-1", 1, 1),
                    _pd("damage-2", "mold", "room-1/wall-3", 1, 1)])
    assert [f.evidence for f in fired["RULE_MOLD_01"]] == [["damage-0"], ["damage-2"]]


def test_ceiling_01_fires_on_sagging_ceiling():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 4, (6000, 2))
    z_flat = 2.5 + rng.normal(0, 0.003, 6000)
    z_sag = z_flat - 0.06 * (xy[:, 0] > 3.0)  # one strip 6 cm lower
    floor = np.column_stack([rng.uniform(0, 4, (3000, 2)), rng.normal(0, 0.003, 3000)])

    def room(z):
        room = _flat_room()
        room.point_cloud = PointCloud(points=np.vstack([floor, np.column_stack([xy, z])]))
        return room

    assert check_concealed_damage([room(z_flat)], []) == []
    (flag,) = check_concealed_damage([room(z_sag)], [])
    assert flag.rule_id == "RULE_CEILING_01" and flag.surface_id == "room-1/ceiling" and flag.evidence == []


def test_scope_items_and_flag_items():
    damages = [_pd("damage-0", "water_stain", "room-1/wall-1", 1.0, 0.2, w=0.5, h=0.4),
               _pd("damage-1", "peeling_paint", "room-1/wall-2", 1.0, 1.5)]
    flags = check_concealed_damage([_flat_room()], damages)
    items = generate_scope(damages, flags)
    by_region = {i.damage_region_id: i for i in items if not i.description.startswith("Further")}
    assert by_region["damage-0"].priority == "high"
    assert "0.20 m²" in by_region["damage-0"].description
    assert by_region["damage-1"].priority == "low"
    further = [i for i in items if i.description.startswith("Further investigation recommended: ")]
    assert len(further) == len(flags) == 1 and further[0].priority == "high"
    assert all(i.estimated_area.confidence_low <= i.estimated_area.value <= i.estimated_area.confidence_high
               for i in items)


def test_no_damage_gives_empty_lists():
    assert generate_scope([], []) == []
    assert check_concealed_damage([_flat_room()], []) == []
