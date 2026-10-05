"""Floor tile scale recovery: rendered tiled floors seen through a perspective camera, and the scale chain."""

import cv2
import numpy as np
import pytest

from src.geometry.scale_recovery import detect_tile_grid, recover_scale_from_tiles, snap_tile_size
from src.room_ir import CameraPose, PointCloud
from src.tiers import colmap_utils as cu

W, H, FOCAL = 1280, 960, 1000.0
K = np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1.0]])


def camera(center, target):
    """World-to-camera rotation and translation for a camera at `center` looking at `target` (Z up)."""
    forward = np.asarray(target, float) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    rot = np.vstack([right, np.cross(forward, right), forward])
    return rot, -rot @ center


def render_floor(tile_m, center, target, rotation_deg=20.0, seed=0, grout=0.012):
    """Image of a floor of `tile_m` square tiles (shaded at random, dark grout), metres, camera pose."""
    rng = np.random.default_rng(seed)
    rot, trans = camera(np.asarray(center, float), target)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays = (rot.T @ np.linalg.inv(K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])).T
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -center[2] / rays[:, 2]
    hit = np.asarray(center, float)[:2] + lam[:, None] * rays[:, :2]
    a = np.deg2rad(rotation_deg)
    gx = hit[:, 0] * np.cos(a) + hit[:, 1] * np.sin(a)
    gy = -hit[:, 0] * np.sin(a) + hit[:, 1] * np.cos(a)
    ix, iy = np.floor(gx / tile_m).astype(int), np.floor(gy / tile_m).astype(int)
    shade = rng.integers(170, 215, size=(64, 64))[ix % 64, iy % 64]
    on_grout = (np.mod(gx, tile_m) < grout) | (np.mod(gy, tile_m) < grout)
    img = np.where(on_grout, 60, shade).astype(np.float32)
    img[(rays[:, 2] >= 0) | (lam <= 0)] = 150  # above the horizon: plain wall
    img = img.reshape(H, W) + rng.normal(0, 2, (H, W))
    return cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0), rot, trans


def room_cloud(units_per_meter, seed=0):
    """Floor, ceiling and wall points of a 5 x 4 m room, 2.5 m high, in arbitrary units."""
    rng = np.random.default_rng(seed)
    floor = np.c_[rng.uniform(0, 5, 4000), rng.uniform(0, 4, 4000), rng.normal(0, 0.005, 4000)]
    ceiling = np.c_[rng.uniform(0, 5, 1500), rng.uniform(0, 4, 1500), 2.5 + rng.normal(0, 0.005, 1500)]
    walls = np.c_[np.r_[np.zeros(800), np.full(800, 5.0), rng.uniform(0, 5, 800), rng.uniform(0, 5, 800)],
                  np.r_[rng.uniform(0, 4, 800), rng.uniform(0, 4, 800), np.zeros(800), np.full(800, 4.0)],
                  rng.uniform(0, 2.5, 3200)]
    return PointCloud(points=np.vstack([floor, ceiling, walls]) * units_per_meter)


def pose_for(rot, trans, units_per_meter, path="x.jpg"):
    return CameraPose(image_path=path, rotation=rot, translation=trans * units_per_meter, intrinsics=K)


@pytest.mark.parametrize("tile_m", [0.30, 0.45, 0.60, 0.80])
def test_tile_grid_gives_the_scale_at_arbitrary_units(tile_m):
    units = 0.37  # cloud units per metre
    image, rot, trans = render_floor(tile_m, [2.5, -0.6, 1.5], [2.5, 2.0, 0.0])
    result = detect_tile_grid(image, pose_for(rot, trans, units), room_cloud(units))
    assert result is not None
    assert result.tile_size_m == tile_m
    assert result.scale_factor == pytest.approx(1 / units, rel=0.06)  # metres per unit
    assert result.confidence > 0.5


def test_snapping_follows_the_bands():
    assert [snap_tile_size(s) for s in (0.27, 0.43, 0.55, 0.62, 0.75)] == [0.30, 0.45, 0.60, 0.60, 0.80]
    assert snap_tile_size(0.38) is None and snap_tile_size(1.0) is None


def test_a_rough_prior_still_snaps_to_the_right_tile():
    units = 2.9
    image, rot, trans = render_floor(0.60, [2.5, -0.6, 1.5], [2.5, 2.0, 0.0], rotation_deg=-12)
    cloud = room_cloud(units)
    for off in (0.85, 1.07):  # a 0.60 m tile reads 0.51 to 0.64 m: inside its 0.50-0.65 band
        result = detect_tile_grid(image, pose_for(rot, trans, units), cloud, prior_scale=off / units)
        assert result is not None and result.tile_size_m == 0.60
        assert result.scale_factor == pytest.approx(1 / units, rel=0.06)  # the scale comes from the tile, not the prior


def test_noise_floor_has_no_grid():
    rng = np.random.default_rng(1)
    noise = cv2.GaussianBlur(rng.integers(0, 255, (H, W)).astype(np.uint8), (5, 5), 0)
    rot, trans = camera(np.array([2.5, -0.6, 1.5]), [2.5, 2.0, 0.0])
    assert detect_tile_grid(noise, pose_for(rot, trans, 1.0), room_cloud(1.0)) is None


def test_parallel_planks_are_not_tiles():
    img, rot, trans = render_floor(0.60, [2.5, -0.6, 1.5], [2.5, 2.0, 0.0], rotation_deg=0)
    rng = np.random.default_rng(0)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    planks = np.where((u // 40) % 2 == 0, 90, 190).astype(np.uint8) + rng.integers(0, 5, (H, W), dtype=np.uint8)
    assert detect_tile_grid(planks, pose_for(rot, trans, 1.0), room_cloud(1.0)) is None


def test_missing_image_or_wrong_height_gives_none(tmp_path):
    rot, trans = camera(np.array([2.5, -0.6, 1.5]), [2.5, 2.0, 0.0])
    pose = pose_for(rot, trans, 1.0, str(tmp_path / "nothing.jpg"))
    assert detect_tile_grid(pose.image_path, pose, room_cloud(1.0)) is None
    assert recover_scale_from_tiles([pose], room_cloud(1.0)) is None


def test_several_images_agree(tmp_path):
    units = 0.37
    poses = []
    for i, (cx, tx) in enumerate([(1.5, 1.8), (2.5, 2.5), (3.5, 3.1)]):
        image, rot, trans = render_floor(0.60, [cx, -0.6, 1.5], [tx, 2.0, 0.0], seed=i)
        path = tmp_path / f"{i}.png"
        cv2.imwrite(str(path), image)
        poses.append(pose_for(rot, trans, units, str(path)))
    result = recover_scale_from_tiles(poses, room_cloud(units))
    assert result is not None and result.n_images == 3 and result.tile_size_m == 0.60
    assert result.scale_factor == pytest.approx(1 / units, rel=0.05)


# ---------------------------------------------------------------- the scale chain

def _fake_tiles(factor, confidence):
    from src.geometry.scale_recovery import TileGridResult
    return TileGridResult(scale_factor=factor, confidence=confidence, tile_size_m=0.6, period_units=0.6 / factor,
                          computed_size_m=0.6, method="hough")


def test_confident_tiles_win_and_are_logged(monkeypatch, caplog):
    from test_sfm_helpers import sfm_like_scene
    pts, poses, _ = sfm_like_scene(unit_per_meter=2.7)
    monkeypatch.setattr(cu, "recover_scale_from_tiles", lambda *a, **k: _fake_tiles(0.37, 0.85))
    with caplog.at_level("INFO", logger="src.tiers.colmap_utils"):
        _, _, meta = cu.make_metric_point_cloud(pts, poses)
    assert meta["scale_method"] == "floor-tiles" and meta["scale_factor"] == pytest.approx(0.37)
    assert meta["scale_confidence"] == pytest.approx(0.85)
    assert "Scale recovered from floor tiles (0.60m × 0.60m grid detected, confidence 0.85)" in caplog.text
    assert [m for m, _ in meta["scale_tried"]] == ["floor-tiles"]  # nothing after the first confident method ran


def test_weak_tiles_fall_through_to_the_next_method(monkeypatch):
    from test_sfm_helpers import sfm_like_scene
    pts, poses, _ = sfm_like_scene(unit_per_meter=2.7)
    monkeypatch.setattr(cu, "recover_scale_from_tiles", lambda *a, **k: _fake_tiles(0.37, 0.4))  # not above 0.5
    _, _, meta = cu.make_metric_point_cloud(pts, poses)
    assert meta["scale_method"] in {"door", "ceiling-height-prior"}
    assert meta["scale_tried"][0] == ["floor-tiles", 0.4]
    assert meta["scale_confidence"] > 0.5


def test_no_images_means_the_old_chain_is_unchanged():
    from test_sfm_helpers import sfm_like_scene
    pts, poses, _ = sfm_like_scene(unit_per_meter=0.4)  # image files do not exist: tiles give None
    _, _, meta = cu.make_metric_point_cloud(pts, poses)
    assert meta["scale_method"] != "floor-tiles" and "floor-tiles" not in [m for m, _ in meta["scale_tried"]]


def test_low_confidence_everywhere_uses_the_longest_wall_last_resort(monkeypatch):
    from test_sfm_helpers import sfm_like_scene
    pts, poses, _ = sfm_like_scene()
    monkeypatch.setattr(cu, "_outline_area", lambda cloud: 3.0)  # every method leaves a room under 4 m2
    monkeypatch.setattr(cu, "_longest_wall", lambda cloud: 2.0)
    _, _, meta = cu.make_metric_point_cloud(pts, poses)
    assert meta["scale_method"] == "longest-wall-prior" and meta["scale_confidence"] == pytest.approx(0.2)


def test_ceiling_candidate_is_chosen_from_a_visible_door():
    ctx = cu._ScaleContext(aligned=np.zeros((1, 3)), poses=[], factor0=1.0, method0="vertical-extent")
    ctx.door_widths_units = [0.86]  # in a cloud 3.0 units tall: a 0.86 m door only if the ceiling is 3.0 m
    assert cu._pick_ceiling(ctx, extent=3.0) == (3.0, True)
    ctx.door_widths_units = [1.03]  # 1.03 x 2.5 / 3.0 = 0.86 m: the default fits
    assert cu._pick_ceiling(ctx, extent=3.0) == (2.5, True)
    ctx.door_widths_units = []
    assert cu._pick_ceiling(ctx, 3.0) == (2.5, False)  # no door seen: the default prior


def test_end_to_end_tiled_floor_through_the_chain(tmp_path):
    """Rendered 0.60 m tiles, cloud rotated and scaled arbitrarily: the chain returns the true scale."""
    units = 0.37
    rng = np.random.default_rng(3)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = 1.1
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    q = np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx
    cloud = room_cloud(units).points
    poses = []
    for i, (cx, tx) in enumerate([(1.5, 1.8), (2.5, 2.5), (3.5, 3.1)]):  # cameras looking down at the floor
        image, rot, trans = render_floor(0.60, [cx, 0.3, 1.5], [tx, 3.0, 0.2], seed=i)
        path = tmp_path / f"{i}.png"
        cv2.imwrite(str(path), image)
        poses.append(CameraPose(str(path), rot @ q.T, trans * units, K))
    for i, (cx, tz) in enumerate([(1.0, 1.5), (2.0, 2.2), (3.0, 1.6), (4.0, 2.3)]):  # level or upward: no image needed
        rot, trans = camera(np.array([cx, 0.3, 1.4]), [cx, 3.0, tz])
        poses.append(CameraPose(str(tmp_path / f"none{i}.png"), rot @ q.T, trans * units, K))
    pts = (cloud @ q.T) + rng.normal(0, 0.003, cloud.shape)
    out, _, meta = cu.make_metric_point_cloud(pts, poses)
    assert meta["scale_method"] == "floor-tiles", meta
    assert meta["scale_factor"] * units == pytest.approx(1.0, rel=0.06)  # metres per unit, relative to the truth
    assert np.ptp(np.percentile(out.points[:, 2], [1, 99])) == pytest.approx(2.5, rel=0.08)  # tiles never saw the ceiling
