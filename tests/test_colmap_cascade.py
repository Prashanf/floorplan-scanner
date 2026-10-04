"""COLMAP configuration cascade, image preparation and keyframe cascade (no real COLMAP run needed)."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.tiers import colmap_utils
from src.tiers.colmap_utils import COLMAP_CONFIGS, ColmapError, ColmapResult, run_colmap_reconstruction
from src.tiers.preprocessing import (PreparedImage, prepare_images_for_colmap, restore_pose, restore_prior)


def test_config_table_matches_the_specification():
    assert [c["name"] for c in COLMAP_CONFIGS] == ["strict", "relaxed", "aggressive", "desperate"]
    strict, relaxed, aggressive, desperate = COLMAP_CONFIGS
    assert (strict["max_num_features"], strict["max_image_size"], strict["camera_model"], strict["single_camera"],
            strict["min_num_matches"], strict["min_model_size"]) == (8192, 3200, "SIMPLE_RADIAL", True, 15, 3)
    assert (relaxed["max_num_features"], relaxed["camera_model"], relaxed["min_num_matches"],
            relaxed["min_model_size"]) == (16384, "SIMPLE_PINHOLE", 5, 2)
    assert aggressive["single_camera"] is False and aggressive["extra"]["Mapper.multiple_models"] is True
    assert desperate["camera_model"] == "OPENCV" and desperate["min_num_matches"] == 2
    assert desperate["extra"]["Mapper.init_min_tri_angle"] == 1.0


def _fake_result(name):
    return ColmapResult(points=np.zeros((50, 3)), poses=[], config_name=name)


def test_cascade_stops_at_the_first_working_config(tmp_path, monkeypatch):
    tried = []

    def fake_try(image_dir, workspace, config, matcher="exhaustive", camera_prior=None):
        tried.append(config["name"])
        return _fake_result(config["name"]) if config["name"] == "aggressive" else None

    monkeypatch.setattr(colmap_utils, "try_colmap", fake_try)
    result = colmap_utils.run_colmap_with_fallback(str(tmp_path), str(tmp_path / "work"))
    assert tried == ["strict", "relaxed", "aggressive"] and result.config_name == "aggressive"


def test_cascade_rejects_tiny_models_and_reports_total_failure(tmp_path, monkeypatch):
    tiny = ColmapResult(points=np.zeros((10, 3)), poses=[], config_name="x")  # needs more than 10 points
    monkeypatch.setattr(colmap_utils, "try_colmap", lambda *a, **k: tiny)
    assert colmap_utils.run_colmap_with_fallback(str(tmp_path), str(tmp_path / "w")) is None
    with pytest.raises(ColmapError, match="all configurations failed"):
        run_colmap_reconstruction(str(tmp_path), str(tmp_path / "w"))


def test_real_colmap_on_unmatchable_images_runs_every_config(tmp_path, caplog):
    import logging

    caplog.set_level(logging.INFO)
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        cv2.imwrite(str(images / f"{i:03d}.jpg"), (rng.random((120, 160, 3)) * 255).astype(np.uint8))
    with pytest.raises(ColmapError):
        run_colmap_reconstruction(str(images), str(tmp_path / "w"))
    attempts = [r.message for r in caplog.records if r.message.startswith("COLMAP attempt")]
    assert attempts == [f"COLMAP attempt: {n}" for n in ("strict", "relaxed", "aggressive", "desperate")]


def test_prepare_images_rotates_portrait_resizes_and_renames(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (60, 100), (200, 30, 30)).save(src / "b_portrait.jpg")  # portrait
    Image.new("RGB", (4000, 1000), (30, 200, 30)).save(src / "a_huge.jpg")  # larger than 3200
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (100, 60), (30, 30, 200)).save(src / "c_exif.jpg", exif=exif)  # displays as 60 x 100
    paths = [str(src / n) for n in ("a_huge.jpg", "b_portrait.jpg", "c_exif.jpg")]
    items = prepare_images_for_colmap(paths, str(tmp_path / "out"))
    assert [i.name for i in items] == ["000.jpg", "001.jpg", "002.jpg"]
    assert [i.rotated for i in items] == [False, True, True]
    assert items[0].resize_scale == pytest.approx(0.8) and items[1].resize_scale == 1.0
    sizes = [Image.open(tmp_path / "out" / i.name).size for i in items]
    assert sizes == [(3200, 800), (100, 60), (100, 60)]  # every image is landscape, none above 3200
    assert (src / "b_portrait.jpg").exists() and Image.open(src / "b_portrait.jpg").size == (60, 100)


def test_prepare_images_skips_unreadable_files(tmp_path):
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    Image.new("RGB", (80, 60)).save(tmp_path / "ok.jpg")
    items = prepare_images_for_colmap([str(tmp_path / "bad.jpg"), str(tmp_path / "ok.jpg")], str(tmp_path / "o"))
    assert len(items) == 1 and items[0].source.endswith("ok.jpg")


def test_restore_pose_undoes_rotation_and_resize():
    """Project 3D points with an original camera and with its rotated, resized version; restoring the second
    camera must reproduce the first."""
    width, height, f = 600, 800, 700.0  # portrait original
    k = np.array([[f, 0, 310.0], [0, f, 395.0], [0, 0, 1.0]])
    angle = 0.3
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    translation = np.array([0.1, -0.2, 0.5])
    rng = np.random.default_rng(0)
    world = rng.uniform(-1, 1, (30, 3)) + [0, 0, 4]

    def project(r, t, kk, pts):
        cam = pts @ r.T + t
        uv = cam @ kk.T
        return uv[:, :2] / uv[:, 2:3]

    original_uv = project(rotation, translation, k, world)
    # what COLMAP would see after "rotate 90 deg counterclockwise, then scale by 0.5"
    m = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    k_rot = np.array([[k[1, 1], 0, k[1, 2]], [0, k[0, 0], width - k[0, 2]], [0, 0, 1.0]])
    k_staged = k_rot.copy()
    k_staged[:2, :] *= 0.5
    item = PreparedImage("x.jpg", "000.jpg", rotated=True, resize_scale=0.5, original_size=(width, height))
    r2, t2, k2 = restore_pose(m @ rotation, m @ translation, k_staged, item)
    assert np.allclose(r2, rotation) and np.allclose(t2, translation) and np.allclose(k2, k)
    assert np.allclose(project(r2, t2, k2, world), original_uv)
    # the prior goes the other way: original (f, cx, cy) -> staged image
    f2, cx2, cy2 = restore_prior((f, k[0, 2], k[1, 2]), item)
    assert (f2, cx2, cy2) == pytest.approx((f * 0.5, k[1, 2] * 0.5, (width - k[0, 2]) * 0.5))


def test_video_keyframe_cascade_gets_denser(tmp_path, monkeypatch):
    from src.tiers import video

    path = tmp_path / "clip"
    path.mkdir()
    writer = cv2.VideoWriter(str(path / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    rng = np.random.default_rng(0)
    for _ in range(120):
        writer.write((rng.random((120, 160, 3)) * 255).astype(np.uint8))
    writer.release()
    sizes = []

    def failing(image_dir, workspace, matcher="exhaustive", camera_prior=None):
        sizes.append(len(list(Path(image_dir).iterdir())))
        raise ColmapError("no model")

    monkeypatch.setattr(video, "run_colmap_reconstruction", failing)
    with pytest.raises(ColmapError):
        video.process_video(str(path))
    assert len(sizes) == 4  # motion, then every 10th, 5th and 3rd frame
    assert sizes[1] < sizes[2] < sizes[3]
