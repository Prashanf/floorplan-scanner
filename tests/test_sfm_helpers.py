"""SfM helpers without running COLMAP: parsing, gravity alignment, scale recovery, keyframes."""

import numpy as np
import pytest
from PIL import Image

from create_test_photos import look_at
from src.geometry.openings import OpeningDetection
from src.geometry.wall_fitting import WallSegment
from src.room_ir import CameraPose, PointCloud
from src.tiers import colmap_utils as cu
from src.tiers.preprocessing import list_room_images, stage_upright_copy
from test_synthetic_room import make_room_cloud


def door(width: float) -> OpeningDetection:
    return OpeningDetection(wall_index=0, type="door", position_along_wall=1.0, width=width,
                            height=2.1, confidence=1.0)


def wall(length: float) -> WallSegment:
    return WallSegment(start=(0, 0), end=(length, 0), length=length, direction=0.0, inlier_count=100)


def test_quaternion_identity_and_rotation():
    assert cu._quaternion_to_rotation(1, 0, 0, 0) == pytest.approx(np.eye(3))
    r = cu._quaternion_to_rotation(np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4))  # 90 deg about z
    assert r @ [1, 0, 0] == pytest.approx([0, 1, 0])
    assert r @ r.T == pytest.approx(np.eye(3))


def test_parse_colmap_text_model(tmp_path):
    (tmp_path / "cameras.txt").write_text(
        "# Camera list\n1 SIMPLE_RADIAL 1280 960 900.0 640.0 480.0 0.01\n")
    (tmp_path / "images.txt").write_text(
        "# Image list\n1 1 0 0 0 0.5 0.0 2.0 1 000.jpg\n10.0 20.0 -1\n"
        "2 1 0 0 0 1.0 0.0 2.0 1 001.jpg\n\n")
    (tmp_path / "points3D.txt").write_text(
        "# 3D point list\n1 1.0 2.0 3.0 10 20 30 0.4 1 5 2 7\n"   # kept: error 0.4, track 2
        "2 9.0 9.0 9.0 10 20 30 3.0 1 5 2 7\n"                    # dropped: error 3 px
        "3 5.0 5.0 5.0 10 20 30 0.2 1 5\n")                       # dropped: track length 1
    points, poses = cu.parse_colmap_sparse(str(tmp_path))
    assert points.tolist() == [[1.0, 2.0, 3.0]]
    assert [p.image_path for p in poses] == ["000.jpg", "001.jpg"]
    assert poses[0].intrinsics[0, 0] == 900.0 and poses[0].intrinsics[0, 2] == 640.0
    assert poses[1].translation.tolist() == [1.0, 0.0, 2.0]


def test_recover_scale_from_door():
    est = cu.recover_scale(PointCloud(np.zeros((1, 3))), [door(0.78), door(0.9)], [wall(3.0)])
    assert est.method == "door" and est.factor == pytest.approx(0.86 / 0.9)


def test_recover_scale_ignores_implausible_door_then_uses_longest_wall():
    est = cu.recover_scale(PointCloud(np.zeros((1, 3))), [door(0.2)], [wall(3.0), wall(5.0)])
    assert est.method == "longest-wall" and est.factor == pytest.approx(4.0 / 5.0)


def test_recover_scale_nothing_usable():
    est = cu.recover_scale(PointCloud(np.zeros((1, 3))), [], None)
    assert est.method == "none" and est.factor == 1.0


def sfm_like_scene(seed: int = 0, unit_per_meter: float = 2.7):
    """The synthetic room as an SfM tool would hand it over: sparse, rotated, arbitrary scale."""
    rng = np.random.default_rng(seed)
    cloud = make_room_cloud(seed=seed)
    pts = cloud.points[rng.choice(len(cloud), 3500, replace=False)] + rng.normal(0, 0.01, (3500, 3))

    eyes, targets = [], []
    for x, y in [(0.5, 0.5), (2.0, 0.5), (3.5, 0.5), (3.5, 1.5), (3.5, 2.5), (2.0, 2.5), (0.5, 2.5), (0.5, 1.5)]:
        eyes.append(np.array([x, y, 1.4]))
        targets.append(np.array([2.0, 1.5, 1.2]))

    # random similarity: rotation Q, scale, shift
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.5, 2.5)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    Q = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    shift = np.array([3.0, -2.0, 5.0])
    pts_u = unit_per_meter * pts @ Q.T + shift

    poses = []
    for i, (eye, target) in enumerate(zip(eyes, targets)):
        c2w = Q @ look_at(eye, target)
        center = unit_per_meter * Q @ eye + shift
        r_w2c = c2w.T
        poses.append(CameraPose(image_path=f"{i:03d}.jpg", rotation=r_w2c, translation=-r_w2c @ center,
                                intrinsics=np.eye(3)))
    return pts_u, poses, Q


def test_align_to_gravity_recovers_vertical():
    pts, poses, Q = sfm_like_scene()
    A = cu.align_to_gravity(pts, poses)
    true_up_in_scene = Q @ np.array([0, 0, 1.0])
    assert A @ true_up_in_scene == pytest.approx([0, 0, 1], abs=0.03)
    assert np.linalg.det(A) == pytest.approx(1.0)


def test_metric_cloud_from_sparse_scene():
    pts, poses, _ = sfm_like_scene(unit_per_meter=2.7)
    cloud, new_poses, meta = cu.make_metric_point_cloud(pts, poses)
    z = cloud.points[:, 2]
    assert np.ptp(np.percentile(z, [1, 99])) == pytest.approx(2.5, abs=0.3)   # ceiling-ish height
    assert meta["scale_factor"] == pytest.approx(1 / 2.7, rel=0.08)
    assert len(new_poses) == len(poses)
    # camera heights above the floor are preserved by the metric pose transform (about 1.4 m)
    heights = [(-p.rotation.T @ p.translation)[2] for p in new_poses]
    assert np.ptp(heights) < 0.1


def test_stage_upright_copy_rotates_without_touching_original(tmp_path):
    src = tmp_path / "portrait.jpg"
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 deg clockwise to display
    Image.new("RGB", (40, 20), (200, 10, 10)).save(src, exif=exif)
    before = src.read_bytes()
    dst = tmp_path / "staged.jpg"
    stage_upright_copy(str(src), str(dst))
    assert Image.open(dst).size == (20, 40)
    assert src.read_bytes() == before

    plain = tmp_path / "plain.jpg"
    Image.new("RGB", (40, 20)).save(plain)
    link = tmp_path / "plain_staged.jpg"
    stage_upright_copy(str(plain), str(link))
    assert link.is_symlink()


def test_list_room_images_dedupes_heic_twin(tmp_path):
    for name in ("a.jpg", "b.heic", "b.jpg", "c.heic", "notes.txt", ".hidden.jpg"):
        (tmp_path / name).write_bytes(b"x")
    names = [p.split("/")[-1] for p in list_room_images(str(tmp_path))]
    assert names == ["a.jpg", "b.jpg", "c.heic"]


def test_extract_keyframes_prefers_sharp_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from src.tiers.video import extract_keyframes

    rng = np.random.default_rng(0)
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for i in range(120):
        frame = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
        if i % 4:  # three of every four frames are blurred
            frame = cv2.GaussianBlur(frame, (0, 0), 6)
        writer.write(frame)
    writer.release()

    paths = extract_keyframes(str(video), str(tmp_path / "frames"), target=10)
    assert 8 <= len(paths) <= 12
    sharpness = [cv2.Laplacian(cv2.imread(p, 0), cv2.CV_64F).var() for p in paths]
    blurred_ref = cv2.Laplacian(cv2.GaussianBlur((rng.random((240, 320)) * 255).astype(np.uint8), (0, 0), 6),
                                cv2.CV_64F).var()
    assert min(sharpness) > 10 * blurred_ref


def test_heic_roundtrip(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    from src.tiers.preprocessing import convert_heic_to_jpeg

    src = tmp_path / "IMG_1.heic"
    try:
        Image.new("RGB", (64, 48), (10, 120, 200)).save(src)
    except Exception as exc:  # encoder missing in this libheif build
        pytest.skip(f"cannot write HEIC here: {exc}")
    out = convert_heic_to_jpeg(str(src), str(tmp_path / "IMG_1.jpg"))
    assert Image.open(out).size == (64, 48)
