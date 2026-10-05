"""Tests for Step 1 Vision Reasoning & Step 2 Metric Snapping (VLM-guided photo stitching)."""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.geometry.openings import OpeningDetection
from src.geometry.wall_fitting import WallSegment
from src.pipeline import run_pipeline
from src.room_ir import PointCloud, PropertyIR, RoomIR
from src.stitching import photo_stitch as ps
from src.stitching.vlm_adjacency import (
    build_vlm_prompt,
    infer_room_adjacency_vlm,
    parse_vlm_adjacency_json,
    resolve_model_name,
    select_protocol_photos,
)


def make_test_room(room_id: str, w: float, h: float, doors=()) -> RoomIR:
    pts = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    walls = [
        WallSegment(
            start=pts[i],
            end=pts[(i + 1) % 4],
            length=math.dist(pts[i], pts[(i + 1) % 4]),
            direction=math.atan2(pts[(i + 1) % 4][1] - pts[i][1], pts[(i + 1) % 4][0] - pts[i][0]),
            inlier_count=100,
        )
        for i in range(4)
    ]
    openings = [
        OpeningDetection(
            wall_index=wall,
            type="door",
            position_along_wall=pos,
            width=width,
            height=2.0,
            confidence=1.0,
        )
        for wall, pos, width in doors
    ]
    return RoomIR(
        room_id=room_id,
        point_cloud=PointCloud(points=np.zeros((1, 3))),
        tier="photo",
        wall_segments=walls,
        floor_polygon=pts,
        openings=openings,
    )


# --------------------------------------------------------------------------- Protocol selection

def test_select_protocol_photos():
    room_images = {
        "room1": ["img_01.jpg", "img_02.jpg", "img_03.jpg", "img_04.jpg", "img_05.jpg", "img_06.jpg"],
        "room4": [f"hall_{i:02d}.jpg" for i in range(1, 10)],  # 9 images
    }
    picks = select_protocol_photos(room_images, max_photos_per_room=4)
    assert len(picks["room1"]) == 2
    assert picks["room1"][0] == ("img_01.jpg", "entry_door")
    assert picks["room1"][-1] == ("img_06.jpg", "doorway_exit_to_hallway")

    # Hallway room picks entry, exit, and intermediate sightlines
    assert len(picks["room4"]) >= 3
    assert picks["room4"][0] == ("hall_01.jpg", "entry_door")
    assert picks["room4"][-1] == ("hall_09.jpg", "doorway_exit_to_hallway")


def test_resolve_model_name():
    assert resolve_model_name("qwen2-vl-2b") == "Qwen/Qwen2-VL-2B-Instruct"
    assert resolve_model_name("qwen2-vl-7b") == "Qwen/Qwen2-VL-7B-Instruct"
    assert resolve_model_name("custom/my-model") == "custom/my-model"


# --------------------------------------------------------------------------- JSON parsing

def test_parse_vlm_adjacency_json_clean():
    raw = json.dumps({
        "connections": [
            {"room_a": "room1", "room_b": "room2", "shared_opening": "interior_door"},
            {"room_a": "room2", "room_b": "room3", "shared_opening": "hallway"},
            {"room_a": "room1", "room_b": "room1", "shared_opening": "interior_door"},  # self-loop ignored
            {"room_a": "room2", "room_b": "room1", "shared_opening": "interior_door"},  # duplicate ignored
            {"room_a": "room1", "room_b": "nonexistent", "shared_opening": "interior_door"},  # invalid room ignored
        ]
    })
    parsed = parse_vlm_adjacency_json(raw, ["room1", "room2", "room3"])
    conns = parsed["connections"]
    assert len(conns) == 2
    assert {"room_a": "room1", "room_b": "room2", "shared_opening": "interior_door"} in conns
    assert {"room_a": "room2", "room_b": "room3", "shared_opening": "hallway"} in conns


def test_parse_vlm_adjacency_json_markdown_wrapped():
    raw = """
Here is the adjacency map based on the hallway sightlines:
```json
{
  "connections": [
    {"room_a": "Room1", "room_b": "Room4", "shared_opening": "interior_door"},
    {"room_a": "Room2", "room_b": "Room4", "shared_opening": "hallway"}
  ]
}
```
Hope this helps!
"""
    parsed = parse_vlm_adjacency_json(raw, ["room1", "room2", "room4"])
    assert len(parsed["connections"]) == 2
    assert parsed["connections"][0]["room_a"] == "room1"
    assert parsed["connections"][0]["room_b"] == "room4"


def test_parse_vlm_adjacency_json_malformed():
    raw = "I cannot determine the connections from these images."
    parsed = parse_vlm_adjacency_json(raw, ["room1", "room2"])
    assert parsed == {"connections": []}


# --------------------------------------------------------------------------- Step 2 Metric Snapping

def test_metric_snapping_with_vlm_adjacency():
    """Verify that VLM adjacency correctly snaps room polygons along shared openings."""
    # room-1: 4x3m with door on east wall (wall 1, position 1.05, width 0.9m)
    r1 = make_test_room("room-1", 4.0, 3.0, doors=[(1, 1.05, 0.9)])
    # room-2: 3x3m with door on west wall (wall 3, position 1.05, width 1.2m)
    # Note: 1.2m vs 0.9m exceeds classical 20% width tolerance (diff = 0.3 / 1.2 = 25%),
    # so classical CV alone would reject this pair! But VLM asserts they are connected.
    r2 = make_test_room("room-2", 3.0, 3.0, doors=[(3, 1.05, 1.2)])

    vlm_map = {
        "connections": [
            {"room_a": "room-1", "room_b": "room-2", "shared_opening": "interior_door"}
        ]
    }

    transforms, adjacencies = ps.stitch_photos([r1, r2], vlm_adjacency=vlm_map)
    assert "room-1" in transforms and "room-2" in transforms
    assert len(adjacencies) == 1
    assert adjacencies[0]["room_a_id"] == "room-1"
    assert adjacencies[0]["room_b_id"] == "room-2"

    p1 = ps._room_polygon(r1, transforms["room-1"])
    p2 = ps._room_polygon(r2, transforms["room-2"])
    assert p1.intersection(p2).area < 0.01  # No intersection
    assert p2.centroid.x > p1.centroid.x   # room-2 is placed to the east of room-1


def test_metric_snapping_multi_room_star_topology():
    """A central hallway (room-4) connects to room-1, room-2, and room-3."""
    r1 = make_test_room("room-1", 4.0, 3.0, doors=[(1, 1.0, 0.9)])
    r2 = make_test_room("room-2", 4.0, 3.0, doors=[(3, 1.0, 0.9)])
    r3 = make_test_room("room-3", 3.0, 3.0, doors=[(0, 1.0, 0.9)])
    r4 = make_test_room("room-4", 6.0, 2.0, doors=[(3, 0.5, 0.9), (1, 0.5, 0.9), (2, 2.0, 0.9)])

    vlm_map = {
        "connections": [
            {"room_a": "room-1", "room_b": "room-4", "shared_opening": "hallway"},
            {"room_a": "room-2", "room_b": "room-4", "shared_opening": "hallway"},
            {"room_a": "room-3", "room_b": "room-4", "shared_opening": "hallway"},
        ]
    }

    transforms, adjacencies = ps.stitch_photos([r1, r2, r3, r4], vlm_adjacency=vlm_map)
    assert len(transforms) == 4
    assert len(adjacencies) >= 3

    # Check that all rooms are positioned without large overlaps
    polys = [ps._room_polygon(r, transforms[r.room_id]) for r in [r1, r2, r3, r4]]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            assert polys[i].intersection(polys[j]).area < 0.05


def test_infer_room_adjacency_vlm_caching(tmp_path):
    """Verify cache loading avoids running the model."""
    cache_file = tmp_path / "vlm_cache.json"
    cache_data = {
        "connections": [
            {"room_a": "room1", "room_b": "room2", "shared_opening": "interior_door"}
        ]
    }
    cache_file.write_text(json.dumps(cache_data))

    res = infer_room_adjacency_vlm(
        room_images={"room1": ["fake1.jpg"], "room2": ["fake2.jpg"]},
        cache_path=cache_file,
    )
    assert res == cache_data


def test_pipeline_photo_tier_with_vlm_adjacency(tmp_path):
    """Verify run_pipeline with photo_stitcher='vlm' and vlm_adjacency produces valid report and floor plan."""
    test_photos_dir = Path(__file__).resolve().parents[1] / "test_photos"
    if not (test_photos_dir / "room-1").is_dir():
        pytest.skip("test_photos/room-1 not found")

    out_dir = tmp_path / "vlm_photo_out"
    vlm_map = {
        "connections": [
            {"room_a": "room-1", "room_b": "room-2", "shared_opening": "interior_door"},
            {"room_a": "room-2", "room_b": "room-3", "shared_opening": "interior_door"},
        ]
    }
    vlm_json_path = tmp_path / "vlm_map.json"
    vlm_json_path.write_text(json.dumps(vlm_map))

    report = run_pipeline(
        capture_dir=str(test_photos_dir),
        tier="photo",
        output_dir=str(out_dir),
        render=True,
        photo_stitcher="vlm",
        vlm_adjacency=str(vlm_json_path),
    )
    assert (out_dir / "floor_plan.png").is_file()
    assert (out_dir / "report.json").is_file()
    assert report.capture_tier == "photo"
    assert report.room_count >= 1
    assert len(report.adjacencies) >= 1

