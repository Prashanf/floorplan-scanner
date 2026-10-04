# floorplan-scanner

Turns a handheld iPhone capture into a dimensioned, stitched whole-property floor plan: walls, ceiling height, floor area, openings, per-surface damage regions, concealed-damage flags, scope line items, and a confidence interval on every measurement. Three input tiers share one output contract:

| Tier | Input | Folder layout |
|------|-------|---------------|
| `lidar` | PLY/OBJ from 3D Scanner App (Pro iPhones) | `lidar/*.ply` |
| `video` | one walkthrough clip, iPhone 15+ | `video/*.mov` |
| `photo` | 2 to 8 stills per room, iPhone 15+ | `photos/room-1/`, `photos/room-2/`, ... |

How to capture: [`capture_protocol.md`](capture_protocol.md). Hardware and accuracy: [`device_matrix.md`](device_matrix.md).

## Prerequisites
- Python 3.10+
- COLMAP (photo and video tiers): `brew install colmap`, or nothing: `pycolmap` (in requirements.txt) is used when the binary is missing. `FLOORPLAN_COLMAP_BACKEND=cli|pycolmap` forces one.
- macOS: `brew install libusb` (Open3D loads it)

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Usage (one command per capture)
```bash
python run.py ./captures/lidar  --tier lidar
python run.py ./captures/video  --tier video
python run.py ./captures/photos --tier photo
```
Options: `--output-dir ./output/`, `--no-drift-correction` (ablation), `--render/--no-render`, `--verbose`.

### Try it without a phone
```bash
python tests/create_test_ply.py          # writes test_data/apartment.ply + ground_truth.yaml
python run.py ./test_data/ --tier lidar

python tests/create_test_photos.py       # renders test_photos/room-1..3 and test_video/walkthrough.mp4
python run.py ./test_photos/ --tier photo
python run.py ./test_video/ --tier video # about 3 minutes (COLMAP runs single-threaded so results repeat exactly)
```
`pytest` runs the fast tests; `pytest -m slow` also runs COLMAP on rendered rooms.
On macOS, Open3D also needs `brew install libusb`.

## Output
In `--output-dir`:
- `report.json`: `PropertyReport` (see `src/models.py`); JSON Schema published at `schema/output_schema.json`.
- `floor_plan.png`: rendered stitched plan.

Every measurement is `{value, confidence_low, confidence_high, unit}`. Empty lists (no damage found) are present, never missing.

## Layout
- `src/tiers/` capture front-ends and preprocessing (HEIC to JPEG)
- `src/geometry/` walls, ceiling, openings, floor area
- `src/stitching/` multi-room stitching and drift correction
- `src/damage/` damage detection, concealed-damage rules, scope
- `src/calibration/` confidence intervals
- `src/output/` JSON writer and renderer
- `benchmark/` ground truth, captures, evaluation, head-to-head
- `fix_loop/` fix declaration, before and after runs

## Status
Working end to end: all three tiers (LiDAR, video, photo), geometry, JSON report, rendered plan. Photo and video clouds are sparse, so they are scaled from a floor-to-ceiling prior and refined with a 0.86 m door when one is found (`RoomIR.metadata` records which). Stubs: stitching and drift correction (photo rooms each keep their own frame and overlap in the plan until Session 5), damage detection, concealed-damage rules, scope.
