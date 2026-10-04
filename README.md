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
- COLMAP (photo and video tiers): `brew install colmap`
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
```
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
Working end to end: LiDAR tier (PLY/OBJ, room segmentation), geometry, JSON report, rendered plan. Stubs: photo and video front-ends, stitching and drift correction, damage detection, concealed-damage rules and scope. Until stitching exists, rooms keep their capture-frame coordinates and `adjacencies` is empty.
