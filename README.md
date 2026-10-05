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

## Damage detectors (two modes)

`--damage-model opencv` (default; `--damage-detector` and the value `heuristic` are the same thing) uses the OpenCV rules: cracks, stains, mold, peeling paint and holes from edges and colour ranges. No extra install.

`--damage-model mobilesam` uses OWLv2 text prompts for boxes and MobileSAM for masks (mask area, crack angle and thickness). Setup:

```bash
pip install -r requirements-damage-model.txt
python tools/download_mobile_sam.py          # weights/mobile_sam.pt, about 40 MB (without it, box areas are used)
python run.py <capture_dir> --tier photo --damage-model mobilesam [--damage-threshold 0.3] [--owl-model google/owlvit-base-patch32]
```

To see which one does better on your images, run both on the same files:

```bash
python tools/compare_damage_detectors.py <images or folders> --out output/damage_compare [--labels labels.json]
```

It writes `comparison.md` / `comparison.json` (counts per class, agreement, speed) and `overlays/` (blue = heuristic, red = model). With hand labels (`{"IMG.jpg": [{"class": "crack", "bbox": [x1, y1, x2, y2]}], "clean.jpg": []}`) it also reports precision and recall per detector. The model threshold (`MODEL_DAMAGE_THRESHOLD` in `src/config.py`) is provisional until it is tuned on labelled captures.

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

## Sample data (the three provided captures)

The provided zips (`single_room`, `single_scan_floor_only`, `single_scan_with_ceiling`) are raw iPhone LiDAR logs: `depth/*.png` (256 x 192, mm), `confidence/*.png`, `odometry.csv` (ARKit pose and intrinsics per frame), `camera_matrix.csv`, `imu.csv` and `rgb.mp4`. They are not in the repository (about 900 MB). To rerun:

```bash
mkdir -p data_raw && unzip single_room.zip -d data_raw/single_room        # likewise for the other two
# one folder per capture and tier under sample_data_run/ (raw data is linked, not copied):
#   sample_data_run/lidar/<name>/capture/   depth/ confidence/ odometry.csv camera_matrix.csv ... (links)
#   sample_data_run/video/<name>/           rgb.mp4 camera_matrix.csv (links)
#   sample_data_run/photos/<name>/room-N/   stills cut from the video, see below (committed)
python tools/make_photo_sets.py sample_data_run/lidar/<name>/capture sample_data_run/photos/<name>
PYTHON=python tools/run_sample_data.sh              # lidar, photo (with drift ablation), video
python tools/validate_sample_output.py              # schema and field check of output/sample_*/
```

- **LiDAR tier** reads the raw depth log directly (`src/tiers/depth_stream.py`): every depth pixel is back-projected with the logged intrinsics and pose, so no PLY export is needed.
- **Photo tier** has no photos in the provided data, so `tools/make_photo_sets.py` cuts stills from `rgb.mp4`: rooms come from the LiDAR run, and each room gets up to 8 sharp frames from one continuous stay, rotated upright. The photo tier itself sees only the JPEGs (no depth, no poses).
- **Video tier** runs on `rgb.mp4` with the intrinsics from `camera_matrix.csv` as the COLMAP starting point.
- Results and known problems on this data are in `report.md`. The data has no tape-measure ground truth, so no accuracy number is claimed from it.
- Plans are drawn in a wall-aligned frame (the cloud is rotated about Z so walls are axis-aligned), not north-up.
