# floorplan-scanner

Turns a handheld phone capture into a dimensioned, stitched whole-property floor plan: walls, ceiling height, floor area, openings, per-surface damage regions, concealed-damage flags, scope line items, and a confidence interval on every measurement. Three input tiers share one output contract:

| Tier | Input | Folder layout |
|------|-------|---------------|
| `lidar` | PLY/OBJ from 3D Scanner App (Pro iPhones), or a raw depth log (`depth/`, `odometry.csv`, intrinsics) | `lidar/*.ply` or `lidar/capture/` |
| `video` | one walkthrough clip, iPhone 15+ | `video/*.mov` (also `.mp4`, `.mkv`, `.avi`) |
| `photo` | 2 to 8 stills per room, iPhone 15+ | `photos/room-1/`, `photos/room-2/`, ... |

How to capture: [`capture_protocol.md`](capture_protocol.md). Hardware and accuracy claims: [`device_matrix.md`](device_matrix.md). Results, failure modes and the fix loop: [`technical_report.md`](technical_report.md) and [`fix_loop/declaration.md`](fix_loop/declaration.md). Requirement coverage: [`compliance_matrix.md`](compliance_matrix.md). Where the raw benchmark data lives: [`benchmark/DATA.md`](benchmark/DATA.md).

Video and photos from any phone are accepted as input. Our own test captures came from a non-LiDAR camera phone, so the LiDAR tier was tested only on the three provided sample captures (see "Sample data" and "Status").

## Prerequisites
- Python 3.10+
- COLMAP (photo and video tiers): `brew install colmap`, or nothing: `pycolmap` (in requirements.txt) is used when the binary is missing. `FLOORPLAN_COLMAP_BACKEND=cli|pycolmap` forces one.
- macOS: `brew install libusb` (Open3D loads it)
- Optional: `ffmpeg` (re-encodes a video OpenCV cannot open), `rawpy` (DNG photos; already in requirements.txt, DNG files are skipped without it)

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
Options: `--output-dir ./output/`, `--no-drift-correction` (ablation), `--render/--no-render`, `--verbose`, `--ceiling-height M` (1 to 10 m, photo and video: scale the reconstruction so its vertical extent equals the known ceiling height; ignored with a warning for LiDAR), `--damage-model`, `--damage-threshold`, `--owl-model` (see below).

When no room can be reconstructed the command still writes a valid report (`room_count` 0, a `warnings` list, a blank plan) and exits 0. An invalid capture folder exits with a one-line error.

Environment variables: `FLOORPLAN_COLMAP_BACKEND=cli|pycolmap`, `FLOORPLAN_COLMAP_DEVICE=auto|cpu|cuda`, `FLOORPLAN_COLMAP_THREADS=N` (default 1, so results repeat exactly), `FLOORPLAN_KEEP_WORKSPACE=1` (keep the COLMAP workspace for inspection), `FLOORPLAN_UP_AXIS=x|y|z` (force the cloud's up axis), `FLOORPLAN_MOBILE_SAM_WEIGHTS`.

### Try it without a phone
```bash
python tests/create_test_ply.py          # writes test_data/apartment.ply + ground_truth.yaml
python run.py ./test_data/ --tier lidar

python tests/create_test_photos.py       # renders test_photos/room-1..3 and test_video/walkthrough.mp4
python run.py ./test_photos/ --tier photo
python run.py ./test_video/ --tier video # about 3 minutes (COLMAP runs single-threaded so results repeat exactly)
```
`pytest` skips tests marked `slow`; `pytest -m slow` also runs COLMAP on rendered rooms. `tests/test_photo_pipeline.py` and `tests/test_real_capture_photo.py` run the whole photo pipeline with COLMAP and are not marked `slow`, so they take minutes; the second one skips itself unless `benchmark/captures/photos/real_capture/` exists. For a quick run: `pytest -q --ignore=tests/test_photo_pipeline.py --ignore=tests/test_real_capture_photo.py` (about 25 s).
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
- `src/tiers/` capture front-ends (LiDAR depth logs and PLY/OBJ, video, photo), COLMAP wrapper with its configuration cascade, image preparation (HEIC and other formats to JPEG), room segmentation (free-space watershed), single-image fallback
- `src/geometry/` walls, room outline, ceiling, openings, floor area, scale recovery (floor tiles, door, ceiling prior, longest wall), rectangle fit
- `src/stitching/` multi-room stitching, drift correction, photo stitching, room merging (video and LiDAR: fragment merge; LiDAR also merges rooms under 2 m²)
- `src/damage/` damage detection (OpenCV and MobileSAM + OWL), concealed-damage rules, scope, detector comparison
- `src/calibration/` confidence intervals
- `src/output/` JSON writer and renderer
- `src/config.py` every tunable threshold
- `benchmark/` ground truth (`ground_truth/room-1.yaml` is the tape-measured bedroom), captures (videos are gitignored), `evaluate.py`, `head_to_head.py`, `results_saved/` (results kept from real runs)
- `fix_loop/` fix declaration, before and after runs
- `notebooks/` Kaggle notebooks (CPU and GPU) that run the pipeline end to end
- `tools/` sample-data runner, photo-set cutter, output validator, notebook builders, damage-detector comparison

## Status
All three tiers run end to end and write `report.json` and `floor_plan.png`. Stitching with drift correction, damage detection, concealed-damage rules, scope items and calibration are implemented. What the saved runs show:

| Tier | Data | Result |
|------|------|--------|
| LiDAR | three provided iPhone LiDAR logs | 6 room segments (56.9 m²), 7 (32.4 m²) and 3 (8.6 m²), 7 to 18 s each; plans are recognisable. Over-split fragments are merged (before: 10, 9 and 3 segments). No ground truth, so accuracy and the true room counts are not verified. |
| Video | our own 5-minute walkthroughs, camera phone | partial plans: landscape 2 rooms, 11.5 m² against 12.8 m² measured (first room 2.63 m × 2.85 m against 4.06 m × 3.16 m); portrait 7 rooms, 25.4 m². 8 to 17 minutes with a GPU. |
| Photo | our own four-room photo set | 0 rooms (COLMAP cannot match white, low-texture walls); a valid empty report is written. |

Known limits: photo and video scale is a prior or a floor-tile measurement, not a measurement of the room; the ceiling is often not reconstructed (ceiling height is then a lower bound); room segmentation can still over-split (LiDAR fragments under 2 m² and fragments sharing a wall without a door are now merged, which also removes some adjacencies and openings; true room counts are unverified); damage detection has no labelled evaluation. Benchmark gates pass only on the synthetic apartment (`benchmark/results/gates_report.md`); there is no real LiDAR repeatability or head-to-head result, and `benchmark/competitor/` holds a mock file. Details in [`technical_report.md`](technical_report.md).

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
- Results and known problems on this data are in `technical_report.md` (`report.md` is a stub pointing there). The data has no tape-measure ground truth, so no accuracy number is claimed from it.
- Plans are drawn in a wall-aligned frame (the cloud is rotated about Z so walls are axis-aligned), not north-up.
