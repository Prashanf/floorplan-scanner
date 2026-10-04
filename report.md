# floorplan-scanner: Technical Report

Status as of 2026-10-04. Pipeline version 0.1.0. This report says how each requirement of the Applied AI case study is met, which file implements it, and what is still unproven.

**Read this first.** Every number in this report comes from synthetic data (a generated apartment point cloud, rendered photos, a rendered video). No real iPhone capture has been run yet, no real tape or laser ground truth exists, and the "competitor" file used for the head-to-head is a mock, not a Polycam export. The LiDAR tier is shown to work end to end on clean synthetic input. The photo and video tiers run, but the photo tier does not yet pass the whole-property stitch on the synthetic set. The fix loop has not been run.

---

## 1. What the system does

One command per capture turns a handheld iPhone capture into a dimensioned, stitched whole-property plan:

```bash
python run.py <capture_dir> --tier lidar|video|photo
```

Output, in `--output-dir`:

- `report.json`: a `PropertyReport` (`src/models.py`), validated against the published schema in `schema/output_schema.json`.
- `floor_plan.png`: the rendered stitched plan.

Every measurement is `{value, confidence_low, confidence_high, unit}`. Empty lists (for example, no damage found) are present, never missing.

## 2. Architecture

```
capture folder
  -> preprocess (HEIC->JPEG, validate layout)            src/tiers/preprocessing.py
  -> tier front-end -> PropertyIR (rooms as point clouds) src/tiers/{lidar,photo,video}.py
  -> geometry per room (walls, ceiling, openings, area)   src/geometry/
  -> stitching (adjacency, drift correction)              src/stitching/
  -> damage -> concealed rules -> scope                   src/damage/
  -> calibration (intervals)                              src/calibration/confidence.py
  -> JSON + rendered plan                                 src/output/
```

The design idea is that the three tiers differ only in the front-end. Each front-end produces the same intermediate representation, a metric Z-up point cloud per room (`RoomIR` in `src/room_ir.py`). Everything downstream (geometry, stitching, damage, calibration, output) is shared. That is how all three tiers return "the same output contract" and why photos are a first-class path rather than a special case. `src/pipeline.py` orchestrates the steps and prints a timing summary.

## 3. How each objective is achieved

### 3.1 Capture route (Route 2: stock capture protocol)

- **What:** `capture_protocol.md` is a one-page protocol for a non-engineer using stock iPhone apps: 3D Scanner App (LiDAR), the Camera app (video and photos).
- **How:** It gives phone height (chest, about 1.4 m), walking speed (one step per second for LiDAR, half a step per second for video), overlap for photos (60%), a door-in-frame photo per room for scale, what to avoid (mirrors, glass, bright windows, fast turns), and the hand-off folder layout (`lidar/`, `video/`, `photos/room-N/`).
- **Status:** Written; never followed by a real person yet.

### 3.2 Device matrix

- **What:** `device_matrix.md` maps each tier to minimum hardware, sensor, and claimed accuracy.
- **How:** LiDAR needs a Pro-class iPhone. Video and photo run on any iPhone 15 or newer. The claimed accuracy widens as sensor data thins:

| Tier | Wall length | Ceiling height | Opening width |
|------|-------------|----------------|---------------|
| LiDAR | ±2 cm or ±0.5% | ±1.5 cm | ±2 cm |
| Video | ±3% | ±3% | ±5 cm |
| Photo | ±8% | ±8% | ±8 cm |

- **Status:** These are design targets, not measurements. They must be replaced by numbers from real captures.

### 3.3 Input tiers

**LiDAR** (`src/tiers/lidar.py`, `src/tiers/room_segmentation.py`)

1. Load PLY/OBJ with Open3D, merge files, voxel-downsample to 2 cm, estimate normals.
2. Segment rooms. The wall-height band is projected to 2D, free space is split with a watershed over the distance transform, and each point is assigned to a room. This replaced a plain DBSCAN, because rooms joined by a doorway are one connected cluster to DBSCAN.
3. Emit one `RoomIR` per room with point density.

**Video** (`src/tiers/video.py`)

1. Extract keyframes with OpenCV (histogram-difference sampling, about 50 to 150 frames).
2. Run COLMAP (`src/tiers/colmap_utils.py`) on the keyframes.
3. Align to gravity, scale, then split into rooms with the same segmentation as LiDAR.

**Photo** (`src/tiers/photo.py`)

1. One COLMAP reconstruction per room folder (2 to 8 stills). A room that fails to reconstruct is skipped with a warning, not fatal.
2. HEIC is converted to JPEG in preprocessing.
3. Rooms have no shared coordinate frame, so they are stitched later by door matching (section 3.5).

**Shared SfM helpers** (`src/tiers/colmap_utils.py`)

- Backend: the `colmap` binary if present, otherwise `pycolmap`. `FLOORPLAN_COLMAP_BACKEND=cli|pycolmap` forces one.
- Fixed seed and single-threaded COLMAP so repeated runs give the same result.
- Gravity alignment from camera up-vectors refined by plane fits.
- Scale: a prior from vertical extent (typical 2.2 to 2.5 m) is refined with a 0.86 m door when a plausible door is found. A door-based correction outside 0.8x to 1.25x is rejected as a false door. The method used is recorded in `RoomIR.metadata`.
- Limitation: SfM scale is a prior, not a measurement. A non-standard door biases the whole room.

### 3.4 Per-room geometry

| Output | File | Method |
|--------|------|--------|
| Walls | `src/geometry/wall_fitting.py` | Detect floor and ceiling, slice 0.8 to 1.5 m above the floor, project to 2D, iterative RANSAC lines (3 cm inlier threshold for LiDAR, 6 cm for SfM), snap to 90°, intersect adjacent lines into a closed counter-clockwise polygon |
| Ceiling height | `src/geometry/ceiling.py` | Z-histogram (2 cm bins); floor peak in the lower quarter, ceiling peak in the upper quarter |
| Openings | `src/geometry/openings.py` | Per wall, project nearby points to (along-wall, vertical), bin along the wall, find empty gaps wider than 0.4 m; a gap starting near the floor is a door, one starting above 0.7 m is a window |
| Floor area | `src/geometry/floor_area.py` | Shapely polygon area |

A room whose fit has fewer than 4 walls or an area under 1 m² is dropped as a failed fit (`src/pipeline.py`). `tests/test_synthetic_room.py` checks a 4 × 3 × 2.5 m room with one 0.86 m door against ground truth.

### 3.5 Stitched multi-room plan and adjacency

**LiDAR and video** (`src/stitching/multi_room.py`): rooms already share a frame. Two rooms are adjacent when openings of both lie within 0.5 m of each other. The result is recorded as `Adjacency(room_a_id, room_b_id, shared_opening_id)`. Room polygons are checked for overlap with Shapely.

**Photo** (`src/stitching/photo_stitch.py`): rooms have no common frame, so the folders are taken in walk order (`room-1`, `room-2`, ...). Room N is linked to room N+1 through the first pair of doors of similar width (within 15%). Each room is rotated and translated so its door coincides with the neighbour's door and the shared wall is coincident. Overlaps are detected with Shapely and the room is nudged.

### 3.6 Drift accountability

The case study says "poses used as-is" fails this row, and requires an on/off ablation.

- **What we do:** Plane-anchored correction (`src/stitching/drift_correction.py`). For each adjacent room pair, find shared walls (roughly parallel within 10°, midpoints within 0.3 m, lengths within 20%). Then minimise the squared discrepancy over per-room XY translations with `scipy.optimize.minimize` (first room fixed). Offset along a wall counts as drift. Offset across a wall of up to 0.15 m is treated as wall thickness (two scans of one wall) and is free.
- **Ablation switch:** `--no-drift-correction` runs raw placement. The flag value is printed and logged.
- **Evidence:** `tests/test_drift_ablation.py` builds the synthetic apartment with injected drift (`tests/create_test_ply.py --drift DX DY`, 0.15 m of drift per room walked, which accumulates). Room origin error against the true layout:

| Room | Correction off | Correction on |
|------|----------------|---------------|
| room-1 | 0.00 m | 0.00 m |
| room-2 | 0.15 m | 0.001 m |
| room-3 | 0.30 m | 0.00 m |

  On clean input (no drift) the correction leaves the plan untouched (error under 0.05 m).
- **Bug found and fixed during integration (commit `92a124f`):** the shared-wall finder contained an angle test that rejected every parallel pair, so correction never ran and the flag changed nothing. The ablation test would have caught this, and now guards against it.
- **Limits:** translation only (no rotation drift, no loop closure). Correction needs shared walls between adjacent rooms; with none found it does nothing and logs it.

### 3.7 Damage, concealed-damage flags, and scope

**Damage detection** (`src/damage/detection.py`): OpenCV heuristics, no model download. Cracks come from long thin Canny contours, water stains and mold from HSV colour ranges, peeling paint from texture variance, holes from dark circular regions. Overlapping same-class detections are merged by non-maximum suppression. It is designed for obvious staged damage (tape cracks, paper stains).

**Surface projection** (`src/damage/surface_projection.py`): a ray from the camera pose through the box centre is intersected with the room's wall planes. The closest hit gives the surface (`<room>/wall-<i>`), and the pixel size becomes metres through distance and focal length. Without a camera pose, a fallback assumes the camera is 1.5 m from the wall.

**Concealed-damage rules** (`src/damage/concealed_rules.py`): five rules, each emitting a flag with the rule id and the evidence region ids:

| Rule | Fires when |
|------|-----------|
| `RULE_WATER_01` | A water stain lies within 0.3 m of the wall-floor junction |
| `RULE_WATER_02` | Two or more stains on one wall are vertically aligned (within 0.2 m) |
| `RULE_CRACK_01` | A crack is wider than 3 mm or diagonal (30° to 60°) |
| `RULE_MOLD_01` | Mold is on an exterior wall (first or last wall of the polygon) |
| `RULE_CEILING_01` | Ceiling height varies by more than 2 cm across the room |

**Scope** (`src/damage/scope.py`): a lookup table maps each damage class to a line item and priority, keyed to the surface id, with the area interval. Each concealed flag becomes a high-priority "further investigation" item.

### 3.8 Confidence interval on every measurement

`src/calibration/confidence.py` sets `confidence_low` and `confidence_high` on every `Measurement` (wall lengths and heights, openings, ceiling, floor area, damage extents, scope areas, total area). The half-width is a base uncertainty for the measurement type times a tier multiplier:

| Measurement | Base | | Tier | Multiplier |
|-------------|------|-|------|-----------|
| Wall length | 1 cm | | LiDAR | 1.5 |
| Ceiling height | 1 cm | | Video | 3.0 |
| Opening width | 2 cm | | Photo | 6.0 |
| Floor area | 0.1 m² | | | |
| Damage extent | 5 cm | | | |

So intervals widen from LiDAR to photo. If a `calibration_data.json` exists in the project root, fitted half-widths per tier and measurement type replace the hardcoded ones. That is the hook for replacing the multipliers with values fitted on real benchmark data.

Honest status: the multipliers are design assumptions. They have not been validated against real error distributions, so they do not yet count as calibrated.

### 3.9 Output contract, schema, rendered plan

- **JSON:** Pydantic v2 models in `src/models.py` with field descriptions and examples. `src/output/json_writer.py` writes `report.json` and regenerates `schema/output_schema.json`. `tests/test_schema.py` checks that a sample report validates.
- **Plan:** `src/output/renderer.py` draws one colour per room, walls labelled with length in metres (2 decimals), doors as open gaps with jambs, windows as dashed blue lines, red X damage markers by class, a legend, a north arrow, a round scale bar, and room name with area.

### 3.10 Benchmark, gates, and head-to-head

- **Gate evaluation** (`benchmark/evaluate.py`): reads ground-truth YAML and pipeline JSON, matches rooms by id, and scores the gates: opening width within 2 cm on at least 85% (a missed opening and a phantom opening each count as a miss), ceiling height within 1.5 cm, repeatability, photo-tier footprint within 8%, and the share of 90% intervals that contain the truth. It writes `benchmark/results/gates_report.md`.
- **Head-to-head** (`benchmark/head_to_head.py`): per shared dimension, our error against the competitor's error, and the share where we beat or tie.
- **Result so far (synthetic apartment, LiDAR tier):**

| Gate | Value | Threshold | Result |
|------|-------|-----------|--------|
| Opening widths | 4 of 4 within 2 cm (max error 1.77 cm) | at least 85% | pass |
| Ceiling height | 0.11 cm max error | 1.5 cm | pass |
| Wall length | 0.08 cm max error | n/a | n/a |
| Missed / phantom openings | 0 / 0 | 0 | pass |
| Repeatability | no repeat captures | 1 cm or 0.5% | not run |
| Photo-tier stitch | no passing photo result | within 8% | not run |

  A perfect score on noise-free synthetic data is a software sanity check, not evidence about real accuracy.
- **Head-to-head:** `benchmark/competitor/apartment_polycam.json` is a mock file I wrote to exercise the script. It is not a real Polycam or magicplan export. The 70% beat-or-tie requirement is untested.

### 3.11 Timing

LiDAR on the 3-room synthetic apartment: about 1.2 s end to end. Photo (3 rooms, pycolmap): about 33 s. Video (rendered walkthrough, pycolmap): about 167 s. COLMAP dominates the SfM tiers.

## 4. Error budget

| Source | Effect | How it is handled |
|--------|--------|-------------------|
| Sensor noise (LiDAR, a few mm) | Wall and ceiling error at the mm level | RANSAC over thousands of points; the 1.5x multiplier leaves margin |
| SfM scale (photo, video) | Whole-room scale error, the largest term in the SfM tiers | Vertical-extent prior refined by the 0.86 m door; the interval is wide (6x for photo) |
| Sparse SfM points | Noisier wall lines, missed or false openings | Larger RANSAC threshold (6 cm), door-height gap rule for openings |
| Pose drift across rooms | Footprint distortion | Shared-wall translation correction (section 3.6) |
| Wall thickness | A gap between rooms that is real, not drift | 0.15 m tolerance across shared walls |
| Damage extent from one image | Depth error scales the metric size | 5 cm base error widened per tier; fallback distance when poses are absent |

## 5. Calibration analysis

Calibration here means that stated intervals should contain the truth at the stated rate. The mechanism exists (per-type, per-tier intervals, with a fitted-parameter path). `benchmark/evaluate.py` reports the interval coverage on a 90% target. On the synthetic LiDAR run it is 100%, which only shows the intervals are not too narrow for noise-free data. Real calibration needs real captures at all three tiers, then fitting the half-widths into `calibration_data.json`. Until then, SfM-tier intervals are assumptions.

## 6. Fix loop (25% of the score)

Not done. `fix_loop/declaration.md` is an unfilled template and `fix_loop/before/` and `fix_loop/after/` are empty. This needs real benchmark numbers: pick the worst gate, state the root-cause hypothesis and a predicted number, ship the fix, and keep regenerable before and after runs.

The photo-tier stitch failure below is a likely candidate, but only after it is confirmed on real photos.

## 7. Compliance matrix

| Requirement | File path | Artifact | Status |
|-------------|-----------|----------|--------|
| Capture route (stock protocol) | `capture_protocol.md` | One-page protocol | Written, untested with a real user |
| Device matrix | `device_matrix.md` | Tier to hardware to accuracy table | Written, accuracy is claimed not measured |
| Three tiers, one contract | `src/tiers/`, `src/models.py` | `report.json` from each tier | Run on synthetic data; photo loses rooms (section 8) |
| Photo tier: per-room folders, stitched plan | `src/tiers/photo.py`, `src/stitching/photo_stitch.py` | Stitched plan from photo folders | Implemented; fails on the synthetic set (1 of 3 rooms) |
| Per-room walls, ceiling, area, openings | `src/geometry/` | Fields in `report.json` | Done, tested on synthetic |
| Stitched multi-room plan with adjacency | `src/stitching/multi_room.py` | `adjacencies` in JSON, plan PNG | Done on LiDAR; 2 links found on the 3-space apartment |
| Drift accountability and ablation | `src/stitching/drift_correction.py`, `tests/test_drift_ablation.py` | On/off footprint table (section 3.6) | Done on synthetic drift; real-capture ablation pending |
| Per-surface damage with class and extent | `src/damage/detection.py`, `surface_projection.py` | `damage_regions` | Implemented; heuristic, many false positives on synthetic photos |
| Concealed-damage flags with rule id | `src/damage/concealed_rules.py` | `concealed_damage_flags` | Five rules implemented |
| Scope line items keyed to surfaces | `src/damage/scope.py` | `scope_line_items` | Done |
| Confidence interval on every measurement | `src/calibration/confidence.py` | `confidence_low/high` | Done; values are assumptions until fitted |
| One command per capture | `run.py` | CLI | Done |
| JSON to the published schema | `schema/output_schema.json`, `tests/test_schema.py` | Schema file | Done |
| Rendered plan | `src/output/renderer.py` | `floor_plan.png` | Done |
| Benchmark gates | `benchmark/evaluate.py` | `benchmark/results/gates_report.md` | Script done; run on synthetic data only |
| Head-to-head vs a consumer app | `benchmark/head_to_head.py` | Comparison table | Script done; competitor data is a mock |
| Benchmark set (multi-room, damage room, all tiers, repeat, ground truth) | `benchmark/captures/`, `benchmark/ground_truth/` | Raw captures and tape measurements | Not started (needs the phone and a tape measure) |
| Fix loop bundle | `fix_loop/` | Declaration, before, after, diff | Not started |
| Technical report (6 pages) | `report.md` | This document | Draft; needs real results, and a 6-page cut |
| Process evidence | git history | Commits per session | 9 commits so far; unpushed |
| README to running in 15 minutes | `README.md` | Install and usage | Written; not timed on a clean machine |
| Mirrors, glass, low light | `capture_protocol.md`, section 8 | Avoid rules and failure modes | Documented only, no handling in code |

## 8. Known failure modes and open problems

1. **Photo tier loses rooms (blocks the stitch gate).** On the synthetic set, room-1 reconstructs (8 of 8 images), but room-2 registers only 4 of 8 images (scale factor 0.029, degenerate fit) and room-3 gets a wrong door width (0.72 m instead of 0.86 m, which skews scale) and a degenerate fit. Only room-1 is stitched. The synthetic renders are random coloured shapes with few features, so this may not reproduce on real photos. Confirm on real captures before changing code.
2. **False windows and damage on SfM tiers.** The synthetic photo and video scenes produce many window gaps and hundreds of damage detections. The thresholds were tuned on nothing real.
3. **Scale rests on an assumed 0.86 m door** (or a 2.2 to 2.5 m vertical prior). A non-standard door shifts the whole room.
4. **Damage detection is heuristic.** `RULE_CRACK_01` fires on almost any visible crack (3 mm is near pixel size). `RULE_MOLD_01` checks only the first and last wall, and no plumbing-fixture detector exists. Damage on the floor or ceiling is dropped, because the ray hits no wall.
5. **Drift correction is translation only.** No rotation drift and no loop closure. It needs shared walls between adjacent rooms.
6. **Mirrors, glass, wet-look surfaces and low light** degrade every tier. They are covered by the capture protocol only. Nothing in the code detects or rejects them.
7. **Repeatability gate formula in `benchmark/evaluate.py` is wrong.** The relative check (0.5% per wall) is computed and then not used; the pass condition reduces to "max difference at most 1 cm". It must be corrected before the gate is trusted.
8. **Interval calibration is an assumption** until fitted on real data.
9. **Video tier is slow** (about 3 minutes for a short clip) because COLMAP runs single-threaded, which is the price of deterministic repeats.

## 9. What remains

1. Capture the benchmark set with an iPhone (multi-room set with three or more rooms plus a connector, a furnished room with two staged damage classes, all three tiers on the same rooms, one room twice at the same tier), with tape or laser ground truth.
2. Run Polycam (or magicplan) on two of the rooms and replace the mock competitor file with the real export, naming the app and version.
3. Run `benchmark/evaluate.py` and `benchmark/head_to_head.py` on the real data. Correct the repeatability formula first.
4. Fit `calibration_data.json` from the real errors.
5. Fill `fix_loop/declaration.md` from the worst real gate, ship the fix, and keep the before and after runs.
6. Run the pipeline cold on a space not used for tuning, for all three tiers (the walk-in test).
7. Cut this report to six pages with real numbers.

## 10. Reproduce

```bash
pip install -r requirements.txt
pytest -q                                   # 49 fast tests
python tests/create_test_ply.py             # synthetic apartment + ground truth
python run.py ./test_data/ --tier lidar
python run.py ./test_data/ --tier lidar --no-drift-correction
python tests/create_test_ply.py --out-dir /tmp/drifted --drift 0.0 0.15
python tests/create_test_photos.py
python run.py ./test_photos/ --tier photo
python run.py ./test_video/ --tier video
python benchmark/evaluate.py --results-dir benchmark/results/ --ground-truth-dir benchmark/ground_truth/
```
