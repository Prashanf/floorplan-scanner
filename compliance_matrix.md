# Compliance matrix

Requirement (from `Applied_AI_Case_Study.pdf`) → file path → artifact → status. Status values: **Done**, **Partial**, **Not done**. "Real data" means a capture we made or were given, as opposed to the synthetic apartment.

## Part 1: capture and tiers

| Requirement | File path | Artifact | Status |
|---|---|---|---|
| Capture route (stock-capture protocol, one page for a non-engineer) | `capture_protocol.md` | One-page protocol per tier, with the 360° turn, ceiling tilt and doorway pause | Done. Never followed by a third party. |
| Device matrix: tier, hardware, honest accuracy | `device_matrix.md` | Table of claimed accuracy | Partial. Claims, not measurements; measured results noted at the bottom of the file. |
| Tier 1: photos, 2 to 8 per room, per-room folders, stitched plan | `src/tiers/photo.py`, `src/stitching/photo_stitch.py` | `benchmark/results_saved/photos_real_capture/` | Partial. Runs; produced 0 rooms on our real photos. |
| Tier 2: video walkthrough | `src/tiers/video.py` | `benchmark/results_saved/video_test3/` | Partial. Partial plan only (11.5 m² against 12.8 m² measured). |
| Tier 3: LiDAR depth, poses, intrinsics | `src/tiers/lidar.py`, `depth_stream.py` | `benchmark/results_saved/lidar_segmentation_v2/` | Done on the provided samples only; not run on our own rooms (no LiDAR device). |

## Part 2: output contract and gates

| Requirement | File path | Artifact | Status |
|---|---|---|---|
| Per-room walls, ceiling height, floor area, openings | `src/geometry/` | `report.json` | Done |
| Stitched multi-room plan with adjacency | `src/stitching/` | `floor_plan.png` | Partial. Adjacency often from shared boundary; counts unverified. |
| Per-surface damage regions with class and metric extent | `src/damage/` | `damage_regions` in `report.json` | Partial. Heuristic, no labelled evaluation. |
| Concealed-damage flags with the rule that fired | `src/damage/concealed_rules.py` | `concealed_damage_flags` | Done (5 rules); no real damaged-room test |
| Scope line items keyed to surfaces | `src/damage/scope.py` | `scope_line_items` | Done |
| Confidence interval on every measurement | `src/calibration/confidence.py` | `{value, confidence_low, confidence_high, unit}` | Partial. Fixed multipliers, not fitted; a real video result fell outside its interval. |
| One command per capture, JSON to published schema, rendered plan | `run.py`, `schema/output_schema.json`, `src/output/` | `report.json`, `floor_plan.png` | Done |
| Gate: opening widths ≤2 cm on ≥85% | `benchmark/evaluate.py` | `benchmark/results/gates_report.md` | Not measured on real data (synthetic only: pass) |
| Gate: ceiling ≤1.5 cm per room; repeat spread ≤1 cm | `benchmark/evaluate.py` | video: 0.61 m low on room-1 | Fail on real data; no repeat capture |
| Gate: repeatability ≤1 cm or 0.5% per wall | `benchmark/evaluate.py` | none | Not measured; the 0.5% rule is not applied in the script (bug) |
| Gate: drift accountability, with ablation | `src/stitching/drift_correction.py`, `tests/test_drift_ablation.py` | `technical_report.md` §3 | Partial. Ablation on synthetic drift only. |
| Gate: photo-tier whole-property stitch within ±8% | `src/stitching/photo_stitch.py` | none on real data | Fail (0 rooms on our photos) |
| Video walls ±3%, photo walls ±8%, calibration scored per tier | `benchmark/evaluate.py` | `technical_report.md` §4 | Fail / not fitted (room-1 walls −35% and −10%) |

## Benchmark set

| Requirement | Artifact | Status |
|---|---|---|
| One multi-room capture, 3+ rooms plus a connector | Our own four photo folders; video walkthrough | Partial. No tape ground truth for the multi-room capture. |
| One furnished room with staged damage in two classes | none | Not done |
| Same rooms at all three tiers | video and photo of our room; LiDAR only on provided samples | Not done (no LiDAR device) |
| At least one room captured twice at the same tier | none | Not done |
| Laser or tape ground truth on everything | `benchmark/ground_truth/room-1.yaml` (one bedroom) | Partial |
| Raw data and test data, with measurements, submitted | `benchmark/DATA.md`; our test data (videos and photos we recorded): https://drive.google.com/drive/folders/1Cb9Y-XzCeFcEkPu4zT8gaxw45lG32aVz?usp=drive_link | Partial. Test data (our videos and photos) is on Drive; the raw data is the provided LiDAR samples only. |

## Part 3 to 5

| Requirement | File path | Artifact | Status |
|---|---|---|---|
| Head-to-head vs a consumer scanning app on 2 rooms, beat or tie on ≥70% | `benchmark/head_to_head.py`, `benchmark/competitor/apartment_polycam.json` | mock file only | Not done (needs a LiDAR export of our own room) |
| Fix declaration: worst gate, root cause, fix, predicted number | `fix_loop/declaration.md` | declaration | Done, but written after the fix |
| Before and after runs, regenerable, with a diff | `fix_loop/before/`, `fix_loop/after/`, `git diff d65568f 57c22eb` | Kaggle runs | Done. The gate stayed FAIL. |
| Process evidence: commit as you work | git history | 37 commits over 4 to 6 October | Done |

## Deliverables

| # | Deliverable | File path | Status |
|---|---|---|---|
| 1 | Compliance matrix | this file | Done |
| 2 | Capture route and device matrix | `capture_protocol.md`, `device_matrix.md` | Done |
| 3 | Repo, README to running in under 15 min, one command per capture | `README.md` | Partial. Not timed on a clean machine. |
| 4 | Reproduction bundle (regenerate every number from raw inputs) | `benchmark/DATA.md`, `tools/` | Partial. Test data (our videos and photos) is on the shared Drive folder linked in `benchmark/DATA.md`; the raw data is the provided samples. Not rerun from a clean download. |
| 5 | Benchmark report: gates at three tiers, repeatability table, head-to-head table, timing | `benchmark/results/gates_report.md`, `technical_report.md` | Partial. Repeatability and head-to-head tables are empty; gates are synthetic. |
| 6 | Fix loop bundle | `fix_loop/` | Done |
| 7 | Technical report, max 6 pages | `technical_report.md` (5 pages) | Done |
| 8 | Raw benchmark data (sensor logs, ground truth, app exports): raw data plus our test data | `benchmark/DATA.md` | Partial. Test data (our videos and photos) is on the shared Drive folder (linked in `benchmark/DATA.md`); the app export is a mock. |

## Constraints
- Handheld consumer capture only: yes. Our own captures came from a non-iPhone camera phone; stated in the report.
- Pretrained models and datasets disclosed: MobileSAM and OWLv2 are optional (`requirements-damage-model.txt`, `README.md`). COLMAP (SIFT) is used for video and photo.
- Everything runs without calling our infrastructure: yes.
- Weights fetched by script: `tools/download_mobile_sam.py`.
- Mirrors, glass, wet-look surfaces, low light "covered in the submission": listed as a failure mode and avoided in the protocol; no dedicated test. **Partial.**
- Walk-in test: LiDAR is the tier most likely to work on a new space; video takes 8 to 17 minutes and gives partial plans; photo failed on our real photos.
