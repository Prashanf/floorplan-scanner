# Floor Plan Scanner: Technical Report

Branch `dev`, pipeline version 0.1.0. Every number below comes from a saved run in this repository (`output/`, `benchmark/results_saved/`, `fix_loop/`). Where a number is a design target rather than a measurement, the text says so.

**Summary.** The LiDAR tier produces recognisable multi-room plans from real iPhone depth logs. The video tier reconstructs with COLMAP but the plans are wrong in scale and room count. The photo tier produced no rooms on the one real photo set we ran. No gate has been measured against tape or laser ground truth on a full benchmark set. The only gate numbers that pass are on synthetic data. The head-to-head comparison uses a mock competitor file.

## 1. Architecture

Three front-ends feed one shared room representation (`RoomIR`); everything downstream runs once, whatever the tier.

- **LiDAR** (`src/tiers/depth_stream.py`, `lidar.py`): 16-bit depth PNGs, confidence maps, ARKit odometry and intrinsics are back-projected into a point cloud. PLY/OBJ exports also load. The cloud is voxel-downsampled and rotated so walls are axis-aligned. Rooms are separated by DBSCAN on a 5 cm grid sliced at wall height.
- **Video** (`src/tiers/video.py`): motion-based keyframes (cap 400), then COLMAP structure-from-motion, a scale step, and the same room segmentation as LiDAR.
- **Photo** (`src/tiers/photo.py`): one COLMAP reconstruction per room folder, scale step, then door-matched placement of the rooms in walk order (`src/stitching/photo_stitch.py`).

Shared stages: wall fitting (RANSAC lines, 90° snap, with an occupancy-grid outline for partial scans) → ceiling height (Z histogram) → openings (gaps in the wall point band) → stitching and drift correction → damage detection (OpenCV heuristics by default) → five concealed-damage rules → scope line items → confidence calibration → JSON to `schema/output_schema.json` and a rendered plan. Every measurement is `{value, confidence_low, confidence_high, unit}`; empty lists are present, never missing. When a tier yields no usable room, the pipeline writes a valid report with `room_count` 0, a `warnings` list and a blank plan, and exits 0.

**Capture route 2** (stock apps, no custom app): `capture_protocol.md`, one page. LiDAR uses 3D Scanner App; video and photos use the native Camera app. The protocol covers phone height, walking speed (half a step per second), a slow pass through each doorway, a glance at floor and ceiling in each corner, and what to avoid (mirrors, glass doors, bright windows, fast turns). It has not changed since the first commit. It does not ask for a slow 360° rotation or a deliberate ceiling sweep in the video route. Section 5 shows why that matters.

## 2. Tier design and device matrix

From `device_matrix.md`. The accuracy columns are claims, not measurements.

| Tier | Minimum device | Sensor | Wall length | Ceiling | Opening width |
|---|---|---|---|---|---|
| LiDAR | iPhone 12 Pro or newer (Pro class) | LiDAR depth + camera + pose | ±2 cm or ±0.5% | ±1.5 cm | ±2 cm |
| Video | iPhone 15 or newer | Wide camera, handheld walkthrough, COLMAP | ±3% | ±3% | ±5 cm |
| Photo | iPhone 15 or newer | 2 to 8 stills per room, COLMAP | ±8% | ±8% | ±8 cm |

The matrix text still says photo and video scale comes from a 0.86 m door prior. The code now tries, in order: floor-tile grid, door, ceiling-height prior, longest wall, each with a confidence, and `--ceiling-height M` overrides all of them. The matrix has not been updated.

**What each tier delivered on real data**

| Run | Result | Time |
|---|---|---|
| LiDAR `single_scan_with_ceiling` | 10 rooms, 56.3 m², 7 adjacencies, 12 openings | 14.6 s |
| LiDAR `single_scan_floor_only` | 9 rooms, 32.5 m², 3 adjacencies, 2 openings | 9.3 s |
| LiDAR `single_room` | 3 rooms, 8.6 m², 0 adjacencies, 0 openings | 4.5 s |
| Video, landscape (`test-video3-ldscp.mp4`, `--ceiling-height 3.15`) | 2 rooms, 11.5 m² | 504 s (COLMAP on GPU) |
| Video, portrait (`test-video3-prtrt.mp4`, `--ceiling-height 3.15`) | 7 rooms, 25.4 m², 3 adjacencies | 995 s |
| Photo, real capture (room1 to room4) | 0 rooms | 1046 s |
| Photo, three sample captures (Kaggle, photos cut from video) | 0 rooms in all three | 7 to 89 s |

LiDAR is the strongest tier, but no tape ground truth exists for the three sample captures, so no accuracy figure is claimed for it. The plans look like apartments. "Correct adjacency" is unverified: adjacency falls back to shared boundary contact because few doors are detected (14 openings across 22 rooms). Room segmentation also over-splits hallways into fragments under 2 m². `single_room` is not one room: the scan has two disconnected chunks about 5 m apart.

**Video COLMAP detail.** Landscape: 9,884 frames to 400 keyframes, 126 registered (32%), 27,239 points. Portrait: 9,211 frames to 400 keyframes, 301 registered (75%), 76,053 points. Both succeeded with the first (`strict`) configuration, so the cascade (strict → relaxed → aggressive → desperate) was not exercised. Front-end time dominates: 499 s of 504 s (landscape) and 978 s of 995 s (portrait). A cold walk-in run on video would take 8 to 17 minutes, even with a GPU.

**Photo detail.** The real photo set (four rooms) failed in all four rooms: "neither wall lines nor a room outline found". On the earlier Kaggle photo sets, COLMAP registered 2 to 6 of 8 images with 14 to 368 points per room. The rooms are white, glossy and glass-heavy, so there are few matchable features. The door-matching stitcher has never run on a real reconstruction. It works on the synthetic rooms only, where one of three rooms reconstructs.

## 3. Drift handling

`src/stitching/drift_correction.py`: translation-only correction with `scipy.optimize.minimize` (L-BFGS-B).

- **Shared walls**: two walls from different rooms count as shared when parallel within 10°, midpoints within 0.3 m, lengths within 20%.
- **Cost**: for each shared pair, offset along the wall squared plus the part of the across-wall offset beyond 0.15 m (one wall thickness), squared, plus 0.05 times the sum of squared room offsets. The wall length is used to select pairs, not in the cost. Each room offset is bounded to ±0.5 m. The first room is fixed.
- **Ablation**: `--no-drift-correction` exists. `tests/test_drift_ablation.py` injects 0.15 m of drift per space along y into the synthetic apartment. Room origin error is 0.00 / 0.15 / 0.30 m with correction off and 0.00 / 0.001 / 0.00 m with it on. Clean input is left untouched (error under 0.05 m).
- **Limits**: this is proven on synthetic drift only. An earlier version rejected every parallel wall pair, so correction never ran and the flag changed nothing; that bug was found and fixed on 4 Oct (`92a124f`). On the real LiDAR scans the rooms already share one frame, and the effect was not quantified. Rotation drift is not corrected. The photo tier has no shared frame, so it relies on door matching instead.

## 4. Error budget and calibration

Intervals are `half-width = base uncertainty × tier multiplier` (`src/config.py`): base 1 cm for walls and ceiling, 2 cm for openings, 0.1 m² for floor area, 5 cm for damage extent. Multipliers: LiDAR 1.5, video 3.0, photo 6.0. A wall therefore gets ±1.5 cm (LiDAR), ±3 cm (video) or ±6 cm (photo). These are fixed design targets, not fitted. The code reads `calibration_data.json` if present; no such file exists, because we have no measured ground truth to fit it on.

**Dominant error sources**

- LiDAR: sensor noise near 1 cm, unscanned wall stretches, room-boundary choices, missing ceilings (most rooms report the highest observed point as a lower bound, and the interval opens upward by up to 1 m).
- Video: monocular scale, sparse clouds on textureless walls, an unscanned ceiling, room over-segmentation.
- Photo: the video errors, plus no shared frame between rooms.

**Measured calibration failure.** The author tape-measured one room at 4.06 m × 3.16 m (12.8 m²) with a 3.15 m ceiling. The raw tape notes are not in the repository, and which room in the plan corresponds to it was not recorded. The landscape video was run with `--ceiling-height 3.15`. Its first room is 2.63 m × 2.85 m, 7.5 m². Against the tape that is:

- area −41%, with the two walls −35% (2.63 vs 4.06) and −10% (2.85 vs 3.16);
- ceiling reported 2.54 m against 3.15 m, an error of −0.61 m, because the ceiling was not scanned and the pipeline reports the highest observed point.

The two wall errors differ, so this is not a uniform scale error: part of the room was not reconstructed. The plan's second room (4.01 m²) may be the rest of the same room; the total of 11.5 m² would then be −10%. Either way the output is outside its stated interval: the total area is 11.51 m² with an interval of 11.21 to 11.81 m² (±0.3 m²), against 12.8 m² measured. That is a confident interval around a wrong number, which the case study penalises. The ceiling gate (1.5 cm) fails by 0.61 m even though the user supplied the true height.

**What we cannot claim.** No repeated capture exists, so the repeatability gate is unmeasured, and `benchmark/evaluate.py` has a bug: the 0.5% part of the rule is computed and never used. The gate table in `benchmark/results/gates_report.md` passes (openings 4 of 4, ceiling error 0.1 cm, wall error 0.08 cm maximum), but it is computed on the synthetic apartment where the pipeline's own generator supplies the ground truth. It says nothing about real accuracy. The head-to-head file `benchmark/competitor/apartment_polycam.json` is a mock, not a real export. The 70% head-to-head gate is unmeasured.

## 5. Fix loop

### 5.1 The declared loop (`fix_loop/declaration.md`)

**Gate:** photo and video whole-property stitch. **Before** (Kaggle, commit `d65568f`): all 9 photo and video runs on the three sample captures ended in a `RuntimeError` with no report. **Hypothesis:** the sparse COLMAP models are too small to contain a room (white, glossy, glass-heavy rooms, close-up views). **Fix** (commits `90fb7f5`, `57c22eb`): a four-configuration COLMAP cascade, image preparation (EXIF, portrait to landscape, size cap), graceful degradation to a valid empty report, a single-image fallback with ±50% intervals, and denser video keyframe retries. **Prediction:** the gate stays FAIL; only the failure mode changes.

| | Before | After |
|---|---|---|
| Photo and video runs with a valid report | 0 of 9 | 9 of 9 |
| Runs ending in an error | 9 of 9 | 0 of 9 |
| Rooms reconstructed | 0 | 0 |
| Gate | FAIL | FAIL |

The prediction held, and it is a weak one: the fix repaired crashing, not reconstruction. The declaration was written after the fix, which the file states. Three shortcomings found in the after run are still open: the single-image fallback runs only when COLMAP produces no model; the keyframe cascade triggers only on COLMAP failure, not on a low share of registered keyframes; and the "insufficient feature matches" warning is also written when COLMAP did produce a small model.

### 5.2 Follow-up on the real video

This part was not predicted in advance and is not a controlled comparison.

- **Before:** the first Samsung floor-tour video (`Floorplan_test.mp4`, 500 s). 3 rooms of 1.00, 1.81 and 1.04 m² (3.85 m² total), 32 damage detections, 13 concealed flags.
- **After:** the re-recorded landscape video with `--ceiling-height 3.15`. 2 rooms of 7.50 and 4.01 m² (11.51 m² total).
- Outputs: `benchmark/results_saved/video_before_fixes/` and `benchmark/results_saved/video_test3/landscape/`.

The two runs use different recordings, and the second also uses the user-supplied ceiling height. We cannot attribute the improvement to a code change, to the re-recording, or to the flag. The "3× to 40% error" comparison is not valid: the first video has no tape measurement, so its error is unknown. The second error is computed above and is largely a missing-geometry error, not a scale error. The earlier run of `Floorplan_test.mp4` with the flag also gave the same small rooms, which suggests room segmentation and the unscanned ceiling limit the result more than the scale step does. The ceiling was not scanned in either recording (warnings: "the ceiling was not scanned" for every room), so the capture still lacks what the vertical-extent method needs. Of the five fixes listed in 5.1, none addressed this.

**Assessment.** The loop moved the failure from a crash to an empty report, and the video output went from fragments to a partial plan. No gate moved from fail to pass. The remaining problem is monocular scale and coverage. A metric depth model (for example Depth Anything V2 or DepthPro) could remove scale ambiguity, but we have not tried one; it is a proposal, not a result.

## 6. Known failure modes

1. **COLMAP on low-texture interiors.** White painted walls and glossy floors give too few SIFT matches. This ended photo-tier reconstruction on all real data. A tiny model (2 to 3 images, over 10 points) is accepted by the cascade, so stronger configurations may never run; the earlier retry ladder demanded 3 images and 30 points. Not fixed.
2. **Scale ambiguity.** Monocular SfM has no absolute scale. Tile, door and ceiling-prior scale are workarounds. Floor-tile scale has never been validated on a real tiled floor. A non-standard door or a wrong ceiling prior biases the whole room.
3. **Unscanned ceilings.** In both new video runs and most LiDAR rooms the ceiling was not observed, so ceiling height is a lower bound. This breaks the ceiling gate in principle, not just by tuning.
4. **Room over-segmentation.** The portrait video gave 7 rooms with walls as short as 0.29 m. A merge heuristic (`room_merge.py`) and a rectangle fallback exist, with synthetic tests only.
5. **Damage false positives.** OpenCV heuristics fire on grout, furniture edges and shadows. The landscape run had 129 detections in one room (capped at the 10 strongest), and the report lists 24 regions, 16 concealed flags and 40 scope items; the portrait run lists 67, 31 and 98. We have no labelled images, so no precision or recall exists, and most of these are probably false. The MobileSAM + OWL option needs a GPU and a threshold chosen on labelled data. `RULE_CRACK_01` fires on almost any visible crack; `RULE_MOLD_01` checks only the first and last wall.
6. **Mirrors, glass, low light.** The sample frames contain mirrors, glass and a person reflected in one. The protocol tells users to avoid them. Their effect on the output was not measured.
7. **Run time.** Video takes 8 to 17 minutes even with a GPU, and the photo run took 17 minutes to produce no rooms.

## 7. What is still missing

A benchmark set with tape or laser ground truth (the case study requires a multi-room capture, a damaged furnished room, all three tiers on the same rooms, and one repeated room); a real competitor export; a calibration fit on measured data; a corrected repeatability gate; an updated device matrix; and a protocol revision that asks for a ceiling sweep. Until these exist, the accuracy claims in section 2 remain unverified.
