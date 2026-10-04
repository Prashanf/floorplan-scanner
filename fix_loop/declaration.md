# Fix Loop Declaration

Note on timing: the case study asks for this page before the fix is shipped. This one was written after the fix was committed and after the Kaggle runs, so the "predicted" outcome is what the fix was designed to deliver, not a number recorded in advance. Everything marked (measured) comes from the Kaggle notebook runs saved in `fix_loop/before/` and `fix_loop/after/`.

## 1. Worst gate

**Photo-tier whole-property stitch: FAIL**, and the video tier is in the same state.

Before the fix, on the three provided sample captures (measured, `before/digest.txt`):
- All 9 photo and video runs ended with an error and wrote no report: the 2 `single_room` photo runs (with and without drift correction) with `RuntimeError: no room could be reconstructed; check photo overlap and lighting`, the other 7 with `RuntimeError: geometry failed for every room`.
- Result: 0 of 3 captures stitched at the photo tier, 0 of 3 at the video tier (gate: a stitched plan with correct adjacency, footprint within ±8% for photos, ±3% for video).

The photo sets were cut from each capture's `rgb.mp4` by `tools/make_photo_sets.py` (the provided data contains no stills): up to 8 frames per room, 1440×1920 portrait, 120 images in total.

## 2. Root-cause hypothesis

The sparse structure-from-motion reconstructions are too small to contain a room. The rooms are white, glossy, glass-heavy bathrooms and small rooms, and the sampled views are close-ups of fixtures and surfaces, so the images share few matchable features.

Evidence (measured, `after/` logs, unless stated):
- Photo tier: per room, COLMAP registered only 2 to 6 of 8 images with 14 to 368 points (single_scan_floor_only: 4, 6, 3, 4 and 4 images; single_scan_with_ceiling: 2, 3, 2, 5, 2, 2 and 2 images). Room geometry could not be fitted on any of them: `neither wall lines nor a room outline found`.
- Video tier: with the logged intrinsics as the focal length prior, COLMAP registered only 24 of 143, 32 of 400 and 11 of 400 keyframes, with 411, 1371 and 269 points.
- Two photo rooms (single_scan_floor_only room-6 and room-7) failed in all four COLMAP configurations. The single-image check found no vanishing points in their best photo (23 and 9 line segments).
- Earlier local experiment (not in the saved runs): spacing the photos 0.5 s to 1.5 s apart in time instead of spreading them over the room made no difference.
- Not verified: per-image SIFT feature counts. Portrait orientation is unlikely to be the cause, since SIFT matching does not depend on orientation; the preprocessing rotated all 56 photos of each multi-room capture to landscape (`rotated_images` 56 in the digest) without changing the outcome.

## 3. Fix and prediction

Fix shipped (commits `90fb7f5`, `57c22eb`):
1. COLMAP auto-tuning cascade, `COLMAP_CONFIGS` in `src/tiers/colmap_utils.py`: strict, relaxed, aggressive, desperate, tried in order by `run_colmap_with_fallback()`; up to 32768 features; SIMPLE_RADIAL, SIMPLE_PINHOLE and OPENCV camera models; per-image cameras; match thresholds down to 2.
2. Image preparation before every run (`prepare_images_for_colmap`): EXIF orientation, portrait to landscape (camera poses mapped back afterwards), size cap 3200 px, normalized file names.
3. Graceful degradation (`src/pipeline.py`): when no room can be reconstructed the pipeline writes a valid report with `room_count` 0, a `warnings` list and a blank plan, and exits 0.
4. Single-image fallback (`src/tiers/single_image.py`): a flagged ±50% rough room, only if the best photo passes a vanishing-point box test.
5. Video: motion-based keyframes, then every 10th, 5th and 3rd frame.

Prediction:
- The gate stays **FAIL** on this data: the cascade cannot create image overlap that the capture does not contain.
- What changes is the failure mode: from a crash with no output to a valid, schema-compliant report with 0 rooms and diagnostic warnings, exit code 0.
- On captures that follow the capture protocol (overlapping views, textured surfaces) the photo tier should reconstruct rooms. This is **not demonstrated**: on the synthetic test rooms it reconstructed 1 of 3 rooms, and no protocol-following real photo set has been run.

## Result

- Before run: Kaggle, commit `d65568f`, `fix_loop/before/` (digest, validation log, all photo and video logs).
- After run: Kaggle, commit `57c22eb`, CUDA build of COLMAP (`COLMAP device: GPU` in the logs), `fix_loop/after/` (digest, validation log, logs, and `report.json` plus `floor_plan.png` for every photo and video run). The 98 unit tests passed in that environment. LiDAR results were unchanged (3, 9 and 10 rooms; 8.6, 32.4 and 56.2 m²).
- Actual numbers (measured):

| | Before | After |
|---|---|---|
| Photo and video runs that wrote a valid report | 0 of 9 | 9 of 9 |
| Runs that ended with an error | 9 of 9 | 0 of 9 |
| Rooms reconstructed (photo and video) | 0 | 0 |
| Gate | FAIL | FAIL |

  After the fix every photo and video run has `room_count` 0 plus warnings and a blank plan reading "Reconstruction failed — insufficient data". Run times in the saved logs: photo 7 to 89 s, video 55 to 211 s.
- Prediction vs. actual: the failure-mode prediction held (crash to a valid empty report, 9 of 9). The reconstruction prediction cannot be judged until a protocol-following photo set exists.
- Shortcomings of the fix found in the after run (not fixed yet):
  1. The warning text "COLMAP reconstruction failed: insufficient feature matches" is also written when COLMAP did produce a model but it was too small for geometry (all video runs, most photo rooms). It is imprecise in those cases.
  2. The single-image fallback only runs when COLMAP produces no model. For rooms where COLMAP "succeeded" with 2 to 6 images and geometry then failed, the fallback never ran, so those rooms are simply missing. The photo `single_room` run earlier gave a flagged rough room locally on the CPU, but not on Kaggle, where COLMAP found a 2-image, 26-point model instead.
  3. The video keyframe cascade also only triggers when COLMAP fails, not when it registers a small share of the keyframes (24 of 143, 32 of 400, 11 of 400), so the denser keyframe sets were never tried.
- Diff: `git diff d65568f 57c22eb`.
- To regenerate: run `notebooks/floorplan_scanner_kaggle.ipynb` on Kaggle at each commit (or `tools/run_sample_data.sh photo video` locally).
