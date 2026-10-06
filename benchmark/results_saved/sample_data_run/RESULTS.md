# Sample-data run: all three tiers on the provided captures

Run on 2026-10-06 (local Mac, 8 GB RAM, 8 cores, CPU only, no `colmap` binary so `pycolmap` was used, `FLOORPLAN_COLMAP_THREADS=4` except where noted), code at `main` commit `33da191`. Command per run: `python run.py <dir> --tier <tier> --output-dir <out> --verbose`.

Inputs are the three provided raw iPhone LiDAR captures (`data_raw/`: `single_room`, `single_scan_floor_only`, `single_scan_with_ceiling`). The LiDAR tier reads the raw depth logs. The video tier reads each capture's `rgb.mp4`. The photo tier reads stills cut from `rgb.mp4` by `tools/make_photo_sets.py` (the provided data has no real stills). None of the captures has tape or laser ground truth, so **no accuracy is claimed from these runs**; they show what each tier produces.

Each folder holds `report.json` and `floor_plan.png`. Verbose logs are gzipped (`*_log.txt.gz`).

## Results

| Tier | Capture | Rooms | Total area | Adjacencies | Openings | Damage / flags / scope | Time |
|---|---|---|---|---|---|---|---|
| LiDAR | `single_room` | 3 | 8.6 m² | 0 | 0 | 0 / 0 / 0 | 8 s |
| LiDAR | `single_scan_floor_only` | 9 | 32.5 m² | 3 | 2 | 0 / 0 / 0 | 12 s |
| LiDAR | `single_scan_with_ceiling` | 10 | 56.3 m² | 7 | 12 | 0 / 8 / 8 | 17 s |
| Photo | `single_room` | 1 (rough estimate) | 10.8 m² | 0 | 0 | skipped | 919 s (1 thread) |
| Photo | `single_scan_floor_only` | 0 | 0 | 0 | 0 | 0 / 0 / 0 | 182 s |
| Photo | `single_scan_with_ceiling` | 1 | 1.8 m² | 0 | 4 | 7 / 3 / 10 | 175 s |
| Video | `single_room` | 0 | 0 | 0 | 0 | 0 / 0 / 0 | 124 s |
| Video | `single_scan_floor_only` | 0 | 0 | 0 | 0 | 0 / 0 / 0 | 362 s |
| Video | `single_scan_with_ceiling` | 0 | 0 | 0 | 0 | 0 / 0 / 0 | 370 s |

All nine runs wrote a valid report and exited 0.

## What the results show

**LiDAR works on all three captures.** The depth logs are back-projected, rooms segmented, and walls, area and openings fitted, in 8 to 17 s. The numbers match the earlier local and Kaggle runs exactly (3, 9 and 10 rooms; 8.6, 32.5 and 56.3 m²), so the tier is repeatable on this data. Limits: no ground truth; adjacency often rests on shared boundaries because few doors are found (14 openings across 22 rooms); `single_room` is two disconnected scan chunks, not one room; ceilings were scanned in most rooms of `single_scan_with_ceiling` but not in `single_scan_floor_only` (all nine rooms report the highest observed point as a lower bound). Three rooms of `single_scan_with_ceiling` are under 2 m², which looks like over-segmentation.

**Video fails on all three captures.** COLMAP succeeded with the `strict` configuration but registered only 14 to 27 images with 357 to 1,269 points. That is too little to fit a room, so each run produced a valid empty report with the warning "COLMAP reconstruction failed: insufficient feature matches". The scenes are white, glossy and glass-heavy.

**Photo is not usable either.**
- `single_room`: COLMAP failed in all four configurations, so the single-image fallback produced a stand-in room with typical dimensions (3.6 × 3.0 m, intervals ±50%). The 10.8 m² is a prior, not a measurement.
- `single_scan_floor_only`: 0 rooms. in the rooms checked in the log, COLMAP registered 4 to 5 of 8 images with 198 to 280 points.
- `single_scan_with_ceiling`: 1 room of 1.8 m² with a 4.08 m ceiling; the other photo folders gave no geometry. This is implausible and should be read as a failed reconstruction.

**Run issue.** The first photo run on `single_room` was killed by the system (exit 137, probably memory with 4 COLMAP threads on an 8 GB machine). It was re-run with `FLOORPLAN_COLMAP_THREADS=1`, which finished in 919 s.

## Comparison with earlier runs
These results agree with the Kaggle GPU runs in `fix_loop/after/`: LiDAR unchanged, photo and video reconstructing no usable rooms. They do not change the conclusions in `technical_report.md`.

## Reproduce
```bash
# data_raw/ and sample_data_run/ must be set up as in README ("Sample data")
FLOORPLAN_COLMAP_THREADS=4 python run.py sample_data_run/lidar/single_room --tier lidar --output-dir out/lidar_single_room
# likewise for photos/<name> (--tier photo) and video/<name> (--tier video)
```
