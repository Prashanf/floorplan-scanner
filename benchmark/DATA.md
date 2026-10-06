# Benchmark data: what exists and where it lives

Large raw data is kept **outside git** (videos, photos and LiDAR logs total about 3 GB). Reports, plans, logs and ground truth that were produced from it are in the repo under `benchmark/results_saved/`.

| Data | Where | In git? |
|---|---|---|
| Provided LiDAR captures (`single_room`, `single_scan_floor_only`, `single_scan_with_ceiling`): depth, confidence, odometry, intrinsics, IMU, `rgb.mp4` | `data_raw/` in the project root (original zips alongside) | No |
| Photos cut from those captures (committed, small) | `sample_data_run/photos/` | Yes |
| Our room, video: `test-video3-ldscp.mp4` (1080p, 329 s), `test-video3-prtrt.mp4` (1080p portrait, 307 s), earlier `Floorplan_test.mp4`, `floorscanner_test2.mp4` | `benchmark/captures/video/` | No (`*.mp4` is gitignored) |
| Our room, photos: room1 to room4, 5 to 6 JPEGs each | `benchmark/captures/photos/real_capture/` | No |
| Tape measurement of one bedroom (4.06 m × 3.16 m, ceiling 3.15 m, 2026-10-05) | `benchmark/ground_truth/room-1.yaml` | Yes |
| Competitor (scanning app) export | `benchmark/competitor/apartment_polycam.json` | Yes, but a **mock**, not a real export |

**Not hosted.** There is no public download link for the raw files. To reproduce from raw inputs, the author must hand over the raw archive as a separate volume (not yet uploaded). To rerun without it, use the synthetic generators: `python tests/create_test_ply.py` and `python tests/create_test_photos.py`.

## Regenerate the saved results
```bash
# sample LiDAR, photo and video (after setting up data_raw/ and sample_data_run/ as in README "Sample data")
tools/run_sample_data.sh lidar photo video
# our own videos
python run.py <folder holding one video> --tier video --ceiling-height 3.15 --output-dir out/
# our own photos
python run.py benchmark/captures/photos/real_capture --tier photo --output-dir out/
```
Video runs took 8 to 17 minutes on a GPU (Kaggle); photo took 17 minutes on CPU for four rooms.

## Not collected
No staged-damage room, no LiDAR capture of our own rooms, no repeat capture of one room at one tier, no real competitor export. See `compliance_matrix.md`.
