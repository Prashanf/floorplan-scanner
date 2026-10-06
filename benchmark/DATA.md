# Benchmark data: what exists and where it lives

Two kinds of data are used, and they are kept apart:

- **Raw data:** the three sample captures provided with the task (raw iPhone LiDAR logs). They are in `data_raw/` in the project root and are not in git (about 900 MB).
- **Test data:** the videos and photos we recorded ourselves of one floor, for our own test cases. They are on Google Drive: **https://drive.google.com/drive/folders/1Cb9Y-XzCeFcEkPu4zT8gaxw45lG32aVz?usp=drive_link**
  Download them into `benchmark/captures/video/` and `benchmark/captures/photos/real_capture/` to rerun the video and photo results. They are not in git (`*.mp4` and `*.jpg` are gitignored).

Reports, plans, logs and ground truth produced from both are in the repo under `benchmark/results_saved/`.

| Data | Kind | Where | In git? |
|---|---|---|---|
| Provided LiDAR captures (`single_room`, `single_scan_floor_only`, `single_scan_with_ceiling`): depth, confidence, odometry, intrinsics, IMU, `rgb.mp4` | raw data | `data_raw/` (original zips alongside) | No |
| Photos cut from those captures | raw data (derived) | `sample_data_run/photos/` | Yes (small) |
| Our room, video: `test-video3-ldscp.mp4` (1080p, 329 s), `test-video3-prtrt.mp4` (1080p portrait, 307 s), earlier `Floorplan_test.mp4`, `floorscanner_test2.mp4` | test data | Google Drive (link above); `benchmark/captures/video/` locally | No |
| Our room, photos: room1 to room4, 5 to 6 JPEGs each | test data | Google Drive (link above); `benchmark/captures/photos/real_capture/` locally | No |
| Tape measurement of one bedroom (4.06 m × 3.16 m, ceiling 3.15 m, 2026-10-05) | test ground truth | `benchmark/ground_truth/room-1.yaml` | Yes |
| Competitor (scanning app) export | none | `benchmark/competitor/apartment_polycam.json` | Yes, but a **mock**, not a real export |

To rerun without either data set, use the synthetic generators: `python tests/create_test_ply.py` and `python tests/create_test_photos.py`.

## Regenerate the saved results
```bash
# raw data: the provided samples (set up data_raw/ and sample_data_run/ as in README "Sample data")
tools/run_sample_data.sh lidar photo video
# test data: our own videos (one folder per video) and photos
python run.py <folder holding one video> --tier video --ceiling-height 3.15 --output-dir out/
python run.py benchmark/captures/photos/real_capture --tier photo --output-dir out/
```
Video runs took 8 to 17 minutes on a GPU (Kaggle); the photo run took 17 minutes on CPU for four rooms.

## Not collected
No staged-damage room, no LiDAR capture of our own rooms, no repeat capture of one room at one tier, no real competitor export. See `compliance_matrix.md`.
