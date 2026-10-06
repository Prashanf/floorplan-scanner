# LiDAR room segmentation: cleanup of over-split rooms (branch `lidar-segmentation`)

LiDAR tier only, on the three provided captures (`data_raw/`). Video and photo were not run and are unchanged. No ground truth exists for these captures, so a lower room count is not proof of a better plan; it is checked by looking at the plans.

## Change
- `src/stitching/multi_room.py`: the existing fragment merge (`merge_oversegmented_rooms`: two rooms sharing at least 80% of the shorter room's wall and no opening are one room) now also runs for LiDAR.
- `src/stitching/room_merge.py`: new `merge_small_rooms`: a room under `LIDAR_MIN_ROOM_AREA` (2.0 m²) is merged into the neighbour it shares the longest boundary with (at least 0.6 m).
- `src/tiers/room_segmentation.py` takes `persistence` and `min_room_radius` arguments; LiDAR passes `LIDAR_ROOM_PERSISTENCE` and `LIDAR_MIN_ROOM_RADIUS` (`src/config.py`), set equal to the old values (0.1 m, 0.3 m). The video tier is untouched.

## Threshold sweep (rooms, total m² per capture: single_room | floor_only | with_ceiling)
Merging was on in every row.

| persistence | min radius | single_room | floor_only | with_ceiling |
|---|---|---|---|---|
| 0.1 | 0.3 | 3, 8.6 | 7, 32.4 | 6, 56.9 |
| 0.1 | 0.4 | 1, 21.6 | 5, 23.0 | 5, 47.7 |
| 0.1 | 0.5 | 1, 21.6 | 1, 72.4 | 4, 42.1 |
| 0.2 | 0.3 | 3, 8.6 | 7, 32.4 | 6, 57.9 |
| 0.2 | 0.4 | 1, 21.6 | 5, 23.0 | 5, 47.8 |
| 0.2 | 0.5 | 1, 21.6 | 1, 72.4 | 4, 42.1 |
| 0.3 | 0.3 | 3, 8.6 | 8, 33.5 | 6, 57.9 |

- Persistence 0.1 and 0.2 give identical results; it has little effect here.
- A larger minimum radius (0.4 or more) is harmful: it removes real rooms and distorts the area (`single_room` jumps from 8.6 to 21.6 m², `floor_only` to 23.0 or 72.4 m²). It was rejected. The first attempt used 0.5 and collapsed `floor_only` to one 72.4 m² room.
- The merge steps alone, with the old thresholds, consolidate the fragments without changing the total area.

## Result (before = `sample_data_run/`, after = this folder)

| Capture | Before | After |
|---|---|---|
| single_room | 3 rooms, 8.6 m², areas 2.3, 2.4, 4.0 | 3 rooms, 8.6 m² (unchanged: the pieces do not touch) |
| single_scan_floor_only | 9 rooms, 32.5 m², 3 adjacencies, 2 openings | 7 rooms, 32.4 m², 1 adjacency, 2 openings; no room under 2.4 m² |
| single_scan_with_ceiling | 10 rooms, 56.3 m², 7 adjacencies, 12 openings | 6 rooms, 56.9 m², 3 adjacencies, 7 openings; areas 3.8, 3.9, 7.4, 12.0, 12.3, 17.6 |

The `with_ceiling` plan no longer has the three 1.6 to 1.9 m² fragments and looks like a plausible flat. Costs: the adjacency and opening counts fell because merged rooms lose the interface between them (7 to 3 adjacencies, 12 to 7 openings in `with_ceiling`), and one merged room is 17.6 m², which may now combine a hallway with a room. Without a labelled room count we cannot say whether 6 is right; watching `rgb.mp4` or counting rooms in the scan would settle it.

Tests: 174 pass (two tests that asserted "LiDAR is never merged" were rewritten for the new behaviour, and one new test covers the small-room rule). Logs are gzipped.
