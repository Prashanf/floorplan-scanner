# Device Matrix

Which tier runs on which hardware, and the accuracy each tier **claims** for a typical furnished room (2 to 5 m walls) under the capture protocol. Values are design targets to be replaced by measured benchmark numbers (`benchmark/results/gates_report.md`). Intervals widen as sensor data thins.

| Tier | Minimum device | Sensor used | Wall length | Ceiling height | Opening width |
|------|----------------|-------------|-------------|----------------|---------------|
| LiDAR | iPhone 12 Pro / 13 Pro / 14 Pro / 15 Pro / 16 Pro or newer (LiDAR scanner) | LiDAR depth + camera + IMU via 3D Scanner App (PLY/OBJ export) | ±2 cm or ±0.5% | ±1.5 cm | ±2 cm |
| Video | iPhone 15 or newer (any model) | Wide camera, 4K 30 fps handheld walkthrough, SfM (COLMAP) | ±3% | ±3% | ±5 cm |
| Photo | iPhone 15 or newer (any model) | Wide camera, 2 to 8 stills per room, SfM (COLMAP) with door-width scale prior | ±8% | ±8% | ±8 cm |

Notes
- Photo and video scale is a prior, not a measurement. The code tries, in order: floor-tile grid, a door of 0.86 m, a ceiling-height prior (2.5 m, or `--ceiling-height M` when the height is known), then the longest wall. A non-standard door or a wrong ceiling height biases the whole room.
- Whole-property footprint: LiDAR ±2%, Video ±3%, Photo ±8%.
- Mirrors, glass, wet-look surfaces and low light degrade every tier; see `capture_protocol.md`.
- Non-LiDAR iPhones (base 15 and newer) run the video and photo tiers only.
- **Measured so far, not just claimed.** LiDAR: only on the three provided sample captures (no ground truth, so no accuracy measured). Video: one tape-measured room, walls 35% and 10% short and ceiling 0.61 m low, so the ±3% claim is **not met**. Photo: 0 rooms on our real photos, so the ±8% claim is **not met**. See `technical_report.md` and `compliance_matrix.md`.
- Our own video and photos were captured on a non-LiDAR Android camera phone, not an iPhone 15 or newer; behaviour on the target hardware is untested.
