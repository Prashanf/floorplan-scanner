# Capture Protocol (one page, no engineering needed)

Follow this page literally. Pick **one** tier per visit. Phone: iPhone 15 or newer (LiDAR needs a Pro model: 12 Pro or newer). Charge above 50%, free 5 GB of storage, wipe the lens, turn on all room lights, open no curtains onto direct sun.

## Rules for every tier
- **Avoid:** mirrors and glass doors (point the phone away or cover with a sheet), windows with bright sun in the frame, moving people and pets, fast turns, covered lens.
- **Open every interior door** you want in the plan (flat against the wall). Close nothing halfway.
- **Do not** use zoom, portrait mode, filters, night mode or the front camera.
- Keep **walk order**: start at the front door, go room to room in the order you walk, and name things in that order.
- **Per-room routine (LiDAR and video).** In every room, in this order:
  1. **Doorways:** stand in each doorway for **3 seconds** when you enter and again when you leave.
  2. **360° turn:** stand in the **center** and rotate slowly through a full 360° over **10 to 12 seconds**.
  3. **Ceiling tilt:** after the turn, tilt the phone **up at the ceiling** for **3 seconds**.
- **Why:** the video tier needs the ceiling in view to recover scale; skipping these steps can give wrong room sizes. They help but do not guarantee a result: a plain white ceiling gives few points.

## Tier A: LiDAR (iPhone Pro) with "3D Scanner App" (free, by Laan Labs)
1. Install **3D Scanner App** from the App Store. Allow camera access.
2. Tap the red record button, choose **LiDAR** mode, resolution **Medium**, range **5 m**.
3. Hold the phone at **chest height (about 1.4 m)**, screen facing you. Walk **slowly: about 1 step per second**.
4. In each room do the **per-room routine** above. Scan each wall floor to ceiling in one smooth sweep, then the floor. Pass each doorway slowly **twice** (in and out). Finish at the start point so the loop closes.
5. Keep 1 to 3 m from walls. About 2 to 3 minutes per room. Stop when the live mesh covers all walls.
6. Tap **Save**, then **Share, Point Cloud, PLY** (or OBJ). One file for the whole property.
7. Hand-off: AirDrop to the computer. Put the file in a folder named `lidar/`.

## Tier B: Video (any iPhone 15+) with the Camera app
1. Open **Camera, Video**, set **4K at 30 fps** (Settings, Camera, Record Video). Turn the phone **sideways (landscape)**.
2. Hold at **chest height**, elbows in. Start recording **in the doorway of the first room**.
3. Walk **slowly: half a step per second**. Turn the body, not the wrist. Pan left to right along the walls.
4. Walk through every doorway slowly, door frame centered. Walk the perimeter.
5. In each room do the **per-room routine** above.
6. One continuous clip, **60 to 120 seconds per room**, do not stop recording. End where you started.
7. Hand-off: AirDrop the `.mov` (choose **Options, All Photos Data, Most Compatible** if prompted). Put it in a folder named `video/`.

## Tier C: Photos (any iPhone 15+) with the Camera app
1. One **folder per room**, named in walk order: `room-1`, `room-2`, `room-3`, ...
2. **2 to 8 photos per room.** Stand in a corner, face the opposite corner, hold at **chest height (about 1.4 m)**, camera horizontal. Take one photo per corner, so each wall appears in at least two photos with **60% overlap** between neighbours.
3. In every room, take **one photo with a whole door visible**, floor to top of frame (this gives the scale). Include some floor and ceiling in each shot.
4. Hold still, wait for focus, tap to expose on the wall (not the window).
5. Hand-off: AirDrop or USB the photos (HEIC or JPG both work). Put each room's photos in its folder, all under one folder `photos/`.

## Hand-off to the pipeline
Put `lidar/`, `video/` or `photos/` on the computer, then run **one command**:
`python run.py <folder> --tier lidar|video|photo` (see README). Results appear in `output/`.
If unclear, re-shoot slower: slow and overlapping beats fast and short.
