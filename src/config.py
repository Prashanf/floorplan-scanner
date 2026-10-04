"""Pipeline configuration: every tunable threshold in one file.

All parameters are meant to work for any residential property. They are not fitted to one
capture; the modules import their constants from here, so tuning is a one-file change.
Units are metres unless a name says otherwise.
"""

# --- Wall fitting (src/geometry/wall_fitting.py, room_outline.py) -----------------------------
RANSAC_INLIER_THRESHOLD = 0.03  # a point within this of a line supports it (LiDAR)
RANSAC_MIN_INLIERS = 20  # minimum points to accept a wall line
RANSAC_ITERATIONS = 400
MAX_WALL_LINES = 16
WALL_HEIGHT_BAND = (0.8, 1.5)  # above the floor; clear of floor clutter and the ceiling edge
SNAP_ANGLE_TOLERANCE_DEG = 15  # lines within this of a multiple of 90 degrees snap onto the grid
WALL_MERGE_OFFSET = 0.05  # parallel lines closer than this are one wall
MIN_WALL_LENGTH = 0.3
MIN_WALL_COVERAGE = 0.4  # share of a wall's extent that must hold points
MAX_LINE_WALLS = 8  # more fragments than this: the room is not a clean box, use the outline
MAX_AREA_DISAGREEMENT = 0.2  # line polygon vs occupancy outline area before the outline wins
OUTLINE_CELL = 0.05  # occupancy grid resolution for the room outline
OUTLINE_CLOSE_RADIUS = 0.30  # holes smaller than this (furniture, unscanned patches) are filled
OUTLINE_SIMPLIFY = 0.12  # Douglas-Peucker tolerance on the outline
OUTLINE_MIN_EDGE = 0.20  # shorter outline edges are merged into their neighbours

# --- Opening detection (src/geometry/openings.py) ---------------------------------------------
MIN_OPENING_WIDTH = 0.4  # gaps narrower than this are not openings
MIN_OPENING_HEIGHT = 0.4  # a column is open if it has an empty vertical run this tall
DOOR_MAX_HEIGHT_START = 0.2  # a gap starting within this of the floor is a door
WINDOW_MIN_HEIGHT_START = 0.7  # a gap starting above this is a window
DOOR_MIN_HEIGHT = 1.6  # shorter floor-level gaps are unscanned wall, not doorways
WINDOW_MAX_HEIGHT = 2.0
WINDOW_LINTEL_MIN = 0.1  # a window must end this far below the top of the wall
OPENING_PLANE_DISTANCE = 0.15  # points this close to the wall plane belong to the wall
OPENING_BIN_WIDTH = 0.05

# --- Scale recovery for photo and video (src/tiers/colmap_utils.py) ---------------------------
DEFAULT_DOOR_WIDTH = 0.86  # US interior door. Indian and many other doors are 0.75 to 0.90:
# the interval widens by tier, but a non-standard door biases the whole room.
TYPICAL_CEILING = 2.5
DOOR_WIDTH_RANGE = (0.6, 1.2)  # detected "doors" outside this are ignored
DOOR_CORRECTION_RANGE = (0.8, 1.25)  # a bigger door-based correction is a false door
SFM_WALL_INLIER_THRESHOLD = 0.06  # SfM points scatter more around a wall than LiDAR

# --- Room segmentation (src/tiers/room_segmentation.py) ---------------------------------------
DBSCAN_EPS = 0.5  # distance between separate scan parts
DBSCAN_MIN_SAMPLES = 50
SEGMENTATION_CELL = 0.05
ROOM_PERSISTENCE = 0.1  # how much a room's free-space peak must rise above the doorway to its neighbour
MIN_ROOM_RADIUS = 0.3
MIN_ROOM_POINTS = 500

# --- Point cloud processing (src/tiers/lidar.py, depth_stream.py) -----------------------------
VOXEL_SIZE = 0.02
NORMAL_SEARCH_RADIUS = 0.1
NORMAL_MAX_NN = 30
ALIGN_TO_WALLS = True  # rotate the cloud about Z so walls are axis-aligned (plan is not north-up)
DEPTH_MIN = 0.2  # raw LiDAR depth frames: ignore returns closer than this
DEPTH_MAX = 5.0  # ... or farther than this (ARKit LiDAR range)
DEPTH_MIN_CONFIDENCE = 1  # 0 low, 1 medium, 2 high
DEPTH_TARGET_FRAMES = 500  # frames used per capture

# --- Video keyframes (src/tiers/video.py) -----------------------------------------------------
KEYFRAME_SECONDS = 0.5  # one keyframe per this many seconds
KEYFRAME_MAX_COUNT = 400
KEYFRAME_MAX_LONG_EDGE = 1600  # pixels

# --- Confidence calibration (src/calibration/confidence.py) -----------------------------------
CONFIDENCE_MULTIPLIER = {"lidar": 1.5, "video": 3.0, "photo": 6.0}
BASE_UNCERTAINTY = {
    "wall_length": 0.01,
    "ceiling_height": 0.01,
    "opening_width": 0.02,
    "floor_area": 0.1,  # m2
    "damage_extent": 0.05,
}
UNOBSERVED_CEILING_EXTRA = 1.0  # no ceiling scanned: the true height lies up to this far above the highest point

# --- Damage detection (src/damage/detection.py) -----------------------------------------------
CRACK_EDGE_MIN_LENGTH = 50  # pixels at the 1280 px working size
STAIN_MIN_AREA = 500
MOLD_MIN_AREA = 300
WATER_STAIN_HSV = {"lo": (15, 30, 100), "hi": (35, 150, 220)}  # H, S, V
MOLD_HSV = {"lo": (35, 20, 0), "hi": (85, 100, 100)}
