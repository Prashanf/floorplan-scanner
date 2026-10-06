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

# --- Scale recovery for photo and video (src/tiers/colmap_utils.py)
MIN_PLAUSIBLE_AREA = 4.0  # m2: a reconstruction smaller than this is no real room, so the next scale prior is tried
MIN_ROOM_AREA_WARN = 2.0  # a room outside [MIN_ROOM_AREA_WARN, MAX_ROOM_AREA_WARN] m2 gets a scale warning
MAX_ROOM_AREA_WARN = 100.0
TYPICAL_LONGEST_WALL = 3.5  # last-resort prior: the longest wall is about this long (m) ---------------------------
DEFAULT_DOOR_WIDTH = 0.86  # US interior door. Indian and many other doors are 0.75 to 0.90:
# the interval widens by tier, but a non-standard door biases the whole room.
TYPICAL_CEILING = 2.5
DOOR_WIDTH_RANGE = (0.6, 1.2)  # detected "doors" outside this are ignored
DOOR_CORRECTION_RANGE = (0.8, 1.25)  # a bigger door-based correction is a false door
SFM_WALL_INLIER_THRESHOLD = 0.06  # SfM points scatter more around a wall than LiDAR
MIN_SCALE_CONFIDENCE = 0.5  # the scale chain stops at the first method above this
CEILING_CANDIDATES = (2.4, 2.5, 2.7, 3.0)  # ceiling heights tried by the ceiling prior; a visible door picks one

# --- Floor tile scale recovery (src/geometry/scale_recovery.py) --------------------------------
TILE_SIZES = (0.30, 0.45, 0.60, 0.80)  # standard square tiles (m): 12, 18, 24, 32 inch
TILE_BANDS = {0.30: (0.25, 0.35), 0.45: (0.40, 0.50), 0.60: (0.50, 0.65), 0.80: (0.70, 0.90)}  # computed size -> tile
TILE_MAX_IMAGES = 6  # images tried per room (evenly spaced over the registered ones)
TILE_IMAGE_LONG_EDGE = 1600  # px; larger images are downscaled before rectification
TILE_ORTHO_MAX = 1400  # px; longest side of the rectified floor image
TILE_MIN_DEPRESSION_DEG = 12.0  # floor pixels closer than this to the horizon are too oblique to use
TILE_MIN_LINES = 3  # distinct grout lines needed in each direction
TILE_SQUARE_TOLERANCE = 0.10  # the two line spacings must agree within this share (tiles are square)

# --- Room segmentation (src/tiers/room_segmentation.py) ---------------------------------------
DBSCAN_EPS = 0.5  # distance between separate scan parts
DBSCAN_MIN_SAMPLES = 50
SEGMENTATION_CELL = 0.05
ROOM_PERSISTENCE = 0.1  # how much a room's free-space peak must rise above the doorway to its neighbour
MIN_ROOM_RADIUS = 0.3
MIN_ROOM_POINTS = 500
# LiDAR tier only: stricter room splitting and cleanup of fragments (the video tier keeps the values above)
LIDAR_ROOM_PERSISTENCE = 0.1  # same as the video tier: 0.1 to 0.2 gave identical rooms on the sample captures
LIDAR_MIN_ROOM_RADIUS = 0.3  # same as the video tier: 0.4 or more collapsed rooms and distorted the area on the sample captures
LIDAR_MIN_ROOM_AREA = 2.0  # m2; a smaller room is merged into the neighbour it shares the longest boundary with

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
KEYFRAME_SECONDS = 0.5  # windowed mode: one keyframe per this many seconds
KEYFRAME_MAX_COUNT = 400
KEYFRAME_MAX_LONG_EDGE = 1920  # pixels; keyframes are downscaled to this before COLMAP
KEYFRAME_MIN_GAP = 10  # motion mode: at least every 10th frame at most
KEYFRAME_MAX_GAP = 60  # ... and at least one keyframe in this many frames even without motion
KEYFRAME_MOTION_FRACTION = 0.05  # median feature displacement (share of frame width) that starts a new keyframe
VIDEO_RETRY_MAX_FRAMES = 600  # retry after a failed reconstruction: denser sampling, capped at this many frames
VIDEO_RETRY_STEP = 5  # ... every 5th frame, or sparser if the video is long enough to exceed the cap

# --- Confidence calibration (src/calibration/confidence.py) -----------------------------------
CONFIDENCE_MULTIPLIER = {"lidar": 1.5, "video": 3.0, "photo": 6.0}
BASE_UNCERTAINTY = {
    "wall_length": 0.01,
    "ceiling_height": 0.01,
    "opening_width": 0.02,
    "floor_area": 0.1,  # m2
    "damage_extent": 0.05,
}
ROUGH_ESTIMATE_RELATIVE_ERROR = 0.5  # single-image fallback rooms: +/-50 % on every length
UNOBSERVED_CEILING_EXTRA = 1.0  # no ceiling scanned: the true height lies up to this far above the highest point

# --- Damage detection (src/damage/detection.py) -----------------------------------------------
CRACK_EDGE_MIN_LENGTH = 150  # pixels at the 1280 px working size (tile grout and furniture edges are shorter)
STAIN_MIN_AREA = 2000
MOLD_MIN_AREA = 1500
MAX_DETECTION_AREA_FRACTION = 0.2  # a detection covering more of the image than this is a wall or object
MAX_DETECTIONS_PER_IMAGE = 5  # strongest only
PERSISTENT_FRAME_LIMIT = 3  # video: a detection at the same image position in more than this many consecutive frames is dropped
PERSISTENT_IOU = 0.5  # overlap that counts as "the same position"
ROOM_DETECTION_CAP = 20  # more detections than this in one room: assume false positives ...
ROOM_DETECTION_KEEP = 10  # ... and keep only the strongest this many
WATER_STAIN_HSV = {"lo": (15, 30, 100), "hi": (35, 150, 220)}  # H, S, V
MOLD_HSV = {"lo": (35, 20, 0), "hi": (85, 100, 100)}

# --- Model damage detection (src/damage/model_detection.py) ------------------------------------
MODEL_DAMAGE_THRESHOLD = 0.3  # OWL score above which a box is kept. Provisional: set from 9 photos, tune on labelled captures
MODEL_OWL_NAME = "google/owlv2-base-patch16-ensemble"  # "google/owlvit-base-patch32" (v1, scores much lower) also works

# --- Photo stitching, room merging, rectangle fallback ------------------------------------------
DOORWAY_PHOTO_WINDOW_S = 30  # photo tier: an image taken within this many seconds of the next room's first image is the doorway photo
DOORWAY_MAX_TILT_DEG = 10.0  # a shared-camera transform whose "up" axes differ by more than this is rejected
DOOR_WIDTH_MATCH = 0.15  # two doors can be the same doorway when their widths differ by less than this share
MERGE_SHARED_FRACTION = 0.8  # video tier: rooms sharing a boundary this long (vs the shorter room's facing wall) are one room
RECT_MAX_WALLS = 6  # a room with more wall segments than this is tried against a bounding rectangle
RECT_EDGE_TOLERANCE = 0.15  # metres: a wall point this close to a rectangle edge is explained by it
RECT_MIN_EXPLAINED = 0.7  # share of wall points the rectangle must explain
