"""
ACES — all physical and tunable constants, in one place.

Rule: no magic number anywhere else in the codebase. If the robot behaves
badly, you change a value here, not code. Every module imports this.

    from config import settings as S
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ══════════════════════════════════════════════════ CHASSIS GEOMETRY
ROBOT_WIDTH_CM        = 32.0
ROW_GAP_CM            = 42.0     # clear soil gap between crop walls
ROW_GAP_MIN_CM        = 40.0     # worst case you must survive
WHEEL_DIAM_CM         = 6.5
WHEEL_BASE_CM         = 22.0
TICKS_PER_REV         = 780

# The crop-crush budget is DERIVED, never typed by hand.
#   40 - 32 = 8 cm total slack -> 4 cm per side, minus 2 cm margin = 2 cm.
LAT_SAFETY_MARGIN_CM  = 2.0
LAT_LIMIT_CM          = max(0.0, (ROW_GAP_MIN_CM - ROBOT_WIDTH_CM) / 2.0
                                 - LAT_SAFETY_MARGIN_CM)
LAT_TRACKING_TOL_CM   = (ROW_GAP_MIN_CM - ROBOT_WIDTH_CM) / 2.0

# ══════════════════════════════════════════════════ LINKS
ESP32_PORT            = "/dev/serial0"
ESP32_BAUD            = 115200
LINK_TIMEOUT_S        = 0.5

GPS_PORT              = "/dev/ttyAMA1"
GPS_BAUD              = 9600
GPS_MIN_SATS          = 5
GPS_MAX_HDOP          = 2.5

# ══════════════════════════════════════════════════ NAV CAMERA
NAV_CAM_INDEX         = 0
NAV_FRAME_W, NAV_FRAME_H = 320, 240
NAV_FPS               = 15
BAND_NEAR             = (0.72, 0.88)
BAND_FAR              = (0.52, 0.66)
PX_PER_CM_NEAR        = 2.10     # MEASURE with tools/calibrate_row.py
PX_PER_CM_FAR         = 1.35     # MEASURE with tools/calibrate_row.py
LOOKAHEAD_NEAR_CM     = 35.0
LOOKAHEAD_FAR_CM      = 70.0
NAV_CAM_X_OFFSET_CM   = 0.0

EXG_THRESHOLD         = 20       # 0 = Otsu
MORPH_KERNEL          = 5
MIN_CORRIDOR_CM       = ROBOT_WIDTH_CM * 0.9
MAX_CORRIDOR_CM       = ROW_GAP_CM * 1.6

CONF_MIN_DRIVE        = 0.35
CONF_BLIND_TIMEOUT_S  = 1.5

# ══════════════════════════════════════════════════ SPEEDS
CRUISE_SPEED_CMS      = 12.0     # narrow gap -> slow. 4 cm of error is all you get.
SLOW_SPEED_CMS        = 6.0
CAPTURE_SPEED_CMS     = 5.0

# ══════════════════════════════════════════════════ SONAR (3x HC-SR04)
SONAR_MIN_CM          = 3.0
SONAR_MAX_CM          = 250.0
OBST_SLOW_CM          = 55.0
OBST_STOP_CM          = 25.0
OBST_CONFIRM_HITS     = 3
OBST_CLEAR_HITS       = 5
OBST_WAIT_S           = 8.0
SIDE_NOMINAL_CM       = (ROW_GAP_CM - ROBOT_WIDTH_CM) / 2.0
SIDE_VALID_MAX_CM     = 40.0
SIDE_FUSE_WEIGHT      = 0.35

# ══════════════════════════════════════════════════ DISEASE CAMERA
DISEASE_CAM_ANGLE_DEG = 45.0
DISEASE_CAM_LEAD_CM   = 22.0
PREVIEW_W, PREVIEW_H  = 640, 360
CAPTURE_W, CAPTURE_H  = 2304, 1296
SHARPNESS_MIN         = 90.0
CAPTURE_RETRIES       = 3
BACKUP_EXTRA_CM       = 8.0

# ══════════════════════════════════════════════════ DETECTOR
# Replace these with YOUR percentiles from tools/pixel_probe.py.
DETECTOR = {
    # --- leaf segmentation (hue only used to SEED the leaf, never to judge) --
    "healthy_h": (33, 88),
    "healthy_s_min": 45,
    "healthy_v_min": 35,
    "exg_thresh": 15,          # LOWER = more permissive. Tune this FIRST:
                               # nothing downstream works until the blue leaf
                               # outline hugs the real leaf.
    "fill_holes": True,
    "close_k": 9,
    "min_leaf_frac": 0.02,
    "max_leaf_frac": 0.92,     # a "leaf" bigger than this = growth ran away

    # How the leaf outline is found:
    #   "grow"  (default) green core, then grow into anything that is not
    #           background and is attached to it. Catches lesions on the leaf
    #           MARGIN, which hole-filling can never recover.
    #   "green" legacy: green mask + hole fill only. Interior lesions only.
    "leaf_method": "grow",
    "bg_border_frac": 0.07,    # frame border ring used to learn the background
    "bg_k": 3.0,               # LOWER -> leaf grows more eagerly into the
                               # background. RAISE if the outline leaks out.
    "bg_L_weight": 0.35,       # lightness counts less than colour, so shadow
                               # does not get mistaken for background

    # --- the abnormality decision (v3: illumination-invariant) ---------------
    # d = (G-R)/(R+G+B). See perception/detector.py for why.
    # Two greenness bounds, doing different jobs:
    #   below min_d_ref         -> not a plant at all, reject the frame
    #   min_d_ref .. d_healthy  -> a plant, but the WHOLE leaf is diseased
    #   above d_healthy         -> healthy tissue present, normal comparison
    "min_d_ref": 0.02,         # a cable, a wall or a hand sits near 0.000
    "d_healthy_foliage": 0.12, # living green foliage is +0.15 to +0.30
    "ref_percentile": 75,      # which percentile of d counts as "healthy here"
    "k_weak": 1.8,             # relative gate, in robust std devs below d_ref
    "k_strong": 3.0,           # seed gate for hysteresis
    "d_abs_max": 0.075,        # ABSOLUTE gate. Raise -> more sensitive AND more
                               # false positives. This is the single most
                               # important number in the detector.
    "d_abs_strong": 0.055,

    # --- exclusions ----------------------------------------------------------
    "shadow_v_max": 32,
    "glare_s_max": 28,
    "glare_v_min": 225,
    "k_specular": 3.0,         # adaptive specular removal, in robust std devs
                               # of min(R,G,B) above this leaf's own median.
                               # LOWER = removes more highlight (and more leaf).

    # --- blob acceptance -----------------------------------------------------
    "min_blob_px": 400,
    "min_blob_frac_of_leaf": 0.004,
    "max_blob_frac_of_leaf": 0.85,
}
# tools/tune_live.py writes config/detector_tuned.json. If that file exists it
# overrides the values above, so your tuning survives a code update and this
# file never has to be hand-edited. Delete the json to go back to defaults.
_tuned = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "detector_tuned.json")
if os.path.exists(_tuned):
    try:
        import json as _json
        with open(_tuned) as _fh:
            _t = _json.load(_fh)
        if isinstance(_t.get("healthy_h"), list):
            _t["healthy_h"] = tuple(_t["healthy_h"])
        DETECTOR.update(_t)
        print(f"[settings] detector overrides loaded from {_tuned}")
    except Exception as _e:
        print(f"[settings] could not load {_tuned}: {_e}")

DETECT_RATIO_MIN      = 0.02     # ratio above which we call it a detection
SEVERITY_BANDS        = [(0.02, "trace"), (0.08, "mild"),
                         (0.20, "moderate"), (1.01, "severe")]

# ══════════════════════════════════════════════════ CLASSIFIER
TFLITE_MODEL          = os.path.join(ROOT, "models", "plant_disease.tflite")
LABELS_FILE           = os.path.join(ROOT, "models", "labels.txt")
CLASSIFIER_INPUT      = 224
CONF_UNCERTAIN        = 0.60     # below this -> data/uncertain, no Telegram

# ══════════════════════════════════════════════════ ROW END / TURN
ROW_LENGTH_CM         = 3000.0
ROW_LENGTH_TOL_CM     = 250.0
ROWEND_GREEN_FRAC     = 0.04
ROWEND_CONFIRM_FRAMES = 8
TURN_DEGREES          = 180.0
PASSES_PER_ROW        = 2        # one 45-deg camera -> each row twice
HEADLAND_CLEAR_CM     = 60.0     # a 32 cm robot cannot spin inside a 42 cm gap

# ══════════════════════════════════════════════════ STORAGE / OUTPUT
SD_ROOT               = os.environ.get("ACES_DATA", os.path.join(ROOT, "data"))
LOG_CSV               = os.path.join(SD_ROOT, "log.csv")
DUP_RADIUS_M          = 3.0      # suppress repeat detections within this radius
STREAM_PORT           = 8080
MAP_PORT              = 8081

TELEGRAM_ENABLED      = True
SHEETS_ENABLED        = False    # turn on once credentials.json is in place
SHEET_NAME            = "ACES Field Log"
