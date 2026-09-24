"""
ACES full mission: follow the row, find diseased leaves, pin them on a map.

Everything in one loop:

    WEBCAM   -> row centreline -> steer the bot down the middle
             -> obstacle in the way -> stop until it clears
    PI CAM   -> diseased leaf -> stop, photograph, GPS-stamp, pin, Telegram
             -> no plants at all -> row has ended, stop
    GPS      -> live position on the map
    OUTPUT   -> SD card, live map at :8081, Telegram group

Two browser tabs:
    http://<pi-ip>:8080    what the cameras see
    http://<pi-ip>:8081    live map with detection pins

Run it
------
    python3 run_mission.py                          # everything
    python3 run_mission.py --no-drive               # perception only, no motors
    python3 run_mission.py --port /dev/ttyACM0
    python3 run_mission.py --no-telegram

CH5 HIGH on the transmitter = autonomous. CH5 LOW = manual RC, instantly.
The firmware enforces that; this script cannot override it.
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings as S                      # noqa: E402
from navigation.row_vision import RowFollower         # noqa: E402
from perception import detector as D                  # noqa: E402
from telemetry import map_server                      # noqa: E402
from telemetry.storage import Storage                 # noqa: E402
from telemetry.telegram_bot import Telegram           # noqa: E402

try:
    from perception.classifier import DiseaseClassifier
except Exception:
    DiseaseClassifier = None
try:
    from telemetry.gps_reader import GPSReader, haversine_m
except Exception:
    GPSReader = None
    def haversine_m(*a):
        return 1e9

# ---------------------------------------------------------------- tunables
OBST_X0, OBST_X1 = 0.34, 0.66
OBST_Y0          = 0.45
OBST_BRIGHT      = 235
OBST_FRACTION    = 0.55
OBST_CONFIRM     = 3
OBST_CLEAR       = 5

PLANT_STRIP      = 0.45
PLANT_D_MIN      = 0.05
PLANT_FRAC_MIN   = 0.10
PLANT_BLOB_MIN   = 0.06
ROW_END_CONFIRM  = 15

# ---- losing the row is not the same as reaching the end ------------------
# Declaring ROW_END the moment the walls vanish is wrong: a gap in the
# planting, a washed-out frame, or a patch of glare all look identical to
# actually running out of row. A real row end is the walls staying gone.
#
# So when the walls disappear the bot COASTS instead: it keeps driving
# straight on its last known heading, still watching. If the row comes back
# it carries on as if nothing happened. Only after ROW_SEARCH_S of continuous
# absence does it accept that the row has ended.
ROW_SEARCH_S      = 12.0   # how long to keep looking before giving up
ROW_REACQUIRE     = 5      # frames of walls needed to call it found again
ROW_COAST_STEER   = 0.5    # decay the last steer while coasting -- keep some
                           # of the correction, but drift toward straight

# ---- end-of-row turning -------------------------------------------------
# Off by default. A half-working turn will eat your whole session, and the
# straight run is a complete demo on its own. Enable with --turn-at-end.
#
# Four phases, and the important one is that the TURN ends on VISION, not on
# a timer. A timed spin is wrong the moment the surface, the slope or the
# battery level changes, and you would re-tune it every session. Spinning
# until the camera sees a row again is self-correcting and needs no
# calibration.
TURN_DIR          = +1     # +1 right, -1 left
HEADLAND_CLEAR_S  = 4.0    # drive forward before spinning. The bot sweeps
                           # ~34 cm turning and cannot rotate inside a 60 cm
                           # track without clipping the last plants.
TURN_TIMEOUT_S    = 8.0    # give up if no row is found
TURN_REACQUIRE    = 6      # consecutive frames of "walls both" before we
                           # believe the turn is finished. One lucky frame
                           # mid-spin must not end it.

CAPTURE_SETTLE_S = 1.0      # let the chassis stop moving before the still
RESUME_DELAY_S   = 1.0
DETECT_COOLDOWN  = 6.0

# ---- stop-and-scan -------------------------------------------------------
# Driving past a plant and hoping a frame lands on the lesion is a poor way
# to inspect it: at speed the leaf is blurred, off-centre, or gone by the
# time detection runs. Stopping to look is both more reliable and slower on
# average, which incidentally gives the row follower more frames per metre.
#
# The bot drives until the side camera says a plant is beside it, stops,
# holds still for SCAN_HOLD_S while detection runs over several steady
# frames, then either captures or moves on.
# ---- recentring at each scan stop ---------------------------------------
# A skid-steer robot cannot slide sideways. To recover from drift it has to
# turn toward the centreline, drive a little, and turn back -- a dogleg. Done
# at cruise speed that overshoots and weaves.
#
# The scan stop is the ideal moment: the bot is already stationary, the
# camera has steady frames to measure from, and there is no forward momentum
# to fight. So after inspecting a plant, if the bot has drifted, it makes
# short low-speed correction pulses with a strong differential and re-checks
# the centreline between each one. It only resumes once it is back near the
# middle, or after RECENTRE_MAX_S if the row is too ambiguous to fix.
# ---- steering trim -------------------------------------------------------
# A chassis that is not quite square, or motors that are not quite matched,
# makes the bot pull constantly to one side. The vision controller then
# spends its whole authority fighting that bias instead of following the row.
#
# Trim cancels it: a fixed offset added to every steering command, so the
# controller starts from "straight" rather than from "drifting left".
#
# To find your value: run with --trim 0 on a flat floor with no row in view,
# see which way it curves, and add trim in the OPPOSITE direction until it
# runs straight. Positive = steer right.
STEER_TRIM       = 0        # override at runtime with --trim

RECENTRE_ENABLED = True
RECENTRE_TOL     = 0.06    # offset, as fraction of frame width, that counts
                           # as "centred enough". ~6% of the view.
RECENTRE_GAIN    = 2.2     # multiplier on the normal steering while doing it
RECENTRE_PULSE_S = 0.35    # length of each correction burst
RECENTRE_GAP_S   = 0.45    # settle and re-measure between bursts
RECENTRE_MAX_S   = 6.0     # give up and carry on rather than dither forever

SCAN_ENABLED     = True
SCAN_HOLD_S      = 2.0     # how long to sit still and look
SCAN_COOLDOWN_S  = 5.0     # don't re-scan the plant you just looked at
SCAN_MIN_FRAMES  = 8       # frames that must agree before deciding

# ---- pulsed travel -------------------------------------------------------
# A DC motor needs a certain PWM just to overcome stiction, so you cannot
# simply turn the speed down -- below roughly 70 it buzzes and does not turn.
# To go slower, drive in short bursts instead: full torque while moving,
# stationary in between. Effective speed is CRUISE * ON/(ON+OFF).
# 0.35 s of drive proved too short: the bot barely overcame its own inertia
# before being told to stop again, so it twitched instead of travelling.
# A burst has to be long enough to actually accelerate the chassis.
PULSE_ENABLED    = True
PULSE_ON_S       = 0.9
PULSE_OFF_S      = 0.6

STREAM_PORT      = S.STREAM_PORT
JPEG_Q           = 85



def find_esp32_port(preferred=None):
    """
    Find the ESP32 without being told which port it is on.

    The device number changes every time the board is reflashed or the USB
    connection glitches -- ttyACM0 today, ttyACM1 after the next upload. That
    keeps showing up as a mysterious "no-link", so stop hardcoding it.

    Strategy: try the port the user asked for, then every ttyACM* and
    ttyUSB*, and keep the first one that actually sends us a '#T' or '#E'
    line. Talking is the test, not existing.
    """
    import glob
    import serial

    cands = []
    if preferred and os.path.exists(preferred):
        cands.append(preferred)
    for pat in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        for dev in sorted(glob.glob(pat)):
            if dev not in cands:
                cands.append(dev)

    if not cands:
        print("[esp32] no serial devices found at all.")
        print("        check the USB cable carries data, then:  dmesg | tail")
        return None

    for dev in cands:
        try:
            ser = serial.Serial()
            ser.port = dev
            ser.baudrate = 115200
            ser.timeout = 0.3
            ser.dtr = False          # do NOT reset the ESP32 on open
            ser.rts = False
            ser.open()
            ser.reset_input_buffer()
            t0 = time.time()
            while time.time() - t0 < 2.5:
                ser.write(b"$P\n")
                line = ser.readline().decode("ascii", "ignore")
                if line.startswith("#T,") or line.startswith("#E,"):
                    ser.close()
                    print(f"[esp32] found on {dev}")
                    return dev
            ser.close()
            print(f"[esp32] {dev} opened but stayed silent")
        except Exception as e:
            print(f"[esp32] {dev}: {e}")
    return None


# Modes the ESP32 can report that mean "ready to be driven".
# The no-RC firmware says IDLE/RUN. The older receiver firmware said
# AUTO_IDLE/AUTO_RUN once CH5 was high. Accepting only the second set meant
# the mission waited for ever with the no-RC firmware and never sent $GO.
DRIVABLE_MODES = ("IDLE", "RUN", "TURN", "AUTO_IDLE", "AUTO_RUN")


# ---------------------------------------------------------------- link
class Link:
    def __init__(self, port, baud=115200, enabled=True):
        self.ok = False
        self.mode = "no-link"
        self._lock = threading.Lock()
        if not enabled:
            print("[esp32] disabled (--no-drive): motors are never commanded")
            return
        try:
            import serial
            # DO NOT let pyserial toggle DTR/RTS on open. On ESP32 boards
            # those lines are wired to the auto-reset circuit, so simply
            # opening the port can hold the chip in reset -- or drop it into
            # the bootloader. The port then opens fine and you get silence,
            # which looks exactly like a dead board.
            self.ser = serial.Serial()
            self.ser.port = port
            self.ser.baudrate = baud
            self.ser.timeout = 0.2
            self.ser.dtr = False
            self.ser.rts = False
            self.ser.open()
            self.ser.reset_input_buffer()
            self.ok = True
            threading.Thread(target=self._rx, daemon=True).start()

            # Prove the board is actually talking before we trust the link.
            t0 = time.time()
            while time.time() - t0 < 4.0:
                if self.read_mode() != "no-link":
                    break
                self.send("$P")
                time.sleep(0.2)
            if self.read_mode() == "no-link":
                print(f"[esp32] port {port} opened but NO TELEMETRY received.")
                print("        The board is not sending #T lines. Try:")
                print("          - press the ESP32 reset button now")
                print("          - check the sketch really is aces_noRC.ino")
                print("          - a charge-only USB cable powers it but "
                      "carries no data")
            else:
                print(f"[esp32] connected on {port}, "
                      f"telemetry OK (mode {self.read_mode()})")
        except Exception as e:
            print(f"[esp32] NOT connected ({e}) — perception only")

    def _rx(self):
        while True:
            try:
                ln = self.ser.readline().decode("ascii", "ignore").strip()
            except Exception:
                time.sleep(0.1)
                continue
            if ln.startswith("#T,"):
                p = ln[3:].split(",")
                if len(p) >= 2:
                    with self._lock:
                        self.mode = p[1]
            elif ln.startswith("#E,"):
                print(f"[esp32] {ln[3:]}")

    def send(self, cmd):
        if self.ok:
            try:
                self.ser.write((cmd + "\n").encode())
            except Exception:
                pass

    def read_mode(self):
        with self._lock:
            return self.mode

    def close(self):
        if self.ok:
            self.send("$STOP")
            time.sleep(0.2)
            self.ser.close()


# ---------------------------------------------------------------- cameras
def open_webcam(idx, width, height, fps, exposure=None):
    """
    Open a camera. `idx` is either a /dev/videoN index, or a URL.

    A phone running an IP-camera app works as a source: pass its stream URL
    instead of an index. Worth knowing the trade-off before you rely on it --
    a WiFi stream adds latency (typically 200-500 ms) and a control loop
    steering on half-second-old frames will overshoot. It also adds the
    hotspot and the app as things that can fail mid-run. Fine for checking
    the vision, riskier as the thing your demo depends on.
    """
    url = isinstance(idx, str) and not idx.isdigit()
    if url:
        print(f"[webcam] opening stream {idx}")
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print("[webcam] could not open that URL. Check the phone and the")
            print("         Pi are on the same network, and that the URL ends")
            print("         in /video for IP Webcam.")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always take the freshest frame
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[webcam] stream {aw}x{ah}")
        return cap
    idx = int(idx)
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"[webcam] cannot open index {idx}. Try --webcam 1")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Pull the exposure DOWN. A webcam metering a bright concrete floor
    # over-exposes the plants: they wash out to pale grey-green and the
    # vegetation mask finds almost nothing. Darker frames keep the leaf
    # colour saturated, which is what the mask actually needs. Not every
    # webcam honours these, hence the try.
    if exposure is not None:
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # manual on most UVC
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            cap.set(cv2.CAP_PROP_GAIN, 0)
        except Exception as e:
            print(f"[webcam] exposure control unavailable: {e}")

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[webcam] {aw}x{ah} @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps  "
          f"exposure {'auto' if exposure is None else exposure}")
    return cap


def open_picam():
    try:
        from picamera2 import Picamera2
        pc = Picamera2()
        pc.configure(pc.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"}))
        pc.start()
        time.sleep(2.0)
        try:
            from libcamera import controls
            pc.set_controls({"AfMode": controls.AfModeEnum.Manual,
                             "LensPosition": 100.0 / 40.0})
            time.sleep(1.0)
            md = pc.capture_metadata()
            pc.set_controls({"AwbEnable": False,
                             "ColourGains": md.get("ColourGains", (1.8, 1.8))})
        except Exception as e:
            print(f"[picam] controls: {e}")
        print("[picam] open, focus fixed ~40 cm, AWB locked")
        return pc
    except Exception as e:
        print(f"[picam] unavailable ({e})")
        return None


# ---------------------------------------------------------------- vision
def check_obstacle(frame):
    h, w = frame.shape[:2]
    x0, x1 = int(w * OBST_X0), int(w * OBST_X1)
    y0 = int(h * OBST_Y0)
    band = frame[y0:, x0:x1]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, OBST_BRIGHT, 255, cv2.THRESH_BINARY)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8))
    occ = float(cv2.bitwise_or(bright, edges).mean()) / 255.0
    return occ > OBST_FRACTION, occ


def check_plants(frame):
    """Plants beside us AND close enough. Largest blob is the distance proxy."""
    if frame is None:
        return True, 0.0, 0.0
    h, w = frame.shape[:2]
    strip = frame[int(h * 0.15):int(h * 0.9), :int(w * PLANT_STRIP)]
    b, g, r = cv2.split(strip.astype(np.float32))
    d = (g - r) / (r + g + b + 1e-6)
    mask = (d > PLANT_D_MIN).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    total = mask.shape[0] * mask.shape[1]
    gf = float(mask.sum()) / 255.0 / total
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    bf = float(stats[1:, cv2.CC_STAT_AREA].max()) / total if n > 1 else 0.0
    return (gf >= PLANT_FRAC_MIN and bf >= PLANT_BLOB_MIN), gf, bf


# ---------------------------------------------------------------- stream
_views = {"webcam": None, "picam": None, "text": "starting"}
_vlock = threading.Lock()
try:
    from flask import Flask, Response
    _app = Flask(__name__)
    HAVE_FLASK = True
except ImportError:
    HAVE_FLASK = False

if HAVE_FLASK:
    def _gen(key):
        while True:
            with _vlock:
                f = _views.get(key)
            if f is not None:
                ok, buf = cv2.imencode(".jpg", f,
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                if ok:
                    yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buf.tobytes() + b"\r\n")
            time.sleep(0.07)

    @_app.route("/webcam.mjpg")
    def _w():
        return Response(_gen("webcam"),
                        mimetype="multipart/x-mixed-replace; boundary=f")

    @_app.route("/picam.mjpg")
    def _p():
        return Response(_gen("picam"),
                        mimetype="multipart/x-mixed-replace; boundary=f")

    @_app.route("/readout")
    def _r():
        with _vlock:
            return _views["text"]

    @_app.route("/")
    def _i():
        return """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ACES mission</title>
<style>
:root{--loam:#12160f;--panel:#1b2116;--rule:#2f3a26;--crop:#7fb069;
      --ink:#e8ece3;--dim:#8b9680;
      --mono:ui-monospace,"DejaVu Sans Mono",Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--loam);color:var(--ink);font:13px/1.5 var(--mono)}
header{padding:11px 15px;border-bottom:1px solid var(--rule);background:var(--panel);
  display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
h1{margin:0;font-size:12px;letter-spacing:.26em;font-weight:700}
header a{color:var(--crop);font-size:11px;letter-spacing:.1em;text-decoration:none;
  border-bottom:1px solid #3c4a31}
main{display:grid;gap:1px;background:var(--rule);grid-template-columns:1fr}
@media(min-width:780px){main{grid-template-columns:1fr 1fr}}
figure{margin:0;background:var(--panel);padding:12px}
figcaption{font-size:10px;letter-spacing:.14em;color:var(--dim);margin-bottom:7px}
img{width:100%;display:block;border:1px solid var(--rule);background:#000}
pre{margin:0;padding:13px 15px;background:var(--panel);
  border-top:1px solid var(--rule);white-space:pre-wrap;font-size:12.5px}
</style>
<header><h1>ACES &mdash; MISSION</h1>
 <a href="http://localhost:8081" target="_blank">OPEN LIVE MAP &rarr;</a></header>
<main>
 <figure><figcaption>WEBCAM &mdash; row centreline &amp; obstacles</figcaption>
  <img src="/webcam.mjpg" alt="Forward webcam with row edges"></figure>
 <figure><figcaption>PI CAMERA &mdash; disease detection</figcaption>
  <img src="/picam.mjpg" alt="Side camera with abnormality mask"></figure>
</main>
<pre id="r">connecting</pre>
<script>
setInterval(async()=>{try{
 document.getElementById('r').textContent=await(await fetch('/readout')).text();
}catch(e){document.getElementById('r').textContent='link lost';}},350);
</script>"""

    def start_stream():
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        threading.Thread(target=lambda: _app.run(
            host="0.0.0.0", port=STREAM_PORT, threaded=True,
            debug=False, use_reloader=False), daemon=True).start()
        print(f"[stream] http://0.0.0.0:{STREAM_PORT}")
else:
    def start_stream():
        print("[stream] flask missing. pip install flask")


# ---------------------------------------------------------------- main
def main():
    global JPEG_Q
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--webcam", default="0",
                    help="/dev/videoN index, OR a stream URL from a phone "
                         "camera app, e.g. http://192.168.1.5:8080/video")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--proc-width", type=int, default=480)
    ap.add_argument("--jpeg", type=int, default=85)
    ap.add_argument("--exposure", type=int, default=None,
                    help="webcam exposure. LEFT ALONE by default -- the camera "
                         "keeps its own auto exposure. Pass a value only if "
                         "you want to override it (lower = darker).")
    ap.add_argument("--no-drive", action="store_true")
    ap.add_argument("--no-telegram", action="store_true")
    ap.add_argument("--no-classify", action="store_true")
    ap.add_argument("--only-confident", action="store_true",
                    help="send ONLY high-confidence detections to Telegram. "
                         "Off by default: an uncertain photo costs a glance, "
                         "a missed lesion costs a crop.")
    ap.add_argument("--no-plant-check", action="store_true",
                    help="do not use the Pi Camera to decide the row has "
                         "ended. Use this when the side camera is aimed for "
                         "disease scanning rather than at the crop wall -- "
                         "otherwise a frame of bare wall ends the mission.")
    # These are OPT-IN. The default is plain continuous driving steered by
    # the centreline -- the behaviour that actually worked. Pulsed travel
    # made the bot lurch and stop repeatedly, which looked like a fault.
    ap.add_argument("--trim", type=int, default=STEER_TRIM,
                    help="constant steering offset to cancel a chassis pull. "
                         "POSITIVE steers right. Start at 0, watch which way "
                         "it drifts, correct the other way.")
    ap.add_argument("--scan-stops", action="store_true",
                    help="stop beside each plant to inspect it")
    ap.add_argument("--pulse", action="store_true",
                    help="drive in short bursts for a lower average speed")
    ap.add_argument("--scan-hold", type=float, default=SCAN_HOLD_S,
                    help="seconds to hold still while inspecting a plant")
    ap.add_argument("--turn-at-end", action="store_true",
                    help="turn and keep going when the row ends, instead of "
                         "stopping. Leave OFF until the straight run works.")
    ap.add_argument("--turn-dir", type=int, default=TURN_DIR,
                    help="+1 right, -1 left")
    ap.add_argument("--passes", type=int, default=2,
                    help="how many rows to do before stopping for good")
    a = ap.parse_args()
    JPEG_Q = a.jpeg
    if a.no_telegram:
        S.TELEGRAM_ENABLED = False

    port = a.port
    if not a.no_drive:
        found = find_esp32_port(a.port)
        if found:
            port = found
    link = Link(port, enabled=not a.no_drive)
    webcam = open_webcam(a.webcam, a.width, a.height, a.fps, a.exposure)
    picam = open_picam()
    rows = RowFollower()
    store = Storage()
    tg = Telegram()
    gps = GPSReader() if GPSReader else None
    clf = None
    if DiseaseClassifier and not a.no_classify:
        clf = DiseaseClassifier()
    start_stream()
    map_server.start()

    print(f"\n  cameras : http://<pi-ip>:{STREAM_PORT}")
    print(f"  map     : http://<pi-ip>:{S.MAP_PORT}")
    print("\n  ctrl-c stops the bot -- the ESP32 also halts by itself if")
    print("  this script goes quiet for one second\n")

    state = "WAITING"
    turn_t0 = 0.0
    reacq = 0
    passes_done = 0
    obs_hits = obs_clear = no_plant = 0
    no_wall_frames = 0
    row_end_sent = False
    last_hit = 0.0
    t_state = time.time()
    found = 0
    seen = []
    last_scan = 0.0
    scan_t0 = 0.0
    scan_frames = 0
    scan_hits = 0
    pulse_t0 = time.time()
    pulse_on = True
    recentre_t0 = 0.0
    recentre_pulse_on = False
    recentre_t = 0.0
    search_t0 = 0.0
    search_reacq = 0
    coast_steer = 0

    def is_dup(fx):
        if not (fx and fx.fix_ok):
            return False
        return any(haversine_m(fx.lat, fx.lon, la, lo) < S.DUP_RADIUS_M
                   for la, lo in seen)

    try:
        while True:
            # ---- webcam: navigation ----------------------------------
            ok, wfull = webcam.read()
            if not ok:
                time.sleep(0.05)
                continue
            if wfull.shape[1] > a.proc_width:
                sc = a.proc_width / wfull.shape[1]
                wsmall = cv2.resize(wfull, None, fx=sc, fy=sc,
                                    interpolation=cv2.INTER_AREA)
            else:
                wsmall = wfull
            is_obs, occ = check_obstacle(wsmall)
            row = rows.update(wsmall)
            wdbg = cv2.resize(row.debug, (wfull.shape[1], wfull.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

            # ---- pi camera: plants + disease --------------------------
            pfull = picam.capture_array() if picam else None
            present, gfrac, bfrac = check_plants(pfull)
            if pfull is not None:
                psmall = cv2.resize(pfull, (800, 450))
                res = D.detect(psmall)
                pdbg = D.overlay(psmall, res)
            else:
                res = None
                pdbg = np.zeros((450, 800, 3), np.uint8)

            fix = gps.read() if gps else None

            # ---- debounce ---------------------------------------------
            if is_obs:
                obs_hits += 1
                obs_clear = 0
            else:
                obs_clear += 1
                if obs_clear >= OBST_CLEAR:
                    obs_hits = 0
            obstacle = obs_hits >= OBST_CONFIRM

            # Row-end needs BOTH signals by default. But the Pi Camera is
            # angled for disease scanning, and if it happens to be looking at
            # bare wall the mission ends before it starts. --no-plant-check
            # falls back to the webcam walls alone.
            # Only count "row missing" frames while we are actually driving.
            # A capture stop takes a few seconds, and letting the counter run
            # during it meant the bot finished photographing a leaf and
            # immediately declared the row over -- it had been parked, not
            # driving past the end of anything.
            if state in ("STOPPING", "CAPTURING", "RESUMING", "RECENTRE",
                         "SCANNING"):
                pass                       # parked: the row has not moved
            else:
                no_wall_frames = 0 if row.walls != "none" else no_wall_frames + 1
                no_plant = 0 if present else no_plant + 1
            if a.no_plant_check:
                row_over = (row.walls == "none") and no_wall_frames >= ROW_END_CONFIRM
            else:
                row_over = no_plant >= ROW_END_CONFIRM

            diseased = (res is not None and res.trusted and res.blobs
                        and res.ratio >= S.DETECT_RATIO_MIN)

            esp = link.read_mode()

            # Say out loud why a detection did or did not act. Reading the
            # ratio off the overlay is not enough: the overlay paints every
            # abnormal pixel regardless of whether the state machine was in a
            # position to do anything about it.
            if diseased:
                why = []
                if state not in ("RUNNING", "SEARCHING"):
                    why.append(f"state is {state}")
                if (time.time() - last_hit) <= DETECT_COOLDOWN:
                    why.append(f"cooldown {DETECT_COOLDOWN - (time.time()-last_hit):.1f}s left")
                if is_dup(fix):
                    why.append("duplicate within 3 m")
                if why:
                    print(f"[mission] diseased {res.ratio:.1%} IGNORED: "
                          + ", ".join(why))

            # ---- state machine ----------------------------------------
            # Walls gone? Coast and search rather than giving up.
            if state == "RUNNING" and row.walls == "none":
                state = "SEARCHING"
                search_t0 = time.time()
                search_reacq = 0
                coast_steer = row.steer
                print("[mission] lost the row -> coasting and searching")

            elif state == "SEARCHING":
                # A diseased leaf found while hunting for the row still
                # counts. The old code only checked in RUNNING, so on a row
                # that flickers in and out the bot drove straight past
                # obvious lesions -- it was simply not looking at the result.
                if diseased and (time.time() - last_hit) > DETECT_COOLDOWN \
                        and not is_dup(fix):
                    state = "STOPPING"
                    link.send("$STOP")
                    t_state = time.time()
                    no_wall_frames = 0
                    no_plant = 0
                    print(f"[mission] diseased leaf while searching "
                          f"({res.ratio:.1%}) -> stopping")
                elif row.walls != "none":
                    search_reacq += 1
                    if search_reacq >= ROW_REACQUIRE:
                        state = "RUNNING"
                        no_wall_frames = 0
                        print(f"[mission] row reacquired after "
                              f"{time.time()-search_t0:.1f}s")
                else:
                    search_reacq = 0
                if state == "SEARCHING" \
                        and time.time() - search_t0 > ROW_SEARCH_S:
                    if a.turn_at_end and passes_done + 1 < a.passes:
                        state = "CLEARING"
                        t_state = time.time()
                        print("[mission] row really has ended -> clearing")
                    else:
                        state = "ROW_END"
                        link.send("$ROW_END")
                        row_end_sent = True
                        print(f"[mission] no row for {ROW_SEARCH_S:.0f}s "
                              f"-> stopping")

            # ROW_END is only reachable from a state that was actually
            # driving. Reaching it from WAITING means we never started.
            elif state in ("RUNNING", "OBSTACLE") and row_over:
                if a.turn_at_end and passes_done + 1 < a.passes:
                    state = "CLEARING"
                    t_state = time.time()
                    print("[mission] row ended -> clearing headland")
                else:
                    state = "ROW_END"
                    link.send("$ROW_END")
                    row_end_sent = True
                    print("[mission] row ended -> stopping")

            elif state == "CLEARING":
                # keep driving straight, just far enough that the whole
                # chassis is past the last plants before we spin
                link.send(f"$S,{a.trim}")
                link.send("$GO")
                if time.time() - t_state > HEADLAND_CLEAR_S:
                    state = "TURNING"
                    turn_t0 = time.time()
                    reacq = 0
                    link.send(f"$T,{a.turn_dir}")
                    print(f"[mission] turning {'right' if a.turn_dir>0 else 'left'}")

            elif state == "TURNING":
                # THE TURN ENDS WHEN THE CAMERA SEES A ROW, not on a clock.
                if row.walls == "both" and abs(row.offset_frac) < 0.25:
                    reacq += 1
                else:
                    reacq = 0
                if reacq >= TURN_REACQUIRE:
                    passes_done += 1
                    no_plant = 0
                    state = "RUNNING"
                    print(f"[mission] row reacquired, pass {passes_done+1}")
                elif time.time() - turn_t0 > TURN_TIMEOUT_S:
                    state = "ROW_END"
                    link.send("$ROW_END")
                    row_end_sent = True
                    print("[mission] turn timed out, no row found -> stopping")
                else:
                    link.send(f"$T,{a.turn_dir}")

            elif state == "WAITING":
                if (esp in DRIVABLE_MODES or not link.ok) \
                        and not obstacle:
                    state = "RUNNING"
                    # Clear the counters. They have been ticking up while we
                    # sat in WAITING pointed at whatever happened to be in
                    # front of us, and carrying that count into RUNNING made
                    # the mission declare ROW_END before it had moved.
                    no_plant = 0
                    no_wall_frames = 0
                    print("[mission] running")

            elif state == "RUNNING":
                if obstacle:
                    state = "OBSTACLE"
                    link.send("$STOP")
                    print(f"[mission] obstacle ({occ:.2f})")
                elif diseased and (time.time() - last_hit) > DETECT_COOLDOWN \
                        and not is_dup(fix):
                    state = "STOPPING"
                    link.send("$STOP")
                    t_state = time.time()
                    no_wall_frames = 0
                    no_plant = 0
                    print("[mission] diseased leaf -> stopping")
                elif (a.scan_stops and present
                      and (time.time() - last_scan) > SCAN_COOLDOWN_S):
                    # A plant is beside us. Stop and look properly rather than
                    # hoping a frame lands on it at speed.
                    state = "SCANNING"
                    scan_t0 = time.time()
                    scan_frames = scan_hits = 0
                    link.send("$STOP")
                    print("[mission] plant alongside -> stopping to inspect")

            elif state == "SCANNING":
                link.send("$STOP")
                scan_frames += 1
                if diseased:
                    scan_hits += 1
                done_time = (time.time() - scan_t0) > a.scan_hold
                enough = scan_frames >= SCAN_MIN_FRAMES
                if done_time and enough:
                    last_scan = time.time()
                    if scan_hits >= max(2, SCAN_MIN_FRAMES // 4) \
                            and (time.time() - last_hit) > DETECT_COOLDOWN \
                            and not is_dup(fix):
                        state = "STOPPING"
                        t_state = time.time()
                        print(f"[mission] diseased ({scan_hits}/{scan_frames} "
                              f"frames agreed) -> capturing")
                    elif (RECENTRE_ENABLED and row.walls == "both"
                          and abs(row.offset_frac) > RECENTRE_TOL):
                        state = "RECENTRE"
                        recentre_t0 = time.time()
                        recentre_t = time.time()
                        recentre_pulse_on = True
                        print(f"[mission] healthy, but off centre by "
                              f"{row.offset_frac:+.3f} -> recentring")
                    else:
                        state = "RUNNING"
                        print(f"[mission] healthy ({scan_hits}/{scan_frames}) "
                              f"-> moving on")

            elif state == "RECENTRE":
                back = abs(row.offset_frac) <= RECENTRE_TOL
                gave_up = time.time() - recentre_t0 > RECENTRE_MAX_S
                if row.walls != "both":
                    # cannot measure the centre reliably; do not guess
                    state = "RUNNING"
                    print("[mission] lost both walls while recentring "
                          "-> carrying on")
                elif back or gave_up:
                    state = "RUNNING"
                    print(f"[mission] {'centred' if back else 'gave up'} at "
                          f"{row.offset_frac:+.3f} -> resuming")
                else:
                    # short bursts with a strong differential, re-measuring
                    # between each one
                    now_r = time.time()
                    if recentre_pulse_on and now_r - recentre_t > RECENTRE_PULSE_S:
                        recentre_pulse_on = False
                        recentre_t = now_r
                    elif not recentre_pulse_on and now_r - recentre_t > RECENTRE_GAP_S:
                        recentre_pulse_on = True
                        recentre_t = now_r
                    st = int(np.clip(row.steer * RECENTRE_GAIN + a.trim,
                                     -100, 100))
                    link.send(f"$S,{st}")
                    link.send("$GO" if recentre_pulse_on else "$STOP")

            elif state == "OBSTACLE":
                if not obstacle:
                    state = "RUNNING"
                    print("[mission] clear")

            elif state == "STOPPING":
                link.send("$STOP")
                if time.time() - t_state > CAPTURE_SETTLE_S:
                    state = "CAPTURING"

            elif state == "CAPTURING":
                link.send("$STOP")
                shot = picam.capture_array() if picam else None
                if shot is not None:
                    s2 = cv2.resize(shot, (800, 450))
                    r2 = D.detect(s2)
                    ov = D.overlay(s2, r2)
                    disease, conf = "abnormal_leaf", 1.0
                    if clf and clf.ok:
                        sx, sy = shot.shape[1] / 800.0, shot.shape[0] / 450.0
                        bl = [{"x": int(b["x"] * sx), "y": int(b["y"] * sy),
                               "w": int(b["w"] * sx), "h": int(b["h"] * sy)}
                              for b in r2.blobs]
                        disease, conf, _ = clf.predict(shot, bl)
                    rec = {
                        "disease": disease, "confidence": conf,
                        "severity": r2.severity, "ratio": r2.ratio,
                        "blobs": len(r2.blobs), "trusted": r2.trusted,
                        "sharpness": 0.0, "note": r2.note,
                        "lat": fix.lat if (fix and fix.fix_ok) else None,
                        "lon": fix.lon if (fix and fix.fix_ok) else None,
                        "sats": fix.sats if fix else 0,
                        "hdop": fix.hdop if fix else 99.9,
                        "pass_idx": 0, "odo_cm": 0.0, "t": time.time(),
                    }
                    path, confident = store.save(shot, ov, rec)
                    rec["image"] = path
                    map_server.add_pin(rec)
                    if fix and fix.fix_ok:
                        seen.append((fix.lat, fix.lon))
                    # Send everything, marking the doubtful ones. Holding back
                    # an uncertain detection means a real lesion can go
                    # unreported, which is the expensive error here. The
                    # caption says UNVERIFIED so nobody is misled.
                    if confident or not a.only_confident:
                        tg.detection(path, rec)
                    found += 1
                    print(f"  [{found}] {disease} {conf:.0%} {r2.severity} "
                          f"{r2.ratio:.1%} -> {os.path.basename(path)}")
                last_hit = time.time()
                state = "RESUMING"
                t_state = time.time()

            elif state == "RESUMING":
                link.send("$STOP")
                if time.time() - t_state > RESUME_DELAY_S:
                    state = "RUNNING"
                    no_wall_frames = 0

            # ---- drive -------------------------------------------------
            if state == "RUNNING":
                steer_out = int(np.clip(row.steer + a.trim, -100, 100))
                link.send(f"$S,{steer_out}")
                if not a.pulse:
                    link.send("$GO")
                else:
                    # Duty-cycle the motion. Full torque while moving, so the
                    # motors never sit below their stiction threshold, but a
                    # much lower average speed.
                    now_p = time.time()
                    if pulse_on and now_p - pulse_t0 > PULSE_ON_S:
                        pulse_on = False
                        pulse_t0 = now_p
                    elif not pulse_on and now_p - pulse_t0 > PULSE_OFF_S:
                        pulse_on = True
                        pulse_t0 = now_p
                    link.send("$GO" if pulse_on else "$STOP")
            elif state == "ROW_END":
                # send it once, not every frame -- the ESP32 acknowledges each
                # one and the console fills with ROW_END_ACK
                if not row_end_sent:
                    link.send("$ROW_END")
                    row_end_sent = True
            elif state == "SEARCHING":
                # keep moving on a decaying version of the last good steer
                coast_steer = int(coast_steer * ROW_COAST_STEER)
                link.send(f"$S,{int(np.clip(coast_steer + a.trim, -100, 100))}")
                now_p = time.time()
                if pulse_on and now_p - pulse_t0 > PULSE_ON_S:
                    pulse_on = False; pulse_t0 = now_p
                elif not pulse_on and now_p - pulse_t0 > PULSE_OFF_S:
                    pulse_on = True; pulse_t0 = now_p
                link.send("$GO" if (pulse_on or not a.pulse) else "$STOP")

            elif state in ("CLEARING", "TURNING", "SCANNING", "RECENTRE"):
                pass                       # those states send their own commands
            else:
                link.send("$STOP")

            # ---- publish ------------------------------------------------
            if fix:
                map_server.update_bot(lat=fix.lat or None, lon=fix.lon or None,
                                      fix_ok=fix.fix_ok, sats=fix.sats,
                                      state=state)
            hw = f"{row.half_width:.0f}px" if row.half_width else "not learned"
            gtxt = (f"{fix.sats} sats  hdop {fix.hdop}  fix {fix.fix_ok}"
                    if fix else "no gps")
            text = (
                f"state      {state}          esp32 {esp}\n"
                f"\n"
                f"ROW        walls {row.walls}   steer {row.steer:+d}"
                + (f" {a.trim:+d} trim = {row.steer + a.trim:+d}"
                   if a.trim else "")
                + f"   conf {row.conf:.2f}\n"
                f"           offset {row.offset_frac:+.3f} of width   "
                f"half-width {hw}\n"
                f"           {row.note}\n"
                f"OBSTACLE   {'YES' if obstacle else 'no'}   "
                f"occupied {occ:.2f}/{OBST_FRACTION:.2f}\n"
                f"PLANTS     {'YES' if present else 'NO'}   "
                f"green {gfrac:.2f}  blob {bfrac:.2f}   "
                f"row-end {no_plant}/{ROW_END_CONFIRM}\n"
                + (f"SEARCHING  looking for {time.time()-search_t0:.1f}s "
                   f"of {ROW_SEARCH_S:.0f}s   reacquire {search_reacq}/"
                   f"{ROW_REACQUIRE}\n" if state == "SEARCHING" else "")
                + f"TURNING    {'enabled' if a.turn_at_end else 'off'}   "
                f"pass {passes_done+1}/{a.passes}"
                + (f"   reacquire {reacq}/{TURN_REACQUIRE}"
                   if state == "TURNING" else "") + "\n"
                f"\n"
                + (f"RECENTRE   correcting {row.offset_frac:+.3f} toward "
                   f"{RECENTRE_TOL:.2f}   "
                   f"{time.time()-recentre_t0:.1f}s of {RECENTRE_MAX_S:.0f}s\n"
                   if state == "RECENTRE" else "")
                + f"SCAN       {'on' if a.scan_stops else 'off'}"
                + (f"   inspecting {scan_frames} frames, "
                   f"{scan_hits} say diseased" if state == "SCANNING" else "")
                + f"   pulse {'on' if a.pulse else 'off'}\n"
                f"DISEASE    {'YES' if diseased else 'no'}"
                + (f"   {res.severity} {res.ratio:.1%}  d_ref {res.d_ref:+.3f}"
                   f"  trusted {res.trusted}" if res else "")
                + f"\n"
                f"GPS        {gtxt}\n"
                f"FOUND      {found} leaves logged\n"
            )
            with _vlock:
                _views["webcam"] = wdbg
                _views["picam"] = pdbg
                _views["text"] = text

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        print("\nstopping...")
        for _ in range(5):
            link.send("$STOP")
            time.sleep(0.1)
        link.close()
        tg.session_summary(store.summary())
        tg.drain(timeout=25)
        if gps:
            gps.close()
        webcam.release()
        if picam:
            picam.stop()
        print(f"done. {found} detections.")


if __name__ == "__main__":
    main()
