"""
ACES autonomous run — webcam navigates the row, Pi Camera watches for plants.

The logic
---------
The WEBCAM does two jobs from one frame:

  * ROW FOLLOWING. Crop grows on both sides, the furrow does not. Find the
    inner edge of each crop wall, put the centreline between them, and steer
    so that centreline sits in the middle of the image. If one wall slides
    out of frame we have drifted toward the OTHER one, and the remembered
    row half-width still gives us a real setpoint to steer back to.

  * OBSTACLES. Something filling the centre of the lower half is in the way.

The PI CAMERA looks 45 degrees left and answers one question: are there still
plants beside us, close enough to scan? When there are not, for several
consecutive frames, the row has ended.

    webcam -> steer to centre     -> $S,<steer>
    webcam -> obstacle ahead?     -> $STOP / $GO
    pi cam -> plants still there? -> $ROW_END

Everything else lives on the ESP32, which is a motor controller with a kill
switch and nothing more.

Run it
------
    python3 run_auto.py                       # ESP32 on /dev/ttyUSB0
    python3 run_auto.py --no-link             # cameras only, motors dead
    python3 run_auto.py --webcam 1            # if the webcam is not video0

    http://<pi-ip>:8080   both views, live, with every number on screen

CH5 HIGH on the transmitter = autonomous.  CH5 LOW = manual RC, instantly.
This script cannot override that; the firmware enforces it.
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from navigation.row_vision import RowFollower

# ---------------------------------------------------------------- tunables
# Every one of these is meant to be changed in the field. The browser view
# prints the live value next to each, so you can see what to move.

# --- obstacle (webcam) ---
OBST_X0, OBST_X1 = 0.34, 0.66   # centre band, fraction of width
OBST_Y0          = 0.45         # from here down, fraction of height
OBST_BRIGHT      = 160          # absolute grey level counted as "close object"
OBST_FRACTION    = 0.35         # this much of the band occupied -> obstacle
OBST_CONFIRM     = 3            # consecutive frames before believing it
OBST_CLEAR       = 5            # consecutive clear frames before resuming

# --- plants (pi camera) ---
# The Pi Cam is angled left at the row. "Plants present" means: enough green
# tissue, AND that tissue is big enough in frame to be within ~30-50 cm.
PLANT_STRIP      = 0.45         # leftmost fraction of frame to examine
PLANT_D_MIN      = 0.05         # (G-R)/(R+G+B) above this counts as plant
PLANT_FRAC_MIN   = 0.10         # this much of the strip must be plant
PLANT_BLOB_MIN   = 0.06         # largest connected plant blob, as frac of strip
                                # this is the "close enough" test: a distant
                                # row is many small specks, a near plant is
                                # one large mass
ROW_END_CONFIRM  = 15           # consecutive plant-free frames -> row over

STREAM_PORT      = 8080


# ---------------------------------------------------------------- link
class Link:
    """Talks to the ESP32. Degrades to a no-op if it is not there."""

    def __init__(self, port, baud=115200, enabled=True):
        self.ok = False
        self.mode = "no-link"
        self.events = []
        self._lock = threading.Lock()
        if not enabled:
            print("[esp32] disabled (--no-link): motors are never commanded")
            return
        try:
            import serial
            self.ser = serial.Serial(port, baud, timeout=0.2)
            time.sleep(2.0)          # the ESP32 resets when the port opens
            self.ok = True
            threading.Thread(target=self._rx, daemon=True).start()
            print(f"[esp32] connected on {port}")
        except Exception as e:
            print(f"[esp32] NOT connected ({e}) — cameras only")

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
                with self._lock:
                    self.events.append(ln[3:])
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
def open_webcam(idx, width=1280, height=720, fps=30):
    """
    Open the webcam at the requested resolution.

    THE MJPG LINE IS THE IMPORTANT ONE. Over USB 2.0 a webcam streaming raw
    YUYV can only manage about 640x480 at 30 fps -- the bus runs out of
    bandwidth. Most cameras will silently give you a lower resolution than you
    asked for rather than telling you. Switching the camera to MJPG (it
    compresses on-board) is what actually unlocks 720p and above.

    Note what we get back, not what we asked for. Cameras lie about this
    constantly, so the code prints the ACTUAL negotiated mode.
    """
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"[webcam] cannot open index {idx}")
        print("         try:  ls /dev/video*   then pass --webcam N")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # always take the freshest frame

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    af = cap.get(cv2.CAP_PROP_FPS)
    cc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((cc >> 8 * i) & 0xFF) for i in range(4))
    print(f"[webcam] index {idx}: asked {width}x{height}, got {aw}x{ah} "
          f"@ {af:.0f}fps  {fourcc}")
    if (aw, ah) != (width, height):
        print("         the camera refused that mode. Run:")
        print(f"           v4l2-ctl -d /dev/video{idx} --list-formats-ext")
        print("         to see what it actually supports.")
    return cap


def open_picam():
    try:
        from picamera2 import Picamera2
        pc = Picamera2()
        pc.configure(pc.create_preview_configuration(
            main={"size": (640, 360), "format": "RGB888"}))
        pc.start()
        time.sleep(2.0)
        try:
            from libcamera import controls
            # Fixed focus at the row distance. Autofocus would hunt while
            # driving and every frame would be differently sharp.
            pc.set_controls({"AfMode": controls.AfModeEnum.Manual,
                             "LensPosition": 100.0 / 40.0})   # ~40 cm
            time.sleep(1.0)
        except Exception as e:
            print(f"[picam] focus: {e}")
        print("[picam] Pi Camera open, focus fixed at ~40 cm")
        return pc
    except Exception as e:
        print(f"[picam] unavailable ({e}) — row-end detection OFF")
        return None


# ---------------------------------------------------------------- vision
def check_obstacle(frame):
    """
    Is something in the way?

    We look only at the centre band of the lower half — that is the patch of
    ground the bot is about to drive over. A close object is brighter (more
    reflected light) and has strong edges; open ground ahead is neither.

    Returns (is_obstacle, occupied_fraction, annotated_frame).
    """
    h, w = frame.shape[:2]
    x0, x1 = int(w * OBST_X0), int(w * OBST_X1)
    y0 = int(h * OBST_Y0)
    band = frame[y0:, x0:x1]

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, OBST_BRIGHT, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    occupied = float(cv2.bitwise_or(bright, edges).mean()) / 255.0

    is_obs = occupied > OBST_FRACTION

    dbg = frame.copy()
    col = (0, 0, 255) if is_obs else (0, 220, 0)
    cv2.rectangle(dbg, (x0, y0), (x1, h), col, 2)
    cv2.rectangle(dbg, (0, 0), (w, 18), (0, 0, 0), -1)
    cv2.putText(dbg, f"{'OBSTACLE' if is_obs else 'clear'}  "
                     f"{occupied:.2f}/{OBST_FRACTION:.2f}",
                (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
    return is_obs, occupied, dbg


def check_plants(frame):
    """
    Are there plants beside us, close enough to scan?

    Two conditions, and the second is what gives you the 30-50 cm range
    without any distance sensor:

      1. enough green pixels in the left strip                  -> a row exists
      2. the LARGEST connected green blob is big enough         -> it is NEAR

    A row two metres away breaks into many small specks in frame. A plant at
    40 cm is one large continuous mass. Thresholding the biggest blob is a
    crude but reliable proxy for distance with a single fixed camera.

    Returns (present, green_frac, blob_frac, annotated_frame).
    """
    if frame is None:
        return True, 0.0, 0.0, np.zeros((180, 320, 3), np.uint8)

    h, w = frame.shape[:2]
    x1 = int(w * PLANT_STRIP)
    strip = frame[int(h * 0.15):int(h * 0.9), :x1]

    b, g, r = cv2.split(strip.astype(np.float32))
    d = (g - r) / (r + g + b + 1e-6)
    mask = (d > PLANT_D_MIN).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    total = mask.shape[0] * mask.shape[1]
    green_frac = float(mask.sum()) / 255.0 / total

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    blob_frac = 0.0
    if n > 1:
        blob_frac = float(stats[1:, cv2.CC_STAT_AREA].max()) / total

    present = (green_frac >= PLANT_FRAC_MIN) and (blob_frac >= PLANT_BLOB_MIN)

    dbg = cv2.resize(frame, (320, 180))
    dh, dw = dbg.shape[:2]
    sx1 = int(dw * PLANT_STRIP)
    col = (0, 220, 0) if present else (0, 0, 255)
    overlay = cv2.resize(mask, (sx1, int(dh * 0.75)))
    region = dbg[int(dh * 0.15):int(dh * 0.15) + overlay.shape[0], :sx1]
    region[overlay > 0] = (0.5 * region[overlay > 0]
                           + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.rectangle(dbg, (0, int(dh * 0.15)), (sx1, int(dh * 0.9)), col, 2)
    cv2.rectangle(dbg, (0, 0), (dw, 16), (0, 0, 0), -1)
    cv2.putText(dbg, f"{'PLANTS' if present else 'NO PLANTS'}  "
                     f"g{green_frac:.2f} b{blob_frac:.2f}",
                (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
    return present, green_frac, blob_frac, dbg


# ---------------------------------------------------------------- stream
JPEG_Q = 85
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
<title>ACES autonomous</title>
<style>
:root{--loam:#12160f;--panel:#1b2116;--rule:#2f3a26;--crop:#7fb069;
      --ink:#e8ece3;--dim:#8b9680;
      --mono:ui-monospace,"DejaVu Sans Mono",Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--loam);color:var(--ink);font:13px/1.5 var(--mono)}
header{padding:11px 15px;border-bottom:1px solid var(--rule);background:var(--panel)}
h1{margin:0;font-size:12px;letter-spacing:.26em;font-weight:700}
main{display:grid;gap:1px;background:var(--rule);grid-template-columns:1fr}
@media(min-width:760px){main{grid-template-columns:1fr 1fr}}
figure{margin:0;background:var(--panel);padding:12px}
figcaption{font-size:10px;letter-spacing:.14em;color:var(--dim);margin-bottom:7px}
img{width:100%;display:block;border:1px solid var(--rule);background:#000}
pre{margin:0;padding:13px 15px;background:var(--panel);
    border-top:1px solid var(--rule);white-space:pre-wrap;font-size:12.5px}
</style>
<header><h1>ACES &mdash; AUTONOMOUS RUN</h1></header>
<main>
 <figure><figcaption>WEBCAM &mdash; row centreline &amp; obstacles</figcaption>
  <img src="/webcam.mjpg" alt="Forward webcam with row edges and centreline"></figure>
 <figure><figcaption>PI CAMERA &mdash; plants beside us</figcaption>
  <img src="/picam.mjpg" alt="Side camera with plant mask"></figure>
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
        print("[stream] flask missing — no browser view. pip install flask")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--webcam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--proc-width", type=int, default=480,
                    help="detection runs at this width. Row following gains "
                         "nothing above ~480 and costs CPU; the DISPLAY stays "
                         "full resolution either way.")
    ap.add_argument("--jpeg", type=int, default=85,
                    help="stream JPEG quality 1-100")
    ap.add_argument("--no-link", action="store_true")
    a = ap.parse_args()

    global JPEG_Q
    JPEG_Q = a.jpeg

    link = Link(a.port, enabled=not a.no_link)
    webcam = open_webcam(a.webcam, a.width, a.height, a.fps)
    picam = open_picam()
    rows = RowFollower()
    start_stream()

    print(f"\n  open  http://<pi-ip>:{STREAM_PORT}")
    print("  CH5 HIGH = autonomous,  CH5 LOW = manual RC")
    print("  ctrl-c stops the bot\n")

    state = "WAITING"        # WAITING | RUNNING | OBSTACLE | ROW_END
    obs_hits = obs_clear = 0
    no_plant = 0
    last_tx = 0.0

    try:
        while True:
            ok, wframe_full = webcam.read()
            if not ok:
                time.sleep(0.05)
                continue

            # Detection runs small, display stays big. The column-profile
            # method gains nothing from extra pixels -- it averages whole
            # columns anyway -- but a 1280-wide ExG + morphology pass on a
            # Pi 4 would eat your frame rate for no benefit.
            if wframe_full.shape[1] > a.proc_width:
                sc = a.proc_width / wframe_full.shape[1]
                wframe = cv2.resize(wframe_full, None, fx=sc, fy=sc,
                                    interpolation=cv2.INTER_AREA)
            else:
                sc = 1.0
                wframe = wframe_full

            is_obs, occ, _ = check_obstacle(wframe)
            row = rows.update(wframe)          # steering from the same frame
            # upscale the annotated view back for display
            wdbg = cv2.resize(row.debug, (wframe_full.shape[1],
                                          wframe_full.shape[0]),
                              interpolation=cv2.INTER_NEAREST) if sc != 1.0 \
                else row.debug
            # draw the obstacle band on top of the row-following view
            h, w = wdbg.shape[:2]
            ox0, ox1 = int(w * OBST_X0), int(w * OBST_X1)
            oy0 = int(h * OBST_Y0)
            cv2.rectangle(wdbg, (ox0, oy0), (ox1, h),
                          (0, 0, 255) if is_obs else (90, 90, 90), 1)

            pframe = picam.capture_array() if picam else None
            present, gfrac, bfrac, pdbg = check_plants(pframe)

            # ---- debounce both signals ------------------------------------
            if is_obs:
                obs_hits += 1
                obs_clear = 0
            else:
                obs_clear += 1
                if obs_clear >= OBST_CLEAR:
                    obs_hits = 0
            obstacle = obs_hits >= OBST_CONFIRM

            if present:
                no_plant = 0
            else:
                no_plant += 1
            row_over = no_plant >= ROW_END_CONFIRM

            esp = link.read_mode()

            # ---- state machine --------------------------------------------
            if state != "ROW_END" and row_over:
                state = "ROW_END"
                link.send("$ROW_END")
                print("[nav] no plants -> ROW_END")

            elif state == "WAITING":
                # start only once the ESP32 reports it is in AUTO (CH5 high)
                if esp in ("AUTO_IDLE", "AUTO_RUN") or not link.ok:
                    if not obstacle:
                        state = "RUNNING"
                        print("[nav] starting")

            elif state == "RUNNING" and obstacle:
                state = "OBSTACLE"
                link.send("$STOP")
                print(f"[nav] obstacle ({occ:.2f}) -> STOP")

            elif state == "OBSTACLE" and not obstacle:
                state = "RUNNING"
                print("[nav] clear -> GO")

            # ---- command the ESP32, and keep the watchdog fed --------------
            now = time.time()
            if now - last_tx > 0.1:
                last_tx = now
                if state == "RUNNING":
                    link.send(f"$S,{row.steer}")
                    link.send("$GO")
                elif state == "ROW_END":
                    link.send("$ROW_END")
                else:
                    link.send("$STOP")

            # ---- publish ---------------------------------------------------
            hw = f"{row.half_width:.0f}px" if row.half_width else "not learned"
            text = (
                f"state      {state}\n"
                f"esp32      {esp}\n"
                f"\n"
                f"ROW        walls {row.walls}   steer {row.steer:+d}   "
                f"conf {row.conf:.2f}\n"
                f"           offset {row.offset_px:+.0f}px "
                f"({row.offset_frac:+.3f} of width)   "
                f"heading {row.heading_px:+.0f}px   half-width {hw}\n"
                f"           {row.note}\n"
                f"\n"
                f"OBSTACLE   {'YES' if obstacle else 'no'}   "
                f"occupied {occ:.2f} / {OBST_FRACTION:.2f}   "
                f"({obs_hits}/{OBST_CONFIRM} hits)\n"
                f"PLANTS     {'YES' if present else 'NO'}    "
                f"green {gfrac:.2f}/{PLANT_FRAC_MIN:.2f}   "
                f"nearest blob {bfrac:.2f}/{PLANT_BLOB_MIN:.2f}\n"
                f"row-end    {no_plant}/{ROW_END_CONFIRM} plant-free frames\n"
                f"\n"
                f"weaving          -> lower kp_offset in navigation/row_vision.py\n"
                f"drifts and stays -> raise kp_offset\n"
                f"cuts corners     -> raise kp_heading\n"
                f"walls never seen -> lower veg_min_col"
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
        webcam.release()
        if picam:
            picam.stop()
        print("done.")


if __name__ == "__main__":
    main()
