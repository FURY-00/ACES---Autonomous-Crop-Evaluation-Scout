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
OBST_BRIGHT      = 160
OBST_FRACTION    = 0.35
OBST_CONFIRM     = 3
OBST_CLEAR       = 5

PLANT_STRIP      = 0.45
PLANT_D_MIN      = 0.05
PLANT_FRAC_MIN   = 0.10
PLANT_BLOB_MIN   = 0.06
ROW_END_CONFIRM  = 15

CAPTURE_SETTLE_S = 1.0      # let the chassis stop moving before the still
RESUME_DELAY_S   = 1.0
DETECT_COOLDOWN  = 6.0

STREAM_PORT      = S.STREAM_PORT
JPEG_Q           = 85


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
            self.ser = serial.Serial(port, baud, timeout=0.2)
            time.sleep(2.0)
            self.ok = True
            threading.Thread(target=self._rx, daemon=True).start()
            print(f"[esp32] connected on {port}")
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
def open_webcam(idx, width, height, fps):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"[webcam] cannot open index {idx}. Try --webcam 1")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[webcam] {aw}x{ah} @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps")
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
    ap.add_argument("--webcam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--proc-width", type=int, default=480)
    ap.add_argument("--jpeg", type=int, default=85)
    ap.add_argument("--no-drive", action="store_true")
    ap.add_argument("--no-telegram", action="store_true")
    ap.add_argument("--no-classify", action="store_true")
    a = ap.parse_args()
    JPEG_Q = a.jpeg
    if a.no_telegram:
        S.TELEGRAM_ENABLED = False

    link = Link(a.port, enabled=not a.no_drive)
    webcam = open_webcam(a.webcam, a.width, a.height, a.fps)
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
    print("\n  CH5 HIGH = autonomous,  CH5 LOW = manual RC")
    print("  ctrl-c stops the bot\n")

    state = "WAITING"
    obs_hits = obs_clear = no_plant = 0
    last_hit = 0.0
    t_state = time.time()
    found = 0
    seen = []

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

            no_plant = 0 if present else no_plant + 1
            row_over = no_plant >= ROW_END_CONFIRM

            diseased = (res is not None and res.trusted and res.blobs
                        and res.ratio >= S.DETECT_RATIO_MIN)

            esp = link.read_mode()

            # ---- state machine ----------------------------------------
            if state != "ROW_END" and row_over:
                state = "ROW_END"
                link.send("$ROW_END")
                print("[mission] row ended")

            elif state == "WAITING":
                if (esp in ("AUTO_IDLE", "AUTO_RUN") or not link.ok) \
                        and not obstacle:
                    state = "RUNNING"
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
                    print("[mission] diseased leaf -> stopping")

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
                    if confident:
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

            # ---- drive -------------------------------------------------
            if state == "RUNNING":
                link.send(f"$S,{row.steer}")
                link.send("$GO")
            elif state == "ROW_END":
                link.send("$ROW_END")
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
                f"ROW        walls {row.walls}   steer {row.steer:+d}   "
                f"conf {row.conf:.2f}\n"
                f"           offset {row.offset_frac:+.3f} of width   "
                f"half-width {hw}\n"
                f"           {row.note}\n"
                f"OBSTACLE   {'YES' if obstacle else 'no'}   "
                f"occupied {occ:.2f}/{OBST_FRACTION:.2f}\n"
                f"PLANTS     {'YES' if present else 'NO'}   "
                f"green {gfrac:.2f}  blob {bfrac:.2f}   "
                f"row-end {no_plant}/{ROW_END_CONFIRM}\n"
                f"\n"
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
