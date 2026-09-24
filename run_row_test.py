"""
ACES row-following test — no receiver, no GPS, no disease detection.

One job: drive down the middle of the track, and stop when the green walls
run out.

    webcam -> both walls visible -> steer to the centreline, keep going
           -> one wall visible   -> steer back toward the middle
           -> no walls for N frames -> STOP, done

Run it
------
    python3 run_row_test.py                      # ESP32 on /dev/ttyACM0
    python3 run_row_test.py --no-drive           # vision only, motors dead
    python3 run_row_test.py --port /dev/ttyUSB0
    python3 run_row_test.py --webcam 1

    http://<pi-ip>:8080   live view, with every number that matters

STOPPING IT
-----------
Ctrl-C. The ESP32 also stops on its own if this script goes quiet for one
second, so closing the terminal, killing the SSH session or unplugging the
USB all stop the bot. With no transmitter, that watchdog is your kill switch —
keep the terminal in front of you.
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from navigation.row_vision import RowFollower       # noqa: E402

# ---------------------------------------------------------------- tunables
NO_WALL_STOP   = 15      # consecutive frames with no wall -> stop
ONE_WALL_MAX   = 60      # frames on a single wall before giving up
START_DELAY_S  = 3.0     # countdown before it moves, so you can step back
STREAM_PORT    = 8080
JPEG_Q         = 85



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
            print(f"[esp32] NOT connected ({e})")
            print("        check:  ls /dev/ttyACM* /dev/ttyUSB*")

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
            for _ in range(5):
                self.send("$STOP")
                time.sleep(0.05)
            self.ser.close()


# ---------------------------------------------------------------- camera
def open_webcam(idx, width, height, fps):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"[webcam] cannot open index {idx}")
        print("         ls /dev/video*   then try --webcam 1")
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


# ---------------------------------------------------------------- stream
_views = {"cam": None, "text": "starting"}
_vlock = threading.Lock()
try:
    from flask import Flask, Response
    _app = Flask(__name__)
    HAVE_FLASK = True
except ImportError:
    HAVE_FLASK = False

if HAVE_FLASK:
    @_app.route("/cam.mjpg")
    def _c():
        def gen():
            while True:
                with _vlock:
                    f = _views["cam"]
                if f is not None:
                    ok, buf = cv2.imencode(".jpg", f,
                                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                    if ok:
                        yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n"
                               + buf.tobytes() + b"\r\n")
                time.sleep(0.07)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

    @_app.route("/readout")
    def _r():
        with _vlock:
            return _views["text"]

    @_app.route("/")
    def _i():
        return """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ACES row test</title>
<style>
:root{--loam:#12160f;--panel:#1b2116;--rule:#2f3a26;--crop:#7fb069;
      --ink:#e8ece3;--dim:#8b9680;
      --mono:ui-monospace,"DejaVu Sans Mono",Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--loam);color:var(--ink);font:13px/1.55 var(--mono)}
header{padding:11px 15px;border-bottom:1px solid var(--rule);background:var(--panel)}
h1{margin:0;font-size:12px;letter-spacing:.26em;font-weight:700}
main{padding:14px}
img{width:100%;display:block;border:1px solid var(--rule);background:#000}
pre{margin:14px 0 0;padding:13px 15px;background:var(--panel);
  border:1px solid var(--rule);white-space:pre-wrap;font-size:12.5px}
</style>
<header><h1>ACES &mdash; ROW FOLLOWING TEST</h1></header>
<main>
 <img src="/cam.mjpg" alt="Webcam with detected row edges and centreline">
 <pre id="r">connecting</pre>
</main>
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
        print("[stream] flask missing.  pip install flask")


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
    ap.add_argument("--delay", type=float, default=START_DELAY_S)
    a = ap.parse_args()
    JPEG_Q = a.jpeg

    port = a.port
    if not a.no_drive:
        found = find_esp32_port(a.port)
        if found:
            port = found
    link = Link(port, enabled=not a.no_drive)
    cam = open_webcam(a.webcam, a.width, a.height, a.fps)
    rows = RowFollower()
    start_stream()

    print(f"\n  view: http://<pi-ip>:{STREAM_PORT}")
    print("  ctrl-c stops the bot\n")

    state = "STARTING"
    no_wall = one_wall = 0
    t0 = time.time()

    try:
        while True:
            ok, full = cam.read()
            if not ok:
                time.sleep(0.05)
                continue

            if full.shape[1] > a.proc_width:
                sc = a.proc_width / full.shape[1]
                small = cv2.resize(full, None, fx=sc, fy=sc,
                                   interpolation=cv2.INTER_AREA)
            else:
                small = full
            row = rows.update(small)
            dbg = cv2.resize(row.debug, (full.shape[1], full.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

            # ---- wall bookkeeping ---------------------------------------
            if row.walls == "both":
                no_wall = one_wall = 0
            elif row.walls in ("left", "right"):
                one_wall += 1
                no_wall = 0
            else:
                no_wall += 1

            # ---- state ---------------------------------------------------
            if state == "STARTING":
                left = a.delay - (time.time() - t0)
                if left <= 0:
                    if row.walls == "none":
                        # Do not start blind. If it cannot see the track at
                        # the start it will not find it by driving forward.
                        state = "NO_TRACK"
                    else:
                        state = "RUNNING"
                        print("[row] running")

            elif state == "RUNNING":
                if no_wall >= NO_WALL_STOP:
                    state = "END_OF_TRACK"
                    print(f"[row] no walls for {NO_WALL_STOP} frames -> stopping")
                elif one_wall >= ONE_WALL_MAX:
                    state = "LOST"
                    print("[row] stuck on one wall too long -> stopping")

            elif state == "NO_TRACK":
                if row.walls != "none":
                    state = "RUNNING"
                    print("[row] track found -> running")

            # ---- command ---------------------------------------------------
            if state == "RUNNING":
                link.send(f"$S,{row.steer}")
                link.send("$GO")
            elif state in ("END_OF_TRACK", "LOST"):
                link.send("$ROW_END")
            else:
                link.send("$STOP")

            # ---- publish -----------------------------------------------------
            hw = f"{row.half_width:.0f}px" if row.half_width else "not learned"
            le = f"{row.left_edge:.0f}" if row.left_edge is not None else "-"
            re_ = f"{row.right_edge:.0f}" if row.right_edge is not None else "-"
            ce = f"{row.center:.0f}" if row.center is not None else "-"
            countdown = ""
            if state == "STARTING":
                countdown = f"   starting in {max(0, a.delay-(time.time()-t0)):.1f}s"

            text = (
                f"state      {state}{countdown}\n"
                f"esp32      {link.read_mode()}\n"
                f"\n"
                f"walls      {row.walls}     conf {row.conf:.2f}\n"
                f"edges      left {le}   right {re_}   centre {ce}   "
                f"(image centre {small.shape[1]//2})\n"
                f"half-width {hw}\n"
                f"offset     {row.offset_frac:+.3f} of width\n"
                f"STEER      {row.steer:+d}   "
                f"{'<<< LEFT' if row.steer < -3 else 'RIGHT >>>' if row.steer > 3 else 'straight'}\n"
                f"{row.note}\n"
                f"\n"
                f"no-wall {no_wall}/{NO_WALL_STOP}    one-wall {one_wall}/{ONE_WALL_MAX}\n"
                f"\n"
                f"walls never seen -> lower veg_min_col in navigation/row_vision.py\n"
                f"weaving          -> lower kp_offset\n"
                f"drifts and stays -> raise kp_offset\n"
                f"too fast         -> lower CRUISE in the .ino"
            )
            with _vlock:
                _views["cam"] = dbg
                _views["text"] = text

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        print("\nstopping...")
        link.close()
        cam.release()
        print("done.")


if __name__ == "__main__":
    main()
