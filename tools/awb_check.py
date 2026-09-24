"""
White-balance diagnostic, with live preview and lock/unlock buttons.

No Flask -- only Python's standard library http.server, because the Flask
views kept hanging on the Pi.

WHY THIS EXISTS
---------------
auto_calibrate.py locks white balance at the start of a session so colour
does not drift between frames. Right in principle -- but if you start it
POINTING AT PLANTS, auto white balance has already decided the scene is "too
green" and boosted red and blue to cancel it. Those wrong gains then get
frozen for the whole session, and every leaf reads far less green than it is.

Your calibration set peaked at d = +0.085 where living foliage should be
+0.15 to +0.30. This tool tells you whether that is the cause.

Usage
-----
    python3 tools/awb_check.py
    python3 tools/awb_check.py --port 8000

Open  http://<pi-ip>:8000

THE EXPERIMENT
--------------
    1. point at a green leaf on AUTO, note the number
    2. tap LOCK while still pointing at the leaf
    3. the number should DROP  <- that is the bug
    4. tap AUTO, then point at grey concrete or white paper
    5. tap LOCK, then point back at the leaf
    6. the number should now stay HIGH

Locking on grey is what a photographer's grey card does, and for the same
reason: the camera cannot tell "this scene IS green" from "this scene has a
green cast I should remove".
"""

import argparse
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_st = {"jpg": None, "text": "starting", "cmd": None}
_lock = threading.Lock()


def greenness(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    return float(np.percentile((g - r) / (r + g + b + 1e-6), 75))


PAGE = b"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ACES white balance</title>
<style>
:root{--loam:#12160f;--panel:#1b2116;--rule:#2f3a26;--crop:#7fb069;--ink:#e8ece3}
*{box-sizing:border-box}
body{margin:0;background:var(--loam);color:var(--ink);
 font:13px/1.55 ui-monospace,"DejaVu Sans Mono",monospace}
header{padding:11px 15px;background:var(--panel);border-bottom:1px solid var(--rule)}
h1{margin:0;font-size:12px;letter-spacing:.26em}
main{padding:14px}
img{width:100%;display:block;border:1px solid var(--rule);background:#000}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}
button{padding:20px 8px;font:inherit;font-size:12px;letter-spacing:.14em;
 font-weight:700;border-radius:3px;cursor:pointer;
 background:var(--panel);color:var(--ink);border:1px solid var(--rule)}
button.go{background:var(--crop);color:var(--loam);border:0}
button:active{opacity:.7}
pre{margin:14px 0 0;padding:13px 15px;background:var(--panel);
 border:1px solid var(--rule);white-space:pre-wrap}
</style>
<header><h1>ACES &mdash; WHITE BALANCE CHECK</h1></header>
<main>
 <img src="/stream" alt="Live camera preview">
 <div class="btns">
  <button onclick="c('auto')">BACK TO AUTO</button>
  <button class="go" onclick="c('lock')">LOCK ON THIS SCENE</button>
 </div>
 <pre id="t">loading</pre>
</main>
<script>
function c(x){fetch('/cmd?'+x);}
setInterval(async()=>{try{
 document.getElementById('t').textContent=await(await fetch('/text')).text();
}catch(e){}},350);
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        jpg = _st["jpg"]
                    if jpg:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg + b"\r\n")
                    time.sleep(0.08)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path.startswith("/cmd"):
            cmd = self.path.split("?", 1)[1] if "?" in self.path else ""
            with _lock:
                _st["cmd"] = cmd
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
        elif self.path.startswith("/text"):
            with _lock:
                t = _st["text"].encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(t)))
            self.end_headers()
            self.wfile.write(t)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    from picamera2 import Picamera2
    pc = Picamera2()
    pc.configure(pc.create_preview_configuration(
        main={"size": (640, 360), "format": "RGB888"}))
    pc.start()
    time.sleep(2.5)
    try:
        from libcamera import controls
        pc.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        time.sleep(1.0)
    except Exception:
        pass

    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"\n  open  http://<pi-ip>:{a.port}")
    print("  point at a green leaf and watch the greenness number")
    print("  the number is also printed here, once a second\n")

    locked = None
    last = 0.0
    try:
        while True:
            with _lock:
                cmd = _st["cmd"]
                _st["cmd"] = None
            if cmd == "auto":
                pc.set_controls({"AwbEnable": True})
                locked = None
                print("\n[awb] back to AUTO")
            elif cmd == "lock":
                md = pc.capture_metadata()
                g = md.get("ColourGains", (1.8, 1.8))
                pc.set_controls({"AwbEnable": False, "ColourGains": g})
                locked = tuple(round(x, 2) for x in g)
                print(f"\n[awb] LOCKED at {locked}")

            f = pc.capture_array()
            d = greenness(f)
            md = pc.capture_metadata()
            gains = md.get("ColourGains", (0, 0))

            verdict = ("LOOKS RIGHT" if d >= 0.15
                       else "LOW - green being cancelled" if d >= 0.06
                       else "VERY LOW")
            col = (0, 220, 0) if d >= 0.15 else (0, 140, 255)

            disp = f.copy()
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(disp, f"greenness {d:+.3f}   "
                              f"{'LOCKED' if locked else 'AUTO'}",
                        (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with _lock:
                    _st["jpg"] = buf.tobytes()
                    _st["text"] = (
                        f"greenness   {d:+.3f}   {verdict}\n"
                        f"            foliage should read +0.15 to +0.30\n"
                        f"\n"
                        f"awb         {'LOCKED at ' + str(locked) if locked else 'AUTO'}\n"
                        f"gains       R/G {gains[0]:.2f}   B/G {gains[1]:.2f}\n"
                        f"\n"
                        f"1. point at a green leaf on AUTO, note the number\n"
                        f"2. tap LOCK while still on the leaf\n"
                        f"3. the number should DROP  <- that is the bug\n"
                        f"4. tap AUTO, point at grey concrete or white paper\n"
                        f"5. tap LOCK, then point back at the leaf\n"
                        f"6. the number should now stay HIGH")
            if time.time() - last > 1.0:
                last = time.time()
                print(f"  greenness {d:+.3f}  "
                      f"gains R/G {gains[0]:.2f} B/G {gains[1]:.2f}  "
                      f"{'LOCKED' if locked else 'AUTO  '}  {verdict}      ",
                      end="\r")
            time.sleep(0.07)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        pc.stop()
        print("\nstopped.")


if __name__ == "__main__":
    main()
