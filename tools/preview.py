"""
Dead-simple camera preview. No Flask, no dependencies beyond picamera2.

Built because the Flask-based views kept hanging. This uses only Python's
standard library http.server, so there is one less thing that can go wrong.

Usage
-----
    python3 tools/preview.py                 # browser stream on :8000
    python3 tools/preview.py --port 8000
    python3 tools/preview.py --snap out.jpg  # just save ONE photo and exit
    python3 tools/preview.py --usb           # use the USB webcam instead

Then open  http://<pi-ip>:8000

The greenness number is drawn on the image and printed to the terminal, so
you can check white balance even if the browser refuses to cooperate.

    living foliage should read  +0.15 to +0.30
    below +0.10 means the green is being cancelled out
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

_latest = {"jpg": None, "text": ""}
_lock = threading.Lock()


def greenness(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    return float(np.percentile((g - r) / (r + g + b + 1e-6), 75))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass                                    # keep the terminal readable

    def do_GET(self):
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        jpg = _latest["jpg"]
                    if jpg:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.08)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path.startswith("/snap.jpg"):
            with _lock:
                jpg = _latest["jpg"] or b""
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
        elif self.path.startswith("/text"):
            with _lock:
                t = _latest["text"].encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(t)
        else:
            page = b"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ACES preview</title>
<style>
body{margin:0;background:#12160f;color:#e8ece3;
 font:13px/1.55 ui-monospace,monospace}
header{padding:11px 15px;background:#1b2116;border-bottom:1px solid #2f3a26}
h1{margin:0;font-size:12px;letter-spacing:.26em}
main{padding:14px}
img{width:100%;display:block;border:1px solid #2f3a26;background:#000}
pre{margin:14px 0 0;padding:13px 15px;background:#1b2116;
 border:1px solid #2f3a26;white-space:pre-wrap}
</style>
<header><h1>ACES &mdash; CAMERA PREVIEW</h1></header>
<main>
 <img src="/stream" alt="Live camera preview">
 <pre id="t">loading</pre>
</main>
<script>
setInterval(async()=>{try{
 document.getElementById('t').textContent=await(await fetch('/text')).text();
}catch(e){}},400);
</script>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)


def open_cam(usb):
    if not usb:
        try:
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
            print("[cam] Pi Camera")
            return ("picam", pc)
        except Exception as e:
            print(f"[cam] Pi Camera failed ({e}), trying USB")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    if not cap.isOpened():
        print("[cam] no camera at all")
        sys.exit(1)
    print("[cam] USB webcam")
    return ("cv2", cap)


def grab(cam):
    kind, dev = cam
    if kind == "picam":
        return dev.capture_array()
    ok, f = dev.read()
    return f if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--usb", action="store_true")
    ap.add_argument("--snap", help="save one photo to this path and exit")
    a = ap.parse_args()

    cam = open_cam(a.usb)

    if a.snap:
        for _ in range(5):
            f = grab(cam)
            time.sleep(0.2)
        if f is not None:
            cv2.imwrite(a.snap, f)
            print(f"saved {a.snap}   greenness {greenness(f):+.3f}")
        if cam[0] == "picam":
            cam[1].stop()
        else:
            cam[1].release()
        return

    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"\n  open  http://<pi-ip>:{a.port}")
    print(f"  or a single frame:  http://<pi-ip>:{a.port}/snap.jpg")
    print("  ctrl-c to stop\n")

    last_print = 0.0
    try:
        while True:
            f = grab(cam)
            if f is None:
                time.sleep(0.1)
                continue
            d = greenness(f)
            V = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:, :, 2]
            blown = float((V > 250).mean())
            sharp = float(cv2.Laplacian(
                cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

            verdict = ("looks right" if d >= 0.15
                       else "LOW - green being cancelled" if d >= 0.06
                       else "VERY LOW")
            disp = f.copy()
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(disp, f"greenness {d:+.3f}   sharp {sharp:.0f}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 220, 0) if d >= 0.15 else (0, 140, 255), 2)
            ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with _lock:
                    _latest["jpg"] = buf.tobytes()
                    _latest["text"] = (
                        f"greenness  {d:+.3f}   {verdict}\n"
                        f"           foliage should read +0.15 to +0.30\n"
                        f"sharpness  {sharp:6.0f}\n"
                        f"blown out  {blown:.1%}")
            if time.time() - last_print > 1.0:
                last_print = time.time()
                print(f"  greenness {d:+.3f}  sharp {sharp:6.0f}  "
                      f"blown {blown:.1%}   {verdict}      ", end="\r")
            time.sleep(0.06)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        if cam[0] == "picam":
            cam[1].stop()
        else:
            cam[1].release()
        print("\nstopped.")


if __name__ == "__main__":
    main()
