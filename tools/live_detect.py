"""
STEP 2: point the camera at a leaf and watch it detect. Nothing else.

No motors, no GPS, no ESP32, no classifier, no Telegram. Just: camera in,
mask out. This is the tool to use while you tune.

It streams to a browser at http://<pi-ip>:8080, so it works over SSH with
no display attached. If you ARE on the Pi's own screen, add --window for a
local OpenCV window too.

Usage
-----
    python tools/live_detect.py                 # stream only
    python tools/live_detect.py --window        # + local window
    python tools/live_detect.py --save          # press SPACE / click Save to
                                                #   write the current frame
Keys (only with --window)
-------------------------
    SPACE   save the current raw frame to captures/
    ESC     quit

What to look for on screen
--------------------------
    blue outline   the leaf the detector found. If this is not hugging your
                   leaf, fix exg_thresh FIRST. Nothing downstream can be
                   right until the leaf outline is right.
    red fill       tissue judged abnormal
    grey fill      shadow or glare: "cannot judge", deliberately not counted
    yellow boxes   accepted lesion blobs
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, Response, jsonify          # noqa: E402

from config import settings as S                    # noqa: E402
from perception import detector as D                # noqa: E402

app = Flask(__name__)
_shared = {"view": None, "masks": None, "stats": {}}
_lock = threading.Lock()


# ------------------------------------------------------------------ camera
def set_focus(cam, spec):
    """
    Pi Camera v3 has a motorised lens. If you never set AfMode it stays
    wherever it was left, which is why your leaf looked soft: the lens was
    parked, not broken.

    spec:  "continuous" | "auto" | a distance in cm, e.g. "25"
    LensPosition is in dioptres = 1 / distance_in_metres, so 25 cm -> 4.0
    and 0.0 is infinity.
    """
    kind, dev = cam
    if kind != "picam":
        return
    try:
        from libcamera import controls
    except ImportError:
        print("[focus] libcamera controls unavailable")
        return
    try:
        if spec == "continuous":
            dev.set_controls({"AfMode": controls.AfModeEnum.Continuous,
                              "AfSpeed": controls.AfSpeedEnum.Fast})
            print("[focus] continuous autofocus")
        elif spec == "auto":
            dev.set_controls({"AfMode": controls.AfModeEnum.Auto})
            dev.autofocus_cycle()
            print("[focus] one-shot autofocus done")
        else:
            cm = float(spec)
            lens = 100.0 / max(cm, 1.0)          # dioptres
            dev.set_controls({"AfMode": controls.AfModeEnum.Manual,
                              "LensPosition": lens})
            print(f"[focus] manual, locked at {cm:.0f} cm "
                  f"(LensPosition {lens:.2f})")
    except Exception as e:
        print(f"[focus] could not set focus: {e}")


def open_camera():
    try:
        from picamera2 import Picamera2
        pc = Picamera2()
        pc.configure(pc.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"}))
        pc.start()
        time.sleep(2.0)
        print("[cam] Pi Camera via picamera2 (RGB888 -> array is already BGR)")
        return ("picam", pc)
    except Exception as e:
        print(f"[cam] picamera2 unavailable ({e}); trying USB")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print("[cam] NO CAMERA. Run tools/check_camera.py")
            sys.exit(1)
        print("[cam] USB webcam via cv2")
        return ("cv2", cap)


SWAP_RB = False        # set by --swap-rb


def grab(cam):
    kind, dev = cam
    if kind == "picam":
        f = dev.capture_array()          # already BGR - do NOT convert
    else:
        ok, f = dev.read()
        if not ok:
            return None
    if SWAP_RB:
        f = f[:, :, ::-1].copy()
    return f


def channel_report(frame):
    b, g, r = [float(c.mean()) for c in cv2.split(frame)]
    warn = ""
    if b > r * 1.25:
        warn = "  <- BLUE >> RED. Channels are probably swapped: try --swap-rb"
    return f"B {b:5.1f}  G {g:5.1f}  R {r:5.1f}{warn}"


def lock_awb(cam):
    """Freeze white balance so hue stops moving between frames."""
    kind, dev = cam
    if kind != "picam":
        dev.set(cv2.CAP_PROP_AUTO_WB, 0)
        print("[cam] AWB off (webcam)")
        return
    md = dev.capture_metadata()
    gains = md.get("ColourGains", (1.8, 1.8))
    dev.set_controls({"AwbEnable": False, "ColourGains": gains})
    print(f"[cam] AWB locked at gains {tuple(round(g, 2) for g in gains)}")


# ------------------------------------------------------------------ web
PAGE = r"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ACES leaf detection</title>
<style>
:root{--loam:#12160f;--panel:#1b2116;--rule:#2f3a26;--crop:#7fb069;
      --ink:#e8ece3;--dim:#8b9680;--rust:#c25a3a;
      --mono:ui-monospace,"DejaVu Sans Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--loam);color:var(--ink);font:14px/1.5 var(--mono)}
header{padding:12px 16px;border-bottom:1px solid var(--rule);background:var(--panel);
       display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
h1{margin:0;font-size:13px;letter-spacing:.26em;font-weight:700}
header span{color:var(--dim);font-size:11px;letter-spacing:.08em}
main{padding:14px;display:grid;gap:14px;grid-template-columns:1fr}
@media(min-width:900px){main{grid-template-columns:2fr 1fr}}
img{width:100%;display:block;border:1px solid var(--rule);background:#000}
.key{background:var(--panel);border:1px solid var(--rule);padding:14px}
.key h2{margin:0 0 10px;font-size:10px;letter-spacing:.2em;color:var(--dim);font-weight:600}
.k{display:flex;gap:9px;align-items:center;margin:7px 0;font-size:12px}
.sw{width:15px;height:15px;border-radius:2px;flex:none;border:1px solid #0006}
pre{margin:12px 0 0;padding:12px;background:var(--panel);border:1px solid var(--rule);
    white-space:pre-wrap;font-size:12.5px;color:var(--ink)}
.hint{color:var(--dim);font-size:11.5px;margin-top:12px;line-height:1.6}
.hint b{color:var(--crop)}
</style>
<header><h1>ACES</h1><span>live leaf detection &mdash; camera only, no motors</span></header>
<main>
  <div>
    <img src="/stream.mjpg" alt="Live camera with detection overlay">
    <img src="/mask.mjpg" style="margin-top:14px"
         alt="Leaf mask diagnostic panels">
  </div>
  <div>
    <div class="key">
      <h2>OVERLAY KEY</h2>
      <div class="k"><span class="sw" style="background:#00c8ff"></span>leaf outline the detector found</div>
      <div class="k"><span class="sw" style="background:#c25a3a"></span>judged abnormal</div>
      <div class="k"><span class="sw" style="background:#787878"></span>shadow or glare &mdash; not judged</div>
      <div class="k"><span class="sw" style="background:#e0d64a"></span>accepted lesion blob</div>
    </div>
    <pre id="r">connecting</pre>
    <p class="hint">Check panel <b>2</b> below the main view. Your lesion must
      be WHITE there. If it is black, the leaf outline missed it and the
      detector never looked at it &mdash; that is a segmentation problem, and
      tuning thresholds will not fix it. Try lowering <b>bg_k</b>, then
      <b>exg_thresh</b>, in config/settings.py.</p>
  </div>
</main>
<script>
setInterval(async()=>{try{
  const d=await (await fetch('/stats')).json();
  document.getElementById('r').textContent=d.text;
}catch(e){document.getElementById('r').textContent='link lost';}},400);
</script>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/stats")
def stats():
    with _lock:
        return jsonify(_shared["stats"])


def mask_panels(bgr, res):
    """
    Four panels that answer one question: is the lesion INSIDE the leaf mask?

    If the lesion shows black in panel 2, the leaf outline missed it and the
    detector never even looked at it. That is a segmentation problem, not a
    threshold problem, and no amount of tuning d_abs_max will fix it.
    """
    def lab(img, text):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img.copy(), (320, 180))
        cv2.rectangle(img, (0, 156), (320, 180), (0, 0, 0), -1)
        cv2.putText(img, text, (5, 173), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1)
        return img

    inside = bgr.copy()
    if res.leaf_mask is not None:
        outside = res.leaf_mask == 0
        inside[outside] = (inside[outside] * 0.18).astype(np.uint8)

    return np.vstack([
        np.hstack([lab(bgr, "1 input"),
                   lab(res.leaf_mask, "2 LEAF mask - lesion must be WHITE")]),
        np.hstack([lab(inside, "3 what the detector actually judges"),
                   lab(res.abnormal_mask, "4 abnormal")]),
    ])


@app.route("/mask.mjpg")
def maskstream():
    def gen():
        while True:
            with _lock:
                v = _shared["masks"]
            if v is None:
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
            time.sleep(0.12)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/stream.mjpg")
def stream():
    def gen():
        while True:
            with _lock:
                v = _shared["view"]
            if v is None:
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", action="store_true", help="also show a local window")
    ap.add_argument("--save", action="store_true", help="enable SPACE to save frames")
    ap.add_argument("--outdir", default="captures")
    ap.add_argument("--focus", default="continuous",
                    help="'continuous', 'auto', or a distance in cm e.g. 25")
    ap.add_argument("--no-lock-awb", action="store_true")
    ap.add_argument("--swap-rb", action="store_true",
                    help="swap red and blue if your stack delivers RGB")
    args = ap.parse_args()

    global SWAP_RB
    SWAP_RB = args.swap_rb

    cam = open_camera()
    time.sleep(0.5)
    set_focus(cam, args.focus)
    time.sleep(1.5)                  # let the lens actually travel
    if not args.no_lock_awb:
        lock_awb(cam)
    if args.save:
        os.makedirs(args.outdir, exist_ok=True)

    threading.Thread(target=lambda: app.run(
        host="0.0.0.0", port=S.STREAM_PORT, threaded=True,
        debug=False, use_reloader=False), daemon=True).start()
    probe = grab(cam)
    if probe is not None:
        print(f"[cam] channel means: {channel_report(probe)}")
        print("[cam] sanity check: on screen, skin should look like skin.")
        print("      Purple hands / magenta walls = swapped channels.\n")
    print(f"  open  http://<your-pi-ip>:{S.STREAM_PORT}  in a browser")
    print("  ctrl-c to stop\n")

    saved, t0, frames, fps = 0, time.time(), 0, 0.0
    try:
        while True:
            frame = grab(cam)
            if frame is None:
                time.sleep(0.05)
                continue
            small = cv2.resize(frame, (960, 540))
            res = D.detect(small)
            view = D.overlay(small, res)

            frames += 1
            if time.time() - t0 > 1.0:
                fps = frames / (time.time() - t0)
                frames, t0 = 0, time.time()

            unk = (float(res.unknown_mask.sum()) / 255 / max(res.leaf_px, 1)
                   if res.unknown_mask is not None and res.leaf_px else 0.0)
            text = (
                f"leaf found     {res.leaf_px:,} px "
                f"({100*res.leaf_px/(small.shape[0]*small.shape[1]):.1f}% of frame)\n"
                f"abnormal       {res.abnormal_px:,} px  "
                f"= {res.ratio:.2%} of leaf\n"
                f"severity       {res.severity}\n"
                f"lesion blobs   {len(res.blobs)}\n"
                f"unjudged       {unk:.1%} (shadow/glare)\n"
                f"trusted        {res.trusted}"
                f"{'  -> ' + res.note if res.note else ''}\n"
                f"detection      "
                f"{'YES' if res.ratio >= S.DETECT_RATIO_MIN and res.blobs else 'no'}"
                f"   (threshold {S.DETECT_RATIO_MIN:.0%})\n"
                f"channels       {channel_report(small)}\n"
                f"leaf method    {D.P.get('leaf_method','grow')}\n"
                f"focus          {args.focus}\n"
                f"fps            {fps:.1f}   saved {saved}")

            with _lock:
                _shared["view"] = view
                _shared["masks"] = mask_panels(small, res)
                _shared["stats"] = {"text": text, "ratio": res.ratio}

            if args.window:
                cv2.imshow("ACES live detection", view)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
                if k == 32 and args.save:
                    p = os.path.join(args.outdir,
                                     f"leaf_{time.strftime('%H%M%S')}.jpg")
                    cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved += 1
                    print(f"  saved {p}")
            else:
                time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        if cam[0] == "picam":
            cam[1].stop()
        else:
            cam[1].release()
        cv2.destroyAllWindows()
        print(f"\nstopped. {saved} frames saved.")


if __name__ == "__main__":
    main()
