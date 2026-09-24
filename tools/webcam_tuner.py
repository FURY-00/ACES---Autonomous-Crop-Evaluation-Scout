"""
Adjust the webcam's exposure, saturation and friends live, in a browser,
while watching what the row detector makes of the result.

WHY THIS AND NOT tune_live.py
-----------------------------
tune_live.py tunes the DETECTOR -- the thresholds applied to an image. This
tunes the CAMERA -- what image you get in the first place. When the frame is
washed out, no detector threshold can recover colour the sensor never
recorded, so this is the one that matters.

The panel shows the row-following mask beside the raw frame, so you can see
the effect of a camera change on the thing you actually care about: whether
the crop walls are found.

Usage
-----
    python3 tools/webcam_tuner.py                 # auto-detect the device
    python3 tools/webcam_tuner.py --device /dev/video1
    python3 tools/webcam_tuner.py --port 8000

Then open  http://<pi-ip>:8000

Every control is read from the driver, so you get exactly the ones your
camera supports, each with its real range and default. RESET puts everything
back. SAVE prints a ready-to-paste command line.

Note these settings live in the DRIVER, not in this program: whatever you
leave them at persists into every later run until something resets them or
you replug the camera.
"""

import argparse
import glob
import json
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from navigation.row_vision import RowFollower, vegetation_mask, CFG  # noqa

_st = {"jpg": None, "txt": "starting", "cmd": None}
_lock = threading.Lock()

# controls worth exposing, in a sensible order
WANTED = ["auto_exposure", "exposure_time_absolute", "exposure_absolute",
          "exposure_dynamic_framerate", "gain",
          "brightness", "contrast", "saturation", "sharpness",
          "white_balance_automatic", "white_balance_temperature",
          "backlight_compensation", "hue", "gamma"]


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True,
                          text=True).stdout


def find_webcam():
    out = sh("v4l2-ctl --list-devices")
    for chunk in out.split("\n\n"):
        if "USB" in chunk and "Camera" in chunk:
            for n in re.findall(r"(/dev/video\d+)", chunk):
                if "brightness" in sh(f"v4l2-ctl -d {n} --list-ctrls"):
                    return n
    for n in sorted(glob.glob("/dev/video*")):
        if "brightness" in sh(f"v4l2-ctl -d {n} --list-ctrls"):
            return n
    return None


def read_ctrls(dev):
    """Every writable control the driver reports, with range and default."""
    out = sh(f"v4l2-ctl -d {dev} --list-ctrls")
    ctrls = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\((int|bool|menu)\)\s*:(.*)",
                     line)
        if not m:
            continue
        name, kind, rest = m.groups()
        g = lambda k: (int(re.search(rf"{k}=(-?\d+)", rest).group(1))
                       if re.search(rf"{k}=(-?\d+)", rest) else None)
        lo = g("min") if kind != "bool" else 0
        hi = g("max") if kind != "bool" else 1
        ctrls[name] = dict(name=name, kind=kind, min=lo, max=hi,
                           step=g("step") or 1, default=g("default"),
                           value=g("value"))
    return ctrls


def set_ctrl(dev, name, val):
    r = subprocess.run(f"v4l2-ctl -d {dev} --set-ctrl={name}={int(val)}",
                       shell=True, capture_output=True, text=True)
    return r.returncode == 0


PAGE = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ACES webcam tuner</title>
<style>
:root{--loam:#12160f;--panel:#1b2116;--rule:#2f3a26;--crop:#7fb069;
      --ink:#e8ece3;--dim:#8b9680;--rust:#c25a3a}
*{box-sizing:border-box}
body{margin:0;background:var(--loam);color:var(--ink);
 font:13px/1.5 ui-monospace,"DejaVu Sans Mono",monospace}
header{padding:11px 15px;background:var(--panel);border-bottom:1px solid var(--rule);
 display:flex;gap:12px;align-items:center;flex-wrap:wrap;position:sticky;top:0;z-index:9}
h1{margin:0;font-size:12px;letter-spacing:.26em;font-weight:700}
button{background:var(--panel);border:1px solid var(--rule);color:var(--ink);
 font:inherit;font-size:11px;letter-spacing:.12em;padding:7px 14px;cursor:pointer}
button:hover{border-color:var(--crop)}
button.go{background:var(--crop);color:var(--loam);border:0;font-weight:700}
main{display:grid;gap:14px;padding:14px;grid-template-columns:1fr}
@media(min-width:1000px){main{grid-template-columns:1.3fr 1fr}}
img{width:100%;display:block;border:1px solid var(--rule);background:#000}
pre{margin:12px 0 0;padding:12px;background:var(--panel);
 border:1px solid var(--rule);white-space:pre-wrap;font-size:12.5px}
.sl{padding:9px 12px;border-bottom:1px solid #232c1c;background:var(--panel)}
.sl:first-child{border-top:1px solid var(--rule)}
.top{display:flex;justify-content:space-between;gap:8px}
.val{color:var(--crop);font-weight:700}
.dflt{color:var(--dim);font-size:10.5px}
input[type=range]{width:100%;margin:5px 0 0;accent-color:var(--crop)}
#msg{color:var(--crop);font-size:11px}
</style>
<header><h1>WEBCAM TUNER</h1>
 <button onclick="reset()">RESET TO DEFAULTS</button>
 <button class="go" onclick="save()">SHOW COMMAND</button>
 <span id="msg"></span></header>
<main>
 <div>
  <img src="/stream" alt="Webcam with row detection overlay">
  <pre id="t">loading</pre>
 </div>
 <div id="panel"></div>
</main>
<script>
let C={};
async function load(){
  C=await (await fetch('/ctrls')).json();
  const p=document.getElementById('panel'); p.innerHTML='';
  for(const [k,c] of Object.entries(C)){
    const d=document.createElement('div'); d.className='sl';
    d.innerHTML=`<div class="top"><span>${k}</span>
      <span class="val" id="v_${k}">${c.value}</span></div>
      <input type="range" id="s_${k}" min="${c.min}" max="${c.max}"
             step="${c.step}" value="${c.value}">
      <div class="dflt">default ${c.default} &nbsp; range ${c.min}..${c.max}</div>`;
    p.appendChild(d);
  }
  document.querySelectorAll('input[type=range]').forEach(el=>{
    const k=el.id.slice(2);
    el.addEventListener('input',()=>{
      document.getElementById('v_'+k).textContent=el.value;
      fetch('/set?'+k+'='+el.value);
    });
  });
}
async function reset(){ await fetch('/reset'); await load();
  document.getElementById('msg').textContent='reset to defaults'; }
async function save(){
  const t=await (await fetch('/save')).text();
  document.getElementById('msg').textContent='printed in the terminal';
  alert(t);
}
setInterval(async()=>{try{
 document.getElementById('t').textContent=await(await fetch('/text')).text();
}catch(e){}},400);
load();
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="text/plain"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path
        if p.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        j = _st["jpg"]
                    if j:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(j)}\r\n\r\n".encode())
                        self.wfile.write(j + b"\r\n")
                    time.sleep(0.08)
            except Exception:
                pass
        elif p.startswith("/ctrls"):
            self._send(json.dumps(CTRLS), "application/json")
        elif p.startswith("/set?"):
            k, v = p.split("?", 1)[1].split("=")
            if set_ctrl(DEV, k, v):
                CTRLS[k]["value"] = int(v)
            self._send("ok")
        elif p.startswith("/reset"):
            for k, c in sorted(CTRLS.items(),
                               key=lambda kv: 0 if "auto" in kv[0] else 1):
                set_ctrl(DEV, k, c["default"])
                c["value"] = c["default"]
            self._send("ok")
        elif p.startswith("/save"):
            parts = [f"v4l2-ctl -d {DEV} --set-ctrl={k}={c['value']}"
                     for k, c in CTRLS.items() if c["value"] != c["default"]]
            txt = ("\n".join(parts) if parts
                   else "everything is at its default")
            print("\n--- run these before run_mission.py ---")
            print(txt)
            print("---------------------------------------\n")
            self._send(txt)
        elif p.startswith("/text"):
            with _lock:
                self._send(_st["txt"])
        else:
            self._send(PAGE, "text/html")


def main():
    global DEV, CTRLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--device")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    DEV = a.device or find_webcam()
    if not DEV:
        print("no webcam found. check:  v4l2-ctl --list-devices")
        sys.exit(1)
    all_ctrls = read_ctrls(DEV)
    CTRLS = {k: v for k, v in all_ctrls.items() if k in WANTED} or all_ctrls
    print(f"device {DEV}, {len(CTRLS)} controls\n")

    idx = int(re.search(r"(\d+)$", DEV).group(1))
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"could not open {DEV} for capture")
        sys.exit(1)

    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"  open  http://<pi-ip>:{a.port}\n")

    rows = RowFollower()
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            small = cv2.resize(f, (480, 270))
            row = rows.update(small)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            mask = vegetation_mask(small, CFG)

            view = np.hstack([
                cv2.resize(row.debug, (480, 270)),
                cv2.cvtColor(cv2.resize(mask, (480, 270)), cv2.COLOR_GRAY2BGR)])
            okj, buf = cv2.imencode(".jpg", view,
                                    [cv2.IMWRITE_JPEG_QUALITY, 80])
            if okj:
                with _lock:
                    _st["jpg"] = buf.tobytes()
                    _st["txt"] = (
                        f"walls      {row.walls}    conf {row.conf:.2f}\n"
                        f"steer      {row.steer:+d}\n"
                        f"\n"
                        f"brightness {hsv[:,:,2].mean():5.0f}   "
                        f"(150-190 is a good working range)\n"
                        f"saturation {hsv[:,:,1].mean():5.0f}   "
                        f"(leaves need 60+)\n"
                        f"veg mask   {100*mask.mean()/255:4.1f}% of frame\n"
                        f"\n"
                        f"AIM FOR:  walls both,  saturation above 60,\n"
                        f"          brightness under 190, mask covering the\n"
                        f"          plants rather than scattered specks.\n"
                        f"\n"
                        f"left panel = row detection, right = vegetation mask")
            time.sleep(0.06)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        cap.release()
        print("\nstopped. settings stay in the driver until reset or replug.")


if __name__ == "__main__":
    main()
