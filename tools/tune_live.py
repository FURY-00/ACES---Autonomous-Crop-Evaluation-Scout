"""
Live tuner: adjust every threshold in the browser, on the real camera feed.

    python3 tools/tune_live.py

Then open  http://<pi-ip>:8080  on a laptop or phone.

Why in a browser: the Pi in the field has no monitor, and OpenCV trackbar
windows need one. This works over plain SSH.

What you get
------------
  * the live overlay, plus FOUR separate mask views (background, leaf,
    healthy, abnormal) so you can see which stage is wrong
  * a slider for every parameter, grouped by the stage it affects
  * FREEZE, so you tune against one still frame instead of chasing a
    moving picture
  * CLICK-TO-SAMPLE: click anywhere on the image and it reports that
    pixel's real numbers, and tells you which threshold to move
  * SAVE, which writes config/detector_tuned.json. settings.py picks that
    up automatically on the next run. Your settings.py is never edited.

The order to tune in
--------------------
This order is not a suggestion. Each stage feeds the next, so tuning a
later stage against a broken earlier one just bakes in the error.

    1. BACKGROUND   panel 1: background white, leaf black.  -> bg_k
    2. LEAF         panel 2: the whole leaf white, lesion INCLUDED.
                    -> exg_thresh, close_k
    3. HEALTHY      panel 3: only clearly-green tissue white.
                    -> healthy_h, healthy_s_min
    4. ABNORMAL     panel 4: only the lesion white.  -> d_abs_max
"""

import argparse
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, Response, jsonify, request   # noqa: E402

from config import settings as S                      # noqa: E402
from perception import detector as D                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNED = os.path.join(ROOT, "config", "detector_tuned.json")

app = Flask(__name__)
_lock = threading.Lock()
_st = {"frame": None, "views": {}, "stats": "", "frozen": None,
       "params": dict(S.DETECTOR), "sample": ""}

SWAP_RB = False

# label, key, min, max, step, group, hint
SLIDERS = [
    ("Background sensitivity", "bg_k", 1.0, 8.0, 0.1, "1 BACKGROUND",
     "LOWER = leaf grows more into the background. Raise if the outline leaks."),
    ("Lightness weight", "bg_L_weight", 0.0, 1.0, 0.05, "1 BACKGROUND",
     "How much brightness counts vs colour. Keep low so shadow isn't background."),
    ("Border ring size", "bg_border_frac", 0.02, 0.20, 0.01, "1 BACKGROUND",
     "How much of the frame edge is sampled to learn the background."),

    ("Green core threshold", "exg_thresh", 0.0, 80.0, 1.0, "2 LEAF",
     "LOWER = more pixels count as the green seed. 0 = automatic (Otsu)."),
    ("Gap closing", "close_k", 3.0, 25.0, 2.0, "2 LEAF",
     "Bridges gaps in the outline. Too big and the leaf swells."),
    ("Min leaf size", "min_leaf_frac", 0.002, 0.20, 0.002, "2 LEAF",
     "Blobs smaller than this fraction of the frame are not leaves."),
    ("Max leaf size", "max_leaf_frac", 0.30, 0.99, 0.01, "2 LEAF",
     "If the 'leaf' exceeds this, growth ran away into the background."),

    ("Healthy hue low", "healthy_h_lo", 0.0, 90.0, 1.0, "3 HEALTHY",
     "OpenCV hue 0-179. Green is roughly 35-85."),
    ("Healthy hue high", "healthy_h_hi", 20.0, 179.0, 1.0, "3 HEALTHY", ""),
    ("Healthy min saturation", "healthy_s_min", 0.0, 200.0, 5.0, "3 HEALTHY",
     "RAISE to stop pale washed-out green counting as healthy."),
    ("Healthy min value", "healthy_v_min", 0.0, 200.0, 5.0, "3 HEALTHY", ""),

    ("Plant floor (min_d_ref)", "min_d_ref", -0.15, 0.20, 0.005, "4 ABNORMAL",
     "Below this greenness the frame is 'not a plant' and is rejected. "
     "Read d_ref off your leaf in the stats box, then set this ~0.03 BELOW it."),
    ("Healthy foliage level", "d_healthy_foliage", -0.05, 0.35, 0.005, "4 ABNORMAL",
     "Above this, the leaf has healthy tissue and normal comparison runs. "
     "BETWEEN the plant floor and this, the WHOLE leaf is called diseased. "
     "Set it above your leaf's d_ref to flag an entirely yellow leaf."),
    ("Absolute gate", "d_abs_max", -0.05, 0.30, 0.005, "4 ABNORMAL",
     "THE key number. A pixel greener than this is never abnormal. "
     "RAISE to catch more; too high and healthy tissue gets flagged."),
    ("Absolute gate (seed)", "d_abs_strong", -0.10, 0.25, 0.005, "4 ABNORMAL",
     "Keep about 0.02 below the absolute gate."),
    ("Relative gate", "k_weak", 0.5, 6.0, 0.1, "4 ABNORMAL",
     "How far below THIS leaf's healthy tissue counts. Lower = more sensitive."),
    ("Relative gate (seed)", "k_strong", 1.0, 8.0, 0.1, "4 ABNORMAL", ""),
    ("Min lesion size", "min_blob_px", 0.0, 4000.0, 50.0, "4 ABNORMAL",
     "Specks smaller than this are discarded."),

    ("Specular removal", "k_specular", 1.0, 10.0, 0.1, "5 EXCLUSIONS",
     "LOWER removes more glare (and more leaf with it)."),
    ("Shadow cutoff", "shadow_v_max", 0.0, 120.0, 2.0, "5 EXCLUSIONS", ""),
    ("Glare max saturation", "glare_s_max", 0.0, 120.0, 2.0, "5 EXCLUSIONS", ""),
    ("Glare min brightness", "glare_v_min", 120.0, 255.0, 5.0, "5 EXCLUSIONS", ""),
]


# ---------------------------------------------------------------- camera
def open_camera():
    try:
        from picamera2 import Picamera2
        pc = Picamera2()
        pc.configure(pc.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"}))
        pc.start()
        time.sleep(2.0)
        try:
            from libcamera import controls
            pc.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            time.sleep(1.2)
            md = pc.capture_metadata()
            pc.set_controls({"AwbEnable": False,
                             "ColourGains": md.get("ColourGains", (1.8, 1.8))})
        except Exception as e:
            print(f"[cam] focus/awb: {e}")
        print("[cam] Pi Camera (RGB888 array is already BGR)")
        return ("picam", pc)
    except Exception as e:
        print(f"[cam] picamera2 unavailable ({e}); trying USB")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print("[cam] no camera. Run tools/check_camera.py")
            sys.exit(1)
        return ("cv2", cap)


def grab(cam):
    kind, dev = cam
    if kind == "picam":
        f = dev.capture_array()
    else:
        ok, f = dev.read()
        if not ok:
            return None
    return f[:, :, ::-1].copy() if SWAP_RB else f


# ---------------------------------------------------------------- params
def to_detector(ui):
    """UI keys are flat; the detector wants healthy_h as a tuple."""
    p = dict(S.DETECTOR)
    for k, v in ui.items():
        if k in ("healthy_h_lo", "healthy_h_hi"):
            continue
        p[k] = int(v) if k in ("close_k", "min_blob_px", "exg_thresh",
                               "shadow_v_max", "glare_s_max", "glare_v_min",
                               "healthy_s_min", "healthy_v_min") else v
    lo = int(ui.get("healthy_h_lo", S.DETECTOR["healthy_h"][0]))
    hi = int(ui.get("healthy_h_hi", S.DETECTOR["healthy_h"][1]))
    p["healthy_h"] = (lo, max(lo + 1, hi))
    if p.get("close_k", 9) % 2 == 0:
        p["close_k"] += 1
    return p


def initial_ui():
    d = dict(S.DETECTOR)
    ui = {k: d[k] for _, k, *_ in SLIDERS if k in d}
    ui["healthy_h_lo"] = d["healthy_h"][0]
    ui["healthy_h_hi"] = d["healthy_h"][1]
    for _, k, lo, hi, st, *_ in SLIDERS:
        ui.setdefault(k, lo)
    return ui


# ---------------------------------------------------------------- routes
def encode(img):
    if img is None:
        img = np.zeros((180, 320), np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return buf.tobytes() if ok else b""


def mjpeg(name):
    def gen():
        while True:
            with _lock:
                v = _st["views"].get(name)
            if v is not None:
                yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n"
                       + encode(v) + b"\r\n")
            time.sleep(0.08)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/v/<name>.mjpg")
def view(name):
    return mjpeg(name)


@app.route("/stats")
def stats():
    with _lock:
        return jsonify({"text": _st["stats"], "sample": _st["sample"],
                        "frozen": _st["frozen"] is not None})


@app.route("/set", methods=["POST"])
def setp():
    d = request.get_json(force=True)
    with _lock:
        _st["params"].update({k: float(v) for k, v in d.items()})
    return jsonify(ok=True)


@app.route("/freeze", methods=["POST"])
def freeze():
    with _lock:
        if _st["frozen"] is None:
            _st["frozen"] = _st["frame"].copy() if _st["frame"] is not None else None
        else:
            _st["frozen"] = None
        return jsonify(frozen=_st["frozen"] is not None)


@app.route("/sample", methods=["POST"])
def sample():
    d = request.get_json(force=True)
    with _lock:
        f = _st["frozen"] if _st["frozen"] is not None else _st["frame"]
        if f is None:
            return jsonify(text="no frame")
        h, w = f.shape[:2]
        x = int(np.clip(d["x"] * w, 1, w - 2))
        y = int(np.clip(d["y"] * h, 1, h - 2))
        patch = f[y - 1:y + 2, x - 1:x + 2].reshape(-1, 3).mean(axis=0)
        b, g, r = patch
        hsv = cv2.cvtColor(np.uint8([[patch]]), cv2.COLOR_BGR2HSV)[0][0]
        dv = (g - r) / (r + g + b + 1e-6)
        exg = max(0.0, 2 * g - r - b)
        advice = []
        if dv < 0.05:
            advice.append(f"d={dv:+.3f} is LOW -> looks abnormal. "
                          f"Set the absolute gate just above it, ~{dv+0.02:.3f}")
        else:
            advice.append(f"d={dv:+.3f} is HIGH -> looks healthy. "
                          f"Keep the absolute gate BELOW {dv-0.02:.3f}")
        advice.append(f"ExG={exg:.0f} -> green-core threshold must be under "
                      f"this for the pixel to seed the leaf")
        _st["sample"] = (f"({x},{y})  B{b:.0f} G{g:.0f} R{r:.0f}   "
                         f"H{hsv[0]} S{hsv[1]} V{hsv[2]}\n" + "\n".join(advice))
        return jsonify(text=_st["sample"])


@app.route("/save", methods=["POST"])
def save():
    with _lock:
        p = to_detector(_st["params"])
    out = {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}
    with open(TUNED, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[tuner] SAVED -> {TUNED}")
    print(f"[tuner]   min_d_ref={out.get('min_d_ref')}  "
          f"d_healthy_foliage={out.get('d_healthy_foliage')}  "
          f"d_abs_max={out.get('d_abs_max')}  bg_k={out.get('bg_k')}")
    return jsonify(ok=True, path=TUNED,
                   min_d_ref=out.get("min_d_ref"),
                   d_abs_max=out.get("d_abs_max"))


@app.route("/")
def index():
    groups = {}
    for label, key, lo, hi, st, grp, *hint in SLIDERS:
        groups.setdefault(grp, []).append(
            dict(label=label, key=key, lo=lo, hi=hi, step=st,
                 hint=(hint[0] if hint else "")))
    # Serve the values that are LIVE right now, not the ones this process
    # started with. Otherwise a browser refresh shows stale sliders, and the
    # first slider you touch pushes all of them back and silently undoes your
    # tuning. On a phone, where refreshes happen by accident, that is brutal.
    with _lock:
        live = dict(_st["params"])
    return Response(PAGE.replace("__GROUPS__", json.dumps(groups))
                        .replace("__INIT__", json.dumps(live)),
                    mimetype="text/html")


PAGE = r"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ACES tuner</title>
<style>
:root{--loam:#12160f;--panel:#1b2116;--rule:#2f3a26;--crop:#7fb069;
      --ink:#e8ece3;--dim:#8b9680;--rust:#c25a3a;
      --mono:ui-monospace,"DejaVu Sans Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--loam);color:var(--ink);font:13px/1.5 var(--mono)}
header{padding:11px 15px;border-bottom:1px solid var(--rule);background:var(--panel);
  display:flex;gap:12px;align-items:center;flex-wrap:wrap;position:sticky;top:0;z-index:9}
h1{margin:0;font-size:12px;letter-spacing:.26em;font-weight:700}
button{background:var(--panel);border:1px solid var(--rule);color:var(--ink);
  font:inherit;font-size:11px;letter-spacing:.12em;padding:6px 13px;cursor:pointer}
button:hover{border-color:var(--crop)}
button.on{background:var(--crop);color:var(--loam);font-weight:700}
main{display:grid;gap:14px;padding:14px;grid-template-columns:1fr}
@media(min-width:1000px){main{grid-template-columns:1.25fr 1fr}}
img{width:100%;display:block;border:1px solid var(--rule);background:#000;cursor:crosshair}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:12px}
figure{margin:0}
figcaption{font-size:10px;letter-spacing:.13em;color:var(--dim);padding:5px 0 4px}
pre{margin:11px 0 0;padding:11px;background:var(--panel);border:1px solid var(--rule);
  white-space:pre-wrap;font-size:12px}
.grp{margin-bottom:16px;border:1px solid var(--rule);background:var(--panel)}
.grp h2{margin:0;padding:8px 12px;font-size:10px;letter-spacing:.2em;
  border-bottom:1px solid var(--rule);color:var(--crop);font-weight:700}
.sl{padding:9px 12px;border-bottom:1px solid #232c1c}
.sl:last-child{border-bottom:0}
.sl .top{display:flex;justify-content:space-between;gap:8px;font-size:12px}
.sl .val{color:var(--crop);font-weight:700}
.sl input{width:100%;margin:5px 0 0;accent-color:var(--crop)}
.sl .hint{font-size:10.5px;color:var(--dim);margin-top:3px;line-height:1.45}
</style>
<header>
  <h1>ACES TUNER</h1>
  <button id="fz">FREEZE</button>
  <button id="sv">SAVE</button>
  <span id="msg" style="color:var(--dim);font-size:11px"></span>
</header>
<main>
  <div>
    <figure><figcaption>OVERLAY &mdash; click anywhere to sample a pixel</figcaption>
      <img id="main" src="/v/overlay.mjpg" alt="Live detection overlay"></figure>
    <div class="grid2">
      <figure><figcaption>1 BACKGROUND (white = background)</figcaption>
        <img src="/v/bg.mjpg" alt="Background mask"></figure>
      <figure><figcaption>2 LEAF (lesion must be WHITE)</figcaption>
        <img src="/v/leaf.mjpg" alt="Leaf mask"></figure>
      <figure><figcaption>3 HEALTHY</figcaption>
        <img src="/v/healthy.mjpg" alt="Healthy mask"></figure>
      <figure><figcaption>4 ABNORMAL</figcaption>
        <img src="/v/abnormal.mjpg" alt="Abnormal mask"></figure>
    </div>
    <pre id="stats">connecting</pre>
    <pre id="sample">click the overlay to sample a pixel</pre>
  </div>
  <div id="panel"></div>
</main>
<script>
const G=__GROUPS__, V=__INIT__;
const panel=document.getElementById('panel');
for(const [name,items] of Object.entries(G).sort()){
  const d=document.createElement('div'); d.className='grp';
  d.innerHTML=`<h2>${name}</h2>`+items.map(it=>`
    <div class="sl">
      <div class="top"><span>${it.label}</span>
        <span class="val" id="v_${it.key}">${(+V[it.key]).toFixed(3)}</span></div>
      <input type="range" id="s_${it.key}" min="${it.lo}" max="${it.hi}"
             step="${it.step}" value="${V[it.key]}">
      ${it.hint?`<div class="hint">${it.hint}</div>`:''}
    </div>`).join('');
  panel.appendChild(d);
}
let timer=null;
function push(){
  clearTimeout(timer);
  timer=setTimeout(()=>fetch('/set',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(V)}),90);
}
document.querySelectorAll('input[type=range]').forEach(el=>{
  const k=el.id.slice(2);
  el.addEventListener('input',()=>{
    V[k]=+el.value;
    document.getElementById('v_'+k).textContent=(+el.value).toFixed(3);
    push();
  });
});
document.getElementById('fz').onclick=async e=>{
  const r=await (await fetch('/freeze',{method:'POST'})).json();
  e.target.classList.toggle('on',r.frozen);
  e.target.textContent=r.frozen?'LIVE':'FREEZE';
};
document.getElementById('sv').onclick=async()=>{
  const el=document.getElementById('msg');
  try{
    const r=await (await fetch('/save',{method:'POST'})).json();
    el.style.color='var(--crop)';
    el.textContent=`SAVED  min_d_ref=${r.min_d_ref}  d_abs_max=${r.d_abs_max}`;
  }catch(e){
    el.style.color='var(--rust)';
    el.textContent='SAVE FAILED - '+e;
  }
  setTimeout(()=>{el.textContent='';},8000);
};
document.getElementById('main').onclick=async ev=>{
  const b=ev.target.getBoundingClientRect();
  const r=await (await fetch('/sample',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({x:(ev.clientX-b.left)/b.width,
                         y:(ev.clientY-b.top)/b.height})})).json();
  document.getElementById('sample').textContent=r.text;
};
setInterval(async()=>{try{
  const d=await (await fetch('/stats')).json();
  document.getElementById('stats').textContent=d.text;
}catch(e){}},500);
</script>"""


# ---------------------------------------------------------------- loop
def main():
    global SWAP_RB
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap-rb", action="store_true")
    a = ap.parse_args()
    SWAP_RB = a.swap_rb

    cam = open_camera()
    with _lock:
        _st["params"] = initial_ui()

    threading.Thread(target=lambda: app.run(
        host="0.0.0.0", port=S.STREAM_PORT, threaded=True,
        debug=False, use_reloader=False), daemon=True).start()
    print(f"\n  open  http://<your-pi-ip>:{S.STREAM_PORT}\n")

    try:
        while True:
            with _lock:
                frozen = _st["frozen"]
                ui = dict(_st["params"])
            frame = frozen if frozen is not None else grab(cam)
            if frame is None:
                time.sleep(0.05)
                continue
            small = cv2.resize(frame, (800, 450))
            p = to_detector(ui)
            r = D.detect(small, p)

            b, g, rr = [float(c.mean()) for c in cv2.split(small)]
            warn = "  <- B>>R, channels may be swapped" if b > rr * 1.25 else ""
            unk = (float(r.unknown_mask.sum()) / 255 / max(r.leaf_px, 1)
                   if r.unknown_mask is not None and r.leaf_px else 0)
            txt = (f"leaf      {r.leaf_px:,} px "
                   f"({100*r.leaf_px/(800*450):.1f}% of frame)\n"
                   f"abnormal  {r.abnormal_px:,} px = {r.ratio:.2%} of leaf   "
                   f"blobs {len(r.blobs)}\n"
                   f"severity  {r.severity}    trusted {r.trusted}"
                   f"{'  ' + r.note if r.note else ''}\n"
                   f"d_ref     {r.d_ref:+.4f}   applied threshold "
                   f"{r.d_thresh:+.4f}   spread {r.debug.get('spread','-')}\n"
                   f"unjudged  {unk:.1%}   channels B{b:.0f} G{g:.0f} R{rr:.0f}{warn}\n"
                   f"DETECTION {'YES' if r.ratio >= S.DETECT_RATIO_MIN and r.blobs else 'no'}"
                   f"  (needs >= {S.DETECT_RATIO_MIN:.0%})")

            with _lock:
                _st["frame"] = frame
                _st["stats"] = txt
                _st["views"] = {
                    "overlay": D.overlay(small, r),
                    "bg": r.bg_mask,
                    "leaf": r.leaf_mask,
                    "healthy": cv2.bitwise_and(r.healthy_mask, r.leaf_mask)
                    if r.healthy_mask is not None else None,
                    "abnormal": r.abnormal_mask,
                }
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if cam[0] == "picam":
            cam[1].stop()
        else:
            cam[1].release()


if __name__ == "__main__":
    main()
