"""
Field debug stream. Open http://<pi-ip>:8080 on a phone while the robot runs.

Two feeds: the navigation view (vegetation mask + corridor) and the disease
view (HSV abnormal mask). Plus a live text readout so you can tell whether a
misbehaviour is a vision problem or a control problem without stopping.
"""

import threading

import cv2
from flask import Flask, Response

from config import settings as config

app = Flask(__name__)
_state = {"nav": None, "dis": None, "text": "waiting for first frame"}
_lock = threading.Lock()


def publish(nav_frame=None, disease_frame=None, text=None):
    with _lock:
        if nav_frame is not None:
            _state["nav"] = nav_frame
        if disease_frame is not None:
            _state["dis"] = disease_frame
        if text is not None:
            _state["text"] = text


def _mjpeg(key):
    while True:
        with _lock:
            f = _state[key]
        if f is None:
            continue
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/nav.mjpg")
def nav():
    return Response(_mjpeg("nav"), mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/disease.mjpg")
def disease():
    return Response(_mjpeg("dis"), mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/readout")
def readout():
    with _lock:
        return _state["text"]


PAGE = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>ACES field view</title>
<style>
 :root{--ink:#e6e9e4;--dim:#7d8878;--bg:#14171a;--panel:#1c2024;--rule:#2c3238}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);
   font:14px/1.5 "IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
 header{padding:14px 16px;border-bottom:1px solid var(--rule);
   display:flex;align-items:baseline;gap:12px}
 h1{margin:0;font-size:15px;letter-spacing:.14em;text-transform:uppercase;font-weight:600}
 header span{color:var(--dim);font-size:12px}
 main{display:grid;gap:1px;background:var(--rule);grid-template-columns:1fr}
 @media(min-width:760px){main{grid-template-columns:1fr 1fr}}
 figure{margin:0;background:var(--panel);padding:12px}
 figcaption{color:var(--dim);font-size:11px;letter-spacing:.12em;
   text-transform:uppercase;margin-bottom:8px}
 img{width:100%;display:block;background:#000;border:1px solid var(--rule)}
 pre{margin:0;padding:12px 16px;background:var(--panel);color:var(--ink);
   border-top:1px solid var(--rule);white-space:pre-wrap;font-size:13px}
</style>
<header><h1>ACES</h1><span>row&nbsp;view / leaf&nbsp;view</span></header>
<main>
 <figure><figcaption>Navigation &mdash; vegetation &amp; corridor</figcaption>
  <img src="/nav.mjpg" alt="Navigation camera with vegetation mask"></figure>
 <figure><figcaption>Disease &mdash; HSV abnormal tissue</figcaption>
  <img src="/disease.mjpg" alt="Side camera with abnormal-tissue mask"></figure>
</main>
<pre id=r>connecting</pre>
<script>
 setInterval(async()=>{try{
   document.getElementById('r').textContent=await(await fetch('/readout')).text();
 }catch(e){document.getElementById('r').textContent='link lost \u2014 check the Pi';}},400);
</script>"""


@app.route("/")
def index():
    return PAGE


def start():
    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=config.STREAM_PORT,
                               threaded=True, debug=False, use_reloader=False),
        daemon=True).start()
