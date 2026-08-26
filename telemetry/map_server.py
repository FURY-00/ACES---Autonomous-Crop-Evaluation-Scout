"""
Live field map: the robot's position plus a pin at every detection.

Open http://<pi-ip>:8081 from a phone on the same hotspot.

The offline problem, and what this does about it
------------------------------------------------
Map tiles come from the internet. In a field you often have none, and a
Leaflet map with no tiles is a grey rectangle with your markers floating on
it — technically working, practically useless.

So this page has two views. "Satellite" uses real tiles when they load.
"Survey" is a self-contained SVG plot of everything in local metres relative
to the first GPS fix, drawn from data the Pi already has. It needs no
network at all and is arguably the more useful view in a crop row, because
it shows the run as a track through the rows rather than as a blob on a
world map.
"""

import json
import threading
import time

from flask import Flask, Response, jsonify

from config import settings as S

app = Flask(__name__)

_state = {
    "bot": {"lat": None, "lon": None, "fix_ok": False, "sats": 0,
            "heading": 0.0, "state": "IDLE", "pass_idx": 0, "odo_cm": 0.0},
    "track": [],        # [[lat, lon], ...] breadcrumb
    "pins": [],         # detections
    "started": time.time(),
}
_lock = threading.Lock()


def update_bot(**kw):
    with _lock:
        _state["bot"].update(kw)
        lat, lon = _state["bot"].get("lat"), _state["bot"].get("lon")
        if lat and lon and _state["bot"].get("fix_ok"):
            t = _state["track"]
            if not t or abs(t[-1][0] - lat) > 1e-6 or abs(t[-1][1] - lon) > 1e-6:
                t.append([lat, lon])
                if len(t) > 4000:
                    del t[:1000]


def add_pin(record, image_url=None):
    with _lock:
        _state["pins"].append({
            "lat": record.get("lat"), "lon": record.get("lon"),
            "disease": record.get("disease", "unknown"),
            "severity": record.get("severity", "unknown"),
            "ratio": round(record.get("ratio", 0.0), 4),
            "confidence": round(record.get("confidence", 0.0), 3),
            "trusted": record.get("trusted", True),
            "t": time.strftime("%H:%M:%S"),
            "image": image_url,
        })


@app.route("/state")
def state():
    with _lock:
        return jsonify(_state)


PAGE = r"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>ACES field map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
:root{
  --loam:#12160f;        /* deep wet soil, the page ground */
  --loam-2:#1b2116;
  --rule:#2f3a26;
  --crop:#7fb069;        /* living leaf */
  --crop-bright:#a7d189;
  --chlorosis:#e0b64a;   /* yellowing */
  --necrosis:#c25a3a;    /* rust brown */
  --ink:#e8ece3;
  --dim:#8b9680;
  --mono:ui-monospace,"DejaVu Sans Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{
  background:var(--loam);color:var(--ink);
  font:14px/1.45 var(--mono);
  display:grid;grid-template-rows:auto 1fr;
}
/* ---- header: reads like the top of a survey sheet ---- */
header{
  display:flex;align-items:center;gap:14px;flex-wrap:wrap;
  padding:10px 14px;border-bottom:1px solid var(--rule);background:var(--loam-2);
}
h1{margin:0;font-size:13px;letter-spacing:.28em;font-weight:700}
.stat{color:var(--dim);font-size:11px;letter-spacing:.06em}
.stat b{color:var(--ink);font-weight:600}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;
     background:var(--necrosis);vertical-align:middle;margin-right:5px}
.dot.ok{background:var(--crop);box-shadow:0 0 0 0 var(--crop);
        animation:pulse 2.4s infinite}
@keyframes pulse{70%{box-shadow:0 0 0 7px rgba(127,176,105,0)}
                 100%{box-shadow:0 0 0 0 rgba(127,176,105,0)}}
.toggle{margin-left:auto;display:flex;border:1px solid var(--rule);border-radius:2px}
.toggle button{
  background:none;border:0;color:var(--dim);font:inherit;font-size:11px;
  letter-spacing:.14em;padding:6px 12px;cursor:pointer}
.toggle button[aria-pressed=true]{background:var(--crop);color:var(--loam);font-weight:700}
.toggle button:focus-visible{outline:2px solid var(--crop-bright);outline-offset:2px}

/* ---- body: map + ledger ---- */
main{display:grid;grid-template-columns:1fr;min-height:0}
@media(min-width:820px){main{grid-template-columns:1fr 300px}}
#stage{position:relative;min-height:0;background:var(--loam-2)}
#map,#survey{position:absolute;inset:0;width:100%;height:100%}
#survey{display:none}
.leaflet-container{background:var(--loam-2)}

/* ---- the ledger: a field survey log, newest at top ---- */
#ledger{border-top:1px solid var(--rule);overflow-y:auto;background:var(--loam-2)}
@media(min-width:820px){#ledger{border-top:0;border-left:1px solid var(--rule)}}
#ledger h2{margin:0;padding:10px 14px;font-size:10px;letter-spacing:.22em;
  color:var(--dim);font-weight:600;border-bottom:1px solid var(--rule);
  position:sticky;top:0;background:var(--loam-2)}
.entry{padding:9px 14px;border-bottom:1px solid var(--rule);cursor:pointer}
.entry:hover,.entry:focus-visible{background:#222a1c;outline:none}
.entry .row1{display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.entry .name{font-size:12px;font-weight:600}
.entry .time{font-size:10px;color:var(--dim)}
.entry .meta{font-size:10px;color:var(--dim);margin-top:3px}
/* severity as a filled bar, because a word is not a quantity */
.bar{height:3px;background:var(--rule);margin-top:6px;border-radius:2px;overflow:hidden}
.bar i{display:block;height:100%}
.sev-trace i{background:var(--crop);width:12%}
.sev-mild i{background:var(--chlorosis);width:35%}
.sev-moderate i{background:var(--chlorosis);width:65%}
.sev-severe i{background:var(--necrosis);width:100%}
.untrusted{opacity:.55}
.gmap{display:inline-block;margin-top:7px;font-size:10px;letter-spacing:.08em;
  color:var(--crop);text-decoration:none;border-bottom:1px solid #3c4a31;
  padding-bottom:2px}
.gmap:hover{border-color:var(--crop)}
.nogps{display:inline-block;margin-top:7px;font-size:10px;color:var(--dim)}
.empty{padding:22px 14px;color:var(--dim);font-size:12px}
</style>

<header>
  <h1>ACES</h1>
  <span class="stat"><span class="dot" id="fixdot"></span>GPS <b id="sats">–</b> sats</span>
  <span class="stat">STATE <b id="botstate">–</b></span>
  <span class="stat">PASS <b id="pass">–</b></span>
  <span class="stat">ODO <b id="odo">–</b></span>
  <span class="stat">FOUND <b id="count">0</b></span>
  <div class="toggle" role="group" aria-label="Map view">
    <button id="btnSat" aria-pressed="true">SATELLITE</button>
    <button id="btnSur" aria-pressed="false">SURVEY</button>
  </div>
</header>

<main>
  <div id="stage">
    <div id="map"></div>
    <svg id="survey" role="img" aria-label="Local survey plot of the robot track and detections"></svg>
  </div>
  <section id="ledger">
    <h2>DETECTION LOG</h2>
    <div id="entries"><p class="empty">No detections yet. Pins appear here as they are found.</p></div>
  </section>
</main>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const SEV_COLOR={trace:'#7fb069',mild:'#e0b64a',moderate:'#e08a3a',severe:'#c25a3a'};
let map,botMarker,trackLine,pinLayer,centred=false,mode='sat',last=null;

function initMap(){
  map=L.map('map',{zoomControl:false,attributionControl:false}).setView([23.78,90.40],18);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:22}).addTo(map);
  L.control.zoom({position:'bottomright'}).addTo(map);
  trackLine=L.polyline([],{color:'#7fb069',weight:2,opacity:.75}).addTo(map);
  pinLayer=L.layerGroup().addTo(map);
}

function setMode(m){
  mode=m;
  document.getElementById('map').style.display   = m==='sat'?'block':'none';
  document.getElementById('survey').style.display= m==='sur'?'block':'none';
  document.getElementById('btnSat').setAttribute('aria-pressed',m==='sat');
  document.getElementById('btnSur').setAttribute('aria-pressed',m==='sur');
  if(m==='sat'&&map) setTimeout(()=>map.invalidateSize(),50);
  if(last) render(last);
}

/* Survey view: everything in metres from the first fix. No network needed. */
function drawSurvey(d){
  const svg=document.getElementById('survey');
  const W=svg.clientWidth||600,H=svg.clientHeight||400;
  const pts=d.track, pins=d.pins.filter(p=>p.lat&&p.lon);
  if(!pts.length){svg.innerHTML=`<text x="50%" y="50%" fill="#8b9680"
     font-family="monospace" font-size="12" text-anchor="middle">Waiting for a GPS fix</text>`;return;}
  const [lat0,lon0]=pts[0], mPerLat=111320, mPerLon=111320*Math.cos(lat0*Math.PI/180);
  const xy=(la,lo)=>[(lo-lon0)*mPerLon,(la-lat0)*mPerLat];
  const all=[...pts.map(p=>xy(p[0],p[1])),...pins.map(p=>xy(p.lat,p.lon))];
  let xs=all.map(a=>a[0]),ys=all.map(a=>a[1]);
  let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const pad=Math.max(3,(Math.max(x1-x0,y1-y0))*0.12);
  x0-=pad;x1+=pad;y0-=pad;y1+=pad;
  const sc=Math.min(W/(x1-x0||1),H/(y1-y0||1));
  const px=x=>(x-x0)*sc, py=y=>H-(y-y0)*sc;
  let g=`<rect width="100%" height="100%" fill="#1b2116"/>`;
  for(let m=Math.ceil(x0/5)*5;m<x1;m+=5)
    g+=`<line x1="${px(m)}" y1="0" x2="${px(m)}" y2="${H}" stroke="#2f3a26" stroke-width="1"/>`;
  for(let m=Math.ceil(y0/5)*5;m<y1;m+=5)
    g+=`<line x1="0" y1="${py(m)}" x2="${W}" y2="${py(m)}" stroke="#2f3a26" stroke-width="1"/>`;
  g+=`<polyline fill="none" stroke="#7fb069" stroke-width="2" opacity=".8" points="${
      pts.map(p=>{const[a,b]=xy(p[0],p[1]);return px(a)+','+py(b)}).join(' ')}"/>`;
  pins.forEach(p=>{const[a,b]=xy(p.lat,p.lon);
    g+=`<circle cx="${px(a)}" cy="${py(b)}" r="6" fill="${SEV_COLOR[p.severity]||'#8b9680'}"
        stroke="#12160f" stroke-width="2"/>`;});
  if(d.bot.lat){const[a,b]=xy(d.bot.lat,d.bot.lon);
    g+=`<circle cx="${px(a)}" cy="${py(b)}" r="9" fill="none" stroke="#a7d189" stroke-width="2"/>
        <circle cx="${px(a)}" cy="${py(b)}" r="3.5" fill="#a7d189"/>`;}
  g+=`<text x="12" y="${H-12}" fill="#8b9680" font-family="monospace" font-size="10">
      grid 5 m · origin at first fix</text>`;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  svg.innerHTML=g;
}

function render(d){
  last=d;
  const b=d.bot;
  document.getElementById('sats').textContent=b.sats??'–';
  document.getElementById('botstate').textContent=b.state||'–';
  document.getElementById('pass').textContent=(b.pass_idx??0)+1;
  document.getElementById('odo').textContent=((b.odo_cm||0)/100).toFixed(1)+' m';
  document.getElementById('count').textContent=d.pins.length;
  document.getElementById('fixdot').className='dot'+(b.fix_ok?' ok':'');

  if(mode==='sur'){drawSurvey(d);}
  else if(map){
    trackLine.setLatLngs(d.track);
    if(b.lat&&b.lon){
      if(!botMarker){
        botMarker=L.circleMarker([b.lat,b.lon],{radius:7,color:'#a7d189',
          fillColor:'#a7d189',fillOpacity:.9,weight:2}).addTo(map);
      } else botMarker.setLatLng([b.lat,b.lon]);
      if(!centred){map.setView([b.lat,b.lon],19);centred=true;}
    }
    pinLayer.clearLayers();
    d.pins.filter(p=>p.lat&&p.lon).forEach(p=>{
      L.circleMarker([p.lat,p.lon],{radius:8,weight:2,color:'#12160f',
        fillColor:SEV_COLOR[p.severity]||'#8b9680',fillOpacity:.95})
       .bindPopup(`<b>${p.disease.replace(/_/g,' ')}</b><br>${p.severity}
         · ${(p.ratio*100).toFixed(1)}% of leaf<br>${p.t}<br>
         <a target="_blank" rel="noopener"
            href="https://www.google.com/maps/search/?api=1&query=${p.lat},${p.lon}"
            >Open in Google Maps</a>`)
       .addTo(pinLayer);
    });
  }

  const box=document.getElementById('entries');
  if(!d.pins.length){
    box.innerHTML='<p class="empty">No detections yet. Pins appear here as they are found.</p>';
  } else {
    box.innerHTML=d.pins.slice().reverse().map((p,i)=>`
      <article class="entry sev-${p.severity}${p.trusted?'':' untrusted'}"
               tabindex="0" data-i="${d.pins.length-1-i}">
        <div class="row1"><span class="name">${p.disease.replace(/_/g,' ')}</span>
          <span class="time">${p.t}</span></div>
        <div class="meta">${p.severity} · ${(p.ratio*100).toFixed(1)}% of leaf ·
          ${(p.confidence*100).toFixed(0)}% conf${p.trusted?'':' · UNVERIFIED'}</div>
        <div class="bar"><i></i></div>
        ${p.lat&&p.lon?`<a class="gmap" target="_blank" rel="noopener"
           href="https://www.google.com/maps/search/?api=1&query=${p.lat},${p.lon}"
           onclick="event.stopPropagation()">OPEN IN GOOGLE MAPS
           &nbsp;${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}</a>`
          :'<span class="nogps">no GPS fix at capture</span>'}
      </article>`).join('');
    box.querySelectorAll('.entry').forEach(el=>el.onclick=()=>{
      const p=d.pins[+el.dataset.i];
      if(p.lat&&map&&mode==='sat'){map.setView([p.lat,p.lon],21);}
    });
  }
}

async function poll(){
  try{ render(await (await fetch('/state')).json()); }
  catch(e){ document.getElementById('botstate').textContent='LINK LOST'; }
}
initMap();
document.getElementById('btnSat').onclick=()=>setMode('sat');
document.getElementById('btnSur').onclick=()=>setMode('sur');
addEventListener('resize',()=>{if(mode==='sur'&&last)drawSurvey(last)});
poll(); setInterval(poll,1000);
</script>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


def start():
    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=S.MAP_PORT,
                               threaded=True, debug=False, use_reloader=False),
        daemon=True).start()
    print(f"[map] http://0.0.0.0:{S.MAP_PORT}")
