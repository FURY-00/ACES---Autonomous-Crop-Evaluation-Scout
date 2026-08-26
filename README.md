# ACES — Autonomous Crop Evaluation Scout

Complete software for the crop-scouting UGV: row navigation, obstacle
avoidance, diseased-leaf detection and classification, SD-card logging,
Telegram alerts, Google Sheets, and a live field map.

---

## What's here

```
aces/
├── config/
│   ├── settings.py            EVERY constant. No magic numbers elsewhere.
│   └── secrets.example.py     → copy to secrets.py, add tokens
├── perception/
│   ├── detector.py            leaf-first HSV/ExG abnormality detection
│   ├── classifier.py          TFLite MobileNetV2, classifies the lesion crop
│   └── disease_camera.py      45° Pi Cam v3, sharpness check, hi-res stills
├── navigation/
│   ├── row_follower.py        forward webcam → offset_cm, heading_deg
│   ├── obstacle_policy.py     3 sonars → speed, dodge, escalation
│   └── serial_link.py         ASCII protocol Pi ↔ ESP32
├── telemetry/
│   ├── gps_reader.py          NEO-M8N, with a real fix-quality gate
│   ├── storage.py             SD layout, CSV log, duplicate suppression
│   ├── telegram_bot.py        farmer's group, non-blocking queue
│   ├── sheets.py              Google Sheets, buffers when offline
│   ├── map_server.py          live map + pins        :8081
│   └── stream_server.py       live mask views        :8080
├── tools/
│   ├── capture_dataset.py     shoot the dataset with LOCKED white balance
│   ├── check_dataset.py       verify it before you tune anything
│   ├── pixel_probe.py         what your lesion pixels actually are
│   ├── stage_debug.py         where in the pipeline pixels die
│   ├── tune_detector.py       live sliders, old vs new side by side
│   ├── evaluate.py            precision / recall / F1, not opinions
│   └── calibrate_row.py       px-per-cm for the nav camera
├── firmware/aces_drive_esp32/ ESP32: motors, encoders, sonar, 100 Hz steering
├── models/                    put plant_disease.tflite + labels.txt here
├── run_bench.py               full pipeline, NO motors  ← start here
└── run_field.py               the whole robot
```

### What you still need to supply

- `models/plant_disease.tflite` and `models/labels.txt` from your training run
- `config/secrets.py` (copy the example, add your Telegram token)
- Real measured values for `PX_PER_CM_NEAR` / `PX_PER_CM_FAR` in settings
- Your own detector thresholds from `tools/pixel_probe.py`

---

## Install

```bash
sudo apt install python3-opencv python3-picamera2 python3-flask
pip install -r requirements.txt --break-system-packages
sudo raspi-config          # enable serial port, DISABLE the serial console
cp config/secrets.example.py config/secrets.py   # then edit it
```

Always run from the repo root, so the `config` / `perception` / `navigation`
packages resolve:

```bash
cd ~/aces && python run_bench.py
```

---

## Bring-up order

Each step fails loudly on its own. Skip one and they fail silently together.

### 1 — Detector (no hardware beyond a camera)

```bash
python tools/capture_dataset.py       # lock WB, shoot diseased/ and healthy/
python tools/check_dataset.py testset/
python tools/stage_debug.py testset/diseased/     # find where pixels die
python tools/pixel_probe.py testset/diseased/     # get real percentiles
# paste percentiles into config/settings.py -> DETECTOR
python tools/tune_detector.py testset/diseased/
python tools/evaluate.py testset/ --sweep
```

Target: recall > 0.90 on `diseased/`, at most one false positive on `healthy/`.

### 2 — Bench (camera + GPS + outputs, no motors)

```bash
python run_bench.py
```

Open `:8080` for masks and `:8081` for the map. Confirm a detection lands on
the SD card, appears as a pin, and arrives in the Telegram group. **Everything
except driving is proven here.** Do not move on until it is.

### 3 — ESP32

Flash `firmware/aces_drive_esp32/`. Serial monitor at 115200; you should see
`#T,...` at 20 Hz.

- Send `$T,90` → the robot should spin exactly 90°. If it spins 70, fix
  `TICKS_PER_REV` / `WHEEL_BASE_CM`, not the gains.
- Send `$Z`, push the robot exactly 100 cm, read `odo_cm`. Must be 100 ± 2.
  The reverse-and-retry feature depends on this.

**Pin conflict warning.** The firmware's default encoder pins (GPIO 32/33)
collide with your existing BTS7960 wiring, which already uses 32/33 for the
left driver and 14/26 for the right. Reconcile the pin map against your real
harness before flashing anything onto the assembled bot.

### 4 — Calibrate the nav camera

```bash
python tools/calibrate_row.py near
python tools/calibrate_row.py far
python tools/calibrate_row.py live
```

In `live`, centre the robot in a row → `offset` ≈ 0. Shift it 10 cm right by
hand → it must read about **+10**. A flipped sign here produces a robot that
steers confidently into the crop, and finding that later costs a day.

### 5 — Wheels off the ground

```bash
python run_field.py --dry-run     # perception only, never commands motors
python run_field.py               # then, wheels still raised
```

### 6 — In the row

Set `CRUISE_SPEED_CMS = 8` for the first run. Walk beside it with a hand on
the kill switch.

---

## The geometry that constrains everything

```
row gap (worst case)   40.0 cm
robot width            32.0 cm
─────────────────────────────────
total slack             8.0 cm   →  4.0 cm per side
safety margin           2.0 cm
usable dodge           ~2.0 cm   ← LAT_LIMIT_CM, derived in settings.py
```

1. **Tracking tolerance is 4 cm.** Exceed it and you are in the plants.
2. **There is no driving around an obstacle.** 2 cm is a nudge, not a
   manoeuvre. The robot slows, stops, waits 8 s, then reverses out and
   messages the group.
3. **The u-turn cannot happen in the row.** A 32 cm robot sweeps ~34 cm. It
   clears `HEADLAND_CLEAR_CM` past the row end first.
4. **Cruise is 12 cm/s, not 18.** At higher speed a 4 cm error is
   unrecoverable inside the reaction distance.

---

## Design decisions worth knowing

**The Pi never writes a PWM value.** Linux is not real-time; if the TFLite
classifier stalls for 300 ms mid-row, a Pi-driven steering loop drives you
into the crop. The ESP32 runs a 100 Hz loop on the last vision estimate,
halves speed when it goes stale, and stops itself after 1.5 s of silence.

**The side sonars are the best sensor on this robot.** At a 5 cm standoff
HC-SR04 is accurate to millimetres and indifferent to light. `fuse_sonar()`
blends `(left − right)/2` into the vision estimate, weighted up as vision
confidence falls. When the camera is blinded by dust or low sun, the sonars
quietly take over centring.

**Never share a sonar trigger line.** Three HC-SR04 on one trigger hear each
other's bursts and invent walls. The firmware fires them round-robin, 60 ms
apart, each with its own 5-sample median ring.

**The detector finds the leaf without using hue.** Necrotic tissue fails any
plant-hue test by definition — that was the original bug. A brown lesion
inside a green leaf is a *hole* in the green mask, so filling holes recovers
the true leaf outline, lesion included. Measured on a synthetic leaf: brown
necrosis recovery went from 4.7% to 97.6%.

**The system is allowed to say "I don't know."** Shadow and glare go to an
`unknown` mask instead of being counted as disease. Over 25% unjudgeable, or
classifier confidence under 60%, and the result goes to `data/uncertain/` and
is never sent to the farmer. A wrong pin costs trust; a missing pin costs one
leaf.

**Duplicate suppression is not optional.** The bot sees the same lesion on
both passes and across many frames. Without the 3 m radius check, one sick
plant becomes thirty pins and the map stops being useful.

**The map works offline.** Tiles need internet, which a field does not have.
The "Survey" view plots the run in local metres from the first GPS fix as a
self-contained SVG — no network, and arguably more useful in a crop row.

---

## Known gaps

- **Row-end IR sensors are the weakest signal.** `_rowend()` needs 2 of 3
  votes (vision, odometry, IR) held for 8 frames because IR alone sees glare
  and bare patches.
- **Dead reckoning drifts** on damp soil. Trusted for ~2 s only; vision
  re-anchors it the moment confidence returns.
- **A pure spin turn needs headland space.** Under ~60 cm you need a
  three-point turn; the `S_TURN` case in the firmware is where it goes.
- **Only one row is covered per run.** Multi-row coverage (turn, translate to
  the next row, repeat) is not implemented.
- **`labels.txt` order is unverified.** Check it against one known image
  before any field run.
