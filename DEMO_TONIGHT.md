# Demo tomorrow — tonight's plan

No nav webcam. Autonomy comes from the **two side ultrasonics**: their
difference IS the cross-track error, in centimetres, with no calibration.

    error = (left - right) / 2      + = right of centre

## The demo you are aiming for

> The bot drives itself down a corridor, keeping centred with no human input.
> When its camera sees a diseased leaf, it stops on its own, photographs it,
> drops a pin on a live map, and carries on. At the end of the corridor it
> stops and reports.

That is genuinely autonomous, uses only working hardware, and shows off the
detector you spent two weeks on.

---

## Time budget — do these in order, stop at whatever you reach

| # | Task | Time | Skip if short on time? |
|---|---|---|---|
| 1 | Flash demo firmware, verify RC still works | 45 min | NO |
| 2 | Wire + verify the two sonars | 45 min | NO |
| 3 | Corridor test, tune KP | 60 min | NO |
| 4 | Connect Pi, full loop | 45 min | NO |
| 5 | GPS + map pins | 30 min | yes — pins can be indoor-fake |
| 6 | Telegram | 15 min | yes |
| 7 | Dry run x3 | 30 min | NO |

**Hard rule: if step 3 is not working by 22:00, fall back.** See the bottom.

---

## Step 1 — Firmware (45 min)

Flash `firmware/aces_demo_esp32/aces_demo_esp32.ino`.

It keeps your existing PPM RC decode on GPIO 16 and your BTS7960 pins
(32/33, 14/26). It adds one thing:

- **CH5 LOW  = MANUAL** — your normal RC control
- **CH5 HIGH = AUTO** — the robot drives itself

**Test with the wheels off the ground.**

1. CH5 low, sticks — all four motors respond as before.
2. CH5 high — motors stop, serial prints `#E,AUTO`.
3. Drop CH5 — sticks work instantly again.

If the transmitter is off, the robot will not move at all. That is deliberate.

> Your reverse-side diagonals were mirrored in RC. In AUTO the mixing is done
> in this firmware, so that bug does not affect the demo.

## Step 2 — Sonars (45 min)

    Left  sonar: TRIG 5,  ECHO 18
    Right sonar: TRIG 17, ECHO 23

**Put a 1k/2k divider on each ECHO line.** They are 5 V and the ESP32 is 3.3 V.

Open the serial monitor at 115200. You get `#T,...` at 10 Hz:

    #T,<ms>,<mode>,<left_cm>,<right_cm>,<error>,<wall_l>,<wall_r>

Hold a book 20 cm from the left sensor — `left_cm` should read ~20 and stay
steady. Wandering by more than 2 cm means a loose echo wire or a bad divider.
**Do not proceed on noisy sonar readings.** Everything downstream is built on
these two numbers.

## Step 3 — Corridor (60 min) — the make-or-break step

Build a corridor from cardboard boxes, books, or chairs:

- **55–60 cm wide** (your bot is 32 cm — do not use 42 cm for a first demo)
- **at least 3 m long**
- walls at least 15 cm tall so the sonars see them
- flat, hard floor

Set the bot in the middle, CH5 high. It will not move until the Pi says go,
so for a firmware-only test temporarily change `AUTO_IDLE` to go straight to
`AUTO_RUN`, or just run step 4 first.

Tuning, one at a time:

- **weaves side to side** → lower `KP` (6.0 → 4.0)
- **drifts into a wall and stays** → raise `KP` (6.0 → 8.0)
- **jitters fast** → raise `KD`
- **too fast to control** → lower `BASE_PWM` (100 → 80)

## Step 4 — Pi in the loop (45 min)

USB cable from Pi to ESP32.

    ls /dev/ttyUSB*          # usually /dev/ttyUSB0
    cd ~/Desktop/acesss/aces
    python3 run_demo.py --no-drive     # FIRST: proves perception, motors dead
    python3 run_demo.py                # then the real thing

Two tabs: `:8080` detection, `:8081` map.

Hold your tuned leaf in front of the camera. The bot should stop, log, and
resume. Check the readout says `state STOPPING` then `CAPTURING`.

## Step 4b — Your trained model (20 min)

The classifier is now wired into `run_demo.py`. It runs ONLY at the moment of
capture, when the robot is already stopped, so it costs nothing while driving.

    cp your_model.tflite  ~/Desktop/acesss/aces/models/plant_disease.tflite
    cp your_labels.txt    ~/Desktop/acesss/aces/models/labels.txt
    pip install tflite-runtime --break-system-packages
    python3 tools/check_model.py testset/diseased/

`check_model.py` catches the silent killer: if `labels.txt` is not in the same
order as the model's output indices, the robot names the WRONG disease with
total confidence and nothing crashes. Run it against images whose class you
already know and check the names come out right.

If the model will not load, the demo still runs — detections are logged as
`abnormal_leaf` with no disease name. Nothing breaks. You can also force that
with `--no-classify`.

## Step 5 — GPS (30 min, optional)

NEO-M8N needs **open sky** and 30–90 s for a cold fix. It will not fix
indoors. If you are demoing inside, say so plainly — the pins will show at
the last known fix, and that is fine.

## Step 7 — Dry runs (30 min) — do not skip

Run it end to end **three times**. Time it. Watch what breaks. A demo that has
never been run twice will fail in front of your teacher.

---

## FALLBACK — if sonar following is not working by 22:00

Stop. Do not keep debugging. Demo this instead:

    python3 run_demo.py --no-drive

and **drive the bot manually by RC** while the Pi autonomously detects,
photographs, logs, pins on the map and messages Telegram.

Then say exactly this: *"Navigation is sonar-based and is on the bench; the
perception and telemetry loop you are seeing is fully autonomous."*

That is an honest, complete demo of the hard part. Overclaiming and having it
fail is much worse than scoping honestly.

---

## What to say when they ask "why no camera navigation?"

> We use two side ultrasonics for row centring. Their difference gives
> cross-track error directly in centimetres, with no calibration and no
> lighting dependence — in a 42 cm row gap that is more accurate than the
> camera would be. The camera is dedicated to disease detection, where colour
> actually matters.

That is a design decision, not an excuse, and it is true.

---

## Pre-demo checklist

- [ ] LiPo fully charged, spare ready — your pack sags at full throttle
- [ ] `BASE_PWM` at 100 or below (a brownout mid-demo looks terrible)
- [ ] Transmitter ON, in someone's hand, CH5 as the kill switch
- [ ] Corridor 55–60 cm, 3 m, on flat floor
- [ ] Your tuned diseased leaf, plus the flashlight you tuned under
- [ ] Pi and laptop on the same hotspot; know the Pi's IP (`hostname -I`)
- [ ] Both browser tabs open BEFORE you start
- [ ] `config/detector_tuned.json` present on the Pi
- [ ] `models/plant_disease.tflite` + `labels.txt` in place, `check_model.py` clean
- [ ] Run once, right before they walk in
