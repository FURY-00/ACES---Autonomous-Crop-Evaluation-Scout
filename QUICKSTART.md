# Quickstart — detecting a leaf today

Pi + Pi Camera only. No motors, no ESP32, no GPS.

## 1. Open a terminal and go to the project

```bash
cd ~/Desktop/acesss/aces
ls
```

You should see `run_bench.py`, `config`, `perception`, `tools`.
If instead you see another `aces` folder, go one level deeper:
`cd aces`. The rule: **you must be in the folder that contains `config/`.**

## 2. Install what's needed

```bash
sudo apt update
sudo apt install -y python3-opencv python3-picamera2 python3-flask python3-numpy
```

Everything else (GPS, serial, TFLite) is NOT needed today.

## 3. Prove the camera works

```bash
python3 tools/check_camera.py
```

Must print `Pi Camera works`. If not, stop and fix that first —
`rpicam-hello -t 3000` is the fastest check that the hardware is alive.

**Note:** `cv2.VideoCapture(0)` does not open a Pi Camera v3 on Bookworm.
If your old code used it, that alone explains a black frame.

## 4. Detect

```bash
python3 tools/live_detect.py
```

Then open `http://<pi-ip>:8080` in any browser on the same network.
Find the IP with `hostname -I`.

On the Pi's own monitor you can add `--window --save` for a local window
where SPACE saves the current frame.

### Reading the overlay

| colour | meaning |
|---|---|
| blue outline | the leaf the detector found |
| red fill | judged abnormal |
| grey fill | shadow or glare — deliberately not judged |
| yellow box | accepted lesion blob |

**Fix the blue outline first.** If it isn't hugging your leaf, nothing
downstream can be right. Edit `exg_thresh` in `config/settings.py`
(lower = more permissive) and re-run.

## 5. When it's wrong, tune it properly

```bash
python3 tools/capture_dataset.py          # shoot diseased/ and healthy/
python3 tools/check_dataset.py testset/
python3 tools/stage_debug.py testset/diseased/
python3 tools/pixel_probe.py testset/diseased/
# paste the percentiles into config/settings.py -> DETECTOR
python3 tools/tune_detector.py testset/diseased/
python3 tools/evaluate.py testset/ --sweep
```

The windowed tools need a screen. Over SSH use `ssh -X`, or enable VNC
(`sudo raspi-config` → Interface Options → VNC), or work on the Pi's monitor.
`live_detect.py` is the only one that needs no display at all.
