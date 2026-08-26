"""
Angle stability test.

Your problem (c) -- "at different angles the abnormality region keeps
changing" -- is the one that actually matters, because a robot photographs
every leaf from whatever angle it happens to drive past. A detector that is
right on average but swings 15% between shots is useless for severity.

This measures the swing directly. Photograph the SAME leaf from several
angles, then this reports how much the detected ratio moves. The leaf did
not change between shots, so every bit of movement is detector error.

Usage
-----
    python tools/angle_test.py --capture           # shoot the series
    python tools/angle_test.py angles/leaf01/      # score an existing series

Capture protocol
----------------
Keep the SAME leaf, the same distance, and the same light. Change ONLY the
angle. Take 8 shots: 4 rotations of the leaf in its own plane (0/90/180/270)
and 4 tilts toward and away from the light. Press SPACE for each.

Reading the result
------------------
    ratio swing   < 3 percentage points   good enough for severity bands
                  3-8 pp                  usable for detection, not severity
                  > 8 pp                  the detector is measuring the light,
                                          not the leaf

If the swing is large, the "unjudged" column usually tells you why: specular
highlights moving across the leaf. Lower k_specular in settings, or diffuse
your light source (shoot in open shade, or put paper over the lamp).
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings as S        # noqa: E402
from perception import detector as D    # noqa: E402


def capture(outdir):
    os.makedirs(outdir, exist_ok=True)
    try:
        from libcamera import controls
        from picamera2 import Picamera2
        pc = Picamera2()
        pc.configure(pc.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"}))
        pc.start()
        time.sleep(2)
        pc.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        time.sleep(1.5)
        md = pc.capture_metadata()
        pc.set_controls({"AwbEnable": False,
                         "ColourGains": md.get("ColourGains", (1.8, 1.8))})
        grab = lambda: pc.capture_array()  # RGB888 array is already BGR
        stop = pc.stop
    except Exception as e:
        print(f"picamera2 unavailable ({e}), using USB")
        cap = cv2.VideoCapture(0)
        grab = lambda: cap.read()[1]
        stop = cap.release

    print("\nSame leaf, same distance, same light. Change ONLY the angle.")
    print("SPACE = shoot   ESC = done\n")
    n = 0
    while True:
        f = grab()
        if f is None:
            continue
        r = D.detect(cv2.resize(f, (960, 540)))
        v = D.overlay(cv2.resize(f, (960, 540)), r)
        cv2.putText(v, f"shot {n}   SPACE to capture", (8, 530),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("angle test", v)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        if k == 32:
            cv2.imwrite(os.path.join(outdir, f"angle_{n:02d}.jpg"), f,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  shot {n}")
            n += 1
    stop()
    cv2.destroyAllWindows()
    print(f"\n{n} shots -> {outdir}")
    return outdir


def score(folder):
    paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if len(paths) < 3:
        print("need at least 3 shots of the same leaf")
        return

    print(f"\n{'shot':>5s} {'ratio':>8s} {'blobs':>6s} {'leaf%':>7s} "
          f"{'unjudged':>9s} {'d_ref':>8s} {'trusted':>8s}")
    print("-" * 62)
    rows = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, (960, 540))
        r = D.detect(img)
        unk = (float(r.unknown_mask.sum()) / 255 / max(r.leaf_px, 1)
               if r.unknown_mask is not None else 0)
        leafpct = 100 * r.leaf_px / (img.shape[0] * img.shape[1])
        rows.append((r.ratio, len(r.blobs), unk, r.d_ref, r.trusted))
        print(f"{os.path.basename(p)[-6:-4]:>5s} {r.ratio:8.2%} "
              f"{len(r.blobs):6d} {leafpct:6.1f}% {unk:9.1%} "
              f"{r.d_ref:+8.3f} {str(r.trusted):>8s}")

    ratios = np.array([r[0] for r in rows])
    unks = np.array([r[2] for r in rows])
    swing = (ratios.max() - ratios.min()) * 100
    print("-" * 62)
    print(f"  mean ratio    {ratios.mean():.2%}")
    print(f"  ratio SWING   {swing:.1f} percentage points   "
          f"(min {ratios.min():.2%}, max {ratios.max():.2%})")
    print(f"  blob count    {min(r[1] for r in rows)} to {max(r[1] for r in rows)}")
    print(f"  unjudged      {unks.mean():.1%} mean, {unks.max():.1%} worst")

    print()
    if swing < 3:
        print("  GOOD. Stable enough to report severity bands.")
    elif swing < 8:
        print("  USABLE for detection, NOT for severity. The severity band")
        print("  will flip between shots of the same leaf.")
    else:
        print("  TOO UNSTABLE. The detector is measuring your lighting, not")
        print("  the leaf.")
    if unks.max() > 0.2:
        print("  Specular highlights are the likely cause "
              f"({unks.max():.0%} of the leaf unjudged at worst).")
        print("  Try: shoot in open shade, diffuse the lamp with paper, or")
        print("  lower k_specular in config/settings.py.")
    if any(not r[4] for r in rows):
        n_bad = sum(1 for r in rows if not r[4])
        print(f"  {n_bad}/{len(rows)} shots were self-flagged untrusted - the")
        print("  detector already knew those were bad. On the robot they would")
        print("  trigger a re-shoot rather than being reported.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="angles/leaf01")
    ap.add_argument("--capture", action="store_true")
    a = ap.parse_args()
    if a.capture:
        capture(a.folder)
    score(a.folder)
