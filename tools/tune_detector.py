"""
STEP 4 TOOL: live tuner.

Shows the new detector and your old two-stage logic side by side on the same
frame, with every parameter on a trackbar. Tune while looking at the truth.

Usage
-----
    python tune_v2.py <image_or_folder>          # from files
    python tune_v2.py cam                        # live from the Pi camera

Controls
--------
    n / p     next / previous image
    d         dump the current parameters as a Python dict, ready to paste
    ESC       quit

The one rule of tuning
----------------------
Change one slider at a time and watch ONE image you understand completely.
Tuning against a folder of 200 images by eye means you are averaging your
own confusion. Get one image perfect, then check it did not break the others.
"""

import os
import sys

import cv2
import numpy as np

from perception import detector as V

WIN = "tuner"

SLIDERS = [
    ("healthy_h_lo",  33,  179),
    ("healthy_h_hi",  88,  179),
    ("healthy_s_min", 45,  255),
    ("healthy_v_min", 35,  255),
    ("exg_thresh",    15,  120),
    ("close_k",        9,   31),
    ("shadow_v_max",  32,  120),
    ("glare_s_max",   28,  120),
    ("glare_v_min",  225,  255),
    ("min_blob_px",  400, 5000),
]


def read_params():
    g = lambda n: cv2.getTrackbarPos(n, WIN)
    k = max(1, g("close_k") | 1)          # kernel must be odd
    return {
        "healthy_h": (g("healthy_h_lo"), max(g("healthy_h_lo") + 1, g("healthy_h_hi"))),
        "healthy_s_min": g("healthy_s_min"),
        "healthy_v_min": g("healthy_v_min"),
        "exg_thresh": g("exg_thresh"),
        "close_k": k,
        "shadow_v_max": g("shadow_v_max"),
        "glare_s_max": g("glare_s_max"),
        "glare_v_min": g("glare_v_min"),
        "min_blob_px": g("min_blob_px"),
    }


def old_logic(bgr):
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    plant = cv2.inRange(hsv, (20, 30, 25), (95, 255, 255))
    healthy = cv2.inRange(hsv, (36, 60, 40), (90, 255, 255))
    ab = cv2.subtract(plant, healthy)
    k = np.ones((5, 5), np.uint8)
    ab = cv2.morphologyEx(ab, cv2.MORPH_OPEN, k)
    out = bgr.copy()
    out[ab > 0] = (0.35 * out[ab > 0] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, f"OLD two-stage   {int(ab.sum()/255):,} px", (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return out


def main(target):
    if target == "cam":
        cap = cv2.VideoCapture(0)
        paths = None
    else:
        cap = None
        paths = ([os.path.join(target, f) for f in sorted(os.listdir(target))
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                 if os.path.isdir(target) else [target])
        if not paths:
            print("no images found")
            return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    for name, default, hi in SLIDERS:
        cv2.createTrackbar(name, WIN, default, hi, lambda _v: None)

    idx = 0
    while True:
        if cap:
            ok, img = cap.read()
            if not ok:
                break
        else:
            img = cv2.imread(paths[idx])
            if img is None:
                idx = (idx + 1) % len(paths)
                continue
        s = min(1.0, 620 / max(img.shape[:2]))
        img = cv2.resize(img, None, fx=s, fy=s)

        p = read_params()
        r = V.detect(img, p)
        new = V.overlay(img, r)
        old = old_logic(img)

        h = max(new.shape[0], old.shape[0])
        pad = lambda a: cv2.copyMakeBorder(a, 0, h - a.shape[0], 0, 0,
                                           cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.imshow(WIN, np.hstack([pad(old), pad(new)]))

        k = cv2.waitKey(30 if cap else 60) & 0xFF
        if k == 27:
            break
        elif k == ord("n") and paths:
            idx = (idx + 1) % len(paths)
        elif k == ord("p") and paths:
            idx = (idx - 1) % len(paths)
        elif k == ord("d"):
            print("\nP = {")
            for kk, vv in p.items():
                print(f'    "{kk}": {vv!r},')
            print("}\n")

    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
