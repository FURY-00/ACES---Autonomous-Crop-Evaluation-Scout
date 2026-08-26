"""
STEP 1 TOOL: pixel forensics.

Before you touch a single threshold, you must know what your lesion pixels
actually ARE. Not what you think they are. Not what a tutorial says brown is.
What YOUR camera, on YOUR leaf, under YOUR light, actually records.

Usage
-----
    python pixel_probe.py <image_or_folder>

Controls
--------
    left click   sample a pixel, tagged with the current class
    1            set current class = LESION   (yellow/brown/black tissue)
    2            set current class = HEALTHY  (clearly green tissue)
    3            set current class = BACKGROUND (soil, sky, paper, hand)
    n / p        next / previous image
    u            undo last sample
    s            save samples to probe_samples.json and print the report
    ESC          quit

What you do with it
-------------------
Sample at least 30 lesion pixels across at least 5 different images, spread
over pale yellow, orange-brown, and dark necrotic black. Then press 's'.
The report prints the 5th-95th percentile of H, S and V for each class.
Those percentiles ARE your thresholds. You never guess again.
"""

import json
import os
import sys

import cv2
import numpy as np

CLASSES = {1: "LESION", 2: "HEALTHY", 3: "BACKGROUND"}
COLORS = {"LESION": (0, 0, 255), "HEALTHY": (0, 255, 0), "BACKGROUND": (255, 0, 0)}

samples = []          # list of dicts
cur_class = "LESION"
cur_img = None
cur_name = ""


def on_mouse(event, x, y, flags, _):
    global samples
    if event != cv2.EVENT_LBUTTONDOWN or cur_img is None:
        return
    # Average a 3x3 patch. A single pixel is sensor noise, not evidence.
    h, w = cur_img.shape[:2]
    x0, x1 = max(0, x - 1), min(w, x + 2)
    y0, y1 = max(0, y - 1), min(h, y + 2)
    patch = cur_img[y0:y1, x0:x1]

    bgr = patch.reshape(-1, 3).mean(axis=0)
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]
    b, g, r = bgr
    exg = 2 * g - r - b

    samples.append({
        "image": cur_name, "x": x, "y": y, "cls": cur_class,
        "B": float(b), "G": float(g), "R": float(r),
        "H": int(hsv[0]), "S": int(hsv[1]), "V": int(hsv[2]),
        "L": int(lab[0]), "a": int(lab[1]), "b_": int(lab[2]),
        "ExG": float(exg),
    })
    print(f"  {cur_class:10s} H={hsv[0]:3d} S={hsv[1]:3d} V={hsv[2]:3d}   "
          f"L={lab[0]:3d} a={lab[1]:3d} b={lab[2]:3d}   ExG={exg:+6.1f}")


def report():
    if not samples:
        print("no samples yet")
        return
    print("\n" + "=" * 64)
    print("PERCENTILE REPORT  (5th - 95th)   <- these are your thresholds")
    print("=" * 64)
    for cls in ("LESION", "HEALTHY", "BACKGROUND"):
        rows = [s for s in samples if s["cls"] == cls]
        if not rows:
            continue
        print(f"\n{cls}   n={len(rows)}")
        for key in ("H", "S", "V", "L", "a", "b_", "ExG"):
            v = np.array([r[key] for r in rows], dtype=float)
            print(f"   {key:4s}  p5={np.percentile(v, 5):7.1f}   "
                  f"median={np.median(v):7.1f}   p95={np.percentile(v, 95):7.1f}")

    les = [s for s in samples if s["cls"] == "LESION"]
    hea = [s for s in samples if s["cls"] == "HEALTHY"]
    if les and hea:
        print("\n" + "-" * 64)
        print("SEPARABILITY  (how cleanly each channel splits lesion from healthy)")
        print("-" * 64)
        for key in ("H", "S", "V", "a", "b_", "ExG"):
            l = np.array([s[key] for s in les], float)
            h = np.array([s[key] for s in hea], float)
            # Fisher-style score: gap between means over pooled spread.
            score = abs(l.mean() - h.mean()) / (np.sqrt(l.var() + h.var()) + 1e-6)
            bar = "#" * int(min(score, 6) * 6)
            print(f"   {key:4s}  {score:5.2f}  {bar}")
        print("\n   The channel with the highest score is the one to threshold on.")
        print("   If Hue is NOT at the top, your two-stage HSV approach is")
        print("   fighting the data, and that alone explains a bad mask.")


def main(target):
    global cur_img, cur_name, cur_class
    paths = ([os.path.join(target, f) for f in sorted(os.listdir(target))
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
             if os.path.isdir(target) else [target])
    if not paths:
        print("no images found")
        return

    idx = 0
    cv2.namedWindow("probe")
    cv2.setMouseCallback("probe", on_mouse)

    while True:
        img = cv2.imread(paths[idx])
        if img is None:
            idx = (idx + 1) % len(paths)
            continue
        scale = min(1.0, 900 / max(img.shape[:2]))
        img = cv2.resize(img, None, fx=scale, fy=scale)
        cur_img, cur_name = img, os.path.basename(paths[idx])

        while True:
            disp = cur_img.copy()
            for s in samples:
                if s["image"] == cur_name:
                    cv2.circle(disp, (s["x"], s["y"]), 4, COLORS[s["cls"]], -1)
                    cv2.circle(disp, (s["x"], s["y"]), 5, (255, 255, 255), 1)
            n_les = sum(1 for s in samples if s["cls"] == "LESION")
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(disp, f"[{idx+1}/{len(paths)}] {cur_name}   class={cur_class}"
                              f"   lesion samples={n_les}", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cur_class], 1)
            cv2.imshow("probe", disp)

            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                report()
                cv2.destroyAllWindows()
                return
            if k in (ord("1"), ord("2"), ord("3")):
                cur_class = CLASSES[k - ord("0")]
            elif k == ord("u") and samples:
                samples.pop()
            elif k == ord("s"):
                with open("probe_samples.json", "w") as fh:
                    json.dump(samples, fh, indent=1)
                print(f"\nsaved {len(samples)} samples -> probe_samples.json")
                report()
            elif k == ord("n"):
                idx = (idx + 1) % len(paths)
                break
            elif k == ord("p"):
                idx = (idx - 1) % len(paths)
                break


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
