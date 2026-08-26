"""
Run this the moment the photo session ends, BEFORE you start tuning.

Ten minutes here saves you from fitting thresholds to a broken dataset and
then spending two days wondering why the robot disagrees with your laptop.

Usage
-----
    python check_dataset.py testset/

Checks
------
  counts        enough images, and a real healthy set, not an afterthought
  white balance did the lock actually hold across the session
  exposure      are shots clipped black or blown white
  sharpness     motion blur that will destroy small lesion detail
  colour split  do diseased and healthy images actually differ in colour
  printed       photos of paper sneaking back into the set
"""

import os
import sys
from collections import Counter

import cv2
import numpy as np


def load_paths(root):
    out = {}
    for cls in ("diseased", "healthy"):
        d = os.path.join(root, cls)
        out[cls] = ([os.path.join(d, f) for f in sorted(os.listdir(d))
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                    if os.path.isdir(d) else [])
    return out


def stats(path):
    img = cv2.imread(path)
    if img is None:
        return None
    small = cv2.resize(img, (480, 360))
    b, g, r = [float(c.mean()) for c in cv2.split(small)]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return dict(
        path=path,
        rg=r / max(g, 1e-6), bg=b / max(g, 1e-6),
        v_mean=float(V.mean()),
        clip_lo=float((V < 8).mean()), clip_hi=float((V > 250).mean()),
        sat_p95=float(np.percentile(S, 95)),
        sharp=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        white_frac=float(((S < 40) & (V > 195)).mean()),
        hue_med=float(np.median(H[S > 50])) if (S > 50).any() else 0.0,
        tag=os.path.basename(path).split("_")[0],
    )


def section(title):
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def main(root):
    paths = load_paths(root)
    rows = {c: [s for s in (stats(p) for p in paths[c]) if s] for c in paths}
    nd, nh = len(rows["diseased"]), len(rows["healthy"])
    problems = []

    section("COUNTS")
    print(f"  diseased  {nd}")
    print(f"  healthy   {nh}")
    if nd < 25:
        problems.append(f"only {nd} diseased images - aim for 30+")
    if nh < 15:
        problems.append(f"only {nh} healthy images - you need these to catch "
                        "false positives")
    if nh and nd / max(nh, 1) > 4:
        problems.append("healthy set is far smaller than diseased - precision "
                        "numbers will be meaningless")

    tags = Counter(s["tag"] for c in rows for s in rows[c])
    print("  lighting tags:", dict(tags))
    if len(tags) < 2:
        problems.append("all images shot under one lighting condition - "
                        "thresholds will not survive the field")

    section("WHITE BALANCE  (did the lock hold?)")
    allr = [s for c in rows for s in rows[c]]
    rg = np.array([s["rg"] for s in allr])
    bg = np.array([s["bg"] for s in allr])
    print(f"  R/G  mean {rg.mean():.3f}   spread (p5-p95) "
          f"{np.percentile(rg,5):.3f} - {np.percentile(rg,95):.3f}")
    print(f"  B/G  mean {bg.mean():.3f}   spread (p5-p95) "
          f"{np.percentile(bg,5):.3f} - {np.percentile(bg,95):.3f}")
    spread = (np.percentile(rg, 95) - np.percentile(rg, 5)) + \
             (np.percentile(bg, 95) - np.percentile(bg, 5))
    if spread > 0.35:
        problems.append(f"white balance varies a lot across the set "
                        f"(spread {spread:.2f}) - AWB was probably still on, "
                        "or you re-locked mid-session")
    else:
        print("  OK - white balance is consistent")

    section("EXPOSURE AND SHARPNESS")
    for c in ("diseased", "healthy"):
        if not rows[c]:
            continue
        v = np.array([s["v_mean"] for s in rows[c]])
        sh = np.array([s["sharp"] for s in rows[c]])
        print(f"  {c:9s} brightness {v.mean():5.1f}   "
              f"sharpness median {np.median(sh):7.0f}")
    dark = [s for s in allr if s["clip_lo"] > 0.25]
    blown = [s for s in allr if s["clip_hi"] > 0.12]
    blurry = [s for s in allr if s["sharp"] < 90]
    for name, lst in (("crushed blacks", dark), ("blown highlights", blown),
                      ("blurry", blurry)):
        if lst:
            problems.append(f"{len(lst)} images with {name}")
            for s in lst[:4]:
                print(f"    {name:18s} {os.path.basename(s['path'])}")

    section("COLOUR SEPARATION  (do the two classes actually differ?)")
    if nd and nh:
        for key, label in (("hue_med", "median hue"), ("sat_p95", "sat p95")):
            d = np.array([s[key] for s in rows["diseased"]])
            h = np.array([s[key] for s in rows["healthy"]])
            sep = abs(d.mean() - h.mean()) / (np.sqrt(d.var() + h.var()) + 1e-6)
            print(f"  {label:12s} diseased {d.mean():6.1f}   "
                  f"healthy {h.mean():6.1f}   separation {sep:.2f}")
        d = np.array([s["hue_med"] for s in rows["diseased"]])
        h = np.array([s["hue_med"] for s in rows["healthy"]])
        if abs(d.mean() - h.mean()) < 3:
            problems.append("diseased and healthy images have nearly identical "
                            "colour - check you did not mix up the folders, or "
                            "the lesions are too small a fraction of the frame")

    section("PRINTED-PAPER CHECK")
    printed = [s for s in allr if s["white_frac"] > 0.18 and s["sat_p95"] < 165]
    if printed:
        problems.append(f"{len(printed)} images still look like photos of paper")
        for s in printed[:5]:
            print(f"    {os.path.basename(s['path'])}  white={s['white_frac']:.2f} "
                  f"sat_p95={s['sat_p95']:.0f}")
    else:
        print("  OK - none detected")

    print("\n" + "=" * 60)
    if problems:
        print("FIX THESE BEFORE TUNING:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("Dataset looks good. Next:  python pixel_probe.py "
              f"{os.path.join(root, 'diseased')}")
    print("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "testset")
