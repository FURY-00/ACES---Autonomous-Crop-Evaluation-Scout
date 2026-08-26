"""
Offline tuner: search for parameter values that work across a WHOLE FOLDER.

Runs on your laptop. No Pi, no camera. This is how tuning should be done --
sliders in a field optimise for the one leaf in front of you, which is
exactly how you end up with settings that fail on the next leaf.

Setup
-----
    dataset/
      diseased/     images that DO have abnormal tissue
      healthy/      clean leaves, no disease
      notleaf/      hands, soil, sky, shirts, buildings   <- optional but
                    this is the folder that stops false alarms

Usage
-----
    python3 tools/offline_tune.py dataset/
    python3 tools/offline_tune.py dataset/ --quick        # coarse, ~1 min
    python3 tools/offline_tune.py dataset/ --write        # save the winner

Scoring
-------
    recall     fraction of diseased images correctly flagged
    precision  of everything flagged, how much was really diseased
    F-beta     precision weighted more heavily than recall (beta 0.5)

Precision is weighted higher on purpose. On a field robot a false positive
costs a wasted stop, a wasted capture, and a wrong pin on the farmer's map.
A miss costs one leaf, and the robot will pass hundreds.
"""

import argparse
import itertools
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings as S            # noqa: E402
from perception import detector as D        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "config", "detector_tuned.json")

GRID_FULL = {
    "bg_k":              [2.0, 3.0, 4.5, 6.0],
    "exg_thresh":        [8, 15, 25],
    "min_d_ref":         [-0.06, -0.02, 0.02, 0.06],
    "d_healthy_foliage": [0.05, 0.09, 0.12, 0.16],
    "d_abs_max":         [0.04, 0.06, 0.075, 0.10],
}
GRID_QUICK = {
    "bg_k":              [3.0, 5.0],
    "exg_thresh":        [15],
    "min_d_ref":         [-0.02, 0.02],
    "d_healthy_foliage": [0.09, 0.12],
    "d_abs_max":         [0.06, 0.09],
}


def load(folder):
    sets = {}
    for cls in ("diseased", "healthy", "notleaf"):
        dd = os.path.join(folder, cls)
        paths = ([os.path.join(dd, f) for f in sorted(os.listdir(dd))
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                 if os.path.isdir(dd) else [])
        imgs = []
        for pth in paths:
            im = cv2.imread(pth)
            if im is not None:
                imgs.append((os.path.basename(pth), cv2.resize(im, (640, 360))))
        sets[cls] = imgs
    return sets


def fires(img, p):
    r = D.detect(img, p)
    return (r.ratio >= S.DETECT_RATIO_MIN and bool(r.blobs) and r.trusted), r


def evaluate(sets, p):
    tp = sum(1 for _, im in sets["diseased"] if fires(im, p)[0])
    fn = len(sets["diseased"]) - tp
    fp = sum(1 for _, im in sets["healthy"] if fires(im, p)[0])
    fp += sum(1 for _, im in sets["notleaf"] if fires(im, p)[0])
    tn = len(sets["healthy"]) + len(sets["notleaf"]) - fp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    beta = 0.5
    f = ((1 + beta ** 2) * prec * rec / max(beta ** 2 * prec + rec, 1e-9))
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, precision=prec, recall=rec, f=f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()

    sets = load(a.folder)
    n = {k: len(v) for k, v in sets.items()}
    print(f"\nimages: diseased {n['diseased']}, healthy {n['healthy']}, "
          f"notleaf {n['notleaf']}")
    if n["diseased"] == 0:
        print(f"\nNo images under {a.folder}/diseased/. Nothing to tune against.")
        return
    if n["healthy"] + n["notleaf"] == 0:
        print("\n! No healthy/ or notleaf/ images. Precision cannot be measured,")
        print("  so the search will happily pick settings that flag EVERYTHING.")
        print("  Add negatives before trusting any result.")

    grid = GRID_QUICK if a.quick else GRID_FULL
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"testing {len(combos)} combinations x {sum(n.values())} images "
          f"= {len(combos)*sum(n.values()):,} detections\n")

    base = dict(S.DETECTOR)
    results = []
    for i, vals in enumerate(combos):
        p = {**base, **dict(zip(keys, vals))}
        p["d_abs_strong"] = p["d_abs_max"] - 0.02
        m = evaluate(sets, p)
        results.append((m, dict(zip(keys, vals))))
        if (i + 1) % 10 == 0 or i == len(combos) - 1:
            print(f"  {i+1}/{len(combos)}", end="\r")

    results.sort(key=lambda x: (-x[0]["f"], x[0]["fp"], -x[0]["recall"]))

    print("\n\n" + "=" * 78)
    print(f"{'F':>5s} {'prec':>5s} {'rec':>5s} {'TP':>3s} {'FN':>3s} {'FP':>3s}"
          f"   bg_k  exg  min_d_ref  d_healthy  d_abs_max")
    print("=" * 78)
    for m, v in results[:a.top]:
        print(f"{m['f']:5.2f} {m['precision']:5.2f} {m['recall']:5.2f} "
              f"{m['tp']:3d} {m['fn']:3d} {m['fp']:3d}   "
              f"{v['bg_k']:4.1f} {v['exg_thresh']:4d} "
              f"{v['min_d_ref']:+9.3f} {v['d_healthy_foliage']:9.3f} "
              f"{v['d_abs_max']:9.3f}")

    best_m, best_v = results[0]
    print("\n" + "=" * 78)
    print("BEST")
    print("=" * 78)
    for k, v in best_v.items():
        print(f'   "{k}": {v},')
    print(f'   "d_abs_strong": {best_v["d_abs_max"] - 0.02:.3f},')
    print(f"\n   precision {best_m['precision']:.2f}   "
          f"recall {best_m['recall']:.2f}   "
          f"false alarms {best_m['fp']}")

    if best_m["fp"] > 0:
        print("\n   Still firing on negatives. Which ones:")
        p = {**base, **best_v}
        p["d_abs_strong"] = p["d_abs_max"] - 0.02
        for cls in ("healthy", "notleaf"):
            for name, im in sets[cls]:
                hit, r = fires(im, p)
                if hit:
                    print(f"     {cls:8s} {name:32s} d_ref {r.d_ref:+.3f} "
                          f"ratio {r.ratio:.1%}")
        print("   Look at those images. Usually one bad one is dragging the")
        print("   whole search -- a blurred frame, or a mislabelled file.")

    if best_m["fn"] > 0:
        print("\n   Missed diseased images:")
        p = {**base, **best_v}
        p["d_abs_strong"] = p["d_abs_max"] - 0.02
        for name, im in sets["diseased"]:
            hit, r = fires(im, p)
            if not hit:
                print(f"     {name:32s} d_ref {r.d_ref:+.3f} "
                      f"ratio {r.ratio:5.1%}  {r.note[:30]}")
        print("   Run tools/explain.py on one of these to see which gate "
              "rejected it.")

    if a.write:
        p = {**base, **best_v}
        p["d_abs_strong"] = p["d_abs_max"] - 0.02
        out = {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}
        with open(OUT, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n   written -> {OUT}")
        print("   copy it to the Pi:")
        print(f"     scp config/detector_tuned.json "
              f"pi@<pi-ip>:~/acesss/aces/config/")
    else:
        print("\n   (add --write to save this as config/detector_tuned.json)")


if __name__ == "__main__":
    main()
