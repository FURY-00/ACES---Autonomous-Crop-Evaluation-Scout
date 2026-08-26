"""
Explain one image, gate by gate.

The tuner shows you WHAT happened. This shows you WHY, and tells you the
numbers to set. Run it on your laptop with no Pi and no camera.

    python3 tools/explain.py leaf.jpg
    python3 tools/explain.py leaf.jpg --save out.png    # writes a visual too

What you get
------------
  * the distribution of d = (G-R)/(R+G+B) inside the leaf, as a text histogram
  * exactly which gate accepted or rejected the frame, and by how much
  * a recommended setting for every parameter that mattered

Reading the d histogram is the single most useful skill for this detector.
Everything the detector decides is a threshold on that one number.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings as S            # noqa: E402
from perception import detector as D        # noqa: E402


def histogram(vals, lo=-0.25, hi=0.45, bins=28, width=52, marks=None):
    counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    top = max(counts.max(), 1)
    marks = marks or {}
    out = []
    for i, c in enumerate(counts):
        a, b = edges[i], edges[i + 1]
        bar = "#" * int(round(c / top * width))
        tag = ""
        for name, v in marks.items():
            if a <= v < b:
                tag += f"  <-- {name}"
        out.append(f"  {a:+.3f} |{bar:<{width}}| {c:7d}{tag}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--save", help="write an annotated PNG here")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        print(f"could not read {a.image}")
        return
    img = cv2.resize(img, (800, 450))
    p = dict(S.DETECTOR)
    r = D.detect(img, p)

    print("=" * 70)
    print(f"  {os.path.basename(a.image)}")
    print("=" * 70)

    # ---- stage 1: leaf --------------------------------------------------
    frac = r.leaf_px / (800 * 450)
    print(f"\n1  LEAF SEGMENTATION")
    print(f"   leaf found      {r.leaf_px:,} px  ({frac:.1%} of frame)")
    if frac < 0.05:
        print("   ! tiny. The leaf should fill a good part of the frame.")
        print("     Hold it closer, or lower bg_k so the mask grows more.")
    elif frac > 0.85:
        print("   ! almost the whole frame. The mask has probably leaked into")
        print("     the background. Raise bg_k.")
    else:
        print("   looks reasonable")

    if r.leaf_px < 500:
        print("\n   STOPPED: no leaf. Nothing else can be judged.")
        return

    # ---- stage 2: exclusions -------------------------------------------
    unk = float(r.unknown_mask.sum()) / 255 / r.leaf_px
    print(f"\n2  EXCLUSIONS (glare and shadow)")
    print(f"   unjudged        {unk:.1%} of the leaf")
    if unk > 0.25:
        print("   ! too much. Shoot in open shade, or raise k_specular so less")
        print("     is removed as glare.")
    else:
        print("   fine")

    # ---- stage 3: the d distribution ------------------------------------
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    d = D.green_red_index(blur)
    jm = (r.leaf_mask > 0) & (r.unknown_mask == 0)
    dv = d[jm]

    print(f"\n3  GREENNESS  d = (G-R)/(R+G+B)   inside the leaf")
    print(f"   min {dv.min():+.3f}   p05 {np.percentile(dv,5):+.3f}   "
          f"p25 {np.percentile(dv,25):+.3f}   median {np.median(dv):+.3f}")
    print(f"   p75 {np.percentile(dv,75):+.3f}   p95 {np.percentile(dv,95):+.3f}"
          f"   max {dv.max():+.3f}")
    print()
    print(histogram(dv, marks={
        "min_d_ref": p.get("min_d_ref", 0.02),
        "d_healthy": p.get("d_healthy_foliage", 0.12),
        "d_abs_max": p.get("d_abs_max", 0.075),
        "d_ref": r.d_ref,
    }))
    print("\n   Two humps = healthy tissue and lesion, cleanly separable.")
    print("   One hump  = the leaf is uniformly one thing (all healthy, or")
    print("               all diseased). The whole-leaf path handles the latter.")

    # ---- stage 4: which gate decided ------------------------------------
    print(f"\n4  THE DECISION")
    print(f"   d_ref (p{p.get('ref_percentile',75)} of the above) = {r.d_ref:+.4f}")
    min_dr = p.get("min_d_ref", 0.02)
    d_h = p.get("d_healthy_foliage", 0.12)

    if r.d_ref < min_dr:
        print(f"   REJECTED as not-a-plant: {r.d_ref:+.4f} < min_d_ref {min_dr:+.4f}")
        print(f"   -> if this IS a leaf, set min_d_ref to about "
              f"{r.d_ref - 0.03:+.3f}")
    elif r.d_ref < d_h:
        print(f"   WHOLE-LEAF path: {min_dr:+.3f} <= {r.d_ref:+.4f} < "
              f"d_healthy {d_h:+.3f}")
        print(f"   the entire leaf is called diseased -> ratio {r.ratio:.1%}")
        print(f"   -> if this leaf has healthy tissue you want spared, set")
        print(f"      d_healthy_foliage below {r.d_ref:+.3f}")
    else:
        print(f"   NORMAL path: healthy tissue present ({r.d_ref:+.4f} >= "
              f"{d_h:+.3f})")
        print(f"   applied threshold {r.d_thresh:+.4f}   "
              f"spread {r.debug.get('spread','-')}")
        below = float((dv < r.d_thresh).mean())
        print(f"   {below:.1%} of the leaf is below that threshold")
        if r.ratio == 0 and below > 0.02:
            print("   ! pixels qualify but no blob survived. Lower min_blob_px,")
            print("     or the lesions are scattered specks.")

    print(f"\n   RESULT   {r.severity}   {r.ratio:.2%} of leaf   "
          f"{len(r.blobs)} blobs   trusted={r.trusted}")
    if r.note:
        print(f"   note     {r.note}")

    # ---- stage 5: recommendations ---------------------------------------
    print(f"\n5  SUGGESTED VALUES FOR THIS IMAGE")
    print(f'   "min_d_ref": {max(-0.15, r.d_ref - 0.03):.3f},')
    if r.d_ref >= d_h:
        p25 = np.percentile(dv, 25)
        print(f'   "d_abs_max": {p25 + 0.01:.3f},        '
              f'# just above the lesion cluster')
        print(f'   "d_abs_strong": {p25 - 0.01:.3f},')
    else:
        print(f'   "d_healthy_foliage": {r.d_ref + 0.03:.3f},'
              f'   # so this leaf takes the whole-leaf path')
    print("\n   Do NOT paste these blindly. They fit THIS image. Use")
    print("   tools/offline_tune.py to find values that fit a whole folder.")

    if a.save:
        cv2.imwrite(a.save, D.overlay(img, r))
        print(f"\n   annotated image -> {a.save}")


if __name__ == "__main__":
    main()
