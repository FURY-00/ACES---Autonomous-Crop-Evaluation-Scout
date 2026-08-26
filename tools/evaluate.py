"""
STEP 5 TOOL: score it, do not eyeball it.

Once you have images sorted into two folders, this gives you a number instead
of an opinion. "It looks better" is how projects lose a week.

Folder layout
-------------
    testset/
        diseased/     images that DO contain abnormal tissue
        healthy/      images of clean leaves, no disease
                      (this folder is the one people forget, and it is the
                       one that catches a detector that just says "yes")

Usage
-----
    python evaluate.py testset/            # score old vs new
    python evaluate.py testset/ --sweep    # find the best severity threshold

It also flags images that look like photographs of printed paper, because
those have compressed, desaturated colour and will mislead every threshold
you set from them.
"""

import os
import sys

import cv2
import numpy as np

from perception import detector as V

DETECT_RATIO = 0.02        # ratio above which we call an image "diseased"


def old_ratio(bgr):
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    plant = cv2.inRange(hsv, (20, 30, 25), (95, 255, 255))
    healthy = cv2.inRange(hsv, (36, 60, 40), (90, 255, 255))
    ab = cv2.morphologyEx(cv2.subtract(plant, healthy), cv2.MORPH_OPEN,
                          np.ones((5, 5), np.uint8))
    denom = max(int(plant.sum() / 255), 1)
    return int(ab.sum() / 255) / denom


def looks_printed(bgr):
    """
    Photographs of printed paper share three tells: a large near-white
    region (the page), unusually low peak saturation (ink gamut is small),
    and a strong regular high-frequency component (halftone dots).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, Vc = hsv[:, :, 1], hsv[:, :, 2]
    white_frac = float(((S < 40) & (Vc > 195)).mean())
    sat_p95 = float(np.percentile(S, 95))
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hf = float(cv2.Laplacian(g, cv2.CV_64F).var())
    score = (white_frac > 0.18) + (sat_p95 < 165) + (hf > 900)
    return score >= 2, dict(white=round(white_frac, 3),
                            sat_p95=round(sat_p95, 1), hf=round(hf, 0))


def load(folder):
    out = []
    for label, sub in (("diseased", "diseased"), ("healthy", "healthy")):
        d = os.path.join(folder, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append((os.path.join(d, f), label))
    return out


def score(items, thresh=DETECT_RATIO):
    rows = []
    for path, label in items:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, None, fx=min(1.0, 700 / max(img.shape[:2])),
                         fy=min(1.0, 700 / max(img.shape[:2])))
        r = V.detect(img)
        printed, tells = looks_printed(img)
        rows.append(dict(path=path, label=label, new=r.ratio,
                         old=old_ratio(img), trusted=r.trusted,
                         printed=printed, tells=tells, note=r.note))
    return rows


def confusion(rows, key, thresh):
    tp = sum(1 for r in rows if r["label"] == "diseased" and r[key] >= thresh)
    fn = sum(1 for r in rows if r["label"] == "diseased" and r[key] < thresh)
    fp = sum(1 for r in rows if r["label"] == "healthy" and r[key] >= thresh)
    tn = sum(1 for r in rows if r["label"] == "healthy" and r[key] < thresh)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return tp, fp, fn, tn, prec, rec, f1


def main(folder, sweep=False):
    items = load(folder)
    if not items:
        print(f"no images under {folder}/diseased and {folder}/healthy")
        return
    rows = score(items)

    printed = [r for r in rows if r["printed"]]
    if printed:
        print(f"\n!! {len(printed)}/{len(rows)} images look like photos of "
              f"PRINTED PAPER.")
        print("   Ink cannot reproduce leaf saturation, so every threshold you")
        print("   fit to these will be wrong on a real leaf. Examples:")
        for r in printed[:5]:
            print(f"     {os.path.basename(r['path']):40s} {r['tells']}")

    print("\n" + "=" * 62)
    for key in ("old", "new"):
        tp, fp, fn, tn, prec, rec, f1 = confusion(rows, key, DETECT_RATIO)
        print(f"{key.upper():4s}  TP={tp:3d} FP={fp:3d} FN={fn:3d} TN={tn:3d}   "
              f"precision={prec:.2f}  recall={rec:.2f}  F1={f1:.2f}")
    print("=" * 62)

    untrusted = [r for r in rows if not r["trusted"]]
    if untrusted:
        print(f"\n{len(untrusted)} images flagged untrusted:")
        for r in untrusted[:8]:
            print(f"   {os.path.basename(r['path']):40s} {r['note']}")

    if sweep:
        print("\nthreshold sweep (new detector)")
        best = (0, 0)
        for t in np.arange(0.005, 0.20, 0.005):
            _, _, _, _, p_, r_, f1 = confusion(rows, "new", t)
            if f1 > best[1]:
                best = (t, f1)
            print(f"   {t:.3f}   P={p_:.2f}  R={r_:.2f}  F1={f1:.2f}")
        print(f"\n   best threshold = {best[0]:.3f}  (F1 {best[1]:.2f})")

    print("\nworst misses:")
    miss = sorted([r for r in rows if r["label"] == "diseased"],
                  key=lambda r: r["new"])[:5]
    for r in miss:
        print(f"   {os.path.basename(r['path']):40s} "
              f"old={r['old']:.3f}  new={r['new']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "testset",
         "--sweep" in sys.argv)
