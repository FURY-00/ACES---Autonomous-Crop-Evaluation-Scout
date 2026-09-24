#!/usr/bin/env python3
"""
Work out why the row is not being detected -- by measuring, not guessing.

Run it pointed down your test row:

    python3 tools/mask_probe.py

or on a saved image:

    python3 tools/mask_probe.py --image some_frame.jpg

It answers three questions in order, and the order matters. There is no point
tuning a threshold if the vegetation is not where the bands are looking, and
no point moving the bands if the mask is not finding vegetation anywhere.

  1. IS THERE ANY GREEN AT ALL?  Excess-Green statistics for the frame. If
     the 99th percentile is in single digits, the camera is not delivering a
     usable colour image and no threshold will save it -- that is an exposure,
     white-balance or dirty-lens problem.

  2. WHERE IS THE GREEN, VERTICALLY?  A row-by-row profile. Crop walls have
     to appear in the NEAR band or the follower cannot see them. Short plants
     and a high, level camera put all the leaves in the top half of the frame,
     where the near band never looks. This is the failure that looks like "the
     mask is broken" but is really "the bands are pointed at the floor".

  3. WHAT THRESHOLD AND WHAT BANDS WOULD WORK?  A sweep. For each
     combination it reports the left and right column peaks and whether walls
     would be found, so you can read the answer off a table instead of
     restarting the mission twenty times.

Everything is written to data/probe/ as images you can look at or send on.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navigation.row_vision import (CFG, band_edges,  # noqa: E402
                                   flatten_illumination)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "probe")


def exg_of(bgr, flatten=True):
    src = flatten_illumination(bgr, CFG) if flatten else bgr
    b, g, r = cv2.split(src.astype(np.int16))
    return (2 * g - r - b).clip(0, 255).astype(np.uint8)


def mask_at(exg, thr):
    _, m = cv2.threshold(exg, thr, 255, cv2.THRESH_BINARY)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="probe a saved image instead of the webcam")
    ap.add_argument("--webcam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--proc-width", type=int, default=480)
    ap.add_argument("--warm", type=int, default=15,
                    help="frames to throw away so auto-exposure settles")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    if a.image:
        frame = cv2.imread(a.image)
        if frame is None:
            sys.exit(f"could not read {a.image}")
    else:
        cap = cv2.VideoCapture(a.webcam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        for _ in range(a.warm):          # let auto-exposure settle
            cap.read()
        ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit("webcam gave no frame")

    frame = cv2.resize(frame, (a.proc_width,
                               int(frame.shape[0] * a.proc_width
                                   / frame.shape[1])))
    h, w = frame.shape[:2]
    cv2.imwrite(os.path.join(OUT, "00_raw.jpg"), frame)

    # ---------------------------------------------------- 1. any green?
    print("=" * 66)
    print("1. IS THERE ANY GREEN?")
    print("=" * 66)
    b, g, r = cv2.split(frame.astype(np.int16))
    print(f"   frame            {w} x {h}")
    print(f"   mean B,G,R       {b.mean():.0f}, {g.mean():.0f}, {r.mean():.0f}")
    sat = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    print(f"   mean saturation  {sat.mean():.0f} of 255", end="")
    print("   <- WASHED OUT, fix exposure first" if sat.mean() < 40 else "")

    raw = exg_of(frame, flatten=False)
    flat = exg_of(frame, flatten=True)
    for name, e in (("raw     ", raw), ("flattened", flat)):
        p = np.percentile(e, [50, 90, 99, 99.9])
        print(f"   ExG {name}   median {p[0]:5.1f}   p90 {p[1]:5.1f}   "
              f"p99 {p[2]:5.1f}   p99.9 {p[3]:5.1f}")
    cv2.imwrite(os.path.join(OUT, "01_exg.jpg"), flat)

    if np.percentile(flat, 99) < 12:
        print()
        print("   VERDICT: there is essentially no excess green anywhere in")
        print("   this frame. No threshold will fix that. Causes, in order of")
        print("   likelihood: overexposure washing the colour out; a")
        print("   protective film or smudge on the lens; auto white balance")
        print("   pulling the greens grey. Fix the picture, then re-probe.")

    # ------------------------------------------- 2. where is it, vertically?
    print()
    print("=" * 66)
    print("2. WHERE IS THE GREEN, VERTICALLY?")
    print("=" * 66)
    # Use the same rule the mission uses -- Otsu with a floor -- rather than
    # a fixed percentile. A percentile above the brightest green marks
    # nothing at all and makes the profile look empty when it is not.
    otsu, _ = cv2.threshold(flat, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_ref = max(10, int(otsu))
    m = mask_at(flat, thr_ref)
    if m.mean() < 1.0:                     # still nothing: back right off
        thr_ref = 10
        m = mask_at(flat, thr_ref)
    rows_frac = m.mean(axis=1) / 255.0
    print(f"   (using ExG threshold {thr_ref}, from Otsu)")
    print()
    print("   band of frame      green cover   bar")
    for i in range(10):
        y0, y1 = int(i * h / 10), int((i + 1) * h / 10)
        v = float(rows_frac[y0:y1].mean())
        bar = "#" * int(v * 50)
        tag = ""
        if 0.70 <= (i + 0.5) / 10 <= 0.92:
            tag = "  <- NEAR band (decides walls)"
        elif 0.45 <= (i + 0.5) / 10 <= 0.65:
            tag = "  <- far band (heading only)"
        print(f"   {i/10:.1f}-{(i+1)/10:.1f}          {v:5.3f}      "
              f"{bar}{tag}")

    best = int(np.argmax([rows_frac[int(i * h / 10):int((i + 1) * h / 10)].mean()
                          for i in range(10)]))
    print()
    print(f"   Most vegetation sits at {best/10:.1f}-{(best+1)/10:.1f} of "
          "frame height.")
    if best < 6 and rows_frac.mean() > 0.001:
        near_lo = max(0.05, min(0.70, best / 10 - 0.05))
        near_hi = min(0.98, near_lo + 0.25)
        far_lo = max(0.0, near_lo - 0.22)
        far_hi = max(far_lo + 0.10, near_lo)
        print("   That is ABOVE the default near band (0.70-0.92), which is")
        print("   why walls are not found. Try:")
        print(f"       --band-near {near_lo:.2f},{near_hi:.2f} "
              f"--band-far {far_lo:.2f},{far_hi:.2f}")

    # --------------------------------------------------- 3. sweep
    print()
    print("=" * 66)
    print("3. WHAT WOULD ACTUALLY FIND WALLS?")
    print("=" * 66)
    print("   thr   band          L peak  R peak   result")
    bands = [(0.70, 0.92), (0.55, 0.80), (0.45, 0.70), (0.35, 0.60),
             (0.25, 0.50)]
    hits = []
    for thr in (40, 30, 25, 20, 15, 10):
        mm = mask_at(flat, thr)
        for bd in bands:
            stats = {}
            le, re = band_edges(mm, bd, CFG, stats)
            lp, rp = stats.get("left_peak", 0), stats.get("right_peak", 0)
            if le is not None and re is not None \
                    and (re - le) >= CFG["min_corridor_px"]:
                res = f"BOTH  L={le:.0f} R={re:.0f}  gap {re-le:.0f}px"
                # Rank by the WEAKER side, not the gap. A wide gap can mean
                # one edge was found on noise; a strong weaker-side peak
                # means both walls are genuinely there. The wall that is
                # about to disappear is the one that decides whether the run
                # survives.
                hits.append((min(lp, rp), thr, bd, re - le))
            elif le is not None:
                res = f"left only ({le:.0f})"
            elif re is not None:
                res = f"right only ({re:.0f})"
            else:
                res = "none"
            print(f"   {thr:3d}   {bd[0]:.2f}-{bd[1]:.2f}    {lp:5.2f}   "
                  f"{rp:5.2f}   {res}")
        mm2 = cv2.addWeighted(frame, 1.0,
                              cv2.merge([np.zeros_like(mm), mm,
                                         np.zeros_like(mm)]), 0.4, 0)
        cv2.imwrite(os.path.join(OUT, f"02_mask_thr{thr:02d}.jpg"), mm2)

    print()
    if hits:
        weak, thr, bd, gap = max(hits, key=lambda x: x[0])
        print("   WORKS. Best combination -- ranked by the WEAKER wall, "
              "since that")
        print("   is the one that vanishes first when the bot drifts:")
        print(f"       --exg {thr} --band-near {bd[0]:.2f},{bd[1]:.2f} "
              f"--band-far {max(0.0, bd[0]-0.20):.2f},{bd[0]:.2f}")
        print(f"   weaker-side peak {weak:.2f}, corridor {gap:.0f}px")
        if weak < 0.30:
            print()
            print(f"   CAUTION: {weak:.2f} is thin. A real crop wall reads "
                  "0.6-0.9. This")
            print("   will track in still air and lose a wall as soon as the "
                  "bot drifts")
            print("   toward the other side. Add foliage to the weak side "
                  "before")
            print("   trusting a long run.")
    else:
        print("   NOTHING found two walls at any threshold or band position.")
        print("   The problem is the scene, not the settings. The near band")
        print("   needs CONTINUOUS green on BOTH sides -- separate pots with")
        print("   floor visible between them do not form a wall, however")
        print("   close together you push them.")

    print()
    print(f"   images written to {OUT}")


if __name__ == "__main__":
    main()
