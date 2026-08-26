"""
STEP 2 TOOL: stage autopsy.

Your bug report has always been the same sentence: "the abnormal parts are
treated as background." This tool proves or disproves that in one screen.

It runs your ORIGINAL two-stage logic and shows every intermediate result
side by side, with pixel counts. You watch a lesion travel through the
pipeline and see exactly which stage kills it.

Usage
-----
    python stage_debug.py <image_or_folder>

Controls
--------
    n / p     next / previous image
    m         cycle which panel is shown full-screen
    ESC       quit

Reading the output
------------------
Panel 2 (PLANT mask) is the one to stare at. If the lesion is BLACK in
panel 2, the lesion never entered the pipeline at all -- it was rejected as
non-plant before the subtraction ever happened. No amount of tuning the
healthy-green range in panel 3 can recover it. That is a stage-1 bug, and
it is by far the most common cause of this exact symptom.
"""

import os
import sys

import cv2
import numpy as np

# ---- your original ranges. Edit these to match your current code exactly. ----
PLANT_LO   = (20, 30, 25)     # "any plant material"
PLANT_HI   = (95, 255, 255)
HEALTHY_LO = (36, 60, 40)     # "clearly green"
HEALTHY_HI = (90, 255, 255)


def panelise(img, label, count=None, total=None):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 24), (w, h), (0, 0, 0), -1)
    txt = label
    if count is not None:
        pct = 100.0 * count / max(total, 1)
        txt += f"   {count:,} px  ({pct:.1f}%)"
    cv2.putText(img, txt, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)
    return img


def analyse(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    total = bgr.shape[0] * bgr.shape[1]

    plant = cv2.inRange(hsv, PLANT_LO, PLANT_HI)
    healthy = cv2.inRange(hsv, HEALTHY_LO, HEALTHY_HI)
    abnormal = cv2.subtract(plant, healthy)

    k = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(abnormal, cv2.MORPH_OPEN, k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k)

    # The diagnostic panel: what does the PLANT mask actually reject?
    rejected = cv2.bitwise_not(plant)
    rej_vis = bgr.copy()
    rej_vis[rejected > 0] = (0, 0, 255)      # red = never entered the pipeline

    # Hue histogram, so you can see where the lesion hues actually sit.
    hist = cv2.calcHist([H], [0], None, [180], [0, 180]).flatten()
    hplot = np.zeros((160, 360, 3), np.uint8)
    if hist.max() > 0:
        hist = hist / hist.max() * 150
    for i in range(180):
        colour = cv2.cvtColor(np.uint8([[[i, 200, 220]]]),
                              cv2.COLOR_HSV2BGR)[0][0].tolist()
        cv2.rectangle(hplot, (i * 2, 160), (i * 2 + 1, 160 - int(hist[i])),
                      colour, -1)
    for x, lab in ((PLANT_LO[0], "P.lo"), (PLANT_HI[0], "P.hi"),
                   (HEALTHY_LO[0], "H.lo"), (HEALTHY_HI[0], "H.hi")):
        cv2.line(hplot, (x * 2, 0), (x * 2, 160), (255, 255, 255), 1)
        cv2.putText(hplot, lab, (x * 2 + 2, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (255, 255, 255), 1)

    panels = [
        panelise(bgr, "1  input"),
        panelise(plant, "2  PLANT mask  <- lesion must be WHITE here",
                 int(plant.sum() / 255), total),
        panelise(healthy, "3  HEALTHY mask", int(healthy.sum() / 255), total),
        panelise(abnormal, "4  PLANT minus HEALTHY",
                 int(abnormal.sum() / 255), total),
        panelise(cleaned, "5  after morphology",
                 int(cleaned.sum() / 255), total),
        panelise(rej_vis, "6  RED = rejected by the PLANT mask"),
    ]
    return panels, hplot


def grid(panels, cell=(300, 225)):
    ps = [cv2.resize(p, cell) for p in panels]
    rows = [np.hstack(ps[i:i + 3]) for i in range(0, len(ps), 3)]
    return np.vstack(rows)


def main(target):
    paths = ([os.path.join(target, f) for f in sorted(os.listdir(target))
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
             if os.path.isdir(target) else [target])
    if not paths:
        print("no images found")
        return

    idx, solo = 0, -1
    while True:
        img = cv2.imread(paths[idx])
        if img is None:
            idx = (idx + 1) % len(paths)
            continue
        img = cv2.resize(img, None, fx=min(1.0, 800 / max(img.shape[:2])),
                         fy=min(1.0, 800 / max(img.shape[:2])))
        panels, hplot = analyse(img)

        view = grid(panels) if solo < 0 else cv2.resize(panels[solo], (900, 675))
        cv2.imshow("stage autopsy", view)
        cv2.imshow("hue histogram", hplot)
        print(f"[{idx+1}/{len(paths)}] {os.path.basename(paths[idx])}")

        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break
        elif k == ord("n"):
            idx = (idx + 1) % len(paths)
        elif k == ord("p"):
            idx = (idx - 1) % len(paths)
        elif k == ord("m"):
            solo = solo + 1 if solo < len(panels) - 1 else -1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
