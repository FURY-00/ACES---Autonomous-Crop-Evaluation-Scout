"""
Disease camera: 45-degree side-mounted Pi Camera v3.

Two jobs:
  1. Watch a low-res preview stream for abnormal (yellow/brown) tissue.
  2. When something is found, take a HIGH-res still and prove it is sharp.

The reverse-and-retry trick
---------------------------
The camera looks 45 deg sideways, so by the time detection fires, is
handled, and the wheels actually stop, the leaf is already behind the
lens. We therefore stamp the odometer at the moment of detection and
reverse to (odo_at_detect + lead + extra) rather than reversing a blind
fixed amount. Odometry is very accurate over 30 cm, so this works well.
"""

import json
import os
import time

import cv2
import numpy as np

from config import settings as config

try:
    from picamera2 import Picamera2
    HAVE_PICAM = True
except ImportError:
    HAVE_PICAM = False


def abnormal_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    plant = cv2.inRange(hsv, (20, 30, 25), (95, 255, 255))       # all leaf tissue
    healthy = cv2.inRange(hsv, (36, 60, 40), (90, 255, 255))     # clearly green
    abnormal = cv2.subtract(plant, healthy)
    abnormal = cv2.bitwise_and(
        abnormal, cv2.inRange(hsv, (10,60,50),(35,255,255)))
    k = np.ones((5, 5), np.uint8)
    abnormal = cv2.morphologyEx(abnormal, cv2.MORPH_OPEN, k)
    abnormal = cv2.morphologyEx(abnormal, cv2.MORPH_CLOSE, k)
    return abnormal


def largest_blob(mask):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    a = stats[i, cv2.CC_STAT_AREA]
    if a < config.ABNORMAL_AREA_MIN_PX:
        return None
    x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                  stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
    return (x, y, w, h, int(a))


def sharpness(bgr, roi=None):
    """Variance of Laplacian. Measured on the ROI, not the whole frame --
    a sharp background with a motion-blurred leaf must score LOW."""
    img = bgr
    if roi:
        x, y, w, h = roi[:4]
        pad = 20
        img = bgr[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
    if img.size == 0:
        return 0.0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


class DiseaseCamera:
    def __init__(self):
        self.preview = None
        self.overlay = None
        if HAVE_PICAM:
            self.cam = Picamera2()
            cfg = self.cam.create_video_configuration(
                main={"size": (640, 360), "format": "RGB888"},
                lores=None,
                controls={"FrameRate": 20},
            )
            self.cam.configure(cfg)
            self.cam.start()
            time.sleep(1.5)
            try:
                from libcamera import controls
                # The 45-deg mount looks at a roughly fixed distance, so a
                # FIXED focus is better than autofocus here: the lens never
                # hunts mid-capture and every frame is comparable.
                self.cam.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": 100.0 / max(config.DISEASE_CAM_LEAD_CM, 1.0)})
            except Exception as e:
                print(f"[disease-cam] focus: {e}")
        else:
            self.cam = cv2.VideoCapture(1)

    def grab(self, hires=False):
        if HAVE_PICAM:
            if hires:
                req = self.cam.switch_mode_and_capture_array(
                    self.cam.create_still_configuration(
                        main={"size": (config.CAPTURE_W, config.CAPTURE_H)}))
                return req                        # RGB888 array is already BGR
            return self.cam.capture_array()   # RGB888 array is already BGR
        ok, f = self.cam.read()
        return f if ok else None

    def scan(self):
        """Cheap per-frame check. Returns (found, roi, frame)."""
        f = self.grab(hires=False)
        if f is None:
            return False, None, None
        m = abnormal_mask(f)
        roi = largest_blob(m)
        ov = f.copy()
        ov[m > 0] = (0.4 * ov[m > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        if roi:
            x, y, w, h, a = roi
            cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(ov, f"{a}px", (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        self.preview, self.overlay = f, ov
        return roi is not None, roi, f

    def shoot(self, meta: dict):
        """Full-res still. Returns (path, sharp_ok, score)."""
        best, best_score, best_roi = None, -1.0, None
        for _ in range(config.CAPTURE_RETRIES):
            f = self.grab(hires=True)
            if f is None:
                continue
            roi = largest_blob(abnormal_mask(f))
            s = sharpness(f, roi)
            if s > best_score:
                best, best_score, best_roi = f, s, roi
            if s >= config.SHARPNESS_MIN:
                break
        if best is None:
            return None, False, 0.0

        ts = time.strftime("%Y%m%d_%H%M%S")
        day = time.strftime("%Y%m%d")
        outdir = os.path.join(config.SD_ROOT, day,
                              "images" if best_score >= config.SHARPNESS_MIN
                              else "uncertain")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"leaf_{ts}.jpg")
        cv2.imwrite(path, best, [cv2.IMWRITE_JPEG_QUALITY, 92])

        meta = dict(meta)
        meta.update({"sharpness": round(best_score, 1),
                     "sharp_ok": best_score >= config.SHARPNESS_MIN,
                     "roi": best_roi, "file": os.path.basename(path)})
        with open(path.replace(".jpg", ".json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        return path, best_score >= config.SHARPNESS_MIN, best_score

    def close(self):
        if HAVE_PICAM:
            self.cam.stop()
        else:
            self.cam.release()
