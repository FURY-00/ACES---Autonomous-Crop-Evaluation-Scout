"""
STEP 0 TOOL: capture the dataset properly.

Read this before you take a single photo, because one setting decides
whether the whole session is usable.

LOCK THE WHITE BALANCE.
-----------------------
Auto white balance re-decides what "white" means on every frame. Point the
camera at a mostly-green leaf and AWB pulls the whole image magenta to
compensate. Point it at soil and it pulls green. Your hue values then shift
by 10-20 degrees between shots of the SAME leaf, and hue thresholds fitted
to one shot fail on the next.

This is the single most common reason a colour-threshold detector works in
the lab and fails in the field. This script locks AWB, exposure and gain
before the first frame and reports if they drift.

Usage
-----
    python capture_dataset.py                 # writes into ./testset/
    python capture_dataset.py --out mydata

Controls
--------
    d          save current frame to  testset/diseased/
    h          save current frame to  testset/healthy/
    l          re-lock white balance and exposure on the CURRENT scene
    t          cycle the lighting tag: sun / shade / overcast / indoor
    ESC        quit and print a session summary

Filenames encode the tag, so check_dataset.py can tell you whether you have
covered enough lighting conditions.
"""

import argparse
import os
import time

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    HAVE_PICAM = True
except ImportError:
    HAVE_PICAM = False

TAGS = ["sun", "shade", "overcast", "indoor"]


class Camera:
    """Wraps Picamera2 or a USB webcam behind one interface, with locking."""

    def __init__(self):
        self.locked = False
        if HAVE_PICAM:
            self.cam = Picamera2()
            cfg = self.cam.create_preview_configuration(
                main={"size": (1280, 720), "format": "RGB888"})
            self.cam.configure(cfg)
            self.cam.start()
            time.sleep(2.0)          # let AE/AWB settle before we freeze them
            self.set_focus("continuous")
        else:
            self.cam = cv2.VideoCapture(0)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            time.sleep(1.5)

    def set_focus(self, spec="continuous"):
        """continuous | auto | distance in cm. See tools/live_detect.py."""
        if not HAVE_PICAM:
            return
        try:
            from libcamera import controls
            if spec == "continuous":
                self.cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            elif spec == "auto":
                self.cam.set_controls({"AfMode": controls.AfModeEnum.Auto})
                self.cam.autofocus_cycle()
            else:
                cm = float(spec)
                self.cam.set_controls({"AfMode": controls.AfModeEnum.Manual,
                                       "LensPosition": 100.0 / max(cm, 1.0)})
            print(f"[focus] {spec}")
        except Exception as e:
            print(f"[focus] {e}")

    def lock(self):
        """Freeze AWB, exposure and gain at their current auto-chosen values."""
        if HAVE_PICAM:
            md = self.cam.capture_metadata()
            gains = md.get("ColourGains", (1.8, 1.8))
            self.cam.set_controls({
                "AwbEnable": False,
                "ColourGains": gains,
                "AeEnable": False,
                "ExposureTime": md.get("ExposureTime", 8000),
                "AnalogueGain": md.get("AnalogueGain", 1.0),
            })
            print(f"  locked: gains={tuple(round(g,2) for g in gains)}  "
                  f"exp={md.get('ExposureTime')}us  gain={md.get('AnalogueGain'):.2f}")
        else:
            self.cam.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            print("  locked: AWB and auto-exposure disabled on the webcam")
        self.locked = True

    def read(self):
        if HAVE_PICAM:
            return self.cam.capture_array()   # RGB888 array is already BGR
        ok, f = self.cam.read()
        return f if ok else None

    def close(self):
        if HAVE_PICAM:
            self.cam.stop()
        else:
            self.cam.release()


def wb_signature(bgr):
    """Mean channel ratios. If these drift between shots, the lock failed."""
    b, g, r = [float(c.mean()) for c in cv2.split(bgr)]
    return r / max(g, 1e-6), b / max(g, 1e-6)


def sharpness(bgr):
    return float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="testset")
    args = ap.parse_args()

    for sub in ("diseased", "healthy"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    cam = Camera()
    print("\nSettling... point the camera at a TYPICAL scene, then press 'l' "
          "to lock.\n")
    tag_i = 0
    counts = {"diseased": 0, "healthy": 0}
    ref_wb = None

    while True:
        frame = cam.read()
        if frame is None:
            continue

        sh = sharpness(frame)
        rg, bg = wb_signature(frame)
        if ref_wb is None and cam.locked:
            ref_wb = (rg, bg)
        drift = (abs(rg - ref_wb[0]) + abs(bg - ref_wb[1])) if ref_wb else 0.0

        disp = cv2.resize(frame, None, fx=0.7, fy=0.7)
        h, w = disp.shape[:2]
        cv2.rectangle(disp, (0, 0), (w, 52), (0, 0, 0), -1)
        lock_txt = "LOCKED" if cam.locked else "NOT LOCKED - press 'l'"
        lock_col = (0, 255, 0) if cam.locked else (0, 0, 255)
        cv2.putText(disp, f"{lock_txt}   tag={TAGS[tag_i]}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, lock_col, 1)
        sh_col = (0, 255, 0) if sh > 120 else (0, 165, 255)
        cv2.putText(disp,
                    f"sharp={sh:6.0f}   wb drift={drift:.3f}   "
                    f"diseased={counts['diseased']}  healthy={counts['healthy']}",
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sh_col, 1)
        if drift > 0.08:
            cv2.putText(disp, "WHITE BALANCE DRIFTING - re-lock", (8, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("capture", disp)

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord("l"):
            cam.lock()
            ref_wb = wb_signature(cam.read())
        elif k == ord("t"):
            tag_i = (tag_i + 1) % len(TAGS)
        elif k in (ord("d"), ord("h")):
            if not cam.locked:
                print("  ! lock the white balance first (press 'l')")
                continue
            if sh < 90:
                print(f"  ! too blurry (sharpness {sh:.0f}) - not saved")
                continue
            cls = "diseased" if k == ord("d") else "healthy"
            counts[cls] += 1
            name = f"{TAGS[tag_i]}_{counts[cls]:03d}_{int(time.time())}.jpg"
            path = os.path.join(args.out, cls, name)
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  saved {cls:9s} {name}   sharp={sh:.0f}")

    cam.close()
    cv2.destroyAllWindows()
    print(f"\nsession: {counts['diseased']} diseased, {counts['healthy']} healthy")
    print(f"next:  python check_dataset.py {args.out}")


if __name__ == "__main__":
    main()
