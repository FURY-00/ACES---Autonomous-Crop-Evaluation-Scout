"""
Calibration. Run this BEFORE nav_controller.py. Nothing else will work
properly until PX_PER_CM_* in config.py are real measured numbers.

Procedure
---------
1. Park the robot in a row, wheels straight, on flat ground.
2. Lay a tape measure across the row, on the ground, at 35 cm ahead of the
   wheel axle (the NEAR band). Run:  python calibrate_row.py near
3. Click two points 20 cm apart on the tape in the frozen frame.
   The script prints PX_PER_CM_NEAR. Paste it into config.py.
4. Repeat at 70 cm ahead with:  python calibrate_row.py far
5. Then run: python calibrate_row.py live  -- drive the robot by hand down
   a row and check that `offset` reads near 0 in the middle and roughly
   +10 when you physically shift the robot 10 cm to the right.

Step 5 is the one people skip and then spend a week wondering why the PID
oscillates. The controller is only as honest as this number.
"""

import sys

import cv2

from config import settings as config
from navigation.row_follower import RowFollower

pts = []


def on_click(ev, x, y, flags, _):
    if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
        pts.append((x, y))


def calibrate(band_name):
    cap = cv2.VideoCapture(config.NAV_CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.NAV_FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.NAV_FRAME_H)
    print("SPACE freezes the frame, then click two points 20 cm apart.")
    frozen = None
    while True:
        if frozen is None:
            ok, f = cap.read()
            if not ok:
                break
            band = config.BAND_NEAR if band_name == "near" else config.BAND_FAR
            h = f.shape[0]
            cv2.line(f, (0, int(band[0] * h)), (f.shape[1], int(band[0] * h)),
                     (0, 255, 255), 1)
            cv2.line(f, (0, int(band[1] * h)), (f.shape[1], int(band[1] * h)),
                     (0, 255, 255), 1)
            show = f
        else:
            show = frozen.copy()
            for p in pts:
                cv2.circle(show, p, 4, (0, 0, 255), -1)
        cv2.imshow("calib", show)
        cv2.setMouseCallback("calib", on_click)
        k = cv2.waitKey(20) & 0xFF
        if k == ord(" ") and frozen is None:
            frozen = f.copy()
        if k == 27:
            break
        if len(pts) == 2:
            d = ((pts[0][0] - pts[1][0]) ** 2 + (pts[0][1] - pts[1][1]) ** 2) ** 0.5
            print(f"\nPX_PER_CM_{band_name.upper()} = {d / 20.0:.3f}")
            break
    cap.release()
    cv2.destroyAllWindows()


def live():
    rf = RowFollower()
    while True:
        e = rf.update()
        print(f"off {e.offset_cm:+6.1f} cm  hdg {e.heading_deg:+6.1f}  "
              f"corridor {e.corridor_cm:5.1f}  L {e.left_room_cm:5.1f}  "
              f"R {e.right_room_cm:5.1f}  conf {e.conf:.2f}", end="\r")
        cv2.imshow("row", rf.debug)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break
    rf.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "live"
    live() if mode == "live" else calibrate(mode)
