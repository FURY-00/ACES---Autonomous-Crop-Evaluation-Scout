"""
Row following from the forward-facing webcam.

Idea in one line: the crop is green, the furrow is not, so the widest
non-green vertical channel in front of the robot IS the path. We measure
where the centre of that channel sits in two horizontal bands -- a near
band and a far band -- and that gives us both how far off centre we are
and which way we are pointing.

Outputs are in centimetres and degrees, never pixels. The controller
should never have to know anything about the camera.
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from config import settings as config


def fuse_sonar(est, tlm):
    """
    Blend the side-sonar cross-track estimate into the vision estimate.

    Vision sees far ahead but is fooled by shadow, dust and bare patches.
    The side sonars see only right here, but at a 5 cm standoff they are
    accurate to a few millimetres and do not care about light at all.
    Weighted average, with the weight rising as vision confidence falls --
    so when the camera gives up, the sonars quietly take over centring.
    """
    from navigation.obstacle_policy import sonar_offset          # local import: no cycle
    s_off, s_conf = sonar_offset(tlm.left_cm, tlm.right_cm)
    if s_off is None:
        return est
    w = config.SIDE_FUSE_WEIGHT * s_conf * (1.0 - 0.6 * est.conf)
    w = float(np.clip(w, 0.0, 0.85))
    est.offset_cm = (1 - w) * est.offset_cm + w * s_off
    est.conf = float(np.clip(max(est.conf, 0.55 * s_conf), 0.0, 1.0))
    est.valid = est.conf >= config.CONF_MIN_DRIVE
    # Sonar measures the walls directly, so it also gives the truest room figures.
    half = config.ROBOT_WIDTH_CM / 2.0
    if config.SONAR_MIN_CM <= tlm.left_cm <= config.SIDE_VALID_MAX_CM:
        est.left_room_cm = max(0.0, tlm.left_cm)
    if config.SONAR_MIN_CM <= tlm.right_cm <= config.SIDE_VALID_MAX_CM:
        est.right_room_cm = max(0.0, tlm.right_cm)
    return est


@dataclass
class RowEstimate:
    offset_cm: float = 0.0     # + = robot is right of row centre
    heading_deg: float = 0.0   # + = robot is pointing right of the row
    left_room_cm: float = 0.0  # free space from robot edge to left crop wall
    right_room_cm: float = 0.0
    corridor_cm: float = 0.0
    conf: float = 0.0
    valid: bool = False


def _exg_mask(bgr):
    """Excess-Green vegetation mask. Robust to the sun going behind a cloud."""
    b, g, r = cv2.split(bgr.astype(np.int16))
    exg = (2 * g - r - b).clip(0, 255).astype(np.uint8)
    if config.EXG_THRESHOLD > 0:
        _, m = cv2.threshold(exg, config.EXG_THRESHOLD, 255, cv2.THRESH_BINARY)
    else:
        _, m = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((config.MORPH_KERNEL, config.MORPH_KERNEL), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


def _corridor(mask, band, px_per_cm):
    """
    Find the free channel inside one horizontal band.

    Returns (centre_px, width_cm, left_edge_px, right_edge_px) or None.
    """
    h, w = mask.shape
    y0, y1 = int(band[0] * h), int(band[1] * h)
    strip = mask[y0:y1, :]

    # Fraction of the band height that is vegetation, per column.
    veg = strip.mean(axis=0) / 255.0
    veg = np.convolve(veg, np.ones(9) / 9.0, mode="same")   # smooth out leaf gaps
    free = veg < 0.25                                        # soil-dominant columns

    # Longest run of free columns that contains, or is nearest to, image centre.
    best, run_start = None, None
    cx = w // 2
    for x in range(w + 1):
        is_free = x < w and free[x]
        if is_free and run_start is None:
            run_start = x
        elif not is_free and run_start is not None:
            run = (run_start, x - 1)
            length = run[1] - run[0]
            # score = length, penalised by distance of the run centre from image centre
            score = length - 0.6 * abs((run[0] + run[1]) / 2 - cx)
            if best is None or score > best[0]:
                best = (score, run)
            run_start = None
    if best is None:
        return None

    l, r = best[1]
    width_cm = (r - l) / px_per_cm
    return (l + r) / 2.0, width_cm, l, r


class RowFollower:
    def __init__(self):
        self.cap = cv2.VideoCapture(config.NAV_CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.NAV_FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.NAV_FRAME_H)
        self.cap.set(cv2.CAP_PROP_FPS, config.NAV_FPS)
        self._off = 0.0
        self._head = 0.0
        self._last_good = 0.0
        self.debug = None
        self.frame = None

    def _blend(self, new, old, a=0.45):
        return a * new + (1 - a) * old

    def update(self) -> RowEstimate:
        ok, frame = self.cap.read()
        if not ok:
            return RowEstimate(conf=0.0, valid=False)
        self.frame = frame
        mask = _exg_mask(frame)
        h, w = mask.shape
        est = RowEstimate()

        near = _corridor(mask, config.BAND_NEAR, config.PX_PER_CM_NEAR)
        far = _corridor(mask, config.BAND_FAR, config.PX_PER_CM_FAR)

        if near is None:
            est.conf = 0.0
            self._draw(frame, mask, est, near, far)
            return est

        n_cx, n_w_cm, n_l, n_r = near

        # --- cross-track error -------------------------------------------
        off_px = n_cx - (w / 2.0)
        off_cm = off_px / config.PX_PER_CM_NEAR + config.NAV_CAM_X_OFFSET_CM
        # sign: corridor centre appearing to the LEFT of image centre means the
        # robot has drifted RIGHT, so offset is positive.
        off_cm = -off_cm

        # --- heading error ------------------------------------------------
        if far is not None:
            f_cx = far[0]
            f_off_cm = -((f_cx - w / 2.0) / config.PX_PER_CM_FAR)
            dy = config.LOOKAHEAD_FAR_CM - config.LOOKAHEAD_NEAR_CM
            head = np.degrees(np.arctan2(f_off_cm - off_cm, dy))
        else:
            head = self._head

        # --- how much room before we crush something ----------------------
        left_edge_cm = (n_cx - n_l) / config.PX_PER_CM_NEAR
        right_edge_cm = (n_r - n_cx) / config.PX_PER_CM_NEAR
        half = config.ROBOT_WIDTH_CM / 2.0
        est.left_room_cm = max(0.0, left_edge_cm + off_cm - half)
        est.right_room_cm = max(0.0, right_edge_cm - off_cm - half)

        # --- confidence ----------------------------------------------------
        conf = 1.0
        if not (config.MIN_CORRIDOR_CM <= n_w_cm <= config.MAX_CORRIDOR_CM):
            conf *= 0.25                                   # corridor width implausible
        if far is None:
            conf *= 0.7
        veg_frac = mask.mean() / 255.0
        if veg_frac < 0.05:
            conf *= 0.3                                    # nothing green = no row
        jump = abs(off_cm - self._off)
        if jump > 12.0:
            conf *= 0.4                                    # teleporting corridor = noise
        conf = float(np.clip(conf, 0.0, 1.0))

        # --- smoothing ------------------------------------------------------
        a = 0.5 * conf + 0.15
        self._off = self._blend(off_cm, self._off, a)
        self._head = self._blend(head, self._head, a)

        est.offset_cm = self._off
        est.heading_deg = self._head
        est.corridor_cm = n_w_cm
        est.conf = conf
        est.valid = conf >= config.CONF_MIN_DRIVE
        if est.valid:
            self._last_good = time.time()

        self._draw(frame, mask, est, near, far)
        return est

    def blind_for(self) -> float:
        return time.time() - self._last_good if self._last_good else 0.0

    def side_vegetation(self) -> float:
        """Vegetation fraction in the outer thirds -- used for end-of-row voting."""
        if self.frame is None:
            return 1.0
        m = _exg_mask(self.frame)
        h, w = m.shape
        band = m[int(0.55 * h):int(0.9 * h), :]
        third = w // 3
        left, right = band[:, :third], band[:, -third:]
        return float(max(left.mean(), right.mean()) / 255.0)

    def _draw(self, frame, mask, est, near, far):
        dbg = frame.copy()
        h, w = mask.shape
        green = np.zeros_like(dbg)
        green[:, :, 1] = mask
        dbg = cv2.addWeighted(dbg, 1.0, green, 0.35, 0)
        for band, res, col in ((config.BAND_NEAR, near, (0, 255, 255)),
                               (config.BAND_FAR, far, (255, 180, 0))):
            y = int((band[0] + band[1]) / 2 * h)
            cv2.line(dbg, (0, y), (w, y), (60, 60, 60), 1)
            if res:
                cx, _, l, r = res
                cv2.line(dbg, (int(l), y), (int(r), y), col, 2)
                cv2.circle(dbg, (int(cx), y), 4, col, -1)
        cv2.line(dbg, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        cv2.putText(dbg, f"off {est.offset_cm:+.1f}cm  hdg {est.heading_deg:+.1f}deg"
                         f"  conf {est.conf:.2f}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        self.debug = dbg

    def release(self):
        self.cap.release()
