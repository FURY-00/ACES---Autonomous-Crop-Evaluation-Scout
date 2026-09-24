"""
Row following from the forward webcam.

The idea
--------
Crop grows on both sides. The furrow between them does not. So:

    1. mark every vegetation pixel
    2. in a horizontal band, walk in from the left until the crop wall ends
       -> that is the LEFT INNER EDGE
    3. walk in from the right the same way -> RIGHT INNER EDGE
    4. the centreline is halfway between them
    5. steer so the centreline sits at the middle of the image

Two bands, not one
------------------
A single band tells you WHERE you are but not WHICH WAY you are pointing. A
bot that is centred but angled will drive straight out of the row while
reporting zero error. So we measure a NEAR band (cross-track error) and a FAR
band (heading error) and use both. This is the single biggest difference
between a bot that weaves and one that tracks.

Losing a wall is information, not failure
-----------------------------------------
Get the direction right, because it is the opposite of what feels obvious.

Drift LEFT, toward the left crop wall, and that wall gets CLOSER -- so it
grows in the frame and its inner edge slides toward the middle. Meanwhile the
right wall recedes, shrinks, and slides off the right of the frame. So:

    only the LEFT wall visible   ->  you drifted LEFT   ->  steer RIGHT
    only the RIGHT wall visible  ->  you drifted RIGHT  ->  steer LEFT

Getting this backwards drives the bot straight into the crop, confidently.

The naive response is then to turn blindly. The better one: while both walls
were visible we measured the row's half-width in pixels. So when one wall is
lost we can still compute a real setpoint -- "centre is half a row-width in
from the wall I can still see" -- and steer to an actual target rather than
guessing. That remembered half-width is what makes recovery smooth instead of
a lurch.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------- config
CFG = {
    # vegetation
    "exg_thresh": 0,          # 0 = Otsu (adapts to light automatically)
    "veg_min_col": 0.12,      # a column is "crop wall" if this much is green.
                          # LOWER for sparse or low walls: leaves lying
                          # near the ground fill little of the band.
    "smooth_cols": 9,         # column-profile smoothing window

    # bands, as fractions of frame height
    "band_near": (0.70, 0.92),
    "band_far":  (0.45, 0.65),

    # geometry
    "min_corridor_px": 30,    # narrower than this is not a real corridor
    "min_wall_px": 12,        # a run of crop shorter than this is noise,
                              # not a wall

    # control
    # Gains act on error expressed as a FRACTION OF FRAME WIDTH, not pixels.
    # In pixels the same physical offset reads 7 units at 240px wide and 41 at
    # 1280 -- so a gain tuned at one resolution would be 5x too strong at
    # another. Normalising means your tuning survives a resolution change.
    # An offset of 0.25 (a quarter of the frame) with kp_offset 260 gives a
    # steer of 65.
    # kp_heading was originally larger than kp_offset, which meant a noisy
    # heading estimate could cancel a real cross-track error -- the bot sat
    # 15% off centre reporting a steer of +2. Offset is the reliable signal;
    # heading is a refinement. Weight them accordingly.
    "kp_offset": 260.0,       # error as fraction of width -> steer units
    "kp_heading": 120.0,
    "steer_max": 90,
    "smooth": 0.45,           # temporal blend, 0..1 (higher = more responsive)

    # recovery
    "halfwidth_memory": 0.9,  # how strongly to trust the remembered row width
    "recover_steer": 40,      # steer applied when NO wall is visible at all
}


@dataclass
class RowState:
    left_edge: float = None       # px, inner edge of left crop wall (near band)
    right_edge: float = None      # px, inner edge of right crop wall
    center: float = None          # px, centreline of the row
    offset_px: float = 0.0        # + = bot is RIGHT of centre
    heading_px: float = 0.0       # + = bot is pointing RIGHT
    offset_frac: float = 0.0      # offset as a fraction of frame width
    half_width: float = None      # px, half the row width (remembered)
    steer: int = 0                # -100..100, + = turn right
    walls: str = "none"           # "both" | "left" | "right" | "none"
    conf: float = 0.0
    note: str = ""
    debug: np.ndarray = None
    veg_frac: float = 0.0


def vegetation_mask(bgr, cfg):
    """
    Excess-Green with Otsu. ExG = 2G - R - B is far more stable across
    sun and shade than an HSV hue range, and Otsu re-picks the threshold
    every frame so a passing cloud does not change the answer.
    """
    b, g, r = cv2.split(bgr.astype(np.int16))
    exg = (2 * g - r - b).clip(0, 255).astype(np.uint8)
    if cfg["exg_thresh"] > 0:
        _, m = cv2.threshold(exg, cfg["exg_thresh"], 255, cv2.THRESH_BINARY)
    else:
        _, m = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


def band_edges(mask, band, cfg):
    """
    Find the inner edges of the crop walls inside one horizontal band.

    Returns (left_edge, right_edge). Either may be None.

    IMPORTANT: we do NOT require the crop to start at the very edge of the
    frame. An earlier version walked inward from column 0, which meant a
    strip of floor -- or an operator's leg -- between the frame edge and the
    start of the crop made the whole wall invisible. That is fine deep inside
    a real row, where the crop fills the frame edges, but it fails on a short
    test track where the camera sees past the ends of the walls.

    Instead: look at each half separately, find every run of crop columns,
    and take the run NEAREST THE CENTRE that is substantial enough to be a
    wall. Its inner end is the edge we want. Gaps at the frame edge no longer
    matter.
    """
    h, w = mask.shape
    y0, y1 = int(band[0] * h), int(band[1] * h)
    strip = mask[y0:y1, :]
    if strip.size == 0:
        return None, None

    prof = strip.mean(axis=0) / 255.0
    k = max(3, cfg["smooth_cols"])
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    is_crop = prof >= cfg["veg_min_col"]

    min_run = cfg.get("min_wall_px", 12)

    def runs(a, offset=0):
        out, start = [], None
        for i, v in enumerate(a):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= min_run:
                    out.append((start + offset, i - 1 + offset))
                start = None
        if start is not None and len(a) - start >= min_run:
            out.append((start + offset, len(a) - 1 + offset))
        return out

    mid = w // 2
    left_runs = runs(is_crop[:mid], 0)
    right_runs = runs(is_crop[mid:], mid)

    # nearest-to-centre run on each side; its INNER end is the wall edge
    left_edge = float(max(r[1] for r in left_runs)) if left_runs else None
    right_edge = float(min(r[0] for r in right_runs)) if right_runs else None

    return left_edge, right_edge


class RowFollower:
    def __init__(self, cfg=None):
        self.cfg = {**CFG, **(cfg or {})}
        self._offset = 0.0
        self._heading = 0.0
        self._half_width = None
        self._last_seen = "none"

    def update(self, frame) -> RowState:
        cfg = self.cfg
        st = RowState()
        h, w = frame.shape[:2]
        mid = w / 2.0

        mask = vegetation_mask(frame, cfg)
        st.veg_frac = float(mask.mean()) / 255.0

        nl, nr = band_edges(mask, cfg["band_near"], cfg)
        fl, fr = band_edges(mask, cfg["band_far"], cfg)

        # ---- 1. what can we see? ------------------------------------
        if nl is not None and nr is not None and (nr - nl) >= cfg["min_corridor_px"]:
            st.walls = "both"
        elif nl is not None:
            st.walls = "left"
        elif nr is not None:
            st.walls = "right"
        else:
            st.walls = "none"

        # ---- 2. centreline -------------------------------------------
        if st.walls == "both":
            st.left_edge, st.right_edge = nl, nr
            st.center = (nl + nr) / 2.0
            # learn the row half-width while we can see both walls
            hw = (nr - nl) / 2.0
            self._half_width = hw if self._half_width is None else (
                cfg["halfwidth_memory"] * self._half_width
                + (1 - cfg["halfwidth_memory"]) * hw)
            st.conf = 1.0
            st.note = "both walls"

        elif st.walls == "left" and self._half_width:
            # Only the left wall: the right wall has slid out of frame, which
            # happens when we move TOWARD the left wall. We drifted LEFT.
            st.left_edge = nl
            st.center = nl + self._half_width
            st.conf = 0.6
            st.note = "left wall only - drifted LEFT, steer right"

        elif st.walls == "right" and self._half_width:
            st.right_edge = nr
            st.center = nr - self._half_width
            st.conf = 0.6
            st.note = "right wall only - drifted RIGHT, steer left"

        elif st.walls in ("left", "right"):
            # One wall but we have never seen both, so no width to work from.
            st.center = None
            st.conf = 0.25
            st.note = f"{st.walls} wall only, row width unknown"

        else:
            st.center = None
            st.conf = 0.0
            st.note = "no crop visible"

        st.half_width = self._half_width

        # ---- 3. cross-track and heading ------------------------------
        if st.center is not None:
            raw_offset = st.center - mid          # + = centre is right of us
            # invert: if the row centre appears to our right, WE are left of it
            offset = -raw_offset

            # heading from the far band, when we have it
            far_center = None
            if fl is not None and fr is not None:
                far_center = (fl + fr) / 2.0
            elif fl is not None and self._half_width:
                far_center = fl + self._half_width
            elif fr is not None and self._half_width:
                far_center = fr - self._half_width

            if far_center is not None:
                heading = -((far_center - mid) - raw_offset)
            else:
                heading = self._heading * 0.7      # decay, do not invent

            a = cfg["smooth"] * max(st.conf, 0.3)
            self._offset = a * offset + (1 - a) * self._offset
            self._heading = a * heading + (1 - a) * self._heading

            st.offset_px = self._offset
            st.heading_px = self._heading
            st.offset_frac = self._offset / float(w)

            # normalise to fraction of frame width so gains are
            # resolution-independent
            off_frac = st.offset_px / float(w)
            hdg_frac = st.heading_px / float(w)
            steer = (cfg["kp_offset"] * off_frac
                     + cfg["kp_heading"] * hdg_frac)
            st.steer = int(np.clip(-steer, -cfg["steer_max"], cfg["steer_max"]))
            # sign: offset positive means the row centre is to our LEFT,
            # so we must steer LEFT, i.e. negative steer.

        else:
            # No usable centre. If we saw a wall recently, keep turning away
            # from the side we can still see. Otherwise hold straight and let
            # the caller decide to stop.
            # Only the left wall visible -> the right wall slid out of frame
            # -> we moved toward the left -> steer RIGHT to come back.
            if st.walls == "left":
                st.steer = +cfg["recover_steer"]
                st.note += " | blind recovery: steering RIGHT"
            elif st.walls == "right":
                st.steer = -cfg["recover_steer"]
                st.note += " | blind recovery: steering LEFT"
            else:
                st.steer = 0

        st.debug = self._draw(frame, mask, st, (nl, nr), (fl, fr))
        return st

    # ---------------------------------------------------------------- draw
    def _draw(self, frame, mask, st, near, far):
        dbg = frame.copy()
        h, w = mask.shape
        green = np.zeros_like(dbg)
        green[:, :, 1] = mask
        dbg = cv2.addWeighted(dbg, 1.0, green, 0.30, 0)

        for band, (le, re), col, tag in (
                (self.cfg["band_near"], near, (0, 255, 255), "near"),
                (self.cfg["band_far"], far, (255, 170, 0), "far")):
            y = int((band[0] + band[1]) / 2 * h)
            cv2.line(dbg, (0, y), (w, y), (70, 70, 70), 1)
            if le is not None:
                cv2.line(dbg, (int(le), y - 9), (int(le), y + 9), col, 2)
            if re is not None:
                cv2.line(dbg, (int(re), y - 9), (int(re), y + 9), col, 2)
            if le is not None and re is not None:
                cv2.line(dbg, (int(le), y), (int(re), y), col, 1)

        # image centre (where we want the row centre to be)
        cv2.line(dbg, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        # detected row centre
        if st.center is not None:
            cx = int(st.center)
            cv2.line(dbg, (cx, int(h * 0.4)), (cx, h), (0, 120, 255), 2)
            cv2.circle(dbg, (cx, int(h * 0.82)), 5, (0, 120, 255), -1)

        cv2.rectangle(dbg, (0, 0), (w, 32), (0, 0, 0), -1)
        wall_col = {"both": (0, 255, 0), "left": (0, 200, 255),
                    "right": (0, 200, 255), "none": (0, 0, 255)}[st.walls]
        cv2.putText(dbg, f"walls {st.walls}   steer {st.steer:+d}", (5, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, wall_col, 1)
        cv2.putText(dbg, f"off {st.offset_px:+.0f}px  hdg {st.heading_px:+.0f}px"
                         f"  conf {st.conf:.2f}", (5, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        return dbg
