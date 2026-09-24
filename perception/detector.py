"""
Leaf abnormality detector, v4 -- hue-based.

WHY HUE, AFTER ALL THAT
-----------------------
v3 decided abnormality from d = (G-R)/(R+G+B), chosen because it is
invariant to uniform illumination change. Sound reasoning -- but measured on
real leaves from this project it is beaten by plain HSV hue:

    index    healthy   chlorotic   necrotic    worst-case separation
    hue        40.8       32.0       17.2        2.01   clean
    d          +0.08      +0.02      -0.11       1.75   usable
    b*        162.8      178.5      154.1        0.73   overlapping
    a*        107.5      111.3      134.9        0.70   overlapping
    ExG        97.4      111.4        9.5        0.56   overlapping

The pixel distributions barely touch:

    healthy      p5 37   p50 40   p95 48
    chlorotic    p5 27   p50 32   p95 36
    necrotic     p5 13   p50 17   p95 21

Chlorotic tops out at 36 and healthy starts at 37, so ONE threshold near 36
separates healthy from both yellowing and browning. That is the whole rule.

THE OLD HUE BUG IS NOT BACK
---------------------------
The week-5 detector also used hue and failed, because it used hue TWICE: once
to decide "is this plant at all" and again to decide "is this diseased".
Necrotic tissue failed the first test, so it was discarded as background
before the second ever ran.

Here hue decides only the second question. The leaf is still found without
hue -- green core plus growth into anything that is not background, plus
hole filling -- so brown tissue is inside the leaf mask before we judge it.
That separation of concerns is what makes hue safe to use now.

WHAT IS STILL TRUE FROM v3
--------------------------
Everything except the decision rule: leaf-first segmentation, adaptive
specular removal, shadow exclusion, hysteresis, blob filtering, and the
refusal to judge a frame that is mostly glare. d is still computed and
reported, because it is a useful sanity check on whether a region is
vegetation at all.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from config import settings as S

P = dict(S.DETECTOR)
SEVERITY_BANDS = S.SEVERITY_BANDS


@dataclass
class Result:
    leaf_mask: np.ndarray = None
    healthy_mask: np.ndarray = None
    abnormal_mask: np.ndarray = None
    unknown_mask: np.ndarray = None
    blobs: list = field(default_factory=list)
    leaf_px: int = 0
    abnormal_px: int = 0
    ratio: float = 0.0
    severity: str = "none"
    trusted: bool = True
    note: str = ""
    d_ref: float = 0.0          # this image's own healthy greenness
    d_thresh: float = 0.0       # the threshold actually applied
    core_mask: np.ndarray = None    # the green seed the leaf grew from
    bg_mask: np.ndarray = None      # what was judged background
    d_map: np.ndarray = None        # the raw (G-R)/(R+G+B) field
    debug: dict = field(default_factory=dict)


# ---------------------------------------------------------------- indices
def green_red_index(bgr):
    """d = (G - R) / (R + G + B). Invariant to uniform illumination change."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    return (g - r) / (r + g + b + 1e-6)


def excess_green(bgr):
    b, g, r = cv2.split(bgr.astype(np.int16))
    return (2 * g - r - b).clip(0, 255).astype(np.uint8)


def fill_holes(mask):
    """Flood from the border; what the flood cannot reach is a lesion
    surrounded by leaf. This recovers necrosis no hue test can find."""
    h, w = mask.shape
    ff = mask.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(ff))


def specular_index(bgr):
    """
    min(R,G,B) per pixel.

    A specular highlight is the light source reflected off the leaf's waxy
    surface WITHOUT being coloured by it -- so it adds roughly equal amounts
    to all three channels. Healthy leaf tissue has a low blue channel, so the
    minimum channel is low; add white and the minimum shoots up. That makes
    min(R,G,B) a direct measure of "how much white is mixed into this pixel".

    This matters because additive white pulls (G-R)/(R+G+B) toward zero, which
    looks exactly like chlorosis. A glossy leaf tilted to the sun will be
    reported as diseased unless these pixels are excluded first.
    """
    return bgr.min(axis=2).astype(np.float32)


def robust_spread(x):
    """MAD, scaled to be comparable with a standard deviation."""
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


# ---------------------------------------------------------------- leaf
def background_model(bgr, border_frac=0.07):
    """
    Learn the background colour from the border ring of the frame.

    The leaf is the thing in the middle; whatever is around the outside is
    table, soil or hand. Sampling the ring gives a background reference with
    no user input and no fixed colour assumption.
    """
    h, w = bgr.shape[:2]
    bh, bw = max(2, int(h * border_frac)), max(2, int(w * border_frac))
    ring = np.concatenate([
        bgr[:bh].reshape(-1, 3), bgr[-bh:].reshape(-1, 3),
        bgr[:, :bw].reshape(-1, 3), bgr[:, -bw:].reshape(-1, 3)])
    lab = cv2.cvtColor(ring.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    lab = lab.astype(np.float32)
    med = np.median(lab, axis=0)
    mad = np.median(np.abs(lab - med), axis=0) * 1.4826
    return med, np.maximum(mad, 4.0)


def segment_leaf(bgr, p):
    """
    Find the WHOLE leaf, including necrotic edges.

    Why this is not just a green mask
    ---------------------------------
    A green mask plus hole-filling recovers a lesion that is SURROUNDED by
    healthy tissue -- it is an interior hole, so the flood cannot reach it.
    But a lesion on the leaf MARGIN is not a hole. It is a bite taken out of
    the outline, open to the background, and flood-filling can never recover
    it. The lesion then sits outside the leaf mask entirely and is never
    judged -- which looks exactly like "it thinks the abnormal part is
    outside the leaf".

    So the leaf is grown outward from a green core instead:

      1. core     = confidently-green tissue (this is only a SEED)
      2. allowed  = everything that does not look like the background
      3. leaf     = the connected regions of `allowed` that touch the core

    Step 3 is a geodesic reconstruction done in one pass with connected
    components. A brown edge lesion is not green, so it is not in the core --
    but it is not background either, and it is physically attached to the
    leaf, so it is swept in. Nothing about its colour is assumed.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    healthy_seed = cv2.inRange(
        hsv, (p["healthy_h"][0], p["healthy_s_min"], p["healthy_v_min"]),
        (p["healthy_h"][1], 255, 255))

    exg = excess_green(bgr)
    if p["exg_thresh"] > 0:
        _, veg = cv2.threshold(exg, p["exg_thresh"], 255, cv2.THRESH_BINARY)
    else:
        _, veg = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    core = cv2.bitwise_or(veg, healthy_seed)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    total = bgr.shape[0] * bgr.shape[1]

    # keep only substantial green regions as seeds
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    seed = np.zeros_like(core)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= p["min_leaf_frac"] * total * 0.5:
            seed[lab == i] = 255

    allowed = None
    if p.get("leaf_method", "grow") == "green" or seed.sum() == 0:
        leaf = core
    else:
        # --- 2. what is NOT background -------------------------------------
        med, mad = background_model(bgr, p.get("bg_border_frac", 0.07))
        labimg = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Chroma distance matters more than lightness: shadow changes L a lot
        # but a and b hardly at all, so a shaded leaf still reads as not-soil.
        wl = p.get("bg_L_weight", 0.35)
        dist = (np.abs(labimg[:, :, 0] - med[0]) / mad[0] * wl
                + np.abs(labimg[:, :, 1] - med[1]) / mad[1]
                + np.abs(labimg[:, :, 2] - med[2]) / mad[2])
        allowed = (dist > p.get("bg_k", 3.0)).astype(np.uint8) * 255
        allowed = cv2.morphologyEx(allowed, cv2.MORPH_CLOSE,
                                   np.ones((p["close_k"], p["close_k"]), np.uint8))
        allowed = cv2.bitwise_or(allowed, seed)      # the core is always allowed

        # --- 3. keep only what is connected to the green core --------------
        n, lab = cv2.connectedComponents(allowed, 8)
        keep_ids = [i for i in np.unique(lab[seed > 0]) if i != 0]
        leaf = (np.isin(lab, keep_ids).astype(np.uint8) * 255
                if keep_ids else core)

        # --- 3b. BOUND THE GROWTH ------------------------------------------
        # Connectivity alone is not enough. A leaf touching a stem touching a
        # wall touching a roof is all one connected region, so the mask
        # happily swallowed an entire building and reported "leaf = 69% of
        # frame". Then the tissue test correctly rejected it, and the leaf we
        # actually wanted went with it.
        #
        # Physical constraint: diseased tissue is ON a leaf, so it is never
        # far from green tissue. Growth is therefore capped at a fixed
        # distance from the green core. A brown edge lesion is ~20 px from
        # something green. A building is not.
        grow_px = int(p.get("grow_max_px", 40))
        if grow_px > 0:
            k = 2 * grow_px + 1
            reach = cv2.dilate(seed, np.ones((k, k), np.uint8))
            leaf = cv2.bitwise_and(leaf, reach)

    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE,
                            np.ones((p["close_k"], p["close_k"]), np.uint8))
    if p["fill_holes"]:
        leaf = fill_holes(leaf)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(leaf, 8)
    keep = np.zeros_like(leaf)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if p["min_leaf_frac"] * total <= a <= p.get("max_leaf_frac", 0.92) * total:
            keep[lab == i] = 255
    if keep.sum() == 0:                    # growth ran away or found nothing
        keep = core
    bg = cv2.bitwise_not(allowed) if p.get("leaf_method", "grow") != "green" \
        and seed.sum() > 0 else cv2.bitwise_not(core)
    return keep, healthy_seed, seed, bg


# ---------------------------------------------------------------- main
def detect(bgr, p=None):
    p = {**P, **(p or {})}
    res = Result()
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    Hc, Sc, Vc = cv2.split(hsv)

    leaf, healthy_seed, core, bg = segment_leaf(blur, p)
    res.leaf_mask = leaf
    res.healthy_mask = healthy_seed
    res.core_mask = core
    res.bg_mask = bg
    res.leaf_px = int(leaf.sum() / 255)
    res.abnormal_mask = np.zeros_like(leaf)
    res.unknown_mask = np.zeros_like(leaf)

    if res.leaf_px < 500:
        res.trusted = False
        res.note = "no leaf found"
        return res

    # ---- 1. exclusions, computed BEFORE the reference -----------------
    # Specular glare on a tilted glossy leaf is bright AND colourless; deep
    # shadow carries no colour either. Both would poison the reference.
    lm = leaf > 0
    # (i) hard glare: blown out and colourless, by absolute thresholds
    glare = cv2.bitwise_and(cv2.inRange(Sc, 0, p["glare_s_max"]),
                            cv2.inRange(Vc, p["glare_v_min"], 255))
    # (ii) adaptive specular: much more white mixed in than the rest of THIS
    #      leaf. Absolute thresholds miss this on a dim shot and over-trigger
    #      on a bright one, which is why it is measured relative to the leaf.
    spec = specular_index(blur)
    sv = spec[lm]
    spec_ref = float(np.median(sv))
    spec_spread = max(robust_spread(sv), 4.0)
    spec_mask = ((spec > spec_ref + p.get("k_specular", 3.0) * spec_spread)
                 & lm).astype(np.uint8) * 255
    spec_mask = cv2.morphologyEx(spec_mask, cv2.MORPH_CLOSE,
                                 np.ones((9, 9), np.uint8))

    # Only true black is treated as unjudgeable shadow. The old threshold of
    # 32 removed dark necrotic tissue along with the shadows.
    shadow = cv2.inRange(Vc, 0, p.get("shadow_v_max", 18))
    unknown = cv2.bitwise_and(
        cv2.bitwise_or(cv2.bitwise_or(glare, spec_mask), shadow), leaf)
    res.unknown_mask = unknown

    judgeable = cv2.bitwise_and(leaf, cv2.bitwise_not(unknown))
    jm = judgeable > 0
    if jm.sum() < 400:
        res.trusted = False
        res.note = "leaf is almost entirely glare or shadow"
        return res

    # ---- 2. invariant index + this image's own reference ---------------
    d = green_red_index(blur)
    res.d_map = d
    dv = d[jm]
    d_ref = float(np.percentile(dv, p.get("ref_percentile", 75)))

    upper = dv[dv >= np.median(dv)]              # spread of the HEALTHY side
    spread = max(robust_spread(upper), 0.008)    # so lesions can't inflate it

    # ---- 2a. IS THIS EVEN A PLANT? --------------------------------------
    # Leaf tissue -- healthy, yellowing or brown -- lives in hue 8..95.
    # Skin, sky, concrete and painted surfaces do not. This replaces the old
    # greenness gate, which rejected a genuinely diseased leaf for the crime
    # of not being green enough: exactly the bug we were trying to detect.
    # Hue alone is NOT enough. Measured on this project's own images:
    #
    #     material          hue   saturation
    #     leaf healthy       41       143
    #     leaf chlorotic     32       132
    #     leaf necrotic      17       140
    #     brick wall         10        32
    #     concrete floor     13        21
    #     white wall        123         9
    #     red container       1       229
    #
    # Brick and concrete land in the same hue band as necrotic tissue, which
    # is why a brick wall was reported as a diseased leaf. SATURATION is what
    # separates them: every leaf type sits near 130-145 while background
    # material is far below or far above. Living tissue holds pigment;
    # masonry does not, and painted plastic overshoots.
    hue_lo = p.get("hue_min_tissue", 8.0)
    hue_hi = p.get("hue_max_tissue", 95.0)
    sat_lo = p.get("sat_min_tissue", 55.0)
    sat_hi = p.get("sat_max_tissue", 205.0)
    hue_all = Hc.astype(np.float32)
    sat_all = Sc.astype(np.float32)
    tissue = ((hue_all >= hue_lo) & (hue_all <= hue_hi)
              & (sat_all >= sat_lo) & (sat_all <= sat_hi))

    # PRUNE, do not reject.
    #
    # The leaf mask grows outward from green tissue, and in a cluttered scene
    # that growth follows leaf -> stem -> pole -> roof and swallows the whole
    # frame. The old code then measured "only 41% of this looks like leaf"
    # and threw the frame away -- taking the real leaf with it.
    #
    # Cutting the non-tissue parts OUT of the mask is strictly better: the
    # building disappears, the leaf survives, and the ratio afterwards is
    # computed over actual foliage.
    tis_u8 = (tissue.astype(np.uint8) * 255)
    tis_u8 = cv2.morphologyEx(tis_u8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    tis_u8 = cv2.morphologyEx(tis_u8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    leaf = cv2.bitwise_and(leaf, tis_u8)

    # drop anything left that is too small to be a leaf
    total_px = bgr.shape[0] * bgr.shape[1]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(leaf, 8)
    pruned = np.zeros_like(leaf)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= p["min_leaf_frac"] * total_px:
            pruned[lab == i] = 255
    leaf = pruned
    res.leaf_mask = leaf
    res.leaf_px = int(leaf.sum() / 255)

    if res.leaf_px < 500:
        res.d_ref = d_ref
        res.trusted = False
        res.note = "no plant tissue found (nothing with a leaf-like hue and saturation)"
        res.severity = "none"
        return res

    # recompute the judgeable region against the pruned mask
    unknown = cv2.bitwise_and(unknown, leaf)
    res.unknown_mask = unknown
    jm = (leaf > 0) & (unknown == 0)
    if jm.sum() < 400:
        res.d_ref = d_ref
        res.trusted = False
        res.note = "leaf is almost entirely glare or shadow"
        return res

    # ---- 2c. THE DECISION: hue ------------------------------------------
    # Healthy foliage sits above hue_healthy_min. Yellowing drops into the
    # low 30s, browning into the teens. One threshold catches both.
    #
    # A relative term rides on top: a pixel also counts as abnormal if it is
    # well below the hue of THIS leaf's own healthy tissue. That keeps the
    # detector working on a naturally pale or naturally dark cultivar without
    # re-tuning the absolute number.
    hue = hue_all
    hue_abs = p.get("hue_healthy_min", 36.0)
    hue_abs_strong = p.get("hue_strong", 30.0)
    k_rel = p.get("hue_k_rel", 2.0)

    hv = hue[jm]
    hue_ref = float(np.percentile(hv, p.get("hue_ref_percentile", 75)))
    upper = hv[hv >= np.median(hv)]
    hue_spread = max(robust_spread(upper), 2.0)

    # The relative term is OFF by default, and that is deliberate.
    #
    # A reference taken from the leaf itself sounds smart, and it is what v3
    # did. But on a leaf that is diseased ALL OVER, the reference is itself
    # low, the threshold slides down with it, and nothing is flagged. That
    # exact failure produced the "whole leaf" special case in v3 and cost
    # days of debugging.
    #
    # The measured distributions do not need it. Healthy starts at 37,
    # chlorotic ends at 36: a fixed threshold is cleanly correct and cannot
    # slide out from under you. Turn it on only if you have a cultivar whose
    # healthy hue genuinely sits below 37.
    if p.get("hue_use_relative", False):
        hue_rel = hue_ref - k_rel * hue_spread
        # never let it go BELOW the absolute rule, only extend above it
        thr = float(np.clip(max(hue_abs, hue_rel), hue_abs, hue_abs + 8.0))
        thr_strong = thr - (hue_abs - hue_abs_strong)
    else:
        thr, thr_strong = hue_abs, hue_abs_strong

    res.d_ref = d_ref
    res.d_thresh = float(thr)
    res.debug = {"hue_ref": round(hue_ref, 1),
                 "hue_spread": round(hue_spread, 2),
                 "hue_thresh": round(thr, 1),
                 "d_ref": round(d_ref, 3),
                 "spread": round(spread, 4),
                 "spec_ref": round(spec_ref, 1)}

    # Very low hue wraps toward red/brown, which is exactly necrosis, so we
    # do NOT exclude it. But hue above ~90 is blue/purple -- sky, a shirt,
    # water -- and is never leaf tissue, so it is not called diseased either.
    # A pixel is only a candidate lesion if it could be leaf tissue at all.
    plausible = tissue

    strong = (hue < thr_strong) & plausible & jm
    weak = (hue < thr) & plausible & jm

    # ---- 3. hysteresis: grow strong seeds into weak neighbours ---------
    weak_u8 = (weak * 255).astype(np.uint8)
    n, lab = cv2.connectedComponents(weak_u8, 8)
    keep_ids = [i for i in np.unique(lab[strong]) if i != 0]
    abnormal = (np.isin(lab, keep_ids).astype(np.uint8) * 255
                if keep_ids else np.zeros_like(weak_u8))

    abnormal = cv2.morphologyEx(abnormal, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    abnormal = cv2.morphologyEx(abnormal, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # ---- 4. blob filtering ---------------------------------------------
    n, lab, stats, cent = cv2.connectedComponentsWithStats(abnormal, 8)
    kept = np.zeros_like(abnormal)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        frac = a / res.leaf_px
        if a < p["min_blob_px"] or frac < p["min_blob_frac_of_leaf"]:
            continue
        # A single blob covering nearly the whole leaf is normal here: a
        # fully chlorotic leaf IS one big lesion. Only flag it as suspicious,
        # do not discard it.
        if frac > p["max_blob_frac_of_leaf"]:
            res.note = "lesion covers almost the whole leaf"
        kept[lab == i] = 255
        res.blobs.append({
            "x": int(stats[i, cv2.CC_STAT_LEFT]), "y": int(stats[i, cv2.CC_STAT_TOP]),
            "w": int(stats[i, cv2.CC_STAT_WIDTH]), "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
            "area": int(a), "frac": round(float(frac), 4),
            "cx": float(cent[i][0]), "cy": float(cent[i][1]),
            "mean_d": round(float(d[lab == i].mean()), 4),
        })

    res.abnormal_mask = kept
    res.abnormal_px = int(kept.sum() / 255)
    res.ratio = res.abnormal_px / res.leaf_px

    unk_frac = float(unknown.sum() / 255) / res.leaf_px
    if unk_frac > 0.25:
        res.trusted = False
        res.note = (f"{unk_frac:.0%} of the leaf is glare or shadow - "
                    "move the light or the leaf")
    elif spread > 0.035:
        res.trusted = False
        res.note = "leaf colour is very uneven - probably harsh side lighting"

    for lim, name in SEVERITY_BANDS:
        if res.ratio < lim:
            res.severity = name if res.ratio > 0 else "none"
            break
    return res


# ---------------------------------------------------------------- draw
def overlay(bgr, res):
    out = bgr.copy()
    if res.leaf_mask is not None:
        cnts, _ = cv2.findContours(res.leaf_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 200, 0), 2)
    if res.unknown_mask is not None and res.unknown_mask.any():
        m = res.unknown_mask > 0
        out[m] = (0.55 * out[m] + 0.45 * np.array([120, 120, 120])).astype(np.uint8)
    if res.abnormal_mask is not None and res.abnormal_mask.any():
        m = res.abnormal_mask > 0
        out[m] = (0.35 * out[m] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    for b in res.blobs:
        cv2.rectangle(out, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]),
                      (0, 255, 255), 2)
    tag = (f"{res.severity}  {res.ratio:.1%} of leaf  blobs={len(res.blobs)}  "
           f"d_ref={res.d_ref:+.3f}")
    if not res.trusted:
        tag += "  [UNTRUSTED]"
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, tag, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255) if res.trusted else (0, 165, 255), 1)
    return out


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    r = detect(img)
    print(f"severity={r.severity} ratio={r.ratio:.3f} blobs={len(r.blobs)} "
          f"d_ref={r.d_ref:+.4f} thresh={r.d_thresh:+.4f} "
          f"trusted={r.trusted} {r.note}")
    cv2.imshow("result", overlay(img, r))
    cv2.waitKey(0)
