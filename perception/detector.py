"""
Leaf abnormality detector, v3.

Three problems from real leaves drove this rewrite:

  (a) healthy tissue was being called abnormal
  (b) the abnormal region changed every time the leaf angle changed
  (c) thresholds tuned on one shot failed on the next

All three have one root cause: HSV hue and saturation are NOT invariant to
illumination. Tilt a glossy leaf and the light reaching the sensor changes
by a factor of two or more across the surface. Hue wobbles, saturation
collapses on the bright side, and any fixed threshold slides around
underneath you. You were not mis-tuning. You were tuning a moving target.

The fix has three parts.

1. AN ILLUMINATION-INVARIANT INDEX

       d = (G - R) / (R + G + B)

   Scale every channel by the same factor -- exactly what shading, tilt and
   exposure do to a matte surface -- and d does not change: the factor
   cancels top and bottom. Hue loses this property once saturation is low,
   which is precisely where diseased tissue lives.

   d is also monotone in disease progression, which is a happy bonus:
       healthy green     d ~ +0.10 .. +0.20
       chlorotic yellow  d ~  0.00 .. +0.06
       necrotic brown    d ~ -0.06 .. +0.02

2. A REFERENCE TAKEN FROM THE IMAGE ITSELF

   Instead of "is this pixel greener than hue 33", we ask "is this pixel
   much less green than the healthy tissue ON THIS LEAF, IN THIS SHOT". The
   reference is the 75th percentile of d inside the leaf. Change the angle,
   the light or the camera and the reference moves with it.

3. AN ABSOLUTE GATE -- this is what kills the false positives

   A purely relative test ALWAYS finds the least-green part of anything.
   Show it a perfectly healthy leaf and it will confidently outline the
   slightly-less-green 5%. That is problem (a), exactly.

   So a pixel must fail BOTH tests: much less green than this leaf's own
   healthy tissue, AND below an absolute greenness that healthy tissue never
   reaches. Relative alone gives false positives; absolute alone gives back
   the old lighting sensitivity. The AND of the two is stable.

Hysteresis on top: strong seeds grow into weakly-abnormal neighbours, so
lesion edges stop flickering from frame to frame.
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
    _, Sc, Vc = cv2.split(hsv)

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

    shadow = cv2.inRange(Vc, 0, p["shadow_v_max"])
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
    # d_ref is the greenness of the tissue we are treating as healthy. Real
    # foliage sits around +0.15 to +0.30. A cable, a wall, a hand or a dark
    # room gives something near zero or negative.
    #
    # Without this check the relative gate happily finds "the least green
    # part" of ANY object and reports it as severe disease. Every stage
    # downstream is conditional on there actually being a leaf here.
    min_d_ref = p.get("min_d_ref", 0.02)
    if d_ref < min_d_ref:
        res.d_ref = d_ref
        res.trusted = False
        res.note = (f"not vegetation (greenness {d_ref:+.3f} < "
                    f"{min_d_ref:+.3f}) - nothing plant-like in frame")
        res.severity = "none"
        return res

    # ---- 2b. THE WHOLE-LEAF CASE ---------------------------------------
    # The relative test compares tissue against the healthy tissue on the
    # SAME leaf. A leaf that is yellow edge to edge has no healthy tissue to
    # compare against, so the relative test finds nothing and the detector
    # reports a severely diseased leaf as perfectly fine.
    #
    # But that leaf is not ambiguous at all -- it is uniformly far below what
    # living foliage looks like. So when the leaf's own reference is itself
    # below healthy-foliage greenness, the answer is not "no disease", it is
    # "all of it".
    #
    # The two bounds are doing different jobs:
    #   below min_d_ref        -> not a plant, reject
    #   min_d_ref .. d_healthy -> a plant, and the WHOLE leaf is abnormal
    #   above d_healthy        -> healthy tissue exists, use the normal path
    d_healthy = p.get("d_healthy_foliage", 0.12)
    if d_ref < d_healthy:
        whole = cv2.bitwise_and(judgeable, judgeable)
        whole = cv2.morphologyEx(whole, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        res.d_ref = d_ref
        res.d_thresh = d_healthy
        res.abnormal_mask = whole
        res.abnormal_px = int(whole.sum() / 255)
        res.ratio = res.abnormal_px / res.leaf_px
        n, lab, stats, cent = cv2.connectedComponentsWithStats(whole, 8)
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            if a < p["min_blob_px"]:
                continue
            res.blobs.append({
                "x": int(stats[i, cv2.CC_STAT_LEFT]),
                "y": int(stats[i, cv2.CC_STAT_TOP]),
                "w": int(stats[i, cv2.CC_STAT_WIDTH]),
                "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
                "area": int(a), "frac": round(a / res.leaf_px, 4),
                "cx": float(cent[i][0]), "cy": float(cent[i][1]),
                "mean_d": round(float(d[lab == i].mean()), 4)})
        res.note = (f"whole leaf is chlorotic/necrotic "
                    f"(greenness {d_ref:+.3f}, healthy foliage is "
                    f"{d_healthy:+.3f}+)")
        for lim, name in SEVERITY_BANDS:
            if res.ratio < lim:
                res.severity = name if res.ratio > 0 else "none"
                break
        return res

    k_strong = p.get("k_strong", 3.0)
    k_weak = p.get("k_weak", 1.8)
    d_abs = p.get("d_abs_max", 0.075)            # the absolute gate
    d_abs_strong = p.get("d_abs_strong", 0.055)

    strong = (d < (d_ref - k_strong * spread)) & (d < d_abs_strong) & jm
    weak = (d < (d_ref - k_weak * spread)) & (d < d_abs) & jm

    res.d_ref = d_ref
    res.d_thresh = float(min(d_ref - k_weak * spread, d_abs))
    res.debug = {"spread": round(spread, 4),
                 "spec_ref": round(spec_ref, 1),
                 "strong_px": int(strong.sum()), "weak_px": int(weak.sum())}

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
        if frac > p["max_blob_frac_of_leaf"]:
            res.trusted = False
            res.note = "one blob covers almost the whole leaf - check segmentation"
            continue
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
