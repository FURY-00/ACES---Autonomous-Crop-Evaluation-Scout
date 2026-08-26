"""
Auto-calibration: let the bot look at ~100 plants and work out its own
thresholds.

Why this beats sliders
----------------------
Sliders optimise for the one leaf in front of you. That is exactly how you
end up with settings that work indoors under a flashlight and fail outdoors
an hour later. This collects a spread of real frames in the real light and
fits the numbers to all of them at once.

Two phases
----------
    COLLECT   grab N frames a second or two apart while you drive/carry the
              bot past plants. Everything is saved, so you can re-fit later
              without going outside again.

    FIT       pool the greenness distribution across every frame, find where
              healthy tissue sits and where lesions sit, and put the
              thresholds between them.

Usage
-----
    python3 tools/auto_calibrate.py                    # collect 100, then fit
    python3 tools/auto_calibrate.py --count 60
    python3 tools/auto_calibrate.py --interval 1.5
    python3 tools/auto_calibrate.py --fit-only         # re-fit saved frames
    python3 tools/auto_calibrate.py --fit-only --dir dataset/autocal_0819
    python3 tools/auto_calibrate.py --write            # save the result

    # override any fitted value by hand
    python3 tools/auto_calibrate.py --fit-only --write \\
        --set d_abs_max=0.085 --set min_d_ref=-0.01

Nothing is written unless you pass --write. Without it you get the numbers
printed and can decide.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings as S            # noqa: E402
from perception import detector as D        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "config", "detector_tuned.json")


# ---------------------------------------------------------------- collect
def open_camera():
    try:
        from picamera2 import Picamera2
        pc = Picamera2()
        pc.configure(pc.create_still_configuration(
            main={"size": (1536, 864), "format": "RGB888"}))
        pc.start()
        time.sleep(2.0)
        try:
            from libcamera import controls
            pc.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            time.sleep(1.2)
            md = pc.capture_metadata()
            # Lock white balance for the whole session. If AWB drifts between
            # frames the pooled distribution is a blur of different colour
            # renderings and the fit is meaningless.
            pc.set_controls({"AwbEnable": False,
                             "ColourGains": md.get("ColourGains", (1.8, 1.8))})
            print(f"[cam] AWB locked at {tuple(round(g,2) for g in md.get('ColourGains',(1.8,1.8)))}")
        except Exception as e:
            print(f"[cam] controls: {e}")
        return ("picam", pc)
    except Exception as e:
        print(f"[cam] picamera2 unavailable ({e}); trying USB")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print("[cam] no camera")
            sys.exit(1)
        return ("cv2", cap)


def grab(cam):
    kind, dev = cam
    if kind == "picam":
        return dev.capture_array()
    ok, f = dev.read()
    return f if ok else None


def sharpness(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def collect(outdir, count, interval, min_sharp):
    os.makedirs(outdir, exist_ok=True)
    cam = open_camera()
    print(f"\nCollecting {count} frames, one every {interval:.1f}s.")
    print("Carry or drive the bot past plants. Vary the angle, the distance,")
    print("and include some healthy AND some diseased leaves.")
    print("Ctrl-C to stop early and fit what you have.\n")

    kept = rejected = 0
    try:
        while kept < count:
            f = grab(cam)
            if f is None:
                time.sleep(0.2)
                continue
            sh = sharpness(f)
            if sh < min_sharp:
                rejected += 1
                print(f"  [{kept:3d}/{count}] skipped, blurry ({sh:.0f})",
                      end="\r")
                time.sleep(0.4)
                continue
            path = os.path.join(outdir, f"cal_{kept:04d}.jpg")
            cv2.imwrite(path, f, [cv2.IMWRITE_JPEG_QUALITY, 92])
            kept += 1
            print(f"  [{kept:3d}/{count}] saved  sharpness {sh:6.0f}   "
                  f"({rejected} blurry skipped)", end="\r")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  stopped early")
    finally:
        if cam[0] == "picam":
            cam[1].stop()
        else:
            cam[1].release()
    print(f"\n\n{kept} frames -> {outdir}")
    return outdir


# ---------------------------------------------------------------- fit
def pooled_stats(folder, max_frames=400):
    """
    Walk every frame, segment the leaf, and pool the greenness values.

    We keep per-frame d_ref separately from the pooled pixel values. They
    answer different questions: d_ref tells us what a whole leaf looks like
    (used for the plant/not-plant and whole-leaf gates), while the pooled
    pixels tell us where lesion tissue sits relative to healthy tissue.
    """
    paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))[:max_frames]
    if not paths:
        print(f"no images in {folder}")
        return None

    all_d = []
    d_refs = []
    leaf_fracs = []
    unk_fracs = []
    used = 0

    base = dict(S.DETECTOR)
    base["min_d_ref"] = -1.0            # accept everything while measuring
    base["d_healthy_foliage"] = -1.0

    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, (640, 360))
        r = D.detect(img, base)
        if r.leaf_px < 2000:
            continue
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        d = D.green_red_index(blur)
        jm = (r.leaf_mask > 0)
        if r.unknown_mask is not None:
            jm &= (r.unknown_mask == 0)
        vals = d[jm]
        if vals.size < 500:
            continue
        all_d.append(vals.astype(np.float32))
        d_refs.append(r.d_ref)
        leaf_fracs.append(r.leaf_px / (640 * 360))
        unk_fracs.append(float(r.unknown_mask.sum()) / 255 / max(r.leaf_px, 1))
        used += 1
        if used % 10 == 0:
            print(f"  analysed {used}/{len(paths)}", end="\r")

    if used == 0:
        print("\nNo frames had a usable leaf. Check the framing and lighting.")
        return None

    print(f"  analysed {used}/{len(paths)} frames" + " " * 20)
    return dict(d=np.concatenate(all_d), d_refs=np.array(d_refs),
                leaf=np.array(leaf_fracs), unk=np.array(unk_fracs), used=used)


def otsu_1d(vals, lo, hi, bins=120):
    """
    Otsu's method on a 1-D histogram. Splits the values into two groups so
    that the spread WITHIN each group is as small as possible -- which is
    exactly the question "where does healthy tissue end and lesion begin".
    """
    h, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    h = h.astype(np.float64)
    total = h.sum()
    if total == 0:
        return None
    p = h / total
    centres = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / denom
    sigma_b[~np.isfinite(sigma_b)] = 0
    return float(centres[int(np.argmax(sigma_b))])


def histogram(vals, lo, hi, bins=26, width=46, marks=None):
    counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    top = max(counts.max(), 1)
    marks = marks or {}
    out = []
    for i, c in enumerate(counts):
        a, b = edges[i], edges[i + 1]
        tag = "".join(f"  <-- {n}" for n, v in marks.items() if a <= v < b)
        out.append(f"  {a:+.3f} |{'#' * int(round(c / top * width)):<{width}}|{tag}")
    return "\n".join(out)


def fit(stats, p_over=None):
    d = stats["d"]
    d_refs = stats["d_refs"]

    p_lo = float(np.percentile(d_refs, 5))
    p_med = float(np.median(d_refs))

    # Where healthy tissue sits: upper part of the pooled distribution
    healthy_mode = float(np.percentile(d, 75))
    split = otsu_1d(d, float(np.percentile(d, 1)), float(np.percentile(d, 99)))

    fitted = {}
    # A frame whose whole leaf is less green than the 5th-percentile leaf we
    # actually saw is probably not a leaf at all. Margin of 0.04 below that.
    fitted["min_d_ref"] = round(max(-0.15, p_lo - 0.04), 3)
    # Below the typical leaf's greenness, treat the whole leaf as diseased.
    fitted["d_healthy_foliage"] = round(max(0.02, p_med - 0.05), 3)
    # The lesion/healthy split from Otsu, with a small margin.
    if split is not None:
        fitted["d_abs_max"] = round(min(split + 0.01, healthy_mode - 0.02), 3)
        fitted["d_abs_strong"] = round(fitted["d_abs_max"] - 0.02, 3)
    if p_over:
        fitted.update(p_over)
    return fitted, dict(p_lo=p_lo, p_med=p_med, healthy_mode=healthy_mode,
                        split=split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--interval", type=float, default=1.5)
    ap.add_argument("--min-sharp", type=float, default=40.0)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--set", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="override a fitted value, e.g. --set d_abs_max=0.085")
    a = ap.parse_args()

    folder = a.dir or os.path.join(
        ROOT, "dataset", "autocal_" + time.strftime("%m%d_%H%M"))

    if not a.fit_only:
        folder = collect(folder, a.count, a.interval, a.min_sharp)
    elif a.dir is None:
        cands = sorted(d for d in os.listdir(os.path.join(ROOT, "dataset"))
                       if d.startswith("autocal_")) \
            if os.path.isdir(os.path.join(ROOT, "dataset")) else []
        if not cands:
            print("no autocal folders found. Run without --fit-only first.")
            return
        folder = os.path.join(ROOT, "dataset", cands[-1])
        print(f"using most recent: {folder}")

    print(f"\nfitting from {folder}")
    stats = pooled_stats(folder)
    if stats is None:
        return

    overrides = {}
    for kv in a.set:
        if "=" not in kv:
            print(f"  ignoring bad --set '{kv}' (need KEY=VALUE)")
            continue
        k, v = kv.split("=", 1)
        try:
            overrides[k.strip()] = float(v)
        except ValueError:
            print(f"  ignoring non-numeric --set '{kv}'")

    fitted, info = fit(stats, overrides)

    print("\n" + "=" * 62)
    print(f"POOLED GREENNESS  d = (G-R)/(R+G+B)   from {stats['used']} frames")
    print("=" * 62)
    print(histogram(stats["d"], -0.2, 0.45, marks={
        "abs gate": fitted.get("d_abs_max", 0),
        "healthy": info["healthy_mode"]}))
    print(f"\n  per-frame leaf greenness: p5 {info['p_lo']:+.3f}   "
          f"median {info['p_med']:+.3f}")
    print(f"  healthy tissue sits near  {info['healthy_mode']:+.3f}")
    print(f"  otsu split healthy/lesion {info['split']:+.3f}"
          if info["split"] is not None else "  otsu split: not found")
    print(f"  leaf fills {stats['leaf'].mean():.0%} of frame on average")
    print(f"  {stats['unk'].mean():.0%} unjudgeable (glare/shadow) on average")
    if stats["unk"].mean() > 0.2:
        print("  ! a lot of glare. Collect again in open shade for a better fit.")

    print("\n" + "=" * 62)
    print("FITTED VALUES")
    print("=" * 62)
    for k, v in fitted.items():
        mark = "  (your override)" if k in overrides else ""
        print(f'   "{k}": {v},{mark}')

    if a.write:
        p = dict(S.DETECTOR)
        p.update(fitted)
        out = {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}
        with open(OUT, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n   written -> {OUT}")
        print("   the next run picks this up automatically")
    else:
        print("\n   (add --write to save; add --set KEY=VALUE to override)")
    print(f"\n   frames kept in {folder} — re-fit any time with:")
    print(f"     python3 tools/auto_calibrate.py --fit-only --dir {folder}")


if __name__ == "__main__":
    main()
