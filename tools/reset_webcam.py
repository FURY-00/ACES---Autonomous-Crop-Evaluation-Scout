"""
Reset the webcam to its factory defaults.

Why this is needed: V4L2 controls live in the DRIVER, not in the program that
set them. Run something once with a manual exposure and the camera keeps that
setting after the script exits, after Ctrl-C, and across every later run --
until something explicitly puts it back or the device is replugged.

So a frame that looks over-exposed or over-saturated is not necessarily the
current program's doing. It may be a leftover from a session hours ago.

Usage
-----
    python3 tools/reset_webcam.py                  # auto-detect the device
    python3 tools/reset_webcam.py --device /dev/video1
    python3 tools/reset_webcam.py --show           # just list current values

It reads each control's own default from the driver and writes it back, so it
works whatever camera you have rather than assuming particular numbers.
"""

import argparse
import glob
import re
import subprocess
import sys


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True,
                          text=True).stdout


def find_webcam():
    out = run("v4l2-ctl --list-devices")
    block = None
    for chunk in out.split("\n\n"):
        if "USB" in chunk and "Camera" in chunk:
            block = chunk
            break
    if block:
        nodes = re.findall(r"(/dev/video\d+)", block)
        for n in nodes:
            # the first node that actually reports capture controls
            if "brightness" in run(f"v4l2-ctl -d {n} --list-ctrls"):
                return n
        if nodes:
            return nodes[0]
    for n in sorted(glob.glob("/dev/video*")):
        if "brightness" in run(f"v4l2-ctl -d {n} --list-ctrls"):
            return n
    return None


def parse_ctrls(dev):
    """Return [(name, current, default), ...] for every writable control."""
    out = run(f"v4l2-ctl -d {dev} --list-ctrls")
    ctrls = []
    for line in out.splitlines():
        m = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\((int|bool|menu)\)\s*:(.*)",
                     line)
        if not m:
            continue
        name, _, rest = m.groups()
        dm = re.search(r"default=(-?\d+)", rest)
        vm = re.search(r"value=(-?\d+)", rest)
        if dm and vm:
            ctrls.append((name, int(vm.group(1)), int(dm.group(1))))
    return ctrls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    dev = a.device or find_webcam()
    if not dev:
        print("No USB webcam found. Check it is plugged in:  v4l2-ctl --list-devices")
        sys.exit(1)
    print(f"device: {dev}\n")

    ctrls = parse_ctrls(dev)
    if not ctrls:
        print("No controls reported. Is this the capture node? Try the other")
        print("/dev/videoN listed under your camera in:  v4l2-ctl --list-devices")
        sys.exit(1)

    changed = [c for c in ctrls if c[1] != c[2]]
    print(f"{'control':32s} {'current':>9s} {'default':>9s}")
    print("-" * 54)
    for name, cur, dflt in ctrls:
        mark = "  <- not default" if cur != dflt else ""
        print(f"{name:32s} {cur:9d} {dflt:9d}{mark}")

    if a.show:
        return

    if not changed:
        print("\nEverything is already at its default. If the image still")
        print("looks wrong, it is the lighting, not the camera settings.")
        return

    print(f"\nresetting {len(changed)} control(s)...")
    # auto modes first: a manual exposure cannot be cleared while the
    # auto flag is still off
    order = sorted(changed, key=lambda c: 0 if "auto" in c[0] else 1)
    for name, cur, dflt in order:
        r = subprocess.run(f"v4l2-ctl -d {dev} --set-ctrl={name}={dflt}",
                           shell=True, capture_output=True, text=True)
        ok = "ok" if r.returncode == 0 else f"failed: {r.stderr.strip()[:40]}"
        print(f"  {name:30s} {cur} -> {dflt}   {ok}")

    print("\nDone. Re-check with:  python3 tools/reset_webcam.py --show")
    print("If a control refuses to move, unplug and replug the webcam --")
    print("that always restores factory defaults.")


if __name__ == "__main__":
    main()
