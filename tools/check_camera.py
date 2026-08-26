"""
STEP 1: prove the camera works, before anything else.

The single most common Pi Camera failure is silent: on Raspberry Pi OS
Bookworm, cv2.VideoCapture(0) does NOT open a Pi Camera v3. It either
returns False or, worse, grabs some other video device and hands you black
frames. Your week-5 code used cv2.VideoCapture for the Pi Cam, so if that
never worked, this is why.

The Pi Camera goes through picamera2 (libcamera). USB webcams go through
cv2.VideoCapture. They are different stacks.

Usage
-----
    python tools/check_camera.py

It reports which backends are available, grabs a frame from each, saves it
to camera_test_*.jpg, and prints brightness and sharpness so you can tell a
working camera from a lens cap you forgot to remove.
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def describe(frame, name):
    if frame is None or frame.size == 0:
        print(f"  {name}: NO FRAME")
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright = float(gray.mean())
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sat = float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
    out = f"camera_test_{name}.jpg"
    cv2.imwrite(out, frame)
    print(f"  {name}: {frame.shape[1]}x{frame.shape[0]}  "
          f"brightness {bright:5.1f}  sharpness {sharp:7.0f}  "
          f"saturation {sat:5.1f}")
    print(f"     saved -> {out}")
    if bright < 12:
        print("     ! almost black. Lens cap on? Ribbon cable seated the "
              "right way round?")
    elif bright > 245:
        print("     ! blown out. Pointed at the sky or a lamp?")
    if sharp < 40:
        print("     ! very soft. The v3 has autofocus - give it a second, and "
              "check nothing is closer than about 10 cm.")
    return True


def try_picamera2():
    print("\n[picamera2 / libcamera]  <- this is the one for a Pi Camera v3")
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("  not installed.  sudo apt install -y python3-picamera2")
        return False
    try:
        cams = Picamera2.global_camera_info()
        if not cams:
            print("  no cameras detected by libcamera.")
            print("  check:  rpicam-hello --list-cameras")
            return False
        for i, c in enumerate(cams):
            print(f"  found: [{i}] {c.get('Model','?')}")
        pc = Picamera2()
        pc.configure(pc.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"}))
        pc.start()
        time.sleep(2.5)                    # let AE/AWB and autofocus settle
        frame = cv2.cvtColor(pc.capture_array(), cv2.COLOR_RGB2BGR)
        ok = describe(frame, "picamera2")
        md = pc.capture_metadata()
        print(f"     exposure {md.get('ExposureTime')}us  "
              f"gain {md.get('AnalogueGain', 0):.2f}  "
              f"lens {md.get('LensPosition', 'n/a')}")
        pc.stop()
        return ok
    except Exception as e:
        print(f"  failed: {e}")
        return False


def try_opencv():
    """
    On a Pi, /dev/video0..N are libcamera/ISP nodes, NOT webcams. cv2 can
    open() them but never gets a frame, so each probe blocks for ~10 s and
    prints 'select() timeout'. Those warnings are harmless and expected --
    they mean 'this is not a webcam', not 'your camera is broken'.
    We therefore skip this scan entirely when the Pi Camera already worked.
    """
    print("\n[cv2.VideoCapture]  <- USB webcams only, NOT the Pi Camera")
    found = False
    for idx in (0, 1, 2):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(5):
            ok, frame = cap.read()
        cap.release()
        if ok:
            found = True
            describe(frame, f"usb{idx}")
    if not found:
        print("  no USB cameras responded (fine if you only have the Pi Cam)")
    return found


def check_display():
    print("\n[display]")
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        print("  a display is available - the windowed tools will work")
        return True
    print("  NO display detected. You are probably on SSH without X.")
    print("  The windowed tools (pixel_probe, tune_detector, stage_debug)")
    print("  need a screen. Either:")
    print("    a) work on the Pi's own monitor, or")
    print("    b) use VNC (sudo raspi-config -> Interface Options -> VNC), or")
    print("    c) ssh -X pi@<ip>   (slow but works), or")
    print("    d) use tools/live_detect.py, which streams to a browser and")
    print("       needs no display at all.")
    return False


if __name__ == "__main__":
    print("=" * 62)
    print("ACES camera check")
    print("=" * 62)
    pi = try_picamera2()
    if pi and "--usb" not in sys.argv:
        print("\n[cv2.VideoCapture]  skipped - the Pi Camera already works.")
        print("  (pass --usb to scan for USB webcams too; on a Pi this takes")
        print("   ~30 s and prints harmless 'select() timeout' warnings)")
        usb = False
    else:
        usb = try_opencv()
    check_display()
    print("\n" + "=" * 62)
    if pi:
        print("Pi Camera works. Next:  python tools/live_detect.py")
    elif usb:
        print("USB camera works, Pi Camera did not. live_detect.py will")
        print("fall back to the USB camera automatically.")
    else:
        print("NO WORKING CAMERA. Fix this before anything else:")
        print("  rpicam-hello --list-cameras     # does libcamera see it?")
        print("  rpicam-hello -t 3000            # does it preview?")
        print("  check the ribbon cable: contacts face the board on the Pi")
        print("  end, and face away from the lens on the camera end")
    print("=" * 62)
