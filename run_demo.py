"""
ACES demo runner.

The whole loop, using only hardware you have working today:

    ESP32 drives itself down a corridor using the two side sonars
      -> Pi Camera watches for diseased tissue
      -> lesion found: Pi tells the ESP32 to STOP
      -> full-res photo, GPS stamp, saved to SD, pin dropped on the live map
      -> Pi tells the ESP32 to GO
      -> end of corridor: stop, report

No nav camera. No encoders. No IR.

Run it
------
    python3 run_demo.py                      # everything
    python3 run_demo.py --no-drive           # perception only, motors never move
    python3 run_demo.py --port /dev/ttyUSB0

Two browser tabs:
    http://<pi-ip>:8080    live detection view
    http://<pi-ip>:8081    live map with detection pins

SAFETY
------
CH5 on the transmitter is the master switch. CH5 low = manual RC, instantly.
The ESP32 also stops on its own if this script goes quiet for 2 seconds, so
closing the terminal stops the robot.
"""

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings as S            # noqa: E402
from perception import detector as D        # noqa: E402
from perception.classifier import DiseaseClassifier   # noqa: E402
from telemetry import map_server, stream_server   # noqa: E402
from telemetry.storage import Storage       # noqa: E402
from telemetry.telegram_bot import Telegram  # noqa: E402

try:
    from telemetry.gps_reader import GPSReader
except Exception:
    GPSReader = None


# ------------------------------------------------------------------ link
@dataclass
class Tlm:
    mode: str = "?"
    left: float = 999.0
    right: float = 999.0
    err: float = 0.0
    wall_l: bool = False
    wall_r: bool = False
    stamp: float = 0.0

    @property
    def fresh(self):
        return (time.time() - self.stamp) < 1.0


class Link:
    """Line protocol to the demo firmware. Degrades to a no-op if absent."""

    def __init__(self, port, baud=115200, enabled=True):
        self.tlm = Tlm()
        self.events = []
        self.ok = False
        self._lock = threading.Lock()
        if not enabled:
            print("[esp32] disabled (--no-drive): motors are never commanded")
            return
        try:
            import serial
            self.ser = serial.Serial(port, baud, timeout=0.2)
            time.sleep(2.0)              # ESP32 resets when the port opens
            self.ok = True
            threading.Thread(target=self._rx, daemon=True).start()
            print(f"[esp32] connected on {port}")
        except Exception as e:
            print(f"[esp32] NOT connected ({e}) - running perception only")

    def _rx(self):
        while True:
            try:
                ln = self.ser.readline().decode("ascii", "ignore").strip()
            except Exception:
                time.sleep(0.1)
                continue
            if ln.startswith("#T,"):
                p = ln[3:].split(",")
                if len(p) == 7:
                    try:
                        with self._lock:
                            self.tlm = Tlm(mode=p[1], left=float(p[2]),
                                           right=float(p[3]), err=float(p[4]),
                                           wall_l=p[5] == "1", wall_r=p[6] == "1",
                                           stamp=time.time())
                    except ValueError:
                        pass
            elif ln.startswith("#E,"):
                with self._lock:
                    self.events.append(ln[3:])
                print(f"[esp32] {ln[3:]}")

    def read(self):
        with self._lock:
            return self.tlm

    def pop_events(self):
        with self._lock:
            e, self.events = self.events, []
        return e

    def _tx(self, s):
        if self.ok:
            try:
                self.ser.write((s + "\n").encode())
            except Exception:
                pass

    def go(self):   self._tx("$G")
    def stop(self): self._tx("$S")
    def turn(self, d=1): self._tx(f"$T,{d}")


# ------------------------------------------------------------------ camera
def open_camera():
    try:
        from picamera2 import Picamera2
        pc = Picamera2()
        pc.configure(pc.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"}))
        pc.start()
        time.sleep(2.0)
        try:
            from libcamera import controls
            pc.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            time.sleep(1.2)
            md = pc.capture_metadata()
            pc.set_controls({"AwbEnable": False,
                             "ColourGains": md.get("ColourGains", (1.8, 1.8))})
        except Exception as e:
            print(f"[cam] focus/awb: {e}")
        return ("picam", pc)
    except Exception as e:
        print(f"[cam] picamera2 unavailable ({e})")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[cam] NO CAMERA")
            sys.exit(1)
        return ("cv2", cap)


def grab(cam):
    kind, dev = cam
    if kind == "picam":
        return dev.capture_array()       # RGB888 array is already BGR
    ok, f = dev.read()
    return f if ok else None


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--no-drive", action="store_true")
    ap.add_argument("--no-telegram", action="store_true")
    ap.add_argument("--cooldown", type=float, default=6.0,
                    help="seconds before the same area can trigger again")
    ap.add_argument("--no-classify", action="store_true",
                    help="skip the TFLite model, just report 'abnormal_leaf'")
    ap.add_argument("--min-sharp", type=float, default=0.0,
                    help="reject frames blurrier than this. DEFAULT OFF: read "
                         "the 'focus' number off your own feed first, then set "
                         "it to about half what a good in-focus leaf shows.")
    args = ap.parse_args()
    if args.no_telegram:
        S.TELEGRAM_ENABLED = False

    link = Link(args.port, enabled=not args.no_drive)
    cam = open_camera()
    store = Storage()
    tg = Telegram()
    # Loaded once at startup. If the model or tflite-runtime is missing it
    # prints why and degrades to "abnormal_leaf" rather than crashing --
    # detection still works, you just do not get a disease name.
    clf = None if args.no_classify else DiseaseClassifier()
    gps = GPSReader() if GPSReader else None
    stream_server.start()
    map_server.start()

    print(f"\n  detection view : http://<pi-ip>:{S.STREAM_PORT}")
    print(f"  field map      : http://<pi-ip>:{S.MAP_PORT}")
    print("\n  CH5 HIGH = autonomous.  CH5 LOW = manual RC.")
    print("  ctrl-c stops the robot.\n")

    state = "SCOUTING"
    last_hit = 0.0
    found = 0
    t_state = time.time()
    seen = []          # [(lat, lon)] of pins already dropped

    def is_dup(fx):
        """
        The camera sees the same leaf over many frames, and you will hold it
        there while explaining it. Without this, one plant becomes fifteen
        pins and the map stops meaning anything.
        """
        if not (fx and fx.fix_ok):
            return False
        from telemetry.gps_reader import haversine_m
        return any(haversine_m(fx.lat, fx.lon, a, b) < S.DUP_RADIUS_M
                   for a, b in seen)

    try:
        while True:
            frame = grab(cam)
            if frame is None:
                time.sleep(0.05)
                continue
            small = cv2.resize(frame, (800, 450))
            res = D.detect(small)
            tlm = link.read()
            fix = gps.read() if gps else None

            for e in link.pop_events():
                if e == "ROW_END":
                    state = "ROW_END"
                    t_state = time.time()

            # ---- the decision --------------------------------------------
            # Three independent gates. All must pass.
            #   1. the detector found real vegetation and trusts the result
            #   2. enough of the leaf is abnormal
            #   3. the frame is actually in focus -- a blurred frame cannot be
            #      judged, and an out-of-focus false alarm is worse than silence
            sharp = float(cv2.Laplacian(
                cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            in_focus = args.min_sharp <= 0 or sharp >= args.min_sharp
            hit = (res.ratio >= S.DETECT_RATIO_MIN and res.blobs
                   and res.trusted and in_focus)

            if state == "SCOUTING":
                link.go()
                if hit and (time.time() - last_hit) > args.cooldown \
                        and not is_dup(fix):
                    state = "STOPPING"
                    link.stop()
                    t_state = time.time()

            elif state == "STOPPING":
                link.stop()
                # let the chassis actually come to rest before shooting
                if time.time() - t_state > 1.0:
                    state = "CAPTURING"

            elif state == "CAPTURING":
                link.stop()
                shot = grab(cam)
                if shot is not None:
                    r2 = D.detect(cv2.resize(shot, (800, 450)))
                    ov = D.overlay(cv2.resize(shot, (800, 450)), r2)

                    # Classify only here, never per frame. The robot is
                    # already stopped, so inference time is free -- and the
                    # detector has just told us WHERE the lesion is, so the
                    # model gets a tight crop instead of a wide field shot
                    # that is mostly floor.
                    disease, conf, top = "abnormal_leaf", 1.0, []
                    if clf is not None and clf.ok:
                        sx = shot.shape[1] / 800.0
                        sy = shot.shape[0] / 450.0
                        blobs = [{"x": int(b["x"] * sx), "y": int(b["y"] * sy),
                                  "w": int(b["w"] * sx), "h": int(b["h"] * sy)}
                                 for b in r2.blobs]
                        t0 = time.time()
                        disease, conf, top = clf.predict(shot, blobs)
                        ms = (time.time() - t0) * 1000
                        print(f"      classified in {ms:.0f} ms: "
                              + ", ".join(f"{n} {p:.0%}" for n, p in top))

                    rec = {
                        "disease": disease,
                        "confidence": conf,
                        "top3": top,
                        "severity": r2.severity, "ratio": r2.ratio,
                        "blobs": len(r2.blobs), "trusted": r2.trusted,
                        "sharpness": 0.0, "note": r2.note,
                        "lat": fix.lat if (fix and fix.fix_ok) else None,
                        "lon": fix.lon if (fix and fix.fix_ok) else None,
                        "sats": fix.sats if fix else 0,
                        "hdop": fix.hdop if fix else 99.9,
                        "pass_idx": 0, "odo_cm": 0.0, "t": time.time(),
                    }
                    path, confident = store.save(shot, ov, rec)
                    rec["image"] = path
                    map_server.add_pin(rec)
                    if fix and fix.fix_ok:
                        seen.append((fix.lat, fix.lon))
                    if confident:
                        tg.detection(path, rec)
                    found += 1
                    flag = "" if confident else "  (low confidence -> uncertain/)"
                    print(f"  [{found}] {disease} {conf:.0%}  "
                          f"{r2.severity} {r2.ratio:.1%} -> "
                          f"{os.path.basename(path)}{flag}")
                last_hit = time.time()
                state = "RESUMING"
                t_state = time.time()

            elif state == "RESUMING":
                if time.time() - t_state > 1.0:
                    state = "SCOUTING"
                else:
                    link.stop()

            elif state == "ROW_END":
                link.stop()

            # ---- publish --------------------------------------------------
            wall = (f"L {tlm.left:5.1f}{'*' if tlm.wall_l else ' '}  "
                    f"R {tlm.right:5.1f}{'*' if tlm.wall_r else ' '}")
            gtxt = (f"{fix.sats} sats hdop {fix.hdop} fix {fix.fix_ok}"
                    if fix else "no gps")
            stream_server.publish(
                nav_frame=D.overlay(small, res),
                disease_frame=D.overlay(small, res),
                text=(f"state     {state}\n"
                      f"esp32     {tlm.mode}   link {'ok' if tlm.fresh else 'STALE'}\n"
                      f"sonar     {wall}   error {tlm.err:+5.1f} cm\n"
                      f"leaf      {res.leaf_px:,} px   abnormal {res.ratio:.2%}   "
                      f"blobs {len(res.blobs)}\n"
                      f"severity  {res.severity}   trusted {res.trusted}"
                      f"{'  ' + res.note if res.note else ''}\n"
                      f"greenness d_ref {res.d_ref:+.3f}   "
                      f"(real foliage is +0.15 to +0.30)\n"
                      f"focus     {sharp:.0f}"
                      f"{'' if in_focus else '  TOO BLURRY - ignoring'}\n"
                      f"model     {'loaded' if (clf and clf.ok) else 'NOT loaded'}\n"
                      f"gps       {gtxt}\n"
                      f"found     {found}"))
            if fix:
                map_server.update_bot(lat=fix.lat or None, lon=fix.lon or None,
                                      fix_ok=fix.fix_ok, sats=fix.sats,
                                      state=state)
            time.sleep(0.06)

    except KeyboardInterrupt:
        pass
    finally:
        print("\nstopping...")
        for _ in range(5):
            link.stop()
            time.sleep(0.1)
        tg.session_summary(store.summary())
        tg.drain(timeout=20)
        if gps:
            gps.close()
        if cam[0] == "picam":
            cam[1].stop()
        else:
            cam[1].release()
        print(f"done. {found} detections.")


if __name__ == "__main__":
    main()
