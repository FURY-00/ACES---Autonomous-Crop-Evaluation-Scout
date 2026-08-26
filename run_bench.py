"""
Bench runner: the complete perception + reporting pipeline, no motors.

Run this on a table with the Pi camera pointed at a leaf. It exercises
everything the field run does except driving, so you can prove the detector,
the classifier, the SD-card layout, the Telegram group and the map all work
before you ever put the robot in a field.

    python run_bench.py                 # live camera
    python run_bench.py --dir testset/diseased    # replay a folder
    python run_bench.py --no-telegram

Open http://<pi-ip>:8080  (masks)  and  http://<pi-ip>:8081  (map).
"""

import argparse
import os
import time

import cv2

from config import settings as S
from perception import detector as D
from perception.classifier import DiseaseClassifier
from telemetry import map_server, stream_server
from telemetry.gps_reader import GPSReader
from telemetry.sheets import Sheets
from telemetry.storage import Storage
from telemetry.telegram_bot import Telegram


def build_record(res, disease, conf, top, fix, sharp, note=""):
    return {
        "disease": disease, "confidence": conf, "top3": top,
        "severity": res.severity, "ratio": res.ratio,
        "blobs": len(res.blobs), "sharpness": sharp,
        "trusted": res.trusted, "note": note or res.note,
        "lat": fix.lat if fix.fix_ok else None,
        "lon": fix.lon if fix.fix_ok else None,
        "sats": fix.sats, "hdop": fix.hdop,
        "pass_idx": 0, "odo_cm": 0.0, "t": time.time(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="replay images from a folder instead of the camera")
    ap.add_argument("--no-telegram", action="store_true")
    ap.add_argument("--interval", type=float, default=3.0,
                    help="seconds between accepted detections")
    args = ap.parse_args()

    if args.no_telegram:
        S.TELEGRAM_ENABLED = False

    store = Storage()
    tg = Telegram()
    sheets = Sheets()
    clf = DiseaseClassifier()
    gps = GPSReader()
    stream_server.start()
    map_server.start()

    print(f"\nmask stream : http://<pi-ip>:{S.STREAM_PORT}")
    print(f"field map   : http://<pi-ip>:{S.MAP_PORT}")
    print("ctrl-c to stop\n")

    paths = None
    cap = None
    if args.dir:
        paths = [os.path.join(args.dir, f) for f in sorted(os.listdir(args.dir))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"replaying {len(paths)} images from {args.dir}")
    else:
        try:
            from picamera2 import Picamera2
            pc = Picamera2()
            pc.configure(pc.create_preview_configuration(
                main={"size": (S.PREVIEW_W, S.PREVIEW_H), "format": "RGB888"}))
            pc.start()
            time.sleep(2)
            cap = ("picam", pc)
        except ImportError:
            cap = ("cv2", cv2.VideoCapture(0))

    i, last_hit = 0, 0.0
    try:
        while True:
            if paths:
                if i >= len(paths):
                    break
                frame = cv2.imread(paths[i])
                label = os.path.basename(paths[i])
                i += 1
                if frame is None:
                    continue
                frame = cv2.resize(frame, None,
                                   fx=min(1.0, 900 / max(frame.shape[:2])),
                                   fy=min(1.0, 900 / max(frame.shape[:2])))
            else:
                kind, dev = cap
                if kind == "picam":
                    frame = dev.capture_array()       # RGB888 array is already BGR
                else:
                    ok, frame = dev.read()
                    if not ok:
                        continue
                label = "live"

            res = D.detect(frame)
            ov = D.overlay(frame, res)
            fix = gps.read()

            stream_server.publish(
                nav_frame=ov, disease_frame=ov,
                text=(f"source   {label}\n"
                      f"severity {res.severity}   ratio {res.ratio:.2%}\n"
                      f"blobs    {len(res.blobs)}   leaf px {res.leaf_px:,}\n"
                      f"trusted  {res.trusted}   {res.note}\n"
                      f"gps      {fix.sats} sats  hdop {fix.hdop}  "
                      f"fix_ok {fix.fix_ok}\n"
                      f"saved    {store.count}"))
            map_server.update_bot(lat=fix.lat or None, lon=fix.lon or None,
                                  fix_ok=fix.fix_ok, sats=fix.sats,
                                  state="BENCH")

            hit = res.ratio >= S.DETECT_RATIO_MIN and res.blobs
            if hit and (time.time() - last_hit) > args.interval:
                last_hit = time.time()
                disease, conf, top = clf.predict(frame, res.blobs)
                sharp = float(cv2.Laplacian(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                rec = build_record(res, disease, conf, top, fix, sharp)

                if store.is_duplicate(rec["lat"], rec["lon"], disease):
                    print(f"  duplicate within {S.DUP_RADIUS_M} m, skipped")
                else:
                    path, confident = store.save(frame, ov, rec)
                    rec["image"] = path
                    store.remember(rec["lat"], rec["lon"], disease)
                    map_server.add_pin(rec)
                    sheets.append(rec)
                    if confident:
                        tg.detection(path, rec)
                    print(f"  [{store.count}] {disease} {conf:.0%} "
                          f"{res.severity} {res.ratio:.1%} "
                          f"-> {os.path.basename(path)}"
                          f"{'' if confident else '  (uncertain, not sent)'}")

            if not paths:
                time.sleep(0.15)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nshutting down...")
        tg.session_summary(store.summary())
        tg.drain()
        sheets.drain()
        gps.close()
        if cap and cap[0] == "cv2":
            cap[1].release()
        elif cap:
            cap[1].stop()
        print(store.summary())


if __name__ == "__main__":
    main()
