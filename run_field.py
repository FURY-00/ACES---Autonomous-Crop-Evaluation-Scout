"""
Field runner: the whole robot.

    FOLLOW ──obstacle──> BLOCKED ──won't clear──> ESCALATE (reverse out)
       │
       ├─abnormal leaf──> CAPTURE ──blurry──> REVERSE ──> CAPTURE ──> FOLLOW
       │
       └─row end──> headland ──> TURN ──> FOLLOW (pass 2) ──> DONE

The Pi decides WHAT to do. The ESP32 decides HOW to move.
The Pi never writes a PWM value; the ESP32 never opens a camera.

Before running this you must have completed, in order:
    tools/calibrate_row.py        (PX_PER_CM_* in settings are real)
    run_bench.py                  (detector + classifier + outputs all work)
    a wheels-off-the-ground run   (steering responds, nothing runs away)
"""

import argparse
import time

import cv2

from config import settings as S
from navigation.obstacle_policy import ObstaclePolicy
from navigation.row_follower import RowFollower, fuse_sonar
from navigation.serial_link import SerialLink
from perception import detector as D
from perception.classifier import DiseaseClassifier
from perception.disease_camera import DiseaseCamera
from telemetry import map_server, stream_server
from telemetry.gps_reader import GPSReader
from telemetry.sheets import Sheets
from telemetry.storage import Storage
from telemetry.telegram_bot import Telegram


class Mission:
    def __init__(self, dry_run=False):
        self.dry = dry_run
        self.link = SerialLink()
        self.eyes = RowFollower()
        self.leafcam = DiseaseCamera()
        self.policy = ObstaclePolicy()
        self.clf = DiseaseClassifier()
        self.gps = GPSReader()
        self.store = Storage()
        self.tg = Telegram()
        self.sheets = Sheets()

        self.state = "FOLLOW"
        self.pass_idx = 0
        self.rowend_votes = 0
        self.detect_odo = None
        self.note = ""

        stream_server.start()
        map_server.start()

    # ------------------------------------------------------------ helpers
    def _rowend(self, tlm):
        """Three independent votes, two must agree. Any single one lies."""
        v_vision = self.eyes.side_vegetation() < S.ROWEND_GREEN_FRAC
        v_odo = abs(tlm.odo_cm) > (S.ROW_LENGTH_CM - S.ROW_LENGTH_TOL_CM)
        v_ir = not (tlm.ir_left or tlm.ir_right)
        self.rowend_votes = self.rowend_votes + 1 \
            if sum((v_vision, v_odo, v_ir)) >= 2 else 0
        return self.rowend_votes >= S.ROWEND_CONFIRM_FRAMES

    def _readout(self, est, tlm, fix):
        return (f"state   {self.state}   pass {self.pass_idx+1}/{S.PASSES_PER_ROW}\n"
                f"offset  {est.offset_cm:+6.1f} cm  (limit {S.LAT_LIMIT_CM:.1f})"
                f"   heading {est.heading_deg:+6.1f}   conf {est.conf:.2f}\n"
                f"sonar   F {tlm.front_cm:5.1f}  L {tlm.left_cm:5.1f}  "
                f"R {tlm.right_cm:5.1f} cm\n"
                f"odo     {tlm.odo_cm:7.1f} cm   esp {tlm.state}\n"
                f"gps     {fix.sats} sats  hdop {fix.hdop}  fix {fix.fix_ok}\n"
                f"found   {self.store.count}   {self.note}")

    # ------------------------------------------------------------ actions
    def handle_detection(self, frame, res, tlm, fix, sharp, retried=False):
        disease, conf, top = self.clf.predict(frame, res.blobs)
        rec = {
            "disease": disease, "confidence": conf, "top3": top,
            "severity": res.severity, "ratio": res.ratio,
            "blobs": len(res.blobs), "sharpness": sharp,
            "trusted": res.trusted, "note": res.note + (" retry" if retried else ""),
            "lat": fix.lat if fix.fix_ok else None,
            "lon": fix.lon if fix.fix_ok else None,
            "sats": fix.sats, "hdop": fix.hdop,
            "pass_idx": self.pass_idx, "odo_cm": tlm.odo_cm, "t": time.time(),
        }
        if self.store.is_duplicate(rec["lat"], rec["lon"], disease):
            self.note = "duplicate, skipped"
            return
        ov = D.overlay(frame, res)
        path, confident = self.store.save(frame, ov, rec)
        rec["image"] = path
        self.store.remember(rec["lat"], rec["lon"], disease)
        map_server.add_pin(rec)
        self.sheets.append(rec)
        if confident:
            self.tg.detection(path, rec)
        self.note = (f"{disease} {conf:.0%} {res.severity}"
                     + ("" if confident else " (uncertain)"))

    def do_capture(self, tlm, fix):
        """Stop, shoot full-res, and reverse-and-retry if it came out blurry."""
        self.state = "CAPTURE"
        if not self.dry:
            self.link.mode("IDLE")
        time.sleep(0.35)

        frame = self.leafcam.grab(hires=True)
        if frame is None:
            self.state = "FOLLOW"
            return
        res = D.detect(frame)
        sharp = float(cv2.Laplacian(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

        if sharp < S.SHARPNESS_MIN and self.detect_odo is not None and not self.dry:
            # The 45-deg camera means the leaf is already behind the lens by
            # the time we stop. Odometry over 30 cm is excellent, so reverse
            # by a measured amount rather than a blind guess.
            back = (tlm.odo_cm - self.detect_odo) \
                + S.DISEASE_CAM_LEAD_CM + S.BACKUP_EXTRA_CM
            back = max(5.0, min(60.0, back))
            self.state = "REVERSE"
            self.note = f"blurry ({sharp:.0f}), reversing {back:.0f} cm"
            self.link.back(back)
            self.link.wait_event("BACK_DONE", timeout=15)
            time.sleep(0.5)                       # let the chassis settle
            frame2 = self.leafcam.grab(hires=True)
            if frame2 is not None:
                s2 = float(cv2.Laplacian(
                    cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                if s2 > sharp:
                    frame, res, sharp = frame2, D.detect(frame2), s2

        if res.ratio >= S.DETECT_RATIO_MIN and res.blobs:
            self.handle_detection(frame, res, tlm, fix, sharp,
                                  retried=(self.state == "REVERSE"))
        self.detect_odo = None
        self.state = "FOLLOW"
        if not self.dry:
            self.link.mode("DRIVE")
            self.link.speed(S.CRUISE_SPEED_CMS)

    def do_turn(self):
        # A 32 cm robot cannot spin inside a 42 cm gap without clipping the
        # last plants. Clear the headland first.
        self.state = "TURN"
        self.note = "clearing headland"
        if not self.dry:
            self.link.speed(S.SLOW_SPEED_CMS)
            time.sleep(S.HEADLAND_CLEAR_CM / max(S.SLOW_SPEED_CMS, 1.0))
            self.link.mode("IDLE")
            time.sleep(0.3)
            self.link.turn(S.TURN_DEGREES)
            self.link.wait_event("TURN_DONE", timeout=30)
            self.link.zero_odo()
        self.policy.reset()
        self.rowend_votes = 0
        self.pass_idx += 1
        self.state = "FOLLOW" if self.pass_idx < S.PASSES_PER_ROW else "DONE"
        if self.state == "FOLLOW" and not self.dry:
            self.link.mode("DRIVE")
            self.link.speed(S.CRUISE_SPEED_CMS)
        self.note = f"pass {self.pass_idx+1}"

    # ------------------------------------------------------------ main
    def run(self):
        self.gps.wait_for_fix(timeout=90)
        self.tg.message("\U0001F916 ACES starting a run.")
        if not self.dry:
            self.link.zero_odo()
            self.link.speed(S.CRUISE_SPEED_CMS)
            self.link.mode("DRIVE")

        while self.state != "DONE":
            est = self.eyes.update()
            tlm = self.link.read()
            est = fuse_sonar(est, tlm)          # side sonars refine centring
            fix = self.gps.read()

            self.link.vision(est.offset_cm, est.heading_deg, est.conf)

            if self.eyes.blind_for() > S.CONF_BLIND_TIMEOUT_S:
                self.state = "BLOCKED"
                self.link.mode("IDLE")
                self.note = "row lost - vision blind"

            elif self.state == "FOLLOW":
                d = self.policy.update(tlm, est)
                self.link.speed(d.speed_cms)
                self.link.lateral(d.lateral_cm)
                self.note = d.reason
                if d.stop:
                    self.state = "ESCALATE" if d.escalate else "BLOCKED"
                    self.link.mode("IDLE")
                else:
                    found, roi, _ = self.leafcam.scan()
                    if found and self.detect_odo is None:
                        self.detect_odo = tlm.odo_cm
                        self.link.speed(S.CAPTURE_SPEED_CMS)
                        self.do_capture(tlm, fix)
                    elif self._rowend(tlm):
                        self.do_turn()

            elif self.state == "BLOCKED":
                d = self.policy.update(tlm, est)
                self.note = d.reason
                if not d.stop and est.valid:
                    self.state = "FOLLOW"
                    self.link.mode("DRIVE")
                elif d.escalate:
                    self.state = "ESCALATE"

            elif self.state == "ESCALATE":
                # Nothing in a 40 cm gap can be driven around. Back out and
                # tell a human rather than sitting in the row until the battery dies.
                self.note = "obstacle will not clear - reversing out"
                self.tg.message("\u26A0\uFE0F ACES is blocked in the row and "
                                "is reversing out. Needs a look.")
                self.link.back(min(abs(tlm.odo_cm), 150.0))
                self.link.wait_event("BACK_DONE", timeout=40)
                self.state = "DONE"

            map_server.update_bot(
                lat=fix.lat or None, lon=fix.lon or None, fix_ok=fix.fix_ok,
                sats=fix.sats, heading=est.heading_deg, state=self.state,
                pass_idx=self.pass_idx, odo_cm=tlm.odo_cm)
            stream_server.publish(self.eyes.debug, self.leafcam.overlay,
                                  self._readout(est, tlm, fix))
            time.sleep(1.0 / S.NAV_FPS)

        self.shutdown()

    def shutdown(self):
        try:
            self.link.mode("IDLE")
            time.sleep(0.2)
        finally:
            self.tg.session_summary(self.store.summary())
            self.tg.drain()
            self.sheets.drain()
            self.eyes.release()
            self.leafcam.close()
            self.gps.close()
            self.link.close()
            print(self.store.summary())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="perception and reporting only, never commands the motors")
    a = ap.parse_args()
    m = Mission(dry_run=a.dry_run)
    try:
        m.run()
    except KeyboardInterrupt:
        m.shutdown()
