"""
Storage: SD card layout, CSV log, and duplicate suppression.

Layout
------
    data/
      2026-08-15/
        images/      confident detections, renamed to the predicted disease
        uncertain/   low classifier confidence or untrusted detection
        raw/         the original frame, always kept
      log.csv        one row per detection, append-only

Why the raw copy is always kept: when you later discover a threshold was
wrong, the annotated overlay is useless for re-analysis. The raw frame lets
you re-run the whole pipeline on a season's worth of data offline.
"""

import csv
import json
import os
import time

import cv2

from config import settings as S
from telemetry.gps_reader import haversine_m

CSV_HEADER = ["timestamp", "date", "lat", "lon", "sats", "hdop",
              "disease", "confidence", "severity", "abnormal_ratio",
              "blobs", "sharpness", "trusted", "pass_idx", "odo_cm",
              "image", "note"]


class Storage:
    def __init__(self, root=S.SD_ROOT):
        self.root = root
        self.day = time.strftime("%Y-%m-%d")
        self.dirs = {k: os.path.join(root, self.day, k)
                     for k in ("images", "uncertain", "raw")}
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
        self.csv_path = os.path.join(root, "log.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as fh:
                csv.writer(fh).writerow(CSV_HEADER)
        self._seen = []          # [(lat, lon, disease)] for dup suppression
        self.count = 0

    # ------------------------------------------------------------------
    def is_duplicate(self, lat, lon, disease):
        """
        The bot sees the same lesion on both passes of a row, and often over
        several consecutive frames. Without this, one sick plant becomes
        thirty pins on the farmer's map and the map stops being useful.
        """
        if not lat and not lon:
            return False
        for plat, plon, pdis in self._seen:
            if pdis == disease and haversine_m(lat, lon, plat, plon) < S.DUP_RADIUS_M:
                return True
        return False

    def remember(self, lat, lon, disease):
        self._seen.append((lat, lon, disease))

    # ------------------------------------------------------------------
    def save(self, raw_bgr, overlay_bgr, record: dict):
        """
        record needs: disease, confidence, severity, ratio, blobs, sharpness,
        trusted, pass_idx, odo_cm, lat, lon, sats, hdop, note
        Returns the path of the saved primary image.
        """
        ts = time.strftime("%Y%m%d_%H%M%S")
        conf = record.get("confidence", 0.0)
        trusted = record.get("trusted", True)
        confident = trusted and conf >= S.CONF_UNCERTAIN

        disease = (record.get("disease") or "unknown").replace(" ", "_")
        stem = f"{disease}_{ts}_{self.count:04d}"
        bucket = "images" if confident else "uncertain"

        primary = os.path.join(self.dirs[bucket], stem + ".jpg")
        cv2.imwrite(primary, overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(os.path.join(self.dirs["raw"], stem + "_raw.jpg"),
                    raw_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(primary.replace(".jpg", ".json"), "w") as fh:
            json.dump(record, fh, indent=2)

        with open(self.csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow([
                time.time(), self.day,
                record.get("lat", ""), record.get("lon", ""),
                record.get("sats", ""), record.get("hdop", ""),
                disease, round(conf, 3), record.get("severity", ""),
                round(record.get("ratio", 0.0), 4), record.get("blobs", 0),
                round(record.get("sharpness", 0.0), 1), trusted,
                record.get("pass_idx", ""), round(record.get("odo_cm", 0), 1),
                os.path.relpath(primary, self.root), record.get("note", ""),
            ])
        self.count += 1
        return primary, confident

    def summary(self):
        return {"detections": self.count, "day": self.day,
                "csv": self.csv_path}
