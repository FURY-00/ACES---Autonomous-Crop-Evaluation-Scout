"""
Google Sheets appender.

Same non-blocking design as the Telegram notifier, plus one addition: rows
that fail to upload are written to data/sheets_pending.jsonl and retried on
the next run. A field robot with intermittent signal must not silently lose
records, and the CSV on the SD card is the real source of truth anyway —
Sheets is a convenience view for people who are not going to SSH into a Pi.

Setup: see config/secrets.example.py.
"""

import json
import os
import queue
import threading
import time

from config import settings as S

try:
    from config.secrets import GOOGLE_CREDENTIALS
except ImportError:
    GOOGLE_CREDENTIALS = None

HEADER = ["timestamp", "date", "time", "latitude", "longitude", "disease",
          "confidence", "severity", "abnormal_ratio", "trusted", "image",
          "maps_link"]


class Sheets:
    def __init__(self):
        self.enabled = False
        self.q = queue.Queue()
        self.pending = os.path.join(S.SD_ROOT, "sheets_pending.jsonl")
        self.ws = None
        if not S.SHEETS_ENABLED:
            print("[sheets] disabled in settings")
            return
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            scopes = ["https://www.googleapis.com/auth/spreadsheets",
                      "https://www.googleapis.com/auth/drive"]
            creds = Credentials.from_service_account_file(
                GOOGLE_CREDENTIALS, scopes=scopes)
            gc = gspread.authorize(creds)
            try:
                sh = gc.open(S.SHEET_NAME)
            except Exception:
                sh = gc.create(S.SHEET_NAME)
            self.ws = sh.sheet1
            if not self.ws.get_all_values():
                self.ws.append_row(HEADER)
            self.enabled = True
            threading.Thread(target=self._worker, daemon=True).start()
            self._flush_pending()
            print(f"[sheets] connected to '{S.SHEET_NAME}'")
        except Exception as e:
            print(f"[sheets] disabled: {e}")
            print("[sheets] did you share the sheet with the service account "
                  "client_email from the JSON key?")

    # ------------------------------------------------------------------
    def _row(self, rec):
        lat, lon = rec.get("lat"), rec.get("lon")
        return [
            round(time.time(), 1), time.strftime("%Y-%m-%d"),
            time.strftime("%H:%M:%S"),
            lat or "", lon or "",
            rec.get("disease", "unknown"),
            round(rec.get("confidence", 0), 3),
            rec.get("severity", ""),
            round(rec.get("ratio", 0), 4),
            rec.get("trusted", True),
            os.path.basename(rec.get("image", "")),
            f"https://maps.google.com/?q={lat},{lon}" if lat and lon else "",
        ]

    def _worker(self):
        while True:
            rec = self.q.get()
            row = self._row(rec)
            try:
                self.ws.append_row(row, value_input_option="USER_ENTERED")
            except Exception as e:
                print(f"[sheets] append failed, buffering: {e}")
                with open(self.pending, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            self.q.task_done()

    def _flush_pending(self):
        if not os.path.exists(self.pending):
            return
        with open(self.pending) as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        if not rows:
            return
        try:
            self.ws.append_rows([self._row(r) for r in rows],
                                value_input_option="USER_ENTERED")
            os.remove(self.pending)
            print(f"[sheets] flushed {len(rows)} buffered rows")
        except Exception as e:
            print(f"[sheets] could not flush buffer: {e}")

    # ------------------------------------------------------------------
    def append(self, record):
        if self.enabled:
            self.q.put(record)
        elif S.SHEETS_ENABLED:
            with open(self.pending, "a") as fh:
                fh.write(json.dumps(record) + "\n")

    def drain(self, timeout=30):
        t0 = time.time()
        while self.enabled and not self.q.empty() and time.time() - t0 < timeout:
            time.sleep(0.5)
