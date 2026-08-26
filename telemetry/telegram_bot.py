"""
Telegram notifier for the farmer's group.

Design constraint: the field has bad signal. A blocking HTTP POST on a 2G
connection can hang for 30 seconds, and if that call sits in the control
loop the robot stops steering. So every send goes onto a queue and a worker
thread drains it. If the network is down the queue simply grows; nothing
upstream ever waits, and nothing is lost until the process exits.

Setup: see config/secrets.example.py.
"""

import queue
import threading
import time

import requests

from config import settings as S

try:
    from config.secrets import TELEGRAM_CHAT_ID, TELEGRAM_TOKEN
except ImportError:
    TELEGRAM_TOKEN = TELEGRAM_CHAT_ID = None

API = "https://api.telegram.org/bot{}/{}"


class Telegram:
    def __init__(self):
        self.enabled = bool(S.TELEGRAM_ENABLED and TELEGRAM_TOKEN
                            and not TELEGRAM_TOKEN.startswith("0000"))
        self.q = queue.Queue()
        self.sent = 0
        self.failed = 0
        self.queued = 0        # total ever enqueued
        if self.enabled:
            threading.Thread(target=self._worker, daemon=True).start()
            print("[telegram] enabled")
        else:
            print("[telegram] disabled (no token in config/secrets.py)")

    # ------------------------------------------------------------------
    def _worker(self):
        while True:
            kind, payload = self.q.get()
            for attempt in range(3):
                try:
                    if kind == "photo":
                        path, caption = payload
                        with open(path, "rb") as fh:
                            r = requests.post(
                                API.format(TELEGRAM_TOKEN, "sendPhoto"),
                                data={"chat_id": TELEGRAM_CHAT_ID,
                                      "caption": caption,
                                      "parse_mode": "HTML"},
                                files={"photo": fh}, timeout=30)
                    elif kind == "location":
                        lat, lon = payload
                        r = requests.post(
                            API.format(TELEGRAM_TOKEN, "sendLocation"),
                            data={"chat_id": TELEGRAM_CHAT_ID,
                                  "latitude": lat, "longitude": lon},
                            timeout=20)
                    else:
                        r = requests.post(
                            API.format(TELEGRAM_TOKEN, "sendMessage"),
                            data={"chat_id": TELEGRAM_CHAT_ID,
                                  "text": payload, "parse_mode": "HTML"},
                            timeout=20)
                    if r.status_code == 200:
                        self.sent += 1
                        break
                    print(f"[telegram] HTTP {r.status_code}: {r.text[:120]}")
                except Exception as e:
                    print(f"[telegram] attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)          # 1s, 2s, 4s
            else:
                self.failed += 1
            self.q.task_done()

    # ------------------------------------------------------------------
    def _enqueue(self, item):
        self.queued += 1
        self.q.put(item)

    def message(self, text):
        if self.enabled:
            self._enqueue(("text", text))

    def detection(self, image_path, record):
        """Send the leaf photo, then a map pin as a separate message."""
        if not self.enabled:
            return
        lat, lon = record.get("lat"), record.get("lon")
        sev = record.get("severity", "?")
        emoji = {"trace": "\U0001F7E1", "mild": "\U0001F7E0",
                 "moderate": "\U0001F534", "severe": "\u26A0\uFE0F"}.get(sev, "\U0001F50D")

        caption = (
            f"{emoji} <b>{record.get('disease','unknown').replace('_',' ')}</b>\n"
            f"Severity: {sev} ({record.get('ratio',0):.1%} of leaf)\n"
            f"Confidence: {record.get('confidence',0):.0%}\n"
            f"Time: {time.strftime('%H:%M:%S')}"
        )
        if lat and lon:
            caption += (f"\nLocation: {lat:.6f}, {lon:.6f}"
                        f"\nhttps://maps.google.com/?q={lat},{lon}")
        else:
            caption += "\nLocation: no GPS fix"

        self._enqueue(("photo", (image_path, caption)))
        if lat and lon:
            self._enqueue(("location", (lat, lon)))

    def session_summary(self, storage_summary, extra=""):
        self.message(
            f"\U0001F916 <b>ACES run finished</b>\n"
            f"Detections: {storage_summary['detections']}\n"
            f"Date: {storage_summary['day']}\n{extra}")

    def drain(self, timeout=60):
        """
        Block until every enqueued message has actually finished sending.

        Waiting on `q.empty()` is NOT enough and was a real bug: the worker
        removes an item from the queue BEFORE the HTTP request completes, so
        the queue reads empty while the POST is still in flight. The script
        then exits, the daemon thread is killed mid-request, and nothing ever
        arrives. Wait on the completion counters instead.
        """
        if not self.enabled:
            return
        t0 = time.time()
        while (self.sent + self.failed) < self.queued \
                and (time.time() - t0) < timeout:
            time.sleep(0.2)
        pending = self.queued - self.sent - self.failed
        print(f"[telegram] sent={self.sent} failed={self.failed} "
              f"pending={pending}")
        if pending:
            print("[telegram] still sending when we gave up - slow network?")
