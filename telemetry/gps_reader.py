"""
NEO-M8N GPS reader.

Runs in its own thread so a slow or silent GPS never stalls the control loop.

The important part is `fix_ok`. A GPS module reports a position long before
that position is any good — indoors it will happily give you coordinates
400 m away. Every detection pin on the farmer's map is only as trustworthy
as this gate, so we require satellite count AND HDOP, not just "got a line".
"""

import threading
import time
from dataclasses import dataclass

import pynmea2
import serial

from config import settings as S


@dataclass
class Fix:
    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0
    sats: int = 0
    hdop: float = 99.9
    quality: int = 0
    stamp: float = 0.0

    @property
    def fix_ok(self) -> bool:
        return (self.quality > 0
                and self.sats >= S.GPS_MIN_SATS
                and self.hdop <= S.GPS_MAX_HDOP
                and (time.time() - self.stamp) < 5.0)

    def as_dict(self):
        return {"lat": self.lat, "lon": self.lon, "sats": self.sats,
                "hdop": self.hdop, "fix_ok": self.fix_ok}


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres. Used for duplicate suppression."""
    from math import asin, cos, radians, sin, sqrt
    r = 6371000.0
    p1, p2 = radians(lat1), radians(lat2)
    dp, dl = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * r * asin(sqrt(a))


class GPSReader:
    def __init__(self, port=S.GPS_PORT, baud=S.GPS_BAUD):
        self.fix = Fix()
        self._lock = threading.Lock()
        self._run = True
        try:
            self.ser = serial.Serial(port, baud, timeout=1.0)
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
            self.available = True
        except Exception as e:
            print(f"[gps] not available: {e}")
            self.available = False

    def _loop(self):
        while self._run:
            try:
                line = self.ser.readline().decode("ascii", "ignore").strip()
            except Exception:
                time.sleep(0.2)
                continue
            if not line.startswith("$"):
                continue
            try:
                msg = pynmea2.parse(line)
            except pynmea2.ParseError:
                continue
            if isinstance(msg, pynmea2.types.talker.GGA):
                if msg.latitude and msg.longitude:
                    with self._lock:
                        self.fix = Fix(
                            lat=float(msg.latitude), lon=float(msg.longitude),
                            alt_m=float(msg.altitude or 0),
                            sats=int(msg.num_sats or 0),
                            hdop=float(msg.horizontal_dil or 99.9),
                            quality=int(msg.gps_qual or 0),
                            stamp=time.time())

    def read(self) -> Fix:
        with self._lock:
            return self.fix

    def wait_for_fix(self, timeout=120):
        """Call this once at startup. Cold start outdoors takes 30-90 s."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            f = self.read()
            if f.fix_ok:
                print(f"[gps] fix: {f.lat:.6f},{f.lon:.6f} "
                      f"sats={f.sats} hdop={f.hdop}")
                return True
            print(f"[gps] waiting... sats={f.sats} hdop={f.hdop}", end="\r")
            time.sleep(1.0)
        print("\n[gps] no usable fix - detections will be logged without "
              "coordinates")
        return False

    def close(self):
        self._run = False
        if self.available:
            self.ser.close()
