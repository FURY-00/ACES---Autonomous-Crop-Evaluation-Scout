"""
Line protocol between the Raspberry Pi (brain) and the ESP32 (reflexes).

Pi -> ESP32                                     meaning
  $V,<off_cm>,<head_deg>,<conf>                 vision update, sent at ~15 Hz
  $M,<IDLE|DRIVE|TURN|BACK|ESTOP>               mode request
  $S,<cm_s>                                     cruise speed
  $L,<cm>                                       lateral setpoint (dodge), |cm| <= 5
  $T,<deg>                                      turn in place, signed
  $B,<cm>                                       reverse this many cm then hold
  $Z                                            zero the odometer

ESP32 -> Pi (20 Hz)
  #T,<ms>,<state>,<odo_cm>,<lat_cm>,<head_deg>,<front_cm>,<left_cm>,<right_cm>,<flags>
  #E,<event text>                               one-shot events (TURN_DONE, ...)

Everything is ASCII, newline terminated. If you can read the traffic with a
plain serial monitor you can debug the robot in a field with no laptop.
"""

import threading
import time
from dataclasses import dataclass, field

import serial

from config import settings as config


@dataclass
class Telemetry:
    t_ms: int = 0
    state: str = "IDLE"
    odo_cm: float = 0.0        # signed distance travelled since last $Z
    lat_cm: float = 0.0        # ESP32's dead-reckoned cross-track estimate
    head_deg: float = 0.0
    front_cm: float = 999.0    # forward ultrasonic, 999 = no echo
    left_cm: float = 999.0     # left ultrasonic -> distance to the left crop wall
    right_cm: float = 999.0    # right ultrasonic -> distance to the right crop wall
    flags: int = 0             # bit0 IR-left row present, bit1 IR-right row present
    stamp: float = field(default_factory=time.time)

    @property
    def ir_left(self) -> bool:
        return bool(self.flags & 0x01)

    @property
    def ir_right(self) -> bool:
        return bool(self.flags & 0x02)

    @property
    def fresh(self) -> bool:
        return (time.time() - self.stamp) < config.LINK_TIMEOUT_S


class SerialLink:
    def __init__(self, port=config.ESP32_PORT, baud=config.ESP32_BAUD):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.tlm = Telemetry()
        self.events = []
        self._lock = threading.Lock()
        self._run = True
        self._rx = threading.Thread(target=self._reader, daemon=True)
        self._rx.start()

    # ---------------------------------------------------------------- rx
    def _reader(self):
        while self._run:
            try:
                line = self.ser.readline().decode("ascii", "ignore").strip()
            except Exception:
                time.sleep(0.05)
                continue
            if not line:
                continue
            if line.startswith("#T,"):
                p = line[3:].split(",")
                if len(p) == 9:
                    try:
                        with self._lock:
                            self.tlm = Telemetry(
                                t_ms=int(p[0]), state=p[1], odo_cm=float(p[2]),
                                lat_cm=float(p[3]), head_deg=float(p[4]),
                                front_cm=float(p[5]), left_cm=float(p[6]),
                                right_cm=float(p[7]), flags=int(p[8]),
                            )
                    except ValueError:
                        pass
            elif line.startswith("#E,"):
                with self._lock:
                    self.events.append((time.time(), line[3:]))

    def read(self) -> Telemetry:
        with self._lock:
            return self.tlm

    def pop_events(self):
        with self._lock:
            e, self.events = self.events, []
        return e

    def wait_event(self, name, timeout=20.0):
        """Block until the ESP32 reports `name` (e.g. TURN_DONE)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            for _, txt in self.pop_events():
                if txt.startswith(name):
                    return True
            time.sleep(0.02)
        return False

    # ---------------------------------------------------------------- tx
    def _send(self, s: str):
        self.ser.write((s + "\n").encode("ascii"))

    def vision(self, off_cm, head_deg, conf):
        self._send(f"$V,{off_cm:.2f},{head_deg:.2f},{conf:.2f}")

    def mode(self, m):        self._send(f"$M,{m}")
    def speed(self, cms):     self._send(f"$S,{cms:.1f}")
    def lateral(self, cm):
        cm = max(-config.LAT_LIMIT_CM, min(config.LAT_LIMIT_CM, cm))  # belt & braces
        self._send(f"$L,{cm:.2f}")
    def turn(self, deg):      self._send(f"$T,{deg:.1f}")
    def back(self, cm):       self._send(f"$B,{cm:.1f}")
    def zero_odo(self):       self._send("$Z")

    def close(self):
        self._run = False
        try:
            self.mode("IDLE")
        finally:
            self.ser.close()
