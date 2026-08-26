"""
Obstacle policy for a 32 cm robot in a 40-45 cm gap.

Read this comment before you read the code, because the geometry decides
the whole design:

    worst-case gap            40.0 cm
    robot width               32.0 cm
    total slack                8.0 cm   ->  4.0 cm per side
    minus safety margin        2.0 cm
    usable dodge              ~2.0 cm   (config.LAT_LIMIT_CM)

Two cm is not an avoidance manoeuvre. It is a nudge. So this robot does
NOT drive around obstacles: it slows, stops, waits (birds, people and
dogs move), and if the thing is still there it hands the decision back to
the mission layer. Sideways dodging is left in the code only for the case
where the side sonars actively prove there is room, which in a 40 cm gap
will almost never happen.

The three sonars earn their keep in a different way. FRONT finds
obstacles. LEFT and RIGHT measure the distance to the crop walls, which
gives a second, independent row-centring signal that keeps working when
the camera is blinded by dust, low sun or a shadow.
"""

import time
from dataclasses import dataclass

from config import settings as config


def _valid(d):
    return config.SONAR_MIN_CM <= d <= config.SIDE_VALID_MAX_CM


@dataclass
class Decision:
    speed_cms: float
    lateral_cm: float          # setpoint for the row-centre offset
    stop: bool = False
    escalate: bool = False     # blocked long enough that the mission must act
    reason: str = ""


def sonar_offset(left_cm, right_cm):
    """
    Cross-track offset from the two side sonars, in cm, + = right of centre.

    Both walls visible: the difference is the offset directly, and it is
    immune to lighting. Only one wall visible: fall back to holding the
    nominal standoff from that wall. Neither: return None.
    """
    l_ok, r_ok = _valid(left_cm), _valid(right_cm)
    if l_ok and r_ok:
        return (left_cm - right_cm) / 2.0, 0.9
    if l_ok:
        return left_cm - config.SIDE_NOMINAL_CM, 0.5
    if r_ok:
        return config.SIDE_NOMINAL_CM - right_cm, 0.5
    return None, 0.0


class ObstaclePolicy:
    def __init__(self):
        self._hits = 0
        self._clear = 0
        self._committed = 0.0
        self._blocked_since = None

    def update(self, tlm, est) -> Decision:
        front = tlm.front_cm

        # ---- 1. crop-wall guard, runs before anything else ---------------
        # If a side sonar says a plant is closer than the safety margin, the
        # only correct action is to move away from it, whatever else is going on.
        for d, sign, side in ((tlm.left_cm, +1.0, "left"),
                              (tlm.right_cm, -1.0, "right")):
            if _valid(d) and d < config.LAT_SAFETY_MARGIN_CM:
                return Decision(config.SLOW_SPEED_CMS,
                                sign * config.LAT_LIMIT_CM, False, False,
                                f"crop wall {d:.1f}cm on the {side}, easing off")

        # ---- 2. de-bounce the front sonar ---------------------------------
        # Foliage scatters the cone and produces phantom short echoes.
        if config.SONAR_MIN_CM <= front < config.OBST_SLOW_CM:
            self._hits += 1
            self._clear = 0
        else:
            self._clear += 1
            if self._clear >= config.OBST_CLEAR_HITS:
                self._hits = 0
                self._committed = 0.0
                self._blocked_since = None

        if self._hits < config.OBST_CONFIRM_HITS:
            return Decision(config.CRUISE_SPEED_CMS, 0.0, False, False, "clear")

        # ---- 3. obstacle believed ------------------------------------------
        if front >= config.OBST_STOP_CM:
            # Still some distance: creep and keep looking.
            return Decision(config.SLOW_SPEED_CMS, self._committed, False, False,
                            f"obstacle {front:.0f}cm ahead, creeping")

        # ---- 4. stopped in front of it --------------------------------------
        if self._blocked_since is None:
            self._blocked_since = time.time()
        waited = time.time() - self._blocked_since

        # Can the side sonars actually prove there is room? Rarely, but check.
        room_l = min(config.LAT_LIMIT_CM,
                     max(0.0, tlm.left_cm - config.LAT_SAFETY_MARGIN_CM)) \
            if _valid(tlm.left_cm) else 0.0
        room_r = min(config.LAT_LIMIT_CM,
                     max(0.0, tlm.right_cm - config.LAT_SAFETY_MARGIN_CM)) \
            if _valid(tlm.right_cm) else 0.0

        if max(room_l, room_r) >= 1.5 and est.conf >= config.CONF_MIN_DRIVE \
                and self._committed == 0.0:
            self._committed = -room_l if room_l >= room_r else room_r
            return Decision(config.SLOW_SPEED_CMS, self._committed, False, False,
                            f"nudging {self._committed:+.1f}cm (all the room there is)")

        return Decision(0.0, 0.0, True, waited > config.OBST_WAIT_S,
                        f"blocked at {front:.0f}cm for {waited:.0f}s"
                        + (" - escalating" if waited > config.OBST_WAIT_S else ""))

    def reset(self):
        self._hits = self._clear = 0
        self._committed = 0.0
        self._blocked_since = None
