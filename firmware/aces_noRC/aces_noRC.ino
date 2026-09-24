/*  ACES autonomous firmware -- NO RECEIVER.
 *
 *  Everything comes from the Pi over USB. No FlySky, no PPM, no channel pins.
 *
 *  Pi -> ESP32
 *    $GO          drive forward at cruise
 *    $STOP        stop
 *    $S,<steer>   steering, -100..100, + = turn RIGHT
 *    $ROW_END     no walls left -- stop and stay stopped
 *    $T,<dir>     spin in place: +1 = right, -1 = left. Runs until the Pi
 *                 sends $GO or $STOP. The Pi ends the turn when the CAMERA
 *                 sees a row again, not on a timer -- a timed spin is wrong
 *                 the moment the surface or the battery changes.
 *    $P           keepalive
 *
 *  ESP32 -> Pi at 10 Hz
 *    #T,<ms>,<mode>,<steer>
 *    #E,<event>
 *
 *  SAFETY, since there is no transmitter any more
 *  ----------------------------------------------
 *  1. It boots STOPPED and stays stopped until the Pi sends $GO.
 *  2. If the Pi goes quiet for 1 second the motors stop by themselves. So
 *     Ctrl-C on the Pi, a crashed script, or a yanked USB cable all stop the
 *     bot. Your keyboard IS the kill switch now.
 *  3. CRUISE is deliberately low. Raise it only once you trust the tracking.
 *
 *  Wire a physical switch in the LiPo line to the drivers if you can. With no
 *  transmitter that is your only hardware stop.
 *
 *  Motor pins, unchanged:
 *    Left  BTS7960  RPWM 32  LPWM 33
 *    Right BTS7960  RPWM 14  LPWM 26
 *    Enables        every spare GPIO held high -- see ENABLE_PINS below
 */

#include <Arduino.h>

// Pins as actually wired on this bot.
// GPIO 34/35/36/39 are INPUT-ONLY on the ESP32 and can never drive an
// output. The left driver was on 35, which is why that side never ran.
#define L_RPWM 32
#define L_LPWM 33
#define R_RPWM 25
#define R_LPWM 27

// DIRECTION
// ---------
// On a BTS7960 the two PWM inputs are just "drive this way" and "drive the
// other way" -- which one is forward depends entirely on how the motor
// leads are landed on M+/M-. Rather than swap wires, flip it here.
//
// If the bot drives BACKWARD, set the matching value to -1.
// If ONE SIDE runs the wrong way (the bot spins on the spot instead of
// going straight), flip only that side.
const int LEFT_DIR  = +1;
const int RIGHT_DIR = +1;

// DRIVER ENABLES
// ---------------
// motor_test.ino ran the motors fine and this sketch did not. The only
// difference between them was that motor_test drove EVERY spare GPIO high,
// while this one drove just two. So the BTS7960 enable pins are wired to a
// GPIO neither of us has identified.
//
// Rather than keep guessing one pin at a time, do what motor_test does:
// hold every output-capable GPIO that is not a motor pin high at boot. They
// are unused, so driving them high is harmless, and it guarantees the
// enables are asserted whichever pin they are actually on.
//
// If you later trace the real enable pins, replace this list with just those
// two -- but there is no functional need to.
const int ENABLE_PINS[] = {2, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19,
                           21, 22, 23, 26};
const int N_ENABLE = sizeof(ENABLE_PINS) / sizeof(ENABLE_PINS[0]);

#define CH_LR 0
#define CH_LL 1
#define CH_RR 2
#define CH_RL 3

// Forward speed. Do NOT drop this below ~75: a DC motor needs a minimum PWM
// just to overcome stiction, and below that it buzzes without turning. To go
// slower, the Pi pulses $GO/$STOP instead -- full torque while moving, much
// lower average speed. See PULSE_ON_S in run_mission.py.
// Slow and continuous beats fast and pulsed. Pulsing looked like a fault and
// gave the vision loop nothing steady to work with. 62 is close to the
// stiction floor on this chassis -- if it buzzes without turning, go back up
// in steps of 4.
const int CRUISE      = 62;
const int MAX_PWM     = 170;   // allow the outer side to exceed cruise
const int STEER_SCALE = 100;   // full +/-100 steer -> this much differential

// A four-wheel skid-steer robot turns by SCRUBBING its tyres sideways. That
// takes real force. With a small differential the inner wheels simply spin a
// little slower and the bot keeps going straight, which looks exactly like
// "the steering is not working".
//
// ALLOW_PIVOT lets the inner side run BACKWARDS for large corrections. That
// turns a weak differential into a genuine pivot and is usually the
// difference between a bot that corrects and one that does not. Set it to 0
// if you would rather it never reverses a wheel.
#define ALLOW_PIVOT 1

// Below this the motors do not overcome their own stiction and just buzz.
// Anything smaller is snapped to zero so we are not sending noise.
const int MIN_MOVE_PWM = 35;
const uint32_t PI_TIMEOUT_MS = 1000;
const int TURN_PWM = 85;             // spin speed. Low: the Pi has to SEE the
                                     // row to stop, so give it frames to work
                                     // with. Too fast and it overshoots.

enum Mode { IDLE, RUN, TURN, DONE };
Mode mode = IDLE;
int  turnDir = 1;                    // +1 right, -1 left

int      steerCmd = 0;
uint32_t piLast   = 0;
bool     everHeard = false;

void side(int fwd, int rev, int pwm) {
  pwm = constrain(pwm, -255, 255);
  ledcWrite(fwd, pwm >= 0 ?  pwm : 0);
  ledcWrite(rev, pwm <  0 ? -pwm : 0);
}
void drive(int l, int r) {
  side(CH_LR, CH_LL, constrain(l * LEFT_DIR,  -MAX_PWM, MAX_PWM));
  side(CH_RR, CH_RL, constrain(r * RIGHT_DIR, -MAX_PWM, MAX_PWM));
}
void stopAll() { drive(0, 0); }
void event(const char* e) { Serial.printf("#E,%s\n", e); }

void handleLine(String ln) {
  if (ln.length() < 2 || ln[0] != '$') return;
  piLast = millis();
  everHeard = true;
  if      (ln.startsWith("$GO"))      { if (mode == IDLE) { mode = RUN; event("RUNNING"); } }
  else if (ln.startsWith("$STOP"))    { if (mode == RUN)  { mode = IDLE; stopAll(); steerCmd = 0; event("STOPPED"); } }
  else if (ln.startsWith("$ROW_END")) { mode = DONE; stopAll(); event("ROW_END_ACK"); }
  else if (ln.startsWith("$T,"))      { turnDir = ln.substring(3).toInt() >= 0 ? 1 : -1;
                                        mode = TURN; event("TURNING"); }
  else if (ln.startsWith("$S,"))      { steerCmd = constrain(ln.substring(3).toInt(), -100, 100); }
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < N_ENABLE; i++) {
    pinMode(ENABLE_PINS[i], OUTPUT);
    digitalWrite(ENABLE_PINS[i], HIGH);
  }
  ledcSetup(CH_LR, 15000, 8); ledcAttachPin(L_RPWM, CH_LR);
  ledcSetup(CH_LL, 15000, 8); ledcAttachPin(L_LPWM, CH_LL);
  ledcSetup(CH_RR, 15000, 8); ledcAttachPin(R_RPWM, CH_RR);
  ledcSetup(CH_RL, 15000, 8); ledcAttachPin(R_LPWM, CH_RL);
  stopAll();
  event("BOOT_NO_RC_v6_SLOW");
  // Print the pin map at boot so you can confirm from miniterm which build
  // is actually on the board, instead of assuming the upload took.
  Serial.printf("#E,PINS L=%d/%d R=%d/%d CRUISE=%d DIR L%+d R%+d\n",
                L_RPWM, L_LPWM, R_RPWM, R_LPWM, CRUISE, LEFT_DIR, RIGHT_DIR);
  Serial.print("#E,ENABLES");
  for (int i = 0; i < N_ENABLE; i++) Serial.printf(" %d", ENABLE_PINS[i]);
  Serial.println();
}

void loop() {
  static uint32_t tCtl = 0, tTlm = 0;
  static String buf;
  uint32_t now = millis();

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handleLine(buf); buf = ""; }
    else if (c != '\r' && buf.length() < 48) buf += c;
  }

  if (now - tCtl >= 20) {                          // 50 Hz
    tCtl = now;
    // Pi silence watchdog -- this is the kill switch
    if ((mode == RUN || mode == TURN) && everHeard
        && (now - piLast) > PI_TIMEOUT_MS) {
      stopAll(); mode = IDLE; steerCmd = 0; event("PI_TIMEOUT");
    }
    if (mode == TURN) {
      // Counter-rotate: tightest possible spin, so the bot stays inside the
      // headland instead of arcing out into the crop.
      drive(TURN_PWM * turnDir, -TURN_PWM * turnDir);
    } else if (mode == RUN) {
      int st = (steerCmd * STEER_SCALE) / 100;
      int l = CRUISE + st;
      int r = CRUISE - st;
#if !ALLOW_PIVOT
      if (l < 0) l = 0;
      if (r < 0) r = 0;
#endif
      // kill buzzing: a wheel commanded below stiction does nothing useful
      if (l > 0 && l < MIN_MOVE_PWM) l = 0;
      if (r > 0 && r < MIN_MOVE_PWM) r = 0;
      if (l < 0 && l > -MIN_MOVE_PWM) l = 0;
      if (r < 0 && r > -MIN_MOVE_PWM) r = 0;
      drive(l, r);
    } else {
      stopAll();
    }
  }

  if (now - tTlm >= 100) {                         // 10 Hz
    tTlm = now;
    const char* ms = mode == RUN ? "RUN" : mode == TURN ? "TURN"
                   : mode == DONE ? "DONE" : "IDLE";
    // report the actual per-side PWM so you can see whether the differential
    // is big enough, without guessing
    int st = (mode == RUN) ? (steerCmd * STEER_SCALE) / 100 : 0;
    Serial.printf("#T,%lu,%s,%d,%d,%d\n", now, ms, steerCmd,
                  (mode == RUN) ? CRUISE + st : 0,
                  (mode == RUN) ? CRUISE - st : 0);
  }
}
