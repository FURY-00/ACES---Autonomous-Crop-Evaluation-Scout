/*  ACES demo firmware  --  ESP32
 *
 *  Autonomous corridor following using the TWO SIDE ULTRASONICS. No camera,
 *  no encoders, no IR. Everything here is hardware you already have wired.
 *
 *  WHY THIS WORKS WITHOUT A NAV CAMERA
 *  -----------------------------------
 *  Two side sonars measure the distance to each wall. Their difference IS
 *  the cross-track error:
 *
 *      error = (left - right) / 2        + means you are right of centre
 *
 *  No calibration, no pixels, no lighting dependence. In a narrow corridor
 *  this is more accurate than the camera would have been.
 *
 *  SAFETY: CH5 ON YOUR TRANSMITTER IS THE KILL SWITCH
 *  --------------------------------------------------
 *  CH5 low  -> MANUAL. Your existing RC control, unchanged.
 *  CH5 high -> AUTO.   The robot drives itself.
 *  Drop CH5 at any moment and you have the sticks back instantly. Do not run
 *  the demo without the transmitter powered on and in your hand.
 *
 *  If PPM is missing entirely (no receiver), the robot stays in MANUAL and
 *  refuses to move -- it will not run away on you.
 *
 *  WIRING (matches your existing build)
 *  ------------------------------------
 *    PPM in .............. GPIO 16
 *    Left BTS7960 ........ RPWM 32, LPWM 33
 *    Right BTS7960 ....... RPWM 14, LPWM 26
 *    Left sonar .......... TRIG 5,  ECHO 18
 *    Right sonar ......... TRIG 17, ECHO 23
 *    Pi ..................  USB cable  (/dev/ttyUSB0, 115200)
 *
 *  Sonar echo pins are 5 V. Use a divider (1k / 2k) into the ESP32, or you
 *  will slowly damage the pin.
 */

#include <Arduino.h>

// ------------------------------------------------------------------ pins
#define PPM_PIN     16
#define L_RPWM      32
#define L_LPWM      33
#define R_RPWM      14
#define R_LPWM      26
#define SON_L_TRIG   5
#define SON_L_ECHO  18
#define SON_R_TRIG  17
#define SON_R_ECHO  23

// ------------------------------------------------------------------ ledc
#define CH_LR 0
#define CH_LL 1
#define CH_RR 2
#define CH_RL 3

// ------------------------------------------------------------------ tuning
// Your LiPo sags at full throttle and the motors cut out. The demo runs well
// below that. Do not raise this to "look faster" -- a brownout mid-demo looks
// far worse than a slow robot.
const int   MAX_AUTO_PWM   = 130;   // of 255
const int   BASE_PWM       = 100;   // cruise
const int   TURN_PWM       = 110;   // spin speed

const float KP             = 6.0f;  // cm of error -> PWM of steer
const float KD             = 2.5f;
const int   STEER_CLAMP    = 70;    // max PWM difference between sides

const float WALL_MAX_CM    = 60.0f; // beyond this, no wall is seen
const float WALL_MIN_CM    = 8.0f;  // closer than this, back off
const float STOP_CM        = 10.0f; // both walls this close = wedged
const int   ENDROW_FRAMES  = 12;    // consecutive frames with no walls

// ------------------------------------------------------------------ state
enum Mode { MANUAL, AUTO_IDLE, AUTO_RUN, AUTO_TURN, AUTO_DONE };
Mode mode = MANUAL;

volatile uint16_t ppm[8];
volatile uint32_t ppmLast = 0;
volatile uint8_t  ppmIdx = 0;

float leftCm = 999, rightCm = 999;
float ring[2][5]; int ringIdx[2] = {0, 0}; int sonarTurn = 0;
float prevErr = 0;
int   noWallCount = 0;
uint32_t turnStart = 0;
int   turnDir = 1;
bool  piWantsRun = false;
uint32_t piLast = 0;

// ------------------------------------------------------------------ PPM
void IRAM_ATTR ppmISR() {
  uint32_t now = micros();
  uint32_t dt = now - ppmLast;
  ppmLast = now;
  if (dt > 3000) { ppmIdx = 0; return; }       // frame gap
  if (ppmIdx < 8) ppm[ppmIdx++] = dt;
}

bool ppmAlive() { return (micros() - ppmLast) < 100000UL; }   // 100 ms

int ch(int i) {                                  // 1000..2000 us
  if (i < 0 || i > 7) return 1500;
  uint16_t v = ppm[i];
  return (v > 800 && v < 2200) ? v : 1500;
}

// ------------------------------------------------------------------ motors
void side(int chFwd, int chRev, int pwm) {
  pwm = constrain(pwm, -255, 255);
  if (pwm >= 0) { ledcWrite(chFwd, pwm);  ledcWrite(chRev, 0); }
  else          { ledcWrite(chFwd, 0);    ledcWrite(chRev, -pwm); }
}

void drive(int l, int r) {
  side(CH_LR, CH_LL, l);
  side(CH_RR, CH_RL, r);
}

void stopMotors() { drive(0, 0); }

// ------------------------------------------------------------------ sonar
float pingOnce(int trig, int echo) {
  digitalWrite(trig, LOW);  delayMicroseconds(3);
  digitalWrite(trig, HIGH); delayMicroseconds(10);
  digitalWrite(trig, LOW);
  unsigned long us = pulseIn(echo, HIGH, 25000UL);
  return us == 0 ? 999.0f : us / 58.0f;
}

float medianOf(int i) {
  float v[5]; memcpy(v, ring[i], sizeof(v));
  for (int a = 1; a < 5; a++) {
    float k = v[a]; int b = a - 1;
    while (b >= 0 && v[b] > k) { v[b + 1] = v[b]; b--; }
    v[b + 1] = k;
  }
  return v[2];
}

// Fired one at a time, 60 ms apart. Two sonars on a shared trigger hear each
// other's burst and invent walls that are not there.
void serviceSonar() {
  const int trig[2] = {SON_L_TRIG, SON_R_TRIG};
  const int echo[2] = {SON_L_ECHO, SON_R_ECHO};
  int i = sonarTurn;
  ring[i][ringIdx[i]] = pingOnce(trig[i], echo[i]);
  ringIdx[i] = (ringIdx[i] + 1) % 5;
  leftCm = medianOf(0); rightCm = medianOf(1);
  sonarTurn ^= 1;
}

// ------------------------------------------------------------------ protocol
void event(const char* e) { Serial.printf("#E,%s\n", e); }

void telemetry() {
  const char* m = mode == MANUAL ? "MANUAL" : mode == AUTO_IDLE ? "AUTO_IDLE"
                 : mode == AUTO_RUN ? "AUTO_RUN" : mode == AUTO_TURN ? "AUTO_TURN"
                 : "AUTO_DONE";
  bool lw = leftCm < WALL_MAX_CM, rw = rightCm < WALL_MAX_CM;
  float err = (lw && rw) ? (leftCm - rightCm) / 2.0f : 0.0f;
  Serial.printf("#T,%lu,%s,%.1f,%.1f,%.2f,%d,%d\n",
                millis(), m, leftCm, rightCm, err, lw ? 1 : 0, rw ? 1 : 0);
}

void handleLine(String ln) {
  if (ln.length() < 2 || ln[0] != '$') return;
  char c = ln[1];
  piLast = millis();
  if      (c == 'G') { piWantsRun = true;  }         // go
  else if (c == 'S') { piWantsRun = false; }         // stop (leaf found etc.)
  else if (c == 'T') { mode = AUTO_TURN; turnStart = millis();
                       turnDir = ln.substring(3).toFloat() >= 0 ? 1 : -1; }
  else if (c == 'P') { Serial.printf("#E,PONG\n"); }
}

// ------------------------------------------------------------------ control
void autoStep() {
  bool lw = leftCm < WALL_MAX_CM;
  bool rw = rightCm < WALL_MAX_CM;

  // ---- wedged: something right in front of a side sensor ----
  if ((lw && leftCm < STOP_CM) || (rw && rightCm < STOP_CM)) {
    stopMotors(); event("BLOCKED"); return;
  }

  // ---- end of corridor: both walls gone ----
  if (!lw && !rw) {
    if (++noWallCount >= ENDROW_FRAMES) {
      stopMotors(); mode = AUTO_IDLE; noWallCount = 0; event("ROW_END");
      return;
    }
  } else noWallCount = 0;

  // ---- cross-track error from the two sonars ----
  // Both walls: the difference IS the error, and it needs no calibration.
  // One wall: hold a fixed standoff from the wall we can still see.
  float err;
  if (lw && rw)      err = (leftCm - rightCm) / 2.0f;
  else if (lw)       err = leftCm - 22.0f;
  else if (rw)       err = 22.0f - rightCm;
  else               err = 0.0f;

  float dErr = err - prevErr;
  prevErr = err;
  int steer = (int)constrain(KP * err + KD * dErr, -STEER_CLAMP, STEER_CLAMP);

  int base = BASE_PWM;
  if (fabs(err) > 8.0f) base = BASE_PWM * 0.75f;      // slow when badly off

  int l = constrain(base + steer, -MAX_AUTO_PWM, MAX_AUTO_PWM);
  int r = constrain(base - steer, -MAX_AUTO_PWM, MAX_AUTO_PWM);
  drive(l, r);
}

// ------------------------------------------------------------------ setup
void setup() {
  Serial.begin(115200);

  pinMode(PPM_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PPM_PIN), ppmISR, RISING);

  ledcSetup(CH_LR, 15000, 8); ledcAttachPin(L_RPWM, CH_LR);
  ledcSetup(CH_LL, 15000, 8); ledcAttachPin(L_LPWM, CH_LL);
  ledcSetup(CH_RR, 15000, 8); ledcAttachPin(R_RPWM, CH_RR);
  ledcSetup(CH_RL, 15000, 8); ledcAttachPin(R_LPWM, CH_RL);

  pinMode(SON_L_TRIG, OUTPUT); pinMode(SON_L_ECHO, INPUT);
  pinMode(SON_R_TRIG, OUTPUT); pinMode(SON_R_ECHO, INPUT);
  for (int i = 0; i < 2; i++) for (int j = 0; j < 5; j++) ring[i][j] = 999.0f;

  stopMotors();
  event("BOOT");
}

// ------------------------------------------------------------------ loop
void loop() {
  static uint32_t tSon = 0, tCtl = 0, tTlm = 0;
  static String buf;

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handleLine(buf); buf = ""; }
    else if (c != '\r' && buf.length() < 48) buf += c;
  }

  uint32_t now = millis();
  if (now - tSon > 60) { tSon = now; serviceSonar(); }

  if (now - tCtl >= 20) {                       // 50 Hz
    tCtl = now;

    // ---- CH5 is the master switch, and RC always wins ----
    bool rc = ppmAlive();
    bool wantAuto = rc && (ch(4) > 1600);       // CH5 high

    if (!wantAuto) {
      if (mode != MANUAL) { mode = MANUAL; stopMotors(); event("MANUAL"); }
    } else if (mode == MANUAL) {
      mode = AUTO_IDLE; stopMotors(); prevErr = 0; noWallCount = 0;
      event("AUTO");
    }

    switch (mode) {
      case MANUAL: {
        if (!rc) { stopMotors(); break; }       // no receiver -> do not move
        int thr = ch(1) - 1500;                 // CH2 forward/back
        int str = ch(0) - 1500;                 // CH1 left/right
        if (abs(thr) < 40) thr = 0;
        if (abs(str) < 40) str = 0;
        int t = map(thr, -500, 500, -MAX_AUTO_PWM, MAX_AUTO_PWM);
        int s = map(str, -500, 500, -STEER_CLAMP, STEER_CLAMP);
        drive(t + s, t - s);
        break;
      }
      case AUTO_IDLE:
        stopMotors();
        // The Pi says go. If the Pi is silent for 2 s, stay stopped.
        if (piWantsRun && (millis() - piLast) < 2000) mode = AUTO_RUN;
        break;

      case AUTO_RUN:
        if (!piWantsRun || (millis() - piLast) > 2000) {
          stopMotors(); mode = AUTO_IDLE; break;
        }
        autoStep();
        break;

      case AUTO_TURN: {
        if (millis() - turnStart > 1800) {      // timed spin, tune on the day
          stopMotors(); mode = AUTO_IDLE; prevErr = 0; event("TURN_DONE");
        } else drive(TURN_PWM * turnDir, -TURN_PWM * turnDir);
        break;
      }
      default: stopMotors();
    }
  }

  if (now - tTlm >= 100) { tTlm = now; telemetry(); }   // 10 Hz
}
