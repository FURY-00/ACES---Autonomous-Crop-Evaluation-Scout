/*  ACES drive controller  --  ESP32
 *
 *  Owns everything that must happen on time: motor PWM, quadrature encoders,
 *  the ultrasonic ping, the row-presence IR pair, and a 100 Hz steering loop.
 *  It never opens a camera and never makes a mission decision.
 *
 *  Why split it this way: Linux is not real-time. If the Pi's classifier hogs
 *  the CPU for 300 ms mid-row, a Pi-driven PWM loop drives you into the crop.
 *  The ESP32 keeps steering on its last known good vision estimate, decays the
 *  speed, and stops on its own if the Pi stays quiet.
 *
 *  Protocol: see serial_link.py.
 */

#include <Arduino.h>

// ------------------------------------------------------------------ pins
#define ENC_L_A 34
#define ENC_L_B 35
#define ENC_R_A 32
#define ENC_R_B 33

#define MOT_L_PWM 25
#define MOT_L_DIR 26
#define MOT_R_PWM 27
#define MOT_R_DIR 14

// Three HC-SR04. One shared trigger is tempting but causes crosstalk:
// sensor A hears sensor B's burst and reports a wall that is not there.
// Separate triggers, fired strictly one at a time, 60 ms apart.
#define SON_F_TRIG 5
#define SON_F_ECHO 18
#define SON_L_TRIG 4
#define SON_L_ECHO 16
#define SON_R_TRIG 17
#define SON_R_ECHO 23
#define IR_LEFT    19   // side facing, LOW = crop wall present
#define IR_RIGHT   21

// ------------------------------------------------------------------ geometry
const float WHEEL_DIAM_CM   = 6.5f;
const int   TICKS_PER_REV   = 780;      // gearbox output x quadrature
const float WHEEL_BASE_CM   = 22.0f;
const float CM_PER_TICK     = (PI * WHEEL_DIAM_CM) / TICKS_PER_REV;

// Geometry: 32 cm robot in a 40-45 cm gap. 4 cm per side of slack, 2 cm of
// which is safety margin. These MUST match config.py.
const float ROBOT_WIDTH_CM  = 32.0f;
const float LAT_LIMIT_CM    = 2.0f;     // usable dodge
const float LAT_ABORT_CM    = 4.0f;     // beyond this we are touching plants
const float SIDE_MIN_CM     = 2.0f;     // side sonar closer than this = stop
const float MAX_SPEED_CMS   = 20.0f;

// ------------------------------------------------------------------ gains
float Kp_e   = 2.6f;    // cross-track error  (cm  -> steer units)
float Kd_e   = 0.9f;
float Kp_psi = 1.1f;    // heading error      (deg -> steer units)
const float STEER_CLAMP = 55.0f;   // percent of full scale

// ------------------------------------------------------------------ state
enum State { S_IDLE, S_DRIVE, S_BACK, S_TURN, S_ESTOP };
State state = S_IDLE;

volatile long tickL = 0, tickR = 0;
long odoZeroL = 0, odoZeroR = 0;

float visOff = 0, visHead = 0, visConf = 0;
uint32_t visStamp = 0;

float latSet = 0;                 // dodge setpoint, always within +/-5 cm
float cruise = 15.0f;
float prevErr = 0;

float headEst = 0;                // integrated from wheel difference
float latEst  = 0;                // dead-reckoned cross-track, used when blind

long  turnTargetTicks = 0; int turnDir = 0;
long  backTargetTicks = 0;

float frontCm = 999, leftCm = 999, rightCm = 999;

// ------------------------------------------------------------------ encoders
void IRAM_ATTR isrL() { digitalRead(ENC_L_B) ? tickL-- : tickL++; }
void IRAM_ATTR isrR() { digitalRead(ENC_R_B) ? tickR++ : tickR--; }

float odoCm() {
  return ((tickL - odoZeroL) + (tickR - odoZeroR)) * 0.5f * CM_PER_TICK;
}

// ------------------------------------------------------------------ motors
void drive(float leftPct, float rightPct) {
  leftPct  = constrain(leftPct, -100, 100);
  rightPct = constrain(rightPct, -100, 100);
  digitalWrite(MOT_L_DIR, leftPct  >= 0);
  digitalWrite(MOT_R_DIR, rightPct >= 0);
  ledcWrite(0, (int)(fabs(leftPct)  * 2.55f));
  ledcWrite(1, (int)(fabs(rightPct) * 2.55f));
}

// ------------------------------------------------------------------ sonar
// Round robin, one sensor per slot, so no sensor ever hears another's burst.
// Each sensor keeps its own 5-sample ring and reports the median: a single
// HC-SR04 ping in foliage lies constantly, because leaves scatter the cone.
float ring[3][5]; int ringIdx[3] = {0, 0, 0}; int sonarTurn = 0;

float pingOnce(int trig, int echo) {
  digitalWrite(trig, LOW);  delayMicroseconds(3);
  digitalWrite(trig, HIGH); delayMicroseconds(10);
  digitalWrite(trig, LOW);
  unsigned long us = pulseIn(echo, HIGH, 25000UL);
  if (us == 0) return 999.0f;
  return us / 58.0f;
}

float medianOf(int i) {
  float v[5];
  memcpy(v, ring[i], sizeof(v));
  for (int a = 1; a < 5; a++) {
    float k = v[a]; int b = a - 1;
    while (b >= 0 && v[b] > k) { v[b + 1] = v[b]; b--; }
    v[b + 1] = k;
  }
  return v[2];
}

void serviceSonar() {
  const int trig[3] = {SON_F_TRIG, SON_L_TRIG, SON_R_TRIG};
  const int echo[3] = {SON_F_ECHO, SON_L_ECHO, SON_R_ECHO};
  int i = sonarTurn;
  ring[i][ringIdx[i]] = pingOnce(trig[i], echo[i]);
  ringIdx[i] = (ringIdx[i] + 1) % 5;
  frontCm = medianOf(0); leftCm = medianOf(1); rightCm = medianOf(2);
  sonarTurn = (sonarTurn + 1) % 3;
}

// ------------------------------------------------------------------ protocol
void sendTelemetry() {
  uint8_t flags = (digitalRead(IR_LEFT) == LOW ? 1 : 0) |
                  (digitalRead(IR_RIGHT) == LOW ? 2 : 0);
  const char* s = state == S_IDLE ? "IDLE" : state == S_DRIVE ? "DRIVE" :
                  state == S_BACK ? "BACK" : state == S_TURN ? "TURN" : "ESTOP";
  Serial.printf("#T,%lu,%s,%.1f,%.2f,%.1f,%.1f,%.1f,%.1f,%u\n",
                millis(), s, odoCm(), latEst, headEst,
                frontCm, leftCm, rightCm, flags);
}

void event(const char* e) { Serial.printf("#E,%s\n", e); }

void handleLine(String ln) {
  if (ln.length() < 2 || ln[0] != '$') return;
  char c = ln[1];
  String rest = ln.substring(3);

  if (c == 'V') {
    int a = rest.indexOf(','), b = rest.indexOf(',', a + 1);
    visOff  = rest.substring(0, a).toFloat();
    visHead = rest.substring(a + 1, b).toFloat();
    visConf = rest.substring(b + 1).toFloat();
    visStamp = millis();
    if (visConf > 0.5f) { latEst = visOff; headEst = visHead; }  // vision re-anchors
  }
  else if (c == 'M') {
    if      (rest.startsWith("DRIVE")) state = S_DRIVE;
    else if (rest.startsWith("ESTOP")) { state = S_ESTOP; drive(0, 0); }
    else                                { state = S_IDLE;  drive(0, 0); }
  }
  else if (c == 'S') cruise = constrain(rest.toFloat(), 0, MAX_SPEED_CMS);
  else if (c == 'L') latSet = constrain(rest.toFloat(), -LAT_LIMIT_CM, LAT_LIMIT_CM);
  else if (c == 'T') {
    float deg = rest.toFloat();
    turnDir = deg >= 0 ? 1 : -1;
    float arc = fabs(deg) / 360.0f * PI * WHEEL_BASE_CM;
    turnTargetTicks = (long)(arc / CM_PER_TICK);
    tickL = tickR = 0;
    state = S_TURN;
  }
  else if (c == 'B') {
    backTargetTicks = (long)(fabs(rest.toFloat()) / CM_PER_TICK);
    tickL = tickR = 0;
    state = S_BACK;
  }
  else if (c == 'Z') { odoZeroL = tickL; odoZeroR = tickR; latEst = 0; headEst = 0; }
}

// ------------------------------------------------------------------ control
void steerLoop(float dt) {
  bool visionFresh = (millis() - visStamp) < 400 && visConf >= 0.35f;

  if (!visionFresh) {
    // Dead reckon: integrate the wheel difference into a heading, and the
    // heading into a lateral drift. Good for ~2 seconds, not for 20.
    static long pl = 0, pr = 0;
    float dl = (tickL - pl) * CM_PER_TICK, dr = (tickR - pr) * CM_PER_TICK;
    pl = tickL; pr = tickR;
    headEst += degrees((dr - dl) / WHEEL_BASE_CM);
    latEst  += ((dl + dr) * 0.5f) * sin(radians(headEst));
  }

  float err = latEst - latSet;                 // where we are vs where we want to be
  float dErr = (err - prevErr) / max(dt, 1e-3f);
  prevErr = err;

  float steer = -(Kp_e * err + Kd_e * dErr + Kp_psi * headEst);
  steer = constrain(steer, -STEER_CLAMP, STEER_CLAMP);

  // Speed policy: slow down when badly off centre or when flying blind.
  float v = cruise / MAX_SPEED_CMS * 100.0f;
  if (fabs(err) > 3.0f)  v *= 0.7f;
  if (!visionFresh)      v *= 0.5f;
  if (millis() - visStamp > 1500) { state = S_IDLE; drive(0, 0); event("VISION_TIMEOUT"); return; }
  if (frontCm < 25.0f)   { drive(0, 0); event("SONAR_STOP"); return; }
  // Last line of defence: a side sonar inside SIDE_MIN_CM means a plant is
  // about to be crushed. The Pi may not have noticed yet. Stop anyway.
  if ((leftCm > 1.0f && leftCm < SIDE_MIN_CM) ||
      (rightCm > 1.0f && rightCm < SIDE_MIN_CM)) {
    drive(0, 0); event("CROP_GUARD"); return;
  }
  if (fabs(err) > LAT_ABORT_CM && (millis() - visStamp) > 800) {
    drive(0, 0); event("LATERAL_ABORT"); return;
  }

  drive(v - steer, v + steer);
}

// ------------------------------------------------------------------ setup
void setup() {
  Serial.begin(115200);
  pinMode(ENC_L_A, INPUT); pinMode(ENC_L_B, INPUT);
  pinMode(ENC_R_A, INPUT); pinMode(ENC_R_B, INPUT);
  attachInterrupt(ENC_L_A, isrL, RISING);
  attachInterrupt(ENC_R_A, isrR, RISING);

  pinMode(MOT_L_DIR, OUTPUT); pinMode(MOT_R_DIR, OUTPUT);
  ledcSetup(0, 20000, 8); ledcAttachPin(MOT_L_PWM, 0);
  ledcSetup(1, 20000, 8); ledcAttachPin(MOT_R_PWM, 1);

  pinMode(SON_F_TRIG, OUTPUT); pinMode(SON_F_ECHO, INPUT);
  pinMode(SON_L_TRIG, OUTPUT); pinMode(SON_L_ECHO, INPUT);
  pinMode(SON_R_TRIG, OUTPUT); pinMode(SON_R_ECHO, INPUT);
  for (int i = 0; i < 3; i++) for (int j = 0; j < 5; j++) ring[i][j] = 999.0f;
  pinMode(IR_LEFT, INPUT_PULLUP); pinMode(IR_RIGHT, INPUT_PULLUP);
  drive(0, 0);
  event("BOOT");
}

// ------------------------------------------------------------------ loop
void loop() {
  static uint32_t tCtl = 0, tTlm = 0, tSon = 0;
  static String buf;

  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n') { handleLine(buf); buf = ""; }
    else if (ch != '\r' && buf.length() < 64) buf += ch;
  }

  uint32_t now = millis();

  // 60 ms per sensor -> each sensor refreshes every 180 ms, no crosstalk.
  if (now - tSon > 60) { tSon = now; serviceSonar(); }

  if (now - tCtl >= 10) {                     // 100 Hz
    float dt = (now - tCtl) / 1000.0f;
    tCtl = now;
    switch (state) {
      case S_DRIVE: steerLoop(dt); break;
      case S_TURN: {
        long done = (labs(tickL) + labs(tickR)) / 2;
        if (done >= turnTargetTicks) { drive(0, 0); state = S_IDLE; event("TURN_DONE"); }
        else drive(35.0f * turnDir, -35.0f * turnDir);
        break;
      }
      case S_BACK: {
        long done = (labs(tickL) + labs(tickR)) / 2;
        if (done >= backTargetTicks) { drive(0, 0); state = S_IDLE; event("BACK_DONE"); }
        else drive(-25, -25);
        break;
      }
      default: drive(0, 0);
    }
  }

  if (now - tTlm >= 50) { tTlm = now; sendTelemetry(); }   // 20 Hz
}
