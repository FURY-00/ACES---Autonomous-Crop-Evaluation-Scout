/*  ACES autonomous firmware -- no sonar. Webcam + Pi Cam drive everything.
 *
 *  The ESP32 does ONE job: run the motors when the Pi says GO, stop when it
 *  says STOP or ROW_END. All the seeing and deciding happens on the Pi.
 *  This board is a motor controller with a kill switch, nothing more.
 *
 *  CH5 LOW  = MANUAL RC   your existing sticks, always available
 *  CH5 HIGH = AUTO        takes commands from the Pi over USB
 *
 *  Pi -> ESP32
 *    $GO          drive forward at cruise
 *    $STOP        stop (obstacle ahead)
 *    $ROW_END     no plants left -- stop and stay stopped
 *    $S,<steer>   steering, -100..100, + = turn RIGHT
 *    $P           keepalive
 *
 *  Steering is differential: left = cruise + steer, right = cruise - steer.
 *  The Pi recomputes it every frame from the row centreline it sees.
 *
 *  ESP32 -> Pi at 10 Hz
 *    #T,<ms>,<mode>
 *    #E,<event>
 *
 *  Pins are unchanged from your working RC build:
 *    Left  BTS7960  RPWM 32  LPWM 33
 *    Right BTS7960  RPWM 14  LPWM 26
 *    PPM input GPIO 16
 *
 *  Safety: in AUTO, if the Pi goes quiet for 2 seconds the ESP32 stops by
 *  itself. The Pi sends something every 100 ms, so a normal loop never gets
 *  close to that -- but a crashed script or a yanked USB cable stops the bot
 *  instead of leaving it driving.
 */

#include <Arduino.h>

// ---- pins ----------------------------------------------------------------
#define PPM_PIN   16
#define L_RPWM    32
#define L_LPWM    33
#define R_RPWM    14
#define R_LPWM    26
// BTS7960 enable pins. Your notes say these are GPIO-driven -- set the numbers
// to match your harness, or tie them high on the board and ignore these.
#define L_EN      27
#define R_EN      25

// ---- speed ---------------------------------------------------------------
// Start LOW. With no sonar centring the bot will drift, and you want time to
// grab CH5. 75 is a sane indoor number; raise toward 100 once you trust it.
const int CRUISE  = 75;
const int MAX_PWM = 130;

// ---- ledc channels -------------------------------------------------------
#define CH_LR 0
#define CH_LL 1
#define CH_RR 2
#define CH_RL 3

// ---- state ---------------------------------------------------------------
enum Mode { MANUAL, AUTO_IDLE, AUTO_RUN, AUTO_DONE };
Mode mode = MANUAL;

volatile uint16_t ppm[8];
volatile uint32_t ppmLast = 0;
volatile uint8_t  ppmIdx  = 0;
uint32_t piLast = 0;
int steerCmd = 0;                      // -100..100, + = turn right

// ---- PPM -----------------------------------------------------------------
void IRAM_ATTR ppmISR() {
  uint32_t now = micros(), dt = now - ppmLast; ppmLast = now;
  if (dt > 3000) { ppmIdx = 0; return; }
  if (ppmIdx < 8) ppm[ppmIdx++] = (uint16_t)dt;
}
bool ppmAlive() { return (micros() - ppmLast) < 100000UL; }
int  ch(int i)  { uint16_t v = ppm[i]; return (v > 800 && v < 2200) ? v : 1500; }

// ---- motors --------------------------------------------------------------
void side(int fwd, int rev, int pwm) {
  pwm = constrain(pwm, -255, 255);
  ledcWrite(fwd, pwm >= 0 ?  pwm : 0);
  ledcWrite(rev, pwm <  0 ? -pwm : 0);
}
void drive(int l, int r) {
  side(CH_LR, CH_LL, constrain(l, -MAX_PWM, MAX_PWM));
  side(CH_RR, CH_RL, constrain(r, -MAX_PWM, MAX_PWM));
}
void stopAll() { drive(0, 0); }

// ---- protocol ------------------------------------------------------------
void event(const char* e) { Serial.printf("#E,%s\n", e); }

void handleLine(String ln) {
  if (ln.length() < 2 || ln[0] != '$') return;
  piLast = millis();
  if      (ln.startsWith("$GO"))      { if (mode == AUTO_IDLE) { mode = AUTO_RUN;  event("RUNNING"); } }
  else if (ln.startsWith("$STOP"))    { if (mode == AUTO_RUN)  { mode = AUTO_IDLE; stopAll(); steerCmd = 0; event("STOPPED"); } }
  else if (ln.startsWith("$ROW_END")) { mode = AUTO_DONE; stopAll(); event("ROW_END_ACK"); }
  else if (ln.startsWith("$S,"))      { steerCmd = constrain(ln.substring(3).toInt(), -100, 100); }
  else if (ln.startsWith("$P"))       { /* keepalive only */ }
}

// ---- setup ---------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  pinMode(L_EN, OUTPUT); pinMode(R_EN, OUTPUT);
  digitalWrite(L_EN, HIGH); digitalWrite(R_EN, HIGH);

  ledcSetup(CH_LR, 15000, 8); ledcAttachPin(L_RPWM, CH_LR);
  ledcSetup(CH_LL, 15000, 8); ledcAttachPin(L_LPWM, CH_LL);
  ledcSetup(CH_RR, 15000, 8); ledcAttachPin(R_RPWM, CH_RR);
  ledcSetup(CH_RL, 15000, 8); ledcAttachPin(R_LPWM, CH_RL);

  pinMode(PPM_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PPM_PIN), ppmISR, RISING);
  memset((void*)ppm, 0, sizeof(ppm));

  stopAll();
  event("BOOT");
}

// ---- loop ----------------------------------------------------------------
void loop() {
  static uint32_t tCtl = 0, tTlm = 0;
  static String buf;
  uint32_t now = millis();

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handleLine(buf); buf = ""; }
    else if (c != '\r' && buf.length() < 48) buf += c;
  }

  if (now - tCtl >= 20) {                       // 50 Hz
    tCtl = now;
    bool rc = ppmAlive();
    bool wantAuto = (rc && ch(4) > 1600);       // CH5 high

    if (!wantAuto) {
      if (mode != MANUAL) { mode = MANUAL; stopAll(); event("MANUAL"); }
    } else if (mode == MANUAL) {
      mode = AUTO_IDLE; stopAll(); piLast = now; event("AUTO");
    }

    // Pi silence watchdog
    if ((mode == AUTO_RUN || mode == AUTO_IDLE) && (now - piLast) > 2000) {
      stopAll(); mode = AUTO_IDLE; event("PI_TIMEOUT");
    }

    switch (mode) {
      case MANUAL: {
        if (!rc) { stopAll(); break; }
        int t = ch(1) - 1500, s = ch(0) - 1500;
        if (abs(t) < 40) t = 0;
        if (abs(s) < 40) s = 0;
        int v  = map(t, -500, 500, -MAX_PWM, MAX_PWM);
        int st = map(s, -500, 500, -40, 40);
        drive(v + st, v - st);
        break;
      }
      case AUTO_RUN: {
        // Differential steering. Scaled so a full +/-100 command gives a
        // firm but not violent turn -- a spin-on-the-spot correction inside
        // a crop row puts wheels through plants.
        int st = (steerCmd * 45) / 100;
        drive(CRUISE + st, CRUISE - st);
        break;
      }
      case AUTO_DONE: stopAll(); break;
      default:        stopAll();
    }
  }

  if (now - tTlm >= 100) {                      // 10 Hz
    tTlm = now;
    const char* ms = mode == MANUAL    ? "MANUAL"
                   : mode == AUTO_RUN  ? "AUTO_RUN"
                   : mode == AUTO_DONE ? "AUTO_DONE" : "AUTO_IDLE";
    Serial.printf("#T,%lu,%s,%d\n", now, ms, steerCmd);
  }
}
