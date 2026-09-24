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
 *  RECEIVER: two wiring options, pick one with RX_MODE below.
 *
 *    RX_PPM  (one signal wire)   PPM/CH1 port  -> GPIO 16
 *    RX_PWM  (three signal wires) CH1 -> GPIO 35
 *                                 CH2 -> GPIO 34
 *                                 CH6 -> GPIO 39
 *    Either way the receiver still needs 5V and GND from the buck converter,
 *    and its ground must be common with the ESP32's.
 *
 *  Motor pins, unchanged from your working RC build:
 *    Left  BTS7960  RPWM 32  LPWM 33
 *    Right BTS7960  RPWM 14  LPWM 26
 *
 *  Safety: in AUTO, if the Pi goes quiet for 2 seconds the ESP32 stops by
 *  itself. The Pi sends something every 100 ms, so a normal loop never gets
 *  close to that -- but a crashed script or a yanked USB cable stops the bot
 *  instead of leaving it driving.
 */

#include <Arduino.h>

// ---- receiver mode -------------------------------------------------------
// Set to 1 if all three receiver wires go to the PPM/CH1 port (one signal
// line carrying every channel). Set to 0 if you have separate signal wires
// from CH1, CH2 and CH6.
#define RX_MODE_PPM 0

// ---- pins ----------------------------------------------------------------
#define PPM_PIN   16          // used only when RX_MODE_PPM is 1

// individual-channel pins, used when RX_MODE_PPM is 0.
// GPIO 34/35/39 are input-only on the ESP32, which is exactly what we want.
#define RX_CH1    35          // steering stick
#define RX_CH2    34          // throttle stick
#define RX_CH6    39          // the AUTO switch
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

// individual-channel capture: each ISR times the high pulse on its own pin
volatile uint16_t pwmVal[3]  = {1500, 1500, 1000};
volatile uint32_t pwmRise[3] = {0, 0, 0};
volatile uint32_t pwmLast[3] = {0, 0, 0};

// ---- PPM -----------------------------------------------------------------
void IRAM_ATTR ppmISR() {
  uint32_t now = micros(), dt = now - ppmLast; ppmLast = now;
  if (dt > 3000) { ppmIdx = 0; return; }
  if (ppmIdx < 8) ppm[ppmIdx++] = (uint16_t)dt;
}
// ---- individual-channel ISRs --------------------------------------------
// On the rising edge note the time; on the falling edge the difference IS
// the pulse width, which is the channel value in microseconds.
void IRAM_ATTR pwmISR(int i, int pin) {
  uint32_t now = micros();
  if (digitalRead(pin)) {
    pwmRise[i] = now;
  } else if (pwmRise[i]) {
    uint32_t wdt = now - pwmRise[i];
    if (wdt > 800 && wdt < 2200) { pwmVal[i] = (uint16_t)wdt; pwmLast[i] = now; }
  }
}
void IRAM_ATTR isrCh1() { pwmISR(0, RX_CH1); }
void IRAM_ATTR isrCh2() { pwmISR(1, RX_CH2); }
void IRAM_ATTR isrCh6() { pwmISR(2, RX_CH6); }

bool rxAlive() {
#if RX_MODE_PPM
  return (micros() - ppmLast) < 100000UL;
#else
  // Alive if the AUTO-switch channel is still updating. That channel is the
  // one that matters for safety, so it is the one we watch.
  return (micros() - pwmLast[2]) < 100000UL;
#endif
}

// ch(0)=steering, ch(1)=throttle, ch(4)=the AUTO switch.
// The indices stay the same in both modes so the rest of the code does not
// care which wiring you used.
int ch(int i) {
#if RX_MODE_PPM
  uint16_t v = ppm[i];
  return (v > 800 && v < 2200) ? v : 1500;
#else
  if (i == 0) return pwmVal[0];
  if (i == 1) return pwmVal[1];
  if (i == 4) return pwmVal[2];
  return 1500;
#endif
}

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

#if RX_MODE_PPM
  pinMode(PPM_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PPM_PIN), ppmISR, RISING);
  memset((void*)ppm, 0, sizeof(ppm));
  event("RX_PPM");
#else
  pinMode(RX_CH1, INPUT); pinMode(RX_CH2, INPUT); pinMode(RX_CH6, INPUT);
  attachInterrupt(digitalPinToInterrupt(RX_CH1), isrCh1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(RX_CH2), isrCh2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(RX_CH6), isrCh6, CHANGE);
  event("RX_PWM_CH1_CH2_CH6");
#endif

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
    bool rc = rxAlive();
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
    // The extra fields make wiring problems obvious from the terminal:
    // if ch1/ch2/ch6 sit at 1500/1500/1500 the receiver is not being read.
    Serial.printf("#T,%lu,%s,%d,%d,%d,%d\n",
                  now, ms, steerCmd, ch(0), ch(1), ch(4));
  }
}
