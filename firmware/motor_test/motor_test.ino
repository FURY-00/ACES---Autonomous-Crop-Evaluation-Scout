/*  ACES motor test -- no vision, no Pi, no receiver.
 *
 *  Just drives the motors so you can find out whether the hardware works and
 *  which pins the BTS7960 enables are actually on.
 *
 *  It runs a repeating cycle: forward, stop, left, stop, right, stop, and
 *  prints what it is doing over serial at 115200.
 *
 *  THE ENABLE-PIN HUNT
 *  -------------------
 *  A BTS7960 does nothing at all unless R_EN and L_EN are high. If those are
 *  on GPIOs the sketch never sets, the PWM is perfect and the wheels sit
 *  still -- exactly the symptom you have.
 *
 *  ENABLE_HUNT below sets EVERY safe spare GPIO high. If the wheels move
 *  with this sketch but not with the real firmware, an enable pin is the
 *  cause, and you can then binary-search which one by commenting entries out
 *  of the list.
 *
 *  Motor pins, unchanged:
 *    Left  BTS7960  RPWM 32  LPWM 33
 *    Right BTS7960  RPWM 14  LPWM 26
 */

#include <Arduino.h>

// Pins as actually wired on this bot.
// NOTE: GPIO 34/35/36/39 are INPUT-ONLY on the ESP32 and can never drive a
// motor driver. The left driver was on 35, which is why that side was dead.
#define L_RPWM 32
#define L_LPWM 33
#define R_RPWM 25
#define R_LPWM 27

#define CH_LR 0
#define CH_LL 1
#define CH_RR 2
#define CH_RL 3

const int SPEED = 90;

// Every output-capable GPIO that is NOT one of the four PWM pins above and
// is safe to drive high on boot. 34/35/36/39 are input-only so they are not
// here. 6-11 are the flash pins and must never be touched.
// Every output-capable GPIO that is not one of the four motor pins above.
// If the driver enables are wired to the ESP32 somewhere, this finds them.
const int ENABLE_HUNT[] = {2, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19,
                           21, 22, 23, 26};
const int N_HUNT = sizeof(ENABLE_HUNT) / sizeof(ENABLE_HUNT[0]);

void side(int fwd, int rev, int pwm) {
  pwm = constrain(pwm, -255, 255);
  ledcWrite(fwd, pwm >= 0 ?  pwm : 0);
  ledcWrite(rev, pwm <  0 ? -pwm : 0);
}
void drive(int l, int r) {
  side(CH_LR, CH_LL, l);
  side(CH_RR, CH_RL, r);
}

void setup() {
  Serial.begin(115200);
  delay(500);

  ledcSetup(CH_LR, 15000, 8); ledcAttachPin(L_RPWM, CH_LR);
  ledcSetup(CH_LL, 15000, 8); ledcAttachPin(L_LPWM, CH_LL);
  ledcSetup(CH_RR, 15000, 8); ledcAttachPin(R_RPWM, CH_RR);
  ledcSetup(CH_RL, 15000, 8); ledcAttachPin(R_LPWM, CH_RL);
  drive(0, 0);

  Serial.println("\n=== ACES motor test ===");
  Serial.print("driving these pins HIGH as possible enables: ");
  for (int i = 0; i < N_HUNT; i++) {
    pinMode(ENABLE_HUNT[i], OUTPUT);
    digitalWrite(ENABLE_HUNT[i], HIGH);
    Serial.printf("%d ", ENABLE_HUNT[i]);
  }
  Serial.println("\n");
  Serial.println("If the wheels move now but not with aces_noRC.ino, an");
  Serial.println("enable pin is your problem. Comment out half this list,");
  Serial.println("reflash, and narrow it down.\n");
  Serial.println("If the wheels do NOT move even now, it is not the pins:");
  Serial.println("  - is the LiPo connected to the driver B+/B- terminals?");
  Serial.println("  - do the BTS7960 boards have any LED lit?");
  Serial.println("  - is the ESP32 ground tied to the driver ground?");
  Serial.println("  - are the motor wires on M+/M- of each driver?\n");
  delay(2000);
}

void loop() {
  Serial.println("FORWARD  (both sides forward)");
  drive(SPEED, SPEED);   delay(2000);
  Serial.println("stop");
  drive(0, 0);           delay(1000);

  Serial.println("LEFT WHEELS ONLY");
  drive(SPEED, 0);       delay(2000);
  Serial.println("stop");
  drive(0, 0);           delay(1000);

  Serial.println("RIGHT WHEELS ONLY");
  drive(0, SPEED);       delay(2000);
  Serial.println("stop");
  drive(0, 0);           delay(1000);

  Serial.println("REVERSE");
  drive(-SPEED, -SPEED); delay(2000);
  Serial.println("stop\n");
  drive(0, 0);           delay(2000);
}
