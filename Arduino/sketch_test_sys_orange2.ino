#include <Servo.h>

// ============================================================
// РАСПИНОВКА
// ============================================================

// --- Энкодер (только пины 2 и 3 поддерживают прерывания на Uno) ---
#define ENC_A 2
#define ENC_B 3

// --- Драйвер мотора движения ---
#define ENA 6
#define IN1 7
#define IN2 8

// --- Рулевое серво (45 = право, 90 = прямо, 135 = лево) ---
#define SERVO_STEER_PIN 12
#define STEER_MIN 45
#define STEER_MAX 135

// --- Шаговый двигатель (перенесён на A0..A3) ---
#define STEP_IN1 A0
#define STEP_IN2 A1
#define STEP_IN3 A2
#define STEP_IN4 A3

// --- Сервы камеры 2DOF ---
#define SERVO_PAN_PIN  13
#define SERVO_TILT_PIN 11

Servo steeringServo;
Servo panServo;
Servo tiltServo;

// ============================================================
// ПЕРЕМЕННЫЕ
// ============================================================
volatile long encoderCount = 0;
bool lastA = false;

// ============================================================
// ЭНКОДЕР
// ============================================================
void encoderISR() {
  bool a = digitalRead(ENC_A);
  bool b = digitalRead(ENC_B);
  if (a != lastA) {
    if (b != a) encoderCount++;
    else        encoderCount--;
    lastA = a;
  }
}

// ============================================================
// МОТОР ДВИЖЕНИЯ
// ============================================================
void motorControl(int speed, bool forward) {
  speed = constrain(speed, 0, 255);
  if (speed == 0) {
    analogWrite(ENA, 0);
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    return;
  }
  analogWrite(ENA, speed);
  digitalWrite(IN1, forward ? HIGH : LOW);
  digitalWrite(IN2, forward ? LOW : HIGH);
}

// ============================================================
// ПЛАВНЫЙ ПОВОРОТ РУЛЕВОГО СЕРВО (45..135)
// ============================================================
void smoothSteer(int targetAngle, int stepDelay = 10) {
  targetAngle = constrain(targetAngle, STEER_MIN, STEER_MAX);
  int current = steeringServo.read();
  int step = (targetAngle > current) ? 1 : -1;

  while (current != targetAngle) {
    current += step;
    steeringServo.write(current);
    delay(stepDelay);
  }
  steeringServo.write(targetAngle);
}

// ============================================================
// ПЛАВНОЕ ДВИЖЕНИЕ КАМЕРЫ (pan + tilt одновременно)
// ============================================================
void smoothCamera(int targetPan, int targetTilt, int stepDelay = 10) {
  targetPan  = constrain(targetPan,  0, 180);
  targetTilt = constrain(targetTilt, 0, 180);

  int pan  = panServo.read();
  int tilt = tiltServo.read();

  while (pan != targetPan || tilt != targetTilt) {
    if (pan  < targetPan)  pan++;
    else if (pan > targetPan)  pan--;

    if (tilt < targetTilt) tilt++;
    else if (tilt > targetTilt) tilt--;

    panServo.write(pan);
    tiltServo.write(tilt);
    delay(stepDelay);
  }
}

// ============================================================
// ШАГОВЫЙ ДВИГАТЕЛЬ
// ============================================================
void stepMotor(int steps, bool forward) {
  byte seq[4][4] = {
    {HIGH, LOW,  HIGH, LOW },
    {LOW,  HIGH, HIGH, LOW },
    {LOW,  HIGH, LOW,  HIGH},
    {HIGH, LOW,  LOW,  HIGH}
  };

  for (int i = 0; i < abs(steps); i++) {
    int idx = forward ? (i % 4) : ((4 - (i % 4)) % 4);
    digitalWrite(STEP_IN1, seq[idx][0]);
    digitalWrite(STEP_IN2, seq[idx][1]);
    digitalWrite(STEP_IN3, seq[idx][2]);
    digitalWrite(STEP_IN4, seq[idx][3]);
    delay(15);
  }

  digitalWrite(STEP_IN1, LOW);
  digitalWrite(STEP_IN2, LOW);
  digitalWrite(STEP_IN3, LOW);
  digitalWrite(STEP_IN4, LOW);
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(9600);

  // Мотор
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  motorControl(0, true);

  // Энкодер
  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);
  lastA = digitalRead(ENC_A);
  attachInterrupt(digitalPinToInterrupt(ENC_A), encoderISR, CHANGE);

  // Рулевое серво
  steeringServo.attach(SERVO_STEER_PIN);
  steeringServo.write(90);
  delay(300);

  // Шаговый
  pinMode(STEP_IN1, OUTPUT);
  pinMode(STEP_IN2, OUTPUT);
  pinMode(STEP_IN3, OUTPUT);
  pinMode(STEP_IN4, OUTPUT);
  digitalWrite(STEP_IN1, LOW);
  digitalWrite(STEP_IN2, LOW);
  digitalWrite(STEP_IN3, LOW);
  digitalWrite(STEP_IN4, LOW);

  // Сервы камеры
  panServo.attach(SERVO_PAN_PIN);
  tiltServo.attach(SERVO_TILT_PIN);
  panServo.write(90);
  tiltServo.write(90);
  delay(300);

  Serial.println("READY");
}

// ============================================================
// ПАРСИНГ КОМАНД
// ============================================================
// S,<угол>          — рулевое серво (45..135)
// C,<pan>,<tilt>    — камера (0..180)
// M,<speed>,<dir>   — мотор (speed 0..255, dir 0/1)
// T,<steps>,<dir>   — шаговый (steps, dir 0/1)
// E                 — запрос энкодера
// R                 — сброс энкодера в 0
void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() < 2) return;

  char type = cmd.charAt(0);
  String rest = cmd.substring(2);

  if (type == 'S') {
    int angle = rest.toInt();
    angle = constrain(angle, STEER_MIN, STEER_MAX);
    smoothSteer(angle);
    Serial.print("OK S,"); Serial.println(angle);
  }
  else if (type == 'C') {
    int comma = rest.indexOf(',');
    if (comma > 0) {
      int pan  = rest.substring(0, comma).toInt();
      int tilt = rest.substring(comma + 1).toInt();
      smoothCamera(pan, tilt);
      Serial.print("OK C,"); Serial.print(pan);
      Serial.print(",");     Serial.println(tilt);
    } else {
      Serial.println("ERR C format");
    }
  }
  else if (type == 'M') {
    int comma = rest.indexOf(',');
    if (comma > 0) {
      int spd = rest.substring(0, comma).toInt();
      int dir = rest.substring(comma + 1).toInt();
      motorControl(spd, dir == 1);
      Serial.print("OK M,"); Serial.print(spd);
      Serial.print(",");     Serial.println(dir);
    } else {
      Serial.println("ERR M format");
    }
  }
  else if (type == 'T') {
    int comma = rest.indexOf(',');
    if (comma > 0) {
      int steps = rest.substring(0, comma).toInt();
      int dir   = rest.substring(comma + 1).toInt();
      stepMotor(steps, dir == 1);
      Serial.print("OK T,"); Serial.print(steps);
      Serial.print(",");     Serial.println(dir);
    } else {
      Serial.println("ERR T format");
    }
  }
  else if (type == 'E') {
    noInterrupts();
    long c = encoderCount;
    interrupts();
    Serial.print("ENC,"); Serial.println(c);
  }
  else if (type == 'R') {
    noInterrupts();
    encoderCount = 0;
    interrupts();
    Serial.println("OK R,0");
  }
  else {
    Serial.print("ERR unknown: "); Serial.println(cmd);
  }
}

// ============================================================
// LOOP
// ============================================================
void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }
}
