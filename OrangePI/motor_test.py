import serial
import time
import sys

PORT = '/dev/ttyACM0'
BAUD = 9600

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # Arduino перезагружается при открытии порта
    print(f"Подключено к {PORT}")
except Exception as e:
    print(f"Ошибка подключения: {e}")
    sys.exit(1)

def send(cmd, wait=0.1):
    """Отправить команду и прочитать ответ Arduino."""
    ser.write((cmd + "\n").encode())
    time.sleep(wait)
    if ser.in_waiting:
        resp = ser.readline().decode(errors='ignore').strip()
        print(f"→ {cmd:20s} ← {resp}")
        return resp
    print(f"→ {cmd:20s} ← (нет ответа)")
    return None

# ---------- Обёртки ----------

def steer(angle):
    """45 = право, 90 = прямо, 135 = лево."""
    angle = max(45, min(135, angle))
    return send(f"S,{angle}")

def camera(pan, tilt):
    """Плавное движение камеры."""
    pan  = max(0, min(180, pan))
    tilt = max(0, min(180, tilt))
    return send(f"C,{pan},{tilt}", wait=2.0)  # плавно, ждём дольше

def motor(speed, forward):
    """speed 0..255, forward True/False."""
    speed = max(0, min(255, speed))
    return send(f"M,{speed},{1 if forward else 0}", wait=0.2)

def step(steps, forward):
    """Шаговый двигатель: steps, направление."""
    return send(f"T,{steps},{1 if forward else 0}", wait=0.2)

def encoder_read():
    return send("E")

def encoder_reset():
    return send("R")

# ============================================================
# ТЕСТ ВСЕХ СИСТЕМ
# ============================================================

print("\n=== 1. РУЛЕВОЕ СЕРВО (45..135) ===")
steer(90);  time.sleep(0.5)   # прямо
steer(45);  time.sleep(0.5)   # право
steer(135); time.sleep(0.5)   # лево
steer(90);  time.sleep(0.5)   # прямо

print("\n=== 2. КАМЕРА PAN/TILT ===")
camera(90, 90);    time.sleep(0.5)  # центр
camera(45, 120);   time.sleep(0.5)  # влево-вниз
camera(135, 60);   time.sleep(0.5)  # вправо-вверх
camera(90, 90);    time.sleep(0.5)  # центр

print("\n=== 3. МОТОР ДВИЖЕНИЯ + ЭНКОДЕР ===")
encoder_reset()
motor(200, True)    # вперёд
time.sleep(3)
motor(0, True)      # стоп
encoder_read()

motor(200, False)   # назад
time.sleep(3)
motor(0, False)     # стоп
encoder_read()

print("\n=== 4. ШАГОВЫЙ ДВИГАТЕЛЬ ===")
step(200, True)     # вперёд
time.sleep(0.5)
step(200, False)    # назад
time.sleep(0.5)

print("\n=== ТЕСТ ЗАВЕРШЁН ===")
ser.close()