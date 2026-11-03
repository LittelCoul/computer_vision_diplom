import socket
import cv2
import numpy as np
import keyboard
import threading
import time

# --- Настройки ---
ORANGE_PI_IP = '192.168.1.38'  # <-- ЗАМЕНИТЕ НА IP ВАШЕГО ORANGE PI
PORT = 5000

# --- Подключение к Orange Pi ---
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client.connect((ORANGE_PI_IP, PORT))
    print(f"Подключено к Orange Pi {ORANGE_PI_IP}:{PORT}")
except Exception as e:
    print(f"Ошибка подключения: {e}")
    exit(1)


# --- Функция для отправки команд ---
def send_command(cmd):
    try:
        client.sendall(cmd.encode())
    except Exception as e:
        print(f"Ошибка отправки: {e}")


# --- Поток для приема видео ---
def receive_video():
    while True:
        try:
            # Читаем размер кадра (4 байта)
            raw_size = client.recv(4)
            if not raw_size:
                break
            size = int.from_bytes(raw_size, 'big')

            # Читаем сам кадр
            data = b''
            while len(data) < size:
                packet = client.recv(size - len(data))
                if not packet:
                    break
                data += packet

            # Декодируем и показываем
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow('Orange Pi Camera', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Ошибка видео: {e}")
            break
    cv2.destroyAllWindows()


# Запускаем поток видео
video_thread = threading.Thread(target=receive_video, daemon=True)
video_thread.start()

print("Управление: W-вперед, S-назад, A-влево, D-вправо, Q-шаговый вперед, E-шаговый назад")
print("Нажмите ESC для выхода.")

# --- Основной цикл обработки клавиш ---
try:
    while True:
        if keyboard.is_pressed('w'):
            send_command('W')
            time.sleep(0.1)  # Небольшая задержка, чтобы не спамить
        elif keyboard.is_pressed('s'):
            send_command('S')
            time.sleep(0.1)
        elif keyboard.is_pressed('a'):
            send_command('A')
            time.sleep(0.1)
        elif keyboard.is_pressed('d'):
            send_command('D')
            time.sleep(0.1)
        elif keyboard.is_pressed('q'):
            send_command('Q')
            time.sleep(0.5)  # Шаговый мотор медленный, даем время
        elif keyboard.is_pressed('e'):
            send_command('E')
            time.sleep(0.5)
        elif keyboard.is_pressed('esc'):
            print("Выход...")
            break
except KeyboardInterrupt:
    pass
finally:
    client.close()
    cv2.destroyAllWindows()