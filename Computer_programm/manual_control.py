import socket
import cv2
import numpy as np
import keyboard
import threading
import time

ORANGE_PI_IP = '192.168.1.38'  # <-- ЗАМЕНИТЕ НА IP ORANGE PI
PORT = 5000

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client.connect((ORANGE_PI_IP, PORT))
    print(f"Подключено к Orange Pi {ORANGE_PI_IP}:{PORT}")
except Exception as e:
    print(f"Ошибка подключения: {e}")
    exit(1)

def send_command(cmd):
    try:
        client.sendall(cmd.encode())
    except Exception as e:
        print(f"Ошибка отправки: {e}")

def receive_video():
    while True:
        try:
            raw_size = client.recv(4)
            if not raw_size:
                break
            size = int.from_bytes(raw_size, 'big')
            data = b''
            while len(data) < size:
                packet = client.recv(size - len(data))
                if not packet:
                    break
                data += packet
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow('Orange Pi Camera', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Ошибка видео: {e}")
            break
    cv2.destroyAllWindows()

threading.Thread(target=receive_video, daemon=True).start()

print("Управление: W-вперёд, S-назад, A-влево, D-вправо,")
print("Q-шаговый вперёд, E-шаговый назад, Z-стоп колёс, C-колёса прямо")
print("Нажмите ESC для выхода.")

# Отслеживаем состояние клавиш, чтобы не спамить
last_state = {'W': False, 'S': False, 'A': False, 'D': False}

try:
    while True:
        # --- W / S с автостопом при отпускании ---
        w_pressed = keyboard.is_pressed('w')
        s_pressed = keyboard.is_pressed('s')

        if w_pressed and not last_state['W']:
            send_command('W')
        if not w_pressed and last_state['W']:
            send_command('Z')   # отпустили W — стоп
        last_state['W'] = w_pressed

        if s_pressed and not last_state['S']:
            send_command('S')
        if not s_pressed and last_state['S']:
            send_command('Z')   # отпустили S — стоп
        last_state['S'] = s_pressed

        # --- A / D (однократно при нажатии) ---
        a_pressed = keyboard.is_pressed('a')
        d_pressed = keyboard.is_pressed('d')

        if a_pressed and not last_state['A']:
            send_command('A')
        last_state['A'] = a_pressed

        if d_pressed and not last_state['D']:
            send_command('D')
        last_state['D'] = d_pressed

        # --- Q / E / Z / C (однократно) ---
        if keyboard.is_pressed('q'):
            send_command('Q')
            time.sleep(0.5)
        if keyboard.is_pressed('e'):
            send_command('E')
            time.sleep(0.5)
        if keyboard.is_pressed('z'):
            send_command('Z')
            time.sleep(0.2)
        if keyboard.is_pressed('c'):
            send_command('C')
            time.sleep(0.2)

        if keyboard.is_pressed('esc'):
            print("Выход...")
            break

        time.sleep(0.02)

except KeyboardInterrupt:
    pass
finally:
    send_command('Z')  # на всякий случай стоп при выходе
    client.close()
    cv2.destroyAllWindows()