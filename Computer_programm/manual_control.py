import socket
import cv2
import numpy as np
import keyboard
import threading
import time

ORANGE_PI_IP = '192.168.1.38'
PORT = 5000

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client.connect((ORANGE_PI_IP, PORT))
    print(f"Подключено к Orange Pi {ORANGE_PI_IP}:{PORT}")
except Exception as e:
    print(f"Ошибка подключения: {e}")
    exit(1)

# Флаг остановки для потока видео
stop_event = threading.Event()

# Текущие углы камеры (pan, tilt)
cam_pan = 90
cam_tilt = 90
CAM_STEP = 10  # шаг поворота камеры в градусах


def send_command(cmd):
    try:
        client.sendall(cmd.encode())
    except Exception as e:
        print(f"Ошибка отправки: {e}")


def receive_video():
    while not stop_event.is_set():
        try:
            client.settimeout(1.0)
            raw_size = client.recv(4)
            if not raw_size:
                print("Соединение закрыто сервером")
                break
            size = int.from_bytes(raw_size, 'big')
            if size <= 0 or size > 10_000_000:
                print(f"Некорректный размер кадра: {size}")
                break

            data = b''
            while len(data) < size and not stop_event.is_set():
                packet = client.recv(size - len(data))
                if not packet:
                    break
                data += packet

            if len(data) != size:
                continue

            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow('Orange Pi Camera', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    stop_event.set()
                    break
        except socket.timeout:
            continue
        except OSError:
            break
        except Exception as e:
            print(f"Ошибка видео (продолжаем): {e}")
            time.sleep(0.1)
            continue


video_thread = threading.Thread(target=receive_video, daemon=True)
video_thread.start()

print("Управление:")
print("  W/S — вперёд/назад (автостоп при отпускании)")
print("  A/D — колёса влево/вправо")
print("  Z   — стоп колёс")
print("  C   — колёса прямо")
print("  Q/E — шаговый мотор вперёд/назад")
print("  8/2 — камера вверх/вниз")
print("  4/6 — камера влево/вправо")
print("  ESC — выход")

last_state = {'W': False, 'S': False, 'A': False, 'D': False}
last_q = last_e = last_z = last_c = False
last_8 = last_2 = last_4 = last_6 = False

try:
    while not stop_event.is_set():
        # --- W / S с автостопом ---
        w_pressed = keyboard.is_pressed('w')
        s_pressed = keyboard.is_pressed('s')

        if w_pressed and not last_state['W']:
            send_command('W')
        if not w_pressed and last_state['W']:
            send_command('Z')
        last_state['W'] = w_pressed

        if s_pressed and not last_state['S']:
            send_command('S')
        if not s_pressed and last_state['S']:
            send_command('Z')
        last_state['S'] = s_pressed

        # --- A / D ---
        a_pressed = keyboard.is_pressed('a')
        d_pressed = keyboard.is_pressed('d')
        if a_pressed and not last_state['A']:
            send_command('A')
        last_state['A'] = a_pressed
        if d_pressed and not last_state['D']:
            send_command('D')
        last_state['D'] = d_pressed

        # --- Q / E / Z / C ---
        q_pressed = keyboard.is_pressed('q')
        if q_pressed and not last_q:
            send_command('Q')
        last_q = q_pressed

        e_pressed = keyboard.is_pressed('e')
        if e_pressed and not last_e:
            send_command('E')
        last_e = e_pressed

        z_pressed = keyboard.is_pressed('z')
        if z_pressed and not last_z:
            send_command('Z')
        last_z = z_pressed

        c_pressed = keyboard.is_pressed('c')
        if c_pressed and not last_c:
            send_command('C')
        last_c = c_pressed

        # --- Камера: 8/2/6/4 ---
        k8 = keyboard.is_pressed('8')
        if k8 and not last_8:
            cam_tilt = max(0, cam_tilt - CAM_STEP)   # вверх
            send_command(f'CAM,{cam_pan},{cam_tilt}')
        last_8 = k8

        k2 = keyboard.is_pressed('2')
        if k2 and not last_2:
            cam_tilt = min(180, cam_tilt + CAM_STEP)  # вниз
            send_command(f'CAM,{cam_pan},{cam_tilt}')
        last_2 = k2

        k6 = keyboard.is_pressed('6')
        if k6 and not last_6:
            cam_pan = min(180, cam_pan + CAM_STEP)    # вправо
            send_command(f'CAM,{cam_pan},{cam_tilt}')
        last_6 = k6

        k4 = keyboard.is_pressed('4')
        if k4 and not last_4:
            cam_pan = max(0, cam_pan - CAM_STEP)      # влево
            send_command(f'CAM,{cam_pan},{cam_tilt}')
        last_4 = k4

        # --- Выход ---
        if keyboard.is_pressed('esc'):
            print("Выход...")
            stop_event.set()
            break

        time.sleep(0.02)

except KeyboardInterrupt:
    stop_event.set()

finally:
    try:
        send_command('Z')
    except Exception:
        pass

    stop_event.set()
    video_thread.join(timeout=2.0)

    try:
        client.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        client.close()
    except Exception:
        pass

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("Программа завершена.")