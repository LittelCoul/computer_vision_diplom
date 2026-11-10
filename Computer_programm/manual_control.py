import socket
import cv2
import numpy as np
import keyboard
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk

ORANGE_PI_IP = '192.168.1.38'
PORT = 5000

# ============ ПОДКЛЮЧЕНИЕ ============
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client.connect((ORANGE_PI_IP, PORT))
    print(f"Подключено к Orange Pi {ORANGE_PI_IP}:{PORT}")
except Exception as e:
    print(f"Ошибка подключения: {e}")
    exit(1)

stop_event = threading.Event()

# Углы камеры
cam_pan = 90
cam_tilt = 90
CAM_STEP = 10

# Режим: 'manual' или 'auto'
current_mode = 'manual'


def send_command(cmd):
    try:
        client.sendall(cmd.encode())
    except Exception as e:
        print(f"Ошибка отправки: {e}")


# ============ ПРИЁМ ВИДЕО ============
latest_frame = None
frame_lock = threading.Lock()


def receive_video():
    global latest_frame
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
                with frame_lock:
                    latest_frame = frame
        except socket.timeout:
            continue
        except OSError:
            break
        except Exception as e:
            print(f"Ошибка видео (продолжаем): {e}")
            time.sleep(0.1)
            continue


threading.Thread(target=receive_video, daemon=True).start()


# ============ АВТОМАТИЧЕСКИЙ РЕЖИМ ============
def auto_mode_loop():
    """Простая заготовка авторежима: едет вперёд, если перед камерой нет крупного объекта.
    Замените логику на свою (объезд препятствий, слежение за объектом и т.д.)."""
    while not stop_event.is_set():
        if current_mode != 'auto':
            time.sleep(0.2)
            continue

        # Заготовка: едем вперёд
        send_command('W')
        time.sleep(0.5)
        send_command('Z')
        time.sleep(0.5)


threading.Thread(target=auto_mode_loop, daemon=True).start()


# ============ РУЧНОЕ УПРАВЛЕНИЕ ============
def manual_control_loop():
    last_state = {'W': False, 'S': False, 'A': False, 'D': False}
    last_q = last_e = last_z = last_c = False
    last_8 = last_2 = last_4 = last_6 = False

    while not stop_event.is_set():
        if current_mode != 'manual':
            time.sleep(0.05)
            continue

        w = keyboard.is_pressed('w')
        s = keyboard.is_pressed('s')
        if w and not last_state['W']:
            send_command('W')
        if not w and last_state['W']:
            send_command('Z')
        last_state['W'] = w

        if s and not last_state['S']:
            send_command('S')
        if not s and last_state['S']:
            send_command('Z')
        last_state['S'] = s

        a = keyboard.is_pressed('a')
        d = keyboard.is_pressed('d')
        if a and not last_state['A']:
            send_command('A')
        last_state['A'] = a
        if d and not last_state['D']:
            send_command('D')
        last_state['D'] = d

        q = keyboard.is_pressed('q')
        if q and not last_q:
            send_command('Q')
        last_q = q

        e = keyboard.is_pressed('e')
        if e and not last_e:
            send_command('E')
        last_e = e

        z = keyboard.is_pressed('z')
        if z and not last_z:
            send_command('Z')
        last_z = z

        c = keyboard.is_pressed('c')
        if c and not last_c:
            send_command('C')
        last_c = c

        global cam_pan, cam_tilt

        k8 = keyboard.is_pressed('8')
        if k8 and not last_8:
            cam_tilt = max(0, cam_tilt - CAM_STEP)
            send_command(f'CAM,{cam_pan},{cam_tilt}')
        last_8 = k8

        k2 = keyboard.is_pressed('2')
        if k2 and not last_2:
            cam_tilt = min(180, cam_tilt + CAM_STEP)
            send_command(f'CAM,{cam_pan},{cam_tilt}')
        last_2 = k2

        k6 = keyboard.is_pressed('6')
        if k6 and not last_6:
            cam_pan = min(180, cam_pan + CAM_STEP)
            send_command(f'CAM,{cam_pan},{cam_tilt}')
        last_6 = k6

        k4 = keyboard.is_pressed('4')
        if k4 and not last_4:
            cam_pan = max(0, cam_pan - CAM_STEP)
            send_command(f'CAM,{cam_pan},{cam_tilt}')
        last_4 = k4

        time.sleep(0.02)


threading.Thread(target=manual_control_loop, daemon=True).start()


# ============ GUI ============
root = tk.Tk()
root.title("Управление роботом")
root.geometry("900x600")

# Левая часть — видео
video_label = tk.Label(root, bg='black')
video_label.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

# Правая часть — панель управления
panel = tk.Frame(root, width=220, bg='#f0f0f0')
panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
panel.pack_propagate(False)

mode_label = tk.Label(panel, text="Режим: РУЧНОЙ", font=('Arial', 12, 'bold'), bg='#f0f0f0')
mode_label.pack(pady=(10, 20))


def set_manual():
    global current_mode
    current_mode = 'manual'
    mode_label.config(text="Режим: РУЧНОЙ", fg='black')
    send_command('Z')  # на всякий случай стоп


def set_auto():
    global current_mode
    current_mode = 'auto'
    mode_label.config(text="Режим: АВТО", fg='blue')


btn_manual = tk.Button(panel, text="Ручное управление", command=set_manual,
                       width=20, height=2, font=('Arial', 11))
btn_manual.pack(pady=5)

btn_auto = tk.Button(panel, text="Автоматическое управление", command=set_auto,
                     width=20, height=2, font=('Arial', 11))
btn_auto.pack(pady=5)

hint = tk.Label(panel,
                text=("Ручной режим:\n"
                      "W/S — вперёд/назад\n"
                      "A/D — колёса влево/вправо\n"
                      "Z — стоп, C — прямо\n"
                      "Q/E — шаговый вперёд/назад\n"
                      "8/2 — камера вверх/вниз\n"
                      "4/6 — камера влево/вправо"),
                justify=tk.LEFT, bg='#f0f0f0', font=('Arial', 9))
hint.pack(pady=20, anchor='w')


def update_frame():
    if not stop_event.is_set():
        with frame_lock:
            frame = latest_frame
        if frame is not None:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((640, 480), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        root.after(30, update_frame)


def on_close():
    stop_event.set()
    try:
        send_command('Z')
    except Exception:
        pass
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)
update_frame()
root.mainloop()


# ============ ЗАВЕРШЕНИЕ ============
stop_event.set()
try:
    client.shutdown(socket.SHUT_RDWR)
except Exception:
    pass
try:
    client.close()
except Exception:
    pass
cv2.destroyAllWindows()
print("Программа завершена.")