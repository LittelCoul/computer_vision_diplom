import socket
import serial
import cv2
import numpy as np
import threading
import time

HOST = '0.0.0.0'
PORT = 5000
ARDUINO_PORT = '/dev/ttyACM0'
BAUD = 9600

# --- Подключение к Arduino ---
try:
    ser = serial.Serial(ARDUINO_PORT, BAUD, timeout=1)
    time.sleep(2)
    print(f"Arduino подключен к {ARDUINO_PORT}")
except Exception as e:
    print(f"Ошибка подключения к Arduino: {e}")
    exit(1)

# --- Инициализация камеры ---
cap = cv2.VideoCapture('/dev/video1', cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit(1)


def send_to_arduino(cmd):
    try:
        ser.write((cmd + '\n').encode())
        print(f"-> Arduino: {cmd}")
    except Exception as e:
        print(f"Ошибка отправки: {e}")


def handle_client(client_socket):
    """Принимает команды от компьютера. Сокет НЕ закрывает — это делает stream_video."""
    while True:
        try:
            data = client_socket.recv(1024).decode().strip()
            if not data:
                break
            print(f"<- Компьютер: {data}")

            if data == 'W':
                send_to_arduino("M,200,0")
            elif data == 'S':
                send_to_arduino("M,200,1")
            elif data == 'A':
                send_to_arduino("S,135")
            elif data == 'D':
                send_to_arduino("S,45")
            elif data == 'Q':
                send_to_arduino("T,200,1")
            elif data == 'E':
                send_to_arduino("T,200,0")
            elif data == 'Z':
                send_to_arduino("M,0,1")
            elif data == 'C':
                send_to_arduino("S,90")
            elif data.startswith('CAM,'):
                try:
                    parts = data.split(',')
                    pan = int(parts[1])
                    tilt = int(parts[2])
                    pan = max(0, min(180, pan))
                    tilt = max(0, min(180, tilt))
                    send_to_arduino(f"C,{pan},{tilt}")
                except (ValueError, IndexError):
                    print(f"Некорректная команда камеры: {data}")
            elif data == 'STOP':
                send_to_arduino("M,0,1")
        except Exception as e:
            print(f"Ошибка клиента: {e}")
            break
    print("Поток команд завершён.")


def stream_video(client_socket):
    """Отправляет видео. Именно этот поток закрывает сокет в finally."""
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            data = buffer.tobytes()
            client_socket.sendall(len(data).to_bytes(4, 'big'))
            client_socket.sendall(data)
    except (BrokenPipeError, ConnectionResetError, OSError) as e:
        print(f"Соединение потеряно: {e}")
    except Exception as e:
        print(f"Ошибка видео: {e}")
    finally:
        try:
            client_socket.close()
            print("Сокет закрыт.")
        except Exception:
            pass


# --- Сервер ---
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
print(f"Сервер запущен на {HOST}:{PORT}. Ожидание подключения...")

client_sock, addr = server.accept()
print(f"Подключен клиент: {addr}")

threading.Thread(target=handle_client, args=(client_sock,), daemon=True).start()
stream_video(client_sock)