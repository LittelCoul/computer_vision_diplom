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

try:
    ser = serial.Serial(ARDUINO_PORT, BAUD, timeout=1)
    time.sleep(2)
    print(f"Arduino подключен к {ARDUINO_PORT}")
except Exception as e:
    print(f"Ошибка подключения к Arduino: {e}")
    exit(1)

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
    while True:
        try:
            data = client_socket.recv(1024).decode().strip()
            if not data:
                break
            print(f"<- Компьютер: {data}")

            if data == 'W':          # Вперёд (инвертировано)
                send_to_arduino("M,200,0")
            elif data == 'S':        # Назад (инвертировано)
                send_to_arduino("M,200,1")
            elif data == 'A':        # Колёса влево
                send_to_arduino("S,135")
            elif data == 'D':        # Колёса вправо
                send_to_arduino("S,45")
            elif data == 'Q':        # Шаговый вперёд
                send_to_arduino("T,200,1")
            elif data == 'E':        # Шаговый назад
                send_to_arduino("T,200,0")
            elif data == 'Z':        # СТОП колёс
                send_to_arduino("M,0,1")
            elif data == 'C':        # Колёса прямо
                send_to_arduino("S,90")
            elif data == 'STOP':     # Экстренная остановка
                send_to_arduino("M,0,1")
        except Exception as e:
            print(f"Ошибка клиента: {e}")
            break
    client_socket.close()

def stream_video(client_socket):
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            data = buffer.tobytes()
            client_socket.sendall(len(data).to_bytes(4, 'big'))
            client_socket.sendall(data)
    except Exception as e:
        print(f"Ошибка видео: {e}")
    finally:
        client_socket.close()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
print(f"Сервер запущен на {HOST}:{PORT}. Ожидание подключения...")

client_sock, addr = server.accept()
print(f"Подключен клиент: {addr}")

threading.Thread(target=handle_client, args=(client_sock,), daemon=True).start()
stream_video(client_sock)