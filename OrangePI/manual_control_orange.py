import socket
import serial
import cv2
import numpy as np
import threading
import time

# --- Настройки ---
HOST = '0.0.0.0'  # Слушать все интерфейсы
PORT = 5000
ARDUINO_PORT = '/dev/ttyACM0'  # Ваш порт Arduino
BAUD = 9600

# --- Подключение к Arduino ---
try:
    ser = serial.Serial(ARDUINO_PORT, BAUD, timeout=1)
    time.sleep(2)  # Дать Arduino перезагрузиться
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


# --- Функция для отправки команд в Arduino ---
def send_to_arduino(cmd):
    try:
        ser.write((cmd + '\n').encode())
        print(f"-> Arduino: {cmd}")
    except Exception as e:
        print(f"Ошибка отправки: {e}")


# --- Функция для обработки команд от компьютера ---
def handle_client(client_socket):
    while True:
        try:
            data = client_socket.recv(1024).decode().strip()
            if not data:
                break
            print(f"<- Компьютер: {data}")

            # Маппинг команд с клавиатуры на команды Arduino
            if data == 'W':  # Вперед
                send_to_arduino("M,200,1")
            elif data == 'S':  # Назад
                send_to_arduino("M,200,0")
            elif data == 'A':  # Колеса влево
                send_to_arduino("S,135")  # 135 = лево в вашем коде
            elif data == 'D':  # Колеса вправо
                send_to_arduino("S,45")  # 45 = право
            elif data == 'Q':  # Шаговый вперед
                send_to_arduino("T,200,1")
            elif data == 'E':  # Шаговый назад
                send_to_arduino("T,200,0")
            elif data == 'STOP':  # Экстренная остановка
                send_to_arduino("M,0,1")
        except Exception as e:
            print(f"Ошибка клиента: {e}")
            break
    client_socket.close()


# --- Функция для отправки видео ---
def stream_video(client_socket):
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Кодируем кадр в JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            data = buffer.tobytes()
            # Отправляем размер кадра, затем сам кадр
            client_socket.sendall(len(data).to_bytes(4, 'big'))
            client_socket.sendall(data)
    except Exception as e:
        print(f"Ошибка видео: {e}")
    finally:
        client_socket.close()


# --- Основной цикл сервера ---
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
print(f"Сервер запущен на {HOST}:{PORT}. Ожидание подключения...")

client_sock, addr = server.accept()
print(f"Подключен клиент: {addr}")

# Запускаем обработку команд и видео в отдельных потоках
threading.Thread(target=handle_client, args=(client_sock,), daemon=True).start()
stream_video(client_sock)