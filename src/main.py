import cv2
import numpy as np
from ultralytics import YOLO
from pyzbar.pyzbar import decode  # Для распознавания QR-кодов

train_dir_name = 'train'
model_path = f"../models/run/{train_dir_name}/weights/best.pt"
classes_path = "../models/pallet.names"


def load_model(model_path):
    model = YOLO(model_path)
    return model


def detect_objects(frame, model, conf_threshold=0.1):
    results = model(frame, verbose=False)
    boxes = []
    confidences = []
    class_ids = []

    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())

            if confidence > conf_threshold:
                boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
                confidences.append(confidence)
                class_ids.append(class_id)

    return boxes, confidences, class_ids, np.arange(len(boxes))


def draw_predictions(frame, boxes, confidences, class_ids, indices, classes):
    for i in indices:
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def detect_qr_codes(frame):
    # Распознаем QR-коды
    qr_codes = decode(frame)
    qr_info = []

    for qr in qr_codes:
        # Получаем данные и координаты QR-кода
        data = qr.data.decode('utf-8')
        points = qr.polygon

        # Если QR-код имеет 4 угла, рисуем его контур
        if len(points) == 4:
            pts = np.array([(point.x, point.y) for point in points], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        # Выводим текст с данными QR-кода
        x, y = qr.rect.left, qr.rect.top
        cv2.putText(frame, data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        qr_info.append(data)

    return qr_info


def main():
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    model = load_model(model_path)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция объектов YOLOv8
        boxes, confidences, class_ids, indices = detect_objects(frame, model)
        draw_predictions(frame, boxes, confidences, class_ids, indices, classes)

        # Детекция QR-кодов
        qr_info = detect_qr_codes(frame)
        if qr_info:
            print("Найденные QR-коды:", qr_info)

        cv2.imshow("Object and QR Code Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()