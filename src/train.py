from ultralytics import YOLO
import os

# дополнительные установки для перехода на GPU
# CUDA: https://developer.nvidia.com/cuda-12-4-0-download-archive
# pip uninstall torch torchvision -y
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'{project_path=}')

yaml_text = f'''
train: {project_path}/dataset/images/train
val: {project_path}/dataset/images/val

names:
  0: pallet
  1: load_pallet
  2: pallet_front
'''

# Загрузка предобученной модели YOLOv8n
model = YOLO("../models/yolov8n/yolov8n.pt")

with open("data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_text)

results = model.train(
    data="data.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    device="cpu",  # "cpu" или "0" (GPU)
    project=project_path+'/models/run'
)