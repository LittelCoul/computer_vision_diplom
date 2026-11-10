from ultralytics import YOLO
import os

# Дополнительные установки для перехода на GPU (если нужно)
# CUDA: https://developer.nvidia.com/cuda-12-4-0-download-archive
# pip uninstall torch torchvision -y
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'{project_path=}')

# Определяем классы: сначала COCO классы, которые хотим сохранить, затем свои
# Номера классов должны соответствовать их оригинальным номерам в COCO (0-79)
# COCO class IDs for truck, cat, dog: 7, 15, 16
# Наши новые классы будут с ID 80, 81, 82 (после стандартных 80 классов COCO)

yaml_text = f'''
train: {project_path}/dataset/images/train
val: {project_path}/dataset/images/val

names:
  0: person  
  1: bicycle  
  2: car  
  3: motorbike  
  4: aeroplane  
  5: bus  
  6: train  
  7: truck  
  8: boat  
  9: traffic light  
  10: fire hydrant  
  11: stop sign  
  12: parking meter  
  13: bench  
  14: bird  
  15: cat  
  16: dog  
  17: horse  
  18: sheep  
  19: cow  
  20: elephant  
  21: bear  
  22: zebra  
  23: giraffe  
  24: backpack  
  25: umbrella  
  26: handbag  
  27: tie  
  28: suitcase  
  29: frisbee  
  30: skis  
  31: snowboard  
  32: sports ball  
  33: kite  
  34: baseball bat  
  35: baseball glove  
  36: skateboard  
  37: surfboard  
  38: tennis racket  
  39: bottle  
  40: wine glass  
  41: cup  
  42: fork  
  43: knife  
  44: spoon  
  45: bowl  
  46: banana  
  47: apple  
  48: sandwich  
  49: orange  
  50: broccoli  
  51: carrot  
  52: hot dog  
  53: pizza  
  54: donut  
  55: cake  
  56: chair  
  57: sofa  
  58: pottedplant  
  59: bed  
  60: diningtable  
  61: toilet  
  62: tvmonitor  
  63: laptop  
  64: mouse  
  65: remote  
  66: keyboard  
  67: cell phone  
  68: microwave  
  69: oven  
  70: toaster  
  71: sink  
  72: refrigerator  
  73: book  
  74: clock  
  75: vase  
  76: scissors  
  77: teddy bear  
  78: hair drier  
  79: toothbrush
  80: pallet
  81: load_pallet
  82: pallet_front

nc: 83  # общее количество классов (80 COCO + 3 новых)
'''

# Загрузка предобученной модели YOLOv8n
model = YOLO("../models/yolov8n/yolov8n.pt")

with open("data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_text)

# Важно: при дообучении с новыми классами нужно разморозить слои модели
# Также убедитесь, что ваши аннотации используют правильные ID классов
results = model.train(
    data="data.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    device="cpu",  # "cpu" или "0" (GPU)
    project=project_path+'/models/run',
    # Добавляем параметры для дообучения с новыми классами
    freeze=[0],  # Размораживаем все слои (можно указать конкретные слои)
    pretrained=True,
    optimizer='auto',
    # lr0=0.01,  # Начальная скорость обучения
    # lrf=0.01,  # Конечная скорость обучения
    # weight_decay=0.0005,
    # warmup_epochs=3,
    # warmup_momentum=0.8,
    # box=7.5,  # Вес для loss по bounding box
    # cls=0.5,  # Вес для loss по классификации (уменьшаем, так как добавляем новые классы)
    # dfl=1.5,  # Вес для Distribution Focal Loss
)


# Важные замечания:
#
# 1. В файле `data.yaml` мы сохраняем только нужные классы из COCO (truck=7, cat=15, dog=16) и добавляем новые (80, 81, 82).
#
# 2. Параметр `nc` (number of classes) должен быть равен общему количеству классов (в данном случае 83: 80 оригинальных + 3 новых).
#
# 3. При подготовке датасета:
#    - Для изображений с объектами из COCO используйте оригинальные ID классов (7, 15, 16)
#    - Для новых классов используйте ID 80, 81, 82
#
# 4. Для лучшего дообучения рекомендуется:
#    - Использовать аугментации
#    - Настроить параметры обучения (learning rate, weight decay и т.д.)
#    - Возможно, заморозить часть слоев в начале обучения
#
# 5. Если вы хотите полностью заменить все классы COCO на свои, подход будет немного другим (нужно будет модифицировать архитектуру модели).