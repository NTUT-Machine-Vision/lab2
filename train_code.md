# YOLO

## Train 1 : default

``` python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
path_group2 = "/home/ntut/CV_NTUT/Group2/datasets/pdfdataset.yaml"
results = model.train(data=path_group2, epochs=1000, imgsz=640, plots=True, save=True, name='yolo11n_1000epochs_original')
```

path = "/home/ntut/CV_NTUT/Group2/runs/detect/yolo11n_1000epochs_original/weights/best.pt"

## Train 2 : default + large batch size

``` python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
path_group2 = "/home/ntut/CV_NTUT/Group2/datasets/pdfdataset.yaml"
results = model.train(data=path_group2, epochs=1000, imgsz=640, batch=100, plots=True, save=True, name='yolo11n_1000epochs_batch100')
```

path = "/home/ntut/CV_NTUT/Group2/runs/detect/yolo11n_1000epochs_batch100/weights/best.pt"

## Train 3 : default + AdamW

``` python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
path_group2 = "/home/ntut/CV_NTUT/Group2/datasets/pdfdataset.yaml"
results = model.train(data=path_group2, epochs=1000, imgsz=640, optimizer='AdamW', plots=True, save=True, name='yolo11n_1000epochs_adamw')
```

path = "/home/ntut/CV_NTUT/Group2/runs/detect/yolo11n_1000epochs_adamw/weights/best.pt"

# RTDETR

``` python 
from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")

# Display model information (optional)
model.info()


data_path = "/home/ntut/CV_NTUT/Group2/datasets/pdfdataset.yaml"

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data=data_path, epochs=1000, imgsz=640, plots=True, save=True, name='rtdetr_1000epochs_original')
```

path = "/home/ntut/CV_NTUT/Group2/rtdetr/runs/detect/rtdetr_1000epochs_original/weights/best.pt"