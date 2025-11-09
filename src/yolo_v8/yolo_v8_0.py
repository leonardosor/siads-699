# Fine tuning script
# --- Data integrity check ---
import cv2, os, glob
from ultralytics import YOLO
from pathlib import Path

image_dir = "/home/joehiggi/siads-699/data/input/training/images"
for f in glob.glob(os.path.join(image_dir, "*.jpg")):
    img = cv2.imread(f)
    if img is None:
        print(f"Removing unreadable: {f}")
        os.remove(f)

# Load the model
# model = YOLO('/workspace/src/yolo_v8/models/yolov8n.pt')
 
# Pointing to model
home = str(Path.home())
model = YOLO(f'{home}/siads-699/src/yolo_v8/models/yolov8n.pt')

# Fine tuning (and traning)
results = model.train(
   data=f'{home}/siads-699/src/yolo_v8/finance-image-parser.yaml',
   imgsz=850,
   epochs=3,
   batch=4,
   name='finance-image-parser'
)
<<<<<<< HEAD
=======


>>>>>>> ad576b537cedccdee05b1656d72db5f3d9329e8c
