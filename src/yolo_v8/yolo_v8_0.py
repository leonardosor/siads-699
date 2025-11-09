# Fine tuning script

# Importing packages
from ultralytics import YOLO
 
# Load the model
model = YOLO('/workspace/src/yolo_v8/models/yolov8n.pt')
 
# Fine tuning (and traning)
results = model.train(
   data='finance-image-parser.yaml',
   imgsz=850,
   epochs=50,
   batch=8,
   name='finance-image-parser'
)