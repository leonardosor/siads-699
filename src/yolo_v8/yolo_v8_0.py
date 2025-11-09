# Fine tuning script

# Importing packages
from ultralytics import YOLO
 
# Load the model
model = YOLO('/workspace/src/yolo_v8/models/yolov8n.pt')
 
# Fine tuning (and traning)
results = model.train(
   data='finance-image-parser.yaml',
   imgsz=850,
   epochs=5,
   batch=4,
   name='finance-image-parser'
)