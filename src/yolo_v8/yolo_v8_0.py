# Fine tuning script

# Importing packages
from ultralytics import YOLO
 
# Load the model
model = YOLO('/workspace/src/yolov8/models/yolov8n.pt')
 
# Fine tuning (and traning)
results = model.train(
   data='finance-image-parser.yaml',
   imgsz=640,
   epochs=50,
   batch=8,
   name='finance-image-parser'
)