# Fine tuning script
# --- Data integrity check ---
import glob
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

DATASET_ROOT = Path(
    os.environ.get(
        "FINANCE_DATASET_ROOT",
        Path(__file__).resolve().parents[3] / "models" / "training-kit" / "data" / "input",
    )
)
image_dir = DATASET_ROOT / "training" / "images"

for f in glob.glob(str(image_dir / "*.jpg")):
    img = cv2.imread(f)
    if img is None:
        print(f"Removing unreadable: {f}")
        os.remove(f)

home = Path.home()
model = YOLO(str(home / "siads-699" / "src" / "yolo_v8" / "models" / "yolov8n.pt"))

results = model.train(
    data=str(Path(__file__).with_name("finance-image-parser.yaml")),
    imgsz=850,
    epochs=3,
    batch=4,
    name="finance-image-parser",
)
