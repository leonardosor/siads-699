"""
Evaluate the pretrained YOLOv8n model on the testing set only.
"""

from pathlib import Path
from ultralytics import YOLO
import json

# Define paths
REPO_ROOT = Path(__file__).parent.parent
PRETRAINED_MODEL = REPO_ROOT / "models" / "pretrained" / "yolov8n.pt"
DATA_CONFIG = REPO_ROOT / "src" / "training" / "finance-image-parser.yaml"
OUTPUT_DIR = REPO_ROOT / "data" / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PRETRAINED MODEL EVALUATION ON TESTING SET")
print("=" * 80)

# Verify files exist
if not PRETRAINED_MODEL.exists():
    print(f"❌ Pretrained model not found: {PRETRAINED_MODEL}")
    exit(1)

if not DATA_CONFIG.exists():
    print(f"❌ Data config not found: {DATA_CONFIG}")
    exit(1)

print(f"\nModel: {PRETRAINED_MODEL}")
print(f"Config: {DATA_CONFIG}")
print(f"\nLoading model...")

# Load the pretrained model
model = YOLO(str(PRETRAINED_MODEL))

print("✓ Model loaded successfully")
print("\nRunning evaluation on TEST set...")
print("(This may take a few minutes depending on test set size)\n")

# Validate on test split only
results = model.val(
    data=str(DATA_CONFIG),
    split='test',
    batch=16,
    imgsz=640,
    verbose=True
)

# Extract metrics
print("\n" + "=" * 80)
print("TEST SET RESULTS - PRETRAINED MODEL")
print("=" * 80)

metrics = {
    'model': 'yolov8n.pt (pretrained)',
    'dataset_split': 'test',
    'mAP50': float(results.box.map50),
    'mAP50-95': float(results.box.map),
    'precision': float(results.box.mp),
    'recall': float(results.box.mr)
}

print(f"\nPerformance Metrics:")
print(f"  mAP50:     {metrics['mAP50']:.4f} ({metrics['mAP50']*100:.2f}%)")
print(f"  mAP50-95:  {metrics['mAP50-95']:.4f} ({metrics['mAP50-95']*100:.2f}%)")
print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")

# Performance assessment
print("\nPerformance Assessment:")
if metrics['mAP50'] > 0.95:
    print("  ⭐⭐⭐⭐⭐ EXCELLENT - Very high accuracy")
elif metrics['mAP50'] > 0.90:
    print("  ⭐⭐⭐⭐ VERY GOOD - Strong performance")
elif metrics['mAP50'] > 0.85:
    print("  ⭐⭐⭐ GOOD - Reliable performance")
elif metrics['mAP50'] > 0.75:
    print("  ⭐⭐ ACCEPTABLE - Functional performance")
else:
    print("  ⭐ NEEDS IMPROVEMENT - Low accuracy")

# Save results
output_file = OUTPUT_DIR / "pretrained_model_test_results.json"
with open(output_file, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")

# Additional per-class metrics if available
if hasattr(results.box, 'ap_class_index'):
    print("\n" + "=" * 80)
    print("PER-CLASS PERFORMANCE")
    print("=" * 80)
    
    class_names = ['header', 'body', 'footer']
    
    for i, class_name in enumerate(class_names):
        if i < len(results.box.ap50):
            ap50 = results.box.ap50[i]
            print(f"\n{class_name.upper()}:")
            print(f"  AP50: {ap50:.4f} ({ap50*100:.2f}%)")

print("\n" + "=" * 80)
print("Evaluation complete!")
print("=" * 80)
