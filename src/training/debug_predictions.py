#!/usr/bin/env python3
"""
Debug script to see raw model predictions regardless of threshold.
"""
import sys
import tempfile
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

if len(sys.argv) < 2:
    print("Usage: python debug_predictions.py <path_to_test_image_or_pdf>")
    sys.exit(1)

# Load model
model_path = Path("models/trained/best.pt")
if not model_path.exists():
    print(f"Model not found at {model_path}")
    sys.exit(1)

model = YOLO(str(model_path))

# Load test image
test_file_path = sys.argv[1]
print(f"\nTesting on: {test_file_path}")
print("=" * 70)

# Convert PDF to images if needed
test_images = []
if test_file_path.lower().endswith(".pdf"):
    try:
        from pdf2image import convert_from_path

        print("Converting PDF to images...")
        images = convert_from_path(test_file_path, dpi=200)
        print(f"  Extracted {len(images)} page(s)")
        test_images = images
    except ImportError:
        print("ERROR: pdf2image not installed. Install with: pip install pdf2image")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR converting PDF: {e}")
        sys.exit(1)
else:
    # Single image file
    test_images = [Image.open(test_file_path)]

# Run inference with very low confidence to see all predictions
all_results = []
for img in test_images:
    result = model.predict(
        img, conf=0.01, iou=0.5, verbose=False  # Very low threshold to see everything
    )
    all_results.extend(result)

results = all_results

# Print results
for i, result in enumerate(results):
    print(f"\nImage {i+1}:")
    print(f"  Shape: {result.orig_shape}")

    if len(result.boxes) == 0:
        print("  ❌ NO DETECTIONS AT ALL (even at 1% confidence)")
        print("  This suggests:")
        print("    - Image is very different from training data")
        print("    - Model doesn't recognize any regions")
        print("    - Image quality/format issue")
    else:
        print(f"  ✓ Found {len(result.boxes)} detection(s):")
        for box in result.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_names = {0: "header", 1: "body", 2: "footer"}
            class_name = class_names.get(cls, f"class_{cls}")

            # Color code by confidence
            if conf >= 0.35:
                status = "✓ ABOVE DEFAULT (0.35)"
            elif conf >= 0.15:
                status = "⚠ MEDIUM (try lowering threshold)"
            else:
                status = "❌ LOW (probably noise)"

            print(f"    - {class_name}: {conf:.3f} confidence {status}")

print("\n" + "=" * 70)
print("Recommendations:")
print("  • If you see detections with 0.15-0.35 confidence, lower the threshold")
print("  • If all detections are < 0.10, the model doesn't recognize this image type")
print("  • If NO detections at all, verify the image is a financial document")
print("=" * 70)
