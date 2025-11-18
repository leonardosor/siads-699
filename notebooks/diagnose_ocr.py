"""
Quick OCR Diagnostic Tool
Tests OCR on detected regions and shows what's working/not working
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytesseract
from PIL import Image
from ultralytics import YOLO

from src.utils.ocr_enhancement import extract_text_from_bbox


def diagnose_image(image_path: str, confidence: float = 0.35):
    """Run full diagnostic on an image"""
    print("=" * 80)
    print(f"DIAGNOSING: {image_path}")
    print("=" * 80)

    # Load image
    image = Image.open(image_path)
    print(f"\n1. IMAGE INFO:")
    print(f"   Size: {image.width}x{image.height}")
    print(f"   Mode: {image.mode}")
    print(f"   Format: {image.format}")

    # Load YOLO model
    model_path = Path(__file__).resolve().parents[1] / "models" / "trained" / "best.pt"
    if not model_path.exists():
        print(f"\n   ERROR: Model not found at {model_path}")
        return

    print(f"\n2. LOADING YOLO MODEL:")
    print(f"   Path: {model_path}")
    model = YOLO(str(model_path))

    # Run YOLO detection
    print(f"\n3. RUNNING YOLO DETECTION (confidence={confidence}):")
    results = model.predict(image, conf=confidence, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        print(f"   ❌ NO DETECTIONS FOUND!")
        print(f"   Try lowering confidence threshold")
        return

    print(f"   ✓ Found {len(boxes)} detections")

    # Test OCR on each detection
    print(f"\n4. TESTING OCR ON EACH DETECTION:")
    print("-" * 80)

    for i, box in enumerate(boxes):
        coords = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = results[0].names.get(cls_id, f"class_{cls_id}")

        bbox = (float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))

        print(f"\n   Detection #{i+1}:")
        print(f"   Label: {label}")
        print(f"   Confidence: {conf:.1%}")
        print(f"   BBox: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")
        print(f"   Size: {bbox[2]-bbox[0]:.1f}x{bbox[3]-bbox[1]:.1f} pixels")

        # Test standard OCR
        print(f"\n   Testing STANDARD OCR (PSM 7):")
        x1, y1, x2, y2 = bbox
        padding = 10
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(image.width, int(x2) + padding)
        y2 = min(image.height, int(y2) + padding)
        cropped = image.crop((x1, y1, x2, y2))

        try:
            text = pytesseract.image_to_string(cropped, config="--psm 7").strip()
            data = pytesseract.image_to_data(
                cropped, output_type=pytesseract.Output.DICT
            )
            confidences = [int(c) for c in data["conf"] if c != "-1" and int(c) > 0]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0

            if text:
                print(f"   ✓ Text: '{text}'")
                print(f"   ✓ OCR Confidence: {avg_conf:.1f}%")
                print(f"   ✓ Characters: {len(text)}")
            else:
                print(f"   ❌ NO TEXT EXTRACTED")
                print(f"   Confidence: {avg_conf:.1f}%")

                # Try different PSM modes
                print(f"\n   Trying other PSM modes:")
                for psm in [6, 7, 8, 11, 13]:
                    alt_text = pytesseract.image_to_string(
                        cropped, config=f"--psm {psm}"
                    ).strip()
                    if alt_text:
                        print(f"      PSM {psm}: '{alt_text[:50]}'")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")

        # Test enhanced OCR
        print(f"\n   Testing ENHANCED OCR:")
        try:
            result = extract_text_from_bbox(image, bbox, enhanced=True)
            if result["text"]:
                print(f"   ✓ Text: '{result['text']}'")
                print(f"   ✓ OCR Confidence: {result['confidence']:.1f}%")
                print(f"   ✓ Method: {result.get('method', 'N/A')}")
                print(f"   ✓ PSM Mode: {result.get('psm_mode', 'N/A')}")
            else:
                print(f"   ❌ NO TEXT EXTRACTED")
                print(f"   Method tried: {result.get('method', 'N/A')}")
                print(f"   Error: {result.get('error', 'None')}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")

        print("-" * 80)

    print(f"\n5. SUMMARY:")
    print(f"   Total detections: {len(boxes)}")
    print(f"   Check if bounding boxes look correct in Streamlit")
    print(f"   If boxes are wrong, retrain YOLO or adjust confidence")
    print(f"   If boxes are right but OCR fails, might need different preprocessing")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose OCR issues")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument(
        "--confidence", type=float, default=0.35, help="YOLO confidence threshold"
    )

    args = parser.parse_args()

    diagnose_image(args.image, args.confidence)
