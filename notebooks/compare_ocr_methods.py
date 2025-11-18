"""
Compare OCR methods: Basic vs Enhanced
Demonstrates the improvements in character extraction accuracy
"""

import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytesseract
from PIL import Image, ImageDraw, ImageFont

from src.utils.ocr_enhancement import OCREnhancer, extract_text_from_bbox


def compare_ocr_on_image(image_path: str, bbox: tuple = None):
    """
    Compare basic vs enhanced OCR on an image

    Args:
        image_path: Path to image file
        bbox: Optional bounding box (x1, y1, x2, y2) to test specific region
    """
    print("=" * 80)
    print("OCR Comparison: Basic vs Enhanced")
    print("=" * 80)

    # Load image
    image = Image.open(image_path)
    print(f"\nImage: {image_path}")
    print(f"Size: {image.width}x{image.height}")

    if bbox:
        print(f"Testing region: {bbox}")
        test_image = image.crop(bbox)
    else:
        print("Testing full image")
        test_image = image
        bbox = (0, 0, image.width, image.height)

    print("\n" + "-" * 80)
    print("METHOD 1: BASIC TESSERACT (Original)")
    print("-" * 80)

    # Basic OCR - original method
    try:
        basic_text = pytesseract.image_to_string(test_image, config="--psm 6").strip()
        basic_data = pytesseract.image_to_data(
            test_image, config="--psm 6", output_type=pytesseract.Output.DICT
        )
        basic_confidences = [
            int(c) for c in basic_data["conf"] if c != "-1" and int(c) > 0
        ]
        basic_conf = (
            sum(basic_confidences) / len(basic_confidences) if basic_confidences else 0
        )

        print(f"Text: '{basic_text}'")
        print(f"Length: {len(basic_text)} characters")
        print(f"Confidence: {basic_conf:.1f}%")
        print(f"Word count: {len(basic_text.split())}")
    except Exception as e:
        print(f"Error: {e}")
        basic_text = ""
        basic_conf = 0

    print("\n" + "-" * 80)
    print("METHOD 2: ENHANCED OCR (New)")
    print("-" * 80)

    # Enhanced OCR
    try:
        result = extract_text_from_bbox(image, bbox, enhanced=True)

        print(f"Text: '{result['text']}'")
        print(f"Length: {len(result['text'])} characters")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Word count: {result['word_count']}")
        print(f"\nBest preprocessing method: {result.get('method', 'N/A')}")
        print(f"Best PSM mode: {result.get('psm_mode', 'N/A')}")
        print(f"OEM used: {result.get('oem', 'N/A')}")
        print(f"High-confidence words: {result.get('high_conf_word_count', 0)}")

        enhanced_text = result["text"]
        enhanced_conf = result["confidence"]
    except Exception as e:
        print(f"Error: {e}")
        enhanced_text = ""
        enhanced_conf = 0

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    # Calculate improvements
    char_improvement = len(enhanced_text) - len(basic_text)
    conf_improvement = enhanced_conf - basic_conf

    print(f"\nCharacter extraction:")
    print(f"  Basic:    {len(basic_text)} characters")
    print(f"  Enhanced: {len(enhanced_text)} characters")
    print(
        f"  Change:   {'+' if char_improvement >= 0 else ''}{char_improvement} characters"
    )

    print(f"\nConfidence scores:")
    print(f"  Basic:    {basic_conf:.1f}%")
    print(f"  Enhanced: {enhanced_conf:.1f}%")
    print(f"  Change:   {'+' if conf_improvement >= 0 else ''}{conf_improvement:.1f}%")

    print(f"\nText comparison:")
    if enhanced_text != basic_text:
        print(f"  Texts differ!")
        print(
            f"  Basic:    '{basic_text[:100]}{'...' if len(basic_text) > 100 else ''}'"
        )
        print(
            f"  Enhanced: '{enhanced_text[:100]}{'...' if len(enhanced_text) > 100 else ''}'"
        )
    else:
        print(f"  Texts are identical")

    if char_improvement > 0 or conf_improvement > 0:
        print(f"\n✓ Enhanced OCR performed better!")
    elif char_improvement == 0 and conf_improvement == 0:
        print(f"\n= Both methods produced identical results")
    else:
        print(f"\n⚠ Basic OCR performed better (unusual - check image quality)")

    print("\n" + "=" * 80)


def demonstrate_preprocessing():
    """Show the preprocessing steps visually"""
    print("\n" + "=" * 80)
    print("PREPROCESSING METHODS AVAILABLE")
    print("=" * 80)

    methods = {
        "adaptive": "Adaptive Gaussian thresholding (best for varying lighting)",
        "otsu": "Otsu's automatic thresholding",
        "gaussian": "Gaussian blur + Otsu thresholding",
        "bilateral": "Bilateral filter (edge-preserving) + Otsu",
        "clahe": "CLAHE contrast enhancement + Otsu",
    }

    print("\nThe enhanced OCR tests all these methods and picks the best:")
    for i, (method, description) in enumerate(methods.items(), 1):
        print(f"\n{i}. {method.upper()}")
        print(f"   {description}")

    print("\n" + "=" * 80)
    print("ADVANCED FEATURES")
    print("=" * 80)
    print(
        """
✓ 3x upscaling (vs 2x basic)
✓ Deskewing (rotation correction)
✓ Border removal
✓ CLAHE contrast enhancement
✓ Automatic inversion detection (white-on-black → black-on-white)
✓ Multiple PSM modes: single_line, single_word, sparse_text, single_block
✓ Multiple OEM modes: Default + LSTM
✓ Post-processing error correction (l→1, O→0, etc.)
✓ Confidence-based filtering
✓ Increased padding (20px vs 5px)

TOTAL COMBINATIONS TESTED: Up to 40
(5 methods × 4 PSM modes × 2 OEM modes)
    """
    )


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Compare basic vs enhanced OCR")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument(
        "--bbox",
        type=str,
        help="Bounding box as 'x1,y1,x2,y2' (optional, tests specific region)",
    )
    parser.add_argument(
        "--show-methods", action="store_true", help="Show preprocessing methods"
    )

    args = parser.parse_args()

    if args.show_methods:
        demonstrate_preprocessing()

    # Parse bbox if provided
    bbox = None
    if args.bbox:
        try:
            bbox = tuple(map(int, args.bbox.split(",")))
            if len(bbox) != 4:
                print("Error: bbox must have 4 values (x1,y1,x2,y2)")
                return
        except ValueError:
            print("Error: bbox must be integers separated by commas")
            return

    # Run comparison
    try:
        compare_ocr_on_image(args.image, bbox)
    except FileNotFoundError:
        print(f"Error: Image file not found: {args.image}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Example usage if no args provided
    import sys

    if len(sys.argv) == 1:
        print("OCR Comparison Tool")
        print("=" * 80)
        print("\nUsage:")
        print("  python compare_ocr_methods.py <image_path> [--bbox x1,y1,x2,y2]")
        print("\nExamples:")
        print("  python compare_ocr_methods.py form.jpg")
        print("  python compare_ocr_methods.py form.jpg --bbox 100,200,400,250")
        print("  python compare_ocr_methods.py form.jpg --show-methods")
        print("\n")
        demonstrate_preprocessing()
    else:
        main()
