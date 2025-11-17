#!/usr/bin/env python3
"""
Test script to demonstrate background color augmentation.
"""
import sys
from pathlib import Path

import cv2

# Add src/utils to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "utils"))
from common import adjust_background_color

# Load the example training image
input_image = "example_training_image.jpg"
output_dir = Path("augmentation_examples")
output_dir.mkdir(exist_ok=True)

print("Testing background color augmentation...")
print(f"Input: {input_image}")
print(f"Output directory: {output_dir}")
print()

# Load image
img = cv2.imread(input_image)
if img is None:
    print(f"ERROR: Could not load {input_image}")
    sys.exit(1)

# Generate all background variations
variations = ["white", "cream", "gray", "normalize"]

for variant in variations:
    print(f"  Generating {variant} variant...")
    aug_img = adjust_background_color(img, variant)
    output_path = output_dir / f"example_bg_{variant}.jpg"
    cv2.imwrite(str(output_path), aug_img)
    print(f"    Saved: {output_path}")

# Also save original for comparison
original_path = output_dir / "example_original.jpg"
cv2.imwrite(str(original_path), img)
print(f"  Original saved: {original_path}")

print()
print("=" * 70)
print("DONE! Compare the images:")
print(f"  Original (yellow):  {original_path}")
print(f"  White background:   {output_dir / 'example_bg_white.jpg'}")
print(f"  Cream background:   {output_dir / 'example_bg_cream.jpg'}")
print(f"  Gray background:    {output_dir / 'example_bg_gray.jpg'}")
print(f"  Normalized:         {output_dir / 'example_bg_normalize.jpg'}")
print("=" * 70)
print()
print("These augmented variations will help your model recognize documents")
print("with different background colors (like your white Hilton invoice)!")
