"""
Visualize OCR Bounding Boxes
Shows an image with bounding boxes from both YOLO and Tesseract OCR detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ocr_processor import OCRProcessor


def visualize_image_with_boxes(image_data, yolo_regions=None, tesseract_words=None,
                                label=None, save_path=None):
    """
    Visualize an image with bounding boxes from OCR detection

    Args:
        image_data: Image data from parquet
        yolo_regions: List of YOLO detected regions
        tesseract_words: List of Tesseract detected words
        label: Original label for the image
        save_path: Optional path to save the visualization
    """
    # Extract image
    if isinstance(image_data, dict):
        image_bytes = image_data.get("bytes")
    else:
        image_bytes = image_data

    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Create figure with matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f'Original Image\nLabel: {label}', fontsize=12)
    axes[0].axis('off')

    # Image with YOLO boxes
    axes[1].imshow(image)
    if yolo_regions:
        for region in yolo_regions:
            bbox = region['bbox']
            confidence = region['confidence']
            class_id = region['class']

            # Draw rectangle
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=2,
                edgecolor='red',
                facecolor='none',
                alpha=0.8
            )
            axes[1].add_patch(rect)

            # Add label
            axes[1].text(
                bbox[0], bbox[1] - 5,
                f'C{class_id}: {confidence:.2f}',
                color='red',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )

    axes[1].set_title(f'YOLO Detection\n{len(yolo_regions) if yolo_regions else 0} regions',
                      fontsize=12)
    axes[1].axis('off')

    # Image with Tesseract boxes
    axes[2].imshow(image)
    if tesseract_words:
        for word in tesseract_words:
            bbox = word['bbox']
            confidence = word['confidence']
            text = word['text']

            # Draw rectangle
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=1,
                edgecolor='blue',
                facecolor='none',
                alpha=0.6
            )
            axes[2].add_patch(rect)

            # Add text label for high confidence words
            if confidence > 60:
                axes[2].text(
                    bbox[0], bbox[1] - 2,
                    f'{text[:10]}',
                    color='blue',
                    fontsize=6,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5)
                )

    axes[2].set_title(f'Tesseract OCR\n{len(tesseract_words) if tesseract_words else 0} words',
                      fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()


def visualize_from_parquet(parquet_path, image_index=0, save_output=True):
    """
    Load an image from parquet and visualize with OCR bounding boxes

    Args:
        parquet_path: Path to parquet file
        image_index: Index of image to visualize
        save_output: Whether to save the visualization
    """
    print(f"Loading image {image_index} from {parquet_path}")

    # Load parquet
    df = pd.read_parquet(parquet_path)
    print(f"Total images in file: {len(df)}")

    if image_index >= len(df):
        print(f"Error: Image index {image_index} out of range (max: {len(df)-1})")
        return

    # Get image and label
    row = df.iloc[image_index]
    image_data = row['image']
    label = row.get('label', 'Unknown')

    print(f"Processing image with label: {label}")

    # Initialize OCR processor
    processor = OCRProcessor(
        use_yolo_ocr=True,
        use_tesseract=True,
        save_to_db=False
    )

    # Process image
    print("Running OCR detection...")
    result = processor.process_image(image_data, label=label)

    # Extract results
    yolo_regions = result.get('methods', {}).get('yolo', {}).get('text_regions', [])
    tesseract_data = result.get('methods', {}).get('tesseract', {})
    tesseract_words = tesseract_data.get('words', [])

    print(f"\nDetection Results:")
    print(f"  YOLO regions: {len(yolo_regions)}")
    print(f"  Tesseract words: {len(tesseract_words)}")
    print(f"  Tesseract confidence: {tesseract_data.get('avg_confidence', 0):.1f}%")

    if tesseract_data.get('full_text'):
        print(f"\nExtracted Text:")
        print(f"  {tesseract_data['full_text'][:200]}")

    # Visualize
    save_path = None
    if save_output:
        output_dir = Path("/workspace/data/output/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"ocr_visualization_{Path(parquet_path).stem}_img{image_index}.png"

    visualize_image_with_boxes(
        image_data=image_data,
        yolo_regions=yolo_regions,
        tesseract_words=tesseract_words,
        label=label,
        save_path=save_path
    )


def main():
    """Main execution"""
    # Default parameters
    parquet_file = "/workspace/data/raw/train-00000-of-00005.parquet"
    image_index = 0

    # Parse command line arguments
    if len(sys.argv) > 1:
        parquet_file = sys.argv[1]
    if len(sys.argv) > 2:
        image_index = int(sys.argv[2])

    print("=" * 70)
    print("OCR Bounding Box Visualization")
    print("=" * 70)

    visualize_from_parquet(
        parquet_path=parquet_file,
        image_index=image_index,
        save_output=True
    )

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
