"""
Train custom YOLO models on document images from parquet files
This script prepares data and trains Ultralytics YOLO models for document classification/detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import io
from ultralytics import YOLO
from tqdm import tqdm
import yaml


class YOLODocumentTrainer:
    """Train YOLO models on document images from parquet files"""

    def __init__(
        self, parquet_dir="/workspace/data/raw", output_dir="/workspace/data/output/yolo_training"
    ):
        """
        Initialize YOLO trainer

        Args:
            parquet_dir: Directory containing parquet files
            output_dir: Directory for training data and models
        """
        self.parquet_dir = Path(parquet_dir)
        self.output_dir = Path(output_dir)
        self.dataset_dir = self.output_dir / "dataset"
        self.models_dir = self.output_dir / "models"

        # Create directory structure
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def extract_images_from_parquet(self, split_ratio=0.8, max_images_per_class=None):
        """
        Extract images from parquet files and organize for YOLO training

        Args:
            split_ratio: Train/val split ratio
            max_images_per_class: Optional limit on images per class
        """
        print("=" * 70)
        print("Extracting images from parquet files for YOLO training")
        print("=" * 70)

        # Find all parquet files
        parquet_files = sorted(self.parquet_dir.glob("*.parquet"))

        if not parquet_files:
            print(f"❌ No parquet files found in {self.parquet_dir}")
            return False

        print(f"Found {len(parquet_files)} parquet files")

        # Create train/val directories
        train_dir = self.dataset_dir / "train" / "images"
        val_dir = self.dataset_dir / "val" / "images"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # Collect all data
        all_images = []
        label_counts = {}

        for pf in parquet_files:
            print(f"\nReading: {pf.name}")
            df = pd.read_parquet(pf)

            for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Loading images"):
                label = row.get("label", 0)

                # Track label counts
                label_counts[label] = label_counts.get(label, 0) + 1

                # Check max images per class
                if max_images_per_class and label_counts[label] > max_images_per_class:
                    continue

                all_images.append(
                    {
                        "image_data": row["image"],
                        "label": label,
                        "source_file": pf.name,
                        "row_index": idx,
                    }
                )

        print(f"\nTotal images collected: {len(all_images):,}")
        print("Label distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  Class {label}: {count:,} images")

        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(all_images)

        split_idx = int(len(all_images) * split_ratio)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]

        print(f"\nSplit: {len(train_images):,} train, {len(val_images):,} val")

        # Save images
        print("\nSaving images...")
        self._save_images(train_images, train_dir, "train")
        self._save_images(val_images, val_dir, "val")

        # Create dataset YAML
        self._create_dataset_yaml(label_counts.keys())

        print("✓ Dataset preparation complete!")
        return True

    def _save_images(self, image_list, output_dir, split_name):
        """Save images to disk"""
        for i, img_data in enumerate(tqdm(image_list, desc=f"  Saving {split_name}")):
            try:
                # Extract image
                if isinstance(img_data["image_data"], dict):
                    image_bytes = img_data["image_data"].get("bytes")
                else:
                    image_bytes = img_data["image_data"]

                image = Image.open(io.BytesIO(image_bytes))

                # Convert to RGB if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Save with label in filename for classification
                filename = f"class_{img_data['label']}_img_{i:06d}.jpg"
                image.save(output_dir / filename, "JPEG", quality=95)

            except Exception as e:
                print(f"Error saving image {i}: {e}")

    def _create_dataset_yaml(self, labels):
        """Create YAML configuration for YOLO training"""
        num_classes = len(labels)
        class_names = [f"class_{i}" for i in sorted(labels)]

        yaml_content = {
            "path": str(self.dataset_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "nc": num_classes,
            "names": class_names,
        }

        yaml_path = self.dataset_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        print(f"\n✓ Dataset YAML created: {yaml_path}")
        print(f"  Classes: {num_classes}")
        print(f"  Names: {class_names}")

        return yaml_path

    def train_classification_model(
        self, model_size="n", epochs=50, imgsz=224, batch=16, device="0"
    ):
        """
        Train YOLO classification model

        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            device: Device to train on ('0' for GPU, 'cpu' for CPU)
        """
        print("\n" + "=" * 70)
        print("Training YOLO Classification Model")
        print("=" * 70)

        # Initialize model
        model_name = f"yolov8{model_size}-cls.pt"
        print(f"Loading model: {model_name}")
        model = YOLO(model_name)

        # Train
        print(f"\nStarting training:")
        print(f"  Epochs: {epochs}")
        print(f"  Image size: {imgsz}")
        print(f"  Batch size: {batch}")
        print(f"  Device: {device}")

        results = model.train(
            data=str(self.dataset_dir),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(self.models_dir),
            name="document_classifier",
            exist_ok=True,
            pretrained=True,
            optimizer="Adam",
            verbose=True,
            patience=10,  # Early stopping
            save=True,
            plots=True,
        )

        print("\n✓ Training complete!")
        print(f"Model saved to: {self.models_dir / 'document_classifier'}")

        return results

    def train_detection_model(self, model_size="n", epochs=50, imgsz=640, batch=16, device="0"):
        """
        Train YOLO detection model (for text region detection)
        Note: This requires annotated bounding boxes

        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            device: Device to train on
        """
        print("\n" + "=" * 70)
        print("Training YOLO Detection Model")
        print("=" * 70)
        print("⚠ Note: Detection training requires bounding box annotations")
        print("  This basic setup uses classification data")
        print("  For text detection, you'll need to annotate text regions")

        # Initialize model
        model_name = f"yolov8{model_size}.pt"
        print(f"Loading model: {model_name}")
        model = YOLO(model_name)

        # For detection, you would need a properly formatted dataset with labels
        # This is a placeholder for future implementation
        print("\n⚠ Detection model training requires:")
        print("  1. Annotated bounding boxes (YOLO format)")
        print("  2. Label files (.txt) for each image")
        print("  3. Updated dataset.yaml with proper paths")
        print("\nConsider using Label Studio or other annotation tools")

        return None

    def evaluate_model(self, model_path):
        """
        Evaluate trained model

        Args:
            model_path: Path to trained model weights
        """
        print("\n" + "=" * 70)
        print("Evaluating Model")
        print("=" * 70)

        model = YOLO(model_path)

        # Run validation
        results = model.val()

        print("\n✓ Evaluation complete!")
        return results

    def export_model(self, model_path, format="onnx"):
        """
        Export model to different format

        Args:
            model_path: Path to trained model
            format: Export format ('onnx', 'torchscript', 'coreml', etc.)
        """
        print(f"\nExporting model to {format.upper()}...")

        model = YOLO(model_path)
        export_path = model.export(format=format)

        print(f"✓ Model exported to: {export_path}")
        return export_path


def main():
    """Main execution function"""
    print("=" * 70)
    print("YOLO Document Classification Training Pipeline")
    print("=" * 70)

    # Initialize trainer
    trainer = YOLODocumentTrainer(
        parquet_dir="/workspace/data/raw", output_dir="/workspace/data/output/yolo_training"
    )

    # Step 1: Extract images and prepare dataset
    print("\nStep 1: Preparing dataset...")
    success = trainer.extract_images_from_parquet(
        split_ratio=0.8, max_images_per_class=1000  # Limit for testing
    )

    if not success:
        print("❌ Failed to prepare dataset")
        return

    # Step 2: Train classification model
    print("\nStep 2: Training classification model...")
    print("Starting with YOLOv8 nano for quick testing...")

    # Uncomment to train:
    results = trainer.train_classification_model(
        model_size="n",  # nano model for quick training
        epochs=10,  # Start with few epochs for testing
        imgsz=224,
        batch=16,
        device="0",  # Use '0' for GPU, 'cpu' for CPU
    )

    # Step 3: Export model (optional)
    # model_path = trainer.models_dir / "document_classifier" / "weights" / "best.pt"
    # if model_path.exists():
    #     trainer.export_model(model_path, format='onnx')

    print("\n" + "=" * 70)
    print("Training pipeline complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review training results in the models directory")
    print("2. Evaluate model performance on validation set")
    print("3. Use trained model for inference on new images")
    print("4. Consider training larger models (s, m, l) for better accuracy")


if __name__ == "__main__":
    main()
