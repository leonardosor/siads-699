"""
OCR Processor using Ultralytics YOLO and Tesseract
Processes images from parquet files and extracts text using multiple OCR methods
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import io
import pytesseract
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import json
import os
from datetime import datetime
from sqlalchemy import create_engine, text
import time
import uuid
import torch


class OCRProcessor:
    """Process images from parquet files using Ultralytics models and Tesseract"""

    def __init__(
        self,
        parquet_dir="/workspace/data/raw",
        output_dir="/workspace/data/output/ocr_results",
        use_yolo_ocr=True,
        use_tesseract=True,
        save_to_db=True,
        database_url=None,
    ):
        """
        Initialize OCR Processor

        Args:
            parquet_dir: Directory containing parquet files
            output_dir: Directory to save OCR results
            use_yolo_ocr: Whether to use YOLO for text detection
            use_tesseract: Whether to use Tesseract for OCR
            save_to_db: Whether to save results to PostgreSQL database
            database_url: Database connection URL (defaults to env var)
        """
        self.parquet_dir = Path(parquet_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_yolo_ocr = use_yolo_ocr
        self.use_tesseract = use_tesseract
        self.save_to_db = save_to_db

        self.db_engine = None
        if self.save_to_db:
            self._initialize_database(database_url)

        self.yolo_model = None
        if self.use_yolo_ocr:
            self._initialize_yolo()

    def _initialize_database(self, database_url):
        """Initialize database connection"""
        try:
            db_url = database_url or os.getenv(
                "DATABASE_URL", "postgresql://postgres:123@db:5432/postgres"
            )

            print(
                f"Connecting to database: {db_url.split('@')[1] if '@' in db_url else 'localhost'}"
            )
            self.db_engine = create_engine(db_url)

            with self.db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                conn.commit()

            print("Database connection established")
        except Exception as e:
            print(f"Could not connect to database: {e}")
            print("  Results will only be saved to files")
            self.save_to_db = False
            self.db_engine = None

    def _initialize_yolo(self):
        """Initialize YOLO model for text detection/recognition"""
        try:
            print("Loading YOLO model for text detection...")
            self.yolo_model = YOLO("yolov8n.pt")
            if torch.cuda.is_available():
                self.yolo_model.to("cuda")
                print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("  GPU not available, using CPU")
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Could not load YOLO model: {e}")
            print("  Continuing with Tesseract OCR only")
            self.use_yolo_ocr = False

    def _extract_image_from_row(self, image_data):
        """
        Extract PIL Image from parquet row data

        Args:
            image_data: Image data from parquet (dict or bytes)

        Returns:
            PIL.Image object
        """
        if isinstance(image_data, dict):
            image_bytes = image_data.get("bytes")
        else:
            image_bytes = image_data

        return Image.open(io.BytesIO(image_bytes))

    def _yolo_detect_text_regions(self, image):
        """
        Use YOLO to detect text regions in image

        Args:
            image: PIL Image

        Returns:
            List of detected text regions (bounding boxes)
        """
        if not self.yolo_model:
            return []

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            img_array = np.array(image)
            results = self.yolo_model(img_array, conf=0.25, device="cuda")

            text_regions = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    text_regions.append(
                        {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": confidence,
                            "class": class_id,
                        }
                    )

            return text_regions
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []

    def _tesseract_ocr(self, image):
        """
        Use Tesseract to extract text from image

        Args:
            image: PIL Image

        Returns:
            Dictionary with OCR results
        """
        try:
            if image.mode not in ["RGB", "L", "RGBA"]:
                image = image.convert("RGB")

            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            full_text = pytesseract.image_to_string(image)

            confidences = [int(conf) for conf in data["conf"] if conf != "-1"]
            avg_confidence = np.mean(confidences) if confidences else 0

            words = []
            n_boxes = len(data["text"])
            for i in range(n_boxes):
                if int(data["conf"][i]) > 0:
                    word = {
                        "text": data["text"][i],
                        "confidence": int(data["conf"][i]),
                        "bbox": [
                            data["left"][i],
                            data["top"][i],
                            data["left"][i] + data["width"][i],
                            data["top"][i] + data["height"][i],
                        ],
                    }
                    if word["text"].strip():
                        words.append(word)

            return {
                "full_text": full_text.strip(),
                "words": words,
                "avg_confidence": float(avg_confidence),
                "total_words": len(words),
            }
        except Exception as e:
            print(f"Error in Tesseract OCR: {e}")
            return {
                "full_text": "",
                "words": [],
                "avg_confidence": 0,
                "total_words": 0,
                "error": str(e),
            }

    def process_image(self, image_data, label=None):
        """
        Process a single image with all available OCR methods

        Args:
            image_data: Image data from parquet
            label: Original label from dataset

        Returns:
            Dictionary with OCR results
        """
        start_time = time.time()
        result = {"label": label, "timestamp": datetime.now().isoformat(), "methods": {}}

        try:
            image = self._extract_image_from_row(image_data)
            result["image_size"] = image.size
            result["image_mode"] = image.mode

            if self.use_yolo_ocr:
                text_regions = self._yolo_detect_text_regions(image)
                result["methods"]["yolo"] = {
                    "text_regions": text_regions,
                    "num_regions": len(text_regions),
                }

            if self.use_tesseract:
                tesseract_result = self._tesseract_ocr(image)
                result["methods"]["tesseract"] = tesseract_result

            result["status"] = "success"

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        result["processing_time"] = time.time() - start_time
        return result

    def _save_to_database(self, results_list):
        """
        Save OCR results to PostgreSQL database

        Args:
            results_list: List of OCR result dictionaries
        """
        if not self.save_to_db or not self.db_engine:
            return

        saved_count = 0
        try:
            with self.db_engine.connect() as conn:
                for result in tqdm(results_list, desc="  Saving to database"):
                    try:
                        if result["status"] != "success":
                            continue

                        if "tesseract" in result["methods"]:
                            ocr_id = self._insert_ocr_result(conn, result, "tesseract")

                            if ocr_id and "words" in result["methods"]["tesseract"]:
                                self._insert_ocr_words(
                                    conn, ocr_id, result["methods"]["tesseract"]["words"]
                                )

                        if (
                            "yolo" in result["methods"]
                            and result["methods"]["yolo"]["num_regions"] > 0
                        ):
                            yolo_id = self._insert_ocr_result(conn, result, "yolo")

                            if yolo_id:
                                self._insert_yolo_regions(
                                    conn, yolo_id, result["methods"]["yolo"]["text_regions"]
                                )

                        conn.commit()
                        saved_count += 1

                    except Exception:
                        conn.rollback()
                        continue

            print(f"Saved {saved_count}/{len(results_list)} results to database")

        except Exception as e:
            print(f"Error saving to database: {e}")

    def _insert_ocr_result(self, conn, result, engine_type):
        """Insert main OCR result record"""
        try:
            ocr_id = str(uuid.uuid4())

            tesseract_data = result["methods"].get("tesseract", {})
            yolo_data = result["methods"].get("yolo", {})

            sql = text(
                """
                INSERT INTO parquet_ocr_results (
                    id, parquet_file, row_index, label,
                    image_size_width, image_size_height, image_mode,
                    ocr_engine,
                    tesseract_full_text, tesseract_confidence, tesseract_word_count,
                    yolo_region_count,
                    processing_status, processing_error, processing_time_seconds
                ) VALUES (
                    CAST(:id AS UUID), :parquet_file, :row_index, :label,
                    :width, :height, :mode,
                    :engine,
                    :tess_text, :tess_conf, :tess_words,
                    :yolo_regions,
                    :status, :error, :proc_time
                )
                ON CONFLICT (parquet_file, row_index, ocr_engine) DO UPDATE SET
                    tesseract_full_text = EXCLUDED.tesseract_full_text,
                    tesseract_confidence = EXCLUDED.tesseract_confidence,
                    tesseract_word_count = EXCLUDED.tesseract_word_count,
                    yolo_region_count = EXCLUDED.yolo_region_count,
                    processing_status = EXCLUDED.processing_status,
                    processing_time_seconds = EXCLUDED.processing_time_seconds,
                    processed_at = CURRENT_TIMESTAMP
            """
            )

            conn.execute(
                sql,
                {
                    "id": ocr_id,
                    "parquet_file": result.get("parquet_file", ""),
                    "row_index": result.get("row_index", 0),
                    "label": result.get("label"),
                    "width": result.get("image_size", [0, 0])[0]
                    if result.get("image_size")
                    else None,
                    "height": result.get("image_size", [0, 0])[1]
                    if result.get("image_size")
                    else None,
                    "mode": result.get("image_mode"),
                    "engine": engine_type,
                    "tess_text": tesseract_data.get("full_text"),
                    "tess_conf": tesseract_data.get("avg_confidence"),
                    "tess_words": tesseract_data.get("total_words", 0),
                    "yolo_regions": yolo_data.get("num_regions", 0),
                    "status": result.get("status", "success"),
                    "error": result.get("error"),
                    "proc_time": result.get("processing_time"),
                },
            )

            return ocr_id

        except Exception:
            return None

    def _insert_ocr_words(self, conn, ocr_result_id, words):
        """Insert word-level OCR details"""
        try:
            for i, word in enumerate(words):
                sql = text(
                    """
                    INSERT INTO parquet_ocr_words (
                        id, ocr_result_id, word_text, confidence,
                        x_min, y_min, x_max, y_max, sequence_number
                    ) VALUES (
                        CAST(:id AS UUID), CAST(:ocr_id AS UUID), :text, :conf,
                        :x_min, :y_min, :x_max, :y_max, :seq
                    )
                """
                )

                conn.execute(
                    sql,
                    {
                        "id": str(uuid.uuid4()),
                        "ocr_id": ocr_result_id,
                        "text": word.get("text", ""),
                        "conf": word.get("confidence", 0),
                        "x_min": word.get("bbox", [0, 0, 0, 0])[0],
                        "y_min": word.get("bbox", [0, 0, 0, 0])[1],
                        "x_max": word.get("bbox", [0, 0, 0, 0])[2],
                        "y_max": word.get("bbox", [0, 0, 0, 0])[3],
                        "seq": i,
                    },
                )

        except Exception:
            pass

    def _insert_yolo_regions(self, conn, ocr_result_id, regions):
        """Insert YOLO detected regions"""
        try:
            for region in regions:
                sql = text(
                    """
                    INSERT INTO parquet_yolo_regions (
                        id, ocr_result_id, class_id, confidence,
                        x_min, y_min, x_max, y_max
                    ) VALUES (
                        CAST(:id AS UUID), CAST(:ocr_id AS UUID), :class_id, :conf,
                        :x_min, :y_min, :x_max, :y_max
                    )
                """
                )

                bbox = region.get("bbox", [0, 0, 0, 0])
                conn.execute(
                    sql,
                    {
                        "id": str(uuid.uuid4()),
                        "ocr_id": ocr_result_id,
                        "class_id": region.get("class", 0),
                        "conf": region.get("confidence", 0),
                        "x_min": bbox[0],
                        "y_min": bbox[1],
                        "x_max": bbox[2],
                        "y_max": bbox[3],
                    },
                )

        except Exception:
            pass

    def process_parquet_file(self, parquet_file, sample_size=None):
        """
        Process all images in a parquet file

        Args:
            parquet_file: Path to parquet file
            sample_size: Optional number of images to process (None = all)

        Returns:
            List of OCR results
        """
        print(f"\nProcessing: {parquet_file.name}")

        df = pd.read_parquet(parquet_file)

        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"  Processing sample of {sample_size} images")
        else:
            print(f"  Processing all {len(df)} images")

        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="  OCR Processing"):
            result = self.process_image(image_data=row["image"], label=row.get("label"))
            result["parquet_file"] = parquet_file.name
            result["row_index"] = idx
            results.append(result)

        if self.save_to_db and results:
            print(f"  Saving {len(results)} results to database...")
            self._save_to_database(results)

        return results

    def process_all_parquets(self, sample_size=None):
        """
        Process all parquet files in the directory

        Args:
            sample_size: Optional number of images to process per file

        Returns:
            DataFrame with all OCR results
        """
        parquet_files = sorted(self.parquet_dir.glob("*.parquet"))

        if not parquet_files:
            print(f"No parquet files found in {self.parquet_dir}")
            return pd.DataFrame()

        print(f"Found {len(parquet_files)} parquet files")

        all_results = []

        for pf in parquet_files:
            results = self.process_parquet_file(pf, sample_size=sample_size)
            all_results.extend(results)

        return pd.DataFrame(all_results)

    def save_results(self, results_df, format="parquet"):
        """
        Save OCR results to file

        Args:
            results_df: DataFrame with OCR results
            format: Output format ('parquet', 'json', 'csv')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "parquet":
            output_file = self.output_dir / f"ocr_results_{timestamp}.parquet"
            results_df.to_parquet(output_file, index=False)
        elif format == "json":
            output_file = self.output_dir / f"ocr_results_{timestamp}.json"
            results_df.to_json(output_file, orient="records", indent=2)
        elif format == "csv":
            output_file = self.output_dir / f"ocr_results_{timestamp}.csv"
            simplified_df = results_df.copy()
            if "methods" in simplified_df.columns:
                simplified_df["tesseract_text"] = simplified_df["methods"].apply(
                    lambda x: x.get("tesseract", {}).get("full_text", "")
                    if isinstance(x, dict)
                    else ""
                )
                simplified_df["tesseract_confidence"] = simplified_df["methods"].apply(
                    lambda x: x.get("tesseract", {}).get("avg_confidence", 0)
                    if isinstance(x, dict)
                    else 0
                )
                simplified_df["yolo_regions"] = simplified_df["methods"].apply(
                    lambda x: x.get("yolo", {}).get("num_regions", 0) if isinstance(x, dict) else 0
                )
                simplified_df = simplified_df.drop(columns=["methods"])
            simplified_df.to_csv(output_file, index=False)

        print(f"Results saved to: {output_file}")
        return output_file

    def print_summary(self, results_df):
        """Print summary statistics of OCR results"""
        print("\n" + "=" * 70)
        print("OCR Processing Summary")
        print("=" * 70)

        total = len(results_df)
        successful = len(results_df[results_df["status"] == "success"])

        print(f"Total images processed: {total:,}")
        print(f"Successful: {successful:,} ({successful/total*100:.1f}%)")

        if "methods" in results_df.columns:
            tesseract_texts = results_df["methods"].apply(
                lambda x: x.get("tesseract", {}).get("full_text", "") if isinstance(x, dict) else ""
            )
            non_empty_texts = tesseract_texts[tesseract_texts.str.len() > 0]

            print(f"\nTesseract OCR:")
            print(
                f"  Images with text detected: {len(non_empty_texts):,} ({len(non_empty_texts)/total*100:.1f}%)"
            )

            avg_confidences = results_df["methods"].apply(
                lambda x: x.get("tesseract", {}).get("avg_confidence", 0)
                if isinstance(x, dict)
                else 0
            )
            print(f"  Average confidence: {avg_confidences.mean():.1f}%")

            yolo_regions = results_df["methods"].apply(
                lambda x: x.get("yolo", {}).get("num_regions", 0) if isinstance(x, dict) else 0
            )
            if yolo_regions.sum() > 0:
                print(f"\nYOLO Text Detection:")
                print(f"  Total text regions detected: {int(yolo_regions.sum()):,}")
                print(f"  Average regions per image: {yolo_regions.mean():.2f}")

        print("=" * 70)


def main():
    """Main execution function"""
    print("=" * 70)
    print("OCR Processor - Ultralytics + Tesseract + Database\n" + "=" * 70)

    processor = OCRProcessor(
        parquet_dir="/workspace/data/raw",
        output_dir="/workspace/data/output/ocr_results",
        use_yolo_ocr=True,
        use_tesseract=True,
        save_to_db=True,
    )

    print("\nProcessing sample of images (10 per file)...")
    results_df = processor.process_all_parquets(sample_size=10)

    if len(results_df) > 0:
        processor.print_summary(results_df)
        processor.save_results(results_df, format="parquet")
        processor.save_results(results_df, format="json")
        processor.save_results(results_df, format="csv")

        print("\n" + "=" * 70)
        print("OCR processing complete!")
        if processor.save_to_db:
            print("Results saved to database\n")
            print("Query with: SELECT * FROM parquet_ocr_results LIMIT 10;")
        print("=" * 70 + "\n")
    else:
        print("No results to save")


if __name__ == "__main__":
    main()
