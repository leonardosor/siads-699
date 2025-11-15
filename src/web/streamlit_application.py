from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import pytesseract
import json

UM_BLUE = "#00274C"
UM_MAIZE = "#FFCB05"
_model_env = os.getenv("MODEL_PATH")
DEFAULT_MODEL_PATH = (
    Path(_model_env)
    if _model_env
    else Path(__file__).resolve().parents[2] / "models" / "trained" / "best.pt"
)
ALLOWED_TYPES = ("jpg", "jpeg", "png", "pdf")


@st.cache_resource(show_spinner=False)
def load_model(weights_path: Path) -> YOLO:
    """Load YOLO weights once per Streamlit session."""
    return YOLO(str(weights_path))


@st.cache_data(show_spinner=False)
def _load_font(size: int = 18) -> ImageFont.FreeTypeFont:
    """Use a clean sans font when available; fall back to default."""
    for font_name in ("Inter-Regular.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _extract_text_from_bbox(
    image: Image.Image, bbox: Tuple[float, float, float, float]
) -> dict:
    """
    Extract text from a specific bounding box region using Tesseract OCR

    Args:
        image: PIL Image
        bbox: Tuple of (x1, y1, x2, y2) coordinates

    Returns:
        Dictionary with extracted text and confidence
    """
    try:
        x1, y1, x2, y2 = bbox
        # Add small padding around the box to improve OCR accuracy
        padding = 5
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(image.width, int(x2) + padding)
        y2 = min(image.height, int(y2) + padding)

        # Crop the region
        cropped = image.crop((x1, y1, x2, y2))

        # Extract text using Tesseract
        text = pytesseract.image_to_string(cropped, config="--psm 6").strip()

        # Get confidence scores
        data = pytesseract.image_to_data(cropped, output_type=pytesseract.Output.DICT)
        confidences = [
            int(conf) for conf in data["conf"] if conf != "-1" and int(conf) > 0
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "text": text,
            "ocr_confidence": avg_confidence,
            "word_count": len([w for w in text.split() if w]),
        }
    except Exception as e:
        return {"text": "", "ocr_confidence": 0, "word_count": 0, "error": str(e)}


def _format_detections(
    result, image: Image.Image = None, extract_text: bool = False
) -> pd.DataFrame:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame(
            columns=[
                "label",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "extracted_text",
                "ocr_confidence",
            ]
        )

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy()
    names = result.names or {}

    rows: List[dict] = []
    for coords, confidence, cls_id in zip(xyxy, conf, cls_ids):
        row_data = {
            "label": names.get(int(cls_id), f"class_{int(cls_id)}"),
            "confidence": float(confidence),
            "x1": float(coords[0]),
            "y1": float(coords[1]),
            "x2": float(coords[2]),
            "y2": float(coords[3]),
        }

        # Extract text if requested and image is provided
        if extract_text and image is not None:
            bbox = (coords[0], coords[1], coords[2], coords[3])
            ocr_result = _extract_text_from_bbox(image, bbox)
            row_data["extracted_text"] = ocr_result["text"]
            row_data["ocr_confidence"] = ocr_result["ocr_confidence"]
            row_data["word_count"] = ocr_result["word_count"]

        rows.append(row_data)

    df = pd.DataFrame.from_records(rows)
    return df.sort_values("confidence", ascending=False).reset_index(drop=True)


def _annotate_umich(image: Image.Image, detections: pd.DataFrame) -> Image.Image:
    if detections.empty:
        return image

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = _load_font()

    for _, row in detections.iterrows():
        box = (row["x1"], row["y1"], row["x2"], row["y2"])
        label = row["label"]
        conf = row["confidence"]
        text = f"{label} {conf:.2f}"

        draw.rectangle(box, outline=UM_BLUE, width=4)

        text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        text_width = text_bbox[2] - text_bbox[0]
        label_y = max(box[1] - text_height - 10, 0)
        background = (
            box[0],
            label_y,
            box[0] + text_width + 14,
            label_y + text_height + 10,
        )
        draw.rectangle(background, fill=UM_MAIZE)
        draw.text((background[0] + 6, background[1] + 4), text, fill=UM_BLUE, font=font)

    return annotated


def _run_inference(
    image: Image.Image, confidence: float, iou: float, extract_text: bool = True
) -> Tuple[Image.Image, pd.DataFrame]:
    model = load_model(DEFAULT_MODEL_PATH)
    result = model.predict(image, conf=confidence, iou=iou, verbose=False)[0]
    detections = _format_detections(result, image=image, extract_text=extract_text)
    annotated = _annotate_umich(image, detections)
    return annotated, detections


def _bytes_to_image(data: bytes, filename: str = "") -> List[Image.Image]:
    """Convert uploaded file bytes to PIL Images. Returns list to handle multi-page PDFs."""
    if filename.lower().endswith(".pdf"):
        # Convert PDF to images (one image per page)
        images = convert_from_bytes(data, dpi=200)
        return [img.convert("RGB") for img in images]
    else:
        # Single image file
        return [Image.open(io.BytesIO(data)).convert("RGB")]


def main() -> None:
    st.set_page_config(page_title="Financial Form Text Extractor", layout="wide")
    st.title("Financial Form Text Extractor")
    st.caption(
        "Upload a JPG/PNG/PDF to generate Michigan-branded bounding boxes from your YOLOv8 model."
    )

    if not DEFAULT_MODEL_PATH.exists():
        st.error(
            f"Model weights not found at `{DEFAULT_MODEL_PATH}`. "
            "Copy your trained checkpoint there or set the MODEL_PATH environment variable."
        )
        return

    with st.sidebar:
        st.header("Inference Controls")
        confidence = st.slider(
            "Confidence threshold", min_value=0.1, max_value=0.9, value=0.35, step=0.05
        )
        iou = st.slider(
            "IoU threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05
        )

        st.header("OCR Settings")
        extract_text = st.checkbox(
            "Extract text from detected regions",
            value=True,
            help="Use Tesseract OCR to extract text from each detected bounding box",
        )

        st.write("Weights file:")
        st.code(str(DEFAULT_MODEL_PATH))

    uploads = st.file_uploader(
        "Upload one or more JPG/PNG/PDF files",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
    )

    if not uploads:
        st.info(
            "Waiting for uploads… drag a JPG/PNG/PDF into the widget above to begin."
        )
        return

    for uploaded in uploads:
        st.markdown(f"### {uploaded.name}")
        images = _bytes_to_image(uploaded.getvalue(), uploaded.name)

        # Process each page/image
        for page_num, raw_image in enumerate(images, start=1):
            page_suffix = f" (Page {page_num}/{len(images)})" if len(images) > 1 else ""

            st.image(
                raw_image, caption=f"Original{page_suffix}", use_container_width=True
            )
            spinner_text = (
                f"Running YOLOv8 inference and OCR{page_suffix}…"
                if extract_text
                else f"Running YOLOv8 inference{page_suffix}…"
            )
            with st.spinner(spinner_text):
                annotated, detections = _run_inference(
                    raw_image, confidence, iou, extract_text=extract_text
                )

            if detections.empty:
                st.warning(
                    f"No bounding boxes detected with the current thresholds{page_suffix}."
                )
                continue

            annotated_buffer = io.BytesIO()
            annotated.save(annotated_buffer, format="JPEG")

            st.image(
                annotated_buffer.getvalue(),
                caption=f"UM-Branded Bounding Boxes{page_suffix}",
                use_container_width=True,
            )

            download_filename = (
                f"{Path(uploaded.name).stem}_page{page_num}_umich_bboxes.jpg"
                if len(images) > 1
                else f"{Path(uploaded.name).stem}_umich_bboxes.jpg"
            )
            st.download_button(
                label=f"Download annotated JPG{page_suffix}",
                data=annotated_buffer.getvalue(),
                file_name=download_filename,
                mime="image/jpeg",
                key=f"download-{uploaded.name}-page{page_num}",
            )

            st.subheader(f"Detections{page_suffix}")

            # Format the detections dataframe
            pretty = detections.copy()
            pretty["confidence"] = (pretty["confidence"] * 100).round(1).astype(
                str
            ) + "%"

            # If OCR was performed, format those columns too
            if extract_text and "ocr_confidence" in pretty.columns:
                pretty["ocr_confidence"] = (
                    pretty["ocr_confidence"].round(1).astype(str) + "%"
                )
                # Reorder columns to show extracted text prominently
                display_cols = [
                    "label",
                    "extracted_text",
                    "confidence",
                    "ocr_confidence",
                    "word_count",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                ]
                display_cols = [col for col in display_cols if col in pretty.columns]
                pretty = pretty[display_cols]

            st.dataframe(pretty, use_container_width=True, hide_index=True)

            # Add download buttons for OCR results
            if (
                extract_text
                and "extracted_text" in detections.columns
                and len(detections) > 0
            ):
                col1, col2 = st.columns(2)

                with col1:
                    # CSV download
                    csv_buffer = io.StringIO()
                    detections.to_csv(csv_buffer, index=False)
                    csv_filename = (
                        f"{Path(uploaded.name).stem}_page{page_num}_ocr_results.csv"
                        if len(images) > 1
                        else f"{Path(uploaded.name).stem}_ocr_results.csv"
                    )
                    st.download_button(
                        label="📥 Download OCR Results (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name=csv_filename,
                        mime="text/csv",
                        key=f"csv-{uploaded.name}-page{page_num}",
                    )

                with col2:
                    # JSON download
                    json_data = detections.to_dict(orient="records")
                    json_filename = (
                        f"{Path(uploaded.name).stem}_page{page_num}_ocr_results.json"
                        if len(images) > 1
                        else f"{Path(uploaded.name).stem}_ocr_results.json"
                    )
                    st.download_button(
                        label="📥 Download OCR Results (JSON)",
                        data=json.dumps(json_data, indent=2),
                        file_name=json_filename,
                        mime="application/json",
                        key=f"json-{uploaded.name}-page{page_num}",
                    )

            # Show extracted text summary if OCR was performed
            if extract_text and "extracted_text" in detections.columns:
                texts_found = detections[detections["extracted_text"].str.len() > 0]
                if len(texts_found) > 0:
                    st.subheader(f"Extracted Text Summary{page_suffix}")
                    st.info(
                        f"Found text in {len(texts_found)} out of {len(detections)} detections"
                    )

                    # Display each extracted text with its label
                    for idx, row in texts_found.iterrows():
                        with st.expander(
                            f"📄 {row['label']} (confidence: {row['confidence']:.1%})"
                        ):
                            st.write(row["extracted_text"])
                            st.caption(
                                f"OCR Confidence: {row['ocr_confidence']:.1f}% | Words: {row['word_count']}"
                            )
                else:
                    st.warning(
                        f"No text extracted from detected regions{page_suffix}. Try adjusting confidence thresholds or check image quality."
                    )


if __name__ == "__main__":
    main()
