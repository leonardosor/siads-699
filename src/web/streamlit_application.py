from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytesseract
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


try:
    import numpy as np
except ImportError:
    pass  

try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import pytesseract
    PYTESSERACT_IMPORTED = True
except ImportError:
    PYTESSERACT_IMPORTED = False

UM_BLUE = "#00274C"
UM_MAIZE = "#FFCB05"
ALLOWED_TYPES = ("jpg", "jpeg", "png", "pdf")

_model_env = os.getenv("MODEL_PATH")
DEFAULT_MODEL_PATH = (
    Path(_model_env)
    if _model_env
    else Path(__file__).resolve().parents[2] / "models" / "best.pt"
)

st.set_page_config(
    page_title="Financial Form Text Extractor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


TESSERACT_AVAILABLE = False
if PYTESSERACT_IMPORTED:
    # Try common Tesseract paths
    tesseract_paths = [
        "/usr/local/bin/tesseract",  
        "/opt/homebrew/bin/tesseract",  
        "/usr/bin/tesseract",  
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",  
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break
    
    try:
        pytesseract.get_tesseract_version()
        TESSERACT_AVAILABLE = True
    except Exception as e:
        st.error(f"⚠️ Tesseract not found. Please install: `brew install tesseract` (macOS) or `sudo apt-get install tesseract-ocr` (Linux)")
        TESSERACT_AVAILABLE = False
else:
    st.warning("⚠️ Pytesseract not installed. Install with: `pip install pytesseract`")

try:
    import numpy as np
except ImportError:
    st.error("❌ Install numpy: `pip install numpy`")
    st.stop()



def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {{ font-family: 'Inter', sans-serif; }}
        
        #MainMenu, footer, header {{visibility: hidden;}}
        .stDeployButton {{display: none;}}
        
        /* Main */
        .main {{ padding: 0.5rem 3rem 2rem 3rem; background: #f5f5f7; }}
        
        /* Hero - Navy gradient */
        .hero-section {{
            background: linear-gradient(135deg, #003d5c 0%, #00274C 100%);
            border-radius: 20px;
            padding: 1.5rem 3rem;
            margin-bottom: 1.5rem;
            color: white;
            position: relative;
        }}
        
        .hero-section h1 {{
            color: white !important;
            font-size: 2.75rem !important;
            font-weight: 700 !important;
            margin: 0 0 0.5rem 0 !important;
        }}
        
        .hero-subtitle {{
            color: {UM_MAIZE} !important;
            font-size: 1.1rem !important;
            font-weight: 500 !important;
            margin: 0 0 1rem 0 !important;
        }}
        
        .team-names {{
            color: rgba(255,255,255,0.8);
            font-size: 0.95rem;
        }}
        
        .version-badge {{
            position: absolute;
            top: 2.5rem;
            right: 3rem;
            background: {UM_MAIZE};
            color: {UM_BLUE};
            padding: 0.4rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.875rem;
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: {UM_BLUE};
            color: white;
        }}
        
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
            color: white !important;
            font-weight: 600 !important;
        }}
        
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{
            color: rgba(255,255,255,0.9) !important;
        }}
        
        [data-testid="stSidebar"] .stMarkdown {{
            color: rgba(255,255,255,0.9);
        }}
        
        /* Expanders in sidebar */
        [data-testid="stSidebar"] .streamlit-expanderHeader {{
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            color: white !important;
            font-weight: 600;
        }}
        
        [data-testid="stSidebar"] [data-testid="stExpander"] {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
        }}
        
        /* Cards matching Figma */
        .upload-card {{
            background: white;
            border: 2px solid {UM_MAIZE};
            border-radius: 16px;
            padding: 2.5rem;
            text-align: center;
            transition: all 0.3s ease;
            height: 100%;
            cursor: pointer;
        }}
        
        .upload-card:hover {{
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transform: translateY(-5px);
        }}
        
        /* Make card buttons look like cards */
        div[data-testid="column"] .stButton > button {{
            background: white !important;
            border: 2px solid {UM_MAIZE} !important;
            border-radius: 16px !important;
            padding: 2.5rem !important;
            text-align: center !important;
            height: auto !important;
            min-height: 280px !important;
            color: #1e293b !important;
            font-weight: 400 !important;
            white-space: pre-line !important;
            line-height: 1.6 !important;
            font-size: 1rem !important;
        }}
        
        div[data-testid="column"] .stButton > button:hover {{
            box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
            transform: translateY(-5px) !important;
            background: white !important;
            border-color: {UM_MAIZE} !important;
        }}
        
        div[data-testid="column"] .stButton > button strong {{
            font-size: 1.5rem;
            font-weight: 600;
            display: block;
            margin: 1rem 0;
        }}
        
        .icon-circle {{
            width: 80px;
            height: 80px;
            border-radius: 16px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
        }}
        
        .icon-blue {{ background: #e8f1f8; }}
        .icon-maize {{ background: #fef9e7; }}
        
        .card-title {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 0.75rem;
        }}
        
        .card-description {{
            color: #64748b;
            margin-bottom: 0;
        }}
        
        /* Buttons matching Figma */
        .stButton > button {{
            width: 100%;
            border-radius: 12px;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            border: none;
            transition: all 0.2s ease;
        }}
        
        /* Primary buttons - UM Blue */
        button[kind="primary"] {{
            background: {UM_BLUE} !important;
            color: white !important;
        }}
        
        button[kind="primary"]:hover {{
            background: #003366 !important;
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(0,39,76,0.3);
        }}
        
        /* Architecture cards */
        .arch-card {{
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: left;
        }}
        
        .arch-icon {{
            width: 48px;
            height: 48px;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }}
        
        /* Slider styling - UM Maize (WORKING from v3) */
        .stSlider > div > div > div > div {{
            background-color: {UM_MAIZE} !important;
        }}
        
        .stSlider > div > div > div {{
            background-color: rgba(255, 203, 5, 0.3) !important;
        }}
        
        /* Min/Max labels white text */
        [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
        [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {{
            color: white !important;
        }}
        
        /* Download buttons */
        .stDownloadButton > button {{
            border-radius: 8px;
            font-weight: 600;
        }}
        
        /* Expanders */
        .streamlit-expanderHeader {{
            background: #f8fafc;
            border-radius: 8px;
            color: #1e293b;
            font-weight: 600;
        }}
        
        /* File uploader */
        [data-testid="stFileUploader"] {{
            border: 2px dashed #d1d5db;
            border-radius: 12px;
            padding: 2rem;
            background: white;
        }}
        
        h2 {{
            color: #1e293b !important;
            font-weight: 600 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_resource(show_spinner=False)
def load_model(weights_path: Path) -> YOLO:
    return YOLO(str(weights_path))

@st.cache_data(show_spinner=False)
def _load_font(size: int = 18) -> ImageFont.FreeTypeFont:
    for font_name in ("Inter-Regular.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()

def _extract_text_from_bbox(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    label: str = "",
) -> dict:
    """Extract text from a bounding box region using Tesseract OCR."""
    if not TESSERACT_AVAILABLE:
        return {"text": "", "ocr_confidence": 0, "word_count": 0}
    
    try:
        x1, y1, x2, y2 = bbox
        padding = 10
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(image.width, int(x2) + padding)
        y2 = min(image.height, int(y2) + padding)

        cropped = image.crop((x1, y1, x2, y2))
        
        if cropped.mode != 'RGB':
            cropped = cropped.convert('RGB')

        width = x2 - x1
        height = y2 - y1
        area_ratio = (width * height) / (image.width * image.height)

        # Choose PSM mode based on region type and size
        if area_ratio > 0.3 or label.lower() in ["body", "document", "page", "form", "table"]:
            psm_mode = 3  # Fully automatic page segmentation
        elif label.lower() in ["header", "title", "heading", "footer"]:
            psm_mode = 3
        elif area_ratio > 0.15:
            psm_mode = 6  # Assume uniform block of text
        elif height > width * 2:
            psm_mode = 6  # Tall region
        elif width > height * 3:
            psm_mode = 7  # Single line
        else:
            psm_mode = 7  # Single line default

        text = pytesseract.image_to_string(
            cropped, config=f"--psm {psm_mode}"
        ).strip()

        data = pytesseract.image_to_data(
            cropped, output_type=pytesseract.Output.DICT
        )
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
        st.error(f"OCR Error for {label}: {str(e)}")
        return {"text": "", "ocr_confidence": 0, "word_count": 0, "error": str(e)}

def _format_detections(result, image: Image.Image = None, extract_text: bool = False) -> pd.DataFrame:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        base_cols = ["label", "confidence", "x1", "y1", "x2", "y2"]
        if extract_text:
            base_cols.extend(["extracted_text", "ocr_confidence", "word_count"])
        return pd.DataFrame(columns=base_cols)
    
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
            label = row_data["label"]
            ocr_result = _extract_text_from_bbox(image, bbox, label=label)
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
        background = (box[0], label_y, box[0] + text_width + 14, label_y + text_height + 10)
        draw.rectangle(background, fill=UM_MAIZE)
        draw.text((background[0] + 6, background[1] + 4), text, fill=UM_BLUE, font=font)
    return annotated

def _run_inference(image: Image.Image, confidence: float, iou: float, extract_text: bool = True) -> Tuple[Image.Image, pd.DataFrame, float]:
    model = load_model(DEFAULT_MODEL_PATH)
    start_time = time.time()
    result = model.predict(image, conf=confidence, iou=iou, verbose=False)[0]
    inference_time = time.time() - start_time
    detections = _format_detections(result, image=image, extract_text=extract_text)
    annotated = _annotate_umich(image, detections)
    return annotated, detections, inference_time

def _bytes_to_image(data: bytes, filename: str = "") -> Image.Image:
    if filename.lower().endswith('.pdf'):
        if not PDF_SUPPORT:
            raise ImportError("PDF support requires pdf2image")
        images = convert_from_bytes(data, first_page=1, last_page=1)
        return images[0].convert("RGB")
    else:
        return Image.open(io.BytesIO(data)).convert("RGB")


def main() -> None:
    st.set_page_config(
        page_title="Financial Form Text Extractor",
        layout="wide",
    )
    st.title(
        "University of Michigan \n Master of Applied Data Science - Capstone Project \nFinancial Form Text Extractor"
    )
    st.caption(
        "Upload a JPG/PNG/PDF from a financial form to generate bounding boxes and extract text using Optical Character Recognition (OCR)."
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

        enhanced_ocr = False

        st.info(
            "Using optimized Tesseract OCR with smart PSM mode selection:\n"
            "- Large regions (body, tables): PSM 3 (automatic)\n"
            "- Headers/footers: PSM 3 (automatic)\n"
            "- Form fields: PSM 7 (single line)"
        )

        st.write("Weights file:")
        st.code(str(DEFAULT_MODEL_PATH))

        if st.button(
            "🔄 Reload Model", help="Clear cache and reload model weights from disk"
        ):
            st.cache_resource.clear()
            st.success("Model cache cleared! Reloading...")
            st.rerun()

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
                    raw_image,
                    confidence,
                    iou,
                    extract_text=extract_text,
                    enhanced_ocr=enhanced_ocr,
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
                # caption=f"UM-Branded Bounding Boxes{page_suffix}",
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
                col1, col2, col3 = st.columns(3)

                with col1:
                    # Text download - clean formatted text only
                    text_buffer = io.StringIO()
                    text_buffer.write(f"Extracted Text from {uploaded.name}\n")
                    if len(images) > 1:
                        text_buffer.write(f"Page {page_num} of {len(images)}\n")
                    text_buffer.write("=" * 60 + "\n\n")

                    for _, row in detections.iterrows():
                        if (
                            row["extracted_text"]
                            and len(row["extracted_text"].strip()) > 0
                        ):
                            text_buffer.write(f"[{row['label'].upper()}]\n")
                            text_buffer.write(f"{row['extracted_text']}\n")
                            text_buffer.write(
                                f"(Confidence: {row['confidence']:.1%}, OCR: {row['ocr_confidence']:.1f}%)\n"
                            )
                            text_buffer.write("-" * 60 + "\n\n")

                    text_filename = (
                        f"{Path(uploaded.name).stem}_page{page_num}_extracted_text.txt"
                        if len(images) > 1
                        else f"{Path(uploaded.name).stem}_extracted_text.txt"
                    )
                    st.download_button(
                        label="📥 Download Extracted Text",
                        data=text_buffer.getvalue(),
                        file_name=text_filename,
                        mime="text/plain",
                        key=f"txt-{uploaded.name}-page{page_num}",
                    )

                with col2:
                    # CSV download
                    csv_buffer = io.StringIO()
                    detections.to_csv(csv_buffer, index=False)
                    csv_filename = (
                        f"{Path(uploaded.name).stem}_page{page_num}_ocr_results.csv"
                        if len(images) > 1
                        else f"{Path(uploaded.name).stem}_ocr_results.csv"
                    )
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_buffer.getvalue(),
                        file_name=csv_filename,
                        mime="text/csv",
                        key=f"csv-{uploaded.name}-page{page_num}",
                    )

                with col3:
                    # JSON download
                    json_data = detections.to_dict(orient="records")
                    json_filename = (
                        f"{Path(uploaded.name).stem}_page{page_num}_ocr_results.json"
                        if len(images) > 1
                        else f"{Path(uploaded.name).stem}_ocr_results.json"
                    )
                    st.download_button(
                        label="📥 Download JSON",
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
