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
    else Path(__file__).resolve().parents[2] / "models" / "production" / "best.pt"
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


def render_sidebar(confidence: float, iou: float, show_controls: bool = False):
    """Render left sidebar"""
    with st.sidebar:
        if show_controls:

            st.markdown("## ⚙️ Inference Controls")
            
            st.markdown(f"**Confidence Threshold** `{confidence:.2f}`")
            new_conf = st.slider("", 0.10, 0.90, confidence, 0.05, label_visibility="collapsed", key="conf_sl")
            
            st.markdown(f"**IoU Threshold** `{iou:.2f}`")
            new_iou = st.slider("", 0.10, 0.90, iou, 0.05, label_visibility="collapsed", key="iou_sl")
            
            st.markdown("---")
            
            with st.expander("📁 Model Information", expanded=True):
                st.markdown("**Model:** YOLOv8")
                if DEFAULT_MODEL_PATH.exists():
                    size_mb = DEFAULT_MODEL_PATH.stat().st_size / (1024 * 1024)
                    st.markdown(f"**Status:** ✅ Loaded")
                    st.markdown(f"**Size:** {size_mb:.1f} MB")
                    st.markdown(f"**Path:** `{DEFAULT_MODEL_PATH.name}`")
                else:
                    st.markdown("**Status:** ❌ Not found")
            
            with st.expander("💡 About this tool", expanded=False):
                st.markdown("""
                    This application uses YOLOv8 to detect and extract text regions (header, body, footer) 
                    from financial form images. Supported formats: JPG, PNG, PDF. Upload your scanned forms 
                    to get accurate bounding boxes and detailed detection data.
                """)
            
            with st.expander("ℹ️ Quick Guide", expanded=True):
                st.markdown("""
                    **How to use:**
                    
                    1. Choose upload mode on home
                    2. Upload JPG/PNG/PDF images
                    3. Adjust thresholds if needed
                    4. Download results
                    
                    **Detection Classes:**
                    - 📋 Header
                    - 📝 Body  
                    - 🔽 Footer
                """)
        else:
            with st.expander("💡 About this tool", expanded=True):
                st.markdown("""
                    This application uses YOLOv8 to detect and extract text regions (header, body, footer) 
                    from financial form images. Supported formats: JPG, PNG, PDF. Upload your scanned forms 
                    to get accurate bounding boxes and detailed detection data.
                """)
            
            with st.expander("ℹ️ Quick Guide", expanded=True):
                st.markdown("""
                    **How to use:**
                    
                    1. Choose upload mode on home
                    2. Upload JPG/PNG/PDF images
                    3. Adjust thresholds if needed
                    4. Download results
                    
                    **Detection Classes:**
                    - 📋 Header
                    - 📝 Body  
                    - 🔽 Footer
                """)
            
            with st.expander("📁 Model Information", expanded=False):
                st.markdown("**Model:** YOLOv8")
                if DEFAULT_MODEL_PATH.exists():
                    size_mb = DEFAULT_MODEL_PATH.stat().st_size / (1024 * 1024)
                    st.markdown(f"**Status:** ✅ Loaded")
                    st.markdown(f"**Size:** {size_mb:.1f} MB")
                    st.markdown(f"**Path:** `{DEFAULT_MODEL_PATH.name}`")
                else:
                    st.markdown("**Status:** ❌ Not found")
            
            new_conf = confidence
            new_iou = iou
    
    return new_conf, new_iou


def render_landing_page(confidence: float, iou: float) -> None:
    """Landing page matching Figma design."""
    st.markdown(
        f"""
        <div class='hero-section'>
            <h1>Financial Form Text Extractor</h1>
            <p class='hero-subtitle'>University of Michigan MADS Capstone Fall 2025</p>
            <p class='team-names'>Joey Higgins • Leonardo Cedeno • Kajal Dattatray Raut • Denesh Chandrahasan</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<h2 style='text-align: center; color: #1e293b;'>Choose Your Upload Mode</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; margin-bottom: 2rem;'>Select how you want to process your financial forms</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown(
            """
            <div class='upload-card'>
                <div class='icon-circle icon-blue'>🎯</div>
                <div class='card-title'>Try Me Out</div>
                <p class='card-description'>Test the app instantly with a preloaded demo form</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("⚡ Launch Demo", key="demo", type="primary"):
            st.session_state.page = "demo"
            st.rerun()
    
    with col2:
        st.markdown(
            """
            <div class='upload-card'>
                <div class='icon-circle icon-maize'>📤</div>
                <div class='card-title'>Upload Your Forms</div>
                <p class='card-description'>Process your own forms (single or multiple files)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📂 Start Upload", key="upload", type="primary"):
            st.session_state.page = "upload"
            st.rerun()


def render_demo_page(confidence: float, iou: float) -> None:
    col_back, col_spacer = st.columns([1, 5])
    with col_back:
        if st.button("← Back to Home", key="back_demo"):
            st.session_state.page = "home"
            st.rerun()
    
    demo_path = Path(__file__).parent / "demo_form.jpg"
    if not demo_path.exists():
        st.error("Demo form not found")
        return
    
    raw_image = Image.open(demo_path).convert("RGB")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.markdown(
            """
            <style>
            button[key="run_demo"] {
                font-size: 1.2rem !important;
                font-weight: 700 !important;
                padding: 0.75rem 1.5rem !important;
                width: 100% !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        if st.button("Run Detection", key="run_demo", type="primary"):
            with st.spinner("Running inference..."):
                annotated, detections, inf_time = _run_inference(raw_image, confidence, iou)
            
            st.session_state.demo_results = {
                'annotated': annotated,
                'detections': detections,
                'inf_time': inf_time,
                'raw_image': raw_image
            }
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        import base64
        buffered = io.BytesIO()
        raw_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        st.markdown(
            f"""
            <div style='border: 3px solid {UM_BLUE}; border-radius: 8px; padding: 0.5rem; background: white; margin-top: 1rem;'>
                <img src='data:image/jpeg;base64,{img_str}' style='width: 100%; border-radius: 4px;' />
                <p style='text-align: center; color: #64748b; margin-top: 0.5rem; margin-bottom: 0; font-size: 0.9rem;'>Demo Financial Statement</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    if 'demo_results' in st.session_state:
        results = st.session_state.demo_results
        detections = results['detections']
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            buf = io.BytesIO()
            results['annotated'].save(buf, format="JPEG", quality=95)
            st.download_button(
                "Download annotated image",
                buf.getvalue(),
                "demo_annotated.jpg",
                "image/jpeg",
                key="download_annotated",
                type="primary",
                use_container_width=True
            )
        
        st.markdown("### Results")
        c1, c2 = st.columns(2)
        with c1:
            st.image(results['raw_image'], caption="Original", use_column_width=True)
        with c2:
            st.image(results['annotated'], caption="Detected", use_column_width=True)
        
        st.subheader("Detections")
        
        if not detections.empty:
            display_df = detections.copy()
            
            display_df['confidence'] = (display_df['confidence'] * 100).round(1).astype(str) + '%'
            
            if 'ocr_confidence' in display_df.columns:
                display_df['ocr_confidence'] = display_df['ocr_confidence'].round(1).astype(str) + '%'
            
            cols_order = ['label']
            if 'extracted_text' in display_df.columns:
                cols_order.append('extracted_text')
            cols_order.extend(['confidence'])
            if 'ocr_confidence' in display_df.columns:
                cols_order.append('ocr_confidence')
            if 'word_count' in display_df.columns:
                cols_order.append('word_count')
            cols_order.extend(['x1', 'y1', 'x2', 'y2'])
            
            display_df = display_df[[col for col in cols_order if col in display_df.columns]]
            
            st.dataframe(display_df, use_container_width=True)
        
        if 'extracted_text' in detections.columns:
            col_left, col1, col2, col3, col_right = st.columns([1, 1, 1, 1, 1])
            
            with col1:
                text_buffer = io.StringIO()
                for idx, row in detections.iterrows():
                    if row.get('extracted_text') and len(row['extracted_text'].strip()) > 0:
                        text_buffer.write(f"{row['label'].upper()}\n")
                        text_buffer.write(f"{row['extracted_text']}\n")
                        text_buffer.write(f"(Confidence: {row['confidence']:.1%}, OCR: {row['ocr_confidence']:.1f}%)\n")
                        text_buffer.write("\n" + "="*50 + "\n\n")
                
                st.download_button(
                    "📥 Download Extracted Text",
                    text_buffer.getvalue(),
                    "demo_extracted_text.txt",
                    "text/plain",
                    key="download_text",
                    type="primary",
                    use_container_width=True
                )
            
            with col2:
                csv_buffer = io.StringIO()
                detections.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_buffer.getvalue(),
                    "demo_detections.csv",
                    "text/csv",
                    key="download_csv",
                    type="primary",
                    use_container_width=True
                )
            
            with col3:
                json_str = detections.to_json(orient='records', indent=2)
                st.download_button(
                    "📥 Download JSON",
                    json_str,
                    "demo_detections.json",
                    "application/json",
                    key="download_json",
                    type="primary",
                    use_container_width=True
                )
        
        if 'extracted_text' in detections.columns:
            texts_found = detections[detections['extracted_text'].str.len() > 0]
            if len(texts_found) > 0:
                st.subheader("Extracted Text Summary")
                st.info(f"Found text in {len(texts_found)} out of {len(detections)} detections")
                
                # Display each extracted text with its label
                for idx, row in texts_found.iterrows():
                    with st.expander(
                        f"📄 {row['label']} (confidence: {row['confidence']:.1%})"
                    ):
                        st.write(row['extracted_text'])
                        st.caption(
                            f"OCR Confidence: {row['ocr_confidence']:.1f}% | Words: {row['word_count']}"
                        )
            else:
                st.warning("No text extracted from detected regions. Try adjusting confidence thresholds.")


def render_upload_page(confidence: float, iou: float) -> None:
    col_back, col_spacer = st.columns([1, 5])
    with col_back:
        if st.button("← Back to Home", key="back_upload"):
            st.session_state.page = "home"
            st.rerun()
    
    st.markdown("<h2 style='text-align: center;'>Upload Your Forms</h2>", unsafe_allow_html=True)
    
    uploads = st.file_uploader("", type=ALLOWED_TYPES, accept_multiple_files=True, label_visibility="collapsed")
    
    if uploads:
        for idx, uploaded in enumerate(uploads):
            try:
                raw_image = _bytes_to_image(uploaded.getvalue(), uploaded.name)
                
                result_key = f"upload_result_{idx}_{uploaded.name}"
                
                if result_key not in st.session_state:
                    with st.spinner(f"Processing {uploaded.name}..."):
                        annotated, detections, inf_time = _run_inference(raw_image, confidence, iou)
                    
                    st.session_state[result_key] = {
                        'annotated': annotated,
                        'detections': detections,
                        'inf_time': inf_time,
                        'raw_image': raw_image,
                        'filename': uploaded.name
                    }
                
                results = st.session_state[result_key]
                detections = results['detections']
                
                st.markdown("---")
                st.markdown(f"<h4 style='text-align: center; font-size: 1.1rem;'>{uploaded.name}</h4>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    buf = io.BytesIO()
                    results['annotated'].save(buf, format="JPEG", quality=95)
                    st.download_button(
                        "Download annotated image",
                        buf.getvalue(),
                        f"{Path(uploaded.name).stem}_annotated.jpg",
                        "image/jpeg",
                        key=f"download_annotated_{idx}",
                        type="primary",
                        use_container_width=True
                    )
                
                st.markdown("### Results")
                c1, c2 = st.columns(2)
                with c1:
                    st.image(results['raw_image'], caption="Original", use_column_width=True)
                with c2:
                    st.image(results['annotated'], caption="Detected", use_column_width=True)
                
                st.subheader("Detections")
                
                if not detections.empty:
                    display_df = detections.copy()
                    
                    display_df['confidence'] = (display_df['confidence'] * 100).round(1).astype(str) + '%'
                    
                    if 'ocr_confidence' in display_df.columns:
                        display_df['ocr_confidence'] = display_df['ocr_confidence'].round(1).astype(str) + '%'
                    
                    cols_order = ['label']
                    if 'extracted_text' in display_df.columns:
                        cols_order.append('extracted_text')
                    cols_order.extend(['confidence'])
                    if 'ocr_confidence' in display_df.columns:
                        cols_order.append('ocr_confidence')
                    if 'word_count' in display_df.columns:
                        cols_order.append('word_count')
                    cols_order.extend(['x1', 'y1', 'x2', 'y2'])
                    
                    display_df = display_df[[col for col in cols_order if col in display_df.columns]]
                    
                    st.dataframe(display_df, use_container_width=True)
                
                if 'extracted_text' in detections.columns:
                    col_left, col1, col2, col3, col_right = st.columns([1, 1, 1, 1, 1])
                    
                    with col1:
                        text_buffer = io.StringIO()
                        for _, row in detections.iterrows():
                            if row.get('extracted_text') and len(row['extracted_text'].strip()) > 0:
                                text_buffer.write(f"{row['label'].upper()}\n")
                                text_buffer.write(f"{row['extracted_text']}\n")
                                text_buffer.write(f"(Confidence: {row['confidence']:.1%}, OCR: {row['ocr_confidence']:.1f}%)\n")
                                text_buffer.write("\n" + "="*50 + "\n\n")
                        
                        st.download_button(
                            "📥 Download Extracted Text",
                            text_buffer.getvalue(),
                            f"{Path(uploaded.name).stem}_extracted_text.txt",
                            "text/plain",
                            key=f"download_text_{idx}",
                            type="primary",
                            use_container_width=True
                        )
                    
                    with col2:
                        csv_buffer = io.StringIO()
                        detections.to_csv(csv_buffer, index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv_buffer.getvalue(),
                            f"{Path(uploaded.name).stem}_detections.csv",
                            "text/csv",
                            key=f"download_csv_{idx}",
                            type="primary",
                            use_container_width=True
                        )
                    
                    with col3:
                        json_str = detections.to_json(orient='records', indent=2)
                        st.download_button(
                            "📥 Download JSON",
                            json_str,
                            f"{Path(uploaded.name).stem}_detections.json",
                            "application/json",
                            key=f"download_json_{idx}",
                            type="primary",
                            use_container_width=True
                        )
                
                if 'extracted_text' in detections.columns:
                    texts_found = detections[detections['extracted_text'].str.len() > 0]
                    if len(texts_found) > 0:
                        st.subheader("Extracted Text Summary")
                        st.info(f"Found text in {len(texts_found)} out of {len(detections)} detections")
                        
                        # Display each extracted text with its label
                        for _, row in texts_found.iterrows():
                            with st.expander(
                                f"📄 {row['label']} (confidence: {row['confidence']:.1%})"
                            ):
                                st.write(row['extracted_text'])
                                st.caption(
                                    f"OCR Confidence: {row['ocr_confidence']:.1f}% | Words: {row['word_count']}"
                                )
                    else:
                        st.warning("No text extracted from detected regions. Try adjusting confidence thresholds.")
                
            except Exception as e:
                st.error(f"Error processing {uploaded.name}: {str(e)}")



def main() -> None:
    inject_custom_css()
    
    if not DEFAULT_MODEL_PATH.exists():
        st.error("❌ Model weights not found")
        return
    
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "confidence" not in st.session_state:
        st.session_state.confidence = 0.45
    if "iou" not in st.session_state:
        st.session_state.iou = 0.50
    
    if st.session_state.page in ["demo", "upload"]:
        new_conf, new_iou = render_sidebar(st.session_state.confidence, st.session_state.iou, show_controls=True)
        st.session_state.confidence = new_conf
        st.session_state.iou = new_iou
    else:
        render_sidebar(st.session_state.confidence, st.session_state.iou, show_controls=False)
    
    if st.session_state.page == "home":
        render_landing_page(st.session_state.confidence, st.session_state.iou)
    elif st.session_state.page == "demo":
        render_demo_page(st.session_state.confidence, st.session_state.iou)
    elif st.session_state.page == "upload":
        render_upload_page(st.session_state.confidence, st.session_state.iou)


if __name__ == "__main__":
    main()
