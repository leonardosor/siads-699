from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

UM_BLUE = "#00274C"
UM_MAIZE = "#FFCB05"
_model_env = os.getenv("MODEL_PATH")
DEFAULT_MODEL_PATH = (
    Path(_model_env)
    if _model_env
    else Path(__file__).resolve().parents[2] / "models" / "best.pt"
)
ALLOWED_TYPES = ("jpg", "jpeg", "png")


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


def _format_detections(result) -> pd.DataFrame:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame(columns=["label", "confidence", "x1", "y1", "x2", "y2"])

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy()
    names = result.names or {}

    rows: List[dict] = []
    for coords, confidence, cls_id in zip(xyxy, conf, cls_ids):
        rows.append(
            {
                "label": names.get(int(cls_id), f"class_{int(cls_id)}"),
                "confidence": float(confidence),
                "x1": float(coords[0]),
                "y1": float(coords[1]),
                "x2": float(coords[2]),
                "y2": float(coords[3]),
            }
        )

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


def _run_inference(image: Image.Image, confidence: float, iou: float) -> Tuple[Image.Image, pd.DataFrame]:
    model = load_model(DEFAULT_MODEL_PATH)
    result = model.predict(image, conf=confidence, iou=iou, verbose=False)[0]
    detections = _format_detections(result)
    annotated = _annotate_umich(image, detections)
    return annotated, detections


def _bytes_to_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def main() -> None:
    st.set_page_config(page_title="Financial Form Text Extractor", layout="wide")
    st.title("Financial Form Text Extractor")
    st.caption("Upload a JPG/PNG to generate Michigan-branded bounding boxes from your YOLOv8 model.")

    if not DEFAULT_MODEL_PATH.exists():
        st.error(
            f"Model weights not found at `{DEFAULT_MODEL_PATH}`. "
            "Copy your trained checkpoint there or set the MODEL_PATH environment variable."
        )
        return

    with st.sidebar:
        st.header("Inference Controls")
        confidence = st.slider("Confidence threshold", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
        iou = st.slider("IoU threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        st.write("Weights file:")
        st.code(str(DEFAULT_MODEL_PATH))

    uploads = st.file_uploader(
        "Upload one or more JPG/PNG files", type=ALLOWED_TYPES, accept_multiple_files=True
    )

    if not uploads:
        st.info("Waiting for uploads… drag a JPG/PNG into the widget above to begin.")
        return

    for uploaded in uploads:
        st.markdown(f"### {uploaded.name}")
        raw_image = _bytes_to_image(uploaded.getvalue())

        st.image(raw_image, caption="Original", use_column_width=True)
        with st.spinner("Running YOLOv8 inference…"):
            annotated, detections = _run_inference(raw_image, confidence, iou)

        if detections.empty:
            st.warning("No bounding boxes detected with the current thresholds.")
            continue

        annotated_buffer = io.BytesIO()
        annotated.save(annotated_buffer, format="JPEG")

        st.image(annotated_buffer.getvalue(), caption="UM-Branded Bounding Boxes", use_column_width=True)
        st.download_button(
            label="Download annotated JPG",
            data=annotated_buffer.getvalue(),
            file_name=f"{Path(uploaded.name).stem}_umich_bboxes.jpg",
            mime="image/jpeg",
            key=f"download-{uploaded.name}",
        )

        st.subheader("Detections")
        pretty = detections.assign(
            confidence=lambda df: (df["confidence"] * 100).round(1).astype(str) + "%"
        )
        st.dataframe(pretty, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
