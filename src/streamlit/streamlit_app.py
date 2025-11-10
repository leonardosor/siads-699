import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pdf2image import convert_from_bytes
import tempfile, os, cv2, numpy as np

st.set_page_config(page_title="YOLOv8 Inference", layout="wide")
st.title("📄 PDF / Image Object Detection")

model = YOLO("models/best.pt")

uploaded_file = st.file_uploader("Upload a JPG, PNG, or PDF", type=["jpg","jpeg","png","pdf"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        pages = convert_from_bytes(uploaded_file.read())
        image = pages[0]
    else:
        image = Image.open(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model.predict(source=tmp.name, conf=0.25, save=False)
        annotated = results[0].plot()
        st.image(annotated[:, :, ::-1], caption="Detected Objects", use_column_width=True)
