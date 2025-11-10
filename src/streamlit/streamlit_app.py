import os, numpy as np, streamlit as st
from ultralytics import YOLO
from PIL import Image
from pdf2image import convert_from_bytes

st.set_page_config(page_title="YOLO Diagnostic", layout="wide")
st.title("YOLO Diagnostic – trace model output")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/best.pt")
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}")
    st.stop()

st.write(f"Using model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
conf = st.slider("Confidence", 0.01, 0.9, 0.1, 0.01)
uploaded = st.file_uploader("Upload image or PDF", type=["jpg","jpeg","png","pdf"])

def pdf_to_pil(pdf_bytes):
    return convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=1)[0].convert("RGB")

if uploaded:
    if uploaded.type == "application/pdf":
        pil = pdf_to_pil(uploaded.read())
    else:
        pil = Image.open(uploaded).convert("RGB")

    st.write("Running inference...")
    arr = np.array(pil)
    try:
        results = model.predict(arr, conf=conf, verbose=True, device="cpu")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    if not results:
        st.error("Model returned no results object at all.")
    else:
        r = results[0]
        st.write("Raw result type:", type(r))
        if not hasattr(r, "boxes") or r.boxes is None:
            st.error("Result has no boxes attribute – likely bad weights or wrong model file.")
        else:
            st.success(f"Boxes tensor found: shape {r.boxes.xyxy.shape}")
            boxes = r.boxes.xyxy.cpu().numpy()
            st.write("First few boxes:")
            st.json(boxes.tolist()[:5])
            st.image(arr, caption=f"Image shape: {arr.shape}")

