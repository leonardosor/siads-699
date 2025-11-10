#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────────────────────────────
# Phase 1 – Sanity checks & environment setup
# ────────────────────────────────────────────────
need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1"; exit 1; }; }
need_cmd docker
compose_cmd="docker compose"
if ! docker compose version >/dev/null 2>&1; then
  if command -v docker-compose >/dev/null 2>&1; then compose_cmd="docker-compose"; else
    echo "Docker Compose not found"; exit 1
  fi
fi

repo_root="$(pwd)"
mkdir -p archive/docs archive/src_utilities docs src/docker src/streamlit scripts data/output models

# ────────────────────────────────────────────────
# Phase 2 – Lightweight repo hygiene
# (moves clutter into archive, keeps reproducibility)
# ────────────────────────────────────────────────
for f in docs/utilities docs/feedback_0.txt docs/common_commands*.txt docs/build_fast.txt; do
  [ -e "$f" ] && mv "$f" archive/docs/ || true
done
[ -d src/utilities ] && mv src/utilities archive/src_utilities || true
[ -f docker-compose.yml ] && mv docker-compose.yml archive/ || true
[ -f src/docker-compose.yml ] && mv src/docker-compose.yml archive/ || true
echo "→ Repo cleanup complete → see archive/"

# ────────────────────────────────────────────────
# Phase 3 – Core environment files
# ────────────────────────────────────────────────
if [ ! -f ".env" ]; then
  cat > .env <<EOF
APP_PORT=8501
MODEL_PATH=/app/models/best.pt
EOF
fi

cat > src/docker/requirements.txt <<'EOF'
ultralytics==8.2.103
streamlit==1.39.0
pillow>=10.3.0
numpy>=1.26.4
opencv-python-headless>=4.10.0.84
pdf2image>=1.17.0
EOF

cat > src/docker/Dockerfile <<'EOF'
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libtesseract-dev poppler-utils libglib2.0-0 libgl1 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY src/docker/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt
COPY src/streamlit /app/src/streamlit
ENV MODEL_PATH=/app/models/best.pt
EXPOSE 8501
CMD ["bash","-lc","streamlit run /app/src/streamlit/streamlit_app.py --server.port=8501 --server.address=0.0.0.0"]
EOF

cat > compose.yml <<'EOF'
name: siads-699
services:
  app:
    build:
      context: .
      dockerfile: src/docker/Dockerfile
    env_file: .env
    ports:
      - "${APP_PORT:-8501}:8501"
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data:ro
      - app_output:/app/output
    healthcheck:
      test: ["CMD-SHELL","curl -sSf http://localhost:8501/_stcore/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 10
volumes:
  app_output:
EOF

# ────────────────────────────────────────────────
# Phase 4 – Streamlit application
# ────────────────────────────────────────────────
cat > src/streamlit/streamlit_app.py <<'EOF'
import os, numpy as np, cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pdf2image import convert_from_bytes

st.set_page_config(page_title="Document Object Detection", layout="wide")
st.title("PDF / Image Object Detection Pipeline")

model_path = os.getenv("MODEL_PATH", "/app/models/best.pt")
if not os.path.exists(model_path):
    st.error(f"Model weights not found → {model_path}")
    st.stop()
model = YOLO(model_path)

uploaded = st.file_uploader("Upload a JPG/PNG/PDF", type=["jpg","jpeg","png","pdf"])
save_out = st.checkbox("Save annotated output", value=True)
conf = st.slider("Confidence", 0.1, 0.9, 0.25, 0.05)

def run_infer(pil_image):
    arr = np.array(pil_image.convert("RGB"))
    res = model.predict(source=arr, conf=conf, verbose=False)
    if len(res)==0:
        return arr
    plot = res[0].plot()
    return cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)

if uploaded:
    if uploaded.type == "application/pdf":
        pages = convert_from_bytes(uploaded.read(), dpi=200, first_page=1, last_page=1)
        img = pages[0]
    else:
        img = Image.open(uploaded).convert("RGB")
    out = run_infer(img)
    st.image(out, caption="Annotated", use_container_width=True)
    if save_out:
        os.makedirs("/app/output", exist_ok=True)
        base = os.path.splitext(uploaded.name)[0]
        out_path = f"/app/output/{base}_annotated.png"
        Image.fromarray(out).save(out_path)
        st.success(f"Saved → {out_path}")
        with open(out_path,"rb") as f:
            st.download_button("Download annotated image",f,file_name=os.path.basename(out_path))
EOF

cat > docs/citations_automated.txt <<'EOF'
This deployment script and containerized environment were developed collaboratively with OpenAI GPT-5 (Assistant, 2025).
Citation format: OpenAI. “GPT-5 Assistant for Containerized ML Workflows.” Collaborative Engineering Guidance 2025. URL: https://openai.com/
EOF

# ────────────────────────────────────────────────
# Phase 5 – Build and launch
# ────────────────────────────────────────────────
echo "→ Building container image..."
$compose_cmd -f compose.yml build
echo "→ Launching Streamlit application..."
$compose_cmd -f compose.yml up -d
$compose_cmd ps
echo
echo "App available at http://localhost:${APP_PORT:-8501}"
echo "Use '$compose_cmd logs -f app' to monitor runtime output."

