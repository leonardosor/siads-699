#!/usr/bin/env bash
set -euo pipefail

repo_root="$(pwd)"
need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }; }
need_cmd docker

compose_cmd="docker compose"
if ! docker compose version >/dev/null 2>&1; then
  if command -v docker-compose >/dev/null 2>&1; then compose_cmd="docker-compose"; else
    echo "Docker Compose plugin not found; install Docker Desktop or docker-compose." >&2; exit 1
  fi
fi

mkdir -p docs src/streamlit scripts data/output
touch docs/README.md

if [ ! -f ".env" ]; then
  pass="${RANDOM}${RANDOM}${RANDOM}"
  cat > .env <<EOF
POSTGRES_DB=appdb
POSTGRES_USER=appuser
POSTGRES_PASSWORD=${pass}
PGDATA_VOLUME=pgdata
APP_PORT=8501
DB_PORT=5432
MODEL_PATH=/app/models/best.pt
EOF
fi

if [ ! -s "src/streamlit/streamlit_app.py" ]; then
  cat > src/streamlit/streamlit_app.py <<'EOF'
import os, tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pdf2image import convert_from_bytes
import cv2

st.set_page_config(page_title="Document Object Detection", layout="wide")
st.title("PDF/JPG/PNG Object Detection")

model_path = os.getenv("MODEL_PATH", "/app/models/best.pt")
if not os.path.exists(model_path):
    st.error(f"Model not found at {model_path}")
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
    plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
    return plot

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
        st.success(f"Saved: {out_path}")
        with open(out_path, "rb") as f:
            st.download_button("Download annotated image", f, file_name=os.path.basename(out_path))
st.caption("For PDF multi-page support, extend first/last_page parameters.")
EOF
fi

cat > src/docker/requirements.txt <<'EOF'
ultralytics==8.2.103
streamlit==1.39.0
pillow>=10.3.0
numpy>=1.26.4
opencv-python-headless>=4.10.0.84
pdf2image>=1.17.0
psycopg2-binary>=2.9.9
EOF

mkdir -p src/docker
cat > src/docker/Dockerfile <<'EOF'
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      tesseract-ocr libtesseract-dev poppler-utils \
      libglib2.0-0 libgl1 ca-certificates curl && \
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
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL","curl -sSf http://localhost:8501/_stcore/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 12

  db:
    image: postgres:16
    env_file: .env
    ports:
      - "${DB_PORT:-5432}:5432"
    volumes:
      - ${PGDATA_VOLUME:-pgdata}:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL","pg_isready -U ${POSTGRES_USER:-appuser} -d ${POSTGRES_DB:-appdb}"]
      interval: 10s
      timeout: 5s
      retries: 12

volumes:
  app_output:
  pgdata:
EOF

if [ ! -s "scripts/init-db.sql" ]; then
  cat > scripts/init-db.sql <<'EOF'
CREATE TABLE IF NOT EXISTS inference_runs (
  id BIGSERIAL PRIMARY KEY,
  filename TEXT NOT NULL,
  saved_path TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
EOF
fi

if [ ! -f ".dockerignore" ]; then
  cat > .dockerignore <<'EOF'
.git
**/__pycache__
**/*.pyc
runs
weights
notebooks
docs
*.ipynb_checkpoints*
EOF
fi

if [ -d "models" ] && [ ! -f "models/best.pt" ]; then
  echo "WARNING: models/best.pt not found in ./models. Place your trained weights there." >&2
fi

mkdir -p docs
cat > docs/citations_automated.txt <<'EOF'
This deployment stack was prepared in collaboration with an AI assistant (OpenAI GPT-5 Thinking).
If referenced academically, a generic software-collaboration citation format is appropriate, e.g.:

OpenAI. “GPT-5 Thinking (Assistant),” collaborative systems engineering guidance for containerized ML pipelines (YOLOv8/Tesseract/PostgreSQL/Streamlit), 2025. Contribution: environment orchestration, reproducible builds, and runbook generation. URL: https://openai.com/

For reproducibility statements, include Dockerfile/Compose digests and package versions recorded in src/docker/requirements.txt and compose.yml.
EOF

echo "Building images…"
$compose_cmd -f compose.yml build

echo "Starting services…"
$compose_cmd -f compose.yml up -d

echo "Waiting for services to report healthy…"
$compose_cmd -f compose.yml ps

echo
echo "App:       http://localhost:${APP_PORT:-8501}"
echo "Postgres:  localhost:${DB_PORT:-5432}  db=${POSTGRES_DB:-appdb} user=${POSTGRES_USER:-appuser}"
echo
echo "Follow logs:"
echo "  $compose_cmd logs -f db"
echo "  $compose_cmd logs -f app"

