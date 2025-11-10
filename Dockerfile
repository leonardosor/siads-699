FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl tesseract-ocr tesseract-ocr-eng libtesseract-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ultralytics==8.0.196 pdf2image pillow pytesseract streamlit psycopg2-binary numpy==1.26.4

WORKDIR /workspace
COPY src/streamlit /workspace/src/streamlit
COPY models /workspace/models

EXPOSE 8501
CMD ["streamlit", "run", "src/streamlit/streamlit_app.py", "--server.port=8501"]
