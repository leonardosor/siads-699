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
COPY src/environments/requirements-prod.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

COPY src/web /app/src/web

ENV MODEL_PATH=/app/models/trained/best.pt
EXPOSE 8501
CMD ["bash","-lc","streamlit run /app/src/web/streamlit_application.py --server.port=8501 --server.address=0.0.0.0"]
