# Rewritten by ChatGPT

# Lightweight Python base
FROM python:3.12-slim

# Set up working directory
WORKDIR /workspace

# Install system dependencies (OCR, compilers, Postgres client libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Optional: prevent pip cache buildup
ENV PIP_NO_CACHE_DIR=1

# Copy dependencies and install
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Keep container alive for VS Code Dev Container use
CMD ["sleep", "infinity"]