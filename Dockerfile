# SIADS 699 Capstone - PDF OCR & CNN Training Container
# Base image with PyTorch, CUDA, and cuDNN
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata \
    PATH=/opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    # Tesseract OCR and language data
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    # Image processing libraries
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # PDF processing
    poppler-utils \
    # PostgreSQL client
    postgresql-client \
    libpq-dev \
    # Other utilities
    zip \
    unzip \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create tessdata directory structure
RUN mkdir -p /usr/share/tesseract-ocr/4.00/tessdata && \
    ln -s /usr/share/tesseract-ocr/4.00/tessdata /usr/share/tessdata

# Upgrade pip and install Python build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies for ML, OCR, and data processing
RUN pip install --no-cache-dir \
    # Deep Learning & Computer Vision
    ultralytics==8.0.196 \
    opencv-python==4.8.1.78 \
    opencv-contrib-python==4.8.1.78 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    # OCR & PDF Processing
    pytesseract==0.3.10 \
    pdf2image==1.16.3 \
    PyPDF2==3.0.1 \
    pdfplumber==0.10.3 \
    # Data Science
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    scipy==1.11.3 \
    # Database
    psycopg2-binary==2.9.9 \
    sqlalchemy==2.0.23 \
    # Visualization & Monitoring
    matplotlib==3.7.3 \
    seaborn==0.13.0 \
    tensorboard==2.15.1 \
    # Jupyter & Interactive
    jupyter==1.0.0 \
    jupyterlab==4.0.9 \
    ipywidgets==8.1.1 \
    # Utilities
    tqdm==4.66.1 \
    pillow==10.1.0 \
    requests==2.31.0 \
    pyyaml==6.0.1 \
    python-dotenv==1.0.0

# Install Label Studio SDK for annotation integration
RUN pip install --no-cache-dir label-studio-sdk==0.0.30

# Set working directory
WORKDIR /workspace

# Create directories for data and outputs
RUN mkdir -p /workspace/data/input/ground-truth \
             /workspace/data/input/training \
             /workspace/data/output \
             /workspace/docs \
             /workspace/src/docker \
             /workspace/src/models

# Expose ports for services
EXPOSE 5432 6006 8888 8890

# Configure bash history persistence
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && echo "$SNIPPET" >> "/root/.bashrc"

# Set default command
CMD ["/bin/bash"]
