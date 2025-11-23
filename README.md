# SIADS 699 - MADS Capstone
## Financial Form Text Extractor

### Overview
A full-stack text extraction application for scanned financial forms, combining YOLOv8 object detection, Tesseract OCR, PostgreSQL storage, and a Streamlit web interface. The system uses manually labeled training data (Label Studio) to fine-tune a convolutional neural network for detecting document regions (header, body, footer), enabling targeted text extraction from form bodies.

### AI Assistance Disclosure
Consistent with MADS academic integrity guidelines, assume that OpenAI's ChatGPT 5 (large language model, 2024 release) materially assisted in producing the source code within this repository. When citing or reusing this work, please include the acknowledgement: *OpenAI. (2024). ChatGPT 5 (large language model) [Computer software]. Assistance provided to the SIADS 699 Financial Form Text Extractor project.*

## Table of Contents
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Key Components](#key-components)
- [Common Workflows](#common-workflows)
- [Documentation](#documentation)
- [Environment Configuration](#environment-configuration)

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Trained YOLOv8 model weights at [models/trained/best.pt](models/trained/best.pt)

### Launch the Application

```bash
# 1. Start all services
docker compose -f src/environments/docker/compose.yml up --build --remove-orphans

# 2. Access Streamlit UI
open http://localhost:8501

# 3. Tear down when finished
docker compose -f src/environments/docker/compose.yml down --volumes
```

### What You Can Do
- Upload JPG/PNG document images for YOLOv8 inference
- Adjust confidence and IoU thresholds in real-time
- View detections with University of Michigan branded annotations
- Download annotated images

See [docs/README.md](docs/README.md) for detailed quickstart guide and [src/web/README.md](src/web/README.md) for Streamlit usage.

## Architecture

### Logical Architecture
![Logical Architecture](/docs/imgs/architecture_1.png)

### Physical Architecture
![Physical Architecture](/docs/imgs/architecture_0.png)

### Technology Stack
- **Object Detection**: YOLOv8 (Ultralytics)
- **OCR**: Tesseract 5
- **Web Interface**: Streamlit
- **Database**: PostgreSQL 15
- **Containerization**: Docker Compose
- **Training Environment**: Great Lakes HPC (University of Michigan)

## Repository Structure

```
src/
├── training/           # Model training scripts
├── processing/         # OCR and document processing
├── web/                # Streamlit web application
├── database/           # Database schemas and notebooks
├── utils/              # Utility scripts organized by purpose
│   ├── dataset/        # Dataset labeling and validation
│   ├── models/         # Model management utilities
│   ├── monitoring/     # Training monitoring scripts
│   └── deployment/     # Deployment automation
├── environments/       # Environment configurations
│   ├── docker/         # Docker/Compose for local development
│   └── conda/          # Conda environment for Great Lakes HPC
└── config/             # Configuration files
```

### Product Architecture - Logical
![Logical](/docs/imgs/architecture_0.png)

### Product Architecture - Physical
![Physical](/docs/imgs/architecture_1.png)

### Dockerized Development
- All container build assets live in `src/docker/` (`Dockerfile`, `compose.yml`, dependency lock, and helper scripts).
- `src/docker/compose.yml` builds the Streamlit image, mounts `./models` and `./src`, and provisions PostgreSQL with `src/scripts/init-db.sql`.
- Default credentials/ports are injected via environment variables (`APP_PORT`, `DB_PORT`, `POSTGRES_*`, `MODEL_PATH`). Override them inline or inject a `.env`.
- Ensure your YOLO weights live at `models/best.pt` (or point `MODEL_PATH` elsewhere).
- For local installs outside Docker run `pip install -r src/docker/requirements.txt`; Conda users can apply `src/env/environment.yml`.


## Quick Start
1. Clone the repo and enter the directory.
2. Copy your best YOLO weights to `models/trained/best.pt` (alternatively set `MODEL_PATH`).
3. Launch everything: `docker compose -f src/environments/docker/compose.yml up --build --remove-orphans`.
4. Open [http://localhost:8501](http://localhost:8501) to use the Streamlit UI.
5. Tear down when finished: `docker compose -f src/environments/docker/compose.yml down --volumes`.

Spin up the full stack with a single command:

```bash
docker compose -f src/environments/docker/compose.yml up --build --remove-orphans
```

Shut everything down with `docker compose -f src/environments/docker/compose.yml down --volumes` once you're finished testing.

### Streamlit Inference Workflow
1. Place your best YOLO checkpoint at `models/trained/best.pt` (compose mounts it inside `/app/models`).
2. Launch the stack with the compose command above and browse to `http://localhost:8501`.
3. Upload one or more JPG/PNG scans. The app runs YOLOv8 inference, draws maize labels with blue outlines over every detection, and lists coordinates/confidence in a table.
4. Adjust confidence/IoU sliders in the sidebar to tighten or loosen detections.
5. Download the UM-branded annotated image directly from the UI for documentation or model comparisons.

For detailed usage instructions, see [src/web/README.md](src/web/README.md).

### Managing YOLO Runs & Archives
- **Successful training runs** live under `models/runs/<run-name>/` (metrics, configs, preview images, weights). Git tracks the metadata, while `.pt/.pth` binaries remain ignored.
- **Production models** live in `models/trained/`:
  - `best.pt` - Currently deployed model
  - `active_run.txt` - Name of the training run currently in use
- **Historical/incomplete runs** are archived in `models/archive/` to keep `runs/` clean.
- **Base models** (pre-trained YOLOv8) are in `models/pretrained/`.

See [models/README.md](models/README.md) for detailed model management documentation.

### Local Development with Docker

1. **Build and start services:**
   ```bash
   docker compose -f src/environments/docker/compose.yml up --build -d
   ```

2. **Access Streamlit:**
   - Open http://localhost:8501
   - Upload images to run YOLOv8 inference

3. **Access PostgreSQL:**
   ```bash
   docker compose -f src/environments/docker/compose.yml exec db psql -U appuser -d appdb
   ```

### Training on Great Lakes

1. **Setup conda environment:**
   ```bash
   conda env create -f src/environments/conda/great-lakes-env.yml
   conda activate yolov8-env
   ```

2. **Submit training job:**
   ```bash
   sbatch src/training/batch_job.sh
   ```

3. **Monitor progress:**
   ```bash
   src/utils/monitoring/tail_latest_training_log.sh
   ```

4. **Download results:**
   ```bash
   src/utils/deployment/sync_yolov8_run.sh <run-name>
   ```

## Component Details

### Training (`training/`)
Fine-tune YOLOv8 models on financial document datasets.
- See [training/README.md](training/README.md) for detailed instructions

### Processing (`processing/`)
OCR processing pipeline combining YOLO and Tesseract:
- **ocr_processor.py** - Batch OCR processing from parquet files
- Saves results to PostgreSQL database
- Exports to parquet/JSON/CSV formats

### Web Application (`web/`)
Interactive Streamlit interface for model inference:
- Upload document images
- Adjust confidence/IoU thresholds
- View detection results
- Download annotated images
- See [web/README.md](web/README.md) for usage

### Database (`database/`)
PostgreSQL schema and exploration tools:
- **init-db.sql** - Complete database schema (PDF + parquet workflows)
- **postgresql.ipynb** - Database exploration notebook

Schema includes:
- PDF workflow: documents, pages, OCR results, annotations
- Parquet workflow: parquet_ocr_results, parquet_ocr_words, parquet_yolo_regions
- Training: models, training_runs, training_epochs, predictions

### Utilities (`utils/`)

#### Dataset Tools (`utils/dataset/`)
- **count_yolo_labels.py** - Count class occurrences in YOLO datasets
- **remap_yolo_labels.py** - Fix mislabeled class IDs
- **preview_yolo_labels.py** - Visualize bounding boxes for QC

#### Model Management (`utils/models/`)
- **list_model_runs.py** - Summarize training runs with metrics
- **set_active_run.sh** - Switch active Streamlit model

#### Monitoring (`utils/monitoring/`)
- **tail_latest_training_log.sh** - Monitor Great Lakes training logs

#### Deployment (`utils/deployment/`)
- **sync_yolov8_run.sh** - Download models from Great Lakes
- **reset_streamlit_model.sh** - Rollback to baseline model

### Environments (`environments/`)

#### Docker (`environments/docker/`)
Local development environment:
- **Dockerfile** - Streamlit application image
- **compose.yml** - Multi-container orchestration (app + PostgreSQL)
- **requirements.txt** - Python dependencies

#### Conda (`environments/conda/`)
Great Lakes HPC environment:
- **great-lakes-env.yml** - Full conda environment with GPU support

## Workflow Overview

### 1. Data Preparation
```bash
# Validate dataset
src/utils/dataset/count_yolo_labels.py data/input/training/labels

# Visual QC
src/utils/dataset/preview_yolo_labels.py --image path/to/image.jpg --label path/to/label.txt

# Fix mistakes
src/utils/dataset/remap_yolo_labels.py --mapping "0:1,1:2,2:0" data/input/
```

### 2. Model Training
```bash
# On Great Lakes
sbatch src/training/batch_job.sh

# Monitor
src/utils/monitoring/tail_latest_training_log.sh

# Download results
src/utils/deployment/sync_yolov8_run.sh finance-parser-{timestamp}
```

### 3. Model Deployment
```bash
# Switch active model
src/utils/models/set_active_run.sh finance-parser-{timestamp}

# Restart Streamlit
docker compose -f src/environments/docker/compose.yml restart app
```

### 4. OCR Processing
```bash
# Process parquet files
python src/processing/ocr_processor.py

# Query results
docker compose -f src/environments/docker/compose.yml exec db psql -U appuser -d appdb
> SELECT * FROM parquet_ocr_results LIMIT 10;
```

## Environment Variables

Create `src/config/.env` for local overrides:
```bash
# Docker configuration
APP_PORT=8501
DB_PORT=5432

# Database
POSTGRES_USER=appuser
POSTGRES_PASSWORD=apppassword
POSTGRES_DB=appdb

# Model
MODEL_PATH=/app/models/trained/best.pt
```

## Common Tasks

### Reset Everything
```bash
# Stop containers
docker compose -f src/environments/docker/compose.yml down -v

# Rebuild and restart
docker compose -f src/environments/docker/compose.yml up --build -d
```

### View Logs
```bash
# Streamlit logs
docker compose -f src/environments/docker/compose.yml logs -f app

# Database logs
docker compose -f src/environments/docker/compose.yml logs -f db
```

### Database Operations
```bash
# Reinitialize database
docker compose -f src/environments/docker/compose.yml down -v
docker compose -f src/environments/docker/compose.yml up -d db

# Backup database
docker compose -f src/environments/docker/compose.yml exec db pg_dump -U appuser appdb > backup.sql
```
