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
.
├── src/                      # Source code
│   ├── training/             # Model training scripts and configs
│   ├── processing/           # OCR processing pipeline
│   ├── web/                  # Streamlit web application
│   ├── database/             # Database schemas and init scripts
│   ├── utils/                # Utility scripts
│   │   ├── dataset/          # Dataset validation and labeling tools
│   │   ├── models/           # Model management utilities
│   │   ├── monitoring/       # Training monitoring tools
│   │   └── deployment/       # Deployment automation scripts
│   └── environments/         # Environment configurations
│       ├── docker/           # Docker and Compose files
│       └── conda/            # Conda environment for HPC
├── models/                   # Model weights and training runs
│   ├── pretrained/           # Base YOLOv8 weights
│   ├── trained/              # Production models
│   ├── runs/                 # Training experiments
│   └── archive/              # Historical experiments
├── data/                     # Datasets
│   ├── input/                # Training/validation/test data
│   ├── output/               # OCR results and exports
│   └── raw/                  # Raw dataset downloads
├── docs/                     # Documentation and diagrams
├── tests/                    # Test suite
└── notebooks/                # Jupyter notebooks for exploration
```


## Key Components

### 1. Web Application ([src/web/](src/web/))
Interactive Streamlit interface for document inference. Upload images, adjust detection thresholds, and download annotated results with UM branding.

**Key Features:**
- Multi-image upload support
- Real-time YOLOv8 inference
- Adjustable confidence/IoU thresholds
- University of Michigan branded output
- See [src/web/README.md](src/web/README.md) for detailed usage

### 2. Model Training ([src/training/](src/training/))
Scripts for training YOLOv8 models locally or on Great Lakes HPC. Includes hyperparameter optimization with Optuna.

**Capabilities:**
- Local GPU/CPU training
- SLURM batch jobs for HPC
- Hyperparameter optimization
- Early stopping and checkpointing
- See [src/training/README.md](src/training/README.md) for training guide

### 3. Model Management ([models/](models/))
Organized storage for model weights, training runs, and experiments.

**Structure:**
- `trained/` - Production models (`best.pt`, `active_run.txt`)
- `runs/` - Successful training experiments with metrics
- `pretrained/` - Base YOLOv8 weights
- `archive/` - Historical/incomplete runs
- See [models/README.md](models/README.md) for management guide

### 4. OCR Processing ([src/processing/](src/processing/))
Batch OCR processing pipeline combining YOLO detection with Tesseract text extraction. Processes parquet files and stores results in PostgreSQL.

### 5. Database ([src/database/](src/database/))
PostgreSQL schema supporting both PDF and parquet workflows, with tables for documents, OCR results, annotations, and training metadata.

### 6. Testing ([tests/](tests/))
Comprehensive test suite with unit and integration tests. Includes fixtures for common testing scenarios.
- See [tests/README.md](tests/README.md) for testing guide

### 7. Utilities ([src/utils/](src/utils/))
- **dataset/** - Label counting, remapping, and visualization
- **models/** - List runs, switch active models
- **monitoring/** - Track training progress on Great Lakes
- **deployment/** - Sync models from HPC, reset to baseline

## Common Workflows

### 1. Train a New Model on Great Lakes

```bash
# Submit training job
sbatch src/training/batch_job.sh

# Monitor progress
src/utils/monitoring/tail_latest_training_log.sh

# Download and deploy results
src/utils/deployment/sync_yolov8_run.sh finance-parser-20251112_143826
```

See [src/training/README.md](src/training/README.md) for detailed training documentation.

### 2. Switch Active Model

```bash
# Switch to a different trained model
src/utils/models/set_active_run.sh finance-parser-20251112_143826

# Restart Streamlit to load new model
docker compose -f src/environments/docker/compose.yml restart app
```

### 3. Run OCR Processing

```bash
# Process parquet files through OCR pipeline
python src/processing/ocr_processor.py

# Query results in PostgreSQL
docker compose -f src/environments/docker/compose.yml exec db psql -U appuser -d appdb
```

### 4. Validate Dataset

```bash
# Count label occurrences
python src/utils/dataset/count_yolo_labels.py data/input/training/labels

# Preview labels on images
python src/utils/dataset/preview_yolo_labels.py --image path/to/image.jpg --label path/to/label.txt

# Remap incorrect labels
python src/utils/dataset/remap_yolo_labels.py --mapping "0:1,1:2,2:0" data/input/
```

### 5. Database Operations

```bash
# Access PostgreSQL
docker compose -f src/environments/docker/compose.yml exec db psql -U appuser -d appdb

# Backup database
docker compose -f src/environments/docker/compose.yml exec db pg_dump -U appuser appdb > backup.sql

# Reinitialize database
docker compose -f src/environments/docker/compose.yml down -v
docker compose -f src/environments/docker/compose.yml up -d db
```

## Documentation

Detailed documentation is available in component-specific README files:

| Component | Location | Description |
|-----------|----------|-------------|
| **Quickstart Guide** | [docs/README.md](docs/README.md) | Comprehensive quickstart and usage guide |
| **Web Application** | [src/web/README.md](src/web/README.md) | Streamlit interface usage, configuration, troubleshooting |
| **Model Training** | [src/training/README.md](src/training/README.md) | Training scripts, hyperparameter tuning, Great Lakes HPC |
| **Model Management** | [models/README.md](models/README.md) | Model organization, switching models, evaluation |
| **Testing** | [tests/README.md](tests/README.md) | Test suite structure, running tests, writing new tests |
| **Environments** | [src/environments/README.md](src/environments/README.md) | Requirements files, Docker, Conda configurations |
| **Dataset** | [data/raw/rvl-cdip-invoice/README.md](data/raw/rvl-cdip-invoice/README.md) | RVL-CDIP invoice dataset information |

## Environment Configuration

### Docker Environment Variables

Key environment variables (set in `src/config/.env` or inline):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/models/trained/best.pt` | Path to YOLOv8 weights |
| `APP_PORT` | `8501` | Streamlit port |
| `DB_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_USER` | `appuser` | Database username |
| `POSTGRES_PASSWORD` | `apppassword` | Database password |
| `POSTGRES_DB` | `appdb` | Database name |

See [src/environments/README.md](src/environments/README.md) for detailed environment configuration.

### Local Development (Non-Docker)

```bash
# Install dependencies
pip install -r src/environments/docker/requirements.txt

# Or use Conda (for Great Lakes HPC)
conda env create -f src/environments/conda/great-lakes-env.yml
```

## Troubleshooting

### View Logs
```bash
# Streamlit logs
docker compose -f src/environments/docker/compose.yml logs -f app

# Database logs
docker compose -f src/environments/docker/compose.yml logs -f db
```

### Reset Everything
```bash
docker compose -f src/environments/docker/compose.yml down -v
docker compose -f src/environments/docker/compose.yml up --build -d
```

### Model Not Loading
```bash
# Verify model exists
ls -la models/trained/best.pt

# Check active run
cat models/trained/active_run.txt

# List available models
python src/utils/models/list_model_runs.py
```

## External Resources

- **Project Data**: [Google Drive](https://drive.google.com/drive/folders/1ibqk_GzowWrwybOqg8wA88Q95gKQnrM1?usp=share_link)
- **RVL-CDIP Dataset**: [HuggingFace](https://huggingface.co/datasets/chainyo/rvl-cdip-invoice)
- **YOLOv8 Documentation**: [Ultralytics](https://docs.ultralytics.com/)
- **Tesseract OCR**: [Documentation](https://tesseract-ocr.github.io/tessdoc/)
