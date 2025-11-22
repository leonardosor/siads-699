# Source Code Directory

This directory contains all source code, utilities, configurations, and deployment files for the SIADS 699 Capstone project.

## Directory Structure

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

## Quick Start

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

Key differences from Docker:
- Full PyTorch (not CPU-only)
- Includes pycocotools
- GPU-enabled packages

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
src/utils/deployment/sync_yolov8_run.sh finance-parser-20251112_143826
```

### 3. Model Deployment
```bash
# Switch active model
src/utils/models/set_active_run.sh finance-parser-20251112_143826

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

## Development Guidelines

1. **Never commit secrets** - Use environment variables
2. **Test locally before Great Lakes** - Use Docker for validation
3. **Document significant changes** - Update relevant READMEs
4. **Use git mv for file moves** - Preserves history
5. **Keep paths relative** - Scripts should work in any environment

## Support

For issues or questions:
- Check component-specific READMEs (training/, web/)
- Review Docker logs for runtime errors
- Verify environment variables are set correctly
