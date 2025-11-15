# Technology Stack

## Core Technologies

- **Python 3.12+**: Primary language
- **YOLOv8 (Ultralytics)**: Object detection for document region identification
- **Tesseract 5**: OCR text extraction
- **PyTorch**: Deep learning framework (via Ultralytics)
- **PostgreSQL 16**: Relational database for OCR results and metadata
- **Streamlit**: Web application framework
- **Docker Compose**: Container orchestration

## Key Libraries

### Computer Vision & ML
- `ultralytics==8.3.226` - YOLOv8 implementation
- `opencv-python-headless==4.10.0.84` - Image processing
- `torch` - Deep learning (via ultralytics)
- `optuna>=3.5.0` - Hyperparameter optimization

### OCR & Document Processing
- `pytesseract>=0.3.10` - Tesseract Python wrapper
- `pdf2image>=1.16.3` - PDF to image conversion
- `PyPDF2>=3.0.1` - PDF manipulation
- `pdfplumber>=0.10.3` - PDF text extraction

### Data & Database
- `pandas>=2.0.3` - Data manipulation
- `numpy>=1.24.3,<2.0` - Numerical operations
- `psycopg2-binary>=2.9.9` - PostgreSQL adapter
- `sqlalchemy>=2.0.23` - Database ORM

### Web & UI
- `streamlit>=1.30.0` - Web interface
- `pillow>=10.1.0` - Image handling

## Build & Development Commands

### Docker (Local Development)
```bash
# Start full stack (Streamlit + PostgreSQL)
docker compose -f src/environments/docker/compose.yml up --build -d

# View logs
docker compose -f src/environments/docker/compose.yml logs -f app

# Stop services
docker compose -f src/environments/docker/compose.yml down --volumes

# Restart after code changes
docker compose -f src/environments/docker/compose.yml restart app
```

### Dataset Preparation
```bash
# Generate augmented training dataset (5k samples)
python src/utils/dataset/prepare_augmented_groundtruth.py

# Generate large dataset (10k samples)
python src/utils/dataset/prepare_augmented_groundtruth.py --augmentations-per-image 100

# Count class distribution
python src/utils/dataset/count_yolo_labels.py --data-config src/training/finance-image-parser.yaml

# Preview labels visually
python src/utils/dataset/preview_yolo_labels.py --data-config src/training/finance-image-parser.yaml

# Fix mislabeled classes
python src/utils/dataset/remap_yolo_labels.py --root data/input --map 0:1 1:2 2:0
```

### Model Training
```bash
# Quick test training (10 epochs)
python src/training/train.py --epochs 10 --batch 16

# Full training (100 epochs)
python src/training/train.py --epochs 100 --batch 16 --cache

# With hyperparameter optimization
python src/training/train.py --optimize --n-trials 20 --epochs 50

# On Great Lakes HPC
sbatch src/training/batch_job.sh
```

### Model Management
```bash
# List all training runs with metrics
python src/utils/models/list_model_runs.py

# Download model from Great Lakes and deploy
src/utils/deployment/sync_yolov8_run.sh finance-parser-20251112_143826

# Switch active model locally
src/utils/models/set_active_run.sh <run-name>

# Reset to baseline model
src/utils/deployment/reset_streamlit_model.sh

# Monitor training on Great Lakes
src/utils/monitoring/tail_latest_training_log.sh
```

### Database Operations
```bash
# Access PostgreSQL shell
docker compose -f src/environments/docker/compose.yml exec db psql -U appuser -d appdb

# Backup database
docker compose -f src/environments/docker/compose.yml exec db pg_dump -U appuser appdb > backup.sql
```

## Environment Configuration

Environment variables are set via `.env` file in project root or via Docker Compose defaults:

- `APP_PORT` - Streamlit port (default: 8501)
- `DB_PORT` - PostgreSQL port (default: 5432)
- `MODEL_PATH` - Path to YOLO weights (default: /app/models/trained/best.pt)
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` - Database credentials

## Platform-Specific Notes

### Great Lakes HPC
- Uses Conda environment: `src/environments/conda/great-lakes-env.yml`
- GPU-enabled PyTorch for training
- SLURM batch job submission via `src/training/batch_job.sh`

### Local Development
- Docker Compose for consistent environment
- CPU-only PyTorch (sufficient for inference)
- Volume mounts for hot-reloading code changes

## Testing

No formal test suite currently implemented. Manual testing via:
- Streamlit UI for inference validation
- Visual inspection of training results
- Database queries for data integrity
