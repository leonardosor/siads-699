# Quick Start Guide

This guide provides quick reference for the most common tasks. For comprehensive documentation, see the [main README](../README.md).

## Prerequisites
- Docker and Docker Compose installed
- Trained YOLOv8 model at `models/trained/best.pt`
- For training: Access to Great Lakes HPC (optional)

## Launch the Application

```bash
# Start all services
docker compose -f src/environments/docker/compose.yml up --build --remove-orphans

# Access Streamlit UI
open http://localhost:8501

# Stop services
docker compose -f src/environments/docker/compose.yml down --volumes
```

## Streamlit Inference Workflow

1. **Prepare Model**
   - Place your best YOLO checkpoint at `models/trained/best.pt`
   - Docker automatically mounts this at `/app/models`

2. **Launch & Access**
   - Start stack: `docker compose -f src/environments/docker/compose.yml up --build`
   - Browse to [http://localhost:8501](http://localhost:8501)

3. **Upload & Detect**
   - Upload one or more JPG/PNG images
   - App runs YOLOv8 inference automatically
   - View detections with blue outlines and confidence scores

4. **Adjust Settings**
   - Use sidebar sliders to adjust confidence/IoU thresholds
   - Lower confidence = more detections (more false positives)
   - Higher confidence = fewer detections (higher precision)

5. **Download Results**
   - Download UM-branded annotated images
   - Use for documentation or model comparisons

For detailed usage, see [src/web/README.md](../src/web/README.md).

## Managing YOLO Models

### Model Directory Structure
```
models/
├── trained/
│   ├── best.pt           # Currently deployed model
│   └── active_run.txt    # Name of active training run
├── runs/
│   └── <run-name>/       # Training experiments with metrics
├── pretrained/
│   └── yolov8n.pt        # Base model
└── archive/              # Historical/incomplete runs
```

See [models/README.md](../models/README.md) for detailed model management.

### Download and Deploy from Great Lakes

Pull a run from Great Lakes and deploy in one step:

```bash
src/utils/deployment/sync_yolov8_run.sh finance-parser-20251112_143826
```

This automatically:
- Downloads run to `models/runs/<run-name>/`
- Copies `best.pt` to `models/trained/best.pt`
- Updates `models/trained/active_run.txt`
- Rebuilds and restarts Streamlit

**Options:**
- `--no-best` - Skip copying weights
- `--no-restart` - Skip Streamlit restart

**Environment overrides:**
- `REMOTE_USER` (default: `joehiggi`)
- `REMOTE_HOST` (default: `greatlakes.arc-ts.umich.edu`)
- `REMOTE_PROJECT` (default: `/home/$REMOTE_USER/siads-699`)

### Switch Models Locally

Switch to a different already-downloaded model:

```bash
src/utils/models/set_active_run.sh finance-parser-20251112_143826
docker compose -f src/environments/docker/compose.yml restart app
```

### Reset to Baseline

Revert to known-good baseline model:

```bash
src/utils/deployment/reset_streamlit_model.sh
```

Options:
- `--weights <path>` - Use custom checkpoint
- `--no-restart` - Skip Streamlit restart
- Set `BASELINE_RUN` environment variable for custom baseline

## Training on Great Lakes

### Quick Training

```bash
# Default configuration (150 epochs, batch=4, imgsz=1024)
sbatch src/training/batch_job.sh

# Custom configuration
EPOCHS=200 BATCH=8 IMGSZ=640 sbatch src/training/batch_job.sh

# With custom run name
RUN_NAME=my-custom-run sbatch src/training/batch_job.sh
```

### Monitor Training

```bash
# View latest log in real-time
src/utils/monitoring/tail_latest_training_log.sh

# Check job status
squeue -u $USER

# View full log file
cat ~/team12_yolov8_main-<job-id>.log
```

See [src/training/README.md](../src/training/README.md) for comprehensive training documentation.

## Common Tasks

### View Logs

```bash
# Streamlit logs
docker compose -f src/environments/docker/compose.yml logs -f app

# Database logs
docker compose -f src/environments/docker/compose.yml logs -f db
```

### Reset Everything

```bash
# Stop and remove all containers and volumes
docker compose -f src/environments/docker/compose.yml down -v

# Rebuild and restart
docker compose -f src/environments/docker/compose.yml up --build -d
```

### Access Database

```bash
# PostgreSQL shell
docker compose -f src/environments/docker/compose.yml exec db psql -U appuser -d appdb

# Backup database
docker compose -f src/environments/docker/compose.yml exec db pg_dump -U appuser appdb > backup.sql
```

### List Model Runs

```bash
# Show all training runs with metrics
python src/utils/models/list_model_runs.py
```

### Validate Dataset

```bash
# Count labels
python src/utils/dataset/count_yolo_labels.py data/input/training/labels

# Preview labels
python src/utils/dataset/preview_yolo_labels.py --image path/to/image.jpg --label path/to/label.txt

# Remap labels
python src/utils/dataset/remap_yolo_labels.py --mapping "0:1,1:2,2:0" data/input/
```

## Troubleshooting

### Model Not Found
```bash
# Verify model exists
ls -la models/trained/best.pt

# Check active run
cat models/trained/active_run.txt
```

### Port Already in Use
```bash
# Kill existing containers
docker compose -f src/environments/docker/compose.yml down
```

### No Detections
- Lower confidence threshold (try 0.1)
- Verify correct model is loaded
- Check model performance metrics

## Next Steps

- **Detailed Training Guide**: [src/training/README.md](../src/training/README.md)
- **Streamlit Configuration**: [src/web/README.md](../src/web/README.md)
- **Model Management**: [models/README.md](../models/README.md)
- **Testing**: [tests/README.md](../tests/README.md)
- **Full Documentation**: [README.md](../README.md)
