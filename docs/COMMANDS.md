# Common Commands

## Open Streamlit Application

```bash
# macOS
open -a safari -u http://localhost:8501

# Linux/WSL
xdg-open http://localhost:8501

# Windows
start http://localhost:8501
```

## Label Management

### Fix class-ID mix-up after label QA

```bash
python src/utils/dataset/remap_yolo_labels.py \
  --root data/raw/rvl-cdip-invoice \
  --map 0:1 1:2 2:0

python src/utils/dataset/count_yolo_labels.py \
  --data-config src/training/finance-image-parser.yaml
```

*Note: The example mapping swaps header→body, body→footer, footer→header.*

### Preview YOLO labels

```bash
python src/utils/dataset/preview_yolo_labels.py \
  --data-config src/training/finance-image-parser.yaml
```

### Count YOLO labels

```bash
python src/utils/dataset/count_yolo_labels.py \
  --data-config src/training/finance-image-parser.yaml
```

## Model Training

### Train YOLOv8 model

```bash
python src/training/train.py \
  --data src/training/finance-image-parser.yaml \
  --epochs 100 \
  --batch 16
```

### Monitor training

```bash
bash src/utils/monitoring/tail_latest_training_log.sh
```

## Model Management

### List model runs

```bash
python src/utils/models/list_model_runs.py
```

### Set active model run

```bash
bash src/utils/models/set_active_run.sh <run_name>
```

## Deployment

### Sync model to Streamlit

```bash
bash src/utils/deployment/sync_yolov8_run.sh
```

### Reset Streamlit model

```bash
bash src/utils/deployment/reset_streamlit_model.sh
```

## Docker

### Start services

```bash
cd src/environments/docker
docker-compose up -d
```

### Stop services

```bash
cd src/environments/docker
docker-compose down
```

### View logs

```bash
docker-compose logs -f app
```

### Rebuild after changes

```bash
cd src/environments/docker
docker-compose up --build
```
