#!/bin/bash
#SBATCH --job-name=team12_yolov8_smoke
#SBATCH --mail-user=joehiggi@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=siads699f25_class
#SBATCH --partition=gpu
#SBATCH --output=/home/%u/%x-%j.log

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/${USER}/siads-699/models/yolov8-run}
RUN_NAME=${RUN_NAME:-smoke-test-$(date +%Y%m%d_%H%M%S)}

echo "=== YOLOv8 GPU Smoke Test (${RUN_NAME}) ==="
module load mamba/py3.12
source /sw/pkgs/arc/mamba/py3.12/etc/profile.d/conda.sh
conda activate yolov8-env

mkdir -p "${PROJECT_ROOT}/runs/detect"
cd "${PROJECT_ROOT}"

python src/yolo_v8/train.py \
  --weights models/yolov8n.pt \
  --data-config src/yolo_v8/finance-image-parser.yaml \
  --epochs 10 \
  --batch 4 \
  --imgsz 640 \
  --device 0 \
  --workers 2 \
  --project runs/detect \
  --name "${RUN_NAME}" \
  --clean-broken

echo "Smoke test finished. Artifacts under ${PROJECT_ROOT}/runs/detect/${RUN_NAME}"
