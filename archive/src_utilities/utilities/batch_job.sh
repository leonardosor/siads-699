#!/bin/bash
#SBATCH --job-name=team12_yolov8_gpu
#SBATCH --account=siads699f25_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1              # one GPU only
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --mail-user=joehiggi@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/%u/%x-%j.log

# ================================================================
# YOLOv8 GPU Batch Script — Safe Credit Mode
# ------------------------------------------------
# Runs YOLOv8 on 1 GPU for a short 3-epoch training test.
# Verifies CUDA works, limits resource burn, and logs everything.
# ================================================================

echo "Starting GPU YOLOv8 training job..."
module load mamba/py3.12
source /sw/pkgs/arc/mamba/py3.12/etc/profile.d/conda.sh
conda activate yolov8-env

# Confirm CUDA device visible
echo "Python executable: $(which python)"
python - <<'PY'
import torch, cv2
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("OpenCV:", cv2.__version__)
PY

# Run lightweight YOLOv8 training (3 epochs, small image size)
python /home/joehiggi/siads-699/src/yolo_v8/yolo_v8_0.py \
  --device 0 \
  --epochs 3 \
  --batch 2 \
  --imgsz 512 \
  --workers 1 \
  --cache False

echo ">>> Training job finished."

