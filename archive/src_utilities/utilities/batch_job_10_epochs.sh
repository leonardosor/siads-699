#!/bin/bash
#SBATCH --job-name=team12_yolov8_gpu_10
#SBATCH --mail-user=joehiggi@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --account=siads699f25_class
#SBATCH --partition=gpu
#SBATCH --output=/home/%u/%x-%j.log

echo "=== YOLOv8 GPU Training (10 Epochs) ==="
module load mamba/py3.12
source /sw/pkgs/arc/mamba/py3.12/etc/profile.d/conda.sh
conda activate yolov8-env

echo "Python and CUDA setup check:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

# run the training
python /home/joehiggi/siads-699/src/yolo_v8/yolo_v8_0.py \
  --device 0 \
  --epochs 10 \
  --batch 2 \
  --imgsz 640 \
  --workers 2 \
  --cache False

echo ">>> GPU training job finished."

