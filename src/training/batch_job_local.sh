#!/bin/bash
#SBATCH --job-name=capstone_local
#SBATCH --partition=mypartition
#SBATCH --output=/workspace/logs/%x-%j.log

# Local SLURM batch script for YOLOv8 training
# Executes training inside the devcontainer where all packages are installed

set -euo pipefail

# Configuration variables (can be overridden via environment)
PROJECT_ROOT=${PROJECT_ROOT:-/workspace}
RUN_NAME=${RUN_NAME:-finance-parser-$(date +%Y%m%d_%H%M%S)}
EPOCHS=${EPOCHS:-250}
BATCH=${BATCH:-4}
IMGSZ=${IMGSZ:-1024}
PATIENCE=${PATIENCE:-60}
HYPERPARAMS=${HYPERPARAMS:-"--cache"}

# Optuna hyperparameter optimization
USE_OPTUNA=${USE_OPTUNA:-0}  # Set to 1 to enable Optuna
N_TRIALS=${N_TRIALS:-20}      # Number of Optuna trials

RUNS_DIR="${PROJECT_ROOT}/models/experiments/active"
ARTIFACT_DIR="${PROJECT_ROOT}/models/artifacts"

echo "=========================================="
echo "Starting YOLOv8 Training Job (Local SLURM)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Run Name: ${RUN_NAME}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${BATCH}"
echo "Image Size: ${IMGSZ}"
echo "Use Optuna: ${USE_OPTUNA}"
echo "=========================================="

# Create necessary directories
mkdir -p "${RUNS_DIR}" "${ARTIFACT_DIR}" "${PROJECT_ROOT}/logs"

# Display dataset summary
echo "=========================================="
echo "Dataset Summary"
echo "=========================================="
docker exec 699-devcontainer bash -c "
cd /workspace
echo 'Training set:'
find data/input/ground-truth-augmented -type f -name '*.jpg' -o -name '*.png' 2>/dev/null | wc -l | xargs echo '  Images:'
find data/input/ground-truth-augmented -type f -name '*.txt' 2>/dev/null | wc -l | xargs echo '  Labels:'

echo 'Validation set:'
find data/input/validation/images -type f -name '*.jpg' -o -name '*.png' 2>/dev/null | wc -l | xargs echo '  Images:'
find data/input/validation/labels -type f -name '*.txt' 2>/dev/null | wc -l | xargs echo '  Labels:'

echo 'Test set:'
find data/input/testing/images -type f -name '*.jpg' -o -name '*.png' 2>/dev/null | wc -l | xargs echo '  Images:'
find data/input/testing/labels -type f -name '*.txt' 2>/dev/null | wc -l | xargs echo '  Labels:'
"
echo "=========================================="

# Execute training inside devcontainer where Python packages are installed
echo "Executing training in devcontainer..."

# Build the training command
if [[ "${USE_OPTUNA}" == "1" ]]; then
    echo "=========================================="
    echo "Running with Optuna optimization"
    echo "Trials: ${N_TRIALS}"
    echo "Epochs per trial: ${EPOCHS}"
    echo "=========================================="
    
    TRAIN_CMD="python src/training/train.py \
        --weights models/pretrained/yolov8n.pt \
        --data-config src/training/finance-image-parser.yaml \
        --epochs ${EPOCHS} \
        --device 0 \
        --optimize \
        --n-trials ${N_TRIALS} \
        --clean-broken \
        ${HYPERPARAMS}"
else
    echo "=========================================="
    echo "Running standard training"
    echo "Epochs: ${EPOCHS}"
    echo "Batch Size: ${BATCH}"
    echo "=========================================="
    
    TRAIN_CMD="python src/training/train.py \
        --weights models/pretrained/yolov8n.pt \
        --data-config src/training/finance-image-parser.yaml \
        --epochs ${EPOCHS} \
        --batch ${BATCH} \
        --device 0 \
        --name ${RUN_NAME} \
        --clean-broken \
        ${HYPERPARAMS}"
fi

# Execute the command in the devcontainer
docker exec 699-devcontainer bash -c "cd /workspace && ${TRAIN_CMD}"

# Find and package the results
if [[ "${USE_OPTUNA}" == "1" ]]; then
    # For Optuna, find the most recent non-trial directory
    RUN_PATH=$(find "${RUNS_DIR}" -maxdepth 1 -type d -name "finance-parser-*" ! -name "trial_*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [[ -z "${RUN_PATH}" ]]; then
        echo "Warning: Could not find final Optuna model directory"
        RUN_PATH="${RUNS_DIR}/${RUN_NAME}"
    fi
else
    RUN_PATH="${RUNS_DIR}/${RUN_NAME}"
fi

# Package artifacts if the run directory exists
if [[ -d "${RUN_PATH}" ]]; then
    RUN_BASENAME=$(basename "${RUN_PATH}")
    TAR_PATH="${ARTIFACT_DIR}/${RUN_BASENAME}.tar.gz"
    
    echo "=========================================="
    echo "Packaging training artifacts"
    echo "=========================================="
    
    tar -czf "${TAR_PATH}" -C "$(dirname "${RUN_PATH}")" "${RUN_BASENAME}"
    
    echo "Success! Training completed."
    echo "Run directory: ${RUN_PATH}"
    echo "Artifact package: ${TAR_PATH}"
    echo "=========================================="
else
    echo "=========================================="
    echo "Warning: Training directory not found: ${RUN_PATH}"
    echo "Training may have failed or output to a different location"
    echo "=========================================="
fi

echo "Job completed at: $(date)"
