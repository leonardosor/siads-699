#!/bin/bash
#SBATCH --job-name=capstone_team12
#SBATCH --account=siads699f25_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=01:00:00  # Increase to 08:00:00 for Optuna optimization
#SBATCH --mail-user=lcedeno@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/%u/%x-%j.log

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/${USER}/699/siads-699}
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

echo "Starting YOLOv8 training job: ${RUN_NAME}"
if [[ -n "${CUDA_MODULE:-}" ]]; then
  echo "Loading CUDA module ${CUDA_MODULE}"
  module load "${CUDA_MODULE}" || echo "Warning: failed to load ${CUDA_MODULE}; continuing without explicit CUDA module"
fi
# If CUDA pulled in a python module, unload it before loading mamba
if module list 2>&1 | grep -qi "python"; then
  module purge python >/dev/null 2>&1 || module unload python >/dev/null 2>&1 || true
fi
module load mamba/py3.12
source /sw/pkgs/arc/mamba/py3.12/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate capstone

mkdir -p "${RUNS_DIR}" "${ARTIFACT_DIR}"
cd "${PROJECT_ROOT}"

echo "Python executable: $(which python)"
python -c "import torch; print('Torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Build training command based on Optuna usage
if [[ "${USE_OPTUNA}" == "1" ]]; then
  echo "Running with Optuna optimization (${N_TRIALS} trials, ${EPOCHS} epochs per trial)"
  python src/training/train.py \
    --weights models/pretrained/yolov8n.pt \
    --data-config src/training/finance-image-parser.yaml \
    --epochs "${EPOCHS}" \
    --device 0 \
    --optimize \
    --n-trials "${N_TRIALS}" \
    --clean-broken \
    ${HYPERPARAMS}
else
  echo "Running standard training (${EPOCHS} epochs, batch=${BATCH})"
  python src/training/train.py \
    --weights models/pretrained/yolov8n.pt \
    --data-config src/training/finance-image-parser.yaml \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --device 0 \
    --name "${RUN_NAME}" \
    --clean-broken \
    ${HYPERPARAMS}
fi

# Find the final model directory (for Optuna, it's not RUN_NAME)
if [[ "${USE_OPTUNA}" == "1" ]]; then
  # Find most recent non-trial directory
  RUN_PATH=$(find "${RUNS_DIR}" -maxdepth 1 -type d -name "finance-parser-*" ! -name "trial_*" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
  if [[ -z "${RUN_PATH}" ]]; then
    echo "Warning: Could not find final Optuna model directory"
    RUN_PATH="${RUNS_DIR}/${RUN_NAME}"
  fi
else
  RUN_PATH="${RUNS_DIR}/${RUN_NAME}"
fi

if [[ -d "${RUN_PATH}" ]]; then
  RUN_BASENAME=$(basename "${RUN_PATH}")
  TAR_PATH="${ARTIFACT_DIR}/${RUN_BASENAME}.tar.gz"
  tar -czf "${TAR_PATH}" -C "$(dirname "${RUN_PATH}")" "${RUN_BASENAME}"
  echo "Packaged artifacts -> ${TAR_PATH}"

  echo "Training run saved to ${RUN_PATH}"
  echo "Retrieve results via:"
  echo "  scp -r ${USER}@login.greatlakes.arc-ts.umich.edu:${RUN_PATH} ./models/experiments/active/"
  echo "or download the tarball:"
  echo "  scp ${USER}@login.greatlakes.arc-ts.umich.edu:${TAR_PATH} ./"
else
  echo "Warning: Training directory not found: ${RUN_PATH}"
fi
