#!/bin/bash
#SBATCH --job-name=post-optuna-final
#SBATCH --account=siads699f25_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --mail-user=lcedeno@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/%u/%x-%j.log

# Final Training Script
#
# Runs final training using the best Optuna parameters from optuna_study.db
# Similar to batch_job.sh but uses pre-optimized hyperparameters instead of running Optuna.
#
# Environment overrides (export before submitting):
#   PROJECT_ROOT   - path to repo (default /home/$USER/699/siads-699)
#   RUN_NAME       - name for this training run
#   EPOCHS         - number of epochs (default: 100)
#   DEVICE         - CUDA device string (default: 0)
#   HYPERPARAMS    - extra flags for train_final.py (default: "--cache")
#   PARAMS_JSON    - path to pre-exported params JSON (optional)

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/${USER}/699/siads-699}
RUN_NAME=${RUN_NAME:-finance-parser-final-$(date +%Y%m%d_%H%M%S)}
EPOCHS=${EPOCHS:-100}
DEVICE=${DEVICE:-0}
HYPERPARAMS=${HYPERPARAMS:-"--cache"}
PARAMS_JSON=${PARAMS_JSON:-""}

RUNS_DIR="${PROJECT_ROOT}/models/experiments/active"
ARTIFACT_DIR="${PROJECT_ROOT}/models/artifacts"
ACTIVE_DB="${PROJECT_ROOT}/models/experiments/active/optuna_study.db"
BEST_PARAMS_JSON="${PROJECT_ROOT}/models/experiments/active/best_params.json"

echo "Starting final training job: ${RUN_NAME}"
echo "Project root : ${PROJECT_ROOT}"
echo "Epochs       : ${EPOCHS}"
echo "Device       : ${DEVICE}"
echo

# Load environment (Great Lakes style)
if [[ -n "${CUDA_MODULE:-}" ]]; then
  echo "Loading CUDA module ${CUDA_MODULE}"
  module load "${CUDA_MODULE}" || echo "Warning: failed to load ${CUDA_MODULE}; continuing without explicit CUDA module"
fi
if module list 2>&1 | grep -qi "python"; then
  module purge python >/dev/null 2>&1 || module unload python >/dev/null 2>&1 || true
fi
set +u
module load mamba/py3.12
source /sw/pkgs/arc/mamba/py3.12/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate capstone
set -u

mkdir -p "${RUNS_DIR}" "${ARTIFACT_DIR}"
cd "${PROJECT_ROOT}"

echo "Python executable: $(which python)"
python -c "import torch; print('Torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Export best Optuna params if JSON not provided and not existing
if [[ -z "${PARAMS_JSON}" ]]; then
  if [[ ! -f "${BEST_PARAMS_JSON}" ]]; then
    if [[ -f "${ACTIVE_DB}" ]]; then
      echo "Exporting best Optuna parameters from ${ACTIVE_DB}"
      python src/training/show_best_params.py --study-db "${ACTIVE_DB}" --save-json "${BEST_PARAMS_JSON}" || echo "Warning: failed to export params"
    else
      echo "Warning: Optuna DB not found (${ACTIVE_DB}); proceeding without optimized params"
    fi
  else
    echo "Using existing best params from ${BEST_PARAMS_JSON}"
  fi
  PARAMS_JSON="${BEST_PARAMS_JSON}"
fi

# Run final training with best Optuna parameters
echo "Running final training with best Optuna parameters (${EPOCHS} epochs)"
TRAIN_CMD="python src/training/train_final.py --epochs ${EPOCHS} --device ${DEVICE} --name ${RUN_NAME} --clean-broken ${HYPERPARAMS}"
if [[ -f "${PARAMS_JSON}" ]]; then
  TRAIN_CMD="${TRAIN_CMD} --params-json ${PARAMS_JSON}"
  echo "Using parameters from ${PARAMS_JSON}"
else
  echo "No params JSON found; using defaults"
fi

echo "Command: ${TRAIN_CMD}"
eval "${TRAIN_CMD}"

# Find the run directory
RUN_PATH="${RUNS_DIR}/${RUN_NAME}"
if [[ ! -d "${RUN_PATH}" ]]; then
  echo "Warning: Expected run directory not found at ${RUN_PATH}"
  echo "Searching for most recent run..."
  RUN_PATH=$(find "${RUNS_DIR}" -maxdepth 1 -type d -name "*final*" ! -name "trial_*" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
fi

# Package artifacts into tarball
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
