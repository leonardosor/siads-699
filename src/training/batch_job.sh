#!/bin/bash
#SBATCH --job-name=team12_yolov8_main
#SBATCH --account=siads699f25_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --mail-user=lcedeno@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/%u/%x-%j.log

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/${USER}/siads-699}
RUN_NAME=${RUN_NAME:-finance-parser-$(date +%Y%m%d_%H%M%S)}
EPOCHS=${EPOCHS:-150}
BATCH=${BATCH:-4}
IMGSZ=${IMGSZ:-1024}
PATIENCE=${PATIENCE:-60}
HYPERPARAMS=${HYPERPARAMS:-"--mosaic 0 --cache"}

RUNS_DIR="${PROJECT_ROOT}/models/experiments/active"
ARTIFACT_DIR="${PROJECT_ROOT}/models/artifacts"

echo "Starting YOLOv8 training job: ${RUN_NAME}"
if [[ -n ":${CUDA_MODULE:-}:" && "${CUDA_MODULE:-}" != ":" ]]; then
  echo "Loading CUDA module ${CUDA_MODULE}"
  module load "${CUDA_MODULE}" || echo "Warning: failed to load ${CUDA_MODULE}; continuing without explicit CUDA module"
fi
# If CUDA pulled in a python module, unload it before loading mamba
if module list 2>&1 | grep -qi "python"; then
  module purge python >/dev/null 2>&1 || module unload python >/dev/null 2>&1 || true
fi
module load mamba/py3.12
source /sw/pkgs/arc/mamba/py3.12/etc/profile.d/conda.sh
conda activate yolov8-env

mkdir -p "${RUNS_DIR}" "${ARTIFACT_DIR}"
cd "${PROJECT_ROOT}"

echo "Python executable: $(which python)"
python -c "import torch; print('Torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"

python src/training/train.py \
  --weights models/pretrained/yolov8n.pt \
  --data-config src/training/finance-image-parser.yaml \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --device 0 \
  --name "${RUN_NAME}" \
  --clean-broken \
  ${HYPERPARAMS}

RUN_PATH="${RUNS_DIR}/${RUN_NAME}"
if [[ -d "${RUN_PATH}" ]]; then
  TAR_PATH="${ARTIFACT_DIR}/${RUN_NAME}.tar.gz"
  tar -czf "${TAR_PATH}" -C "$(dirname "${RUN_PATH}")" "$(basename "${RUN_PATH}")"
  echo "Packaged artifacts -> ${TAR_PATH}"
fi

echo "Training run saved to ${RUN_PATH}"
echo "Retrieve results via:"
echo "  scp -r ${USER}@login.greatlakes.arc-ts.umich.edu:${RUN_PATH} ./runs/"
echo "or download the tarball:"
echo "  scp ${USER}@login.greatlakes.arc-ts.umich.edu:${TAR_PATH} ./"
