#!/bin/bash
#SBATCH --job-name=team12_yolov8_main
#SBATCH --account=siads699f25_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --mail-user=joehiggi@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/%u/%x-%j.log

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/${USER}/siads-699/models/training-kit}
RUN_NAME=${RUN_NAME:-finance-parser-$(date +%Y%m%d_%H%M%S)}
EPOCHS=${EPOCHS:-150}
BATCH=${BATCH:-4}
IMGSZ=${IMGSZ:-1024}
PATIENCE=${PATIENCE:-60}
HYPERPARAMS=${HYPERPARAMS:-"--mosaic 0 --cache"}

RUNS_DIR="${PROJECT_ROOT}/runs/detect"
ARTIFACT_DIR="${PROJECT_ROOT}/artifacts"

echo "Starting YOLOv8 training job: ${RUN_NAME}"
module load mamba/py3.12
source /sw/pkgs/arc/mamba/py3.12/etc/profile.d/conda.sh
conda activate yolov8-env

mkdir -p "${RUNS_DIR}" "${ARTIFACT_DIR}"
cd "${PROJECT_ROOT}"

echo "Python executable: $(which python)"
python -c "import torch; print('Torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"

python src/train.py \
  --weights models/yolov8n.pt \
  --data-config src/finance-image-parser.yaml \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --imgsz "${IMGSZ}" \
  --device 0 \
  --workers 4 \
  --patience "${PATIENCE}" \
  --project runs/detect \
  --name "${RUN_NAME}" \
  --cos-lr \
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
