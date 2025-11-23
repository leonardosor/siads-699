#!/bin/bash
#SBATCH --job-name=final_train_batch
#SBATCH --account=siads699f25_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-user=lcedeno@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/%u/%x-%j.log

# Batch Final Training Script
#
# Runs multiple final-training jobs using the best Optuna parameters.
# Each run can vary by seed, epochs, and optional extra flags.
# Consolidates artifacts and produces a summary report at the end.
#
# Environment overrides (export before submitting):
#   PROJECT_ROOT   - path to repo (default /home/$USER/699/siads-699)
#   SEEDS          - space-separated list of seeds (default: "42 1337 2025")
#   EPOCHS_LIST    - space-separated list of epochs per run (default: "300")
#   EXTRA_FLAGS    - extra flags for train_final.py (default: "--cache")
#   DEVICE         - CUDA device string (default: 0)
#   DEPLOY_MODE    - 'first' to deploy first run only, 'best' to deploy best mAP, 'all' to deploy each, 'none'
#   PARAMS_JSON    - path to pre-exported params JSON (optional)

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/${USER}/699/siads-699}
SEEDS=${SEEDS:-"42 1337 2025"}
EPOCHS_LIST=${EPOCHS_LIST:-"10"}
EXTRA_FLAGS=${EXTRA_FLAGS:-"--cache"}
DEVICE=${DEVICE:-0}
DEPLOY_MODE=${DEPLOY_MODE:-best}
PARAMS_JSON=${PARAMS_JSON:-""}

RUN_BASE_DIR="${PROJECT_ROOT}/models/experiments/final"
ACTIVE_DB="${PROJECT_ROOT}/models/experiments/active/optuna_study.db"
BEST_PARAMS_JSON="${PROJECT_ROOT}/models/experiments/active/best_params.json"
PRODUCTION_DIR="${PROJECT_ROOT}/models/production"
SUMMARY_FILE="${PROJECT_ROOT}/models/experiments/final/batch_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "Batch final training starting"
echo "Project root : ${PROJECT_ROOT}"
echo "Seeds        : ${SEEDS}"; echo "Epochs list  : ${EPOCHS_LIST}"; echo "Device       : ${DEVICE}"; echo "Deploy mode  : ${DEPLOY_MODE}"; echo

# Load environment (Great Lakes style)
set +u
module load mamba/py3.12 || echo "Warning: mamba module not found"
source /sw/pkgs/arc/mamba/py3.12/etc/profile.d/conda.sh || true
eval "$(conda shell.bash hook)" || true
conda activate capstone || { echo "Failed to activate env 'capstone'"; exit 1; }
set -u

cd "${PROJECT_ROOT}"
echo "Python: $(which python)"; python -c "import torch; print('Torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Export best Optuna params if JSON not provided and not existing
if [[ -z "${PARAMS_JSON}" ]]; then
  if [[ ! -f "${BEST_PARAMS_JSON}" ]]; then
    if [[ -f "${ACTIVE_DB}" ]]; then
      echo "Exporting best Optuna parameters to ${BEST_PARAMS_JSON}"
      python src/training/show_best_params.py --study-db "${ACTIVE_DB}" --save-json "${BEST_PARAMS_JSON}" || echo "Warning: failed to export params"
    else
      echo "Warning: Optuna DB not found (${ACTIVE_DB}); proceeding with defaults"
    fi
  fi
  PARAMS_JSON="${BEST_PARAMS_JSON}"  # may or may not exist
fi

mkdir -p "${RUN_BASE_DIR}" "${PRODUCTION_DIR}"

declare -A RUN_MAP  # run_name -> best_mAP (float)
BEST_MAP=-1
BEST_RUN=""

touch "${SUMMARY_FILE}"; echo "Batch Final Training Summary" >> "${SUMMARY_FILE}"; echo "Started: $(date -Iseconds)" >> "${SUMMARY_FILE}"; echo >> "${SUMMARY_FILE}"

for seed in ${SEEDS}; do
  for epochs in ${EPOCHS_LIST}; do
    RUN_NAME="final-batch-seed${seed}-e${epochs}-$(date +%H%M%S)"
    echo "=== Running: ${RUN_NAME} (seed=${seed}, epochs=${epochs}) ==="
    TRAIN_CMD=(python src/training/train_final.py --epochs "${epochs}" --device "${DEVICE}" --name "${RUN_NAME}" ${EXTRA_FLAGS})
    if [[ -f "${PARAMS_JSON}" ]]; then
      TRAIN_CMD+=(--params-json "${PARAMS_JSON}")
    fi
    # Do not deploy yet; will decide post-run
    echo "Command: ${TRAIN_CMD[*]}"
    SEED_ENV="${seed}"
    # Use seed by setting PYTHONHASHSEED + torch seed inside script (script may not set; rely on determinism limited)
    export PYTHONHASHSEED="${seed}"
    "${TRAIN_CMD[@]}" || echo "Warning: training failed for ${RUN_NAME}" >> "${SUMMARY_FILE}"

    RUN_DIR="${RUN_BASE_DIR}/${RUN_NAME}"
    if [[ -d "${RUN_DIR}" ]]; then
      MAP_FILE="${RUN_DIR}/training_metadata.json"
      if [[ -f "${MAP_FILE}" ]]; then
        MAP_VAL=$(python - <<'PY'
import json, sys; import pathlib
f=pathlib.Path(sys.argv[1])
data=json.load(open(f))
print(data.get('parameter_source',{}).get('best_map50_95',-1))
PY
"${MAP_FILE}" 2>/dev/null || echo -1)
      else
        MAP_VAL=-1
      fi
      echo "Run ${RUN_NAME} mAP50-95 (from metadata): ${MAP_VAL}" | tee -a "${SUMMARY_FILE}" 
      RUN_MAP["${RUN_NAME}"]="${MAP_VAL}" 
      # Track best
      awk 'BEGIN{exit ARGV[1] > ARGV[2] ? 0 : 1}' "${MAP_VAL}" "${BEST_MAP}" 2>/dev/null || true
      if python - <<'PY'
import sys
try:
  cur=float(sys.argv[1]); best=float(sys.argv[2])
  sys.exit(0 if cur>best else 1)
except: sys.exit(1)
PY
"${MAP_VAL}" "${BEST_MAP}"; then
        BEST_MAP="${MAP_VAL}"; BEST_RUN="${RUN_NAME}"
      fi
    else
      echo "Warning: expected run directory not found: ${RUN_DIR}" | tee -a "${SUMMARY_FILE}"
    fi
    echo >> "${SUMMARY_FILE}"
  done
done

echo "All runs complete." | tee -a "${SUMMARY_FILE}"
echo "Best run: ${BEST_RUN} (mAP50-95=${BEST_MAP})" | tee -a "${SUMMARY_FILE}"

# Deployment logic
deploy_run=""
case "${DEPLOY_MODE}" in
  first) deploy_run=$(printf '%s\n' "${!RUN_MAP[@]}" | head -1) ;;
  best) deploy_run="${BEST_RUN}" ;;
  all)  deploy_run="ALL" ;;
  none) deploy_run="" ;;
  *)    deploy_run="${BEST_RUN}" ;;
esac

deploy_function() {
  local run_name="$1"
  local run_dir="${RUN_BASE_DIR}/${run_name}"
  if [[ -d "${run_dir}" ]]; then
    echo "Deploying ${run_name}" | tee -a "${SUMMARY_FILE}"
    python src/training/train_final.py --name "${run_name}" --deploy --epochs 0 || true
    # Above call won't re-train because epochs=0 but deploy logic triggers for existing weights? If not, copy manually:
    if [[ -f "${run_dir}/weights/best.pt" ]]; then
      cp "${run_dir}/weights/best.pt" "${PRODUCTION_DIR}/best.pt"
      echo "Copied best.pt to production for ${run_name}" | tee -a "${SUMMARY_FILE}"
    fi
  else
    echo "Cannot deploy; run directory missing: ${run_dir}" | tee -a "${SUMMARY_FILE}"
  fi
}

if [[ -n "${deploy_run}" ]]; then
  if [[ "${deploy_run}" == "ALL" ]]; then
    for rn in "${!RUN_MAP[@]}"; do deploy_function "${rn}"; done
  else
    deploy_function "${deploy_run}" 
  fi
else
  echo "Deployment skipped (DEPLOY_MODE=${DEPLOY_MODE})" | tee -a "${SUMMARY_FILE}"
fi

echo "Summary file: ${SUMMARY_FILE}"; echo "Done."; 
