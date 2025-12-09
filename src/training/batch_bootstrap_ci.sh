#!/bin/bash
#SBATCH --job-name=bootstrap-ci
#SBATCH --account=siads699f25_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-user=lcedeno@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/%u/%x-%j.log

# Bootstrap Confidence Intervals Batch Job
#
# Computes bootstrap confidence intervals for mAP50 comparison between
# baseline and fine-tuned models using GPU acceleration.
#
# Environment overrides (export before submitting):
#   PROJECT_ROOT       - path to repo (default /home/$USER/699/siads-699)
#   BASELINE_MODEL     - path to baseline model weights (default: models/pretrained/yolov8n.pt)
#   FINETUNED_MODEL    - path to fine-tuned model weights (required)
#   N_BOOTSTRAP        - number of bootstrap iterations (default: 10000)
#   CONFIDENCE_LEVEL   - confidence level 0-1 (default: 0.95)
#   OUTPUT_DIR         - output directory (default: data/output/bootstrap)

set -euo pipefail

# Configuration
PROJECT_ROOT=${PROJECT_ROOT:-/home/${USER}/699/siads-699}
BASELINE_MODEL=${BASELINE_MODEL:-models/pretrained/yolov8n.pt}
FINETUNED_MODEL=${FINETUNED_MODEL:-""}
DATA_CONFIG=${DATA_CONFIG:-src/training/finance-image-parser.yaml}
N_BOOTSTRAP=${N_BOOTSTRAP:-10000}
CONFIDENCE_LEVEL=${CONFIDENCE_LEVEL:-0.95}
RANDOM_SEED=${RANDOM_SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-data/output/bootstrap}
ARTIFACT_DIR=${ARTIFACT_DIR:-models/artifacts}

echo "========================================"
echo "Bootstrap Confidence Intervals Job"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Project Root: ${PROJECT_ROOT}"
echo "Baseline Model: ${BASELINE_MODEL}"
echo "Fine-tuned Model: ${FINETUNED_MODEL}"
echo "Bootstrap Iterations: ${N_BOOTSTRAP}"
echo "Confidence Level: ${CONFIDENCE_LEVEL}"
echo "========================================"

# Validate required parameters
if [[ -z "${FINETUNED_MODEL}" ]]; then
    echo "ERROR: FINETUNED_MODEL is required!"
    echo "Usage: FINETUNED_MODEL=path/to/model.pt sbatch batch_bootstrap_ci.sh"
    exit 1
fi

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

# Navigate to project root
cd "${PROJECT_ROOT}"

# Create output directories
mkdir -p "${OUTPUT_DIR}" "${ARTIFACT_DIR}"

# Display environment info
echo ""
echo "========================================"
echo "Environment Information"
echo "========================================"
echo "Python executable: $(which python)"
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi
echo "========================================"

# Verify model files exist
echo ""
echo "Verifying model files..."
if [[ ! -f "${BASELINE_MODEL}" ]]; then
    echo "ERROR: Baseline model not found: ${BASELINE_MODEL}"
    exit 1
fi
echo "✓ Baseline model found: ${BASELINE_MODEL}"

if [[ ! -f "${FINETUNED_MODEL}" ]]; then
    echo "ERROR: Fine-tuned model not found: ${FINETUNED_MODEL}"
    exit 1
fi
echo "✓ Fine-tuned model found: ${FINETUNED_MODEL}"

if [[ ! -f "${DATA_CONFIG}" ]]; then
    echo "ERROR: Data config not found: ${DATA_CONFIG}"
    exit 1
fi
echo "✓ Data config found: ${DATA_CONFIG}"

# Run bootstrap confidence intervals
echo ""
echo "========================================"
echo "Running Bootstrap Analysis"
echo "========================================"
echo "This may take 10-30 minutes depending on GPU and number of iterations..."
echo ""

python src/training/bootstrap_confidence_intervals.py \
    --baseline-model "${BASELINE_MODEL}" \
    --finetuned-model "${FINETUNED_MODEL}" \
    --data-config "${DATA_CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-bootstrap ${N_BOOTSTRAP} \
    --confidence-level ${CONFIDENCE_LEVEL} \
    --random-seed ${RANDOM_SEED}

# Check if results were generated
if [[ -f "${OUTPUT_DIR}/bootstrap_confidence_intervals.json" ]]; then
    echo ""
    echo "========================================"
    echo "Bootstrap Analysis Complete!"
    echo "========================================"
    echo "Results saved to: ${OUTPUT_DIR}"

    # Create tarball of results
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    TAR_NAME="bootstrap_ci_${TIMESTAMP}.tar.gz"
    TAR_PATH="${ARTIFACT_DIR}/${TAR_NAME}"

    echo "Creating results tarball..."
    tar -czf "${TAR_PATH}" -C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")"

    echo ""
    echo "Artifacts packaged: ${TAR_PATH}"
    echo ""
    echo "Retrieve results via:"
    echo "  scp ${USER}@login.greatlakes.arc-ts.umich.edu:${OUTPUT_DIR}/bootstrap_confidence_intervals.json ./"
    echo "or download the tarball:"
    echo "  scp ${USER}@login.greatlakes.arc-ts.umich.edu:${TAR_PATH} ./"
    echo ""

    # Display key results
    echo "========================================"
    echo "Key Results Summary"
    echo "========================================"
    python -c "
import json
with open('${OUTPUT_DIR}/bootstrap_confidence_intervals.json', 'r') as f:
    results = json.load(f)

print(f\"Baseline mAP50:    {results['baseline']['point_estimate']:.4f} [{results['baseline']['ci_lower']:.4f}, {results['baseline']['ci_upper']:.4f}]\")
print(f\"Fine-tuned mAP50:  {results['finetuned']['point_estimate']:.4f} [{results['finetuned']['ci_lower']:.4f}, {results['finetuned']['ci_upper']:.4f}]\")
print(f\"Improvement:       {results['improvement']['point_estimate']:.4f} [{results['improvement']['ci_lower']:.4f}, {results['improvement']['ci_upper']:.4f}]\")
print(f\"P-value:           {results['improvement']['p_value']:.6f}\")
print(f\"Iterations:        {results['config']['n_bootstrap']}\")
print(f\"Device:            {results['config']['device']}\")
"
    echo "========================================"
else
    echo ""
    echo "ERROR: Bootstrap analysis failed - no results file generated"
    exit 1
fi

echo ""
echo "Job completed at: $(date)"
