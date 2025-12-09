#!/bin/bash
# Quick submission script for bootstrap confidence intervals
#
# Usage:
#   ./submit_bootstrap_ci.sh path/to/finetuned/model.pt [n_bootstrap]
#
# Examples:
#   ./submit_bootstrap_ci.sh models/experiments/final/best.pt
#   ./submit_bootstrap_ci.sh models/experiments/final/best.pt 1000
#   ./submit_bootstrap_ci.sh models/experiments/final/best.pt 50000

set -euo pipefail

# Check arguments
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <finetuned_model_path> [n_bootstrap]"
    echo ""
    echo "Examples:"
    echo "  $0 models/experiments/final/best.pt"
    echo "  $0 models/experiments/final/best.pt 1000"
    echo "  $0 models/experiments/final/best.pt 50000"
    echo ""
    exit 1
fi

FINETUNED_MODEL="$1"
N_BOOTSTRAP="${2:-10000}"  # Default 10000 if not specified

# Verify model exists
if [[ ! -f "${FINETUNED_MODEL}" ]]; then
    echo "ERROR: Model file not found: ${FINETUNED_MODEL}"
    exit 1
fi

echo "========================================"
echo "Bootstrap CI Job Submission"
echo "========================================"
echo "Fine-tuned Model: ${FINETUNED_MODEL}"
echo "Bootstrap Iterations: ${N_BOOTSTRAP}"
echo "========================================"
echo ""

# Submit the job
echo "Submitting SLURM job..."
FINETUNED_MODEL="${FINETUNED_MODEL}" \
N_BOOTSTRAP="${N_BOOTSTRAP}" \
sbatch src/training/batch_bootstrap_ci.sh

echo ""
echo "Job submitted successfully!"
echo ""
echo "Monitor your job:"
echo "  squeue -u \$USER"
echo ""
echo "View log (after job starts):"
echo "  tail -f ~/bootstrap-ci-<job-id>.log"
echo ""
echo "Expected runtime:"
if [[ ${N_BOOTSTRAP} -le 1000 ]]; then
    echo "  ~2-5 minutes"
elif [[ ${N_BOOTSTRAP} -le 10000 ]]; then
    echo "  ~15-30 minutes"
else
    echo "  ~1-2 hours"
fi
echo ""
