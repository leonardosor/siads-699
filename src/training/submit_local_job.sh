#!/bin/bash
# Helper script to submit training jobs to local SLURM cluster
# Usage: ./submit_local_job.sh [OPTIONS]

set -euo pipefail

# Default values
EPOCHS=${EPOCHS:-250}
BATCH=${BATCH:-4}
USE_OPTUNA=${USE_OPTUNA:-0}
N_TRIALS=${N_TRIALS:-20}
SCRIPT_PATH="/workspace/src/training/batch_job_local.sh"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --optuna)
            USE_OPTUNA=1
            shift
            ;;
        --trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --epochs N       Number of training epochs (default: 250)"
            echo "  --batch N        Batch size (default: 4)"
            echo "  --optuna         Enable Optuna hyperparameter optimization"
            echo "  --trials N       Number of Optuna trials (default: 20)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Convert script to Unix line endings if needed
sed -i 's/\r$//' "${SCRIPT_PATH}" 2>/dev/null || true

# Submit the job
echo "Submitting job to SLURM..."
echo "  Epochs: ${EPOCHS}"
echo "  Batch: ${BATCH}"
echo "  Optuna: ${USE_OPTUNA}"
if [[ "${USE_OPTUNA}" == "1" ]]; then
    echo "  Trials: ${N_TRIALS}"
fi

JOB_ID=$(sbatch \
    --export=ALL,EPOCHS="${EPOCHS}",BATCH="${BATCH}",USE_OPTUNA="${USE_OPTUNA}",N_TRIALS="${N_TRIALS}" \
    "${SCRIPT_PATH}" | awk '{print $4}')

echo ""
echo "Job submitted successfully!"
echo "Job ID: ${JOB_ID}"
echo ""
echo "Monitor job status with:"
echo "  squeue"
echo ""
echo "View job output with:"
echo "  tail -f /workspace/logs/capstone_local-${JOB_ID}.log"
echo ""
echo "Cancel job with:"
echo "  scancel ${JOB_ID}"
