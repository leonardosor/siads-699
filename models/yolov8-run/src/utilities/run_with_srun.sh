#!/bin/bash
# Launch YOLOv8 training via srun so logs stream to the terminal.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/${USER}/siads-699/models/yolov8-run}
JOB_NAME=${JOB_NAME:-yolov8-live}
PARTITION=${PARTITION:-gpu}
GRES=${GRES:-gpu:1}
CPUS=${CPUS:-4}
MEM=${MEM:-16G}
TIME=${TIME:-04:00:00}
MAIL_USER=${MAIL_USER:-joehiggi@umich.edu}
MAIL_TYPE=${MAIL_TYPE:-END,FAIL}

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "Requesting ${PARTITION} with ${GRES}, ${CPUS} CPU(s), ${MEM} RAM, ${TIME} walltime..."
echo "Streaming run logs as soon as the node is allocated."

srun \
  --job-name="${JOB_NAME}" \
  --partition="${PARTITION}" \
  --gres="${GRES}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --mail-type="${MAIL_TYPE}" \
  --mail-user="${MAIL_USER}" \
  --output="${LOG_DIR}/${JOB_NAME}-%j.out" \
  bash -lc "
    cd ${PROJECT_ROOT}
    src/utilities/batch_job.sh
  "
