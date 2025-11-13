#!/bin/bash
# Snapshot GPU queues, then launch YOLOv8 training via srun with live logs.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/${USER}/siads-699/models/yolov8-run}
PARTITION_LIST=${PARTITION_LIST:-"gpu spgpu"}
JOB_NAME=${JOB_NAME:-yolov8-live}
GRES=${GRES:-gpu:1}
CPUS=${CPUS:-4}
MEM=${MEM:-16G}
TIME=${TIME:-04:00:00}
QUEUE_LINES=${QUEUE_LINES:-10}
MAIL_USER=${MAIL_USER:-joehiggi@umich.edu}
MAIL_TYPE=${MAIL_TYPE:-END,FAIL}

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

partition_avail() {
  local part=$1
  sinfo -h -p "${part}" -o "%a" 2>/dev/null | head -n1
}

echo "=== Queue snapshot ($(date)) ==="
for part in ${PARTITION_LIST}; do
  echo "-- Partition: ${part}"
  sinfo -p "${part}" || true
  squeue -p "${part}" -o "%9i %8P %8j %5D %7T %8R %10M" | head -n $((QUEUE_LINES + 1)) || true
  echo
done

attempt_partition() {
  local partition=$1
  local avail_lower
  avail_lower=$(partition_avail "${partition}")
  if [[ -z "${avail_lower}" ]]; then
    echo "Partition ${partition} not found in sinfo; skipping."
    return 1
  fi
  avail_lower=$(echo "${avail_lower}" | tr '[:upper:]' '[:lower:]')
  case "${avail_lower}" in
    down*|drain*|drng*|maint*|fail*)
      echo "Partition ${partition} availability '${avail_lower}'; skipping."
      return 1
      ;;
    *)
      ;;
  esac

  echo ">>> Attempting srun on partition '${partition}'"
  srun \
    --job-name="${JOB_NAME}-${partition}" \
    --partition="${partition}" \
    --gres="${GRES}" \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    --mail-type="${MAIL_TYPE}" \
    --mail-user="${MAIL_USER}" \
    --output="${LOG_DIR}/${JOB_NAME}-${partition}-%j.out" \
    bash -lc "
      cd ${PROJECT_ROOT}
      src/utilities/batch_job.sh
    "
}

for part in ${PARTITION_LIST}; do
  if attempt_partition "${part}"; then
    exit 0
  else
    echo "Partition ${part} did not start; trying next..."
  fi
done

echo "No partitions in PARTITION_LIST (${PARTITION_LIST}) could run the job."
exit 1
