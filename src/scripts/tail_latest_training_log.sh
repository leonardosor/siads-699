#!/usr/bin/env bash
# Tail the most recent YOLOv8 training log on Great Lakes.

set -euo pipefail

USER_NAME=${1:-$USER}
LOG_GLOB=${LOG_GLOB:-/home/${USER_NAME}/team12_yolov8_main-*.log}
LINES=${LINES:-50}

LATEST_LOG=$(ls -t ${LOG_GLOB} 2>/dev/null | head -n 1 || true)

if [[ -z "${LATEST_LOG}" ]]; then
  echo "No logs found matching ${LOG_GLOB}"
  exit 1
fi

echo "Tailing ${LATEST_LOG}"
tail -n "${LINES}" "${LATEST_LOG}"
