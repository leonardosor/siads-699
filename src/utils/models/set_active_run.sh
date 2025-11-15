#!/usr/bin/env bash
# Copy a run's best checkpoint into models/best.pt and mark it active.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run-name>"
  echo "Example: $0 body-focus2"
  exit 1
fi

RUN_NAME=$1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="${REPO_ROOT}/models/experiments/active/${RUN_NAME}"
BEST_SRC="${RUN_DIR}/weights/best.pt"
BEST_DST="${REPO_ROOT}/models/production/best.pt"
ACTIVE_FILE="${REPO_ROOT}/models/production/active_run.txt"

if [[ ! -f "${BEST_SRC}" ]]; then
  echo "Error: ${BEST_SRC} not found. Make sure the run has been synced locally."
  exit 1
fi

cp "${BEST_SRC}" "${BEST_DST}"
echo "${RUN_NAME}" > "${ACTIVE_FILE}"
echo "Set active run to ${RUN_NAME} (copied ${BEST_SRC} -> ${BEST_DST})"
