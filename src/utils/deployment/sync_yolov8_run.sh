#!/usr/bin/env bash
# Sync a YOLOv8 run from Great Lakes into models/experiments/active/, optionally refresh
# models/production/best.pt, and restart Streamlit so the new weights take effect.

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 <run-name> [--no-best] [--no-restart]

Examples:
  $0 finance-parser-20251112_143826
  REMOTE_USER=someone $0 finance-run --no-restart
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

RUN_NAME=$1
shift || true

COPY_BEST=1
RESTART_STREAMLIT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-best)
      COPY_BEST=0
      ;;
    --no-restart)
      RESTART_STREAMLIT=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

REMOTE_USER=${REMOTE_USER:-joehiggi}
REMOTE_HOST=${REMOTE_HOST:-greatlakes.arc-ts.umich.edu}
REMOTE_PROJECT=${REMOTE_PROJECT:-/home/${REMOTE_USER}/siads-699}
REMOTE_PATH="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT}/models/experiments/active/${RUN_NAME}/"

LOCAL_RUNS_DIR=${LOCAL_RUNS_DIR:-${REPO_ROOT}/models/experiments/active}
LOCAL_RUN_PATH="${LOCAL_RUNS_DIR}/${RUN_NAME}"

mkdir -p "${LOCAL_RUNS_DIR}"

echo "Syncing ${REMOTE_PATH} -> ${LOCAL_RUN_PATH}"
rsync -av --progress "${REMOTE_PATH}" "${LOCAL_RUN_PATH}"

if [[ ${COPY_BEST} -eq 1 ]]; then
  BEST_SRC="${LOCAL_RUN_PATH}/weights/best.pt"
  BEST_DST="${REPO_ROOT}/models/production/best.pt"
  ACTIVE_FILE="${REPO_ROOT}/models/production/active_run.txt"
  if [[ -f "${BEST_SRC}" ]]; then
    cp "${BEST_SRC}" "${BEST_DST}"
    echo "Updated ${BEST_DST} from ${BEST_SRC}"
    echo "${RUN_NAME}" > "${ACTIVE_FILE}"
  else
    echo "Warning: ${BEST_SRC} not found; skipped copying best weights."
  fi
fi

if [[ ${RESTART_STREAMLIT} -eq 1 ]]; then
  if command -v docker >/dev/null 2>&1; then
    echo "Rebuilding Streamlit image and restarting containers..."
    pushd "${REPO_ROOT}" >/dev/null
    docker compose -f src/environments/docker/compose.yml up --build --remove-orphans -d
    popd >/dev/null
    echo "Streamlit is rebuilding with the new weights."
  else
    echo "Docker not found on PATH; skipping Streamlit restart."
  fi
else
  echo "Skipping Streamlit restart (--no-restart supplied)."
fi

echo "Done. Local run artifacts are in ${LOCAL_RUN_PATH}"
