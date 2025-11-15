#!/usr/bin/env bash
# Restore Streamlit to a known-good YOLO checkpoint and rebuild the containers.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: reset_streamlit_model.sh [--weights <path>] [--no-restart]

Defaults:
  BASELINE_RUN     : finance-image-parser4
  BASELINE_WEIGHTS : models/experiments/archived/${BASELINE_RUN}/weights/best.pt

Examples:
  ./src/scripts/reset_streamlit_model.sh
  BASELINE_RUN=finance-parser-20251112_143826 ./src/scripts/reset_streamlit_model.sh
  ./src/scripts/reset_streamlit_model.sh --weights /path/to/custom.pt --no-restart
EOF
}

WEIGHTS_OVERRIDE=""
RESTART_STREAMLIT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights)
      WEIGHTS_OVERRIDE=${2:-}
      shift
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
BASELINE_RUN=${BASELINE_RUN:-finance-image-parser4}
BASELINE_DEFAULT="${REPO_ROOT}/models/experiments/archived/${BASELINE_RUN}/weights/best.pt"
SOURCE_WEIGHTS=${WEIGHTS_OVERRIDE:-${BASELINE_DEFAULT}}
TARGET_WEIGHTS="${REPO_ROOT}/models/production/best.pt"

if [[ ! -f "${SOURCE_WEIGHTS}" ]]; then
  echo "Baseline weights not found at ${SOURCE_WEIGHTS}."
  echo "Set BASELINE_RUN or use --weights to point at a valid .pt file."
  exit 1
fi

cp "${SOURCE_WEIGHTS}" "${TARGET_WEIGHTS}"
echo "Restored ${TARGET_WEIGHTS} from ${SOURCE_WEIGHTS}"

if [[ ${RESTART_STREAMLIT} -eq 1 ]]; then
  if command -v docker >/dev/null 2>&1; then
    pushd "${REPO_ROOT}" >/dev/null
    docker compose -f src/environments/docker/compose.yml up --build --remove-orphans -d
    popd >/dev/null
    echo "Streamlit containers restarted with the baseline model."
  else
    echo "Docker not found; skipping Streamlit restart."
  fi
else
  echo "Skipping Streamlit restart (--no-restart supplied)."
fi
