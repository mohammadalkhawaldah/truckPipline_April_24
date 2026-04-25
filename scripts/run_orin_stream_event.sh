#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /absolute/path/to/video.mp4 [extra main.py args...]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv_orin"

VIDEO_PATH="$1"
shift

if [ ! -f "${VIDEO_PATH}" ]; then
  echo "Video not found: ${VIDEO_PATH}" >&2
  exit 1
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Missing ${VENV_DIR}/bin/python. Run scripts/setup_orin_nano.sh first." >&2
  exit 1
fi

cd "${REPO_ROOT}"

"${VENV_DIR}/bin/python" main.py \
  --video-path "${VIDEO_PATH}" \
  --mode stream_event \
  --every_n 1 \
  --show 0 \
  --size-show 0 \
  --preview-scale 0.25 \
  --size-preview-every 999999 \
  --summary-only 1 \
  --non-interactive-model-select \
  --device auto \
  "$@"
