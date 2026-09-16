#!/usr/bin/env bash
# Last validation of this phase. Existing weights/results are read, never changed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash scripts/run_phase_closeout.sh REPLICATION_RESULTS ORIGINAL_CONTINUATION_RESULTS [cuda|cpu]' >&2
  exit 2
fi
cd "$ROOT"
REPLICATION="$(realpath "$1")"
PREVIOUS="$(realpath "$2")"
DEVICE="${3:-cuda}"
if [[ "$DEVICE" != cuda && "$DEVICE" != cpu ]]; then
  echo 'Device must be cuda or cpu.' >&2
  exit 2
fi
PY="${UNIVERSA_PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" scripts/check_release.py
mkdir -p runs
WORK="$(mktemp -d "$ROOT/runs/phase-closeout-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
"$PY" -u scripts/phase_closeout.py \
  --replication-dir "$REPLICATION" --previous-dir "$PREVIOUS" \
  --output-dir "$WORK/results" --device "$DEVICE" \
  2>&1 | tee "$WORK/console.log"
