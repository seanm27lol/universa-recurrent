#!/usr/bin/env bash
# Reuse a trusted v2 checkpoint; do not reinstall PyTorch or alter saved weights.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash scripts/run_dual_study.sh /absolute/path/to/neural_v2.pt [cpu|cuda]' >&2
  exit 2
fi
CHECKPOINT="$(realpath "$1")"
test -f "$CHECKPOINT" || { echo 'Checkpoint not found' >&2; exit 2; }
DEVICE="${2:-cuda}"
case "$DEVICE" in cpu|cuda) ;; *) echo 'Device must be cpu or cuda' >&2; exit 2;; esac
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
test -x "$PY" || { echo 'Activate/install this checkout in .venv first.' >&2; exit 2; }
"$PY" scripts/check_release.py
mkdir -p runs
WORK="$(mktemp -d "$ROOT/runs/dual-output-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
"$PY" -m universa_recurrent.neural.dual_cli compare \
  --device "$DEVICE" --checkpoint "$CHECKPOINT" \
  --output-dir "$WORK/results" 2>&1 | tee "$WORK/console.log"
echo "Completed. Reports: $WORK/results"
