#!/usr/bin/env bash
# Run a source snapshot with the Spark's existing Python environment and weights.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPLICATION="${1:-$HOME/projects/universa-recurrent-git/runs/replication-20260909-173545-3Lc0us/results}"
DEVICE="${2:-cuda}"
export UNIVERSA_PYTHON="${UNIVERSA_PYTHON:-$HOME/projects/universa-recurrent-git/.venv/bin/python}"
if [[ ! -x "$UNIVERSA_PYTHON" ]]; then
  echo "Python environment missing: $UNIVERSA_PYTHON" >&2
  echo 'Set UNIVERSA_PYTHON to the existing experiment environment.' >&2
  exit 2
fi
if [[ ! -d "$REPLICATION" ]]; then
  echo "Replication results missing: $REPLICATION" >&2
  exit 2
fi
export PYTHONPATH="$ROOT/src"
cd "$ROOT"
"$UNIVERSA_PYTHON" scripts/check_release.py
"$UNIVERSA_PYTHON" -m pytest -q
bash scripts/run_verified_pipeline_study.sh "$REPLICATION" "$DEVICE"
