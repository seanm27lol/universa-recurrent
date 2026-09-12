#!/usr/bin/env bash
# Use the DGX's existing environment, without changing its Git checkout or weights.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPLICATION="${1:-$HOME/projects/universa-recurrent-git/runs/replication-20260909-173545-3Lc0us/results}"
DEVICE="${2:-cuda}"
export UNIVERSA_PYTHON="${UNIVERSA_PYTHON:-$HOME/projects/universa-recurrent-git/.venv/bin/python}"
if [[ ! -x "$UNIVERSA_PYTHON" ]]; then
  echo "Python environment missing: $UNIVERSA_PYTHON" >&2
  exit 2
fi
if [[ ! -d "$REPLICATION" ]]; then
  echo "Replication directory missing: $REPLICATION" >&2
  exit 2
fi
REPLICATION="$("$UNIVERSA_PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$REPLICATION")"
export PYTHONPATH="$ROOT/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ROOT"
"$UNIVERSA_PYTHON" - "$DEVICE" <<'PY'
import sys
import torch
if sys.argv[1] not in ('cpu', 'cuda'):
    raise SystemExit('Device must be cpu or cuda.')
if sys.argv[1] == 'cuda' and not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in the selected Python environment.')
print(f'Preflight: Python {sys.version.split()[0]}, PyTorch {torch.__version__}', flush=True)
if sys.argv[1] == 'cuda':
    print(f'Device: {torch.cuda.get_device_name(0)}', flush=True)
PY
"$UNIVERSA_PYTHON" scripts/check_release.py
"$UNIVERSA_PYTHON" -m pytest -q
bash scripts/run_claim_reliability_study.sh "$REPLICATION" "$DEVICE"
