#!/usr/bin/env bash
# Fixed protocol, frozen weights, and fresh held-out structural-claim evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash scripts/run_claim_reliability_study.sh /path/to/replication/results [cuda|cpu]' >&2
  exit 2
fi
REPLICATION="$1"
DEVICE="${2:-cuda}"
if [[ "$DEVICE" != cuda && "$DEVICE" != cpu ]]; then
  echo 'Device must be cuda or cpu.' >&2
  exit 2
fi
if [[ ! -d "$REPLICATION" ]]; then
  echo "Replication directory missing: $REPLICATION" >&2
  exit 2
fi
PY="${UNIVERSA_PYTHON:-$ROOT/.venv/bin/python}"
REPLICATION="$("$PY" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$REPLICATION")"
export PYTHONPATH="$ROOT/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ROOT"
mkdir -p runs
WORK="$(mktemp -d "$ROOT/runs/claim-reliability-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
"$PY" -u scripts/run_claim_reliability.py \
  --replication-dir "$REPLICATION" --output-dir "$WORK/results" --device "$DEVICE" \
  2>&1 | tee "$WORK/console.log"
"$PY" - "$WORK" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import sys

root = Path(sys.argv[1])
summary = root / 'results/summary.json'
if not summary.is_file():
    raise SystemExit('No completed summary; refusing to package an incomplete run.')
report = json.loads(summary.read_text())
if report['format'] != 'universa-recurrent.claim-reliability.v1.complete':
    raise SystemExit('Invalid completion report.')
destination = Path.home() / 'Downloads'
destination.mkdir(exist_ok=True, parents=True)
archive = destination / f'{root.name}-reports.zip'
files = sorted((root / 'results').iterdir())
if any(not p.is_file() or p.suffix not in ('.json', '.npz') for p in files):
    raise SystemExit('Unexpected result file; refusing to package it.')
with ZipFile(archive, 'x', ZIP_DEFLATED) as output:
    for path in files:
        output.write(path, path.relative_to(root))
print(f'Completed. Upload this reports ZIP: {archive}', flush=True)
print(f'SHA-256: {hashlib.sha256(archive.read_bytes()).hexdigest()}', flush=True)
print('Contains synthetic observations and predictions; no model weights.', flush=True)
PY
