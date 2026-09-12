#!/usr/bin/env bash
# Frozen checkpoints, fresh inputs, and complete receipt checking; no training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash scripts/run_verified_pipeline_study.sh /path/to/replication/results [cuda|cpu]' >&2
  exit 2
fi
REPLICATION="$(realpath "$1")"
DEVICE="${2:-cuda}"
if [[ "$DEVICE" != cuda && "$DEVICE" != cpu ]]; then
  echo 'Device must be cuda or cpu.' >&2
  exit 2
fi
test -d "$REPLICATION"
cd "$ROOT"
PY="${UNIVERSA_PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" scripts/check_release.py
mkdir -p runs
WORK="$(mktemp -d "$ROOT/runs/verified-pipeline-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" -u scripts/benchmark_verified_pipeline.py \
  --replication-dir "$REPLICATION" --output-dir "$WORK/results" --device "$DEVICE" \
  --record-counts 1 16 64 256 --repeats 5 --test-seed 36000 \
  2>&1 | tee "$WORK/console.log"
"$PY" - "$WORK" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import sys

root = Path(sys.argv[1])
summary = root / 'results/summary.json'
if not summary.is_file():
    raise SystemExit('No completed summary; refusing to package an incomplete run.')
if json.loads(summary.read_text())['format'] != 'universa-recurrent.verified-pipeline.v1.complete':
    raise SystemExit('Invalid completion report.')
destination = Path.home() / 'Downloads'
destination.mkdir(exist_ok=True, parents=True)
archive = destination / f'{root.name}-reports.zip'
with ZipFile(archive, 'x', ZIP_DEFLATED) as output:
    for path in sorted((root / 'results').glob('*.json')):
        output.write(path, path.relative_to(root))
print(f'Upload these reports (all saved receipts; no weights): {archive}')
PY
