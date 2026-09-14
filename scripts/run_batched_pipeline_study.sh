#!/usr/bin/env bash
# Compare grouped processing with prepared scalar checking; never train or push.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash scripts/run_batched_pipeline_study.sh /path/to/replication/results [cuda|cpu]' >&2
  exit 2
fi
REPLICATION="$(realpath "$1")"
DEVICE="${2:-cuda}"
case "$DEVICE" in cuda|cpu) ;; *) echo 'Device must be cuda or cpu.' >&2; exit 2 ;; esac
[[ -d "$REPLICATION" ]] || { echo 'Replication folder missing.' >&2; exit 2; }
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || { echo 'Install this checkout in .venv first.' >&2; exit 2; }
"$PY" scripts/check_release.py
mkdir -p runs
WORK="$(mktemp -d "$ROOT/runs/batched-pipeline-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" -u scripts/benchmark_batched_pipeline.py \
  --replication-dir "$REPLICATION" --output-dir "$WORK/results" --device "$DEVICE" \
  --record-counts 1 16 64 256 --models shared fixed_depth_4 direct \
  --test-seed 46000 --order-seed 48000 --repeats 10 --warmup 2 --cpu-threads 1 \
  2>&1 | tee "$WORK/console.log"
"$PY" - "$WORK" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import sys
root = Path(sys.argv[1])
results = root / 'results'
summary = json.loads((results / 'summary.json').read_text())
if (summary.get('format') != 'universa-recurrent.batched-pipeline.v1.complete'
    or summary.get('all_outputs_and_receipts_agree') is not True
    or summary.get('all_rejection_tests_passed') is not True):
    raise SystemExit('Incomplete/failed run: keep the console log for review.')
for worker in summary['worker_reports']:
    if not (results / worker).is_file():
        raise SystemExit('A worker report is missing.')
destination = Path.home() / 'Downloads'
destination.mkdir(parents=True, exist_ok=True)
archive = destination / f'{root.name}-reports.zip'
with ZipFile(archive, 'x', compression=ZIP_DEFLATED) as bundle:
    for path in sorted(results.glob('*.json')):
        bundle.write(path, path.relative_to(root))
print(f'\nCOMPLETED. Upload reports, not weights:\n{archive}')
PY
