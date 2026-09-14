#!/usr/bin/env bash
# Keep the network/checker frozen. Compare host conversion size and observe pauses.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash scripts/run_bounded_receipt_study.sh /path/to/replication/results [cuda|cpu]' >&2
  exit 2
fi
REPLICATION="$(realpath "$1")"
DEVICE="${2:-cuda}"
case "$DEVICE" in cuda|cpu) ;; *) echo 'Device must be cuda or cpu.' >&2; exit 2;; esac
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
"$PY" scripts/check_release.py
test -d "$REPLICATION"
mkdir -p runs
WORK="$(mktemp -d "$ROOT/runs/bounded-receipts-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" -u scripts/benchmark_bounded_receipts.py \
  --replication-dir "$REPLICATION" --output-dir "$WORK/results" --device "$DEVICE" \
  --record-counts 64 256 --models shared fixed_depth_4 \
  --repeats 30 --diagnostic-repeats 10 --warmup 2 \
  --test-seed 46000 --order-seed 49000 --cpu-threads 1 \
  2>&1 | tee "$WORK/console.log"
"$PY" - "$WORK" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import sys
root = Path(sys.argv[1])
result = root / 'results'
summary = json.loads((result / 'summary.json').read_text())
if (summary.get('format') != 'universa-recurrent.bounded-receipts.v1.complete'
    or summary.get('all_outputs_and_bytes_agree') is not True
    or summary.get('all_gc_policies_unchanged') is not True):
    raise SystemExit('Incomplete results: keep the logs for review.')
destination = Path.home() / 'Downloads'
destination.mkdir(parents=True, exist_ok=True)
archive = destination / f'{root.name}-reports.zip'
with ZipFile(archive, 'x', ZIP_DEFLATED) as bundle:
    for path in sorted(result.glob('*.json')):
        bundle.write(path, path.relative_to(root))
print(f'COMPLETED. Upload these reports (no weights): {archive}')
PY
