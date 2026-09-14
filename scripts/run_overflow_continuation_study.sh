#!/usr/bin/env bash
# No retraining: preserve prior quantization ranges and use raw states on overflow.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash scripts/run_overflow_continuation_study.sh /path/to/replication/results /path/to/previous-continuation/results [cuda|cpu]' >&2
  exit 2
fi
REPLICATION="$(realpath "$1")"
PREVIOUS="$(realpath "$2")"
DEVICE="${3:-cuda}"
case "$DEVICE" in cuda|cpu) ;; *) echo 'Device must be cuda or cpu.' >&2; exit 2;; esac
cd "$ROOT"
PY="${UNIVERSA_PYTHON:-$ROOT/.venv/bin/python}"
"$PY" scripts/check_release.py
for SEED in 6100 7100 8100 9100 10100; do
  for FILE in "$REPLICATION/weights-$SEED.pt" "$REPLICATION/study-$SEED/calibration.json" "$PREVIOUS/seed-$SEED.json" "$PREVIOUS/seed-$SEED.npz"; do
    test -f "$FILE" || { echo "Missing required file: $FILE" >&2; exit 2; }
  done
done
mkdir -p runs
WORK="$(mktemp -d "$ROOT/runs/overflow-continuation-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" -u scripts/benchmark_overflow_continuation.py \
  --replication-dir "$REPLICATION" --previous-dir "$PREVIOUS" \
  --output-dir "$WORK/results" --device "$DEVICE" \
  2>&1 | tee "$WORK/console.log"
"$PY" - "$WORK" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import sys
root = Path(sys.argv[1]); results = root/'results'
summary = json.loads((results/'summary.json').read_text())
if (summary.get('format') != 'universa-recurrent.overflow-continuation.v1.complete'
    or summary.get('checkpoint_runs') != 5
    or summary.get('all_exact_fallback_gates_passed') is not True):
    raise SystemExit('Run incomplete; keep the logs for review.')
for path in results.glob('seed-*.json'):
    report = json.loads(path.read_text())
    arrays = results/report['per_example_file']
    if hashlib.sha256(arrays.read_bytes()).hexdigest() != report['per_example_sha256']:
        raise SystemExit('Per-example archive hash mismatch.')
destination = Path.home()/'Downloads'; destination.mkdir(exist_ok=True, parents=True)
archive = destination/f'{root.name}-reports.zip'
with ZipFile(archive, 'x', ZIP_DEFLATED) as bundle:
    for path in sorted(results.iterdir()):
        if path.suffix in ('.json', '.npz'):
            bundle.write(path, path.relative_to(root))
print(f'COMPLETED. Upload reports and numerical arrays (no weights): {archive}')
PY
