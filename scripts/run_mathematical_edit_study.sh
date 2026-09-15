#!/usr/bin/env bash
# Known-field edits through a frozen numerical descriptor. No training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash scripts/run_mathematical_edit_study.sh REPLICATION_RESULTS ORIGINAL_CONTINUATION_RESULTS [cuda|cpu]' >&2
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
  for FILE in "$REPLICATION/weights-$SEED.pt" "$REPLICATION/study-$SEED/calibration.json" "$PREVIOUS/seed-$SEED.json"; do
    test -f "$FILE" || { echo "Missing required file: $FILE" >&2; exit 2; }
  done
done
mkdir -p runs
WORK="$(mktemp -d "$ROOT/runs/mathematical-edits-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" -u scripts/benchmark_mathematical_edits.py \
  --replication-dir "$REPLICATION" --previous-dir "$PREVIOUS" \
  --output-dir "$WORK/results" --device "$DEVICE" \
  --models shared fixed_depth_4 --n 512 --test-seed 83000 \
  2>&1 | tee "$WORK/console.log"
"$PY" - "$WORK" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import sys
root=Path(sys.argv[1]);results=root/'results'
summary=json.loads((results/'summary.json').read_text())
if (summary.get('format')!='universa-recurrent.mathematical-edits.v1.complete'
    or summary.get('checkpoint_runs')!=5 or summary.get('all_raw_gates_passed') is not True):
    raise SystemExit('Completion checks failed; preserve logs rather than packaging a success.')
for name in summary['worker_files']:
    worker=json.loads((results/name).read_text())
    array=results/worker['per_example_file']
    if hashlib.sha256(array.read_bytes()).hexdigest()!=worker['per_example_sha256']:
        raise SystemExit('Numerical-array checksum mismatch.')
destination=Path.home()/'Downloads';destination.mkdir(parents=True,exist_ok=True)
archive=destination/f'{root.name}-reports.zip'
with ZipFile(archive,'x',ZIP_DEFLATED) as z:
    for path in sorted(results.iterdir()):
        if path.suffix in ('.json','.npz'):z.write(path,path.relative_to(root))
print(f'COMPLETE. Reported nonfinite experimental cases: {summary["failed_cases"]}')
print(f'Upload reports and numerical arrays (no weights): {archive}')
PY
