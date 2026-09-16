#!/usr/bin/env bash
# Frozen numerical descriptions; paired single and joint edits, no retraining.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash scripts/run_edit_composition_study.sh REPLICATION_RESULTS ORIGINAL_CONTINUATION_RESULTS [cuda|cpu]' >&2
  exit 2
fi
REPLICATION="$(realpath "$1")"
PREVIOUS="$(realpath "$2")"
DEVICE="${3:-cuda}"
[[ "$DEVICE" == cuda || "$DEVICE" == cpu ]] || { echo 'Device must be cuda or cpu.' >&2; exit 2; }
cd "$ROOT"
PY="${UNIVERSA_PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" scripts/check_release.py
for SEED in 6100 7100 8100 9100 10100; do
  test -f "$REPLICATION/weights-$SEED.pt"
  test -f "$REPLICATION/study-$SEED/calibration.json"
  test -f "$PREVIOUS/seed-$SEED.json"
done
mkdir -p runs
WORK="$(mktemp -d "$ROOT/runs/edit-composition-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" -u scripts/benchmark_edit_composition.py \
  --replication-dir "$REPLICATION" --previous-dir "$PREVIOUS" \
  --output-dir "$WORK/results" --device "$DEVICE" --n 256 --test-seed 85000 \
  --models shared fixed_depth_4 --scales 0.25 0.5 1.0 \
  2>&1 | tee "$WORK/console.log"
"$PY" - "$WORK" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib,json,sys
root=Path(sys.argv[1]);results=root/'results'
summary=json.loads((results/'summary.json').read_text())
if (summary.get('format')!='universa-recurrent.edit-composition.v1.complete'
    or summary.get('checkpoint_runs')!=5 or summary.get('all_raw_gates_passed') is not True):
    raise SystemExit('Incomplete experiment. Keep the console log.')
for name in summary['worker_files']:
    worker=json.loads((results/name).read_text())
    numeric=results/worker['per_example_file']
    if hashlib.sha256(numeric.read_bytes()).hexdigest()!=worker['per_example_sha256']:
        raise SystemExit('Numerical report checksum mismatch.')
downloads=Path.home()/'Downloads';downloads.mkdir(parents=True,exist_ok=True)
archive=downloads/f'{root.name}-reports.zip'
with ZipFile(archive,'x',compression=ZIP_DEFLATED) as z:
    for p in sorted(results.iterdir()):
        if p.is_file() and p.suffix in ('.json','.npz'):z.write(p,p.relative_to(root))
print('Completed. Numerical failures, if any, remain in the reports.')
print('Upload this reports-only ZIP (no weights):',archive)
PY
