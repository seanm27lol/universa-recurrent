#!/usr/bin/env bash
# Frozen weights: pause, encode, reconstruct, resume. No training or threshold refit.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash scripts/run_state_continuation_study.sh /path/to/replication/results [cuda|cpu]' >&2
  exit 2
fi
REPLICATION="$(realpath "$1")"
DEVICE="${2:-cuda}"
case "$DEVICE" in cuda|cpu) ;; *) echo 'Device must be cuda or cpu.' >&2; exit 2;; esac
cd "$ROOT"
PY="${UNIVERSA_PYTHON:-$ROOT/.venv/bin/python}"
"$PY" scripts/check_release.py
for SEED in 6100 7100 8100 9100 10100; do
  test -f "$REPLICATION/weights-$SEED.pt" || { echo "Missing checkpoint for $SEED" >&2; exit 2; }
  test -f "$REPLICATION/study-$SEED/calibration.json" || { echo "Missing claim calibration for $SEED" >&2; exit 2; }
done
mkdir -p runs
WORK="$(mktemp -d "$ROOT/runs/state-continuation-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" -u scripts/benchmark_state_continuation.py \
  --replication-dir "$REPLICATION" --output-dir "$WORK/results" --device "$DEVICE" \
  --models shared fixed_depth_4 --bits 4 8 12 16 \
  --calibration-size 1024 --n 1024 \
  --calibration-seed 61000 --test-seed 62000 --rotation-seed 63000 \
  2>&1 | tee "$WORK/console.log"
"$PY" - "$WORK" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import sys
root=Path(sys.argv[1]); results=root/'results'
summary=json.loads((results/'summary.json').read_text())
if (summary.get('format')!='universa-recurrent.state-continuation.v1.complete'
        or summary.get('checkpoint_runs')!=5 or summary.get('all_restoration_gates_passed') is not True):
    raise SystemExit('Incomplete study; reports are preserved but no completion ZIP is created.')
for name in summary['worker_files']:
    worker=json.loads((results/name).read_text())
    array=results/worker['per_example_file']
    if hashlib.sha256(array.read_bytes()).hexdigest()!=worker['per_example_sha256']:
        raise SystemExit('Per-example data checksum mismatch.')
destination=Path.home()/'Downloads';destination.mkdir(parents=True,exist_ok=True)
archive=destination/f'{root.name}-reports.zip'
with ZipFile(archive,'x',ZIP_DEFLATED) as z:
    for path in sorted(results.iterdir()):
        if path.suffix in ('.json','.npz'):
            z.write(path,path.relative_to(root))
print(f'COMPLETE. Upload these reports and per-example arrays (no weights): {archive}')
PY
