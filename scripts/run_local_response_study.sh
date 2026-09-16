#!/usr/bin/env bash
# Existing weights and frozen ranges; local response rules, not neural training.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash scripts/run_local_response_study.sh REPLICATION_RESULTS ORIGINAL_CONTINUATION_RESULTS [cuda|cpu]' >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPLICATION="$(realpath "$1")"
PREVIOUS="$(realpath "$2")"
DEVICE="${3:-cuda}"
[[ "$DEVICE" == cuda || "$DEVICE" == cpu ]] || { echo 'Device must be cuda or cpu.' >&2; exit 2; }
test -d "$REPLICATION" && test -d "$PREVIOUS"
cd "$ROOT"
PY="${UNIVERSA_PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" scripts/check_release.py
mkdir -p runs
WORK="$(mktemp -d "$ROOT/runs/local-response-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
"$PY" -u scripts/benchmark_local_response.py \
  --replication-dir "$REPLICATION" --previous-dir "$PREVIOUS" \
  --output-dir "$WORK/results" --device "$DEVICE" --n 256 --test-seed 87000 \
  2>&1 | tee "$WORK/console.log"
"$PY" - "$WORK" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json, sys
root=Path(sys.argv[1]);results=root/'results'
summary=json.loads((results/'summary.json').read_text())
if summary.get('format')!='universa-recurrent.local-response.v1.complete' or not summary.get('all_raw_gates_passed'):
    raise SystemExit('Incomplete report; refusing to package.')
# Failed local approximations are experimental results, not omitted data.
print('Failed fits:',summary['failed_fits'],'; failed targets:',summary['failed_target_cases'])
folder=Path.home()/'Downloads';folder.mkdir(parents=True,exist_ok=True)
archive=folder/(root.name+'-reports.zip')
with ZipFile(archive,'x',compression=ZIP_DEFLATED) as z:
    for path in sorted(results.iterdir()):
        if path.suffix in ('.json','.npz'):z.write(path,path.relative_to(root))
print('COMPLETED. Upload numerical reports, not model weights:',archive)
PY
