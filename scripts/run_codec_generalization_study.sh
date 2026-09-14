#!/usr/bin/env bash
# Frozen codecs and checkpoints; new input seeds; no training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash scripts/run_codec_generalization_study.sh REPLICATION_RESULTS ORIGINAL_CONTINUATION_RESULTS [cuda|cpu]' >&2
  exit 2
fi
REPLICATION="$(realpath "$1")"
PREVIOUS="$(realpath "$2")"
DEVICE="${3:-cuda}"
if [[ "$DEVICE" != cuda && "$DEVICE" != cpu ]]; then
  echo 'Device must be cuda or cpu.' >&2
  exit 2
fi
cd "$ROOT"
PY="${UNIVERSA_PYTHON:-$ROOT/.venv/bin/python}"
"$PY" scripts/check_release.py
mkdir -p runs
WORK="$(mktemp -d "$ROOT/runs/codec-generalization-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" -u scripts/benchmark_codec_generalization.py \
  --replication-dir "$REPLICATION" --previous-dir "$PREVIOUS" \
  --output-dir "$WORK/results" --device "$DEVICE" --n 1024 \
  --data-seeds 71000 72000 --conditions matched noise_x3 sparse_040 \
  2>&1 | tee "$WORK/console.log"
"$PY" - "$WORK" <<'PY'
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED
root=Path(sys.argv[1]);results=root/'results'
summary=json.loads((results/'summary.json').read_text())
if (summary.get('format')!='universa-recurrent.codec-generalization.v1.complete' or
    not summary.get('all_raw_restoration_gates_passed') or
    not summary.get('all_successful_fallback_subset_checks_passed')):
    raise SystemExit('Incomplete experiment. Preserve logs; do not package as complete.')
for name in summary['worker_files']:
    if Path(name).name!=name or not (results/name).is_file() or not (results/name).with_suffix('.npz').is_file():
        raise SystemExit('Missing worker JSON or numeric arrays.')
destination=Path.home()/'Downloads';destination.mkdir(parents=True,exist_ok=True)
archive=destination/f'{root.name}-reports.zip'
with ZipFile(archive,'x',ZIP_DEFLATED) as z:
    for p in sorted(results.iterdir()):
        if p.is_file() and p.suffix in ('.json','.npz'):
            z.write(p,p.relative_to(root))
print(f'\nCompleted with {summary["failed_cases"]} reported numerical failures.')
print('Changed decisions or numerical failures are study results, not grounds to discard a run.')
print(f'Upload this reports-only ZIP (no weights):\n{archive}')
PY
