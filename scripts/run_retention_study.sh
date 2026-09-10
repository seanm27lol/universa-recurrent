#!/usr/bin/env bash
# Reuse frozen replication weights; do not install dependencies or retrain.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash scripts/run_retention_study.sh /path/to/replication/results [cpu|cuda]' >&2
  exit 2
fi
REPLICATION="$(realpath "$1")"
DEVICE="${2:-cuda}"
case "$DEVICE" in cpu|cuda) ;; *) echo 'Device must be cpu or cuda' >&2; exit 2;; esac
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
test -x "$PY" || { echo 'Install this checkout in .venv first.' >&2; exit 2; }
"$PY" scripts/check_release.py
mkdir -p runs
WORK="$(mktemp -d "$ROOT/runs/retention-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
"$PY" -u -m universa_recurrent.neural.retention_study \
  --replication-dir "$REPLICATION" --output-dir "$WORK/results" \
  --device "$DEVICE" --batch-sizes 1024 4096 --test-seed 32000 \
  --n 4000 --warmup 5 --repeats 20 --audit-samples 4 --audit-repeats 3 \
  2>&1 | tee "$WORK/console.log"
"$PY" - "$WORK" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import sys
root = Path(sys.argv[1])
results = root / 'results'
if not (results / 'summary.json').is_file():
    raise SystemExit('Study did not finish; do not label partial reports complete.')
destination = Path.home() / 'Downloads'
destination.mkdir(parents=True, exist_ok=True)
archive = destination / f'{root.name}-reports.zip'
with ZipFile(archive, 'x', compression=ZIP_DEFLATED) as bundle:
    for path in sorted(results.rglob('*.json')):
        bundle.write(path, path.relative_to(root))
print(f'\nCOMPLETED. Upload this report archive:\n{archive}')
PY
