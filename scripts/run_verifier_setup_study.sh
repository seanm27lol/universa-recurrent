#!/usr/bin/env bash
# Reuse receipts and trusted local checkpoints; no inference or training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 2 ]]; then
  echo 'Usage: bash scripts/run_verifier_setup_study.sh /path/to/retention-reports.zip /path/to/replication/results' >&2
  exit 2
fi
REPORTS="$(realpath "$1")"
REPLICATION="$(realpath "$2")"
test -f "$REPORTS" && test -d "$REPLICATION"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
"$PY" scripts/check_release.py
mkdir -p runs
WORK="$(mktemp -d "$ROOT/runs/verifier-setup-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
"$PY" -u scripts/benchmark_verifier_setup.py \
  --reports "$REPORTS" --replication-dir "$REPLICATION" \
  --output-dir "$WORK/results" --record-counts 1 4 16 64 --repeats 5 \
  2>&1 | tee "$WORK/console.log"
"$PY" - "$WORK" <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import sys
root = Path(sys.argv[1])
if not (root/'results/summary.json').is_file():
    raise SystemExit('No completed summary; refusing to package an incomplete run.')
destination = Path.home()/'Downloads'
destination.mkdir(exist_ok=True, parents=True)
archive = destination/f'{root.name}-reports.zip'
with ZipFile(archive, 'x', ZIP_DEFLATED) as z:
    for path in sorted((root/'results').rglob('*.json')):
        z.write(path, path.relative_to(root))
print(f'Upload these reports (no weights): {archive}')
PY
