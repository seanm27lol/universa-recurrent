#!/usr/bin/env bash
# One benchmark, fresh processes, existing weights. Nothing is trained or pushed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash scripts/run_execution_audit.sh /path/to/replication/results [cuda|cpu]' >&2
  exit 2
fi
INPUT="$(realpath "$1")"
DEVICE="${2:-cuda}"
case "$DEVICE" in cuda|cpu) ;; *) echo 'Choose cuda or cpu' >&2; exit 2;; esac
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
"$PY" scripts/check_release.py
mkdir -p runs
WORK="$(mktemp -d "$ROOT/runs/execution-audit-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
"$PY" -u scripts/rebenchmark_execution.py "$INPUT" --device "$DEVICE" \
  --passes 2 --warmup 5 --repeats 20 --output-dir "$WORK/results" 2>&1 | tee "$WORK/console.log"
echo "All reports: $WORK/results-reports.zip"
