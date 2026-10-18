#!/usr/bin/env bash
# Run the frozen lookup-task initialization check (research/ti_tnn/protocols/adaptive_compute_hops_init_check.md).
# Run from this worktree's root; results go to the frozen queue's run directory.
set -uo pipefail
RUNS=${RUNS:-/home/seanjazm27/projects/universa-recurrent-gemma}
PY=${PY:-$RUNS/.venv-phase2/bin/python}
OUT=$RUNS/runs/titnn_efficiency/hops_init_check
gate() { if [ -n "${GATE:-}" ]; then $GATE; fi; }
for SEED in 0 1 2; do gate
  $PY research/ti_tnn/bench/hops.py --ffn moe_top1 --expert-init dense --seed "$SEED" --out "$OUT" \
    || echo "FAILED hops init check seed $SEED"
done
echo "HOPS INIT CHECK DONE"
