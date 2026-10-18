#!/usr/bin/env bash
# Run the frozen part A v2 (research/ti_tnn/protocols/adaptive_compute_part_a_v2.md). Run from this
# worktree's root; results go to the frozen queue's run directory.
set -uo pipefail
RUNS=${RUNS:-/home/seanjazm27/projects/universa-recurrent-gemma}
PY=${PY:-$RUNS/.venv-phase2/bin/python}
OUT=$RUNS/runs/titnn_efficiency/hops_v2
gate() { if [ -n "${GATE:-}" ]; then $GATE; fi; }
MOE="--expert-init dense --shared-expert-frac 0.5"
SKIP="--cap-warmup-frac 0.5 --skip-keeps-kv"
variants=("dense --layers 2" "dense --layers 3" "dense --layers 4" "dense --layers 6" "dense --layers 8"
          "moe_top1 $MOE"
          "mod --target-compute 0.5 $SKIP" "mod --target-compute 0.3 $SKIP"
          "ti_tnn_adaptive --target-compute 0.5 $MOE $SKIP" "ti_tnn_adaptive --target-compute 0.3 $MOE $SKIP")
for SEED in 200 201 202 203 204; do for v in "${variants[@]}"; do gate
  $PY research/ti_tnn/bench/hops.py --ffn $v --steps 18000 --seed "$SEED" --out "$OUT" || echo "FAILED part A v2 $v seed $SEED"
done; done
echo "PART A V2 DONE"
