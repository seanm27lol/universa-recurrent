#!/usr/bin/env bash
# Run the expert-initialization control (research/ti_tnn/protocols/adaptive_compute_init_control.md),
# after the frozen adaptive-compute queue has finished. Run from this worktree's root. RUNS points at the
# run directory of the frozen queue (results, data and the GPU environment live there).
set -uo pipefail
RUNS=${RUNS:-/home/seanjazm27/projects/universa-recurrent-gemma}
PY=${PY:-$RUNS/.venv-phase2/bin/python}
DATA=$RUNS/runs/titnn_data/wikitext103
OUT=$RUNS/runs/titnn_efficiency/lm_init_control
gate() { if [ -n "${GATE:-}" ]; then $GATE; fi; }

for SEED in 0 1 2; do for v in "ti_tnn_adaptive 0.3" "moe_top1 1.0"; do set -- $v; gate
  save=""; [ "$1" = ti_tnn_adaptive ] && save="--save-model"
  $PY research/ti_tnn/bench/lm.py --ffn "$1" --target-compute "$2" --seed "$SEED" --tokens 1e8 --expert-init dense \
    --data "$DATA" --out "$OUT" $save || echo "FAILED init control $1 seed $SEED"
done; done

gate
$PY research/ti_tnn/bench/latency.py "$RUNS/runs/titnn_bench/lm" "$RUNS/runs/titnn_efficiency/lm" "$OUT" --data "$DATA" \
  --out "$RUNS/runs/titnn_efficiency/latency_init_control.json" || echo "FAILED latency"
$PY research/ti_tnn/bench/init_diagnostics.py --data "$DATA" --out "$RUNS/runs/titnn_efficiency/init_diagnostics.json" \
  || echo "FAILED diagnostics"
echo "CONTROL QUEUE DONE"
