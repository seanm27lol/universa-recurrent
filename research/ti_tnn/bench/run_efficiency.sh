#!/usr/bin/env bash
# Run the frozen TI-TNN adaptive-compute queue (research/ti_tnn/protocols/adaptive_compute.md).
# Run from the repository root. Each run waits for an idle GPU when GATE is set to a gate command
# (a script that returns once the GPU is free); otherwise runs start immediately.
# dense, moe_top1 and moe_top2 at the reference size are reused from the real-benchmark runs
# (runs/titnn_bench/lm), whose code path is unchanged.
set -uo pipefail
PY=${PY:-.venv-phase2/bin/python}
HOPS_ARGS=${HOPS_ARGS:-}
gate() { if [ -n "${GATE:-}" ]; then $GATE; fi; }

# Part A: variable-hop lookups (minutes per run)
hops_variants=("dense 1.0 2" "dense 1.0 3" "dense 1.0 4" "dense 1.0 6" "dense 1.0 8" "moe_top1 1.0 8"
               "mod 0.5 8" "mod 0.3 8" "mod 0.15 8" "ti_tnn_adaptive 0.5 8" "ti_tnn_adaptive 0.3 8" "ti_tnn_adaptive 0.15 8")
for SEED in 0 1 2; do for v in "${hops_variants[@]}"; do set -- $v; gate
  $PY research/ti_tnn/bench/hops.py --ffn "$1" --target-compute "$2" --layers "$3" --seed "$SEED" $HOPS_ARGS \
    || echo "FAILED hops $1 $2 L$3 seed $SEED"
done; done

# Part B: WikiText-103 at the 100M-token budget of the real benchmark
lm_variants=("dense 1.0 512 4" "dense 1.0 384 8" "dense 1.0 256 8" "mod 0.5 512 8" "mod 0.3 512 8"
             "ti_tnn_adaptive 0.5 512 8" "ti_tnn_adaptive 0.3 512 8")
for SEED in 0 1 2; do for v in "${lm_variants[@]}"; do set -- $v; gate
  save=""; case "$1" in mod|ti_tnn_adaptive) save="--save-model";; esac
  $PY research/ti_tnn/bench/lm.py --ffn "$1" --target-compute "$2" --d-model "$3" --layers "$4" --seed "$SEED" \
    --tokens 1e8 --out runs/titnn_efficiency/lm $save || echo "FAILED lm $1 $2 d$3 L$4 seed $SEED"
done; done

# Batch-1 decoding latency of every frontier variant, measured together at the end
gate
$PY research/ti_tnn/bench/latency.py runs/titnn_bench/lm runs/titnn_efficiency/lm || echo "FAILED latency"
echo "QUEUE DONE"
