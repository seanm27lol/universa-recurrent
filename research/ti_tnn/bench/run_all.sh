#!/usr/bin/env bash
# Run the frozen TI-TNN real-benchmark queue (research/ti_tnn/protocols/real_benchmarks.md).
# Run from the repository root. Each run waits for an idle GPU when GATE is set to a gate command
# (a script that returns once the GPU is free); otherwise runs start immediately.
set -uo pipefail
PY=${PY:-.venv-phase2/bin/python}
gate() { if [ -n "${GATE:-}" ]; then $GATE; fi; }
variants=("dense 0.01" "moe_top1 0.01" "moe_top2 0.01" "ti_tnn 0.01" "ti_tnn 0.0")
splits=(simple length addprim_jump addprim_turn_left)
lm_seed() {
  for v in "${variants[@]}"; do set -- $v; gate
    $PY research/ti_tnn/bench/lm.py --ffn "$1" --bridge-cost "$2" --seed "$SEED" --tokens 1e8 || echo "FAILED lm $1 $2 seed $SEED"
  done
}
scan_seed() {
  for split in "${splits[@]}"; do for v in "${variants[@]}"; do set -- $v; gate
    $PY research/ti_tnn/bench/scan.py --split "$split" --ffn "$1" --bridge-cost "$2" --seed "$SEED" --steps 5000 \
      || echo "FAILED scan $split $1 $2 seed $SEED"
  done; done
}
for SEED in 0 1 2; do lm_seed; scan_seed; done
