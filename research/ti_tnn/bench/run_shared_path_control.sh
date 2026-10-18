#!/usr/bin/env bash
# Run the frozen shared-path control (research/ti_tnn/protocols/adaptive_compute_shared_path_control.md).
# Run from the run directory's repository root (where runs/ lives), so recorded paths stay relative.
set -uo pipefail
CODE=${CODE:-$(cd "$(dirname "$0")/../../.." && pwd)}
PY=${PY:-.venv-phase2/bin/python}
gate() { if [ -n "${GATE:-}" ]; then $GATE; fi; }
variants=("dense --ffn-mult 2"
          "ti_tnn_adaptive --target-compute 0.3 --expert-init dense --shared-expert-frac 0.5 --cap-warmup-frac 0.5 --skip-keeps-kv --experts 1")
for SEED in 200 201 202 203 204; do for v in "${variants[@]}"; do gate
  $PY "$CODE/research/ti_tnn/bench/hops.py" --ffn $v --steps 18000 --seed "$SEED" \
    --out runs/titnn_efficiency/hops_shared_control || echo "FAILED shared control $v seed $SEED"
done; done
echo "SHARED CONTROL DONE"
