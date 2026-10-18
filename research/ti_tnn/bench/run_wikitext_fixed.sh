#!/usr/bin/env bash
# Run the frozen WikiText-103 fixed-recipe experiment (research/ti_tnn/protocols/adaptive_compute_wikitext_fixed.md).
# Run from the run directory's repository root (where runs/ lives), so recorded paths stay relative;
# CODE points at this branch's checkout.
set -uo pipefail
CODE=${CODE:-$(cd "$(dirname "$0")/../../.." && pwd)}
PY=${PY:-.venv-phase2/bin/python}
gate() { if [ -n "${GATE:-}" ]; then $GATE; fi; }
for SEED in 0 1 2; do gate
  $PY "$CODE/research/ti_tnn/bench/lm.py" --ffn ti_tnn_adaptive --target-compute 0.3 --expert-init dense \
    --shared-expert-frac 0.5 --cap-warmup-frac 0.5 --skip-keeps-kv --seed "$SEED" --tokens 1e8 --save-model \
    --out runs/titnn_efficiency/lm_wikitext_fixed || echo "FAILED wikitext fixed seed $SEED"
done
gate
$PY "$CODE/research/ti_tnn/bench/latency.py" runs/titnn_bench/lm runs/titnn_efficiency/lm_wikitext_fixed \
  --out runs/titnn_efficiency/latency_wikitext_fixed.json || echo "FAILED latency"
echo "WIKITEXT FIXED DONE"
