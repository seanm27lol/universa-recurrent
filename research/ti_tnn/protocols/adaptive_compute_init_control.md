# TI-TNN adaptive compute: the expert-initialization control (frozen brief)

**Status: FROZEN 2026-10-16 (UTC), before any part B result of the adaptive-compute queue existed.**
Branch `ti-tnn-init-control`, based on `ti-tnn-adaptive-compute` at `d9a8c1f`. No run of this control
has started.

## The question

In every mixture model so far, experts have been initialized differently from the dense MLP:

| Weights | Experts ("legacy") | Dense MLP |
|---|---|---|
| Up-projection | std D^-1/2 = 0.044 | std 0.02 |
| Down-projection | std H^-1/2 = 0.022 | std 0.02 / √(2L) = 0.005 (GPT-2 residual scaling) |

So two comparisons in the [adaptive-compute benchmark](adaptive_compute.md) also compare
initializations:
- `ti_tnn_adaptive` (experts) against `mod` (dense MLP), its prediction 7;
- the mixtures against dense.

Do those comparisons survive when experts start like the dense MLP?

## Design

**Intervention.** Both expert banks, small and full, use the dense MLP's rule: up std 0.02, down std
0.005 at 8 layers. Only the standard deviation changes at the same sampling sites. So the same random
numbers are drawn, every other parameter is bitwise identical, and each expert weight is the legacy
draw rescaled (tested).

**Fixed.** Everything else is the frozen benchmark's part B: data, data order, the 100M-token budget,
optimizer, schedule, compute cap, losses and evaluation.

**Runs** (6), seeds 0, 1, 2:

| New run family | Compared with (legacy initialization, from the frozen queue) |
|---|---|
| `ti_tnn_adaptive_c30_initdense` (saves weights for latency) | `ti_tnn_adaptive_c30`, `mod_c30`, `dense` |
| `moe_top1_initdense` | `moe_top1`, `dense` |

- **Why c30.** The frozen protocol already singles out the 0.3 cap, for latency. A conclusion about
  both caps would need c50 too (9 runs). This control covers c30 only.
- **Estimated time.** About 7 GPU-hours: 3 × 58 min (top-1) + 3 × 77 min (adaptive), plus latency and
  diagnostics.
- **When it runs.** It starts after the frozen queue finishes, on the idle-GPU gate. It runs from this
  branch, whose training code differs from the frozen one only by the initialization option, a
  source-version record, and a step-0 diagnostic. The diagnostic draws no random numbers and changes no
  parameters (tested).
- **Recorded per run.** The initialization policy; the source commit; and, at step 0 on the first
  training batch, the residual stream's RMS after each block and each layer's routing probabilities.
  `bench/init_diagnostics.py` reconstructs the same quantities for the legacy runs, from the same seed
  and first batch; a test checks it matches `lm.py`'s record exactly.
- **Activation scale.** Matching weight scales does not match activation scales: a top-1 mixture
  multiplies its expert's output by the router probability. The diagnostics are reported, and no
  direction of the confound is assumed.

## Predictions (recorded before any run; means over seeds 0–2; validation perplexity)

1. **Adaptive against skipping survives.** The sign of perplexity(`ti_tnn_adaptive_c30`) −
   perplexity(`mod_c30`) is the same with dense-matched experts as with legacy experts.
2. **Mixture against dense survives.** The sign of perplexity(`moe_top1`) − perplexity(`dense`) is the
   same with dense-matched experts as with legacy experts.

Each prediction is untestable, not held, unless all its runs exist for seeds 0–2 exactly once, with the
right initialization and the protocol's settings.

**Reported whatever happens:**
- per seed: perplexity under each initialization, decoder MACs and total MACs (output layer
  included), and realized compute;
- option shares;
- the initial diagnostics;
- latency of the dense-initialized c30 model, under the frozen protocol's validity rule.

Nominally equal caps do not guarantee equal realized compute, so compute is reported per run.

**Separation.** These runs are named `..._initdense`. The frozen benchmark's evaluator excludes them,
so they cannot change its verdict (tested). They are analysed with
`frontier.py --lm runs/titnn_bench/lm runs/titnn_efficiency/lm --init-control
runs/titnn_efficiency/lm_init_control --latency runs/titnn_efficiency/latency_init_control.json`.

## Limits

- One cap (0.3) and one alternative initialization; three seeds.
- Answers whether these two comparisons keep their sign, not which initialization is better in
  general.
- Other training fixes (SCAN padding masks, token-weighted gate statistics) are separate experiments
  and are not applied here.

**Run:** `GATE=<gate script> bash research/ti_tnn/bench/run_init_control.sh`, after the frozen queue's
`QUEUE DONE`.
