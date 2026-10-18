# WikiText-103 initialization control: TI-TNN's lead over plain skipping grows; top-1's small mean edge over dense reverses under matched initialization — 2026-10-18

**Protocol:** [`protocols/adaptive_compute_init_control.md`](../protocols/adaptive_compute_init_control.md),
frozen at `630cab2` before any run. Its execution record lists one bookkeeping fix, which does not change
the verdict. **Raw results:** [`experiments/results/ti_tnn/adaptive_compute_init_control/`](../../../experiments/results/ti_tnn/).

## The question

Mixtures of experts in this project started with their experts initialized far larger than the dense
MLP: up std 0.044 against 0.02, down std 0.022 against 0.005 at width 512 and 8 layers. So two
comparisons in the [adaptive-compute benchmark](adaptive_compute.md) also compared initializations:
- TI-TNN adaptive against plain layer skipping (`mod`);
- the top-1 mixture of experts against dense.

Do those comparisons keep their sign when experts start like the dense MLP? The intervention rescales
the same random draws, so every other parameter is identical.

## Answer

| WikiText-103, 3 seeds (validation perplexity) | Legacy experts | Dense-matched experts | Comparison | Sign kept? |
|---|---:|---:|---|---|
| `ti_tnn_adaptive_c30` | 49.09 | **46.94** | against `mod_c30` 52.95: −3.85, then −6.01 | **Held** |
| `moe_top1` | 39.57 | 39.99 | against `dense` 39.85: −0.27, then +0.14 | **Failed** |

- **TI-TNN improves on every seed with matched initialization:** 49.21 → 48.22, 49.39 → 46.02,
  48.67 → 46.57. Its lead over plain skipping at the same cap grows, from 3.9 to 6.0 perplexity.
  - Realized compute is unchanged: 0.30 of a block, 8.3M decoder and 34.1M total MACs per token.
  - It is still far from the dense reference (39.85) and from the 4-layer dense model (43.69 at 13.6M
    decoder MACs).
  - Parameters (202M against 51M) and the small-expert option remain uncontrolled.
- **The top-1 mixture loses its small edge over dense.** It is worse on every seed with matched
  initialization (39.04 → 39.82, 39.81 → 40.09, 39.87 → 40.05) and ends 0.14 above dense. Its small
  mean edge over dense reverses under matched initialization. Both gaps are within seed variation (the
  legacy mixture's sd is 0.46), so this shows the edge was not robust to the initialization, not that
  initialization caused it. The frozen prediction asked only whether the sign held, and it did not.
- **Latency is untestable.** The dense-initialized c30 model decoded at 1.84 against dense's 2.21 ms per
  token (0.83×), but its seed-1 checkpoint failed the execution check (logits 2.34 apart on a scale of
  28.1).

**Initial residual-stream scale** (RMS after the last block, first training batch, mean over seeds):

| Family | Legacy experts | Dense-matched |
|---|---:|---:|
| `dense` | 0.184 | — |
| `mod_c30` | 0.127 | — |
| `moe_top1` | 0.420 | 0.105 |
| `ti_tnn_adaptive_c30` | 0.339 | 0.062 |

- **Legacy experts started the residual stream at 2–3× the dense model's scale.** Dense-matched experts
  start it below dense, because a mixture also multiplies its expert's output by the router probability
  (about 0.22 at initialization with 8 experts).
- **So neither initialization matches dense's activation scale.** The control tests the weight rule
  the protocol fixed, not equal activations.

## Predictions (frozen before any run)

| # | Prediction | Outcome |
|---|---|---|
| 1 | Adaptive against skipping keeps its sign | **Held** (−3.85 → −6.01) |
| 2 | Mixture against dense keeps its sign | **Failed** (−0.27 → +0.14) |

## What it changes

- **The adaptive-compute report's prediction 7.** TI-TNN beat plain skipping at the 30% cap. That
  survives the initialization confound and gets stronger. The parameter difference and the
  small-expert option are still not isolated.
- **The real benchmark's "top-1 beats dense".** Its small mean edge reverses under matched
  initialization; that report's errata already flagged the confound.
- **What it doesn't change.** The adaptive-compute verdict, *not validated*, stands. Even with better
  initialization, TI-TNN at 30% compute is 18% worse than dense, outside the 2% bar.

## Execution

- **Runs.** 6 runs, from `630cab2`, 2026-10-17 22:47 to 2026-10-18 05:59 UTC, on the GB10 under the
  idle-GPU gate, with no failures. Each records its source commit and step-0 diagnostics.
- **Initial diagnostics for the legacy runs.** Reconstructed by `bench/init_diagnostics.py` from the same
  seeds and first batch.
- **Analysis.** `frontier.py --init-control`. Its `run_key` fix aligns the latency rows' absolute paths
  with the manifest's relative ones; the verdict is the same under either version.
