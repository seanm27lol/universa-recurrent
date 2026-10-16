# TI-TNN on real benchmarks: WikiText-103 and SCAN — 2026-10-16

**Protocol:** [`protocols/real_benchmarks.md`](../protocols/real_benchmarks.md), frozen
at `e25b261` before any full run. The 100M-token budget was fixed at `316c18f` after
a false start that produced no result; it is recorded in the protocol.

## The question

On established benchmarks, how does a TI-TNN feed-forward layer compare with standard
architectures? The TI-TNN layer is top-1 experts plus a Noul-gated sheaf bridge to
the runner-up expert. The comparison is at a fixed budget: the same decoder, data
order, optimizer and schedule, with only the feed-forward block differing. No variant
gets routing labels, because real text has none.

## Answer

**The bridge does not earn its compute on real data. Standard mixtures of experts
do at least as well.**
- **WikiText-103.** Top-2 experts is clearly best: perplexity 37.92 against 39.57 for
  top-1 and 39.85 for dense.
- **Charged TI-TNN.** It shuts its gate off (0.1–0.2% of tokens fire) and becomes
  top-1: 39.59.
- **Free TI-TNN.** It fires on 44% of tokens, using 1.44 expert runs per token at
  inference and twice top-1's compute in training. That buys only 39.24.
  Interpolating between top-1 and top-2 at the same 1.44 expert runs gives about
  38.84. So the bridge is a *less* efficient way to spend the extra expert than
  top-2's plain weighted average.
- **SCAN.** No architecture solves the compositional splits. Every variant scores
  about 1.0 on the random split, about 0 on add-jump and under 0.04 on length.
  TI-TNN shows no advantage. On add-turn-left, every mixture variant, TI-TNN
  included, generalizes worse than the dense model.

This agrees with the toy benchmarks. There, the bridge's composition result came from
isolation enforced by construction, which real data does not provide.

## Results

### WikiText-103 (3 seeds; perplexity, lower is better)

| Variant | Validation | Test | Expert runs per token (inference) | FFN MACs per token per layer, inference / training | Training tokens/s |
|---|---|---|---:|---|---:|
| `dense` | 39.85 ± 0.10 | 39.63 ± 0.25 | 1 (dense) | 2.10M / 2.10M | 45,191 |
| `moe_top1` | 39.57 ± 0.46 | 39.69 ± 0.51 | 1.00 | 2.10M / 2.10M | 29,850 |
| `moe_top2` | **37.92 ± 0.75** | **38.11 ± 0.56** | 2.00 | 4.20M / 4.20M | 26,890 |
| `ti_tnn_charged` | 39.59 ± 0.81 | 39.76 ± 0.90 | 1.00 (fires on 0.1–0.2%) | 2.10M / 4.40M | 24,907 |
| `ti_tnn_free` | 39.24 ± 0.18 | 39.38 ± 0.45 | 1.44 (fires on 44%) | 3.12M / 4.40M | 25,898 |

**Per-seed validation perplexity:**

| Variant | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| `dense` | 39.90 | 39.74 | 39.91 |
| `moe_top1` | 39.04 | 39.81 | 39.87 |
| `moe_top2` | 37.66 | 38.76 | 37.34 |
| `ti_tnn_charged` | 39.70 | 38.73 | 40.35 |
| `ti_tnn_free` | 39.30 | 39.04 | 39.38 |

**Model size.** The mixture variants have 169–170M parameters, against 51M for dense.
Peak GPU memory is 14.7–18.8 GB.

**Where the free TI-TNN's gate fires.** By layer 0–7, averaged over the three seeds:
0.47, 0.07, 0.00, 0.07, 0.56, 0.67, 0.82, 0.88. It fires mostly in the last four
layers and the first.

**Load balance.** No expert takes more than 22% of tokens in any layer of any
mixture run, against 12.5% for perfectly even routing over 8 experts.

### SCAN (3 seeds; test exact match)

| Split | `dense` | `moe_top1` | `moe_top2` | `ti_tnn_charged` | `ti_tnn_free` |
|---|---|---|---|---|---|
| simple (random) | 0.9999 ± 0.0001 | 0.9997 ± 0.0003 | 0.9999 ± 0.0001 | 0.9999 ± 0.0001 | 0.9998 ± 0.0001 |
| length | 0.021 ± 0.013 | 0.002 ± 0.002 | 0.001 ± 0.001 | 0.021 ± 0.017 | 0.004 ± 0.007 |
| addprim_jump | 0.0002 ± 0.0004 | 0.0002 ± 0.0003 | 0.0003 ± 0.0005 | 0.0007 ± 0.0012 | 0.0000 ± 0.0000 |
| addprim_turn_left | **0.965 ± 0.019** | 0.926 ± 0.047 | 0.826 ± 0.015 | 0.860 ± 0.021 | 0.845 ± 0.063 |

Every run reached 1.000 on a 2,000-example sample of its training set. The failures
are generalization, not fitting.

## Predictions (frozen before any full run)

| # | Prediction | Outcome |
|---|---|---|
| WikiText 1 | `moe_top1` below `dense` on every seed | **Failed.** Seed 1: 39.81 against 39.74. The means differ by only 0.28 (39.57 against 39.85). |
| WikiText 2 | `moe_top2` below `moe_top1` on every seed | **Held** on all three seeds. |
| WikiText 3 | `ti_tnn_charged` fires on under 5% and is within 2% of `moe_top1` on every seed | **Failed as worded.** It fires on 0.1–0.2% and the means agree to within 0.1%. But on seed 1 it was 2.7% *lower* (better), outside a two-sided band. |
| WikiText 4 | `ti_tnn_free` fires on over 10%; its mean is below `moe_top1`'s and not below `moe_top2`'s | **Held:** it fires on 43.6–45.1%, with a mean of 39.24, between 37.92 and 39.57. On seed 0 it was above `moe_top1`. |
| WikiText 5 | Throughput: `dense` > `moe_top1` > `moe_top2` ≥ TI-TNN | **Held** in the means and on seeds 1 and 2. **Failed on seed 0**: `ti_tnn_free` ran at 25,836 tokens/s against `moe_top2`'s 25,226. |
| SCAN 1 | Every variant at least 0.95 on `simple` | **Held** (at least 0.9995 on every seed). |
| SCAN 2 | Every variant at most 0.20 on `length` | **Held** (at most 0.035). |
| SCAN 3 | Every variant at most 0.20 on `addprim_jump` | **Held** (at most 0.002). |
| SCAN 4 | No TI-TNN variant beats every baseline on `addprim_jump` by more than 0.10 | **Held**. |

**Lessons.**
- WikiText 3 is the third prediction this project lost to a two-sided "within N"
  band. Future predictions should say which direction matters.
- The WikiText 1 failure is real. At 100M tokens and this size, top-1 experts barely
  beat dense, even with 3.3× the parameters.

## Execution

- **Runs.** All 75 completed with no failures: 15 WikiText (5 variants × 3 seeds) and
  60 SCAN (5 variants × 4 splits × 3 seeds). They ran on the NVIDIA GB10 under the
  GPU gate, from 2026-10-15 19:54 to 2026-10-16 16:36 UTC.
- **Data.** WikiText-103 hashes are in `runs/titnn_data/wikitext103/meta.json`. The
  SCAN file hashes are in each result.
- **Summary.** `runs/titnn_bench/summary.json` has sha256 `223662cf…`. It was built
  by `bench/summarize.py` from the per-run `result.json` files, which are in the
  git-ignored `runs/titnn_bench/`. Byte-identical copies of all 75 and the summary are in
  [`experiments/results/ti_tnn/real_benchmarks/`](../../../experiments/results/ti_tnn/), where
  `make_tables.py` rebuilds the summary from them.
- **Co-tenancy.** A GPU log ran from 01:36 UTC on 2026-10-16, which covers seeds 1
  and 2. Beyond the rf-moe dashboard (451 MiB), three small processes (229–375 MiB)
  appeared once each, for one 30-second sample. The seed-0 WikiText runs started
  before the log, so their co-tenancy is unknown. Their throughput was about 10%
  lower (dense 42.3k against 46.6k tokens/s), which is why the throughput prediction
  failed on seed 0. Accuracies and perplexities do not depend on co-tenancy.

## What this does not establish

- **Small scale.** One model size (about 25M active, 170M total parameters), one
  token budget and one hyperparameter set, shared by every variant and tuned for
  none. Larger models, longer training or per-architecture tuning could change the
  ordering.
- **Three seeds and no significance tests.** Differences smaller than the seed spread
  (for example, `ti_tnn_free` against `moe_top1`) are not established.
- **Only this bridge.** The test is of this bridge as one design choice (expert
  outputs as stalks, a tanh coboundary, a straight-through gate). It does not cover
  the categorical version (`cdl.py`), which was not run here.
- **Compute.** Counted MACs and measured training throughput of unoptimized expert
  loops. Inference speed was not measured.

## Errata and caveats (added 2026-10-16, after an outside review)

None of these changes a perplexity or an exact-match score above.

- **Firing rates and expert runs are batch-averaged.** `evaluate` averages each
  batch's rate with equal weight. The 482 validation windows form 15 batches of 32 and
  one of 2, so the last two windows count about 15 times too much. The error is at most
  (1/16 − 2/482) ≈ 0.058 times the spread of the per-batch rates. For the charged
  TI-TNN (0.1–0.2%) that is negligible. For the free TI-TNN (44%) it is a fraction of
  a point. The per-batch rates were not saved, so the exact token-weighted values
  cannot be recomputed. Mixture expert runs (exactly 1 or 2) are unaffected.
- **Experts and the dense MLP start from different initializations.** Experts are
  initialized at std D^-1/2 (up) and H^-1/2 (down), and miss the GPT-2 residual
  scaling the dense MLP gets. The dense down-projection starts at std 0.005, against
  0.0221 for experts. So dense against the mixtures also compares two
  initializations. Which way this cuts is not established.
- **SCAN's routing losses include padding.** Cross-entropy ignores padded positions,
  but the balance loss and the gate charge average over them. Right padding cannot
  change any real token's output, but batch composition changes the routing
  objective slightly. It affects the mixture and TI-TNN variants, not dense.
- **The interpolation is an estimate.** "About 38.84 at the same compute" interpolates
  linearly between the measured top-1 and top-2 models; no model was measured there.
- **The bridge is an MLP.** W_e tanh(F_u o_u − F_v o_v) is algebraically a
  one-hidden-layer MLP on the two expert outputs side by side. Its failure here is a
  result about that fusion layer, not about sheaf structure in general.

Code: [`bench/`](../bench/): `ti_tnn_models.py`, `lm.py`, `scan.py`, `prep_wikitext.py`,
`summarize.py`, `run_all.sh`.
