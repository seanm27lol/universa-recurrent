# TI-TNN adaptive compute on WikiText-103 with the lookup-task fixes (frozen brief)

**Status: FROZEN 2026-10-18 (UTC), before any run of this experiment.** Branch `ti-tnn-wikitext-fixed`,
based on `ti-tnn-learnfix-dev` at `3e5d55b`.

## The question

On the lookup task, three fixes let the mixture-of-experts parts of TI-TNN learn
([part A v2](../reports/adaptive_compute_part_a_v2.md)):
- dense-matched expert initialization;
- a shared expert;
- for the compute choice, a cap warm-up, plus skipped tokens that keep their keys and values.

On WikiText-103 the [initialization control](../reports/adaptive_compute_init_control.md) showed the
first fix alone improves `ti_tnn_adaptive_c30` from 49.09 to 46.94 perplexity. Do the other fixes help
on real text too, and does the fixed model get closer to the efficiency bar?

## Design (`research/ti_tnn/bench/lm.py`)

- **Runs.** `ti_tnn_adaptive` at the 0.3 cap with `--expert-init dense --shared-expert-frac 0.5
  --cap-warmup-frac 0.5 --skip-keeps-kv`, seeds 0, 1, 2, weights saved for latency.
- **Fixed.** Everything else as the frozen benchmark's part B: data, data order, the 100M-token budget,
  optimizer, schedule, losses and evaluation.
- **Compared with existing runs.**
  - the frozen benchmark's planned baselines: dense at four sizes, the mixtures, and `mod` at 0.5 and
    0.3;
  - `ti_tnn_adaptive_c30` (legacy experts);
  - the initialization control's `ti_tnn_adaptive_c30_initdense`.
- **Compute.** A visible skipped token costs 2D² (1/6 of a block), and the cap counts it. Compute is
  measured from the decisions made.
- **Latency.** `latency.py` times the new checkpoints alongside the dense reference, under the frozen
  execution-validity rule.

## Predictions (recorded before any run; one-sided; means over seeds 0–2; validation perplexity)

1. **The fixes help on text.** The fixed model's perplexity is lower than the initialization-only run
   (46.94).
2. **Still ahead of plain skipping.** It is lower than `mod_c30` (52.95).
3. **The efficiency bar, part B's frozen rule.** It is within 2% of dense (≤ 40.65), at ≤ half the
   compute of the cheapest baseline as good.
4. **Equal compute.** It beats every planned baseline using no more decoder compute.
5. **Latency.** With every execution check valid, it decodes in ≤ 0.6× dense's median time per token.

**What I expect** (stated, not predicted): 1, 2 and 4 hold; 3 fails by a wide margin. The fixes address
learning, not the gap to dense.

`frontier.py --wikitext-fixed` applies these. It was written, with a test, before any run.

## Limits

- **One cap and three seeds.** The parameter difference (TI-TNN has about 4× dense's) is still not
  controlled.
- **Compute differs from `mod_c30`.** Visible skipping changes the compute accounting, while `mod_c30`
  skips without keeping keys and values.

**Run:** `GATE=<gate script> bash research/ti_tnn/bench/run_wikitext_fixed.sh`, from the run directory's
repository root.

## Execution record (appended after the run; nothing above was changed)

- **The run.** 3 runs from `16bba5c`, 2026-10-18 14:21 to 17:52 UTC, with no failures. The latency pass ran
  at the same commit.
- **Predictions 1–4 failed** (57.21 validation perplexity).
- **Prediction 5 was first reported untestable because no candidate rows were found.** `latency.py` at
  `16bba5c` named runs without the warm-up, visible-skip and shared-expert suffixes (a bookkeeping bug).
  `latency.run_name` now uses `lm.py`'s own naming.
- **Re-judged with corrected names, on the same recorded rows, prediction 5 is still untestable.** Every
  checkpoint fails the frozen execution check (logits up to 19% of their scale apart; top-1 agreement
  0.95–0.98 on some windows).
- **The decoder used here also computed an uncharged query projection** for visible skipped tokens. That
  is fixed in `decode.py` since; the timing was not repeated.

