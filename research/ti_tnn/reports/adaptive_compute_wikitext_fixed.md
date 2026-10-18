# TI-TNN on WikiText-103 with the lookup-task fixes: substantially worse on text — 2026-10-18

**Protocol:** [`protocols/adaptive_compute_wikitext_fixed.md`](../protocols/adaptive_compute_wikitext_fixed.md),
frozen at `16bba5c` before any run. Its analysis (`frontier.py --wikitext-fixed`) was written and tested
before any run. **Raw results:** [`experiments/results/ti_tnn/`](../../../experiments/results/ti_tnn/)
(`adaptive_compute_wikitext_fixed/`).

## The question

On the synthetic lookup task, three fixes let TI-TNN's mixture-of-experts parts learn
([part A v2](adaptive_compute_part_a_v2.md)):
- dense-matched expert initialization;
- a shared expert;
- for the compute choice, a cap warm-up and visible skipped tokens.

On WikiText-103, the first fix alone had improved TI-TNN at the 30% cap from 49.09 to 46.94 perplexity
([initialization control](adaptive_compute_init_control.md)). Do the other fixes help on real text too?

## Answer

**No. With all the fixes, TI-TNN is much worse on text: perplexity 57.21, against 46.94 with the
initialization fix alone.** All five frozen predictions failed or were untestable. It is now worse than
plain layer skipping at the same cap (52.95), and worse than a smaller dense model that uses less compute
(the 8 × 256 dense model, 55.27 at 7.34M decoder MACs).

| WikiText-103, 3 seeds | Validation perplexity | Test | Decoder MACs per token | Realized block compute | Parameters |
|---|---:|---:|---:|---:|---:|
| `ti_tnn_adaptive_c30`, original | 49.09 ± 0.37 | 48.89 | 8.33M | 0.300 | 202M |
| `ti_tnn_adaptive_c30`, dense-matched init only | 46.94 | — | 8.3M | 0.30 | 202M |
| **`ti_tnn_adaptive_c30`, all lookup fixes** | **57.21 ± 0.75** | 56.40 | **9.27M** | 0.336 | 129M |
| `mod_c30` (plain skipping) | 52.95 ± 0.49 | 52.61 | 7.86M | 0.297 | 51M |
| `dense`, 8 × 256 | 55.27 ± 1.26 | 54.80 | 7.34M | 1.00 | 19M |
| `dense` (reference) | 39.85 ± 0.10 | 39.63 | 27.27M | 1.00 | 51M |

- **Per seed.** Validation perplexity 57.99, 56.49, 57.15; test 57.18, 55.49, 56.52.
- **How it spent compute.** It skipped 62% of block evaluations (each still paying for its key and
  value), used the small experts for 29% and the full experts for only 8%.
- **Over the cap.** Realized compute was 0.33–0.34 of a full block, slightly above the cap (0.30).
- **Fewer parameters.** The shared expert halves the routed experts' width, so the model has 129M
  parameters against the original's 202M.

## Predictions (frozen before any run)

| # | Prediction | Outcome |
|---|---|---|
| 1 | The fixes help on text: below the initialization-only run (46.94) | **Failed** (57.21) |
| 2 | Still ahead of plain skipping: below `mod_c30` (52.95) | **Failed** (57.21) |
| 3 | Part B's efficiency bar: ≤ 40.65 at ≤ half the cheapest-as-good compute | **Failed.** Far from the bar; its compute is 1.26× that of the cheapest baseline at least as good (the 256-wide dense model) |
| 4 | Equal compute: beats every baseline using no more decoder compute | **Failed.** `mod_c30` (52.95 at 7.86M) beats it |
| 5 | Latency ≤ 0.6× dense, with valid execution checks | **Untestable.** See below |

**Latency.** The frozen analysis reported it untestable because no candidate rows were found: the
latency script at `16bba5c` named runs without the warm-up, visible-skip and shared-expert suffixes. That
naming bug is fixed (`latency.run_name` now uses `lm.py`'s own naming). With the corrected names
applied to the same recorded rows, it is still untestable, for a different reason:
- every checkpoint fails the frozen execution check: logits differ by up to 19% of their scale against
  the 5% tolerance, and top-1 agreement on some windows is 0.95–0.98;
- the decoder used here also computed the uncharged query projection for visible skipped tokens (fixed
  since);
- raw, the model decoded at 0.94× dense's time per token (2.08 against 2.21 ms).

## Why might the fixes hurt here? (hypotheses, not tested)

The frozen protocol fixed no explanation in advance. Candidates, each untested:
- **Visible skipping costs budget.** At 1/6 of a block per skip, skipped tokens consume about 0.10 of
  the 0.30 cap, leaving less for real computation. Here the model ran full blocks for only 8% of
  evaluations, against 17% for the initialization-only model.
- **The shared expert halves each routed expert.** On text, narrower routed experts may simply be
  weaker; the initialization-only model kept full-width experts.
- **The cap warm-up** trains at nearly full compute for half the run, then must shed most of it.
- **The lookup task and text want different things.** Facts that later tokens must read are central to
  the lookup task. In language modeling, most context information can apparently be reached through
  the layers tokens do run.

Separating these would take one run per fix on top of the initialization-only model (about 1.2 GPU-hours
each, per seed). Per the [integration plan](../integration_plan.md), no further language-model run is
planned until the typed executor earns one.

## What it means

- **Fixes for one task can hurt another.** The lookup-task fixes made learning possible there and cost
  10 perplexity points on text. Recipes chosen on a synthetic task need confirming on text before they
  are adopted.
- **The best TI-TNN on WikiText-103 is still the initialization-only model** (46.94 at the 30% cap):
  better than plain skipping, far from dense, and not efficient by the frozen bar.

## Execution

- **Runs.** 3 runs from `16bba5c` (each records its commit, with no tracked changes), 2026-10-18 14:21 to
  17:52 UTC, on the GB10 under the idle-GPU gate with a co-tenancy log; no failures.
- **Latency.** `latency.py` at the same commit.
- **Analysis.** `frontier.py --wikitext-fixed`. The latency re-judgment with corrected names is
  recorded in the protocol's execution record.
