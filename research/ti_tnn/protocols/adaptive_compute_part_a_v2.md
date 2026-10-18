# TI-TNN adaptive compute, part A v2: the lookup task with learning fixed (frozen brief)

**Status: FROZEN 2026-10-18 (UTC), before any run on its evaluation seeds.** Branch `ti-tnn-learnfix-dev`.

## Why a second version

[Part A](adaptive_compute.md) of the adaptive-compute benchmark failed in a way that measured
learning, not efficiency:
- the adaptive models and a full-compute mixture of experts mostly never learned the variable-hop
  lookup task;
- the dense reference's perfect score was partly selected: the task was designed on the evaluation
  seeds (see the [report](../reports/adaptive_compute.md)).

Diagnostics on development seeds (100–103, never evaluation seeds) found three separate problems, each
with a fix:

| Problem | Evidence (development seeds) | Fix |
|---|---|---|
| Chaining needs more training, even for dense | 3-hop learned on 0 of 4 seeds at 6,000 steps, 3 of 4 at 18,000 | 18,000 steps for every model |
| Routing tokens across experts blocks the lookup circuit | Top-1 mixture with 1 expert learns chaining; with 2 or 8 it never leaves the "queries at chance" plateau, at 6,000 or 18,000 steps | Dense-matched expert initialization, plus a shared expert (half of each layer's width always on, so total compute is unchanged): 3-hop 0.97 and 1.00 on seeds 100 and 101 |
| Skipping hides facts that later queries need, and the router gets no gradient for keeping them (a third outside review) | Without these fixes the skipping models stayed at chance | Cap warm-up (full compute, falling to the cap over the first half of training), and a skipped token still writes its key and value (charged 2D², 1/6 of a block) |

Ruled out on development seeds:
- a balanced loss (it broke the dense model's chaining);
- a hop curriculum (it didn't help with longer training);
- longer training alone for the mixture;
- no balance loss, fewer experts, top-2 routing or a lower learning rate.

The full list of development runs is in the report.

## Design (`research/ti_tnn/bench/hops.py`)

**Fixed for all runs.**
- Part A's task: 8 facts, hops 1–3, half of the queries one-hop, 12 queries per 128 tokens.
- Optimizer AdamW (0.9, 0.95), weight decay 0.1, learning rate 1e-3, 2% warmup, cosine to 10%, batch 64.
- Part A's 2,048 fixed evaluation sequences.

**Changed from part A.** 18,000 steps.

**Seeds:** 200, 201, 202, 203 and 204, used by no earlier run.

| Variant | Recipe |
|---|---|
| `dense` at 2, 3, 4, 6 and 8 layers | plain; 8 layers is the reference |
| `moe_top1` (fixed) | dense-matched expert initialization, shared expert 0.5 |
| `mod` at caps 0.5 and 0.3 | cap warm-up 0.5, skipped tokens keep their key and value |
| `ti_tnn_adaptive` at caps 0.5 and 0.3 | dense-matched expert initialization, shared expert 0.5, cap warm-up 0.5, skipped tokens keep their key and value |

The 0.15 cap from part A is dropped to keep the run within a day. Compute is measured as in part A:
non-embedding MACs per token, from the decisions made.

## Predictions (recorded before any run on seeds 200–204; one-sided; means over the 5 seeds)

A run "learns the task" if its 3-hop accuracy is at least 0.9.

**The learning fixes:**
- **0. Validity** (a condition, not a TI-TNN prediction): the 8-layer dense reference's hard accuracy
  is at least 0.90, and it learns the task on at least 4 of 5 seeds.
- **1.** The fixed `moe_top1` reaches a hard accuracy of at least 0.90.

**Efficiency, with part A's frozen rules** (reference = 8-layer dense; quality = hard accuracy):
- **2.** Some `ti_tnn_adaptive` variant is within 0.02 of the reference, using at most half the compute
  of the cheapest baseline at least as good.
- **3.** Every `ti_tnn_adaptive` variant beats every baseline using no more compute. Untestable, not
  held, for a variant cheaper than every baseline.
- **4.** At each cap, `ti_tnn_adaptive` beats `mod`.
- **5.** In every `ti_tnn_adaptive` variant, 3-hop queries get more compute than 1-hop queries, and
  easy tokens get less than queries of any hop count.

**Decision rule.** The learning fixes work if 0 and 1 hold. TI-TNN passes the favorable task if 2 and 3
hold.

**What I expect** (stated, not predicted): on development seeds, `ti_tnn_adaptive_c30` learned 2-hop
chaining but not 3-hop, so prediction 2 will most likely fail at 0.3. The 0.5 cap was not tried on
development seeds.

## Limits

- **One synthetic task;** facts in context, not in weights.
- **The fixes were chosen on 2–4 development seeds,** so their reliability is estimated here, on 5
  fresh ones.
- **Compute is not equal across the fixes.**
  - The shared expert keeps a mixture's compute equal to dense.
  - Visible skipping makes skipping cost 1/6 of a block rather than 0, and the cap counts that.
- **Not tested:** WikiText-103 with these fixes (the initialization control there is separate).

**Run:** `GATE=<gate script> bash research/ti_tnn/bench/run_part_a_v2.sh`.

## Execution record (appended after the run; nothing above was changed)

- **The run.** 50 runs completed (2026-10-18 07:01–13:48 UTC); the evaluator was frozen at `dcf115d`
  before any result was examined.
- **A fourth outside review: the "ruled out" list above overstates what was shown.**
  - The balanced loss, no balance loss, 2 experts, top-2 and half the learning rate were each tried
    where the matched dense control also failed. They only failed to rescue learning under the tested
    settings.
  - The hop-curriculum runs were affected by a vocabulary bug: each stage shrank the task, shifting
    token IDs. It is fixed in `hops.py`, so those runs do not test a curriculum.
  - The "routing fragments the lookup circuit" explanation in the table above is a hypothesis. The
    one-expert evidence is partial (3-hop 0.76 and 0.20), and the shared expert might do the work alone.
