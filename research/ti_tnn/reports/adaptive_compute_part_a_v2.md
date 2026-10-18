# Part A v2, the lookup task with learning fixed: the mixture-of-experts fix works; the reference is too unreliable to judge TI-TNN — 2026-10-18

**Protocol:** [`protocols/adaptive_compute_part_a_v2.md`](../protocols/adaptive_compute_part_a_v2.md), frozen at
`2a34728`. Its evaluator (`frontier.py --hops-v2`) was frozen at `dcf115d`, after the queue started and
before any v2 result was examined. **Raw results:**
[`experiments/results/ti_tnn/`](../../../experiments/results/ti_tnn/) (`adaptive_compute_part_a_v2/`,
`lookup_init_check/`, `lookup_learning_diagnostics/`). `make_tables.py` prints every number below from
those files.

## The question

[Part A](adaptive_compute.md) of the adaptive-compute benchmark failed on learning, not efficiency:
- the adaptive models and a full-compute mixture of experts mostly never learned the variable-hop
  lookup task;
- the dense reference's success there had been selected in design.

Diagnostics on development seeds found three problems and a fix for each. Part A v2 applies the fixes
to every model and asks:
1. Do the fixes make the task learnable?
2. If so, does TI-TNN pass the efficiency test it was built for?

The runs use 18,000 steps and fresh seeds 200–204.

## Answer

**By the frozen decision rule, the learning fixes are not established, and TI-TNN's efficiency test is
untestable.** The validity condition failed: the 8-layer dense reference learned the task on only 3 of
5 seeds (needed: 4 of 5, with mean hard accuracy ≥ 0.90; it got 0.873).

Within that verdict, two results are clear:
- **The mixture-of-experts fix works decisively.** With dense-matched initialization and a shared
  expert, the top-1 mixture learns the task on **5 of 5** seeds (0.990). In part A it learned nothing.
  It is more reliable than every dense model.
- **TI-TNN can now learn the whole task at 30% of the compute, but not reliably.**
  - On 2 of 5 seeds, `ti_tnn_adaptive_c30` reaches 0.98 and 1.000 hard accuracy at 0.51M decoder
    MACs per token, 30% of the reference's 1.70M. That is the efficient solution the task was designed
    to reward, and it is reachable.
  - On the other 3 seeds it learns only 1- and 2-hop lookups.
  - It beats plain layer skipping at both caps; plain skipping learns the task on 0 of 5 seeds.

## Results (seeds 200–204; hard accuracy, chance about 0.14)

| Variant | Mean | Per seed | Learned the task (3-hop ≥ 0.9) | Decoder MACs per token |
|---|---:|---|---:|---:|
| `dense`, 2 layers | 0.467 | 0.45, 0.45, 0.45, 0.45, 0.54 | 0/5 | 0.43M |
| `dense`, 3 layers | 0.817 | 0.96, 0.76, 0.77, 0.87, 0.74 | 0/5 | 0.64M |
| `dense`, 4 layers | 0.918 | 1.00, 1.00, 1.00, 1.00, 0.60 | 4/5 | 0.85M |
| `dense`, 6 layers | 0.943 | 0.73, 1.00, 1.00, 0.99, 1.00 | 4/5 | 1.28M |
| **`dense`, 8 layers (reference)** | **0.873** | 0.44, 1.00, 1.00, 0.93, 1.00 | **3/5** | 1.70M |
| **`moe_top1`, fixed** | **0.990** | 0.99, 1.00, 0.99, 0.99, 0.99 | **5/5** | 1.71M |
| `mod`, 50% cap, fixed | 0.612 | 0.65, 0.89, 0.65, 0.44, 0.42 | 0/5 | 0.83M |
| `mod`, 30% cap, fixed | 0.576 | 0.79, 0.68, 0.74, 0.43, 0.24 | 0/5 | 0.50M |
| `ti_tnn_adaptive`, 50% cap, fixed | 0.793 | 1.00, 0.74, 1.00, 0.74, 0.48 | 2/5 | 0.87M |
| **`ti_tnn_adaptive`, 30% cap, fixed** | **0.776** | **0.98**, 0.73, **1.00**, 0.62, 0.55 | **2/5** | **0.51M** |

- **The fixes applied:**
  - every model: 18,000 steps;
  - mixtures: dense-matched expert initialization and a shared expert (half of each layer's width always
    on, so total compute matches dense);
  - adaptive models: a cap warm-up, and skipped tokens still write their keys and values.
- **Compute follows difficulty at the 30% cap.** 3-hop queries get 0.52 of a block against 0.43 for
  1-hop, and easy tokens 0.29, less than any query. At the 50% cap, 3-hop queries got less than 1-hop
  (0.54 against 0.65).

## Predictions (frozen before any v2 run)

| # | Prediction | Outcome |
|---|---|---|
| 0 | Validity: the reference ≥ 0.90 and learns on ≥ 4 of 5 seeds | **Failed** (0.873; 3 of 5) |
| 1 | The fixed `moe_top1` ≥ 0.90 | **Held** (0.990; 5 of 5) |
| 2 | Some TI-TNN variant within 0.02 of the reference at ≤ half the cheapest-as-good compute | Not judged (validity failed). Computed anyway: fails. 0.776 at the 30% cap is far below the reference |
| 3 | Every TI-TNN variant beats every baseline using no more compute | Not judged. Computed anyway: fails at 50%, where 4-layer dense reaches 0.918 at 0.85M against 0.793 at 0.87M; holds at 30% |
| 4 | TI-TNN beats `mod` at each cap | Not judged. Computed anyway: holds (0.793 > 0.612; 0.776 > 0.576) |
| 5 | Compute follows difficulty in every TI-TNN variant | Not judged. Computed anyway: holds at 30%, fails at 50% |

**Decision rules:** "the learning fixes work" needs 0 and 1, so **not established**. "TI-TNN passes"
needs validity, so **untestable**.

## What it shows

- **This task is a fragile test.** Even with 18,000 steps, the 8-layer dense model learns multi-step
  chaining on only 3 of 5 fresh seeds. Shallower dense models do better (4 and 6 layers: 4 of 5), and
  the fixed mixture of experts does best (5 of 5). Chaining is an all-or-nothing learning event. So
  per-seed success rates, not means, are the informative measure. Five seeds are few for estimating
  them.
- **The mixture-of-experts failure is understood and fixed.** Development seeds showed that routing
  tokens across two or more experts kept the model on the "queries at chance" plateau, while one
  expert escaped it. A shared expert removes that: 5 of 5.
- **TI-TNN's efficient solution exists but is learned unreliably.** When it learns, it chains 3 hops
  at 30% compute, and it spends more compute on harder queries. It learns on 2 of 5 seeds, against 3
  of 5 for the reference and 5 of 5 for the full-compute mixture.
- **Not shown:**
  - that adaptive compute is efficient here, since the test is untestable;
  - anything about WikiText-103 with these fixes. They were not applied there; the separate
    [initialization control](adaptive_compute_init_control.md) is the only WikiText follow-up.

## Development diagnostics

These runs are exploratory. They were used to choose the fixes, on development seeds 100–103 (plus one
seed-0 reproduction of part A's dense run), and are not pre-registered results. Every run is in
`lookup_learning_diagnostics/` and printed by `make_tables.py`. Accuracy is given as 1-hop / 2-hop /
3-hop.

| Round | What was tried | What it showed |
|---|---|---|
| 1 (`hops_dev`) | Cap warm-up, router bias toward full, both; for `mod` and TI-TNN at 30% | Nothing chains. Warm-up plus bias gets TI-TNN 1-hop on both seeds; warm-up alone sends `mod`'s multi-hop answers below chance (it applies one hop to every query) |
| 2 (`hops_dev`) | Visible skipped tokens; a balanced loss; MoE with longer training | Visible skipping plus warm-up gives `mod` reliable 1-hop and TI-TNN 2-hop on one seed. **The balanced loss breaks the dense model's chaining** (1.00 / 0.14 / 0.14), so it was dropped |
| 3 (`hops_dev3`) | The dense reference on fresh seeds | **3-hop on 0 of 4 seeds** at 6,000 steps. Part A's 1.000 was selected in design (disclosed in that report). Part A's dense seed 0 reproduces exactly |
| 4 (`hops_dev4`) | Dense with a hop curriculum, 18,000 steps, or both | 18,000 steps: 3-hop on 3 of 4. Curriculum: 2 of 4. Both: 0 of 4 |
| 5 (`hops_dev5`) | The adaptive recipes at 18,000 steps | `mod` 2-hop on 1 seed; TI-TNN 1-hop only; **MoE gets worse with longer training** |
| 6 (`hops_dev6`) | MoE without balance loss, with 2 experts, top-2, or half the learning rate (6,000 steps) | None chains. Uninformative, because dense does not chain on these seeds at 6,000 steps either |
| 7 (`hops_dev7`) | MoE with 1 or 2 experts, 18,000 steps | **1 expert chains (0.92, 0.62); 2 experts stay at chance.** The training loss shows 2 experts never leaving the "queries at chance" plateau |
| 8 (`hops_dev8`) | A shared expert (half the width) | **MoE 3-hop 0.97 and 1.00.** TI-TNN at 30%: 2-hop on both seeds, 3-hop on neither |

The frozen **lookup initialization check** (seeds 0–2, 6,000 steps;
[report](adaptive_compute_hops_init_check.md)) showed dense-matched initialization alone restores 1-hop
lookups (0.9995) but not chaining.

## Limits

- **The fixes were chosen on 2–4 development seeds** and evaluated on 5 fresh ones; success rates from
  5 seeds are coarse.
- **18,000 steps is 3× part A's budget,** so v2 is not comparable step for step with part A.
- **Compute differs from part A.** Visible skipped tokens cost 1/6 of a block, which the cap counts; the
  shared expert keeps a mixture's compute equal to dense.
- **One synthetic task,** with facts in context, not in weights.

## Execution

- **Runs.** 50 runs, from `2a34728` (each records its commit, with no tracked changes), 2026-10-18 07:01
  to 13:48 UTC, on the GB10 under the idle-GPU gate with a co-tenancy log; no failures.
- **Evaluation.** `frontier.py --hops-v2` at `dcf115d`.
- **Hashes.** `frontier_part_a_v2.json` lists each result file's hash; `make_tables.py` checks them
  against the bundle, using each file's original hash where a local path prefix was removed.
