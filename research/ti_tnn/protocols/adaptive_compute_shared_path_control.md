# Lookup task: does the shared path do the work by itself? (frozen brief)

**Status: FROZEN 2026-10-18 (UTC), before any run of this control.** Branch `ti-tnn-learnfix-dev`.

## Why

In [part A v2](../reports/adaptive_compute_part_a_v2.md), a top-1 mixture of experts with dense-matched
initialization and a shared expert learned the lookup task on 5 of 5 seeds; without those fixes it
learned nothing. Its feed-forward output is

y = f_shared(x) + p_e(x) · f_e(x),

where f_shared is an always-on dense MLP of half the width, and f_e is the expert the router picks,
weighted by its probability p_e. A fourth outside review pointed out that the result does not say which
term does the work:
- the shared path may let the routed experts cooperate;
- or the shared path may carry the lookup computation by itself.

Likewise, TI-TNN's edge over plain skipping (`mod`) mixes expert routing with its extra "small" option.

## Design (`research/ti_tnn/bench/hops.py`)

Part A v2's task, training (18,000 steps) and fresh seeds 200–204:

| Variant | What it is | Isolates |
|---|---|---|
| `dense_d128_l8_m2_s18000` | an 8-layer dense model whose MLP has the shared path's width (2D) | whether the shared path alone learns the task |
| `ti_tnn_adaptive_c30_initdense_warm50_kv_sh50_e1_s18000` | v2's TI-TNN at the 0.3 cap with one routed expert per branch (router probability always 1) | expert routing, holding the skip/small/full choice fixed |

They are compared with v2's runs on the same seeds: the fixed `moe_top1` (5 of 5), the 8-expert TI-TNN
(2 of 5) and the dense reference (3 of 5).

## Predictions (recorded before any run; one-sided)

A run learns the task if its 3-hop accuracy is at least 0.9.

1. **The shared path alone is not enough.** `dense_d128_l8_m2_s18000` learns the task on at most 4 of 5
   seeds, fewer than the fixed mixture's 5.
2. **Routing helps TI-TNN.** The one-expert TI-TNN learns the task on fewer seeds than the 8-expert one
   (2).

**How to read them.**
- **If 1 fails** (5 of 5), the shared path alone matches the fixed mixture. The routed experts are then
  not shown to add anything to learning this task.
- **If 1 holds,** they add learning reliability. That would still be with more parameters, not by any
  shown mechanism.
- **Prediction 2 is weak at 5 seeds.** Two of five against fewer is a coarse comparison, and the counts
  are reported either way.

`frontier.py --shared-control` applies these. It was written and tested before any run.

## Limits

- **Parameters differ.** The half-width dense model has fewer parameters than the mixture, which also
  has eight routed experts.
- **Five seeds.**
- **This is a synthetic task.** These runs say nothing about WikiText-103.

**Run:** `GATE=<gate script> bash research/ti_tnn/bench/run_shared_path_control.sh`, from the run
directory's repository root.
