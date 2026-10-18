# Lookup task: does expert initialization explain why the mixture of experts never learned? (frozen brief)

**Status: FROZEN 2026-10-17 (UTC), before any run of this check.** Branch `ti-tnn-learnable-adaptive`,
based on `ti-tnn-init-control` at `630cab2`.

## Why

In part A of the [adaptive-compute benchmark](adaptive_compute.md), the 8-layer dense model learned
the variable-hop lookup task perfectly (1.000 on every seed). `moe_top1` stayed at chance on every hop
count and every seed: hard accuracy 0.14, with 1-hop also 0.14. It has the same attention layers and
the same compute.

One difference is initialization. The experts are drawn far larger than the dense MLP:

| Weights (width 128, 8 layers) | Experts ("legacy") | Dense MLP |
|---|---|---|
| Up-projection | std 0.088 | std 0.02 |
| Down-projection | std 0.044 | std 0.005 |

That mismatch was already declared as a confound for WikiText-103. This check asks whether it also
explains the failure to learn here.

## Design (`research/ti_tnn/bench/hops.py`)

- **Runs.** `moe_top1` with `--expert-init dense`: up std 0.02, down std 0.02/√(2L) = 0.005. The same
  random draws are rescaled; every other parameter is identical (tested).
- **Fixed.** Everything else is the frozen part A: task, 6,000 steps, optimizer, schedule and the
  fixed evaluation sequences.
- **Seeds and hardware.** Seeds 0, 1, 2, on the GB10, under the idle-GPU gate, after the frozen queue
  finishes.
- **Compared with.** Part A's `moe_top1` (legacy initialization) and `dense`, from the same hardware.

## Prediction (recorded before any run; one-sided)

1. `moe_top1_initdense` reaches a mean hard accuracy of at least 0.90 over seeds 0–2. Part A's
   `moe_top1` was 0.14.

**How to read it.**
- **If it holds**, the legacy expert initialization is enough to stop a full-compute mixture of experts
  from learning this task. Part A's expert-based results then reflect that initialization, not
  routing or adaptive compute.
- **If it fails**, the cause is elsewhere: routing, the load-balancing loss, or the probability
  scaling of expert outputs.

Accuracy by hop count and per seed is reported either way.

## Limits

- **Scope.** One task and one alternative initialization; three seeds.
- **What it doesn't test.** It does not touch why plain layer skipping (`mod`, which has no experts)
  also failed to learn. That is diagnosed separately, on development seeds, before a separately frozen
  follow-up.

**Run:** `GATE=<gate script> bash research/ti_tnn/bench/run_hops_init_check.sh`.
