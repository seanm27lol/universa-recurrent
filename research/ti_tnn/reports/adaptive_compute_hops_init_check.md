# Lookup task: expert initialization explains part of the mixture-of-experts failure, not all — 2026-10-17

**Protocol:** [`protocols/adaptive_compute_hops_init_check.md`](../protocols/adaptive_compute_hops_init_check.md),
frozen at `1ac2b1c` before any run.

## The question

In part A of the [adaptive-compute benchmark](adaptive_compute.md), the top-1 mixture of experts
(`moe_top1`) never learned the variable-hop lookup task: chance on every hop count, at full compute.
The 8-layer dense model learned it perfectly. The experts start much larger than the dense MLP: up std
0.088 against 0.02, and down std 0.044 against 0.005. Does starting them like the dense MLP fix it?

## Answer

**Partly. The prediction failed.** With dense-matched expert initialization, `moe_top1` learns every
1-hop lookup (0.9995), where before it stayed at chance (0.14). It still fails 2-hop and 3-hop chains
(0.17 and 0.16, about chance). So the legacy initialization blocked even the basic lookup. Something
else in the sparse-expert recipe also blocks chaining.

| Model (part A task, 6,000 steps) | Hard accuracy | 1 hop | 2 hops | 3 hops |
|---|---:|---:|---:|---:|
| `dense` (8 layers) | 1.000 | 1.00 | 1.00 | 1.00 |
| `moe_top1`, legacy initialization (part A) | 0.143 | 0.14 | 0.14 | 0.14 |
| **`moe_top1`, dense-matched initialization** | **0.440 ± 0.014** | **1.00** | **0.17** | **0.16** |

**Per seed (0, 1, 2).** Hard accuracy 0.434, 0.430, 0.456. 1-hop is 0.9999, 0.9993, 0.9994 on the
three seeds; easy tokens 1.000 on all.

**This is the same pattern as the 2-layer dense model** (1.00, 0.19, 0.17). The mixture behaves as if
it had too little depth for chaining, despite having all 8 layers.

## Prediction (frozen before any run)

| # | Prediction | Outcome |
|---|---|---|
| 1 | `moe_top1_initdense` mean hard accuracy ≥ 0.90 | **Failed** (0.440) |

## What it means

- **The legacy expert initialization is a real defect for this task.** It stops the mixture of experts
  from learning even single-step lookups. Every expert-based result in part A, `ti_tnn_adaptive`
  included, carries that confound.
- **It is not the whole explanation.** Multi-step chaining still fails at full compute. Candidates not
  tested here include:
  - the loss weighting (hard queries are about 12% of the scored positions);
  - slower learning;
  - the router-probability scaling of expert outputs;
  - routing instability.

  Exploratory diagnostics on development seeds (100, 101) are looking at these before any fix is
  frozen.

## Execution

- **Hardware and timing.** Three runs on the GB10 under the idle-GPU gate, 2026-10-17 between 16:06 and 16:25 UTC,
  after the frozen queue. They took 193–204 s each.
- **Provenance.** Every run records its source commit (`1ac2b1c`, no tracked changes).
- **Results.** `runs/titnn_efficiency/hops_init_check/`; result.json sha256 prefixes `7dbe5436…`,
  `e0653619…`, `5ca54421…` (seeds 0, 1, 2).
