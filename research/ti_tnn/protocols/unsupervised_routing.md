# TI-TNN: can the router find question types without labels? (frozen brief)

**Status: FROZEN 2026-10-15 (UTC), before any training run of this experiment.
Branch `ti-tnn-unsupervised-routing`, based on `main` at `a24d3f3`.**

## The question

In the routing benchmark the router is told each query's type. Its Brier
loss targets a branch per type (for example `capital_of` → Branch 2), and the
bridge gate is targeted per type too. A real model gets no such labels.
This experiment removes them:
- **Training signal.** Only the answer's cross-entropy plus the compute
  penalty.
- **Cost.** The penalty now also charges each expected bridge firing one
  branch-cost unit (`--bridge-cost 1.0`).
- **Question.** Does the router still route by question type?

## Design

`python research/ti_tnn/ti_tnn_train.py --benchmark unsupervised`, with
default settings: 3 seeds, 3,000 steps, batch 512, the routing benchmark's
world and training stream. NVIDIA GB10, `.venv-phase2`.

| Variant | Labels in training | Bridge |
|---|---|---|
| `supervised` | routing and gate labels (the routing benchmark's routed + bridge model) | yes |
| `unsup_bridge` | none | yes, firings charged |
| `unsup_nobridge` | none | no |

**Measures.** All are label-free, on 8,192 fresh-context queries:
- **Task accuracy**, overall and by type.
- **Routing purity.** For each query type, the share routed to its most common
  branch, averaged over types.
- **Information.** Normalized mutual information between query type and branch.
- **Distinct majority branches.**
- **Separation.** Whether the two pairs that share an entity kind
  (`region_of`/`capital_of`, `size_of`/`country_of`) have different majority
  branches.
- **Compute.** Branch runs per query and the bridge firing rate.

A token-wise branch sees only the entity token. **Without the bridge**,
two relations on the same entity kind must go to different branches for
both to be answered. **With the bridge**, the gate reads the question, so a
branch plus a gated bridge could in principle answer two relations itself.

## Predictions (recorded before any run; all one-sided)

1. **The task forces routing.** `unsup_nobridge` and `unsup_bridge` each reach
   fresh-context accuracy of at least 0.95 on at least 2 of 3 seeds.
2. **By type.** On every `unsup_nobridge` seed with accuracy of at least
   0.95:
   - both pairs are separated;
   - routing purity is at least 0.9.
3. **Cheap.** `unsup_bridge` fires the bridge on at most 20% of queries on
   every seed. The bridge isn't needed on this task, and firing is charged.
4. **Reference.** `supervised` reproduces the routing benchmark: accuracy
   1.000 and 1.40 branch runs per query on every seed.

Every measure is reported whatever happens, including any seed where the
router collapses onto too few branches.

## Limits

- One toy world.
- Unsupervised routing here is still per query, with a router that pools
  the whole query.
- The compute penalty and bridge cost are fixed in advance; nothing is
  tuned after seeing results.

**Code:** `research/ti_tnn/ti_tnn_train.py` (`run_unsupervised`,
`compute_unsupervised_loss`, `routing_structure`).
