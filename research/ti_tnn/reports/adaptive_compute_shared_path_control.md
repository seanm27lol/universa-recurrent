# Lookup task, shared-path control: both predictions hold, each by one seed — 2026-10-18

**Protocol:** [`protocols/adaptive_compute_shared_path_control.md`](../protocols/adaptive_compute_shared_path_control.md),
frozen at `9da9c82` before any run, with its analysis (`frontier.py --shared-control`) and a test. **Raw
results:** [`experiments/results/ti_tnn/`](../../../experiments/results/ti_tnn/)
(`adaptive_compute_shared_path_control/`).

## The question

In [part A v2](adaptive_compute_part_a_v2.md), a top-1 mixture of experts with a shared expert learned the
lookup task on 5 of 5 seeds. Its feed-forward output is

y = f_shared(x) + p_e(x) · f_e(x)

where:
- f_shared is an always-on dense network of half the width;
- f_e is the expert the router picks;
- p_e(x) is the router's probability for that expert.

**Does the shared path do the work by itself?** And separately: does TI-TNN's routing among 8 experts
help it, beyond its choice between skipping, a small block and a full block?

## Answer

**By the frozen rules, both predictions hold, and each holds by exactly one seed.**
- **The shared path alone learned the task on 4 of 5 seeds,** against the mixture's 5 of 5. This is a
  half-width dense model: the mixture without its routed experts.
- **TI-TNN with one expert per branch learned it on 1 of 5,** against 2 of 5 with 8 experts.

**Neither difference is distinguishable from chance at 5 seeds.** For both 5 of 5 against 4 of 5, and 2
of 5 against 1 of 5, a two-sided Fisher exact test gives p = 1.0. The test is a post hoc check, not part
of the frozen protocol.

**So the routed experts are not shown to add much to learning this task.** The shared path alone:
- comes within one seed of the full mixture;
- has a fifth of its parameters and 69% of its compute;
- learns more often than the full-width dense reference (3 of 5).

| Lookup task, seeds 200–204 | Learned (3-hop ≥ 0.9) | Hard accuracy, mean | 3-hop accuracy by seed (200, 201, 202, 203, 204) | Decoder MACs per token | Parameters |
|---|---:|---:|---|---:|---:|
| **Shared path alone** (`dense_d128_l8_m2`, half-width MLP) | **4 / 5** | 0.982 | 0.946, 0.972, **0.858**, 1.000, 0.960 | 1.18M | 1.08M |
| Fixed mixture (`moe_top1_initdense_sh50`, part A v2) | 5 / 5 | 0.990 | 0.969, 0.987, 0.960, 0.970, 0.969 | 1.71M | 5.28M |
| Dense reference (`dense`, full-width MLP, part A v2) | 3 / 5 | 0.873 | 0.166, 1.000, 0.986, 0.792, 0.985 | 1.71M | 1.60M |
| **TI-TNN c30, one expert per branch** | **1 / 5** | 0.732 | 0.211, 0.202, 0.166, 0.958, 0.246 | 0.50M | 1.87M |
| TI-TNN c30, 8 experts per branch (part A v2) | 2 / 5 | 0.776 | 0.967, 0.202, 1.000, 0.198, 0.155 | 0.51M | 6.47M |

**About the columns.**
- **Bold rows** are this control's new runs; the others are part A v2's runs on the same seeds.
- **Parameters** are counted from each model as built. Part A v2's run records list negative
  active-parameter counts, from a counting bug fixed in #41.

**The two TI-TNN models fail on different seeds.**
- The one-expert model learned only seed 203; the 8-expert model learned seeds 200 and 202.
- Which seeds succeed looks like luck in training, not a property of either design.
- Both learn 1-hop on every seed (0.976 or above). Where they fail, 3-hop stays near chance
  (0.16–0.25).

## Predictions (frozen before any run)

| # | Prediction | Outcome |
|---|---|---|
| 1 | The shared path alone learns the task on at most 4 of 5 seeds, fewer than the fixed mixture's 5 | **Holds** (4 of 5; seed 202 reached 0.858 on 3-hop) |
| 2 | The one-expert TI-TNN learns the task on fewer seeds than the 8-expert one (2) | **Holds** (1 of 5) |

**The brief's reading of prediction 1.** If it holds, "the routed experts add learning reliability",
though with more parameters and by no shown mechanism. Here that reliability amounts to one seed.

**On prediction 2.** The brief already called it "weak at 5 seeds".

## What it means

- **The mixture recipe's success is mostly the shared path's.** A plain dense model of the shared path's
  width gets within one seed of it, with 69% of the decoder compute and a fifth of the parameters. Five
  seeds cannot show whether the routed experts add the remaining seed's worth.
- **More width did not help the dense model.** The half-width dense model learned the task more often
  than the full-width one (4 against 3 of 5). At 5 seeds, differences of a seed or two are within noise,
  which also limits part A v2's validity verdict.
- **Routing does not explain TI-TNN's success either way.** With 1 or 8 experts, TI-TNN learns on a
  minority of seeds at about 30% compute, the same compute either way. Settling it would take more seeds.

## Limits

- **Five seeds, one synthetic task.** Nothing here concerns WikiText-103, where the lookup-task recipe
  made TI-TNN worse ([#44](https://github.com/seanm27lol/universa-recurrent/pull/44)).
- **Not matched in parameters.** The mixture has about 5× the shared path's parameters; the 8-expert
  TI-TNN about 3.5× the one-expert one.
- **No compute-matched shared-only model.** The [integration plan](../integration_plan.md) lists a
  second control: a shared-only model matched to the mixture's full active width. It was not run.

## Execution

- **Runs.** 10 runs from `9da9c82` (each records its commit, with no tracked changes), 2026-10-18 17:52 to
  19:13 UTC, on the GB10.
- **The GPU gate.** It ran with a co-tenancy log. The gate reported no other GPU process, and waited
  four times (about 1.5 minutes each) for utilization to fall. No run failed.
- **Analysis.** `frontier.py --shared-control` as frozen. `make_tables.py` rebuilds its output from the
  bundled runs.
