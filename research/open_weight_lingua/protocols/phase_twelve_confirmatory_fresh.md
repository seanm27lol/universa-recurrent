# Phase Twelve confirmatory: the most recent intact assignment, on a third fresh split

**Status: FROZEN 2026-10-12 (UTC), before the confirmatory programs are
generated and before any model forward on them. Branch
`phase-twelve-selection`, at the commit that adds this file, after the
exploratory report (`9e6481d`). Every hypothesis and analysis choice is
fixed here.**

## The question

```text
x = 2              ← v2
y = 3
x = x + 9          ← v1 = 11
y = 5
x = 7              ← v0 = 7, the answer
What is x? Reply with only the integer.
```

The exploratory Phase Twelve ([report](../reports/phase_twelve_selection.md))
found, in both models, that the late read takes the **most recent intact**
assignment of the asked variable.
- Degrading the final assignment makes the model say v1.
- Degrading the final and the previous assignment makes it say v2.
- Degrading only the previous assignment does not hurt.

This study tests those claims once on new programs.

## Data: a third fresh split

- **Exclusions.** The code rebuilds the frozen Phase Two plan and both
  earlier fresh splits: `fresh_2026_10` (seed 206001) and `fresh_2026_10b`
  (seed 207001). It verifies their inventories (`8e0d30ab…f963`,
  `67dc17bb…2a39`) and excludes every canonical program from all of them.
- **Generation.** **1,536 new groups** under split `fresh_2026_10c`, seed
  **208001**, with the unchanged generator. The inventory hash is recorded
  before any forward.
- **Eligible prompts.** As in exploratory Phase Twelve: at least three
  assignments of the asked variable, a literal final one, and v0, v1, v2
  pairwise distinct. The original format. The exploratory rate predicts
  about 455 eligible prompts.

## Settings

As in exploratory Phase Twelve:
- degrading a statement blocks its tokens from every earlier program token,
  from layer 0;
- conditions `baseline`, `final`, `final_and_previous`, `previous`;
- teacher-forced scoring of v0, v1 and v2;
- the bitwise no-op control.

## Confirmatory hypotheses

Per prompt (each "says" is 0 or 1):
- `v1_gain` = says v1 under `final` − says v1 at baseline;
- `v2_gain` = says v2 under `final_and_previous` − says v2 under `final`;
- `loss_previous` = says v0 at baseline − says v0 under `previous`.

**Six tests, Bonferroni-controlled.** Each uses a two-sided percentile
interval at coverage 1 − 0.05/6, from 10,000 whole-group bootstrap resamples
(seed 805100 Gemma, 805101 Qwen).

| ID | Model | Measure | Supported iff |
|---|---|---|---|
| G1 | Gemma-3-12B | `v1_gain` | lower bound > 0.10 |
| G2 | Gemma-3-12B | `v2_gain` | lower bound > 0.10 |
| G3 | Gemma-3-12B | `loss_previous` | upper bound < 0.05 |
| Q1 | Qwen2.5-7B | `v1_gain` | lower bound > 0.10 |
| Q2 | Qwen2.5-7B | `v2_gain` | lower bound > 0.10 |
| Q3 | Qwen2.5-7B | `loss_previous` | upper bound < 0.05 |

## Predictions (recorded before generation)

| ID | Exploratory | Expected half-width at about 455 prompts | Prediction |
|---|---:|---:|---|
| G1 | +0.474 | 0.084 | supported |
| G2 | +0.359 | 0.083 | supported |
| G3 | −0.030 | 0.033 | supported |
| Q1 | +0.602 | 0.087 | supported |
| Q2 | +0.751 | 0.070 | supported |
| Q3 | −0.173 | 0.070 | supported |

## Execution and limits

- **Freeze first.** This protocol, the seed addition, the module, its tests
  and the launcher are committed and pushed before generation. Tests use
  test-only seeds.
- **Runs.** Gemma, then Qwen, each only on an idle GPU. Foreign processes
  are logged, never acted on. An operational failure with zero results may
  be relaunched once and recorded. Partial runs never enter the evaluation.
  One CPU evaluation over both complete runs.
- **Budget.** At most 1 GPU-hour.
- **Limits.** The degrading block is an off-distribution intervention.
  Literal final statements only; one task family. The rule describes the
  read's behaviour when statements lose their context.

**Code:** `src/open_weight_lingua/selection_confirm.py`, launched with
`scripts/run_selection_confirm.sh`.
