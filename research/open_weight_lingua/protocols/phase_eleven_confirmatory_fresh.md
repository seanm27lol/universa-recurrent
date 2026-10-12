# Phase Eleven confirmatory: a disrupted final assignment, on fresh programs

**Status: FROZEN 2026-10-12 (UTC), before the confirmatory programs are
generated and before any model forward on them. Branch
`phase-ten-dependency-knockout`, at the commit that adds this file, after
the Phase Eleven exploratory report (`c9a397d`). Every hypothesis and
analysis choice is fixed here.**

## The question

```text
x = 4
y = 3
x = 7              ← the asked variable's final statement (a literal)
y = 5
What is x? Reply with only the integer.      → 7   (superseded value: 4)
```

Phases Ten and Eleven, exploratory on reused prompts
([Ten](../reports/phase_ten_dependency_knockout.md),
[Eleven](../reports/phase_eleven_supersession.md)), found the following.
- Blocking the asked variable's final statement from earlier lines damages
  the answer, whether the blocked lines belong to the asked variable or to
  the other one.
- When it is damaged, the model mostly falls back to the **superseded
  value**, the asked variable's previous assignment.

This study tests those claims once on new programs, for literal final
statements, where the answer does not depend on earlier lines at all.

## Data: a second fresh split

- **Exclusions.** The code rebuilds the frozen Phase Two plan (hash
  `ff060012…667f`) and Phase Nine's fresh split `fresh_2026_10` (seed
  206001, 256 groups). It verifies the latter's inventory hash equals
  `8e0d30ab8915e08f465ed29c2861fdd194f4a02c7e73afaabc6676cb9efff963`, then
  excludes every canonical program from both.
- **Generation.** **512 new groups** under split `fresh_2026_10b`, seed
  **207001**, with the unchanged generator. The run records the inventory
  hash before any forward.
- **Prompts.** The original format; both sides, both questions. Only prompts
  whose asked variable's final statement is a literal with earlier
  assignments, and whose superseded value differs from the answer, are run.
  The exploratory rate predicts about 350 such prompts.

## Settings

As in exploratory Phase Eleven:
- teacher-forced digit accuracy for the answer and the superseded value;
- the knockout hook and the bitwise no-op control;
- line assignment by each token's first character;
- both blocks from layer 0:
  - `same_variable`: the final statement blocked from the asked variable's
    earlier lines;
  - `other_variable`: the final statement blocked from the other
    variable's earlier lines.

## Confirmatory hypotheses

Per prompt:
- **loss** is baseline correctness minus correctness under the block;
- **superseded gain** is superseded-correctness under the same-variable
  block minus superseded-correctness at baseline.

**Six tests, Bonferroni-controlled.** Each uses a two-sided percentile
interval at coverage 1 − 0.05/6, from 10,000 whole-group bootstrap resamples
(seed 804100 Gemma, 804101 Qwen).

| ID | Model | Measure | Supported iff |
|---|---|---|---|
| G1 | Gemma-3-12B | loss under `same_variable` | lower bound > 0.05 |
| G2 | Gemma-3-12B | loss under `other_variable` | lower bound > 0.05 |
| G3 | Gemma-3-12B | superseded gain under `same_variable` | lower bound > 0.10 |
| Q1 | Qwen2.5-7B | loss under `same_variable` | lower bound > 0.05 |
| Q2 | Qwen2.5-7B | loss under `other_variable` | lower bound > 0.05 |
| Q3 | Qwen2.5-7B | superseded gain under `same_variable` | lower bound > 0.10 |

## Predictions (recorded before generation)

These are exploratory values on the same prompt type. Expected 99.17%
half-widths at 512 fresh groups come from the exploratory spread.

| ID | Exploratory | Expected half-width | Prediction |
|---|---:|---:|---|
| G1 | +0.519 | about 0.10 | supported |
| G2 | +0.365 | about 0.10 | supported |
| G3 | +0.435 | about 0.10 | supported |
| Q1 | +0.342 | about 0.09 | supported |
| Q2 | +0.473 | about 0.10 | supported |
| Q3 | +0.327 | about 0.09 | supported |

## Descriptive (not tested)

- Accuracy, superseded rate and preference for each condition.
- The share of wrong answers that are the superseded value, per block.

## Execution and limits

- **Freeze first.** This protocol, the seed addition, the module, its tests
  and the launcher are committed and pushed before generation. Tests use a
  test-only seed.
- **Runs.** Gemma, then Qwen, each only on an idle GPU. Foreign processes
  are logged, never acted on. An operational failure with zero results may
  be relaunched once and recorded. Partial runs never enter the evaluation.
  One CPU evaluation over both complete runs.
- **Budget.** At most 1 GPU-hour.
- **Limits.**
  - Layer-0 blocks only; an off-distribution intervention; teacher-forced
    scoring; one task family.
  - The "most recent intact assignment" reading is an interpretation, not
    tested here.

**Code:** `src/open_weight_lingua/supersession_confirm.py`, launched with
`scripts/run_supersession_confirm.sh`.
