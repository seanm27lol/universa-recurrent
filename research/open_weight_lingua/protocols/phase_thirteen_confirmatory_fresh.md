# Phase Thirteen confirmatory: the account at Gemma-3-27B, on a fourth fresh split

**Status: FROZEN 2026-10-12 (UTC), before the confirmatory programs are
generated and before any model forward on them. Branch
`phase-thirteen-scale`, at the commit that adds this file, after the
exploratory report (`aa30041`). Every hypothesis and analysis choice is
fixed here.**

## The question

The exploratory Phase Thirteen ([report](../reports/phase_thirteen_scale.md))
found that Gemma-3-27B follows the account confirmed on the smaller models:
- **Routing.** Its answer positions never need the program directly.
  Blocking them at every layer costs about 1 point. Every post-program
  position needs the program until layer 46 of 62: blocking from 46 costs
  about 3 points, blocking from 38 about 19.
- **Selection.** Degrading the final assignment gives v1, degrading the
  final and the previous gives v2, and degrading only the previous costs
  nothing.

This study tests those claims once, on Gemma-3-27B, on new programs.

## Data: a fourth fresh split

- **Exclusions.** The code rebuilds the frozen Phase Two plan and the three
  earlier fresh splits: `fresh_2026_10` (206001), `fresh_2026_10b` (207001)
  and `fresh_2026_10c` (208001). It verifies their inventories
  (`8e0d30ab…f963`, `67dc17bb…2a39`, `c56dfba4…a254`) and excludes every
  canonical program from all of them.
- **Generation.** **1,536 new groups** under split `fresh_2026_10d`, seed
  **209001**. The inventory hash is recorded before any forward.
- **Routing prompts.** Every prompt of the **first 512 generated groups**
  (2,048 prompts), in the original format.
- **Selection prompts.** Every eligible prompt of all 1,536 groups, by
  Phase Twelve's rule.

## Settings

As in exploratory Phase Thirteen: Gemma-3-27B under its lock, teacher-forced
digit accuracy, the Phase Eight knockout and the Phase Twelve degrading
block, and the bitwise no-op control.
- **Routing conditions.** Baseline; the answer positions blocked from the
  program from layer 0 (`answer@0`); every post-program position blocked
  from layer 38 (`after_program@38`) and from layer 46
  (`after_program@46`).
- **Selection conditions.** As in Phase Twelve.

## Confirmatory hypotheses

**Six tests, Bonferroni-controlled.** Each uses a two-sided percentile
interval at coverage 1 − 0.05/6, from 10,000 whole-group bootstrap resamples
(seed 806100). Groups are resampled within each claim's prompts.

| ID | Measure | Prompts | Supported iff |
|---|---|---|---|
| R1 | loss under `answer@0` | routing | upper bound < 0.05 |
| R2 | loss under `after_program@46` | routing | upper bound < 0.05 |
| R3 | loss under `after_program@38` | routing | lower bound > 0.05 |
| S1 | v1 gain when the final assignment is degraded | selection | lower bound > 0.10 |
| S2 | v2 gain when the previous is degraded too | selection | lower bound > 0.10 |
| S3 | loss when only the previous is degraded | selection | upper bound < 0.05 |

## Predictions (recorded before generation)

| ID | Exploratory | Expected interval | Prediction |
|---|---:|---|---|
| R1 | +0.010 | about [−0.003, +0.024] at 512 groups | supported |
| R2 | +0.029 | about [+0.014, +0.043] | supported, **the closest call** |
| R3 | +0.195 | about [+0.165, +0.224] | supported |
| S1 | +0.775 | wide margin | supported |
| S2 | +0.769 | wide margin | supported |
| S3 | −0.009 | wide margin | supported |

## Execution and limits

- **Freeze first.** This protocol, the seed addition, the module, its tests
  and the launcher are committed and pushed before generation. Tests use
  test-only seeds.
- **Run.** One Gemma-3-27B run, only on an idle GPU with enough free memory.
  Foreign processes are logged, never acted on. An operational failure with
  zero results may be relaunched once and recorded. A partial run never
  enters the evaluation. One CPU evaluation.
- **Budget.** At most 2.5 GPU-hours: about 10,240 routing forwards and about
  5,400 selection forwards, at about 0.34 s each.
- **Limits.** One model; knockout limits as before; one task family.

**Code:** `src/open_weight_lingua/scale_confirm.py`, launched with
`scripts/run_scale_confirm.sh`.
