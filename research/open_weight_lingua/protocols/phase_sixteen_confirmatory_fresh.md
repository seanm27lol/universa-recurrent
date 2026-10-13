# Phase Sixteen confirmatory: with the question first, the program carries the asked variable, on a sixth fresh split

**Status: FROZEN 2026-10-13 (UTC), before the confirmatory programs are
generated and before any model forward on them. Branch
`phase-sixteen-question-route`, at the commit that adds this file, after the
Phase Sixteen exploratory report (`5f06812`). Every hypothesis and analysis
choice is fixed here.**

## The question

```text
What is x? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          → 7  (y ends at 14)
```

The exploratory Phase Sixteen
([report](../reports/phase_sixteen_question_route.md)) found an unpredicted
result about prompts with the question first. When the program tokens
cannot attend to the question (`program_blind`), the late read falls back to
the most recent assignment of either variable:
- programs ending on the other variable are answered with that variable's
  value;
- programs ending on the asked variable stay mostly right.

This study tests that once, in both models, on new programs.

## Data: a sixth fresh split

- **Exclusions.** The code rebuilds the frozen Phase Two plan and the five
  earlier fresh splits (seeds 206001–210001). It verifies their inventories
  (the four recorded by the Phase Fifteen confirmatory protocol, plus
  `fresh_2026_10e`'s `4f744c7c0f1b…`) and excludes every canonical program
  from all of them.
- **Generation.** **768 new groups** under split `fresh_2026_10f`, seed
  **211001**. The inventory hash is recorded before any forward. Both
  families receive the same programs.
- **Prompts.** All 3,072 prompts, question first (Phase Seven's
  `original_before`).
- **Main stratum.** Programs ending on the other variable, excluding
  prompts whose answer passes through a copy of it.
- **Asked-last stratum.** Programs ending on the asked variable. Each
  program is asked about both variables, so it usually contributes one
  prompt to each stratum.

## Settings

Gemma-3-12B and Qwen2.5-7B under their locks. Teacher-forced scoring of the
answer and of the other variable's final value. The bitwise no-op control
runs on every prompt. Blocks apply from layer 0:

| Condition | Queries | Keys |
|---|---|---|
| `baseline` | — | — |
| `program_blind` | every program token | every question token |
| `reader_blind_clause` (descriptive) | every post-program position | the tokens of `What is x?` only, leaving the answer-format instruction visible |

## Confirmatory hypotheses

Measures:
- `other_rise`: `program_blind` minus `baseline`, in the rate of saying the
  other variable's final value. Main stratum; prompts where that value
  differs from the answer.
- `loss`: `baseline` minus `program_blind`, in accuracy. Main stratum.
- `stratum_contrast`: per program, the main-stratum prompt's `loss` minus
  the asked-last prompt's `loss`. Programs with a prompt in both strata.

**Six tests, Bonferroni-controlled.** Each uses a two-sided percentile
interval at coverage 1 − 0.05/6, from 10,000 whole-group bootstrap resamples
(seed 808100 Gemma, 808101 Qwen). Groups are resampled within each claim's
units.

| ID | Family | Measure | Supported iff | Claim |
|---|---|---|---|---|
| G1 | Gemma-3-12B | `other_rise` | lower > 0.05 | without the question, the read takes the other variable's value |
| G2 | Gemma-3-12B | `loss` | lower > 0.10 | ... and loses accuracy on programs ending on it |
| G3 | Gemma-3-12B | `stratum_contrast` | lower > 0 | ... more than on programs ending on the asked variable (the recency default) |
| Q1 | Qwen2.5-7B | `other_rise` | lower > 0.05 | as G1 |
| Q2 | Qwen2.5-7B | `loss` | lower > 0.10 | as G2 |
| Q3 | Qwen2.5-7B | `stratum_contrast` | lower > 0 | as G3 |

## Predictions (recorded before generation; all one-sided)

All six are predicted to be supported.

| ID | Exploratory | Expected interval at 768 groups |
|---|---:|---|
| G1 | +0.585 | about [+0.54, +0.63] |
| G2 | +0.619 | about [+0.57, +0.66] |
| G3 | +0.567 | about [+0.51, +0.63] |
| Q1 | +0.169 | about [+0.13, +0.21] |
| Q2 | +0.235 | about [+0.19, +0.28] |
| Q3 | +0.080 | about [+0.02, +0.14], **the closest call** |

`reader_blind_clause` is descriptive, with no prediction. It tests the direct
route without hiding the answer format. If the read took the variable
straight from the question, this block would raise other-value answers on
the main stratum.

## Execution and limits

- **Freeze first.** This protocol, the seed addition, the module, its tests
  and the launcher are committed and pushed before generation. Tests use
  test-only seeds.
- **Runs.** One run per family, each only on an idle GPU with enough free
  memory.
  - Foreign processes are logged, never acted on.
  - An operational failure with zero results may be relaunched once and
    recorded.
  - A partial run never enters the evaluation.
  - One CPU evaluation over both runs.
- **Budget.** At most 2 GPU-hours: about 3,072 × 7 forwards per family.
- **Limits.**
  - Off-distribution blocks; one wording.
  - `program_blind` hides the whole question from the program. It does not
    say whether the program tokens tag statements or carry the variable
    forward.

**Code:** `src/open_weight_lingua/route_confirm.py`, launched with
`scripts/run_route_confirm.sh`.
