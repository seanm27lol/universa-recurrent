# Phase Seventeen confirmatory: where the program marks the asked variable, on a seventh fresh split

**Status: FROZEN 2026-10-13 (UTC), before the confirmatory programs are
generated and before any model forward on them. Branch
`phase-seventeen-question-marking`, at the commit that adds this file, after
the Phase Seventeen exploratory report (`b65ec9e`). Every hypothesis and
analysis choice is fixed here.**

## The question

```text
What is x? Reply with only the integer.
x = 7              ← the asked variable's line
y = 16             ← the other variable's lines
y = y + 1
y = y - 3          ← the last line   → 7  (y ends at 14)
```

The exploratory Phase Seventeen
([report](../reports/phase_seventeen_question_marking.md)) studied prompts
with the question first. It blocked parts of the program from the question
and found:
- blocking only the asked variable's lines, or only the other variable's
  lines, each raises other-value answers on programs ending on the other
  variable;
- in Gemma, blocking only the last line does much less than blocking the
  other variable's lines;
- in Qwen, the last line alone does more than half.

This study tests that once, in both models, on new programs.

## Data: a seventh fresh split

- **Exclusions.** The code rebuilds the frozen Phase Two plan and the six
  earlier fresh splits (seeds 206001–211001). It verifies their inventories
  (the five recorded by the Phase Sixteen confirmatory protocol, plus
  `fresh_2026_10f`'s `4758be932923…`) and excludes every canonical program
  from all of them.
- **Generation.** **768 new groups** under split `fresh_2026_10g`, seed
  **212001**. The inventory hash is recorded before any forward. Both
  families receive the same programs.
- **Prompts.** All 3,072 prompts, question first (Phase Seven's
  `original_before`).
- **Units for every claim.** Main-stratum prompts (programs ending on the
  other variable, whose answer does not depend on it) where the other
  variable's final value differs from the answer.

## Settings

Gemma-3-12B and Qwen2.5-7B under their locks. Teacher-forced scoring of the
answer and of the other variable's final value. The bitwise no-op control
runs on every prompt. From layer 0, a subset of program tokens is blocked
from every question token:

| Condition | Program tokens blocked from the question |
|---|---|
| `baseline` | none |
| `asked_blind` | the asked variable's lines |
| `other_blind` | the other variable's lines |
| `last_line_blind` | the program's last line |

## Confirmatory hypotheses

Per prompt, "says other" is 1 when the other variable's final value is the
argmax answer under teacher forcing, and 0 otherwise.
- `asked_rise` = says other under `asked_blind` − under `baseline`;
- `other_rise` = says other under `other_blind` − under `baseline`;
- `last_rise` = says other under `last_line_blind` − under `baseline`;
- `other_minus_last` = says other under `other_blind` − under
  `last_line_blind`.

**Six tests, Bonferroni-controlled.** Each uses a two-sided percentile
interval at coverage 1 − 0.05/6, from 10,000 whole-group bootstrap resamples
(seed 809100 Gemma, 809101 Qwen).

| ID | Family | Measure | Supported iff | Claim |
|---|---|---|---|---|
| G1 | Gemma-3-12B | `asked_rise` | lower > 0.05 | the asked statements carry a mark |
| G2 | Gemma-3-12B | `other_rise` | lower > 0.05 | the other variable's statements carry a mark |
| G3 | Gemma-3-12B | `other_minus_last` | lower > 0.05 | Gemma's mark on the other variable is not confined to its last line |
| Q1 | Qwen2.5-7B | `asked_rise` | lower > 0.05 | as G1 |
| Q2 | Qwen2.5-7B | `other_rise` | lower > 0.05 | as G2 |
| Q3 | Qwen2.5-7B | `last_rise` | lower > 0.05 | in Qwen, the last line alone carries a mark |

## Predictions (recorded before generation; all one-sided)

All six are predicted to be supported.

| ID | Exploratory | Expected interval at 768 groups |
|---|---:|---|
| G1 | +0.357 | about [+0.31, +0.40] |
| G2 | +0.295 | about [+0.25, +0.34] |
| G3 | +0.213 | about [+0.18, +0.25] |
| Q1 | +0.155 | about [+0.12, +0.20] |
| Q2 | +0.131 | about [+0.10, +0.17] |
| Q3 | +0.096 | about [+0.07, +0.13], **the closest call** |

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
- **Budget.** At most 2.5 GPU-hours: about 3,072 × 9 forwards per family.
- **Limits.**
  - Off-distribution blocks; one wording.
  - On these prompts the last line belongs to the other variable, so Q3
    cannot separate "carried to the end" from "the other's latest
    statement is marked".

**Code:** `src/open_weight_lingua/marking_confirm.py`, launched with
`scripts/run_marking_confirm.sh`.
