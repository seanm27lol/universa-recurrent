# Phase Nineteen confirmatory: the question-first two-stage account at Gemma-3-27B, on a ninth fresh split

**Status: FROZEN 2026-10-13 (UTC), before the confirmatory programs are
generated and before any model forward on them. Branch
`phase-nineteen-scale-question-first`, at the commit that adds this file,
after the Phase Nineteen exploratory report (`46a1d49`). Every hypothesis
and analysis choice is fixed here.**

## The question

```text
What is x? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          → 7  (y ends at 14)
```

The exploratory Phase Nineteen
([report](../reports/phase_nineteen_scale_question_first.md)) found that,
with the question first, Gemma-3-27B's program tokens carry the asked
variable to the late read. They read the question in layers 15–30 of 62:
- blocking all of them from the question makes 80% of answers the other
  variable's value;
- blocking only from layer 31 on, or only below layer 7, is harmless;
- blocking from layer 23 on, or below layer 23, is not.

This study tests that once, on new programs.

## Data: a ninth fresh split

- **Exclusions.** The code rebuilds the frozen Phase Two plan and the eight
  earlier fresh splits (seeds 206001–213001). It verifies their inventories
  (the seven recorded by the Phase Eighteen confirmatory protocol, plus
  `fresh_2026_10h`'s `fb17767b3304…`) and excludes every canonical program
  from all of them.
- **Generation.** **768 new groups** under split `fresh_2026_10i`, seed
  **214001**. The inventory hash is recorded before any forward.
- **Prompts.** The question-first prompts where the mark decides the
  answer, by Phase Eighteen's rule:
  - programs ending on the other variable;
  - with an answer that does not depend on it;
  - with an other value distinct from the answer.

## Settings

Gemma-3-27B under its lock (`ca078e52`). Teacher-forced scoring of the
answer and of the other variable's final value. The bitwise no-op control
runs on every prompt. Every program token is blocked from every question
token in these layers:

| Condition | Blocked layers |
|---|---|
| `program_blind` | 0–61 |
| `from_31` | 31–61 |
| `from_23` | 23–61 |
| `until_23` | 0–22 |
| `until_7` | 0–6 |

## Confirmatory hypotheses

Per prompt, the measure is the **rise** in saying the other variable's final
value: 1 or 0 under the condition, minus 1 or 0 under `baseline`.

**Five tests, Bonferroni-controlled.** Each uses a two-sided percentile
interval at coverage 1 − 0.05/5, from 10,000 whole-group bootstrap resamples
(seed 811100).

| ID | Condition | Supported iff | Claim |
|---|---|---|---|
| S1 | `program_blind` | lower > 0.30 | at 27B, the program carries the asked variable to the read |
| S2 | `from_31` | upper < 0.05 | the marking is finished by layer 31 of 62 (half the depth) |
| S3 | `from_23` | lower > 0.05 | ... but not by layer 23 |
| S4 | `until_23` | lower > 0.05 | it has begun below layer 23 |
| S5 | `until_7` | upper < 0.05 | the first seven layers do not mark |

## Predictions (recorded before generation; all one-sided)

All five are predicted to be supported.

| ID | Exploratory | Expected interval at 768 groups |
|---|---:|---|
| S1 | +0.751 | about [+0.71, +0.79] |
| S2 | −0.001 | about [−0.004, 0.000] |
| S3 | +0.082 | about [+0.058, +0.109], **the closest call** |
| S4 | +0.674 | about [+0.63, +0.71] |
| S5 | +0.004 | about [+0.001, +0.009] |

## Execution and limits

- **Freeze first.** This protocol, the seed addition, the module, its tests
  and the launcher are committed and pushed before generation. Tests use
  test-only seeds.
- **Run.** One Gemma-3-27B run, only on an idle GPU with at least 75 GiB
  available.
  - Foreign processes are logged, never acted on.
  - An operational failure with zero results may be relaunched once and
    recorded.
  - A partial run never enters the evaluation.
  - One CPU evaluation.
- **Budget.** At most 2.5 GPU-hours: about 1,400 prompts × 13 forwards at
  about 0.34 s.
- **Limits.**
  - One model.
  - Grid resolution only.
  - Off-distribution blocks; one wording.

**Code:** `src/open_weight_lingua/scale_marking_confirm.py`, launched with
`scripts/run_scale_marking_confirm.sh`.
