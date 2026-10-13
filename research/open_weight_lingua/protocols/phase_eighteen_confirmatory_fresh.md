# Phase Eighteen confirmatory: the layers where the program reads the question, on an eighth fresh split

**Status: FROZEN 2026-10-13 (UTC), before the confirmatory programs are
generated and before any model forward on them. Branch
`phase-eighteen-marking-layers`, at the commit that adds this file, after
the Phase Eighteen exploratory report (`142a2bc`). Every hypothesis and
analysis choice is fixed here.**

## The question

```text
What is x? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          → 7  (y ends at 14)
```

The exploratory Phase Eighteen
([report](../reports/phase_eighteen_marking_layers.md)) studied prompts with
the question first. It found that the program tokens read the question in
the middle layers, before the late read. When that attention is blocked:
- from layer 24 on (Gemma, of 48) or from 17 on (Qwen, of 28), nothing
  changes;
- from 18 or 14 on, or below layer 12 or 10, the late read takes the other
  variable's value more often.

This study tests those window edges once, in both models, on new programs.

## Data: an eighth fresh split

- **Exclusions.** The code rebuilds the frozen Phase Two plan and the seven
  earlier fresh splits (seeds 206001–212001). It verifies their inventories
  (the six recorded by the Phase Seventeen confirmatory protocol, plus
  `fresh_2026_10g`'s `6ed1d8b3ffbd…`) and excludes every canonical program
  from all of them.
- **Generation.** **768 new groups** under split `fresh_2026_10h`, seed
  **213001**. The inventory hash is recorded before any forward. Both
  families receive the same programs.
- **Prompts.** The question-first prompts (Phase Seven's `original_before`)
  where the mark decides the answer, by Phase Eighteen's rule:
  - programs ending on the other variable;
  - with an answer that does not depend on it;
  - with an other value distinct from the answer.

## Settings

Gemma-3-12B and Qwen2.5-7B under their locks. Teacher-forced scoring of the
answer and of the other variable's final value. The bitwise no-op control
runs on every prompt. Every program token is blocked from every question
token in these layers:

| Family | Condition | Blocked layers |
|---|---|---|
| Gemma-3-12B | `from_24` | 24–47 |
| Gemma-3-12B | `from_18` | 18–47 |
| Gemma-3-12B | `until_12` | 0–11 |
| Qwen2.5-7B | `from_17` | 17–27 |
| Qwen2.5-7B | `from_14` | 14–27 |
| Qwen2.5-7B | `until_10` | 0–9 |

## Confirmatory hypotheses

Per prompt, the measure is the **rise** in saying the other variable's final
value: 1 or 0 under the condition, minus 1 or 0 under `baseline`.

**Six tests, Bonferroni-controlled.** Each uses a two-sided percentile
interval at coverage 1 − 0.05/6, from 10,000 whole-group bootstrap resamples
(seed 810100 Gemma, 810101 Qwen).

| ID | Family | Condition | Supported iff | Claim |
|---|---|---|---|---|
| G1 | Gemma-3-12B | `from_24` | upper < 0.05 | the marking is finished by layer 24 |
| G2 | Gemma-3-12B | `from_18` | lower > 0.05 | ... but not by layer 18 |
| G3 | Gemma-3-12B | `until_12` | lower > 0.05 | it has begun below layer 12 |
| Q1 | Qwen2.5-7B | `from_17` | upper < 0.05 | the marking is finished by layer 17 |
| Q2 | Qwen2.5-7B | `from_14` | lower > 0.05 | ... but not by layer 14 |
| Q3 | Qwen2.5-7B | `until_10` | lower > 0.05 | it has begun below layer 10 |

## Predictions (recorded before generation; all one-sided)

All six are predicted to be supported.

| ID | Exploratory | Expected interval at 768 groups |
|---|---:|---|
| G1 | +0.014 | about [+0.003, +0.025] |
| G2 | +0.092 | about [+0.065, +0.118], **the closest call** |
| G3 | +0.269 | about [+0.229, +0.309] |
| Q1 | −0.011 | about [−0.023, −0.001] |
| Q2 | +0.100 | about [+0.073, +0.129] |
| Q3 | +0.118 | about [+0.082, +0.156] |

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
- **Budget.** At most 1.5 GPU-hours: about 1,440 prompts × 9 forwards per
  family.
- **Limits.**
  - Grid resolution only: the edges are confirmed at the tested layers,
    not to the single layer.
  - Off-distribution blocks; one wording.

**Code:** `src/open_weight_lingua/layers_confirm.py`, launched with
`scripts/run_layers_confirm.sh`.
