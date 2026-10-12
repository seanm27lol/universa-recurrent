# Phase Nine confirmatory: which statements the late read uses, on freshly generated programs

**Status: FROZEN 2026-10-12 (UTC), before the confirmatory programs are
generated and before any model forward on them. Branch
`phase-nine-span-knockout`, at the commit that adds this file, after the
exploratory report (`7ca8756`). Every hypothesis, threshold and analysis
choice is fixed here; none may change after generation.**

## The question

```text
x = 7              ← the asked variable's line
y = 16             ← the other variable's lines
y = y + 1
y = y - 3          ← ends with the program's last token "3"
What is x? Reply with only the integer.      → 7
```

The exploratory span knockout ([report](../reports/phase_nine_span_knockout.md)),
on reused calibration and pilot prompts, found the same pattern in both
models. Inside each model's confirmed late route:
- blocking the asked variable's own statements is as damaging as blocking
  the whole program;
- blocking the other variable's statements costs nothing;
- when the program ends on the other variable, blocking the last token
  costs nothing.

This study tests those claims once, on programs that do not yet exist.

## Data: a fresh split

Both reserved validation blocks are spent (Phases Seven and Eight), so the
programs are new.

- **Exclusions.** The code rebuilds the frozen Phase Two plan, requires its
  hash to equal `ff060012d35d92c480a51400a02597bea7421a525c44d97043a56dd3f3de667f`,
  and excludes every canonical program in its five splits.
- **Generation.** It then generates **256 new groups** with the unchanged
  generator (`tasks.generate_groups`), under the new split name
  `fresh_2026_10` and seed **206001**. That is the next value in the
  existing series 201001–205001; it is fixed here and has never been used.
- **Prompts.** 1,024 per family: both sides, both questions, original
  format (question after the program).
- **Recorded before any forward.** The run records the inventory hash of
  the generated groups and the stimuli hash before loading the model.

## Settings

As in exploratory Phase Nine:
- **Routes.** Qwen: the answer positions, from layer 21. Gemma: every
  post-program position, from layer 24.
- **Mechanics.** Line assignment, the knockout hook, teacher-forced digit
  accuracy and the bitwise no-op control.
- **Conditions.** Baseline, no-op, `asked_lines_minus_last`,
  `other_lines_minus_last`, `last_token`.

## Confirmatory hypotheses

**Loss** is baseline correctness minus knockout correctness, per prompt.

**Six tests, Bonferroni-controlled.** Each uses a two-sided percentile
interval at coverage 1 − 0.05/6, from 10,000 whole-group bootstrap resamples
(NumPy `default_rng`; seed 803100 Gemma, 803101 Qwen). Groups are resampled
whole.
- "Damaged" needs the interval's lower bound above 0.05.
- "Survives" needs its upper bound below 0.05.

| ID | Model | Blocked span | Prompts | Claim |
|---|---|---|---|---|
| G1 | Gemma-3-12B | the asked variable's lines (minus the last token) | all 1,024 | damaged |
| G2 | Gemma-3-12B | the other variable's lines (minus the last token) | all 1,024 | survives |
| G3 | Gemma-3-12B | the last token | the 512 whose last line assigns the other variable | survives |
| Q1 | Qwen2.5-7B | the asked variable's lines (minus the last token) | all 1,024 | damaged |
| Q2 | Qwen2.5-7B | the other variable's lines (minus the last token) | all 1,024 | survives |
| Q3 | Qwen2.5-7B | the last token | the 512 whose last line assigns the other variable | survives |

Every group contributes two prompts to the G3 and Q3 subset: one per side,
the question about the variable that side does not end on.

## Predictions (recorded before generation)

These are exploratory losses. Expected intervals at 256 groups use the
exploratory group-level spread, under a normal approximation.

| ID | Exploratory loss | Expected interval | Prediction |
|---|---:|---|---|
| G1 | +0.701 | about [+0.65, +0.75] | supported |
| G2 | −0.005 | about [−0.020, +0.011] | supported |
| G3 | +0.001 | about [−0.006, +0.009] | supported |
| Q1 | +0.599 | about [+0.55, +0.65] | supported |
| Q2 | −0.012 | about [−0.027, +0.003] | supported |
| Q3 | −0.012 | about [−0.030, +0.007] | supported |

## Descriptive (not tested)

- Baseline accuracy on the fresh programs.
- Accuracy and loss for each span.
- Breakdowns by the asked variable's last update, by the last writer, and
  by whether the asked variable copies the other.

## Execution

- **Freeze first.** This protocol, the seed addition, the module, its tests
  and the launcher are committed and pushed before generation. Generation
  happens inside the run command, after this freeze.
- **Start.** Gemma, then Qwen, each only on an idle GPU. Foreign processes
  that appear later are logged, never acted on.
- **Failures.** An operational failure with zero results may be relaunched
  once, into a new directory, and recorded. A source bug permits only a
  regression test plus a visibly versioned rerun. Partial runs never enter
  the evaluation.
- **Evaluation.** One CPU pass over both complete runs, reporting all
  outcomes.
- **Budget.** At most 1 GPU-hour. Five forwards per prompt: about 11
  minutes for Gemma and 7 for Qwen, plus setup.

## Limits

- **Knockout limits.** An off-distribution intervention; teacher-forced
  digit accuracy; one route per model, one prompt format, one task family.
- **Mechanism inferred.** That copied values sit at the copy statement by
  the critical layer is an inference, not tested here.
- **Same distribution.** The fresh programs come from the same generator,
  so this tests generalization to new programs of the same distribution,
  not to other tasks.

**Code:** `src/open_weight_lingua/span_knockout_confirm.py`, launched with
`scripts/run_span_knockout_confirm.sh`.
