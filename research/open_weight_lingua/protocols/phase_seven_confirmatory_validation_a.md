# Phase Seven confirmatory: question position × wording on validation_a

**Status: FROZEN 2026-10-05 (UTC), before `validation_a` is opened and
before any model forward on it. Branch `phase-seven-confirmatory`, based on
`main` at `1d1aa64`. Every hypothesis, threshold and analysis choice below
is fixed here, from the exploratory Phase Seven results; none may change
after the split is opened.**

## The question

```text
x = 7
y = 16
y = y + 1
y = y - 3          x is 7; the program ends on y (14)

original wording:  What is x? Reply with only the integer.
expanded wording:  What is x at the end of this program? Reply with only the integer.
```

The exploratory Phase Seven factorial, on reused calibration and pilot
programs, separated two causes of Phase Six's accuracy loss
([report](../reports/phase_seven_prompt_factorial.md)).
- Qwen2.5-7B's loss came from placing the question before the program
  (about −0.30), with no resolved wording effect.
- Gemma-3-12B's came from position (−0.097) plus a wording × position
  interaction (−0.105): "at the end of this program" hurt only when it came
  first.

This study tests those specific claims once, on data no analysis has seen:
the reserved `validation_a` split. `validation_b` stays reserved.

## Data

- **Split.** `validation_a`: 256 groups from the frozen Phase Two split
  plan. The code rebuilds the plan and requires its hash to equal
  `ff060012d35d92c480a51400a02597bea7421a525c44d97043a56dd3f3de667f`, the
  value both closed calibration manifests record. The programs are
  therefore the pre-planned ones, disjoint from smoke, calibration and
  pilot.
- **Base prompts.** Every group's two programs (sides A and B), each asked
  about x and about y: 1,024 base prompts per family. Each is rendered in
  four formats with Phase Seven's exact strings (`render_prompt`), which
  makes 4,096 answers per family.

| Format | User message |
|---|---|
| `original_after` | program, newline, `What is {v}? Reply with only the integer.` |
| `original_before` | that question, newline, program |
| `expanded_after` | program, newline, `What is {v} at the end of this program? Reply with only the integer.` |
| `expanded_before` | that question, newline, program |

- **Fixed settings, identical across formats and to exploratory Phase
  Seven.**
  - Models: Gemma-3-12B-it and Qwen2.5-7B-Instruct under their existing,
    hash-verified locks.
  - Tokenization: each family's chat template, with one user message and
    the generation prompt.
  - Decoding: greedy, at most 8 new tokens, the pinned 128-token bucket, no
    KV cache, TF32 off.
  - Stopping: at any end-of-sequence token the model declares
    (`Target.greedy` with the multi-EOS fix).
  - Scoring: each family's frozen convention through `answer_text_matches`,
    `rstrip` for Gemma and `raw` for Qwen.
  - No retries, no repairs, no exclusions. A prompt that does not fit the
    bucket fails the run.

## Confirmatory hypotheses

`A(w, p)` is exact-answer accuracy for wording `w` and position `p`, over
the 1,024 base prompts of one model. The contrasts are paired on the same
prompts:

- `position_original = A(original, before) − A(original, after)`
- `wording_before = A(expanded, before) − A(original, before)`
- `interaction = [A(expanded, before) − A(expanded, after)] − [A(original, before) − A(original, after)]`

**Family of six tests, Bonferroni-controlled.** Each has a two-sided
percentile interval at coverage 1 − 0.05/6 (quantiles 0.004167 and
0.995833). The intervals come from 10,000 whole-group bootstrap resamples:
NumPy `default_rng`, seed 801100 for Gemma and 801101 for Qwen. Each
resample draws 256 groups with replacement, keeping a group's four base
prompts and four formats together.

| ID | Model | Contrast | Claim | Supported iff |
|---|---|---|---|---|
| G1 | Gemma-3-12B | position_original | Placing the question first lowers accuracy | interval entirely below 0 |
| G2 | Gemma-3-12B | wording_before | With the question first, the expanded wording lowers accuracy | interval entirely below 0 |
| G3 | Gemma-3-12B | interaction | The expanded wording's harm depends on position (more harm before) | interval entirely below 0 |
| Q1 | Qwen2.5-7B | position_original | Placing the question first lowers accuracy | interval entirely below 0 |
| Q2 | Qwen2.5-7B | wording_before | With the question first, wording changes accuracy by less than 5 points | interval entirely inside (−0.05, +0.05) |
| Q3 | Qwen2.5-7B | interaction | The position effect differs between wordings by less than 5 points | interval entirely inside (−0.05, +0.05) |

For G1–G3 and Q1 the report also states whether the interval lies entirely
beyond −0.05. That is the 5-point practical reference used since Phase Two;
it is reported but does not change the verdict.

## Predictions (recorded before opening the split)

These come from the exploratory estimates and the precision expected at 256
groups. The precision was computed from the exploratory group-level spread,
under a normal approximation.

| ID | Exploratory estimate | Expected interval at 256 groups | Prediction |
|---|---|---|---|
| G1 | −0.097 | about ±0.038 | supported, likely beyond −0.05 |
| G2 | −0.118 | about ±0.042 | supported, beyond −0.05 |
| G3 | −0.105 | about ±0.046 | supported, likely beyond −0.05 |
| Q1 | −0.301 | about ±0.049 | supported, beyond −0.05 |
| Q2 | −0.007 | about ±0.042 | **borderline**: even if the exploratory value is true, the interval reaches about −0.049; roughly an even chance of establishing equivalence |
| Q3 | +0.003 | about ±0.045 | **borderline**, for the same reason |

A failed Q2 or Q3 would mean "equivalence not established at this sample
size". It would not mean that Qwen has a wording effect, unless its
interval excludes 0.

## Secondary analyses (pre-specified, not in the family)

These are nominal 95% intervals from the same bootstrap, labelled
secondary.
- **The other contrasts for both models:** `position_expanded`,
  `wording_after`, and Gemma's Q2/Q3-type and Qwen's G2/G3-type contrasts.
- **Last-line concentration.** For each model and wording: the position
  effect on prompts whose last line assigns the *other* variable, minus the
  position effect on prompts whose last line assigns the asked variable.
  Each group contributes two prompts to each side. Exploratory results
  predict a negative value.
- **Other-value answers.** How often the answer equals the other variable's
  final value (when that differs from the asked one), by format and last
  writer.

**Breakdowns (descriptive, with denominators).** For each model and format,
accuracy by:
- requested variable (`x`, `y`);
- whether the last line assigns the asked variable;
- the asked variable's last update (`assign`, `add`, `subtract`, `copy`,
  and as literal / arithmetic / copy);
- the joint table of requested variable × last writer;
- termination.

## Execution

- **Freeze first.** This protocol, `prompt_factorial_confirm.py`, its tests
  and launcher are committed and pushed before `validation_a` is built.
  Building the plan, rendering and tokenizing happen inside the run
  command, which records the stimuli hash and the protocol and source hashes
  in its manifest *before* loading the model.
- **Start condition.** Each family starts only when the GPU has no other
  workload: no compute process except the rf-moe dashboard, and
  utilization under 20% on three checks 30 s apart. Foreign GPU processes
  that appear during a run are logged and do not stop it; greedy outputs do
  not depend on them.
- **Order.** Gemma, then Qwen, one run each. Each run writes every answer to
  `generations.jsonl` as it is produced, and writes a `completion.json` with
  status, full traceback on failure, and timings.
- **Failure.** A source bug permits only a regression test plus a visibly
  versioned rerun. An operational failure that produced zero answers may be
  relaunched once into a new directory; that is recorded. Partial runs never
  enter the confirmatory analysis.
- **Budget.** At most 2 GPU-hours. Expected: about 29 minutes of generation
  for Gemma (0.425 s per answer) and 15 for Qwen (0.218 s), plus a few
  minutes of setup each.
- **Evaluation.** `evaluate` runs on CPU over both complete runs. It writes
  every cell, contrast, interval and verdict to `summary.json`. All outcomes
  are reported, including unsupported and inconclusive ones.

## Limits

- **Behaviour only.** These contrasts concern answer accuracy under these
  formats. They say nothing about where or whether the models store
  variable values.
- **One pair of wordings, two models, one task family** (integers 0–19,
  short straight-line programs).
- `validation_a` is used once, here. A second confirmatory question would
  need `validation_b` and its own frozen protocol.

**Code:** `src/open_weight_lingua/prompt_factorial_confirm.py`, launched with
`scripts/run_prompt_factorial_confirm.sh`.
