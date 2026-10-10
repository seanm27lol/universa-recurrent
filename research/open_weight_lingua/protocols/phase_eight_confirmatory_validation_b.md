# Phase Eight confirmatory: answer routing on validation_b (attention knockout)

**Status: FROZEN 2026-10-06 (UTC), before `validation_b` is opened and
before any model forward on it. Branch `phase-eight-knockout`, at the commit
that adds this file, after the exploratory report (`f50560e`). Every
hypothesis, threshold and analysis choice is fixed here, from the
exploratory results; none may change after the split is opened.
`validation_b` is the last reserved block; this study uses it once.**

## The question

```text
x = 7
y = 16
y = y + 1
y = y - 3
What is x? Reply with only the integer.      → 7
```

The exploratory knockout ([report](../reports/phase_eight_knockout.md)), on
reused calibration and pilot prompts, found that the two models consult the
program late, by different routes.
- **Qwen2.5-7B:** the answer position reads the program itself, in layers
  21–23 of 28.
- **Gemma-3-12B:** the answer positions never need the program directly.
  The question and template tokens read it in layers 24–29 of 48.

Every release layer chosen on calibration replicated exactly on pilot.
This study tests those claims once, on prompts no analysis has seen.

## Data and settings

- **Split.** `validation_b`: 256 groups. The code rebuilds the frozen Phase
  Two split plan and requires its hash to equal
  `ff060012d35d92c480a51400a02597bea7421a525c44d97043a56dd3f3de667f`.
- **Prompts.** Every group's two programs, each asked about x and about y,
  in the original format (question after the program): 1,024 per family.
  Each is tokenized with the family's chat template and generation prompt.
- **Exactly as in the exploratory run:**
  - the program span is located by decoding, and a span that does not
    decode exactly to the program fails the run;
  - teacher-forced digit accuracy;
  - the per-layer additive-mask knockout in all heads;
  - the bitwise no-op control on every prompt;
  - the pinned 128-token bucket, no KV cache, BF16, eager attention, TF32
    off.
- **Conditions.** Baseline, no-op, and only the knockouts the hypotheses
  need:
  - Gemma: `after_program` from 24 and from 30; `answer` from 0.
  - Qwen: `answer` from 21 and from 24; `after_program` from 24.

## Confirmatory hypotheses

For each prompt, **loss** is baseline correctness minus knockout
correctness (−1, 0 or +1). The **mean loss** is the drop in digit accuracy
caused by the knockout.

**Family of six tests, Bonferroni-controlled.** Each has a two-sided
percentile interval at coverage 1 − 0.05/6, from 10,000 whole-group
bootstrap resamples (NumPy `default_rng`; seed 802100 Gemma, 802101 Qwen).
Each resample draws 256 groups with replacement, keeping a group's four
prompts together.

| ID | Model | Knockout | Claim | Supported iff |
|---|---|---|---|---|
| G1 | Gemma-3-12B | every position after the program, from layer 30 | the answer survives | interval upper bound < 0.05 |
| G2 | Gemma-3-12B | every position after the program, from layer 24 | the answer is damaged | interval lower bound > 0.05 |
| G3 | Gemma-3-12B | answer positions only, from layer 0 | the answer positions never need the program | interval upper bound < 0.05 |
| Q1 | Qwen2.5-7B | answer positions only, from layer 24 | the answer survives | interval upper bound < 0.05 |
| Q2 | Qwen2.5-7B | answer positions only, from layer 21 | the answer is damaged | interval lower bound > 0.05 |
| Q3 | Qwen2.5-7B | every position after the program, from layer 24 | the answer survives | interval upper bound < 0.05 |

## Predictions (recorded before opening the split)

These are the exploratory mean losses. Expected intervals at 256 groups use
the exploratory group-level spread, under a normal approximation.

| ID | Exploratory loss | Expected interval | Prediction |
|---|---:|---|---|
| G1 | +0.014 | about [−0.001, +0.030] | supported |
| G2 | +0.743 | about [+0.69, +0.79] | supported |
| G3 | +0.012 | about [−0.005, +0.028] | supported |
| Q1 | +0.001 | about [−0.006, +0.009] | supported |
| Q2 | +0.585 | about [+0.53, +0.64] | supported |
| Q3 | +0.000 | about [−0.007, +0.007] | supported |

## Descriptive (not tested)

- Baseline digit accuracy and accuracy under each knockout.
- Breakdowns by the asked variable's last update (literal / arithmetic /
  copy) and by whether the last line assigns the asked variable.

## Execution

- **Freeze first.** This protocol, `answer_knockout_confirm.py`, its tests
  and launcher are committed and pushed before `validation_b` is built. The
  run command builds the plan, tokenizes and locates the spans, then records
  the stimuli hash and the protocol and source hashes in its manifest. Only
  after that does it load the model.
- **Start.** Gemma, then Qwen, each only when the GPU has no other
  workload. Foreign GPU processes that appear later are logged, never acted
  on.
- **Failures.** An operational failure with zero results may be relaunched
  once, into a new directory, and recorded. A source bug permits only a
  regression test and a visibly versioned rerun. Partial runs never enter
  the evaluation.
- **Evaluation.** One CPU evaluation over both complete runs, reporting all
  outcomes.
- **Budget.** At most 1.5 GPU-hours. Expected about 6 forwards per prompt:
  roughly 16 minutes for Gemma and 9 for Qwen, plus setup.

## Limits

- **Knockout is an off-distribution intervention.** The claims concern this
  intervention's effect on teacher-forced digit accuracy.
- **The blocked span is the whole program.** These tests do not say which
  program tokens are read: raw statements or something stored at them.
- **One prompt format and one task family.** After this, no reserved split
  remains. Further confirmatory claims would need newly generated programs
  under a new, frozen plan.

**Code:** `src/open_weight_lingua/answer_knockout_confirm.py`, launched with
`scripts/run_answer_knockout_confirm.sh`.
