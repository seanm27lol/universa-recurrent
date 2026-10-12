# Phase Fifteen: are question-first errors the late read choosing the wrong statements? (exploratory)

**Status: FROZEN 2026-10-12 (UTC), before any Phase Fifteen model forward.
Branch `phase-fourteen-question-order`, after the Phase Fourteen report
(`02a75ca`). Exploratory, on reused calibration and pilot prompts. A
confirmatory test would use a fresh split under its own frozen protocol.**

## The question

```text
What is x? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          → 7  (y ends at 14)
```

**What is known.**
- **Phases Six and Seven.** Asking the question before the program costs
  accuracy (Gemma-3-12B 0.944 → 0.847, Qwen2.5-7B 0.810 → 0.508). The loss
  concentrates on programs that end on the *other* variable, and many wrong
  answers are that variable's value.
- **Phase Fourteen.** The question order does not change what the late
  read uses. And with the question first, letting the post-program
  positions see only a literal last line of the asked variable raises
  Qwen's accuracy from 0.587 to 0.971.

**Hypothesis.** The question-first errors come from the late read taking
the wrong statements: the other variable's, or the asked variable's
superseded ones. If so, hiding those statements from the post-program
positions should restore accuracy. If the errors arise earlier, for
instance in how the program tokens themselves are processed when the
question precedes them, hiding statements from the read should not help.

## Design

- **Models.** Gemma-3-12B and Qwen2.5-7B, locks hash-verified.
- **Prompts.** Phase Eight's 1,536 calibration and pilot prompts per
  family, each in both orders (`after`, and `first` = Phase Seven's
  `original_before`), as in Phase Fourteen: 3,072 per family.
- **Conditions.** All from layer 0, each blocking every post-program
  position from a set of program tokens:
  - `baseline`;
  - `block_other`: every token of the other variable's lines;
  - `block_superseded`: the asked variable's lines before its final
    assignment. Only for prompts where that final assignment is a literal
    and an earlier assignment exists (280 per order).
  - `block_both`: both, on the same prompts.
- **Scoring.** Teacher-forced, for three values, each when distinct from
  the answer:
  - the answer;
  - the other variable's final value;
  - the asked variable's value before its final assignment.

  The bitwise no-op control runs on every prompt.
- **Main stratum.** Programs whose last line assigns the other variable,
  excluding the 12 prompts per order whose answer depends on a copy of the
  other variable: 758 per order.

## Predictions (recorded before any forward)

1. The no-op control is exact.
2. **Question after: hiding statements costs nothing.**
   - `block_other` costs at most 3 points on prompts whose answer does not
     depend on the other variable.
   - `block_superseded` costs at most 3 points on its prompts.
3. **Question first, main stratum: hiding the other variable's lines
   helps.**
   - `block_other` raises accuracy by at least 10 points in each model.
   - It at least halves the rate at which the other variable's final value
     is said.
4. **Question first, main stratum: most of the order effect is
   recovered.** Under `block_other`, accuracy closes at least half of the
   gap between the question-first and question-after baselines, in each
   model.
5. **Question first, superseded-literal prompts.** Under `block_both`,
   accuracy is within 5 points of the question-after baseline on the same
   prompts, in each model.

If 3 and 4 fail, the question-first errors are not a choice made by the
late read. All conditions, orders and strata are reported, including
`block_superseded` alone and the previous-value rates.

## Budget, honesty, limits

- **Budget.** At most 2.5 GPU-hours: about 21,600 forwards per family,
  about 55 minutes for Gemma and 35 for Qwen.
  - Each family starts only on an idle GPU with enough free memory.
  - Foreign processes are logged, never acted on.
  - An operational failure with zero results may be relaunched once and
    recorded.
- **Limits.**
  - Exploratory, on opened prompts.
  - Hiding statements from every layer is an off-distribution intervention.
    It shows what the read *can* do without them, not what it does
    normally.
  - One wording.

**Code:** `src/open_weight_lingua/variable_choice_knockout.py`, launched with
`scripts/run_variable_choice_knockout.sh`.
