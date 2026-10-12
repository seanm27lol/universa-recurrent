# Phase Fourteen: does knowing the question first change the late read? (exploratory)

**Status: FROZEN 2026-10-12 (UTC), before any Phase Fourteen model forward.
Branch `phase-fourteen-question-order`, based on `phase-thirteen-scale` at
`fac33db`. Exploratory, on reused calibration and pilot prompts. A
confirmatory test would use a fresh split under its own frozen protocol.**

## The question

```text
question after:  x = 2 / y = 3 / x = x + 9 / y = 5 / What is x? Reply with only the integer.   → 11
question first:  What is x? Reply with only the integer. / x = 2 / y = 3 / x = x + 9 / y = 5   → 11
```

Phases Eight to Thirteen established and confirmed an account for programs
followed by the question:
- After reading the question, the model looks back at the program late.
- It reads the asked variable's own statements.
- It takes the most recent intact assignment.

Phase Nine also showed two things about the read:
- **Literal values** are read from the final assignment alone.
- **Computed values** (`x = x + 9`) also need the earlier lines.

When the question comes after the program, the model cannot know which
variable to follow while reading it. When the question comes first (Phase
Seven's `original_before` format), it can. Two things could then change:
1. **(i) The value is computed at the statement.** The asked variable's
   final statement holds its computed value, so the late read no longer
   needs the earlier lines.
2. **(ii) An end-of-program summary.** The program's last line holds the
   asked variable's value even when it assigns the other variable.

Or the late read could stay as it is. This phase tests that.

## Design

- **Models.** Gemma-3-12B and Qwen2.5-7B, locks hash-verified.
- **Prompts.** Phase Eight's calibration and pilot prompts (1,536 per
  family), each in two orders, giving 3,072 per family:
  - `after`: the original prompt; its recorded input ids are checked
    against the chat template.
  - `first`: the same program and question, rendered by Phase Seven's
    `original_before`.
- **Scoring.** Teacher-forced digit accuracy, as in Phase Eight.
- **Conditions, per prompt and order.** The bitwise no-op control, plus:
  - `baseline`;
  - `after_program@k` for k on Phase Eight's grid (floor(j·L/8)): every
    post-program position is blocked from every program token from layer
    k. `after_program@0` is the floor.
  - `only_last_line`: from layer 0, every post-program position is blocked
    from every program token before the program's last line. The last
    line's own tokens are untouched and still see the whole program.
- **Strata.** Whether the last line assigns the asked variable, and if it
  does, the kind of that final update: literal, arithmetic or copy.
- **Release layer.** As in Phase Eight: chosen on calibration, checked on
  pilot, per order.

## Predictions (recorded before any forward)

The account predicts that the late read is unchanged by the question's
position.

1. The no-op control is exact.
2. **Reproduction.** In the `after` order, `baseline` and every
   `after_program@k` cell reproduce Phase Eight's recorded per-prompt
   correctness.
3. **Literal final.** When the last line assigns the asked variable a
   literal, `only_last_line` stays within 10 points of baseline in both
   orders.
4. **(i) No value at the statement.** When the last line assigns the asked
   variable arithmetically, `only_last_line` costs at least 30 points in
   both orders.
5. **(ii) No summary.** When the last line assigns the other variable,
   `only_last_line` is within 10 points of the floor in both orders.
6. **Still late.** In the `first` order, the release layer of
   `after_program` is in the second half of the network (k ≥ L/2), as in
   the `after` order (Gemma 30 of 48, Qwen 24 of 28).

If 4 or 5 fails only in the `first` order, knowing the question first
changes the read. Accuracies for every stratum, order and condition are
reported whatever happens.

## Budget, honesty, limits

- **Budget.** At most 3 GPU-hours: 33,792 forwards per family, about
  1.5 hours for Gemma-12B and 0.9 hours for Qwen.
  - Each family starts only on an idle GPU with enough free memory.
  - Foreign processes are logged, never acted on.
  - An operational failure with zero results may be relaunched once and
    recorded.
- **Limits.**
  - Exploratory, on opened prompts.
  - One wording.
  - Knockout limits apply as before: an off-distribution intervention and
    teacher-forced scoring.
  - "Holds the value" means the post-program positions can read the answer
    from those tokens. It does not say how it is represented.

**Code:** `src/open_weight_lingua/question_order_knockout.py`, launched with
`scripts/run_question_order_knockout.sh`.
