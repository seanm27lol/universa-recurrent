# Phase Sixteen: with the question first, how does the asked variable reach the late read? (exploratory)

**Status: FROZEN 2026-10-12 (UTC), before any Phase Sixteen model forward.
Branch `phase-sixteen-question-route`, based on `main` at `d8a9f58`.
Exploratory, on reused calibration and pilot prompts, question-first order
only.**

## The question

```text
What is x? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          → 7  (y ends at 14)
```

With the question first, the late read happens at the chat-template
positions after the program (Phase Fourteen). It chooses the asked
variable's statements, but less reliably than with the question after
(Phase Fifteen). How does it learn which variable was asked? Two routes are
possible:
- **Direct.** The post-program positions (the "reader") attend to the
  question tokens.
- **Through the program.** The program tokens come after the question and
  can attend to it. They could mark the asked variable's statements as
  they are read.

## Design

- **Models.** Gemma-3-12B and Qwen2.5-7B, locks hash-verified.
- **Prompts.** Phase Eight's 1,536 calibration and pilot prompts per
  family, in the question-first order (Phase Seven's `original_before`).
- **Spans.** The question span (`What is x? Reply with only the integer.`)
  and its variable-name token are located by decoding.
- **Conditions.** Attention is blocked from layer 0, at every layer:
  - `baseline`;
  - `reader_blind`: post-program positions → the question;
  - `program_blind`: program tokens → the question;
  - `both_blind`: both;
  - the same three with only the variable-name token blocked
    (`*_name`). These are descriptive, because later question tokens can
    already carry the name.
- **Scoring.** Teacher-forced, for the answer and for the other variable's
  final value. The bitwise no-op control runs on every prompt.
- **Strata.**
  - Main: programs ending on the other variable, whose answer does not
    pass through a copy of it.
  - Programs ending on the asked variable.

## Predictions (recorded before any forward; all one-sided)

The account's reading is that the read takes the variable directly from the
question. If it cannot see the question, it falls back to the most recent
assignment of either variable.

1. The no-op control is exact.
2. **Direct route.** On the main stratum, `reader_blind` lowers accuracy by
   at least 30 points, and raises the rate of saying the other variable's
   value by at least 20 points.
3. **No marking through the program.** `program_blind` costs at most 5
   points on all prompts.
4. `both_blind` is at most 5 points below `reader_blind` on all prompts.
5. **Recency default.** Under `reader_blind`, accuracy on programs ending
   on the asked variable exceeds accuracy on the main stratum by at least
   30 points.

If 3 fails, the program tokens carry the asked variable to the read.

## Budget, honesty, limits

- **Budget.** At most 2.5 GPU-hours: about 22,500 forwards per family,
  about 1 hour for Gemma and 35 minutes for Qwen. Same gate and rules as
  before.
- **Limits.**
  - Exploratory.
  - The `reader_blind` condition also hides the answer-format instruction.
    Prediction 5 separates a lost variable from a lost format: a format
    loss would hurt both strata alike.
  - Off-distribution blocks; one wording.

**Code:** `src/open_weight_lingua/question_route_knockout.py`, launched with
`scripts/run_question_route_knockout.sh`.
