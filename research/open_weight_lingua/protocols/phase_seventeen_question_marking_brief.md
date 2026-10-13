# Phase Seventeen: which program tokens carry the asked variable when the question comes first? (exploratory)

**Status: FROZEN 2026-10-13 (UTC), before any Phase Seventeen model forward.
Branch `phase-seventeen-question-marking`, based on `main` at `fa29713`.
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

Phase Sixteen confirmed that, with the question first, the asked variable
reaches the late read through the program tokens. When every program token
is blocked from the question, Gemma's read falls back to the most recent
assignment of either variable. Qwen's variable is confused too, but less
specifically. Which program tokens matter? Three
hypotheses make different predictions:
- **Tagging the asked statements.** The asked variable's lines, read with
  the question in view, are marked as relevant.
- **Suppressing the other's.** The other variable's lines are marked as
  irrelevant.
- **Carrying to the end.** The program's last line holds which variable
  was asked, for the read to use.

## Design

- **Models.** Gemma-3-12B and Qwen2.5-7B, locks hash-verified.
- **Prompts.** Phase Eight's 1,536 calibration and pilot prompts per
  family, question first (Phase Seven's `original_before`).
- **Conditions.** From layer 0, a subset of program tokens is blocked from
  every question token:
  - `program_blind`: every program token. This is Phase Sixteen's
    condition, run again here as the reference.
  - `asked_blind`: the asked variable's lines.
  - `other_blind`: the other variable's lines.
  - `last_line_blind`: the program's last line.
- **Scoring.** Teacher-forced, for the answer and for the other variable's
  final value. The bitwise no-op control runs on every prompt.
- **Strata.**
  - Main: programs ending on the other variable, whose answer does not
    depend on it. There, the last line belongs to the other variable.
  - Asked last: programs ending on the asked variable.

The effect of a block is its rise in other-value answers on the main
stratum, as a share of `program_blind`'s rise.

## Predictions (recorded before any forward; all one-sided)

The prediction follows the tagging hypothesis.

1. The no-op control is exact.
2. `program_blind` reproduces Phase Sixteen's effect: on the main stratum,
   other-value answers rise by at least 30 points (Gemma) and 8 points
   (Qwen).
3. **Tagging.** `asked_blind` produces at least half of `program_blind`'s
   rise, in each model.
4. `other_blind` produces less than half of it, in each model.
5. `last_line_blind` produces less than a quarter of it, in each model.

If 4 fails and 3 holds, both kinds of marking act. If 3 fails and 4 holds,
the model suppresses the other variable instead. If 5 fails, the variable
is carried to the end. Every condition and stratum is reported whatever
happens.

## Budget, honesty, limits

- **Budget.** At most 2 GPU-hours: about 16,500 forwards per family. Same
  gate and rules as before.
- **Limits.**
  - Exploratory.
  - Off-distribution blocks; one wording.
  - On the main stratum, `other_blind` includes the last line, so
    predictions 4 and 5 overlap there. The asked-last stratum separates
    them.

**Code:** `src/open_weight_lingua/question_marking_knockout.py`, launched
with `scripts/run_question_marking_knockout.sh`.
