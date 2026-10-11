# Phase Nine: which program tokens does the late read use? (partial-span knockout, exploratory)

**Status: FROZEN 2026-10-11 (UTC), before any Phase Nine model forward.
Branch `phase-nine-span-knockout`, based on `main` at `16600bf`. This is an
exploratory study on reused calibration and pilot prompts. Both reserved
validation blocks are already spent, so any confirmatory test will need
newly generated programs under a new, separately frozen split plan.**

## The question

```text
x = 7              ← line 1 (asked variable)
y = 16             ← line 2
y = y + 1          ← line 3
y = y - 3          ← line 4, ends with the program's last token "3"
What is x? Reply with only the integer.      → 7
```

Phase Eight confirmed that both models must look back at the program tokens
after reading the question
([report](../reports/phase_eight_confirmatory.md)).
- **Qwen2.5-7B:** the answer positions must do it in layers 21–23.
- **Gemma-3-12B:** the positions after the program must do it in layers
  24–29.

That knockout blocked the *whole* program, so it could not say what is
read:
- **raw statements:** the asked variable's own lines, distributed across
  the program;
- **an end-of-program summary:** something stored at the last program token
  while the program was processed.

Phase Nine blocks only parts of the program, inside each model's confirmed
critical route, and asks which part the answer needs.

## Design

- **Models, prompts and scoring.** As in Phase Eight:
  - Gemma-3-12B and Qwen2.5-7B;
  - the 1,536 calibration and pilot prompts in the original format;
  - teacher-forced digit accuracy;
  - the additive-mask knockout in all heads;
  - the bitwise no-op control on every prompt.
- **Route,** fixed to each model's confirmed damaging condition:
  - Qwen: the answer positions, from layer 21 to the last.
  - Gemma: every position after the program, from layer 24 to the last.
- **Program lines.** Each program token belongs to exactly one line, the
  line containing its first character. A newline token belongs to the line
  it ends. All 3,072 prompts were checked with the real tokenizers before
  this freeze: no token spans two lines' content.
- **Spans** (keys blocked):

| Span | Tokens | What it tests |
|---|---|---|
| `full` | the whole program | reproduces Phase Eight's damage |
| `last_token` | the program's last token only | is an end-of-program summary necessary? |
| `all_minus_last` | the whole program except its last token | can the last token alone carry the answer? |
| `asked_lines_minus_last` | every line assigning the asked variable, except the last token | are the asked variable's own statements read? |
| `other_lines_minus_last` | every line assigning the other variable, except the last token | control: the other variable's statements |
| `asked_final_line_minus_last` | the asked variable's last assignment line, except the last token | the single most direct source line |

- **Cost.** 8 forwards per prompt: baseline, no-op and the 6 spans.

## Summary statistics (descriptive; no gate)

- Accuracy and mean loss against baseline for each span, by split
  (calibration, pilot).
- Breakdowns:
  - the asked variable's last update (literal / arithmetic / copy);
  - whether the last line assigns the asked variable;
  - whether the asked variable's history includes a copy of the other
    variable (then the other variable's lines carry the asked value).

## Predictions (recorded before any forward)

1. The no-op control is exact.
2. `full` reproduces Phase Eight: accuracy about 0.2 on both models.
3. `all_minus_last` is within 0.10 of `full`: the answer does not come from
   the last token alone.
4. `last_token` alone costs less than 0.10: no single end-of-program summary
   is necessary.
5. `asked_lines_minus_last` costs more than `other_lines_minus_last`. The
   read targets the asked variable's statements, except where the asked
   value was copied from the other variable.

## What follows

If calibration names a clear pattern and pilot replicates it, a
confirmatory test on new programs follows. That test needs:
- a new split generated with a new seed, excluding every program in the
  five Phase Two splits;
- its own frozen protocol before the programs are generated.

## Budget, honesty, limits

- **Budget:** at most 1.5 GPU-hours. Expected about 20–30 minutes for Gemma
  and 15 for Qwen. Each run starts only on an idle GPU. Foreign processes are
  logged, never acted on.
- **Limits.**
  - The last token is the final operand digit or variable. Blocking it also
    removes part of the last line's content.
  - Spans differ in size, so a larger span can matter more for that reason
    alone.
  - Blocking only within the confirmed route says nothing about other
    routes.
  - Knockout is an off-distribution intervention, and the scoring is
    teacher-forced digit accuracy.
  - Reused data: exploratory.

**Code:** `src/open_weight_lingua/span_knockout.py`, launched with
`scripts/run_span_knockout.sh`.
