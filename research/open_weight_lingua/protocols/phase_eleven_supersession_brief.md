# Phase Eleven: does the final statement mark itself as current? (supersession knockout, exploratory)

**Status: FROZEN 2026-10-12 (UTC), before any Phase Eleven model forward.
Branch `phase-ten-dependency-knockout`, after the Phase Ten report
(`31d00bd`). This is exploratory, on reused calibration and pilot prompts.**

## The question

```text
x = 4
y = 3
x = 7              ← the asked variable's final statement (a literal)
y = 5
What is x? Reply with only the integer.      → 7   (superseded value: 4)
```

Phase Ten ([report](../reports/phase_ten_dependency_knockout.md)) found a
surprise. Blocking the asked variable's final statement from the earlier
assignments of that variable damages the answer even when the statement is
a literal, whose value needs nothing earlier. Two accounts:

- **Supersession.** The final statement looks at earlier assignments of
  the same variable to mark itself as the current one. Without that, the
  late read can retrieve an outdated value: here 4 instead of 7.
- **Disruption.** Blocking any part of a statement's context degrades it,
  whatever that part contains.

The two make different predictions:
- **Control block.** Blocking the final statement from the *other*
  variable's earlier lines should not hurt under supersession, but should
  hurt under disruption.
- **Superseded value.** Supersession predicts that the same-variable block
  shifts probability toward the superseded value (4).

## Design

- **Models, scoring and controls.** Gemma-3-12B and Qwen2.5-7B;
  teacher-forced digit accuracy; the bitwise no-op control; the Phase
  Eight–Ten knockout and line assignment.
- **Prompts.** The calibration and pilot prompts whose asked variable's
  final statement is a literal with earlier asked lines (280), or
  arithmetic (674).
- **Superseded value.** The asked variable's value just before its final
  statement, from the reference interpreter. Prompts where it equals the
  answer are excluded from preference measures.
- **Knockouts.** Queries are every token of the final statement; the block
  runs from layer 0 to the last.

| Condition | Keys blocked |
|---|---|
| `same_variable` | every earlier line assigning the asked variable (Phase Ten's block) |
| `other_variable` | every earlier line assigning the other variable (control) |

- **Measures, per condition.**
  - Correct: every answer digit is the argmax under teacher forcing.
  - Superseded-correct: every digit of the superseded value is the argmax
    when *it* is teacher-forced, meaning the model would output the old
    value.
  - Preference: log P(superseded digits) − log P(answer digits).
- **Key sizes** are reported, because the two key sets differ in size.

## Predictions (recorded before any forward)

1. The no-op control is exact.
2. `same_variable` reproduces Phase Ten's damage: a loss of more than 25
   points for literals.
3. **`other_variable` costs less than 10 points** for both kinds (the
   supersession account; it fails under disruption).
4. Under `same_variable`, **the preference for the superseded value rises**
   by more than 2 nats on average for literals. Many wrong answers become
   the superseded value, so superseded-correct exceeds 25% of literal
   prompts.

## Budget, honesty, limits

- **Budget:** at most 1 GPU-hour. About 7 forwards per prompt: roughly 15
  minutes for Gemma and 10 for Qwen. Each run starts only on an idle GPU;
  foreign processes are logged.
- **Limits.**
  - The two key sets differ in size and position, so a size effect cannot
    be ruled out entirely. Sizes are reported.
  - Knockout from layer 0 is the strongest version; other layers are not
    tested here.
  - Reused data, exploratory.

**Code:** `src/open_weight_lingua/supersession_knockout.py`, launched with
`scripts/run_supersession_knockout.sh`.
