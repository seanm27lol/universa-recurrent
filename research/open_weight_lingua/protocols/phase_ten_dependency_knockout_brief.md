# Phase Ten: when does a statement collect what it depends on? (dependency knockout, exploratory)

**Status: FROZEN 2026-10-12 (UTC), before any Phase Ten model forward.
Branch `phase-ten-dependency-knockout`, based on `main` at `1021264`. This
is exploratory, on reused calibration and pilot prompts. A confirmatory test
would use another freshly generated split under its own frozen protocol.**

## The question

```text
x = 7
y = 3
x = x + 2          ← the asked variable's final statement: 9 = (7 from line 1) + 2
y = 5
What is x? Reply with only the integer.      → 9
```

Phase Nine confirmed, on fresh programs, that the late read retrieves the
asked variable's own statements, and only those
([report](../reports/phase_nine_confirmatory.md)). For a computed value, it
needs the earlier asked lines as well as the final one. Two accounts fit
that:
- **Eager.** The final statement `x = x + 2` collects its operand (7) from
  line 1 while the program is read, and holds the result.
- **Lazy.** The late read combines the raw statements at answer time.

Phase Ten asks whether the asked variable's final statement must collect
what it depends on, and by when. From layer k onward, it blocks attention
from that statement's tokens to the lines it depends on. If the answer
survives even from layer 0, the statement's own read is never needed: the
lazy account. If blocking hurts until some layer, the statement collects
its operand before that layer: the eager account.

A correction made the same day shaped this design. Most "copy" statements
in these programs are self-copies (`x = x`), and genuine cross-variable
copies (`x = y`) are rare. See the correction notes in the Phase Nine
reports.

## Design

- **Models, prompts and scoring.** As in Phases Eight and Nine:
  - Gemma-3-12B and Qwen2.5-7B;
  - the 1,536 calibration and pilot prompts in the original format;
  - teacher-forced digit accuracy;
  - the additive-mask knockout in all heads;
  - the bitwise no-op control;
  - line assignment by each token's first character.
- **Queries.** Every token of the asked variable's final statement (the
  last line assigning it).
- **Keys (dependencies)**, by the kind of that final statement:

| Final statement | Keys | Role | Prompts |
|---|---|---|---:|
| arithmetic (`x = x + 2`) | every earlier line assigning the asked variable | the operand's source | 674 |
| self-copy (`x = x`) | every earlier line assigning the asked variable | the copied value's source | 148 |
| cross-variable copy (`x = y`) | every earlier line assigning the copied variable | the copied value's source; **descriptive only**, too few | 8 |
| literal (`x = 7`) with earlier asked lines | every earlier line assigning the asked variable | **placebo:** a literal needs nothing earlier | 280 |
| literal with no earlier asked line | none | not run | 426 |

- **Layer grid.** k = floor(j·L/8), j = 0…7, as in Phase Eight.
  - Gemma: 0, 6, 12, 18, 24, 30, 36, 42.
  - Qwen: 0, 3, 7, 10, 14, 17, 21, 24.
- **Cost.** 10 forwards per prompt: baseline, no-op and 8 knockouts.

## Summary statistics (descriptive; no gate)

- Accuracy and loss against baseline for each k, by kind and split.
- **Dependency release layer**, per model and kind: the smallest grid k
  such that blocking from k and every later grid k keeps calibration
  accuracy within 5 points of baseline. "0" means the statement's own read
  is never needed; "none" means it is needed through the last grid layer.
  It is chosen on calibration and checked on pilot.

## Predictions (recorded before any forward)

1. The no-op control is exact.
2. **Literal placebo:** release layer 0. Blocking a literal statement from
   earlier asked lines costs under 5 points at every k.
3. **Arithmetic:** blocking from layer 0 costs under 10 points; the lazy
   account. Phase Nine showed that the late read uses the earlier asked
   lines directly, so the final statement need not collect its operand.
   This prediction is uncertain. If it fails, statements compute eagerly.
4. **Self-copy:** as arithmetic.

## Budget, honesty, limits

- **Budget:** at most 1.5 GPU-hours. About 15,400 forwards per family:
  roughly 35 minutes for Gemma and 25 for Qwen. Each run starts only on an
  idle GPU; foreign processes are logged, never acted on.
- **Limits.**
  - The self-copy subset is modest (148 prompts), and cross-variable
    copies (8) are descriptive only.
  - The late read is left open. A statement blocked from its source could
    in principle be bypassed by reading the source late, which would make
    the release layer early. It cannot make it late.
  - Knockout is an off-distribution intervention, and scoring is
    teacher-forced digit accuracy.
  - Reused data, exploratory.

**Code:** `src/open_weight_lingua/dependency_knockout.py`, launched with
`scripts/run_dependency_knockout.sh`.
