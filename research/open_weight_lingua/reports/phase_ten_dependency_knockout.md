# Phase Ten: the final statement needs to see earlier assignments, even when it is a literal — exploratory — 2026-10-12

```text
x = 7
y = 3
x = x + 2          ← the asked variable's final statement
y = 5
What is x? Reply with only the integer.      → 9
```

Phase Nine confirmed that the late read retrieves the asked variable's own
statements ([report](phase_nine_confirmatory.md)). Phase Ten
([brief](../protocols/phase_ten_dependency_knockout_brief.md)) asked whether
the asked variable's **final** statement must collect what it depends on.
From layer k onward, it blocked that statement's tokens from the lines it
depends on.

**The design's placebo failed, and with it the brief's interpretation.**
- A *literal* final statement (`x = 7`) needs no earlier line to know its
  value. Yet blocking it from the earlier lines assigning x damages the
  answer about as much as blocking an arithmetic statement does (Gemma
  0.963 → 0.437 from layer 0; Qwen 0.884 → 0.547).
- So these knockouts do not isolate "collecting an operand". Something else
  the final statement does with earlier same-variable lines matters. One
  possibility is marking itself as the current assignment. Another is that
  blocking any of its context disrupts it.
- Phase Eleven is designed to tell these apart.

These are exploratory results on reused calibration and pilot prompts.

## What ran

- **Brief.** Frozen and pushed at `f25602f` before any Phase Ten forward.
  Both manifests record that commit, the brief's hash `82a6f30fd6ba…`, and
  source hashes identical to the frozen code.
- **Setup.** 1,536 calibration and pilot prompts per family;
  teacher-forced digit accuracy; the bitwise no-op control, which held on
  every prompt. 426 prompts whose final asked statement is a first literal
  had nothing to block and were skipped.
- **Knockout.** Queries are every token of the asked variable's final
  statement. Keys are the lines it depends on: the asked variable's earlier
  lines, or the copied variable's earlier lines for a cross-variable copy.
  The block runs from grid layer k to the last.

| Family | Run | Manifest / results / summary sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p10-dependency-gemma3-12b-20261012T032640Z-27aa963f` | `a1c28586…` / `2b1e445a…` / `5b2fa681…` | 1,799 s / 1,947 s |
| Qwen2.5-7B | `p10-dependency-qwen2.5-7b-20261012T040817Z-4440773a` | `e4ee91ef…` / `965b27aa…` / `3df4d281…` | 1,024 s / 1,111 s |

Both runs were admitted to an idle GPU. One 375 MiB process from another
project appeared once and was logged.

## Results (digit accuracy; calibration, pilot in parentheses where informative)

**Gemma-3-12B** (grid 0, 6, …, 42):

| Final statement (calibration n) | Baseline | k=0 | 6 | 12 | 18 | 24 | 30 | Release layer (cal / pilot) |
|---|---|---|---|---|---|---|---|---|
| Literal, with earlier asked lines (190) | 0.963 | **0.437** | 0.626 | 0.737 | 0.916 | 0.963 | 0.968 | 18 / 18 |
| Arithmetic (450) | 0.889 | **0.598** | 0.627 | 0.704 | 0.833 | 0.882 | 0.891 | 24 / 24 |
| Self-copy `x = x` (90) | 0.989 | 0.989 | 0.989 | 0.978 | 0.989 | 0.989 | 0.989 | 0 / 12 |

**Qwen2.5-7B** (grid 0, 3, 7, 10, 14, 17, 21, 24):

| Final statement (calibration n) | Baseline | k=0 | 3 | 7 | 10 | 14 | 17 | Release layer (cal / pilot) |
|---|---|---|---|---|---|---|---|---|
| Literal, with earlier asked lines (190) | 0.884 | **0.547** | 0.621 | 0.674 | 0.716 | 0.811 | 0.858 | 17 / 17 |
| Arithmetic (450) | 0.693 | **0.329** | 0.387 | 0.507 | 0.529 | 0.600 | 0.653 | 17 / 17 |
| Self-copy `x = x` (90) | 0.956 | 0.800 | 0.811 | 0.911 | 0.944 | 0.944 | 0.956 | 7 / 7 |

The pilot curves track calibration (for example, Gemma literal from layer 0
is 0.567, arithmetic 0.558).

- **Cross-variable copies** (`x = y`): only 8 prompts, with baselines of
  0.25–0.50 (Gemma) and 0 (Qwen). They are uninformative and are not
  interpreted.
- **Release layers:** every literal and arithmetic release layer
  replicated exactly on pilot. The self-copy release layers did not
  replicate for Gemma (0 vs 12): its curve barely moves.

**Predictions.**
1. The no-op control is exact. *Held.*
2. The literal placebo has release layer 0. **Wrong:** it is 18 (Gemma) and
   17 (Qwen), the same window as arithmetic.
3. Arithmetic costs under 10 points from layer 0. **Wrong:** it costs 29
   (Gemma) and 36 (Qwen) points.
4. Self-copies behave as arithmetic. **Wrong:** they cost Gemma nothing
   and Qwen up to 16 points.

## What this establishes (exploratory)

- **In layers before about 17–24, the asked variable's final statement
  needs to see the earlier assignments of that variable.** This holds even
  when its own value does not depend on them (a literal). Blocking that view
  from layer 0 removes 34–53 points of accuracy for literals and 29–36 for
  arithmetic. Blocking it only from layer 18 (Gemma) or 17 (Qwen) onward
  costs little.
- **The dependency-collection reading of the knockout is not supported.**
  The literal result shows that the effect is not specific to operands.

## What this does not establish

- **Why the view matters.** Two accounts remain:
  - *Supersession:* the final statement marks itself as replacing earlier
    assignments. The late read then retrieves the current one, and without
    the mark it may retrieve an outdated value.
  - *Disruption:* blocking part of any statement's context degrades it.
  
  A control block (the final statement blocked from the *other* variable's
  earlier lines) and a measure of preference for the *superseded* value can
  separate them.
- **Self-copies** behave differently in the two models. Few prompts; not
  interpreted further.
- **Knockout limits.** An off-distribution intervention; teacher-forced
  digit accuracy; reused data; exploratory.

Code: [dependency_knockout.py](../src/open_weight_lingua/dependency_knockout.py),
launched with `scripts/run_dependency_knockout.sh`.
