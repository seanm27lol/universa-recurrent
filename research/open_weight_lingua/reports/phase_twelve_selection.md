# Phase Twelve: the late read takes the most recent intact assignment — exploratory — 2026-10-12

```text
x = 2              ← v2: the value two assignments back
y = 3
x = x + 9          ← v1: 11, the previous value
y = 5
x = 7              ← v0: 7, the final assignment and the answer
What is x? Reply with only the integer.
```

Phases Ten and Eleven confirmed, on fresh programs, that degrading the
asked variable's final assignment makes the model answer its previous value
([report](phase_eleven_confirmatory.md)). Phase Twelve
([brief](../protocols/phase_twelve_selection_brief.md)) tested the
selection rule behind that: does the late read take the **most recent
intact** assignment?

**Yes, in both models.**
- Degrade the final assignment, and the answer falls back to v1.
- Degrade the final *and* the previous assignment, and it falls back to v2.
- Degrade only the previous assignment, and the answer stays v0.

These are exploratory results. The pool reuses programs opened before; its
reused and fresh parts agree.

## What ran

- **Brief.** Frozen and pushed at `daee4e9` before any Phase Twelve
  forward. Both manifests record that commit, the brief's hash
  `480e2d7a2fa9…`, and source hashes identical to the frozen code.
- **Pool: 329 eligible prompts.**
  - 101 from Phase Two's calibration and pilot prompts.
  - 228 from the two fresh splits, regenerated and verified against their
    recorded inventories.
  - Eligible means: at least three assignments of the asked variable, a
    literal final one, and v0, v1, v2 pairwise distinct.
- **Degrading a statement** means blocking its tokens from every earlier
  program token, from layer 0. Each value was scored by teacher forcing.
  The bitwise no-op control held on every prompt.

| Family | Run | Manifest / results / summary sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p12-selection-gemma3-12b-20261012T064741Z-ba521d33` | `2a9ada74…` / `645f42f4…` / `6e6c34ff…` | 752 s / 899 s |
| Qwen2.5-7B | `p12-selection-qwen2.5-7b-20261012T071201Z-8189e3b8` | `3aafa974…` / `8d5d811f…` / `8813eff7…` | 424 s / 509 s |

Both runs were admitted to an idle GPU, with no foreign processes.

## Results

Rate at which each value is said (all 329 prompts):

| Degraded | Gemma: v0 / v1 / v2 | Qwen: v0 / v1 / v2 |
|---|---|---|
| Nothing (baseline) | 0.970 / 0.012 / 0.003 | 0.729 / 0.134 / 0.043 |
| The final assignment | 0.277 / **0.486** / 0.046 | 0.112 / **0.736** / 0.097 |
| The final and the previous assignment | 0.377 / 0.155 / **0.404** | 0.079 / 0.091 / **0.848** |
| Only the previous assignment | **1.000** / 0.000 / 0.000 | **0.903** / 0.006 / 0.067 |

**Mean log-probability** of v0 / v1 / v2:
- Gemma, final degraded: −6.7 / −4.5 / −11.0; both degraded: −5.2 / −9.0 /
  −4.6.
- Qwen, final degraded: −12.3 / −1.6 / −10.3; both degraded: −13.8 / −15.2
  / −0.9.

**By source.** The reused (101) and fresh (228) subsets agree. For
example, with both assignments degraded, Gemma says v2 0.307 (reused) and
0.447 (fresh); Qwen 0.851 and 0.846.

**Predictions.**
1. The no-op control is exact. *Held.*
2. Degrading the final assignment costs more than 25 points, and v1 is the
   most common wrong answer. *Held:* −69 (Gemma) and −62 (Qwen) points;
   v1 makes up 61% and 76% of wrong answers.
3. Degrading both raises v2 by more than 10 points over degrading the final
   alone, and v2 exceeds v1. *Held:* +36 and +75 points; v2 0.404 against
   v1 0.155 (Gemma), 0.848 against 0.091 (Qwen).
4. Degrading only the previous assignment costs under 10 points. *Held:*
   Gemma 0.970 → 1.000, and Qwen *improves* from 0.729 to 0.903.

## What this establishes (exploratory)

- **The late read selects the most recent intact assignment of the asked
  variable.**
  - Degrading assignments from the end pushes the answer back one
    assignment at a time.
  - Degrading an earlier assignment while leaving the final one intact
    changes nothing.
- **An earlier assignment interferes in Qwen.** Degrading the previous
  assignment raises Qwen's accuracy by 17 points. Most of Qwen's baseline
  errors are the previous value (45% of its wrong answers), so removing
  that competitor helps.
- **With Phases Eight to Eleven:**
  - after reading the question, the model looks back at the asked
    variable's statements in a late window;
  - it retrieves the most recent intact assignment.

  That is a lookup with recency, not a running state maintained while
  reading.

## What this does not establish

- **Confirmation.** The pool reuses opened programs. A confirmatory test
  needs a third fresh split.
- **What "intact" means mechanistically.** The degrading block is
  off-distribution. The rule describes what the read does when a statement
  has lost its context, not how statements are marked as current in normal
  operation.
- **Scope.** Literal final statements only; one task family.

## Next

A confirmatory test on a third fresh split (a new seed, excluding every
earlier program) of three claims per model:
- degrading the final assignment raises v1 answers;
- degrading both raises v2 answers above the final-only level;
- degrading the previous assignment alone does not hurt.

Code: [selection_knockout.py](../src/open_weight_lingua/selection_knockout.py),
launched with `scripts/run_selection_knockout.sh`.
