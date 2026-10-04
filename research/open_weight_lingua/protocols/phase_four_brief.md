# Phase Four: does the model track state while it reads?

**Status: FROZEN 2026-10-04 (UTC), before any Phase Four model forward.
Branch `phase-four-trace`, based on `main` at `bec1fea`. Every threshold,
gate and stopping rule below is a design choice locked here, not a result.**

## The concrete question

```text
x = 9          after this line: x = 9
y = 8          after this line: x = 9,  y = 8
y = 3          after this line: x = 9,  y = 3
y = y + 3      after this line: x = 9,  y = 6
x = x + 1      after this line: x = 10, y = 6
What is x?
```

Phase Three found that at the end of the program, neither final value is
linearly readable. After the question, only the asked variable's value
appears, late in depth
([reports/phase_three_stage0.md](../reports/phase_three_stage0.md)).

That suggests the models compute the answer on demand. But it does not rule
out a running state kept at each statement boundary and overwritten, or
local to each line. Phase Four asks directly: at the token that completes
each statement, can a linear probe read the variables' values *as of that
line*?

The positions are every statement boundary of every program, fixed by the
question. They are not chosen from Phase Three's results. Phases Two and
Three stay closed and unchanged.

## Precedent

- **Implicit entity states in text** (Li, Nye & Andreas 2021, *Implicit
  Representations of Meaning in Neural Language Models*, ACL). Models encode
  the changing states of entities while they read narratives, and probes can
  recover them.
- **Entity tracking under state changes** (Kim & Schuster 2023, *Entity
  Tracking in Language Models*, ACL). Tracking entity states through a
  sequence of operations is a measurable capability that varies with the
  model.
- **Probe selectivity** (Hewitt & Liang 2019). Probes are always compared
  with controls.

## Design

- **Models.** Gemma-3-12B-it is primary and gates; Qwen2.5-7B-Instruct is
  reported only. The locks are unchanged and hash-verified at start.
- **Programs.** The source and counterfactual programs of the 256
  calibration and 128 pilot groups: 768 distinct programs, one per group
  side. The two prompts of a side share every token through the program, so
  each program is encoded once. The two validation splits remain unopened and
  are reserved for Stage 1.
- **Positions.** For each statement `i ≥ 2` (both variables are defined from
  line 2 on), the token whose decoded prefix first contains statements 1..i.
  The last statement's boundary is Phase Three's `program_end`.
- **Labels.** x and y after statements 1..i, from the reference interpreter.
  Each (statement, variable) pair is one of:
  - *updated by a literal* (`x = 5`);
  - *updated by arithmetic* (`x = x + 1`), which requires computation;
  - *updated by a copy* (`x = y`);
  - *carried*: not touched on line `i`, so its value has to be held from
    earlier.

  A value is *computed* if it appears nowhere in the program text up to
  line `i`.
- **Probes.** One ridge one-hot probe (20 classes) per layer and variable,
  pooled over all boundaries `i ≥ 2`, so it tests a position-invariant code.
  Ridge strength comes from Phase Three's grid, chosen on the selection
  split. Probes are fitted on the GPU in float64; the method is unchanged.
- **Splits.** Calibration groups 0–191 train, 192–255 select; the pilot
  groups are held out.
- **Controls.**
  - a label-permutation probe;
  - a copy baseline: the last literal assigned to the variable up to
    line `i`;
  - accuracies reported per category, and on the computed subset.

## Readings and gate (frozen)

The best layer for each reading is chosen on the selection split. Reported
numbers are held out.

| Reading | Category and layer choice | Supported iff, held out at that layer |
|---|---|---|
| **R1, carried state is readable** | layer maximizing `min_v` carried accuracy | carried accuracy ≥ 0.80 for both x and y, **and** carried *computed* accuracy ≥ 0.50 for both |
| **R2, computed updates are readable** | layer maximizing `min_v` arithmetic-update accuracy | arithmetic-update accuracy ≥ 0.80 for both x and y |

**Incremental state tracking is supported iff R1 and R2 both hold** on
Gemma-3-12B. That is gate G4.

- If G4 passes, a Stage 1 causal edit test at statement boundaries follows.
  It would edit a variable's value at the last boundary before the question,
  the answer must follow, and the other variable must stay intact. Its own
  protocol is frozen before its forwards, with Phase Three's S1–S3
  thresholds, on the validation splits.
- If G4 fails, Phase Four closes after this survey.

**Predictions, recorded before any forward (not thresholds):**

- **G4 fails.** Phase Three found both final values barely readable at the
  last boundary (0.24 and 0.15 for Gemma).
- **R2 fails:** literal updates read well, but arithmetic results do not.
- **R1 fails:** carried values sit near the copy baseline or below.

If the predictions hold, Phase Four is the direct test of the on-demand
reading: no running state at any statement boundary.

## Budget, stopping, honesty

- **Budget:** at most 6 GPU-hours for the survey on both families.
- **One run per family.** A source bug permits only a regression test plus a
  visibly versioned rerun.
- **Every run** starts only when the GPU has no other workload. Other
  people's jobs are never stopped.
- **Limits:**
  - Linear probes only.
  - One task family; values 0–19.
  - Pooling across boundaries favours position-invariant codes. A code that
    changes with the line number would be missed, and this is stated as
    untested.
  - "Readable" is a property of a probe, not proof that the model uses the
    representation. Only Stage 1 would test use.
