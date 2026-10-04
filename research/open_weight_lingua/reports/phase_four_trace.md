# Phase Four: no running state at statement boundaries — 2026-10-04

```text
x = 9          after this line: x = 9
y = 8          after this line: x = 9,  y = 8
y = 3          after this line: x = 9,  y = 3     ← y updated by a literal
y = y + 3      after this line: x = 9,  y = 6     ← y updated by arithmetic; x carried
x = x + 1      after this line: x = 10, y = 6
What is x?
```

Phase Three found that neither final value is readable at the end of the
program, and that the asked value appears only after the question. Phase
Four ([brief](../protocols/phase_four_brief.md)) asked the direct follow-up:
at the token that completes each statement, can a linear probe read x and y
*as of that line*? Accuracy is split by how each variable changed on that
line.

**Gate G4 failed on Gemma-3-12B, as predicted, and Phase Four closes.**
Qwen2.5-7B, reported only, shows the same pattern. Every earlier outcome
stands unchanged.

## What ran

- **Brief.** Frozen and pushed at `af4e648`, before any Phase Four forward;
  both manifests record its hash, `98feb9f1c9ab…`.
- **Start.** Both runs started only after the GPU was idle; the first waited
  for another run to finish.
- **Data.** 768 programs, one per calibration and pilot group side, give
  2,548 statement boundaries from line 2 on: 1,246 train, 432 select and 870
  held out. The validation splits stay unopened.
- **Probe.** One ridge one-hot probe per layer and variable, pooled over all
  boundaries, using Phase Three's method and grid, fitted on the GPU in
  float64.

| Family | Run | Manifest sha256 | Wall clock |
|---|---|---|---:|
| Gemma-3-12B (gating) | `p4-trace-gemma3-12b-20261004T161553Z-517f9189` | `f5bc8defadb59e4a…` | 342 s |
| Qwen2.5-7B (reported only) | `p4-trace-qwen2.5-7b-20261004T164315Z-27667f5a` | `e8d7ac3de7c663f3…` | 182 s |

Of Gemma's 342 s, capture took 280 s and probe fitting 33 s. Phase Three's
CPU fitting took 3,329 s.

## Result (held out)

| Reading, at its selected layer | Gemma-3-12B x / y | Qwen2.5-7B x / y | Copy-the-literal baseline x / y | Needed |
|---|---|---|---|---|
| R1: carried values (L7 Gemma; L3 Qwen) | 0.672 / 0.383 | 0.707 / 0.375 | 0.904 / 0.661 | 0.80 |
| R1: carried *computed* values | **0.000 / 0.077** | **0.000 / 0.108** | — | 0.50 |
| R2: arithmetic results (L18 Gemma; L26 Qwen) | **0.176 / 0.206** | **0.184 / 0.184** | 0.044 / 0.018 | 0.80 |
| Literal updates, at the R2 layer | 0.794 / 0.906 | 0.706 / 0.909 | 1.000 / 1.000 | — |

**G4 = R1 and R2: not met on Gemma-3-12B. Phase Four closes.**

**Across depth** (descriptive). The best value of `min(x, y)` at any layer,
held out:

| Category | Gemma-3-12B | Qwen2.5-7B |
|---|---:|---:|
| Carried | 0.383 | 0.375 |
| Carried computed | 0.056 | 0.037 |
| Arithmetic | 0.221 | 0.206 |
| Label-permutation control (never above) | 0.089 | 0.071 |

No layer of either model comes close to either reading.

**Predictions: right.** The brief predicted that G4 fails; that literal
updates read well but arithmetic results do not; and that carried values
sit near the copy baseline or below. All three held. Carried values in fact
sit *below* the copy baseline, and carried computed values are essentially
absent.

## What this establishes

- **At each statement boundary, the models hold what is written on that
  line.** A literal update is readable from layer 0, because the boundary
  token is the number itself. The result of the line's arithmetic is barely
  readable (about 0.2, above the controls but far below the gate). A value
  computed on an earlier line and carried forward is essentially not
  readable at all.
- **With Phase Three, this is a consistent picture on this task.** Neither
  model keeps a linearly readable running state of the program, either
  while reading it or at its end. The asked variable's value appears only
  after the question, late in depth. Answers look computed on demand, at
  question time, from the program text.
- This is why the released NLA describers in Phase Two could only predict
  answers, and why a typed state record could not be built at any site
  surveyed.

## What this does not establish

- **Linear probes only.** A nonlinear code is untested.
- **Pooling across lines.** A code whose geometry changes from one line to
  the next would be missed. The weak arithmetic signal (about 0.2) could be
  such a code, or partial information such as an operand.
- **One task family,** integers 0–19, short programs.
- **Readability is not use.** Only a causal test (Phase Four's Stage 1) could
  show what the model uses, and it was not run because G4 failed.

## Closure

Per the brief, a failed G4 closes Phase Four after its survey. The
validation splits remain unopened. GPU time was about 7 minutes of capture
and under a minute of probe fitting, against a 6-hour budget.

Taken together, Phases Three and Four answer the question Phase Two left
open. On this task, these models don't appear to keep a readable variable
state, so neither a released describer nor a probe-built record can state
it. A future phase would need a task that forces state to persist, for
example several questions about one program, or a model trained to keep it.
