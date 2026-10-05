# Phase Six: the question-first format fails usability; tracking remains unresolved — 2026-10-05

```text
What is x at the end of this program? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          ← the program ends on y
```

Gemma-3-12B answers `14`. That is y's final value; x is 7. With the same
program followed by the original, shorter question, `What is x? Reply with
only the integer.`, Gemma answers 7.

Phases Three to Five did not meet their program-state readout thresholds
while the model reads a program, at its end, or with a small nonlinear probe
([Phase Five](phase_five_nonlinear.md)). In all three the question came last,
so while reading, the model could not know which variable would matter.
Phase Six ([brief](../protocols/phase_six_brief.md)) kept the programs and
linear probe method, but moved the question first and added "at the end of
this program" to its wording. The 1,536 program/question pairs test whether
this format makes the asked variable easier to read at statement boundaries.
It does not isolate the effect of moving an otherwise identical question.

**Gate U6 failed on Gemma-3-12B, so G6 and E6 are void and Phase Six closes
after Stage 0.**
- With the question first, Gemma answers 0.729 of the prompts correctly
  (U6 needed 0.80), against 0.944 with the question after.
- Qwen2.5-7B, reported only, falls from 0.810 to 0.502.
- The probe results below are descriptive only. Neither model reaches the
  planned tracking thresholds. The asked-minus-not-asked arithmetic intervals
  include zero and small positive effects; they do not establish no effect.
- Every earlier outcome stands unchanged.

**Review addendum — 2026-10-05.** This report's interpretation was revised
after review of commit `c4a6e73`. The original text attributed the loss to
question placement and inferred late computation from poor probe readout.
Position and wording changed together, and neither the readouts nor the
answer errors identify that mechanism. The revised text also distinguishes
Gemma's error pattern from Qwen's and notes reuse of the pilot data. Recorded
numbers, gates, the frozen brief and run artifacts are unchanged.

## What ran

- **Brief.** Frozen and pushed at `1c61018`, before any Phase Six forward.
  Both manifests record that commit, the brief's hash `b0a2cc36b5ee…`, and
  source hashes identical to the frozen code.
- **Start.** Both runs started only after the GPU was idle.
- **Prompts.** 1,536 prompts per family: 768 train, 256 select, 512 held
  out. That gives 5,096 statement boundaries, twice Phase Four's 2,548
  because each program is asked both ways. The validation splits stay
  unopened. "Held out" here means excluded from the current probe fit and
  selection: the pilot was already examined in earlier phases. Groups and
  complete programs are disjoint across these splits, but some intermediate
  token prefixes recur. This is an exploratory survey, not a fresh
  confirmatory sample or a test in which every boundary input is novel.
- **Answers.** Greedy decoding in both formats, scored under each family's
  frozen convention (`rstrip` for Gemma, `raw` for Qwen).
- **Probes.** Phase Four's linear probe at every boundary, separately for
  the asked and the not-asked variable on the same programs.

| Family | Run | Manifest sha256 | Wall clock (answers / capture / probes) |
|---|---|---|---:|
| Gemma-3-12B (gating) | `p6-question-first-gemma3-12b-20261004T231457Z-96041d06` | `294d6f840c606218…` | 1,699 s (1,250 / 248 / 35) |
| Qwen2.5-7B (reported only) | `p6-question-first-qwen2.5-7b-20261004T235340Z-4f429ce9` | `9406bc43fa7ff187…` | 1,430 s (1,027 / 255 / 34) |

## Stage 0: the answers (gate U6)

| Exact-answer accuracy | Gemma-3-12B | Qwen2.5-7B |
|---|---:|---:|
| Question after (the original format), all 1,536 prompts | 0.944 | 0.810 |
| Question after, pilot split (the pilots' P0) | **0.951 (487/512)** | **0.811 (415/512)** |
| **Question first, all 1,536 prompts (U6 needs ≥ 0.80 on Gemma)** | **0.729** | 0.502 |
| Question first, asking about x / about y | 0.647 / 0.810 | 0.402 / 0.602 |

The sanity check holds exactly. On the pilot split, the original format
reproduces both pilots' P0 to the prompt: 487/512 and 415/512.

**U6: not met** (0.729 < 0.80). The brief predicted it would pass; that
prediction was wrong.

## The probes (descriptive, since U6 failed)

Each probe was fitted on the asked variable's prompts, and again on the
other variable's prompts. These are the same programs and boundaries; only
the question differs.

| At each reading's layer, held out (x / y) | Gemma asked | Gemma, Phase Four | Qwen asked | Qwen, Phase Four | G6 needed |
|---|---|---|---|---|---|
| R1: carried values | 0.653 / 0.379 (L9) | 0.672 / 0.383 (L7) | 0.693 / 0.379 (L4) | 0.707 / 0.375 (L3) | 0.80 |
| R1: carried computed values | 0.000 / 0.123 | 0.000 / 0.077 | 0.000 / 0.062 | 0.000 / 0.108 | 0.50 |
| R2: arithmetic results | 0.103 / 0.228 (L16) | 0.176 / 0.206 (L18) | 0.154 / 0.254 (L5) | 0.184 / 0.184 (L26) | 0.80 |
| Permuted-label control | 0.062–0.064 | — | 0.048–0.056 | — | ≤ 0.20 |

**Asked versus not asked, paired** (E6's statistic, void). This is held-out
accuracy at R2's layer, pooled over x and y, with a whole-group bootstrap
95% interval:

| Boundaries updated by | Gemma: asked − not asked | Qwen: asked − not asked |
|---|---|---|
| Arithmetic (E6) | +0.014 [−0.014, +0.043] (0.181 vs 0.168) | +0.016 [−0.006, +0.039] (0.217 vs 0.201) |
| Carried | −0.002 [−0.030, +0.024] | −0.011 [−0.028, +0.006] |
| Literal | −0.032 [−0.054, −0.010] | +0.005 [−0.010, +0.023] |

Descriptively maximizing over layers, the lower of the x and y arithmetic
scores reaches 0.228 (Gemma) and 0.206 (Qwen) when asked; not asked, 0.237 and
0.199. These are post hoc held-out maxima, not additional selected readings.

**The final value** (held-out x / y; layer chosen to maximize the lower x/y
accuracy on the selection split):

| Position | Gemma asked | Gemma not asked | Qwen asked | Qwen not asked |
|---|---|---|---|---|
| The program's last token | 0.301 / 0.180 | 0.289 / 0.191 | 0.277 / 0.250 | 0.254 / 0.223 |
| The answer position | 0.594 / 0.855 | 0.352 / 0.227 | 0.391 / 0.566 | 0.223 / 0.238 |

The brief predicted a marked improvement in the asked final value's
readability at the program's last token over Phase Three's 0.24. These
scores do not show that pattern consistently across variables. The asked
value is read more accurately at the answer position. This locates where
these probes work better; it does not establish when the model computes or
stores the value.

## Why the answers fail (post hoc, descriptive)

The brief did not plan this analysis, and it can never change a gate.
`scripts/question_first_errors.py RUN_DIR` reproduces both tables from a
run's saved generations. A majority of Gemma's wrong answers equal the
*other variable's* final value. That explains 28% of Qwen's errors, not a
majority. Matching that value does not by itself identify the internal cause
of the error, and the remaining errors are not classified here:

| Wrong answers that equal the other variable's final value | Gemma | Qwen |
|---|---:|---:|
| Question first | 273 of 417 (65%) | 213 of 765 (28%) |
| Question after | 3 of 86 | 29 of 292 |

The error pattern is associated with the last assignment. Each program is
asked both ways, so in exactly half the prompts the last line assigns the
asked variable:

| Accuracy (answered with the other variable's value) | Gemma, question first | Gemma, question after | Qwen, question first | Qwen, question after |
|---|---|---|---|---|
| Last line assigns the asked variable (768) | 0.918 (0.4%) | 0.923 (0.3%) | 0.667 (2.5%) | 0.759 (1.8%) |
| Last line assigns the other variable (768) | **0.539 (35.2%)** | 0.965 (0.1%) | **0.337 (25.3%)** | 0.861 (2.0%) |

When the program ends on the asked variable, Gemma's accuracy is similar
across formats (0.918 versus 0.923), but Qwen's falls from 0.759 to 0.667.
Both models have a larger accuracy loss when the last assignment is to the
other variable, and answers matching that variable become more frequent.

## What this establishes

- **This question-first format reduces exact-answer accuracy.** The loss
  relative to the original question-after format is 0.22 (Gemma) and 0.31
  (Qwen), with both question position and wording changed. Gemma fails U6.
- **The tested linear readouts do not reach the planned tracking
  thresholds.** The descriptive asked-minus-not-asked arithmetic intervals
  span zero and allow gains up to about 4.3 points (Gemma) and 3.9 points
  (Qwen). This comparison is inconclusive about small effects, rather than
  evidence of equivalence.
- **Asked-value readout is stronger at the answer position.** Together
  with Phases Three to Five, this motivates a hypothesis of computation or
  retrieval near the answer. The surveys alone cannot distinguish that
  hypothesis from state stored elsewhere, a code the probes miss, or a
  failure to use an available value when producing the answer.

## What this does not establish

- **U6 failed.** The probe numbers are descriptive and cannot pass or fail
  G6 or E6. Their interpretation leans on the paired asked/not-asked design,
  not on the gate.
- **Position and wording are confounded.** The question-first prompt adds
  "at the end of this program" to the original question. Its effect, the
  position effect and their interaction are not separated. The last-line
  analysis is post hoc.
- **Linear probes only,** which Phase Five found to match a small MLP on
  its question-after task. Poor readout does not prove absent state, and
  successful readout would not establish that the model uses it. The small
  MLP result does not cover arbitrary codes or the new question-first format.
- **Reused exploratory pilot.** Group separation protects the current fit
  from direct group leakage, but earlier phases already informed this
  study's design. Repeated token prefixes also limit a claim about novel
  intermediate states. Reserved validation remains unopened.
- **One task family:** integers 0–19, short programs that can always be
  re-read.

## Closure

Per the brief, a failed U6 voids G6 and E6 and closes Phase Six. The
validation splits remain unopened. GPU time was about 52 minutes, against a
3-hour budget. Each run keeps its captured activations locally (`raw/`,
4.6 GB for Gemma and 2.5 GB for Qwen); they are not committed.

These four surveys constrain what the tested readouts recover; they do not
settle how the models retain or compute state. A task that restricts access
to earlier program information, together with appropriate interventions,
could help distinguish storage, retrieval and recomputation. Such a test
requires a new design and its own controls.
