# Phase Six: does asking the question first make the model track the variable?

**Status: FROZEN 2026-10-04 (UTC), before any Phase Six model forward.
Branch `phase-six-question-first`, based on `main` at `8b0cd26`. Every
threshold, gate and stopping rule below is a design choice locked here, not a
result.**

## The concrete question

```text
Phases Three to Five (question after)      Phase Six (question first)

x = 9                                      What is y at the end of this program?
y = 8                                      Reply with only the integer.
y = 3                                      x = 9
y = y + 3      ← y is 6 here               y = 8
x = x + 1                                  y = 3
What is y? Reply with only the integer.    y = y + 3      ← is 6 readable here now?
                                           x = x + 1
```

Phases Three to Five found no readable program state, for linear and small
nonlinear probes, at the end of the program or at any statement boundary.
Only the asked variable's value appears, and only after the question
([Phase Three](../reports/phase_three_stage0.md),
[Phase Four](../reports/phase_four_trace.md),
[Phase Five](../reports/phase_five_nonlinear.md)).

But in all three phases the question came *last*. While reading the program
the model could not know which variable would matter, and both are cheap to
re-read from the text later. Computing lazily, when asked, is then the
natural strategy. Phase Six changes the task, not the probe: the same 768
programs, with the question moved before the program. If the models'
laziness is only for lack of knowing what will be asked, the asked variable's
value should now become readable line by line. The variable that was not
asked, on the same programs, is the control.

## Precedent

- **Entity states in text** (Li, Nye & Andreas 2021, *Implicit
  Representations of Meaning in Neural Language Models*, ACL): probes recover
  changing entity states from models reading narratives.
- **Entity tracking** (Kim & Schuster 2023, *Entity Tracking in Language
  Models*, ACL): tracking states through a sequence of operations is a
  measurable, model-dependent capability.
- **Lookup at query time** (Feng & Steinhardt 2024, *How do Language Models
  Bind Entities in Context?*, ICLR): models store binding information at the
  entity tokens and retrieve it when queried. That is a lazy mechanism, the
  pattern Phases Three to Five found.

## Design

- **Models.** Gemma-3-12B-it is primary and gates; Qwen2.5-7B-Instruct is
  reported only. Locks unchanged, hash-verified at start.
- **Prompts.** Every calibration and pilot prompt of Phase Two (256 + 128
  groups, both sides, both questions: 1,536 prompts). The program is
  unchanged. The user message becomes
  `What is {v} at the end of this program? Reply with only the integer.`,
  then a newline and the program. It is tokenized with the same chat template
  and generation prefix. The validation splits remain unopened.
- **Splits.** As in Phases Three to Five: calibration groups 0–191 train,
  192–255 select, the pilot split held out. Each split keeps both questions
  of a group together.
- **Positions.**
  - Every statement boundary from line 2: the token whose decoded prefix
    first contains the question line and statements 1..i. That is 5,096
    boundaries, 2,548 per asked variable.
  - The answer position (the last token of the generation prefix).
- **Labels.** x and y as of each line, with Phase Four's categories
  (literal, arithmetic, copy, carried) and computed flags, from the reference
  interpreter.

### Stage 0: is the new format usable? (gate U6)

Greedy decoding, as in the pilots, on all 1,536 question-first prompts and
on the same 1,536 prompts in the original format. Answers are scored under
each family's frozen convention: `rstrip` for Gemma (the 2026-09-27
amendment), `raw` for Qwen.

- **U6:** Gemma-3-12B's question-first accuracy is ≥ 0.80.
- **If U6 fails,** the new format is unusable and G6 and E6 are void. The
  probe numbers are then reported as descriptive only, and Phase Six closes.
- **Sanity check (descriptive):** the original-format accuracy on the pilot
  split should reproduce the pilots' P0: Gemma 0.951 under `rstrip`, Qwen
  0.811 under `raw`. A mismatch is reported, not corrected.

### Stage 1: probes

Phase Four's linear probe, unchanged: dual ridge one-hot, fitted on the GPU
in float64, with alpha from Phase Three's grid chosen on the selection
split. One probe per layer, variable v and condition:

- **asked:** trained and scored on the prompts that ask about v;
- **not asked:** trained and scored on the prompts that ask about the other
  variable. These are the same programs and boundaries; only the question
  differs.

**Readings, on the asked condition.** Phase Four's rule
(`state_trace_survey.readings`) chooses each reading's layer on the
selection split and applies its thresholds:

- **R1 (carried):** carried ≥ 0.80 and carried-computed ≥ 0.50, both
  variables.
- **R2 (arithmetic):** arithmetic results ≥ 0.80, both variables.
- At each reading's layer, a permuted-label control is trained at the chosen
  alpha on labels shuffled with seed 301100, and scored against the true
  held-out labels. It must be ≤ 0.20 for both variables.

**Gate G6:** with U6 met, the model tracks the asked variable iff R1 **and**
R2 hold on Gemma-3-12B. These are G4's conditions, met by the asked variable
once the question comes first.

**Effect E6 (secondary).** Does knowing the question raise the readability
of arithmetic results at all, even far below the gate?

- **Where:** at R2's layer, on held-out arithmetic boundaries of both
  variables.
- **Statistic:** per boundary and variable, asked-probe correctness minus
  not-asked-probe correctness, pooled over both variables.
- **Interval:** a whole-group bootstrap (3,000 resamples, seed 601100).
- **Support:** E6 holds iff the two-sided 95% interval lies above 0.

**Descriptive, never gating:**
- the same asked-minus-not-asked differences for every category;
- the final value at the program's last token and at the answer position,
  asked and not asked;
- Phase Four's question-after accuracies on the same programs, for
  comparison.

## Predictions (not thresholds), recorded before any forward

- **U6 passes:** question-first accuracy within a few points of the original
  format.
- **G6 fails:** the asked variable's arithmetic results stay below 0.80, and
  its carried computed values below 0.50.
- **E6 holds, but small:** asked arithmetic results read better than not
  asked, but stay below 0.5.
- **The final value moves earlier:** at the program's last token, before
  any answer cue, the asked variable's final value is readable well above
  Phase Three's 0.24. Once the question and the whole program are present,
  computing on demand can happen there.

## Budget, stopping, honesty

- **Budget:** at most 3 GPU-hours. One run per family. A source bug permits
  only a regression test plus a visibly versioned rerun.
- **Start:** each run starts only when the GPU has no other workload.
- **Data:** the validation splits stay unopened.
- **Limits:**
  - One question wording.
  - Linear probes, which Phase Five found to match a small MLP on this task.
  - Readability is not use.
  - One task family: integers 0–19, short programs. A short program can
    always be re-read; a task that makes re-reading impossible is out of
    scope.

**Code:** `src/open_weight_lingua/state_question_first.py`, launched with
`scripts/run_state_question_first.sh`.
