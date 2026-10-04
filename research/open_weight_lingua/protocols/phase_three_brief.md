# Phase Three: a state record that a real model actually reads?

**Status: FROZEN 2026-10-04 (UTC), before any Phase Three model forward.
Branch `phase-three-state`, based on `main` at `3323926`. Every threshold,
gate and stopping rule below is a design choice locked here, not a result.**

## The concrete question

```text
x = 9
y = 8
y = 3
y = y + 3
x = x + 1
What is x? Reply with only the integer.
```

The answer is 10, and when the question comes y is 6. Phase Two asked
whether released Natural Language Autoencoders (NLAs) describe such a state
in English, so that editing "x is 10" to "x is 11" would make the model
answer 11. They do not. At the site they were trained for, their
descriptions predict the next token ("expecting "10" or "11"") and never
state variable values. That left the edit hypothesis untested on 0/128
groups for every family
([reports/edit_eligibility_structure.md](../reports/edit_eligibility_structure.md)).

Phase Three builds the missing piece itself and tests it. Is there a
position and layer where the model holds *both* variables' current values?
Can they be read out as a typed statement, `x is currently 10; y is
currently 6`? And does editing that statement change exactly the answer it
should?

**This is a new phase.** Phase Two and every result in it stay closed and
unchanged. Nothing here relabels a Phase Two outcome.

## Intuition, and the honest catch

If a probe trained on ground-truth labels writes the statement, the
statement mentions variables by construction. So a readable statement alone
proves little. The test that matters is causal and specific:

- **Targeted:** editing x's stated value must move the x answer to the new
  value.
- **Specific:** it must leave the y answer intact. Phase Two could never run
  this wrong-variable control, because its descriptions never mentioned the
  other variable.
- **Beyond noise:** a random direction of the same size must do neither.

That is the standard of known-field work. The first two are the precedents;
the rest are methods this phase uses.

- **Board-state probes and edits** (Li et al. 2023, *Emergent World
  Representations*, ICLR). Probes recovered an Othello board from a sequence
  model's activations, and editing those activations changed its legal-move
  predictions as the edited board implied.
- **Linear world representations** (Nanda, Lee & Wattenberg 2023, *Emergent
  Linear Representations in World Models of Self-Supervised Sequence
  Models*). The same board state turned out to be linearly decodable.
- **Probe selectivity** (Hewitt & Liang 2019, *Designing and Interpreting
  Probes with Control Tasks*). Probe accuracy alone overstates what is
  represented; a control is required.
- **Binding** (Feng & Steinhardt 2024, *How do Language Models Bind Entities
  in Context?*, ICLR). How a model ties entities to their attributes is
  itself a mechanism to be found, not assumed.
- **Training-free verbalization** (Ghandeharioun et al. 2024, *Patchscopes*,
  ICML; Chen et al. 2024, *SelfIE*). A model can be asked to put its own
  activation into words.
- **The NLA recipe** (Fraser-Taliente, Kantamneni, Ong et al. 2026). Its
  released pairs were the Phase Two instruments.

## Models, data, splits

- **Primary model:** Gemma-3-12B-it under the existing lock
  `configs/model-lock-gemma3-12b.json`, hash-verified at the start of every
  run. Qwen2.5-7B-Instruct (`configs/model-lock.json`) is surveyed in Stage 0
  as a secondary family. Its numbers are reported but never gate anything.
- **Task:** the unchanged paired-program generator (`tasks.py`). Values are
  0–19, and both variables are always defined.
- **Splits:**
  - **Stage 0 probe training:** the first 192 calibration groups (768
    prompts).
  - **Stage 0 site selection:** the last 64 calibration groups (256 prompts).
  - **Stage 0 held-out check:** the 128 pilot groups (512 prompts).

  The calibration and pilot splits were used in Phase Two. Only their prompts
  and tokenization are reused here, never any behavioral outcome.
  `validation_a` and `validation_b` (256 groups each) have never been opened,
  and are reserved for Stage 2's locked validation.

## Stage 0 — where does the model hold both values? (survey)

- **Positions,** each on every prompt's own tokens:
  1. `program_end`: the token completing the program text;
  2. `query_variable`: the variable token in "What is x";
  3. `question_mark`: the token completing "What is x?";
  4. `user_end`: the token completing the whole user message;
  5. `answer`: the last assistant-prefix token, the Phase Two site.
- **Layers:** the output of every transformer block (`hidden_states[1..L]`).
  The last entry includes the final norm, and this is recorded.
- **Probes.** For each position, layer and variable, a least-squares
  one-hot linear classifier over the 20 values (ridge, dual form). Inputs
  are centred on the training mean. Ridge strength is chosen from a fixed
  grid on the selection split. Secondary: a scalar ridge regression, rounded
  to the nearest value.
- **Controls:**
  - a label-permutation control probe, whose accuracy should sit near chance
    (Hewitt & Liang's selectivity);
  - a copy baseline that predicts the variable's last literally assigned
    value;
  - the *computed* subset: prompts whose true value appears nowhere in the
    prompt text. These are reported separately, so that copying literals
    cannot pass for state.
- **Site selection (frozen).** `s*` is the (position, layer) maximizing
  `min(acc_x, acc_y)` on the selection split under the primary probe. Ties
  go to the higher computed-subset minimum *on the selection split*, then to
  the earlier layer, then to the earlier position. The held-out pilot split
  never influences selection.
- **Gate G0, to proceed to Stage 1.** At `s*`, on the held-out pilot split:
  `min(acc_x, acc_y) ≥ 0.80`, and the computed-subset `min(acc_x, acc_y) ≥
  0.50`. Otherwise Phase Three stops after Stage 0. Its finding would be that
  no surveyed site linearly exposes both values at this level. A nonlinear
  representation would remain possible and would be stated as untested.

**Predictions (not thresholds).** Both values are decodable at
`program_end` and `user_end` in middle layers. At `answer`, the queried
value is decodable but the other variable's less so (Phase Two's
descriptions dropped it). G0 passes on Gemma-3-12B.

## Stage 1 — can the model verbalize the state itself? (training-free)

If G0 passes, the activation at `s*` is patched, Patchscopes-style, into a
fixed inspection prompt that asks for a variable's value, and the model's
own greedy completion is read as an integer. The inspection prompts and
patch layer are frozen in a Stage 1 protocol *before* any Stage 1 forward.
That protocol is written after Stage 0 is read, and its rules are fixed
here:

- **Readout accuracy** per variable on the pilot split.
- **Controls:** the unpatched inspection prompt, and the activation of a
  different program (shuffled).
- **Gate G1:** Stage 1 "works" iff the readout accuracy's one-sided 95%
  lower bound (prompt-group bootstrap, 3,000 resamples) exceeds the
  shuffled control's accuracy by 0.2, for both variables.

**Stage 1 never blocks Stage 2.** It answers "can the model say it?"; Stage 2
answers "does the model use it?".

## Stage 2 — does editing the typed statement change exactly what it should?

Stage 2 runs if G0 passes, whatever G1's outcome.

- **Describer.** The Stage 0 probes at `s*` write `x is currently X; y is
  currently Y`.
- **Reconstructor.** A linear decoder fitted on the Stage 0 training split
  maps each variable's value to a direction: the class-mean difference
  vectors at `s*`.
- **The edit.** Editing the statement from `x is currently X` to `x is
  currently X′` patches `h′ = h + D_x(X′) − D_x(X)` at `s*`, and likewise
  for y.
- **Measured** on the pilot split for development; final numbers come from a
  locked validation on `validation_a` plus `validation_b`, opened once.
  1. Targeted flip: the x-query answer becomes `X′`.
  2. Integrity: the y-query answer is unchanged.
  3. The same for edits to y.
  4. A wrong-variable control: editing y must not move the x answer.
  5. A random direction with the same norm as the edit.

**Frozen success criteria for the locked validation,** for both
directions (x edits and y edits):

| ID | Criterion | Threshold |
|---|---|---|
| S1 | targeted flip rate | ≥ 0.50 |
| S2 | other-variable answer intact | ≥ 0.90 |
| S3 | wrong-variable and random controls move the target answer | < 0.10 |

The edit hypothesis is **supported at this site** iff S1, S2 and S3 all hold
on the locked validation. Edit sizes and the exact protocol are frozen in a
Stage 2 protocol before any Stage 2 forward. That protocol may not weaken
S1–S3.

## Explicitly out of scope

Training an English describer and reconstructor (LoRA) is a possible Stage 3.
It is proposed only if Stage 2's hypothesis is supported, and it needs its own
brief and approval. Changing the task, model families or thresholds after
results are read is also out of scope.

## Budget, stopping, honesty

- **GPU budget:** at most 24 GPU-hours for Stages 0–2 together, measured;
  exceeding it stops the phase.
- **Each stage runs once.** A failed gate closes the phase with that stage's
  report. A source bug permits only a regression test plus a visibly
  versioned rerun. Each stage's protocol is committed and pushed before its
  forwards, and records its predictions.
- **Every run** starts only when the GPU has no other active workload. Other
  people's jobs are never stopped.
- **Limits, stated in advance:**
  - Probe-written statements mention variables by construction; only the
    causal tests count as evidence.
  - Linear probes miss nonlinear codes.
  - One task family, one primary model.
  - Success would show that a typed state record at one site can be read
    and edited causally. It would not show general interpretability,
    faithfulness of free-form English, or anything about the released NLAs.
