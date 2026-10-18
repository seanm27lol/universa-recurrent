# Questions, a learned interpreter and calibrated abstention (frozen brief, integration plan PR 6)

**Status: FROZEN 2026-10-18 (UTC), before any run on the confirmation worlds.** Branch `ti-tnn-routing`,
stacked on the execution-records PR (#47).

## The question

Plans PR 4 and 5 supplied the correct program. Here a learned interpreter (Jev's first role in the plan)
reads a controlled-language question and proposes the program; the typed runtime runs it with learned
modules. A calibrated threshold then decides whether to answer. Three questions:
- **End to end:** does the system answer supported questions as well as when the program is supplied?
- **Refusals:** does it refuse questions it should not answer: ill-typed, unsupported, unknown entity,
  ambiguous?
- **Risk under shift:** does a threshold calibrated on one distribution keep accepted answers reliable
  when the phrasing shifts?

## Design

Code: `research/ti_tnn/integrated/routing.py`, `calibration.py`; `research/ti_tnn/bench/typed_questions.py`.

**Worlds.** `generate_world` with seeds **400, 401, 402**, unused by any earlier run, and 48 countries
(twice PR 4's 24). The extra size is needed so that a 1% risk bound can be certified at all: zero errors
in fewer than 299 answers cannot get a 95% upper bound below 1%.

**Questions.**
- **Instances.** Every well-typed path of length 1–3 from every start value is one instance, split by
  instance into train 30%, calibration 30% and test 40%.
- **Supported questions.** Each instance becomes one question in seen phrasings. A question is either
  all prefix ("the capital of the country of X") or all possessive ("X 's country 's capital"), never
  mixed, so every question has exactly one reading.
- **Questions to refuse:** ill-typed, unsupported, unknown entity and ambiguous, 6% of each split
  each.
- **Shift split.** Every test instance again, plus refusal questions, in phrasings and a frame never used
  in training.
- **Path families.** Paths with a round trip (`capital_of` then `country_of`) are their own family, as in
  PR 4.

**Systems.** All are trained on the same train split.

| System | What it is |
|---|---|
| **`typed_specialists` (gated)** | The interpreter, the runtime and learned specialist modules (PR 4) |
| `typed_shared_dense` | The same interpreter with shared-network modules |
| `typed_table` | The same interpreter with table modules (diagnostic: interpreter error only) |
| `answer_only` | The same encoder classifying the answer directly. It is trained on the train questions plus every atomic fact in every seen phrasing and frame |

**Training.**
- **Interpreter:** BiGRU encoder (width 48, hidden 64), entity head and attention decoder; 30 epochs,
  Adam 3e-3, word dropout 0.1, init seeds 0–4.
- **Modules:** as in PR 4.
- **Answer-only:** 30 epochs.

**Calibration** (`calibration.py`).
- **Event:** "the policy's answer is correct", on every question the policy would answer before
  thresholding.
- **Fit:** a Platt transform of the interpreter's score.
- **Threshold:** the largest-coverage threshold whose one-sided 95% Clopper–Pearson upper bound on
  calibration-split error is at most 1%.
- **Use:** fitted on the calibration split only, bound to the interpreter, registry and policy
  identities, and applied once to test and shift.

**Units.** Each (world, init) is a unit. Test questions are treated as independent draws conditional on
that unit, and bounds are never pooled across units.

## Development pilot (disclosed)

Worlds 100–101, inits 100–101, 4 units. Three problems were found and fixed before freezing:
- **Ambiguous questions.** The first version mixed prefix and possessive phrasings in one question, so
  some strings had two readings with different programs ("the country containing X 's capital"). That
  capped the interpreter at about 0.94–0.985. Questions now use one style.
- **Platt fit diverged** on separable calibration data (a = 54,745). A stronger prior and a line search
  fixed it.
- **The answer-only baseline was undertrained.** It answered only 0.55–0.65 of one-step questions,
  because it had seen each fact in one phrasing. It now sees every fact in every seen phrasing, and
  answers every one-step question correctly.

**Pilot results after the fixes,** for `typed_specialists`:

| Measure | Result |
|---|---|
| Interpreter exact match | 0.995–1.000 |
| Test coverage | 0.994–1.000, with zero accepted errors (upper bound 0.0034) |
| Every refusal kind | 100% refused |
| Answer-only on non-cancelling 2- and 3-step questions | 0.13–0.30 |
| Answer-only on round trips (a shortcut) | 0.82–0.84 |
| **Under shift:** interpreter exact match | 0.12–0.36 |
| **Under shift:** coverage | 0–0.026 |
| **Under shift:** accepted answers | 0/0, 3/9, 13/23 and 0/0 wrong |

## Predictions (recorded before any confirmation run; one-sided)

A prediction holds for a world when it holds on at least 4 of its 5 inits, unless stated otherwise.

1. **Interpreter competence.** Exact program and entity on at least 0.99 of supported test questions.
2. **Selective answering** (the plan's bar). On test, the 95% upper bound on accepted-answer error is at
   most 1%, with coverage of supported questions at least 0.90.
3. **Refusals.** Pooled over each world's 5 inits, at least 0.99 of each refusal kind is refused on
   test.
4. **End to end equals supplied.** Unthresholded end-to-end accuracy on supported test questions is
   within 0.01 of the supplied-program accuracy, with the same modules.
5. **Answer-only fails to chain.** Below 0.5 on non-cancelling 2- and 3-step test questions, in every
   run.
6. **Risk under shift.** Under the held-out phrasings, accepted-answer error is at most 0.05 (or nothing
   is answered).

**What I expect** (stated, not predicted): 1–5 hold, and 6 fails. In the pilot, two of four units
accepted a few shifted questions with high error. A threshold calibrated in distribution does not
transfer, even though coverage collapses.

**What holds by construction or design:**
- **3, for ill-typed and unknown-entity questions,** follows from the runtime's type check and entity
  binding, given a faithful parse.
- **4** follows from 1, given competent modules.
- **5** is expected, since answer-only never sees most chains.
- **1 and 2 are easy here:** in-distribution controlled language is templated.

**The informative outcomes are 2 and 6:** whether calibrated abstention certifies a low-risk operating
point, and whether it survives a phrasing shift.

## Limits

- **Controlled, templated language.** This is not natural language. The held-out phrasings are only
  one kind of shift.
- **The interpreter is the only learned decision-maker.** The types, relations and entity vocabulary
  are supplied; entities are copied, not recognized.
- **The calibration event is end-to-end answer correctness.** Route-level confidence (which relation
  next) is not calibrated separately.
- **No repair:** an ill-typed proposal is refused, never re-ranked.

**Run:**
`python research/ti_tnn/bench/typed_questions.py --worlds 400 401 402 --inits 0 1 2 3 4`, which also
writes `verdict.json` via `evaluate()`.
