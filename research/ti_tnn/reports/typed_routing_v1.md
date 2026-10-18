# A learned interpreter with calibrated abstention: reliable in distribution, not under a phrasing shift — 2026-10-18

**Protocol:** [`protocols/typed_routing_v1.md`](../protocols/typed_routing_v1.md), frozen at `fd0aa20` before
any run on the confirmation worlds, with the analysis (`evaluate()` in `bench/typed_questions.py`) and its
test. **Raw results:**
[`experiments/results/ti_tnn/typed_routing/`](../../../experiments/results/ti_tnn/typed_routing/).

## The question

In plans PR 4 and 5, the correct program was supplied. Here a learned interpreter (Jev's first role)
reads a controlled-language question, such as "what is the operator of the airport of the largest city of
country_3 ?", and proposes the program. The typed runtime runs it with learned modules. A threshold,
calibrated on a separate split, decides whether to answer or abstain.

Three questions:
- **End to end:** does the whole system answer as well as when the program is supplied?
- **Refusals:** does it refuse questions it should not answer?
- **Risk under shift:** does calibrated abstention keep accepted answers reliable when the phrasing
  changes?

## Answer

**In distribution, yes. Under a phrasing shift, no.**

- **In distribution:**
  - **Coverage and risk:** the system answered 98.2–100% of supported test questions. Across all 15
    units, 2 of 13,714 accepted answers were wrong. In every unit, the 95% upper bound on accepted error
    was 0.0073 or lower, under the plan's 1% bar.
  - **Refusals:** every ill-typed, unsupported, unknown-entity and ambiguous test question was refused.
  - **Supplied programs:** end-to-end accuracy matched them.
- **Under a phrasing shift** (phrasings and a question frame never seen in training):
  - **Coverage:** the system answered at most 5% of supported questions; 8 of 15 units answered
    nothing at all.
  - **Accepted answers:** 59 of the 100 it did accept were wrong.
  - **The 5% bar:** 7 of 15 units failed it, one at 18 wrong out of 18, and the prediction failed in all
    three worlds. That is the one prediction that failed, as the protocol said I expected.

**Why the shifted errors happen.** Most come from questions that should have been refused: an unseen
phrasing of an ambiguous or unsupported relation. For example:
- "tell me the principal city of country_36 ." was parsed as `capital_of`, with calibrated probability
  0.84;
- "tell me the firm that runs the airfield that serves X ." was parsed as `country_of, largest_city_of`,
  a well-typed chain that the runtime runs without complaint.

A calibration fit in distribution cannot see this. There, correct parses score near 1 and the few errors
score low, so the threshold lands anywhere from 0.77 to 0.997 depending on the init. Shifted questions
that happen to clear it are answered.

## Results

**Accepted error and coverage, by system.** In distribution (test), then under shift.

| System | Test coverage of supported questions | Test accepted errors | Shift coverage | Shift accepted errors |
|---|---|---:|---|---:|
| **`typed_specialists`** (the interpreter, the runtime and learned specialist modules) | 0.982–1.000 | **2 / 13,714** | 0–0.050 | **59 / 100** |
| `typed_shared_dense` (same interpreter, shared-network modules) | identical | identical | identical | identical |
| `typed_table` (same interpreter, table modules) | identical | identical | identical | identical |
| `answer_only` (same encoder, answer classified directly) | 0 (no threshold certifiable) | — | 0 | — |

**Accuracy without a threshold, and calibration error.**

| System | Test, no threshold | Shift, no threshold | Calibration error, test / shift |
|---|---|---|---|
| **`typed_specialists`** | 0.994–1.000 | 0.002–0.384 | ≤ 0.006 / 0.09–0.41 |
| `answer_only` | 0.57–0.63 | 0.15–0.32 | — |

- **Units.** 3 worlds (48 countries each) × 5 inits. Each unit has 878–950 supported test questions and
  53–57 of each refusal kind; the shift split matches. Ranges run over units.
- **The modules made no errors.** The three typed systems behaved identically: every error came from
  the interpreter, so the module choice made no difference.
- **Calibration error** is the expected calibration error of the calibrated probability. It is computed
  on questions the policy would answer before thresholding.

**Answer-only, by path family** (test, no threshold):

| Family | Accuracy |
|---|---|
| one step | 0.996–1.000 |
| non-cancelling 2-step | 0.11–0.20 |
| non-cancelling 3-step | 0.20–0.31 |
| round trips | 0.80–0.88 |

The round trips are a shortcut: "the capital of the country of the capital of X" is the capital of X. Its
calibration split never allowed a threshold certifying 1%, so it answered nothing.

**The interpreter under shift.** Exact program and entity on 0.001–0.383 of supported shifted questions.

## Predictions (frozen before any confirmation run)

| # | Prediction | Outcome |
|---|---|---|
| 1 | Interpreter exact on ≥ 0.99 of supported test questions (≥ 4 of 5 inits per world) | **Holds** (14 of 15 units at 0.999–1.000; world 400, init 1 at 0.9897) |
| 2 | Accepted-error 95% upper bound ≤ 1% with coverage ≥ 0.90 (≥ 4 of 5) | **Holds** (every unit: bound ≤ 0.0073, coverage ≥ 0.982) |
| 3 | ≥ 0.99 of each refusal kind refused on test (pooled per world) | **Holds** (all refused) |
| 4 | End-to-end within 0.01 of supplied-program accuracy (≥ 4 of 5) | **Holds** (lowest 0.994 against 1.000) |
| 5 | Answer-only below 0.5 on non-cancelling 2- and 3-step questions (every run) | **Holds** (0.11–0.31) |
| 6 | Under shift, accepted error ≤ 0.05 or nothing answered (≥ 4 of 5) | **Fails in every world:** 2, 3 and 3 of 5 units pass |

**Which are findings.**
- **By construction or by design** (the protocol said so beforehand):
  - prediction 3, for ill-typed and unknown-entity questions;
  - prediction 4, given 1;
  - prediction 5.
- **Predictions 1 and 2 are findings, but easy ones:** in-distribution controlled language is templated.
- **Prediction 6 is the informative outcome.**

**Shift results per world** (units passing / accepted errors):

| World | Units passing | Accepted errors in the failing units |
|---|---|---|
| 400 | 2 of 5 | 18/18, 4/6, 5/5 |
| 401 | 3 of 5 | 1/6, 3/4 |
| 402 | 3 of 5 | 2/2, 26/59 |

Every passing unit answered nothing under shift.

## What it means

- **Typed execution with a learned interpreter works where the language is familiar.** It certified a
  low-risk operating point at nearly full coverage. Every ill-typed question was refused, by the
  runtime's type check rather than by the interpreter.
- **Calibration does not protect against unfamiliar inputs.** The threshold is fit to in-distribution
  scores. An interpreter that maps an unseen phrasing onto a plausible well-typed program can be as
  confident as it is on familiar questions. The type check cannot catch a well-typed misreading, so
  type validity does not imply correctness here either.
- **Coverage collapsing is not the same as risk staying bounded.** Under shift the system answered almost
  nothing, yet over half of what it did answer was wrong. A gate on coverage alone would have hidden this.
- **What would address it:** an explicit test of whether the input is in distribution, separate from the
  interpreter's confidence; calibration on a split that includes shifted phrasings; or an interpreter that
  marks unknown words. None was tested here.

## The two in-distribution errors

Both came from world 400, init 1, where the threshold was 0.78, among the lowest of any unit. They were not
diagnosed further.

## Diagnosis after the run (outside the protocol)

I reran world 400, init 1 and world 402, init 4 and listed every accepted shifted question. Both
reproduced the recorded counts exactly: 18 accepted, all wrong; 59 accepted, 26 wrong.

**World 400, init 1:**
- 14 of the 18 errors were the held-out ambiguous phrasing ("the principal city of X"), parsed as
  `capital_of`;
- the other 4 were supported questions misparsed. "the airfield that serves X" became
  `country_of, capital_of`, and "the most populous city of X" became `capital_of, country_of`.

**World 402, init 4:**
- 13 errors were misparsed supported questions;
- 8 were ambiguous questions answered;
- 5 were unsupported questions answered.

## Limits

- **Controlled, templated language;** the shift is one kind (new words for the same relations).
- **The interpreter is the only learned decision-maker.** Entities are copied, and the types, relations
  and vocabulary are supplied.
- **Only end-to-end answer correctness is calibrated,** not the route-level confidence (which relation
  next).
- **Units are independent** conditional on the world and trained model. No bound is pooled across
  units.

## Execution

- **Source.** Committed as `fd0aa20` immediately before the run, with a clean tree. The run file does not
  record its commit.
- **Run.** 2026-10-18, finished 20:09 UTC, CPU only (4 threads, `nice`). It ran from the run directory's
  checkout of `main` with the frozen branch's script and the protocol's arguments.
- **Software.** PyTorch 2.10.0, Python 3.11.14, aarch64.
- **Verdict.** `verdict.json` was written by the frozen `evaluate()`; `make_tables.py` rebuilds it from the
  bundled runs.
- **Development pilot.** It ran on worlds 100–101 before freezing; its three fixes are disclosed in the
  protocol.

## After review (2026-10-18): what limits the interpreter, and what calibration added

An outside review found four things; each was checked independently before being recorded here.

**1. Unseen words collapse to one token, so some shifted questions are indistinguishable.**
- **The cause.** The vocabulary comes from training questions, and every unfamiliar word becomes `<unk>`.
  "tell me the nation that includes X ." (`country_of`) and "tell me the airfield that serves X ." (`airport_of`)
  become the same input, with the same copied entity.
- **The size.** Questions with identical inputs and entities but different required answers form 97, 94 and
  111 groups in worlds 400–402. Answering every supported shifted question would force at least 11.0%,
  9.9% and 12.0% errors before any learning question arises.
- **What a fix needs.** Character, byte or subword input would keep unfamiliar spellings apart. It would
  not teach what an unseen synonym means.

**2. A simpler control would have refused the whole shift.** Every supported shifted question contains an
unfamiliar non-entity word, and no in-distribution test question does. Rejecting such words would refuse
every shifted question and keep full in-distribution coverage. That is conservative refusal, not
interpretation. The frozen shift rule (accepted error ≤ 5%, or nothing answered) would count it as a pass.
A claim of competence under shift needs a coverage requirement too.

**3. The calibrated threshold was barely needed in distribution.**

| Policy, test split | Errors / answers | Units meeting the frozen 1% bound with coverage ≥ 0.90 |
|---|---:|---:|
| No threshold | 6 / 13,749 | 14 / 15 (4, 5 and 5 per world, so prediction 2's rule still holds) |
| Fitted threshold (reported) | 2 / 13,714 | 15 / 15 |

The supported conclusion is narrower than "calibration is needed for this operating point": the
interpreter was already accurate enough. The threshold removed four errors.

**4. The threshold search has no selection-corrected guarantee.**
- **The problem.** `choose_threshold` scans many cutoffs and keeps the largest whose single-cutoff 95%
  Clopper–Pearson bound meets the target. Selecting among cutoffs by their outcomes breaks that per-cutoff
  guarantee.
- **The size.** In a constructed case where every cutoff's true error is 1.01%, with 650 calibration
  examples, a simulation of 4,000 repetitions found an apparently certifiable cutoff 9.1% of the time.
  The review's exact calculation gave 9.03%. Either is above the nominal 5%.
- **What it does not affect.** The test-split bounds reported above are separate, fixed-policy evidence.
- **The fix.** Select the policy on one split and certify it on another untouched one, or use a
  simultaneous procedure (for example, Learn then Test).

**Also:** 4, 11 and 8 refusal question strings (worlds 400–402) occur in both training and calibration or
test. Supported program-and-entity instances were correctly separated. This is a small cleanup, not an
explanation of the shift result.

**What follows** (integration plan section 12): an interpreter robustness experiment separating three
things:
- new structure with familiar words;
- alternate wording that was taught;
- abstention on unfamiliar wording.

Its controls run first: no threshold, a validly certified raw-score threshold, the calibrated threshold,
and lexical-novelty rejection.
