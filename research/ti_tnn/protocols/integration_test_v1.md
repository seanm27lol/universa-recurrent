# Integration test: one question through recurrence, a checked update mid-run, and resume (DRAFT)

**Status: DRAFT for review, 2026-10-19. Nothing has been run and none of the new code exists yet.**
- **The numbers are proposals.** Every number here is a starting proposal. A bounded development pilot
  (see "Development pilot") may revise them before the protocol is frozen.
- **Decisions are open.** Choices that need a decision are marked **For review** and collected at the end.

This is item 3 of the [integration plan's section 12](../integration_plan.md#12-second-review-amendments-2026-10-18-after-prs-4150)
order: "question, interpreted bounded query, recurrence, pause, a checked module update, then resume under
the right version policy."

## A concrete example

A route-network world has airports, each with one onward connection (`connects_to`), and an operator for
each airport (`operator_of`). Someone asks:

> which is the first airport within 6 connections of country_12_city_0_airport that operator_3 operates ?

1. **Interpreting the question.** The interpreter turns this into a bounded query:
   - start: `country_12_city_0_airport`;
   - target: `operator_3`;
   - step and test relations: `connects_to` and `operator_of`;
   - budget: 6.

   The runtime type-checks the query before anything runs. A certified threshold decides whether to answer
   or abstain.
2. **Recurrence.** Recurrence runs under the active snapshot S0. After two connections, the run pauses and
   its continuation state is written out as JSON.
3. **An update while the run is paused.** An update changes one fact: `country_40_city_2_airport` is now
   operated by `operator_3`. That airport is the second one the paused run already visited and tested,
   against S0's facts. The update passes its gates and is activated, so the active snapshot is now S1.
4. **The resume question.** What should the resumed run answer?

| What the system does on resume | Answer | Its record |
|---|---|---|
| continue on S0 (**pin**) | the airport S0's facts lead to, four connections later | every application on S0: a valid S0 run |
| start again on S1 (**restart**) | `country_40_city_2_airport`, found at the second connection | every application on S1: a valid S1 run |
| keep the paused state, continue on S1 (**naive**) | the same airport as pin: the second airport was tested under S0 and is never tested again | applications from S0, then S1. The run finishes on S1 and reports S1, yet S1's facts give a different answer |

The naive row is the failure this test exists to rule out:
- **A mixed trace.** The record is not the execution of any single version.
- **A broken contract.** The answer contradicts the snapshot the run reports.
- **An answer from no version at all.** With two facts changed at once, one consumed before the pause and
  one after, naive continuation can return an answer that no single version of the system gives. One
  changed fact cannot do this: the run then ends with S0's outcome or S1's, depending on which side of the
  pause it falls. The test therefore includes two-fact updates.

Today `recurrence.resume` refuses any registry other than the one a state is bound to. So after an update, a
paused run currently cannot continue at all, and nothing yet says what it should do.

## Why this test, and what is new in it

Each component passed its own frozen protocol:

| Plan PR | Component | Protocol |
|---|---|---|
| 6 | learned interpretation with calibrated abstention | [typed_routing_v1](typed_routing_v1.md) and [interpreter_robustness_v1](interpreter_robustness_v1.md) |
| 7 | bounded recurrence with exact pause/resume | [typed_recurrence_v1](typed_recurrence_v1.md) |
| 8 | checked offline updates with atomic activation and exact rollback | [local_updates_v1](local_updates_v1.md) |

The seams between them have never been exercised together:

1. **Interpreter → query.** A parsed query must be type-checked and bound to real entities before it runs.
   The existing interpreters produce fixed relation paths, not until-queries with a start, a target and a
   budget.
2. **Continuation state ↔ versions.** A paused state is bound to `registry.identity`, which hashes the
   modules' *declared* version strings. A module whose parameters change in place after registration keeps
   its declared identity. `resume` would accept it, even though the snapshot digest (`snapshots.py`, #52)
   would not.
3. **Calibration ↔ modules.** The interpreter's calibration is bound to the registry it was fit with, and
   its event is final-answer correctness. An update therefore makes it stale, and
   `calibration_and_resources` blocks activation until it is refit.
4. **Update gates ↔ recurrent queries.** PR 8's behaviour gates are exhaustive over paths of up to three
   relations. A `connects_to` update can change until-queries up to the 16-iteration budget, which those
   gates never execute.

**The question.** When composed, do the components keep their guarantees at these seams? And under a
declared version policy, is every completed run's outcome one that a single snapshot produces, named in its
record and confirmed by an independent checker?

**What holds by construction and what is a finding.**
- **By construction:** much of this holds given pure modules and deterministic code. Examples: pin and
  restart reproducing uninterrupted runs, and stale states being refused. The protocol labels those
  predictions *construction checks*. They verify the implementation and are not reported as findings.
- **Findings:** interpreter and module competence on these queries, before and after updates; noninterference
  over full-length queries; the refit calibration's coverage; how often naive continuation breaks its
  contract or gives a no-version outcome.

## New code (written only after this draft is approved)

| File | What it adds |
|---|---|
| `integrated/sessions.py` | **Version policies**: `pin`, `restart` and `migrate` (below). Each wraps a recurrence state with the digest of the snapshot it runs on, and refuses to resume on any other. A deployment retains its active snapshot and history; pin resumes on the retained snapshot after verifying its digest. |
| `integrated/query_interpreter.py` | An interpreter for until-questions: #48's encoder and decoder, with a second entity-copy head (start and target) and budget outputs (below). It does not change `interpreters.py`, which is frozen for the robustness study. |
| `bench/integration_test.py` | The experiment. |
| `bench/integration_test_verify.py` | An independent verifier, in the form the #55 review asked for: complete certificate comparison, trace and record agreement without trusting the producer's own lists, binding to the frozen source, and a rebuild from a clean checkout. |
| `tests/test_ti_tnn_integration.py` | Success and failure cases, including every fault below. |

Frozen components (`recurrence.py`, `updates.py`, `update_checks.py`, `snapshots.py`, `calibration.py`,
`interpreters.py`) are used as they are. If one needs a change, it gets its own PR with regression tests
first.

## Design

### Worlds and units

- **Worlds:** `route_world` seeds **800, 801, 802**, 48 countries (about 170 airports), as in PR 7. No run
  configuration or protocol has used them; this was checked.
- **Inits:** **0–4**. A unit is one (world, init).
- **Development:** worlds 100–101 × inits 100–101 only.

### Questions

- **Until-questions** come from a small set of templates (**For review**: whether questions state their
  budget). For example:
  - "which is the first airport within N connections of S that T operates ?";
  - "starting from S , name the first airport run by T within N connections .";
  - the same without "within N connections", with the default budget of 16.

  N runs from 1 to 16. The budget in the question then matters: the same start and target can be found
  within 6 connections but not within 3.
- **Refusals**, each with its expected handling:
  - **ill-typed start:** S is a city or a country. The parse is complete, but the runtime's `query_problems`
    rejects it;
  - **unknown entity:** an airport or operator not in the world;
  - **unsupported condition:** "… whose runway is longest …";
  - **out-of-contract budget:** N > 16.
- **Sampling** follows the robustness study's independent design:
  - **training:** a fixed share of (start, target, budget) instances;
  - **calibration and test:** independent draws, with replacement, from declared distributions over the
    held-out instances;
  - **variants:** matched variants are never pooled as independent observations.
- **Out of scope:** unfamiliar wording (already measured by the robustness study) and fixed-path questions.

### Interpreter

The learned query interpreter (**For review**: learned or deterministic):
- **Encoder:** #48's word encoder: training vocabulary, `<unk>` otherwise, entity dropout 0.1.
- **Outputs:** two copy heads, one for the start and one for the target. The decoder emits the step and test
  relations, a budget token (1–16 or `default`), then `<end>`; or `<unsupported>`.
- **Training:** a fixed recipe: the robustness study's 1,500 Adam updates of 64, learning rate 3e-3. The pilot
  may change only the number of updates.
- **Score:** the product of chosen-step and copy probabilities.
- **Certification:** fixed-sequence on the calibration split, with the robustness study's grid, risk and
  level.
- **Calibration event:** the system's final outcome is correct. That means the reference answer, or the
  reference's stopping reason when it abstains. This event depends on the modules, so it is bound to the
  snapshot.

Two conditions are always reported separately:
- **supplied query:** the reference query, with no interpreter;
- **interpreted query:** questions through the learned interpreter.

The supplied condition isolates execution and versioning from interpretation.

### Modules

- **Learned specialists:** PR 4/7's recipe (one network per relation, `connects_to` included), trained per
  init on atomic facts.
- **Table modules:** a diagnostic system that isolates the version machinery from learned error.

### Updates

- **Draws:** per world, 12 updates:
  - 4 operator changes, in which an airport gets another operator;
  - 4 route changes, in which an airport connects onward to another airport;
  - 4 two-fact updates: two operator changes on airports that at least one query visits in turn. Only these
    let naive continuation produce an outcome that no single snapshot gives.
- **Choosing them:** draws are made from the facts that change at least one until-query outcome, so that
  every update is consequential.
- **For each update and init:**
  1. **Candidate:** fine-tune the affected specialist on the updated facts (PR 8's `finetune` recipe).
  2. **Calibration:** refit the interpreter's calibration under the candidate. The calibration questions are
     the same; their correctness labels come from the updated world.
  3. **Gates:** check with PR 8's `Checker`, all mandatory gates.
  4. **Activation:** activate through `Deployment` and its `Authority`.
- **Added check, run at the integration level and not as a change to PR 8's gates:**
  - **query-level noninterference:** after activation, every until-query outcome under S1 equals the
    updated world's reference. This covers every (airport, operator) pair at the default budget, and checks
    the answer, the stopping reason and the visited path.

### Episodes: pause, update, resume

- **Episode queries, for each update:**
  - every query whose reference outcome or visited path differs between the two worlds;
  - a fixed random sample of 200 that do not differ.
- **Each episode query:**
  1. Run it under S0 and pause after every possible iteration k: from 0 (before the first step) up to the
     uninterrupted run's length.
  2. Write the state out and read it back as JSON.
  3. Activate the checked update. This happens once per update; every paused state of that update spans the
     activation.
  4. Resume under each policy and under the naive control.
- **Both conditions.** Episodes run with supplied queries, and with the test questions whose interpreted
  query the certified policy accepted.
- The pilot measures the cost of this. If it is too large, the protocol fixes a sample of pause points before
  freezing.

### Version policies

| Policy | Rule on resume after an activation | Expected outcome | Answers reflect |
|---|---|---|---|
| `pin` | continue on the snapshot the run started on, retained by the deployment, after verifying its digest | identical final state to the uninterrupted S0 run | the facts when the question was asked |
| `restart` | discard the paused state and run from the start on the active snapshot | identical to a fresh S1 run | the facts when the answer is given |
| `migrate` | continue on S1 only if no application in the paused prefix consumed an input whose output the update record declares changed; otherwise restart | the same answer, stopping reason and visited path as a fresh S1 run | the facts when the answer is given, at lower cost |
| `naive` (a control, never deployed) | continue on S1 regardless | a mixed trace; sometimes a contract violation; with a two-fact update, sometimes a no-version outcome | no single snapshot |

Three failures are distinguished, from weakest to strongest:
- **Mixed trace:** a completed run whose applications come from more than one snapshot. This happens whenever
  naive continuation applies a module after the activation.
- **Contract violation:** the outcome (answer, stopping reason or visited path) differs from the uninterrupted
  run on the snapshot the run reports finishing on.
- **No-version outcome:** the outcome differs from the uninterrupted run on every snapshot whose modules it
  applied. A single-fact update cannot cause one, so only two-fact updates can show it.

`migrate` relies on the update record's declared changes being the only changes. The gates check that
claim exhaustively for paths of up to three relations, and the query-level check above covers the until-queries.
The analysis also reports how many episodes were eligible to migrate, so the rule is never vacuous.

**For review:** which policy the deployment declares as its default. The recommendation is `restart`, with
`migrate` as an optimisation that must be equivalent to it and `pin` for audit and replay.

### Faults at the seams (each must be caught)

| Fault | Expected handling |
|---|---|
| A paused state resumed on S1 through `recurrence.resume` | refused: different registry identity |
| A module's parameters changed in place while paused, its declared identity unchanged | `pin` refuses: snapshot digest mismatch. `recurrence.resume` alone would accept it; that is the reason for `sessions.py` |
| Activation with the calibration fit under S0 | blocked: `calibration_and_resources` is insufficient evidence |
| `migrate` forced on an ineligible episode | refused, and restarted |
| A candidate whose record understates its changes, so a migration would rely on a false declaration | rejected at the gates: `intended_behavior` and `identity_and_scope` |
| Rollback while a run is paused | a pinned run continues on S0; a restarted run uses the restored snapshot |
| A record naming one snapshot but containing applications from two | flagged by the independent checker |
| A stale update record, whose base is no longer active, while a run is paused | refused activation; the paused run is unaffected |

## Proposed predictions (one-sided; per unit, a world holds at ≥ 4 of 5 inits, a prediction at every world)

| # | Prediction | Kind |
|---|---|---|
| 1 | **Interpretation.** The interpreted condition is certified, with test coverage ≥ 0.90 at a 95% accepted-error bound ≤ 1%, and abstains on ≥ 0.95 of refusal questions | finding |
| 2 | **Modules before updates.** Every supplied until-query's outcome (answer and stopping reason) matches the reference. This replicates PR 7 on fresh worlds | finding |
| 3 | **Updates.** Every learned update passes every mandatory gate, activates, and passes query-level noninterference; rollback restores S0 exactly | finding (gates and rollback mechanics are construction checks; competence after fine-tuning is the finding) |
| 4 | **Calibration after the update.** The refit calibration is certified with coverage ≥ 0.90 at the same bound | finding |
| 5 | **Policies.** At every pause point of every episode: `pin` equals the uninterrupted S0 run; `restart` equals the fresh S1 run; `migrate` equals the fresh S1 run whenever it migrates | construction check (for `migrate`, given an activated update, whose gates verified that its declared changes are its only atomic changes) |
| 6 | **The test can tell policies apart.** In every world, naive continuation produces contract violations and at least one no-version outcome (from a two-fact update). The independent checker flags every mixed trace and no policy record | non-vacuity requirement plus construction check |
| 7 | **End to end.** With interpreted questions, before and after the update, accepted outcomes meet prediction 1's bounds under the declared default policy | finding |
| 8 | **Faults.** Every fault above is handled as stated | construction check |

Also reported, and judged by no prediction:
- **naive-continuation rates:** contract violations and no-version outcomes, by update kind and pause point;
- **migration eligibility:** the share of episodes eligible to migrate;
- **cost:** module applications under `migrate` against `restart`;
- **continuation state:** its size;
- **accounting:** every refusal and every interpreter failure, in the denominators.

## Analysis and verification

`bench/integration_test_verify.py` rebuilds every reported number from the run's records. It trusts no
stored summary, and it checks:
- **Studies:** each world's question set and update draws are regenerated from their seeds.
- **Certificates:** complete certificates are re-derived from the decision records: every step, the stop
  reason, alpha and delta.
- **Recurrence records:**
  - every application's module identity must belong to the snapshot that the record names;
  - every outcome is re-derived with `reference_until` on that snapshot's world.
- **Policy equalities:** each policy's result is compared with an uninterrupted re-run, recomputed from the
  stored states and not taken from the producer's flags.
- **Source binding:** the run's commit and producer-file hashes must be the frozen ones.
- **Inventory:** exactly the declared worlds, inits, updates and episodes.

The bundle holds records for every unit. Anything too large for git becomes a release asset whose sha256 is
committed first, as #55 did.

## Interpretation limits

- **Seams, not scale.** It tests whether specified components compose without silently breaking their
  guarantees, in a finite synthetic world with supplied types, relations and entities.
- **No general claims.** It makes no claim about natural language, other model families, or the
  safety of updates in general.
- **Construction checks are not findings.** They verify this implementation.
- **"Right" is a declared contract.** The right version policy is the one whose declared consistency
  guarantee the system keeps. Whether answers should reflect question-time or answer-time facts is a design
  decision, not an empirical result.
- **`migrate` depends on complete update records.** It is correct only if update records declare every
  changed fact. That holds here because the gates verify every module's atomic outputs exhaustively in a
  finite world, and activation requires them to pass. It would not hold in a domain where exhaustive
  checking is impossible.
- **One process.** Pause and resume happen in one process with fresh copies, as in PR 7. A real
  cross-process or cross-machine resume is not tested.

## Development pilot (bounded, before freezing)

This uses worlds 100–101 × inits 100–101 only, on CPU (four threads, low priority), with an allowance of
about 4 hours of wall time. The pilot measures:
- **feasibility:** whether the query interpreter is learnable with the fixed recipe, and its coverage;
- **cost:** fine-tuning and checking per update; episodes, and the cost of every pause point;
- **the update draws:** whether operator and route updates both yield episodes eligible and ineligible to
  migrate;
- **non-vacuity:** whether two-fact updates yield no-version outcomes under naive continuation in these
  worlds.

**What it may revise:**
- question templates and sizes;
- interpreter updates;
- pause-point sampling;
- the episode sample size.

**What it may not revise:**
- the predictions' kinds;
- the bars, without explicit review;
- the confirmation worlds.

Pilot results are disclosed in this file's history before the freeze.

## Decisions for review

1. **Interpreter.** Choose between:
   - **learned** (recommended): it tests the calibration seam and the end-to-end condition;
   - **deterministic grammar:** it isolates versioning and drops the calibration seam.
2. **Explicit budgets** ("within N connections") in questions (recommended). They make "bounded" part of the
   interpretation.
3. **Default version policy:** `restart` (recommended), with `migrate` required to be equivalent and `pin`
   for audit.
4. **Updates:** per world, 4 operator changes, 4 route changes and 4 two-fact updates, all chosen to be
   consequential.
5. **Episode set:** every affected query, plus 200 unaffected ones, at every pause point (subject to the
   pilot's cost measurement).
6. **Bars:** coverage ≥ 0.90 at a 1% bound, refusal ≥ 0.95, and exact outcomes for supplied queries. These
   follow PRs 6–8 and the robustness study.
7. **Scope exclusions:** fixed-path questions, unfamiliar wording and cross-process resume.
