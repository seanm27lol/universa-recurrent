# Integration test: one question through recurrence, a checked update mid-run, and resume (DRAFT, revision 3)

**Status: DRAFT, revision 3, 2026-10-22.**
- **What revision 3 adds:** the development pilot's results, under "Development pilot", and the changes it
  proposes for the freeze, which are pending review.
- **No study world has been used.** Nothing has been fitted or run on worlds 800–802.
- **Revision 2** (2026-10-19) is the specification below. It is unchanged except where this file says
  otherwise.
- **What revision 2 records.** It applies the review of revision 1 (`9fb811f`), including its decisions on
  the open choices, collected under "Decisions".
- **The code it specifies** is in three groups:
  - **the prerequisite repairs:** the certificate and checker interface, and the complete serving context;
  - **the deterministic table-module fixtures;**
  - **the independent trace reconstruction.**

  All three come before any learned fit.
- **Before freezing:** a bounded development pilot (below) settles the sample sizes, the threshold
  feasibility and the pause-sampling cost.
- **Reserved:** worlds 800–802 stay untouched until then.

This is item 3 of the [integration plan's section 12](../integration_plan.md#12-second-review-amendments-2026-10-18-after-prs-4150)
order: "question, interpreted bounded query, recurrence, pause, a checked module update, then resume under
the right version policy."

## A concrete example

A route-network world has airports, each with one onward connection (`connects_to`), and an operator for
each airport (`operator_of`). Someone asks:

> which is the first airport within 6 connections of country_12_city_0_airport that operator_3 operates ?

1. **Interpreting the question.** The interpreter turns it into a bounded query:
   - start: `country_12_city_0_airport`;
   - target: `operator_3`;
   - step and test relations: `connects_to` and `operator_of`;
   - budget: 6.

   The query is type-checked and its budget checked against the contract (1–16) before anything runs. The
   serving context's certificate decides whether the system answers.
2. **Recurrence.** The run starts on the active serving context C0 and pauses after two calls to `step`.
3. **An update while the run is paused.** The update changes one fact: `country_40_city_2_airport`, the
   second airport the run already tested under C0, is now operated by `operator_3`. The update passes its
   gates and becomes the active context C1.
4. **The resume question.** What should the resumed run answer?

| On resume | Answer | Its record |
|---|---|---|
| **pin** (continue on C0) | the airport C0 leads to, four connections later | every application from C0: a valid C0 run |
| **restart** (start again on C1) | `country_40_city_2_airport`, at the second connection | every application from C1: a valid C1 run |
| **naive** (keep the state, continue on C1) | the same airport as pin; the second airport was tested under C0 and is never tested again | applications from C0, then C1, with no justification. It reports finishing on C1, but C1 gives a different answer |

**What `recurrence.resume` does today.** It already refuses a registry other than the one a state was
started with, and it can continue a paused run against a retained C0 registry. What is missing is the
choice of which serving context a resumed run should use, and the verification that it used it.

## Why this test, and what is new in it

Each component passed its own frozen protocol:

| Plan PR | Component | Protocol |
|---|---|---|
| 6 | learned interpretation with calibrated abstention | [typed_routing_v1](typed_routing_v1.md), [interpreter_robustness_v1](interpreter_robustness_v1.md) |
| 7 | bounded recurrence with exact pause/resume | [typed_recurrence_v1](typed_recurrence_v1.md) |
| 8 | checked updates with atomic activation and exact rollback | [local_updates_v1](local_updates_v1.md) |

The review of revision 1 found that the seams between these components are incompletely versioned:

1. **The calibration interface.**
   - `fixed_sequence_certify` returns a plain dictionary, while the update checker expects a `Calibration`
     with `binding_problems`.
   - `Checker.check` calls the calibration gate without a policy version, so the gate compares against `""`.
     A certificate honestly bound to a non-empty policy version would therefore fail.
   - The gate passes the certificate's own interpreter identity back to it, so it never establishes which
     interpreter will actually serve.
   - Nothing establishes that a fixed-sequence certificate succeeded for the intended event and sampling
     distribution.
2. **Activation and rollback** swap only the registry snapshot. A certificate's identity is sealed inside the
   check result, but the certificate is never installed, retained or restored with the snapshot.
3. **The behaviour gates** compare the declared changes with the difference between the two *reference
   worlds*. A candidate that also corrects a pre-existing model error outside those changes passes every gate.
   Its actual atomic outputs then differ from S0's at an undeclared input. A migration rule built on the
   declared changes is therefore unsound; see the counterexample under "Migration".
4. **Continuation state** is bound to `registry.identity`, which hashes the modules' declared version
   strings. A module changed in place after registration keeps its declared identity; the snapshot digest
   would detect the change.
5. **The bounded query.** `query_problems` does not enforce a budget contract. `pause_after` counts calls to
   `step`, while `steps_used` counts connections, and the two differ at termination. `reference_until`
   returns only an answer and a stopping reason, not the visited path or the applications.

**The question.** When composed, do the components keep their declared behaviour through an update? Under a
declared version policy, is every released result produced by one complete serving context, or justified as
exactly equivalent to it? Is that confirmed by an independent checker?

## Definitions

### The serving context

A **serving context** is everything that determines what the system releases for a question:

> snapshot (modules) + query certificate + interpreter identity + query-policy and correctness-event
> versions + world identity

- **Identity:** its identity is a hash over all five.
- **Verification:** `verify()` requires that:
  - the snapshot still matches its digest;
  - the certificate is bound to that digest and to the same interpreter, policy, event and distribution.
- **Activation installs a complete context.** Rollback restores the previous complete context. Neither ever
  swaps a snapshot alone.
- **Pinned sessions** keep their own context. A pinned session holds its context even after a rollback
  removes it from the deployment's history.

### The query certificate

A **query certificate** (`integrated/certificates.py`, versioned) holds:
- **the complete fixed-sequence result:** the grid, every step's accepted count, error count, p-value and
  decision, the selected threshold, the stop reason, alpha and delta;
- **bindings:** the interpreter identity, the snapshot digest, the query-policy version, the
  correctness-event version and the sampling-distribution identity;
- **the calibration records** it was computed from: item, raw score and correctness.

The update checker receives the **expected** serving context from the deployment: the interpreter it will
actually serve, and the policy, event, distribution and grid. It compares the certificate against that
context, never against the certificate's own fields. It requires:
- the certificate is bound to the candidate snapshot's digest;
- it re-derives exactly from its own records;
- it certified a rule.

The older `Calibration` path is kept unchanged, so the PR 6 and PR 8 results rebuild exactly as before.

### Completed results and refusals

**Completed semantic result:** one of these, released by the system:
- an airport answer;
- a correct-or-incorrect **negative terminal outcome** of the bounded query (`dead end`, `cycle` or
  `budget exhausted`).

In recurrence these negative outcomes are represented as abstentions, but they are results, not refusals.

**Policy refusal:** the system declines to release a result. The reasons are:
- insufficient confidence (below the effective threshold);
- no certificate;
- unsupported question;
- unknown entity;
- invalid query (ill-typed, or a budget outside 1–16);
- a module that produced no valid output.

**Measures:**

| Measure | Definition |
|---|---|
| Coverage | released results ÷ supported requests |
| Accepted error | incorrect results ÷ released results |
| Refusal performance | the abstention rate on each category of invalid or unsupported request, reported separately |
| Exact parsed-query accuracy | the start, target, relations and budget all match the reference |

Exact parsed-query accuracy is reported because a wrong query can coincidentally produce the right negative
outcome.

**Correctness event** (versioned): a released result is correct when its answer, or its negative terminal
outcome, equals the reference outcome of the reference query under the context's world.

### Which certificate governs a released result

| Policy | Certificate that governs release |
|---|---|
| pin | C0's |
| restart, migrate | C1's |

A question admitted under C0 and released under C1 has passed both gates. Two conditions hold:
- the interpreter and its raw score are unchanged;
- so are the eligible-question set and the threshold grid.

Its effective threshold is then max(t0, t1). C1's fixed-sequence result covers it at a grid point no looser
than t1. The record states the effective threshold and the C1 certificate step that supports it. This argument
does not cover a selection based on old module outcomes; none is used.

### Actual behavioural difference

For every snapshot, the **atomic-output table** records each relation's output on every valid input. It is
finite, contains no weights, and is recorded with the run.

The **actual difference** between S0 and S1 is every (relation, input) whose outputs differ. It is bound to
both snapshot digests and both table digests, and the independent checker re-derives it from the recorded
tables. Its entries are classified, and the classes are reported separately:
- **intended:** a declared world change;
- **correction:** outside the declared changes, the new output is right and the old output was wrong;
- **regression:** anything else. A candidate with a regression fails `unaffected_behavior` and never
  activates.

**The counterexample behind this definition** was found in review and reproduced with table modules.
- **The setup:**
  - the route is S → a → b → c, and the target is T;
  - S0's modules say a = U, b = U, c = T;
  - the old reference world already says a = T, so S0 is wrong at a;
  - the update declares only b → T;
  - the candidate correctly gives a = T, b = T, c = T.
- **The outcomes:** both behaviour gates pass, but the actual difference contains a and b.

| Run | Answer |
|---|---|
| uninterrupted S0 | c |
| fresh S1 | a |
| continuing after S0 tested a | b |

A migration rule using only the declared change would migrate here, and wrongly.

**"One fact" means one actual atomic change.** "A single-fact update cannot produce a no-version outcome"
holds for one *actual* atomic change, not merely one declared world change.

### Migration: provenance and semantic equivalence

Two properties are kept apart:
1. **Execution provenance.** Each application records the snapshot that actually supplied it. Prefix calls
   keep their original S0 identities; they are never relabelled as S1 calls.
2. **Semantic equivalence.** The reused prefix and the final result are equivalent to executing the query on
   S1.

`migrate` continues on S1 only if no prefix application consumed a (relation, input) in the actual
difference S0 → S1. It then records:
- a **migration justification:** the difference's identity, the eligibility check and the
  revalidation;
- **the revalidation itself:** S1's table gives the recorded output for every prefix application.

A valid migration can therefore have **mixed provenance with certified S1 equivalence**. The checker rejects
**unjustified version mixing**: a record whose applications come from two snapshots without a valid
justification. Naive continuation always produces such a record once it applies a module after the
activation.

**Migration's cost** counts:
- verifying the snapshot;
- looking up the prefix's inputs in the difference;
- revalidating the prefix;
- the suffix's module applications.

Revision 2 called migration "potentially" cheaper than restart. The pilot measured that it is **not cheaper
here**:
- **Applications.** Revalidation re-applies S1's module to every prefix input, so the suffix's applications
  plus the revalidations equal restart's applications exactly.
- **Lookups** are extra.

See "Development pilot".

### Three failures of naive continuation

- **Unjustified mixing:** applications from two snapshots with no valid migration justification.
- **Contract violation:** the outcome differs from the uninterrupted run on the snapshot that the run reports
  finishing on.
- **No-version outcome:** the outcome differs from the uninterrupted run on every snapshot that supplied an
  application. One actual atomic change cannot produce this; two can.

**The deterministic witness**, in every world's fixture suite (with table modules):
- **The setup:**
  - the route S → a → b → c is found in the world;
  - c's operator is the target T in both versions;
  - S0 gives a = U and b = U; S1 gives a = T and b = T.
- **The outcomes:**

| Run | First target airport |
|---|---|
| S0 | c |
| S1 | a |
| naive continuation after testing a under S0 | b |

  The witness guarantees that the instrument can detect a no-version outcome. Random draws then measure how
  often one occurs.

### Pause indices and the budget contract

- **Pause index.** Index k means the run is paused after exactly k calls to `step`, counting the call that
  terminates it. That call does not increase `steps_used`. If the uninterrupted run makes n calls, a paused
  run has k from 0 to n − 1; at k = n the run has finished, and nothing is resumed.
- **The budget contract** is an integer from 1 to 16. It is validated and bound into the query before
  execution, separately from `query_problems` (which does not check it). A question asking for more than 16
  connections is refused as an invalid query.

## Design

### Worlds and units

- **Worlds:** `route_world` seeds 800, 801, 802, 48 countries (about 170 airports). They are unused and
  reserved.
- **Inits:** 0–4. A unit is one (world, init).
- **Development:** worlds 100–101 × inits 100–101.

### Questions

- **Primary until-questions** state an explicit budget N from 1 to 16:
  - "which is the first airport within N connections of S that T operates ?";
  - "starting from S , name the first airport run by T within N connections .".
- **A diagnostic, reported separately:** omitted-budget questions with the default budget 16.
- **Refusal categories,** each reported separately:
  - an ill-typed start (a city or a country);
  - an unknown airport or operator;
  - an unsupported condition;
  - a budget over 16.
- **Training:** a fixed share of (start, target, budget) instances.
- **Calibration and test:** independent draws, with replacement, from declared distributions over held-out
  instances.

### Interpreter (learned)

- **Architecture:** #48's word encoder and decoder, with two entity-copy heads (start and target) and budget
  outputs 1–16 (or `default`).
- **Training:** a fixed recipe, the robustness study's 1,500 Adam updates of 64 at a learning rate of 3e-3.
  The pilot may change only the number of updates.
- **Score:** the product of chosen-step and copy probabilities.
- **Two conditions, reported separately:**
  - **supplied query:** no interpreter;
  - **interpreted query.**
- **Table modules** remain a diagnostic system.

### Modules

- **Learned specialists:** PR 4/7's recipe (one network per relation, `connects_to` included), trained per
  init on atomic facts.
- **The S0 atomic-correctness record.** S0's atomic correctness against the reference world is checked
  exhaustively and recorded per unit. Migration does not depend on it, because it uses the actual
  difference.

### Updates: twelve independent branches

- **Branches.** For each unit, 12 updates, each an **independent branch S0 → S1ᵢ** from the same S0:
  - 4 operator changes;
  - 4 route changes;
  - 4 two-fact operator updates.
- **The draws are fixed from the world seed** before any calibration or test outcome is inspected. They come
  from facts that change at least one until-query outcome.
- **For each branch:**
  1. **Candidate:** fine-tune the affected specialist (PR 8's `finetune`).
  2. **Certificate:** compute S1ᵢ's atomic table and its actual difference from S0, then build S1ᵢ's query
     certificate. The certificate uses the same IID calibration questions with labels from the updated world,
     and is bound to S1ᵢ's snapshot digest and the expected serving context.
  3. **Check:** run the repaired `Checker`, all mandatory gates, with the expected context.
  4. **Activate:** activate the complete context C1ᵢ through the serving deployment.
  5. **Roll back to C0** before the next branch, and verify the rollback.

### Certificate scope

The guarantee is **per snapshot**:
- **The claim.** Each of a unit's 13 serving contexts (C0 and the 12 C1ᵢ) is certified at risk 1% and
  level 0.05 on its own.
- **No simultaneous claim** is made across all 13. A simultaneous guarantee would split the error
  probability 13 ways. That raises the minimum accepted calibration observations at zero errors from 299 to
  554, and is not adopted.
- **Every certificate counts.** Every C1ᵢ certificate is reported, including uncertified ones.

### Evaluation populations, kept separate

| Population | What it is for | How it is judged |
|---|---|---|
| **IID held-out test questions** (independent draws) | coverage, accepted error and refusal rates | binomial bounds, per serving context |
| **Episodes** (affected queries, sampled unaffected queries, every pause index) | correctness audits of the version machinery, and descriptive rates of naive continuation's failures | counts and exact requirements. Never pooled into a binomial sample. The test certificate is never transferred to this cohort |

**Episodes for each branch:**
- **The queries:**
  - every query whose S0 and S1ᵢ outcomes, visited paths or applications differ;
  - a fixed random sample of 200 that do not differ.
- **The pause indices:** every index k from 0 to n − 1, for every policy and for naive continuation.
- **Both conditions:**
  - supplied queries;
  - accepted interpreted test questions.
- **The pilot** measures the cost. If the cost is too high, it fixes a pause-index sample before freezing.

### Construction and integration checks across every budget

These check recurrence, budgets, records and version handling, not learned competence.
- **The atomic gates establish atomic correctness.** PR 8's behaviour gates include every length-one path on
  every valid input, so a passing candidate is atomically correct throughout this finite world. Given pure
  deterministic modules and correct recurrence, every until-query then follows the correct atomic
  transitions.
- **The exhaustive check after each branch:** every (start, target) pair at **every budget from 1 to 16**.
  Each must equal the reference outcome, and the independent traversal's visited path and applications. Each
  budget is executed, not derived.

## Predictions (one-sided)

**Construction checks and injected faults:** 100% in every exercised case, in every unit. There is no
allowance.

| # | Check | Requirement |
|---|---|---|
| C1 | **Policies.** At every pause index of every episode: pin equals the uninterrupted C0 run; restart equals the fresh C1ᵢ run; migrate equals the fresh C1ᵢ run whenever it migrates, and restarts otherwise | every case |
| C2 | **Records.** Every policy record passes the independent checker. Every naive record that applied a module after the activation is flagged as unjustified mixing; no policy record is | every case |
| C3 | **Non-vacuity.** The witness yields a no-version outcome under naive continuation, and a correct pin, restart and migrate, in every world. Every branch has both eligible and ineligible migration episodes where they exist | every world |
| C4 | **Complete contexts.** Activation installs C1ᵢ complete; rollback restores C0 complete. A pinned session keeps C0 through any later rollback | every branch |
| C5 | **The exhaustive check.** Every query outcome at every budget matches, under C0 and under every C1ᵢ | every branch |
| F | **Every injected fault** (below) is handled as stated | every case |

**Learned competence:** a world holds at ≥ 4 of 5 inits, and a prediction holds when every world does.
Rejected candidates and uncertified units stay in the denominators.

| # | Prediction |
|---|---|
| L1 | **Interpretation:** C0 is certified; on IID test questions coverage is ≥ 0.90 at a 95% accepted-error bound ≤ 1%; abstention is ≥ 0.95 on every refusal category |
| L2 | **Modules:** S0 is atomically correct, and every supplied until-query outcome matches the reference at every budget |
| L3 | **Updates:** all 12 branches pass every mandatory gate and activate; every C1ᵢ certificate certifies with coverage ≥ 0.90 at the same bound |
| L4 | **End to end:** the same unit meets L1–L3, and its restart-policy results on the IID test questions meet L1's bounds under C0 and every C1ᵢ |

**Also reported:**
- exact parsed-query accuracy;
- the omitted-budget diagnostic;
- the actual-difference classes;
- naive continuation's three failure rates, by update kind and pause index;
- migration eligibility, and its counted costs against restart;
- the continuation-state size.

## Injected faults

| Fault | Required handling |
|---|---|
| Resuming a C0 state on C1 through `recurrence.resume` | refused: different registry identity |
| A module's parameters changed in place while paused, its declared identity unchanged | pin refuses: the snapshot digest does not match |
| A certificate bound to S0 supplied for C1 | the calibration gate fails: wrong snapshot |
| A certificate whose interpreter, policy, event or distribution differs from the expected context | the calibration gate fails |
| A certificate whose stored result does not re-derive from its records, or which certified no rule | the calibration gate fails |
| Activating with a check result whose sealed certificate differs from the context being installed | refused |
| Migrating when the prefix consumed an input in the actual difference (including a correction outside the declared changes) | refused, and restarted |
| A migration justification for a different snapshot pair or difference | refused by the checker |
| A rollback while a run is paused | pin keeps C0; restart and migrate use the restored active context |
| A rollback that removes C0 from the history while a pinned session holds it | the pinned session keeps C0, verified by digest |
| A stale update record (its base is no longer active) | refused activation; paused runs are unaffected |
| A query budget of 0, 17 or a non-integer | refused before execution |

## Analysis and independent verification

`bench/integration_test_verify.py` and `integrated/session_verification.py` share one rule: they never call
`recurrence.py`, `sessions.py` or the producer's checker. They read:
- the run's records;
- every snapshot's atomic-output table;
- the regenerated reference worlds and question sets.

They check:
- **Traversal:** their own traversal re-derives every outcome, visited path and application sequence, from a
  snapshot's table and from the reference world.
- **Records:**
  - every application's provenance and outputs agree with the table of the snapshot it names;
  - every migration justification is valid.
- **Actual differences** are re-derived from the tables and compared with the recorded ones.
- **Certificates:** each is re-derived in full from its records, and its bindings are checked against its
  context.
- **Source binding:** the frozen commit and the producer-file hashes.
- **Inventory:** the declared worlds, inits, branches, episodes and pause indices.

Anything too large for git becomes a release asset with its sha256 committed first, as in #55.

## Interpretation limits

- **The scope.** It tests whether specified components keep their declared behaviour through an update, in
  a finite synthetic world with supplied types, relations and entities. It makes no claim about natural
  language or update safety in general.
- **Construction checks** verify this implementation; they are not findings.
- **The "right" policy** is a declared contract. Restart is the default, because answers then reflect the
  facts when they are given. Pin is for replay.
- **Migration** is exact here because the atomic tables are finite and exhaustively known. That would not
  hold where the actual difference cannot be computed.
- **The guarantees are per snapshot,** not simultaneous across a unit's 13 contexts.
- **Pause and resume** happen in one process, with fresh copies.

## Development pilot

**Cap:** **four hours of wall time, with at most four CPU threads,** at low priority. This is not four
CPU-hours.

**Units:** worlds 100–101 × inits 100–101 only, starting with **one learned unit** (world 100, init 100).

**What it settles:**
- the question sizes;
- the interpreter's update count;
- whether the coverage threshold is feasible;
- the episode sample size;
- the pause-index sampling.

**What it may not change:**
- the definitions;
- the check requirements;
- the bars, without review;
- the confirmation worlds.

### Pilot runs (all disclosed)

| Run | Commit | Units | Wall time | Status |
|---|---|---|---:|---|
| `runs/titnn_integration_pilot/20261019T145308` | `f83e7a4` | 100/100 | 985 s | superseded, see below |
| three smoke runs | uncommitted harness edits, scratch output | 6-country world, 100/100 | about 4 min in all | code checks only, not results |
| `runs/titnn_integration_pilot/20261022T121659` | `67a2f04`, clean tree | 100/100, 100/101, 101/100, 101/101 | 3,830 s | **the pilot result** |

About 1.5 of the 4 hours were used. Two corrections were made between the first and last runs; both are
disclosed here.

1. **Naive continuation's failures.** The independent checker judged contract violations and no-version
   outcomes on the full record, including which version produced each application. This file defines them
   on the outcome: the answer or terminal reason, and the visited path. The stricter test made contract
   violations equal ineligible migrations by construction (1,639 of 1,639 in the first run). The checker now
   follows this file, and a fixture covers the difference. That fixture is an ineligible run whose outcome is
   still S1's.
2. **Resume costs.** The first run's migrate cost included the applications made before the pause, which are
   sunk. Costs now count only work after the pause.

The first run's construction checks and learned results are identical to the final run's unit 100/100.

### Pilot results (`67a2f04`, four units)

**Construction checks, required at 100%:**

| Check | Count | Mismatches |
|---|---:|---:|
| C1/C2 supplied episodes: pin, restart and migrate against uninterrupted runs, with every record independently verified | 46,638 | 0 |
| C1/C2 interpreted episodes | 15,960 | 0 |
| C2: every naive switch flagged; no policy record flagged | 62,598 | 0 |
| C4: complete-context activation and rollback | 48 branches | 0 |
| C5: exhaustive sweep at every budget, against the independent traversal | 13 contexts per unit, 15,360–15,552 queries each | 0 |
| F: injected faults with learned contexts (10 per unit) | 40 | 0 |
| C3: the witness (two worlds) | 2 | 0. Naive continuation gives a no-version outcome; pin, restart and migrate are correct |

**Learned competence.** L1–L4 hold in all four units, **at the ceiling**:
- **Parses** are 100% exact on calibration, test and the default-budget diagnostic.
- **The certificates:** every one (C0 and all 48 C1ᵢ) selects threshold 0.0, with zero errors among 780–789
  calibration results.
- **Test:** coverage is 1.000, errors 0, and the accepted-error bound is 0.0038, under C0 and every C1ᵢ.
- **Refusals:** abstention is 1.000 in every category.
- **Modules:** S0 and every S1ᵢ are atomically correct.
- **Differences:** every actual difference is intended only. There are no corrections and no regressions.

**Naive continuation (supplied episodes):**

| Failure | Episodes | Share |
|---|---:|---:|
| unjustified switches | 46,638 | 100% |
| unjustified mixing | 34,826 | 75% |
| contract violations | 1,566 | 3.4% |
| no-version outcomes | 34 | all from two-fact branches; none from single-fact branches, as defined |

**Migration** was eligible in 41,606 of 46,638 episodes (89%). Where it migrated:

| Policy | Work after the pause |
|---|---|
| migrate | 296,642 suffix applications + 184,816 revalidations = 481,458 applications, plus 184,816 difference lookups |
| restart | 481,458 applications |

**Cost:** 911–1,000 s per unit. Fifteen confirmation units would take about 4 hours at four threads, with
every pause index.

**Records:**
- full session records are about 250 MB of JSON per unit (supplied episodes);
- gzip compresses them about 66 times (sample: 1.58 MB to 24 KB);
- about 5 MB per unit with the interpreted episodes included, or about 75 MB for 15 units.

### Proposed for the freeze (pending review)

1. **State the ceiling.** In these worlds the learned interpreter, modules and certificates are at their
   ceiling, so L1–L4 are expected to hold. The test's informative content is the construction checks at the
   seams. The report must say so, and must not present L1–L4 as evidence of robustness.
2. **Drop the efficiency claim for migration.** It does no less module work than restart; it is kept only as
   an exact-equivalence check (C1, C2). Restart remains the default.
3. **Keep every pause index, and the sizes as specified:**
   - 1,000 calibration and 1,000 test draws;
   - 200 unaffected supplied queries per branch;
   - 50 unaffected interpreted questions per branch.
   No sampling is needed.
4. **Store records in two places:**
   - **in the bundle:** summaries, atomic tables, certificates with their records, and compact per-episode
     outcome rows;
   - **as release assets:** the full gzipped session records, with their sha256 committed before the
     release.
5. **Write the independent verifier before the freeze.** `bench/integration_test_verify.py` must rebuild every
   reported number from those files, never calling the producer, as the #55 review required.
6. **Remove the reserved-world check** in the freezing commit.

## Implementation order (before any learned fit)

1. **Prerequisite repair, in its own PR,** with regression tests:
   - **certificates:** `integrated/certificates.py` (the query certificate and the expected context);
   - **checker:** the `update_checks.py` interface (the expected context passed to the gate, and
     certificate-aware identity);
   - **serving:** `integrated/serving.py` (the serving context, and a deployment that activates and rolls
     back complete contexts).

   The PR 6 and PR 8 results must rebuild unchanged.
2. **The sessions layer:** `integrated/sessions.py`:
   - the budget contract;
   - pause indices by `step` calls;
   - pin, restart and migrate, with provenance and justification;
   - the actual-difference computation from atomic tables.
3. **The independent reconstruction:** `integrated/session_verification.py`.
4. **Deterministic table-module fixtures** for every construction check and injected fault, including the
   review's correction counterexample and the two-fact witness.
5. **The query interpreter and the experiment harness**, then the pilot.

Status, 2026-10-22:
- item 1 is merged (#57);
- items 2–4 are at `62a630e` and `67a2f04`;
- item 5 and the pilot are done (above);
- the record export and the independent verifier are next, before the freeze.

## Decisions (from the review of revision 1)

| Choice | Decision |
|---|---|
| Interpreter | Learned. The supplied-query condition and the table modules are kept to isolate failures |
| Budgets | Explicit budgets of 1–16 in the primary questions. The default-16 questions are a diagnostic, reported separately |
| Default policy | Restart. Pin is kept for replay. Migrate is a secondary optimisation under an exact equivalence contract |
| Updates | 4 operator, 4 route and 4 two-fact updates, as independent branches from the same S0 |
| Bars | 90% coverage at a 1% accepted-error bound, and 95% refusal, with completed results defined as above. 100% on construction checks and injected faults |
| Units | Worlds 800–802 × inits 0–4, reserved. Development on worlds 100–101 first |
| Certificate scope | Per snapshot, not simultaneous |
