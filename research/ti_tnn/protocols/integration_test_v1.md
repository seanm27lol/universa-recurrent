# Integration test: one question through recurrence, a checked update mid-run, and resume (frozen, revision 4)

**Status: FROZEN 2026-10-23 (UTC), revision 4, before any run on confirmation worlds 800–802.**
- **The freezing commit** is the commit that adds this status line. It also:
  - removes the harness's refusal of worlds 800–802;
  - fixes the producer-file hashes in `bench/integration_test_verify.py` (`FROZEN_SOURCES`).
- **The confirmation command:**

      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 10 .venv-phase2/bin/python research/ti_tnn/bench/integration_test.py \
          --worlds 800 801 802 --inits 0 1 2 3 4 --out runs/titnn_integration_confirmation
      .venv-phase2/bin/python research/ti_tnn/bench/integration_test_verify.py runs/titnn_integration_confirmation/<run>

- **Resources:** CPU only, four threads, low priority. One unit takes about 30 minutes; 15 units take about
  7.5 hours.
- **What is authoritative.** The specification immediately below. "History" keeps revisions 1–3, the reviews'
  findings and the development pilot.
- **The code it specifies:**
  - `integrated/certificates.py`, `serving.py`, `sessions.py`, `session_verification.py`, `release.py` and
    `query_interpreter.py`;
  - `bench/integration_test.py` and `bench/integration_test_verify.py`;
  - `update_checks.py`, as repaired in #57.
- **What it builds on:** the components of plan PRs 6–8, unchanged.

This is item 3 of the [integration plan's section 12](../integration_plan.md#12-second-review-amendments-2026-10-18-after-prs-4150)
order: "question, interpreted bounded query, recurrence, pause, a checked module update, then resume under the
right version policy."

## A concrete example

A route-network world has airports, each with one onward connection (`connects_to`), and an operator for each
airport (`operator_of`). Someone asks:

> which is the first airport within 6 connections of country_12_city_0_airport that operator_3 operates ?

1. **Interpreting the question.** The interpreter parses it into a bounded query:
   - start `country_12_city_0_airport`;
   - target `operator_3`;
   - `connects_to`, then `operator_of`;
   - budget 6.

   The policy then binds the entities, checks the types and the budget contract (an integer from 1 to 16),
   and admits the question if its score clears the active serving context's certificate.
2. **Recurrence.** The run starts on context C0 and pauses after two calls to `step`.
3. **An update while the run is paused.** It makes `operator_3` the operator of the second airport, which the
   run already tested under C0. The update passes its gates, and the complete context C1 is activated.
4. **The resume.**

| On resume | Answer | Its record |
|---|---|---|
| **pin** (finish on C0) | the airport C0 leads to | every application from C0 |
| **restart** (start again on C1) | the second airport | every application from C1 |
| **naive** (a control: keep the state, continue on C1) | the same as pin; the second airport was tested under C0 and is never tested again | applications from C0 then C1, with no justification. It reports finishing on C1, but C1 gives a different answer |

**What the test asks.** When connected, do the components keep their contracts through an update? Is every
released result produced by one complete serving context, or justified as exactly equivalent to one, and
released under the right certificate? Is that confirmed by a verifier that rebuilds everything from exported
records?

**What the test is about.** In these worlds the learned interpreter, modules and certificates are expected to
be at their ceiling (see "History: development pilot"). The test's informative content is the **construction
checks** at the seams. Its learned predictions (L1–L4) are reported, but they are **not** evidence of
robustness.

## Definitions

**Serving context.** The unit that serves:

> snapshot (modules) + query certificate + expectation (interpreter, query policy, correctness event,
> sampling distribution, grid) + world identity

- **Activation and rollback** install and restore complete contexts (`serving.ServingDeployment`).
- **Sessions are bound to a complete context.** Continuation and pin require the *complete* recorded context,
  not only its snapshot (`sessions.advance`, `sessions.resume_pin`).

**Query certificate** (`certificates.QueryCertificate`). It holds:
- the complete fixed-sequence result (grid, every step, threshold, stop reason, alpha, delta);
- its calibration records (occurrence id, raw score, correct);
- its bindings (interpreter, snapshot digest, policy, event, distribution).

The update checker judges a certificate against the *expected* serving context, never against the
certificate's own fields. The certificate must be bound to the candidate's snapshot, re-derive from its
records, and have certified a rule.

**Completed result:** an airport answer, or a negative terminal outcome (`dead end`, `cycle`,
`budget exhausted`).

**Policy refusal.** The reasons are:
- insufficient confidence;
- no certificate;
- unsupported;
- unknown entity;
- invalid query (ill-typed, or the budget contract);
- malformed;
- a module with no valid output.

**Correctness event:** a released result equals the reference outcome of the question's reference query
under the context's world. A released result for a question that should be refused is an error.

**Admission and release** (`release.py`, recorded for every decision):
- **admission:** under the active context's certificate (score ≥ its threshold);
- **release on the admitting context** (no change, or pin): at that context's threshold;
- **release on another context** (restart, migrate): at max(t0, t1), with the supporting step of the
  releasing certificate. There is no release unless both certificates share the interpreter, policy, event,
  distribution and grid.

Two C1 test results are kept apart:
- **new requests**, admitted and released by C1 at t1;
- **resumed requests**, admitted under C0 and released on C1 at max(t0, t1).

**Atomic table:** each relation's output on every valid input, for one snapshot. It has no weights.

**Actual difference:** every (relation, input) whose outputs differ between two snapshots' atomic tables. It is
bound to both digests and both table digests. Its entries are classed:
- **intended:** a declared change;
- **correction:** outside the declared changes; the old output was wrong and the new one is right;
- **regression:** anything else.

**Migration.** `migrate` continues on S1 only if no prefix application consumed a key of the actual difference.
Each prefix application is revalidated on S1, and the proof is recorded (difference identity, snapshot pair,
prefix inputs, revalidation count). Otherwise migration restarts, recording the blocking inputs. Its records
keep the prefix's original provenance.
- **Equivalence:** migration is an exact-equivalence check.
- **No efficiency claim:** revalidation makes its module work equal restart's, and its lookups are extra.
- **Default:** restart is the declared default policy. Pin is for replay.

**Pause index:** the number of calls to `step` before the pause, counting a terminating call. A paused run has
k from 0 to n − 1, where n is the uninterrupted run's number of calls.

**Independent checking** (`session_verification.check_session`) makes three separate judgments:
1. **Execution validity.** Replaying the record's segments from the exported tables, at their recorded call
   indices, reproduces every application, visited airport, terminal result, call count and segment boundary.
   Every segment's context resolves to an exported context on that snapshot.
2. **Policy validity.**
   - **pin:** one unchanged segment;
   - **restart:** one fresh segment, with a verifiable discarded prefix;
   - **migrate:** a proof, every field of which re-derives;
   - **naive continuation:** never valid.
3. **Semantic failures, on valid executions only** (outcome = answer or terminal reason, and visited path):
   - **contract violation:** the outcome differs from the snapshot finished on;
   - **no-version outcome:** the outcome differs from every snapshot that supplied an application.

   An **unjustified switch** is a change of snapshot without a valid justification. **Mixing** also needs
   applications from two snapshots; at pause index 0 there is a switch but no mixing.

## Units, study and procedure

- **Worlds and units:**
  - **confirmation:** `route_world` seeds **800, 801, 802**, 48 countries; inits **0–4**; 15 units. These
    worlds are unused, which was checked;
  - **development:** worlds 100–101 only.
- **Questions:**
  - **primary:** until-questions with explicit budgets 1–16 (two templates);
  - **diagnostic, reported separately:** omitted-budget questions, with the default budget 16;
  - **refusal categories:** unsupported, budget over 16, ill-typed start, unknown entity.
- **Sampling and sizes:**
  - **training:** 30% of (start, target, budget) instances, plus 5% per refusal kind;
  - **calibration and test:** 1,000 independent draws each, from one distribution: 20% refusals (5% per
    category), otherwise held-out explicit-budget instances;
  - **diagnostic:** 300 default-budget draws.
- **Interpreter:** `query_interpreter.py`, with 1,500 Adam updates of 64 at a learning rate of 3e-3.
- **Modules:** learned specialists (PR 4/7 recipe: d 32, hidden 64, 1,500 steps, lr 1e-2).
- **Updates:** 12 independent branches from S0 per unit, fixed from the world seed:
  - 4 operator changes;
  - 4 route changes;
  - 4 two-fact operator changes on consecutive airports that some route reaches.

  Each update is fine-tuned with PR 8's recipe (300 steps, lr 3e-3). Its certificate uses the same
  calibration draws relabelled under S1. It is checked by the repaired `Checker` with the expected context,
  then activated as a complete context. Then come the evaluations, then a rollback to C0.
- **Certificate scope:** per snapshot. Each of a unit's 13 contexts is certified at risk 1% and level 0.05
  on its own, with no simultaneous claim. The grid is the robustness study's, from 0.999 to 0.0.

**Episode scope:**

| Population | Queries | Budgets | Judged by |
|---|---|---|---|
| IID test draws | all 1,000 per unit | as drawn | coverage, accepted error, refusal rates per context (binomial bounds) |
| Exhaustive sweep | every (start, target) | **every budget 1–16** | construction (C5) and learned correctness (L2, L3) |
| Supplied episodes | every pair whose S0 and S1ᵢ runs differ, and 200 sampled unaffected pairs, per branch | **16** | C1, C2, descriptive naive rates |
| Interpreted episodes | every admitted test question whose runs differ, and 50 sampled unaffected, per branch | **their parsed budgets** | C1, C2, release decisions |

Episodes run at every pause index under pin, restart and migrate, and under naive continuation. They are never
pooled into a binomial sample.

**Witness.** Per world, the first route S → a → b → c, in sorted order, whose a and b are not run by c's
operator T. Setting a and b to T makes naive continuation after testing a give b, which neither version gives.
It uses table modules.

**Injected faults.** With the unit's learned contexts, in the first activated branch:
- resuming a C0 state on C1 through `recurrence.resume`;
- an in-place parameter change under pin;
- pin on another complete context with the same snapshot;
- a certificate bound to S0;
- a certificate for another interpreter;
- a forged certificate;
- a context with another certificate;
- budgets 0, 17, `True` and 3.0.

The deterministic fixtures in `tests/test_ti_tnn_sessions.py` and `test_ti_tnn_serving.py` cover every fault
and corruption case.

## Predictions and verdicts

**Construction checks: 100% in every exercised case, in every unit.** There is no allowance. The verifier
re-derives each one.

| # | Check |
|---|---|
| C1/C2 | Every pin, restart and migrate record is execution-valid and policy-valid, and equivalent to its expected snapshot (pin: S0; restart and migrate: S1ᵢ). Every naive record is execution-valid and flagged as an unjustified switch. No declared episode record is missing, and none is extra |
| C3 | The witness, in every world: naive continuation gives a no-version outcome, and pin, restart and migrate are valid and correct |
| C4 | Activation installs the complete C1ᵢ, and rollback restores the complete C0. This is the producer's record, with its identities checked |
| C5 | Every exhaustive outcome, in every context, equals the traversal over that context's own table. This is execution, not correctness |
| R | Every recorded admission and release decision re-derives |
| F | Every injected fault is handled as required. Faults are "not exercised" in a unit where no branch activates, and are reported as such |

**Learned predictions.** For each, a unit holds or fails; a world holds at ≥ 4 of 5 inits; the prediction
holds when all three worlds do. The inventory is all 15 units with 12 branches each. A missing unit, a
rejected candidate or an uncertified context counts as a failure, never as missing data.

| # | Holds in a unit when |
|---|---|
| L1 | C0 is certified. Its new-request test has coverage ≥ 0.90 and a 95% accepted-error bound ≤ 1%. Its observed abstention is ≥ 0.95 in each refusal category. This is an observed-rate benchmark over about 50 draws per category, reported with its denominators, and not a population lower bound |
| L2 | S0's atomic table equals the world's facts, and every exhaustive outcome under S0 equals the reference |
| L3 | All 12 branches activate. Each C1ᵢ is certified, and S1ᵢ is atomically correct. Every exhaustive outcome under C1ᵢ equals its world's reference. Each C1ᵢ new-request test (at t1ᵢ) meets L1's coverage and bound |
| L4 | L1–L3 hold, and each C1ᵢ resumed-request test (at max(t0, t1ᵢ)) meets the same coverage and bound |

**Also reported, judged by no prediction:**
- exact parsed-query accuracy;
- the omitted-budget diagnostic;
- the difference classes;
- naive continuation's rates: switches, mixing, contract violations, no-version outcomes;
- migration eligibility and its counted costs;
- releases by policy.

## Export and independent verification

**The export.** Each unit writes `runs/.../units/w<world>_i<init>/`:

| File | Contents |
|---|---|
| `unit.json` | summary, branch records, gates, differences, C4, faults |
| `interpreter.json.gz` | every calibration, test and diagnostic draw: occurrence id, parse, full-precision score and its components, pre-threshold decision, per-context result and label, and admission and release decisions with context and certificate identities |
| `tables.json.gz`, `certificates.json.gz`, `contexts.json.gz` | the atomic tables, certificates and serving contexts |
| `exhaustive.json.gz` | compact rows and an outcome digest per context |
| `episodes.jsonl.gz` | compact episode rows |
| `sessions.jsonl.gz` | every full session record, with its release decision |

**Where it is kept:**
- **in the evidence bundle:** everything except `sessions.jsonl.gz`;
- **as release assets:** the session files, with their sha256 committed before the release (as in #55).

**The verifier.** `bench/integration_test_verify.py` rebuilds every number, construction check and learned
verdict from the export. It never calls the producer's execution or decision code. It uses:
- **the study generator and branch draw,** to regenerate what was asked;
- **`session_verification`,** which has its own traversal;
- **the robustness verifier's independent fixed-sequence certificate.**

It re-derives:
- every score, from its components;
- the pre-threshold decisions;
- every result, by traversal;
- every label, every certificate and every release decision, and every metric;
- the exhaustive digests;
- the declared episode sets, and every session record;
- the differences and their classes;
- the witness.

It raises on any export that does not re-derive. A self-consistent record of a forbidden execution fails the
construction verdict.

**What it cannot re-derive** from weight-free records, and reports as the producer's record:
- the update gates;
- the activation and rollback mechanics (C4);
- the injected faults (F);
- that each table is its snapshot's behaviour. That is checked when the run is produced, and by the
  deterministic fixtures.

**Source binding.** The verifier requires:
- the run's producer-file hashes to equal `FROZEN_SOURCES`, fixed in the freezing commit;
- a research tree that was unmodified when the run started.

The run records its commit, and the execution record names it, since a commit cannot name itself. The verifier
reports that commit.

## Interpretation limits

- **The scope.** Specified components keeping declared contracts through updates, in a finite synthetic world
  with supplied types, relations and entities. There is no claim about natural language, other models, or
  update safety in general.
- **Construction checks verify this implementation.** The learned predictions are expected at the ceiling and
  are not robustness evidence.
- **The "right" policy is a declared contract.** Restart is the default.
- **Migration is exact here** only because atomic tables are finite and exhaustively known.
- **Guarantees are per snapshot.** Refusal rates are observed rates. Pause and resume happen in one process.

## History

**Revision 1** (`9fb811f`) proposed the design. The review of revision 1 found that:
- the calibration interface was not bound to the expected context;
- activation swapped snapshots only;
- the declared changes could differ from the actual module changes (the correction counterexample);
- populations and construction criteria were unclear.

**Revision 2** (`03f0d36`) applied the review's decisions:
- **interpreter:** learned;
- **budgets:** explicit, 1–16;
- **default policy:** restart;
- **updates:** 4 + 4 + 4 independent branches;
- **bars:** 90% coverage at a 1% bound, 95% refusal, and 100% on construction checks;
- **units:** worlds 800–802 × inits 0–4;
- **certificates:** per snapshot.

The prerequisite repair merged as #57.

**Revision 3** (`7d5adbc`) disclosed the development pilot (below). The review of revision 3 found that:
- the checker could count a corrupted naive trace as a genuine failure;
- pin accepted another complete context on the same snapshot;
- the verifier ignored several migration-proof fields;
- the resumed-question path did not exercise the release gate.

It also asked for the wording now in this revision:
- C5 is construction only;
- the episode budget scope is explicit;
- switches and mixing are counted separately;
- the refusal bar is an observed rate.

**Revision 4** (this text) follows the repairs at:
- `bb9816e`: three judgments, complete-context binding and the release adapter;
- `433b208`: the complete export, the independent verifier, and export → rebuild → corruption tests on a
  ceiling run and a non-ceiling run.

The non-ceiling run had imperfect S0 operators, learned corrections, a rejected candidate, non-zero thresholds
and refused releases. Every one of 17 deliberate corruptions was rejected.

### Development pilot (worlds 100–101 × inits 100–101; development diagnostics checked online)

The pilot ran at `f83e7a4` (one unit, superseded) and at `67a2f04` (four units, clean tree). Its checks ran
online, inside the harness. Its harness kept aggregates and a small sample of records, not the complete export
specified above. **Its numbers cannot be rebuilt retrospectively, and they are development diagnostics, not
evidence.**

**Construction checks:** in 46,638 supplied and 15,960 interpreted episodes, the pilot found no pin, restart
or migrate mismatch and no rejected policy record. In each of the 13 contexts per unit, the exhaustive sweep
found no construction mismatch.

**Learned predictions:** at the ceiling, with every certificate at threshold 0.0.

**Naive continuation (supplied episodes):** 3.4% contract violations. The 34 no-version outcomes all came from
two-fact branches.

**Migration** did exactly restart's module work, plus lookups.

**Mid-pilot corrections** (disclosed in revision 3):
- naive failures are judged on outcomes;
- costs count only work after the pause.

**Cost:** about 16 minutes per unit at four threads, before the complete export was added.

**Scale check before the freeze.** One full-size development unit (100/100) ran with the final harness at
`76c37f8`, to measure the complete export before committing to the 15-unit run:
- **cost:** 29.3 minutes, with peak memory of 733 MB;
- **export:** about 0.8 MB of bundle files, and a 4.6 MB session file;
- **verification:** the verifier rebuilt it in 11 seconds;
- **results:** every construction check and L1–L4 held;
- **episodes:** 12,881 supplied and 4,551 interpreted;
- **naive continuation:** 436 contract violations and 9 no-version outcomes among the supplied episodes.

It is a development diagnostic, like the pilot.
