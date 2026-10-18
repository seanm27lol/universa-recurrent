# Execution records, replay and interventions (frozen brief, integration plan PR 5)

**Status: FROZEN 2026-10-18 (UTC), before any run on the confirmation worlds.** Branch `ti-tnn-traces`,
stacked on the learned-operator PR (#45).

**A development pilot ran first,** on worlds 100 and 101 with init 100 (16 runs). Every module set behaved
as designed:
- learned specialists, shared network and mixture: every check passed, and every answer was correct;
- the endpoint control: faithful, but 0.69 on same-endpoint counterfactuals;
- each faulty set was caught by its own check and by no other.

Prediction 3a could not be judged in the pilot (one init per world). The pilot changed nothing in the
design. Its results are not confirmation results.

## The question

Plan PR 4 showed learned operators answering correctly through the typed runtime. Do the recorded traces
describe the computation that produced those answers? Specifically:
- Is every record complete, and does it replay exactly?
- Does each module depend only on the value it is recorded as consuming?
- Can the checks tell four failures apart?
  - a module that reads the question;
  - a module that calls another module unseen;
  - a lawful but wrong module;
  - a faithful module that computes the wrong function.

## Design

Code: `research/ti_tnn/integrated/records.py`, `verification.py`, `interventions.py`, `faults.py`, and
`research/ti_tnn/bench/semantic_interventions.py`.

**Worlds and module sets.**
- **Worlds.** `generate_world` with seeds **300, 301, 302**, unused by any earlier run.
- **Learned sets.** The four PR 4 families, trained exactly as in PR 4 on atomic facts with inits 0–4:
  `specialists`, `shared_dense`, `moe` and `endpoint`.
- **Constructed sets,** once per world:
  - `table`: the lookup tables;
  - three deliberately faulty sets from `faults.py`: `question_reading`, `hidden_dependency` and
    `swapped`.

**What runs.** Every well-typed path of length 1–3, from every start value. Each execution is:
- recorded (schema `ti_tnn_execution_v1`);
- checked by the independent checker against the facts;
- replayed from its record;
- run under a probe on every module object;
- run with every module it does not use disabled.

**Interventions.** On every path of length 2–3, each later step k gets 4 sampled replacement values (fixed
by a hash of the world, path, start and step). Each replaces the value step k consumes. Two comparisons
follow:
- with the same suffix run on its own (history independence);
- with the reference's counterfactual answer, on interventions where the reference answer changes
  (counterfactual semantic accuracy).

**Competent.** A run is competent if every relation's atomic accuracy is at least 0.99.

## Predictions (recorded before any confirmation run; one-sided)

1. **Records complete and replay exactly.** In every run, every execution passes schema, type and
   integrity checks and replays exactly, and every intervention is recorded as intervened.
2. **Pure modules use only their input.** For `specialists`, `shared_dense`, `moe`, `endpoint`, `table`
   and `swapped`:
   - every runtime call receives the recorded object, with no undeclared calls;
   - disabling unexecuted modules changes nothing;
   - every intervention is history independent.
3. **Competence and counterfactuals.**
   - **3a (replication of PR 4 on fresh worlds):** `specialists`, `shared_dense` and `moe` are competent
     on at least 4 of 5 inits in each world.
   - **3b:** every competent run of those families, and `table`, scores at least 0.95 counterfactual
     semantic accuracy.
4. **The question-reading module is caught only by interventions.** All its ordinary answers are
   correct, it replays, the probe and disabling find nothing, and history independence is below 100%.
5. **The hidden dependency is caught by the probe and by disabling.** All its ordinary answers are
   correct; the probe logs undeclared calls; disabling unexecuted modules changes some executions.
6. **The lawful change is classified correctly.** `swapped` has zero law defects, replays exactly and is
   history independent. Its checked `capital_of` accuracy is exactly the share of countries whose
   capital is also their largest city.
7. **The endpoint control is faithful but wrong.** It is history independent and replays exactly, yet
   scores below 0.95 on counterfactuals whose suffix applies `capital_of` or `largest_city_of`.

`evaluate()` in the bench script applies these. It was written, with a test, before any run.

## What is true by construction or by design, and so is not evidence

- **1 and 2 hold by construction** for these implementations. The runtime builds records from what it
  passed and rebuilds every value. Learned and table modules are pure functions of their argument.
- **3b follows from competence.** Given atomic competence, a correct module on any valid replacement
  value gives the counterfactual answer.
- **4–7 hold by design.** Each faulty set was built to break exactly one guarantee.

**What the run tests.** Whether the instruments separate these properties on real learned modules at full
scale, and whether 3a's competence replicates on fresh worlds. **The only empirical prediction is 3a.**

## Limits

- **Disabling is per module object.** For shared networks, the shared parameters stay on, since the
  executed relations need them too.
- **Faults the probe and interventions cannot catch.** A module could read state that never changes
  within these tests. The constructed faults are examples, not an exhaustive search.
- **Supplied programs, finite worlds, tiny models;** provenance is not authenticated.

**Run:**
`python research/ti_tnn/bench/semantic_interventions.py --worlds 300 301 302 --inits 0 1 2 3 4`, which
also writes `verdict.json` via `evaluate()`.

## Execution record (appended after the run; nothing above was changed)

- **The run.** The protocol's command, from `75cb47a`, 2026-10-18, finishing 19:38 UTC. 72 runs, CPU only,
  with no failures. `verdict.json` was written by `evaluate()` as frozen.
- **All seven predictions hold.** See the [report](../reports/semantic_interventions_v1.md).
