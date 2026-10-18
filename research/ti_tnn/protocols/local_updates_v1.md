# Offline module replacement under independent gates (frozen brief, integration plan PR 8)

**Status: FROZEN 2026-10-18 (UTC), before any run on the confirmation worlds.** Branch `ti-tnn-updates`,
stacked on the recurrence PR (#49).

## The question

**Can a candidate update change specified behavior while preserving specified behavior outside its
declared dependencies, and be activated and rolled back safely?** (Plan section 7.)

The plan separates three levels of locality:
- **Write locality:** only the declared state changed.
- **Dependency locality:** what should change, changed.
- **Behavioral noninterference:** everything else still behaves as before.

It also asks that activation be atomic and require separate authority, that stale candidates be refused,
and that rollback be exact.

## Design

Code: `research/ti_tnn/integrated/updates.py`, `update_checks.py`, `modules.finetune`;
`research/ti_tnn/bench/local_updates.py`.

**Worlds.** `generate_world` with seeds **600, 601, 602**, unused by any earlier run (24 countries).

**Updates.** Ten single-fact updates per world, each defining an updated copy of the world:
- five capitals moving to another city of the same country;
- five airports changing operator.

**Learned systems,** inits 0–4, each base trained as in PR 4. The candidate fine-tunes it for 300 steps
at learning rate 3e-3.

| System | Base | Candidate |
|---|---|---|
| `specialists` (proposed) | specialists | only the updated relation's network is fine-tuned on that relation's updated facts; every other module object is reused |
| `shared_same_data` | shared dense | the shared network is fine-tuned on the same facts the specialist sees |
| `shared_rehearsal` | shared dense | the shared network is fine-tuned on every relation's updated facts |

**Gates** (`update_checks.py`; each gives pass, fail or insufficient evidence):

| Gate | What it checks |
|---|---|
| Identity and scope | Changed modules are within the declared scope; declared identities match the actual parameters or tables; finite values; unchanged shapes; no parameter storage shared with an out-of-scope module |
| Types and laws | Every module answers inside its type for every input; the registered round-trip laws hold |
| Intended behavior | The declared changes are exactly the updated world's; every path whose reference answer changed matches the new world (exhaustive, length ≤ 3) |
| Unaffected behavior | Every other path, from every start, matches the reference (exhaustive, length ≤ 3) |
| Trace and continuation | The candidate replays its own records; no undeclared calls; history-independent interventions; records made under the old registry are refused |
| Calibration and resources | A calibration bound to the old registry needs a refit; MACs of changed modules can be read and are within the limit |

**Activation** requires all four:
- the record's base is the active revision;
- every mandatory gate passed;
- the authority permits the proposer;
- the record names this candidate.

**Rollback** restores the exact previous snapshot.

**Faulty suite,** once per world. Each case has the outcome it must produce:
- **valid table edit:** every gate passes, it activates, and rollback is exact.
- **lawful answer flip** (`capital_of` given the largest cities): laws pass, unaffected behavior fails.
- **broken inverse** (a capital city moved to another country): laws fail.
- **ill-typed** (`capital_of` answering airports): types fail.
- **out-of-scope state edit** (an unused entry added to `airport_of`'s table): scope fails while
  behavior passes.
- **stale revision:** every gate passes, and activation is refused as stale.
- **failed rollback** (a tampered snapshot): rollback is refused, and the active registry is unchanged.
- **unauthorized** (claiming confidence 0.999): every gate passes, and activation is refused.
- **stale declared identity** (parameters changed after hashing): scope fails.
- **shared parameter storage** with an out-of-scope module: scope fails.

## Development pilot (disclosed)

Worlds 100–101, inits 100–101.
- **Specialists:** 40 of 40 updates passed every gate, activated and rolled back exactly.
- **Shared, all facts:** this was the first shared design. Every candidate failed write locality, but
  changed no unaffected answer. Because it saw more data than the specialist, I added the same-data
  variant before freezing.
- **Shared, same data:**
  - every `capital_of` update failed unaffected behavior (20 of 20). The answers that moved were
    `largest_city_of`'s, the relation with the same endpoints;
  - every `operator_of` update passed (20 of 20).
- **The faulty suite:** every case behaved as designed.

## Predictions (recorded before any confirmation run; one-sided)

1. **Specialist updates are local.** At least 95% of specialist updates in each world pass every
   mandatory gate, activate and roll back exactly.
2. **Rollback is exact.** Every activation rolls back to the identical snapshot, with identical answers
   on every path.
3. **Shared updates are not write-local.** Every shared candidate, in both variants, fails identity and
   scope (its declared scope is one relation), and none activates.
4. **The faulty suite is caught as designed,** in every world.
5. **Same-data shared fine-tuning interferes through the shared endpoints.** In each world:
   - at least 80% of its `capital_of` updates fail unaffected behavior;
   - at least 80% of its `operator_of` updates pass it.
6. **Rehearsal keeps behavior.** At least 90% of rehearsal shared updates pass unaffected behavior, in
   each world.

**What holds by construction or design, and what does not:**
- **By construction or design:** 2 (snapshots are immutable and swapped whole), 3 (one shared network
  means every module's identity changes) and 4 (each fault was built for its gate).
- **Empirical:**
  - **1:** whether fine-tuning one network learns the change and keeps that relation's other facts;
  - **5:** whether a shared network forgets a relation with the same endpoints, but not one that has
    no counterpart;
  - **6:** whether rehearsal prevents that.

## Limits

- **One-fact updates in a finite world,** with an exhaustive check up to length 3. Larger domains would
  need sampled, separately scoped tests.
- **The authority is a stand-in.** Hashes identify artifacts; they authorize nothing.
- **Candidates are data and parameters only.** No candidate code runs; arbitrary candidate code is out
  of scope, as the plan says.
- **The naturality check in the plan's suite is not applicable here,** because this executor has no
  frame transports.
- **No calibrated component is deployed** in these registries. The calibration gate is exercised in the
  tests only.

**Run:**
`python research/ti_tnn/bench/local_updates.py --worlds 600 601 602 --inits 0 1 2 3 4`, which also writes
`verdict.json` via `evaluate()`.

## Execution record (appended after the run; nothing above was changed)

- **The run.** The protocol's command, from `dd1d937`, 2026-10-18, finishing 20:49 UTC. 450 learned
  updates and 30 faulty-suite cases, CPU only, with no failures. `verdict.json` was written by
  `evaluate()` as frozen.
- **All six predictions hold.** See the [report](../reports/local_updates_v1.md).
