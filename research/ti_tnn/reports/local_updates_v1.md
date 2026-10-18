# Offline module replacement under independent gates: specialist updates are local, shared ones are not — 2026-10-18

**Protocol:** [`protocols/local_updates_v1.md`](../protocols/local_updates_v1.md), frozen at `dd1d937`
before any run on the confirmation worlds, with the analysis (`evaluate()` in `bench/local_updates.py`)
and its test. **Raw results:**
[`experiments/results/ti_tnn/local_updates/`](../../../experiments/results/ti_tnn/local_updates/).

## The question

The plan's question for updates (section 7): **can a candidate update change specified behavior while
preserving specified behavior outside its declared dependencies?** And can it be activated only with
permission, refused when stale, and rolled back exactly?

Each update changes one fact: a capital moves to another city of the same country, or an airport changes
operator. A candidate registry then passes through six independent gates, each answering pass, fail or
insufficient evidence:

| Gate | What it checks |
|---|---|
| Identity and scope | Write locality: only declared modules changed, identities are honest, no shared storage outside the scope |
| Types and laws | Every answer inside its type; the registered round-trip laws hold |
| Intended behavior | Every answer that should change does, exhaustively over paths up to length 3 |
| Unaffected behavior | Every other answer is unchanged, exhaustively |
| Trace and continuation | The candidate's records replay; old records are refused |
| Calibration and resources | A calibration bound to the old registry needs a refit; costs are within limits |

Activation also needs a current base and separate authority.

## Answer

**A specialist update is local in every sense the gates check.** All 150 specialist updates passed every
gate, activated, and rolled back to the identical snapshot:
- 3 worlds × 5 inits × 10 updates;
- in each, only the updated relation's network is fine-tuned, on that relation's facts;
- not one of the 1,013–1,146 unaffected answers per update changed.

**A shared network cannot be updated locally, and without rehearsal it breaks a neighbour.** With the
shared network fine-tuned on the same data the specialist gets:
- **Capital updates:** 73 of 75 broke unaffected answers, 4,387 in total. Every single-step answer that
  moved was `largest_city_of`'s, the relation that shares `capital_of`'s endpoints (Country → City).
  Sixteen of these updates also broke a registered round-trip law.
- **Operator updates:** none of the 75 interfered. `operator_of` has no relation sharing its endpoints.
- **With rehearsal:** fine-tuning on every relation's facts kept all unaffected answers in all 150
  cases.
- **Write locality failed every time.** Every shared candidate changed every module's parameters, so
  none was activated.

**Every deliberately faulty candidate was refused for the right reason,** in all three worlds (30 of 30).

## Results

| 150 updates per system | Every gate passed and activated | Write locality | Unaffected behavior kept | Unaffected answers changed | Intended answers wrong |
|---|---:|---:|---:|---:|---:|
| **`specialists`** | **150** | 150 | 150 | 0 | 0 |
| `shared_same_data` | 0 | 0 | 77 | 4,387 | 0 |
| `shared_rehearsal` | 0 | 0 | 150 | 0 | 0 |

**About the table.**
- Every base model was competent (every answer correct before the update), in all 15 units.
- Intended changes per update: 2–15 answers. Unaffected answers per update: 1,013–1,146, all checked.
- **The shared systems' other gates.** Their calibration-and-resources gate was "insufficient evidence"
  (the cost of a shared module cannot be read off), and their trace gate passed.
- **Speed.** Updates took 0.07–0.18 s each on CPU.

**Same-data shared interference, by update kind:**

| Updated relation | Unaffected behavior kept | Answers changed per update |
|---|---:|---|
| `capital_of` | 2 of 75 | 0–183 |
| `operator_of` | 75 of 75 | 0 |

**The faulty suite** (each case in all three worlds):

| Case | Outcome required, and obtained |
|---|---|
| Valid table edit | Every gate passes, activated, rolled back exactly |
| Lawful answer flip (`capital_of` answers the largest city) | Types and laws pass; unaffected behavior fails |
| Broken inverse (a capital city moved to another country) | Types and laws fail |
| Ill-typed (`capital_of` answers airports) | Types and laws fail |
| Out-of-scope state edit (an unused entry added to another module's table) | Scope fails; behavior passes |
| Stale revision | Every gate passes; activation refused as stale |
| Failed rollback (tampered snapshot) | Refused; the active registry unchanged |
| Unauthorized, claiming confidence 0.999 | Every gate passes; activation refused |
| Stale declared identity (parameters changed after hashing) | Scope fails |
| Parameters shared with an out-of-scope module | Scope fails |

## Predictions (frozen before any confirmation run)

| # | Prediction | Outcome |
|---|---|---|
| 1 | ≥ 95% of specialist updates per world pass every gate, activate and roll back exactly | **Holds** (150 of 150) |
| 2 | Every activation rolls back exactly | **Holds** |
| 3 | Every shared candidate fails identity and scope and is not activated | **Holds** (300 of 300) |
| 4 | The faulty suite is caught as designed in every world | **Holds** (30 of 30) |
| 5 | Same-data shared: ≥ 80% of capital updates fail unaffected behavior, ≥ 80% of operator updates pass it, per world | **Holds** (73 of 75 fail; 75 of 75 pass) |
| 6 | Rehearsal shared: ≥ 90% pass unaffected behavior, per world | **Holds** (150 of 150) |

**Which are findings.**
- **By construction or design** (the protocol said so beforehand): 2, 3 and 4.
- **Empirical:** 1, 5 and 6.
- **5 is the substantive one:** without rehearsal, the shared network forgets precisely the relation
  that shares the updated relation's endpoints.

## What it means

- **Specialists make locality checkable, not just likely.** A specialist update changes one module's
  parameters, and the gates can confirm that nothing else changed in state or behavior. A shared
  network can be updated without behavioral damage, with rehearsal, but its state changes everywhere,
  so write locality can never be shown. The plan distinguishes these levels for this reason.
- **The interference follows relation identity.** Same-data fine-tuning damaged `largest_city_of`, and
  only it, when `capital_of` changed. This is the same-endpoint confusion that plan PR 4 tested, now
  showing up as forgetting. The network kept the two relations apart when trained on both, but drifts
  when trained on one.
- **Laws catch some interference, not all.** Sixteen of the interfering updates broke a registered
  round-trip law. Most did not: the moved `largest_city_of` answers were wrong cities in the right
  country. Only the exhaustive behavior check caught all of them.
- **Gates and authority are separate.** A candidate that passed everything was still refused when it
  was stale or unauthorized, whatever confidence it claimed. A tampered snapshot could not be rolled
  back onto.

## Limits

- **Single-fact updates, a finite world, and an exhaustive check up to length 3.** In larger domains the
  unaffected check would be sampled and separately scoped.
- **Data and parameter candidates only.** No candidate code runs.
- **The authority is a stand-in.** Hashes identify artifacts; they authorize nothing.
- **No calibrated component is deployed.** The calibration gate's refit requirement is shown in the
  tests only.
- **Not applicable here:** naturality checks (this executor has no frame transports).

## Execution

- **Source.** Committed as `dd1d937` immediately before the run, with a clean tree. The run file does not
  record its commit.
- **Run.** 2026-10-18, finished 20:49 UTC, CPU only (4 threads, `nice`). It ran from the run directory's
  checkout of `main` with the frozen branch's script and the protocol's arguments.
- **Software.** PyTorch 2.10.0, Python 3.11.14, aarch64.
- **Verdict.** `verdict.json` was written by the frozen `evaluate()`; `make_tables.py` rebuilds it from the
  bundled runs.
- **Development pilot.** It ran on worlds 100–101 before freezing; it and the same-data variant it
  prompted are disclosed in the protocol.

## Corrections after review (2026-10-18)

The review reproduced three gaps in this PR's machinery. All are repaired with regression tests
(`tests/test_ti_tnn_contract_repairs.py`).

**1. Gate results were not bound to the candidate they checked.**
- **The failure.** A correct candidate's six passing outcomes, passed with a different, wrong
  candidate's matching record, activated the wrong candidate.
- **Context fields.** A record could also name the wrong target world, a different checker version, or a
  required calibration that was never supplied, and still pass.
- **The repair.** A `Checker` checks private published copies and returns a sealed `CheckResult`. The
  result is bound to the record, both snapshot digests, both worlds, the checker version, the
  calibration and the resource limit. A deployment activates only the snapshot inside a result its own
  checker sealed. A new mandatory `record_context` gate rejects a mismatched context.

**2. "Immutable snapshots" could contain mutable state.**
- **The failure.** Registries hashed declared version strings, and rollback kept references to live
  objects. Mutating a NumPy array shared by base and candidate changed the answers but neither registry's
  identity. Activation still succeeded, and rollback reported success without restoring the original
  answers.
- **The repair.** Deployments now hold deep, frozen copies with digests over all behavior-relevant state,
  verified before activation and rollback.

**3. The evaluator accepted incomplete inventories.** A copy holding 90 of the 450 updates and 3 of the
30 fault cases was judged complete, with all six predictions passing. It now requires exactly the
registered cases. The bundled verdict, from the complete inventory, rebuilds unchanged.

**What this does to the reported results.**
- **Nothing in the published runs relied on these gaps.** The harness checked and activated each
  candidate it built, and specialist fine-tuning copied the network it changed.
- **A post-repair rerun reproduced the frozen verdict exactly** (bundled as `local_updates_post_repair/`). It ran with the same worlds, inits and
  command, through the sealed-result and snapshot path (predictions, fault-suite outcomes and every
  per-family count).
- **What the claims rest on now.** Before the repair, "activation requires every gate" and "rollback is
  exact" described what the harness did, not what the interface enforced. Both are now enforced.
