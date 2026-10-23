# Integration test: the assembled system kept every construction contract through 180 checked updates; its learned parts were at their ceiling — 2026-10-23

**Protocol:** [`protocols/integration_test_v1.md`](../protocols/integration_test_v1.md), revision 4, frozen at
`e03075a` before any run on confirmation worlds 800–802.

**Raw results:** [`experiments/results/ti_tnn/integration_test/`](../../../experiments/results/ti_tnn/integration_test/).
That holds the complete export of all 15 units and the verifier's output. The 15 full session-record files are
release assets (see "Evidence").

**Verification:** `bench/integration_test_verify.py`:
- **What it rebuilt:** every number below, from the export, without calling the producer's execution or
  decision code.
- **Source binding:** it required the run's producer-file hashes to equal those fixed in the freezing commit,
  with an unmodified tree. The run recorded commit `e03075a`.

## The question, by example

A route-network world has airports, each with one onward connection, and an operator for each airport. Someone
asks:

> which is the first airport within 6 connections of country_12_city_0_airport that operator_3 operates ?

1. **Interpreting and admitting.** A learned interpreter parses the question into a bounded query (start,
   target, `connects_to` then `operator_of`, budget 6). The policy binds and type-checks it. The active serving
   context's certificate decides whether it is admitted.
2. **Pausing for an update.** The run pauses partway. Meanwhile a checked update (say, a new operator for an
   airport the run already tested) becomes the active context.
3. **The resume.** The run is resumed:
   - **restart**, the default, starts again on the new context;
   - **pin** finishes on the old context;
   - **migrate** continues on the new context only if the paused prefix used nothing the update changed.

   The result is released under the right certificate.

**What the test asks.** When connected, do the components keep their contracts? Is every released result
produced by one complete serving context, or exactly equivalent to one, and released under the right
certificate? Is all of that rebuilt by an independent verifier?

## Result

**Every construction check passed in every unit, and every learned prediction held in 5 of 5 inits in each world.**
The learned interpreter, modules and certificates were at their ceiling, as the protocol expected. These
results test the seams between components. They are not evidence that the learned parts are robust.

### Construction checks (required at 100%; rebuilt by the verifier)

| Check | Exercised | Failures |
|---|---:|---:|
| C1/C2, supplied episodes (budget 16): every pin, restart and migrate record execution-valid, policy-valid and equivalent to its expected snapshot; every naive record execution-valid and flagged | 186,770 episodes; 747,080 records | 0 |
| C1/C2, interpreted episodes (parsed budgets) | 56,670 episodes; 226,680 records | 0 |
| No declared episode record missing, none extra | 973,760 records | 0 |
| C3, the two-fact witness (one per world) | 3 | 0 |
| C4, complete-context activation and rollback (the producer's record) | 180 branches | 0 |
| C5, exhaustive outcomes against each context's own table, at every budget 1–16 | 13 contexts per unit × 15,072–17,664 queries | 0 |
| R, every field of every release decision re-derived (interpreted episodes) | 170,010 policy records | 0 |
| F, injected faults with learned contexts (the producer's record) | 11 faults × 15 units | 0 |

### Learned predictions (a world holds at ≥ 4 of 5 inits)

| Prediction | World 800 | World 801 | World 802 | Holds |
|---|---:|---:|---:|---|
| L1, interpretation (C0) | 5 | 5 | 5 | yes |
| L2, S0 modules (atomic table and every exhaustive outcome) | 5 | 5 | 5 | yes |
| L3, all 12 updates (gates, certificates, S1 atomic and exhaustive, new requests at t1) | 5 | 5 | 5 | yes |
| L4, resumed requests at max(t0, t1) | 5 | 5 | 5 | yes |

**The values, in every unit:**

| Measure | Result |
|---|---|
| Parses | exact on every supported calibration, test and diagnostic draw |
| C0 test | coverage 1.000 with 0 errors over 786–816 released results; 95% accepted-error bound 0.0037–0.0038 |
| Observed refusal | 1.000 in each category, with denominators of 39–65 per category per unit. These are observed rates, not population lower bounds |
| Updates | all 180 passed every mandatory gate and activated. Every certificate (195 in all) selected threshold 0.0 |
| Actual differences | the declared changes only: no corrections, no regressions |
| Default-budget diagnostic | released and correct throughout |

**The five inits of a world behaved identically.** Every learned module was exact, so each init's atomic tables
equal the world's facts, and every episode count is the same across a world's inits. The construction checks
are therefore effectively replicated over **three** worlds, each with 13 contexts. They are not replicated over
15 independent units. The inits differ in weights, identities and certificates' raw scores, which the export
checks, but not in module behaviour.

### What naive continuation would have done (the control, never deployed)

| World | Supplied episodes | Unjustified mixing | Contract violations | No-version outcomes |
|---|---:|---:|---:|---:|
| 800 (each unit) | 10,640 | 7,671 | 520 (4.9%) | 8 |
| 801 (each unit) | 13,289 | 10,486 | 582 (4.4%) | 2 |
| 802 (each unit) | 13,425 | 10,478 | 454 (3.4%) | 0 |

**Interpreted episodes:** 4,280 contract violations and 60 no-version outcomes in all.

**How these are counted.** Every naive record is a valid execution of an unjustified switch, and is counted
only as such. At pause index 0 a switch involves no mixing, which is why mixing is below 100%.

**No-version outcomes** come only from two-fact branches. World 802 had none among its supplied episodes. The
per-world witness shows that the instrument detects them whenever the facts allow.

### Migration and release

- **Eligibility:** migration was possible in 164,940 of 186,770 supplied episodes (88%). Where it migrated, its
  result was always equivalent to the fresh S1 run, by a proof that the verifier re-derived.
- **Cost:** as in the pilot, migration does restart's module work plus lookups. It is an equivalence check, not
  an optimisation. Restart remains the default.
- **Releases:**
  - interpreted pin, restart and migrate records were each released 56,670 times. Every field of every release
    decision was re-derived: the released answer against the session's own outcome, the reason, the effective
    threshold and its supporting step;
  - every threshold was 0.0, so no release was refused by a threshold in the confirmation.
- **The release gate at non-zero thresholds** is covered elsewhere: by the deterministic fixture (a score of
  0.95 admitted at 0.90 and refused at a stricter C1), and by the non-ceiling development run in
  `tests/test_ti_tnn_integration.py`.

## What this establishes, and what it does not

**It establishes**, within a finite synthetic route world with supplied types, relations and entities:
- **The update path:** a learned question → query → recurrence → pause → checked update → resume path kept every
  declared contract. That covers complete-context activation and rollback, binding of paused runs to complete
  contexts, exact pin and restart, proof-carrying migration, and release under the right certificate.
- **The scope of each check:**
  - the declared episode sets passed at every exercised pause index (supplied episodes at budget 16,
    interpreted episodes at their parsed budgets);
  - separate exhaustive traversals covered budgets 1–16;
  - the independent verifier reconstructed execution, policy equivalence and every field of every release
    decision;
  - activation mechanics and injected faults remain producer-record evidence.

**It does not establish:**
- **Learned robustness.** The learned parts were at the ceiling, and the five inits of a world behaved
  identically. L1–L4 are not robustness evidence.
- **Behaviour under imperfect learned parts.** That is exercised only in development: a non-ceiling run with
  imperfect S0 operators, learned corrections, a rejected candidate and non-zero thresholds, rebuilt in the
  test suite.
- **Anything about natural language,** other model families, or the safety of updates in general.
- **That each exported atomic table is its snapshot's behaviour.** That was checked when the run was produced,
  and by deterministic fixtures. The update gates, the activation mechanics (C4) and the injected faults are
  likewise the producer's records.
- **Simultaneous guarantees.** The certificate guarantees are per snapshot, not across a unit's 13 contexts.
  Pause and resume happened in one process.

## How it got here: three reviews

1. **Revision 1** ([draft](../protocols/integration_test_v1.md#history)). The review found four gaps:
   - the calibration interface was not bound to what serves;
   - activation swapped snapshots only;
   - declared changes could differ from actual module changes, so a learned correction of an old error broke
     migration (a counterexample, kept as a fixture);
   - populations were unclear.

   The repair merged as #57.
2. **Revision 3.** The review found four more gaps:
   - the checker could count a corrupted naive trace as a genuine failure;
   - pin accepted another complete context on the same snapshot;
   - several migration-proof fields were never verified;
   - the resumed-question path skipped the release gate.

   These were repaired at `bb9816e`. The complete export and the independent verifier followed at `433b208`,
   with 17 deliberate corruptions each rejected.
3. **The development pilot** (worlds 100–101). It is development diagnostics checked online, with no complete
   export; see the protocol's history. Two definitions were corrected during it:
   - naive failures are judged on outcomes;
   - costs count only work after the pause.

## Evidence

**In the bundle,** at `experiments/results/ti_tnn/integration_test/`:
- `20261022T205617/run.json` and the per-unit export for all 15 units (`unit.json`, `interpreter.json.gz`,
  `tables.json.gz`, `certificates.json.gz`, `contexts.json.gz`, `exhaustive.json.gz`, `episodes.jsonl.gz`);
- `verification.json`, the verifier's complete output;
- `session_release.json`, the sha256 of the 15 session files.

**As release assets:** the session files are in the release `ti-tnn-integration-test-sessions-v1` (70.9 MB).

**Rebuilding everything:**

```bash
gh release download ti-tnn-integration-test-sessions-v1 --repo seanm27lol/universa-recurrent --dir integration_sessions
python experiments/results/ti_tnn/make_tables.py --integration-sessions integration_sessions
```

The second command:
- checks each session file against its committed hash;
- stages it next to the bundled export;
- re-runs the frozen verifier, about five minutes;
- requires the result to equal `verification.json`.

**The run:**
- **source:** `e03075a`, with an unmodified tree;
- **compute:** CPU, four threads, low priority;
- **wall time:** 9.5 hours for 15 units (1,639–2,968 s each). Another session's job on the machine slowed the
  later units; that job was logged and not acted on, and it does not affect results, which are deterministic
  on CPU.

## Corrections (2026-10-23, after an outside review of #58)

No reported number changed. The review downloaded the published session records and reconstructed all 15 units.
It also bound every one of the 973,760 session records to its declared episode, and found no mismatch.

| What changed | Why |
|---|---|
| **Session binding.** The verifier binds each embedded session to its declared episode: question, budget, relations, policy, snapshots and contexts, and pause index where the record carries one. For interpreted episodes, it also binds the item's proposed query and the release decision's score, contexts and certificates. A pin record is the uninterrupted C0 run whatever its pause index, so its content does not depend on it | The verifier checked that a record occupied a declared slot and that it verified on its own, but not that it belonged to that slot. A valid session for another question passed |
| **The frozen inventory.** Frozen verification requires the frozen inventory and configuration (worlds 800–802 × inits 0–4, budgets 1–16, the recipe and the study constants) before verifying anything. No mode accepts an empty, repeated or incomplete inventory | Frozen mode had checked only the source binding. A run with no worlds or units verified as a complete success |
| **Certificates and metadata.** Certificates must carry the expected alpha and delta, and in frozen mode the expectation must carry the frozen ones. Expectation and context identities are recomputed from their parts, and every interpreter record's split and category is checked against its regenerated draw | A certificate for another risk level, with consistent statistics, passed the certificate check |
| **The scope sentence.** The sentence under "It establishes" now separates the scopes of the episode checks, the exhaustive traversals and the producer-record evidence | It had combined them |

**The corrected verifier reproduced the bundled `verification.json` exactly** from the published evidence, with
the session files staged as the release provides them. Each counterexample is a regression test that the
corrected verifier rejects. Run against the verifier merged in #58 (corrected after the review of #59):
- **accepted by it:**
  - a valid session from another case, both supplied and interpreted;
  - a restart from another pause index;
  - an empty run, a missing world, and another configuration;
- **rejected by it for other reasons:**
  - a missing init and a repeated unit, because the units did not match the run's own declaration;
  - a missing budget, a construction failure, because the exhaustive digests no longer re-derived;
- **accepted only by its certificate check:** a re-derived certificate at another alpha. Placed in the export, it
  was rejected, because the context's certificate identity no longer matched.


## Corrections (2026-10-23, after an outside review of #59)

No reported number changed. The review found the three findings of the #58 review resolved: session binding,
the frozen inventory, and certificate risk settings. It also audited every published release record; that
audit supports the published results. It found one remaining omission, in release verification.

| What changed | Why |
|---|---|
| **Release records, field by field.** The verifier re-derives every field of every release record, and requires exactly the declared records (C0's new requests; each activated update's new and resumed requests). For interpreter draws: admission, release, reason, threshold and supporting step. For interpreted sessions, also the released answer and outcome reason, against the session's own outcome | The verifier merged in #59 compared admission and release, and a released record's threshold. A release decision that released an answer its session never produced passed, and a missing record counted as "not released" |
| **Release wording.** The R row, "Releases" and the scope list above now say "every field of every release decision" | They said release decisions were re-derived. The verifier merged in #59 did that only in part |
| **The counterexample list** under the previous corrections now says what the verifier merged in #58 did with each, as run against it | It said that verifier accepted every counterexample. Some it rejected for other reasons, and the certificate counterexample was accepted only by its certificate check |

**The corrected verifier reproduced the bundled `verification.json` exactly** from the published evidence. Each
new counterexample is a regression test. Each was run against both verifiers: the one merged in #59 accepted it
and the corrected one rejects it:
- **an interpreted release decision that differs from its own session in one field**, while still saying
  "released" under the right score, contexts and certificates: the answer, the outcome reason, the effective
  threshold, the supporting step or the reason;
- **an interpreter release record that does not re-derive:** another reason for a pre-execution refusal, a
  threshold on a refused record, no supporting step on a released record, or a missing record.

A further test requires the verifier's release rule, written without the producer's certificate code, to equal
the adapter's on every path, including those the confirmation never reached: certificates without a shared
interpreter, and an uncertified side.
