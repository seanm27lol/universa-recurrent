# Integrated typed execution (TI-TNN × CDL)

**Status: plan PRs 3–8 implemented and tested.**
- **PR 3:** typed contracts, a finite world, an immutable module registry, a type-checked runtime and an
  independent reference evaluator.
- **PR 4:** small learned networks as relation operators (`modules.py`), confirmed against a frozen
  protocol ([report](../reports/typed_composition_v1.md)). They learned every fact of three fresh worlds,
  and a network shared by every relation kept `capital_of` and `largest_city_of` apart.
- **PR 5:** execution records, an independent checker, replay, probes and interventions, confirmed
  against a frozen protocol ([report](../reports/semantic_interventions_v1.md)). On learned modules
  every recorded value was the value consumed. Four deliberately faulty module sets were each caught by
  their own check.
- **PR 6:** a learned interpreter (Jev) that turns controlled-language questions into programs, with
  calibrated abstention ([report](../reports/typed_routing_v1.md)). In distribution it answered 98–100%
  of supported questions with 2 errors in 13,714. Under a phrasing shift, 59 of the 100 answers it
  accepted were wrong.
- **PR 7:** bounded recurrence on a route network, with a continuation state that holds the whole run,
  and cost accounting ([report](../reports/typed_recurrence_v1.md)).
  - Every answer and stopping reason was correct, and runs paused anywhere resumed exactly.
  - Stopping early cost about a quarter of unrolling.
  - Recording the state cost about a fifth of the time.
- **Repairs after review (2026-10-18;** [plan section 12](../integration_plan.md)**):**
  - gate results sealed and bound to the exact snapshot they checked;
  - snapshots as frozen, digested private copies;
  - exact evaluator inventories;
  - values rebuilt from declared domains;
  - matching failure semantics for the fixed-depth control.
- **PR 8:** offline module replacement under independent gates, with atomic activation and exact
  rollback ([report](../reports/local_updates_v1.md)).
  - All 150 specialist updates were local, in state and in behavior.
  - A shared network could not be updated locally. Without rehearsal it corrupted the relation sharing
    the updated one's endpoints.

The [integration plan](../integration_plan.md) says what lands next.

```bash
python research/ti_tnn/integrated/demo.py              # the Ardel example, its trace and the reference
python -m pytest tests/test_ti_tnn_integrated_contracts.py
```

## A concrete example

> Which company operates the airport serving the largest city of Ardel?

Ardel is fictional. The answer takes three typed steps:

| Step | Relation | Type in → type out | Value |
|---|---|---|---|
| 1 | `largest_city_of` | Country → City | Ardel → Port Vey |
| 2 | `airport_of` | City → Airport | Port Vey → Vey International |
| 3 | `operator_of` | Airport → Operator | Vey International → Tessaline Air |

`capital_of` is also Country → City, but Ardel's capital is a different city, Ardel Crossing. Any system
that treats every Country → City relation as the same map gets this wrong. The fixture is built so that
it does.

## The idea in plain language

- **The router (Jev) proposes the operations.** It picks the relations to apply; the runtime checks that
  their types compose before anything runs.
- **Specialists execute them.** Each relation is a module, and modules pass each other explicit typed
  values, here an entity identifier, not a hidden vector.
- **Every value consumed is recorded,** so it can be inspected, replayed, or replaced to test what
  depends on it.
- **The runtime answers or abstains.**

## What different checks establish

| Property | What a passing check establishes |
|---|---|
| Type validity | The declared inputs and outputs compose |
| Algebraic conformance | The declared equations hold: exactly by construction, or numerically within a stated tolerance |
| Calibration | A probability predicts a defined outcome on a defined distribution |
| Semantic correctness | The operations produce the intended answers |
| Trace faithfulness | The recorded intermediate values are the ones the computation actually used |
| Authorization | A separately controlled mechanism permits an update or operation |

None of these substitutes for another. `tests/test_ti_tnn_guarantee_boundaries.py` shows two concrete
cases:
- every categorical law holds and the router's Brier score is unchanged, yet the answer flips;
- a branch's output layer changes while the route probabilities and every categorical defect stay
  identical.

## Layout (done and planned)

| File | Plan PR | Purpose |
|---|---|---|
| [`contracts.py`](contracts.py) | 3 (done) | `TypeSpec`, `RelationSpec`, `ModuleVersion`, `TypedValue`, `Program`, `Decision`, `Budget`, `RunState`, `StepRecord`, `Execution` |
| [`relations.py`](relations.py) | 3 (done) | The finite world: the Ardel fixture, a seeded generator, relations with unique identities, validated facts, restricted-domain laws |
| [`registry.py`](registry.py) | 3 (done) | Immutable module versions; table modules |
| [`runtime.py`](runtime.py) | 3 (done) | Rejects ill-typed programs before any module runs; executes under a budget; records the values passed |
| [`reference.py`](reference.py) | 3 (done) | An independent evaluator that reads facts directly and imports nothing from the runtime |
| [`demo.py`](demo.py) | 3 (done) | The README example, runnable |
| [`modules.py`](modules.py) | 4 (done) | Learned relation operators: specialists, a shared network, a flat top-1 mixture, an endpoint-only control and an answer-only baseline; [`../bench/typed_composition.py`](../bench/typed_composition.py) trains and scores them |
| [`records.py`](records.py) | 5 (done) | Execution-record schema `ti_tnn_execution_v1`, built from the runtime's output, with internal-consistency checks |
| [`verification.py`](verification.py) | 5 (done) | The independent checker: schema, types, semantics against the facts, integrity, provenance (never authenticated) |
| [`interventions.py`](interventions.py) | 5 (done) | Replay, a probe on module calls, disabled unexecuted computation, interventions, law defects |
| [`faults.py`](faults.py) | 5 (done) | Deliberately faulty module sets, used only as positive controls |
| [`routing.py`](routing.py) | 6 (done) | Jev as a planning policy: an interpreter proposes a program (or refuses), and the runtime type-checks and runs it |
| [`calibration.py`](calibration.py) | 6 (done) | Versioned Platt calibration of answer correctness, a Clopper–Pearson-bounded threshold, binding to interpreter, registry and policy |
| [`recurrence.py`](recurrence.py) | 7 (done) | Until-queries with early stopping, dead-end, cycle and budget abstention; a JSON continuation state; exact pause/resume; a fixed-depth control |
| [`costs.py`](costs.py) | 7 (done) | Module applications, MACs from layer shapes, state size, and wall time split into module time and bookkeeping |
| [`updates.py`](updates.py) | 8 (done) | Update records; a deployment with atomic, authority-gated activation and exact rollback |
| [`update_checks.py`](update_checks.py) | 8 (done; repaired after review) | Independent gates (pass, fail or insufficient evidence): record context, identity and scope, types and laws, intended and unaffected behavior, trace and continuation, calibration and resources. A `Checker` seals results bound to what it checked |
| [`snapshots.py`](snapshots.py) | Repairs | Published snapshots: private frozen copies with a digest over all behavior-relevant state, verified before use |
| [`inventory.py`](inventory.py) | Repairs | Exact inventories for frozen evaluators: nothing missing, repeated or extra |

## Limits, already known

- **The structure is supplied.** The types, relations and entity vocabulary come from the benchmark. A
  success shows computation over supplied structure, not that a model discovered it.
- **Several properties hold by construction for a supplied program.** Composition and trace faithfulness
  follow from the discrete interface. The PR 4 protocol labels these outcomes in advance, and the report
  does not count them as findings.
- **Calibration is not integrity.** A router's calibration says nothing about whether a downstream
  module changed.
