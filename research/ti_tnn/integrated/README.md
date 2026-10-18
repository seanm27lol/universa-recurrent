# Integrated typed execution (TI-TNN × CDL)

**Status: planned.** No code here yet; the [integration plan](../integration_plan.md) says what will
land and in which order. Nothing in this directory is an experiment result.

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

## Planned layout

| File | Plan PR | Purpose |
|---|---|---|
| `contracts.py` | 3 | `TypeSpec`, `RelationSpec`, `ModuleVersion`, `TypedValue`, `Program`, `Decision`, `Budget`, `RunState` |
| `relations.py` | 3 | The finite world: types, relations with unique identities, facts |
| `registry.py` | 3 | Immutable module versions |
| `runtime.py` | 3 | Type-checked execution under a budget, recording the values actually consumed |
| `reference.py` | 3 | An independent finite-world evaluator that reads facts directly |
| `modules.py` | 4 | Learned relation operators |
| `records.py`, `verification.py`, `interventions.py` | 5 | Execution records, an independent checker, counterfactual interventions |
| `routing.py`, `calibration.py` | 6 | Jev as a planning policy; versioned calibration and abstention |
| `recurrence.py`, `costs.py` | 7 | Bounded recurrence with explicit continuation state; complete cost accounting |
| `updates.py`, `update_checks.py` | 8 | Offline module replacement with fixed external checks and rollback |

## Limits, already known

- **The structure is supplied.** The types, relations and entity vocabulary come from the benchmark. A
  success shows computation over supplied structure, not that a model discovered it.
- **Several properties hold by construction for a supplied program.** Composition and trace faithfulness
  follow from the discrete interface; the plan's review amendments say how they will be reported.
- **Calibration is not integrity.** A router's calibration says nothing about whether a downstream
  module changed.
