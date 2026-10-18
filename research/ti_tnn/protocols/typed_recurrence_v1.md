# Bounded recurrence, pause/resume and cost accounting (frozen brief, integration plan PR 7)

**Status: FROZEN 2026-10-18 (UTC), before any run on the confirmation worlds.** Branch
`ti-tnn-recurrence`, stacked on the interpreter PR (#48).

## The question

Plans PR 4–6 ran programs of at most three steps. Here the number of steps is not known in advance: "which
is the first airport reached from X that operator O runs?" Each iteration does three things:
- follows the airport's one onward connection;
- stops at a dead end (a terminal) or a cycle (an airport already visited);
- otherwise tests the new airport's operator.

A budget bounds the iterations.

Four questions:
- **Correctness:** does bounded recurrence with learned modules give the reference's answer and
  stopping reason?
- **Continuation state:** is the state complete, so that a run paused after any iteration and resumed
  from JSON with a fresh registry ends identically?
- **Cost:** what does recurrence save over unrolling to the full budget?
- **Overhead:** how much of the time goes to bookkeeping rather than computation?

## Design

Code: `research/ti_tnn/integrated/recurrence.py`, `costs.py`; `route_world` in `relations.py` and
`reference_until` in `reference.py`; and `research/ti_tnn/bench/typed_recurrence.py`.

**Worlds.** `route_world` with seeds **500, 501, 502**, unused by any earlier run:
- 48 countries and about 170 airports;
- each airport has one onward connection, to another airport, or to itself at a terminal (8% of
  airports).

Earlier worlds are unchanged: a test checks that PR 4's recorded world identity still matches.

**Queries.** Every (airport, operator) pair, with a budget of 16 iterations.

**Systems.**

| System | What it is |
|---|---|
| **`recurrent` (gated)** | Iteration with early stopping, using learned specialist modules (PR 4's recipe, now including `connects_to`), inits 0–4 |
| `fixed_depth` | The same modules, unrolled to the full budget: 16 `connects_to` and 16 `operator_of` applications every time, with the first valid match picked afterwards |
| `recurrent_table` | Recurrence with table modules (a diagnostic) |

**Measurements, per query.**
- the answer and stopping reason, against `reference_until`, which reads the facts directly;
- module applications and MACs (from layer shapes), and continuation-state size;
- wall time, the median of 3 timed runs, split into time inside modules and bookkeeping. The run uses
  one CPU thread, so that timing is not mixed with thread scheduling. Each timed run's final state is
  compared with the scored run's.
- for `recurrent`: pause after every possible iteration count, write the state to JSON, and resume
  with a fresh registry (deep-copied networks, whose parameters are re-hashed to the recorded identity).

## Development pilot (disclosed)

Worlds 100–101, inits 100–101, 3 units. It changed nothing in the design.

| Measure | Result |
|---|---|
| Answers and reasons | every query correct |
| Pause/resume | identical at all 2,881 and 2,386 pause points |
| Module applications, recurrent / fixed depth | 6.4–7.7 against 32 (0.20–0.24) |
| Median wall time, recurrent / fixed depth | 323–369 μs against 1,610–1,684 μs |
| Bookkeeping share, recurrent | 0.19 |
| Bookkeeping share, with table modules | 0.97 |

## Predictions (recorded before any confirmation run; one-sided)

A run is competent if every relation's atomic accuracy is at least 0.99.

1. **Atomic competence** (now including `connects_to` over about 170 airports): competent on at least
   4 of 5 inits in each world.
2. **Recurrence matches the reference.** In every competent run, the answer and stopping reason match on
   at least 0.99 of queries.
3. **Fixed depth agrees.** Its answer and reason equal recurrence's on every query, in every run.
4. **Pause/resume is exact.** In every run, every pause point resumes to an identical final state.
5. **Timed equals scored.** Every timed run's final state equals the scored run's, for every system.
6. **Fewer applications.** Recurrence's mean module applications per query are at most half of fixed
   depth's, in every run.
7. **Lower latency.** Recurrence's median wall time per query is below fixed depth's, in every run.

**What holds by construction:**
- **2–5** hold for pure modules, given competence:
  - recurrence reads the same facts the reference does;
  - fixed depth computes the same function;
  - the state is the whole run;
  - timing wraps the same objects.
- **6** follows from the route networks' distances: the budget is 16, and the pilot's mean was under 8
  iterations.
- **7** follows from 6 unless overhead dominates.

**Prediction 1 is the only empirical one.** The others check that the instruments work at full scale.

**The informative measurement** is not a prediction: what bookkeeping costs relative to computation.

## Limits

- **Pure modules only.** The state is complete because the modules keep no state of their own. The
  tests show that a stateful module breaks pause/resume and that the check sees it.
- **The halting test is symbolic** (operator equality after a learned `operator_of`); halting is not
  learned.
- **CPU, one thread, tiny modules.** Latencies are for this executor, not for any accelerator.
- **There is no neural recurrent baseline.** Both systems use the typed executor; the comparison is
  early stopping against unrolling.

**Run:**
`python research/ti_tnn/bench/typed_recurrence.py --worlds 500 501 502 --inits 0 1 2 3 4`, which also
writes `verdict.json` via `evaluate()`.

## Execution record (appended after the run; nothing above was changed)

- **The run.** The protocol's command, from `7ba0b79`, 2026-10-18, finishing 20:28 UTC. 45 system runs (15
  units × 3 systems), one CPU thread, with no failures. `verdict.json` was written by `evaluate()` as
  frozen.
- **All seven predictions hold.** See the [report](../reports/typed_recurrence_v1.md).
