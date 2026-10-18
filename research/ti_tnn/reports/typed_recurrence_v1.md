# Bounded recurrence with a complete continuation state: exact, and a quarter of the unrolled cost — 2026-10-18

**Protocol:** [`protocols/typed_recurrence_v1.md`](../protocols/typed_recurrence_v1.md), frozen at `7ba0b79`
before any run on the confirmation worlds, with the analysis (`evaluate()` in `bench/typed_recurrence.py`)
and its test. **Raw results:**
[`experiments/results/ti_tnn/typed_recurrence/`](../../../experiments/results/ti_tnn/typed_recurrence/).

## The question

Every earlier integration experiment ran programs of at most three steps. Here the number of steps is not
known in advance. The question is "which is the first airport reached from X that operator O runs?", on a
route network where every airport has one onward connection, and some routes end at a terminal or loop
back. The run must find the answer, or abstain for the right reason:
- a dead end;
- a cycle;
- running out of its 16-iteration budget.

Four questions:
- **Correctness:** does bounded recurrence with learned modules get the answers and the stopping reasons
  right?
- **Continuation state:** is the state complete, so that a run can be paused anywhere and resumed
  exactly?
- **Cost:** what does stopping early save over unrolling to the full budget?
- **Overhead:** what share of the time is bookkeeping?

## Answer

**Correctness: every query, every way of stopping.** Learned specialist modules were competent on every
relation in all 15 runs, including the new `connects_to` (about 170 airports). Recurrence matched the
reference's answer and stopping reason on all 15,480 queries (3 worlds × 5 inits; 990–1,062 queries per
world). Every query was found, or abstained at a dead end, a cycle or the budget, exactly as the facts
dictate.

**Continuation state: complete.** Each run was paused after every possible iteration and written to
JSON. It was then resumed with a fresh registry, whose deep-copied networks were re-hashed to the
recorded identity. All 53,970 pause points ended in exactly the uninterrupted final state.

**Cost: early stopping cut it to a quarter.**

| Per query, 15 runs | Recurrence (early stopping) | Fixed depth (unrolled to 16) |
|---|---|---|
| Module applications, mean | 8.0–9.7 | 32 |
| MACs, mean | 62k–77k | 241k–253k |
| Wall time, median | 370–514 μs | 1,674–1,712 μs |
| Share of wall time spent on bookkeeping | 0.19–0.21 | 0.02 |
| Continuation state, mean size | 3.5–4.0 kB | 10.9 kB |
| Accuracy | 1.000 | 1.000 |

Recurrence used 25–30% of fixed depth's module applications and 22–31% of its wall time, at the same
accuracy. Every timed run reproduced its scored run exactly.

**Overhead: bookkeeping is a fifth of the time, and could dominate.**
- **Where it goes.** With learned modules, about 20% of recurrent wall time is outside the modules:
  - type checks;
  - copying the continuation state each iteration;
  - recording each application.
- **Table modules show its size.** Their lookups take 2–3 μs per query, so bookkeeping is 97% of the
  58–89 μs.
- **What it means.** Recording a complete, resumable state costs about 25 times the computation for
  modules as cheap as lookups. For any module much cheaper than these networks, the bookkeeping would
  decide the cost.

## Predictions (frozen before any confirmation run)

| # | Prediction | Outcome |
|---|---|---|
| 1 | Atomic competence, `connects_to` included, on ≥ 4 of 5 inits per world | **Holds** (15 of 15; every relation 1.000) |
| 2 | Recurrence matches the reference on ≥ 0.99 of queries in competent runs | **Holds** (1.000) |
| 3 | Fixed depth gives the same answer and reason on every query | **Holds** |
| 4 | Every pause point resumes to an identical final state | **Holds** (53,970 of 53,970) |
| 5 | Every timed run equals its scored run | **Holds** |
| 6 | Recurrence's mean applications ≤ half of fixed depth's | **Holds** (0.25–0.30) |
| 7 | Recurrence's median wall time below fixed depth's | **Holds** (0.22–0.31 of it) |

**Which are findings.** The protocol said beforehand that 2–5 hold by construction for pure, competent
modules, and that 6 and 7 follow from the networks' distances. **Prediction 1 is the only empirical
one.** The others show the instruments and the continuation contract working at full scale. The
bookkeeping measurement is the informative number.

## How the runs stopped

| World | Found | Dead end | Cycle | Budget exhausted |
|---|---:|---:|---:|---:|
| 500 | 853 | 142 | 52 | 15 |
| 501 | 590 | 286 | 114 | 0 |
| 502 | 660 | 377 | 0 | 7 |

## What it means

- **Bounded recurrence fits the typed runtime cleanly.** With pure modules and a state that holds the
  whole run, a run can stop anywhere, move to another process, and continue identically. That is the
  continuation contract the plan asked for, before any opaque recurrent backend is attempted. The tests
  show a module with hidden state breaking it, and the check catching that.
- **Early stopping is the efficiency lever here,** not module design. The saving, about 75%, is fixed by
  how far answers are along the routes; any executor that stops early would get it.
- **No other baseline qualifies.** Nothing outside the typed executor answers these queries at all, so
  per the plan's efficiency rule the comparison is only within it. This is not an efficiency result
  against neural models.
- **Recording everything has a price.** With cheap modules it is most of the cost. A production
  executor would want cheaper state handling, such as appending rather than copying, or recording
  less. Each option trades against the exact pause/resume shown here.

## Limits

- **Pure modules only.** A module with its own state breaks pause/resume. The runtime cannot prevent
  that; only the check detects it.
- **Halting is symbolic:** operator equality after a learned `operator_of`. Learned halting is not
  tested.
- **CPU, one thread, tiny modules.** Latency is for this Python executor, not an accelerator, and
  includes Python overhead.
- **Training cost is excluded.** It was about 3 s per run for the specialists, and is reported
  separately.

## Execution

- **Source.** Committed as `7ba0b79` immediately before the run, with a clean tree. The run file does not
  record its commit.
- **Run.** 2026-10-18, finished 20:28 UTC. One CPU thread (`OMP_NUM_THREADS=1`, `nice`), with no GPU
  involved. It ran from the run directory's checkout of `main` with the frozen branch's script and the
  protocol's arguments.
- **Software.** PyTorch 2.10.0, Python 3.11.14, aarch64.
- **Verdict.** `verdict.json` was written by the frozen `evaluate()`; `make_tables.py` rebuilds it from the
  bundled runs.
- **Development pilot.** It ran on worlds 100–101 before freezing, and is disclosed in the protocol.
