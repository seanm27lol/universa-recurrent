# Execution records, replay and interventions: every check separates what it should — 2026-10-18

**Protocol:** [`protocols/semantic_interventions_v1.md`](../protocols/semantic_interventions_v1.md), frozen
at `75cb47a` before any run on the confirmation worlds, with the analysis (`evaluate()` in
`bench/semantic_interventions.py`) and its test. **Raw results:**
[`experiments/results/ti_tnn/semantic_interventions/`](../../../experiments/results/ti_tnn/semantic_interventions/).

## The question

[Plan PR 4](typed_composition_v1.md) showed learned operators answering correctly through the typed
runtime. A correct answer does not show that the trace describes how it was computed. This run asks:
- **Records:** is every execution record complete, and does it replay exactly?
- **Interface:** does each module depend only on the value it is recorded as consuming?
- **Separation:** can the checks tell four different failures apart?

## Answer

**Yes, on 83,616 recorded executions and 387,264 interventions, in 72 runs:**
- 3 fresh worlds;
- learned module sets at 5 initializations each (the four PR 4 families);
- the table modules;
- three deliberately faulty sets.

All seven frozen predictions hold.

**What was built.**
- **Records:** an execution-record schema (`ti_tnn_execution_v1`).
- **Checker:** an independent checker that reads only the record and the world's facts.
- **Replay** against the recorded registry.
- **Probe:** logs every module call, including calls one module makes to another.
- **Disabling:** a test that disables every module a program does not use.
- **Interventions,** scored two separate ways:
  - **History independence:** the answer depends on the replacement value alone, not on how the run got
    there.
  - **Counterfactual semantic accuracy:** the answer matches the reference's answer from the replacement
    value.

**Each faulty set was caught by its own check and by no other.**

| Module set | Ordinary answers | Replays | Probe: undeclared calls | Unexecuted modules disabled: unchanged | History independent | Counterfactual accuracy | Law defects |
|---|---|---|---:|---|---|---|---:|
| learned `specialists`, `shared_dense`, `moe` (45 runs) | 1.000 | all | 0 | all | all | 1.000 | 0 |
| learned `endpoint` (15 runs) | 0.77–0.81 | all | 0 | all | all | **0.64–0.72** on same-endpoint suffixes | **0** |
| `table` | 1.000 | all | 0 | all | all | 1.000 | 0 |
| `question_reading`: `country_of` answers with the question's country | **1.000** | all | 0 | all | **0.88–0.90** | 0.88–0.90 | 0 |
| `hidden_dependency`: `capital_of` quietly calls `largest_city_of` | **1.000** | all | **417–477** | **0.67** | all | 1.000 | 0 |
| `swapped`: `capital_of` returns the largest city | 0.77–0.81 | all | 0 | all | all | 0.65–0.72 on same-endpoint suffixes | **0** |

**About the table.**
- Ranges run over runs. "All" means every execution or intervention in every run of that set.
- The worlds have 1,038, 1,238 and 1,208 executions and 4,836, 5,716 and 5,584 interventions per module
  set.

**What each row shows.**
- **The learned operators passed every check, in every run.** Every record replayed. Every module
  received exactly the recorded value and made no other call. Disabling unused modules changed nothing.
  Every intervention was history independent, and every counterfactual answer was correct.
- **The question-reading module answered every ordinary question correctly.** Replay, the probe and
  disabling all passed. Only interventions caught it. In 10–12% of interventions, its answer differed
  from the same suffix run on its own: replacing an intermediate city with a city in another country
  left its answer unchanged.
- **The hidden dependency answered everything correctly and replayed exactly.** The probe logged its
  calls to `largest_city_of`. Disabling that module where programs do not use it changed a third of all
  executions.
- **The swapped module is lawful but wrong.** Both declared round-trip laws hold (country_of of the
  capital, or of the largest city, is the country). Its records replay and its interventions are history
  independent. Only the independent checker, comparing with the facts, flags it. Its `capital_of`
  accuracy is exactly the share of countries whose capital is their largest city (2, 4 and 6 of 24 in
  the three worlds).
- **The learned endpoint control behaved like the swapped module, in all 15 runs.** It gave
  `capital_of` and `largest_city_of` one operator, so it answers both with some city of the right
  country. It therefore has zero law defects and a fully faithful trace, but its answers are wrong. This
  is a learned instance of the [guarantee-boundary counterexample](../integrated/README.md): lawful,
  faithful, and wrong.

## Predictions (frozen before any confirmation run)

| # | Prediction | Outcome | What it is evidence of |
|---|---|---|---|
| 1 | Every record passes schema, type and integrity checks and replays; every intervention is recorded | **Holds** (83,616 executions; 387,264 interventions) | By construction: the instruments work at full scale |
| 2 | Pure module sets: exact consumed values, no undeclared calls, disabling changes nothing, every intervention history independent | **Holds** | By construction for pure modules |
| 3a | Specialists, shared dense and mixture competent on at least 4 of 5 inits per world | **Holds** (15 of 15 each) | **A finding:** PR 4's competence replicates on three more fresh worlds |
| 3b | Counterfactual accuracy at least 0.95 in competent runs | **Holds** (1.000) | Follows from competence |
| 4 | The question-reading module is caught only by interventions | **Holds** | By design: interventions catch what replay and the probe cannot |
| 5 | The hidden dependency is caught by the probe and by disabling | **Holds** | By design |
| 6 | The lawful change is classified correctly: zero law defects, faithful, wrong exactly where the relations differ | **Holds** | By design |
| 7 | The endpoint control is faithful but wrong (below 0.95 on same-endpoint counterfactuals) | **Holds** (0.64–0.72) | By construction, given the shared operator |

## What it means

- **On learned modules, the trace describes the computation.** In this runtime, with these modules,
  every recorded value was the value consumed. The answer depended on nothing else, so intervening on a
  recorded value changes the answer as the reference predicts. This holds by construction here: the
  modules are pure functions, and the runtime rebuilds every value it passes.
- **The checks are not redundant.** Correct answers, exact replay, consumed-value probes, disabled
  computation, interventions and the independent checker each caught a fault the others missed. A
  system that passes only some of them has not shown the rest.
- **Laws and faithfulness do not imply correctness.** Both the swapped module and the learned endpoint
  control satisfy every declared law and have perfectly faithful traces, yet they answer wrongly. Only
  comparison with the facts catches them.

## A runtime fix in this PR

Before this PR, the runtime passed a module's output object on unchanged. A module could return a value
carrying hidden data (an extra attribute, or a subclass), and the next module could read it. The trace
never showed it. The runtime now rebuilds every value as a plain `TypedValue` before recording it and
passing it on. A test checks that smuggled data is dropped. No earlier result depended on this: PR 4's
modules return plain values.

## Limits

- **The faulty sets are examples, not an exhaustive search.** A module could read state these tests
  never vary.
- **Disabling is per module object.** For shared networks, the shared parameters stay on, since the
  executed relations need them.
- **Supplied programs and finite worlds.** No router or interpreter is involved yet (plan PR 6).
- **Provenance is not authenticated.** A record says what was computed, not where or by whom.

## Execution

- **Source.** Committed as `75cb47a` immediately before the run, with a clean tree. The run file does not
  record its commit.
- **Run.** 2026-10-18, finished 19:38 UTC; about 6.6 CPU-minutes on 4 threads (`CUDA_VISIBLE_DEVICES=`
  empty, `nice`). It ran from the run directory's checkout of `main` with the frozen branch's script and
  the protocol's arguments.
- **Software.** PyTorch 2.10.0, Python 3.11.14, aarch64.
- **Verdict.** `verdict.json` was written by the frozen `evaluate()`; `make_tables.py` rebuilds it from the
  bundled runs.
- **Development pilot.** It ran on worlds 100–101 before freezing, and is disclosed in the protocol.

## Corrections after review (2026-10-18)

- **The runtime fix in this PR was incomplete.** It rebuilt the outer `TypedValue` but kept its fields by
  reference. A `str` subclass carrying an extra attribute passed the domain check, survived the rebuild
  and reached the next module. The record showed only the visible string, the probe reported exact
  consumption, and the run answered differently from its own suffix replayed from the record. The repair
  rebuilds every passed value from the declared domain's own plain strings, with a regression test. No
  reported run was affected: the learned and table modules return plain strings.
- **Prediction 3b needs exact atomic correctness, not 0.99 competence.** Every run had atomic accuracy
  exactly 1.000, so it holds as reported.
