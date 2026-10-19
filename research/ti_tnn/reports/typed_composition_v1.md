# Learned relation operators on a finite typed world: all six predictions hold — 2026-10-18

**Protocol:** [`protocols/typed_composition_v1.md`](../protocols/typed_composition_v1.md), frozen at
`ef604a1` before any run on the confirmation worlds, with the analysis (`evaluate()` in
`bench/typed_composition.py`) and its test. **Raw results:**
[`experiments/results/ti_tnn/typed_composition/`](../../../experiments/results/ti_tnn/typed_composition/).

## The question

The [typed runtime](../integrated/README.md) (plan PR 3) ran on lookup tables. This experiment trains
small networks as the relation operators instead, plugs them into the same runtime and checks them against
the same independent reference. Each network is trained on single facts only (one relation applied to one
entity), never on a chain.

Three questions:
- Do these operators reliably learn the facts?
- Does one network shared across every relation keep two relations with the same endpoints apart?
  (`capital_of` and `largest_city_of` both map Country to City.)
- Do the controls fail where they should?

## Answer

**Yes to all three, in 75 runs:** 3 fresh worlds × 5 initializations × 5 families.
- **Every specialist, shared-network and mixture run learned every fact:** 45 of 45 runs at 1.000 on
  every relation.
- **The shared networks never confused the two Country → City relations:** 1.000 on both, on the
  countries where they differ, in all 30 runs.
- **The controls failed as designed.**
  - The endpoint control, which gives the two relations one operator, scored 0.32–0.50 on the weaker of
    the two.
  - The answer-only model, which has no typed interface, answered 4–8% of 2-step chains.

**Only predictions 1 and 2 are findings; 3–6 hold by construction or by design.** The brief recorded which
is which before any run:
- **Prediction 3** holds whenever the two relations differ somewhere: one operator cannot give two answers
  for one input.
- **Predictions 4 and 5** follow from 1: modules pass discrete values, so a chain can only fail through an
  atomic error.
- **Prediction 6** is expected: the answer-only model never saw a chain.

The predictions confirm that the plumbing works with learned modules in place. They do not show learned
composition.

## Results

Each row is 15 runs (3 worlds × 5 inits): the lowest value, then the mean.

| Family | Atomic, weakest relation | Same-endpoint, weaker relation | 2-step chains | 3-step chains | Round trips (reported only) | Intermediate values | Parameters (world 200) |
|---|---|---|---|---|---|---|---:|
| `specialists` | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 37,035 |
| `shared_dense` | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 22,556 |
| `moe` (flat top-1, 4 experts) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 71,956 |
| `endpoint` (negative control) | 0.417 / 0.500 | 0.316 / 0.418 | 0.673 / 0.692 | 0.568 / 0.592 | 0.861 / 0.867 | 0.759 / 0.771 | 29,020 |
| `answer_only` (no interface) | 1.000 / 1.000 | 1.000 / 1.000 | 0.039 / 0.062 | 0.222 / 0.265 | 0.702 / 0.718 | none exist | 26,684 |

**The worlds.**

| World | Countries | Cities | Countries whose capital differs from their largest city | Atomic facts |
|---|---:|---:|---:|---:|
| 200 | 24 | 79 | 21 | 285 |
| 201 | 24 | 79 | 19 | 285 |
| 202 | 24 | 93 | 22 | 327 |

Each world has 6 operators and an airport per city.

**Chains per world.** Every start entity on every well-typed path of length 2 or 3 is scored:
- 2-step: 285 (worlds 200 and 201), 327 (202);
- 3-step: 206, 206, 234;
- round trips: 302, 302, 330.

A round trip is a path that leaves an entity and comes straight back, such as `capital_of` then
`country_of`.

**Answering by chance, without the start entity.**
- A uniform guess within the answer's type scores 0.04–0.06 on chains.
- Always giving each path's most common answer scores 0.10–0.13.

**Training cost.** About 1–3 seconds per run on 4 CPU threads; the whole confirmation took under 2.5
minutes.

**Final training loss.**

| Family | Final training loss |
|---|---|
| specialists, shared dense, answer-only | below 1e-5 |
| `moe` | 0.010–0.015, mostly the balance term, which is at least 0.01 |
| endpoint control | 0.14–0.16, the irreducible conflict between the two relations |

**Parameters.**
- **The families are not matched in parameters,** and the mixture selects one of its 4 experts per query (this implementation computes all four; see the corrections below).
- **Every family is far larger than the roughly 300 facts it stores.** This experiment tests whether the
  facts are learned and kept apart; it says nothing about efficiency.

## Predictions (frozen before any confirmation run)

| # | Prediction | Outcome | What it is evidence of |
|---|---|---|---|
| 1 | Atomic competence (every relation ≥ 0.99) on at least 4 of 5 inits in every world, for specialists, shared dense and mixture | **Holds:** 15 of 15 in every family | A finding: these operators reliably memorize a finite world's facts |
| 2 | Shared dense and mixture ≥ 0.99 on both same-endpoint relations, over the countries where they differ, on at least 4 of 5 inits per world | **Holds:** 1.000 in all 30 runs | A finding: sharing parameters across relations did not collapse them here |
| 3 | The endpoint control scores ≤ 0.6 on at least one of the two relations in every run | **Holds:** at most 0.500 | By construction: shows the test can detect collapse |
| 4 | Every competent run scores ≥ 0.95 on 2- and 3-step chains | **Holds:** 1.000 in all 45 competent runs | By construction, given 1: discrete values pass between modules |
| 5 | Every competent run scores ≥ 0.95 on intermediate values | **Holds:** 1.000 | By construction, given 1 |
| 6 | Answer-only scores < 0.5 on 2- and 3-step chains in every run | **Holds:** at most 0.084 and 0.296 | By design: it never saw a chain |

## The answer-only baseline above chance on 3-step chains

The answer-only model scores 0.22–0.30 on 3-step chains, roughly twice the most-common-answer baseline.
It learned a shortcut. I checked this after the confirmation (an exploratory check outside the frozen
protocol) by retraining it for world 200, init 0, which reproduces the recorded run, and scoring each path
separately.
- **In that run, two paths carry 48 of its 57 correct 3-step answers:** a city's country, then that country's capital (or largest
  city), then that city's airport. On these the model scores 0.30.
- **On both paths it always answers with the start city's own airport** (agreement 1.00). That is right
  exactly when the start city is itself the capital (or the largest city), which is 0.30 of starts.
- **It ignores the first two relations** and applies the last relation it knows how to apply to the
  start entity.

So the number reflects a coincidence in the world, not partial composition.

## Reproducibility

Rerunning world 200, init 0 for all five families on the same machine reproduced every recorded field
exactly, including the final training losses; only the timings differ. That is a single check on a
single machine. CPU results may differ between PyTorch builds or hardware.

## What it does not show

- **Discovery or interpretation.** The types, relations, entity vocabulary and correct program are all
  supplied. Mapping a question to its program is plan PR 6.
- **Generalization to unseen facts.** Every scored chain is built from facts the model was trained on.
  Atomic competence here is memorization of a finite world.
- **That composition was learned.** It is guaranteed once the facts are learned, because the runtime
  passes discrete values.
- **That sparse or specialized operators help.** Specialists, the shared network and the flat mixture
  are indistinguishable here: all are perfect. A task this small cannot separate them.
- **Faithfulness under intervention.** The trace records the values passed, and they match the reference.
  Whether replacing a recorded value changes the answer as predicted is plan PR 5.
- **Categorical laws.** Round trips were scored (1.000 in every competent run), but no law-defect
  measurement or tolerance was registered. The ledger row for declared equations stays unestablished.

## Departures from the plan's PR 4 file list

- **The world generator is in `integrated/relations.py`** (from PR 3), not a separate
  `bench/typed_world.py`.
- **The plan's Experiment B lists more controls** than are run here: capacity-matched variants, soft-law
  penalties, shuffled routing. With every gated family at ceiling, they have nothing to separate here.
  They belong with a harder task.

## Execution

- **Source.** Committed as `ef604a1` immediately before the run, with a clean tree. The run file does not
  record its commit.
- **Run.** 2026-10-18, finished 18:57 UTC. CPU only (`CUDA_VISIBLE_DEVICES=` empty, 4 threads, `nice`),
  run from the run directory's checkout of `main` with the frozen branch's script and the protocol's
  arguments.
- **Software.** PyTorch 2.10.0, Python 3.11.14, aarch64.
- **Verdict.** `verdict.json` was written by the frozen `evaluate()`; `make_tables.py` rebuilds it from the
  bundled runs.
- **Development pilot.** It ran on worlds 100–101 and inits 100–101 before freezing, and is disclosed in
  the protocol. No confirmation world was used before the freeze.

## Corrections after review (2026-10-18)

- **The mixture does not compute sparsely.** The report said it "runs only one of its 4 experts per query".
  It selects one, but `TopOneMoE.forward` computes all four and keeps the selected output. Its compute is
  that of all experts. The memorization results are unaffected; this implementation supports no
  sparse-compute claim.
- **"By construction, given competence" was too strong.** Chain and intermediate accuracy follow by
  construction from *exact* atomic correctness. The registered 0.99 threshold allows a wrong fact, and one
  wrong fact can break many chains. Every run here had atomic accuracy exactly 1.000, so predictions 4 and
  5 still hold as reported, but they rest on that observation, not on prediction 1 alone.
