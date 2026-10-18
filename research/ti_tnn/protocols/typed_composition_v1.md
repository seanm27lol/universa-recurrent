# Learned relation operators on a finite typed world (frozen brief, integration plan PR 4)

**Status: FROZEN 2026-10-18 (UTC), before any run on the confirmation worlds.** Branch
`ti-tnn-typed-operators`, based on the typed runtime (#43).

**A development pilot ran first,** on worlds 100 and 101 (inits 100 and 101; 15 runs). Every family behaved
as designed, with these as the lowest values across pilot runs:
- specialists, shared dense and the mixture of experts: 1.000 on everything;
- the endpoint control: same-endpoint discrimination 0.32–0.42;
- answer-only: 2-step chains 0.03–0.05.

The pilot fixed the budget (1,500 steps is enough). Its results are not confirmation results.

## The question

In the typed runtime (#43), every module was a lookup table. Can small learned operators replace them, so
that the same typed execution, trace and reference comparison work with learned modules? And do the
controls behave as the plan requires? Specifically:
- a shared network must keep two relations with the same endpoints apart;
- an endpoint-only operator must fail to, which shows the test can detect collapse;
- a model with no typed interface must fail to chain.

## Design (`research/ti_tnn/bench/typed_composition.py`, `research/ti_tnn/integrated/modules.py`)

**Worlds.** `generate_world` with seeds **200, 201, 202**, which no earlier run used: 24 countries, 2–5
cities each, an airport per city, 6 operators, and capital and largest city the same in about 20% of
countries. There are about 300 atomic facts per world.

**Training.** Atomic facts only, never a chain. Full batch, Adam, learning rate 1e-2, 1,500 steps,
embedding width 32, hidden width 64, inits 0–4. Outputs are restricted to the target type's domain (the
type system is supplied).

| Family | Relation operators |
|---|---|
| `specialists` | one network per relation (the typed-specialist design) |
| `shared_dense` | one network over (entity, relation identity) for every relation |
| `moe` | `shared_dense` with a flat top-1 mixture of 4 experts (balance loss 0.01) |
| `endpoint` | one network per (source type, target type): `capital_of` and `largest_city_of` share an operator (negative control) |
| `answer_only` | relation sequence and start entity straight to the answer, with no typed intermediate values; trained on single steps |

**Scoring.**
- atomic accuracy per relation on every fact;
- same-endpoint discrimination: `capital_of` and `largest_city_of` on the countries where they differ;
- chains, with the correct program supplied and run through the typed runtime: non-cancelling 2- and
  3-step paths (gated), and round trips such as `capital_of` then `country_of` (reported only);
- intermediate accuracy: every value a chain passes, against the independent reference.

## Predictions (recorded before any confirmation run; one-sided)

A run is **competent** if every relation's atomic accuracy is at least 0.99.

1. **Atomic competence:** `specialists`, `shared_dense` and `moe` are competent on at least 4 of 5 inits
   in each of the 3 worlds.
2. **Same-endpoint discrimination:** `shared_dense` and `moe` score at least 0.99 on both relations,
   over the differing countries, on at least 4 of 5 inits in each world.
3. **The negative control collapses:** `endpoint` scores at most 0.6 on at least one of the two relations,
   over the differing countries, in every run.
4. **Composition, given competence:** every competent run scores at least 0.95 on the 2-step and the
   3-step path families.
5. **Intermediate values, given competence:** every competent run scores at least 0.95.
6. **No interface, no chains:** `answer_only` scores below 0.5 on the 2-step and the 3-step families in
   every run.

## What is true by construction, and so is not evidence

- **Prediction 3** holds whenever a world has countries whose capital and largest city differ: the
  shared operator receives two different answers for one input. It checks that the test can detect
  collapse, not anything learned.
- **Predictions 4 and 5** follow from prediction 1. Modules pass discrete values, so a chain or
  intermediate error needs an atomic error. They check the plumbing with learned modules in place, not
  learned composition.
- **Prediction 6 is expected** because `answer_only` has never seen a chain. It measures what the typed
  interface contributes under atomic-only training, not a property of the typed modules.

**The informative outcomes are 1 and 2:** whether these families reliably learn the facts, and whether a
shared network keeps same-endpoint relations apart.

## Limits

- **Everything is supplied:** the types, relations, entity vocabulary and correct program. Nothing here
  shows discovery or interpretation; the interpreter is plan PR 6.
- **No held-out atomic facts:** atomic competence is memorization of a finite world.
- **The models are tiny,** and run on CPU.

**Run:**
`python research/ti_tnn/bench/typed_composition.py --worlds 200 201 202 --inits 0 1 2 3 4`, which also
writes `verdict.json` via `evaluate()`.
