# Functorial composition controls: geometry-preserving maps compose, the category laws don't matter here — 2026-10-16

**Protocol:** [`protocols/functorial_composition_controls.md`](../protocols/functorial_composition_controls.md),
frozen at `c624e42` before any run.

## The question

The [first run](functorial_composition.md) trained models only on single relations ("capital of
France", "size of Paris"). It then asked them unseen chains ("size of the capital of France").
The "functorial" model got every chain right; unconstrained linear maps got 94%.

An outside review showed that the functorial model is, in capacity, just typed embeddings in one shared
space scored by dot product. Which part of it does the composing? These controls take it apart.

## Answer

**Any relation map that preserves geometry composes perfectly, and so do no maps at all.** The category
laws (inverse tying, flatness) make no difference in this world.

- **No maps at all is enough.** Typed embeddings in one shared space, with identity maps, score 0.9996
  on unseen two-hop chains. That is 2 errors in 2,560 queries over 5 seeds. It also trains about 15
  times faster than the functorial model.
- **Tying the inverse pair doesn't matter.** Orthogonal maps that are not inverses of each other
  (inverse defect 0.376) still score 1.000.
- **A soft penalty works when it targets orthogonality.** Unconstrained maps pulled softly toward being
  orthogonal end up nearly orthogonal (defect 0.001) and score 1.000. The first run's soft penalty
  targeted only the inverse pair. Its maps ended up far from orthogonal (defect 9.61) and scored 0.879.
- **So "exact beats soft" was the wrong lesson.** The soft model lost because its penalty targeted
  the wrong property, not because it was soft.

The first run's linear and soft results reproduce to every digit (0.9398 and 0.8793): their code and
initialization did not change. The functorial model replicates at the current code, which starts its
maps from different random draws: 1.000 again.

## Results (5 seeds, CPU; exact accuracy on every query)

| Variant | Relation maps | **Two-hop** | Size of capital | Region of country | Inverse defect | Orthogonality defect | Seconds per run |
|---|---|---:|---:|---:|---:|---:|---:|
| `functorial` | flat-sheaf transports | **1.000 ± 0.000** | 1.000 | 1.000 | 2e-6 | 2e-6 | 74–76 |
| `ortho_tied` | orthogonal, country_of = capital_ofᵀ | **1.000 ± 0.000** | 1.000 | 1.000 | 2e-6 | 2e-6 | 22–26 |
| `ortho_untied` | independent orthogonal | **1.000 ± 0.000** | 1.000 | 1.000 | 0.376 | 5e-6 | 23–27 |
| `shared` | identity (one shared space) | **0.9996 ± 0.0009** | 1.000 | 0.9992 | 0 | 0 | 4–5 |
| `soft_full` | unconstrained + penalty toward orthogonal and inverse | **1.000 ± 0.000** | 1.000 | 1.000 | 0.001 | 0.001 | 7–8 |
| `soft` | unconstrained + penalty toward inverse only | 0.879 ± 0.021 | 0.873 | 0.885 | 0.005 | 9.61 | 6–7 |
| `linear` | unconstrained | 0.940 ± 0.010 | 0.936 | 0.944 | 9.38 | 10.7 | 5–6 |

- **Fit.** Every variant fits every single-relation fact: one-hop accuracy is 1.000. Three-hop chains
  reduce to round trips in this world (see the first report) and are uninformative: every variant
  scores 0.995 or above.
- **The defects.** Inverse defect is ‖M_country_of M_capital_of − I‖/√d. Orthogonality defect is the
  largest ‖MᵀM − I‖/√d over the four maps, so 0 means every map is a rotation or reflection.

**Results file.** `runs/titnn_composition_controls/20261016T173618/composition.json`, sha256
`76bf11e201d88331bb9e35e97e146e9ba08a13a2923d40c129cbaf90a81a3694`. Run on CPU (4 threads, niced),
PyTorch 2.10.0, while the GPU ran another queue.

## Predictions (frozen before any run)

| # | Prediction | Outcome |
|---|---|---|
| 1 | Replication: `functorial` ≥ 0.98 | **Held** (1.000) |
| 2 | Same function class: `ortho_tied` ≥ `functorial` − 0.02 | **Held** (1.000 against 1.000) |
| 3 | The shared space is enough: `shared` ≥ 0.95 | **Held** (0.9996) |
| 4 | The same equations, softly: `soft_full` ≥ `soft` | **Held** (1.000 against 0.879) |
| 5 | Tying matters: `ortho_untied` < `functorial` | **Failed.** Both are 1.000; tying the inverse pair makes no difference |

## What to read into it

- **The mechanism is geometry, not category theory.** When the relation maps preserve lengths and
  angles (orthogonal), or there are no maps at all, learning each relation separately puts each
  country, its capital and its capital's size near each other in one space. Chains then follow.
  Unconstrained maps can stretch and shear. They fit each single relation, but the distortions
  compound along a chain.
- **The "functorial" name overstated what the first result showed.** The model is functorial, but here
  functoriality is not what makes it work. The simplest model that composes is one shared embedding
  space.
- **This world is too easy to separate the ideas.** It has one relation per pair of entity kinds and
  relations that form a tree. With identity maps, two different relations between the same kinds
  ("capital of" and "largest port of", say) would collapse into one. A world with distinct relations
  between the same kinds, and chains that do not cancel, is the test that could show category
  structure earning its keep. It has not been run.
- **Not efficiency.** These are tiny synthetic models.

Code: [`composition.py`](../composition.py); tests in `tests/test_ti_tnn_composition.py`, including
`test_the_functorial_scores_are_shared_space_dot_products` and
`test_exact_laws_do_not_guarantee_correct_chains`.
