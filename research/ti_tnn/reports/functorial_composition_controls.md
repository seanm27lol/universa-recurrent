# Functorial composition controls: the exact category laws showed no measured advantage here — 2026-10-16

**Protocol:** [`protocols/functorial_composition_controls.md`](../protocols/functorial_composition_controls.md),
frozen at `c624e42` before any run. **Raw results:**
[`experiments/results/ti_tnn/`](../../../experiments/results/ti_tnn/) (byte-identical copies; every count
below is regenerated from them by `make_tables.py` there).

## The question

The [first run](functorial_composition.md) trained models only on single relations ("capital of
France", "size of Paris"). It then asked them unseen chains ("size of the capital of France"). The
"functorial" model got every chain right; unconstrained linear maps got 94%.

An outside review showed the functorial model is, in capacity, just typed embeddings in one shared space
scored by dot product. These controls take it apart, to see which pieces matter.

## Answer

Five training seeds, on one fixed synthetic world:
- **Perfect two-hop accuracy:** the orthogonal variants (tied or untied) and the soft model penalized
  toward the exact model's equations. Shared embeddings with identity maps missed 1 of 2,560 queries.
- **Short of that:** only the models whose maps were free to stretch and shear. Unconstrained linear
  made 154 errors; the inverse-only soft model made 309.
- **No measured advantage** from tying the inverse pair exactly, or from the flat shared-frame
  parametrization.

The controls support geometry preservation as a useful inductive bias here. They do not establish it
as the only mechanism, nor whether categorical structure helps on other tasks. Geometry preservation
is also not sufficient by itself. A model with identity maps can satisfy every law, fit every one-hop
fact, and still miss every size-of-capital chain (`test_exact_laws_do_not_guarantee_correct_chains`).
The results here are what training found, not a guarantee.

Two earlier readings are withdrawn:
- **"Exact category laws explain the advantage."** Untied orthogonal maps, which are not inverses
  (defect 0.376), did as well.
- **"Exact beats soft."** A soft penalty on the exact model's equations (orthogonality of every map,
  plus the inverse pair) scored perfectly. The first run's soft penalty targeted only the inverse pair,
  and its maps ended far from orthogonal (defect 9.61).

## Results (5 seeds; every two-hop query, 512 per seed)

| Variant | Relation maps | **Two-hop correct** | Size of capital | Region of country | Inverse defect | Orthogonality defect | Seconds per run (CPU) |
|---|---|---:|---:|---:|---:|---:|---:|
| `functorial` | flat-sheaf transports | **2,560 / 2,560** | 1,280 / 1,280 | 1,280 / 1,280 | 2e-6 | 2e-6 | 74–76 |
| `ortho_tied` | orthogonal, country_of = capital_ofᵀ | **2,560 / 2,560** | 1,280 / 1,280 | 1,280 / 1,280 | 2e-6 | 2e-6 | 22–26 |
| `ortho_untied` | independent orthogonal | **2,560 / 2,560** | 1,280 / 1,280 | 1,280 / 1,280 | 0.376 | 5e-6 | 23–27 |
| `shared` | identity (one shared space) | **2,559 / 2,560** | 1,280 / 1,280 | 1,279 / 1,280 | 0 | 0 | 4–5 |
| `soft_full` | unconstrained; soft penalty toward orthogonality of every map plus the inverse pair | **2,560 / 2,560** | 1,280 / 1,280 | 1,280 / 1,280 | 0.001 | 0.001 | 7–8 |
| `soft` | unconstrained; soft penalty toward the inverse pair only | 2,251 / 2,560 | 1,118 / 1,280 | 1,133 / 1,280 | 0.005 | 9.61 | 6–7 |
| `linear` | unconstrained | 2,406 / 2,560 | 1,198 / 1,280 | 1,208 / 1,280 | 9.38 | 10.7 | 5–6 |

- **Means.** As mean ± sd of per-seed accuracy: `shared` 0.9996 ± 0.0009, `soft` 0.879 ± 0.021, `linear`
  0.940 ± 0.010, and the rest 1.000.
- **Fit.** Every variant fits every single-relation fact: one-hop accuracy is 1.000. Three-hop chains
  reduce to round trips in this world (see the first report). They are uninformative: every variant
  scores 0.995 or above.
- **The defects.** Inverse defect is ‖M_country_of M_capital_of − I‖/√d. Orthogonality defect is the
  largest ‖MᵀM − I‖/√d over the four maps, so 0 means every map is a rotation or reflection.
- **Against the first run.** `linear` and `soft` keep the same totals: 2,406 and 2,251 correct of 2,560.
  `soft`'s per-path split differs by one query (1,117 / 1,134 then, 1,118 / 1,133 now). The first run
  was on the GPU and this one on the CPU, so the arithmetic differs slightly. `functorial` starts from
  different random draws at the current code (every orthogonal map now starts with determinant +1), and
  again makes no errors.

**Results file.** `composition_controls/20261016T173618/composition.json`, sha256
`76bf11e201d88331bb9e35e97e146e9ba08a13a2923d40c129cbaf90a81a3694`. CPU (4 threads, niced), PyTorch
2.10.0, while the GPU ran another queue.

## Predictions (frozen before any run)

| # | Prediction | Outcome |
|---|---|---|
| 1 | Replication: `functorial` ≥ 0.98 | **Held** (1.000) |
| 2 | Same function class: `ortho_tied` ≥ `functorial` − 0.02 | **Held** (1.000 against 1.000) |
| 3 | The shared space is enough: `shared` ≥ 0.95 | **Held** (0.9996) |
| 4 | The same equations, softly: `soft_full` ≥ `soft` | **Held** (1.000 against 0.879) |
| 5 | Tying matters: `ortho_untied` < `functorial` | **Failed.** Both 1.000; exact inverse tying made no measured difference |

## What to read into it

- **Geometry preservation as an inductive bias.** With maps that preserve lengths and angles, or no
  maps at all, training on each relation separately tended to put a country, its capital and its
  capital's size near one another in one space, and chains followed. Unconstrained maps fit each single
  relation but tended to compound distortions along a chain. That is a pattern on these runs, not a
  mechanism shown to be necessary or sufficient.
- **The "functorial" name overstated what the first result showed.** The model is functorial, but the
  measured result does not depend on it here. The simplest variant that matched it is one shared
  embedding space.
- **Five seeds on one world.** The facts come from one draw (`facts_seed` 1234). Five independently
  generated worlds would test something different.
- **This world cannot separate the ideas.** It has one relation per pair of entity kinds, and its
  relations form a tree. With identity maps, two different relations between the same kinds ("capital
  of" and "largest port of", say) would collapse into one. A world with distinct relations between the
  same kinds, and chains that do not cancel, could show category structure earning its keep. It has not
  been run.
- **Not efficiency.** These are tiny synthetic models.

Code: [`composition.py`](../composition.py); tests in `tests/test_ti_tnn_composition.py`, including
`test_the_functorial_scores_are_shared_space_dot_products` and
`test_exact_laws_do_not_guarantee_correct_chains`.
