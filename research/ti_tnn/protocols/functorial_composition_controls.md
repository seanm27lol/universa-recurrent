# Functorial composition: which part composes? (frozen brief)

**Status: FROZEN 2026-10-16 (UTC), before any run of these controls.** Branch `ti-tnn-second-review`,
based on `main` at `fd25199`. A 300-step CPU timing check of `functorial` and `linear` ran beforehand;
it scored nothing.

## Why these controls

The [first run](../reports/functorial_composition.md) found the "functorial" model perfect on unseen
two-hop chains (1.000 on 5 seeds), against 0.940 for unconstrained linear maps. An outside review
pointed out three things that change what that result can mean:

1. **It is a shared embedding space.** With per-kind orthogonal Q_k and transports
   T(s → t) = Q_tᵀ Q_s, the logits for any path from kind s to kind t are E_s Q_sᵀ Q_t E_tᵀ = Z_s Z_tᵀ,
   where Z_k = E_k Q_kᵀ. So the model has exactly the capacity of typed embeddings in one shared space
   scored by dot product; only the optimization differs. A test checks this numerically.
2. **The laws do not guarantee correct chains.** A model can satisfy every law exactly, answer every
   one-hop fact correctly, and still get every two-hop chain wrong: nearest-match scoring does not force
   a country to align with its capital. A test builds one.
3. **"Exact beats soft" compared different equations.** The soft penalty targeted only the inverse pair,
   while the exact model is also orthogonal.

Also, the four relations form a tree on the four kinds (country–capital, country–region,
capital–size). On a tree, flatness adds nothing beyond tying the inverse pair. So "functorial" here is
the same function class as "orthogonal maps with country_of = capital_ofᵀ".

## Design (`research/ti_tnn/composition.py`, unchanged training)

The same world, training (single-relation facts only, full batch, Adam, learning rate 3e-3, 3,000
steps, d = 128) and scoring as the first run. Seeds 0–4, on CPU (4 threads; the GPU is busy with the
adaptive-compute queue). The variants:

| Variant | Relation maps | Isolates |
|---|---|---|
| `functorial` | flat-sheaf transports (the first run's model) | replication at the current code |
| `ortho_tied` | an orthogonal map per relation, country_of = capital_ofᵀ | the same function class, parametrized directly |
| `ortho_untied` | an independent orthogonal map per relation | whether tying the inverse pair matters |
| `shared` | identity: typed embeddings in one shared space | whether the orthogonal maps matter at all |
| `soft_full` | unconstrained, softly penalized toward every equation `functorial` obeys (each map orthogonal; country_of the inverse of capital_of; weight 1.0) | a soft version of the same equations |
| `soft` | unconstrained, penalized toward the inverse pair only | the first run's soft model |
| `linear` | unconstrained | the first run's baseline |

The code changed since the first run in one way that matters: every orthogonal map now starts with
determinant +1, and the bridge has no output bias (unused here). So `functorial` starts from different
random draws, and this run replicates it rather than repeating it.

## Predictions (recorded before any run; one-sided; means over 5 seeds of two-hop accuracy)

1. **Replication.** `functorial` ≥ 0.98.
2. **Same function class.** `ortho_tied` ≥ `functorial` − 0.02.
3. **The shared space is enough.** `shared` ≥ 0.95.
4. **The same equations, softly.** `soft_full` ≥ `soft`.
5. **Tying matters.** `ortho_untied` < `functorial`.

**How to read them.**
- If 3 holds, what composes here is typed embeddings in one shared space. The orthogonal, "functorial"
  parametrization is then not what the result rests on.
- If 3 fails while 1 and 2 hold, the orthogonal parametrization helps optimization, since the capacity
  is the same.

Every variant and seed is reported whatever happens.

## Limits

- **Same world.** One relation per pair of kinds, and transports depend only on their endpoints, so two
  different relations between the same kinds cannot be told apart. A world with distinct same-endpoint
  relations, and longer chains that do not cancel, is the next test; it is not run here.
- **The three-hop family** still reduces to round trips (as in the first run).

**Run:** `OMP_NUM_THREADS=4 python research/ti_tnn/composition.py --device cpu --variants functorial
ortho_tied ortho_untied shared soft_full soft linear --out runs/titnn_composition_controls`.
