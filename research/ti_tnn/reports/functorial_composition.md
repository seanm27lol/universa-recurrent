# Functorial composition: exact laws let learned relations chain perfectly — 2026-10-16

**Protocol:** [`protocols/functorial_composition.md`](../protocols/functorial_composition.md),
frozen at `4b56a03` before any run.

> **Correction (2026-10-16, after an outside review and a
> [controls run](functorial_composition_controls.md)).** The numbers below stand. Three
> interpretations do not:
> - **"The gap comes from the maps obeying composition laws exactly."** Wrong mechanism. The
>   functorial model is, in capacity, typed embeddings in one shared space. Plain identity maps miss
>   1 of 2,560 chains, and orthogonal maps that are *not* inverses miss none. So exact inverse tying
>   and flatness showed no measured advantage. Geometry-preserving maps look like a useful inductive
>   bias here; they are not shown to be the only mechanism.
> - **"Exact beats soft."** That comparison used different equations. A soft penalty toward the exact
>   model's equations (orthogonality of every map plus the inverse pair) scores 1.000. The soft model
>   here penalized only the inverse pair, and its maps ended far from orthogonal.
> - **The laws do not guarantee the 1.000.** A model satisfying every law can fit every one-hop fact
>   and still miss every chain (`test_exact_laws_do_not_guarantee_correct_chains`). The 1.000 is
>   learned.
>
> The text below is the original report.

## The question

Every model was trained only on single relations ("capital of France", "size of
Paris"), with 256 countries and capitals, 8 regions and 6 sizes. Can it answer
chains it never saw, such as "size of the capital of France"?

The step models share one interface:
- typed embeddings (one table per entity kind);
- one map per relation, applied in order;
- tied scoring against the target kind's embeddings.

They differ only in what a relation map may be. A transformer reading the whole
query is the generic baseline.

## Answer

**With exact functorial structure, the answer is yes, perfectly.**
- **Functorial.** Relations are transports of a flat sheaf, orthogonal with an
  exact inverse law. They answered every unseen two-hop chain correctly on all 5
  seeds: 1.000, zero errors in 2,560 chain queries.
- **The unconstrained maps,** using the same interface, made errors:
  - linear 0.940;
  - linear plus a soft inverse penalty 0.879;
  - a small network 0.717.
- **The transformer,** never shown two relations in a row, was at chance (0.166).

The comparison isolates the constraint, not the interface. Every step model
chains maps the same way, so the gap from 0.94 (or 0.72) to 1.00 comes from the
maps obeying composition laws exactly.

**The soft penalty made composition worse,** 0.879 against 0.940 with no
penalty. Only the exact constraint reached zero error. This repeats HOMYMOLY's
finding: an exact structural constraint beat a soft penalty encoding the same
information.

## Results (5 seeds; exact accuracy on every query of each family)

| Variant | One-hop (fit) | **Two-hop** | Three-hop | Round trip | Inverse defect ‖M_cof M_cap − I‖/√d | Training time |
|---|---:|---:|---:|---:|---:|---:|
| `functorial` | 1.000 | **1.000 ± 0.000** | 1.000 | 1.000 | 2e-6 (float32 round-off) | 41–44 s |
| `linear` | 1.000 | 0.940 ± 0.010 | 0.995 | 1.000 | 9.4 | 6–13 s |
| `soft` | 1.000 | 0.879 ± 0.021 | 1.000 | 1.000 | 0.005 | 6–9 s |
| `mlp` | 1.000 | 0.717 ± 0.012 | 0.648 | 1.000 | — | 10–16 s |
| `transformer` | 1.000 | 0.166 ± 0.045 | 0.271 | 0.040 | — | 37–48 s |

**Per path, two-hop.** Both chains behave alike in every variant:

| Variant | size ∘ capital_of | region ∘ country_of |
|---|---:|---:|
| `functorial` | 1.000 | 1.000 |
| `linear` | 0.936 | 0.944 |
| `soft` | 0.873 | 0.886 |
| `mlp` | 0.725 | 0.709 |

**Chance.** Two-hop chance is 1/6 and 1/8 (mean 0.146). Round-trip chance is
1/256.

**Results file.** `runs/titnn_composition/20261016T152240/composition.json`,
sha256 `c26bd12d…`; a byte-identical copy is in
[`experiments/results/ti_tnn/functorial_composition/`](../../../experiments/results/ti_tnn/).
NVIDIA GB10; the GPU gate opened at 19:13 UTC.

## Predictions (frozen before any run)

| # | Prediction | Outcome |
|---|---|---|
| 1 | Every variant fits single relations (≥ 0.95) | **Held** (1.000 for all). |
| 2 | `functorial` ≥ 0.80 on two-hop chains | **Held** (1.000). |
| 3 | `functorial` beats each unconstrained step model on two-hop by ≥ 0.10 | **Failed for `linear`.** The gap was 0.060, though `linear` was below `functorial` on every seed (0.922–0.949). Held for `soft` (+0.121) and `mlp` (+0.283). |
| 4 | `functorial` ≥ 0.70 on three-hop chains | **Held** (1.000). See the design flaw below. |
| 5 | `transformer` ≤ 0.50 on two-hop chains | **Held** (0.166, about chance). |
| 6 | Round trips: `functorial` ≥ `soft` ≥ `linear` | **Held** as ties at 1.000. It is uninformative: every step model got round trips right. |

## What to read into it, and what not

- **The real finding.** On unseen chains, exact functorial structure is the only
  variant with zero errors.
- **Being inverses is not what does it.**
  - The soft penalty made `capital_of` and `country_of` nearly inverse (defect
    0.005), yet `soft` composed worse than `linear`, whose maps are far from
    inverse (defect 9.4).
  - What `functorial` adds is the rest of the structure: every map is an isometry
    into one shared frame, and the connection is flat. So "size of the capital of
    x" is the same map as a direct country → size transport.
  - The unconstrained maps fit each relation by score ranking alone. They get
    round trips right, but their errors compound on genuine two-hop chains.
- **A design flaw in the three-hop family.** Both chosen three-step chains contain
  an inverse pair (`country_of ∘ capital_of`), so they reduce to a round trip plus
  one relation. That is easier than two-hop, which is why `linear` and `soft`
  score about 1.0 there. A genuinely harder three-step chain needs a world with
  more relations.
- **Not efficiency.** The functorial maps cost the same per step as linear ones
  (d² multiply-accumulates), and training was about 5× slower (orthogonal
  parametrization). This experiment tests compositional generalization only.
- **Scope.** One synthetic world, with a single relation per pair of entity kinds;
  the category is given. On real text no category is supplied (see the real-benchmark
  report), and discovering one is the open problem.

Code: [`composition.py`](../composition.py) (uses [`cdl.py`](../cdl.py)); tests in
`tests/test_ti_tnn_composition.py`.
