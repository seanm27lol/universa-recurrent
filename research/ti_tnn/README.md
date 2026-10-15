# TI-TNN: typed branches, a calibrated router and a sheaf bridge

**Status: exploratory toy benchmarks. Implemented, tested locally, and run on a GPU
with 3 seeds (2026-10-14).** Nothing here is a result about language models.

## A concrete example

Suppose a model has learned, separately, two facts about each country: a trait A
("is it coastal?") and a trait B ("is it landlocked by mountains?"), both yes/no.
Now ask a question it was never trained on for some countries:
"is exactly one of A and B true?", which is XOR(A, B).

If one part of the model only knows A and another part only knows B, neither part
alone can answer. Knowing A tells you nothing about XOR(A, B) when B is a fair coin
flip. You have to bring the two parts together. That is the job of the sheaf
bridge.

## The idea in plain language

- **Router (Jev).** Reads the question and picks one specialist branch. It also
  outputs a calibrated probability (the Noul gate) that the question needs two
  branches bridged.
- **Typed branches.** Small specialists. Only the chosen branch runs, which saves
  compute compared with running all of them (the dense control).
- **Sheaf bridge.** When the gate fires, it wakes the runner-up branch too and
  combines the two branches' internal states (stalks). The comparison is a learned
  version of a cellular-sheaf coboundary.

## The mathematics

Each country c has traits A(c), B(c) in {0, 1}, independent and balanced, and
S(c) = A(c) XOR B(c). For a fixed A, S equals B or 1 − B, so S is independent of A:
the mutual information I(S; A) = 0, and likewise I(S; B) = 0. A predictor that sees
only A, or only B, is therefore right with probability exactly 1/2 on balanced
held-out countries, whatever it learned elsewhere.

The bridge computes, for stalks s_u (Branch 0) and s_v (Branch 1),

```text
bridge(s_u, s_v) = W_e · tanh(F_u s_u − F_v s_v)
```

F_u and F_v are learned restriction maps onto an edge space. F_u s_u − F_v s_v is
the coboundary of a one-edge cellular sheaf with fixed orientation u → v (Hansen and
Ghrist; Bodnar et al.). XOR is not linearly separable (Minsky and Papert), so the
tanh layer is what lets the bridge compute it.

Routing is trained with the straight-through Gumbel-Softmax (Jang, Gu and Poole;
Maddison, Mnih and Teh). Its probabilities are trained with the Brier score, a
proper scoring rule (Brier 1950). Conditional computation follows sparsely gated
mixtures of experts (Shazeer et al.).

**Assumptions the cap depends on** (each is enforced in the code and tested in
[`tests/test_ti_tnn.py`](../../tests/test_ti_tnn.py)):

1. *Private branch embeddings.* Each branch reads its own embedding subspace. With
   one shared table, A and B meet in one vector and the cap no longer holds; the
   `nobridge_shared_emb` variant shows this.
2. *Gradient masking.* A query's gradient reaches only its designated branch, so
   routing exploration cannot send Trait-B supervision into Branch 0.
3. *The router ignores the entity token*, so routing cannot carry trait information.
4. *Held-out countries' XOR queries never appear in training*, and every split is
   balanced in A, B and XOR.
5. *Two-way scoring.* Combined accuracy compares only the two status tokens, so
   chance is 50%.

## Run it

```bash
python research/ti_tnn/ti_tnn_train.py                    # both benchmarks, 3 seeds, GPU if available
python research/ti_tnn/ti_tnn_train.py --benchmark compositional
python research/ti_tnn/ti_tnn_train.py --device cpu --steps 2000 --seeds 0   # quick CPU check
```

Results print as tables and are saved to `runs/titnn/<timestamp>/routing.json` and
`compositional.json`. On the DGX, use the Phase Two environment
(`.venv-phase2/bin/python`); its PyTorch build supports the GB10.

## The two benchmarks

**Routing.** Queries such as `capital_of <country>` and `region_of <country>`. Branches
are token-wise, so only the router can tell the model which function to apply to the
entity: routing is load-bearing. Variants: routed + bridge, routed without a bridge,
and dense.

**Compositional isolation.** 512 countries, split train / validation / test (320 /
64 / 128), stratified by (A, B). `traitA_of` is supervised only through Branch 0 and
`traitB_of` only through Branch 1 for every country. `combined_status_of` (XOR) is
trained only on train countries and scored on held-out validation and test
countries. Variants:

| Variant | What it tests |
|---|---|
| `routed_bridge` | The proposal: XOR queries wake Branch 0 and Branch 1; only the bridge learns XOR |
| `routed_nobridge` | One branch per query; given its best shot (may train its branch on XOR queries) |
| `dense` | Every branch on every query, with the bridge |
| `nobridge_shared_emb` | One branch, but one shared embedding table: the leak the isolation closes |

Controls on `routed_bridge`: the bridge switched off at evaluation, and the bridge
fed another country's Trait-B stalk.

## Evidence so far

- **Tested locally (software properties, not results):**
  [`tests/test_ti_tnn.py`](../../tests/test_ti_tnn.py) checks that conditional
  execution matches running every branch; that gradient masking keeps Trait-B
  supervision out of Branch 0's parameters and embedding subspace; that strict
  isolation trains only the bridge on XOR queries; that held-out countries never
  appear in XOR training queries and every split is balanced; that the router
  ignores the entity token; that the bridge edge orientation is fixed; and that MAC
  counts follow layer shapes.
**GPU run, 2026-10-14:** default settings (3 seeds, 3,000 steps, batch 512), NVIDIA
GB10 (compute capability 12.1), PyTorch 2.10.0+cu130, Python 3.11.14, aarch64. The
configuration is the one first tried on CPU; nothing was tuned after seeing results.
Results files stay on the GPU host (`runs/` is not committed): `routing.json` sha256
`a739895e30c139b7449d8c980928fef3852f789fa8a9764bb5d31dc67fb89726`,
`compositional.json` sha256
`953d2a66563f971755369a26e0fc92b1de0c49747a207b55b61a44a12f5168af`.

- **Compositional isolation benchmark:**

  | Held-out XOR accuracy (chance 0.50) | Test, 128 countries (mean ± sd) | Validation, 64 | Train countries (seen) | Branch + bridge MACs per XOR query |
  |---|---:|---:|---:|---:|
  | `routed_bridge` | **1.000 ± 0.000** | 1.000 ± 0.000 | 1.000 | 442,368 |
  | `routed_nobridge` | 0.492 ± 0.027 | 0.531 ± 0.031 | 1.000 | 196,608 |
  | `dense` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 | 835,584 |
  | `nobridge_shared_emb` | 0.573 ± 0.036 | 0.625 ± 0.068 | 1.000 | 196,608 |

  Per-seed held-out test accuracy with 95% bootstrap intervals over countries (the
  independent unit): `routed_nobridge` 0.477 [0.390, 0.563], 0.523 [0.445, 0.602],
  0.477 [0.391, 0.563], so every interval includes chance; `nobridge_shared_emb`
  0.594 [0.508, 0.680], 0.594 [0.508, 0.680], 0.531 [0.438, 0.625], so two of three
  exclude chance; `routed_bridge` and `dense` 1.000 [1.000, 1.000] on every seed.

  What this shows. One branch memorizes XOR for train countries (1.000) and is at
  chance on held-out ones, as the information argument predicts. The bridge composes
  the separately learned traits on held-out countries on every seed, with 0.529× the
  dense model's branch-and-bridge compute on those queries (2 branches + bridge
  versus 4 + bridge). Controls on `routed_bridge`, identical on every seed: bridge off
  at evaluation 0.500 (the output then carries no Trait-B information, and the test
  set is balanced in XOR within each value of Trait A); bridge fed the next country's
  Trait-B stalk 0.516, exactly the fraction of those partner pairs that share Trait B
  (checked against the data), so the bridge computes XOR of whatever traits it is
  given; XOR queries
  routed Branch 0 then Branch 1 and fired the gate on 100% of queries; conditional
  execution matched running every branch exactly. Single-trait accuracy was 1.000 for
  every variant, so each branch learned its own trait for held-out countries. The
  shared-embedding leak is real but modest at this training budget.

- **Routing benchmark:** every variant reached 1.000 on held-out queries on every
  seed, so this task separates the variants only on compute: 1.40 branch runs per
  query routed with the bridge, 1.00 without, 4.00 dense. Random routing with the
  bridge off fell to 0.576 ± 0.003 (routed + bridge) and 0.583 ± 0.004 (no bridge),
  so routing is load-bearing. The model trained without the bridge matched the one
  with it: the bridge is not needed on this task, unlike the compositional one. On
  longer queries than training saw (length 10), both routed variants stayed at 1.000
  on every seed while dense averaged 0.995 (lowest 0.986); a small difference.

- **Earlier CPU development runs** (one seed, 1,000 to 2,000 steps) showed the same
  pattern: held-out XOR 1.000 with the bridge, 0.555 without, 0.625 with a shared
  embedding.

**Not established:** anything beyond these synthetic tasks, and whether the bridge
helps when the traits are not cleanly isolated by construction.

## Limits

- Synthetic toy tasks. They test whether the mechanism *can* work under stated
  assumptions, not whether it helps a real model.
- The chance-level cap is a consequence of the isolation assumptions, not of the
  architecture alone. Remove them and a single branch can compose
  (`nobridge_shared_emb`).
- Compute is multiply-accumulates counted from layer shapes, not measured speed.
- The depth/score head is not trained by any loss.

## Primary sources

- Jang, Gu and Poole (2017), *Categorical Reparameterization with Gumbel-Softmax*,
  [arXiv:1611.01144](https://arxiv.org/abs/1611.01144); Maddison, Mnih and Teh (2017),
  *The Concrete Distribution*, [arXiv:1611.00712](https://arxiv.org/abs/1611.00712).
- Bengio, Léonard and Courville (2013), *Estimating or Propagating Gradients Through
  Stochastic Neurons*, [arXiv:1308.3432](https://arxiv.org/abs/1308.3432).
- Shazeer et al. (2017), *Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer*, [arXiv:1701.06538](https://arxiv.org/abs/1701.06538).
- Hansen and Ghrist (2019), *Toward a Spectral Theory of Cellular Sheaves*,
  [arXiv:1808.01513](https://arxiv.org/abs/1808.01513); Bodnar et al. (2022),
  *Neural Sheaf Diffusion*, [arXiv:2202.04579](https://arxiv.org/abs/2202.04579).
- Brier (1950), *Verification of Forecasts Expressed in Terms of Probability*,
  Monthly Weather Review 78(1).
- Lake and Baroni (2018), *Generalization without Systematicity*,
  [arXiv:1711.00350](https://arxiv.org/abs/1711.00350).
- Minsky and Papert (1969), *Perceptrons*, MIT Press.
