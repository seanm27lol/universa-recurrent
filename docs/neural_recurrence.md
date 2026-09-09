# Neural v1: route once, then recur in one selected space

> **Historical path.** Neural v1 commits to one structure before recurrence. Neural v2 keeps multiple hypotheses alive; see [neural_v2.md](neural_v2.md).


This is the bridge from the transparent baseline to learned latent computation.
It is intentionally small enough that a curious reader can still see every moving part.

## Familiar picture first

Imagine repeatedly adjusting a noisy circulation estimate. The classical demo chooses the
update rule by hand. Neural v0 learns two decisions from examples:

1. **Which known constraint space fits this observation?**
2. **What coordinate correction should the next recurrent step make?**

```text
partial noisy flow
      │
      ▼
encode observation ──────────────┐
      │                          │
      ├── learned structure route│
      │                          │
      ▼                          │
latent coordinates a₀            │
      │                          │
      ▼                          │
shared learned update Fθ ◄───────┘
      │
 a₁ → a₂ → a₃ → ...
      │
      ▼
state z = Q a   (always inside the selected known subspace)
```

The model carries **2 latent cycle coordinates** rather than 5 ambient edge values.
The same update network is reused at every recurrent step.

## What is structural and what is learned?

| Part | Status |
|---|---|
| Bases `Q` defining the candidate spaces | supplied, explicit mathematics |
| `z = Q a` | exact architectural parameterization up to floating point |
| Observation encoder | learned |
| Structure router | learned |
| Recurrent coordinate update | learned and weight-shared |
| Halting probability | learned from training examples |
| Meaning of hidden neurons/features | **not interpreted yet** |

Lingua therefore records route choices, iteration counts, residual summaries, and
structural feasibility. It does **not** rename hidden neural features with invented human meanings.

## Known precedents

This experiment is related to established ideas rather than appearing from nowhere:

- **Projected / constrained optimization:** represent or project iterates so a known constraint is maintained.
- **Algorithm unrolling / LISTA:** Gregor & LeCun (2010) replaced repeated hand-designed sparse-coding updates with trainable repeated layers.
- **Recurrent neural computation:** one learned update can be reused for multiple computational steps instead of assigning different parameters to every depth.
- **Adaptive computation:** learned halting asks whether different inputs should receive different numbers of steps.

These are precedents, not evidence that this particular implementation is faster or better.
That must be measured against the classical direct solve, the classical recurrent baseline,
and fixed-depth neural controls.

## Why the DGX helps

The neural model uses PyTorch. Training batches and repeated updates are vectorized, so a
CUDA device can run them efficiently. The current network is deliberately tiny; the first
point of the DGX run is scientific validation and instrumentation, not maximizing GPU use.

## First claims we are allowed to test

1. Does the learned router generalize to held-out synthetic flows?
2. Does recurrent reconstruction error improve across steps?
3. Does learned halting reduce mean executed steps without materially hurting error?
4. Are decoded states structurally feasible by construction/check?
5. How much of the neural trajectory can Lingua describe without inventing semantics?

A positive answer to any one does not imply the others.
