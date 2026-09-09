# Learned recurrence: the smallest honest version

## Familiar picture

Imagine correcting a noisy circulation estimate several times. The model must
decide which conservation rule applies, then improve coordinates inside that
space.

```text
observed edges + mask
        │
        ▼
       encoder ────────────────┐
        │                      │
        ├─ structure router    │
        ▼                      │
coordinates a₀                │
        │                      │
        ▼                      │
shared update Fθ ◄─────────────┘
        │
 a₁ → a₂ → a₃ → ...
        │
        ▼
state z = Q a
```

`Q` is an explicit basis for one candidate valid subspace. Carrying `a` means
every decoded state remains in that subspace by construction, up to floating
point.

## One update

For selected basis `Q`, current coordinates `a_k`, observed values `x`, and mask
`m`:

\[
z_k = Q a_k,
\qquad
r_k = m \odot (z_k-x),
\qquad
\bar r_k = Q^\top r_k.
\]

The shared network receives the encoded observation, `a_k`, the reduced residual
`\bar r_k`, and the observed-coordinate RMS. It predicts a correction:

\[
a_{k+1}=a_k+F_\theta(c,a_k,\bar r_k,\|r_k\|_{\text{observed}}).
\]

The residual RMS divides by the number of coordinates actually observed. v0.2
checkpoints used the ambient dimension instead; the loader preserves that legacy
behavior rather than silently changing an old model.

## What the halting head learns

Neural v1 unrolls all candidate trajectories during training. For each step it
asks:

> Would any later available step improve this candidate's synthetic truth MSE by
> more than the declared tolerance?

If no, that step receives a ready-to-stop target. This **future-regret teacher**
can inspect synthetic truth and future unrolled states during training. The
inference model sees neither.

This teacher is useful for a controlled experiment, but it is not yet a method
for unlabeled real-world deployment.

## What “adaptive” means here

| Quantity | Definition |
|---|---|
| Logical mean steps | Average updates assigned to each sample |
| Update examples | Number of sample-update evaluations actually executed |
| Recurrent rounds | Sequential loop depth reached by a batch |
| Wall-clock time | Hardware measurement of the full inference call |

A large batch may contain one difficult example that keeps all eight rounds alive.
The compact path can skip halted samples, but cannot remove the sequential rounds
needed by the remaining examples.

Generated experiment data are validated before inference. The model also offers
`validate_values=True` for untrusted external tensors, but that option reads device
values and therefore synchronizes CUDA; it is intentionally outside the timed hot
path.

Checkpoint loading requires PyTorch 2.10 or newer and uses the restricted
weights-only loader. This narrows the deserialization surface; it is not a reason
to load checkpoints from an unknown source.

## References and controls

The architecture is related to:

- projected and equality-constrained optimization;
- algorithm unrolling and LISTA;
- recurrent weight sharing;
- adaptive computation and early exits;
- mixture/routing systems.

These are precedents, not evidence of success. The evaluator includes:

- fixed depths 1, 2, 4, and 8;
- a non-learned fit-each-structure router;
- a privileged Gaussian reference using the exact toy generator (the posterior
  mixture is squared-error optimal under those assumptions; the hard MAP route
  is only a diagnostic);
- separate error for correct and incorrect routes;
- a wall-clock benchmark for dense and compact execution.

## Why routing cannot be treated as a trivial 100% target

The two toy subspaces are deliberately not disjoint: they share one direction,
and the model sees only a noisy random subset of ambient coordinates. Some
examples are therefore weakly informative about which label generated them.
The evaluator reports a privileged generator-aware posterior reference so a
learned router is judged against the information available in this toy problem,
not against an assumed perfect ceiling.

This is still a synthetic design choice, not evidence that real structural
routing has the same ambiguity.

## Current boundary

The router acts once before recurrence. The model does **not yet** revise its
structure, transport a state between spaces, or invoke Universa discovery. It
also still needs matched ablations against an untied-depth network and a direct
feed-forward estimator before recurrence itself earns causal credit. Those are
the next architectural questions.
