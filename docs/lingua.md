# Lingua: say what ran, and what can be checked

Lingua is not a machine-generated story about what a model “must have thought.”
It is a typed record of explicit operations, measured dynamics, assumptions, and
property-specific evidence.

## Two record families

| Record | Independent checker currently establishes | Does not establish |
|---|---|---|
| Classical full trace | Every retained projected-gradient transition and final constrained optimum, within tolerance | Remote execution attestation |
| Classical compact trace | Final constrained optimum, within tolerance | Discarded intermediate history |
| Neural trace v1, standalone | Final membership in the boundary carried by the record, final observed residual, and event consistency | Whether that boundary came from an untampered checkpoint |
| Neural trace v1 + checkpoint | The standalone checks plus binding to the exact checkpoint hash, candidate names, boundaries, and model configuration | Learned update replay, correct routing, hidden semantics, optimality |

## Neural example

```text
OBSERVED: router selected balanced_flow with probability 0.83
OBSERVED: 5 recurrent updates were assigned
MEASURED: observed-coordinate RMS fell from 0.48 to 0.06
CHECKED: final state satisfies the selected boundary constraint within tolerance
NOT CLAIMED: hidden feature 17 means “circulation”
```

The record embeds the selected boundary matrix and a fingerprint, final state,
observation mask, route probabilities, event summaries, model configuration, and
checkpoint SHA-256. The hash identifies bytes; it does not prove where or when a
remote execution occurred.

The standalone fingerprint only detects accidental inconsistency inside the record:
someone who replaces both the boundary and its fingerprint can create a different
self-consistent claim. Pass `--checkpoint` to `neural-verify` when the record must be
bound to the local model and candidate library.

## Five claims that must stay separate

```text
constraint satisfaction
        ≠ optimization optimality
        ≠ retained-path correctness
        ≠ physical execution provenance
        ≠ faithful semantic interpretation
```

A project may support one without supporting the others.

## Relation to Applied CMCM

Applied CMCM asks when the record of a computation matters. This repo turns that
question into an engineering comparison:

- what information is retained;
- which properties remain checkable;
- how many bytes and how much time the record costs;
- what historical questions become impossible after compression.

More available witness information did not universally predict more learned
benefit in the earlier experiments. Therefore every new Lingua field needs an
explicit use, cost, and test.
