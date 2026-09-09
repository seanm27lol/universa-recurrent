# Neural v1 experiment protocol — DRAFT / NOT SEALED

This design is outcome-informed by neural v0 and remains exploratory. Freeze a
new protocol, code commit, seed blocks, thresholds, and decision rules before any
confirmatory run.

## Question

Can a structure-aware recurrent model achieve a better measured quality–compute
point than every tested fixed depth while retaining a checkable structural
record?

## Current ingredients

- the same two explicit graph-flow subspaces;
- standard-normal latent coordinates and uniform structure labels;
- 70% observation probability, minimum two observed coordinates;
- Gaussian noise std 0.05;
- learned encoder and one-time hard router;
- two latent coordinates in the selected subspace;
- one update MLP shared across a maximum of eight steps;
- residual RMS normalized by observed coordinates;
- all-candidate halt supervision;
- a synthetic future-regret target: stop when later available steps improve
  truth MSE by no more than `1e-4`;
- dense and active-sample-compacted inference implementations.

## Required comparisons

1. fixed depths 1, 2, 4, and 8;
2. non-learned fit-each-structure residual router, including model-device wall time;
3. generator-aware Gaussian Bayes reference, labeled privileged;
4. adaptive dense and adaptive compact execution;
5. route-correct and route-wrong reconstruction error;
6. model-only GPU wall-clock time with declared batch size and inputs already on
   device;
7. independent neural Lingua structural check;
8. evaluation-only true-route runs to separate update quality from routing error;
9. an ambient/unstructured recurrent control before any claim that the
   structural parameterization itself helps;
10. tied recurrent weights versus a parameter-matched untied-depth control;
11. a direct feed-forward estimator, to test whether repeated state updates add
    anything beyond static observation encoding.

## No success from step counts alone

An adaptive result is interesting only when:

- its error is within a frozen quality rule relative to the strongest fixed
  depth; and
- its measured wall-clock cost is lower under the frozen hardware/batch regime.

Logical steps and active update examples are diagnostics, not substitutes.

## Scope

This experiment does not test route revision, transport between structures,
Universa discovery, semantic interpretation of hidden features, real-world data,
or general model reasoning.
