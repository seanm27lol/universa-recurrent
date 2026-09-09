# Neural v2 protocol — DRAFT / NOT SEALED

This design is outcome-informed by the neural v1 DGX run. It is exploratory until the code commit, seed blocks, thresholds, hardware regime, and decision rules are frozen before a confirmatory test block is generated.

## Question

Does keeping multiple structural hypotheses alive, revising their evidence during recurrence, and allowing calibrated abstention improve the quality–compute–coverage tradeoff over early hard routing and the declared controls?

## Architecture

- two explicit equal-dimensional graph-flow subspaces;
- one latent coordinate state per candidate;
- one update network shared across candidates and recurrent depth;
- candidate evidence revised after every step;
- calibrated `continue / commit / abstain` policy;
- committed output is one candidate state;
- abstained output is a probability-weighted provisional mixture.

## Split discipline

- training data: `seed + 1`;
- calibration data: `seed + 2`;
- shuffle RNG: `seed + 3`;
- test seed: separately supplied and absent from the checkpoint;
- policy thresholds chosen only on calibration data.

## Required comparisons

1. calibrated v2 dense and compact execution;
2. max-depth v2 and shared-model truncations;
3. models trained specifically for depths 1, 2, and 4;
4. approximately parameter-matched direct model;
5. approximately parameter-matched untied-depth model;
6. approximately parameter-matched ambient recurrent model;
7. transparent fit-each-structure baseline;
8. privileged Gaussian hard and posterior-mixture references;
9. v1 historical result, labeled outcome-informed and not pooled.

## Metrics

- output MSE;
- top-route and selective-route accuracy;
- coverage and wrong-commit rate;
- committed and abstained MSE;
- mean decision step and route revisions;
- candidate update examples;
- wall-clock time at declared hardware, batch size, warmup, and repetitions;
- Lingua verification success and record size.

## No automatic success

Fewer logical steps are not a speedup. A mixture with lower MSE is not a correct single-structure explanation. A feasible candidate state does not prove the candidate is the true structure. Parameter matching does not equalize sequential compute.

## Confirmatory gate

Before a confirmatory run, freeze:

- one calibration objective and its costs;
- one minimum coverage rule;
- one test seed block verified absent from development;
- primary and secondary comparisons;
- multiplicity handling and failure conditions;
- exact DGX/PyTorch/batch benchmark configuration.
