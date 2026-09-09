# One estimate, one optional structural claim

## Start with a concrete problem

Two flow patterns could explain the measurements. One balances flow at every
junction; the other obeys an alternative constraint. Should the model average
its estimates, or choose one pattern and claim that pattern applies?

**Those are two questions. This release does not force one answer to do both jobs.**

```text
same measurements → existing trained model → candidate estimates + scores
                                                     │
                         ┌───────────────────────────┴──────────────┐
                         ▼                                          ▼
                weighted numerical estimate                optional structural proposal
                returned on every example                   confidence threshold only
                         │                                          │
                check weighted-sum arithmetic               check candidate constraint
```

A forecast and an alert are a familiar analogy: changing an alert threshold
should not silently change the forecast. Here, changing the claim threshold
cannot change the numerical estimate or its computation depth.

## The mathematics

For candidate index `k`, let `z_k` be the candidate estimate and `p_k` its
nonnegative learned weight, with weights summing to one. The returned estimate is

```math
mu = sum_k p_k z_k.
```

Let `j` be the first index attaining the largest weight. A separate proposal
selects `j` when `p_j >= threshold`; otherwise there is no structural proposal.
The proposal carries **`z_j`, not `mu`**, and the candidate boundary matrix `B_j`.
The checker tests `B_j z_j` against a fixed numerical tolerance. It does not prove
that the model chose the right physical rule.

### A counterexample that is in the tests

```text
candidate A: (2,0), constrained to the horizontal axis
candidate B: (0,2), constrained to the vertical axis
weights:     (1/2,1/2)

estimate:    (1,1), which is on NEITHER axis
```

A certificate about A cannot be attached to this average. The record explicitly
states `estimate_has_structural_certificate: false`. That does not assert that
an average can never happen to be feasible; it means this interface makes no
such claim.

For a true signal `y`, elementary expansion gives

```math
sum_k p_k ||z_k-y||^2 = ||mu-y||^2 + sum_k p_k ||z_k-mu||^2.
```

Thus the mixture improves on the weighted average candidate loss, not necessarily
on the loss of whichever individual candidate is best. No universal advantage is
assumed. Learned weights are not automatically Bayesian posterior probabilities.

## Three matched output modes

| Mode | Numerical output | Structural output |
|---|---|---|
| `mixture` | Weighted estimate | Not requested, so coverage is not applicable |
| `mixture_with_claim` | Exactly the same weighted estimate | Accept or abstain under calibrated score cutoff |
| `hard` | Top candidate estimate | Always propose that candidate; legacy comparator |

All modes use the same checkpoint, trained depth, inputs, batching, and candidate
probabilities. They are evaluated AND timed through the same function. The
unstructured model has no invented structural probabilities. The residual-fit
reference has no invented softmax temperature. The Gaussian reference is labeled
privileged because it knows the exact synthetic prior and noise level.

## Symmetric calibration

Every structured model gets its **own numerical cutoff**, selected by the
**same rule on the same calibration observations**. Select the largest inclusive
cutoff that reaches the target empirical coverage, including all cutoff ties.
Labels and reconstruction error do not select the cutoff. This is coverage
calibration, not probability calibration or a guarantee of future coverage.
Calibration and test generator seeds must be distinct and must not match the
stored training or old calibration seeds. Existing checkpoints are not changed.

The primary new comparison intentionally uses fixed computation. A structural
proposal does not halt or replace the estimator. Adaptive stopping needs a
separate quality-versus-cost experiment; a correct proposal is not a certificate
that the numerical estimate has converged.

## Run on the Spark

After `git pull --ff-only` and reinstalling the editable package:

```bash
bash scripts/run_dual_study.sh /absolute/path/to/neural_v2.pt
```

This uses existing weights; no retraining or GPU change is required. It writes
`calibration.json`, `eval.json`, `benchmark.json`, `lingua.json`,
`verification.json`, and a completion marker under a fresh `runs/` directory.
Timing uses the SAME test examples as the reported errors, with warmup,
randomized case order, synchronization, repeated times, median, and quantiles.
Transfers, loading, calibration, metrics, and verification are not in the
inference time. Rollout-history allocation is included. This is not an optimized
minimal-memory kernel or an end-to-end deployment benchmark.

Optional v1 comparison, using identical observations but historically different
weights and training, is available through `compare --v1-checkpoint FILE`. That
is diagnostic, not a controlled experiment on training architecture.

For independent training repetitions:

```bash
python -m universa_recurrent.neural.dual_cli replicate \
  --device cuda --seeds 6100 7100 8100 \
  --output-dir runs/dual-replication-01
```

This trains the existing v2 architectures and controls, then repeats the matched
output study. The summary reports per-seed values, mean, sample standard
deviation, and paired differences across trained-model pairs. It does not count
thousands of test examples as thousands of independent training replicates.

## What the checker establishes

It checks candidate feasibility, mixture arithmetic, first-index tie-breaking,
the recorded threshold rule, and optional binding to trusted local checkpoint
and calibration files. It does **not** replay the neural updates, authenticate
remote execution, prove route correctness, or interpret hidden neurons.

## Known-field foundations

- Model averaging: Hoeting et al., [Bayesian Model Averaging: A Tutorial](https://doi.org/10.1214/ss/1009212519).
  A precedent for retaining uncertainty, not proof that these learned weights are posterior probabilities.
- Classification with rejection: Chow, [On optimum recognition error and reject tradeoff](https://doi.org/10.1109/TIT.1970.1054406).
  Our label-free empirical coverage rule is simpler; it is not a claimed implementation of Chow's optimum.
- GPU measurement: [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html).
  Synchronize when measuring asynchronous device work.

## Limits that remain

The base models have different training objectives and inductive biases even
when parameter counts are similar. Existing training is reused to isolate
output policy, not to prove recurrence causes a gain. This is one tiny synthetic
family. Discovery, transport between structures, semantic interpretation, and a
witness-efficiency advantage remain unestablished. Earlier v2 results stay intact.
