# Dual-output comparison: DRAFT, exploratory, outcome-informed

The prior v2 run motivated this interface. This is not retroactive preregistration.

## Questions

1. Does separating the weighted estimate from a structural proposal preserve the
   mixture's reconstruction quality while allowing useful selective proposals?
2. Which architecture provides the best measured quality-time point when all
   structured models use equivalent output and coverage-calibration rules?
3. Do differences reproduce across independently trained seeds?

## Fixed comparisons

Shared recurrent, direct, untied-depth, dedicated depth 1/2/4, ambient recurrent,
transparent residual-fit, and generator-aware Gaussian references. Compare
mixture, mixture-with-claim, and always-hard output for probabilistic structured
models. Do not assign probabilities or selective claims to controls lacking them.
Do not silently omit missing direct/untied/ambient checkpoint controls.

## Data and policy

Use the checkpoint's noise and mask settings. Default new score-calibration seed:
21000. Default new test seed: 22000. Both differ from all recorded training and old
calibration seeds. Every model sees the same examples on each split. Default
coverage target: 0.75. Select the largest score cutoff achieving that calibration
coverage, accepting all ties. No label or reconstruction loss tunes this cutoff.
The empirical calibration constraint is not an out-of-sample risk guarantee.
Do not retune after observing final test performance.

## Outputs and measurements

A claim about candidate `z_j` cannot be a certificate about mixture `mu`.
Report estimate MSE, claimed fraction, counts, error per claimed example AND per
all examples, and both estimate and candidate MSE on accepted examples. Pure
estimation has no claim-coverage denominator rather than a fictitious 0% coverage.

Quality and timing call the same implementation on the same test cohort and
batch shapes. Warm each case; interleave timed cases under a recorded RNG seed;
synchronize device work; retain all durations and median/p10/p90. Include output
construction. Exclude and name loading, transfers, calibration, metrics, logging,
and verification. No derived FLOPs or speedup from logical iteration counts.

## Repetition

The provided replication command uses fresh training seeds 6100/7100/8100 and
rejects overlapping generator schedules. Models within each replicate share
training data and minibatch order under the existing v2 trainer. They retain
that trainer's different objectives, so the architecture comparison is not a
complete causal ablation. Summarize training-level paired differences; do not
substitute pooled test-example significance for training replication.

## Required negative tests

Changing proposal cutoff must not change estimate or depth. Test ties, zero/full
coverage, mixed states outside either subspace, altered records, nonfinite inputs,
overflow, checkpoint/calibration mismatch, seed reuse, overwrite refusal, exact
timed-versus-evaluated outputs, and no neural replay by the checker.
