# Finish this phase with one locked validation

**Status: implementation, not a new result. No automatic next experiment.**
Version 0.5.2, model weights, named fields, codec ranges and claim thresholds stay
unchanged. The existing local-response evaluator is called without modification.

## The concrete question

A spreadsheet approximation may help estimate a price change without predicting
it accurately enough for a precise invoice. Those are two different standards.
Here the local-response pilot found that a full quadratic rule helped small edits,
but extrapolating it to large edits could be worse than predicting no change.
A more complicated formula is not automatically a more faithful description.

We finish this phase by testing the simpler linear rule on new observations.
The quadratic rules remain comparisons; we do not invent a cubic model or shrink
the test after seeing a bad result.

## Frozen decision, before new answers

The complete settings are in `experiments/phase_closeout_v1.json`. This is an
explicitly post-pilot selection followed by a fresh holdout, not a claim that the
whole research sequence was preregistered.

| Item | Fixed choice |
|---|---|
| Primary model | Dedicated four-step model, paused after step 2 |
| Response horizon | Both original remaining updates |
| Primary rule | Linear, probe radius 1/8 |
| Evaluation edits | All nine noncontrol families, at sizes 1/4, 1/2, 1 |
| Fresh input blocks | Seeds 91000 and 92000, 256 problems each |
| Fits | The same five frozen trained checkpoints |
| Limited-utility criterion | Relative RMS prediction error at most 0.8 |
| Close-prediction criterion | Relative RMS prediction error at most 0.1 |
| Stopping decision | Document the result and close this phase either way |

The two error limits are declared practical choices for this final test, not
universal interpretability standards. A 0.8 ratio means 20% less RMS error than
predicting no effect. It does **not** mean 80% accuracy or faithful understanding.
The stricter 0.1 ratio is reported separately; even satisfying it would not prove
semantic meaning or guarantee individual predictions.

Equal weight is given to every fixed fit, input, noncontrol query and output
component. The pooled statistic is `sqrt(mean squared prediction error / mean
squared actual effect)`. It is not the average of unstable per-example ratios.
Zero and common-evidence controls stay separate from the success denominator.
Both block point estimates and a one-sided 95% bootstrap upper estimate must meet
a threshold to mark that criterion as supported. Inputs are resampled within each
seed block, after averaging the five fits and all fixed queries for each input.
This avoids counting edits, model fits or output components as independent new
problems. The percentile bootstrap is approximate and conditional on these fits
and queries; it does not produce a uniform bound or uncertainty over all models.

All other models, cuts, radii, sizes and approximations are secondary, descriptive
comparisons. They cannot replace the primary after the new output is inspected.
A nonfinite fit/response is retained as a failure and prevents a positive primary
conclusion rather than being omitted from a favorable average.

## Exact costs and scope

The common evaluator still constructs all four approximations from 73 probes per
input, cut, horizon and radius. A new local rule is constructed for EACH input;
that is not retraining neural weights, but it is real computation. It also evaluates
33 targets for scoring. The saved 1,720-byte full rule is not compressed relative
to the 24-byte raw state. A stand-alone linear base/J would have 280 bytes, but
this validation neither implements nor times an optimized 13-probe linear path.
No new inference-speed or total-memory improvement is claimed.

One new run command starts both held-out blocks. Existing raw reports, checkpoint
bytes and codec manifests are never overwritten. Script hashes and reference
file hashes are checked before and after execution. The locked plan is written
before target evaluations. Model predictions are fixed before target responses
inside the unchanged evaluator. The reference already includes original input
context with the solver; this is not an isolated, context-free interpreter.

The portable ZIP contains every worker JSON, input-paired averaged loss arrays,
input values, source/reference hashes, costs, gates and the final decision. It does
not copy checkpoints. The full coefficients/probes/predictions remain on the DGX,
with their NPZ hashes in the report; they are not all repeated in the small upload.
A loss audit is not independent GPU replay or proof of execution authenticity.

## Run and finish

```bash
bash scripts/run_phase_closeout.sh REPLICATION_RESULTS ORIGINAL_CONTINUATION_RESULTS cuda
```

After that report, the deliverable is a demonstration and a technical write-up of
what held up and what did not. No additional mathematical-model family is queued.
A genuine software defect can require repeating an affected comparison; a negative
scientific result does not.

This uses the established local finite-difference approach already explained in
[local response](local_response.md). The earlier preservation and typed-edit
experiments answer different questions; a failure to predict long-range responses
does not erase their narrower engineering results.
