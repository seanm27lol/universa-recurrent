# When two valid edits interact

**Status: opt-in experiment, not a new model or an established result.**
Package version remains 0.5.2. Weights, calibration ranges, claim thresholds,
and the existing parser/codec are unchanged.

## A familiar example

In a spreadsheet, changing price and changing quantity each affects revenue.
Changing both need not equal the sum of those separate changes: price times
quantity contains an interaction. Likewise, two valid mathematical state edits
need not have additive effects on a nonlinear model's final answer.

The previous pilot tested whether a named edit after decoding closely reproduces
the corresponding direct numerical edit. This test asks a different question:
can two *measured individual responses* account for their joint response?

```text
Same paused state, same original context
       |
       +-- no edit --------> y0
       +-- A only ---------> yA
       +-- B only ---------> yB
       +-- A then B -------> yAB (held out of the additive prediction)

Prediction: y0 + (yA - y0) + (yB - y0)
Check:      actual yAB minus that prediction
```

Both commands are applied at ONE cut. There is no recurrence between A and B.
The order is specified; this is not a test of edits at different times.

## Mathematics

Let `F` denote either the immediate mixture readout at the pause point, or the
original remaining updates followed by that readout. The two horizons are kept
separate. Let `h` be the candidate coordinates plus current logits, and let `c`
be the retained observations, mask, encoder context, and prior logits.

Write `y0 = F(h,c)`, `yA = F(A(h),c)`, `yB = F(B(h),c)`, and
`yAB = F(B(A(h)),c)`. The additive prediction is `yA + yB - y0`; the interaction
residual is `yAB - yA - yB + y0`. Every `y` is a vector of numerical estimates.
An exactly affine response has zero residual in exact arithmetic. A nonlinear
mixture readout may already create residuals before any further update.

We report absolute interaction error, a normalized root-mean-square error when
the joint effect is not nearly zero, and no-effect/A-only/B-only comparisons.
Near cancellation, the relative statistic is undefined and stored as null;
absolute errors remain visible. The fixed joint-effect MSE floor is 1e-12.

**This predictor costs a baseline and TWO single-edit model evaluations on each
input.** It is not a learned model that predicts an unseen input's response for
free, and it is not a speed improvement. It never receives the actual joint
response as an argument. This is a controlled interaction diagnostic.

## Protocol

Reuse the five checkpoint/calibration pairs and the sixteen-bit manifests from
the original continuation study. Use 256 common new matched-distribution inputs
at seed 85000. Reject reuse of known prior seeds and checkpoint-reserved splits.
No fitting or tuning uses these inputs. Multiple fits, cuts and edits do not
multiply the number of independent input problems.

At midpoint and penultimate cuts of the shared-eight and dedicated-four models,
use seven fixed pair families: two within-candidate cycle pairs, an across-candidate
cycle pair, two same-candidate cycle/evidence pairs, a common-evidence shift,
and an inverse-cycle pair. Repeat at 1/4, 1/2, and full size of the previous
calibration-based edit magnitude. The inverse and common-evidence pairs are
cancellation controls, not evidence of semantic understanding.

Each path uses both immediate readout and the original remaining-update horizon.
Raw numerical edits are the reference; parsed raw edits must produce the same
state bytes first. The decoded path uses the frozen overflow-safe codec followed
by the same commands, without requantization. Baseline-corrected joint-effect
error measures codec distortion separately from nonadditivity.

Save original/decoded/joint states and all baseline, A, B, and joint estimates,
probabilities and claim indices in numeric NPZ files. Save ranges, commands,
byte costs, input identities and raw-restoration checks in JSON. Reference files
are hash-checked before and after the run. Raw restoration and baseline failures
stop the experiment. Nonfinite edited continuations remain explicit failed path
rows with no paired codec statistic; they are not silently excluded.

## What this can and cannot establish

| Observation | Interpretation |
|---|---|
| Small codec joint-effect error | Approximate preservation of specified numerical interventions |
| Small interaction residual for a tested pair/scale | Local additive response approximation works there |
| Large interaction residual | Individual effects are insufficient; not a broken parser |
| Difference between the two horizons | Different response maps behave differently, not an isolated causal attribution to recurrence |
| Passing all implementation tests | Implemented comparisons function on tested fixtures, not a DGX result |

The descriptions still name fields whose meanings were designed into the model.
Nothing here discovers hidden-neuron concepts, establishes a general reasoning
language, certifies correct routes, or proves causal abstraction. The common
context is retained and counted; no whole-model compression claim is made.

## Run

```bash
bash scripts/run_edit_composition_study.sh \
  /absolute/path/to/replication/results \
  /absolute/path/to/original-state-continuation/results cuda
```

The helper writes a new results folder and a reports-only ZIP in Downloads. It
never modifies weights or old results. The constituent numeric arrays, not model
weights, allow an independent arithmetic audit.

## Established grounding

This is the finite-difference analogue of checking two-factor interactions, not a
new interaction theorem. See NIST's [main and interaction effects](https://itl.nist.gov/div898/handbook/pri/section6/pri615.htm).
Floating-point cancellation and identical mathematical expressions need numerical
rather than universal bitwise assumptions; see PyTorch's [numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
The actual raw/parser byte gate is stronger where the implementations perform
matching float32 additions; it is tested rather than assumed for downstream work.
