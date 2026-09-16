# A local rule with interaction terms

**Status: implemented, regression-tested locally on CPU; trained DGX results unmeasured.**
Models, thresholds, codec ranges and package version 0.5.2 are unchanged.

## Example first

Revenue is price times quantity. Changing both creates a cross term that cannot
be explained by adding the two changes measured separately. A local mathematical
worksheet can include that cross term as well as each variable's individual effect.
This experiment asks whether such a worksheet helps predict this solver's outputs.

The supplied composition run had 256 underlying inputs. At the largest tested
edit size, the pooled RMS error of the additive prediction relative to the joint
effect was about 48% and 61% for the two within-candidate cycle families after
continuation, compared with approximately 0.000018% at the immediate readout.
The correctly decoded continuation had only four claim/abstention differences
versus the raw joint edit in 76,800 repeated targeted-edit comparisons. These are
source-data calculations, not universal bounds, independent trials, or neural
execution replay. Codec fidelity and nonlinear interaction are different issues.

## What changes now

Instead of measuring each requested single-edit response, first probe a small
neighborhood of the paused state. Build a local rule, then predict edits that were
not in those probes. All rules and predictions are fixed before the target
responses are evaluated. The predictor receives numbers, not a neural callback.

```text
Current state + frozen original context
              |
      73 small model probes
              |
    Local mathematical coefficients
              |
   Unqueried signed / joint edits
              |
     Predicted numerical response
              |
 Compare with actual frozen-model response
```

This first prediction test deliberately uses RAW state fields, not a new codec.
It separates whether we can predict a response from the already-measured codec
error. The field names are defined by the model, not discovered neuron meanings.
There is no natural-language encoder, learned English decoder or neural training.

## The mathematics

Let `h` contain the four cycle coordinates and two current evidence logits, and
`c` contain the unchanged context. Let `F(h,c)` return the five-component estimate
at the cut or after all original remaining updates. Each field has a frozen unit
`d_i`: one eighth of its old calibration range for cycle coordinates, and `log(2)`
for each evidence logit. Write `G(u)=F(h + d*u,c)` for a dimensionless edit `u`.

The local predictions are:

- No effect: `G(0)`.
- Linear: `G(0) + sum_i J_i u_i`.
- Diagonal quadratic: add `0.5 * sum_i H_ii u_i^2`.
- Full quadratic: additionally include `sum_(i<j) H_ij u_i u_j`.

The cross terms distinguish full from diagonal curvature. The matrices here are
finite-difference approximations, not certified derivatives. For radius `r`,
`J_i = [G(r e_i)-G(-r e_i)]/(2r)` and
`H_ii = [G(r e_i)-2G(0)+G(-r e_i)]/r^2`.
For `i != j`, use the four signed pair probes divided by `4r^2`.
`e_i` means a vector with a one in field i and zeros elsewhere.

The probes are float32 neural computations. Coefficients and predictions use
float64 arithmetic. The predictor uses the actual representable target change,
not an assumed infinitely precise addition. Float32 rounding and nonsmooth local
behavior can still harm the approximation. There is no uniform error bound.

This is established finite-difference local approximation, not a new theorem.
For numerical differentiation and its step-size/roundoff limitations, see the
[official SciPy finite-difference documentation](https://docs.scipy.org/doc/scipy/reference/differentiate.html)
and [Hessian documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.differentiate.hessian.html).
This implementation uses NumPy directly and does not add SciPy as a dependency.

## Fixed protocol and costs

Use the same five checkpoints and the original continuation manifests. Test 256
new common matched-distribution problems, seed 87000, at midpoint and penultimate
cuts of the shared-eight and dedicated-four models. Other documented data seeds
and checkpoint training/calibration seeds are excluded. No range or policy is fitted.

Build each rule with radius 1/8 (primary) and 1/16 (sensitivity diagnostic). Report
both; do not select a radius after inspecting target answers. Each construction
uses `1 + 2*6 + 4*15 = 73` batched evaluations: one baseline, twelve axial probes,
and sixty mixed probes. There are 33 fixed targets: five two-field families, four
prespecified dense signed directions, zero and common-evidence controls, each at
sizes 1/4, 1/2 and 1. Nonzero targets are outside the stencil; this tests local
extrapolation and may fail. The two controls are not semantic evidence.

Each input needs its own rule. This is NOT yet a global surrogate, a free
explanation, a trained predictor, or a speedup. Each response rule stores 1,720
bytes per input (base/J/full symmetric matrix storage), versus 24 raw dynamic-state
bytes, plus original context and metadata. Redundant symmetric entries are counted.
One unoptimized construction time is descriptive, not a latency benchmark.

Immediate readout is separate from remaining-update continuation. The returned
numerical formula does not forecast or certify discrete structural claims.
Do not pool 5 fits, cuts, radii, targets or horizons as independent input problems.

## Gates, failures and evidence

Raw restoration must match the original rollout and its discrete decisions.
Reference and source files are identified and checked before/after each worker.
No final joint result enters coefficient construction or prediction. Zero energy
makes relative error undefined; its absolute error remains reported. Nonfinite
probe fits and target responses remain explicit failures, not favorable averages.
Reports save raw probe values, coefficients, target states/outputs and predictions
in numeric NPZ files, with JSON metadata and hashes. No weight files are copied.
Neither reported hashes nor raw-restoration gates authenticate remote execution.

Local tests cover exact quadratic/cross-term recovery, stencil counts, invalid
inputs, no hidden prediction callback, cancellation, target/probe separation,
nonfinite fit reporting, the actual frozen-checkpoint worker and subprocess CLI.
Full current-repository CI must pass before merge. No local full-suite claim is made.

## Run

```bash
bash scripts/run_local_response_study.sh REPLICATION_RESULTS ORIGINAL_CONTINUATION_RESULTS cuda
```

A poor quadratic fit is a research result. It may say the edit is too large,
the function is poorly approximated locally, or float32 probes are inadequate.
The next decision should follow measured predictive error and explicit cost,
not the mere fact that a mathematical formula can be written.
