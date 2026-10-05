# Phase Six readout audit: readout choice changes scores; tracking thresholds remain unmet — 2026-10-05

Suppose an activation contains the integer itself: one coordinate is exactly
`x = 7`. Can the probe read 7? On a balanced fixture containing integers
0–19, the frozen categorical ridge probe gets only 10% correct despite that
perfect encoding. A scalar ridge readout recovers every value with moderate
regularization. A poor probe score can therefore reflect the instrument as
well as the representation.

This CPU audit applied both readouts, plus an affine categorical variant,
to the **existing Phase Six activations**. Scalar readout improves some
arithmetic scores, but none of the three methods meets both-variable
tracking thresholds. This is an exploratory sensitivity analysis on reused
pilot data. Phase Six's failed U6 and void G6/E6 remain unchanged.

## What changed in the readout

The frozen probe fits 20 one-hot class targets from centered activations,
then chooses the largest predicted class score. It does not add an
intercept. The two audit alternatives are:

- **Affine one-hot:** add the training class proportions to the predicted
  scores, equivalent to fitting with an unpenalized intercept.
- **Affine scalar:** regress the integer value, with a training-derived
  intercept; round to the nearest integer with ties to even, then clip to
  0–19. This uses the numeric ordering of the labels.

For training features \(X\), let \(X_c=X-\bar X\). For centered targets
\(Y_c\), the affine fit is

\[
W=(X_c^T X_c+\lambda I)^{-1}X_c^T Y_c,\qquad
\hat Y=(X_{\rm eval}-\bar X)W+\bar Y.
\]

Here \(\bar X\) and \(\bar Y\) use training rows only, and
\(\lambda=\alpha\,\operatorname{mean}_i\|X_{c,i}\|^2\).
The implementation uses the equivalent dual solve. Alpha comes from the
same six-value grid, selected by selection-split accuracy, with ties going
to the larger alpha. R1 and R2 layers maximize the lower x/y selection
accuracy for carried and arithmetic values respectively; ties go to the
earlier layer. Pilot scores do not enter these choices.

Code: [state_probe_audit.py](../src/open_weight_lingua/state_probe_audit.py).
The original probe modules and recorded Phase Six artifacts were preserved.

## Positive controls

These are algebraic checks on known, noiseless support, evaluated in reverse
row order. They are not independent generalization tests.

| Known encoding | Frozen one-hot | Affine one-hot | Affine scalar |
|---|---:|---:|---:|
| Balanced integers 0–19 stored directly in one coordinate, tested alpha ≤ 1 | 0.100 | 0.100 | 1.000 |
| Imbalanced binary classes separable by an affine threshold | 0.650 | 1.000 | 1.000 |

The first control exposes a limitation of fitting one-hot least-squares
targets to an ordered scalar code. Adding an intercept alone does not fix
it. The second isolates an intercept-sensitive example.

At the scalar control's strongest regularization, alpha 10, shrinkage
reduces accuracy: the saved eigensolve gives 0.450. An independent direct
solve gives 0.500 because eight of 80 predictions lie at exact half-integers
in real arithmetic and roundoff changes which side they reach. Recovery is
1.000 at every tested alpha ≤ 1 in both implementations. The rounding detail does
not affect the instrument-sensitivity conclusion or any independently
refitted real-feature prediction.

## Readout of the saved model activations

Every accuracy pair below is **x / y on the reused pilot**. Each method
selects its own alpha and layer from the same selection split. These are
descriptive comparisons of fitted procedures, not isolated intercept
effects at a fixed layer or significance tests.

| Family | Probe | R1 layer | Carried | Carried, computed | R2 layer | Arithmetic |
|---|---|---:|---|---|---:|---|
| Gemma-3-12B | Frozen one-hot | 9 | 0.653 / 0.379 | 0.000 / 0.123 | 16 | 0.103 / 0.228 |
| Gemma-3-12B | Affine one-hot | 9 | 0.645 / 0.383 | 0.000 / 0.123 | 44 | 0.206 / 0.189 |
| Gemma-3-12B | Affine scalar | 29 | 0.468 / 0.173 | 0.111 / 0.108 | 18 | 0.250 / 0.285 |
| Qwen2.5-7B | Frozen one-hot | 4 | 0.693 / 0.379 | 0.000 / 0.062 | 5 | 0.154 / 0.254 |
| Qwen2.5-7B | Affine one-hot | 4 | 0.695 / 0.383 | 0.000 / 0.062 | 16 | 0.199 / 0.224 |
| Qwen2.5-7B | Affine scalar | 7 | 0.447 / 0.202 | 0.111 / 0.092 | 8 | 0.309 / 0.241 |

The scalar probe raises both Gemma arithmetic scores and Qwen's x score;
Qwen's y score is slightly lower. It also gives poorer carried-value
readout. The historical thresholds require carried ≥ 0.80 and
carried-computed ≥ 0.50 for both variables (R1), and arithmetic ≥ 0.80 for
both (R2). No method meets them. Selected permuted-label control accuracies
are all at most 0.073, below the historical 0.20 limit. These comparisons do
not reopen a gate that U6 already voided.

Final-value readout also depends on the instrument. At the answer position,
affine one-hot gives 0.602 / 0.855 for Gemma and 0.406 / 0.598 for Qwen;
frozen one-hot gives 0.594 / 0.855 and 0.391 / 0.566. Scalar readout is weaker
there: 0.230 / 0.344 and 0.137 / 0.273. At the program's last token, none of
the methods recovers both asked final values reliably: the largest
selected-method score for either variable is 0.301 on Gemma and 0.277 on
Qwen. Better arithmetic readout therefore does not translate into a general
improvement at every position or for every target.

## Evidence and independent checks

Current local run: `runs/review-readout-audit-20261005-v2`. The
[audit summary](../runs/review-readout-audit-20261005-v2/summary.json),
[input and method record](../runs/review-readout-audit-20261005-v2/protocol.lock.json)
and [independent review](../runs/review-readout-audit-20261005-v2/independent_review.json)
are retained locally with the saved predictions. The
[exact independent checker source](../runs/review-readout-audit-20261005-v2/independent_review.py)
is also saved beside the result and matches the source embedded in that
record. This run used no GPU or
new model forward and took 52.43 seconds; independent checking took another
10.76 seconds. Reserved validation was not opened.

The first run, `review-readout-audit-20261005`, is also retained. A
formatting-only correction to a test file prompted an explicit v2 rerun
because that file was part of the recorded input hashes. The producer,
original features, all result metrics, prompt/record metadata and prediction
arrays are identical between versions; only the spelling of `source_run`
changes from an absolute path to an equivalent repository-relative path.
The [original test bytes](../runs/review-readout-audit-20261005/source_snapshot/test_state_probe_audit.py)
are preserved and verified against v1's hash. V1 took 58.55 seconds, so the
two complete audit executions total about 111 CPU wall-clock seconds;
their completed independent checks took 10.47 and 10.76 seconds. These are
execution costs, not a speed comparison between versions.

The independent check:

- matched all **7,828 original numeric grid entries** exactly: 4,944 for
  Gemma and 2,884 for Qwen, including chosen alphas and final-position
  metrics; selected frozen-reading controls and layers also matched;
- rescored every saved selection and pilot prediction and recomputed every
  reading and final-position layer choice from selection scores alone;
- reconstructed all six alpha fits with direct linear solves at Gemma
  layers 9, 16 and 18, and Qwen layers 4, 5 and 8, for asked x and asked y
  under all three methods. The selected alphas and their predictions
  matched exactly;
- verified that the three original state-probe modules have the same
  hashes as in both Phase Six manifests, and that the inspected artifacts
  stayed unchanged during review.

Other feature fits were checked through their saved predictions and
selection logic, not independently refitted. The producer saves the chosen
alpha's predictions rather than all six alpha grids; independent alpha
reconstruction is limited to the listed layers and conditions.

Group and complete-program splits remain disjoint, but boundary inputs are
not all novel. In each family, 390 of 1,740 pilot boundary rows have exactly
the same causal token prefix as a training row: 366 at line 2, 22 at line 3
and two at line 4. Eight of these rows are in the asked-variable arithmetic
category. This is a limitation on novel-prefix generalization, not evidence
that pilot labels were used in fitting. The audit preserved these rows;
it did not deduplicate the historical data.

## What follows

The useful change is a more sensitive and better-characterized instrument,
with clear tradeoffs across targets. The new measurements still do not
meet the tracking thresholds and cannot establish absent state, causal use
of a decoded value, or computation only at answer time. No confidence
interval or new significance claim is attached to the audit improvements.

The next prompt experiment must also separate question wording from
position; the reviewed [Phase Six report](phase_six_question_first.md)
explains why its existing comparison cannot do that. Any new experiment
needs its own design and evidence, separate from this completed CPU audit.

## Independent cross-check with a second implementation (2026-10-05, CPU, exploratory)

A second review session refit the readouts with separately written code,
`scripts/probe_family_cross_check.py`, which imports none of this audit's
modules. It reads the same saved activations, chooses settings on the
selection split and treats pilot scores as exploratory. Outputs are kept
locally in `runs/review-probe-cross-check-20261005/` (Gemma JSON
`a5b06d5f9d89…`, Qwen `f3038a752126…`). That run took 57 min (Gemma) and
37 min (Qwen) on 8 CPU threads each.

**Agreement.** Where the two implementations fit the same readout, the pilot
scores match to three decimals:
- the frozen one-hot readings (for example Gemma R1 0.653 / 0.379, R2
  0.103 / 0.228);
- the affine scalar readings (Gemma R2 0.250 / 0.285 at layer 18, carried
  0.468 / 0.173 at layer 29);
- the scalar answer-position values (0.230 / 0.344);
- the 390 of 1,740 boundaries whose prefix also occurs in training, 8 of
  them arithmetic.

It also adds four checks this audit did not run.

1. **A logistic affine classifier:** multinomial logistic regression with an
   intercept, L2-penalized, on standardized inputs. It does not beat the
   frozen readout on the running state.
   - Gemma, asked variable: arithmetic 0.213 / 0.246 at layer 19, carried
     0.441 / 0.282.
   - Qwen: arithmetic 0.169 / 0.241, carried 0.590 / 0.323.
   - At the answer position it matches the ridge readouts: Gemma, question
     after, 0.949 / 0.918.
2. **A planted code in the real activations.** At the middle layer
   (Gemma 24, Qwen 14), the true running value was added to the
   question-after boundary activations, as a one-hot or a scalar direction.
   Strength is the planted norm relative to the mean centred activation
   norm. These are held-out accuracies over all boundaries, x / y, Gemma;
   Qwen is similar.

   | Planted code, strength | 0 | 0.02 | 0.04 | 0.08 |
   |---|---|---|---|---|
   | One-hot, frozen one-hot readout | 0.46 / 0.44 | 0.88 / 0.86 | 1.00 / 0.99 | 1.00 / 1.00 |
   | One-hot, logistic readout | 0.43 / 0.47 | 0.68 / 0.68 | 0.92 / 0.91 | 0.99 / 0.99 |
   | Scalar, frozen one-hot readout | 0.46 / 0.44 | 0.47 / 0.45 | 0.47 / 0.45 | 0.47 / 0.45 |
   | Scalar, logistic readout | 0.43 / 0.47 | 0.44 / 0.48 | 0.47 / 0.49 | 0.50 / 0.51 |
   | Scalar, affine scalar readout | 0.38 / 0.38 | 0.58 / 0.56 | 0.78 / 0.78 | 0.95 / 0.96 |

   The frozen readout would have caught a categorical code of a few percent
   of the activation norm. It is blind to a magnitude code at every
   strength tested, and so is the logistic classifier. This is the
   constructed-control finding above, reproduced in realistic noise.
3. **Asked minus not-asked arithmetic accuracy, per readout,** with
   whole-group 95% intervals. Every interval spans zero.
   - Gemma: frozen +0.014 [−0.014, +0.043]; logistic −0.003
     [−0.020, +0.015]; scalar +0.038 [−0.011, +0.091].
   - Qwen: +0.016, +0.000 and +0.008.
4. **Question-after boundaries** (Phase Five's re-capture).
   - 195 of 870 held-out boundaries share a training prefix, 8 of them
     arithmetic.
   - On the 356 arithmetic boundaries with novel prefixes, accuracy is
     0.15–0.18 for the categorical readouts and 0.247 (Gemma) / 0.236
     (Qwen) for the scalar readout.
   - Prefix reuse does not drive the arithmetic scores.

Together these sharpen the bound. A categorical running-state code of more
than a few percent of the activation norm would very likely have been read.
A magnitude code would not have been read by the original instrument. The
scalar readout, which can read one, reaches only 0.19–0.31 on arithmetic
results across both models and both question formats. Neither result establishes that state is absent.
