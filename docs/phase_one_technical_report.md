# Structured-State Lingua: Phase-One Closeout

**Status: phase closed. No further experiment in this sequence is required.**

[Important findings](phase_one_results.md) · [Result summary and provenance](../experiments/results/phase_one_20260916.json)

## Executive result

The locked final validation supports **limited comparative usefulness** for the selected linear response rule, but **does not support close numerical prediction**. The pooled relative root-mean-square (RMS) response-prediction error is **0.690756451**, versus 1.0 for predicting the unmodified output. This is **30.92% less RMS prediction error**, not an increase in task accuracy.

Both fresh seed-block point estimates, 0.675015613 and 0.708026274, are below the preset limited-usefulness limit of 0.8. The one-sided 95th-percentile bootstrap upper estimate is 0.732032068, also below 0.8. The separate close-prediction limit of 0.1 is not met. These were practical criteria fixed after the exploratory pilot but before this holdout, not general standards for interpretability.

The defensible outcome is a structured-state research prototype with approximately faithful numerical encoding and typed interventions, plus a local response approximation with demonstrated limitations. It is not a general language interpreter, a discovery of opaque-neuron meanings, or a universally reliable predictor.

## 1. Concrete question

A model estimates flow values using candidate mathematical structures. Its internal candidate cycle coefficients and current route evidence are explicit numerical fields. Earlier experiments asked whether a compact representation could preserve those fields well enough to continue the computation, and whether named commands could reproduce direct numerical edits. The final question was different: **can a mathematical rule predict the model's response to an edit that was not one of its construction probes?**

The final validation holds the weights, original calibration ranges, candidate spaces, claim thresholds, edit families, and computation budget fixed. It neither trains a new neural model nor repairs the original solver's task errors.

## 2. Locked protocol

| Item | Fixed choice |
|---|---|
| Primary model | Dedicated four-step recurrent model |
| Pause point | After update 2; execute the two remaining original updates |
| Primary predictor | Linear response rule |
| Probe radius | 1/8 in the specified normalized coordinates |
| Edits | Nine noncontrol families at scales 1/4, 1/2, and 1 |
| New input blocks | Seeds 91000 and 92000, 256 inputs each |
| Frozen training runs | 6100, 7100, 8100, 9100, 10100 |
| Limited-usefulness criterion | Each block point and pooled one-sided 95% bootstrap upper estimate <= 0.8 |
| Close-prediction criterion | The same rule at <= 0.1 |
| Bootstrap | 3,000 paired-input resamples within blocks; seed 93000 |
| Stop rule | Document the result and close this phase, regardless of outcome |

All five fits and all queries share each block's input problems. There are **512 distinct held-out input problems**, not 512 times the number of fits, cuts, queries, and radii. The current audit confirms no duplicate observed/mask pairs within or between the two blocks and no overlap with the earlier seed-87000 pilot observations.

The quadratic, alternate-radius, alternate-model, and alternate-cut results remain secondary. None has replaced the locked primary after seeing this holdout.

## 3. Definition of the result

Let F(h,c) be the original remaining computation followed by the numerical mixture readout. Here h is the six-field dynamic state and c is retained input/context. A finite-difference rule produces a forecast y_hat(u) for the edited response y(u)=F(h+d*u,c), where d contains frozen field scales. The no-change forecast is y0=F(h,c).

The reported ratio is:

    R = sqrt(mean((y_hat - y)^2) / mean((y0 - y)^2)).

The mean gives equal weight to each input, fixed fit, noncontrol query, and output component. R=1 means the same pooled squared error as predicting no change. R=0 would mean exact numerical prediction on the measured cases. This ratio is **not a fraction of incorrect outputs, an accuracy percentage, a claim-error rate, or an explanation-validity score**.

Squared errors and effect energies are averaged across queries and fits before the input-level bootstrap. Predictions themselves are not averaged into a neural ensemble. Inputs are resampled jointly for numerator and denominator within each seed block. The uncertainty is conditional on these five fits and these fixed queries; it does not quantify uncertainty over all trained models or all tasks.

## 4. Final primary results

| Quantity | Independently recomputed value |
|---|---:|
| Block 91000 relative RMS error | 0.675015613 |
| Block 92000 relative RMS error | 0.708026274 |
| Pooled relative RMS error | 0.690756451 |
| Paired two-sided percentile 95% interval | [0.635418478, 0.739562322] |
| One-sided 95% upper estimate used for decision | 0.732032068 |
| Prediction MSE | 0.000255458570 |
| No-change prediction MSE | 0.000535390399 |
| Prediction RMSE | 0.015983071366 |
| No-change RMSE | 0.023138504685 |
| Limited-usefulness criterion | MET |
| Close-prediction criterion | NOT MET |

The linear rule has lower query/fit-averaged error than no change on 480 of 512 inputs. This is a post-hoc descriptive count, not an additional success criterion or a per-query success rate.

The pooled decision is not an assertion that every trained fit passes separately. Per-fit primary ratios over both input blocks are:

| Training seed | Relative RMS error |
|---|---:|
| 6100 | 0.822525 |
| 7100 | 0.672505 |
| 8100 | 0.642333 |
| 9100 | 0.663159 |
| 10100 | 0.622463 |

The 6100 fit exceeds 0.8 on its own. The predeclared decision rule required seed-block and pooled results, not a separate pass for every fit. That distinction is retained rather than silently changing the criterion.

## 5. Secondary results: precision of a local rule has a limited range

At the SAME four-step midpoint and primary radius, after remaining updates:

| Approximation | Small edits | Medium edits | Large edits |
|---|---:|---:|---:|
| No change | 100.00% | 100.00% | 100.00% |
| Linear | 30.92% | 51.66% | 74.41% |
| Diagonal quadratic | 21.81% | 52.10% | 116.31% |
| Full quadratic | 14.59% | 46.84% | 114.51% |

These are secondary error ratios, not failure percentages. The full quadratic rule helps at small edits but is worse than predicting no change at the largest edits at this primary site. Across all four model/pause sites, the large-edit full-quadratic error is approximately 227.6%, versus 76.5% for linear. Those pooled-secondary numbers are not substituted into the primary decision.

The holdout therefore reproduces the central limitation from the pilot: more local curvature does not reliably solve extrapolation. This is not an implementation failure requiring an automatically expanded approximation family.

## 6. Costs and limits

The common evaluator constructs all comparisons from **73 probes per input, cut, horizon, and radius**: one baseline, twelve axial probes, and sixty mixed probes. Inputs can be processed in a batch, but their individual response rules still require those model evaluations. The stored full coefficient set uses **1,720 bytes per input**, compared with a 24-byte dynamic state and 304 bytes of retained context. The algebraic size of a standalone linear base/Jacobian would be 280 bytes, but this study did not implement or benchmark an optimized linear-only runtime.

Construction plus all predictions was timed once for each configuration. Those timings are descriptive and are not a controlled performance benchmark. No speedup, memory saving, or inexpensive general-purpose interpreter follows from the primary pass.

This final run tests raw known numerical fields. It does not retest the compressed-state or typed-language path end to end. The original input and encoder context remain available during resumed neural computation. The predictor returns numerical response forecasts, not certified structural claims. It is a local, input-specific approximation, not a universal surrogate.

## 7. What the audit can establish

The uploaded bundle has thirteen members: the locked protocol, ten worker reports, a summary, and a compact per-input loss archive. The independent audit checked ZIP integrity, NPZ hash integrity, numerical shapes/finiteness, all worker report hashes, source fingerprints declared in the protocol, exact source/metadata consistency across reports, original continuation report hashes, frozen scale manifests, query coverage, and input disjointness.

It recomputed **1,920 per-input loss cells and 192 summary cells**, reconciled the averages against **21,120 reported per-query score rows**, and reproduced the primary point, both block points, all 3,000 bootstrap resamples, and the locked pass/fail conclusions. It inspected the reports of **160 finite rule constructions and 40 exact raw-restoration gates**. No discrepancies were found in these checks.

**The full neural probes, predictions, coefficients, and checkpoints are not included in this compact upload.** They remain on the DGX. Consequently this audit did not recalculate each per-input error from raw predictions, rebuild the new holdout coefficients, rerun the model, or verify checkpoint bytes. It verifies the available losses and summaries conditional on the reported neural results. Hashes identify data and do not authenticate remote execution or the time of protocol registration.

## 8. What survives from the earlier phase

Earlier independently reviewed reports support three narrower accomplishments: compact known-field states can approximately preserve continuation; a prepared/batched verifier can reduce recording/checking overhead under its measured conditions; and specified mathematical commands can preserve direct intervention effects. Those are separate from this final forecast result.

The frozen-codec generalization report contained seven changed claim/abstention decisions at sixteen bits across correlated checkpoint/cut/observation comparisons on 2,048 underlying problems. The single-edit report contained five decoded/direct claim differences across correlated comparisons on 512 underlying problems. Thus neither representation nor edits had an all-input decision-preservation guarantee. These earlier findings are summarized from their existing audits, not newly re-audited in full for this closeout.

An appropriate project description is:

> A structured recurrent prototype with checkable numerical records, approximately faithful state descriptions, editable known mathematical fields, and a limited-usefulness local predictor of intervention responses.

An inappropriate description would claim a general lingua franca for opaque models, discovered hidden-neuron meanings, unrestricted English-to-activation decoding, guaranteed correct structure selection, or autonomous discovery and transport between reasoning spaces. This phase does not establish those capabilities.

## 9. Demonstration and stopping decision

The accompanying `results_and_edit_replay.html` shows the locked result and a recorded intervention explorer. The explorer uses checkpoint 6100 and the first sixteen inputs in recorded order from the earlier mathematical-edit pilot. It displays all commands for four model/cut sites, with original, direct-edited, decoded-edited, and wrong-candidate outputs where available. This subset was not selected for particularly good results and is not used to calculate the closeout score.

The explorer is an offline replay of saved responses. It does not run new model inference, implement unrestricted text prompts, or certify a new execution. The selected edit-state arithmetic was checked again when assembling the replay.

**Close this experimental sequence now.** No third run is required by the checks performed, and no new architecture, codec, or approximation has been added. This is a completed research phase with a positive limited result and an explicit negative close-prediction result, not completion of the larger opaque-model interpretability ambition.

## 10. Provenance and reproducibility

This page adapts the completed closeout write-up, not a new experiment or neural
reanalysis. The accompanying replay and full closeout package were supplied
separately; references to them above do not mean they are hosted in this repository.

| Source | Identity / scope |
|---|---|
| Final input archive | `phase-closeout-20260916-165013-qpUqCa-reports.zip` |
| Archive SHA-256 | `5c3037f54583514e9e0644f4cde9c2f08578695bdd945edbbce340e13b391118` |
| Measured project version | 0.5.2 |
| Reported closeout commit | `7d7221eaf133645eabf55f1339bdaa00e1762ff3` |
| Frozen evaluator commit | `3eb96a9d01ae4828a6a697c47c7d861489fa1935` |

The [public result summary](../experiments/results/phase_one_20260916.json)
contains the primary estimates, the primary-site secondary aggregates, per-fit variation,
audit scope and source-archive identities. It does not contain the original
per-input losses, full predictions, model probes or checkpoints. Those artifacts
must be available to reproduce their respective levels of analysis; aggregate
JSON alone is not enough to rebuild the bootstrap or rerun the neural model.

The [locked protocol](../experiments/phase_closeout_v1.json) and
[collector](../scripts/phase_closeout.py) remain unchanged. The original
`audit_closeout.py` and its input-loss evidence were distributed in the separate
closeout package, not newly installed by this documentation update. Reproduction
is optional; no further run is required to close this phase.

Earlier preservation, intervention and engineering statements are summarized
from their existing report audits, not newly re-audited in full for this page.
See [findings and evidence](phase_one_results.md#evidence-and-audit-scope) for their
separate sample counts and limitations. No local filesystem paths, model weights,
private application code or raw uploaded report bundles are published here.
