# Phase one: what the experiments actually established

**Status: complete, 2026-09-16.** This page summarizes the existing reported
experiments and their audits. It introduces no new model, benchmark or required
GPU run. The measured runtime version remains **0.5.2**.

> We can cheaply check specified numerical properties, approximately preserve a
> structured state, and edit its named quantities. A local formula predicts some
> of the consequences, but not closely enough to call it a general explanation.

[Technical report](phase_one_technical_report.md) ·
[Exact aggregate results and source identities](../experiments/results/phase_one_20260916.json) ·
[Claim ledger](claims.md)

## Start with the concrete problem

Imagine estimating flow around a small network from noisy, incomplete readings.
The prototype keeps several candidate structural explanations instead of
immediately declaring one correct. Within each candidate, it stores cycle
coefficients; separate evidence values determine how much weight to give each
candidate. Learned recurrent updates refine these quantities.

A **receipt** is a numerical record of specified calculations and constraints.
A **state description** is an encoding of the current known fields. A **response
rule** forecasts what an edit will do. These answer different questions:

```text
RECORD                   PRESERVE                 EDIT                   PREDICT
Do the listed            Can we restore           Can we change          Can we forecast
calculations check?      useful working values?   the intended field?    the consequences?
       |                        |                        |                       |
       +------------------------+------------------------+-----------------------+
                                None alone proves
                      that the model's answer is correct.
```

For candidate k, `z_k = Q_k a_k`: `a_k` contains cycle coefficients, `Q_k` is its
known basis and `z_k` is the candidate signal. That makes candidate geometry
checkable. It does not certify which candidate generated the observation, and a
weighted mixture of candidates does not inherit a single candidate's constraint.
See the [worked dual-output example](dual_outputs.md).

## 1. Recurrence is useful, not an automatic winner

In the five-training-run study on one shared 4,000-input evaluation cohort:

| Model | Mean task MSE, lower is better |
|---|---:|
| Shared eight-step recurrence | **0.049565** |
| Dedicated four-step recurrence | **0.049900** |
| One-pass model | 0.052676 |

Eight steps had 5.90% lower mean error than one-pass, but only a 0.67% mean-error
advantage over four steps. The eight-step model beat one-pass in all five runs,
but beat four steps and untied depth in only three of five. Its additional work
was not free. Different models also retained different training objectives.

**Implication:** keep four-step recurrence as the practical structured baseline,
eight steps as a depth comparison and one-pass as a fast control. This is not a
universal architecture ranking or an isolated proof that recurrence causes gains.
[Original replication results and timing scope](replication_20260909.md).

## 2. The measured efficiency win is mostly in recording and checking

For a complete **256-input shared-model endpoint request**, batching receipt
creation and checking reduced measured time from **27.13 to 14.15 ms**. Both paths
already used a prepared verifier. The earlier large gains from preparing the
reference once were measured against repeatedly loading the same files; they
must not be described as neural-model speedups.

These are medians of five checkpoint-level medians, each from ten repetitions.
The boundary includes inference, transfers, receipt creation/JSON, parsing and
checking. Models and verifier references are already loaded; network transport,
queueing and report writing are excluded. They are local batched request timings,
not single-user service latency or new accuracy results.

The remaining limitation is visible: for eight-step full trajectories, chunks of
16 gave a **47.23 ms median-of-medians**, but a **101.32 ms median of checkpoint
90th-percentile times**. A separate diagnostic pass observed substantial Python
collection pauses. It did not establish the cause of every uninstrumented spike.

**Implication:** preserve the prepared/batched path and the slow samples. Use bulk
conversion for endpoints and chunk16 as a provisional trajectory option. Do not
advertise a hard latency bound. These are engineering results about the receipt
pipeline, not validation of a general witness theory.
[Batch-check design](batched_receipts.md), [bounded-conversion design](bounded_receipts.md),
[recorded aggregate evidence](../experiments/results/phase_one_20260916.json).

## 3. Approximate state preservation is not exact decision preservation

The codec stores the known state fields at reduced precision. When a value
exceeds its frozen calibration range, it stores the **whole original float32
state** instead of clipping the value. In-range rounding is still present.

On the new-input generalization study:

| Precision | Changed claim/abstention decisions | Worst final component change |
|---|---:|---:|
| 8 bits | 1,646 / 184,320 | 0.08620 |
| 12 bits | 127 / 184,320 | 0.00656 |
| **16 bits** | **7 / 184,320** | **0.000314** |

There are **2,048 underlying problems**, observed under normal, noisier and
sparser conditions. The denominator repeats those problems across five fits,
three conditions and six model-specific cuts; it is not 184,320 independent
trials. The seven sixteen-bit claim changes were threshold crossings between
issuing and withholding a claim, not changes between two issued candidates.
Separately, leading-candidate changes were also measured.

The decoder receives the numerical payload and frozen metadata, not future
states or input context. The resumed solver still has its original context.
The readable text is a display, not a trained English reconstruction channel.

**Implication:** sixteen bits is a useful approximate experimental baseline, not
an all-input guarantee. Full-state overflow fallback fixes clipping, not rounding.
[Codec protocol](codec_generalization.md), [overflow rule](overflow_continuation.md).

## 4. Named edits preserve the intended numerical intervention

A command such as `cycle balanced_flow[0] += 0.125` addresses an explicitly named
coefficient. The pilot compared the command on a decoded state with the same edit
on the original numbers, and with an edit deliberately applied to the wrong
candidate.

| Against the intended direct edit | Correct decoded command | Wrong-candidate control |
|---|---:|---:|
| Mean squared disagreement in the edit's effect | **1.72e-11** | 0.00513 |
| Different claim/abstention decisions | **5 / 122,880** | 31,681 / 122,880 |

Each path's own no-edit result is subtracted before comparing effects. The same
**512 inputs** are reused across fits, cuts and twelve targeted commands. The
five correct-command differences were issuance/abstention crossings. The direct
edits themselves changed 17.1% of decisions in these repeated comparisons, so the
experiment was not only preserving edits that had no effect.

**Implication:** this is an editable mathematical interface, beyond a passive log.
But field names and their meanings were supplied by the architecture. Parser
correctness and numerical fidelity do not discover opaque-neuron semantics.
Commands are applied after decoding without another quantization step.
[Typed-edit implementation and scope](mathematical_edits.md).

## 5. Preserving edits does not make their effects additive

Two individually meaningful edits can interact, just as changing price and
quantity together changes revenue by more than the sum of their separate changes.

The composition study predicted a joint answer as `yA + yB - y0`, using separately
measured single-edit responses. For the two within-candidate cycle-pair families
at full edit size, relative RMS prediction error was **47.9% and 60.9% after
recurrence**, versus near-roundoff interaction at immediate readout. The same
**256 inputs** were reused across fits, cuts, pairs and sizes.

The decoded joint edits still closely matched the direct joint edits. Therefore
this result separates **nonlinear interaction in the model** from **error caused
by the numerical description**. The additive predictor was useful relative to
predicting no change, but not a complete account of joint behavior.

**Implication:** a list of isolated field effects is insufficient as a general
explanation. This diagnostic paid for the baseline and both individual model
responses; it was not a free predictor. [Composition experiment](edit_composition.md).

## 6. The locked final predictor is useful, but not close

The final primary was fixed before its new input results: **four-step model,
paused at step 2, linear rule, probe radius 1/8, all specified noncontrol edit
families and sizes**. The rule predicts the response after the remaining two
updates. This test uses raw state fields, not the compressed typed-edit pipeline.

| Final validation quantity | Relative RMS error |
|---|---:|
| New input block 1 | 67.5% |
| New input block 2 | 70.8% |
| **Pooled result** | **69.1%** |
| One-sided 95% upper estimate used by the criterion | 73.2% |
| Limited-usefulness limit | At most 80%: **met** |
| Close-prediction limit | At most 10%: **not met** |

```text
Exact prediction        Close limit          Observed     Limited limit   No change
       0%                    10%                69.1%           80%           100%
```

The ratio is `sqrt(mean(prediction error squared) / mean(no-change error squared))`.
Thus the result is **30.9% less RMS response-prediction error than no change**, not
30.9% higher task accuracy. It is not a failure rate or an explanation-validity
score. These limits were practical choices made after the pilot but before this
holdout, not universal standards for interpretability.

There are **512 distinct new inputs**, shared across five fixed fits. A paired
bootstrap resamples inputs within the two blocks after averaging losses across
fits and queries; predictions are not averaged into an ensemble. Its uncertainty
is conditional on those fits and queries. One individual fit had an 82.3% ratio,
so the pooled pass does not mean every fit passed separately.

More curvature was not a general solution. At the same primary site, the full
quadratic secondary had **14.6%, 46.8% and 114.5%** relative error for small,
medium and large edits. Above 100% is worse than predicting no change. It helps
locally but extrapolates poorly; it does not replace the locked primary.
[Locked protocol](../experiments/phase_closeout_v1.json),
[full primary and secondary interpretation](phase_one_technical_report.md).

## 7. Preservation, truth and cost must stay separate

| Question | What the evidence permits |
|---|---|
| Is the encoded response close to the original model? | Often, with explicit small failures |
| Is the original model's structural claim true? | Not guaranteed: its errors grow under observation shifts |
| Is a checked numerical record a semantic explanation? | No; it certifies only its stated numerical properties |
| Is the response rule cheap or compact? | Not in this implementation |

For the uncompressed eight-step model, wrong-generating-label rates among issued
claims were **11.2%, 16.1% and 22.8%** under normal, noisier and sparser observations.
Coverage also changed. Preserving those outputs does not repair their errors.

The response evaluator used **73 model probes per input/cut/horizon/radius** and
stored **1,720 bytes of coefficients per input**. The original dynamic state is
24 bytes, with 304 bytes of retained input/context. Even the sixteen-bit encoded
record needs roughly 45 bytes once its identifier and fallback tag are included,
before shared metadata and context. None of these codec/predictor results is a
measured whole-model memory saving or neural speedup.

## What this says about different spaces for reasoning

The prototype operates in explicit candidate spaces and makes their known fields
accessible to checks and edits. That is a concrete part of the original idea.
It does **not** demonstrate autonomous discovery of new spaces, persistent-state
transport between them, a universal benefit from a particular topology, or a
lingua franca that discovers meanings in arbitrary opaque models. HOMYMOLY,
Universa and Applied CMCM remain distinct upstream motivations or components;
this phase does not transfer its measured results to every upstream claim.

A defensible description is:

> A structured recurrent prototype with checkable numerical records,
> approximately faithful state descriptions, editable known mathematical fields,
> and a limited-usefulness local predictor of intervention responses.

## Evidence and audit scope

Every numerical section above is tied to an existing experiment in the
[machine-readable source registry](../experiments/results/phase_one_20260916.json).
Archive names and SHA-256 values identify the supplied evidence; raw archives and
weights are not included in this documentation update.

Earlier audits replayed available quantization, edit, receipt and metric
arithmetic. The final compact upload permits recomputation from saved per-input
losses, not regeneration of its missing probes/predictions or neural outputs.
Its audit reconciled **192 summary cells and 21,120 reported metric rows** and
reproduced the 3,000-resample bootstrap. The complete scope, including what was
not replayed, is in the [technical report](phase_one_technical_report.md#7-what-the-audit-can-establish).

Hashes identify artifacts; they do not authenticate neural execution or prove
when a protocol was registered. Historical experiment descriptions remain for
reproducibility. Their old next-step language is not a new required run list.

**This phase is closed regardless of the negative close-prediction result.**
A future research phase would need a separate task and stopping rule. No new
architecture, parameter search or automatic follow-on experiment is part of
publishing these findings.
