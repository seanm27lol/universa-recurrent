# Roadmap: earn each connection

## Phase-one status: complete

The structured-state experimental sequence closed on 2026-09-16 after its locked
fresh-input validation. The selected linear predictor met the declared
**limited-usefulness** criterion, not the separate **close-prediction** criterion.
See the [important findings](phase_one_results.md) and
[technical report](phase_one_technical_report.md).

| Completed part | What survived evaluation |
|---|---|
| Numerical recording and checking | Workload-specific setup/batching savings; remaining tail latency is reported |
| State descriptions | Approximate continuation preservation with explicit overflow fallback and counted costs |
| Typed mathematical edits | Specified field interventions remain closely aligned with direct edits |
| Response prediction | A useful but inaccurate local linear approximation; quadratic extrapolation can fail |
| Final validation | Locked primary, two fresh input blocks, explicit positive and negative criteria |

**No more GPU runs are required for this phase.** Existing scripts remain for
reproducibility, not as an automatic task queue. Documentation and a recorded
results demonstration close the sequence. The recorded demonstration is a replay,
not a live language model.

The broader goals below remain separate, uncompleted work. This closeout does not
establish discovery of new spaces, transport between spaces, opaque-neuron
semantics or unrestricted natural-language reconstruction. A future phase would
need its own task, baselines, budget and stopping rule.

## Original project stages (not a new run list)

| Stage | Deliverable | Gate before a claim |
|---|---|---|
| 0 | Transparent classical solver and witnesses | Independent numerical checks |
| 1 | Neural v1: one early route, shared recurrence | Exploratory only; failure modes retained |
| 2 | **Neural v2: parallel hypotheses, route revision, commit/abstain** | Full controls, calibration/test separation, DGX benchmark |
| 3 | Typed state transport between structures | Check transport laws and quantify information loss |
| 4 | Structure discovery and admission | Novelty/certification tests and refusal controls |
| 5 | Compact Lingua budgets | Preserve declared checks while reducing storage/check cost |
| 6 | Neural semantic interpretation | Held-out predictive and causal interventions |

## Broader v2 questions retained for future scoping

1. Does delaying commitment reduce wrong-route damage on an untouched test seed?
2. Does the posterior mixture beat hard selection without privileged generator knowledge?
3. Does active compaction lower wall-clock cost at any realistic batch size?
4. Does v2 outperform direct, untied, ambient, and dedicated-depth controls?
5. Are its route probabilities sufficiently calibrated for the declared rejection rule?

A direct solver, a refusal, or a negative latency result can be the correct outcome.
