# Pilot decision — Phase Two open_weight_lingua

**Status: TEMPLATE, published with the Milestone 2 implementation. Every
outcome field is PENDING PILOT RUN.** [The brief](phase_two_brief.md) §11
requires this document from the 128-group pilot: keep/stop, intervention
sensitivity, edit eligibility, final operating choices, projected validation
cost, and any deviations from the brief. The pilot runner writes the
machine-readable counterpart — decision fields in `completion.json` and a
decision section in `report.md`; copy those values here. Do not fill any field
from a partial run, and do not edit this document after the validation
manifest is locked.

## 0. Standing commitments (fixed before the pilot)

- The thresholds in §2 were frozen — copied verbatim from brief §8 as design
  choices — **before any pilot outcome was read**. They are not results, and
  they are never weakened on held-out data.
- At most one locked validation follows this pilot. **A negative pilot does
  not trigger checkpoint shopping**: no search for a different checkpoint,
  site, layer, or task. A repaired source/API bug permits only a regression
  test plus a visibly versioned rerun of the affected pilot (brief §11).
- If edit coverage or intervention sensitivity is inadequate, this phase
  closes with preservation-only results and the edit hypothesis is marked
  explicitly untested or unsupported (brief §8).
- The hashes below identify artifacts; they do not authenticate a remote
  execution.

## 1. Run identity

| Field | Value |
|---|---|
| Pilot run directory | PENDING PILOT RUN |
| Pilot `manifest.json` sha256 | PENDING PILOT RUN |
| Calibration run directory | PENDING PILOT RUN |
| `baseline_fit_identity` (sha256 content hash of the P4 fit) | PENDING PILOT RUN |
| Code and protocol hashes (from the manifest) | PENDING PILOT RUN |
| Frozen edit-rule version and sha256 | PENDING PILOT RUN |
| Bootstrap configuration | Planned: 3,000 whole-group resamples, fixed seed; actual seed PENDING PILOT RUN |
| Date and operator | PENDING PILOT RUN |

## 2. Frozen decision thresholds (design choices, locked before reading outcomes)

Quoted from brief §8. These are proposals locked before validation, not claims
from a paper and not universal definitions of interpretability.

| Gate | Frozen rule |
|---|---|
| 1. Task usable in pilot | P0 unmodified exact-answer accuracy at least 80%, and donor/perturbation controls show this site can affect the intended measurement; otherwise stop with an assay limitation |
| 2. Limited behavioral preservation | One-sided 95% upper estimate of P2 accuracy loss versus P0 at most 5 percentage points, and P2 beats P3 on correct-answer log probability with a positive lower confidence estimate |
| 3. Edit feasibility | At least 32 of the 128 pilot groups permit the frozen edit rule; otherwise close with preservation-only results |
| Uncertainty | Whole-group bootstrap, 3,000 resamples, fixed seed; prompt variants within one group are not independent samples |

For the later locked validation, brief §8 additionally requires the pooled
criteria **and** both 256-group validation-block point estimates in the
intended direction. That clause is restated here so it cannot be forgotten at
validation time.

## 3. Gate 1 — task usability and intervention sensitivity

| Quantity | Value | Threshold | Met? |
|---|---|---|---|
| P0 exact-answer accuracy (intention-to-test: every planned variant, failures count against) | PENDING PILOT RUN | ≥ 0.80 | PENDING PILOT RUN |
| Raw-donor effect on the intended measurement (absolute answer/log-probability shift) | PENDING PILOT RUN | Sensitivity shown, per the gate-1 wording | PENDING PILOT RUN |
| P5 norm-matched random-direction effect | PENDING PILOT RUN | Diagnostic, interpreted together with the donor | PENDING PILOT RUN |

Narrative on whether this site can affect the intended measurement: PENDING
PILOT RUN. Near-zero donor effects are reported as absolute effects; no
ratio-style "fraction recovered" is computed.

## 4. Gate 2 — limited behavioral preservation

| Quantity | Point estimate | One-sided 95% estimate | Threshold | Met? |
|---|---|---|---|---|
| P2 accuracy loss versus P0 | PENDING PILOT RUN | Upper: PENDING PILOT RUN | Upper ≤ 5 percentage points | PENDING PILOT RUN |
| P2 − P3 correct-answer log probability | PENDING PILOT RUN | Lower: PENDING PILOT RUN | Lower > 0 | PENDING PILOT RUN |

P4 (no-more-than-budget PCA) versus P2, descriptive comparison only: PENDING
PILOT RUN. Calibration-median-norm diagnostic, exposing reliance on the
retained norm side channel: PENDING PILOT RUN.

## 5. Gate 3 — edit eligibility and coverage

| Quantity | Value | Threshold | Met? |
|---|---|---|---|
| Groups eligible under the frozen edit rule | PENDING PILOT RUN / 128 | ≥ 32 | PENDING PILOT RUN |

Coverage and exclusion-reason distribution (`eligible` / `absent` /
`ambiguous`) over all 128 groups: PENDING PILOT RUN. Recorded agreement
between the original explicit statement and the reference program — a
diagnostic that never gates an edit: PENDING PILOT RUN.

## 6. Secondary edit evidence (only if gate 3 is met)

On the prespecified eligible groups, as absolute log-probability contrasts:
edited versus unedited reconstruction; edited versus same-sized
wrong-variable edit; raw-donor contrast; unaffected-variable errors: PENDING
PILOT RUN. If gate 3 fails, this section stays empty and the edit hypothesis
is marked untested.

## 7. Per-condition failure accounting (intention-to-test)

| Condition | Successful | Failed | Skipped | Uneditable |
|---|---|---|---|---|
| P0 unmodified | PENDING PILOT RUN | PENDING PILOT RUN | PENDING PILOT RUN | n/a |
| P1 reinsert original | PENDING PILOT RUN | PENDING PILOT RUN | PENDING PILOT RUN | n/a |
| P2 own description | PENDING PILOT RUN | PENDING PILOT RUN | PENDING PILOT RUN | n/a |
| P3 other group's description | PENDING PILOT RUN | PENDING PILOT RUN | PENDING PILOT RUN | n/a |
| P4 PCA baseline | PENDING PILOT RUN | PENDING PILOT RUN | PENDING PILOT RUN | n/a |
| P5 random direction | PENDING PILOT RUN | PENDING PILOT RUN | PENDING PILOT RUN | n/a |
| Raw donor | PENDING PILOT RUN | PENDING PILOT RUN | PENDING PILOT RUN | n/a |
| Calibration-median-norm diagnostic | PENDING PILOT RUN | PENDING PILOT RUN | PENDING PILOT RUN | n/a |
| Edited reconstruction | PENDING PILOT RUN | PENDING PILOT RUN | PENDING PILOT RUN | PENDING PILOT RUN |
| Wrong-variable edit | PENDING PILOT RUN | PENDING PILOT RUN | PENDING PILOT RUN | PENDING PILOT RUN |

Missing generations count against accuracy; failed reconstructions never
manufacture finite KL values; if failures invalidate a claimed aggregate,
that outcome is marked inconclusive/failed.

## 8. Resources and projected validation cost

| Quantity | Value |
|---|---|
| Measured pilot wall time, model calls, allocation peaks | PENDING PILOT RUN |
| Projected 512-group validation wall time (from measured pilot throughput) | PENDING PILOT RUN |
| Eight-hour budget check | PENDING PILOT RUN (projection within or above budget) |

If the projection exceeds the eight-hour budget, report the estimate and stop
for a scope decision before opening validation; do not shrink samples or
conditions midway (brief §9). Validation is never auto-started.

## 9. Final operating choices for validation (locked here before it opens)

| Choice | Locked value |
|---|---|
| P4 rank rule | PENDING PILOT RUN (planned: `rank = min(fitted_rank, floor(text_bytes/2))`, float16 coefficients, no-more-than-budget) |
| Norm policy | PENDING PILOT RUN (planned: retained per-example original norm; calibration median as diagnostic only) |
| Logit tolerances and greedy-answer agreement | PENDING PILOT RUN (planned: brief §4 starting values `atol=1e-5`, `rtol=1e-5`; never relaxed after inspecting validation) |
| Canonical answer formatting and description budget | PENDING PILOT RUN (planned: frozen answer suffix tokenization; one greedy description per activation, 200-new-token ceiling) |
| Numerical editing criteria | PENDING PILOT RUN (brief §8 requires finalizing them here if gate 3 is met; otherwise mark the edit hypothesis untested) |

## 10. Decision

**KEEP / STOP: PENDING PILOT RUN.**

Rule derivation — which gates passed or failed and which frozen rule produces
the decision: PENDING PILOT RUN.

## 11. Deviations from the brief

PENDING PILOT RUN. List every deviation from
[the brief](phase_two_brief.md) with its justification, or state "none".
