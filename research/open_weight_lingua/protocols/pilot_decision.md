# Pilot decision — Phase Two open_weight_lingua

**Status: FILLED from the completed pinned pilot run
`pilot-20260921T235825Z-6164d210` (128/128 groups, independent auditor PASS).
Decision: STOP. The phase is closed under the brief §11 stopping rule; the
locked validation was not run.** [The brief](phase_two_brief.md) §11 requires
this document from the 128-group pilot: keep/stop, intervention sensitivity,
edit eligibility, final operating choices, projected validation cost, and any
deviations from the brief. Values below are copied from the run's
`completion.json`, `summary.json` and `manifest.json`; the machine-readable
counterpart is the decision section of the run's `report.md`. No field was
filled from a partial run.

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
| Pilot run directory | `research/open_weight_lingua/runs/pilot-20260921T235825Z-6164d210` |
| Pilot `manifest.json` sha256 | `cb3ec24ff3187813eb0f2a25ee8397ec7bd94e5b6a8b615588c5615df38b6176` |
| Calibration run directory | `research/open_weight_lingua/runs/calibration-20260921T235017Z-fd111b21` |
| `baseline_fit_identity` (sha256 content hash of the P4 fit) | `37af38fff3df81e016279582b20f179040196c23a84e24d2c9e8d2f8848fec79` |
| Code and protocol hashes (from the manifest) | `source_file_sha256` over 22 files in the pilot `manifest.json`; manifest `git_head` `a3b4b4903ad3a9def4fe3e96faed7723c0ed749d` |
| Frozen edit-rule version and sha256 | `1.0.0` / `e1560d7ec22a2cc5cafda0a46dd05d54d611b7fbf45b3f6b7f6ab36e5bc41fb3` |
| Bootstrap configuration | 3,000 whole-group resamples, fixed seed 203100 (as planned; recorded in each statistic) |
| Date and operator | 2026-09-21; local NVIDIA GB10 run from the `phase-two-milestone-two` worktree (git user `seanm27lol`) |

All target forwards ran under kernel-shape pinning (`TARGET_BUCKET = 128`,
`target_padding_policy` in the manifest; see §11 deviations).

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
validation time. (Moot: the decision is STOP and validation was not opened.)

## 3. Gate 1 — task usability and intervention sensitivity

| Quantity | Value | Threshold | Met? |
|---|---|---|---|
| P0 exact-answer accuracy (intention-to-test: every planned variant, failures count against) | 0.8105 (415/512) | ≥ 0.80 | **Met** |
| Raw-donor effect on the intended measurement (absolute answer/log-probability shift) | accuracy 0.7852 (Δ −2.5 pp vs P0); mean next-token KL 0.173; 39/512 generations differ from P0 | Sensitivity shown, per the gate-1 wording | **Met** (jointly with P5) |
| P5 norm-matched random-direction effect | accuracy 0.0059 (3/512); mean KL 11.75; 508/512 generations differ from P0 | Diagnostic, interpreted together with the donor | Present |

Narrative: this site can affect the intended measurement. A norm-matched
random direction destroys the behavior almost completely (508/512 generations
changed, accuracy 0.6%), while the raw counterfactual donor shifts it modestly
but non-trivially (KL 0.173, 39 changed generations, −2.5 pp accuracy). The
donor effect is reported as an absolute effect; no ratio-style "fraction
recovered" is computed. P0 0.8105 just clears the 0.80 usability floor; it is
recorded as measured, without drama.

## 4. Gate 2 — limited behavioral preservation

| Quantity | Point estimate | One-sided 95% estimate | Threshold | Met? |
|---|---|---|---|---|
| P2 accuracy loss versus P0 | 34.4 pp | Upper: 41.4 pp | Upper ≤ 5 percentage points | **NOT met** |
| P2 − P3 correct-answer log probability | +4.60 | Lower: +3.80 | Lower > 0 | **Met** |

P4 (no-more-than-budget PCA) versus P2, descriptive comparison only: P4
accuracy 0.8027 (411/512), accuracy-loss point 0.78 pp (upper 3.13 pp), mean
KL 0.0157, P0-agreement 0.967 — against P2 accuracy 0.5645, KL 2.71,
P0-agreement 0.615. Read per brief §12 strictly as "generic reconstruction is
competitive under the stated budget"; it is **not** evidence that language is
useless for interpretability in general, and it is not presented as such.
Calibration-median-norm diagnostic, exposing reliance on the retained norm
side channel: accuracy 0.5605 (287/512), mean KL 2.72 — statistically
indistinguishable from P2 with the retained original norm (0.5645, KL 2.71),
so P2's residual accuracy does not rely on the four-byte retained-norm
channel.

## 5. Gate 3 — edit eligibility and coverage

| Quantity | Value | Threshold | Met? |
|---|---|---|---|
| Groups eligible under the frozen edit rule | 0 / 128 | ≥ 32 | **NOT met** |

Coverage and exclusion-reason distribution over all 128 groups: `absent` 128,
`ambiguous` 0, `description_unavailable` 0 (frozen rule v1.0.0, sha256
`e1560d7e…`). Per-variable receiver parses: `x` absent 127 / eligible 1; `y`
absent 128; group-level eligibility requires the affected variable's explicit
current-value statement, which no group received. Recorded agreement between
the original explicit statement and the reference program — a diagnostic that
never gates an edit: 255 of 256 receiver variable-slots absent (no statement
to agree with); the single eligible parse (pilot-0077-A-y, variable x) stated
10 against reference 6, i.e. `agrees: false` — recorded, and it never gated
anything.

## 6. Secondary edit evidence (only if gate 3 is met)

Gate 3 failed (0/128 eligible), so this section stays empty: no edited or
wrong-variable condition was attempted (0 attempted variants for both), and
**the edit hypothesis is marked UNTESTED**. Zero coverage means limited
explicit-variable coverage at this task and site under the frozen rule — it
does not show that descriptions lack editable semantics in general, and no
manual ground-truth description is presented as discovered semantics.

## 7. Per-condition failure accounting (intention-to-test)

All denominators are the 512 planned prompt variants; missing generations
count against accuracy; failed reconstructions never manufacture finite KL
values.

| Condition | Successful | Failed | Skipped | Uneditable |
|---|---|---|---|---|
| P0 unmodified | 512 | 0 | 0 | n/a |
| P1 reinsert original | 512 | 0 | 0 | n/a |
| P2 own description | 512 | 0 | 0 | n/a |
| P3 other group's description | 512 | 0 | 0 | n/a |
| P4 PCA baseline | 512 | 0 | 0 | n/a |
| P5 random direction | 512 | 0 | 0 | n/a |
| Raw donor | 512 | 0 | 0 | n/a |
| Calibration-median-norm diagnostic | 512 | 0 | 0 | n/a |
| Edited reconstruction | 0 | 0 | 512 (not attempted — no eligible group) | 512 variants (128 groups) |
| Wrong-variable edit | 0 | 0 | 512 (not attempted — no eligible group) | 512 variants (128 groups) |

Gate evidence: every identity gate passed bitwise (repeat/noop/raw_restore,
all 512 variants), the greedy identity backstop recorded `exact` for 512/512
variants in both the identity and behavior stages (no `drift_diverged`), and
every `suffix_drift_relative` measurement was exactly 0.0 under kernel-shape
pinning.

## 8. Resources and projected validation cost

| Quantity | Value |
|---|---|
| Measured pilot wall time, model calls, allocation peaks | 11,980.8 s (≈3.33 h) through bundle (setup 155.0 s); stages: target identity 856.9 s, AV 7,892.5 s, AR 106.0 s, behavior 3,084.3 s; forward calls 112,032 (target identity 5,016, AV 72,207, AR 512, target behavior 34,297); GPU peak 15.91 GB allocated / 18.17 GB reserved |
| Projected 512-group validation wall time (from measured pilot throughput) | 47,759.3 s, plus 1,494.2 s projected 256-group calibration: 49,253.5 s total (≈13.7 h); 448,128 projected forward calls; linear per-group scaling — an estimate, not a duration promise |
| Eight-hour budget check | **Above budget**: 49,253.5 s > 28,800 s (`within_budget: false`) |

The projection exceeds the eight-hour budget; per brief §9 this alone would
require stopping for a scope decision before opening validation, without
shrinking samples or conditions midway. Validation was never auto-started,
and the STOP decision makes the question moot.

## 9. Final operating choices for validation (locked here before it opens)

Validation was not opened (STOP). The values below are the as-recorded
operating choices under which the pilot ran — locked, and not editable after
the fact.

| Choice | Locked value |
|---|---|
| P4 rank rule | As applied: `rank = min(fitted_rank, floor(text_bytes/2))` with `fitted_rank = 1023`, float16 coefficients, no-more-than-budget; fit identity `37af38ff…` |
| Norm policy | Retained per-example original float32 norm (4 bytes) for P2/P3/P4/P5/donor patches; frozen calibration median norm 98.8137 as diagnostic only |
| Logit tolerances and greedy-answer agreement | Identity/P1 logit gates `atol=1e-5, rtol=1e-5` (held bitwise all run); greedy identity as an exact/`drift_diverged` recorded backstop under kernel-shape pinning (`TARGET_BUCKET = 128`); `SUFFIX_DRIFT_BOUND = 1e-1` frozen backstop, never re-fitted (manifest `target_padding_policy`, `greedy_identity_gate`) |
| Canonical answer formatting and description budget | Frozen canonical integer tokenization plus EOS, appended without retokenizing the prefix; one greedy description per activation, 200-new-token ceiling |
| Numerical editing criteria | NOT FINALIZED — gate 3 unmet (0/128 eligible); the edit hypothesis is marked untested per the frozen rule |

## 10. Decision

**STOP.**

Rule derivation — gate 1 **met** (P0 0.8105 ≥ 0.80; intervention sensitivity
present); gate 2 **not met** (P2 accuracy-loss one-sided 95% upper 41.4 pp,
above the 5 pp limit; the P2−P3 log-probability lower estimate +3.80 > 0 was
met); gate 3 **not met** (0/128 edit-eligible groups against the 32 floor).
The frozen decision rule — any unmet criterion produces stop — yields STOP
with unmet criteria `p2_accuracy_loss` and `edit_eligible_groups`.
Independently, the projected validation cost (≈13.7 h) exceeds the
eight-hour budget. Per the brief's bounded stopping rule, the phase closes
with this documented negative/bounded result: the locked validation is not
run, and no new checkpoint, site, layer, task, or architecture is tried.

## 11. Deviations from the brief

Two pre-pilot repairs, both documented with justification in
[reports/milestone_two.md](../reports/milestone_two.md) and both frozen
**before** any pilot/validation outcome was inspected:

1. **Suffix-causality gate repair** (after the first released-model smoke,
   `smoke-20260921T185833Z-c264a184`, failed teacher-forced scoring). The
   original `allclose(1e-5)` comparison of the prefix-site vector across
   sequence lengths is unattainable under BF16 cuBLAS kernel re-selection on
   sm_121. It was replaced with a tolerance-free same-length dummy-suffix
   bitwise gate plus the frozen `SUFFIX_DRIFT_BOUND = 1e-1`, justified from
   smoke measurements (max 3.09e-2, safety factor ≈3) per brief §4.
2. **Kernel-shape pinning** (after the first pilot attempt,
   `pilot-20260921T222056Z-69535f26`, failed the identity gate on variant
   pilot-0000-A-x with no conditions executed). Measured unpadded prefix-site
   drift on that variant was 2.58e-1 — a prompt-conditioned heavy tail
   exceeding the frozen bound (fp32 collapse ~3.2e-6 confirmed numerical
   noise, not a leak; the smoke-based bound justification was falsified for
   the pilot distribution and is preserved as such in the claim ledger).
   Rather than re-fit a bound to pilot data, every target forward was pinned
   to the fixed 128-token right-padded bucket (`TARGET_BUCKET = 128`;
   masked pads contribute exactly zero, bitwise-proven; same-shape forwards
   are bitwise deterministic on this backend), eliminating the drift class by
   construction. The bound stays frozen as an untouched backstop, and the
   greedy identity check became a recorded exact/`drift_diverged` backstop.
   The affected pilot was then rerun, visibly versioned (this document's
   pinned run), per brief §11.

Consequences recorded for the reader: the unpadded smoke
(`smoke-20260921T211151Z-a4e4a038`) and calibration
(`calibration-20260921T221636Z-63a6be98`) runs are superseded by the pinned
re-baselines (`smoke-20260921T233021Z-cbe4057e`,
`calibration-20260921T235017Z-fd111b21`); the first pilot attempt is retained
as an explicitly failed run, and its degenerate decision fields are not
outcomes. No other deviations from the brief.
