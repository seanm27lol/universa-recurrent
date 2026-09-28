# Phase two: what the pilot actually established

**Status: complete and closed, decision STOP, pilot run 2026-09-21.** This page
summarizes the completed open-model pilot and its audits. It introduces no new
model, benchmark or required GPU run. The locked 512-group validation was
**not run** and remains unimplemented in the runner, per the brief's stopping
rule and an over-budget cost projection.

> A language description of one Qwen2.5-7B-Instruct residual-stream vector,
> reconstructed through the released NLA verbalizer/reconstructor pair, did
> **not** preserve behavior within the frozen five-percentage-point
> accuracy-loss bound on this task. The description-derived direction was
> clearly on-task (it beat a wrong-description control), and a generic
> calibration-fitted PCA direction nearly matched the unmodified model under
> the same byte budget. The frozen text-edit rule found zero eligible groups,
> so the edit hypothesis is **untested**, not refuted.

[Technical report](phase_two_technical_report.md) ·
[Claim ledger](claims.md) ·
[Pilot decision record](../research/open_weight_lingua/protocols/pilot_decision.md) ·
[Governing brief](../research/open_weight_lingua/protocols/phase_two_brief.md) ·
[Implementation README](../research/open_weight_lingua/README.md)

## Start with the concrete problem

A frozen Qwen2.5-7B-Instruct reads:

```text
x = 3
y = 8
x = x + 2
What is x? Reply with only the integer.
```

The answer is 5. Before the model answers, one internal vector `h` is captured
at the residual-stream output of block index 20, at the last non-padding token
including the assistant-generation prefix. A released activation verbalizer
(AV) describes that vector in English, producing a description `c`; its paired
reconstructor (AR) turns `c` back into a direction `r`; the installed
replacement is `h' = n r / ||r||₂`, where `n = ||h||₂` is a retained four-byte
side channel, **not** something recovered from text. The remaining computation
then runs with the original prompt and all other token vectors untouched.

A paired counterfactual program changes the first assignment to `x = 4`: the
answer for `x` changes to 6 while the answer for `y` stays 8. Each problem
group contributes four prompt variants (source/counterfactual × query x/y).

The headline questions were: does the description-route reconstruction preserve
the measured behavior, and does a minimal edit to an explicit current-value
statement in the description produce a specific, reproducible behavioral
change? The pilot answered the first question **negatively at the frozen
threshold** and could not open the second: no description contained an
editable current-value statement under the frozen rule.

## 1. Three pinned runs, all auditor PASS

| Run | Stage | Result |
|---|---|---|
| `smoke-20260921T233021Z-cbe4057e` | 8-group engineering smoke | COMPLETE: 8/8 groups, identity gates bitwise, drift exactly 0.0 over 336 samples; engineering gate only, not a scientific measurement |
| `calibration-20260921T235017Z-fd111b21` | 256-group numerical calibration | COMPLETE: 1,024/1,024 extractions; P4 PCA fit identity `37af38ff…`; frozen median norm 98.8137 |
| `pilot-20260921T235825Z-6164d210` | 128-group bounded pilot | COMPLETE: 128/128 groups, 0 failed/skipped; decision **STOP**; manifest sha256 `cb3ec24f…` |

All three ran on the pinned kernel-shape code (git head
`a3b4b4903ad3a9def4fe3e96faed7723c0ed749d`, recorded in each manifest) and pass
the independent auditor, which replays saved counts and KL without loading
models. The auditor does not authenticate execution; see
[evidence and audit scope](#evidence-and-audit-scope). The pilot plan hash
`ff060012…` matches the calibration run's, and the pilot consumed the pinned
calibration fit by content hash.

## 2. Per-condition metrics: preservation failed the frozen bound

All denominators are the **512 planned prompt variants** (128 groups × 4),
intention-to-test: missing generations count against accuracy, and failed
reconstructions never manufacture finite KL values. Every condition completed
512/512 valid variants with zero failures.

| Condition | Exact answers /512 | Accuracy | Mean valid next-token KL | Agreement with P0 |
|---|---:|---:|---:|---:|
| P0 unmodified | 415 | 0.8105 | 0.0 | 1.000 |
| P1 original reinserted (adapter gate) | 415 | 0.8105 | 0.0 | 1.000 |
| P2 own description → AR + original norm | 289 | 0.5645 | 2.71 | 0.615 |
| P3 another group's description + receiver norm | 137 | 0.2676 | 7.38 | 0.299 |
| P4 PCA baseline (no-more-than-budget) | 411 | 0.8027 | 0.0157 | 0.967 |
| P5 norm-matched random direction | 3 | 0.0059 | 11.75 | 0.008 |
| Raw donor direction from counterfactual B | 402 | 0.7852 | 0.173 | 0.924 |
| Calibration-median-norm diagnostic | 287 | 0.5605 | 2.72 | 0.611 |

P1 is the mandatory adapter correctness gate and was **identical to P0**: every
identity gate (repeat/no-op/raw-restore) held bitwise on all 512 variants, the
greedy identity backstop recorded `exact` 512/512 in both the identity and
behavior stages, and every `suffix_drift_relative` measurement was exactly 0.0
under kernel-shape pinning.

Three readings, each with its boundary:

- **Specificity control passed.** P2 beats P3 decisively (accuracy 0.5645 vs
  0.2676; frozen log-probability contrast below). The description-derived
  direction carries on-task information; it is not an arbitrary direction.
- **Preservation failed.** P2 answers 289 of 512 variants exactly against
  P0's 415, with mean KL 2.71. The frozen 5-percentage-point loss bound is not
  close to met.
- **The generic baseline nearly matched P0.** P4 accuracy 0.8027 vs P0 0.8105,
  KL 0.0157. Per the brief's claim table this may be read as "generic
  reconstruction is competitive under the stated budget" — **not** as
  "language is useless for interpretability." P4 is *no-more-than-budget*
  under a declared byte rule, not an exactly matched or optimal compression
  baseline.

## 3. Frozen decision statistics and the STOP derivation

Uncertainty uses the frozen whole-group bootstrap: 3,000 resamples, fixed seed
203100; prompt variants within one group are not independent samples. A group
counts as correct only when all four of its variants answer exactly; the
P2−P3 contrast uses per-group mean correct-answer log probability. These
statistics were recomputed by the independent auditor from saved per-group
records.

| Frozen statistic | Point estimate | One-sided 95% estimate | Frozen threshold | Met? |
|---|---:|---:|---|:--:|
| P2 accuracy loss versus P0 | 34.4 pp | upper 41.4 pp | upper ≤ 5 pp | **no** |
| P2 − P3 correct-answer log probability | +4.60 | lower +3.80 | lower > 0 | yes |
| P4 accuracy loss versus P0 (descriptive only) | 0.78 pp | upper 3.13 pp | none | n/a |

The frozen decision rule evaluates five criteria; **any unmet criterion
produces stop**:

| Frozen criterion | Observed value | Met? |
|---|---|:--:|
| P0 unmodified accuracy ≥ 0.80 | 0.8105 (415/512) | yes |
| Intervention sensitivity present at this site | P5 changed 508/512 generations (KL 11.75); raw donor KL 0.173, 39/512 changed, −2.5 pp | yes |
| P2 loss one-sided 95% upper ≤ 5 pp | 41.4 pp | **no** |
| P2 − P3 log-probability lower estimate > 0 | +3.80 | yes |
| Edit-eligible groups ≥ 32 of 128 | 0 | **no** |

**Decision: STOP** — two of five criteria unmet (`p2_accuracy_loss`,
`edit_eligible_groups`). Independently, the projected validation cost exceeds
the eight-hour budget (§5). Per the brief's bounded stopping rule, the locked
validation was not opened, and no new checkpoint, site, layer, task, or
architecture follows the negative pilot. P0's 0.8105 just clears the 0.80
usability floor; it is recorded as measured, without drama.

## 4. Edit coverage was zero: the edit hypothesis is UNTESTED

Under the frozen edit rule (version 1.0.0, sha256 `e1560d7e…`; three
current-value statement forms, canonical values 0–19, value-span-only edits):

| Quantity | Value |
|---|---:|
| Groups eligible under the frozen rule | 0 / 128 (floor: 32) |
| Exclusion reasons over all 128 groups | `absent` 128, `ambiguous` 0, `description_unavailable` 0 |
| Per-variable receiver parses | `x`: absent 127, eligible 1; `y`: absent 128 |
| Edited / wrong-variable conditions | 0 attempted variants; 512 skipped each; 512 uneditable (128 groups) |

The single eligible receiver parse (pilot-0077-A-y, variable `x`) stated 10
against a reference value of 6 — recorded agreement `false`. That agreement
diagnostic never gates an edit, and it gated nothing here. Because no group
was eligible, no edited or wrong-variable reconstruction was attempted and the
editing-effect statistic is marked **untested** ("no edit-eligible group has
complete unedited and edited measurements").

Zero coverage means **limited explicit-variable coverage at this task and site
under the frozen rule**. It does not show that descriptions lack editable
semantics in general, and no manual ground-truth description is presented as
discovered semantics. The targeted-edit question is preserved as an open,
explicitly untested hypothesis — not a silent omission and not a refutation.

## 5. The norm side channel and the P4 comparison

The calibration-median-norm diagnostic replaces the retained per-example norm
with the frozen calibration median (98.8137). It matches P2 almost exactly
(0.5605 vs 0.5645 accuracy; KL 2.72 vs 2.71), so **P2's residual accuracy does
not rely on the four-byte retained-norm channel**. The norm is still counted
as a side channel; nothing here is text-only recovery of the activation.

P4 is a PCA baseline fitted only on pooled, unlabeled calibration unit
directions: rank `min(fitted_rank, floor(text_bytes/2))` with fitted rank 1,023,
float16 coefficients, shared mean/basis bytes reported separately, and the
receiver norm restored at patch time exactly as in P2. Its near-P0 preservation
against P2's loss is a descriptive comparison under the stated budget, not a
verdict on language as an interpretability medium.

## 6. Costs and the validation budget projection

Measured on the DGX Spark (NVIDIA GB10, unified memory), pilot run
`pilot-20260921T235825Z-6164d210`:

| Quantity | Value |
|---|---:|
| Pilot wall time through bundle | 11,980.8 s (≈3.33 h); setup/CPU tests 155.0 s measured separately |
| Stage times | target identity 856.9 s; AV 7,892.5 s; AR 106.0 s; target behavior 3,084.3 s |
| Forward calls | 112,032 total: target identity 5,016, AV 72,207, AR 512, target behavior 34,297 |
| GPU memory peaks | 15.91 GB allocated / 18.17 GB reserved |
| Calibration run wall time | 299.1 s (1,024 target calls) |
| Pinned smoke wall time | 995.1 s (6,739 calls) |

The projected locked-validation cost, scaled linearly from measured pilot
per-group stage times: 512-group validation 47,759.3 s plus 256-group
calibration 1,494.2 s = **49,253.5 s ≈ 13.7 h**, against the 28,800 s
eight-hour budget (`within_budget: false`; 448,128 projected forward calls).
This is an estimate for a scope decision, not a duration promise. Per the
brief, an over-budget projection alone requires stopping before opening
validation, without shrinking samples or conditions midway.

## 7. Superseded runs and the failed first attempt, preserved explicitly

Two pre-pilot numerics repairs are part of this phase's record, and the runs
they superseded are retained rather than deleted:

| Run | Status | Role in the record |
|---|---|---|
| `smoke-20260921T185833Z-c264a184` (main checkout) | FAILED in the behavior stage | First released-model smoke; established that the `allclose(1e-5)` suffix-causality check is unattainable under BF16 kernel re-selection, motivating the same-length bitwise gate and the frozen 1e-1 drift bound |
| `smoke-20260921T211151Z-a4e4a038` | COMPLETE (unpadded), auditor PASS | Repaired-gate smoke; valid measurement of the unpinned regime, **superseded** as a pilot input by the pinned re-baseline |
| `calibration-20260921T221636Z-63a6be98` | COMPLETE (unpadded) | Unpinned calibration; same plan hash, **superseded** by the pinned fit `37af38ff…` |
| `pilot-20260921T222056Z-69535f26` | FAILED at the identity gate | First pilot attempt; died on variant pilot-0000-A-x before any condition executed (1 failed group, 127 skipped). Its decision/summary fields are degenerate artifacts of a dead run, **not outcomes** |

The first attempt's diagnosis — cross-length BF16 drift of 0.258 at the
divergence step, a prompt-conditioned heavy tail beyond the frozen 1e-1 bound
whose smoke-based justification it falsified — led to kernel-shape pinning
(`TARGET_BUCKET = 128`) instead of re-fitting the bound to pilot data. The
bound stays frozen as an untouched backstop. The full mechanism, measurements
and justification are in the
[technical report](phase_two_technical_report.md) and
[milestone two report](../research/open_weight_lingua/reports/milestone_two.md);
the falsified bound is preserved in the [claim ledger](claims.md).

## Evidence and audit scope

Each run directory holds an immutable `manifest.json` (written before
inference; code/protocol/model hashes, frozen settings, every input identity),
`results.json` (every planned variant), `summary.json`/`report.md`
(all-variant denominators; the pilot adds the decision section),
`compatibility.json`, `completion.json`, `inventory.json`, a reports-only
`reports.zip`, and local-only `raw/*.safetensors` plus `wall_clock.json`. The
pilot manifest sha256 is
`cb3ec24ff3187813eb0f2a25ee8397ec7bd94e5b6a8b615588c5615df38b6176`; the P4 fit
identity is `37af38fff3df81e016279582b20f179040196c23a84e24d2c9e8d2f8848fec79`;
the plan hash `ff060012…` binds every group identity.

The independent auditor recomputes saved counts and KL and replays the
bootstrap without loading models or invoking the producer. It does **not**
authenticate execution, regenerate teacher-forced probabilities from raw
logits (full-vocabulary distributions stay local; the reports-only ZIP cannot
independently replay KL), or verify model weights beyond their pinned hashes.
Hashes identify artifacts; they do not authenticate remote execution or prove
when the protocol was registered. The numbers on this page were cross-checked
against the pilot's `summary.json`, `completion.json`, `compatibility.json`,
`wall_clock.json` and `report.md`, and the calibration run's
`completion.json`/`summary.json`.

**This phase is closed with a documented bounded negative result.**
Preservation failed the frozen bound, the edit hypothesis is untested at zero
coverage, and validation was never opened. A future research phase would need
its own task, budget and stopping rule; nothing here triggers an automatic
follow-on run.
