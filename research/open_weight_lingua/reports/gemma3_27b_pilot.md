# Gemma-3-27B pilot outcome and family closeout — 2026-09-26/27

For `x = 3; y = 8; x = x + 2`, the reference answer for `x` is 5. This page
closes the third family run of the activation-description assay: the 27B stack
(62 blocks, width 5376) with the released kitft/nla-gemma3-27b-L41-av/ar pair
(extraction block 41, a full-attention block). **Decision: STOP** — frozen
thresholds, no validation auto-started. The 27B AV is float32-native and was
served BF16 under a lock-declared cast (see the caveat below); read the
floor-effect section before quoting any accuracy number here.

## Pinned runs (all auditor PASS, re-verified 2026-09-27)

| Stage | Run directory | Manifest sha256 (prefix) | Auditor |
|---|---|---|---|
| Smoke (8 groups / 32 variants) | `smoke-20260923T231748Z-068d59ba` | `4c66684004e93796…` | PASS |
| Calibration (256 groups / 1,024 variants) | `calibration-20260926T183645Z-c3374c27` | `a18c40914e9480d0…` | PASS (fit identity `784594e89f6c315b…`, frozen median norm 41,629.3) |
| Pilot (128 groups / 512 variants) | `pilot-20260926T185550Z-d085d53c` | `89583fbe53ca1b0a…` | PASS (decision and statistics reproduce from saved results) |

Model lock `configs/model-lock-gemma3-27b.json` (sha256 `ca078e528a9daaeb…`),
plan hash `ff060012d35d92c4…` (the same deterministic split plan as the Qwen
and 12B pilots). Three earlier attempts are preserved explicitly:
`smoke-20260923T195955Z-e90ad8b5` and `smoke-20260923T201748Z-1687a778`
(kernel-OOM-killed at AV load before the streaming loader existed) and
`smoke-20260923T220025Z-762d65f6` (COMPLETE_WITH_FAILURES; the loader's
missing generation-config merge, since fixed and fixture-pinned).

**Provenance, carried forward:** the official google/gemma-3-27b-it is
gated-manual; the target pins the public unsloth mirror @ `7a5a3053…`. All
twelve weight shards and both tokenizer blobs carry LFS sha256 identical to
the gated official's API record (byte-identical content); the divergent small
files are the mirror's own pinned bytes with the official revision recorded
under `official_source`. Gemma Terms of Use apply to the user regardless of
download source; the HF gate is an access mechanism, not the license itself.

**The AV dtype caveat:** the released 27B AV is float32-native (108.08 GB) and
was served BF16 under the lock's justified `serving_dtype` declaration — the
pinned recipe's own local defaults cast to bf16 the same way
(`load_embedding_only(checkpoint_dir, dtype=torch.bfloat16)` and
`NLACritic(dtype=torch.bfloat16)`), and its SGLang launch sets no `--dtype`.
No local fp32 A/B comparison is possible on this machine, so the cast's
contribution to any behavioral number is unauditable locally and is attached
to every result on this page.

## Decision criteria (frozen thresholds; met/unmet)

| Criterion | Value | Rule | Met |
|---|---:|:---|:---|
| Unmodified exact-answer accuracy | 0.2539 | ≥ 0.80 | **no** |
| Donor/perturbation sensitivity | present | required | yes |
| P2 accuracy loss, one-sided 95% upper | 2.344 pp | ≤ 5 pp | yes |
| P2 − P3 correct-answer log prob, one-sided 95% lower | +1.844 | > 0 | yes |
| Edit-eligible groups | 0 | ≥ 32 | **no** |

Recommendation: **stop**; unmet criteria `unmodified_accuracy` and
`edit_eligible_groups`. Sensitivity detail: P5 changes 510 of 512 greedy
generations (mean KL 13.41); donor changes 39 (KL 0.059). The projected
512-group validation cost is 181,295 s ≈ 50.4 h against the 28,800 s budget
(`within_budget: false`) — an independent second stop reason.

## Conditions (512 variants each, intention-to-test)

| Condition | Exact answers | Accuracy | P0 agreement | Mean valid next-token KL |
|---|---:|---:|---:|---:|
| P0 (unmodified) | 130 | 0.2539 | 1.000 | 0.0 |
| P1 (pinned original) | 130 | 0.2539 | 1.000 | 0.0 |
| P2 (description-derived) | 116 | 0.2266 | 0.816 | 0.216 |
| P3 (shuffled description) | 114 | 0.2227 | 0.574 | 2.809 |
| P4 (calibration PCA baseline) | 134 | 0.2617 | 0.939 | 0.0042 |
| P5 (random direction) | 2 | 0.0039 | 0.004 | 13.408 |
| donor (raw within-group donor) | 136 | 0.2656 | 0.924 | 0.059 |
| calibration_median_norm | 118 | 0.2305 | 0.805 | 0.213 |

`edited` / `wrong_variable_edit`: 0 attempted (all 128 exclusions `absent`
under the unchanged frozen rule v1.0.0). Edit hypothesis: **untested**.

Gates: identity gates bitwise on all 512 variants; greedy identity backstop
exact 512/512 in both stages; all 6,144 suffix-drift measurements exactly 0.0
under kernel-shape pinning; the P0/P1 raw-restoration gate held on reload.
AR round-trip cosine on the pilot: 0.9656–0.9897 (median 0.9823, n = 512).
Frozen statistics: whole-group bootstrap, 3,000 resamples, seed 203100; P2
accuracy-loss point estimate 0.0 pp with one-sided upper 2.34 pp; P2 − P3
contrast point +2.29 with one-sided lower +1.84.

## The floor-effect paragraph (required reading)

At P0 = 25%, the accuracy channel is near its floor and weakly informative.
P2 and P3 land at 0.227 and 0.223 accuracy — a 0.4-point gap that is expected
under floor effects and establishes nothing by itself, and the frozen P2
accuracy-loss upper bound of 2.34 pp is correspondingly easy to satisfy. The
channels that still carry signal are the ones that do not need a correct
answer: P2's agreement with P0 is 0.816 against P3's 0.574, next-token KL is
0.216 against 2.81, and the P2 − P3 correct-answer log-probability contrast
is positive with one-sided lower +1.84. So: the description-derived direction
tracks the model's behavior including its errors, while a shuffled
description does not — and simultaneously, none of this shows the task is
usable (it is not, by the frozen floor) or that the language route would
preserve behavior on a task that is (the one usable task we have failed that
criterion decisively). Neither reading is available to upgrade.

## Costs (measured, not projected)

Pilot runner wall clock 44,681.4 s ≈ 12.4 h (+ 112 s setup/CPU-suite).
Stages: AV 29,248.0 s (76,124 forwards), target identity 2,112.8 s (5,728),
AR 335.9 s (512), target behavior 12,676.8 s (36,498). The 27B AV stage is
~2.5× the 12B's per-call cost at the same call structure.

## Three-family comparison

Same assay, same frozen thresholds, same deterministic split plan, three
closed pilots:

| | Qwen2.5-7B, block 20/28 | Gemma-3-12B, block 32/48 | Gemma-3-27B, block 41/62 |
|---|---:|---:|---:|
| P0 accuracy | 0.811 | 0.520 | 0.254 |
| P2 accuracy-loss upper (pp) | 41.4 | 3.91 | 2.34 |
| P2 agreement with P0 | 0.615 | 0.854 | 0.816 |
| P2 − P3 log-prob lower | +3.80 | +2.61 | +1.84 |
| Edit-eligible groups | 0/128 | 0/128 | 0/128 |
| Decision | stop | stop | stop |
| Task usable (P0 ≥ 0.80) | yes | no | no (near floor) |
| AV round-trip cosine (median) | 0.841 | 0.990 | 0.982 |

Read together, the three closed pilots bracket the language route three ways,
and the pattern must not be misread. On Qwen2.5-7B the task was usable — P0
0.811, above the 0.80 floor — and the preservation criterion failed
decisively (P2 accuracy-loss upper 41.4 pp against the 5 pp bound). On both
Gemma pairs the preservation criterion passes (loss upper 3.91 pp and 2.34
pp, P2 − P3 lower +2.61 and +1.84), but on tasks that are not usable (P0
0.520 and 0.254) — and on 27B the accuracy channel sits near the floor, where
P2 and P3 nearly tie on accuracy as floor effects predict, while the
informative channels (P2 agreement 0.816 and KL 0.216 against P3's 0.574 and
2.81, plus the positive log-probability contrast) still separate description
content from noise. Two upgrades are prohibited: the Gemma passes do not show
that the language route works — the only family with a usable task failed
preservation decisively, and the floors make the passes weak assay evidence —
and the pass margin does not show that larger models preserve better, because
floor effects and the 27B AV's BF16 serving cast (unauditable locally) forbid
that comparison. The edit interface is absent on all three families (0/128
each): the edit hypothesis is untested everywhere. Whether site depth, model
size, family, or the NLA pair's training drives the Gemma pairs' preservation
strength is speculation for a future phase, not a finding; no validation ran
on any family and all three phases are closed.

## Stopping-rule closure for this family

Per the brief's stopping rule, one pilot was run and the locked validation
stage was never started and remains unimplemented; this family does not get a
second pilot from this runner. The operational contribution on record: the
27B stack runs the full gate suite green (identity, causality, sensitivity
controls) at 12.4 h wall clock with the streaming fp32-cast loader, and the
L41 pair round-trips directions at median cosine 0.982. Both facts are
recorded with their denominators; neither is upgraded.

## Limits

Behavioral measurements at one site on one task family; gates and statistics
are engineering instruments, not semantic evidence. Accuracy is
floor-limited here; the agreement/KL/log-probability channels are the
informative ones and are reported as measured. P4's near-P0 behavior (0.262,
KL 0.0042) is a no-more-than-budget baseline comparison, not an optimality
claim. The AV's BF16 serving cast is unauditable locally and attaches to
every AV-dependent number. The mirror's divergent config files were pinned as
fetched; the official bytes remain unverified pending gated access. Auditor
PASS replays saved counts, KL, the frozen statistics and the decision; it
does not authenticate execution or replay AV/AR generation and target
forwards.
