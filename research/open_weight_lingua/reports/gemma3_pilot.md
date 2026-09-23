# Gemma-3-12B pilot outcome and family closeout — 2026-09-23

For `x = 3; y = 8; x = x + 2`, the reference answer for `x` is 5. This page
closes the Gemma-3 second-family replication of the Phase Two assay: does a
released NLA description of one block-output vector carry enough direction
content to preserve measured behavior when the vector is replaced? The Qwen
phase answered with a bounded negative on a usable task; the Gemma pilot
answers on an unusable one — read the contrast section before quoting any
number here.

**Decision: STOP** (frozen thresholds, no validation auto-started). The
replication is complete and closed: smoke → calibration → pilot all ran on the
hash-pinned artifacts with the pipeline's gates green end to end, and the
pilot's keep/stop rule returned stop.

## Pinned runs (all auditor PASS, re-verified 2026-09-23)

| Stage | Run directory | Manifest sha256 (prefix) | Auditor |
|---|---|---|---|
| Smoke (8 groups / 32 variants) | `smoke-20260923T034330Z-4bc7e34a` | `ccdf92f6a70937de…` | PASS |
| Calibration (256 groups / 1,024 variants) | `calibration-20260923T042912Z-d0e9499f` | `047046d1d9153d2a…` | PASS (fit identity `7e60a59773c7b6bc…`, frozen median norm 54,441.53) |
| Pilot (128 groups / 512 variants) | `pilot-20260923T043613Z-f7e71d7b` | `f14dd70859683108…` | PASS (decision and statistics reproduce from saved results) |

Model lock `configs/model-lock-gemma3-12b.json` (sha256 `d49ac3ea596895e5…`),
plan hash `ff060012d35d92c4…`. The failed first smoke
(`smoke-20260923T025530Z-1ec606fa`, AV stop-convention failure, all 32
descriptions `truncated`) is preserved as an explicitly failed run; the repair
is recorded in [gemma3_port_readiness.md](gemma3_port_readiness.md) and follows
the pinned upstream recipe.

**Mirror provenance, carried forward:** the official google/gemma-3-12b-it is
gated-manual; the target role pins the public unsloth mirror @ `9478e665…`.
The five weight shards and both tokenizer blobs carry identical LFS sha256 in
both repos' API records (byte-identical content); the four divergent small
config files are the mirror's own pinned bytes, with the official revision
recorded under `official_source` for later reconciliation. Gemma Terms of Use
apply to the user regardless of download source; the HF gate is an access
mechanism, not the license itself.

## Decision criteria (frozen thresholds; met/unmet)

| Criterion | Value | Rule | Met |
|---|---:|:---|:---|
| Unmodified exact-answer accuracy | 0.5195 | ≥ 0.80 | **no** |
| Donor/perturbation sensitivity | present | required | yes |
| P2 accuracy loss, one-sided 95% upper | 3.906 pp | ≤ 5 pp | yes |
| P2 − P3 correct-answer log prob, one-sided 95% lower | +2.607 | > 0 | yes |
| Edit-eligible groups | 0 | ≥ 32 | **no** |

Recommendation: **stop**; unmet criteria `unmodified_accuracy` and
`edit_eligible_groups`. The projected 512-group validation cost is 75,348 s
≈ 20.9 h against the 28,800 s budget (`within_budget: false`) — an independent
second stop reason, as on the Qwen pilot.

## Conditions (512 variants each, intention-to-test)

| Condition | Exact answers | Accuracy | P0 agreement | Mean valid next-token KL |
|---|---:|---:|---:|---:|
| P0 (unmodified) | 266 | 0.5195 | 1.000 | 0.0 |
| P1 (pinned original) | 266 | 0.5195 | 1.000 | 0.0 |
| P2 (description-derived) | 283 | 0.5527 | 0.854 | 0.309 |
| P3 (shuffled description) | 165 | 0.3223 | 0.361 | 3.574 |
| P4 (calibration PCA baseline) | 270 | 0.5273 | 0.957 | 0.0018 |
| P5 (random direction) | 0 | 0.0000 | 0.000 | 24.672 |
| donor (raw within-group donor) | 272 | 0.5312 | 0.924 | 0.085 |
| calibration_median_norm | 290 | 0.5664 | 0.859 | 0.309 |

`edited` / `wrong_variable_edit`: 0 attempted (no eligible group; all 128
exclusions are `absent` under the frozen rule v1.0.0, unchanged from the Qwen
phase). Edit hypothesis: **untested** — zero coverage, as on Qwen.

Gates: identity gates bitwise on all 512 variants; greedy identity backstop
exact 512/512 in both stages; all 6,144 suffix-drift measurements exactly 0.0
under kernel-shape pinning; the P0/P1 raw-restoration gate held on reload.
AR round-trip cosine on the pilot: 0.9792–0.9942 (median 0.9901, n = 512);
smoke: 0.982–0.994 (median 0.988). Site norms 49.0k–63.7k (median 54.6k),
consistent with the √d-inflated scale behind the 80,000 injection scale.

## Costs (measured, not projected)

Pilot runner wall clock 18,472.7 s ≈ 5.13 h (+ 73.0 s setup/CPU-suite).
Stages: AV 11,859.9 s (70,403 forwards), target identity 990.2 s (5,418),
AR 151.1 s (512), target behavior 5,409.5 s (34,368). Calibration and smoke
costs are in their run bundles. The Gemma pilot is heavier per call than the
Qwen one (12B/48 blocks vs 7B/28) and exceeded its linear projection; the
budget math above is what the runner measured, not an estimate made here.

## The cross-family reading

Read together, the two closed pilots bracket the language route rather than
rank it. On Qwen2.5-7B (block 20 of 28) the task was usable — P0 0.811, above
the 0.80 floor — and the preservation criterion failed decisively (P2 accuracy
loss upper 41.4 pp against the 5 pp bound). On Gemma-3-12B (block 32 of 48)
the task is not usable — P0 0.520 — and the preservation criterion passed
(loss upper 3.91 pp, P2 tracking P0 at 0.854 agreement, errors included, with
a positive P2 − P3 log-probability contrast). The Gemma pass is assay evidence
under an unusable-task caveat: the P0 floor exists precisely so that
preservation numbers mean something, and at 52% unmodified accuracy the
measurement is weak evidence — the description-derived direction preserves a
behavior that is itself mostly wrong. It does not establish that language
reconstruction preserves behavior on a usable task, for either family. Whether
the difference lives in the site (L32-of-48 vs L20-of-28), the model size, the
family, or the NLA pair's training is speculation for a future phase, not a
finding; no validation ran on either family and both phases are closed.

## Stopping-rule closure for this family

Per the brief's stopping rule, one pilot was run and the locked validation
stage was never started and remains unimplemented; this family does not get a
second pilot from this runner. What the replication adds to the record: the
pipeline ported cleanly (architecture registry, mirror lock, the one
recipe-faithful decoding repair), the Gemma NLA pair round-trips directions at
median cosine 0.990 (512 reconstructions, range 0.979–0.994) — above the Qwen
pair's pilot median 0.841 (range 0.547–0.877, same assay) — and the
behavioral-preservation gate passes exactly where the task-usability gate
fails. Both facts are recorded with their denominators; neither is upgraded.

## Limits

Behavioral measurements at one site on one task family; gates and statistics
are engineering instruments, not semantic evidence. P4's near-P0 behavior
(0.527, KL 0.0018) is a no-more-than-budget baseline comparison, not an
optimality claim. The mirror's divergent config files were pinned as fetched;
the official bytes remain unverified pending gated access. Auditor PASS
replays saved counts, KL, the frozen statistics and the decision; it does not
authenticate execution or replay AV/AR generation and target forwards.
