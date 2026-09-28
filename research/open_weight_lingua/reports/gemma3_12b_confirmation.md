# Gemma-3-12B confirmation pilot under the amended answer instrument — 2026-09-27/28

For `x = 3; y = 8; x = x + 2`, the reference answer for `x` is 5. The closed
Gemma-12B pilot failed the task-usability gate (P0 0.520); the failure taxonomy
then showed the instrument, not the model, at fault: the mirror target answers
`digits + "\n" + <end_of_turn>` and the frozen no-strip integer metric rejected
228 of 512 variants, 221 of them with the correct value. This page records the
confirmation run under the frozen amendment. **The closed pilot
`pilot-20260923T043613Z-f7e71d7b` keeps its frozen outcome and STOP decision;
nothing here relabels it.** The amendment changed the comparison instrument,
not the generations: all 512 P0 generations in the confirmation run are
bitwise-identical to the closed pilot's, and the text-independent
log-probability contrast is digit-identical (+2.6073…).

## Chain of custody for the instrument fix

1. Failure taxonomy over the closed pilot's saved generations
   (`reports/post_hoc_answer_lens.md`): trailing-whitespace rejections dominate
   (228/512); difficulty probes show content accuracy 0.98–1.00 at every level,
   with frozen-metric accuracy *falling* on easier prompts — a format effect.
2. Post-hoc lens preview (same saved evidence, rstrip-only): P0 0.9512 and the
   P2-loss upper moving to 10.16 pp — a predicted flip of both criteria.
3. **Frozen amendment** `protocols/gemma_answer_convention.md` (committed
   `8ec15cb`, before any confirmation forward): under
   `--answer-convention rstrip`, an answer counts exactly correct iff
   `exact_integer(text.rstrip(), expected)`; the manifest records the
   convention; the scoring channel (digits + EOS, unchanged) and the default
   `raw` convention (byte-identical replay of all closed runs, re-audited PASS)
   are preserved.
4. The confirmation pilot below. One false start is preserved:
   `pilot-20260927T201459Z-0f1f9907` died before any model forward when a
   file-valued `--calibration-fit` argument met a pre-existing runner path
   inconsistency (the run record carries the error; relaunched with the
   directory form).

## Run identity

Pilot `pilot-20260927T202455Z-f5ec3892` — manifest sha256
`91aa7162258c292b25e59ea66e9aeb5ad014d28125e65985e3daeef69d266fd1`, `git_head` `8ec15cb4c01d…` (the frozen amendment commit),
`answer_convention: rstrip`, the same lock (`d49ac3ea…`), the same split plan
(`ff060012d35d…`, same 128 groups as the closed pilot), the same calibration
fit (`7e60a59773c7b6bc…` — the convention cannot affect target-only
extraction). **Independent auditor: PASS** (saved counts, KL, frozen
statistics and the decision reproduce). Gates: identity bitwise 512/512,
greedy backstop exact 512/512 in both stages, all 6,144 suffix-drift
measurements exactly 0.0. Runner wall clock 27,524 s ≈ 7.6 h (+266 s setup);
stages: AV 17,125.1 s (same call structure as the closed pilot), identity
2,381.4 s, AR 166.0 s, behavior 7,720.8 s. Projected validation cost 113,902 s
against the 28,800 s budget — independently over budget.

## Decision under the corrected instrument

| Criterion | Value | Rule | Met |
|---|---:|:---|:---|
| Unmodified exact-answer accuracy | 0.9512 (487/512) | ≥ 0.80 | **yes** |
| Donor/perturbation sensitivity | present | required | yes |
| P2 accuracy loss, one-sided 95% upper | 10.156 pp (point 5.47) | ≤ 5 pp | **no** |
| P2 − P3 correct-answer log prob, one-sided 95% lower | +2.607 | > 0 | yes |
| Edit-eligible groups | 0 | ≥ 32 | **no** |

Decision: **STOP** — unmet: `p2_accuracy_loss` and `edit_eligible_groups`.

## Conditions under the amended instrument (512 variants, ITT)

| Condition | Exact answers | Accuracy | P0 agreement | (Closed pilot, raw) |
|---|---:|---:|---:|---:|
| P0 | 487 | 0.9512 | 1.000 | 0.5195 |
| P1 (pinned original) | 487 | 0.9512 | 1.000 | 0.5195 |
| P2 (description-derived) | 478 | 0.9336 | 0.945 | 0.5527 |
| P3 (shuffled description) | 232 | 0.4531 | 0.451 | 0.3223 |
| P4 (calibration PCA baseline) | 488 | 0.9531 | 0.998 | 0.5273 |
| P5 (random direction) | 0 | 0.0000 | 0.000 | 0.0000 |
| donor | 480 | 0.9375 | 0.977 | 0.5312 |
| calibration_median_norm | 478 | 0.9336 | 0.949 | 0.5664 |

Every amended-convention number matches the post-hoc lens preview exactly
(the generations reproduced bitwise, so the frozen bootstrap on lens-corrected
correctness is the same computation). The lens numbers were descriptive
previews; this table is the frozen-convention record.

## What the confirmation establishes — and what it does not

With a working instrument, the Gemma-12B task is usable (P0 0.951) **and** the
preservation criterion fails on it (P2 accuracy-loss upper 10.16 pp > 5 pp):
the description-derived direction still moves behavior on-task (P2 − P3
log-prob lower +2.61 > 0, agreement 0.945 vs 0.451) but loses too much
accuracy against the frozen 5 pp bound. The earlier Gemma "preservation pass"
was a floor effect of the format rejections — with the floor removed, the 12B
family lands where Qwen landed: usable task, preservation failed. The edit
hypothesis remains untested (0/128 coverage; the AV is unchanged). This run
does not vindicate the language route beyond this family, task and site, and
it does not relabel the closed pilot: the closed pilot's frozen outcome stands
because instruments are frozen per phase. Prohibited upgrades from the
three-family ledger carry forward unchanged. The 27B AV's BF16 serving cast
does not touch this run (12B AV is BF16-native).

## Limits

One confirmation pilot under the amended instrument; no validation ran or is
implemented. The amended instrument measures exact answers after stripping
trailing whitespace only; the teacher-forced scoring channel is unchanged and
still scores the newline-free suffix the model does not prefer — recorded, not
reconciled. Auditor PASS replays saved evidence; it does not authenticate
execution.
