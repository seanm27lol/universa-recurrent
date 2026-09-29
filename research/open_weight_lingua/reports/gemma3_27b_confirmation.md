# Gemma-3-27B confirmation pilot under the amended answer instrument — 2026-09-29

For `x = 3; y = 8; x = x + 2`, the reference answer for `x` is 5. The closed
Gemma-27B pilot failed the task-usability gate (P0 0.2539). The post-hoc lens
traced that to the same instrument artifact as on 12B: the mirror target
answers `digits + "\n" + <end_of_turn>`, and the frozen no-strip integer metric
rejects the trailing newline. From the closed pilot's saved generations, the
lens predicted P0 0.9648 and a P2 accuracy-loss upper of 6.25 pp, which fails
the frozen 5 pp bound. This page records the one confirmation run under the
frozen amendment. **The result matched the prediction exactly: P0 0.9648
(494/512), P2-loss upper 6.25 pp, decision STOP. The closed pilot
`pilot-20260926T185550Z-d085d53c` keeps its frozen outcome and STOP decision;
nothing here relabels it.** As on 12B, the amendment changed the comparison
instrument, not the generations: all 512 P0 generations are
bitwise-identical to the closed pilot's (token ids, text and termination),
and so are the generations of every other condition, all 512 AV descriptions
and every saved answer log-probability. The text-independent
log-probability contrast is therefore digit-identical (lower +1.8438…).

## Chain of custody for the instrument fix

1. Failure taxonomy over the closed 12B pilot's saved generations and the
   frozen amendment `protocols/gemma_answer_convention.md` (commit `8ec15cb`):
   under `--answer-convention rstrip`, an answer counts exactly correct iff
   `exact_integer(text.rstrip(), expected)`. The 12B confirmation
   ([gemma3_12b_confirmation.md](gemma3_12b_confirmation.md),
   `pilot-20260927T202455Z-f5ec3892`) reproduced its lens preview exactly.
2. Post-hoc lens preview for 27B (`reports/post_hoc_answer_lens.md`, same saved
   evidence, rstrip-only): P0 0.9648 (494/512), P2 0.9531, P2-loss point
   2.34 pp and one-sided upper 6.25 pp — a predicted STOP on preservation.
   The same prediction was also recomputed independently, while the run was
   still in its first stage and before any of its results existed, by
   replaying the closed pilot's saved rows through the auditor's `summarize`
   with `answer_convention` set to `rstrip`; every number in the
   per-condition table below came out of that replay unchanged.
3. **Frozen addendum** `protocols/gemma27b_confirmation_addendum.md`
   (commit `15f4210`, pushed before any model forward): applies the amendment
   unchanged to 27B with the same lock, split plan, 128 groups, calibration
   fit, thresholds, bootstrap and BF16 AV serving cast, and records the lens
   numbers as the prediction under test, explicitly not as thresholds.
4. Preconditions, checked before the addendum commit: the calibration run
   `calibration-20260926T183645Z-c3374c27` was present with its fit identity
   `784594e89f6c315b…`, tensors sha256 `7aaa32f1…` and sidecar sha256
   `a1948c1f…` (reused, not refit); the closed pilot's run directory was
   present; the lock's hash verification passed on all 200,581,048,536 model
   bytes (the same byte counts per role as the closed pilot). To free unified
   memory, the idle user-level ollama service (44 GB resident) was stopped for
   the run and restarted afterwards, and a concurrent rf-moe training job in
   another session was stopped by that session after a notice from this one.
   Memory available at runner start was 115.7 GB, against 71.5 GB for the
   closed pilot.
5. The confirmation pilot below, launched once, with the calibration fit
   passed as a directory and the default model cache (the closed pilot's
   flags). There was no false start and no rerun.

## Run identity

Pilot `pilot-20260929T034532Z-4e4d655f` — manifest sha256
`9dee790c1901defb56653890bbd691864a6ca8ae233cf4613f4775162b6b5fc8`,
`git_head` `15f4210931e4…` (the frozen addendum commit),
`answer_convention: rstrip`, the same lock (`ca078e52…`), the same split plan
(`ff060012d35d…`; all 512 manifest inputs identical to the closed pilot's),
the same calibration fit (`784594e89f6c315b…`, frozen median norm 41,629.35)
and the same lock-declared BF16 serving cast of the float32-native AV. The
source files match the 12B confirmation byte for byte (`runner.py`
`7ad60ee5…`, `audit.py` `a91fc8a9…`, `metrics.py` `aedc88e7…`); the only
addition to the hashed set is the addendum itself. **Independent auditor:
PASS** at the run's code state `15f4210` (saved counts, KL, frozen statistics
and the decision reproduce); the auditor on `gemma3-port` after `main` was
merged in, which carries main's revised `audit.py`, also returns PASS.
Gates: identity bitwise 512/512 (no-op, raw restore, repeat), greedy
backstop exact 512/512 in both stages, all 6,144 suffix-drift measurements
exactly 0.0; all 512 AV descriptions `ok`; AR round-trip cosine 0.9656–0.9897
(median 0.9823), identical to the closed pilot.

Costs, measured: runner wall clock 47,280.8 s ≈ 13.1 h (+111 s setup and
CPU suite, 164 tests passed). Stages: target identity 2,331.4 s (5,728
forwards), AV 30,925.3 s (76,124), AR 338.5 s (512), target behavior
13,389.5 s (36,498), hash verification 288.3 s. Every stage's forward-call
count equals the closed pilot's, and peak GPU memory is identical to the byte
(56,976,531,968 B allocated, 57,294,192,640 B reserved). The wall clock is
5.8% above the closed pilot's 44,681.4 s. Projected validation cost 192,132 s
against the 28,800 s budget — independently over budget, as before.

## Decision under the corrected instrument

| Criterion | Value | Rule | Met |
|---|---:|:---|:---|
| Unmodified exact-answer accuracy | 0.9648 (494/512) | ≥ 0.80 | **yes** |
| Donor/perturbation sensitivity | present | required | yes |
| P2 accuracy loss, one-sided 95% upper | 6.250 pp (point 2.34) | ≤ 5 pp | **no** |
| P2 − P3 correct-answer log prob, one-sided 95% lower | +1.844 | > 0 | yes |
| Edit-eligible groups | 0 | ≥ 32 | **no** |

Decision: **STOP** — unmet: `p2_accuracy_loss` and `edit_eligible_groups`.
Sensitivity detail: P5 changes 510 of 512 answers relative to P0 (mean KL
13.41); donor changes 8 (KL 0.059). The closed pilot reported 39 for the
donor with the same KL: P0 agreement is computed under the declared
convention, so under `rstrip` a generation that differs from P0 only by
trailing whitespace now agrees with it.

## Prediction against outcome

| Quantity | Predicted (addendum) | Observed | Match |
|---|---:|---:|:---|
| P0 exact-answer accuracy | 0.9648 (494/512) | 0.9648 (494/512) | exact |
| P2 exact-answer accuracy | 0.9531 (488/512) | 0.9531 (488/512) | exact |
| P2 accuracy loss, point | 2.34 pp | 2.34 pp | exact |
| P2 accuracy loss, one-sided upper | 6.25 pp | 6.25 pp | exact |
| Decision | STOP (preservation fails) | STOP (`p2_accuracy_loss`, `edit_eligible_groups`) | exact |

## Conditions under the amended instrument (512 variants, ITT)

| Condition | Exact answers | Accuracy | P0 agreement | Mean valid next-token KL | (Closed pilot, raw) |
|---|---:|---:|---:|---:|---:|
| P0 | 494 | 0.9648 | 1.000 | 0.0 | 0.2539 |
| P1 (pinned original) | 494 | 0.9648 | 1.000 | 0.0 | 0.2539 |
| P2 (description-derived) | 488 | 0.9531 | 0.967 | 0.216 | 0.2266 |
| P3 (shuffled description) | 367 | 0.7168 | 0.725 | 2.809 | 0.2227 |
| P4 (calibration PCA baseline) | 494 | 0.9648 | 0.996 | 0.0042 | 0.2617 |
| P5 (random direction) | 3 | 0.0059 | 0.004 | 13.408 | 0.0039 |
| donor | 492 | 0.9609 | 0.984 | 0.059 | 0.2656 |
| calibration_median_norm | 489 | 0.9551 | 0.969 | 0.213 | 0.2305 |

Whole-group bootstrap (3,000 resamples, seed 203100): P2 accuracy loss point
2.34 pp, one-sided upper 6.25 pp; P2 − P3 log-probability point +2.285, lower
+1.844; P4 accuracy loss point −0.78 pp, upper 0.0 pp. `edited` /
`wrong_variable_edit`: 0 attempted (all 128 exclusions `absent` under the
unchanged frozen rule v1.0.0). Edit hypothesis: **untested**.

Every amended-convention number matches the post-hoc lens preview exactly
(the generations reproduced bitwise, so the frozen bootstrap on lens-corrected
correctness is the same computation). The lens numbers were descriptive
previews; this table is the frozen-convention record.

## What the confirmation establishes — and what it does not

With a working instrument, the Gemma-27B task is usable (P0 0.965) **and** the
preservation criterion fails on it (P2 accuracy-loss upper 6.25 pp > 5 pp).
The description-derived direction still moves behavior on-task: P2 agrees
with P0 on 0.967 of variants against P3's 0.725, its next-token KL is 0.216
against 2.81, and the P2 − P3 log-probability lower bound is +1.84 > 0. But it
loses too much accuracy against the frozen bound. The closed pilot's 27B
"preservation pass" (upper 2.34 pp at P0 0.254) was a floor effect of the
format rejections. With the floor removed, all three families land in the
same place: usable task, preservation failed (upper 41.4 pp on Qwen2.5-7B,
10.16 pp on Gemma-3-12B, 6.25 pp on Gemma-3-27B). The edit hypothesis remains
untested (0/128; the AV is unchanged).

Two upgrades stay prohibited. **This does not vindicate the language
route**: every family with a usable task now fails preservation, and nothing
here extends beyond this family, task and site. **This does not rank model
sizes**: 6.25 pp on 27B against 10.16 pp on 12B is not evidence that larger
models preserve better. Both fail the bound. The sites differ (block 41 of 62
is a full-attention block; the 12B's block 32 of 48 is a sliding-window
block), the NLA pairs were trained separately, the 27B AV is served under a
BF16 cast that cannot be audited locally, and there is one task and one pilot
per size. The closed pilot is not relabeled: its frozen outcome stands,
because instruments are frozen per phase. P4's near-P0 behavior (0.965, KL
0.0042) is a no-more-than-budget baseline comparison, not an optimality
claim.

## Limits

One confirmation pilot under the amended instrument; no validation ran or is
implemented. The amended instrument measures exact answers after stripping
trailing whitespace only; the teacher-forced scoring channel is unchanged and
still scores the newline-free suffix the model does not prefer — recorded,
not reconciled. The 27B AV's BF16 serving cast attaches to every AV-dependent
number here and remains unauditable locally (no fp32 A/B fits on this
machine). Bitwise reproduction of the closed pilot shows that this software
stack is deterministic on this machine; it does not show that its outputs are
correct. Auditor PASS replays saved counts, KL, the frozen statistics and the
decision; it does not authenticate execution or replay AV/AR generation and
target forwards.
