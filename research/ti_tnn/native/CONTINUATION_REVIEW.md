# Independent continuation evidence and reconstruction review

## Verdict

The completed continuation is internally consistent with its declared fixed-endpoint protocol. All six children are
preserved and completed epoch 64 / update 13,568 / full-label exposure 22,282,560 / cursor 108,160. No intermediate
checkpoint was selected, and the three immutable #78 parents still have their expected hashes. The frozen recipe
outcome is **`retain_full_sequence` for every parent**.

This is an independent implementation/artifact review, not independent training, external replication, confirmation,
or an architecture ranking. It supports a bounded reused-development recipe decision only. It does not support WSP,
a 124M campaign, bitwise or BF16 serving equivalence, semantic faithfulness, or a full evaluation replay.

## Review scope and checks performed

The reviewer read the launch and final run statuses, protocol and fork manifest, all child summaries and reconstruction
reports, checkpoint journals and manifests, answer diagnostics, reservation ledger, exporter, and its test. The actual
run is:

```text
/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_continuation
```

Training source is `1c551fafc7d64e7598d79a0e0b208e4c73e1ca79`, with recorded code identity
`becf66889371b0f2e08b4bbee0c56024295f3a73ffda80cad4a1a1a53d4137f8`. The current worktree HEAD is that commit,
and all 16 protocol-listed training-source file hashes match it. The post-run exporter and its test are separate,
currently uncommitted files; the exporter preserved in the bundle has SHA256
`5b73804da6f658e5708c80e6eb2ec7f24bb1f4854e0f1765b3d6552065c93b6b` and matches the worktree copy.

Read-only checks performed for this review established the following:

- Every actual final, fork, and immutable parent checkpoint file hashes to its recorded SHA256. All six journal hashes
  also match final child status, and each child's 23 source and bundle checkpoint manifests are byte-identical.
- Every journal contains 10,176 sequential continuation batch records, from update 3,393 through 13,568, plus 23
  checkpoint records. Final status reports zero replayed updates. Batch identities, example indices, steps, and
  cursors match across all six children.
- `sha256sum -c SHA256SUMS` passes for the complete weight-free bundle. A fresh bundle-only rescore reproduces the
  preserved diagnostics exactly, including SHA256
  `e43dd0128f973ef8622bfb3296bd3cd315a55243175c561a92f4bb4db4780eaa`.
- The focused test `tests/test_ti_tnn_native_continuation_export.py` passes: **1 passed in 0.21 s**. It exercises a
  synthetic completed/failed export, source preservation, split and intervention accounting, strict-allclose failure
  preservation, hash coverage, overwrite refusal, and fresh bundle reproduction. No full native suite or CI run was
  performed for this review.

No training or frozen study was rerun, no model/evaluation generation was launched, no GPU reservation was opened,
and no completed run artifact was changed.

## Completion, pairing, and resource boundaries

A is `full_sequence`; B is `answer_eos`. Both arms keep the complete prompt visible. B changes only the main LM-loss
label selection to answer tokens plus EOS; it does not mask model inputs or add an inference-time semantic label.
Route, state, and load-balance auxiliary targets and coefficients remain unchanged within each pair.

Each pair starts from the same parent model, optimizer state and parameter groups, RNG state, ordered sampler state,
and data order. The optimizer inheritance excludes only the new common learning rate. The parent cosine scheduler is
not inherited: every child constructs a new constant `3e-5` schedule for 48 additional epochs with no warmup.
Fork files have distinct whole-file hashes because their configurations differ, while the recorded model and inherited
optimizer-state identities match within each pair.

The matched quantity is ordered full-label exposure, with equal complete epochs and updates. It is **not** compute or
main-loss-token matching. Per child, the additional exposure is 16,711,920 tokens and 10,176 updates, whereas additional
main-loss scored tokens are 16,711,920 for A and 3,342,336 for B. The latter comprises 786,432 numeric-value tokens
and 2,555,904 formatting/EOS tokens. Equal update counts do not make the objectives compute-matched.

### Fixed final checkpoints

Let `RUN` be the actual run path above. Every final checkpoint is retained at the listed path below `RUN`; no periodic
checkpoint was promoted or selected after evaluation.

| Child | Final checkpoint path below `RUN` | SHA256 | Bytes |
|---|---|---|---:|
| Dense A | `78-dense-full_sequence/checkpoints/final-update-00013568.pt` | `701a4671c721dcfe6c354baaa3828ac96a808b8dde6aadb1a2b81e8c75a3c7a6` | 262,410,571 |
| Dense B | `78-dense-answer_eos/checkpoints/final-update-00013568.pt` | `334ea1e81cfb57d697f78c9e736606c1841b6774538961a1945500f69ba5ef0f` | 262,410,955 |
| Untyped A | `78-untyped-full_sequence/checkpoints/final-update-00013568.pt` | `e11c0be3731165a5c18f6234d8bd1e6988fe80441b0b01cb851db940c063a532` | 257,051,915 |
| Untyped B | `78-untyped-answer_eos/checkpoints/final-update-00013568.pt` | `e1e28d16ac2e87575b916f6a311ed9c3acb15cb2cabab7824e3f2d387aeef81c` | 257,052,299 |
| Typed A | `78-typed-full_sequence/checkpoints/final-update-00013568.pt` | `73097f2e091d866f39d0f5727706f7856a30308dae308ea1ca01b1372a559bcc` | 258,074,195 |
| Typed B | `78-typed-answer_eos/checkpoints/final-update-00013568.pt` | `4f7749c3ebc191d5144707a5b6d5ae5587159fdbce268733718f54fcf74386e0` | 258,074,515 |

The immutable parents live under
`/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_program_generalization/runs/20261110T210743-921461de/arms/<mode>/train/checkpoints/final-update-00003392.pt`.
Their current bytes and both fork files per pair match these identities:

| Parent | Parent SHA256 | A fork SHA256 | B fork SHA256 |
|---|---|---|---|
| Dense | `be4d3405e278ce6a6afe6c7dc4c002cd0f86f9ac07942a6a158b7e0299596418` | `be81f5a85c31afb050ff13ac29fe85789d7d79531e698f6ba535ff1e364bf042` | `2fa9dc85d6463ef3fa75c523a6d9059029d6b5a0797ca70d5e430fb61e7b33aa` |
| Untyped | `1608739f43fb83a88e851b3e8cc9aeee4ec0b1c42ca7323594d46b66963a2199` | `bd0d03aa2fc18c15e4cd03e20e787ba89157b2e294f4dc8de1a68ab29097e51c` | `5a3d0711dc74e0a2b7383b34228758c48073cab41f52e8228cc4434a88dad941` |
| Typed | `cf7d6bb49a4d67a1ad8c188b4a03252b4f586521181e34cc5f99e5d8b75b187f` | `7772bb07d008694a44cdb546c1c1ebd372a70e3405d88506ead094c9dd8fee63` | `fa52395c48550e6d4ad39312eb368cc924ea710aaa0f595efe8916777bab7d12` |

Reservation `20261112T000458-637e80a7` ends with status `ok`. Final run-status accounting charges
**1.9216755555894487 of 3.0 authorized GPU hours**. Native cumulative use is **6.221312076198515 / 12 hours**, with
**5.778687923801485 hours remaining**. The ledger end record's redundant `hours=1.9216755557849925` differs from the
elapsed-derived run-status charge by about `1.96e-10` hours because elapsed is sampled separately; this has no budget
consequence and is not a second charge.

## Recipe decision and empirical boundary

The predeclared rule uses primary free-generation exact-answer counts on all five fixed splits. B must be strictly
better on every split with no regression; otherwise the parent retains A. A global recipe change requires all three
parents to pass. The following compact counts, each out of 128, are sufficient to explain the outcome; the full split,
CE, register, formatting, intervention, and per-operation results remain in `evidence-final/summary.json`.

| Pair | Train A → B | Dev A → B | Unseen composition A → B | Initial values A → B | Longer length A → B | Outcome |
|---|---:|---:|---:|---:|---:|---|
| Dense | 59 → 102 | 7 → 9 | 1 → 2 | 0 → 0 | 0 → 0 | Retain full sequence |
| Untyped | 12 → 25 | 3 → 2 | 0 → 1 | 0 → 0 | 0 → 0 | Retain full sequence |
| Typed | 16 → 22 | 2 → 4 | 1 → 1 | 0 → 0 | 0 → 0 | Retain full sequence |

Thus B does not satisfy strict improvement everywhere for any parent. Initial-values and longer-length results are
zero ties throughout, untyped dev regresses, and typed unseen composition also ties. The decision does not mean B is
worse on every metric, and CE does not override the exact-answer rule. It selects neither a checkpoint nor an
architecture. These are reused development populations and frozen selections, not fresh confirmation data.

The program source is present in every prompt. Construction baselines remain distinct from neural behavior: exact
lookup is 128/128 on the selected train subset and zero on every other split; the supplied-program executor is
128/128 on every split. Executor correctness validates the supplied programs and targets, not learned execution.

The 32 interventions per split and 30 supplemental train baselines retain their separate denominators. Writer-update
suppression effects measure sensitivity of this fixed computation path only. They are not evidence of semantic
faithfulness, and no observed effect is not evidence of state absence; redundant or untested paths remain possible.

## Numerical reconstruction: pass and failure must coexist

All six reports use unautocast CPU FP32 with the exact restored checkpoint weights. They pass the declared scale
policy: floating-tensor relative RMS error at most `0.05` of reference RMS, exact integer routes where applicable,
and two fixed four-token full-versus-cache generation replays with identical tokens. Observed relative RMS errors span
approximately `1.7e-7` to `7.2e-7`.

All six nevertheless fail strict full-versus-cache `allclose` at `atol=rtol=3e-6` somewhere:

| Child | Strict-failing tensors | Maximum absolute difference |
|---|---|---:|
| Dense A | `logits`: `1.2397766e-5`; `slots`: `5.7220459e-6` | `1.2397766e-5` |
| Dense B | `h`, `logits`, `slots`, `state_logits` | `1.2934208e-5` |
| Untyped A | `logits`, `route_logits`, `slots` | `9.5367432e-6` |
| Untyped B | `logits`, `route_logits` | `1.3351440e-5` |
| Typed A | `logits`, `route_logits` | `8.5830688e-6` |
| Typed B | `logits`, `route_logits` | `8.5830688e-6` |

Scale-policy agreement is therefore not bitwise equality, strict-allclose agreement, or a replay of the full official
evaluation. Two four-token replay agreements do not establish numerical equality over all saved generations.

Earlier numerical history also remains in force. Training used BF16 CUDA autocast with FP32 losses; these CPU FP32
reports do not establish BF16 serving equivalence. The earlier BF16 untyped replay changed one of 1,024 route
decisions (`REPORT.md`). Separately, #78 typed CPU FP32 reconstruction failed strict `route_logits` allclose: first
violation `4.291534423828125e-6` at `[0,1,3,1]`, maximum `6.198883056640625e-6`, and relative RMS
`3.5788078335711026e-7` (`PROGRAM_REVIEW.md`). The new continuation results do not erase either failure.

## Weight-free bundle and independent rescore

The canonical bundle is:

```text
/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_continuation/evidence-final
```

It contains **262 files**, of which **260 have `SHA256SUMS` entries**, totaling **352,701,941 hashed payload bytes**.
Coverage is every file except `SHA256SUMS` and `bundle_manifest.json`. No weight files, NPY/NPZ arrays, packed answer
masks, or raw external corpora are exported. Structured JSONL examples and the answer-mask companion manifest are
retained; the mask arrays are not.

| Artifact | SHA256 |
|---|---|
| `summary.json` | `a5865132a5741f10351089377a436042f16730a7881524725b0e486331137a6c` |
| `bundle_manifest.json` | `7acb5a4ffdfa79ba018db0fccffcc5268b8e61882bbcd1b52b025208f60e0bfb` |
| `SHA256SUMS` | `2d082e77806b07c048edc4780bc15609de79bbadddc1e5d36316a07e83d0a2bb` |
| Source-run rescore, `generation_diagnostics.json` | `bdbddceefc18ea58ce7e5de75ecb980d9cf64cca64e99a345439b983ef49f608` |
| Bundle-only rescore, `generation_diagnostics.reproduced.json` | `e43dd0128f973ef8622bfb3296bd3cd315a55243175c561a92f4bb4db4780eaa` |

The independent scorer imports no producer modules. It covers **4,980 saved outputs**: per child 640 primary, 30
supplemental ordinary, and 160 intervened outputs, or 3,840 + 180 + 960 overall. Selected intervention baselines reuse
primary/supplemental outputs and are not counted again. There are zero score changes. It also executes **4,608**
source programs for target validation and rejects no inconsistent targets. Source-run and bundle-only diagnostics
are equal after removing only the top-level `study` path.

This validates saved text, target consistency, exact/format scoring, and intervention-change accounting. It does not
regenerate outputs from model weights, replay training, or independently reproduce teacher-forced CE/state readouts.

Use the commands in **`evidence-final/REPRODUCE.md`**: they create a fresh temporary output, run the bundle-only
rescore, compare all fields except `study`, and run `sha256sum -c SHA256SUMS`. They do not require the original
weights or packed NPY data and do not overwrite the evidence bundle.

## Reporting boundaries and remaining evidence risks

1. **Artifact consistency is not empirical confirmation.** Matching statuses, hashes, journals, saved-output scores,
   and bounded numerical checks establishes that the preserved artifacts agree under the declared checks. The
   empirical claim remains one fixed continuation per parent on reused development populations. It does not justify
   cross-architecture ranking, statistical superiority, fresh-data confirmation, WSP, or 124M inference.
2. **CE populations must not be silently pooled.** Official selected-evaluation train CE uses 128 of 4,096 train
   documents and 2,176 answer+EOS tokens. The separate teacher-forced diagnostic uses all 4,096 train documents,
   1,690 packed rows, and 69,632 answer+EOS tokens. Their train means need not agree. On the other four splits the
   official and diagnostic populations both contain 128 documents / 2,176 answer+EOS tokens, but the fields remain
   distinct. The summary preserves these populations separately.
3. **Stale prose is not current run status.** `CONTINUATION.md` still says prepared/not launched and that final files
   do not exist; the frozen protocol still says `proposed_not_authorized_or_launched`; evaluation templates say only
   a future trained checkpoint can provide learning results. Supplemental answer diagnostics also retain bounded
   `not_evidence_final` labels. These historical/template strings describe proposal or diagnostic scope, not the
   completed-run state. `launch.json`, final `run_status.json`, child statuses, and the closed ledger establish the
   actual completion. The stale text was not rewritten during this review.
4. **The exporter does not itself authenticate checkpoint payloads.** It copies checkpoint manifests and verifies
   status-reported evidence hashes and journal hashes, but intentionally exports no `.pt` files and does not rehash
   final/fork/parent weights. Its matched-endpoint flag checks updates and exposure, not cursor, and a missing final
   manifest association is not itself a hard failure. For this specific run, the independent read-only checks above
   close those gaps by hashing every final/fork/parent payload and checking final cursors and journals. The exporter
   flags should not be generalized into standalone proof of checkpoint integrity.
5. **Weight-free/content checks are structural, not semantic.** Export uses a fixed allowlist of evidence files and
   rejects known weight/array suffixes. The final suffix scan cannot detect weights or raw data embedded under an
   allowed extension. No such content was observed, but `contains_weights=false` and `contains_raw_external_corpora=false`
   are claims about the declared export formats and inventory, not a general content classifier or security audit.
6. **The focused test is not exhaustive.** Its synthetic fixture covers important success/failure behavior but does not
   test a real six-child export from the original run, corrupted actual checkpoint bytes, or every malformed-manifest
   case. The real bundle hash check and fresh bundle-only rescore supplement it; no full-suite or CI conclusion is made.
7. **Post-run code identity is separate from training identity.** The exporter/test are uncommitted at review time.
   The bundle pins the exact exporter and independent scorer (`61f7224a25d831aeee755c4bc440193ef9a65d09ef62b5681222121b0c1a9b85`),
   so reproduction does not depend on mistaking the training commit for the exporter commit. Hashes identify local
   artifacts; they do not authenticate a remote execution.

No material contradiction was found between the final statuses, checkpoint identities, ordered journals, saved-output
rescore, resource accounting, and the frozen `retain_full_sequence` outcome. The principal risks are overreading those
consistency checks as replication or numerical equivalence, pooling unlike metric populations, and treating stale
proposal prose as current status.

Separate final integration validation by the coordinating agent ran all `tests/test_ti_tnn_native_*.py` with the pinned
native interpreter, CPU-only visibility and bytecode/cache writing disabled: **362 passed in 17.19 seconds**. This is a
local construction/regression result, not independent training or empirical confirmation. PR CI is recorded separately
on #79 rather than asserted by this review.
