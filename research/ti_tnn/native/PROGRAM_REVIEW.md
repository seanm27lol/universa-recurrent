# Independent varied-program artifact review

## Verdict

The study is complete under its declared completion criteria. All three arms reached the registered final checkpoint
at **3,392 updates and 5,570,640 loss tokens**, with no intermediate checkpoint selection and no truncated control.
This is independent implementation/artifact review, not independent model training or external replication. It is
not approval for architecture ranking, scaling, confirmation or weight-space programming.

The reviewer inspected the actual run read-only, rescored saved outputs and independently replayed the numerical
finding on CPU. No files, checkpoints, protocols, data or historical results were changed, and no GPU reservation was
created by this review. Detailed metrics are in `PROGRAM_RESULTS.md` and `evidence-final/summary.json`.

## Verified implementation and evidence

- Training source commit is `2eb2e527be9c4f4db60902d78073fd05c3d23f32`; code identity is
  `92402d297627cd41d0e1f1f176e63aa5fa700006ee041bcbd58e2105adbad35a`.
- Each training journal has 3,392 sequential batches, with the same batch order across arms, cursor 27,040, no
  recovery and no replayed updates. The packed sampler independently derives the same endpoint.
- Final checkpoint hashes are `be4d3405e278ce6a6afe6c7dc4c002cd0f86f9ac07942a6a158b7e0299596418` (dense),
  `1608739f43fb83a88e851b3e8cc9aeee4ec0b1c42ca7323594d46b66963a2199` (untyped) and
  `cf7d6bb49a4d67a1ad8c188b4a03252b4f586521181e34cc5f99e5d8b75b187f` (typed). Their files, manifests and
  internal model/optimizer/scheduler/sampler/RNG state checks agree.
- All 62 manifest artifact hashes and all 12 protocol source hashes match. All five populations have 128 primary
  outputs per arm, with 32 per final operation; every intervention has 32 examples, eight per operation.
- Each arm independently scored 640 primary outputs, 30 supplemental train baselines and 160 intervened outputs:
  **830 per arm and 2,490 total, with zero score changes**. Supplemental train operation counts are
  `set/add/multiply/copy = 6/8/8/8`; the two reused primary baselines are both `set`.
- Independent diagnostics cover canonical exactness, format, per-register values, prerequisites, per-operation
  results, and changed completions/generated suffixes. Answer CE covers 2,176 answer tokens including EOS per
  arm/split. State readout covers all packed rows: 1690/51/54/63/74 for train/dev/initial/composition/longer.
- The original `evidence_boundary` template referring to a future trained checkpoint and the frozen protocol's
  proposed status are historical strings, not current run status. `run_status.json` records `complete=true` and
  `failure=null` for the completed experiment.

## Numerical qualification that must remain visible

Dense and untyped pass strict `allclose` at `atol=rtol=3e-6`. **Typed fails strict allclose for `route_logits`**:
first violation at `[0,1,3,1]`, absolute difference `4.291534423828125e-6`, maximum difference
`6.198883056640625e-6`. Relative RMS error is `3.5788078335711026e-7`.

The saved overall `agreement=true` remains correct under the declared scale-based policy: relative RMS <= 0.05,
exact integer routes and exact fixed-replay generated tokens. An independent CPU replay reproduced the same
violation. This is not bitwise replay or BF16 serving equivalence.

## Holdout and intervention interpretation

- All 512 scalar initial values in the initial-value split are in `[24,32)`, and training initial scalars are in
  `[0,24)`. All 128 programs reuse training instruction sequences. Values 24–31 are nevertheless present elsewhere
  in training: 974 examples contain high `set` immediates, 1,508 have high final-state values and 1,945 have high
  noninitial execution-state values. Do not call those numbers unseen everywhere in training.
- Reserved adjacent `multiply` followed by `add` is absent from train, dev, initial-value and longer-length splits.
  There are 133 such pairs across 128 unseen-composition programs. Only 64/128 examples have a direct local
  multiply-to-add input dependency; 14/32 intervention examples have it. Do not claim direct dependency in every
  example or conflate it with later overwrite semantics.
- Longer-length generation truncation is a result, not an incomplete evaluator. Truncated primary outputs are
  68/128 dense, 116/128 untyped and 110/128 typed. Format-valid/per-register denominators are only 4, 2 and 1.
  Intervention truncations are 12/32, 28/32 and 25/32. No prompt was truncated.
- Four interventions changed outputs: dense train `multiply` twice, dense unseen `set` once and typed unseen `set`
  once. Dense train exact moves 1→0, dense unseen stays 1→1 and typed unseen stays 0→0. One dense training answer
  changes from correct to incorrect; the other three changes are incorrect to incorrect. Untyped has zero changes.
  Every intervention targeted pass two exactly once and suppressed four writer updates with complete forwards.
  The historical `interventions.operation` text saying "zero-based pass two" is inaccurate: the implementation uses
  `PASS_INDEX=1`, meaning one-based pass two (`pass_1`), the first supervised after-state. Effects are sensitivity
  only; they do not prove semantic faithfulness. No effect does not prove state absence.
- Exact-lookup and supplied-program-executor baselines are separate. Lookup scores 128/128 on selected train
  examples and zero elsewhere. The executor scores 128/128 everywhere and is not the neural model.

## Resource and bundle verification

Canonical elapsed-derived accounting confirms reservation `20261110T210743-bd8d1d5b` charged
**0.5136816186119617 hours**, below the authorized 2.5 hours. Native usage is **4.2996365206090665/12 hours**, with
**7.7003634793909335 hours remaining** and no open native authorization-3 reservation. The raw end record's
redundant `hours=0.5136816188741553` differs by about `2.62e-10` hours because elapsed is sampled twice; it is not
a charge mismatch. A pre-existing authorization-1 reservation remains open.

`evidence-final` contains 87 files, with 85 checksummed entries totaling 69,758,769 bytes. It has no weights,
packed NPY arrays or raw external corpus. Every file except `SHA256SUMS` and `bundle_manifest.json` is hash-covered.
The whole-study rescore and a fresh bundle-only rescore agree except for the `study` path. The reproduction
interface uses a fresh temporary output directory, so repeated verification does not overwrite evidence.

## Scientific limits

This is one bounded, single-seed study with 128-example populations and 32-example intervention sets. Train
exactness is memorization fit, not generalization. Equal update and token endpoints do not make the architectures
parameter- and compute-matched. The evidence does not support architecture ranking, statistical superiority, broad
composition-generalization, semantic-faithfulness, BF16 serving equivalence, a 124M campaign or WSP.

The full native suite and PR CI are separate engineering checks; their results do not change these scientific limits.
The detailed report and weight-free reproduction bundle are suitable for review, not for automatic merge.
