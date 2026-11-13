# Native TI-TNN: CPU takeover and launch preparation

A register program can be memorized without learning how to execute a new one.
The next task is to distinguish those failures using saved answers, then prepare
a controlled test of showing the same program several initial states.

## Update: 2026-11-13, draft #80

Remote main was fetched again and remains `52384a6addd7e06922931f583e8f8be13c48eed2`.
The v1 preparation and every historical artifact remain preserved. Composition
contract v2 retains 102/128 examples and deterministically replaces 26: each
multiplication output is functionally initial-dependent, with an actual-initial-state
skip-write witness for its designated final target. Independent exhaustive checks
verify all 128 witnesses. Training and all other evaluation populations are unchanged.

Saved #79 text was rescored into a new v2 directory with changed/unchanged-register
accuracy. Every previous aggregate field and complete-answer score is unchanged.
See the appended [value-change diagnostic](DEPENDENCY_DIAGNOSTIC.md#changed-versus-unchanged-reference-values-v2-2026-11-13).

The [protocol](INITIAL_STATE_STUDY.md) now includes a thin adapter to existing
reservation, checkpoint and journal recovery. Thirty additional CPU tests bring
the native suite to **412 passed in 27.54 s**. Independent selection/scoring and
runner reviews report no unresolved findings; see [review](INITIAL_STATE_REVIEW.md).
Tests include mocked execution of all eight arms, authorization, identity attacks,
failure/recovery, cumulative costs, duplicate launch and final-checkpoint preservation.
They establish engineering behavior, not learning or external replication.

The v2 preview projects about **2.515 hours**, about 3.822 hours with margin and
shutdown allowance. The proposed **four-hour maximum is part of the existing
12-hour native cap**. The current final source/protocol binding, final-head CI,
exact local command and fresh operational snapshot are recorded in the private
owner handoff and draft PR. Next permitted action is owner review and separate
explicit authorization of that exact command. Do not execute it merely because
the dry run passes. No study training or saved-checkpoint inference is authorized.

The remaining sections record the completed November 12 takeover checkpoint.

## Verified baseline

- Remote `seanm27lol/universa-recurrent` was successfully fetched. `origin/main`
  is `52384a6addd7e06922931f583e8f8be13c48eed2`, the merge of #79; no subsequent
  main commits were present. Local `main` remains at `c5c34804`; it was not moved.
- Work uses a new isolated branch, `ti-tnn-native-dependency-preparation`, from
  that remote commit. The relevant native, continuation, varied-program,
  scaling and takeover worktrees were clean at intake. Existing instructions
  are preserved; no nested instructions were found for the changed paths.
- Six final continuation checkpoints, six forks and three immutable #78 parents
  match both the committed report hashes and local checkpoint manifests.
- All 260 evidence payload hash checks pass. The documented bundle-only
  rescore reproduces all fields except the permitted top-level `study` path;
  output SHA256 is `e43dd0128f973ef8622bfb3296bd3cd315a55243175c561a92f4bb4db4780eaa`.
- **Locally rerun:** all 362 native CPU tests passed in 24.12 seconds using the
  existing pinned interpreter, with CUDA hidden and bytecode/cache writes disabled.
- **Remote historical check, read live:** the post-merge
  [Tests run](https://github.com/seanm27lol/universa-recurrent/actions/runs/43288442249)
  succeeded. This is separate from the local test run.
- Live native usage is **6.221312076198515 / 12 hours**, leaving
  **5.778687923801485 hours**. #79's elapsed-derived charge agrees with the
  reported 1.9216755555894487 hours to floating-point rounding. No native
  reservation or study process is active. A historical open authorization-1
  record remains unchanged; no end event was invented. No GPU lock holder was
  observed. An unrelated GPU process was left untouched.
- The existing environment and artifact paths exist. Hardware/storage and
  package versions were recorded privately; all 58 installed lockfile pins match.
  No environment changes were made.

Operational paths, process inventory, ledger snapshot, commands, logs and artifact
hashes are in the owner's private local takeover manifest, outside this repository.
The inventory is a point-in-time observation, not a persistent monitor.

## Scientific state retained

Native randomly initialized dense, shared recurrent, untyped MoE and typed MoE
models already exist. #77 demonstrated 32/32 training memorization and 0/32 dev.
#78 and #79 ran dense/untyped/typed, without the shared control. #79's fixed
decision remains **`retain_full_sequence` for every parent**. Answer-focused loss
improved fitting without establishing useful generalization; this does not show
that it is universally inferior.

The historical numerical result is scale-based CPU FP32 agreement with bounded
exact route/token checks. Every #79 child failed strict allclose somewhere.
Neither those numerical checks nor teacher-forced state readouts were rerun here.
Saved-text rescoring is not output regeneration, training replication, BF16
serving equivalence, or evidence of an internal algorithm.

## Completed CPU work and next permitted action

- [Dependency diagnostic](DEPENDENCY_DIAGNOSTIC.md): all 3,840 primary saved
  completions retain their original exact/format scores. Initial-value failures
  include untouched registers (0–1/171 correct per child); ordinary dev depth-two
  and deeper targets score 6–13/120. These are task categories, not neural mechanisms.
- [Next-study preparation](INITIAL_STATE_STUDY.md): implemented paired data,
  independent saved-output scoring, packing audits and resource projection.
  The full pinned-tokenizer build excludes 76,288 historical program/state pairs;
  all 263 artifact and 15 source hashes pass, and all 1,424 planned paired sampler
  batches match exposure/order. Twenty added construction tests cover the new paths.
  The final native CPU suite passed **382 tests in 24.61 seconds**; the release
  check passed using the pinned interpreter and this worktree's package source.
- Proposed learning comparison: repeat one versus eight initial states per pattern,
  across dense/shared/untyped/typed; 16 epochs, 1,424 updates per run. Projected
  2.479 hours, proposed **4.0-hour maximum**, including margin and recovery.

At that checkpoint, runner integration and explicit owner authorization remained
outstanding. The November 13 update above completes integration; owner authorization
remains required. The historical runners retain their frozen contracts.
No study training, saved-checkpoint output regeneration,
GPU reservation, shaping, WSP, larger campaign or environment upgrade was run.
The native regression suite includes its existing tiny synthetic CPU training tests;
those do not rerun any research study or use its checkpoints.

See [architecture](ARCHITECTURE.md), [acquisition](ACQUISITION.md),
[varied-program results](PROGRAM_RESULTS.md),
[continuation results](CONTINUATION_RESULTS.md) and
[continuation review](CONTINUATION_REVIEW.md). Historical proposal wording in
the frozen documents is preserved rather than treated as live run status.
