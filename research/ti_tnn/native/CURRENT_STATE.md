# Native TI-TNN: current state

## Next native phase: mixed text/code preparation, 2026-11-13

Remote #80 is merged at `11f3aa5051f345d33128a49b75d38a7c38da52fa`, containing
the approved reporting head `a5bf0134acd0d60e0fd98d25fe61b423b31766ec`. The
implementation agent observed this merge and did not perform it. A separate
`ti-tnn-native-mixed-learning` branch starts there; completed study worktrees and
artifacts remain unchanged.

The [mixed text/code work order](MIXED_LEARNING.md) returns language/code learning
to the primary role, with dense/shared/untyped/typed controls and a fixed register
monitor. It reuses validated packing, document positions, target timing, tokenizer,
model, losses, sampler and recovery components. Missing bounded mixed allocation
and capability preparation/scoring are implemented and CPU-tested.

The proposed per-arm endpoint is 40,001,786 loss tokens / 19,862 updates / cursor
158,896. Complete cost projects to 3.186 h raw, **5.054012 h** with the existing
reservation's 50% margin and all allowances. The earlier 35% / 4.501 h draft estimate
is superseded; its historical artifact remains preserved. **Admission is blocked:**
the complete comparison exceeds the proposed allocation by 0.304012 h and the live
native balance by 0.200710 h.
The proposed 4.75 h maximum is drawn from the live **4.853301779298153 h remaining**,
not added to the native cap. No new reservation, research training, inference on
saved checkpoints, larger-model campaign, shaping or WSP was run in this phase.
The thin execution adapter now binds the existing training, reservation, checkpoint
and recovery machinery. CPU fixtures cover explicit authorization, read-only preflight,
all four fixed endpoints, identity rejection, incomplete arms, cumulative recovery,
duplicate comparisons and concurrency admission. Historical runner contracts remain
unchanged. Private runtime bindings stay outside the public repository.
Next permitted action: review the draft PR, frozen binding and budget shortfall.
A revised, reviewed resource work order and separate explicit owner authorization
are required before any execution; the prepared command cannot override admission.

The completed #80 record and earlier preparation snapshots follow.

A register program can be memorized without learning how to execute a new one.
The completed one-state/eight-state comparison separates training fit, transition
transfer, composition and value preservation without establishing useful
complete-program generalization.

## Completed owner-authorized study: 2026-11-13

All eight arms completed the fixed 16-epoch endpoint: 1,424 updates, cursor 11,376
and 2,590,592 full-label tokens each. All nine 128-example populations were
evaluated. Multi-state training-anchor scores are only 0–2/128 versus 38–119/128
for repeat_one. Both registered development recommendations are false. Lower
heldout numeric CE and partial register gains remain visible alongside the poor
fit and failed screen; these do not establish reusable neural transitions.

See [results](INITIAL_STATE_RESULTS.md), [all metrics](INITIAL_STATE_DETAILS.md)
and [independent results review](INITIAL_STATE_RESULTS_REVIEW.md). All 72 saved
score objects reproduce exactly without model calls. Independent reviewers checked
all 9,216 completions, endpoints, identities and ledger accounting with no blocking
findings. The frozen training source is `3808e8209b4b22e36eded29343721195ca682ca4`;
the protocol is `bd108e03f51aefbc5bf8ce1cc0086c874d450b79ecf5668a23b275d1f37f50df`.
Reporting changes are made in a separate worktree; frozen source and preparation
remain unchanged. Remote main was fetched again and remains
`52384a6addd7e06922931f583e8f8be13c48eed2`.

The single reservation closed normally after **0.9253861445033302 hours**, with
no recovery or failed study attempt. Native use is **7.146698220701847 / 12 hours**,
leaving **4.853301779298153 hours**. The authorized four-hour maximum came from
that cap. No comparison process or native reservation remains active at the
completion audit. Runtime paths and operational records remain private.

Next permitted action: review the completed results in draft #80. No merge,
additional run, larger-model campaign, shaping or WSP is authorized. Preparation
test results below are historical local checks; post-run checks are saved-artifact
reconstruction and independent review, not another training replication.

## Historical pre-authorization update: 2026-11-13, draft #80

This section records the preparation checkpoint before the later explicit owner
authorization and completed run above. Its pending-action wording is historical.

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
