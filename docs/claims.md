# Claim ledger

## Phase Two Milestone 1 (2026-09-21)

An isolated [Qwen/NLA engineering implementation](../research/open_weight_lingua/README.md)
transfers the preserve/intervene/measure procedure to one learned activation.
[Source audit](../research/open_weight_lingua/reports/source_compatibility.md) and
[local verification](../research/open_weight_lingua/reports/milestone_one.md) distinguish
implemented interfaces from released-model evidence.

| Statement | Status | Evidence and boundary |
|---|---|---|
| One block-output vector can be captured and reinserted without changing the fixture computation | Tested locally with tiny random Qwen models | Native BF16 equality, hidden-state indexing, single-site mutation, hook cleanup, padding and repeated-input tests; not a released-Qwen result |
| Released NLA metadata/tokenizer conventions are resolved | Source-audited and tokenizer-tested | Immutable model/source lock; actual AV marker/neighbors and AR suffix/depth checked |
| The local AV adapter distinguishes different injected embeddings | Tested locally with fixtures | Cache-free A/B/A calls with identical token IDs; no SGLang equivalence or real AV quality claim |
| AR reconstruction uses the trained value head and omits final normalization | Implemented and fixture-tested | Required safe head loading, shape/dtype validation and final-block test; released AR weights not loaded |
| Text alone reconstructs the complete native activation | NOT CLAIMED | Direction reconstruction restores a separately retained four-byte original norm; all remaining prompt context persists |
| The pinned environment runs on this GB10 | Partially checked | Installation, imports and basic BF16 kernel passed; capability warning remains; full real-model smoke NOT RUN |
| The eight-group runner produces auditable evidence | Fixture-tested | Safe numeric files, all-group accounting, reports-only inventory and independent saved-count/KL replay; no execution authentication |
| Language preserves or specifically edits real Qwen behavior | NOT ESTABLISHED | Released-weight smoke NOT RUN; pilot, calibration/PCA, text-edit coverage and locked validation are later milestones |
| This is a recurrent-depth experiment, a new NLA method, or a whole-state compression result | FALSE | Conventional transformer; external pretrained NLA pair; one selected-token patch |

The bounded scientific stopping rule remains one pilot and at most one locked
validation, followed by a write-up even on failure. Neither stage is executable
from this Milestone 1 runner. Phase One's conclusions below are unchanged.

## Phase Two Milestone 2 (2026-09-21)

Milestone 2 adds the grouped split plan, the P4 calibration-fitted PCA
baseline, the frozen text editor, group-bootstrap decision statistics and the
calibration/pilot runner stages to the same
[isolated implementation](../research/open_weight_lingua/README.md). Smoke
behavior is unchanged, and the
[pilot decision document](../research/open_weight_lingua/protocols/pilot_decision.md)
is published as a template with every outcome field pending.

| Statement | Status | Evidence and boundary |
|---|---|---|
| The five-split plan (smoke, calibration, pilot, two validation blocks) is deterministic and disjoint | Implemented and fixture-tested | Fixed split order, namespaces and seeds; carried canonical-program and tokenized-prompt exclusion inventories; plan hash over every group identity; disjointness is textual and no released-model data exists yet |
| The P4 baseline is fitted on pooled, unlabeled calibration unit directions with a declared byte-budget rank | Implemented and fixture-tested | PCA mean/basis, rank `min(fitted_rank, floor(text_bytes/2))`, float16 coefficients, shared mean/basis bytes reported separately, sha256 fit identity for the manifest; no task labels and no norm restoration inside the fit |
| The frozen text editor classifies and minimally edits explicit current-value statements | Implemented and fixture-tested | Three frozen forms, canonical values 0–19, eligible/absent/ambiguous statuses, value-span-only replacement; agreement with the reference program recorded separately and never gating an edit; rule version and sha256 enter the manifest |
| Pilot decision statistics resample whole groups | Implemented and fixture-tested | 3,000 bootstrap resamples at a fixed seed; one-sided 95% upper/lower estimates for the §8 rules; absolute log-probability contrasts only, no fraction-recovered ratios |
| Calibration and pilot stages run from one CLI | Implemented, fixture-level | `--stage smoke|calibration|pilot` with smoke unchanged; pilot requires `--calibration-fit`; `run_calibration.sh`/`run_pilot.sh` follow the smoke script pattern; real-model execution NOT RUN |
| The 256-group released-model calibration fit exists | COMPLETE (pinned re-baseline) | Pinned run calibration-20260921T235017Z-fd111b21, auditor PASS, 1024/1024 extractions, fit identity `37af38ff…`, frozen median norm 98.8137; supersedes the unpadded run calibration-20260921T221636Z-63a6be98 |
| The 128-group pilot outcome and keep/stop decision exist | COMPLETE (pinned run); decision STOP | Pinned run pilot-20260921T235825Z-6164d210, 128/128 groups, auditor PASS; unmet criteria `p2_accuracy_loss` and `edit_eligible_groups`; first attempt pilot-20260921T222056Z-69535f26 preserved as an explicitly failed pre-conditions run |
| The 512-group locked validation exists | NOT RUN; not implemented | No validation stage in the CLI; the pilot reports a projected cost against the eight-hour budget and never auto-starts it |
| P4 is an optimal compression baseline, or generic reconstruction already matches language | FALSE as framed | The condition is no-more-than-budget under a declared byte rule; the comparison is a pilot question, not a premise |
| Frozen-rule edits establish discovered variable semantics in the activation | NOT CLAIMED | The parser matches frozen surface forms only; a statement's truth never gates its editability |
| Any scientific threshold (80% P0 floor, 5-point P2 loss bound, 32-group edit floor) has been assessed | NOT YET | These are design choices to lock before validation; pilot outcomes are pending |

The bounded stopping rule is unchanged: one pilot, at most one locked
validation, then a write-up even on failure. A negative pilot does not trigger
checkpoint shopping; a repaired source/API bug permits only a regression test
plus a visibly versioned rerun of the affected pilot (brief §11).

## GB10 real-model smoke: suffix-causality repair (2026-09-21)

The first released-model smoke (run smoke-20260921T185833Z-c264a184, raw
measurements in the retained suffix-probe results) passed fetch, all 32
identity gates bitwise, and every AV/AR call, then failed teacher-forced
scoring: the old `allclose(1e-5)` comparison of the prefix-site vector before
and after appending an answer suffix is unattainable whenever the GEMM
reduction order changes with sequence length. The check was re-derived from
the smoke measurements and frozen before the pilot, not relaxed after
inspecting results.

| Statement | Status | Evidence and boundary |
|---|---|---|
| An appended answer suffix cannot change the causal prefix activation at equal sequence length | Proven bitwise on released Qwen2.5-7B-Instruct | Eight same-length pairs with different suffix content bitwise identical (GB10 BF16 and CPU fp32); `score_answer` now enforces a tolerance-free same-length dummy-suffix bitwise equality gate, so a genuine content leak fails while kernel noise cannot |
| Cross-length BF16 drift at the prefix site is bounded kernel noise under a frozen justified bound | Measured on smoke data; bound frozen pre-pilot; coverage **FALSIFIED on the pilot distribution** 2026-09-21 (pre-pilot, no outcomes inspected) | Maximum relative L2 drift 3.09e-2 across 20 cross-length appends (cuBLAS kernel re-selection on sm_121; CPU fp32 collapses to ~4e-6; answer argmax stable 20/20); `SUFFIX_DRIFT_BOUND = 1e-1`, safety factor ≈3 over the measured maximum, justified per brief §4, recorded in the manifest and in per-score `suffix_drift_relative` evidence. Falsification preserved explicitly: pilot-0000-A-x measured 2.58e-1 (a prompt-conditioned heavy tail the smoke sample missed; fp32 collapses it to ~3.2e-6, so noise, not leak). The bound stays frozen as an untouched backstop and was never re-fitted to pilot data; the operative fix is kernel-shape pinning — see the next section |
| The smoke rerun with the repaired gate exists | COMPLETE (engineering smoke) | Run smoke-20260921T211151Z-a4e4a038 on the repaired code: 8/8 groups successful, independent auditor PASS, drift evidence max 3.55e-2 across 336 samples inside the frozen 1e-1 bound; measured P0 25/32 (0.781), P1 gate bitwise, P2 18/32 (0.562, KL 1.41), P3 15/32 (KL 4.63), P5 0/32 (KL 10.49), donor KL 0.098. Eight smoke groups are engineering data, not a scientific result: P0 0.781 sits below the 80% pilot-usability floor, an early signal for the pilot decision — not a smoke failure and not a pilot outcome; calibration/pilot remain NOT RUN |
| The causality gate or the drift measurements establish anything about the semantic content of activations | FALSE | The gate discriminates a causal leak from kernel-reduction noise only; no interpretation, faithfulness, or semantics claim |

## GB10 first pilot attempt: identity-gate failure and kernel-shape pinning (2026-09-21)

After the unpadded calibration completed (run
calibration-20260921T221636Z-63a6be98), the first pilot attempt (run
pilot-20260921T222056Z-69535f26, preserved with `completion.json: FAILED`)
died in `Target.identity_gate` on the first variant, pilot-0000-A-x ("raw
restoration changed greedy answer"), before any condition ran: 1 failed
group, 127 skipped, no pilot or validation outcomes inspected. Gate-level
diagnosis on the same variant (local scratch probes `/tmp/owl_greedy_probe/`,
not committed): the unpatched and pinned greedy paths diverge at generation
step 2 in a near-tie argmax flip (unpatched EOS margin 0.75; pinned "\n"
margin 0.375), and the unpadded prefix site at that sequence differs from
the pinned vector by 2.58e-1 relative — 2.6× the frozen 1e-1 bound and 7×
the smoke maximum. The drift is deterministic (same-length bitwise), appears
at every append length +1…+5 (0.26 on pilot-0000-A-x; 0.12–0.16 on two more
variants, all above the bound at answer-suffix lengths), grows from 6.3e-2
already at the block-0 output, and collapses to ~3.2e-6 in fp32 — pure
kernel-reselection noise with a prompt-conditioned heavy tail, not a
semantic leak, but outside the smoke-fitted bound's cover. `score_answer`'s
bound would have been exceeded in the behavior stage as well. Re-fitting the
bound to pilot data was rejected (a tolerance fitted to the distribution
that broke it, with weaker bug discrimination); the repair instead removes
the drift class by construction.

| Statement | Status | Evidence and boundary |
|---|---|---|
| Unpadded cross-length prefix-site drift on the pilot distribution stays within the frozen 1e-1 bound | FALSE (falsified 2026-09-21, pre-pilot, gate measurements only) | pilot-0000-A-x 2.58e-1 at the divergence step and ~0.26 at every append +1…+5; pilot-0000-A-y 0.144–0.165; pilot-0001-A-x 0.019 (+1) then 0.122–0.127; the smoke's 336 samples (max 3.55e-2) under-sampled a prompt-conditioned tail; fp32 collapses all of it to ~3.2e-6 |
| The first pilot attempt produced condition outcomes or a decision | FALSE; failed run preserved explicitly | Run pilot-20260921T222056Z-69535f26 died in the identity stage on variant 1; its decision/summary fields are degenerate artifacts of a dead run, not measurements |
| Kernel-shape pinning removes the cross-length drift class by construction | Implemented and fixture-tested; real-model gate check PASSED on pilot-0000-A-x | Every target forward right-padded to `TARGET_BUCKET = 128` (masked pads exactly zero, bitwise-proven; same-shape forwards bitwise deterministic on this backend); greedy/scoring write into masked pad slots; loud pre-inference fit check (max prompt 79 + 8 generation + 3 suffix = 90 ≤ 128); `SUFFIX_DRIFT_BOUND` kept frozen as an untouched backstop; the greedy identity comparison is a recorded exact/drift_diverged backstop counted in summaries, never a decision input. Gate check (pinned code, this variant plus two more): identity gates bitwise, greedy "exact", `suffix_drift_relative` exactly 0.0 — details in reports/milestone_two.md |
| The pinned re-baseline runs (smoke, calibration, pilot) exist | COMPLETE | smoke-20260921T233021Z-cbe4057e (8/8 groups, drift exactly 0.0 over 336 samples, greedy backstop exact 32/32 both stages, auditor PASS); calibration-20260921T235017Z-fd111b21 (auditor PASS); pilot-20260921T235825Z-6164d210 (128/128 groups, greedy backstop exact 512/512 both stages, auditor PASS); unpadded runs retained as superseded engineering evidence |

The bounded stopping rule is unchanged: one pilot, at most one locked
validation, then a write-up even on failure. This pre-pilot repeatability
repair inspected gate internals only; no pilot or validation outcomes were
observed, and no tolerance was relaxed.

## Phase Two pilot outcome and phase closeout (2026-09-21)

**This phase is closed.** Read the [findings](phase_two_results.md) and
[technical closeout](phase_two_technical_report.md) before interpreting the
ledger entries above.

The pinned 128-group pilot (`pilot-20260921T235825Z-6164d210`, manifest
sha256 `cb3ec24f…`, auditor PASS) completed and the frozen decision rule
returned **STOP**. The phase closes with this documented bounded negative
result; the locked validation is NOT RUN (stop decision, and independently
the projected validation cost 49,253 s ≈ 13.7 h exceeds the 28,800 s budget).
The human record is
[protocols/pilot_decision.md](../research/open_weight_lingua/protocols/pilot_decision.md).

| Statement | Status | Evidence and boundary |
|---|---|---|
| The pinned pilot executed all 128 groups with every gate green | COMPLETE, auditor PASS | 128/128 groups successful, 0 failed/skipped; identity gates bitwise on all 512 variants; greedy identity backstop exact 512/512 in both stages; every suffix-drift measurement exactly 0.0 under kernel-shape pinning |
| The site can affect the intended measurement (pilot gate 1) | Supported | P5 norm-matched random direction changed 508/512 greedy generations (accuracy 0.006, KL 11.75); raw donor KL 0.173 with 39/512 changed; P0 accuracy 0.8105 ≥ 0.80 floor |
| The language route preserves behavior within the frozen 5-point bound (gate 2) | NOT SUPPORTED at this task/site | P2 accuracy loss versus P0: point 34.4 pp, one-sided 95% upper 41.4 pp > 5 pp limit (P2 0.564 vs P0 0.811, KL 2.71); P2−P3 correct-answer log-prob lower +3.80 > 0 (met) — the description-derived direction is on-task but not preserving |
| Frozen-rule editing is feasible at this task/site (gate 3) | NOT SUPPORTED; edit hypothesis UNTESTED | 0/128 groups eligible (floor 32); exclusions all "absent" (x: 127 absent/1 eligible, y: 128 absent); the single eligible receiver parse stated a value disagreeing with the reference (recorded, never gating). Zero coverage means limited explicit-variable coverage here, not that descriptions lack editable semantics; no manual ground-truth description is presented as discovered semantics |
| Generic PCA reconstruction is competitive under the stated byte budget | Observed, descriptive only | P4 accuracy 0.803 vs P0 0.811 (loss upper 3.13 pp, KL 0.0157, P0-agreement 0.967) against P2 0.564. Per brief §12 this may be read as "generic reconstruction is competitive under the stated budget"; the upgrade to "language is useless for interpretability" is prohibited and not claimed |
| P2's residual accuracy relies on the retained four-byte norm channel | Not supported | Calibration-median-norm diagnostic matches P2 (0.561 vs 0.564; KL 2.72 vs 2.71) |
| The locked 512-group validation exists | NOT RUN; remains unimplemented | Stop decision under the frozen thresholds plus over-budget projection (13.7 h > 8 h); the runner never auto-starts validation and the phase is closed |
| The pilot establishes semantic content, faithfulness, or a compression result | FALSE | Behavioral preservation/coverage measurements at one site on one task family; gates and statistics are engineering instruments, not semantic evidence |

## Post-Phase-Two steering assay (2026-09-22)

A new, separate measurement on the **reused** pilot split: does a description
*difference* — the pilot's saved AR directions (arm 1) or frozen oracle
templates through the live AR (arm 2) — steer behavior under the NLA paper's
reconstructed-difference recipe (`h′ = h + α·‖h‖·Δ/‖Δ‖`)? Frozen protocol:
[protocols/steering_assay_brief.md](../research/open_weight_lingua/protocols/steering_assay_brief.md);
outcome report:
[reports/steering_assay.md](../research/open_weight_lingua/reports/steering_assay.md).
This did not reopen the closed phase; the pilot's recorded outcomes stand.
The single frozen run (`steering-20260922T160432Z-d3b9e4d7`, auditor PASS)
COMPLETED with a negative result, and the assay is closed per its stopping
rule (one run; no follow-up sweeps).

| Statement | Status | Evidence and boundary |
|---|---|---|
| The steering assay has been run on the released models | COMPLETE; negative result | Run steering-20260922T160432Z-d3b9e4d7: 128/128 groups, 256 receiver rows, auditor PASS, wall 6,114 s; manifest `git_head` aabcef5, plan hash ff060012… (reused pilot split) |
| The assay's success criteria are fixed | Frozen pre-run (design choices), never weakened post-outcome | α grid {−1, 0.5, 1, 2}; at α=1: flip-to-B ≥ 0.30, y-integrity ≥ 0.90, control moved-x < 0.10; whole-group bootstrap 3,000 resamples, seed 205100 |
| Any arm shows targeted steering at α=1 | NOT SUPPORTED at this checkpoint/task/site | All three arms failed every frozen criterion: av_difference flip 0.0781 / integrity 0.5703 / control-moved 0.4453; oracle_terse 0.0313 / 0.4219 / 0.4844; oracle_structured 0.0234 / 0.3594 / 0.5625; `successful_arms: []`. Intended deltas flip 2–8% while disturbing the unaffected answer in 43–64%; controls move x as much as intended deltas; higher α buys disruption, not targeting |
| Description-difference L shifts are specific to the intended variable | Not supported | Steered mean L (logP(B)−logP(A)) moves toward B vs P0's −16.08 (av@1 −9.29, terse@1 −9.00, structured@1 −7.15), but matched controls shift comparably (control@1 −10.16 / −11.23 / −9.88) — an indiscriminate push, not targeted control |
| This result refutes NLA steering generally, or establishes a semantic conclusion | FALSE | One checkpoint, one task family, one site; behavioral measurement only. The NLA authors' poetry-planning steering used a different, stronger model, site and task; oracle texts were intervention instruments; no semantic claim in either direction |
| The steering machinery works on tiny fixtures | Tested locally | Delta unit-scaling, patch composition, template determinism and hashes, controls wiring, ITT encoding, manifest locking, audit replay and tamper detection on random Qwen fixtures; never labeled released-model results |

## Phase Two vLLM backend (2026-09-22)

An opt-in vLLM 0.30.0 backend for the AV/AR stages of the
[open_weight_lingua runner](../research/open_weight_lingua/README.md): a pinned
worker venv (`.venv-vllm`, separate because vllm pins transformers 5.x against
the pipeline's 4.57.6), a subprocess worker replicating the eager recipe
(marker context, injection scale, recompute-per-pass greedy decode, identity
final norm, external value head), runner flags `--av-backend`/`--ar-backend`
(default `eager`), and a measured equivalence gate
(`scripts/check_vllm_equivalence.sh`). Target capture/patch stays eager; vLLM
has no public mid-block capture/replacement API for it.

| Statement | Status | Evidence and boundary |
|---|---|---|
| The vLLM worker reproduces the eager AV/AR recipe mechanics | Implemented and fixture-tested | CPU tests cover job validation, EmbedsPrompt construction, value-head math, checkpoint-inventory proof, and gate comparison logic; the worker never imports the pipeline package |
| vLLM AV/AR outputs are interchangeable with eager outputs | GATE FAIL (measured 2026-09-22) | Gate replay of the pinned smoke bundle (`smoke-20260921T233021Z-cbe4057e`, models re-hash-verified): AV greedy continuations 0/32 token-identical (median first divergence at token 10.5; all rows still status-ok paraphrases; eager top-2 logit margin at a sample divergence measured 0.125 — near-tie argmax flips under different BF16 kernels); AR direction cosine min 0.8097 / median 0.99953 / max 0.99979, norm-relative error up to 5.85%. Both bars missed, so vllm is a distinct measurement backend: it stays non-default and its outputs may not be mixed into eager-regime evidence |
| The target model runs under vLLM | FALSE | Target capture, patching, kernel-shape pinning and behavioral scoring remain eager Transformers; only the AV/AR adapters have an opt-in vllm path |
| A faster vLLM decode implies any scientific claim | FALSE | The gate's recompute-per-pass AV discipline makes decode caching unusable by construction (measured 559.5 s for 4,564 forwards vs 522.4 s eager — no speedup on this workload); the phase-two pilot outcome (STOP) is unchanged, and this backend adds no validation result |

## Phase Two second-family port and replication (Gemma-3, 2026-09-22/23)

The closed Phase Two pipeline was ported to a second model family on the
`gemma3-port` worktree branch — Gemma-3-12B-it plus the released
kitft/nla-gemma3-12b-L32-av/ar pair, behind a small audited architecture
registry — and the replication ran to its bounded end: smoke, calibration and
one 128-group pilot, decision STOP. This does not reopen the closed phase's
findings and adds no validation stage. See
[reports/gemma3_port_readiness.md](../research/open_weight_lingua/reports/gemma3_port_readiness.md)
and [reports/gemma3_pilot.md](../research/open_weight_lingua/reports/gemma3_pilot.md).

| Statement | Status | Evidence and boundary |
|---|---|---|
| The pipeline resolves the Gemma-3 family without changing the Qwen2 path | Implemented and fixture-tested | Registry-driven hook paths/widths/depths; all 128 pre-existing tests pass unchanged plus 21 Gemma-3 fixture tests (149 total); the Qwen lock is byte-untouched and no Qwen manifest field changes |
| The Gemma-3 NLA pair's metadata conventions match the pipeline's checks | Source-audited and tokenizer-tested | Real fetched schema-2 sidecars (width 3840, extraction block 32, AV injection scale 80000.0, AR suffix tokens) parsed by the production loader; the public AV/AR tokenizers pass the production av/ar prompt checks including the live Gemma BOS rule |
| The L32 capture site interacts with Gemma-3 interleaved attention | Analyzed and fixture-exercised | Block 32 is a sliding-window block (window 1024, full attention at every 6th index, verified from the fetched configs); the pinned 128-token bucket is always inside the window, so masked pads stay exactly zero and the causality gates are unaffected |
| The Gemma-3 sources are pinned and fetch-verified | COMPLETE for the mirror-sourced lock; official small files UNVERIFIABLE | The official google/gemma-3-12b-it @ 96b6f1ec… is gated-manual (anonymous 401); the target pins the public unsloth mirror @ 9478e665… instead, and its weight shards/tokenizer blobs carry identical LFS sha256 in both repos' API records (byte-identical); four small config files diverge and are pinned as the mirror's own bytes with the official metadata recorded alongside. All 64,857,314,351 bytes downloaded through `preflight.fetch_models` and re-hashed by `verify_models` (PASS per role; 2026-09-22). Gemma Terms of Use apply to the user regardless of download source; the HF gate is an access mechanism, not the license itself. Reconciliation against the official revision remains pending gated access |
| The mirror target tokenizer/template match the NLA pair's conventions | Verified on the real pinned files | tokenizer.model byte-identical to the AV blob (`cmp`); chat_template.jinja byte-identical to the AV/AR file and equal in text to chat_template.json and the embedded tokenizer_config template; BOS-first chat prompts, shared boundary token 107, answer round-trips with eos 106 |
| The target and NLA pair agree on one termination token | FALSE, documented and repaired by recipe fidelity | Mirror target tokenizer declares eos = `<end_of_turn>` (106); the AV/AR declare `<eos>` (1). The first smoke confirmed the AV emits 106 after its answer; the adapter now stops at the checkpoint's declared eos set ([1, 106] for the released Gemma AV), matching the pinned upstream recipe's no-override convention |
| Any released-Gemma forward, identity gate, AV description, AR direction or behavioral measurement exists | COMPLETE for the engineering smoke only | First attempt smoke-20260923T025530Z-1ec606fa preserved explicitly (all 32 AV descriptions `truncated`: the released AV emits `<end_of_turn>` 106 after a well-formed explanation and the eos-1-only stop ran a 106/107 loop to the 200-token ceiling). Repair follows the pinned recipe (kitft/nla-inference 38b802a: `sp = {"temperature": 1.0, "max_new_tokens": 200, "skip_special_tokens": False}` — no stop override, so the checkpoint's declared eos set `[1, 106]` from the hash-pinned generation_config.json governs); `Verbalizer` now reads the stop set from the loaded model's generation_config, with five regression tests including the loop case and the unchanged single-eos Qwen convention. Repaired run smoke-20260923T034330Z-4bc7e34a: 8/8 groups, auditor PASS, identity gates bitwise 32/32, greedy backstop exact 64/64 across both stages, all suffix drifts exactly 0.0, all 32 descriptions ok; AR round-trip cosine 0.982–0.994 (median 0.988). Smoke-stage P0 0.5312 and P2 0.6562 / P3 0.3750 / P5 0.0 are 32-variant engineering numbers, not pilot outcomes; no scientific threshold assessed |
| A Gemma-3 pilot would assess the frozen Phase Two thresholds or edit coverage | ASSESSED under the frozen thresholds; decision STOP | Pilot pilot-20260923T043613Z-f7e71d7b, 128/128 groups, auditor PASS: unmodified accuracy 0.5195 < 0.80 floor (unmet), sensitivity present, P2 accuracy-loss upper 3.91 pp ≤ 5 pp (met), P2−P3 log-prob lower +2.61 > 0 (met), edit-eligible groups 0/128 (unmet; the edit hypothesis is now untested on BOTH families under the unchanged rule v1.0.0); projected validation cost 75,348 s vs the 28,800 s budget, independently over budget. **This frozen outcome stands and is not relabeled by the instrument fix below** |
| The 12B usability failure was model behavior | FALSE — instrument artifact, proven by taxonomy and corrected-instrument rerun | Saved-evidence taxonomy (reports/post_hoc_answer_lens.md): 228/512 P0 rejections were a trailing newline (221 with the correct value; Qwen pilot: zero such rows); content accuracy 0.98–1.00 across a difficulty probe grid. Frozen amendment protocols/gemma_answer_convention.md (commit 8ec15cb, frozen before the run): exact correctness iff exact_integer(text.rstrip(), answer). Confirmation pilot pilot-20260927T202455Z-f5ec3892 (auditor PASS; 512/512 P0 generations bitwise-identical to the closed pilot; same plan/fit/lock): P0 0.9512 (usable) and P2-loss upper 10.16 pp > 5 pp (preservation now fails on a usable task); decision STOP on p2_accuracy_loss + edit_eligible_groups. The earlier Gemma 'preservation passes' were floor effects of the format rejections |
| The confirmation vindicates the language route on Gemma-12B | FALSE | With a working instrument the family lands where Qwen landed: usable task, preservation failed; the edit hypothesis remains untested (0/128; the AV is unchanged); nothing is established beyond this family, task and site, and no closed pilot is relabeled |
| Gemma preservation passing means language reconstruction preserves behavior on a usable task | FALSE as framed; prohibited upgrade | The P0 usability floor exists so preservation numbers mean something; at 0.52 unmodified accuracy the pass is weak assay evidence — P2 tracks P0 at 0.854 agreement including its errors. This does not establish preservation on a usable task, for either family; on Qwen the task was usable (P0 0.811) and preservation failed (loss upper 41.4 pp) |
| The L32-of-48/12B versus L20-of-28/7B contrast has a known cause | NOT ESTABLISHED | Site, size, family and NLA-pair training all differ; the contrast is explicitly labeled speculation for a future phase, not a finding |
| Locked validation on the Gemma family exists | NOT RUN; remains unimplemented | Stop decision plus over-budget projection; the runner never auto-starts validation; the Gemma replication is closed after one pilot, same stopping rule as Qwen |

### Gemma-3-27B extension (2026-09-23)

The user authorized running the kitft/nla-gemma3-27b-L41 pair (extraction
block 41 of 62, width 5376) with the unsloth/gemma-3-27b-it mirror target.

| Statement | Status | Evidence and boundary |
|---|---|---|
| The 27B sources are pinned in a new lock | Implemented; fetch-verified bytes pending the smoke record | configs/model-lock-gemma3-27b.json: mirror target @ 7a5a3053… (twelve shards + both tokenizer blobs LFS-identical to the gated official's API record; divergent small files pinned as the mirror's own), AV @ 4e721238… (108.08 GB), AR @ aa2f2972… (37.60 GB); the sidecar fields are the real fetched values (injection_scale 60000.0, mse_scale √5376, block 41, AR 42 layers) — never extrapolated from the 12B pair |
| The 27B AV is served at its released precision | FALSE — documented deviation, user-authorized | The released AV is float32-native (108.08 GB) and cannot load in the ~65 GB available of the 121 GB pool; it is served BF16 per the lock's justified serving_dtype declaration, following the pinned recipe's own bf16 local defaults (load_embedding_only and NLACritic both default dtype=torch.bfloat16; the SGLang launch sets no --dtype). No local fp32 A/B comparison is possible; the cast is unauditable locally and recorded in the lock, manifests, report and this ledger |
| The 70B NLA pair is runnable on this machine | FALSE, with evidence | kitft/Llama-3.3-70B-NLA-L53-av totals 141.12 GB by its API file inventory (verified 2026-09-23) against 121 GB physical unified memory; no pinned serving precision fits, so no 70B lock exists |
| Block 41 of the 27B stack shares the 12B site's local-attention character | FALSE (different character, documented) | 27B block 41 is a full-attention (global) block (full at every 6th index of 62); the 12B block 32 is sliding-window local. Both windows (1024) exceed the pinned 128-token bucket, so the exact-causality gates are unaffected |
| Any 27B real-model smoke exists | COMPLETE, after two OOM-killed attempts and one loader-bug run, all preserved explicitly | smoke-20260923T195955Z-e90ad8b5 and smoke-20260923T201748Z-1687a778: kernel-OOM-killed at AV shard 22/22 (manifest + partial raw evidence retained; no completion record by construction). smoke-20260923T220025Z-762d65f6: COMPLETE_WITH_FAILURES, all 32 descriptions truncated (the streaming loader lost the checkpoint's generation_config.json eos list [1, 106]; fixed by mirroring from_pretrained's merge). smoke-20260923T231748Z-068d59ba: COMPLETE, 8/8 groups, auditor PASS, identity gates bitwise 32/32, greedy backstop exact 64/64, all 336 suffix drifts exactly 0.0, all 32 descriptions ok, AR cosine 0.9678–0.9891 (median 0.9827); P0 12/32, P2 10/32 (KL 4.1e-7), P3 14/32 (KL 4.24), P5 0/32 — engineering gates only, not scientific results |
| The 27B OOMs were caused by incomplete stage release | FALSE, measured and refuted | Probe evidence (kept in /tmp/owl27_memprobe/): after the identity-stage target forward, del + release_models() returns the pool to baseline (MemAvailable 68.5→67.4 GiB, cuda 0/0 reserved). The true cause: the fp32→BF16 cast of the AV built a ~47 GiB *anonymous* CPU copy that persisted through model.to('cuda') (NVRM out-of-memory alongside; two probes killed with 46–47 GiB anon-rss). The fix streams the cast from mmap'd shards into a meta-built, device-materialized model (no whole-model CPU copy), fixture-proven bitwise-identical to stock loading including config-derived buffers and the declared eos set; AV then loads in 33 s with 14.9 GiB still free |
| The 27B pilot and its decision exist | COMPLETE; decision STOP | Calibration calibration-20260926T183645Z-c3374c27 (auditor PASS, fit 784594e89f6c315b…, median 41,629.3) and pilot pilot-20260926T185550Z-d085d53c (128/128 groups, auditor PASS): unmodified accuracy 0.2539 < 0.80 (unmet), sensitivity present (P5 510/512 generations differ, KL 13.4; donor 39, KL 0.059), P2 accuracy-loss upper 2.34 pp ≤ 5 (met), P2−P3 log-prob lower +1.84 > 0 (met), edit-eligible groups 0/128 (unmet; rule v1.0.0 unchanged); projected validation cost 181,295 s vs the 28,800 s budget, independently over budget; identity gates bitwise 512/512, greedy backstop exact 1024/1024 across stages, all 6,144 suffix drifts exactly 0.0; AR cosine 0.9656–0.9897 (median 0.9823). **This frozen outcome stands and is not relabeled by the confirmation below** |
| On 27B, P2 ≈ P3 on accuracy means language reconstruction failed | FALSE as framed (floor effect) | At P0 = 25% the accuracy channel is floor-limited and weakly informative; the informative channels separate description from noise — P2 agreement 0.816 / KL 0.216 vs P3 0.574 / 2.81, and the positive P2−P3 contrast (+1.84 lower). Equally, the met criteria do not establish task usability: the P0 floor failed, so the preservation pass is weak assay evidence under an unusable-task caveat |
| Language reconstruction preserves behavior on a usable task, for any family | NOT ESTABLISHED anywhere | Three closed pilots, one assay: Qwen2.5-7B/L20 usable task (P0 0.811) with preservation FAILED (loss upper 41.4 pp); Gemma-12B/L32 and Gemma-27B/L41 preservation criteria met (loss upper 3.91/2.34 pp, P2−P3 lower +2.61/+1.84) under unusable tasks (P0 0.520/0.254). The edit interface is absent on all three (0/128 each; edit hypothesis untested everywhere). Under the frozen answer-convention amendment, the Gemma-12B and Gemma-27B confirmation pilots make both tasks usable (P0 0.9512/0.9648) and fail preservation (loss upper 10.16/6.25 pp): every family with a usable task fails the 5 pp bound |
| Bigger models preserve activations better | FALSE as framed; prohibited upgrade | The 12B→27B pass-margin reading is confounded by floor effects at P0 0.254 and by the 27B AV's BF16 serving cast (float32-native, unauditable locally — no fp32 A/B possible on this machine). Whether site depth, model size, family, or the NLA pair's training drives the Gemma pairs' preservation strength is speculation for a future phase, not a finding; no validation ran on any family and all three phases are closed |
| A Gemma-27B confirmation pilot under the answer-convention amendment exists | COMPLETE; decision STOP; the lens prediction matched exactly | Frozen addendum protocols/gemma27b_confirmation_addendum.md (commit 15f4210, pushed before any model forward; the lens numbers recorded as the prediction under test, not as thresholds). Pilot pilot-20260929T034532Z-4e4d655f (128/128 groups, auditor PASS, manifest sha256 9dee790c…): `answer_convention: rstrip`, the same lock (ca078e52…), split plan (ff060012…), 128 groups and calibration fit (784594e89f6c315b…, reused, not refit), and the same lock-declared BF16 AV cast. All 512 P0 generations bitwise-identical to the closed pilot's, as are every other condition's generations and all 512 AV descriptions. P0 0.9648 = 494/512 (usability met); P2 accuracy-loss point 2.34 pp, one-sided upper 6.25 pp > 5 pp (not met); P2−P3 log-prob lower +1.844 (met; digit-identical to the closed pilot); edit-eligible 0/128 (unmet). Identity gates bitwise 512/512, greedy backstop exact 1024/1024 across stages, all 6,144 suffix drifts exactly 0.0; runner wall clock 47,280.8 s; projected validation cost 192,132 s vs the 28,800 s budget. See reports/gemma3_27b_confirmation.md |
| The 27B confirmation vindicates the language route | FALSE | With a working instrument, 27B lands where Qwen and Gemma-12B landed: usable task, preservation failed. The edit hypothesis remains untested (0/128; the AV is unchanged), and nothing is established beyond this family, task and site |
| Under the corrected instrument the larger Gemma preserves better (6.25 vs 10.16 pp) | FALSE as framed; prohibited upgrade | Both fail the frozen 5 pp bound. The sites differ (block 41 of 62 is full-attention; the 12B's block 32 of 48 is sliding-window), the NLA pairs were trained separately, the 27B AV's BF16 serving cast is unauditable locally, and there is one task and one pilot per size |

### Post-hoc answer lens and frozen answer-convention amendment (2026-09-27)

A descriptive re-read of the closed pilots' saved generations, with trailing
ASCII whitespace stripped once before the unchanged exact-integer rule. It
changes no frozen outcome: thresholds and instruments are frozen per phase, and
the STOP decision on each family stands. Confirmation pilots under the frozen
amendment later reproduced the 12B and the 27B lens numbers exactly. See
[reports/post_hoc_answer_lens.md](../research/open_weight_lingua/reports/post_hoc_answer_lens.md),
[protocols/gemma_answer_convention.md](../research/open_weight_lingua/protocols/gemma_answer_convention.md),
[reports/gemma3_12b_confirmation.md](../research/open_weight_lingua/reports/gemma3_12b_confirmation.md),
[protocols/gemma27b_confirmation_addendum.md](../research/open_weight_lingua/protocols/gemma27b_confirmation_addendum.md)
and [reports/gemma3_27b_confirmation.md](../research/open_weight_lingua/reports/gemma3_27b_confirmation.md).

| Statement | Status | Evidence and boundary |
|---|---|---|
| The Gemma usability-gate failures reflect the model's computation | FALSE (instrument artifact) | Gemma-3-12B pilot taxonomy: 228/512 variants rejected for trailing whitespace, 221 of them with the correct value after stripping; the mirror target answers digits + newline + `<end_of_turn>`. The Qwen pilot has zero whitespace-affected rows |
| Under the lens, every family exceeds the 5 pp preservation limit | Observed post hoc; now a frozen outcome for all three families | Lens P2 accuracy-loss one-sided upper: Qwen2.5-7B 41.41, Gemma-3-12B 10.16, Gemma-3-27B 6.25 pp, recomputed from saved generations with the frozen bootstrap. Each is now also a frozen outcome: the Qwen pilot has zero whitespace-affected rows, so its frozen record equals the lens, and the 12B and 27B confirmation pilots (rows below) reproduced the lens values exactly |
| The frozen-metric Gemma "preservation passes" show the language route works on Gemma | FALSE | They were floor effects of the format rejections; under the lens the Gemma tasks become usable (P0 0.951 / 0.965) and preservation fails |
| An answer-convention amendment exists for future runs | Frozen 2026-09-27, before the Gemma-12B confirmation pilot | `--answer-convention rstrip`; the default stays `raw`, byte-identical to the closed pilots' instrument, and remains the convention for Qwen runs |
| A Gemma-12B confirmation pilot under the amendment has been run | COMPLETE; decision STOP | `pilot-20260927T202455Z-f5ec3892`, 128/128 groups, auditor PASS, `answer_convention: rstrip`, same lock, groups and calibration fit as the closed pilot. All 512 P0 generations bitwise-identical to the closed pilot's. P0 0.9512 (usability met); P2 accuracy-loss upper 10.16 pp > 5 pp (not met); P2 − P3 log-prob lower +2.607; edit-eligible 0/128. Every number matches the lens preview exactly. See reports/gemma3_12b_confirmation.md |
| A Gemma-27B confirmation pilot under the amendment has been run | COMPLETE; decision STOP | `pilot-20260929T034532Z-4e4d655f`, 128/128 groups, auditor PASS, `answer_convention: rstrip`, same lock, groups and calibration fit as the closed pilot, under the frozen addendum protocols/gemma27b_confirmation_addendum.md. All 512 P0 generations bitwise-identical to the closed pilot's. P0 0.9648 (usability met); P2 accuracy-loss upper 6.25 pp > 5 pp (not met); P2 − P3 log-prob lower +1.844; edit-eligible 0/128. Every number matches the lens preview, which the addendum recorded before the run as the prediction under test. See reports/gemma3_27b_confirmation.md |

### Post-hoc structure of edit eligibility (2026-09-29)

A descriptive re-read of the saved AV descriptions from the Qwen2.5-7B pilot
and both Gemma confirmation pilots, asking why no group is edit-eligible on any
family. It loads no model and relabels nothing: 0/128 coverage and the STOP
decisions stand. See
[reports/edit_eligibility_structure.md](../research/open_weight_lingua/reports/edit_eligibility_structure.md)
and `scripts/edit_structure_analysis.py`.

| Statement | Status | Evidence and boundary |
|---|---|---|
| The 0/128 edit coverage is an accident of the frozen rule's exact phrasing | FALSE (structural) | Even case- and markdown-insensitive, the three frozen forms occur in 0 of 1,024 Gemma descriptions; Qwen's 3/512 are false values inside quoted narratives. The capture site is the answer position and the AV writes answer predictions, not variable-state statements |
| Gemma descriptions carry the queried answer value | Observed post hoc | Contained in 120/128 (12B) and 127/128 (27B) receivers against an unrelated-answer baseline of 13 and 18, including computed values the prompt never shows (44/50, 49/50); it leads the "Final token" candidate list in 108 and 111 receivers. Qwen: 12 leads, containment near its baseline (38 vs 36). A candidate is the AV's next-token guess, not evidence of reading the activation |
| Receiver descriptions carry the program state the wrong-variable control needs | FALSE at this site | The other variable's computed value appears in 4/45 (12B), 2/45 (27B) and 5/45 (Qwen) receivers |
| An answer-slot edit rule would give usable edit coverage | Syntactic only; behavior not measured | Rule AS-1.0.0 (`answer_slot.py`): 65/128 (12B) and 74/128 (27B) single-candidate receivers, above the 32-group floor on paper, but it edits a described prediction, not a variable's state; 63 and 53 receivers are ambiguous, and 19 and 18 already list the ±1 counterfactual |
| A description's number content transfers to behavior under replacement | Observed on Gemma-12B; weak on Gemma-27B and Qwen (post hoc, P3 natural edits) | When another group's slot lead differs from the own answer, P3 answers it in 220/477 (12B), 35/470 (27B) and 31/440 (Qwen) rows; unrelated-answer baseline 0/401, 3/403, 13/377. The wrong-lead subset (~60 rows per Gemma family) is too small to separate the number from the rest of the text |

### Edit-channel diagnostics (2026-09-30)

Four frozen diagnostic runs on the Gemma pairs, each protocol committed and
pushed before its forwards: D1 rewrites the AV's answer slot on the pilot
receivers (12B, 27B), D2 captures at the end of the program (12B), and D3
replicates the everywhere edit on the fresh 12B calibration split. New
measurements, not pilot reruns; every 0/128 coverage result and STOP decision
stands. See
[reports/edit_channel_diagnostics.md](../research/open_weight_lingua/reports/edit_channel_diagnostics.md),
[protocols/edit_channel_diagnostics.md](../research/open_weight_lingua/protocols/edit_channel_diagnostics.md)
and [protocols/edit_replication.md](../research/open_weight_lingua/protocols/edit_replication.md).

| Statement | Status | Evidence and boundary |
|---|---|---|
| The diagnostics' harness recomputes the pilot's P2 exactly | Verified bitwise | D1: fresh P0 logits/greedy, E0 direction, E0 replacement and E0 greedy equal the saved pilot records on 65/65 (12B) and 74/74 (27B) receivers; D2: pilot-site re-capture 128/128 and AV replay 8/8; D3: calibration re-capture 256/256 and D1 replay 4/4; all audits PASS |
| Rewriting only the AV's answer slot moves the Gemma answer | NOT SUPPORTED (frozen reading unmet on both families) | D1 slot-only edit to the counterfactual: 2/64 (12B) and 0/74 (27B) answers moved, `Δ` lower bound 0.0; the 12B prediction (0.3–0.5) was wrong. Every 12B primary receiver repeats the value outside the slot, so a slot-only edit leaves a self-contradicting text |
| A consistent edit (every mention rewritten) moves the Gemma-3-12B answer to the written value | SUPPORTED out of sample (frozen D3 readings R3a and R3b met) | D1 (secondary condition): 25/64. D3 on 256 fresh groups: 42/121 to the counterfactual (0.347, `Δ` lower 0.281), 43/120 to the other neighbour, 32/121 to a distant value; log-probability shifts +11.9 to +17.6 toward the written value versus +1.2 to +2.6 toward others. Boundary: it shows the AR-patch path carries a consistently written number to the output; it does not show that the AV reads the activation, and it edits a next-token guess, not a variable's state |
| The same holds on Gemma-3-27B | NOT OBSERVED | D1: 0–1 of 74 answers moved under every edit, although each edit shifts the log-probability toward its own written value by +3.4 to +5.5 nats; consistent with the 7% P3 adoption. Not a size ranking: sites, NLA pairs and the 27B AV's BF16 cast differ |
| Capturing at the end of the program yields variable-state descriptions | NOT SUPPORTED (frozen reading unmet; as predicted) | D2 (12B): the affected variable is bound to its true value in 14/128 descriptions (floor 32) and its computed value in 1/50 (floor 0.25); computed values appear at all in 4/50, against 44/50 at the answer position; both variables are named in 54/128 (1/128 at the answer position); frozen-rule hits 0 |
| A hand-written text asserting the number moves the Gemma-3-12B answer as well as the edited AV description | FALSE (frozen D4: R4a met, R4b unmet; the prediction was the reverse) | Protocol protocols/edit_oracle_control.md (commit 32960ce), run editoracle-d4-gemma3-12b-20261002T144025Z-91fd9c75 (D3 E1g replay bitwise 121/121, audit PASS): the edited AV description moves 42/121 answers to the written value; the steering assay's frozen terse and structured templates, a bare answer slot and a neutral mention move 0, 0, 1 and 0. Every oracle still shifts logP toward the value by about 5–7 nats; their non-`a` answers are a leading newline plus the true answer (stripped: `a` in 119–120/121). Boundary: the oracles also differ in length, register and repetition. **Superseded reading:** this row first read D4 as evidence of receiver-specific content; D5 (next row) refutes that — a different program's description works as well |
| The edited description's effect needs content specific to the receiver's own activation | FALSE (frozen D5: R5a unmet, R5b met; as predicted) | Protocol protocols/edit_foreign_control.md (commit 43b8f64), run editforeign-d5-gemma3-12b-20261003T053546Z-0e304bef (D3 E1g replay bitwise 121/121, audit PASS): another same-variable receiver's AV description rewritten to this receiver's counterfactual moves 46/121 answers, against 42/121 for the receiver's own (own minus foreign −0.033, lower −0.099); unedited, that foreign description moves 3/121 to `c` and pulls 39 of 106 answers to its own number. Boundary: what the AR needs at this site is AV-register text about this task stating a number; nothing here concerns faithfulness to an individual activation |
| The flip needs the whole AV description to agree on the written number | SUPPORTED in part (frozen D6: R6a met, R6b unmet; R6b and the dose magnitudes were predicted wrongly) | Protocol protocols/edit_consistency_dose.md (commit 64b7eb1), run editdose-d6-gemma3-12b-20261003T065726Z-3f350865 (D3 E1g replay bitwise 121/121, audit PASS): rewriting the first ¼, ½, ¾ and all of a description's mentions (in text order) moves 0, 0, 11 and 42 of 121 answers, while the log-probability shift toward the written value grows smoothly (+1.15, +3.66, +7.46, +12.51); every mention except the answer slot moves 0, and the slot alone (D3) moves 5. Boundary: one mention order, so the doses also shift location; a property of the AR-patch path, not of how faithfully a description reads its activation |
| These diagnostics test or relabel the Phase Two edit hypothesis | FALSE | The frozen v1.0.0 hypothesis (edit a stated variable-state value) remains untested; 0/128 coverage and STOP stand on every family. D1/D3 test the AV's answer guess, which at this site is the same number, not the same hypothesis |

## Phase Three: is there a readable program state? (2026-10-04, closed at Stage 0)

Phase Three asked whether a position and layer holds both variables' current
values linearly, so that a typed record (`x is currently 10; y is currently 6`)
could be read and edited causally. The brief
([protocols/phase_three_brief.md](../research/open_weight_lingua/protocols/phase_three_brief.md),
frozen at 6d51d0e) set gate G0 before any forward. G0 failed on the primary
model, so the phase closed after Stage 0. See
[reports/phase_three_stage0.md](../research/open_weight_lingua/reports/phase_three_stage0.md).

| Statement | Status | Evidence and boundary |
|---|---|---|
| Some surveyed position linearly exposes both x's and y's current values (G0) | NOT SUPPORTED (frozen gate failed; Phase Three closed) | Gemma-3-12B run p3-stage0-gemma3-12b-20261004T062811Z-745ea17c: the best site (answer position, block 45) reads x 0.500 and y 0.463 held out, computed values 0.457 and 0.415, against G0's 0.80 and 0.50. Qwen2.5-7B (reported only, p3-stage0-qwen2.5-7b-20261004T074740Z-840871a1): 0.436 and 0.381. Copy-the-literal baseline 0.664 (x) and 0.445 (y); permuted-label probes 0.05–0.12. The predicted mid-layer state at the end of the program was wrong: there both values stay near 0.15–0.24 |
| The models hold the asked variable's value, not the program's state | Observed post hoc (descriptive; never part of G0) | Split by the variable asked: Gemma answer position, block 45, asked 0.930/0.875 versus not asked 0.070/0.051; Qwen block 27, 0.746/0.707 versus 0.125/0.055. It becomes readable only in the last third of depth. At Qwen's released-NLA block 20, even the asked value reads 0.316/0.199. This is consistent with on-demand computation and explains why Phase Two's descriptions predicted answers |
| Stages 1–2 (self-verbalization, typed causal edits) were run | FALSE | Closed by the failed gate; the validation splits remain unopened. Nonlinear and per-line codes are untested; searching for them would need a new brief |

## Phase Four: is state tracked at statement boundaries? (2026-10-04, closed after its survey)

Phase Four probed every statement boundary for x's and y's values as of
that line, by how each variable changed on that line. The brief
([protocols/phase_four_brief.md](../research/open_weight_lingua/protocols/phase_four_brief.md),
frozen at af4e648) set gate G4 before any forward. G4 failed on the primary
model as predicted, so the phase closed. See
[reports/phase_four_trace.md](../research/open_weight_lingua/reports/phase_four_trace.md).

| Statement | Status | Evidence and boundary |
|---|---|---|
| Carried values are readable at statement boundaries (R1) | NOT SUPPORTED (frozen; predicted) | Gemma-3-12B run p4-trace-gemma3-12b-20261004T161553Z-517f9189, at layer 7: carried 0.672/0.383 held out, below the copy-the-literal baseline 0.904/0.661; carried computed values 0.000/0.077 (R1 needed 0.80 and 0.50). Qwen2.5-7B (reported only, p4-trace-qwen2.5-7b-20261004T164315Z-27667f5a): 0.707/0.375, computed 0.000/0.108 |
| Arithmetic results are readable at statement boundaries (R2) | NOT SUPPORTED (frozen; predicted) | Gemma 0.176/0.206 at layer 18; Qwen 0.184/0.184 (needed 0.80). No layer of either model exceeds about 0.22, against a copy baseline of 0.044/0.018 and permuted-label probes of at most 0.089. Literal updates read at 0.79–1.00 because the boundary token is the number itself |
| These models keep a readable running state of the program on this task | NOT SUPPORTED by Phases Three and Four together | No linearly readable state was found while reading (Phase Four) or at the end of the program (Phase Three); only the asked value appears, after the question. Linear probes only; codes that vary across lines are untested; readability is not use, and the causal Stage 1 was not run |

## Phase Five: is the state there, but nonlinear? (2026-10-04, closed after its survey)

Phase Five re-asked Phases Three and Four's questions with a
one-hidden-layer MLP probe, under the same site and layer rules, with a
permuted-label control at every selected site. The brief
([protocols/phase_five_brief.md](../research/open_weight_lingua/protocols/phase_five_brief.md),
frozen at 1841db5) set gate G5 before any fit or forward. G5 failed on the
primary model as predicted, so the phase closed. See
[reports/phase_five_nonlinear.md](../research/open_weight_lingua/reports/phase_five_nonlinear.md).

| Statement | Status | Evidence and boundary |
|---|---|---|
| A small nonlinear probe reads both final values at some surveyed site (G5a) | NOT SUPPORTED (frozen; predicted) | Gemma-3-12B run p5-nonlinear-gemma3-12b-20261004T211741Z-421ba7af, on Phase Three's saved activations: the selected site (answer position, block 46) reads x 0.479 and y 0.443 held out, computed 0.371 and 0.387 (G5a needed 0.80 and 0.50); permuted-label control 0.053/0.055. Split by the variable asked: 0.871/0.824 asked, 0.086/0.062 not asked. Qwen2.5-7B (reported only, p5-nonlinear-qwen2.5-7b-20261004T214157Z-354536e6): 0.445/0.322 at block 24 |
| A small nonlinear probe reads the running state at statement boundaries (G5b = R1 and R2) | NOT SUPPORTED (frozen; predicted) | Gemma at Phase Four's re-captured boundaries: carried 0.527/0.319 and carried computed 0.019/0.077 at layer 7 (R1); arithmetic results 0.184/0.219 at layer 32 (R2), against 0.80. Qwen: carried computed 0.019/0.092; arithmetic 0.221/0.237. Controls 0.049–0.064. No layer of either model reads arithmetic results above 0.250 |
| A nonlinear code hides the state that linear probes missed (the Othello-GPT pattern) | NOT SUPPORTED on this task | The MLP matches the linear probes within a few points, and is below them on carried values (by 3–15 points). One small probe family (256 hidden units, 768–1,246 training rows); a much larger probe or more data could differ. Readability is not use. Synthetic tests show this probe reads a sign-flipped code that a linear probe cannot |

## Completed phase-one findings (2026-09-16)

**This sequence is closed.** Read the [findings](phase_one_results.md) and
[technical closeout](phase_one_technical_report.md) before interpreting the
historical implementation/protocol entries below. Their original requirements
remain visible; a broad claim is not upgraded merely because a narrower test passed.

| Statement | Status after the reported experiments | Evidence and boundary |
|---|---|---|
| Prepared/batched receipt processing reduces full local request time in the measured cases | Supported, workload-specific | Shared endpoint pipeline: 27.131 -> 14.149 ms for 256 inputs, models/references already loaded; not a neural speedup |
| Bounded conversion eliminates long full-trajectory requests | Not supported | Chunk16 improves median consistency, but the checkpoint-aggregated p90 remains about 101 ms |
| Frozen 16-bit numerical descriptions approximately preserve continuation | Supported on the tested cohorts | Seven claim/abstention changes in 184,320 correlated comparisons on 2,048 underlying problems; not exact preservation |
| A correct typed field edit closely reproduces a direct intervention | Supported in the pilot | 1.72e-11 effect-disagreement MSE, five claim differences on repeated comparisons of 512 inputs; names supplied by the schema |
| Individual intervention effects always add | Not supported | Paired edits show substantial downstream interaction |
| The locked linear predictor meets the declared limited-usefulness criterion | MET | Relative RMS 0.690756; upper estimate 0.732032 <= 0.8; both block points <= 0.8 |
| The locked predictor meets the separate close-prediction criterion | NOT MET | The same result is not <= 0.1; one individual fit also exceeds 0.8 |
| The response rule is a cheap or compact interpreter | Not established | Common evaluator: 73 probes and 1,720 coefficient bytes per input; raw state: 24 bytes |
| The final run validates a compressed-language pipeline end to end | FALSE | Final prediction evaluation uses raw known fields, not the compressed typed-edit path |
| This phase discovers opaque-neuron meanings or a general language for models | Not established | Known-field numerical operations and local approximations only |

Exact values, denominators and source-archive identities are in the
[public summary](../experiments/results/phase_one_20260916.json). Saved-loss and
receipt audits do not independently rerun checkpoint inference or authenticate
execution. No new model, codec, threshold or automatic follow-on run accompanies
this closeout.

## Historical implementation and protocol ledger

| Statement | Status | Evidence or required test |
|---|---|---|
| Classical solver reuses reduced coordinates and avoids repeated SVD | Implemented | Classical solver tests |
| Neural v1 routes once, then recurs inside one selected space | Implemented; historical exploratory path | v1 source and tests |
| Neural v2 maintains one state per candidate structure | Implemented | Shape, gradient, and inference tests |
| V2 revises candidate probabilities after recurrent updates | Implemented | Retained trajectories and Lingua events |
| V2 can continue, commit, or abstain | Implemented | Dense/compact and policy tests |
| Abstention returns a provisional probability-weighted mixture | Implemented | Arithmetic and tamper tests |
| Policy thresholds use a disjoint calibration split | Implemented | Checkpoint seed metadata and tests |
| Candidate states satisfy their declared linear constraints | Implemented up to floating-point tolerance | `z = Q a` plus independent checker |
| Dense and compact v2 make matching decisions | Tested on listed cases | Equivalence tests within declared tolerances |
| Compact v2 skips candidate update examples | Implemented | Execution counters |
| Compact v2 is faster | **Not established** | DGX wall-clock benchmark at matched quality and batch size |
| V2 improves on v1 | **Not established** | New DGX evaluation on untouched test seeds |
| V2 beats the transparent solver | **Not established** | Held-out evaluation and timing |
| Recurrence itself causes a gain | **Not established** | Direct, dedicated-depth, and untied controls |
| Structural coordinates cause a gain | **Not established** | Ambient recurrent control |
| Route probabilities are calibrated | **Not established** | Reliability/calibration diagnostics on untouched data |
| The calibrated costs are universally appropriate | FALSE | They are declared experimental choices |
| Lingua checks candidate geometry, recorded diagnostics and policy arithmetic | Implemented for retained one-example records | Independent checker and tamper tests |
| Lingua explains hidden-feature semantics | **Not established** | Predictive and causal intervention evidence |
| V2 transports one persistent state between structures | **Not implemented** | Future typed-transport experiment |
| V2 discovers new structures | **Not implemented** | Future Universa integration |
| A passing test is a scientific result | FALSE | A confirmatory protocol must be sealed before its test block is opened |
| Execution benchmarks can avoid inheriting training settings | Implemented in 0.5.1 | Fresh workers, explicit common controls, and stored backend settings |
| Determinism explains the earlier GPU timing shift | **Not established** | Rebenchmark existing checkpoints; historical settings were not fully recorded |
| Switching an execution profile leaves every output unchanged | Must be checked per run | Numeric tolerances and exact discrete-claim comparison; mismatches suppress matched-output ratios |

Keep negative and ambiguous outcomes visible. Do not turn logical step reduction into a latency claim or a valid structural state into a correct modeling assumption.

For the no-retraining profile comparison and its limits, see [Execution audit](execution_audit.md).

## Final-state retention study (0.5.2)

The opt-in final-only adapter retains no history stack and reuses the existing
fixed-depth updates. Tests and the GPU runner compare final tensors and discrete
claims to rollout. This is not adaptive stopping or an established speedup.
Retained-history checks cover candidate geometry and diagnostic arithmetic, not
learned transitions. Sampled Lingua timings and full-cohort inference have distinct
denominators. See [the retention study](final_state_inference.md).

## Prepared verifier and request pipeline

| Statement | Status | Evidence or required test |
|---|---|---|
| Preparing a pinned reference amortizes repeated verification setup | Reported CPU timings, independently recomputed | [September 10 setup audit](verifier_setup_20260910.md); 97.95% less time for 64 shared endpoint receipts, preparation included |
| The setup ZIP establishes new neural accuracy or independently replayed correctness | FALSE | Four saved receipts are repeated; receipt/reference bytes are absent |
| Prepared checking still runs each record's arithmetic and geometry checks | Source-reviewed and regression-tested | Unchanged per-record checker plus exact snapshot matching; no cached verdict |
| Preparing a new reference helps a one-record batch | Not supported in this run | About 2.2% slower with preparation included |
| Prepared references reduce measured full request costs | Reported DGX timings; locally audited reports and saved receipts | [September 11 pipeline results](verified_pipeline_20260911.md): 80.61x paired speedup for 256 shared endpoint records, preparation included |
| Every saved pipeline receipt passes the existing property checker | Independently replayed locally | 6,400 receipts, 256 distinct inputs; no checkpoint binding or neural execution replay |
| A passing property check establishes a correct structural choice | FALSE | Shared claims choose the wrong synthetic label about 13.4% of the time in the small post-hoc cohort |
| The fresh pipeline cohort establishes confirmatory model superiority | FALSE | Post-hoc 256-input comparison; shared/four-step differences are small and mixed |

## Prospective claim-reliability study

The [fixed next protocol](claim_reliability.md) evaluates the existing five fits
on 20,000 fresh test inputs after every coverage threshold is frozen on
5,000 separate calibration inputs. It specifies a primary shared/four-step
comparison at 75% target coverage and input-paired uncertainty conditional on the
fits and calibration. Software implementation and local smoke tests do not
establish its scientific result. Actual held-out coverage, wrong-claim rates and
numerical-estimate error must be interpreted separately; the protocol promises no
model winner, risk guarantee, equivalence result or new receipt-verification claim.

The [completed study](claim_reliability_20260911.md) was audited from saved
predictions and regenerated truth. Shared minus four-step conditional claim error
at target 75% is +0.0553 percentage points, with a paired 95% interval
[-0.1349, +0.2502]. It establishes neither an eight-step advantage nor equivalence.
At the secondary 50% target, observed wrong-claim rates are 0.1416% shared and
0.0999% four-step, with achieved coverage near 50%; these are not risk guarantees.
The analytic reference's inclusive ties force its nominal 75% policy to 100%
actual coverage. Saved data and metric arithmetic replay; exact checkpoint basis
bytes and GPU execution remain unverified, including a local/reported basis-hash
difference whose cause cannot be established without the checkpoint values.

## State-description continuation (opt-in experiment)

The [state-continuation study](state_continuation.md) implements pause, numerical
state encoding, reconstruction and one-cut resumption of frozen models. It gates
on raw-state restoration and measures lossy errors without hiding failures.
Compression ranges use separate calibration inputs. Context remains with the
solver, never the decoder; its size is reported separately. Named and rotated
controls match numerical payloads, not necessarily common metadata size.
No trained natural-language bottleneck, semantic faithfulness, whole-model
compression or performance improvement is established by this implementation.

## Frozen codec generalization

The [new-input codec study](codec_generalization.md) reuses the overflow-safe
codec without refitting and keeps sixteen bits as the primary precision, with
eight/twelve-bit controls. New matched/noisy/sparse cohorts test preservation and
task error separately. A zero changed-claim count on the previous cohort is not
an all-input guarantee. Sixteen-bit values, identifiers, metadata and retained
solver context remain separately counted; this is not learned language or a
whole-model memory claim. Implementation tests do not establish the new results.

## Typed mathematical edits (opt-in pilot)

The [edit experiment](mathematical_edits.md) compares named commands on decoded
16-bit states with direct numerical interventions and wrong-candidate controls.
Variable meanings and parser alignment are specified, not discovered. Effect
preservation after continuation is an empirical question. No natural-language
interpreter, semantic discovery, formal causal abstraction or speedup is claimed.
