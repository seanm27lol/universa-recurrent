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

## Phase Two second-family port readiness (Gemma-3, 2026-09-22)

The closed Phase Two pipeline was ported to a second model family on the
`gemma3-port` worktree branch: Gemma-3-12B-it plus the released
kitft/nla-gemma3-12b-L32-av/ar pair, behind a small audited architecture
registry. This is engineering-readiness work on the same machinery, not a
reopening of the closed phase and not a result on released Gemma weights.
See [reports/gemma3_port_readiness.md](../research/open_weight_lingua/reports/gemma3_port_readiness.md).

| Statement | Status | Evidence and boundary |
|---|---|---|
| The pipeline resolves the Gemma-3 family without changing the Qwen2 path | Implemented and fixture-tested | Registry-driven hook paths/widths/depths; all 128 pre-existing tests pass unchanged plus 21 Gemma-3 fixture tests (149 total); the Qwen lock is byte-untouched and no Qwen manifest field changes |
| The Gemma-3 NLA pair's metadata conventions match the pipeline's checks | Source-audited and tokenizer-tested | Real fetched schema-2 sidecars (width 3840, extraction block 32, AV injection scale 80000.0, AR suffix tokens) parsed by the production loader; the public AV/AR tokenizers pass the production av/ar prompt checks including the live Gemma BOS rule |
| The L32 capture site interacts with Gemma-3 interleaved attention | Analyzed and fixture-exercised | Block 32 is a sliding-window block (window 1024, full attention at every 6th index, verified from the fetched configs); the pinned 128-token bucket is always inside the window, so masked pads stay exactly zero and the causality gates are unaffected |
| The Gemma-3 sources are pinned and fetch-verified | COMPLETE for the mirror-sourced lock; official small files UNVERIFIABLE | The official google/gemma-3-12b-it @ 96b6f1ec… is gated-manual (anonymous 401); the target pins the public unsloth mirror @ 9478e665… instead, and its weight shards/tokenizer blobs carry identical LFS sha256 in both repos' API records (byte-identical); four small config files diverge and are pinned as the mirror's own bytes with the official metadata recorded alongside. All 64,857,314,351 bytes downloaded through `preflight.fetch_models` and re-hashed by `verify_models` (PASS per role; 2026-09-22). Gemma Terms of Use apply to the user regardless of download source; the HF gate is an access mechanism, not the license itself. Reconciliation against the official revision remains pending gated access |
| The mirror target tokenizer/template match the NLA pair's conventions | Verified on the real pinned files | tokenizer.model byte-identical to the AV blob (`cmp`); chat_template.jinja byte-identical to the AV/AR file and equal in text to chat_template.json and the embedded tokenizer_config template; BOS-first chat prompts, shared boundary token 107, answer round-trips with eos 106 |
| The target and NLA pair agree on one termination token | FALSE, documented | Mirror target tokenizer declares eos = `<end_of_turn>` (106); the AV/AR declare `<eos>` (1). Per-role tokenizers keep the pipeline self-consistent; the AV-side stop convention is a documented watch item for the first smoke, not a silently patched behavior |
| Any released-Gemma forward, identity gate, AV description, AR direction or behavioral measurement exists | NOT RUN | Real GPU stages are handed off with exact commands; nothing beyond fixture/meta-device/load-config checks has executed |
| A Gemma-3 pilot would assess the frozen Phase Two thresholds or edit coverage | NOT ESTABLISHED | Those thresholds were locked for the Qwen phase; applying them to Gemma is a new pilot decision, and locked validation remains unimplemented for both families |

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
