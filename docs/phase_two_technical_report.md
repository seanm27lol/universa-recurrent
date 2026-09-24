# Open-Weight Lingua: Phase-Two Closeout

**Status: phase closed, decision STOP. No further experiment in this sequence is required or permitted by the brief's stopping rule.**

[Important findings](phase_two_results.md) · [Pilot decision record](../research/open_weight_lingua/protocols/pilot_decision.md) · [Governing brief](../research/open_weight_lingua/protocols/phase_two_brief.md) · [Implementation and run instructions](../research/open_weight_lingua/README.md)

## Executive result

The pinned 128-group real-model pilot **completed** (run
`pilot-20260921T235825Z-6164d210`, independent auditor PASS) and the frozen
decision rule returned **STOP**. A language description of one
Qwen2.5-7B-Instruct residual-stream activation, reconstructed through the
released NLA verbalizer/reconstructor pair, **does not support limited
behavioral preservation** on this task: the one-sided 95% upper estimate of
the P2 accuracy loss versus the unmodified model is **41.4 percentage points**
against a frozen 5-point limit (P2 accuracy 0.5645 versus P0 0.8105). The
specificity control passed — P2 beats the wrong-description control P3 on
correct-answer log probability, lower estimate +3.80 — so the
description-derived direction is on-task but not preserving. The frozen
text-edit rule found **0 of 128** groups eligible against a floor of 32, so
the targeted-edit hypothesis is **untested**, not refuted. A no-more-than-budget
PCA baseline nearly matched the unmodified model (accuracy 0.8027, KL 0.0157):
generic reconstruction is competitive under the stated budget, which is not
evidence that language is useless for interpretability. Independently, the
projected locked-validation cost of ≈13.7 hours exceeds the eight-hour budget.
The phase closes with this documented bounded negative result; the locked
512-group validation was **not run** and stays unimplemented in this runner.

## 1. Concrete question

A frozen Qwen2.5-7B-Instruct reads a short straight-line program:

```text
x = 3
y = 8
x = x + 2
What is x? Reply with only the integer.
```

The answer is 5. The experiment captures one internal activation before the
model answers, asks a released activation verbalizer to describe it in
English, reconstructs a direction from that description with the paired
reconstructor, replaces the original vector, and lets the remaining
computation run. A paired counterfactual program changes the first assignment
to `x = 4`; the answer for `x` changes to 6 while the answer for `y` stays 8.

**Does the reconstructed activation preserve the answer distribution? If the
description explicitly states the current value of x, does a minimal edit to
that statement produce a specific, reproducible behavioral change?**

The first question was answered negatively at the frozen threshold. The second
could not be opened: no pilot description contained an editable current-value
statement under the frozen rule, so the edit hypothesis is preserved as
explicitly untested.

## 2. What transferred from Phase One, and what did not

Phase One closed with approximate numerical state preservation and typed edits
on **known mathematical fields** of a structured synthetic recurrent model,
plus a local linear response predictor of limited usefulness (30.9% less RMS
response-prediction error than predicting no change; close prediction not
met). Those findings are specific to that synthetic model. The
[Phase-One closeout](phase_one_technical_report.md) remains their source of
record.

Transferred methodology:

- The **preserve → intervene → continue → measure** loop: capture a state,
  replace it through a description channel, resume the original computation,
  and score behavior rather than plausibility.
- Mandatory **raw-restoration identity gates** before any reconstructed
  condition is interpreted (P1 here; the Phase-One raw-state gate there).
- **Paired counterfactuals and wrong-target controls** (P3 wrong description,
  wrong-variable edit design, raw donor from counterfactual B), mirroring the
  wrong-candidate control.
- **Intention-to-test accounting**: missing generations count against
  accuracy; failures never manufacture finite metric values.
- **Frozen thresholds and a bounded stopping rule**: one pilot, at most one
  locked validation, thresholds never weakened on held-out results, negative
  results preserved.
- An **independent auditor** that replays saved evidence without invoking the
  producer.

What did not transfer:

- **Named fields.** Qwen's hidden coordinates are learned, not named
  quantities with supplied semantics. There is no cycle basis, no `z = Q a`
  constraint to check, and no structural certificate. The interpreter
  establishes program behavior, not hidden-state semantics.
- **The frozen codec.** Phase One's compressor was a designed numerical
  encoding of known fields; here the description channel is an external
  pretrained NLA pair whose coverage and fidelity are the questions under
  test.
- **The response predictor.** No local linear/quadratic response rule was
  fitted; the Phase-One six-dimension machinery is not transplanted into a
  3,584-dimensional activation experiment.
- **Accuracy and compression claims.** Qwen is a conventional transformer,
  not a recurrent-depth experiment; nothing here reopens or extends the
  Phase-One ledger.

## 3. Architecture and information accounting

Let `h` be the target's native BF16 activation at the extraction site,
converted to float32 for measurement. Define `n = ||h||₂` (the retained norm),
`c = AV(h)` (the language description), and `r = AR(c)` (the reconstructed
direction). The installed replacement is:

```text
h′ = cast_native(n · r / ||r||₂)
```

with normalization computed in float32, nonfinite or near-zero `||r||₂`
rejected, and a single cast back to the native dtype. Direction diagnostics
use cosine similarity and the squared unit-direction distance
`||h/||h||₂ − r/||r||₂||₂² = 2(1 − cos)`; neither is raw activation MSE.

The pipeline, with the code that implements each step:

- **Target hook** ([target.py](../research/open_weight_lingua/src/open_weight_lingua/target.py)):
  the site is the residual-stream output of block index 20 at the last
  non-padding token **including the assistant-generation prefix**, confirmed
  against `hidden_states[layer+1]` rather than assumed. One site, native-dtype
  restoration, exception-safe hooks, cache-free full forwards; the patch stays
  at the original position on every recomputed pass during greedy decoding and
  teacher forcing.
- **AV embedding injection**
  ([nla_adapter.py](../research/open_weight_lingua/src/open_weight_lingua/nla_adapter.py)):
  the released `kitft/nla-qwen2.5-7b-L20-av` recipe with its exact loaded
  metadata, marker context and prompt template; cache-free injection with an
  A/B/A regression proving distinct embeddings produce distinct fresh
  computations. AV injection uses its own metadata-defined normalization
  scale, 150, which is not the norm used to reinstall the activation. One
  greedy description per activation, 200-new-token ceiling; truncation or a
  missing closing tag is a reported failure, never retried with answer
  guidance.
- **AR value head** (same module): the released
  `kitft/nla-qwen2.5-7b-L20-ar` recipe — a truncated backbone, final
  normalization removed from the reconstruction path, and a separate learned
  value head. The target's language-model head is never substituted.
- **Norm side channel**
  ([geometry.py](../research/open_weight_lingua/src/open_weight_lingua/geometry.py)):
  `n` is a retained per-example **four-byte** side channel. This is
  text-plus-norm reconstruction; the decoder receives description text and
  fixed metadata, not the target prompt, source activation, correct answer,
  donor activation or future states. Nobody claims the complete vector was
  recovered from text. A calibration-median-norm diagnostic (frozen median
  98.8137) exposes reliance on the channel: it matched P2 almost exactly
  (0.5605 vs 0.5645), so P2's residual accuracy does not rely on it.

Supporting machinery: [tasks.py](../research/open_weight_lingua/src/open_weight_lingua/tasks.py)
(explicit interpreter and paired groups), [splits.py](../research/open_weight_lingua/src/open_weight_lingua/splits.py)
(five deterministic disjoint splits), [controls.py](../research/open_weight_lingua/src/open_weight_lingua/controls.py)
(P4 PCA fit and byte-budget rank), [text_edits.py](../research/open_weight_lingua/src/open_weight_lingua/text_edits.py)
(frozen three-form current-value editor), [metrics.py](../research/open_weight_lingua/src/open_weight_lingua/metrics.py)
(strict integer scoring, multi-token answer plus EOS, float32 full-vocabulary
KL), [stats.py](../research/open_weight_lingua/src/open_weight_lingua/stats.py)
(whole-group bootstrap), [runner.py](../research/open_weight_lingua/src/open_weight_lingua/runner.py)
(stages and conditions), [audit.py](../research/open_weight_lingua/src/open_weight_lingua/audit.py)
(independent replay). The CPU fixture suite (128 tests on tiny random models)
checks software properties only; its outputs are never labeled released-model
measurements.

## 4. Numerics engineering: BF16 kernel reselection on GB10 sm_121

Two pre-pilot repairs are part of the scientific record. Both were frozen
**before** any pilot or validation outcome was inspected, and the runs they
superseded are preserved ([results §7](phase_two_results.md#7-superseded-runs-and-the-failed-first-attempt-preserved-explicitly)).

### Exact masking, and why the same-length gate is bitwise

Qwen's attention mask enters as a finite-minimum additive bias before a
float32 softmax, so masked positions receive exactly zero weight —
independently of kernel scheduling. Eight same-length pairs with different
suffix content were bitwise identical on GB10 BF16 and CPU fp32. The
same-length dummy-suffix bitwise comparison is therefore an exact
discriminator: a genuine suffix-content leak fails it loudly, while kernel
noise cannot occur at equal length. This gate discriminates a causal leak from
kernel-reduction noise only; it establishes nothing about the semantic content
of activations.

### The measured drift

Across **different** sequence lengths, cuBLAS re-selects BF16 GEMM kernels
with different reduction orders on sm_121; the reassociated sums differ at the
1e-2 norm-relative scale and compound across the 21 blocks up to the
extraction site. Measurements:

| Sample | Maximum norm-relative cross-length drift |
|---|---:|
| Smoke probe, 20 cross-length appends (smoke split) | 2.09e-2 – 3.09e-2; CPU fp32 collapses to ~4e-6; answer argmax stable 20/20 |
| Repaired-gate smoke rerun, 336 recorded samples | 3.55e-2 |
| First pilot attempt, pilot-0000-A-x at the divergence step | **0.258**; ~0.26 at every append +1…+5; pilot-0000-A-y 0.144–0.165; pilot-0001-A-x 0.019 then 0.122–0.127 |
| Per-block growth 64→66 tokens, pilot-0000-A-x | 0.063 / 0.065 / 0.104 / 0.084 / 0.258 at layers 0/5/10/15/20 |
| fp32 reload, same sweeps and layers | **~3.2e-6 — collapse** |

The fp32 collapse shows the prefix is mathematically length-independent: this
is numerical kernel-scheduling noise, not a semantic leak. But the noise has a
**prompt-conditioned heavy tail** that the 336-sample smoke justification
badly under-sampled.

### The falsified bound, corrected on record

Repair one (after the first smoke failed teacher-forced scoring) replaced the
unattainable `allclose(1e-5)` cross-length check with the same-length bitwise
gate plus a frozen `SUFFIX_DRIFT_BOUND = 1e-1` — justified from the smoke
maximum 3.09e-2 with a safety factor of about 3, frozen pre-pilot per brief
§4. The first pilot attempt then measured 0.258 on its first variant: the
bound's coverage claim was **falsified for the pilot distribution**, and that
falsification is preserved explicitly in the [claim ledger](claims.md).
Re-fitting a larger bound from pilot-distribution measurements was rejected:
it would fit a tolerance to the very data that broke it and weaken the bound's
wrong-vector-class discrimination. The bound stays frozen as an untouched
backstop, never re-fitted.

### Kernel-shape pinning

Repair two removes the drift class by construction: every target forward in
every stage is right-padded to the fixed bucket `TARGET_BUCKET = 128` (zeros,
`attention_mask` 0 on pads). Each link of the rationale was previously
measured: masked pads contribute exactly zero (the bitwise-proven same-length
control); same-shape forwards are bitwise deterministic on this backend; fp32
shows length-independence (~3.2e-6). With all GEMM shapes pinned, the
unpatched prefix vector is length-stable by construction. Greedy decoding and
teacher-forced scoring write generated/answer tokens into successive masked
pad slots, with argmax and log-probability positions derived from the mask;
site semantics are unchanged. The runner checks loudly before any inference
that every tokenized input plus the generation ceiling and answer suffix fits
the bucket (max observed prompt 79; 79 + 8 + 3 = 90 ≤ 128). The manifest
records `target_bucket`, `target_padding_policy` and `greedy_identity_gate`
verbatim; the greedy identity comparison became a recorded
`exact`/`drift_diverged` backstop, counted per stage, never a decision input.

In the pinned regime: every recorded `suffix_drift_relative` was exactly 0.0
(336 smoke samples; the pilot's per-score evidence), identity gates held
bitwise on all 512 pilot variants, and the greedy backstop recorded `exact`
512/512 in both stages. The affected pilot was rerun, visibly versioned
(`pilot-20260921T235825Z-6164d210`), per brief §11.

## 5. Frozen protocol

| Item | Fixed choice |
|---|---|
| Target / AV / AR | `Qwen/Qwen2.5-7B-Instruct`, `kitft/nla-qwen2.5-7b-L20-av`, `kitft/nla-qwen2.5-7b-L20-ar`, pinned in `configs/model-lock.json` (41,410,059,546 verified bytes); native released BF16 precision |
| Site | Block index 20 residual-stream output; last non-padding token including the assistant prefix |
| Splits | Smoke 8 → calibration 256 → pilot 128 → validation 512 (two 256-group blocks); deterministic, disjoint, carried exclusion inventories; plan hash `ff060012…` |
| Conditions | P0 unmodified; P1 reinsert original; P2 own description; P3 deranged description; P4 calibration-fitted PCA; P5 norm-matched random; raw donor; calibration-median-norm diagnostic; edited/wrong-variable only on eligible groups |
| Edit rule | Frozen version 1.0.0, sha256 `e1560d7e…`; three current-value forms, canonical values 0–19, value-span-only edits; truth agreement recorded, never gating |
| P4 rank rule | `rank = min(fitted_rank, floor(text_bytes/2))`, fitted rank 1,023, float16 coefficients; no-more-than-budget; fit identity `37af38ff…` |
| Thresholds (brief §8 design choices, locked before outcomes) | P0 accuracy ≥ 0.80 with intervention sensitivity; P2 loss one-sided 95% upper ≤ 5 pp and P2−P3 log-prob lower > 0; edit-eligible groups ≥ 32 |
| Uncertainty | Whole-group bootstrap, 3,000 resamples, fixed seed 203100; variants within a group are not independent |
| Accounting | Intention-to-test over all 512 planned variants; failures never manufacture finite KL |
| Budget | Eight hours per scientific stage; an over-budget projection stops for a scope decision before the stage opens |

A group counts as correct only when all four of its variants answer exactly;
the P2−P3 contrast uses per-group mean correct-answer log probability.
Logit identity tolerances `atol=1e-5, rtol=1e-5` held bitwise all run.
Answer formatting is the frozen canonical integer tokenization plus EOS,
appended without retokenizing the prefix.

## 6. Pilot results and the STOP decision

Per-condition metrics over the 512 planned variants (all conditions 512/512
valid, zero failures):

| Condition | Exact /512 | Accuracy | Mean valid next-token KL | Agreement with P0 |
|---|---:|---:|---:|---:|
| P0 unmodified | 415 | 0.8105 | 0.0 | 1.000 |
| P1 original reinserted | 415 | 0.8105 | 0.0 | 1.000 |
| P2 own description → AR + original norm | 289 | 0.5645 | 2.71 | 0.615 |
| P3 another group's description | 137 | 0.2676 | 7.38 | 0.299 |
| P4 PCA baseline (no-more-than-budget) | 411 | 0.8027 | 0.0157 | 0.967 |
| P5 norm-matched random | 3 | 0.0059 | 11.75 | 0.008 |
| Raw donor from counterfactual B | 402 | 0.7852 | 0.173 | 0.924 |
| Calibration-median-norm diagnostic | 287 | 0.5605 | 2.72 | 0.611 |

Frozen decision statistics (3,000 whole-group resamples, seed 203100,
independently replayed by the auditor): P2 accuracy loss point **34.4 pp**,
one-sided 95% upper **41.4 pp** (limit 5 pp — **not met**); P2−P3
correct-answer log probability point **+4.60**, lower **+3.80** (positive —
met); P4 loss point 0.78 pp, upper 3.13 pp (descriptive); editing effect
**untested** (no eligible group).

Decision derivation — gate 1 met (P0 0.8105 ≥ 0.80; sensitivity present: P5
changed 508/512 generations, donor KL 0.173 with 39/512 changed); gate 2 not
met (P2 loss upper 41.4 pp > 5 pp; the P2−P3 clause was met); gate 3 not met
(0/128 edit-eligible against the 32 floor; exclusions `absent` 128,
`ambiguous` 0; per-variable `x` absent 127 / eligible 1, `y` absent 128; the
single eligible parse stated 10 against reference 6 — recorded, never
gating). The frozen rule — any unmet criterion produces stop — yields **STOP**
with unmet criteria `p2_accuracy_loss` and `edit_eligible_groups`.
Independently, the projected validation cost is 47,759.3 s plus 1,494.2 s
calibration = **49,253.5 s ≈ 13.7 h > 28,800 s** (`within_budget: false`).
Measured pilot resources: wall 11,980.8 s (setup 155.0 s separate); stages
856.9 / 7,892.5 / 106.0 / 3,084.3 s; 112,032 forward calls (5,016 target
identity, 72,207 AV, 512 AR, 34,297 target behavior); GPU peaks 15.91 GB
allocated / 18.17 GB reserved.

## 7. What the result does and does not establish

The brief's §12 claim table governs interpretation; it is quoted verbatim,
annotated with what the pilot observed:

| Observation | Defensible interpretation | Prohibited upgrade |
|---|---|---|
| High cosine reconstruction | Similar activation direction. | The model's reasoning was recovered. |
| P2 preserves measured answers | Selected-vector reconstruction preserved behavior on this assay. | The whole hidden state or KV cache was compressed. |
| Correct edits outperform wrong-variable controls | Evidence for targeted influence under the specified intervention. | A unique causal variable or all neuron meanings were discovered. |
| Minimal edits are often unavailable | Limited explicit-variable coverage at this task/site. | Ground-truth text inserted manually was read from the activation. |
| Numerical baseline is as good or better | Generic reconstruction is competitive under the stated budget. | Language is useless for all interpretability. |
| Single-site patch is ineffective | This site/assay did not reveal a measurable contribution. | The model has no relevant internal representation. |
| Small stored description | Small selected-vector payload. | Low total cost despite AV/AR weights, norm, context, calls and metadata. |

Mapping the pilot onto these rows: the observed outcome for preservation is
the **failure** of the second row's premise at the frozen bound — P2 did not
preserve measured answers, so no preservation claim and no prohibited upgrade
arises. The edit row was **not reached**: zero eligible groups is exactly the
fourth row ("minimal edits are often unavailable" → limited explicit-variable
coverage at this task/site), and no manual ground-truth description is
presented as discovered semantics. The fifth row **was** observed (P4 near P0)
and carries only its defensible reading. The sensitivity controls rule out the
sixth row's strong reading at this site: patches clearly affect the intended
measurement. The seventh row stands as a cost warning: the description payload
is small, but total cost includes AV/AR weights (≈15.2 + 10.9 GB verified
artifacts), 112,032 forward calls, the retained norm, and shared metadata.

The pilot is a bounded assay at one checkpoint, one task family and one site —
a pilot, not a proof. Nothing here establishes semantic content, faithfulness,
whole-state compression, or a general language route to interpretability; the
positive specificity contrast shows only that the description-derived
direction is on-task. Qwen remains a conventional transformer; no
recurrent-depth, new-NLA or neuron-semantics claim transfers from Phase One or
from this phase.

## 8. Evidence map and what the audit can establish

| Artifact | Contents and identity |
|---|---|
| Pinned runs (local run directories) | `smoke-20260921T233021Z-cbe4057e`, `calibration-20260921T235017Z-fd111b21`, `pilot-20260921T235825Z-6164d210` under `research/open_weight_lingua/runs/`; superseded/failed runs retained alongside |
| Pilot manifest | Written before inference; all input identities, code/protocol/model hashes, frozen settings; sha256 `cb3ec24ff3187813eb0f2a25ee8397ec7bd94e5b6a8b615588c5615df38b6176`; git head `a3b4b4903ad3a9def4fe3e96faed7723c0ed749d`; plan hash `ff060012…` |
| Calibration fit | `baseline_fit.safetensors` identity `37af38fff3df81e016279582b20f179040196c23a84e24d2c9e8d2f8848fec79`; frozen median norm 98.8137; 1,024/1,024 extractions |
| Per-run files | `results.json` (every planned variant), `summary.json`/`report.md` (all-variant denominators, decision section), `compatibility.json`, `completion.json`, `inventory.json`, `wall_clock.json` (local) |
| Reports ZIPs | Reports-only packaging with explicit inventory; exclude raw vectors, weights and full-vocabulary logits |
| Local-only evidence | `raw/*.safetensors`: native originals, float32 norms, site/inputs/masks, AR directions, actual replacements, full-vocabulary next-token logits |
| Human records | [pilot decision](../research/open_weight_lingua/protocols/pilot_decision.md), [milestone one](../research/open_weight_lingua/reports/milestone_one.md), [milestone two](../research/open_weight_lingua/reports/milestone_two.md), [source/compatibility audit](../research/open_weight_lingua/reports/source_compatibility.md), [claim ledger](claims.md) |

The independent auditor (`audit.py`) replays each bundle: hash integrity,
saved counts, saved KL, and the frozen bootstrap statistics, without loading
models or invoking the producer's code. All three pinned runs PASS. The
auditor **cannot** authenticate execution, regenerate teacher-forced
probabilities from the raw logits (they stay local, so the reports-only ZIP
cannot independently replay KL), or verify model weights beyond their pinned
hashes. Hashes identify artifacts; they do not authenticate remote execution
or prove when the protocol was registered. The numbers in this report were
cross-checked against the pilot's `summary.json`, `completion.json`,
`compatibility.json`, `wall_clock.json` and `report.md`, and the calibration
and smoke runs' `completion.json`/`summary.json`.

## 9. Future directions

Two companion analyses of record accompany this closeout under
`research/open_weight_lingua/reports/`.
[NLA ecosystem notes](../research/open_weight_lingua/reports/nla_ecosystem_notes.md)
(landed 2026-09-22) surveys what else exists in the released NLA ecosystem and
what each follow-up option would cost.
[Pilot description analysis](../research/open_weight_lingua/reports/pilot_description_analysis.md)
(landed 2026-09-22) examines what the 512 pilot descriptions actually
contained, including why the frozen current-value forms never matched: the
verbalizer narrates structure and answer shapes rather than per-variable
values, so the coverage failure is semantic, not phrasing. Any further
experimental work — a different checkpoint, site, layer, task, editor, or the
locked validation — is a separate research decision with its own brief, budget
and stopping rule, not a continuation of this closed phase.

## 10. Provenance and reproducibility

| Source | Identity / scope |
|---|---|
| Measurement host | NVIDIA DGX Spark (GB10, sm_121, unified memory), aarch64 |
| Software | Python 3.11.14, torch 2.10.0+cu130, transformers 4.57.6, numpy 2.2.6, safetensors 0.7.0 (per-run `compatibility.json`) |
| Run dates | 2026-09-21 (all runs); closeout documentation 2026-09-22 |
| Branch | `phase-two-milestone-two`; pilot manifest git head `a3b4b4903ad3a9def4fe3e96faed7723c0ed749d` |
| Model artifacts | 41,410,059,546 pinned bytes, hash-verified on every load; revisions locked in `configs/model-lock.json` |

Run directories, model caches and raw activations are local and ignored; no
weights, credentials or raw activation data are published here. Reproduction
commands and the full evidence contract are in the
[implementation README](../research/open_weight_lingua/README.md). Earlier
Phase-One statements are summarized from their existing audits, not newly
re-audited for this page. A new run is not required to close this phase; the
stopping rule forbids an automatic follow-on.
