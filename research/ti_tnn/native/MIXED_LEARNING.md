# Corrected mixed text/code comparison: draft work order

Given the beginning of a Python file or encyclopedia paragraph, can a small native
model predict a useful continuation? This phase returns language/code learning to
the primary role. A fixed register suite remains a diagnostic; perfect synthetic
execution is not a gate for ordinary language-model experiments.

**Status: CPU preparation, not execution authorization.** The four engineering
architectures and the initial training objective stay fixed. No model, activation,
tokenizer, router, curriculum, shaping or stopping mechanism is redesigned here.

## What the existing evidence supports

The original [mixed bring-up](REPORT.md) completed 20,001,877 loss-bearing tokens
per arm. Reading its saved learning logs again gives the following dev losses;
these are the original bounded mixed dev population, not a new evaluation:

| Mode | Update 7,000 | Update 8,000 | Update 9,000 | Endpoint 9,819 |
|---|---:|---:|---:|---:|
| Dense | 4.15849 | 4.11769 | 4.10018 | 4.08277 |
| Shared | 4.24143 | 4.20116 | 4.18346 | 4.17114 |
| Untyped | 4.24019 | 4.19676 | 4.17981 | 4.16749 |
| Typed | 4.21574 | 4.16794 | 4.15000 | 4.13586 |

The curves were still improving, with diminishing gains, under the original
schedule. That motivates a proposed 40M-token development comparison; it does not
predict a capability threshold or establish an optimal budget. Historical teacher-
forced source losses, failed code suffixes, numerical failures and costs stay intact.
The corrected run would not be an isolated causal estimate of a single historical
bug fix: preparation and evaluation corrections are applied together, and its token
budget is larger. The primary comparison is between its four contemporaneous arms.

The completed [initial-state study](INITIAL_STATE_RESULTS.md) remains a separate
negative development result. Multi-state training failed to acquire complete answers
at the registered endpoint. Each distinct multi-state pair had one eighth the
repetitions of the one-state pair. Held-out numeric CE and some register behavior
improved. This was not a matched-convergence test and does not establish that state
diversity is universally ineffective. Readout quality, answer correctness and causal
state use remain separate claims.

## Corrections relative to the original mixed-data run

| Observed issue | Corrected preparation/evaluation | Existing component reused |
|---|---|---|
| 8,504 train structured examples spanned rows; continuation rows lacked prerequisites | Keep each complete structured example in one row; audit over-context exclusions | `data.pack_runs(packing="intact")`, validated in the [follow-up](FOLLOWUP.md) |
| A final token quota can still cut a structured example before packing | Stop before that example and record the small whole-example underfill | New bounded selection in `mixed_learning.py`; legacy allocator unchanged |
| Row-relative learned positions differed from standalone prompt positions | Reset positions at every document and row boundary; padding position zero | Existing `position_policy="document"`, [acquisition correction](ACQUISITION.md) |
| Old after-state placement lacked a direct later-token cache path | Preserve the already corrected before/after/after/after target schedule in every mode | Current `structured.align_example`; no new loss or neural pathway |
| First 128 packed dev rows can concentrate on a few long documents | Add hash-selected, source-separated 128-document CE populations, one prefix per document | Existing teacher-forced evaluator; new bounded selection only |
| All 64 original code prompts were left-truncated, and final-line exactness was a narrow metric | Add fixed 128-token document beginnings with 32-token continuations, for both text and code | Existing neural greedy decoder; new saved-token scorer |

Ordinary text/code longer than the 256-token context remains chunked. A chunk's
earlier document context is not supplied; no full-file comprehension claim follows.
Register prerequisites are kept intact because they define the reference task.
The original GPT-2 tokenizer, pinned Wikipedia/CodeParrot files, license-label
filter, exact-content deduplication, 1% document dev split and 45/45/10 mixture
remain. Exact deduplication does not exclude near duplicates or semantic overlap.
All raw corpora, arrays, generated strings and weights stay local.

## Fixed comparison

Use fresh random initialization with model/data seed **20261108**, matching the
original declared development seed. Each arm consumes the same prepared arrays
and shuffled batches. Modes remain dense, shared recurrent, untyped MoE and typed
MoE, with the unchanged engineering configuration: vocabulary 50,257, context 256,
hidden width 192, four heads, slot width 48, FF width 768, four passes/slots/experts.
Unique parameter counts remain 21,860,352 / 20,084,544 / 21,415,300 / 21,499,780.

Dense/shared distinguishes recurrent weight sharing; shared/untyped tests the
generic specialist organization; untyped/typed tests the typed interface/routing
combination. These contrasts do not simultaneously match parameter count and
executed cost. Attention, residuals, nonlinear experts, trainable routing and
end-to-end learning remain present. The [architecture](ARCHITECTURE.md) gives the
equations and primary Transformer, Universal Transformer and MoE references.

Keep full-sequence next-token loss, route/state weights 0.1/0.1 and balance weight
0.01. Ordinary text/code has no semantic route/state labels. AdamW remains LR
0.0003, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01, clipping 1.0, batch 8,
warmup fraction 0.02 and final LR fraction 0.1. No answer-only treatment or loss sweep.

Propose **40,000,000 requested full-label tokens per arm**, 160M across all four.
Atomic batches stop at the first count at or above the requested budget; the exact
shared overshoot/update/cursor is derived from the existing sampler during CPU
preparation. Training input allocation is 44M tokens, dev 1M before boundary masks,
with only documented whole-program underfill. No source cycling is introduced.
Checkpoint every 500 updates; periodic dev every 1,000 updates on the fixed 128-row
training-dev screen. Select the final endpoint only. Do not shorten a control,
choose an intermediate checkpoint, or extend based on the diagnostics.

The completed CPU build fixes **40,001,786 loss tokens, 19,862 updates and cursor
158,896 per arm**. This is 160,007,144 loss tokens and 79,448 updates across the
comparison. Each arm consumes 40,206,149 real input tokens and 40,677,376 padded
slots; its loss tokens comprise 18,036,675 Wikipedia, 18,011,928 code and 3,953,183
structured targets. It visits less than one full pass of the 173,894 prepared rows.

Preparation manifest SHA256:
`592017bbf9efbc1888791c19c6750cda3e64641cd238aa11325ab0d60a8484c0`.
The immutable register-monitor protocol remains
`bd108e03f51aefbc5bf8ce1cc0086c874d450b79ecf5668a23b275d1f37f50df`.
Train/dev whole-program underfill is 74/58 input tokens respectively. Their actual
loss counts are 43,776,067 and 994,786. Source token fractions and padding are recorded
separately; a pre-mask 45/45/10 mixture does not imply exact post-mask fractions.

## Evaluation inventory and distinct claims

For source \(s\), report \(L_s=-\sum_{i\in I_s}\log p_\theta(x_i\mid x_{<i})/N_s\),
where \(I_s\) contains that source's nonmasked shifted target positions, \(N_s=|I_s|\),
\(x_i\) is the observed token, and \(\theta\) denotes the trained weights. Report
the sum and count alongside CE and \(\exp(L_s)\). Context is limited to the visible
document prefix or chunk. No cross-source average substitutes for separate results.

| Claim family | Fixed population and endpoint outputs | Limit |
|---|---|---|
| Language/code capability, primary | 128 hash-selected held-out Wikipedia documents and 128 code documents; up to 256 beginning tokens each; full teacher-forced source CE/token accuracy/counts | Development prefixes, not full-corpus, executable-code or general capability evidence |
| Language/code continuation, secondary | 64 documents/source; 128 prompt tokens, exactly 32 possible continuation tokens; save prompt-only generation, 32-token exact match and token agreement with missing/malformed counts | Multiple natural continuations are valid; exact reference match is descriptive, not the sole capability criterion |
| Synthetic transfer | Reuse the immutable #80 multi-state-side nine populations, each 128 examples; 48-token bound; complete answers, changed/unchanged and per-register scores, malformed/EOS/truncation | Historical diagnostic monitoring, not new confirmation. Its `train_anchor`/`train_sample` names refer to #80; neither is mixed-run training fit |
| State readout | Same fixed nine populations; all four passes and slots with explicit counts; answer-value/format/EOS CE separately | Supervised decodability, not proof generation causally uses state |
| Causal interpretation | No new intervention, tensor edit or mechanism experiment in this work order | Deferred to separately authorized research; readout and answers cannot establish causal use |
| Numerical compatibility | Existing bounded CPU FP32 replay, exact applicable routes/tokens, scale and strict-allclose fields retained | Not bitwise or general BF16-serving equivalence |
| Executed cost | Actual cumulative reservation wall, training/checkpoint/evaluation timers, tokens, padding, memory and failed/recovery costs | Proposed estimates are not measured execution |

All populations are frozen before outputs are generated. The corpus split is reused
as development data. The register program/state pairs are checked against the full
mixed structured training/dev inventory and must be absent. No historic evaluation
example is relabeled as new confirmation. No perfect-register threshold blocks
language/code evaluation or is used for stopping/checkpoint selection.

The prepared capability CE denominators are **28,877 Wikipedia tokens** and
**32,640 code tokens**, each across 128 documents. The two continuation populations
each retain 64 examples and 2,048 token decisions, regardless of short, malformed
or missing outputs. All 1,152 fixed register examples remain separate by population.

## Complete-run resource proposal

The live ledger at preparation records **7.146698220701847 / 12 native hours used**,
leaving **4.853301779298153 hours**. No new reservation was created. Propose a
separately authorized **4.75-hour cumulative maximum from that balance**, including
profiling, failures, endpoint evaluation and explicit recovery. It does not increase
the native cap. A full charge would leave **0.103301779298153 hours**.

The original clean engineering profiles measured 27,355.23 / 27,922.86 / 25,726.43 /
25,277.47 loss tokens/s for dense/shared/untyped/typed, including sampler/hash/logging
costs. The projection uses the existing `study.project_arm`, scaling those rates by
the ratio of new to old loss-bearing occupancy per padded slot (0.98867396). This is
a transfer estimate, not new throughput measured on the corrected data.

| Mode | Transferred loss tokens/s | Training, checkpoints, periodic dev, endpoint evaluation and replay, seconds |
|---|---:|---:|
| Dense | 27,045.40 | 2,818.919 |
| Shared | 27,606.60 | 2,749.430 |
| Untyped | 25,435.06 | 2,948.287 |
| Typed | 24,991.17 | 2,952.991 |

Every arm retains 41 checkpoints (initial, 39 periodic, final) and 20 complete
periodic/final training-dev passes, each with 32,455 eligible targets. The projection
charges all checkpoint writes at the measured per-mode rates. Final capability CE
is charged at twice the old 128-row teacher-forced evaluation cost. The 128 new
language/code continuations are conservatively charged at twice the old 64-example,
128-new-token code cost, although the new generation bound is only 32 tokens.
The fixed register generation is charged at 2.25 times the old 512-example cost.
Each arm additionally gets explicit **120 s for answer CE/state readouts** and
**60 s for saved scoring, hashes and logging**. These two entries are allowances,
not fabricated measurements. No state intervention is budgeted or proposed.

The complete raw comparison is **3.186008 h**. A **35% margin** plus **0.15 h for
fresh profiling/preflight** and **0.05 h shutdown allowance** gives **4.501111 h**.
Margin/recovery belongs inside the proposed 4.75 h maximum. If fresh measured
costs or failures no longer permit all four fixed endpoints, stop for owner review;
do not quietly reduce tokens, populations or controls. A second seed or larger
model would need a separate, explicitly budgeted work order.

Retaining all 164 planned checkpoints projects **41,761,987,542 bytes**. Prepared
artifacts occupy **3,407,625,270 bytes**; reserve a further 8 GiB for logs, exports and
recovery bookkeeping, and require at least 64 GiB free before execution. Model
memory planning uses a conservative 12 GiB allowance; this is not a new measured
peak. Old measured memory/cost values in `REPORT.md` remain historical.

Actual work so far: **318.525 s of CPU-side preparation wall time and zero GPU-
reservation hours**. Saved-array audits, tests and review time are separate CPU
work. Neither this projection nor a passing CPU test is an executed learning result.

## Runnable CPU preparation and missing-piece tests

With the pinned native interpreter, CUDA hidden and `PYTHONPATH=research/ti_tnn`:

```bash
python -m native.mixed_learning --cache "$PINNED_LOCAL_CACHE" \
  --historical-data "$ORIGINAL_MIXED_DATA" \
  --register-preparation "$PRESERVED_INITIAL_STATE_V4" --output "$NEW_PREPARATION"
python -m pytest -q -p no:cacheprovider tests/test_ti_tnn_native_mixed_learning.py
```

The preparation refuses an existing destination and uses only verified local
source files. It does not import or call a reservation, train a model, or download
data. It tokenizes only finite source prefixes and reuses the existing packer and
sampler. `evaluate_mixed_learning.py` supplies the future endpoint capability
adapter and saved-output scorer; it does not implement another scheduler.

CPU tests cover complete structured prerequisites at the quota, finite shortfall,
document-local positions, auxiliary target isolation, independent capability rows,
fixed continuation bounds, missing/malformed output denominators, duplicate ids,
gold-field rejection at inference and refusal to overwrite preparation.

Locally run with the pinned environment and CUDA hidden: **423 native CPU tests
passed in 19.22 s**, including 11 new mixed-preparation/evaluation cases. The release
check passed. Endpoint adapter tests use CPU fixtures and preserve incomplete
capability or register evaluations explicitly. Existing tiny synthetic regression
training tests are engineering fixtures; no research training or saved-checkpoint
inference was performed by this preparation task.

The completed local audit verified all **191 artifact hashes**, all five preparation
source hashes, unchanged historical dev document membership, zero packed train/dev
content overlap and 54,820/1,247 intact structured train/dev examples. All copied
register-monitor files are byte-identical to the frozen #80 preparation. This is a
local saved-array check, not independent external replication or a model result.

## Execution and recovery boundary

This work order is a proposal. A launch binding must name the reviewed commit,
prepared manifest and all data/tokenizer/model/sampler identities, pinned environment,
canonical comparison output and one cumulative allocation. The existing shared
ledger/lock, reservation watchdog, checkpoint journals and explicit recovery remain
mandatory. Historical profile identity checks must not be bypassed or relabeled as
new profiles. Transfer projections below are planning evidence, not fresh profile
measurements or a passed launch preflight.

Use the existing training and checkpoint recovery machinery for every arm. Failed
attempts retain their costs; explicit recovery continues the same comparison and
allowance from a valid journaled checkpoint. Preserve incomplete/failed arms and
every attempt. Duplicate comparison admission must fail closed. If the complete
remaining comparison no longer fits, stop and request a revised owner work order;
do not cut the endpoint or omit a control. A later thin execution binding must be
reviewed before owner authorization; none is invoked by these preparation commands.

No GPU launch, larger-model campaign, shaping, WSP or automatic merge is authorized.
