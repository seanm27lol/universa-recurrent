# Same program, several initial states: draft learning protocol

For `copy r0 r1`, seeing one initial vector repeatedly allows memorizing one
answer. Seeing that same instruction with eight different vectors requires
different answers. The proposed experiment changes that diversity while keeping
the program patterns and training exposure fixed.

**Composition contract v2, with a CPU-tested bounded runner; not authorized or launched.** This is a new development
experiment with fresh random weights, not an extension of any completed #79 child.
Its diagnostic basis is the [saved-output analysis](DEPENDENCY_DIAGNOSTIC.md):
even untouched high-valued registers fail, while deeper dependencies perform
poorly on ordinary development. No mechanistic explanation follows from those
reference categories.

## Hypothesis and controlled contrast

Let `P` be an exact instruction sequence, including register operands and set
immediates, and `x` an initial four-register vector. The reference transition
function is `F_P(x)`; its output is serialized as four canonical assignments.
For each training pattern, sample eight distinct initial vectors `x_0,...,x_7`
with coordinates in `[0,24)`. At each of eight matched document slots:

- **repeat_one:** show `(P, x_0, F_P(x_0))` eight times.
- **multi_state:** show `(P, x_i, F_P(x_i))` once for each `i=0,...,7`.

Both repeat that fixed data for 16 epochs. This changes input/state/answer values,
and their diversity, as intended. It does not isolate diversity from the resulting
value-frequency distribution. There is no staged curriculum or loss treatment.
Same-range value transfer tests the proposed benefit without conflating it with
the stronger high-initial-value role shift.

Training has 256 exact patterns: 64 one-instruction transitions and 192 composed
programs of length 2–4. The primitive inventory contains all 16 source/destination
pairs for each of copy/add/multiply, plus 16 set instructions (four sampled
immediates per destination). Every composed train/evaluation program uses only
these primitives. The final read graph of each composed training program has at
least one target of depth two or greater. The 2,048 document slots and program
order match across treatments; only the initial-state choice changes. One quarter
of slots teach individual transitions, and three quarters teach composition.
This mixture is fixed from the beginning in both treatments.

Eight proposed runs are the two conditions crossed with **dense, shared recurrent,
untyped recurrent MoE, and typed recurrent MoE**. Within each architecture, use the
same fresh model seed and sampler seed. The shared control is required for any
discussion of routing versus recurrent sharing. Parameter counts still differ:
21,860,352 dense; 20,084,544 shared; 21,415,300 untyped; 21,499,780 typed. Equal
exposure does not make these architectures parameter- or wall-time-matched.

Keep `configs/engineering.json`, GPT-2 tokenizer revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, context 256, four passes, causal
attention, residuals, learned nonlinear experts, and end-to-end trainable routing.
Use document-local positions, intact prerequisites, blocked cross-document
attention, and all existing auxiliary supervision. Main loss is **full_sequence**
throughout. Balance/route/state weights remain 0.01/0.1/0.1. AdamW starts at
`3e-4`, weight decay 0.01, betas `(0.9,0.999)`, epsilon `1e-8`, gradient clip 1;
the existing scheduler uses 2% warmup and cosine decay to 10% of the initial LR.
These are the existing fresh-training settings, not the inherited-parent #79 LR.

The data seed is 2026111201, model seed 2026111202, and sampler seed 2026111203.
Any change to these or to the hypothesis/data/objective requires a new preparation
identity before launch. Do not select seeds after looking at neural scores.

## Populations and evaluation

Each row has 128 complete-answer observations and 512 register positions.
Populations remain separate. Two state samples of one program are correlated;
counts are not 128 independent program-pattern replications.

| Population | What is tested |
|---|---|
| `train_anchor` | Same 64 primitive and 64 composed training patterns, with their shared `x_0`; fitting only |
| `train_sample` | Same selected patterns, using `x_0` for repeat_one and `x_1` for multi_state; treatment-specific fitting only. Repeat_one overlaps train_anchor completely; never pool them |
| `transition_values_id` | 64 training primitive patterns × two fresh low initial states; individual transition transfer |
| `transition_values_ood` | The same 64 primitive patterns × two fresh states with every initial coordinate in `[24,32)` |
| `program_values_id` | 64 training composed patterns × two fresh low states; value transfer with familiar compositions |
| `program_values_ood` | The same 64 composed patterns × two fresh high states |
| `dev` | 128 new length-2–4 composed patterns, low states, excluding reserved adjacency |
| `composition` | 128 new length-2–4 patterns ending in multiply then add; multiplication output has functional initial support and the actual initial state witnesses a change in the final add target when multiplication is skipped |
| `longer` | 128 new length-6–8 patterns, low states, excluding reserved adjacency; length extrapolation diagnostic |

The reserved adjacent multiply→add pair is absent from all other populations.
Composition contract v2 corrects a limitation in v1. For example,
`set r0 0; multiply r0 r1; add r2 r0` has a multiplication in the final read
graph and a final `r2` that depends on its initial value, yet skipping that
multiplication never changes the answer. A reference read path alone is insufficient.

For a program `P`, let `i` be the index of its penultimate multiplication, `M_i(x)`
its written value at initial vector `x`, and `j` the distinct destination of the
terminal add that reads this multiplication. All arithmetic is modulo 32 and
`x` ranges over `{0,...,31}^4`. Require both:

1. **Functional multiplication output:** there exist `x` and `x'` differing in
   one initial coordinate with `M_i(x) != M_i(x')`. Exhaustive finite-domain
   evaluation checks this, including zero annihilation and modular cancellation.
2. **Actual-state ablation witness:** at the example's recorded initial vector
   `x*`, `F_P(x*)[j] != F_skip_i(P)(x*)[j]`. The ablation skips only multiplication
   write `i`, retaining that destination's previous value, and executes every
   other instruction normally. Record the initial vector, both complete reference
   final vectors, index `i`, target `j`, and functional initial support.

The witness uses the selected example's initial state, not a different convenient
state. This separately rejects a nonconstant but identity multiplication such as
`set r1 1; multiply r0 r1; add r2 r0`, and a normally effective multiplication
that happens to have no effect at the chosen state. These are reference-program
interventions used only for selection, not neural interventions or WSP.

Audit of the preserved v1 population found 6/128 constant multiplication outputs,
21/128 without an actual-state witness, and 26/128 failing their union. Independent
exhaustive checking also found one globally ineffective skip-write multiplication.
V2 retains 102 examples and replaces 26 using a purpose-tagged deterministic RNG
stream derived from the unchanged data seed, after all original populations have
been drawn. It retains 128 observations; no model performance enters selection.
The same revised composition population serves both conditions. All training and
noncomposition artifacts, pattern order, seeds, objective, and endpoint are unchanged.
An independent checker exhaustively verified all 128 revised examples and every
recorded witness. This establishes task selection, not neural composition.

High values are held out **as initial values**; 24–31 may occur in set constants
and intermediate/final values during training.

The preparation command requires an explicit private inventory of historical
structured JSONL files. The local inventory covers original native data, the
followup bundle, both #77 acquisition data roots and the #78/#79 populations:
17 files and **76,288 unique historical program/state pairs**. Followup exports
are 128-example subsets of original sources, not byte-identical copies. Their
files are nevertheless included explicitly. Every new training/heldout pair and
prompt is excluded against that inventory. New dev/composition/longer instruction
patterns are also excluded against historical patterns. Missing files fail closed.
This establishes exclusion relative to the named inventory, not unknown datasets.

Historical records are exclusion inputs only. They are never appended to training
or used as new confirmation data. New data remain **development** data in this
single-seed protocol; fresh examples alone do not make a confirmation study.

At the fixed endpoint, use complete cached greedy generation, at most 48 new
tokens, EOS stopping, and error on an oversized prompt. Neither target length nor
executor state controls decoding. The model-facing export contains only an
example ID and prompt. The symbolic executor is used in data construction and
independent post-generation scoring; it is never an inference component.

Report complete exact answers, all-register and valid-only register accuracy,
changed-register versus unchanged-register accuracy (also per register),
format validity, EOS/truncation, final-write/dependency strata, numeric-value CE,
formatting/EOS CE, and separate state readouts. Free generation and teacher-forced
CE must remain distinct. Malformed strings fail all four register positions;
missing records make an evaluation incomplete rather than shrinking its denominator.
Exact lookup, no-op and supplied-program execution are separately labeled
construction baselines. Lookups score 128/128 on each training selection and
0/128 on each holdout; execution validates all reference targets. These are not
neural results. Numeric error is valid-only, never substituted for CE.

No writer intervention, Goodfire-inspired shaping or WSP is part of this treatment.

## Runnable preparation and what was tested

[`initial_state_study.py`](initial_state_study.py) generates and audits the paired
populations. [`prepare_initial_state_study.py`](prepare_initial_state_study.py)
reuses the existing tokenizer loader, intact packer and packing audit. It emits
train/dev arrays compatible with `PackedDataset`, nine evaluation/target-only
array sets, prompt-only generation exports, construction baselines, source/artifact
hashes and a draft protocol. It has **no execute or reservation option**.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research/ti_tnn \
  "$PY" -m native.prepare_initial_state_study \
  --output "$NEW_PREPARATION" --tokenizer "$TOKENIZER" \
  --history-inventory "$PRIVATE_HISTORY_JSON" --reference-bundle "$BUNDLE" \
  --previous-preparation "$PRESERVED_V1" --execution-output "$CANONICAL_FUTURE_RUN"
```

`PRIVATE_HISTORY_JSON` contains `{"files": ["/explicit/structured/source.jsonl", ...]}`.
All generated datasets and operational manifests stay outside the public repository.
Existing output directories are refused, including incomplete preparations.

[`score_initial_state_study.py`](independent/score_initial_state_study.py) prepares
the independent saved-output evaluation interface. After separately authorized
generation, each prediction must supply `example_id`, `completion_text`, and
boolean `stopped_on_eos`, `truncated`, `prompt_truncated` fields:

```bash
PYTHONPATH=research/ti_tnn "$PY" -m native.independent.score_initial_state_study \
  --source "$PREP/CONDITION/evaluation/POPULATION.jsonl" \
  --predictions "$SAVED_PREDICTIONS" --output "$NEW_SCORE_JSON"
```

The source target joins only after generation. The scorer refuses missing, extra,
duplicate or wrong-target predictions and never imports the producer's executor
or model. Synthetic tests exercise malformed predictions and known correct answers;
they are construction checks, not neural performance.

The original v1 pinned-tokenizer build was performed locally and remains preserved. Both treatments have 711
packed train rows, 163,960 input tokens, 161,912 full-loss tokens and 18,056 padding
tokens per epoch. No example was split, dropped or truncated. The actual paired
position/document/padding/source/route arrays and state masks agree exactly.
State **values** differ as the treatment requires. All 1,424 planned sampler
batches have identical row indexes and exposure counts across conditions.
All 263 prepared artifact hashes and 15 source hashes were checked. Preparation
used about 5.13 CPU seconds before final protocol/hash bookkeeping and occupies
about 50.2 MB. No optimizer update or neural generation was run for this preparation.

Preserved v1 protocol SHA256:
`5e2524b8c7f54b08533fdce501e710b48f0fe23127a6493d9a877b5727cf802f`.
V2 emits a new protocol and output directory. Its preview contains 321 artifacts,
including composition witnesses, fixed replay inputs and evaluation-only answer
masks. The revision checks every old artifact against its recorded hash and verifies
233 old artifact files byte-for-byte; the other 30 old files change only for
composition or aggregate composition metadata. The final protocol additionally
binds the exact clean source commit, native/scaling source hashes, root dependency
lock, all 58 installed pins, tokenizer, canonical output directory, and all eight
complete training configurations (including source/data/model/sampler identities).
The final protocol hash and exact local command are in the private owner handoff.
Timing fields make protocol file hashes vary on a new build; data stay deterministic.

## Endpoint, decision and recovery

Each of the eight runs finishes exactly 16 epochs, **1,424 updates**, cursor
11,376 and **2,590,592 full-label/main-loss tokens**. Input tokens are 2,623,360;
padded computation covers 2,912,256 token slots. Answer+EOS positions are 557,056
(131,072 numeric and 425,984 formatting/EOS), as diagnostics within full loss.
All eight runs total 11,392 updates and 20,724,736 full-label tokens. The final
partial batch is retained. Selection is the fixed final checkpoint only.

Save initial, update-500, update-1000 and final checkpoints. Periodic dev CE is
logged every 500 updates, plus the final pass, with no selection or early stopping.
Save every failure and budget-exhausted status. A low training score is a result;
do not extend the endpoint to obtain fitting. Underfitting makes a negative
generalization result harder to interpret, not permission to train longer.

Predeclare a **development screen**, not a significance test: within each of dense
and shared, multi_state must gain at least 8/128 complete answers on both in-range
value populations (`transition_values_id`, `program_values_id`), without a decrease
on ordinary dev, to recommend further study of value reuse. An additional gain of
at least 8/128 on composition in both controls is required to recommend studying
composition transfer. Report each condition and every population regardless of
this screen. MoE pairs test whether the direction also appears there; no routing
advantage is inferred from dense-versus-MoE alone. High-value and longer results
are separate diagnostics and cannot be pooled to manufacture a pass. These small,
single-seed comparisons need later fresh-data/multiple-seed work for stronger claims.

[`run_initial_state_study.py`](run_initial_state_study.py) binds this protocol to
`native.ledger.Reservation`, `native.train`, and the existing checkpoint journal.
The historical `run_program_study` and `run_continuation` contracts are unchanged.
The default dry run validates source, installed pins, all data, and eight sampler
endpoints, then reads the live anchored ledger. It creates no reservation, training
output, claim, or lock. There is no new scheduler or watchdog.

```bash
# Read-only dry run. Values come from the private frozen owner handoff.
PYTHONPATH=research/ti_tnn "$PY" -m native.run_initial_state_study \
  --preparation "$PREP" --tokenizer "$TOKENIZER" \
  --expected-protocol-sha256 "$PROTOCOL_SHA256"
# Only after separate explicit owner approval, append:
# --execute --authorize-hours 4
# A later explicitly requested recovery also requires --resume.
```

The canonical comparison directory is claimed once and locked exclusively.
Alternative protocol/output identities with the same study ID are rejected against
the ledger. A duplicate launch is refused; explicit resume validates completed
artifacts before skipping them. Completion is rechecked under the lock. All eight
arms must have the exact final endpoint and all nine complete evaluations before
the unchanged development screen can run. Failed/missing arms remain incomplete.

This study's checkpoint configuration has a stable owner binding; reservation
attempt IDs remain in logs/results/ledger without changing checkpoint identity.
An interrupted final-checkpoint/result-write gap adopts the validated journaled
final descriptor without training or regenerating the checkpoint. An unjournaled
or invalid final fails closed. Failed evaluations get a new attempt directory;
prior outputs remain intact and completed training is reused.

The evaluator reuses existing prompt-only neural generation, full-population
answer-component CE, supervised state readout, and CPU FP32 reconstruction APIs.
Its fixed replay checks two 16-token teacher-forced prefixes and two complete
prompts with four generated tokens. Scale-based agreement and exact route/token
checks remain bounded; strict-allclose failures stay in the report. This is not
BF16-serving equivalence or a new scientific replication.

Resume only from a hash-validated checkpoint with identical source, data, model,
optimizer, scheduler, RNG and sampler identities. Never reset the cumulative ledger
or invent an end event. Every failed attempt, replayed update, loading interval,
evaluation and recovery consumes the same allocation. If insufficient time remains
to complete all controls, stop and report incomplete; do not promote a partial
comparison. Do not kill foreign processes or alter historical worktrees.

## Measured resource proposal, not permission

Use the slowest measured #79 **full-sequence** throughput, 19,631.4068 loss
tokens/s. For each new run, training seconds are new full-label tokens divided by
that rate. This timer already includes data, forward/routing, backward and optimizer.
Carry the entire largest unclassified training-invocation overhead (205.996 s)
as a fixed floor, rather than shrinking loading/logging/dev costs with token count.
Budget four checkpoint writes at the largest 23-file average (1.45483 s/write).
Scale the slowest official evaluation by 1,152/830 saved outputs, and retain the
largest full-sequence answer-diagnostic cost. The original official evaluator's
combined generation/CE/state cost is not artificially separated into invented timers.

| Per run component | Dense/untyped/typed planning seconds | Shared proxy seconds |
|---|---:|---:|
| Data + training computation | 131.962 | 164.952 |
| Loading, logging, periodic dev, remaining invocation overhead | 205.996 | 257.495 |
| Four checkpoint writes | 5.819 | 7.274 |
| Generation + selected CE + state readout | 616.610 | 770.763 |
| Answer component diagnostics | 66.690 | 83.363 |
| Independent scoring/dependencies allowance | 10.000 | 12.500 |
| Source/data/checkpoint identity revalidation allowance | 10.000 | 12.500 |
| Bounded CPU FP32 reconstruction allowance | 10.000 | 12.500 |
| Total | **1,057.077** | **1,321.347** |

No comparable new shared-arm timing exists. Its estimate is explicitly a slowest-arm
proxy plus 25%, not a measured rate. All modes still receive the same endpoint.
Across two conditions, plus measured preprocessing and 60 s launch/final
hash/export allowance, the v2 preview projection is **2.514792 hours**, including 8.09 s measured CPU
preparation. A 1.5× watchdog margin is 3.772187 hours; adding 0.05 h shutdown
allowance gives 3.822187 hours. Final measured preparation time is bound in the
final protocol and changes these estimates slightly. The rounded
**proposal is a 4.0-hour maximum**, covering all eight runs and recovery. CPU
evaluation remains charged whenever the GPU reservation is held. Preprocessing
performed before reservation is reported separately but conservatively included
in this projection. Before admission and again under the shared GPU lock, require prior comparison
charges + 1.5× remaining complete-arm projection + 0.05 h <= 4 h, and require
the new margin/shutdown allowance to fit the live native remainder. An unfinished
arm is conservatively budgeted at its full cost on recovery. Open comparison
reservations require owner reconciliation; no synthetic end event is written.
The existing watchdog is bounded additionally by the live cumulative cap.

At takeover, 5.778688 native hours remain. The proposed four hours are **drawn from the existing 12-hour native cap**, never
added to it. A separately approved 4-hour maximum would leave at least 1.778688 hours if no other usage intervenes. This is not an
authorization or an assurance of runtime. Re-read the live ledger before any launch.
Budget exhaustion or an unexpectedly slow shared arm is preserved as an incomplete
result; do not spend the remainder automatically.

Store all 32 checkpoints: conservatively **8.40 GB**, using the largest observed
FP32/optimizer checkpoint payload, plus **2 GiB** for data, logs and exports.
Observed #79 peak reserved CUDA memory is about 9.10 GiB; allow **12 GiB** for
planning, with shared/new-population memory explicitly unmeasured. No environment,
driver, tokenizer or dependency changes are part of this proposal.

## Limits and direction

This tests one data-diversity treatment at the existing engineering scale. It may
improve value transfer while leaving composition or high-value preservation poor.
Passing a reference-task screen would not establish semantic faithfulness, general
intelligence, useful language-model quality, or an architecture ranking. A probe
is a readout; reference dependence is a task property; neither transfers credit
to a causal account of neural computation.

The comparison follows the established practice of testing familiar primitives
in unfamiliar combinations ([Lake and Baroni, 2018](https://proceedings.mlr.press/v80/lake18a.html)).
Its dependency checks use ordinary program slicing
([Weiser, 1984](https://doi.org/10.1109/TSE.1984.5010248)). It claims no new learning
algorithm. Register programs are a bounded diagnostic, not an indefinite gate
before native language-model work toward approximately 125M parameters and beyond.
After this one proposed experiment, report the outcome and revisit broader native
LM priorities; no larger campaign, shaping or WSP is authorized here. The owner's
research conversation retains the weight-space-programmer role.
