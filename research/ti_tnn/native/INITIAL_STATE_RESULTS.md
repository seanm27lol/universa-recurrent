# Initial-state diversity: completed development study

For `copy r0 r1`, repeating one input can teach one answer, while eight initial
states require several answers for the same operation. At the registered fixed
budget, the eight-state condition fitted its training examples poorly and did
not pass either complete-answer development screen. It did improve several
partial metrics, including heldout numeric CE and value preservation.

All eight arms completed. This is a single-seed development result, not a general
claim that multiple initial states cannot help. The main contrast, source, data,
objective, seeds and endpoint were unchanged after approval. See the frozen
[protocol](INITIAL_STATE_STUDY.md), [full metrics](INITIAL_STATE_DETAILS.md), and
[independent results review](INITIAL_STATE_RESULTS_REVIEW.md).

## Registered endpoint and complete answers

Each run received 16 epochs, 1,424 optimizer updates, cursor 11,376 and 2,590,592
full-label tokens. The final endpoint checkpoint was selected in every arm;
there was no intermediate selection, recovery, failed attempt or truncated control.
All nine populations were evaluated with 128 examples and 512 register positions
per population. Every complete-answer cell below has denominator **128**.

| Architecture | Condition | train_anchor | train_sample | transition_values_id | transition_values_ood | program_values_id | program_values_ood | dev | composition | longer |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | repeat_one | 119 | 119 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dense | multi_state | 2 | 2 | 2 | 0 | 2 | 0 | 1 | 0 | 0 |
| shared | repeat_one | 57 | 57 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| shared | multi_state | 1 | 1 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| untyped | repeat_one | 43 | 43 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| untyped | multi_state | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| typed | repeat_one | 38 | 38 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| typed | multi_state | 0 | 1 | 0 | 0 | 2 | 0 | 1 | 0 | 0 |

`train_anchor` uses the same initial vector in both conditions. `train_sample`
uses that vector for repeat_one and a second trained vector for multi_state;
repeat_one's two training populations overlap completely. Do not pool them.
The value, novel-program, composition and longer populations remain separate.

The dense/shared complete-answer deltas (multi_state minus repeat_one) are:

| Control | Individual transitions, in range | Familiar programs, in range | Ordinary dev | Composition |
|---|---:|---:|---:|---:|
| Dense | +2 | +2 | +1 | 0 |
| Shared recurrent | 0 | +1 | 0 | 0 |

The frozen screen required at least +8 on **both** in-range value populations
in **both** controls, with no dev decline. Therefore `recommend_value_reuse=false`
and `recommend_composition_transfer=false`. This rule is unchanged. MoE results
are descriptive controls; they cannot rescue the screen or establish a routing
advantage from dense-versus-MoE alone.

## What changed, and what did not generalize

The multi-state training-anchor scores are 2/128 dense, 1/128 shared and 0/128 for
both MoEs, versus 119, 57, 43 and 38 respectively for repeat_one. Multi_state's
second trained-state sample also scores only 1–2/128. Severe lack of complete-answer
fitting limits what the negative transfer result says about the underlying hypothesis.
The individual-transition half of each multi-state training-anchor population
scores 0/64, so this failure already appears before composing several operations.
With equal document exposure, each repeat_one pair is shown eight times as often
as each distinct multi_state pair. That intended tradeoff is not a comparison at
matched exposure per unique example or matched convergence.

The changed/unchanged split prevents counting preservation alone as improved
transition computation. For the in-range individual-transition population:

| Architecture | Changed values: repeat_one → multi_state | Unchanged values: repeat_one → multi_state |
|---|---:|---:|
| dense | 9/114 → 22/114 | 100/398 → 167/398 |
| shared | 24/114 → 23/114 | 59/398 → 102/398 |
| untyped | 13/114 → 16/114 | 54/398 → 108/398 |
| typed | 18/114 → 21/114 | 52/398 → 97/398 |

For shared recurrence, this gain is entirely in preserved values: changed values
move 24→23/114 while unchanged values move 59→102/398. Dense improves both, but
preservation contributes the larger count gain. Familiar composed programs and
ordinary dev also show partial register gains; their complete answers remain weak.
These categories describe reference outputs, not the neural model's internal algorithm.

Heldout numeric-value CE decreases for all architectures, including the high-initial
and longer populations; this is not equivalent to complete-answer execution.
For ordinary dev, numeric CE is 4.8951→2.9119 dense, 4.5354→3.0432 shared,
4.4169→3.1476 untyped and 4.2305→3.1206 typed. Training-anchor numeric CE instead
worsens in every architecture. Teacher-forced answer/formatting CE and all four
state-readout passes are reported separately in the detailed tables. State readout
measures supervised prediction, not causal use of state during generation.

All 1,024 longer-population completions are malformed: 0/128 format-valid outputs
in every arm. Their zero exact scores cannot be assigned solely to transition
computation or composition. Across all nine populations there are 1,960 malformed
outputs out of 9,216, counted as 7,840 failed register positions, never removed from
the main denominators. Values 24–31 are held out as initial values, not as globally
unseen tokens or intermediate/answer values.
Every arm also scores 0/399 on unchanged registers in the high-initial individual-
transition population. Better in-range preservation does not establish preservation
across this value shift.

This study does not establish useful complete-program generalization or confirm
reusable neural transitions. Lower heldout CE and partial value gains are retained
as evidence alongside the failed screen and underfitting. There is no automatic
extension, objective change or additional treatment. The earlier #79 decisions
remain `retain_full_sequence`; this study did not compare training objectives.
Register programs remain diagnostics rather than an indefinite prerequisite for
all future native language-model work.

## Artifact identity and saved-output reconstruction

Training source: `3808e8209b4b22e36eded29343721195ca682ca4`.
Frozen protocol: `bd108e03f51aefbc5bf8ce1cc0086c874d450b79ecf5668a23b275d1f37f50df`.
The original preparation remains preserved under
`5e2524b8c7f54b08533fdce501e710b48f0fe23127a6493d9a877b5727cf802f`.
Composition v2 retains its 128 independently checked reference ablation witnesses.
No data were regenerated or selected from model performance.

The existing saved-output scorer reconstructed all **72/72 population score
objects exactly**, covering 9,216 completions and 36,864 register positions, in a
new private analysis directory. It made zero model calls. Per-register counts,
valid-only denominators, final-write/dependency categories, no-op baselines and
all saved scoring fields remain in the preserved local JSON. Final checkpoint
content/configuration, source, data and sampler identities were validated, along
with all 321 prepared artifacts and the completed-arm artifact manifests.

Local reconstructed aggregate SHA256:
`51f1264c49122e70e7580723c1b2989287a112935f57c99d5abaae5c12050838`.
Checkpoint hashes are in the detailed table. Runtime paths, process details,
ledger records, checkpoints, raw completions and datasets remain private.

## Numerical scope

All eight registered CPU FP32 replays passed their scale-based check and strict
allclose for these fixed inputs, with exact applicable routes and generated tokens.
Floating tensors still differ bitwise. The scope is two 16-token teacher-forced
prefixes and two full prompts with at most four generated tokens, not the whole
population. It does not establish general BF16-serving equivalence, bitwise
reproduction, remote execution authentication or a new independent learning run.
The historical #79 strict-allclose failures remain unchanged.

## Actual resource accounting

One reservation closed normally, with no recovery or failed attempts. Its complete
wall time, including evaluation and bookkeeping, was **3,331.390120 s =
0.9253861445033302 GPU-reservation hours**, below the approved four-hour cumulative
maximum drawn from the existing native cap. Native usage is now
**7.146698220701847 / 12 h**, with **4.853301779298153 h remaining**. Remaining
hours are not authorization for another run. The historical open authorization-1
record remains preserved; no end event was fabricated.

| Resource | Recorded value |
|---|---:|
| Projected reservation | 2.514782 h |
| Actual reservation | 0.925386 h |
| Recorded data/forward/backward/optimizer timers, all arms | 753.879 s |
| Recorded checkpoint-write timers | 37.008 s |
| Remaining reserved wall time | 2,540.503 s |
| Optimizer updates, all arms | 11,392 |
| Full-label tokens | 20,724,736 |
| Input tokens | 20,986,880 |
| Padded training slots | 23,298,048 |
| Generation prompt tokens | 589,944 |
| Generated tokens, including EOS | 172,637 |
| Maximum permitted generated tokens | 442,368 |
| Preserved checkpoint files | 32 |
| Checkpoint bytes | 6,790,645,254 |
| Maximum recorded CUDA reserved memory | 6,985,613,312 bytes (6.5059 GiB) |

The remaining reserved wall time includes initialization/loading, periodic/final
dev passes, generation, answer CE, state readouts, scoring, replay, hashes and
logging. Those components were not separately timed by this adapter; do not
invent a finer cost split. Allocator reserved memory can carry across sequential
arms and is not an isolated intrinsic model-memory measurement. The conservative
projection exceeded actual cost; this does not by itself establish better GPU
throughput or scalability. Post-reservation saved-score/checkpoint verification
took 10.168 CPU seconds and is reported separately, not charged as GPU occupancy.

Primary background remains the program-slicing and compositional-testing work
linked in the [diagnostic](DEPENDENCY_DIAGNOSTIC.md). These results add a bounded
local development comparison, not a new learning algorithm or a mechanism claim.
