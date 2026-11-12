# Paired continuation: prepared study and allocation proposal

**Status:** prepared, not authorized or launched. No optimizer step has run for this continuation. The completed #78
study, its data, checkpoints, protocol and evidence are preserved. No result was selected by looking at an
intermediate checkpoint, and no 124M campaign or WSP work is proposed.

## 1. Question and fixed comparison

Does another 48 complete epochs improve complete-program learning, and does applying the main token loss only to
answer tokens and EOS use that additional budget more effectively than ordinary full-sequence LM?

For each immutable #78 parent—dense, untyped MoE and typed MoE—there are two independently identified children:

- **A: `full_sequence`.** Continue the existing full-sequence LM objective.
- **B: `answer_eos`.** Keep every prompt token visible, but select the main LM loss only at answer tokens and EOS.

The answer mask is an external loss-label target, never a model input or an inference-time semantic label. Prompt
input IDs, document-local positions, document boundaries, padding, route/state targets and auxiliary coefficients
remain identical within each pair. The main loss in both arms is mean selected-token cross entropy per batch.
Formatting/EOS and numeric-value losses are separately reported diagnostics, not an additional treatment or shaping
penalty.

Every pair inherits the parent's model, AdamW state and moments, sampler state, Python/NumPy/Torch RNG state and
ordered batches. The old cosine scheduler is **not** inherited. All six children use the same new **constant 3e-5
learning rate for 48 complete additional epochs, with no warmup**. Parent acquisition and reservation IDs are kept
as lineage, not mistaken for the child's current reservation.

## 2. Ordered exposure and endpoint

Matching is by complete ordered input exposure, not by unmasked loss-token count:

| Quantity | A: full sequence | B: answer + EOS |
|---|---:|---:|
| Parent state | 16 epochs, 3,392 updates, 5,570,640 exposure tokens, cursor 27,040 | same |
| Additional epochs | 48 | 48 |
| Additional optimizer updates | 10,176 | 10,176 |
| Additional full-label exposure tokens | 16,711,920 | 16,711,920 |
| Additional main-loss tokens | 16,711,920 | 3,342,336 |
| Additional numeric-value tokens | diagnostic only | 786,432 |
| Additional formatting/EOS tokens | diagnostic only | 2,555,904 |
| Final epoch / updates / cursor | 64 / 13,568 / 108,160 | 64 / 13,568 / 108,160 |
| Final full-label exposure | 22,282,560 | 22,282,560 |

The 48 additional epochs are exactly 16,711,920 exposure tokens. The 3,342,336 answer+EOS tokens consist of
786,432 numeric-value tokens and 2,555,904 formatting/EOS tokens. The answer mask includes the first answer token
and EOS; it does not reduce the model's view of the prompt or change the stopping rule. Actual computation and
selected-loss token counts are recorded separately, so the two objectives cannot be called compute-matched merely
because their update counts match.

The protocol is `continuation_protocol.json`, SHA256
`8f806111bc0e24d8fbe703dc7fff061344408fed236769ad02618caaf5ee6129`. Its status is
`proposed_not_authorized_or_launched`. `fork_manifest.json`, SHA256
`665b8ad675541642d2e42d16ead0427244a239cd133dbe8ac21c5fc0c9b3c305`, records each parent hash, explicit fork,
model/optimizer/sampler/RNG identity, new scheduler identity and global `reservation_created = false`. Each child's
`run_config.json` and initial-checkpoint payload records `reservation_id = None`.

| Child run ID | Initial checkpoint SHA256 |
|---|---|
| `78-dense-full_sequence` | `be81f5a85c31afb050ff13ac29fe85789d7d79531e698f6ba535ff1e364bf042` |
| `78-dense-answer_eos` | `2fa9dc85d6463ef3fa75c523a6d9059029d6b5a0797ca70d5e430fb61e7b33aa` |
| `78-untyped-full_sequence` | `bd0d03aa2fc18c15e4cd03e20e787ba89157b2e294f4dc8de1a68ab29097e51c` |
| `78-untyped-answer_eos` | `5a3d0711dc74e0a2b7383b34228758c48073cab41f52e8228cc4434a88dad941` |
| `78-typed-full_sequence` | `7772bb07d008694a44cdb546c1c1ebd372a70e3405d88506ead094c9dd8fee63` |
| `78-typed-answer_eos` | `fa52395c48550e6d4ad39312eb368cc924ea710aaa0f595efe8916777bab7d12` |

Initial files are `<child-run-id>/checkpoints/fork-update-00003392.pt`. The required final file is
`final-update-00013568.pt`, which **does not exist yet**. Parent final hashes remain `be4d3405…`, `1608739f…` and
`cf7d6bb4…`; the complete values are in `PROGRAM_RESULTS.md` and the fork manifest. No parent file is overwritten.

## 3. Frozen populations and evaluation

The existing populations are reused development diagnostics, not confirmation data:

- `train`: 4,096 varied programs, with a fixed 128-example evaluation subset.
- `dev`: 128 new program/initial pairs from the training distribution.
- `initial_values`: 128 training instruction sequences with every scalar initial value in `[24,32)`. Those numbers
  are not absent everywhere in training.
- `unseen_composition`: 128 programs with reserved adjacent `multiply` followed by `add`; a direct local dependency
  is not guaranteed in every example.
- `longer_length`: 128 programs with 8–12 instructions and no reserved composition.

All five splits retain intact examples, `position_policy="document"`, the original tokenizer and ordered sampler.
Official evaluation includes complete greedy answers, answer-token CE, format/EOS/truncation, per-register and
final-operation results, targeted pass-two writer interventions, and independent per-child rescoring. Answer
cross-entropy remains separate from mean absolute register error. Each child gets 640 primary evaluations, 32
interventions per split and 30 supplemental train baselines. No intermediate checkpoint or treatment is selected
by viewing these results.

The companion `manifest.json` has SHA256 `845c80dd237184fb76ba46ee77aa931bade45000af5bab54f927bfd7788ef684`.
The directory mask identity used by the answer+EOS children is
`f20e4986639a3dfcdd7e24fc1cb85067f1742e0cbc80f531c3b9692fc0c3cd55`.
Its manifest records full-label exposure, answer/EOS, numeric-value and formatting/EOS masks for all five splits,
without changing any original array or manifest. Masks have exactly 17 answer+EOS positions, four numeric-value
positions and 13 formatting/EOS positions per document. `native.answer_analysis.AnswerMaskedDataset` preserves the
ordinary `PackedDataset.batch` interface; model-only batches still contain only the four original model inputs.

## 4. What the saved #78 diagnostics show

`answer_diagnostics.json` files in the three parent arm directories and `diagnostic_manifest.json` contain the
bounded CPU-only parent analysis. They are new diagnostic files, not a rewrite of the official evaluations or of
`evidence-final`. Parent checkpoints were not changed. `diagnostic_manifest.json` has SHA256
`fd7660ad21d0f9da20d1d29554c9181c68034c9003449a216ff03554075d7939`.

The CE diagnostic evaluates all 4,096 train documents and all 128 documents in every other split. The no-op and
saved-generation register metrics below use the official 128-example selections. Thus the train CE values below
need not equal the official train subset's CE; dev, initial-value, composition and longer-length selections coincide.

### Answer-value versus formatting/EOS loss

Mean teacher-forced CE in nats/token is split into total answer+EOS, numeric register-value tokens and
formatting/EOS tokens. The partitions contain 16 answer tokens and one EOS per document: four numeric tokens and
13 formatting/EOS tokens. The total CE is token-weighted across all 17.

| Arm | Split | Answer+EOS | Numeric value | Formatting/EOS |
|---|---|---:|---:|---:|
| Dense | train | 0.395267 | 1.677201 | 0.000826 |
| Dense | dev | 0.477592 | 2.024484 | 0.001625 |
| Dense | initial values | 0.931426 | 3.955565 | 0.000922 |
| Dense | unseen composition | 0.592629 | 2.516127 | 0.000783 |
| Dense | longer length | 3.164358 | 3.350870 | 3.106970 |
| Untyped | train | 0.509565 | 2.162696 | 0.000909 |
| Untyped | dev | 0.564119 | 2.393668 | 0.001181 |
| Untyped | initial values | 0.841986 | 3.572650 | 0.001782 |
| Untyped | unseen composition | 0.639380 | 2.714404 | 0.000911 |
| Untyped | longer length | 3.281691 | 3.420220 | 3.239067 |
| Typed | train | 0.497437 | 2.110836 | 0.001006 |
| Typed | dev | 0.554353 | 2.352545 | 0.001064 |
| Typed | initial values | 0.919669 | 3.903534 | 0.001557 |
| Typed | unseen composition | 0.639090 | 2.712697 | 0.001056 |
| Typed | longer length | 3.401804 | 3.352658 | 3.416926 |

Ordinary answers have very low formatting/EOS loss but substantial numeric-value loss. Longer programs make both
kinds of loss substantial. This is why a separate answer-value diagnostic is useful; it does not replace official
complete-answer accuracy or justify a new objective on its own.

### No-op and changed-register results

The no-op baseline returns the four initial registers. Its changed-register accuracy is zero everywhere by
construction: it can only be correct where initial and final values already agree. Saved-generation changed-register
accuracy instead compares predicted target values to the executed target only for registers that actually change;
malformed completions count as incorrect.

| Split | No-op exact | No-op overall registers | Dense changed | Untyped changed | Typed changed |
|---|---:|---:|---:|---:|---:|
| train | 1/128 | 224/512 = 0.437500 | 89/288 = 0.309028 | 36/288 = 0.125000 | 49/288 = 0.170139 |
| dev | 3/128 | 215/512 = 0.419922 | 58/297 = 0.195286 | 17/297 = 0.057239 | 25/297 = 0.084175 |
| initial values | 1/128 | 199/512 = 0.388672 | 62/313 = 0.198083 | 63/313 = 0.201278 | 66/313 = 0.210863 |
| unseen composition | 0/128 | 170/512 = 0.332031 | 61/342 = 0.178363 | 23/342 = 0.067251 | 30/342 = 0.087719 |
| longer length | 0/128 | 55/512 = 0.107422 | 5/457 = 0.010941 | 0/457 = 0 | 1/457 = 0.002188 |

The denominator counts changed register positions, not examples. Exact lookup and supplied-program executor
baselines remain separate; neither substitutes for the neural model.

### Supervision positions

All answer+EOS positions in both train and longer-program packed data have ordinary LM labels. Route and state mask
coverage at those positions is exactly zero. Mean first answer position is 69.001221 in train and 105.203125 in
longer programs; mean final-instruction anchor is 56.001221 and 92.203125 respectively. The final-anchor-to-first-
answer distance is exactly 13 tokens in both. Longer programs move the computation farther downstream in absolute
position but do not change the fixed answer-suffix distance.

These measurements explain the existing results without claiming that a readable state is a faithful explanation or
that an unchanged writer intervention proves state absence.

## 5. Measured six-run allocation proposal

Costs use actual #78 per-mode training, checkpoint and arm-wall measurements. Training and checkpoint costs are
scaled from 16 to 48 additional epochs. Each child includes official evaluation, independent rescoring, CPU FP32
replay, answer diagnostics, periodic/final dev evaluation and fork initialization. The #78 combined wall cost does
not provide a defensible split of official evaluation, rescoring and replay, so those costs remain combined.

| Parent/child mode | Training seconds | Official evaluation/rescore/replay seconds | Answer diagnostics seconds | Checkpoint + fork initialization seconds | Total seconds per child |
|---|---:|---:|---:|---:|---:|
| Dense | 652.527 | 330.530 | 67.667 | 26.821 | 1,079.652 |
| Untyped MoE | 622.785 | 426.078 | 67.667 | 23.552 | 1,142.442 |
| Typed MoE | 631.298 | 435.945 | 67.667 | 23.423 | 1,160.724 |

All six children are budgeted at **6,765.636 seconds = 1.879343 hours**. The projection includes 67.667 seconds of
answer diagnostics per child, based on 203 seconds for all three parents. Maximum watchdog time is 2.819015 hours;
adding 0.05 hours for shutdown gives **2.869015 hours**. The proposed **3.0-hour maximum** therefore fits with
0.130985 hours of allocation headroom and 37.36% planning margin above projected runtime. Live native remaining is
7.700363 hours, within the existing 12-hour cap. No cap increase is requested.

`--resume` requires the original launch, protocol, source, data and authorization identities. It reuses valid stages,
writes versioned recovery results, and includes every previous continuation charge in the same 3-hour limit. It never
makes a truncated control into a completed comparison.

## 6. Predeclared recipe decision rule

No winner is selected before the fixed endpoints. For each parent, `answer_eos` may be recommended for a **fresh
evaluation** only if it improves complete-answer accuracy on every official split—train, dev, initial values, unseen
composition and longer length—relative to its paired `full_sequence` child, with no regression on any split.
Otherwise that parent retains `full_sequence`.

A change of recipe is recommended only if all three parents satisfy that rule. Any mixed result retains the existing
full-sequence recipe. Loss or CE alone does not select a winner, and no additional treatment is introduced after
viewing holdouts. If a recipe is selected, its subsequent evaluation uses fresh data; these reused populations are
not confirmation.

## 7. Exact dry-run and approval gate

Run from `/home/seanjazm27/projects/universa-recurrent-continuation` with the pinned interpreter:

```bash
export PYTHONPATH=research/ti_tnn PYTHONDONTWRITEBYTECODE=1
export TI_TNN_SCALING_RUNS=/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_scaling
PY=/home/seanjazm27/projects/universa-recurrent-native/.venv-native/bin/python

# Default read-only dry-run: no artifacts, reservation or optimizer update.
$PY -m native.run_continuation \
  --repository /home/seanjazm27/projects/universa-recurrent-continuation \
  --python "$PY"

# CPU-only fork creation already performed: no optimizer update or reservation.
$PY -m native.run_continuation \
  --repository /home/seanjazm27/projects/universa-recurrent-continuation \
  --python "$PY" --prepare-only
```

The default dry-run reports `artifacts_created=false`, `reservation_created=false`, `optimizer_updates_performed=0`
and `training_launched=false`. The prepared manifest reports `completed_validated`, also with zero optimizer updates
and no reservation. **No training has been authorized or started for this continuation.**

A future execution requires an explicit owner approval of exactly 3.0 hours and then:

```text
--execute --authorize-hours 3.0
```

The execution path additionally requires clean committed source and the completed fork manifest. GPU forks cannot
open an ordinary fallback reservation; they require the inherited continuation owner with the matching study ID and
3.0-hour authorization. No further study will be started without that approval.
