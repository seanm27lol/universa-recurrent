# Paired continuation study results

**Status:** all six children completed the frozen paired continuation at training source commit
`1c551fafc7d64e7598d79a0e0b208e4c73e1ca79`, with no intermediate-checkpoint selection. Every child reached
**64 completed epochs, update 13,568, ordered full-label exposure 22,282,560 and cursor 108,160**. The run status is
`complete=true`, `failure=null`, with no child failures and zero replayed updates.

Run: `/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_continuation/`

Weight-free evidence: `/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_continuation/evidence-final/`

Reservation `20261112T000458-637e80a7` closed with status `ok`. It charged **1.9216755555894487 of 3.0 authorized
GPU hours**. Native cumulative usage is **6.221312076198515 / 12 hours**, leaving **5.778687923801485 hours**.
`CONTINUATION.md`, the frozen protocol and the evaluator template retain their historical pre-launch or
future-checkpoint wording; this report describes the completed run without altering those records.

This is one bounded, reused-development comparison, not confirmation, an architecture ranking, a 124M campaign or a
weight-space-programming result. **A = `full_sequence`; B = `answer_eos`.** The predeclared recipe outcome is
**`retain_full_sequence` for every parent**. B improves some training and development counts, but fails strict
improvement on every parent: initial-value and longer-length exact counts are zero ties for all three, typed unseen
composition is also a tie, and untyped dev regresses. Train results are fit/memorization diagnostics, not
generalization.

## 1. Launch, lineage and frozen protocol

The execution path was `native.run_continuation --execute --authorize-hours 3.0`, recovery index 0, one reservation,
with output root above. The launch and status records are `launch.json` and `run_status.json` in the run and bundle.
The documented launch was run from the continuation worktree with the frozen protocol, prepared forks and source:

```bash
export PYTHONPATH=research/ti_tnn PYTHONDONTWRITEBYTECODE=1
PY=/home/seanjazm27/projects/universa-recurrent-native/.venv-native/bin/python
$PY -m native.run_continuation \
  --repository /home/seanjazm27/projects/universa-recurrent-continuation \
  --python "$PY" --execute --authorize-hours 3.0
```

This records the completed launch, not authorization for another run. The read-only dry-run uses the same command
without `--execute --authorize-hours 3.0`; it does not regenerate forks or alter the completed results.

| Identity | Value |
|---|---|
| Continuation run | `78-native-continuation` |
| Training source commit | `1c551fafc7d64e7598d79a0e0b208e4c73e1ca79` |
| Training code identity | `becf66889371b0f2e08b4bbee0c56024295f3a73ffda80cad4a1a1a53d4137f8` |
| Protocol SHA256 | `8f806111bc0e24d8fbe703dc7fff061344408fed236769ad02618caaf5ee6129` |
| Fork manifest SHA256 | `665b8ad675541642d2e42d16ead0427244a239cd133dbe8ac21c5fc0c9b3c305` |
| Parent acquisition run / reservation | `20261110T210743-921461de` / `20261110T210743-bd8d1d5b` |
| Parent source commit / code identity | `2eb2e527be9c4f4db60902d78073fd05c3d23f32` / `92402d297627cd41d0e1f1f176e63aa5fa700006ee041bcbd58e2105adbad35a` |

Each pair restores the same parent model tensors, AdamW state and moments, sampler state and ordered batches, and
Python/NumPy/Torch plus saved CUDA RNG state. Optimizer parameter groups are retained except for the new common
learning rate. The old cosine scheduler is **not** inherited: all children use a new **constant LR 3e-5 for 48 complete
additional epochs, with no warmup**. Auxiliary route/state/load-balance objectives and coefficients remain unchanged
(`lambda_route=0.1`, `lambda_state=0.1`, `lambda_balance=0.01`).

The shared sampler identity is `5ecb5165f1d7c0716acda87514ffd73d454c9a357ffde03dd49c4495d1574192`, order identity
is `c159e845dc5d1a9125f95143e3ae63759e0133795b94582d65131e6252361e70`, parent scheduler identity is
`9133db39d155fc55dd44e4cc0f0244ac2f87e033766a7a3d8848272c6bdd0351`, and the new fork scheduler identity is
`2af44d488953b320216cfa95c1db7cbf3b90dc7e7661068967ec0a689a7e6552`.

| Parent | Model identity | Parent optimizer identity | Parent RNG identity |
|---|---|---|---|
| Dense | `ef4fbd49ddbaa80aff823956dd1b49333155323557837ec1aea8769e6c277870` | `6e88e6e1d43e5b39f4bafd874b84ae774f43035497c4d8ddaf9bf8e324e190c5` | `3c06fc75068990a9eb2e655c87bed7c4f7e39bf8a8b61e27646b5f0f263d322d` |
| Untyped | `28f8a0ea0b423d4ef2adc52aa35823716fd5c517401bdc329607de26224c7a36` | `6e311ca97ab72d4b1b0fa52b238ef00547adde3e0b0021fabb2012d18f276f79` | `e55ba23095c693c26bc5a03139f349bc11771c9fc9eee433de769676a38ba08b` |
| Typed | `7218bfe8f5c1bf7d900fe664cd486fe4295e290e00b188889e9b869ee411647f` | `b32079a4c13e155d6ca05d5dfb44d016dbbc5c2d092fbe2858fd8ab00dc22e71` | `debce5b2adb90d04021f2630827e979851b268d63a18cd14b5b2d7067f7e272f` |

### Objective and exposure

Both arms see every prompt token with identical input IDs, document-local positions, document IDs/boundaries, padding,
route/state targets and auxiliary coefficients. A applies the main mean-token LM loss over the full sequence. B keeps
the prompts fully visible but selects main-loss labels only at answer tokens and EOS. The answer mask is never a model
input, inference-time semantic label or stopping signal. Numeric-value and formatting/EOS losses below are separate
diagnostics, not an added shaping penalty.

Matching is by complete ordered input exposure, updates and epochs, **not compute or selected-loss tokens**. Each epoch
has 1,690 packed rows, 212 updates and 348,165 full-label exposure tokens. B has 69,632 answer+EOS positions per epoch:
16,384 numeric-value and 53,248 formatting/EOS positions.

| Quantity | A: full sequence | B: answer + EOS |
|---|---:|---:|
| Parent: epochs / updates / cursor | 16 / 3,392 / 27,040 | same |
| Parent full-label exposure = main-loss tokens | 5,570,640 | 5,570,640 inherited history |
| Additional epochs / updates | 48 / 10,176 | 48 / 10,176 |
| Additional full-label exposure | 16,711,920 | 16,711,920 |
| Additional main-loss scored tokens | 16,711,920 | 3,342,336 |
| Additional numeric-value / formatting-EOS positions | diagnostics | 786,432 / 2,555,904 |
| Final epochs / updates / cursor | 64 / 13,568 / 108,160 | 64 / 13,568 / 108,160 |
| Final full-label exposure | 22,282,560 | 22,282,560 |
| Whole-history actual main-loss scored tokens | 22,282,560 | 8,912,976 |

B's checkpoint field `consumed_loss_tokens=22,282,560` is the legacy **full-label exposure stopping counter**, not its
actual main-loss count. Across all six children, the continuation adds 61,056 updates, 100,271,520 full-label exposure
tokens and 60,162,768 actual main-loss scored tokens; cumulative totals are 133,695,360 and 93,586,608 respectively.

There are 17 answer+EOS positions per document (four value, 13 formatting/EOS), all covered by ordinary full-LM labels;
route/state masks have zero targets at those positions. Mean first-answer position is 69.001221 in train and 105.203125
in longer programs; mean final-instruction anchor is 56.001221 and 92.203125. The anchor-to-answer distance remains 13
tokens, so longer programs shift absolute position rather than changing the answer-suffix distance.

### Data, tokenizer and mask identities

The immutable data root is `/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_program_generalization/`.
Packing is intact with `position_policy="document"` and context length 256. The tokenizer is GPT-2 revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, vocabulary 50,257, EOS token 50,256.

| Artifact | Identity / SHA256 |
|---|---|
| Checkpoint data identity | `57be5b8bffd49638f6d2969122655a682e0c94b952bee2ae58fb17bcd59ecd06` |
| Dev data identity | `947c2ad06b89539f40407685f9d2434fa76793fa8ae65495524b87ebbb2d5cbf` |
| Data manifest | `ce4f2cefc427d6e8a5641e69bc4206d87729794746eb1017a9dd426c2d8ecc05` |
| Data audit / fixed selections | `37e20c9895668a57de04a3b1f34ad78fab804731d3821dc85384a793b11a4153` / `39ecd29706def90208ed5fcaaab17f24cb275ecb322e549f7602facedf4aed2f` |
| Tokenizer | `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6` |
| Replay specification | `387cc38ae4ecd0a27608d27e4b16634496eb52dce6d76bf05fc15c8aa4009c51` |
| Answer-mask directory identity | `f20e4986639a3dfcdd7e24fc1cb85067f1742e0cbc80f531c3b9692fc0c3cd55` |
| Answer-mask manifest | `845c80dd237184fb76ba46ee77aa931bade45000af5bab54f927bfd7788ef684` |

The masks select main-loss labels only. The original model-input arrays (`input_ids`, `positions`, `doc_ids`,
`key_padding_mask`) and route/state targets are unchanged within each pair.

## 2. Exact answers and paired corrections

Official primary scoring uses **128 examples per child and split**, with 32 per final operation. An exact answer is the
complete greedy canonical four-register answer, not a valid prefix or an individually correct register. `B-only` means
A incorrect -> B correct; `A-only` means A correct -> B incorrect, computed from matched per-example records. Thus
`B-A = B-only - A-only`; these transitions expose regressions hidden by net changes.

### Dense

| Split | A /128 | B /128 | B-A | B-only | A-only |
|---|---:|---:|---:|---:|---:|
| Train | 59 | 102 | +43 | 44 | 1 |
| Dev | 7 | 9 | +2 | 4 | 2 |
| Initial values | 0 | 0 | 0 | 0 | 0 |
| Unseen composition | 1 | 2 | +1 | 1 | 0 |
| Longer length | 0 | 0 | 0 | 0 | 0 |

### Untyped MoE

| Split | A /128 | B /128 | B-A | B-only | A-only |
|---|---:|---:|---:|---:|---:|
| Train | 12 | 25 | +13 | 15 | 2 |
| Dev | 3 | 2 | -1 | 1 | 2 |
| Initial values | 0 | 0 | 0 | 0 | 0 |
| Unseen composition | 0 | 1 | +1 | 1 | 0 |
| Longer length | 0 | 0 | 0 | 0 | 0 |

### Typed MoE

| Split | A /128 | B /128 | B-A | B-only | A-only |
|---|---:|---:|---:|---:|---:|
| Train | 16 | 22 | +6 | 10 | 4 |
| Dev | 2 | 4 | +2 | 3 | 1 |
| Initial values | 0 | 0 | 0 | 0 | 0 |
| Unseen composition | 1 | 1 | 0 | 0 | 0 |
| Longer length | 0 | 0 | 0 | 0 | 0 |

The program source is present in every prompt. Exact lookup scores **128/128 on selected train and 0/128 on every other
split**; the supplied-program executor scores **128/128 on every split**. These are construction baselines, not neural
results or a symbolic substitute for the model.

## 3. Answer CE: official selections versus all-train diagnostics

CE is teacher-forced mean nats/token and includes EOS. **Official selected-128 answer CE** below comes from each
child's evaluation selection and has 2,176 answer+EOS tokens per split. The three diagnostic columns come from the
separate answer diagnostic: train covers **all 4,096 training documents** (69,632 answer+EOS, 16,384 value and 53,248
formatting/EOS tokens); every other split covers the same 128 documents (2,176, 512 and 1,664 tokens respectively).
Small last-digit differences on non-train splits are separate evaluator/diagnostic computations.

In particular, `summary.ab_comparisons` uses the **all-train diagnostic** for train answer/value/formatting CE, while
official train exact answers and official train answer CE use the selected 128. The two train populations must not be
conflated. Selected-128 component CEs are not substituted for the all-train component diagnostics.

### Dense A (`78-dense-full_sequence`)

| Split | Official selected-128 answer+EOS | Diagnostic answer+EOS | Numeric value | Formatting/EOS |
|---|---:|---:|---:|---:|
| Train | 0.155300293 | 0.148047799 | 0.628587358 | 0.000189480 |
| Dev | 0.609739497 | 0.609739502 | 2.585706651 | 0.001749605 |
| Initial values | 1.352160480 | 1.352160510 | 5.603210807 | 0.044145008 |
| Unseen composition | 0.818565458 | 0.818565446 | 3.454770297 | 0.007425519 |
| Longer length | 4.840697033 | 4.840697008 | 4.828479618 | 4.844456205 |

### Dense B (`78-dense-answer_eos`)

| Split | Official selected-128 answer+EOS | Diagnostic answer+EOS | Numeric value | Formatting/EOS |
|---|---:|---:|---:|---:|
| Train | 0.069853673 | 0.067311931 | 0.285584326 | 0.000151197 |
| Dev | 0.670844100 | 0.670844096 | 2.836911604 | 0.004361832 |
| Initial values | 1.469732312 | 1.469732256 | 6.223199427 | 0.007127026 |
| Unseen composition | 0.954143003 | 0.954142963 | 4.054631978 | 0.000146439 |
| Longer length | 4.697282681 | 4.697282749 | 5.720777571 | 4.382361100 |

### Untyped A (`78-untyped-full_sequence`)

| Split | Official selected-128 answer+EOS | Diagnostic answer+EOS | Numeric value | Formatting/EOS |
|---|---:|---:|---:|---:|
| Train | 0.366447231 | 0.380436954 | 1.615221890 | 0.000503144 |
| Dev | 0.582526232 | 0.582526228 | 2.473355383 | 0.000732688 |
| Initial values | 1.088411977 | 1.088411969 | 4.565360487 | 0.018581674 |
| Unseen composition | 0.675828226 | 0.675828225 | 2.861177295 | 0.003413158 |
| Longer length | 3.905505554 | 3.905505573 | 3.679380402 | 3.975082718 |

### Untyped B (`78-untyped-answer_eos`)

| Split | Official selected-128 answer+EOS | Diagnostic answer+EOS | Numeric value | Formatting/EOS |
|---|---:|---:|---:|---:|
| Train | 0.284000762 | 0.301744724 | 1.281221235 | 0.000367353 |
| Dev | 0.598091531 | 0.598091550 | 2.526279867 | 0.004802862 |
| Initial values | 1.170404041 | 1.170404070 | 4.938661575 | 0.010940204 |
| Unseen composition | 0.696386664 | 0.696386639 | 2.948448062 | 0.003444735 |
| Longer length | 2.948928811 | 2.948928791 | 4.034070954 | 2.615038922 |

### Typed A (`78-typed-full_sequence`)

| Split | Official selected-128 answer+EOS | Diagnostic answer+EOS | Numeric value | Formatting/EOS |
|---|---:|---:|---:|---:|
| Train | 0.359970617 | 0.368265608 | 1.563799257 | 0.000409120 |
| Dev | 0.566564771 | 0.566564760 | 2.396929681 | 0.003375584 |
| Initial values | 1.235600338 | 1.235600331 | 5.237088621 | 0.004373256 |
| Unseen composition | 0.679858773 | 0.679858748 | 2.874562651 | 0.004565272 |
| Longer length | 4.205938217 | 4.205938073 | 3.839972913 | 4.318542921 |

### Typed B (`78-typed-answer_eos`)

| Split | Official selected-128 answer+EOS | Diagnostic answer+EOS | Numeric value | Formatting/EOS |
|---|---:|---:|---:|---:|
| Train | 0.279697355 | 0.279186238 | 1.185767399 | 0.000238213 |
| Dev | 0.581194771 | 0.581194786 | 2.464291036 | 0.001780577 |
| Initial values | 1.288492188 | 1.288492217 | 5.464772999 | 0.003482735 |
| Unseen composition | 0.745125040 | 0.745125062 | 3.161578983 | 0.001600805 |
| Longer length | 3.093789193 | 3.093789171 | 4.250843942 | 2.737772346 |

B lowers all-train diagnostic CE but raises dev and unseen-composition numeric-value CE for every parent. Longer-length
formatting/EOS CE improves while numeric-value CE worsens. These diagnostics do not override the exact-answer rule.

## 4. Format, truncation, value error and per-register exactness

`Valid` is the canonical format-valid denominator used by both value MAE and per-register accuracy. MAE is mean absolute
register-value error over valid completions only, **not CE**. `Trunc.` counts outputs reaching the 48-token generation
bound without EOS. Each register cell is `correct/valid`; `—` is undefined when there are no valid outputs. Exact-answer
denominators remain 128 regardless of these smaller valid denominators.

### Dense

| Arm | Split | Valid | EOS | Trunc. | Value MAE | r0 / r1 / r2 / r3 correct per valid N |
|---|---|---:|---:|---:|---:|---|
| A | Train | 128 | 128 | 0 | 1.660156 | 101/128 / 108/128 / 105/128 / 109/128 |
| A | Dev | 126 | 128 | 0 | 5.380952 | 63/126 / 63/126 / 51/126 / 60/126 |
| A | Initial values | 117 | 128 | 0 | 10.923077 | 9/117 / 15/117 / 12/117 / 17/117 |
| A | Unseen composition | 125 | 128 | 0 | 7.596000 | 44/125 / 46/125 / 36/125 / 46/125 |
| A | Longer length | 1 | 35 | 93 | 13.000000 | 1/1 / 0/1 / 0/1 / 1/1 |
| B | Train | 128 | 128 | 0 | 0.484375 | 122/128 / 121/128 / 117/128 / 124/128 |
| B | Dev | 127 | 128 | 0 | 4.974409 | 69/127 / 65/127 / 57/127 / 66/127 |
| B | Initial values | 127 | 128 | 0 | 10.576772 | 23/127 / 16/127 / 21/127 / 18/127 |
| B | Unseen composition | 127 | 128 | 0 | 6.803150 | 51/127 / 44/127 / 44/127 / 47/127 |
| B | Longer length | 2 | 26 | 102 | 8.500000 | 1/2 / 0/2 / 0/2 / 0/2 |

### Untyped MoE

| Arm | Split | Valid | EOS | Trunc. | Value MAE | r0 / r1 / r2 / r3 correct per valid N |
|---|---|---:|---:|---:|---:|---|
| A | Train | 128 | 128 | 0 | 4.623047 | 72/128 / 67/128 / 75/128 / 71/128 |
| A | Dev | 128 | 128 | 0 | 6.294922 | 53/128 / 53/128 / 45/128 / 40/128 |
| A | Initial values | 123 | 128 | 0 | 10.768293 | 18/123 / 13/123 / 18/123 / 17/123 |
| A | Unseen composition | 126 | 128 | 0 | 8.017857 | 39/126 / 35/126 / 35/126 / 27/126 |
| A | Longer length | 3 | 18 | 110 | 7.000000 | 1/3 / 1/3 / 0/3 / 0/3 |
| B | Train | 127 | 128 | 0 | 3.669291 | 80/127 / 80/127 / 90/127 / 81/127 |
| B | Dev | 125 | 128 | 0 | 5.988000 | 55/125 / 59/125 / 53/125 / 42/125 |
| B | Initial values | 128 | 128 | 0 | 10.580078 | 20/128 / 12/128 / 24/128 / 21/128 |
| B | Unseen composition | 126 | 128 | 0 | 7.640873 | 44/126 / 40/126 / 38/126 / 28/126 |
| B | Longer length | 3 | 49 | 79 | 4.916667 | 0/3 / 1/3 / 1/3 / 0/3 |

### Typed MoE

| Arm | Split | Valid | EOS | Trunc. | Value MAE | r0 / r1 / r2 / r3 correct per valid N |
|---|---|---:|---:|---:|---:|---|
| A | Train | 126 | 128 | 0 | 4.309524 | 77/126 / 75/126 / 73/126 / 70/126 |
| A | Dev | 128 | 128 | 0 | 6.205078 | 59/128 / 62/128 / 48/128 / 40/128 |
| A | Initial values | 110 | 128 | 0 | 11.829545 | 11/110 / 9/110 / 9/110 / 7/110 |
| A | Unseen composition | 125 | 128 | 0 | 7.944000 | 44/125 / 40/125 / 36/125 / 31/125 |
| A | Longer length | 2 | 18 | 110 | 10.500000 | 1/2 / 1/2 / 0/2 / 0/2 |
| B | Train | 126 | 128 | 0 | 3.299603 | 81/126 / 87/126 / 81/126 / 87/126 |
| B | Dev | 128 | 128 | 0 | 6.060547 | 52/128 / 59/128 / 48/128 / 55/128 |
| B | Initial values | 119 | 128 | 0 | 12.256303 | 9/119 / 8/119 / 13/119 / 14/119 |
| B | Unseen composition | 127 | 128 | 0 | 8.129921 | 33/127 / 31/127 / 36/127 / 40/127 |
| B | Longer length | 0 | 39 | 89 | — | — / — / — / — |

No prompt was truncated. For every child and split, the evaluator reports `program_source_is_in_prompt=true`,
`complete=128` and `truncated_or_incomplete=0` for its **input/target prerequisites**. Generated-output truncation is a
separate measured outcome above. Per-record `first_errors` distinguish format and register-value errors, but each list is
capped at eight illustrative records per child/split; it is not an error-kind incidence denominator. In particular,
longer-length zeros must be read with truncation and tiny valid denominators, not attributed solely to value errors.

## 5. Final-operation exact answers

Every cell is exact answers out of 32 primary examples. Canonical validity by operation is retained separately in the
child artifacts.

### Dense

| Arm | Split | Set | Add | Multiply | Copy |
|---|---|---:|---:|---:|---:|
| A | Train | 19 | 7 | 12 | 21 |
| A | Dev | 3 | 0 | 0 | 4 |
| A | Initial values | 0 | 0 | 0 | 0 |
| A | Unseen composition | 0 | 1 | 0 | 0 |
| A | Longer length | 0 | 0 | 0 | 0 |
| B | Train | 27 | 23 | 22 | 30 |
| B | Dev | 4 | 0 | 1 | 4 |
| B | Initial values | 0 | 0 | 0 | 0 |
| B | Unseen composition | 1 | 1 | 0 | 0 |
| B | Longer length | 0 | 0 | 0 | 0 |

### Untyped MoE

| Arm | Split | Set | Add | Multiply | Copy |
|---|---|---:|---:|---:|---:|
| A | Train | 4 | 1 | 2 | 5 |
| A | Dev | 1 | 0 | 0 | 2 |
| A | Initial values | 0 | 0 | 0 | 0 |
| A | Unseen composition | 0 | 0 | 0 | 0 |
| A | Longer length | 0 | 0 | 0 | 0 |
| B | Train | 9 | 0 | 5 | 11 |
| B | Dev | 1 | 0 | 1 | 0 |
| B | Initial values | 0 | 0 | 0 | 0 |
| B | Unseen composition | 0 | 1 | 0 | 0 |
| B | Longer length | 0 | 0 | 0 | 0 |

### Typed MoE

| Arm | Split | Set | Add | Multiply | Copy |
|---|---|---:|---:|---:|---:|
| A | Train | 5 | 1 | 2 | 8 |
| A | Dev | 0 | 0 | 0 | 2 |
| A | Initial values | 0 | 0 | 0 | 0 |
| A | Unseen composition | 0 | 1 | 0 | 0 |
| A | Longer length | 0 | 0 | 0 | 0 |
| B | Train | 6 | 1 | 3 | 12 |
| B | Dev | 1 | 0 | 1 | 2 |
| B | Initial values | 0 | 0 | 0 | 0 |
| B | Unseen composition | 0 | 1 | 0 | 0 |
| B | Longer length | 0 | 0 | 0 | 0 |

Longer-length valid counts in set/add/multiply/copy order are `0/0/0/1` dense A, `0/0/0/2` dense B,
`1/0/0/2` untyped A, `1/0/0/2` untyped B, `1/0/0/1` typed A and `0/0/0/0` typed B. All longer operation exact counts
are zero.

## 6. Changed registers and no-op baselines

Changed-register micro accuracy conditions on registers whose executed target differs from their initial value. The
`all changed` denominator counts malformed predictions as incorrect and is the quantity used in `ab_comparisons`;
`valid-only changed` is a separate conditional diagnostic. These are register-position, not example, denominators.

### Dense

| Split | A correct/all changed | A correct/valid-only changed | B correct/all changed | B correct/valid-only changed |
|---|---:|---:|---:|---:|
| Train | 203/288 | 203/288 | 260/288 | 260/288 |
| Dev | 61/297 | 61/294 | 82/297 | 82/294 |
| Initial values | 51/313 | 51/287 | 75/313 | 75/309 |
| Unseen composition | 48/342 | 48/333 | 57/342 | 57/339 |
| Longer length | 2/457 | 2/3 | 1/457 | 1/7 |

### Untyped MoE

| Split | A correct/all changed | A correct/valid-only changed | B correct/all changed | B correct/valid-only changed |
|---|---:|---:|---:|---:|
| Train | 87/288 | 87/288 | 130/288 | 130/286 |
| Dev | 36/297 | 36/297 | 43/297 | 43/287 |
| Initial values | 64/313 | 64/303 | 76/313 | 76/313 |
| Unseen composition | 32/342 | 32/336 | 39/342 | 39/336 |
| Longer length | 1/457 | 1/10 | 2/457 | 2/11 |

### Typed MoE

| Split | A correct/all changed | A correct/valid-only changed | B correct/all changed | B correct/valid-only changed |
|---|---:|---:|---:|---:|
| Train | 91/288 | 91/281 | 128/288 | 128/284 |
| Dev | 40/297 | 40/297 | 44/297 | 44/297 |
| Initial values | 36/313 | 36/269 | 44/313 | 44/292 |
| Unseen composition | 40/342 | 40/333 | 38/342 | 38/340 |
| Longer length | 1/457 | 1/6 | 0/457 | — (0/0) |

The no-op baseline returns the initial registers. Its changed-register accuracy is zero by construction; its nonzero
overall-register score comes only from registers that did not change.

| Split | No-op exact /128 | No-op overall registers | No-op changed registers |
|---|---:|---:|---:|
| Train | 1 | 224/512 | 0/288 |
| Dev | 3 | 215/512 | 0/297 |
| Initial values | 1 | 199/512 | 0/313 |
| Unseen composition | 0 | 170/512 | 0/342 |
| Longer length | 0 | 55/512 | 0/457 |

## 7. What the holdouts mean

**Initial values.** Train uses length-2–6 programs with initial scalars in `[0,24)`. This split reuses 128 training
instruction sequences but puts all 512 initial scalars in `[24,32)`. It is an **initial-value-role holdout**, not a claim
that values 24–31 are absent everywhere in training: 974 training examples contain high `set` immediates, 1,508 have high
final-state values and 1,945 have high noninitial execution-state values.

**Unseen composition.** The reserved adjacent pattern is `multiply` immediately followed by `add`, absent from train,
dev, initial-value and longer-length splits. There are 133 such adjacent pairs across the 128 composition programs. Only
64/128 primary examples have a direct local dependency where the add reads the multiply destination through either add
input; 14/32 intervention examples have it. Later overwrites are separate. Adjacent composition therefore does **not**
guarantee direct dependency in every example.

**Longer length.** These are 128 length-8–12 programs without the reserved composition. Output truncation is substantial
in every child and is not an incomplete evaluator. Train fit, dev performance and these reused holdouts are distinct;
none is fresh confirmation evidence.

## 8. Pass-two writer interventions

Each intervention suppresses all four slot-writer output vectors at the explicitly identified final-instruction anchor
in one-based **pass two** (`pass_1`), the first supervised after-state. The incoming state is retained; later passes and
generated suffixes receive no direct suppression. Every intervention targeted exactly once and suppressed four writer
updates per example.

These tables use a **separate 32-example intervention set per split**, eight per final operation. Baseline exact ->
intervened exact, changed completions/suffixes, validity and truncation all retain that 32-example denominator. They are
never pooled with the primary 128-example metrics.

### Dense

| Arm | Split | Baseline -> intervened exact | Changed completions / suffixes | Valid baseline -> intervened | Trunc. baseline -> intervened |
|---|---|---:|---:|---:|---:|
| A | Train | 19 -> 18 | 2 / 2 | 32 -> 32 | 0 -> 0 |
| A | Dev | 0 -> 0 | 2 / 2 | 32 -> 32 | 0 -> 0 |
| A | Initial values | 0 -> 0 | 1 / 1 | 29 -> 29 | 0 -> 0 |
| A | Unseen composition | 0 -> 0 | 5 / 5 | 31 -> 31 | 0 -> 0 |
| A | Longer length | 0 -> 0 | 2 / 2 | 1 -> 1 | 21 -> 21 |
| B | Train | 25 -> 24 | 3 / 3 | 32 -> 32 | 0 -> 0 |
| B | Dev | 2 -> 2 | 3 / 3 | 32 -> 32 | 0 -> 0 |
| B | Initial values | 0 -> 0 | 3 / 3 | 31 -> 31 | 0 -> 0 |
| B | Unseen composition | 1 -> 1 | 8 / 8 | 31 -> 31 | 0 -> 0 |
| B | Longer length | 0 -> 0 | 1 / 1 | 2 -> 2 | 19 -> 19 |

### Untyped MoE

| Arm | Split | Baseline -> intervened exact | Changed completions / suffixes | Valid baseline -> intervened | Trunc. baseline -> intervened |
|---|---|---:|---:|---:|---:|
| A | Train | 2 -> 2 | 1 / 1 | 32 -> 32 | 0 -> 0 |
| A | Dev | 0 -> 0 | 0 / 0 | 32 -> 32 | 0 -> 0 |
| A | Initial values | 0 -> 0 | 0 / 0 | 30 -> 30 | 0 -> 0 |
| A | Unseen composition | 0 -> 0 | 0 / 0 | 31 -> 31 | 0 -> 0 |
| A | Longer length | 0 -> 0 | 0 / 0 | 3 -> 3 | 25 -> 25 |
| B | Train | 2 -> 2 | 0 / 0 | 31 -> 31 | 0 -> 0 |
| B | Dev | 0 -> 0 | 0 / 0 | 30 -> 30 | 0 -> 0 |
| B | Initial values | 0 -> 0 | 0 / 0 | 32 -> 32 | 0 -> 0 |
| B | Unseen composition | 0 -> 0 | 0 / 0 | 32 -> 32 | 0 -> 0 |
| B | Longer length | 0 -> 0 | 2 / 2 | 2 -> 2 | 18 -> 18 |

### Typed MoE

| Arm | Split | Baseline -> intervened exact | Changed completions / suffixes | Valid baseline -> intervened | Trunc. baseline -> intervened |
|---|---|---:|---:|---:|---:|
| A | Train | 1 -> 1 | 0 / 0 | 32 -> 32 | 0 -> 0 |
| A | Dev | 0 -> 0 | 1 / 1 | 32 -> 32 | 0 -> 0 |
| A | Initial values | 0 -> 0 | 2 / 2 | 27 -> 27 | 0 -> 0 |
| A | Unseen composition | 0 -> 0 | 0 / 0 | 31 -> 31 | 0 -> 0 |
| A | Longer length | 0 -> 0 | 1 / 1 | 2 -> 2 | 24 -> 24 |
| B | Train | 7 -> 7 | 1 / 1 | 32 -> 32 | 0 -> 0 |
| B | Dev | 0 -> 0 | 0 / 0 | 32 -> 32 | 0 -> 0 |
| B | Initial values | 0 -> 0 | 3 / 3 | 31 -> 31 | 0 -> 0 |
| B | Unseen composition | 0 -> 0 | 1 / 1 | 32 -> 32 | 0 -> 0 |
| B | Longer length | 0 -> 0 | 0 / 0 | 0 -> 0 | 15 -> 15 |

These are local intervention **sensitivities, not semantic faithfulness**. A changed output does not by itself establish
the model's internal algorithm, and no effect is not evidence of learned-state absence.

## 9. Routing and state readout

Pass-two after-state readout is separate from writer interventions. It measures prediction of supervised state targets
over every packed row: 1,690 train, 51 dev, 54 initial-value, 63 composition and 74 longer-length rows. It does not show
that answer generation uses that state semantically.

### Dense readout (`pass_1`)

| Split | A correct/N (accuracy) | B correct/N (accuracy) |
|---|---:|---:|
| Train | 32002/63876 (0.501002) | 26222/63876 (0.410514) |
| Dev | 839/1944 (0.431584) | 720/1944 (0.370370) |
| Initial values | 170/2024 (0.083992) | 165/2024 (0.081522) |
| Unseen composition | 941/2440 (0.385656) | 802/2440 (0.328689) |
| Longer length | 1388/5008 (0.277157) | 1259/5008 (0.251398) |

### Untyped readout (`pass_1`)

| Split | A correct/N (accuracy) | B correct/N (accuracy) |
|---|---:|---:|
| Train | 26631/63876 (0.416917) | 20213/63876 (0.316441) |
| Dev | 651/1944 (0.334877) | 507/1944 (0.260802) |
| Initial values | 143/2024 (0.070652) | 131/2024 (0.064723) |
| Unseen composition | 814/2440 (0.333607) | 630/2440 (0.258197) |
| Longer length | 1096/5008 (0.218850) | 920/5008 (0.183706) |

### Typed readout (`pass_1`)

| Split | A correct/N (accuracy) | B correct/N (accuracy) |
|---|---:|---:|
| Train | 30074/63876 (0.470818) | 23176/63876 (0.362828) |
| Dev | 768/1944 (0.395062) | 617/1944 (0.317387) |
| Initial values | 164/2024 (0.081028) | 147/2024 (0.072628) |
| Unseen composition | 824/2440 (0.337705) | 690/2440 (0.282787) |
| Longer length | 1241/5008 (0.247804) | 999/5008 (0.199481) |

Dense has zero route targets/loss. For each MoE child, final teacher-forced dev routing has 1,944 route targets and
43,716 balance decisions (10,929 per pass). Scalar diagnostics are:

| Child | Route loss | Balance loss |
|---|---:|---:|
| Untyped A | 0.002082958 | 1.000585714 |
| Untyped B | 0.004993709 | 1.000529208 |
| Typed A | 0.000728398 | 1.001609474 |
| Typed B | 0.001888736 | 1.002574725 |

Selected expert counts in final dev routing, ordered `e0 / e1 / e2 / e3`, are:

| Child | Pass | Selected counts | Load balance |
|---|---:|---|---:|
| Untyped A | 0 | 2983 / 3822 / 2425 / 1699 | 1.077925 |
| Untyped A | 1 | 4124 / 3491 / 2174 / 1140 | 1.184980 |
| Untyped A | 2 | 2524 / 2332 / 3736 / 2337 | 1.046014 |
| Untyped A | 3 | 636 / 1294 / 3131 / 5868 | 1.538402 |
| Untyped B | 0 | 2950 / 3681 / 2564 / 1734 | 1.054293 |
| Untyped B | 1 | 4357 / 3590 / 2165 / 817 | 1.224424 |
| Untyped B | 2 | 2639 / 2268 / 3768 / 2254 | 1.041154 |
| Untyped B | 3 | 1066 / 1475 / 2753 / 5635 | 1.410134 |
| Typed A | 0 | 2798 / 3196 / 2396 / 2539 | 1.006217 |
| Typed A | 1 | 4241 / 2039 / 3025 / 1624 | 1.138412 |
| Typed A | 2 | 2970 / 2074 / 3251 / 2634 | 1.024858 |
| Typed A | 3 | 273 / 3224 / 2861 / 4571 | 1.302026 |
| Typed B | 0 | 2822 / 2940 / 2390 / 2777 | 1.008314 |
| Typed B | 1 | 4126 / 2499 / 2798 / 1506 | 1.116723 |
| Typed B | 2 | 2829 / 2184 / 3419 / 2497 | 1.021078 |
| Typed B | 3 | 383 / 3261 / 2449 / 4836 | 1.321785 |

No final-dev pass reports a starved expert. The pass-specific skew and readout results are diagnostics, not evidence of
architectural superiority or a faithful state-writing mechanism.

## 10. Measured loss, tokens and time

The first and last values below are the continuation's training-log **main LM losses**, from update 3,393 through 13,568,
not parent-to-final losses. Final dev main LM loss is the same per-batch training-log objective and is not the official
selected-128 answer CE above; in particular, A dev loss is full-sequence while B dev loss is answer-selected.

| Child | First -> last train main LM loss | Final dev main LM loss | Train s | Exposure tokens/s | Main-loss tokens/s |
|---|---:|---:|---:|---:|---:|
| Dense A | 0.479358 -> 0.386226 | 0.507365 | 625.028 | 26,737.9 | 26,737.9 |
| Dense B | 0.437990 -> 0.037760 | 0.672300 | 498.815 | 33,503.2 | 6,700.6 |
| Untyped A | 0.499053 -> 0.437041 | 0.499797 | 851.285 | 19,631.4 | 19,631.4 |
| Untyped B | 0.538549 -> 0.230392 | 0.599497 | 425.043 | 39,318.2 | 7,863.5 |
| Typed A | 0.500305 -> 0.427039 | 0.497737 | 634.948 | 26,320.2 | 26,320.2 |
| Typed B | 0.535116 -> 0.230076 | 0.582167 | 447.148 | 37,374.5 | 7,474.8 |

Every child consumed the same 16,711,920 additional exposure tokens; A scored 16,711,920 additional main-loss tokens
and B scored 3,342,336. Equal updates and exposure therefore do not imply equal loss-token throughput or compute.

## 11. Checkpoints

Parent paths below are relative to the #78 parent run
`/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_program_generalization/runs/20261110T210743-921461de/`.

| Parent | Relative checkpoint path | SHA256 |
|---|---|---|
| Dense | `arms/dense/train/checkpoints/final-update-00003392.pt` | `be4d3405e278ce6a6afe6c7dc4c002cd0f86f9ac07942a6a158b7e0299596418` |
| Untyped | `arms/untyped/train/checkpoints/final-update-00003392.pt` | `1608739f43fb83a88e851b3e8cc9aeee4ec0b1c42ca7323594d46b66963a2199` |
| Typed | `arms/typed/train/checkpoints/final-update-00003392.pt` | `cf7d6bb49a4d67a1ad8c188b4a03252b4f586521181e34cc5f99e5d8b75b187f` |

Fork and evaluated-final paths are relative to the continuation run root. Periodic checkpoint manifests and the complete
fork-to-final journals remain in the bundle; no intermediate checkpoint was selected for these results.

| Child | Kind | Relative checkpoint path | SHA256 |
|---|---|---|---|
| Dense A | Fork | `78-dense-full_sequence/checkpoints/fork-update-00003392.pt` | `be81f5a85c31afb050ff13ac29fe85789d7d79531e698f6ba535ff1e364bf042` |
| Dense A | Final | `78-dense-full_sequence/checkpoints/final-update-00013568.pt` | `701a4671c721dcfe6c354baaa3828ac96a808b8dde6aadb1a2b81e8c75a3c7a6` |
| Dense B | Fork | `78-dense-answer_eos/checkpoints/fork-update-00003392.pt` | `2fa9dc85d6463ef3fa75c523a6d9059029d6b5a0797ca70d5e430fb61e7b33aa` |
| Dense B | Final | `78-dense-answer_eos/checkpoints/final-update-00013568.pt` | `334ea1e81cfb57d697f78c9e736606c1841b6774538961a1945500f69ba5ef0f` |
| Untyped A | Fork | `78-untyped-full_sequence/checkpoints/fork-update-00003392.pt` | `bd0d03aa2fc18c15e4cd03e20e787ba89157b2e294f4dc8de1a68ab29097e51c` |
| Untyped A | Final | `78-untyped-full_sequence/checkpoints/final-update-00013568.pt` | `e11c0be3731165a5c18f6234d8bd1e6988fe80441b0b01cb851db940c063a532` |
| Untyped B | Fork | `78-untyped-answer_eos/checkpoints/fork-update-00003392.pt` | `5a3d0711dc74e0a2b7383b34228758c48073cab41f52e8228cc4434a88dad941` |
| Untyped B | Final | `78-untyped-answer_eos/checkpoints/final-update-00013568.pt` | `e1e28d16ac2e87575b916f6a311ed9c3acb15cb2cabab7824e3f2d387aeef81c` |
| Typed A | Fork | `78-typed-full_sequence/checkpoints/fork-update-00003392.pt` | `7772bb07d008694a44cdb546c1c1ebd372a70e3405d88506ead094c9dd8fee63` |
| Typed A | Final | `78-typed-full_sequence/checkpoints/final-update-00013568.pt` | `73097f2e091d866f39d0f5727706f7856a30308dae308ea1ca01b1372a559bcc` |
| Typed B | Fork | `78-typed-answer_eos/checkpoints/fork-update-00003392.pt` | `fa52395c48550e6d4ad39312eb368cc924ea710aaa0f595efe8916777bab7d12` |
| Typed B | Final | `78-typed-answer_eos/checkpoints/final-update-00013568.pt` | `4f7749c3ebc191d5144707a5b6d5ae5587159fdbce268733718f54fcf74386e0` |

## 12. Resource accounting

Peak allocated/reserved CUDA memory is identical for training and final dev within each child. Times are measured
seconds, not projections. Training invocation wall includes training-side work; arm wall additionally covers the fixed
endpoint stages.

| Child | Allocated GiB | Reserved GiB | Checkpoint s | Train invocation s | Arm wall s |
|---|---:|---:|---:|---:|---:|
| Dense A | 1.782956 | 9.097656 | 33.461 | 773.789 | 1129.457 |
| Dense B | 1.052538 | 1.312500 | 33.378 | 648.598 | 1051.142 |
| Untyped A | 1.785014 | 8.507812 | 31.997 | 1089.278 | 1605.027 |
| Untyped B | 1.056232 | 1.312500 | 32.903 | 579.104 | 974.687 |
| Typed A | 1.788953 | 8.828125 | 33.187 | 788.304 | 1173.287 |
| Typed B | 1.062632 | 1.328125 | 32.698 | 599.087 | 983.721 |

| Child | Official evaluation s | Independent rescore s | CPU FP32 replay s | Answer diagnostics s |
|---|---:|---:|---:|---:|
| Dense A | 284.861 | 0.146 | 4.497 | 65.933 |
| Dense B | 329.682 | 0.155 | 4.422 | 68.049 |
| Untyped A | 444.259 | 0.140 | 4.734 | 66.186 |
| Untyped B | 324.333 | 0.143 | 4.662 | 66.081 |
| Typed A | 313.221 | 0.146 | 4.573 | 66.690 |
| Typed B | 303.315 | 0.156 | 4.951 | 75.842 |

The measured training-side component costs below are seconds for the full 48-epoch continuation. The four component
timers sum exactly to each child's training seconds before rounding; checkpoint and total-invocation wall times are
reported separately above.

| Child | Data loading | Forward | Backward | Optimizer |
|---|---:|---:|---:|---:|
| Dense A | 10.080 | 150.352 | 352.002 | 112.594 |
| Dense B | 12.030 | 108.880 | 273.211 | 104.694 |
| Untyped A | 33.986 | 264.398 | 422.110 | 130.791 |
| Untyped B | 9.630 | 103.153 | 210.189 | 102.071 |
| Typed A | 9.149 | 172.225 | 350.428 | 103.146 |
| Typed B | 9.509 | 110.481 | 224.210 | 102.949 |

Training time sums to **3,482.266 seconds**, child arm walls sum to **6,917.321 seconds**, and study wall is
**6,918.069 seconds**. The canonical reservation charge is elapsed-derived; the raw end record's redundant `hours`
field differs by about `2.0e-10` hours and is not a cap or charge discrepancy. No endpoint or evaluation was reduced to
fit the 3.0-hour allocation.

## 13. Numerical reconstruction qualifications

Training used CUDA BF16 autocast with FP32 losses; saved checkpoint parameters restore exactly as FP32. The reports below
are **unautocast CPU FP32 full-sequence versus NativeCache** comparisons at `atol=rtol=3e-6`. The declared scale policy
requires floating-tensor relative RMS error <= 0.05 of reference RMS and exact integer routes. All six pass that policy
with exact checkpoint hashes/weights and exact integer routes. Both fixed replay examples per child also generate the
same four tokens under full and cached execution.

Nevertheless, **every child fails strict full-versus-cache `allclose` somewhere**. Scale-pass and generated-token
agreement are not bitwise equality.

| Child | Strict-failing tensors | Largest finite relative RMS | Reconstruction report SHA256 |
|---|---|---:|---|
| Dense A | `logits`, `slots` | 7.00819054e-7 | `85d7e51a48a91c0279ea0a77a81c7ba648f837f33e2c3514e9ca6367cb750c7f` |
| Dense B | `h`, `logits`, `slots`, `state_logits` | 7.21190527e-7 | `af598bddd95ac4eea0aec57d3cc8efed2a04c90f750bff85b7a1edc2e3baad9d` |
| Untyped A | `logits`, `route_logits`, `slots` | 4.94967437e-7 | `b67f4097291d47d8b5d0415af63e59403b8faaf7226b694d46ed0a72f9ce3085` |
| Untyped B | `logits`, `route_logits` | 4.45863356e-7 | `84e366e87f462d0c7bb5008d5547dde32cd94ed569cb37532ee4a4a9d79768b1` |
| Typed A | `logits`, `route_logits` | 4.49528456e-7 | `3f158346349a416f4b173e26ef7b97d52813360b38f2561e89cf3fe0a109d49f` |
| Typed B | `logits`, `route_logits` | 4.68278830e-7 | `a9aa2529742b9204360925d8b67c8c4e77eb0c44cbfa843ba73314a5eb4a1fdb` |

The earlier #78 typed `route_logits` strict failure remains part of the record: first tolerance violation at `[0,1,3,1]`,
absolute difference `4.291534423828125e-6`, maximum absolute difference `6.198883056640625e-6`, relative RMS
`3.5788078335711026e-7`. Historical BF16 route failures are not erased by these CPU scale-policy passes. No bitwise
replay, BF16 serving-equivalence or full 48-token/full-evaluation replay is claimed; the fixed generation check is only
two four-token examples per child.

## 14. Predeclared recipe decision

The frozen rule allows B for a later **fresh** evaluation only if it strictly improves primary free-generation exact
answers on **every** official split relative to its paired A, with no regression. CE does not select a winner. Any mixed
result retains the existing full-sequence recipe, and no checkpoint or treatment is selected by this rule.

- **Dense:** B has no exact-count regression, but initial values (`0 -> 0`) and longer length (`0 -> 0`) are ties, so the
  strict-every-split test fails.
- **Untyped:** those same zero ties fail, and dev also regresses (`3 -> 2`, B-only/A-only = 1/2).
- **Typed:** the initial/longer zero ties fail; unseen composition is also `1 -> 1`, despite train and dev gains.

Therefore the exact predeclared outcome is **`retain_full_sequence` for dense, untyped and typed parents**. This is a
development recipe decision only. It is not fresh confirmation, evidence that A generalizes from train fit, or a ranking
of dense and MoE architectures.

## 15. Weight-free evidence and reproduction

The bundle contains 262 files, with 260 hashed entries covering everything except `SHA256SUMS` and
`bundle_manifest.json`, totaling **352,701,941 hashed payload bytes** and preserving 253 original files. It contains
original structured programs, generations, diagnostics, failure records, numerical reports, checkpoint manifests and
journals, but **no model weights, packed original NPY arrays, answer-mask NPY arrays or raw external corpora**. Mask and
checkpoint manifests identify external artifacts; hashes do not include weights or authenticate remote execution.

| Bundle artifact | SHA256 |
|---|---|
| `summary.json` | `a5865132a5741f10351089377a436042f16730a7881524725b0e486331137a6c` |
| `bundle_manifest.json` | `7acb5a4ffdfa79ba018db0fccffcc5268b8e61882bbcd1b52b025208f60e0bfb` |
| `generation_diagnostics.json` | `bdbddceefc18ea58ce7e5de75ecb980d9cf64cca64e99a345439b983ef49f608` |
| `generation_diagnostics.reproduced.json` | `e43dd0128f973ef8622bfb3296bd3cd315a55243175c561a92f4bb4db4780eaa` |
| `REPRODUCE.md` | `601172d963032719f576c1f723b93847764bf37b1fba294b20d16e79f2983a15` |

Each child has 640 primary, 160 selected-baseline, 160 intervened and 30 supplemental-train records: **990 per child,
5,940 total**, with zero score changes and matching independent exact counts. The selected baselines **reuse** 160 of
the primary/supplemental outputs rather than adding inference; the independent source rescore therefore reports
**4,980 outputs**, comprising 3,840 primary, 180 supplemental and 960 intervened records. Selected-baseline and
intervened denominators are never pooled with primary accuracy. The preserved whole-continuation, bundle-only rescore
equals the original diagnostics after removing only the top-level `study` path. All 4,608 executed program targets
validate with no rejected inconsistency.

`evidence-final/REPRODUCE.md` gives the exact weight-free interface:

```bash
export BUNDLE=/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_continuation/evidence-final
WORK="$(mktemp -d)"
PYTHONDONTWRITEBYTECODE=1 python "$BUNDLE/code/export_continuation.py" \
  --rescore-bundle "$BUNDLE" --output "$WORK/generation_diagnostics.json"
python -c 'import json, sys; a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2])); a.pop("study", None); b.pop("study", None); assert a == b, "independent rescore differs"; print("independent rescore matches")' \
  "$BUNDLE/generation_diagnostics.json" "$WORK/generation_diagnostics.json"
(cd "$BUNDLE" && sha256sum -c SHA256SUMS)
```

This reproduction rescoring is not training, model inference or full evaluation replay. Reproducing model outputs would
require the separately identified weights and original data. This report did not rerun the frozen training study or its
full training evaluation.

## 16. Interpretation boundaries

B's large selected-train gains coexist with A-only corrections, dev regressions in untyped, worse numeric-value CE on
several holdouts, lower pass-two readout throughout, and zero exact answers on initial values and longer length for every
child. Longer-length conclusions are further limited by 79–110 truncated outputs per child and valid denominators as low
as zero. Intervention sets are small and establish sensitivity only.

These reused development populations and one fixed continuation per parent/arm do not support claims of dense
architectural superiority, broad composition learning, robust learned state usage or confirmed generalization. The
negative counts, strict numerical failures, historical BF16 issues and budget/evidence limits are retained alongside the
successful endpoint and scale-policy checks.
