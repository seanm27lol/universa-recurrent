# Native TI-TNN bring-up: implementation and measured development results

**Status: complete four-arm engineering run, independently reconstructed.** This is a bounded, single-seed
development study, not a confirmation of architectural superiority, a full pretraining campaign, or evidence for
scaling directly to 350M/1B. No WSP edit was performed. Weights and large artifacts remain local.

- Final protocol: `study_v7/protocol.json`, SHA256 `7b900855706c1c17507b6bba22878087b4e000289ed5d4d97b377f18c970eb92`.
- Final summary: `study_v7/summary.json`, SHA256 `d7a67711098d892856e9c3f9b0f587b354ef3f3efc2c4b6b10afc4d99aa5ad23`.
- Independent review: `study_v7/INDEPENDENT_REVIEW.md` and `independent_review.json`, **113 checks, 0 failures**.
- Final endpoint: **20,001,877 loss-bearing tokens and 9,819 updates per arm**, with identical batches, targets,
  initialization seed, optimizer recipe and evaluation populations.

## What changed architecturally

The earlier native 129M/202M adaptive-compute models already trained weights and missed their quality/efficiency
bars. The frozen-Qwen controllers and closed SQL revision studies are separate systems. This phase jointly trains
new token embeddings, a depth-recurrent Transformer, selected specialists, state interfaces and an untied output
head from random initialization. It does not inherit Qwen weights, caches or teacher targets.

The model has four recurrent passes, four slots per token, and four top-1 specialists with selected-only grouped
execution. C/D share the same optional execution-derived route/state labels; ordinary text has no semantic labels.
A/B/C/D isolate different mechanisms:

| Arm | Core and transform | Intended comparison |
|---|---|---|
| A `dense` | Four unique attention/dense cells | A/B: recurrent weight sharing |
| B `shared` | One shared cell called four times | B/C: selected specialist organization |
| C `untyped` | Shared attention, four generic top-1 experts, generic state interface | C/D: type-specific interfaces/routing |
| D `typed` | C plus per-slot readers/writers/codes and slot-conditioned routing | Direct typing comparison |

The direct `h` residual/attention paths bypass slots. Slot names are addresses, not evidence of entity or program
semantics. Four passes always run; there is no adaptive stopping, skip/small/full objective, sheaf bridge, new
optimizer or editability objective. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for equations and information paths.

## Parameters and executed computation

Unique stored parameters count shared weights once. Decoded-token MACs count all four passes, attention,
interfaces, routers, one selected expert and the output head. Active parameters are route-dependent; the nominal
column is not an exact per-token count. Parameter and compute matching are not claimed simultaneously.

| Configuration | Arm | Total parameters | Nominal one-expert active | Decoded-token MACs |
|---|---|---:|---:|---:|
| Engineering | dense | 21,860,352 | 21,860,352 | 13,176,000 |
| Engineering | shared | 20,084,544 | 20,084,544 | 13,176,000 |
| Engineering | untyped | 21,415,300 | 20,085,316 | 13,179,072 |
| Engineering | typed | 21,499,780 | 20,169,796 | 13,182,144 |
| Serious | dense | 130,803,712 | 130,803,712 | 102,356,800 |
| Serious | shared | 97,549,504 | 97,549,504 | 102,356,800 |
| Serious | untyped | 122,485,380 | 97,552,836 | 102,370,112 |
| Serious typed | typed | **124,049,540** | 99,116,996 | 102,383,424 |

Engineering typed/untyped route-union active parameters range from 20,085,316/20,169,796 to all stored parameters.
Typed adds 84,480 stored parameters and 3,072 decoded-token MACs over untyped, so C/D is not parameter-matched.

## Data and split identities

One GPT-2 tokenizer is used everywhere: `openai-community/gpt2`, revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, tokenizer SHA256
`8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`, vocabulary 50,257, EOS 50,256.

- Wikipedia `20231101.en`, revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, two complete Parquet shards;
  card lists CC-BY-SA-3.0/GFDL with article-specific obligations.
- CodeParrot Clean, revision `35a59fb025bc0a102f7d96eac09d145b896d487b`, one complete 100,000-row shard.
  Only Apache-2.0/BSD-2-Clause/BSD-3-Clause/ISC/MIT labels were retained. Labels are provenance filters, not
  file-level legal clearance.
- Synthetic four-register executable programs use set/add/multiply/copy and execution-derived training auxiliaries.
  No Qwen targets, gated Stack data, WikiText evaluation populations or BIRD data are inputs.

The finite mixture is 45%/45%/10% before boundary masks. Exact-content SHA256 deduplication precedes a deterministic
1% document dev split. The independent data audit found 84 duplicates among 370,512 inputs, 370,428 unique documents,
and zero train/dev exact-content overlap. Near duplicates and semantic contamination are not excluded.

| Split | Packed tokens | Shifted loss tokens | Wikipedia loss | Code loss | Structured loss |
|---|---:|---:|---:|---:|---:|
| Train | 22,000,000 | 21,882,582 | 9,859,332 | 9,859,160 | 2,164,090 |
| Dev | 1,000,000 | 994,482 | 447,957 | 448,172 | 98,353 |

Data manifest SHA256: `93ca481cdb586d0dbd274da731f4e196c15d9cd97da86902077580f43301f90a`.
All 20 arrays matched their manifest hashes and dtypes in the independent data audit. Labels at padding, sequence
starts and document transitions are `-100`; document IDs and padding masks block cross-document attention.

The frozen evaluation uses the first 128 dev sequences (32,600 loss tokens), first 128 examples per structured split,
and all 64 code-suffix examples. It is not full-corpus evaluation. Composition-only scoring uses lengths 3–4;
length 5 is reported separately because it also tests unseen length.

## Measured throughput, memory and checkpoints

All four profiles used BF16 CUDA autocast with FP32 parameters. Training measurements below are from the complete
runs; profiling additionally measured data loading, forward, backward, optimizer, diagnostics and durable logging.
Every run wrote update-0, periodic and final checkpoints with model/optimizer/scheduler/RNG/sampler/token state.

| Arm | Training loss tokens/s | Peak allocated | Peak reserved | Stage train wall | Checkpoint total write time |
|---|---:|---:|---:|---:|---:|
| dense | 31,552.97 | 1.96 GB | 2.42 GB | 878.88 s | 24.55 s |
| shared | 33,445.27 | 1.92 GB | 2.40 GB | 838.04 s | 16.09 s |
| untyped | 32,157.31 | 1.96 GB | 2.44 GB | 870.23 s | 15.85 s |
| typed | 31,176.47 | 1.97 GB | 2.44 GB | 891.73 s | 15.59 s |

The separate clean engineering profiles measured 27,355/27,923/25,726/25,277 tokens/s respectively, including the
full profiler's sampler/hash/logging costs. The clean 124.05M typed profile measured **10,822.11 tokens/s**,
4.15 GB allocated, 4.96 GB reserved, and a 1.489 GB checkpoint in 12.47 s.

Twenty measured updates (40,747 loss tokens) decomposed as follows; diagnostics include routing, and persistence
includes the batch journal and fsynced learning log:

| Typed profile | Data loading | Forward | Backward | Optimizer | Diagnostics | Persistence | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Engineering | 0.1118 s | 0.3475 s | 0.7079 s | 0.1909 s | 0.0133 s | 0.2405 s | 1.6120 s |
| Serious 124.05M | 0.1127 s | 0.7393 s | 1.4594 s | 1.0825 s | 0.0151 s | 0.3562 s | 3.7652 s |

Final checkpoints (all 21 checkpoint positions per arm are journaled):

| Arm | Final checkpoint | SHA256 |
|---|---|---|
| dense | `study_v3/arms/dense/train/checkpoints/final-update-00009819.pt` | `80388f620962b96680a5243e73944e1c5568dc92b9e55eaa7f366987b98cb186` |
| shared | `study_v5/arms/shared/train/checkpoints/final-update-00009819.pt` | `0aeb254908b9d633f0f9ef6796bd7218eb4978d94de1b6e69681f2a1651d0201` |
| untyped | `study_v5/arms/untyped/train/checkpoints/final-update-00009819.pt` | `7f640ff77f1b8b64ce0c5ec1771753c5ebbe4eea9c8083de12dbe9caef2122ff` |
| typed | `study_v7/arms/typed/train/checkpoints/final-update-00009819.pt` | `53ee2a900755696ef213abab2062cd66ccf96cf0bd433126016f64b4ee626da4` |

The dense checkpoint was reused only after independent model/trainer-byte and completed-endpoint compatibility
checks. Later arms trained under later instrumentation revisions. No weights were overwritten during recovery.

## Learning behavior

| Arm | Initial → final train LM loss | Step-1000 dev LM | Final dev LM |
|---|---|---:|---:|
| dense | 10.9013 → 3.8580 | 5.1060 | 4.0828 |
| shared | 10.8081 → 3.9494 | 5.1368 | 4.1711 |
| untyped | 10.8084 → 3.9217 | 5.1317 | 4.1675 |
| typed | 10.8088 → 3.8838 | 5.1425 | 4.1359 |

This establishes actual learning behavior, not a tiny overfit alone. It does not establish that recurrence,
specialists or typing improves over dense. Dense has the lowest final train/dev LM losses in this seed.

Teacher-forced dev metrics are kept separate by source:

| Arm | Wikipedia loss | Code loss | Structured loss |
|---|---:|---:|---:|
| dense | **5.9503** | **2.8091** | **0.5467** |
| shared | 6.0351 | 2.9162 | 0.5542 |
| untyped | 6.0280 | 2.9163 | 0.5477 |
| typed | 6.0207 | 2.8533 | 0.5494 |

Dense is best on all three loss channels. Typed is better than shared/untyped on code but worse than dense; no
specialist advantage is established. State supervision was present in every arm and is not architectural evidence.

## Complete free generation

Prompt-only greedy generation did not receive programs, traces, route labels, state targets or target lengths.

| Arm | Train sample | Ordinary dev | Composition only (3–4) | Length-5 composition | Longer length | Code suffix |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0/128 | 0/128 | 0/85 | 0/43 | 1/128 | 0/64 |
| shared | 0/128 | 0/128 | 0/85 | 0/43 | 1/128 | 0/64 |
| untyped | 0/128 | 0/128 | 0/85 | 0/43 | 1/128 | 0/64 |
| typed | 0/128 | 0/128 | 0/85 | 0/43 | 1/128 | 0/64 |

The single longer-length success per arm does not establish composition. Teacher-forced structured loss around
0.55 is insufficient evidence of reliable complete programs. Exact lookup scores 1.0 on train and 0.0 on held-out
splits; the supplied-program executor scores 1.0 by construction. All 64 code prompts required truncation, so the
code metric is a bounded suffix diagnostic, not full-prompt completion.

## Routing usage and state interventions

No expert was completely starved on the frozen dev routes. Minimum per-expert shares were 3.20% for untyped and
6.41% for typed. Mean distinct experts per token were 2.220 and 2.095, respectively. Late-pass normalized entropy
fell to 0.268/0.152, indicating increasingly peaked routing, not a demonstrated human-semantic specialization.
Dataset-wide `4ΣfP` values were 1.0055 and 1.0157.

Pass-1 expert shares were 50.42/24.85/21.52/3.20 for untyped and 43.68/9.04/21.72/25.56 for typed. Complete
per-pass counts, probabilities, starvation flags, entropy and route-union parameter estimates are in the summary.

Zero/shuffle/wrong-slot interventions changed logits in every arm. Only dense zeroing changed one of two sampled
generations; the other 23 of 24 individual prompt generations were unchanged. These two-prompt interventions
show sensitivity to strong off-distribution changes but also redundancy/bypasses. They do not establish faithful
slot semantics or complete causal explanations.

## Numerical and implementation failures

- The original CPU float32 strict-allclose replay failed on dense (maximum logit difference `1.14e-5`), and a
  single-thread retry also failed. Those reports remain preserved.
- BF16 untyped replay changed one of 1,024 route decisions and is a failed BF16 serving check. FP32 serving passed
  with exact routes and exact generated tokens. BF16 serving equivalence is therefore not established.
- Final FP32 serving reconstruction requires relative RMS error ≤0.05 of reference RMS, exact integer routes and
  exact generated tokens. All four passed; strict-allclose mismatches remain separately reported.
- A CLI-only summary `KeyError` was fixed. Earlier protocol versions and failed reservations were not relabeled.
- A first full local suite had 1,730 passes and 11 environment failures (`sqlglot` and project metadata). Those
  prerequisites were installed in the isolated environment and the focused failures passed. The final full local
  suite passed **2,165 tests with 12 skips**; remote CI also passed.
- `uv pip check` reports the `nvidia-cusparselt-cu13` platform incompatibility in both historical and native
  environments; torch/CUDA imports and both correctness smokes passed. This warning is not hidden.

## GPU accounting

The raw independent ledger reconstruction gives **3.134529605 native hours of 12 additional hours**, cumulative
`27.085955314/35.951425709` hours including preserved history. Failed/interrupted attempts and recovery are included.

| Arm | Train hours | Evaluate hours | Reconstruct hours | Total |
|---|---:|---:|---:|---:|
| dense | 0.226557 | 0.076845 | 0.001266 | 0.304668 |
| shared | 0.214774 | 0.074074 | 0.001182 | 0.290031 |
| untyped | 0.224006 | 0.075428 | 0.001297 | 0.300731 |
| typed | 0.229868 | 0.081145 | 0.001430 | 0.312442 |
| **Four arms** | | | | **1.207871** |

The frozen projection was 1.449394 hours raw / 1.739273 with 20% reserve. The completed four-arm total was below
that projection. The remaining native balance is 8.865470395 hours; unused old historical allowance was never added.

## Independent review and 125M proposal

`independent/review.py` uses only the Python standard library and does not import producer modules. It verified 113
checks covering the raw ledger, all final checkpoint hashes, endpoints, metric arithmetic, generation counts, route
conservation, and reconstruction criteria. Its risks and full check inventory are in `study_v7/independent_review.json`.

A direct serious measurement, not an optimistic MAC-scaled extrapolation, is the appropriate planning denominator:

- 125M loss-bearing tokens at 10,822.11 tokens/s: 11,550.42 s.
- 124 checkpoint writes: 1,545.86 s; 62 bounded dev passes: 36.39 s.
- Setup, complete bounded evaluation and reconstruction: 2,262.91 s.
- **Total: 4.27655 GPU-reservation hours; 5.13186 hours with 20% reserve.**
- Effective complete-cost rate: **29.23M loss-bearing tokens per GPU-reservation hour.**

This is a proposal only, not authorization. Retain-all checkpoints would be about 184.6 GB; a required
final-plus-recovery retention policy would be about 2.98 GB. Any serious campaign needs a separate token budget,
allocation, retention policy and evaluation plan. No 350M/1B training is proposed by this result.

## Reproduction

Read [`README.md`](README.md) for environment creation, immutable data preparation, replay generation, profile and
freeze commands. From the repository root:

```bash
export TI_TNN_SCALING_RUNS=/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_scaling
export PYTHONPATH=research/ti_tnn
export PYTHONDONTWRITEBYTECODE=1
PY=.venv-native/bin/python

# Validate existing completed artifacts; this does not rerun completed arms.
$PY -m native.study summarize --study \
  /home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native/study_v7

# Independent standard-library-only reconstruction of the evidence and ledger.
$PY research/ti_tnn/native/independent/review.py

# Native construction tests.
CUDA_VISIBLE_DEVICES='' $PY -m pytest -q tests/test_ti_tnn_native_*.py
```

The checkpoint package contains stable parameter names, shapes, dtypes, recurrent sharing, route/state capture,
complete generated outputs, intervention controls, and exact checkpoint identities for a future separately
authorized ChatGPT-led WSP phase. That phase has not begun.
