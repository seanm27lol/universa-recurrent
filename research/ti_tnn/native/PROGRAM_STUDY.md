# Proposed varied-program acquisition and generalization study

**Status: prepared, not authorized or launched.** This is a protocol and construction/evaluator implementation, not
a learning result. The completed 32-example document-position acquisition remains a pipeline check and is not the
training population for this study. No 124M campaign, WSP edit, Qwen-generated supervision, or MoE removal is involved.

## 1. New executable-program population

The generated data are in `runs/ti_tnn_native_program_generalization/`. The new generator seed is `20261112`; no old
structured-training or evaluation population is read. Every program is executed to produce its target and execution-
derived supervision. Initial values and the complete ordered program remain in the model prompt; inference receives
no execution trace, type label, target text or parser output.

| Split | Examples | Rows | Loss tokens | Role |
|---|---:|---:|---:|---|
| `train` | 4,096 | 1,690 | 348,165 | New length-2..6 programs; scalar initial values in `[0,24)` |
| `dev` | 128 | 51 | 10,801 | New, disjoint program/initial pairs from the same distribution |
| `initial_values` | 128 | 54 | 10,909 | Instruction sequences seen in train, but every scalar initial value is new, in `[24,32)` |
| `unseen_composition` | 128 | 63 | 11,814 | Length-2..6 programs containing reserved `multiply` immediately followed by `add` |
| `longer_length` | 128 | 74 | 15,514 | Length-8..12 programs without the reserved composition |

The reserved adjacent composition is absent from train, dev, initial-value and longer-length splits. Non-initial-
value instruction sequences are unique. Initial-value examples intentionally reuse train instruction sequences with
new scalar initial values. All 4,608 program/initial pairs and all 4,608 complete prompt+target contents are unique.

Final operations are balanced: 1,024 per family in train and 32 per family in every evaluation split. All 4,608
programs passed independent execution. The longest complete document is 144 tokens, below the 256-token context;
every answer and EOS is supervised. No example is dropped or split across rows.

Packing is `intact` with `structured_overflow="error"` and **`position_policy="document"`**. Every document starts at
position zero, padding has position zero and label `-100`, and document transitions cannot leak attention or labels.
The manifest records 50 array hashes; `audit.json` and independent review checked positions, labels, route/state
supervision and full token conservation.

### Identities

- Tokenizer: `openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Tokenizer SHA256: `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`.
- Manifest SHA256: `ce4f2cefc427d6e8a5641e69bc4206d87729794746eb1017a9dd426c2d8ecc05`.
- Audit SHA256: `37e20c9895668a57de04a3b1f34ad78fab804731d3821dc85384a793b11a4153`.
- Selection SHA256: `39ecd29706def90208ed5fcaaab17f24cb275ecb322e549f7602facedf4aed2f`.
- Protocol SHA256: `5066ae93503c6b34e5b49afdcf59a2714d93c51583e64d73d1897e757efddf91`.
- Data-generation branch: `ti-tnn-native-varied-programs`, based on `e876e28`. Protocol source-file hashes are in
  `study_protocol.json`; an actual training launch would additionally record its exact source commit and code identity.

The data are synthetic and use no external text/code corpus. The saved tokenizer is reused under its existing pinned
identity. `manifest.json` preserves the builder-version identities, while `study_protocol.json` binds the current
preparation, evaluation and runner source files separately.

## 2. Fixed model and optimization

Dense, untyped MoE and typed MoE use the existing engineering configuration: 21,860,352, 21,415,300 and 21,499,780
unique trainable parameters respectively. They have different parameter counts and are not claimed to be
simultaneously parameter- and compute-matched. The architecture stays at four recurrent passes and four selected
specialists, with no adaptive stopping.

Every arm uses randomly initialized weights, `seed=model_seed=20261112`, `data_seed=20261113`, batch size eight, and
the existing LM + load-balance + route + state objective. State targets use `before,after,after,after`; route and
state traces supervise training only. There is no answer-focused loss or architecture change.

AdamW uses learning rate `3e-4`, weight decay `0.01`, betas `(0.9,0.999)`, epsilon `1e-8`, gradient clip `1.0`,
2% warmup and cosine decay to 10% of the base rate. Checkpoints occur every 500 updates, with full dev evaluation
every 500 updates and at the endpoint.

One epoch contains **348,165 supervised loss tokens and 212 updates**, including the final partial batch. Sixteen
complete epochs therefore require exactly:

```text
5,570,640 loss tokens and 3,392 updates per arm
16,711,920 loss tokens and 10,176 updates across all three arms
```

The runner independently derives this endpoint from the actual packed sampler, rather than assuming it from the
protocol text.

## 3. Evaluation and intervention contract

Each split has a predeclared, hash-balanced selection of **128 evaluation examples**, 32 per final operation. Train
uses a subset of its 4,096 examples; each other split has exactly 128. Generation receives the complete prompt only,
with at most 48 new tokens and no target-length prompt padding or oracle auxiliary inputs.

Measure separately:

- exact complete answers and canonical format;
- teacher-forced answer-token cross-entropy, including EOS;
- per-register accuracy and mean absolute error over format-valid completions, with coverage counts;
- exact and format accuracy grouped by **final instruction operation**;
- EOS, truncation, first error, and full per-example completions;
- exact-lookup and supplied-program-executor baselines, without substituting the executor for the neural model.

State readout is evaluated over **all packed rows** in each split, including all 1,690 train rows. That population is
not silently conflated with the 128-example generation/answer-CE subset.

Targeted interventions use 32 predeclared examples per split, eight per final operation. The intervention suppresses
the final instruction's pass-two slot-writer update, leaving later passes and suffix computation unchanged except
through the resulting cache. Ordinary generation is preserved for comparison; the 30 extra train baselines outside
the primary 128 are generated and recorded separately. An output effect is not semantic faithfulness; no effect may
reflect redundancy and does not prove state absence.

The endpoint also includes independent generation rescoring of all 830 primary, supplemental and intervened outputs
per arm, including prerequisite completeness and changed-completion/generated-suffix agreement. Each arm is scored
independently, avoiding cumulative whole-study scorer passes. Unautocast CPU FP32 full/cached replay uses exact
integer routes and fixed replay tokens. This is not a claim of BF16 serving equivalence or bitwise reconstruction.

## 4. Measured allocation proposal

The reference is the completed engineering document-position acquisition, not a new timing estimate. Slowest
measured training throughput is 23,379.85 loss tokens/s. Slowest measured evaluation/replay wall cost is
1.27733 s/example; the protocol budgets 1.5 times that, **1.91599 s/example**, for the longer and more varied split.

| Cost per arm | Budgeted seconds |
|---|---:|
| Training, 5,570,640 tokens at the slowest measured rate | 238.267 |
| Seven full periodic/final dev evaluations | 0.608 |
| Checkpoints, using the slowest measured checkpoint total | 8.010 |
| 640 generation/answer-CE/state-evaluation examples | 1,226.233 |
| 30 supplemental train intervention baselines | 57.480 |
| 160 additional targeted-intervention generations | 306.558 |
| 830 independently scored primary/supplemental/intervened outputs | 0.830 |
| Fixed checkpoint reconstruction allowance | 10.000 |
| **Total per arm** | **1,847.986** |

All three arms total **5,543.958 seconds = 1.539988 hours**. The watchdog bound is 2.309983 hours, plus 0.05 hours
for shutdown. The complete plan therefore requires 2.359983 hours before allocation headroom. Independent scoring
uses at least 0.001 seconds per output; the full per-example measurement is recorded in the protocol.

**Proposed allocation: 2.5 GPU-reservation hours.** It provides 0.960012 hours (38.40%) above the projected runtime,
satisfying the 20% planning margin, and 0.140017 hours above the watchdog-plus-shutdown requirement. At preparation,
8.214045098 native hours remain. This is a request only: no cap was increased and no reservation was created. A
future owner decision can accept, reduce with a correspondingly reduced equal-arm protocol, or reject it.

The earlier unlaunched `study_protocol.draft-v1.json`, which omitted supplemental-baseline and independent-scoring
work, is preserved as a draft only. The final `study_protocol.json` and sidecar bind all source and data identities.

## 5. Reproduction and execution gate

Run from `/home/seanjazm27/projects/universa-recurrent-varied-programs` with the pinned environment:

```bash
export PYTHONPATH=research/ti_tnn PYTHONDONTWRITEBYTECODE=1
export TI_TNN_SCALING_RUNS=/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_scaling
PY=/home/seanjazm27/projects/universa-recurrent-native/.venv-native/bin/python
DATA=/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_program_generalization
TOKENIZER=/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native/cache/tokenizer/tokenizer.json

# Construction checks only; no training.
CUDA_VISIBLE_DEVICES='' $PY -m pytest -q tests/test_ti_tnn_native_*.py

# Default dry run: validates protocol, sampler endpoint and live ledger; creates no run or reservation.
$PY -m native.run_program_study \
  --data "$DATA" \
  --repository /home/seanjazm27/projects/universa-recurrent-varied-programs \
  --python "$PY" \
  --tokenizer "$TOKENIZER"
```

The observed dry run reports `artifacts_created=false`, `reservation_created=false`, and
`sampler_endpoint_matches=true`; the `runs/` directory does not exist. The current full native suite has
**310 passing tests**. Independent data/evaluator review found no correctness blocker.

Only after a separate owner allocation should the exact same command add:

```text
--execute --authorize-hours 2.5
```

The runner refuses execution without both flags, refuses a different hour value or dirty source, and uses one
reservation for all three arms. It preserves failures and never completes a candidate by truncating its controls.
There are no trained checkpoints, learning curves, generalization results or intervention-effect results yet.
