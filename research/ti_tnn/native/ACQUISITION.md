# Native acquisition: intact examples, position diagnosis, and a matched correction

A model can have low packed training loss and still fail to generate answers from a standalone prompt. The first
32-example acquisition reached only 11/32 exact training answers in every architecture. Those 11 were exactly the
examples at the start of a packed row. Replaying the other 21 at their trained absolute positions recovered every
answer. This identified a data-position mismatch, not an MoE advantage.

## 1. Frozen acquisition

The first run is `20261110T075533-a8827fef`, at source commit `60fec73`. Its preserved directory is:

```text
runs/ti_tnn_native_acquisition/acquisition/20261110T075533-a8827fef/
```

- Dense, untyped MoE and typed MoE use the existing engineering configuration: 21,860,352, 21,415,300 and
  21,499,780 unique trainable parameters respectively.
- All use four recurrent passes, the same randomly initialized seed (`20261110`), data order and tokenizer.
- Each arm has 4,096 AdamW updates and 5,201,920 loss-bearing tokens: 2,048 repetitions of the same 2,540-target
  packed epoch, batch size eight. The small second batch is retained; no examples are dropped.
- The 32 complete training examples have eight each whose first instruction is set/add/multiply/copy. Initial
  values and all program source are in the prompt. The separate 32-example development set is disjoint.
- The tokenizer is `openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, tokenizer SHA256
  `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`.
- The objective is the existing LM + load-balance + route + state objective. There is no answer-focused loss,
  distillation, Qwen supervision, adaptive stopping, architecture change, or 124M training.

The examples, source manifest, tokenizer, arrays and selection are bound by `source_provenance.json`,
`acquisition_selection.json`, `manifest.json` and the protocol hash. The engineering configuration's nominal
decoded-token MACs are 13,176,000 dense, 13,179,072 untyped and 13,182,144 typed at the declared context rule;
these are not simultaneously parameter-matched or a claim of equal measured execution cost.

## 2. First run: negative acquisition result

All arms completed the same endpoint and passed FP32 full/cached reconstruction. Complete greedy generation used
only the prompt, with at most 48 new tokens. Training-set exact match measures memorization only.

| Metric | Dense | Untyped MoE | Typed MoE |
|---|---:|---:|---:|
| First → last logged train LM loss | 10.8168 → 0.01353 | 10.8877 → 0.01334 | 10.8780 → 0.01338 |
| Whole packed train LM loss | 0.013059 | 0.013044 | 0.013044 |
| Exact training answers, ordinary zero-based generation | 11/32 | 11/32 | 11/32 |
| Exact development answers | 0/32 | 0/32 | 0/32 |
| Train answer-token CE | 2.277051 | 2.759172 | 2.456660 |
| Development answer-token CE | 3.407907 | 4.340158 | 3.771772 |
| Format-valid training generations | 15/32 | 13/32 | 16/32 |
| Training EOS / truncation | 29/32 / 3/32 | 29/32 / 3/32 | 28/32 / 4/32 |
| Training mean absolute register error, valid outputs | 2.2500 | 1.3462 | 2.6250 |
| Training after-state readout, every pass | 100% | 100% | 100% |
| Development state readout range | 3.1–4.5% | 4.5–5.7% | 2.6–3.7% |

Per-register accuracy over format-valid training outputs is 0.7333 for every dense register, 0.8462 for every
untyped register and 0.6875 for every typed register. Full completions, per-register errors, first divergences and
EOS behavior are in each arm's `evaluation.json` and the independently executed `generation_diagnostics.json`.
The program source is in the prompt; these failures are not attributed to a missing parser or truncated prompt.

The final development router distributions include one untyped starved expert in pass four (expert zero, 0/16
packed-row decisions). Typed pass-four expert zero is used in 6.85% of decisions. All per-pass counts, probabilities,
load balancing and starvation diagnostics remain in `train-result.json`; routing differences do not explain the
identical row-first failure pattern.

### Cost and checkpoint identity

| Metric | Dense | Untyped MoE | Typed MoE |
|---|---:|---:|---:|
| Training component time, seconds | 186.278 | 199.595 | 208.148 |
| Actual loss tokens/second | 27,925.6 | 26,062.4 | 24,991.4 |
| Data / forward / backward / optimizer, seconds | 2.97 / 43.73 / 98.46 / 41.12 | 2.92 / 54.30 / 101.50 / 40.88 | 2.93 / 57.44 / 106.36 / 41.42 |
| Checkpoint time, seconds | 8.246 | 6.690 | 7.654 |
| Train + CPU evaluation/replay wall, seconds | 268.768 | 286.875 | 294.207 |
| Peak CUDA allocated / reserved, GiB | 1.75 / 6.35 | 1.76 / 6.14 | 1.76 / 6.13 |
| Final checkpoint SHA256 | `b2393fdb80f05f0758a8dac9791d5a5a65d89ea548150d48f74adc1d77bca8c7` | `718e0b95d0339e0da477318997b16652e0e47dba6e9f221a2b544da9967a9815` | `89161b1c673451c4778b8c900eac8bc954f15f8925e65d9640e9cf26025bf581` |

The complete three-arm run, including loading, warm-up, checkpointing, evaluation and reconstruction, consumed
**0.236201306 GPU-reservation hours** (850.325 seconds). No arm was truncated or retried. Native usage after this
run was 3.539923281/12 hours, leaving 8.460076719 native hours. The separate acquisition allowance retained
0.763798694 hour.

## 3. Why the shared failure occurs

The intact builder keeps each example in one row and blocks cross-document attention. In the original data,
however, an example beginning at column \(s_d\) received position \(p_t=t\). Standalone generation begins at
\(p_t=t-s_d\). Only examples with \(s_d=0\) saw the same positions in training and evaluation.

`position_diagnostic.json` verifies the complete prompt/answer token and label alignment, reproduces every saved
zero-position completion, and then replays with the trained starting position as an explicit diagnostic intervention:

| Arm | Zero-position exact | Trained-position exact | Answer CE, zero → trained | Other-token CE, zero → trained |
|---|---:|---:|---:|---:|
| Dense | 11/32 | 32/32 | 2.277051 → 0.000437 | 1.227538 → 0.016499 |
| Untyped | 11/32 | 32/32 | 2.759172 → 0.000412 | 1.262233 → 0.016487 |
| Typed | 11/32 | 32/32 | 2.456660 → 0.000433 | 1.227123 → 0.016481 |

All 96 zero-position completions match the saved evaluation. The 11 successes are precisely the 11 row-first
examples; trained-position replay produces canonical answers with EOS and no truncation for all 96 arm/examples.
This establishes that the position mismatch is sufficient to explain all 21 non-row-first failures in this run.
It is not a retrained result or evidence of held-out generalization.

The first gradient artifact computed sums of per-unit squared norms, not norms of summed aggregate gradients.
That analysis is preserved as `loss_gradient_diagnostic.json` but is **not used for aggregate-gradient claims**.
`loss_gradient_diagnostic_v2.json` (SHA256 `9200a655d4df141fb0ee2e373fb5455be803ed4a7118ad1d90a9296e456ac55f`)
recomputes the gradients by summing parameter-gradient vectors before taking norms and dot products. Its CE values
independently reproduce the saved answer CE. Neither diagnostic updates a weight; neither uses a GPU reservation.

| Arm | Answer-gradient L2, zero → trained | Full-LM-gradient L2, zero → trained |
|---|---:|---:|
| Dense | 22.814866 → 0.002389 | 7.018528 → 0.006591 |
| Untyped | 14.875066 → 0.002750 | 4.899817 → 0.010155 |
| Typed | 14.854524 → 0.003288 | 4.397263 → 0.008716 |

The table covers all parameters; per-group values and cosine similarities are in the v2 JSON. These are gradients of
global token-count-mean losses, with auxiliary objectives excluded. They diagnose the position-dependent loss and
do not represent an optimizer step or a proposed weight edit.

A perfect training-state probe also does not establish that generation uses the learned state semantically. The
causal construction test establishes a path through later attention, not learned reliance on that path.

## 4. Separately frozen document-position correction

The corrective data directory is `runs/ti_tnn_native_acquisition_local_positions/`. Its protocol is
`native-acquisition-protocol-v2-document-positions`. The only training-input change is:

```text
position_policy="row"  ->  position_policy="document"
```

For every document \(d\), the corrected array uses \(p_t=t-s_d\), including a reset at each row start and zeros in
padding. `position_correction_audit.json` records that only `train/positions.npy` and `dev/positions.npy` changed.
The other 18 arrays, all examples, loss masks, route/state targets and consumed-document indexes are identical.
Selection metadata adds only explicit source provenance. The model, objective, seed, sampler, update/token endpoint,
and scoring are unchanged. The original `row` behavior remains available for exact reproduction; original arrays,
checkpoints, scores and protocols are not rewritten.

The parent protocol, run status and two diagnostic files are hash-linked. The parent ledger charge is 0.236201306
hours. The corrective projection is 0.39 hours, for a combined 0.626201306 hours, retaining over 20% of the
one-hour acquisition allowance. The corrective watchdog limit is 0.713798694 hours; parent + watchdog + a
0.05-hour shutdown allowance is at most one hour. A new run is an explicitly named data-correction diagnostic,
not a relabeling of the first negative result.

## 5. Reproduction interfaces

Run from the follow-up repository with `PYTHONPATH=research/ti_tnn`, `PYTHONDONTWRITEBYTECODE=1`, and the pinned
native Python environment. `RUN` and `DATA` below name the particular preserved attempt; output files refuse to
overwrite existing diagnostics.

```bash
PY=/home/seanjazm27/projects/universa-recurrent-native/.venv-native/bin/python
export TI_TNN_SCALING_RUNS=/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_scaling

# CPU construction and position regression tests.
CUDA_VISIBLE_DEVICES='' $PY -m pytest -q tests/test_ti_tnn_native_*.py

# Independent execution and exact/diagnostic scoring, with a new output path.
$PY research/ti_tnn/native/independent/rescore_generations.py \
  --study "$RUN" --data "$DATA" --output "$RUN/generation_diagnostics.reproduced.json"

# CPU-only position counterfactual and correctly aggregated gradient diagnostics.
CUDA_VISIBLE_DEVICES='' $PY -m native.diagnose_positions --run "$RUN" --data "$DATA"
CUDA_VISIBLE_DEVICES='' $PY -m native.diagnose_loss_gradients --run "$RUN" --data "$DATA"

# Weight-free review export (choose a new output directory).
$PY research/ti_tnn/native/independent/export_acquisition_bundle.py \
  --run "$RUN" --data "$DATA" --output "$RUN/evidence"
```

The exact launch command and corrective measurements are recorded with the corrective run's `launch.json`,
`run_status.json` and this report's results section after execution. No WSP editing is authorized by this study.
