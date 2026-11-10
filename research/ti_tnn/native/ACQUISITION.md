# Native acquisition: intact examples, position diagnosis, and a matched correction

A model can have low packed training loss and still fail to generate answers from a standalone prompt. The first
32-example acquisition reached only 11/32 exact training answers in every architecture. Those 11 were exactly the
examples at the start of a packed row. Replaying the other 21 at their trained absolute positions recovered every
answer. This identified a data-position mismatch, not an MoE advantage.

The separately frozen document-local correction reached **32/32 exact training answers in every arm**, exceeding the
31/32 memorization target. The separate development result is **0/32 in every arm**. A targeted after-state-update
intervention changed **0/96 completions**. Thus complete-program memorization is demonstrated; generalization and a
usable semantic state path are not.

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

The final development router distributions include one untyped starved expert in pass four (expert zero, 0/2,539
token-level decisions, across 11 packed rows and two batches). Typed pass-four expert zero is used in 6.85% of decisions. All per-pass counts, probabilities,
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

## 6. Corrective outcome: memorization passes, generalization and state use do not

The corrective run is `20261110T091534-c25a2b1b`, at source commit `a708fc5`, in:

```text
runs/ti_tnn_native_acquisition_local_positions/acquisition/20261110T091534-c25a2b1b/
```

Every arm again completed exactly 4,096 updates and 5,201,920 loss-bearing tokens, with no failed or truncated arm.
All generated training answers are canonical, EOS-terminated and exact. Answer-token cross-entropy is separate from
register error; both are reported below.

| Metric | Dense | Untyped MoE | Typed MoE |
|---|---:|---:|---:|
| First → last logged train LM loss | 10.8546 → 0.04315 | 10.9056 → 0.04311 | 10.8974 → 0.04304 |
| Exact training answers | **32/32** | **32/32** | **32/32** |
| Training answer-token CE | 0.000442 | 0.000441 | 0.000476 |
| Training format / EOS / truncation | 32 / 32 / 0 | 32 / 32 / 0 | 32 / 32 / 0 |
| Training per-register accuracy / mean absolute error | 1.0 / 0.0 | 1.0 / 0.0 | 1.0 / 0.0 |
| Training state readout, every pass | 100% | 100% | 100% |
| Exact development answers | 0/32 | 0/32 | 0/32 |
| Development answer-token CE | 3.053245 | 3.424107 | 3.443900 |
| Development format / EOS / truncation | 14 / 32 / 0 | 8 / 30 / 2 | 13 / 32 / 0 |
| Development mean absolute register error, valid outputs | 10.5179 | 12.0625 | 11.1154 |
| Development state-readout range | 5.7–8.0% | 5.7–6.3% | 4.8–6.5% |
| Development LM loss | 1.986725 | 2.072663 | 2.167742 |

The exact-lookup baseline is also 32/32 on these repeated training prompts and 0/32 on unseen development prompts.
The supplied-program executor is 32/32 on both, as expected for a reference executor rather than a neural model.
These baselines are in `baselines.json`; neither is substituted for the model. The training result therefore shows
that the corrected pipeline can memorize complete programs, not that it learned the general register algorithm.

Final corrective routing has no zero-count expert in any pass. Untyped pass-four expert zero is used in 3.47% of
development decisions and typed pass-four expert zero in 4.49%; full usage, probability and load diagnostics are
saved. No routing or state-readout difference is credited as an architectural advantage from this single seed.

### Corrective checkpoints and costs

| Metric | Dense | Untyped MoE | Typed MoE |
|---|---:|---:|---:|
| Final checkpoint SHA256 | `4d4c0ede6afcf32c353e228866d8a9084c97b1a30e51c4e6dc45cdb668586b0c` | `bf12f250beaa995d66397e056aa93d43cbb8c85199ad09ad510f627830ed447e` | `b671d0a46f6633deab05d37d6142c14116b3904a5c3b9b5caa65b5e1a0688d67` |
| Training component time, seconds | 222.496 | 199.760 | 208.054 |
| Actual loss tokens/second | 23,379.8 | 26,040.9 | 25,002.7 |
| Data / forward / backward / optimizer, seconds | 3.28 / 47.91 / 130.26 / 41.05 | 2.91 / 54.54 / 102.07 / 40.24 | 3.03 / 57.16 / 106.75 / 41.12 |
| Checkpoint time, seconds | 7.680 | 8.010 | 7.368 |
| Train + CPU evaluation/replay wall, seconds | 303.799 | 284.309 | 297.171 |
| Peak CUDA allocated / reserved, GiB | 1.75 / 6.35 | 1.76 / 6.14 | 1.76 / 6.13 |
| FP32 reconstruction report SHA256 | `47e2235d926db894a741c1095538321e4d114e07d384a732feefc4a31de6172a` | `b2eab95f8f0726b1b2564c9f4456601b9112c56a92481f51c0575e4928d1ef03` | `fa99b495af7ff36443fb89e629cb0854d99f7df8b460d99c608d5dadade6ef6c` |

This reservation consumed **0.246031621 hours** (885.714 seconds), including training, evaluation and replay.
The two acquisition reservations total **0.482232927 hours**, below the one-hour authorization. Native cumulative
usage is now 3.785954902/12 hours, leaving 8.214045098 hours. No additional training was launched after inspecting
these outcomes.

### Learned-state intervention

`learned_state_intervention.json` (SHA256 `f033e4ff16a23b9d128b2be3b93124d72544af2c5aaacfdb3bb15e2c9f67b991`)
identifies the final instruction's last overlapping prompt token and suppresses all four slot-writer output vectors
at that token during pass two only. That is the first supervised after-state in `before,after,after,after`. The
incoming slot state is retained; later passes and suffix forwards receive no direct suppression. Generation uses
the saved attention cache. Baselines run without instrumentation and reproduce all 96 saved completions.

| Arm | Baseline exact | Suppressed exact | Changed completions / generated suffixes |
|---|---:|---:|---:|
| Dense | 32/32 | 32/32 | 0 / 0 |
| Untyped | 32/32 | 32/32 | 0 / 0 |
| Typed | 32/32 | 32/32 | 0 / 0 |

This negative generation-level intervention may reflect redundant token/hidden-state paths. It does **not** prove
state is absent. Conversely, a changed output would have established causal dependence, not semantic faithfulness.
Consequently, this study does not establish a usable learned semantic state interface, despite perfect training
readout and the causal construction test.

## 7. Numerical and independent-review scope

All six parent/corrective checkpoints passed the declared FP32 full/cached reconstruction rule, including exact
integer routes and exact fixed-replay generated tokens. Independent re-review measured maximum relative RMS errors
below approximately \(6\times10^{-7}\), well inside the 0.05 serving tolerance. **Strict `allclose` is false**:
these are tolerance-based FP32 results, not bitwise equality or a reproduction of BF16 CUDA training. The earlier
BF16 route failure and original strict FP32 failures remain preserved; no acceptance history is rewritten.

The independent scorer, which imports no producer modules, executed all 64 saved programs per run and reproduced
both generation-diagnostics artifacts with zero score changes. This is independent execution/rescoring of saved
completions, not independent model regeneration or external replication. Checkpoint SHA256 values identify local
bytes; they do not authenticate a remote execution.

## 8. Next bounded experiment proposal — not launched

A useful next step is a separately authorized acquisition/generalization diagnostic on **new, disjoint programs**,
not more repetitions of these 32 prompts. Freeze equal data, initialization and update budgets for dense, untyped
and typed models; include unseen compositions and longer programs. Report exact complete generation, answer CE,
lookup/executor baselines and per-register/EOS diagnostics separately for train and holdouts.

For state specifically, include counterfactual pairs that require different final states despite a shared visible
suffix. Predeclare targeted after-state interventions and distinguish direct token memorization, redundant paths,
and demonstrated semantic state dependence. A positive training probe alone remains insufficient. Size any such
run from the measured throughput above and request a separate bounded allocation before launching it. No 124M
campaign, WSP edit or scaling claim follows from this memorization result.

## 9. Exact corrective launch and evidence commands

The following is the executed source-bound training interface. A new invocation creates a new immutable attempt
and consumes additional reservation time; it is not a request to relabel or overwrite this completed run.

```bash
export PYTHONPATH=research/ti_tnn
export PYTHONDONTWRITEBYTECODE=1
export TI_TNN_SCALING_RUNS=/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_scaling
PY=/home/seanjazm27/projects/universa-recurrent-native/.venv-native/bin/python

$PY -m native.run_acquisition \
  --data /home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_acquisition_local_positions \
  --repository /home/seanjazm27/projects/universa-recurrent-native-followup \
  --python "$PY" \
  --tokenizer /home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native/cache/tokenizer/tokenizer.json

# CPU-only learned-state intervention; the default output is new and non-overwriting.
CUDA_VISIBLE_DEVICES='' $PY -m native.diagnose_learned_state \
  --run /home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_acquisition_local_positions/acquisition/20261110T091534-c25a2b1b \
  --data /home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_acquisition_local_positions
```

For weight-free reproduction, set `RUN` to the preserved run above and `DATA` to its containing acquisition data
root, then use the commands in section 5. The independent rescorer now refuses to overwrite an existing output;
the diagnostic commands above also default to new, non-overwriting paths.
