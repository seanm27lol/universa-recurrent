# Keep the current worksheet; request the full notebook when needed

**Status: implemented and tested on CPU; GPU speed and memory gains must be measured.**
This is a retention optimization around the existing model, not a new model.

## A familiar example

To calculate a running balance, you need the current balance and the next update.
You do not need to copy every earlier subtotal into a new list on each calculation.
For an audit, however, that list can be useful. Our solver now offers the same choice.

```text
                         SAME weights, updates, inputs and depth
                                      |
                   +------------------+--------------------+
                   |                                       |
          final-only adapter                     original numerical rollout
          current state only                     every requested step retained
                   |                                       |
          estimate + optional claim              SAME estimate + optional claim
                   |                                       |
          optional endpoint receipt              optional trajectory notebook
```

The original model and training implementation are unchanged. The final-only path
is opt-in, via `FinalStateEstimator.from_reference(engine)`. Historical `FixedEstimator`
continues to use rollout, so earlier experiment definitions do not silently change.

## Mathematics and code

For fixed context `c`, the recurrent update is `h_(t+1) = F(h_t,c)`. If inference
needs only the final `h_T`, keeping all `h_1,...,h_T` is unnecessary. Here, the
current state includes all candidate coordinates AND route logits; neither can
be discarded before the next update. Input context remains available throughout.

`neural/final_state.py` reuses the actual `_step` and depth-specific `_step_at_depth`
functions. The ambient control duplicates its short update expression and is
checked against the original rollout. Direct inference has no history to remove
and is a negative control. No early exit, reduced precision, compilation, new
training or change of candidate subspace is introduced.

Trajectory storage scales with depth. This does **not** say that total peak memory
or latency improves by the same factor: contexts, model parameters, temporary
operations and returned tensors still have costs.

## What the study measures

Run after pulling and refreshing the editable package:

```bash
python -m pip install --no-deps -e .
python scripts/check_release.py
python -m pytest -q
bash scripts/run_retention_study.sh /absolute/path/to/replication/results cuda
```

The helper uses the existing five `weights-*.pt` files and their saved
`study-SEED/calibration.json` files. We do not refit thresholds or alter weights.

| Measurement | Denominator and scope |
|---|---|
| Reference vs final-only inference | All 4,000 examples, batch caps 1,024/4,096; estimates, probabilities and discrete decisions compared |
| Peak extra CUDA allocation | Separate warmed call; PyTorch allocated bytes above start, not total device or host memory |
| Endpoint and trajectory generation | First FOUR examples, serial one-example calls; host copies and JSON serialization included |
| Parsing + mathematical checks | Those same four serialized records; no file deserialization or neural replay |
| Inference through record check | Directly measured for those four examples; not a sum of median stage costs |
| Checkpoint/calibration-bound verification | One additional call per record kind, including hashes and deserialization; noisy diagnostic, not a throughput claim |

Lingua costs are sampled for the shared-eight-step and dedicated-four-step models.
They are NOT the cost of recording all 4,000 benchmark examples. File writing,
initial input transfer and checkpoint loading remain outside inference and sampled
pipeline timing. Bound verification is reported separately, not hidden in a cheap
arithmetic check. For the combined sample pipeline, four serial inferences precede conversion and
checking. The first four inputs are fixed by order, not selected for success.

Each checkpoint/batch condition uses a fresh process. Deterministic enforcement
is explicitly OFF by default; `--deterministic` on the module CLI turns it ON.
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off, float32 matmul precision `highest`, and
cuDNN benchmarking off are explicit and recorded. This is not necessarily the
same backend configuration as every historical run. Paired implementations and
checkpoint/batch jobs have recorded randomized orders.

## What a trajectory check does NOT establish

The checker verifies candidate coordinate decoding, constraints, probabilities,
recorded residuals, progress arithmetic and agreement with the final endpoint.
It does not execute the learned update to certify each transition. A geometrically
consistent fabricated path can pass these properties. This is not execution
attestation, hidden-neuron semantics or proof the structural choice was correct.
An endpoint-only record makes no claim about discarded intermediate history.
A mixture never inherits a single candidate's structural certificate.

The numerical check uses fixed tolerances; a caller cannot enlarge them in a record.
The GPU benchmark rejects any disagreement in discrete routes or claim masks even
when the mean error is unchanged. Floating-point tensors must agree within `atol=1e-7,
rtol=1e-6`; per-field maximum discrepancies and exact-equality indicators are saved.

## Established grounding

This is ordinary state reuse and optional execution logging, not a new theorem.
PyTorch's [inference mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)
removes autograd bookkeeping, but does not replace model evaluation mode. We use both.
See [reproducibility settings](https://docs.pytorch.org/docs/stable/notes/randomness.html),
[CUDA timing considerations](https://docs.pytorch.org/docs/stable/notes/cuda.html), and
[peak allocated memory](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html).

## Release checks

The package version has one authoritative literal (`0.5.2`). Source and separately
installed-wheel checks include all three new modules. Tests cover shared/untied/ambient
endpoints, one-pass controls, multiple depths and batch sizes, no-history execution,
unchanged state dictionaries, masking, ties, thresholds, numerical failures, corrupted
records, scoped verification, and a tiny frozen-checkpoint benchmark. CPU tests cannot
substitute for measuring your DGX. A passing test is not a speedup result.
