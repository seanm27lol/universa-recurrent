# Execution-settings timing audit

**Status: controlled re-benchmark only. No retraining, architecture change, or speedup claim.**

## Why this exists

The five-training-seed study ran training and timing in one Python process. Training
explicitly enabled PyTorch deterministic algorithms, so the later timing inherited
that process-global setting. Earlier standalone timing did not record the same state.
Absolute timings from those two studies therefore should not be compared directly.

This experiment asks a narrower question:

> How much do explicit deterministic-algorithm settings and batch size change the
> measured inference time of the already-trained models?

## Familiar analogy

If two cars are timed on different tire settings, a lap-time difference cannot be
credited entirely to the car. Re-run the same cars with the settings written down.

Here the "cars" are frozen checkpoints. The settings are:

- deterministic algorithms OFF versus ON;
- batch size 1,024 versus 4,096;
- one fixed 4,000-example held-out cohort;
- 5 warmup calls and 20 timed calls.

Every checkpoint/setting/batch combination runs in a **fresh Python process**, so
training cannot leave behind global PyTorch state.

## Models timed

The study times the exact numerical-estimate outputs that mattered in the five-seed
comparison:

- shared 8-step recurrence, mixture;
- dedicated 4-step recurrence, mixture;
- untied 8-step model, mixture;
- unstructured recurrent estimate;
- direct one-pass mixture.

Claim generation and Lingua checking are outside this timing question.

## Deterministic mode

When deterministic mode is ON, the child process calls:

```python
torch.use_deterministic_algorithms(True)
```

and receives:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

When it is OFF, the child calls `torch.use_deterministic_algorithms(False)` and
removes that environment setting. `torch.backends.cudnn.benchmark` is pinned OFF
in both modes rather than inherited.

This follows PyTorch's reproducibility guidance:
<https://docs.pytorch.org/docs/stable/notes/randomness.html>.

The setting can affect which kernels are permitted and therefore runtime. It does
not make timing portable across GPUs, drivers, PyTorch versions, or workloads.

## What is held fixed

For each trained checkpoint:

```text
same weights
same 4,000 examples
same output mode
same model depth
same warmup/repeat counts
```

The benchmark excludes checkpoint loading, host-to-device transfer, metric
calculation, calibration, Lingua generation, and checking. Those exclusions are
written into every child report.

## What counts as evidence

The parent summary treats each **trained checkpoint** as the replication unit.
Twenty timing repetitions are measurement repetitions, not twenty independent
models.

Report:

1. each checkpoint's raw timing repetitions;
2. median and 10th–90th timing quantiles;
3. median/mean timing across trained checkpoints;
4. deterministic-ON divided by deterministic-OFF timing;
5. reconstruction MSE and an estimate hash for each setting.

If deterministic ON/OFF produce different numerical estimates, report that rather
than silently treating them as identical.

## Limits

This experiment does not:

- retrain any model;
- prove deterministic mode caused every earlier timing difference;
- measure end-to-end service latency;
- test adaptive recurrence;
- test Lingua cost;
- establish that the fastest model is scientifically best.

Its purpose is to remove one execution-setting ambiguity before interpreting the
accuracy–time frontier.
