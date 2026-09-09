# Execution audit: test the settings, not a new model

## Familiar picture

A car can take longer over the same route because its gearbox is in a different
mode. A fair comparison uses the same car, route, load, and measurement procedure.
Likewise, a GPU benchmark must not silently inherit its settings from training.

The five-run experiment trained and benchmarked in one process. Training sets
`torch.use_deterministic_algorithms(True)`, but the old benchmark did not record
that state. This makes a historical timing comparison uncontrolled. It does NOT
establish that determinism caused the timing change or invalidate the old errors.

```text
five EXISTING checkpoints + saved claim cutoffs + same test cohort
                              |
             fresh process for every checkpoint/profile/pass
                              |
                 +------------+-------------+
                 |                          |
              normal                   deterministic
       enforcement disabled          enforcement enabled
                 |                          |
                 +------------+-------------+
                              |
                 compare time AND actual outputs
```

## Run on the Spark

```bash
bash scripts/run_execution_audit.sh /absolute/path/to/replication/results cuda
```

Point to the folder containing `weights-6100.pt`, the other `weights-*.pt`,
`study-*` folders, and `summary.json`. The report ZIP alone does not contain the
weights. The runner checks all checkpoint hashes before launching any measurement.
No retraining, architecture change, threshold tuning, or package installation is
performed. Keep the GPU otherwise idle.

## What is held fixed?

| Item | Treatment |
|---|---|
| Weights and structural library | Existing checkpoint, checked before and after |
| Claim thresholds | Read from each original `calibration.json`; never recalibrated |
| Observations | Original test seed, noise, mask probability and count; full fingerprint checked |
| Batch shape | Original benchmark's batch size |
| Precision | IEEE float32 backend settings, no autocast or compilation |
| Host threads and cuBLAS workspace | One thread and `:4096:8` in BOTH profiles |
| Device logging | Read-only snapshots, outside the timed region |
| Workload | All 23 original output modes, each at its trained fixed depth |

These controlled common settings are NOT claimed to reconstruct the unrecorded
historical settings. In particular, the normal profile is not a maximum-speed
configuration and disabling deterministic enforcement does not guarantee that
any particular operation becomes nondeterministic.

The deterministic profile retains PyTorch's default memory-fill safeguard.
Filling otherwise uninitialized allocations may itself take time. That is part
of the profile comparison, not an isolated test of one kernel implementation.
Neither profile changes CUDA installations, GPU power limits, or driver settings.

## Measurements and checks

The runner uses two order-reversed passes per checkpoint (20 fresh worker
processes for five checkpoints). Within each worker it warms every output mode
and interleaves 20 measured repeats. Repeats and passes are repeated measurements
of the same trained models, NOT additional independent training runs.

For method `m`, checkpoint `s`, and pass `b`, it reports

`R[m,s,b] = median_time_deterministic / median_time_normal`.

A ratio above one means that deterministic execution was slower in that pair.
Raw timings, quantiles, both MSEs, all configuration flags, profile order, process
IDs, source identity, and checkpoint/input hashes are retained.

All output tensors are compared between profiles after timing. Floating values
must agree within `atol=1e-6, rtol=1e-5`; discrete claims must agree exactly. A
mismatch is reported, not hidden: its matched-output ratio is null. Timings may
still be inspected but must not be called a same-output speedup. Differences
near a claim threshold can matter even when average MSE hardly changes.

Timing retains the existing rollout and output allocations. It excludes loading,
transfers, calibration, Lingua, verification, and process startup. This is NOT an
end-to-end service benchmark, a peak-performance benchmark, or an efficient-witness
result. Before/after device snapshots do not eliminate thermal/load confounds.

## Output

`summary.json` summarizes paired measurements; each worker retains `eval.json`,
`benchmark.json`, `execution.json`, and its log. Raw numerical predictions are
stored locally in non-pickle NPZ form for comparison. The final `*-reports.zip`
contains JSON and logs only, not model weights or prediction arrays. Previous
results are unchanged; an existing output directory is refused.

## Known foundations

- PyTorch [reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html)
  distinguishes random seeds, deterministic algorithms, and performance costs.
- [Deterministic memory filling](https://docs.pytorch.org/docs/stable/deterministic.html)
  explains why otherwise empty allocations may cost more under deterministic mode.
- [CUDA precision controls](https://docs.pytorch.org/docs/stable/notes/cuda.html)
  recommend the new precision API and warn against mixing it with legacy TF32 flags.

These sources justify the controls, not a prediction about which model will win.
CPU integration tests exercise fresh workers, invariants, rejection paths, and
archive scope. The actual GB10 result remains to be measured on the user's GPU.
