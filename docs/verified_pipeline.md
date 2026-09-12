# Time a request through its completed record check

The completed DGX run and local receipt audit are documented in
[the September 11 results](verified_pipeline_20260911.md). The experiment
description below records the original measurement design.

A service receives a batch of observations, computes an estimate, builds a
numerical receipt for every observation, and checks every receipt. The relevant
cost is the time until those checks finish. A fast model or a fast isolated
checker alone does not establish that total.

The [setup study](verifier_setup_20260910.md) found substantial savings from
preparing a trusted reference once. This next exploratory experiment measures
whether that saving remains useful when the complete request work is included.

## Workload

Reuse the five saved replication checkpoints and their calibration artifacts.
Generate a common fresh cohort with seed 36000, distinct from the documented
earlier evaluation seeds. The runner rejects overlap with checkpoint training
and calibration seeds and records the dataset hash. It does not retrain or
choose a new claim threshold.

Default batch counts are 1, 16, 64 and 256, using nested prefixes of the same
256-example cohort. The cohort is shared across checkpoints and timing repeats;
these repetitions do not multiply the number of independent inputs. Actual
observation/mask uniqueness is checked and reported.

Use shared eight-step and dedicated four-step models with both endpoint and
trajectory receipts. Include the direct one-pass model with endpoint receipts
as a control. Comparisons of verification modes are made within a fixed model,
retention type, checkpoint and batch count.

## Measured boundary

```text
CPU input batch already in memory
  -> transfer to compute device
  -> batched inference and output construction
  -> host receipts and JSON serialization for EVERY example
  -> strict JSON parse and bound verification for EVERY example
  -> all checks completed
```

Three modes use this same boundary:

- `files_per_record`: load the trusted reference for each record.
- `prepare_once_per_batch`: prepare inside the measured request, then check all
  records against that reference.
- `reuse_prepared_snapshot`: check against an existing reference; initial
  preparation is separately reported and excluded from this request timing.

GPU synchronization delimits the timings. Total time is directly measured, not
constructed by adding stage medians. Mode order is randomized on each of five
measured repetitions. Model loading is separately reported startup work;
dataset synthesis, network transport, queueing and report-file writing are
outside the request measurement. This is a warmed local request benchmark,
not a deployed-service latency measurement or a cold-start benchmark.

## Gates and evidence

Endpoint and rollout outputs must agree within the existing fixed numerical
tolerances, with identical discrete decisions. Every timed call must verify the
expected number of receipts. Canonical receipts from the maximum batch are
saved, and their acceptance is compared between the original and prepared
checkers. Changes to reference files cause the study to fail.

The reports include inputs/receipt identities, all canonical receipts, check
counts, raw timings and execution order. They exclude model weight files.
Receipts permit an independent arithmetic/geometry audit; file binding still
requires the corresponding trusted checkpoint and calibration bytes.

Local CPU smoke tests are implementation checks. At implementation time the DGX
results were unmeasured; the completed run is linked above. This protocol was not
a new neural-accuracy study and is not preregistered confirmatory evidence.

Local validation on September 11, 2026: 362 tests passed; three CUDA equivalence
tests and the optional upstream integration test were skipped. The release check
and shell syntax checks passed. The 22 new pipeline tests include CPU worker
coverage for all five model/retention cases, a subprocess CLI smoke run, rejected
corrupt records, seed overlap, reference changes, incomplete batches and input
mismatch. Python 3.12, PyTorch 2.14.0, NumPy 2.5.3, macOS arm64. These tests use
tiny local fixture checkpoints; they are not measurements of the user's trained
DGX checkpoints.

## Run

```bash
bash scripts/run_verified_pipeline_study.sh \
  "$HOME/projects/universa-recurrent-git/runs/replication-20260909-173545-3Lc0us/results" cuda
```

The runner creates a fresh output folder and a reports ZIP in Downloads. For a
portable source checkout, set `UNIVERSA_PYTHON` to an existing compatible Python
environment; the runner selects the source shipped with the experiment. The
file-per-record baseline does thousands of reference loads, so this is a longer
run than the earlier isolated setup study. Keep other GPU work idle.

For smaller smoke runs and configurable counts, models, seeds and repeats, use
`python scripts/benchmark_verified_pipeline.py --help`.
