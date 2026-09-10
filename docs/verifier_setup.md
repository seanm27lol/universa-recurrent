# Validate the reference once; check every new calculation

## A familiar example

A laboratory checks its reference manual's edition before a session. It then
checks every calculation against that edition instead of re-reading the whole
manual for each line. Switching to a new edition requires a new session.

This experiment applies that ordinary setup-amortization idea to Lingua. It is
not a new mathematical theorem, neural architecture, or cache of accepted answers.

```text
trusted checkpoint + calibration files
                   |
       hash, load, validate ONCE
                   |
       immutable reference snapshot
                   |
new receipt -> all numerical checks + reference matching -> pass / reject
```

The existing file-bound checker, models, training, tolerances, and record formats
are unchanged. This adds an opt-in research helper under `scripts/`, not a public
serialized trust token. The package version remains 0.5.2.

## What the uploaded retention experiment established

Source: user-supplied retention reports, ten checkpoint/batch conditions, five
frozen training checkpoints, 4,000 common test examples. This is exploratory
reported GPU evidence, not locally rerun training.

| Model | Incremental CUDA tensor allocation reduction, cap 1,024 | Cap 4,096 |
|---|---:|---:|
| Shared depth 8 | 35.0% | 17.4% |
| Dedicated depth 4 | 16.0% | 6.6% |
| Untied depth 8 | 51.4% | 29.3% |

These are warmed peak-extra PyTorch allocated bytes, not total device memory.
All reported paired output fields matched exactly. The shared model's median
paired time reduction was 2.5% / 3.2% for caps 1,024 / 4,096; the four-step model
was about 4.8% slower on the median large-batch pair. The one-pass negative control
also varied, and individual timing spikes exceeded ten times their own row median.
Do not present this as a large or universal speed improvement.

For shared depth 8, in the cap-1,024 workers, median-of-worker medians for FOUR
sampled records were 0.228 ms for endpoint parsing/property checks and 0.955 ms
for trajectory parsing/property checks. Trajectories check more properties, so
these are different guarantees, not an equal-guarantee compression comparison.
Single checkpoint/calibration-bound calls were approximately 12-13 ms each and
include hashing/loading. That suggests a setup-amortization experiment; it does
not demonstrate its speed before the next measurement.

## Three measurements, with explicit denominators

1. **files_per_record**: parse each receipt, perform all property checks, hash/load
   trusted files for each receipt, and match metadata/policy as before.
2. **prepare_once_per_batch**: prepare one snapshot, then parse and fully check each
   receipt. The measured total INCLUDES preparation.
3. **reuse_prepared_snapshot**: fully check each receipt against an already prepared
   snapshot. This is steady-state checking; its timing EXCLUDES preparation.

For N records, the proposed cost comparison is `N*(setup+check)` versus
`setup+N*check`. These are accounting models, not predicted measured speedups.
Raw repetitions and execution order are recorded. Filesystem caches are warm;
file loading does not mean cold-storage latency. Fixed CPU thread settings are
recorded. No GPU computation, inference, new calibration, or training occurs.

The first four saved records per model are cycled to form batches of 1, 4, 16,
and 64. This is intentionally a microbenchmark of repeated verification work,
NOT 64 independent examples, new test accuracy, or complete trajectory coverage.
Only the cap-1,024 records are used, so the duplicate cap-4,096 samples are not
counted as additional evidence.

## Trust and invalidation

Prepared verification still invokes every existing per-record geometry,
probability, mixture, policy, and trajectory diagnostic check. It then matches
checkpoint/calibration hashes, the candidate library, the model/depth, and the
calibrated policy to the pinned snapshot. It NEVER reuses an earlier PASS verdict.

The caller must supply trusted local reference files when preparing. Files are
hashed before and after preparation to detect ordinary concurrent changes. The
snapshot is immutable and does not keep model objects or live file paths.
If a file changes later, existing snapshots still mean the OLD pinned edition;
create a new snapshot for the new version. Prepared mode does not promise to
notice current disk changes. The benchmark checks hashes again before completion.
This is not protection against a hostile process that controls the reference files.

A valid record still does not prove a correct structural choice, learned
transition, human-readable neuron meaning, or authentic remote execution. No
candidate constraint certificate transfers to a mixture.

## Run

```bash
bash scripts/run_verifier_setup_study.sh \
  /path/to/retention-reports.zip /path/to/replication/results
```

No retraining and no new GPU dependency. New reports go into a fresh local folder;
existing receipts and weights are not altered. Tests cover both accepted records
and rejected corruptions, stale snapshots, file changes, immutable snapshots,
no neural replay/file access in steady-state checking, and the complete CLI.

## Established implementation references

- [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html)
  explains checkpoint storage and restricted weights-only loading. We retain the
  repository's restricted loader and its trusted-source requirement.
- [CUDA memory accounting](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)
  distinguishes allocated tensor memory from allocator reservations.

The data suggest a useful memory optimization and a testable verification-cost
opportunity. They do not establish a general latent-reasoning or interpretability
breakthrough. CPU tests are not DGX performance results.
