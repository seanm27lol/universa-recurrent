# Five-training-seed replication: 2026-09-09

**Status: exploratory, user-supplied DGX reports; independently checked report arithmetic, not rerun training.**

## What changed in our understanding

Shared recurrence had lower mixture error than the one-pass model and the ambient recurrent model in all five runs. It beat untied depth and the dedicated four-step model in only three of five. The evidence favors keeping recurrence as a candidate, not declaring it the unique winning architecture.

## Results

All models share one held-out cohort of 4,000 examples. Lower error and time are better.

| Model/output | Mean MSE | Sample SD across training/data seeds | Mean of per-run median milliseconds | Shared lower MSE, out of 5 |
|---|---:|---:|---:|---:|
| shared/mixture | 0.049565 | 0.001412 | 13.282 | — |
| direct/mixture | 0.052676 | 0.001760 | 3.473 | 5 |
| untied/mixture | 0.050916 | 0.002432 | 14.660 | 3 |
| ambient/estimate_only | 0.052247 | 0.000337 | 3.644 | 5 |
| fixed_depth_4/mixture | 0.049900 | 0.000845 | 8.531 | 3 |
| gaussian_generator_reference/mixture | 0.047098 | 0.000000 | 20.668 | 0 |

The Gaussian reference is privileged: it knows the synthetic prior and noise level. Its zero error SD reflects one deterministic reference evaluated on the same cohort, not five independently fitted Gaussian models.

Shared recurrence reduces mean MSE by 5.90% relative to one-pass, but takes 3.82 times as long by the ratio of mean per-run median times. Against dedicated depth 4, the mean error reduction is only 0.67%, with 55.7% more time by that same aggregation. These are descriptive ratios, not significance tests.

## Per-seed accuracy: do not hide reversals

| Training/data seed | Shared | One-pass | Untied | Dedicated depth 4 | Ambient |
|---|---:|---:|---:|---:|---:|
| 6100 | 0.049563 | 0.050793 | 0.049245 | 0.048851 | 0.052827 |
| 7100 | 0.048400 | 0.052906 | 0.050087 | 0.049430 | 0.052169 |
| 8100 | 0.048344 | 0.051499 | 0.050391 | 0.050119 | 0.052126 |
| 9100 | 0.049692 | 0.055394 | 0.049659 | 0.049981 | 0.052165 |
| 10100 | 0.051827 | 0.052787 | 0.055197 | 0.051117 | 0.051947 |

## Audit and provenance

- Package 0.5.0, source commit `808e92dacd1890392654f7f810e26b889bbebaff`.
- Training seeds: 6100, 7100, 8100, 9100, 10100. Each changes initialization and synthetic training samples, not initialization alone.
- New claim-calibration seed 31000; common test seed 32000.
- NVIDIA GB10; batch size 1,024; 20 timing repetitions per case per run. Timing excludes model loading, transfers, calibration, Lingua generation and verification.
- All five report bundles say their Lingua check passed. Evaluation/timing row agreement, claim denominators, calibration-file hashes, timing medians, and supplied summary arithmetic were checked independently.
- Checkpoint bytes were not uploaded. This audit does not repeat checkpoint-bound verification or authenticate GPU execution.

Original report archive SHA-256:

```text
b79190dc4353740ab926cc24ccc7243afa0151d8fe7c9b910cd9a27f53d51ef0
```

Common test-cohort fingerprint:

```text
d5b390bdddde34674c5b76dcf99a6011cbf6fb39910754fa5f73649ca40ee3af
```

Reproduce the report audit using the original ZIP:

```bash
python scripts/audit_replication.py /path/to/replication-reports.zip --output audit.json
```

The script uses only the standard library, reads ZIP members without extracting them, and refuses to replace an existing output. It validates report consistency; it is not a model verifier.

## Limits and next decision

These five fits are not 20,000 independent training experiments. They share a calibration cohort and test cohort, retain differing training objectives across architectures, and do not establish cross-task generalization. Do not compare their absolute timings with previous runs without accounting for run conditions. No timing profiler was supplied.

Keep model code and the dual-output contract frozen while investigating [temporal Lingua](temporal_lingua.md). Include the direct and dedicated-depth-4 controls. A mathematical description, a numerical reconstruction, and a causally faithful explanation must remain separate claims.
