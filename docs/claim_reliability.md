# When a checked answer chooses the wrong structure

The completed DGX run and independent audits are documented in
[the September 11 results](claim_reliability_20260911.md). The original protocol
below and its executable specification are preserved unchanged.

A noisy, partially observed flow can fit more than one structural explanation.
A receipt can verify that a candidate satisfies its declared constraints even
when that candidate is not the structure that generated the observation. The
[September 11 exploratory analysis](verified_pipeline_20260911.md) found this
distinction in practice: all saved receipts passed, while about 13.4% of the
shared model's issued claims selected the wrong synthetic label.

This study asks how often that happens on fresh inputs, and whether eight
recurrent steps improve on the dedicated four-step model. Think of an inspection
system that can send uncertain cases for review: both the frequency of automatic
decisions and the error rate among those decisions matter. This is the familiar
[selective-classification problem](https://arxiv.org/abs/1705.08500), not a new
meaning of mathematical verification.

## Fixed experiment

The executable specification is
[claim_reliability_v1.json](../experiments/claim_reliability_v1.json). Commit that
protocol and the runner before starting the test. Local smoke tests use different
seeds and sizes; they are software checks, not results of this protocol.

| Choice | Prespecified value |
|---|---|
| Frozen fits | Training seeds 6100, 7100, 8100, 9100, 10100 |
| Neural models | Shared eight-step, dedicated four-step, one-pass |
| Additional reference | Gaussian posterior under the known synthetic generator; CPU float64 |
| Fresh calibration | 5,000 observations, seed 41000 |
| Untouched final test | 20,000 observations, seed 42000 |
| Primary coverage target | 75% |
| Other coverage targets | 25%, 50%, 90%; descriptive diagnostics |
| Primary comparison | Shared minus four-step wrong-route rate among issued claims |
| Neural batch size | 1,024, including the final partial batch |
| Uncertainty | 2,000 paired input bootstrap samples, seed 43000 |

The sample counts are fixed practical design choices, not the result of a formal
power calculation. There is no outcome-dependent stopping or seed search. All
fits use the same calibration inputs and the same test inputs. The runner rejects
overlap with stored training/calibration seeds, earlier reserved test seeds, and
known prior data seeds 21000, 22000, 31000, 32000, 33000 and 36000. Seed 41000 has
previously been an execution-order RNG seed; this does not reuse a dataset.

Weights, candidate structures, and trained depths remain frozen. Only the
coverage cutoffs are fitted afresh, using calibration probabilities alone. The
existing replication calibration artifacts are read for provenance and split
exclusion and are not overwritten.

All five fits finish calibration first. Every model's cutoff at every declared
coverage is saved with source, reference and calibration-artifact hashes in
`calibration-lock.json`. Only after that barrier does the runner generate or
evaluate the final test set. New thresholds are never chosen from test labels or
test scores. Amending the experiment after seeing results requires a new protocol
and a fresh test seed; preserve the original result, including failures.

## What the numbers mean

For observation \(i\), let \(p_i(k)\) be the recorded probability of candidate
structure \(k\), and \(y_i\) the synthetic generating label. Set
\(s_i=\max_k p_i(k)\) and \(\hat y_i=\arg\max_k p_i(k)\), breaking a route tie
by the first candidate index, as in the existing model output.

For target coverage \(c\) and calibration size \(m\), choose threshold \(t_c\)
as the \(\lceil cm\rceil\)-th largest calibration score. Issue a claim when
\(A_i=1\{s_i\geq t_c\}\). Include every score tied at the threshold. On a test
set of \(n\) observations, report

\[
\widehat C=\frac{1}{n}\sum_i A_i,\qquad
\widehat R=\frac{\sum_i A_i1\{\hat y_i\ne y_i\}}{\sum_i A_i}.
\]

Here \(\widehat C\) is achieved claim coverage and \(\widehat R\) is the fraction
of issued claims choosing the wrong synthetic label. If no claims are issued,
the conditional error is undefined and reported as `null`. The threshold targets
calibration coverage; it guarantees neither test coverage nor test correctness.
Two models with a 75% target can achieve different held-out coverage. Present
coverage and conditional error together, rather than calling the test coverage
exactly matched. The runner does not tune a threshold to force matching test
coverage.

For the five frozen fits, the primary point estimate is
\(\Delta_R=\frac15\sum_{j=1}^5(\widehat R_{\mathrm{shared},j}-
\widehat R_{\mathrm{four},j})\) at target 75%. Negative values favor shared for
this particular error metric. It is a mean of fit-specific ratios, not a ratio
formed by pooling all claimed examples. The corresponding difference in achieved
coverage and the mixture-estimate MSE difference are reported alongside it.

Each bootstrap replicate draws 20,000 test input indices with replacement and
uses the same indices across both primary models and all five fits. It recomputes the ratios and
their mean difference. The interval uses the 2.5th and 97.5th percentiles of 2,000
replicates. These are pointwise intervals conditional on the existing fits and
frozen calibration policies. They exclude uncertainty from new training runs,
new calibration samples, and distribution changes. Five fits do not turn the
20,000 common test inputs into 100,000 independent observations. Undefined
conditional-error replicates suppress that interval instead of being discarded.

Individual fit results and their variation are retained. Secondary metrics are
top-route accuracy, mean squared error of the probability-weighted numerical
estimate, Brier score, negative log likelihood (probabilities clipped below at
\(10^{-15}\) for this diagnostic), and errors divided by all inputs. Secondary
coverage levels and model comparisons are descriptive; no multiplicity-adjusted
significance or automatic winner is claimed. A wide interval or an interval
covering zero does not establish equivalence. A lower wrong-claim rate paired
with lower coverage is a tradeoff, not automatic superiority.

The Gaussian reference knows the generator's prior and noise. It is privileged,
and repeated identical copies across checkpoint workers are consistency checks,
not five separately trained analytic models. This baseline helps distinguish
model errors from ambiguity present even under the known generator.

## Run and inspect

The provided `run-universa-claim-reliability.sh` contains the complete committed
source snapshot and verifies its SHA-256 before extracting. It launches a
detached process by default and prints its PID and log location. Closing SSH
does not stop that process. The default run reuses the DGX's existing environment
and five replication checkpoints without installing packages or pulling Git.

```bash
bash "$HOME/Downloads/run-universa-claim-reliability.sh"
```

The log records preflight, tests, calibration, the lock, final-test progress and
completion. On success it prints a new `claim-reliability-*-reports.zip` in
Downloads. A launch message alone is not a completed experiment. No completed
summary or upload ZIP is produced when a check fails; partial evidence remains
in its fresh run directory. Use `--foreground` to watch in the same terminal, or
`--extract-only` to inspect the source without starting the study.

From a checkout with the necessary environment:

```bash
bash scripts/run_portable_claim_reliability.sh /path/to/replication/results cuda
```

The reports ZIP contains JSON provenance, frozen policies, numerical results,
and NumPy archives of synthetic inputs, truth, labels, probabilities and numerical
estimates. Array files load with `allow_pickle=False`. This permits local replay
of metric arithmetic and calibration cutoffs without model weights. Hashes
identify the reported artifacts; they do not authenticate remote execution.

The experiment evaluates claim quality. It does not rerun every Lingua receipt
check, prove the learned transitions, measure a complete verified-request latency,
or establish performance on real-world data. The previous verified-pipeline
evidence and its distinct guarantees remain documented separately.
