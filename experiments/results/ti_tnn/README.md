# TI-TNN result evidence

The raw results behind the TI-TNN reports, so anyone can rebuild their tables without rerunning
anything:

```bash
python experiments/results/ti_tnn/make_tables.py   # needs PyTorch (summarize.py imports the model module)
```

The script first checks every file against [`manifest.json`](manifest.json) (sha256). Then it prints
the toy benchmarks, the WikiText-103 and SCAN tables, and both composition runs as exact correct/total
counts. It also rebuilds the real-benchmark summary and the adaptive-compute verdict, and checks them
against the bundled `summary.json` and `frontier.json`.

## The policy exception

Run outputs live in the git-ignored `runs/` directory and are not committed. This folder is a narrow,
reviewed exception:
- **What it holds:** weight-free result JSON only, for every run of each completed experiment.
  Failures and negative results are included.
- **Each file is byte-identical** to the one its run wrote, so the hashes quoted in the reports
  identify these copies.
- **Never here:** model weights, datasets, caches or the working run directory.

A hash identifies an artifact. It does not authenticate where or how the run executed; the manifest
records the declared source version and command.

## Contents

| Experiment | Files | Protocol | Report | Source version |
|---|---:|---|---|---|
| `toy_benchmarks/` | 2 | none (exploratory) | [TI-TNN README](../../../research/ti_tnn/README.md#evidence-so-far) | the working tree committed as `c079459` |
| `unsupervised_routing/` | 1 | [brief](../../../research/ti_tnn/protocols/unsupervised_routing.md) | [TI-TNN README](../../../research/ti_tnn/README.md#evidence-so-far) | `20d3a5c` |
| `real_benchmarks/` | 76 (15 WikiText-103, 60 SCAN, summary) | [brief](../../../research/ti_tnn/protocols/real_benchmarks.md) | [report](../../../research/ti_tnn/reports/real_benchmarks.md) | `316c18f` |
| `functorial_composition/` | 1 | [brief](../../../research/ti_tnn/protocols/functorial_composition.md) | [report](../../../research/ti_tnn/reports/functorial_composition.md) | `4b56a03` |
| `functorial_composition_controls/` | 1 | [brief](../../../research/ti_tnn/protocols/functorial_composition_controls.md) | [report](../../../research/ti_tnn/reports/functorial_composition_controls.md) | `c624e42` |
| `adaptive_compute_init_control/` | 10 (6 WikiText-103 runs, latency, step-0 diagnostics, the analysis in its frozen and path-fixed versions) | [brief](../../../research/ti_tnn/protocols/adaptive_compute_init_control.md) | [report](../../../research/ti_tnn/reports/adaptive_compute_init_control.md) | `630cab2` (recorded in each run) |
| `adaptive_compute_wikitext_fixed/` | 5 (3 WikiText-103 runs, latency, the analysis's output) | [brief](../../../research/ti_tnn/protocols/adaptive_compute_wikitext_fixed.md) | [report](../../../research/ti_tnn/reports/adaptive_compute_wikitext_fixed.md) | `16bba5c` (recorded in each run) |
| `adaptive_compute/` | 59 (36 lookup-task runs, 21 new WikiText-103 runs, `latency.json`, `frontier.json`) | [brief](../../../research/ti_tnn/protocols/adaptive_compute.md) | [report](../../../research/ti_tnn/reports/adaptive_compute.md) | `64ecc86` (training); `d9a8c1f` (evaluation) |
| `lookup_init_check/` | 3 (`moe_top1` with dense-matched experts, seeds 0–2) | [brief](../../../research/ti_tnn/protocols/adaptive_compute_hops_init_check.md) | [report](../../../research/ti_tnn/reports/adaptive_compute_hops_init_check.md) | `1ac2b1c` (recorded in each run) |
| `lookup_learning_diagnostics/` | 75 (every development-seed run; exploratory, not pre-registered) | none (exploratory) | [part A v2 report](../../../research/ti_tnn/reports/adaptive_compute_part_a_v2.md#development-diagnostics) | `1ac2b1c` to `fb39904` (recorded in each run) |
| `adaptive_compute_part_a_v2/` | 51 (50 lookup-task runs, seeds 200–204, and the evaluator's output) | [brief](../../../research/ti_tnn/protocols/adaptive_compute_part_a_v2.md) | [report](../../../research/ti_tnn/reports/adaptive_compute_part_a_v2.md) | `2a34728` (recorded in each run); evaluation at `dcf115d` |
| `semantic_interventions/` | 2 (72 CPU runs in one file, `verdict.json`) | [brief](../../../research/ti_tnn/protocols/semantic_interventions_v1.md) | [report](../../../research/ti_tnn/reports/semantic_interventions_v1.md) | `75cb47a` (committed immediately before the run) |
| `typed_routing/` | 2 (60 CPU system runs in one file, `verdict.json`) | [brief](../../../research/ti_tnn/protocols/typed_routing_v1.md) | [report](../../../research/ti_tnn/reports/typed_routing_v1.md) | `fd0aa20` (committed immediately before the run) |
| `typed_recurrence/` | 2 (45 CPU system runs in one file, `verdict.json`) | [brief](../../../research/ti_tnn/protocols/typed_recurrence_v1.md) | [report](../../../research/ti_tnn/reports/typed_recurrence_v1.md) | `7ba0b79` (committed immediately before the run) |
| `typed_composition/` | 2 (75 CPU runs in one file, `verdict.json`) | [brief](../../../research/ti_tnn/protocols/typed_composition_v1.md) | [report](../../../research/ti_tnn/reports/typed_composition_v1.md) | `ef604a1` (committed immediately before the run) |

**Source versions.**
- The run files do not record their source version. The manifest takes it from each protocol's
  execution record or the run's timing.
- The toy rerun preceded its commit by 16 minutes, so the manifest says "the working tree committed
  as", not "ran from".

**Adaptive compute.** The WikiText-103 comparison reuses `dense`, `moe_top1` and `moe_top2` from
`real_benchmarks/lm/`. `frontier.json` holds the evaluator's output, including its run manifest with every
evaluated file's hash; `make_tables.py` checks those hashes against this bundle. `latency.json` holds the
per-window execution checks behind the "untestable" latency verdict. The 12 adaptive checkpoints
(`model.pt`) are weights and stay out.

**Local paths removed.** Eight files in `adaptive_compute_init_control/` had absolute local paths, because
the run script passed absolute roots. The prefix was removed so every path is repository-relative; no
other byte changed. The manifest gives each such file's original sha256, which the analysis's run manifest
refers to, and `make_tables.py` uses it there.

**Initialization.** Every mixture run in `real_benchmarks/` and `adaptive_compute/` used the legacy expert
initialization
(see the [errata](../../../research/ti_tnn/reports/real_benchmarks.md#errata-and-caveats-added-2026-10-16-after-an-outside-review)).

Experiments still running are added here only after they finish and are reported.
