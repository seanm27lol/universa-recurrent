# TI-TNN result evidence

The raw results behind the TI-TNN reports, so anyone can rebuild their tables without rerunning
anything:

```bash
python experiments/results/ti_tnn/make_tables.py   # needs PyTorch (summarize.py imports the model module)
```

The script first checks every file against [`manifest.json`](manifest.json) (sha256). Then it prints
the toy benchmarks, the WikiText-103 and SCAN tables, and both composition runs as exact correct/total
counts. It also rebuilds the real-benchmark summary and checks it against the bundled `summary.json`.

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

**Source versions.**
- The run files do not record their source version. The manifest takes it from each protocol's
  execution record or the run's timing.
- The toy rerun preceded its commit by 16 minutes, so the manifest says "the working tree committed
  as", not "ran from".

**Initialization.** Every mixture run in `real_benchmarks/` used the legacy expert initialization
(see the [errata](../../../research/ti_tnn/reports/real_benchmarks.md#errata-and-caveats-added-2026-10-16-after-an-outside-review)).

Experiments still running are added here only after they finish and are reported.
