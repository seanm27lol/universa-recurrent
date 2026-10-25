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
| `local_updates/` | 2 (450 learned updates and 30 faulty-suite cases in one file, `verdict.json`) | [brief](../../../research/ti_tnn/protocols/local_updates_v1.md) | [report](../../../research/ti_tnn/reports/local_updates_v1.md) | `dd1d937` (committed immediately before the run) |
| `local_updates_post_repair/` | 2 (the same 450 updates and 30 fault cases, rerun through the repaired interfaces; the verdict must equal the original) | [brief](../../../research/ti_tnn/protocols/local_updates_v1.md) | [report corrections](../../../research/ti_tnn/reports/local_updates_v1.md) | the contract-repair branch |
| `interpreter_robustness/` | 18 (aggregate summaries, `verdict.json`, 15 per-unit decision-record files as gzip-compressed JSON, the verifier's output; about 16 MB) | [brief](../../../research/ti_tnn/protocols/interpreter_robustness_v1.md) | [report](../../../research/ti_tnn/reports/interpreter_robustness_v1.md) | `69b7fd7` (frozen; committed immediately before the run) |
| `typed_composition/` | 2 (75 CPU runs in one file, `verdict.json`) | [brief](../../../research/ti_tnn/protocols/typed_composition_v1.md) | [report](../../../research/ti_tnn/reports/typed_composition_v1.md) | `ef604a1` (committed immediately before the run) |
| `adaptive_compute_shared_path_control/` | 11 (10 lookup-task runs, seeds 200–204, and the evaluator's output) | [brief](../../../research/ti_tnn/protocols/adaptive_compute_shared_path_control.md) | [report](../../../research/ti_tnn/reports/adaptive_compute_shared_path_control.md) | `9da9c82` (recorded in each run) |
| `integration_test/` | 108: the run summary; the 15 units' export (unit summaries, interpreter draws, atomic tables, certificates, contexts, exhaustive digests, episode rows); `verification.json`; `session_release.json` | [protocol](../../../research/ti_tnn/protocols/integration_test_v1.md) | [report](../../../research/ti_tnn/reports/integration_test_v1.md) | `e03075a` (frozen; recorded in the run) |
| `repeated_updates/` | 32: the run summary; the 15 units' `unit.json` and `proposals.jsonl.gz` (every attempt's weight-free record); `verification.json`, with the sampled reproduction | [protocol](../../../research/ti_tnn/protocols/repeated_updates_v1.md) | [report](../../../research/ti_tnn/reports/repeated_updates_v1.md) | `e8c6f99` (frozen; recorded in the run, with a clean tree) |
| `factored_interpreter/` | 55: the no-fit controls run, whole; the pilot's run summary, per-unit summaries and A, F and F+B records; `export_integrity.json` (all 58 original files); the verifiers' outputs; `ak_diagnosis.json`; `pilot_trace_release.json` (the eight A+K step-trace files, bound by sha256) | [protocol](../../../research/ti_tnn/protocols/factored_interpreter_v1.md) (draft, never frozen) | [report](../../../research/ti_tnn/reports/factored_interpreter_v1.md) (development) | `0e1a3c9` (both runs; recorded, clean tree) |
| `s2_diagnosis/` (historical, development) | 36 (the classification of the robustness records; 24 development fits' summaries and decision records; arm A's six trace files; the development analysis; `local_traces.json`; `trace_release.json`) | [brief](../../../research/ti_tnn/protocols/s2_diagnosis_brief.md) | [report](../../../research/ti_tnn/reports/s2_diagnosis_v1.md) | `6435719` (analysis); `ece0f86` (arm A); `d991d12` (B, C, D); recorded in each run |
| `s2_diagnosis/confirmation*` | 47 (45 fits' decision records, the run summary and the analysis with the frozen verdict) | [protocol](../../../research/ti_tnn/protocols/s2_context_confirmation_v1.md) | [report](../../../research/ti_tnn/reports/s2_diagnosis_v1.md) | `07bb97e` (frozen; recorded in the run) |

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

**Interpreter robustness.**
- **Records.** Its decision records are gzip-compressed JSON (`.json.gz`), still weight-free result data.
  The manifest hashes the compressed bytes.
- **Rebuilding.** `research/ti_tnn/bench/interpreter_robustness_verify.py` regenerates each study from its
  world seed and re-derives every reported number from these records. It needs the frozen generator,
  unchanged since `69b7fd7`.
- **What it covers.** Training time, training loss and execution provenance are not derivable from
  decision records and are not covered.

**S2 diagnosis.**
- **Traces.** Only arm A's development traces are bundled (six files, about 8 MB).
- **The other 63 trace files** (85.2 MiB: B–D in development and every confirmation fit) are published as assets of
  the release `ti-tnn-s2-diagnosis-traces-v1`.
  - `s2_diagnosis/local_traces.json`, committed with the original results, gives each file's sha256.
  - `s2_diagnosis/trace_release.json` maps each asset to its file.
- **Rebuilding.** `make_tables.py` re-derives every class count, certificate, unit outcome, decision rule and
  verdict from the bundled decision records, plus arm A's trace tables. Every arm's trace tables also need the
  release:

  ```bash
  gh release download ti-tnn-s2-diagnosis-traces-v1 --repo seanm27lol/universa-recurrent --dir s2_traces
  python experiments/results/ti_tnn/make_tables.py --s2-traces s2_traces
  ```

  The second command refuses any file whose sha256 differs from the committed one, and checks every trace
  against its decision records.
- **Checkpoints** were written locally and are weights, so they are excluded.

**Integration test.**
- **Sessions as release assets.** The 15 full session-record files (70.9 MB) are assets of the release
  `ti-tnn-integration-test-sessions-v1`, and `integration_test/session_release.json` gives their sha256.
- **Rebuilding.** Download them, then run `make_tables.py --integration-sessions <dir>`. That checks each
  file, stages it next to the bundled export, and re-runs the frozen independent verifier, which must
  reproduce `verification.json`.
- **Without the assets,** `make_tables.py` reports the bundled verification.

**Repeated updates.** The whole export is bundled.
- **Re-deriving.** `make_tables.py --repeated-updates` re-runs the frozen independent verifier on it, in
  seconds, and requires `verification.json` exactly.
- **Retraining.** `--repeated-reproduce` also retrains the 114 sampled chains, in about 20 minutes, and requires
  identical records.

**Factored interpreter.** A development engineering result, closed without a confirmation.
- **What re-verifies from the bundle alone:** the no-fit controls, through `make_tables.py`.
- **The pilot's A+K step traces** are the assets of the release `ti-tnn-factored-pilot-traces-v1` (with
  `SHA256SUMS`), bound by sha256 in `factored_interpreter/pilot_trace_release.json`. With them,
  `make_tables.py --factored-traces <dir>` checks every file's hash before anything that needs torch is imported,
  re-verifies the pilot and re-runs the A+K diagnosis. Both must agree with the bundled outputs.
- **What "agree" means:** exactly, except that recomputed binomial bounds and p-values may differ by the declared
  rounding tolerance (`research/ti_tnn/bench/statistical_tolerance.py`). They must be finite probabilities in
  [0, 1] on both sides. The S2 rebuild uses the same rule.

**Initialization.** Every mixture run in `real_benchmarks/` and `adaptive_compute/` used the legacy expert
initialization
(see the [errata](../../../research/ti_tnn/reports/real_benchmarks.md#errata-and-caveats-added-2026-10-16-after-an-outside-review)).

Experiments still running are added here only after they finish and are reported.
