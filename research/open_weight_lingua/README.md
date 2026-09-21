# Can a language description preserve one activation?

```text
x = 3
y = 8
x = x + 2
What is x? Reply with only the integer.
```

The answer is 5. This engineering gate captures one Qwen vector before answering,
describes it using a released activation verbalizer (AV), reconstructs a direction
using its paired reconstructor (AR), and measures the effect of replacing that
vector. A paired program changes the first assignment to `x = 4`; the answer for
`x` changes while the answer for `y` stays fixed.

**Status: Milestone 2 implementation, tested locally with tiny random
fixtures, and the repaired runner has COMPLETED the eight-group real-model
engineering smoke (run `smoke-20260921T211151Z-a4e4a038`, independent auditor
PASS). Released-model calibration/pilot/validation: NOT RUN.** The real
checkpoint metadata and tokenizers have been checked, and all 41.41 GB of
pinned artifacts passed hash verification. The smoke is an engineering gate,
not a scientific result, and its eight groups do not authorize a pilot
decision. [Milestone 1 checks and commands](reports/milestone_one.md);
[Milestone 2 checks and commands](reports/milestone_two.md).

Milestone 2 adds two stages on the same machinery. A target-only
**calibration** stage (256 groups) fits the P4 PCA baseline on pooled unit
directions without task labels and freezes the calibration median norm. A
**pilot** stage (128 groups) runs every condition and control, applies the
frozen text-edit rule, computes whole-group bootstrap decision statistics, and
reports the projected validation cost against the eight-hour budget. The human
decision record is the [pilot decision document](protocols/pilot_decision.md);
locked validation remains a separate milestone this runner cannot launch.

For native activation `h`, retain `n = ||float32(h)||₂` separately. If the AV emits
description `c` and the AR returns nonzero finite vector `r`, install
`h′ = cast_native(n r / ||r||₂)`, normalizing in float32. Here `n` is a four-byte
side channel, **not** something recovered from text. AV injection has its own
metadata-defined norm, 150. We measure cosine and squared distance between unit
directions; neither is raw activation MSE. All other tokens and the original
prompt remain available.

## Run the eight-group engineering smoke

From the repository root, on the DGX:

```bash
bash research/open_weight_lingua/scripts/run_smoke.sh --fetch-models
```

This creates/uses `.venv-phase2`, synchronizes `uv.lock`, runs the CPU suite,
downloads only the immutable artifacts in `configs/model-lock.json`, verifies
their hashes, and runs **eight groups / 32 prompt variants**. It loads target,
AV, AR and target again in separate sequential stages. `--fetch-models` is the
explicit download step; omitting it requires already present weights. Every
invocation uses a fresh directory under `research/open_weight_lingua/runs/`.
Existing run directories are refused, with no automatic resumption.

Prerequisites: `uv`, Python 3.11 or 3.12, Linux ARM64 or x86-64 with a working
CUDA 13 driver, and network access to the public model repositories for the
initial fetch. No remote code or inference server is used. The existing Phase
One environment, root package and drivers are not upgraded.

Resource planning, **not measured model performance**:

| Item | Estimate or measured inventory |
|---|---|
| Pinned model artifacts, including tokenizer/config files | 41,410,059,546 bytes (41.41 GB / 38.57 GiB) |
| Suggested free storage before first setup | 60 GiB, including environment/cache headroom |
| Largest active BF16 weight set | About 15.23 GB; target and AV each have their own weights |
| Suggested free unified memory | 48 GiB for sequential loading, transient copies and buffers; actual peak unmeasured |
| Maximum forward calls under decoding ceilings | 6,400 AV, 32 AR, at most 4,032 target calls |
| Time model | At most 10,464 forward calls; one second per call would be about 2.9 hours, **not a measured throughput estimate**, plus download/load/check/report time |

Unified CPU/GPU memory is one pool. Model calls have different sequence lengths,
so the time model only illustrates scale. The runner records actual calls,
stage times, allocation peaks, process RSS and end-to-end wall time. Do not use
these illustrative numbers to authorize a pilot. A scientific stage would need
its own measured projection including its additional controls, and must respect
the brief's eight-hour budget.

To use pre-existing, locally verified model directories or a reviewed new lock:

```bash
bash research/open_weight_lingua/scripts/run_smoke.sh \
  --lock research/open_weight_lingua/configs/model-lock.json \
  --target-path /path/to/target --av-path /path/to/av --ar-path /path/to/ar
```

The same hashes are enforced for path overrides. The runner never resolves
`main`. `scripts/resolve_sources.py` is an explicit maintainer operation for a
**new** lock; it refuses to overwrite the current one.

## Run the calibration and pilot stages

Planned sample counts, not measurements: calibration uses 256 groups / 1,024
prompt variants; the pilot uses 128 groups / 512 prompt variants.

```bash
bash research/open_weight_lingua/scripts/run_calibration.sh --fetch-models

bash research/open_weight_lingua/scripts/run_pilot.sh --fetch-models \
  --calibration-fit research/open_weight_lingua/runs/<calibration-run>/baseline_fit.safetensors
```

Both scripts follow the `run_smoke.sh` pattern (`uv sync`, the CPU suite, then
the runner) and accept the same lock and model-path flags. The equivalent
direct commands are:

```bash
.venv-phase2/bin/python -m open_weight_lingua.runner --stage calibration [model flags]
.venv-phase2/bin/python -m open_weight_lingua.runner --stage pilot --calibration-fit PATH [model flags]
```

`--stage` defaults to `smoke`, and smoke behavior is unchanged. The pilot
refuses to start without `--calibration-fit PATH` pointing at a completed
calibration run's `baseline_fit.safetensors`; the fit's sha256 content identity
is recorded in the pilot manifest as `baseline_fit_identity`.

The calibration stage loads only the target. It extracts the site activation
for every calibration variant, fits the P4 baseline, and freezes the median
norm. The pilot runs P0–P5, the raw-donor control and the
calibration-median-norm diagnostic, plus `edited` and `wrong_variable_edit`
reconstructions on groups eligible under the frozen text rule. The edit rule
matches only three statement forms — `x is currently 5`, `x is now 5`, `the
current value of x is 5` — with canonical values 0–19; initial assignments
never match; each variable is classified `eligible`, `absent` or `ambiguous`;
and edits replace only the value span. Coverage and exclusion reasons are
reported over all 128 groups, and the rule version and its sha256 go into the
manifest. Failures follow intention-to-test accounting; statistics are a
whole-group bootstrap with 3,000 resamples and a fixed seed. The pilot writes
machine-readable decision fields into `completion.json` and a decision section
into `report.md`, and reports the projected 512-group validation cost against
the eight-hour budget. Validation is never auto-started.

### Pilot decision rules (design choices)

These thresholds are quoted from brief §8 as **proposals to lock before
validation**. They are design choices, not results and not universal
definitions of interpretability; the pilot decision document records the
outcome against them.

| Decision | Frozen proposal |
|---|---|
| Task usable in pilot | P0 unmodified exact-answer accuracy at least 80%, and donor/perturbation controls show this site can affect the intended measurement; otherwise stop with an assay limitation |
| Limited behavioral preservation | One-sided 95% upper estimate of P2 accuracy loss versus P0 at most 5 percentage points, and P2 beats P3 on correct-answer log probability with a positive lower confidence estimate |
| Edit feasibility | At least 32 of the 128 pilot groups eligible under the frozen edit rule; otherwise close with preservation-only results and mark the edit hypothesis untested |
| Uncertainty | Whole-group bootstrap, 3,000 resamples, fixed seed; prompt variants within one group are not independent samples |

Thresholds are never weakened on held-out results. A negative pilot does not
trigger a search for a different checkpoint, site or task (brief §11).

## What the code checks

| Small implementation | What to look for |
|---|---|
| [tasks.py](src/open_weight_lingua/tasks.py) | Explicit interpreter, paired-answer integrity, deterministic groups, canonical-program and tokenized-prompt exclusions |
| [target.py](src/open_weight_lingua/target.py) | Block output versus `hidden_states[layer+1]`, one site, native restoration, exception-safe hooks, suffix causality via a same-length dummy-suffix bitwise gate plus a frozen pre-pilot cross-length drift bound; every target forward right-padded to the fixed 128-token bucket (`TARGET_BUCKET`) for kernel-shape pinning, with greedy/scoring writing into masked pad slots |
| [nla_adapter.py](src/open_weight_lingua/nla_adapter.py) | Exact loaded metadata/templates, marker context, cache-free AV embedding injection, required AR value head, no final norm |
| [geometry.py](src/open_weight_lingua/geometry.py) | Direction and retained norm stay separate; nonfinite/near-zero rejection |
| [metrics.py](src/open_weight_lingua/metrics.py) | Strict integer output, multi-token answer plus EOS scoring, float32 full-vocabulary KL |
| [splits.py](src/open_weight_lingua/splits.py) | Five deterministic disjoint splits (smoke → calibration → pilot → two validation blocks), carried exclusion inventories, plan hash over every group identity |
| [controls.py](src/open_weight_lingua/controls.py) | P4 PCA fit on pooled unlabeled calibration unit directions; byte-budget rank `min(fitted_rank, floor(text_bytes/2))`, float16 coefficients, shared mean/basis bytes separate, no norm restoration |
| [text_edits.py](src/open_weight_lingua/text_edits.py) | Frozen three-form current-value parser, canonical values 0–19, eligible/absent/ambiguous statuses, value-span-only edits; truth agreement recorded separately, never gates an edit |
| [stats.py](src/open_weight_lingua/stats.py) | Whole-group bootstrap (3,000 resamples, fixed seed) and one-sided upper/lower estimates for the decision rules; absolute log-probability contrasts only, no fraction-recovered ratios |
| [runner.py](src/open_weight_lingua/runner.py) | Stages smoke/calibration/pilot; P0–P5, raw donor, per-stage median-norm diagnostic, edit conditions on eligible groups, all variants/failures, sequential loading |
| [audit.py](src/open_weight_lingua/audit.py) | Recompute saved counts and KL without loading models or invoking the producer |

The selected site is the last non-padding token **including the assistant
generation prefix**, at block index 20. Pair alignment uses the last such token
in each context, with each absolute position retained separately; unequal prefix
lengths are allowed and the boundary token IDs must match. There is no filtering
on model answers. During teacher forcing and greedy generation, the patch stays
at that original position on every cache-free forward.

The fixture suite also checks native BF16 equality, unmodified token vectors,
back-to-back samples, cleanup after exceptions, metadata/head failures, safe
artifact loading and overwrite refusal. A three-call A/B/A regression verifies
that distinct injected embeddings with identical prompt IDs produce distinct
fresh model computations and restoring A reproduces its earlier logits.

```bash
.venv-phase2/bin/python -m pytest -q -c research/open_weight_lingua/pyproject.toml research/open_weight_lingua/tests
```

These tests use small random models. They are software checks, not measurements
on the released Qwen or NLA weights, and are not preregistered scientific results.

## Outputs and failure handling

| File in a fresh run | Contents |
|---|---|
| `manifest.json` | Written before model inference: all inputs/IDs, code/protocol hashes, exact revisions, metadata, software and frozen engineering settings; immutable once written, so calibration runs keep `baseline_fit_identity` null and the fit identity is written to `completion.json` and `baseline_fit.json` after inference, while pilot runs record it in the manifest together with the frozen edit-rule version and sha256; `target_bucket`/`target_padding_policy` record the 128-token kernel-shape pinning and `greedy_identity_gate` the greedy backstop policy |
| `results.json` | Every planned prompt variant, identity checks, descriptions, scores, controls, timings, bytes and failures |
| `summary.json`, `report.md` | All-group/all-variant denominators and valid-case distribution metrics; no selection of attractive examples; pilot `report.md` adds a decision section evaluating the frozen thresholds |
| `baseline_fit.safetensors`, `baseline_fit.json` | Calibration stage only: P4 mean/basis plus sidecar (format version, dtype, fitted rank, width) with a sha256 content identity; loaded by the pilot via `--calibration-fit` |
| `compatibility.json` | Packages, backend, memory/storage, downloads, calls and stage times |
| `completion.json` | `COMPLETE`, `COMPLETE_WITH_FAILURES` or `FAILED`; pilot runs add machine-readable decision fields; stages not executed remain `NOT RUN` and validation is never auto-started |
| `raw/*.safetensors` | Native originals, explicit float32 norm, int64 site/inputs/masks, AR directions, actual replacements and full-vocabulary next-token logits; kept local |
| `inventory.json`, `reports.zip` | Explicit packaged-versus-local evidence inventory; ZIP excludes vectors, weights and full-vocabulary logits |
| `wall_clock.json` | Total runner time through ZIP creation and separately measured setup/test time; local, outside ZIP to avoid a hash cycle |

Missing weights yield an actionable `--fetch-models` diagnostic before model
inference. A failed identity gate stops interpretation. Incomplete descriptions
and reconstruction errors count as failed/missing cases, never as zero KL.
Successful group status means the engineering conditions executed; it does not
mean Qwen answered correctly or language reconstruction passed a scientific test.
Payload byte fields count typed values; file headers, JSON, duplicate audit
evidence and shared metadata are additional storage reported by the inventory.

Replay an intact completed local bundle with:

```bash
.venv-phase2/bin/python -m open_weight_lingua.audit /path/to/run
```

The reports-only ZIP cannot independently replay KL because its distributions
stay local. The auditor checks hashes, result counts and saved KL; it does not
authenticate execution or independently regenerate teacher-forced probabilities.

## Limits and established work

The [NLA authors](https://transformer-circuits.pub/2026/nla/) introduced the
pretrained verbalizer/reconstructor and studied descriptions and interventions.
This is an implementation and controlled-transfer assay using that work, not a
new autoencoder or an interpretation theorem. Activation patching is established;
[Zhang and Nanda](https://arxiv.org/abs/2309.16042v2) show that metrics and corruption
choices matter. The interpreter establishes program behavior, not hidden-state
semantics. Qwen is a conventional transformer, not a recurrent-depth experiment.

P4/PCA calibration, the frozen text editor, edit-coverage accounting and the
128-group pilot are now implemented in this runner; the locked 512-group
validation belongs to Milestone 3 and cannot be launched here. The smoke stage
keeps its smoke-local median-norm diagnostic; the pilot uses the frozen
256-group calibration median instead. The P4 baseline is
**no-more-than-budget** under a declared byte rule, not an exactly matched or
optimal compression baseline, and by itself it establishes nothing about
language versus generic reconstruction. If frozen-rule edit coverage or
intervention sensitivity is inadequate, the brief requires closing with
preservation-only results and marking the edit hypothesis untested or
unsupported. There is no SGLang/Transformers backend-equivalence claim. The
primary real-model identity and AV/AR checks have now passed once inside the
eight-group engineering smoke, and every released-model calibration/pilot
outcome remains **NOT RUN**.
[Compatibility/source audit](reports/source_compatibility.md),
[original supplied brief](protocols/phase_two_brief.md),
[pilot decision document](protocols/pilot_decision.md),
[attribution](THIRD_PARTY.md), [claim ledger](../../docs/claims.md).
