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

**Status: Milestone 1 implementation, tested locally with tiny random Qwen
fixtures. Released Qwen/AV/AR inference: NOT RUN.** The real checkpoint metadata
and tokenizers have been checked. A basic BF16 GPU kernel passed; full model
compatibility remains untested. [Actual checks and commands](reports/milestone_one.md).

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
| Maximum forward calls under decoding ceilings | 6,400 AV, 32 AR, at most 3,584 target calls |
| Time model | At most 10,016 forward calls; one second per call would be about 2.8 hours, **not a measured throughput estimate**, plus download/load/check/report time |

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

## What the code checks

| Small implementation | What to look for |
|---|---|
| [tasks.py](src/open_weight_lingua/tasks.py) | Explicit interpreter, paired-answer integrity, deterministic groups, canonical-program and tokenized-prompt exclusions |
| [target.py](src/open_weight_lingua/target.py) | Block output versus `hidden_states[layer+1]`, one site, native restoration, exception-safe hooks, suffix-causality checks |
| [nla_adapter.py](src/open_weight_lingua/nla_adapter.py) | Exact loaded metadata/templates, marker context, cache-free AV embedding injection, required AR value head, no final norm |
| [geometry.py](src/open_weight_lingua/geometry.py) | Direction and retained norm stay separate; nonfinite/near-zero rejection |
| [metrics.py](src/open_weight_lingua/metrics.py) | Strict integer output, multi-token answer plus EOS scoring, float32 full-vocabulary KL |
| [runner.py](src/open_weight_lingua/runner.py) | P0/P1/P2/P3/P5, raw donor, fixed smoke-median-norm diagnostic, all variants/failures, sequential loading |
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
| `manifest.json` | Written before model inference: all inputs/IDs, code/protocol hashes, exact revisions, metadata, software and frozen engineering settings |
| `results.json` | Every planned prompt variant, identity checks, descriptions, scores, controls, timings, bytes and failures |
| `summary.json`, `report.md` | All-group/all-variant denominators and valid-case distribution metrics; no selection of attractive examples |
| `compatibility.json` | Packages, backend, memory/storage, downloads, calls and stage times |
| `completion.json` | `COMPLETE`, `COMPLETE_WITH_FAILURES` or `FAILED`; pilot/validation remain `NOT RUN` |
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

P4/PCA calibration, the frozen text editor, edit coverage, a scientific pilot and
locked validation belong to later milestones and cannot be launched by this
runner. The current norm diagnostic is fitted on the smoke inputs and explicitly
does **not** substitute for the separate 256-group scientific calibration split.
There is no SGLang/Transformers backend-equivalence claim. The primary real-model
identity and AV/AR checks remain **NOT RUN** until the downloaded weights pass
the smoke. [Compatibility/source audit](reports/source_compatibility.md),
[original supplied brief](protocols/phase_two_brief.md),
[attribution](THIRD_PARTY.md), [claim ledger](../../docs/claims.md).
