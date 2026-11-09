# TI-TNN native: reproduction interface

**Status: implemented and locally construction-tested.** The native causal LM is in
[`model.py`](model.py), with training in [`train.py`](train.py), evaluation in
[`evaluate.py`](evaluate.py), and the design contract in [`ARCHITECTURE.md`](ARCHITECTURE.md). The last
full native test run passed **208 tests**. The **full repository test suite is still pending** and is not
claimed here. These checks establish construction and execution behavior, not learning, downstream
quality, or efficiency. **No WSP phase has begun.**

All commands below start at the repository root and use the same shell environment:

```bash
cd /home/seanjazm27/projects/universa-recurrent-native
PY=.venv-native/bin/python
export PYTHONPATH=research/ti_tnn
export PYTHONDONTWRITEBYTECODE=1
```

## 1. Isolated CPython 3.12 uv environment

Use **CPython 3.12 and uv only**: no Conda and no historical-environment upgrade. In particular, do not
install native packages into, or upgrade,
`/home/seanjazm27/projects/universa-recurrent-scaling/.venv-scaling`.

[`requirements-native.txt`](../../../requirements-native.txt) pins the direct dependencies.
[`requirements-native-lock.txt`](../../../requirements-native-lock.txt) records the complete 57-package
environment. From a checkout without `.venv-native`, reproduce it with:

```bash
cd /home/seanjazm27/projects/universa-recurrent-native
uv venv --python /usr/bin/python3.12 .venv-native
uv pip install --python .venv-native/bin/python \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --requirement requirements-native-lock.txt
```

The extra index supplies `torch==2.13.0+cu130`; the remaining pins come from PyPI. Confirm the import
and inspect both environments' package metadata without modifying either:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-native/bin/python -c \
  'import sys, torch; print(sys.version); print(torch.__version__); print(torch.cuda.is_available())'

set +e
uv pip check --python .venv-native/bin/python
uv pip check --python /home/seanjazm27/projects/universa-recurrent-scaling/.venv-scaling/bin/python
set -e
```

**Disclosed environment warning:** both `uv pip check` commands currently exit 1 with:

```text
Found 1 incompatibility
The package `nvidia-cusparselt-cu13` was built for a different platform
```

This warning occurs in **both the historical and new environments**. It is not suppressed or presented
as a clean dependency check. Nevertheless, `torch 2.13.0+cu130` imports and the CUDA smoke check passed
in the new environment. Preserve the warning in environment reports; do not "repair" it by upgrading the
historical environment.

## 2. Prepare the pinned real data

The following is the actual finite-data construction command. It downloads and hash-checks the complete
pinned source objects, then writes packed arrays and audits only under `runs/ti_tnn_native/`:

```bash
"$PY" -m native.prepare_data \
  --cache runs/ti_tnn_native/cache \
  --output runs/ti_tnn_native/data \
  --train-tokens 22000000 \
  --dev-tokens 1000000 \
  --structured-train 65536 \
  --structured-dev 4096 \
  --structured-unseen 1024 \
  --structured-longer 1024 \
  --code-generation-count 64 \
  --code-generation-target-chars 256 \
  --seed 20261108
```

The token budgets are **22,000,000 train and 1,000,000 dev tokens before right padding**. Structured
counts are 65,536 train, 4,096 dev, 1,024 unseen-composition, and 1,024 longer-length examples. The two
holdout sets are not packed into train/dev. Sixty-four held-out dev code prompts are written for bounded
code-suffix generation, with targets capped at 256 characters.

### Immutable source pins

The executable pins are in [`data.py`](data.py); preparation refuses a changed byte stream.

- **Wikipedia:** `wikimedia/wikipedia`, subset `20231101.en`, revision
  `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`:
  - `20231101.en/train-00000-of-00041.parquet`, SHA256
    `382e7f6f09e488b24793a7f7cfc659879d5a22da2cf2efec6491665f0c019677`;
  - `20231101.en/train-00001-of-00041.parquet`, SHA256
    `6cd57a5825312ca966313d4a1b614690b5c4fb53cee9b124bc0e999977eb36cf`.
- **CodeParrot:** `codeparrot/codeparrot-clean`, revision
  `35a59fb025bc0a102f7d96eac09d145b896d487b`;
  `file-000000000001.json.gz`, SHA256
  `3726a0239b5cb7d0ef3ea36886c533d0becc7404217763015559edb546d53c94`.
- **GPT2 tokenizer:** `openai-community/gpt2`, `tokenizer.json`, revision
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`, SHA256
  `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`.
  Vocabulary size is 50,257 and `<|endoftext|>` is token 50,256. The prepared copy is
  `runs/ti_tnn_native/cache/tokenizer/tokenizer.json`.

### Split, mixture, and license limits

- Deduplication uses SHA256 of the **exact UTF-8 document content**, keeping the first occurrence in
  source/file/row order. It happens **before** the dev split.
- For ordinary Wikipedia/CodeParrot documents, dev receives a deterministic 1% of the unique documents:
  sort by `sha256("ti-tnn-native:document-dev:v1:<content-sha256>")`, with content hash as tie-breaker,
  and take `floor(n * 0.01 + 0.5)` records. Structured splits are generated separately.
- Wikipedia/CodeParrot/structured tokens use the **45%/45%/10%** (`9:9:2`) largest-remainder allocation.
  Whole document runs stay contiguous. Sources are never cycled to meet a budget; partial documents and
  unconsumed suffixes are explicit in the audit.
- CodeParrot retains only the labels `apache-2.0`, `bsd-2-clause`, `bsd-3-clause`, `isc`, and `mit`.
  **This license-label filter is a provenance filter, not file-level legal clearance.** Retained files
  can still require individual license and notice review.
- Gated data, WikiText evaluation populations, Qwen targets, and BIRD are forbidden inputs. Only
  `input_ids`, `positions`, `doc_ids`, and `key_padding_mask` enter the model; labels and execution-derived
  arrays remain external training targets.

## 3. Construction tests

Run every native test without writing pytest's cache or Python bytecode:

```bash
"$PY" -m pytest -p no:cacheprovider -q tests/test_ti_tnn_native_*.py
```

The last full native run result was:

```text
208 passed
```

A focused model/routing/cache/replay check is:

```bash
"$PY" -m pytest -p no:cacheprovider -q \
  tests/test_ti_tnn_native_model.py \
  tests/test_ti_tnn_native_losses.py \
  tests/test_ti_tnn_native_generation.py \
  tests/test_ti_tnn_native_reconstruct.py
```

The repository-wide command remains **pending, not passed or claimed**:

```bash
"$PY" -m pytest -p no:cacheprovider -q
```

## 4. Generate the required reconstruction replay

`native.reconstruct` deliberately rejects bare `[256]` teacher vectors. This compact procedure selects
packed dev row 0, preserves every teacher array as rank-2 **`[1,256]`**, and creates exactly two
structured prompts: the first dev record and first unseen-composition record. It includes prompts only,
not programs, target text, route labels, or state labels.

```bash
"$PY" - <<'PY'
import json
from pathlib import Path

from native.checkpoint import atomic_write
from native.data import MODEL_INPUT_ARRAYS, PackedDataset, load_gpt2_tokenizer
from native.evaluate import load_structured_records
from native.reconstruct import REPLAY_FORMAT, load_replay_spec
from native.structured import encode_with_offsets

root = Path("runs/ti_tnn_native/data")
dev = PackedDataset(root, "dev")
teacher = {
    name: dev.arrays[name][0:1].tolist()
    for name in (*MODEL_INPUT_ARRAYS, "labels", "source_ids")
}
assert all(
    len(values) == 1 and len(values[0]) == 256
    for values in teacher.values()
), "every teacher array must retain explicit [1,256] shape"

records, _ = load_structured_records(root / "structured")
tokenizer = load_gpt2_tokenizer(
    "runs/ti_tnn_native/cache/tokenizer/tokenizer.json"
)
prompts = []
for split in ("dev", "unseen_composition"):
    record = records[split][0]
    ids, _ = encode_with_offsets(tokenizer, record["prompt"])
    ids = [int(value) for value in ids]
    if not 0 < len(ids) <= 256 - 48:
        raise ValueError(f"{split}: prompt does not fit with 48 generated tokens")
    prompts.append({
        "example_id": record["example_id"],
        "input_ids": ids,
        "positions": list(range(len(ids))),
    })

payload = {
    "format": REPLAY_FORMAT,
    "teacher_forced": teacher,
    "generation_prompts": prompts,
    "max_new_tokens": 48,
    "eos_token_id": 50256,
}
output = root / "replay.json"
atomic_write(output, (json.dumps(payload, sort_keys=True) + "\n").encode())
spec = load_replay_spec(output)
assert all(spec["teacher_forced"][name].shape == (1, 256) for name in teacher)
assert len(spec["generation_prompts"]) == 2
print(output, spec["spec_sha256"])
PY
```

## 5. Real-data CPU dry-run and anchored CUDA profiles

The common arguments freeze training batch 8, 16 training-dev evaluation batches of 8, complete
teacher-forced evaluation capped at **16 batches × 8**, the first 128 structured examples per split,
64 code examples, at most 48 structured and 128 code generated tokens, eager execution, requested BF16
CUDA autocast, and the replay generated above.

```bash
PROFILE_COMMON=(
  --config research/ti_tnn/native/configs/engineering.json
  --data runs/ti_tnn_native/data
  --batch-size 8
  --evaluation-batches 16
  --evaluation-batch-size 8
  --complete-evaluation-max-batches 16
  --complete-evaluation-batch-size 8
  --structured runs/ti_tnn_native/data/structured
  --code-generation runs/ti_tnn_native/data/code_generation.jsonl
  --tokenizer runs/ti_tnn_native/cache/tokenizer/tokenizer.json
  --structured-limit-per-split 128
  --code-generation-limit 64
  --max-new-tokens 48
  --code-max-new-tokens 128
  --eos-token-id 50256
  --reconstruction-replay runs/ti_tnn_native/data/replay.json
  --seed 20261108
  --execution eager
  --dtype bfloat16
)
```

A real-data CPU dry-run checks the same data, checkpoint, generation, intervention, and replay paths
without opening a reservation. CPU execution uses float32 by implementation; requested BF16 applies to
CUDA autocast. Dry-run timings and losses are not GPU performance or learning evidence.

```bash
MODE=dense
"$PY" -m native.profile \
  --mode "$MODE" \
  --output "runs/ti_tnn_native/dry-runs/$MODE.json" \
  --checkpoint "runs/ti_tnn_native/dry-runs/$MODE.checkpoint.pt" \
  --reconstruction-output "runs/ti_tnn_native/dry-runs/$MODE.reconstruction.json" \
  --device cpu \
  --dry-run \
  --warmup-steps 1 \
  --measured-steps 3 \
  "${PROFILE_COMMON[@]}"
```

A real CUDA profile must use the anchored ledger and lock. Each arm has its own output, durable
checkpoint, reconstruction, and reservation. The shown `1.0` projected-hour bound is an interface
example, not a measured projection or additional authorization; actual profiling needs a justified bound
within the native allowance.

```bash
LEDGER=/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_scaling/gpu_ledger.jsonl
LOCK=/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_scaling/gpu.lock

for MODE in dense shared untyped typed; do
  "$PY" -m native.profile \
    --mode "$MODE" \
    --output "runs/ti_tnn_native/profiles/$MODE.json" \
    --checkpoint "runs/ti_tnn_native/profile-artifacts/$MODE.checkpoint.pt" \
    --reconstruction-output "runs/ti_tnn_native/profile-artifacts/$MODE.reconstruction.json" \
    --device cuda \
    --warmup-steps 5 \
    --measured-steps 20 \
    --projected-hours 1.0 \
    --ledger-path "$LEDGER" \
    --ledger-lock "$LOCK" \
    "${PROFILE_COMMON[@]}"
done
```

Only a successful, cache-validated, complete-evaluation-valid `native-profile-v1` CUDA JSON using
`engineering.json` is eligible for study projection. CPU dry-runs cannot project a study.

## 6. Freeze, run, and summarize the four-arm study

The study runner's built-in checkpoint/dev defaults are both 100. Use this explicit shared recipe to
avoid that excessive schedule: **checkpoint every 500 updates and dev every 1,000 updates**.

```bash
mkdir -p runs/ti_tnn_native
cat > runs/ti_tnn_native/train-hyperparameters.json <<'JSON'
{
  "batch_size": 8,
  "learning_rate": 0.0003,
  "weight_decay": 0.01,
  "betas": [0.9, 0.999],
  "eps": 1e-8,
  "grad_clip": 1.0,
  "lambda_balance": 0.01,
  "lambda_route": 0.1,
  "lambda_state": 0.1,
  "warmup_fraction": 0.02,
  "final_lr_fraction": 0.1,
  "checkpoint_every": 500,
  "dev_every": 1000,
  "dev_batches": 16
}
JSON
```

Freeze from a clean committed repository into a new or empty study directory. **Pass the four engineering
JSONs individually.** Directory ingestion expands every `*.json`, including dry-runs, reconstruction
reports, checkpoint manifests, and other non-profile JSONs; passing the profile directory is therefore
invalid and must not be used.

```bash
"$PY" -m native.study freeze \
  --output runs/ti_tnn_native/study \
  --data runs/ti_tnn_native/data \
  --tokenizer runs/ti_tnn_native/cache/tokenizer/tokenizer.json \
  --replay-spec runs/ti_tnn_native/data/replay.json \
  --engineering-profile runs/ti_tnn_native/profiles/dense.json \
  --engineering-profile runs/ti_tnn_native/profiles/shared.json \
  --engineering-profile runs/ti_tnn_native/profiles/untyped.json \
  --engineering-profile runs/ti_tnn_native/profiles/typed.json \
  --engineering-config research/ti_tnn/native/configs/engineering.json \
  --serious-config research/ti_tnn/native/configs/serious.json \
  --repository /home/seanjazm27/projects/universa-recurrent-native \
  --development-seed 20261108 \
  --provisional-loss-tokens 20000000 \
  --reserve-fraction 0.20 \
  --train-hyperparameters runs/ti_tnn_native/train-hyperparameters.json \
  --ledger-path "$LEDGER" \
  --ledger-lock "$LOCK"
```

Feasibility charges measured setup, training, every scheduled checkpoint and dev pass, complete bounded
evaluation, interventions, and reconstruction, then adds at least a **20% reserve**. If 20M shifted loss
tokens per arm do not fit, freeze may select a smaller **common actual endpoint for every arm before any
comparative outcome is read**. If even the minimum complete curve does not fit, it refuses the study as
infeasible. It must not silently truncate one arm, reallocate another arm's budget after freeze, or
ignore checkpoint storage.

The run stops on the first failed or incomplete arm. Summarization refuses comparative output unless all
four arms match the frozen endpoint and every required artifact and reservation validates.

```bash
"$PY" -m native.study run --study runs/ti_tnn_native/study
"$PY" -m native.study summarize --study runs/ti_tnn_native/study
```

These are reproducible command forms, not a claim that a study has run. The freeze also requires the
profile code identity, source commit, prepared-array hashes, tokenizer, and replay hashes to match.

## 7. A/B/C/D comparisons and corrected counts

The comparisons isolate one intended change at a time; they do not collapse parameter and compute matching.

| Arm | Implementation mode | Construction | Comparison |
|---|---|---|---|
| **A** | `dense` | Four unique attention/transform cells, four passes; no router | A/B isolates recurrent weight sharing while holding four updates fixed |
| **B** | `shared` | One attention/transform cell shared across exactly four passes; no router | B/C isolates generic top-1 specialist organization |
| **C** | `untyped` | Four selected-only experts, shared untyped slot projections, h-only router | C/D isolates type-conditioned interfaces and routing |
| **D** | `typed` | C's expert bank with type-specific slot readers/writers and slot-conditioned router | Direct test of the added typing |

Every arm has four slots and state probes. C/D receive identical optional route labels. None uses adaptive
stopping, skip/small/full compute, a sheaf bridge, Qwen weights/caches/targets, or a new optimizer.

### Corrected parameter and MAC counts

These are construction counts from [`configs/engineering.json`](configs/engineering.json) and
[`configs/serious.json`](configs/serious.json), generated by [`cost.py`](cost.py), not performance
results. MACs are for one decoded token attending to 256 keys and include four passes, attention,
routing, slot/h interfaces, and the untied output head. Probe MACs are separate.

Engineering configuration (`d_model=192`, `d_slot=48`, `d_ff=768`):

| Arm | Total parameters | Nominal one-expert active | Route-union active range | Decoded-token MACs | MACs with probes |
|---|---:|---:|---:|---:|---:|
| A / dense | 21,860,352 | 21,860,352 | 21,860,352 | 13,176,000 | 13,200,576 |
| B / shared | 20,084,544 | 20,084,544 | 20,084,544 | 13,176,000 | 13,200,576 |
| C / untyped | 21,415,300 | 20,085,316 | 20,085,316–21,415,300 | 13,179,072 | 13,203,648 |
| D / typed | 21,499,780 | 20,169,796 | 20,169,796–21,499,780 | 13,182,144 | 13,206,720 |

Serious configuration (`d_model=832`, `d_slot=208`, `d_ff=3328`), a future projection only:

| Arm | Total parameters | Nominal one-expert active | Route-union active range | Decoded-token MACs | MACs with probes |
|---|---:|---:|---:|---:|---:|
| A / dense | 130,803,712 | 130,803,712 | 130,803,712 | 102,356,800 | 102,463,296 |
| B / shared | 97,549,504 | 97,549,504 | 97,549,504 | 102,356,800 | 102,463,296 |
| C / untyped | 122,485,380 | 97,552,836 | 97,552,836–122,485,380 | 102,370,112 | 102,476,608 |
| D / typed | 124,049,540 | 99,116,996 | 99,116,996–124,049,540 | 102,383,424 | 102,489,920 |

For A/B the active range is a single value. For C/D, `active` is a backward-compatible alias for the
**nominal one-expert** count, not a prediction of actual routed execution. Actual per-token active
parameters use the union of experts selected over four passes; repeated selections count once.
`cost.route_active_parameters` computes that union and `active_parameter_report` reports observed values.
Shared recurrent weights count once in stored parameters but four times in executed work. The serious
configuration and its 125M-token proposal do not authorize training or establish a result.

## 8. State/route capture, checkpoints, and the WSP boundary

The current implementation supplies construction artifacts needed to design a future, separately
authorized WSP phase:

- `named_parameters()` and `state_dict()` expose explicit names. A uses `attentions.*` and `transforms.*`;
  B uses one `attention` and one `transform`; C/D use one `attention` and `experts.*`. B/C/D therefore
  reuse the same recurrent weights across passes rather than storing four per-pass copies.
- `return_trace=True` captures `h` and four-slot trajectories. Routed outputs expose route indices;
  `return_aux=True` also captures route logits/probabilities/weights, state logits, and load statistics.
  These are execution records, not evidence that slot names imply semantics or that interventions prove
  faithful explanation.
- [`checkpoint.py`](checkpoint.py) stores model, optimizer, scheduler, sampler, RNG, configuration, and
  code/data/order identities. It writes a SHA256 checkpoint manifest and separate model/optimizer/
  scheduler/sampler state hashes. [`reconstruct.py`](reconstruct.py) verifies checkpoint bytes and records
  full/cached output and generation hashes at the restored dtype.

Names and hashes support exact future matching; they do not make different code or checkpoints
interchangeable. Any WSP phase requires its own authorization, protocol, and evidence. **No WSP has
begun, and no WSP result is claimed.**

## Evidence boundary

The verified evidence is limited to environment/import and CUDA smoke checks, the 208 native construction
tests, and deterministic implementation/counting behavior described above. No native learning curve,
quality comparison, efficiency result, full-repository test result, or WSP outcome is claimed here.
