# Milestone 1 local handoff — 2026-09-21

Implemented the source lock, isolated package/environment, paired program
generator, single-site Qwen capture/restore, fixed-template AV embedding injection,
AR reconstruction through its required value head, separate norm restoration,
safe local evidence, and eight-group smoke runner. This is an engineering
handoff, **not** a scientific finding about the released models.

## Actual checks

| Command/check | Actual result | Scope |
|---|---|---|
| `.venv-phase2/bin/python -m pytest -q -c research/open_weight_lingua/pyproject.toml research/open_weight_lingua/tests` | **48 passed in 18.21 s** on the final implementation | Final CPU suite, including explicit four-byte norm and int64 site evidence |
| `bash research/open_weight_lingua/scripts/run_smoke.sh --device cpu` | **48 tests passed in 18.90 s**, then expected exit 1: missing pinned AR weight shard | Verified the one-command environment/test entrypoint and no-weights diagnostic; real-model inference **NOT RUN** |
| `.venv/bin/python -m pytest -q` | **952 passed, 1 skipped in 130.63 s** | Existing Phase One regression suite; optional commit-pinned Universa adapter is not installed |
| `.venv/bin/python scripts/check_release.py` | **PASS**, version 0.5.2 | Root package/version/editable-install/CLI consistency |
| Metadata-only preflight below | **PASS** | Real pinned tokenizer/config/sidecar files, AV marker context, AR suffix and metadata consistency |
| Preflight BF16 GPU matrix multiplication | **PASS** | Small kernel on NVIDIA GB10; not a full-model smoke; capability warning retained |
| `uvx --from ruff==0.14.14 ruff check research/open_weight_lingua/src research/open_weight_lingua/tests research/open_weight_lingua/scripts` | **PASS** | Static Python checks |
| `bash -n research/open_weight_lingua/scripts/run_smoke.sh` | **PASS** | Shell syntax |
| `git diff --check` | **PASS** | Whitespace in tracked changes |

Exact metadata preflight command executed:

```bash
.venv-phase2/bin/python -m open_weight_lingua.preflight \
  --lock research/open_weight_lingua/configs/model-lock.json \
  --cache research/open_weight_lingua/model-cache --metadata-only --device cuda
```

The CPU tests use randomly initialized tiny Qwen2 models. The eight-group
orchestration test additionally substitutes a fixed fixture description; its
outputs are never labeled released-Qwen or released-NLA measurements. It verifies
all 32 variants, local evidence, reports-only packaging, independent saved-count
and KL replay, and corruption detection. Separate AV tests exercise actual tiny
Qwen forwards with two distinct injected vectors and a return to the first.

The expected no-weights run is retained locally in
`runs/smoke-20260921T073057Z-58761439/`. Its `completion.json` says `FAILED`, with
real-model checks, pilot and validation `NOT RUN`. It produced the actionable
missing-file diagnostic and reports ZIP. No error is relabeled as successful
model execution.

## Concrete remaining blockers and limits

* No target/AV/AR weights were fetched or loaded. The supplied brief makes the
  model fetch an explicit user step. The command below is ready to perform it.
* PyTorch's CUDA 13 wheel warns about GB10 capability 12.1 versus its listed
  maximum 12.0. A basic BF16 kernel passed, but the full model operations remain
  untested. A failure must be diagnosed before any pilot; do not relax identity
  tolerances to hide it.
* There is no measured full-model throughput, allocation peak or behavioral
  preservation result. Use the [resource planning figures](../README.md) as
  estimates only: 41.41 GB pinned artifact download, suggested 60 GiB disk and
  48 GiB free unified-memory headroom for sequential stages.
* Scientific calibration/PCA, deterministic text editing/coverage, pilot and
  locked validation are later milestones. The smoke median norm is explicitly
  local to smoke data, not a calibration-fitted scientific norm.

## Files and repository state

Branch: `phase-two-milestone-one`, based on
`ecc8d79873b6f76b4e896986092c7ce8909cdf75`. This report records the local checks
performed before the handoff commit. Use the delivery branch rather than `main`
to obtain the implementation. No merge, visibility change, or weight publication
is part of this handoff.

| Changed area | Files |
|---|---|
| Root documentation/ignore | `.gitignore`, `README.md`, `docs/claims.md` |
| New isolated environment | `research/open_weight_lingua/pyproject.toml`, `uv.lock`, `.gitignore` |
| Source/compatibility lock | `configs/model-lock.json`, `scripts/resolve_sources.py`, `reports/source_compatibility.md`, `THIRD_PARTY.md`, `third_party/APACHE-2.0.txt` |
| Implementation | `src/open_weight_lingua/{tasks,target,nla_adapter,geometry,metrics,preflight,artifacts,runner,audit}.py` and `__init__.py` |
| Tests | `tests/conftest.py`, four test modules, five pinned upstream config/metadata fixtures |
| Handoff/protocol | `README.md`, this report, original supplied `protocols/phase_two_brief.md`, `scripts/run_smoke.sh` |

Phase One runtime source, dependency declarations, protocols, sealed results,
weights, and completed conclusions are unchanged. Generated environments,
tokenizer/model caches, run directories and numeric data are ignored.

## Next single DGX command

From this checkout's repository root:

```bash
bash research/open_weight_lingua/scripts/run_smoke.sh --fetch-models
```

Expected fresh-run outputs: `manifest.json`, `results.json`, `summary.json`,
`report.md`, `compatibility.json`, `completion.json`, `inventory.json`,
`reports.zip`, local `wall_clock.json`, and local `raw/*.safetensors`.
Preflight failures produce diagnostics instead of a fabricated model result.
See the [source lock](../configs/model-lock.json) and
[full run instructions](../README.md) for path overrides, assumptions and evidence
boundaries. This command cannot start the pilot or locked validation.
