"""Standalone vLLM worker for the opt-in AV/AR backend (pinned vllm==0.30.0).

Runs ONLY under the separate worker interpreter (research/open_weight_lingua/
.venv-vllm, built by scripts/setup_vllm_worker.sh). It deliberately imports the
standard library, safetensors/torch, and vllm's own stack — never the
open_weight_lingua package, because vllm 0.30.0 pins transformers 5.x against
the pipeline's transformers 4.57.6. The NLA recipe replicated here lives in
nla_adapter.py; the parent process owns tokenization conventions (chat
template, prompt IDs) and model lock/hash verification.

Measured stack facts behind the settings below (2026-09-22, this machine):
VLLM_USE_FLASHINFER_SAMPLER=0 is required (system nvcc is CUDA 12.0 and
FlashInfer JIT gates sm_12x on >=12.9); VLLM_BATCH_INVARIANT=1 measured
bitwise-repeatable greedy output at batch 1; enable_prompt_embeds accepts
full-length prompt_embeds with a per-position token/embed mask; pooling task
"token_embed" exposes per-token final hidden states when use_activation=False.

Fork hazard: vLLM forks the engine core from this process at LLM() time, and
torch CPU tensor math before that fork spawns the OMP thread pool — the child
then deadlocks on inherited locks (measured: a pre-boot torch.isfinite over the
25 MB value head hung engine init in UvaBuffer setup). No tensor math before
the engine boots; shape/type checks only.

Recompute-per-pass: vllm has no use_cache=False knob, so the AV decode loop
issues a FRESH request (prefix cache disabled) for every generated token,
matching the eager adapter's fresh full forward per pass.

No cross-backend equivalence is claimed by this file. Equivalence is measured
by scripts/check_vllm_equivalence.sh against a completed eager run.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

JOB_SCHEMA_VERSION = 1
VLLM_PIN = "0.30.0"
# The released AR checkpoint deliberately omits these tensors; the eager loader
# (preflight.load_model) permits exactly this missing set, and the recipe then
# replaces lm_head and the final norm with Identity.
ALLOWED_MISSING_AR_WEIGHTS = frozenset({"model.norm.weight", "lm_head.weight"})
DESCRIPTION_PATTERN = r"<explanation>\s*(.*?)\s*</explanation>"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_direction(vector):
    """Fail-closed replica of open_weight_lingua.geometry.unit_direction."""
    require(vector.ndim == 1 and vector.is_floating_point(), "expected one vector")
    value = vector.float()
    norm = torch.linalg.vector_norm(value)
    require(
        bool(torch.isfinite(value).all()) and bool(torch.isfinite(norm)),
        "nonfinite activation vector",
    )
    require(float(norm) > 1e-12, "near-zero vector norm")
    return value / norm


def scaled_injection(vector, width, scale):
    """Eager AV math: fp32 unit direction x injection_scale, one cast to BF16."""
    require(vector.shape == (width,), "activation vector width mismatch")
    require(
        isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0,
        "invalid injection scale",
    )
    return (unit_direction(vector) * scale).to(torch.bfloat16)


def embeds_prompt(prompt_ids, position, embed_row, width):
    """Full-length EmbedsPrompt payload; the mask selects the injected row.

    Token positions ignore their (zeroed) embed rows and are looked up in the
    model's embedding table, matching the measured bit-exact injection route.
    """
    ids = list(prompt_ids)
    require(
        ids and all(type(i) is int and i >= 0 for i in ids),
        "prompt ids must be nonnegative integers",
    )
    require(0 <= position < len(ids), "injection position out of range")
    require(embed_row.shape == (width,), "injected row width mismatch")
    embeds = torch.zeros(len(ids), width, dtype=torch.bfloat16)
    embeds[position] = embed_row.to(torch.bfloat16)
    mask = [True] * len(ids)
    mask[position] = False
    return {
        "prompt_token_ids": ids,
        "prompt_is_token_ids": mask,
        "prompt_embeds": embeds,
    }


def parse_description_record(
    raw_text, token_ids, *, max_new_tokens, eos_token_id, injection_position
):
    """The eager adapter's exact description extraction and status policy."""
    require(isinstance(raw_text, str) and isinstance(token_ids, list), "bad decode")
    matches = list(re.finditer(DESCRIPTION_PATTERN, raw_text, re.DOTALL))
    if len(token_ids) == max_new_tokens and token_ids[-1] != eos_token_id:
        status, description = "truncated", None
    elif (
        len(matches) != 1
        or raw_text.count("<explanation>") != 1
        or raw_text.count("</explanation>") != 1
        or not matches[0].group(1).strip()
    ):
        status, description = "invalid_explanation_tags", None
    else:
        status, description = "ok", matches[0].group(1).strip()
    return {
        "raw_text": raw_text,
        "description": description,
        "token_ids": token_ids,
        "status": status,
        "injection_position": injection_position,
    }


def validate_ar_prompt_ids(ids, suffix_ids, bos_token_id):
    """The eager ar_prompt checks: terminal suffix convention and BOS policy."""
    require(ids and all(type(i) is int and i >= 0 for i in ids), "bad AR prompt")
    suffix = tuple(suffix_ids)
    require(suffix and all(type(i) is int for i in suffix), "bad AR suffix")
    require(tuple(ids[-len(suffix) :]) == suffix, "AR suffix-token mismatch")
    if bos_token_id is not None:
        require(ids[0] == bos_token_id, "AR BOS convention mismatch")


def read_value_head(path, width):
    """The eager load_value_head checks, returning the raw weight tensor."""
    path = Path(path)
    require(path.is_file(), "missing required value_head.safetensors")
    state = load_file(str(path), device="cpu")
    require(
        set(state) == {"weight"} and state["weight"].shape == (width, width),
        "AR value head must contain only a width-by-width weight",
    )
    weight = state["weight"]
    require(
        weight.is_floating_point() and bool(torch.isfinite(weight).all()),
        "invalid AR value head",
    )
    require(
        weight.dtype == torch.bfloat16,
        "AR value-head dtype differs from native backbone dtype",
    )
    return weight


def value_head_direction(terminal_hidden, weight):
    """Bias-free BF16 linear, then float32 CPU — the eager Reconstructor tail."""
    require(terminal_hidden.ndim == 1, "expected one terminal hidden state")
    vector = (
        torch.nn.functional.linear(terminal_hidden.to(weight.dtype), weight)
        .float()
        .cpu()
    )
    unit_direction(vector)  # fail closed before passing a degenerate direction
    return vector


def _read_json(path, what):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {what}: {exc}") from exc


def _engine_options(job, minimum_len):
    engine = job.get("engine") or {}
    require(type(engine) is dict, "engine options must be a mapping")
    utilization = engine.get("gpu_memory_utilization", 0.25)
    require(
        isinstance(utilization, (int, float))
        and not isinstance(utilization, bool)
        and 0.0 < utilization < 1.0,
        "gpu_memory_utilization must be in (0, 1); it is a fraction of TOTAL "
        "unified memory, so account for other residents",
    )
    max_model_len = engine.get("max_model_len")
    if max_model_len is None:
        max_model_len = minimum_len + 8
    require(
        type(max_model_len) is int and max_model_len >= minimum_len,
        f"max_model_len must cover the worst-case sequence ({minimum_len})",
    )
    seed = engine.get("seed", 1234)
    require(type(seed) is int, "engine seed must be an integer")
    return {
        "gpu_memory_utilization": float(utilization),
        "max_model_len": max_model_len,
        "seed": seed,
        "enforce_eager": True,  # measured configuration; no cudagraph confounds
        "enable_prefix_caching": False,  # recompute-per-pass semantics
    }


def validate_job(job, mode):
    require(type(job) is dict, "job must be a JSON object")
    require(job.get("schema_version") == JOB_SCHEMA_VERSION, "unsupported job schema")
    require(job.get("mode") == mode, f"job mode must be {mode}")
    model_dir = Path(job.get("model_dir") or "")
    require(model_dir.is_dir(), f"model directory missing: {model_dir}")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        require((model_dir / name).is_file(), f"model directory lacks {name}")
    output_dir = Path(job.get("output_dir") or "")
    require(str(output_dir) not in ("", "."), "output_dir required")
    rows = job.get("rows")
    require(type(rows) is list and rows, "job must list at least one row")
    require(
        all(
            type(row) is dict and type(row.get("id")) is str and row["id"]
            for row in rows
        ),
        "every row must be an object with a nonempty string id",
    )
    require(
        len({row["id"] for row in rows}) == len(rows),
        "row ids must be unique",
    )
    metadata = job.get("metadata")
    require(type(metadata) is dict, "job metadata missing")
    return model_dir, output_dir, rows, metadata


def _write_completion(output_dir, completion, outputs):
    completion = dict(completion)
    completion["outputs"] = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in outputs
        if Path(path).is_file()
    }
    path = Path(output_dir) / "completion.json"
    with path.open("x") as stream:
        json.dump(completion, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def _worker_environment():
    import importlib.metadata
    import platform

    return {
        "python": platform.python_version(),
        "vllm": importlib.metadata.version("vllm"),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "env": {
            key: os.environ.get(key)
            for key in (
                "VLLM_USE_FLASHINFER_SAMPLER",
                "VLLM_BATCH_INVARIANT",
                "VLLM_ALLOW_INSECURE_SERIALIZATION",
            )
        },
    }


def _import_vllm():
    import importlib.metadata

    version = importlib.metadata.version("vllm")
    require(
        version == VLLM_PIN,
        f"vllm {version} found; the pinned worker requires {VLLM_PIN} "
        "(rebuild via scripts/setup_vllm_worker.sh)",
    )


def _set_worker_env():
    # FlashInfer's sampler JIT cannot build here (system nvcc CUDA 12.0 gates
    # sm_12x on >=12.9); always use the native sampler.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # Batch-invariant kernels measured bitwise-repeatable at batch 1.
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")


def run_verbalize(job_path):
    job = _read_json(job_path, "verbalize job")
    model_dir, output_dir, rows, metadata = validate_job(job, "verbalize")
    started = time.perf_counter()
    width = metadata.get("width")
    require(type(width) is int and width > 0, "metadata width must be an integer")
    scale = metadata.get("injection_scale")
    prompt_ids = metadata.get("prompt_token_ids")
    require(
        type(prompt_ids) is list
        and prompt_ids
        and all(type(i) is int and i >= 0 for i in prompt_ids),
        "metadata prompt_token_ids must be nonnegative integers",
    )
    position = metadata.get("injection_position")
    injection_id = metadata.get("injection_id")
    require(
        type(position) is int
        and 0 < position < len(prompt_ids) - 1
        and prompt_ids[position] == injection_id
        and prompt_ids.count(injection_id) == 1
        and prompt_ids[position - 1] == metadata.get("left_id")
        and prompt_ids[position + 1] == metadata.get("right_id"),
        "injection marker context mismatch",
    )
    eos_token_id = metadata.get("eos_token_id")
    require(type(eos_token_id) is int, "metadata eos_token_id required")
    max_new_tokens = metadata.get("max_new_tokens", 200)
    require(
        type(max_new_tokens) is int and 1 <= max_new_tokens <= 200,
        "AV budget must be within the frozen 200-token ceiling",
    )
    payload_path = Path(job.get("payload_safetensors") or "")
    if not payload_path.is_absolute():
        payload_path = Path(job_path).parent / payload_path
    require(payload_path.is_file(), f"payload missing: {payload_path}")
    payload = load_file(str(payload_path), device="cpu")
    for row in rows:
        require(type(row.get("id")) is str and row["id"], "row id must be text")
        vector = payload.get(row["id"])
        require(vector is not None, f"payload lacks vector for row {row['id']}")
        # Shape/dtype only: tensor math before LLM() spawns the torch thread
        # pool and deadlocks the forked engine core. Finiteness fails closed
        # per row in scaled_injection/unit_direction after the boot.
        require(
            vector.shape == (width,) and vector.is_floating_point(),
            f"payload vector for row {row['id']} must be a width-{width} vector",
        )
    engine = _engine_options(job, len(prompt_ids) + max_new_tokens)
    config = _read_json(model_dir / "config.json", "AV config")
    require(
        config.get("model_type") == "qwen2"
        and config.get("hidden_size") == width
        and config.get("dtype", config.get("torch_dtype")) == "bfloat16",
        "AV config differs from the audited BF16 Qwen2 backbone",
    )

    _set_worker_env()
    _import_vllm()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_dir),
        enable_prompt_embeds=True,
        **engine,
    )
    tokenizer = llm.get_tokenizer()
    require(
        tokenizer.eos_token_id == eos_token_id,
        "worker tokenizer EOS disagrees with the job metadata",
    )
    records = {}
    row_stats = {}
    sampling = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    for row in rows:
        row_id = row["id"]
        start = time.perf_counter()
        try:
            embed_row = scaled_injection(payload[row_id], width, scale)
            generated = []
            for _ in range(max_new_tokens):
                # A fresh request per pass: full recompute of the injected
                # prefix plus the generated suffix, matching the eager loop.
                prompt = embeds_prompt(
                    [*prompt_ids, *generated], position, embed_row, width
                )
                outputs = llm.generate([prompt], sampling)[0].outputs[0]
                token_ids = list(outputs.token_ids)
                require(len(token_ids) == 1, "expected exactly one sampled token")
                token = token_ids[0]
                generated.append(token)
                if token == eos_token_id:
                    break
            raw = tokenizer.decode(generated, skip_special_tokens=False)
            record = parse_description_record(
                raw,
                generated,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                injection_position=position,
            )
        except (ValueError, RuntimeError) as error:
            record = {"status": "failed", "error": str(error)}
        records[row_id] = record
        row_stats[row_id] = {
            "status": record["status"],
            "seconds": time.perf_counter() - start,
            "sampled_tokens": len(record.get("token_ids") or []),
        }
        print(f"  AV[vllm]: {row_id} {record['status']}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    descriptions_path = output_dir / "descriptions.json"
    with descriptions_path.open("x") as stream:
        json.dump(
            {
                "schema_version": JOB_SCHEMA_VERSION,
                "mode": "verbalize",
                "backend": f"vllm-{VLLM_PIN}",
                "records": records,
            },
            stream,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    completion = {
        "status": "COMPLETE"
        if all(stats["status"] == "ok" for stats in row_stats.values())
        else "COMPLETE_WITH_FAILURES",
        "mode": "verbalize",
        "model_dir": str(model_dir),
        "environment": _worker_environment(),
        "engine": engine,
        "rows": row_stats,
        # Batch-1 fresh requests: one model forward per sampled token, matching
        # the eager adapter's per-pass forward count.
        "model_forward_calls": sum(
            stats["sampled_tokens"] for stats in row_stats.values()
        ),
        "boot_and_generate_seconds": time.perf_counter() - started,
    }
    _write_completion(output_dir, completion, [descriptions_path])
    print(f"COMPLETE: {descriptions_path}", flush=True)


class _IdentityAddNorm(torch.nn.Module):
    """The eager recipe's removed final norm under vllm's fused-residual layers.

    Eager replaces model.norm with torch.nn.Identity, exposing the post-residual
    block-20 output. vllm's decoder layers return (delta, residual) with the
    residual addition deferred into the next norm, so identity here means
    returning the sum: the post-residual hidden state, twice (vllm unpacks the
    second element as the new residual and discards it after the last block).
    """

    def forward(self, hidden_states, residual=None):
        output = hidden_states if residual is None else hidden_states + residual
        return output, output


def _install_identity_norm(model):
    inner = getattr(model, "model", model)
    inner.norm = _IdentityAddNorm()
    return type(inner.norm).__name__


def verify_ar_checkpoint_inventory(model_dir, num_layers):
    """Name-level load-completeness proof for the AR checkpoint.

    vllm's strict weights tracker is disabled for this load (the released AR
    omits model.norm.weight, which the recipe replaces with Identity anyway).
    To keep the eager loader's fail-loud discipline, verify the checkpoint
    inventory directly: every backbone tensor the fused Qwen2 mapping consumes
    must be present, and the only tensors allowed beyond the backbone are the
    documented norm/lm_head omissions (ignored after the identity-norm swap).
    """
    per_layer = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    )
    index = _read_json(model_dir / "model.safetensors.index.json", "AR weight index")
    names = set(index.get("weight_map") or {})
    require(names, "AR weight index is empty")
    expected = {"model.embed_tokens.weight"} | {
        f"model.layers.{layer}.{name}"
        for layer in range(num_layers)
        for name in per_layer
    }
    missing = expected - names
    require(
        not missing,
        f"AR checkpoint lacks expected backbone tensors: {sorted(missing)[:5]}",
    )
    extra = names - expected - ALLOWED_MISSING_AR_WEIGHTS
    require(
        not extra,
        f"AR checkpoint carries unexpected tensors: {sorted(extra)[:5]}",
    )


def _load_ar_engine(model_dir, engine):
    """Single pooling load; checkpoint completeness was proven name-level above."""
    from vllm import LLM
    from vllm.config import PoolerConfig

    llm = LLM(
        model=str(model_dir),
        runner="pooling",
        convert="embed",
        pooler_config=PoolerConfig(task="token_embed"),
        model_loader_extra_config={"enable_weights_track": False},
        **engine,
    )
    return llm, "weights-track off; checkpoint inventory verified name-level"


def run_reconstruct(job_path):
    job = _read_json(job_path, "reconstruct job")
    model_dir, output_dir, rows, metadata = validate_job(job, "reconstruct")
    started = time.perf_counter()
    width = metadata.get("width")
    require(type(width) is int and width > 0, "metadata width must be an integer")
    template = metadata.get("ar_template")
    require(type(template) is str and "{explanation}" in template, "bad AR template")
    suffix_ids = metadata.get("suffix_ids")
    config = _read_json(model_dir / "config.json", "AR config")
    require(
        config.get("model_type") == "qwen2"
        and config.get("hidden_size") == width
        and type(config.get("num_hidden_layers")) is int
        and config.get("dtype", config.get("torch_dtype")) == "bfloat16",
        "AR config differs from the audited truncated BF16 Qwen2",
    )
    head_path = Path(job.get("value_head") or "")
    require(head_path.is_file(), "missing required value_head.safetensors")
    verify_ar_checkpoint_inventory(model_dir, config["num_hidden_layers"])
    for row in rows:
        require(
            type(row.get("description")) is str and row["description"].strip(),
            f"row {row.get('id')}: AR needs a nonempty description",
        )
        expected = row.get("prompt_token_ids")
        require(
            type(expected) is list
            and expected
            and all(type(i) is int and i >= 0 for i in expected),
            f"row {row['id']}: parent-computed prompt ids required",
        )
        validate_ar_prompt_ids(expected, suffix_ids, None)
    longest = max(len(row["prompt_token_ids"]) for row in rows)
    engine = _engine_options(job, longest)

    _set_worker_env()
    # apply_model ships this module's callables to the engine-core process;
    # the worker is a local subprocess running hash-verified jobs only.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    _import_vllm()
    from vllm import PoolingParams

    llm, load_mode = _load_ar_engine(model_dir, engine)
    tokenizer = llm.get_tokenizer()
    swapped = llm.apply_model(_install_identity_norm)
    require(
        swapped == ["_IdentityAddNorm"],
        f"identity-norm injection failed: {swapped}",
    )
    # Post-boot: read_value_head runs torch.isfinite over the 25 MB weight,
    # which spawns the torch thread pool — fatal before the engine-core fork.
    weight = read_value_head(head_path, width)
    pooling = PoolingParams(use_activation=False)
    vectors = {}
    row_stats = {}
    for row in rows:
        row_id = row["id"]
        start = time.perf_counter()
        try:
            ids = tokenizer.encode(
                template.format(explanation=row["description"]),
                add_special_tokens=True,
            )
            validate_ar_prompt_ids(ids, suffix_ids, tokenizer.bos_token_id)
            require(
                ids == row["prompt_token_ids"],
                f"row {row_id}: worker/parent AR tokenization disagree",
            )
            output = llm.encode(
                [{"prompt_token_ids": ids}],
                pooling_params=pooling,
                pooling_task="token_embed",
            )[0]
            hidden = output.outputs.data
            require(
                hidden.ndim == 2
                and hidden.shape[0] == len(ids)
                and hidden.shape[1] == width
                and bool(torch.isfinite(hidden).all()),
                f"row {row_id}: unexpected token_embed output shape",
            )
            vectors[row_id] = value_head_direction(hidden[-1], weight)
        except (ValueError, RuntimeError) as error:
            row_stats[row_id] = {
                "status": "failed",
                "error": str(error),
                "seconds": time.perf_counter() - start,
            }
        else:
            row_stats[row_id] = {
                "status": "ok",
                "seconds": time.perf_counter() - start,
            }
        print(f"  AR[vllm]: {row_id} {row_stats[row_id]['status']}", flush=True)
    require(vectors, "no AR reconstruction succeeded; refusing empty output")
    output_dir.mkdir(parents=True, exist_ok=False)
    directions_path = output_dir / "directions.safetensors"
    save_file(vectors, str(directions_path))
    completion = {
        "status": "COMPLETE"
        if all(stats["status"] == "ok" for stats in row_stats.values())
        else "COMPLETE_WITH_FAILURES",
        "mode": "reconstruct",
        "model_dir": str(model_dir),
        "environment": _worker_environment(),
        "engine": engine,
        "ar_weight_load": load_mode,
        "final_norm": "removed: identity module returning the post-residual hidden state (eager recipe)",
        "rows": row_stats,
        # One pooling encode per description: one full forward each.
        "model_forward_calls": len(vectors),
        "boot_and_reconstruct_seconds": time.perf_counter() - started,
    }
    _write_completion(output_dir, completion, [directions_path])
    print(f"COMPLETE: {directions_path}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verbalize", "reconstruct"))
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == "verbalize":
            run_verbalize(args.job)
        else:
            run_reconstruct(args.job)
    except Exception as exc:  # noqa: BLE001 - the worker boundary reports every failure mode
        print(
            f"vllm worker {args.mode} failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
