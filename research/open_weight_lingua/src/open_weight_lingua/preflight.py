"""Verify immutable models and software before the engineering smoke test."""

import argparse
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import time
import torch

from .architectures import spec_for_config, spec_for_repos, text_config_dict
from .artifacts import sha256_file
from .nla_adapter import ar_prompt, av_prompt, check_pair, load_metadata


def read_lock(path):
    lock = json.loads(Path(path).read_text())
    if lock.get("schema_version") != 1 or not isinstance(lock.get("models"), dict):
        raise ValueError("unsupported source lock")
    spec = spec_for_repos(
        {role: entry.get("repo_id") for role, entry in lock["models"].items()}
    )
    for role, entry in lock["models"].items():
        if entry["repo_id"] != spec.repos[role] or not re.fullmatch(
            r"[0-9a-f]{40}", entry["revision"]
        ):
            raise ValueError("wrong model or non-immutable model revision")
        required = {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        }
        if role in ("av", "ar"):
            required.add("nla_meta.yaml")
        if role == "ar":
            required.add("value_head.safetensors")
        if not required <= entry["files"].keys():
            raise ValueError("source lock is missing required artifacts")
        if "serving_dtype" in entry and (
            entry["serving_dtype"] != "bfloat16"
            or not entry.get("serving_dtype_note")
        ):
            raise ValueError("unsupported or unjustified serving dtype cast")
        pending = []
        for name, file in entry["files"].items():
            if PurePosixPath(name).name != name:
                raise ValueError("unsafe artifact path or missing hash")
            if file["sha256"] is None:
                pending.append(name)
            elif not re.fullmatch(r"[0-9a-f]{64}", file["sha256"]):
                raise ValueError("unsafe artifact path or missing hash")
            if type(file["bytes"]) is not int or file["bytes"] < 1:
                raise ValueError("invalid artifact size")
        if pending:
            raise ValueError(
                "source lock has unresolved gated-file hashes "
                f"({role}: {', '.join(sorted(pending))}); accept the repository "
                "terms, set HF_TOKEN, and run scripts/complete_gemma3_lock.py"
            )
        if entry["download_bytes"] != sum(f["bytes"] for f in entry["files"].values()):
            raise ValueError("source-lock byte total mismatch")
    for entry in lock["sources"].values():
        if not re.fullmatch(r"[0-9a-f]{40}", entry["revision"]):
            raise ValueError("inference/training source revision must be immutable")
    return lock


def model_paths(lock, cache, overrides=None):
    overrides = overrides or {}
    return {
        role: Path(overrides.get(role) or Path(cache) / role / entry["revision"])
        for role, entry in lock["models"].items()
    }


def verify_models(lock, paths, *, metadata_only=False):
    checked = {}
    for role, entry in lock["models"].items():
        checked[role] = 0
        for name, expected in entry["files"].items():
            if metadata_only and name.endswith(".safetensors"):
                continue
            file = paths[role] / name
            if not file.is_file():
                raise ValueError(
                    f"missing pinned {role}/{name}; run with --fetch-models or supply verified model paths"
                )
            if (
                file.stat().st_size != expected["bytes"]
                or sha256_file(file) != expected["sha256"]
            ):
                raise ValueError(
                    f"hash/size mismatch: {role}/{name}; restore the pinned file"
                )
            checked[role] += expected["bytes"]
    return checked


def fetch_models(lock, paths):
    """Only called on the user's explicit --fetch-models step; no moving refs."""
    from huggingface_hub import hf_hub_download

    fetched = 0
    start = time.perf_counter()
    for role, entry in lock["models"].items():
        paths[role].mkdir(parents=True, exist_ok=True)
        remaining = sum(
            f["bytes"]
            for name, f in entry["files"].items()
            if not (paths[role] / name).exists()
        )
        if shutil.disk_usage(paths[role]).free < remaining + 5 * 1024**3:
            raise RuntimeError(
                f"insufficient storage for {role}; need pinned bytes plus 5 GiB headroom"
            )
        for name, file in entry["files"].items():
            local = paths[role] / name
            if local.is_file():
                if (
                    local.stat().st_size == file["bytes"]
                    and sha256_file(local) == file["sha256"]
                ):
                    continue
                raise ValueError(f"refusing to overwrite incompatible {role}/{name}")
            hf_hub_download(
                entry["repo_id"],
                filename=name,
                revision=entry["revision"],
                local_dir=paths[role],
            )
            fetched += file["bytes"]
    return {
        "new_artifact_bytes": fetched,
        "seconds": time.perf_counter() - start,
        "network_transfer_bytes": "not instrumented; Hub transport may deduplicate/retry",
    }


def compatibility(device="cpu"):
    report = {
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "safetensors",
                "numpy",
                "PyYAML",
                "huggingface-hub",
            )
        },
        "cuda_build": torch.version.cuda,
        "device": device,
        "attention": "eager",
        "use_cache": False,
        "native_dtype": "bfloat16",
        "quantized": False,
        "disk_free_bytes": shutil.disk_usage(".").free,
    }
    if Path("/proc/meminfo").exists():
        memory = dict(
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
        report["system_memory_available_bytes"] = (
            int(memory["MemAvailable"].split()[0]) * 1024
        )
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; install the locked ARM CUDA 13 environment on the DGX"
            )
        report.update(
            gpu_name=torch.cuda.get_device_name(),
            gpu_capability=list(torch.cuda.get_device_capability()),
            compiled_architectures=torch.cuda.get_arch_list(),
        )
        # Driver/device discovery alone does not establish executable BF16 kernels.
        values = torch.eye(8, dtype=torch.bfloat16, device=device)
        if not torch.equal(values @ values, values):
            raise RuntimeError("BF16 GPU kernel check failed")
        torch.cuda.synchronize()
        report["bf16_kernel_check"] = "PASS"
    return report


def inspect_metadata(paths, lock=None):
    from transformers import AutoTokenizer

    serving_dtypes = {
        role: entry.get("serving_dtype")
        for role, entry in (lock or {}).get("models", {}).items()
    }
    target_config = json.loads((paths["target"] / "config.json").read_text())
    text_config = text_config_dict(target_config)
    spec = spec_for_config(target_config)
    if (
        text_config["hidden_size"],
        text_config["num_hidden_layers"],
        text_config["model_type"],
    ) != (spec.hidden_size, spec.num_hidden_layers, spec.text_model_type):
        raise ValueError(
            f"target differs from the audited {spec.family} configuration"
        )
    av = load_metadata(
        paths["av"], "av", target_config, serving_dtype=serving_dtypes.get("av")
    )
    ar = load_metadata(
        paths["ar"], "ar", target_config, serving_dtype=serving_dtypes.get("ar")
    )
    check_pair(av, ar)
    if av.layer != spec.extraction_layer:
        raise ValueError(
            f"released pair differs from the audited block {spec.extraction_layer}"
        )
    tokenizers = {
        role: AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        for role, path in paths.items()
    }
    av_ids, position = av_prompt(tokenizers["av"], av)
    ar_ids = ar_prompt(tokenizers["ar"], ar, "tokenizer preflight")
    metadata = {
        "width": av.width,
        "layer": av.layer,
        "injection_scale": av.injection_scale,
        "av_prompt_ids": av_ids,
        "av_injection_position": position,
        "ar_probe_ids": ar_ids,
        "ar_layers": ar.layers,
        "ar_suffix_ids": list(ar.suffix_ids),
        "av_template": av.av_template,
        "ar_template": ar.ar_template,
    }
    if serving_dtypes.get("av"):
        # Lock-declared serving cast; the released artifact's native precision
        # and the cast both go into the manifest (the full note is in the lock).
        av_config = json.loads((paths["av"] / "config.json").read_text())
        metadata["av_native_dtype"] = av_config.get(
            "dtype", av_config.get("torch_dtype")
        )
        metadata["av_serving_dtype"] = serving_dtypes["av"]
    return (tokenizers, av, ar, metadata)


def _load_cast_checkpoint_to_device(path, role, device):
    """Stream a non-BF16-native checkpoint straight onto the device.

    Why this exists (measured 2026-09-23 on the GB10, probes in
    /tmp/owl27_memprobe): the fp32-native 27B AV loaded via the stock path —
    CPU materialization plus ``model.to("cuda")`` — was OOM-killed twice at
    shard 22/22 with ~47 GiB of anonymous CPU RSS (the whole bf16-cast copy)
    coexisting with the growing GPU copy; per-module moves with gc/malloc_trim
    did not drain it. Here the model is built on the meta device, materialized
    uninitialized on the target device, and filled tensor-by-tensor from the
    mmap'd shards (reclaimable file-backed reads, one BF16 cast copy at a
    time), so no whole-model CPU copy ever exists. Fixture-proven
    bitwise-identical to stock from_pretrained + cast, including the
    config-derived non-persistent buffers, which are never in the checkpoint
    and are rebuilt from the config below. gemma3_text only; anything else
    fails closed.
    """
    import copy as copy_module

    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding

    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    if config.model_type != "gemma3_text":
        raise ValueError(
            f"the streaming serving cast is implemented for gemma3_text only, "
            f"not {config.model_type!r}"
        )
    config._attn_implementation = "eager"
    # Match stock from_pretrained(torch_dtype=bf16), which sets the served
    # dtype on the config before building: parameters must be built BF16, not
    # built fp32 and cast on fill.
    config.dtype = torch.bfloat16
    config.torch_dtype = torch.bfloat16
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    # Match stock from_pretrained, which merges the checkpoint's
    # generation_config.json into the loaded model (the released 27B AV
    # declares eos [1, 106] there and eos 1 in config.json; losing the merge
    # silently shrinks the AV stop set — root cause of the 27B smoke's
    # truncated descriptions).
    from transformers import GenerationConfig

    try:
        model.generation_config = GenerationConfig.from_pretrained(str(path))
    except OSError:
        model.generation_config = GenerationConfig.from_model_config(model.config)
    model.to_empty(device=device)
    state = dict(model.state_dict())
    index_path = Path(path) / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]
    loaded = set()
    with torch.no_grad():
        for shard in shards:
            with safe_open(str(Path(path) / shard), framework="pt") as stream:
                for key in stream.keys():
                    if key not in state:
                        raise ValueError(
                            f"{role} checkpoint has unexpected key {key}"
                        )
                    tensor = stream.get_tensor(key)
                    if not tensor.is_floating_point():
                        raise ValueError(f"non-floating checkpoint tensor {key}")
                    target = state[key]
                    if tensor.shape != target.shape:
                        raise ValueError(f"checkpoint shape mismatch on {key}")
                    target.copy_(tensor.to(torch.bfloat16))
                    del tensor
                    loaded.add(key)
    tied = set(getattr(model, "_tied_weights_keys", None) or ())
    allowed_missing = (
        {"lm_head.weight", "model.norm.weight"} if role == "ar" else set()
    ) | tied
    missing = set(state) - loaded - allowed_missing
    if missing:
        raise ValueError(f"{role} checkpoint missing keys: {sorted(missing)}")
    model.tie_weights()
    # Config-derived, non-persistent buffers are never checkpointed; the
    # meta-built copies hold garbage until rebuilt here. Fail closed if a
    # future checkpoint layout adds anything beyond this known set.
    derived = {name for name, _ in model.named_buffers()} - loaded
    expected_derived = {
        "model.embed_tokens.embed_scale",
        "model.rotary_emb.inv_freq",
        "model.rotary_emb_local.inv_freq",
    }
    if derived != expected_derived:
        raise ValueError(f"unexpected non-checkpoint buffers: {sorted(derived)}")
    model.model.rotary_emb = Gemma3RotaryEmbedding(config=config, device=device)
    local_config = copy_module.deepcopy(config)
    local_config.rope_theta = config.rope_local_base_freq
    local_config.rope_scaling = {"rope_type": "default"}
    model.model.rotary_emb_local = Gemma3RotaryEmbedding(
        config=local_config, device=device
    )
    # from_pretrained casts this buffer to the served dtype; match that or the
    # embedding multiply promotes the residual stream to float32.
    model.model.embed_tokens.embed_scale = torch.tensor(
        config.hidden_size**0.5,
        dtype=state["model.embed_tokens.weight"].dtype,
        device=device,
    )
    return model.eval().requires_grad_(False)


def load_model(path, role, device):
    from transformers import AutoModelForCausalLM

    # The pipeline serves BF16. When the pinned checkpoint declares another
    # native dtype (the 27B AV ships float32), the cast is lock-declared; log
    # it loudly rather than letting from_pretrained cast silently.
    config_path = Path(path) / "config.json"
    declared = None
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        declared = config.get("dtype", config.get("torch_dtype"))
    if declared is not None and declared != "bfloat16":
        print(
            f"  Serving dtype cast: {role} checkpoint declares {declared}; "
            "loading BF16 per the lock's serving_dtype declaration",
            flush=True,
        )
        if str(device).startswith("cuda"):
            return _load_cast_checkpoint_to_device(path, role, device)
    model, info = AutoModelForCausalLM.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        output_loading_info=True,
    )
    allowed_missing = {"lm_head.weight", "model.norm.weight"} if role == "ar" else set()
    if (
        set(info["missing_keys"]) - allowed_missing
        or info["unexpected_keys"]
        or info["mismatched_keys"]
        or info["error_msgs"]
    ):
        raise ValueError(f"{role} checkpoint does not load completely: {info}")
    return model.to(device).eval().requires_grad_(False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--fetch-models", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    report = compatibility(args.device)
    if args.fetch_models:
        report["fetch"] = fetch_models(lock, paths)
    report["verified_bytes"] = verify_models(
        lock, paths, metadata_only=args.metadata_only
    )
    _, _, _, report["metadata"] = inspect_metadata(paths, lock)
    report["real_model_inference"] = "NOT RUN"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
