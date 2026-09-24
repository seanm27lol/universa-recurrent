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

from .artifacts import sha256_file
from .nla_adapter import ar_prompt, av_prompt, check_pair, load_metadata

EXPECTED_REPOS = {
    "target": "Qwen/Qwen2.5-7B-Instruct",
    "av": "kitft/nla-qwen2.5-7b-L20-av",
    "ar": "kitft/nla-qwen2.5-7b-L20-ar",
}


def read_lock(path):
    lock = json.loads(Path(path).read_text())
    if lock.get("schema_version") != 1 or set(lock["models"]) != set(EXPECTED_REPOS):
        raise ValueError("unsupported source lock")
    for role, entry in lock["models"].items():
        if entry["repo_id"] != EXPECTED_REPOS[role] or not re.fullmatch(
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
        for name, file in entry["files"].items():
            if PurePosixPath(name).name != name or not re.fullmatch(
                r"[0-9a-f]{64}", file["sha256"]
            ):
                raise ValueError("unsafe artifact path or missing hash")
            if type(file["bytes"]) is not int or file["bytes"] < 1:
                raise ValueError("invalid artifact size")
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


def inspect_metadata(paths):
    from transformers import AutoTokenizer

    target_config = json.loads((paths["target"] / "config.json").read_text())
    if (
        target_config["hidden_size"],
        target_config["num_hidden_layers"],
        target_config["model_type"],
    ) != (3584, 28, "qwen2"):
        raise ValueError("target differs from audited Qwen2.5-7B configuration")
    av = load_metadata(paths["av"], "av", target_config)
    ar = load_metadata(paths["ar"], "ar", target_config)
    check_pair(av, ar)
    if av.layer != 20:
        raise ValueError("released pair differs from audited block 20")
    tokenizers = {
        role: AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        for role, path in paths.items()
    }
    av_ids, position = av_prompt(tokenizers["av"], av)
    ar_ids = ar_prompt(tokenizers["ar"], ar, "tokenizer preflight")
    return (
        tokenizers,
        av,
        ar,
        {
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
        },
    )


def load_model(path, role, device):
    from transformers import AutoModelForCausalLM

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
    _, _, _, report["metadata"] = inspect_metadata(paths)
    report["real_model_inference"] = "NOT RUN"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
