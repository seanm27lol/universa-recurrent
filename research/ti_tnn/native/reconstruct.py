"""CPU reconstruction and first-divergence replay for native checkpoints.

A JSON replay spec fixes token IDs, positions, document boundaries, padding,
and teacher-forced labels. Reconstruction verifies the checkpoint SHA256,
restores its exact configuration and floating dtype, then compares full and
cached execution without silently upcasting the model. Output hashes identify
the replay tensors; first-divergence records retain exact indices and values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .data import MODEL_INPUT_ARRAYS
from .evaluate import _native_config, checkpoint_model_dtype, load_native_model
from .generation import greedy_generate
from .losses import next_token_cross_entropy
from .sampler import identity_hash


REPLAY_FORMAT = "native-reconstruction-replay-v1"
REPORT_FORMAT = "native-reconstruction-report-v1"
OUTPUT_NAMES = ("logits", "h", "slots", "state_logits", "route_logits", "route_probs", "route_indices")


class ReconstructionError(RuntimeError):
    pass


def dtype_tolerances(dtype: torch.dtype) -> dict[str, float]:
    values = {
        torch.float64: (1e-12, 1e-12),
        torch.float32: (3e-6, 3e-6),
        torch.float16: (2e-3, 2e-3),
        torch.bfloat16: (2e-2, 2e-2),
    }
    if dtype not in values:
        raise ReconstructionError(f"no replay tolerances declared for checkpoint dtype {dtype}")
    atol, rtol = values[dtype]
    return {"atol": atol, "rtol": rtol}


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash dtype, shape, and contiguous bytes, including bfloat16 without conversion."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    header = json.dumps([str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    if tensor.numel():
        digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_descriptor(value: torch.Tensor | None) -> dict | None:
    if value is None:
        return None
    return {
        "sha256": tensor_sha256(value),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "device": str(value.device),
    }


def _first_tensor_index(mask: torch.Tensor) -> tuple[int, ...] | None:
    if not mask.any():
        return None
    flat = int(mask.flatten().to(torch.int8).argmax())
    return tuple(int(value) for value in np.unravel_index(flat, mask.shape))


def compare_tensors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict:
    """Compare at the supplied dtype and locate exact and tolerance divergence."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        return {
            "agrees": False,
            "exact": False,
            "shape_dtype_device_match": False,
            "actual": _tensor_descriptor(actual),
            "expected": _tensor_descriptor(expected),
            "first_divergence": None,
            "first_tolerance_violation": None,
        }
    exact_mask = actual.ne(expected)
    if actual.dtype.is_floating_point:
        difference = (actual - expected).abs()
        tolerance_violation = difference > (atol + rtol * expected.abs())
        agrees = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
        maximum = float(difference.max()) if difference.numel() else 0.0
    else:
        difference = (actual.long() - expected.long()).abs()
        tolerance_violation = exact_mask
        agrees = not exact_mask.any()
        maximum = int(difference.max()) if difference.numel() else 0

    def detail(mask: torch.Tensor) -> dict | None:
        index = _first_tensor_index(mask)
        if index is None:
            return None
        return {
            "index": list(index),
            "actual": float(actual[index]) if actual.dtype.is_floating_point else int(actual[index]),
            "expected": float(expected[index]) if expected.dtype.is_floating_point else int(expected[index]),
            "absolute_difference": float(difference[index]),
        }

    return {
        "agrees": agrees,
        "exact": not exact_mask.any(),
        "shape_dtype_device_match": True,
        "different_values": int(exact_mask.sum()),
        "maximum_absolute_difference": maximum,
        "atol": atol,
        "rtol": rtol,
        "first_divergence": detail(exact_mask),
        "first_tolerance_violation": detail(tolerance_violation),
    }


def _model_inputs(sample: Mapping) -> dict[str, torch.Tensor]:
    missing = [name for name in MODEL_INPUT_ARRAYS if name not in sample]
    if missing:
        raise ReconstructionError(f"teacher-forced replay sample is missing model inputs {missing}")
    return {name: sample[name] for name in MODEL_INPUT_ARRAYS}


def _concatenate(pieces: Sequence[torch.Tensor | None]) -> torch.Tensor | None:
    if not pieces or pieces[0] is None:
        return None
    return torch.cat(pieces, dim=1)


@torch.no_grad()
def replay_teacher_forced(
    model,
    sample: Mapping[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> dict:
    """Replay one fixed sample in full and token-by-token cached execution."""
    inputs = _model_inputs(sample)
    if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[1] < 2:
        raise ReconstructionError("teacher-forced input_ids must have shape [B,T], with T >= 2")
    full = model(**inputs, return_aux=True, return_trace=True)
    cache = None
    pieces = []
    for token_index in range(inputs["input_ids"].shape[1]):
        step_inputs = {
            name: value[:, token_index: token_index + 1]
            for name, value in inputs.items()
        }
        step = model(
            **step_inputs,
            cache=cache,
            use_cache=True,
            return_aux=True,
            return_trace=True,
        )
        cache = step.cache
        pieces.append(step)
    comparisons = {}
    full_hashes = {}
    cached_hashes = {}
    for name in OUTPUT_NAMES:
        expected = getattr(full, name)
        actual = _concatenate([getattr(piece, name) for piece in pieces])
        full_hashes[name] = _tensor_descriptor(expected)
        cached_hashes[name] = _tensor_descriptor(actual)
        if expected is None:
            comparisons[name] = None
        else:
            comparisons[name] = compare_tensors(actual, expected, atol=atol, rtol=rtol)
    if "labels" not in sample:
        raise ReconstructionError("teacher-forced replay sample requires labels")
    loss, token_count = next_token_cross_entropy(full.logits, sample["labels"])
    sample_hash = identity_hash({name: sample[name] for name in sorted(sample)})
    return {
        "sample_sha256": sample_hash,
        "input_hashes": {name: identity_hash(value) for name, value in sorted(sample.items())},
        "full_outputs": full_hashes,
        "cached_outputs": cached_hashes,
        "comparisons": comparisons,
        "loss": float(loss),
        "loss_tokens": int(token_count),
        "agrees": all(item is None or item["agrees"] for item in comparisons.values()),
        "execution": "full sequence versus one-token incremental NativeCache at the restored checkpoint dtype/device",
    }


def _prompt_tensors(record: Mapping) -> tuple[torch.Tensor, torch.Tensor]:
    ids = record["input_ids"]
    positions = record["positions"]
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0 or positions.shape != ids.shape:
        raise ReconstructionError("each generation replay needs one nonempty [1,T] prompt and matching positions")
    return ids.long(), positions.long()


def _sequence_divergence(actual: Sequence[int], expected: Sequence[int]) -> dict | None:
    for index, (left, right) in enumerate(zip(actual, expected)):
        if left != right:
            return {"generated_token": index, "cached": int(left), "full": int(right)}
    if len(actual) != len(expected):
        index = min(len(actual), len(expected))
        return {
            "generated_token": index,
            "cached": int(actual[index]) if index < len(actual) else None,
            "full": int(expected[index]) if index < len(expected) else None,
        }
    return None


@torch.no_grad()
def replay_generation_prompts(
    model,
    prompts: Sequence[Mapping],
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> dict:
    """Replay fixed tokenized prompts with full and cached greedy decoding."""
    records = []
    for index, prompt in enumerate(prompts):
        ids, positions = _prompt_tensors(prompt)
        if ids.shape[1] + max_new_tokens > model.config.context_length:
            raise ReconstructionError("generation replay exceeds the restored context_length")
        sequences = {}
        for cached in (False, True):
            sequence = greedy_generate(
                model,
                ids,
                max_new_tokens=max_new_tokens,
                positions=positions,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
                use_cache=cached,
            )
            sequences[cached] = sequence
        full = sequences[False][0].cpu().tolist()
        cached = sequences[True][0].cpu().tolist()
        prompt_length = ids.shape[1]
        divergence = _sequence_divergence(cached[prompt_length:], full[prompt_length:])
        records.append({
            "example_id": prompt.get("example_id", f"prompt-{index}"),
            "prompt_sha256": identity_hash({"input_ids": ids, "positions": positions}),
            "full_sequence_sha256": tensor_sha256(sequences[False]),
            "cached_sequence_sha256": tensor_sha256(sequences[True]),
            "full_generated_ids": full[prompt_length:],
            "cached_generated_ids": cached[prompt_length:],
            "agrees": divergence is None,
            "first_divergence": divergence,
        })
    return {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_token_id,
        "model_inputs": ["input_ids", "positions"],
        "records": records,
        "agrees": all(record["agrees"] for record in records),
    }


def _normalize_teacher_sample(values: Mapping) -> dict[str, torch.Tensor]:
    result = {}
    for name, value in values.items():
        if name == "key_padding_mask" or name.endswith("_mask"):
            result[name] = torch.as_tensor(value, dtype=torch.bool)
        elif name in {"route_targets", "state_targets"}:
            result[name] = torch.as_tensor(value).long()
        else:
            result[name] = torch.as_tensor(value).long()
    return result


def load_replay_spec(path) -> dict:
    """Load a JSON replay spec without tokenization, execution, or network access."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("format") != REPLAY_FORMAT:
        raise ReconstructionError(f"replay spec format must be {REPLAY_FORMAT}")
    if not isinstance(raw.get("teacher_forced"), Mapping):
        raise ReconstructionError("replay spec needs a teacher_forced sample")
    prompts = raw.get("generation_prompts")
    if not isinstance(prompts, list):
        raise ReconstructionError("replay spec needs a generation_prompts list")
    normalized_prompts = []
    for record in prompts:
        if not isinstance(record, Mapping) or "positions" not in record:
            raise ReconstructionError("every fixed generation prompt must explicitly declare positions")
        normalized_prompts.append({
            "example_id": record.get("example_id"),
            "input_ids": torch.as_tensor(record["input_ids"]).long(),
            "positions": torch.as_tensor(record["positions"]).long(),
        })
    max_new_tokens = raw.get("max_new_tokens", 8)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ReconstructionError("max_new_tokens must be a nonnegative integer")
    return {
        "format": REPLAY_FORMAT,
        "teacher_forced": _normalize_teacher_sample(raw["teacher_forced"]),
        "generation_prompts": normalized_prompts,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": raw.get("eos_token_id"),
        "spec_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def compare_reference_replay(actual: Mapping, expected: Mapping) -> dict:
    """Compare stable reconstruction identities and localize the first mismatch."""
    comparisons = []

    def compare(path, left, right):
        if left != right:
            comparisons.append({"path": path, "actual": left, "expected": right})

    compare("checkpoint.sha256", actual["checkpoint"]["sha256"], expected["checkpoint"]["sha256"])
    compare("configuration.sha256", actual["configuration"]["sha256"], expected["configuration"]["sha256"])
    compare("weights.sha256", actual["weights"]["sha256"], expected["weights"]["sha256"])
    teacher_expected = expected["teacher_forced"]
    teacher_actual = actual["teacher_forced"]
    compare("teacher_forced.sample_sha256", teacher_actual["sample_sha256"], teacher_expected["sample_sha256"])
    compare("teacher_forced.loss", teacher_actual["loss"], teacher_expected["loss"])
    for name in OUTPUT_NAMES:
        left = teacher_actual["full_outputs"][name]
        right = teacher_expected["full_outputs"][name]
        compare(f"teacher_forced.full_outputs.{name}", None if left is None else left["sha256"],
                None if right is None else right["sha256"])
    expected_records = expected["generation"]["records"]
    compare("generation.record_count", len(actual["generation"]["records"]), len(expected_records))
    for index, (left, right) in enumerate(zip(actual["generation"]["records"], expected_records)):
        compare(f"generation.{index}.prompt_sha256", left["prompt_sha256"], right["prompt_sha256"])
        compare(f"generation.{index}.full_sequence_sha256", left["full_sequence_sha256"],
                right["full_sequence_sha256"])
    return {
        "agrees": not comparisons,
        "mismatches": comparisons,
        "first_divergence": comparisons[0] if comparisons else None,
    }


def reconstruct_checkpoint(
    checkpoint_path,
    replay_spec: Mapping,
    *,
    expected_sha256: str | None = None,
    expected_report: Mapping | None = None,
) -> dict:
    """Restore and replay entirely at the checkpoint's actual CPU dtype."""
    model, payload = load_native_model(checkpoint_path, expected_sha256=expected_sha256, device="cpu")
    dtype = checkpoint_model_dtype(payload)
    tolerances = dtype_tolerances(dtype)
    teacher = replay_teacher_forced(model, replay_spec["teacher_forced"], **tolerances)
    generation = replay_generation_prompts(
        model,
        replay_spec["generation_prompts"],
        max_new_tokens=replay_spec["max_new_tokens"],
        eos_token_id=replay_spec.get("eos_token_id"),
    )
    config = _native_config(payload["config"])
    report = {
        "format": REPORT_FORMAT,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": payload["sha256"],
            "expected_sha256": expected_sha256,
            "sha256_verified": True,
            "step": payload["step"],
        },
        "configuration": {
            "saved": payload["config"],
            "restored_model": config.to_dict(),
            "sha256": identity_hash(payload["config"]),
            "restored_model_sha256": identity_hash(config.to_dict()),
            "exact": config == _native_config(payload["config"]),
        },
        "weights": {
            "sha256": identity_hash(payload["model"]),
            "tensor_count": len(payload["model"]),
            "dtype": str(dtype),
            "device": str(next(model.parameters()).device),
            "exact_restore": True,
        },
        "replay_spec_sha256": replay_spec.get("spec_sha256", identity_hash(replay_spec)),
        "teacher_forced": teacher,
        "generation": generation,
        "agreement": teacher["agrees"] and generation["agrees"],
        "dtype_device_policy": "no dtype override: full and cached replay both use the floating dtype stored in the checkpoint on CPU",
    }
    if expected_report is not None:
        report["reference_replay"] = compare_reference_replay(report, expected_report)
        report["agreement"] = report["agreement"] and report["reference_replay"]["agrees"]
    return report


def write_reconstruction_report(path, report: Mapping) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m native.reconstruct", description=__doc__.split("\n\n")[0])
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--expected-sha256")
    result.add_argument("--spec", required=True, help="fixed token-ID replay JSON")
    result.add_argument("--expected-report", help="optional previous reconstruction report for hash comparison")
    result.add_argument("--output", required=True)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    spec = load_replay_spec(args.spec)
    expected = json.loads(Path(args.expected_report).read_text()) if args.expected_report else None
    report = reconstruct_checkpoint(
        args.checkpoint,
        spec,
        expected_sha256=args.expected_sha256,
        expected_report=expected,
    )
    digest = write_reconstruction_report(args.output, report)
    print(json.dumps({
        "output": args.output,
        "report_sha256": digest,
        "checkpoint_sha256": report["checkpoint"]["sha256"],
        "dtype": report["weights"]["dtype"],
        "teacher_forced_agrees": report["teacher_forced"]["agrees"],
        "generation_agrees": report["generation"]["agrees"],
        "reference_agrees": report.get("reference_replay", {}).get("agrees"),
    }, sort_keys=True))
    return 0 if report["agreement"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
