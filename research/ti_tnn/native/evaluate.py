"""Checkpointed native evaluation without auxiliary-channel leakage.

Teacher forcing scores ``logits[:, :-1]`` against ``labels[:, 1:]`` and uses
``source_ids[:, 1:]`` to keep Wikipedia, code, and structured losses separate.
Free generation receives only tokenized prompts and positions. Gold targets are
used after decoding, never to choose its length or control its trajectory.

The register parser in this module is deliberately independent of the data
producer and executor. Exact lookup is a memorization baseline, while the
supplied-program executor is an oracle baseline with information the neural
model does not receive. Neither is evidence of learned composition.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import cost
from .checkpoint import load_checkpoint, restore_checkpoint
from .config import NativeConfig
from .data import MODEL_INPUT_ARRAYS, SOURCE_IDS, PackedDataset, load_gpt2_tokenizer
from .generation import greedy_generate
from .losses import IGNORE_INDEX, compute_losses
from .model import NativeModel
from .structured import SPLIT_NAMES, encode_with_offsets, execute
INTERVENTION_KINDS = ("zero", "shuffle", "wrong_slot")
INTERVENTION_LIMITS = (
    "The h attention, residual, and direct output paths remain active; an effect is not proof that slots carry the answer.",
    "Unselected slots and typed slot codes remain active, so redundant or distributed information can bypass the intervention.",
    "Zero, shuffled, and wrong-slot states can be off distribution; sensitivity does not establish a faithful or complete explanation.",
    "Shuffle rolls states across the batch, or across time for a single-row teacher-forced input; it is a no-op for one generated token in one row.",
)
_TARGET_PATTERN = re.compile(
    r"r0=(0|[1-9][0-9]*) r1=(0|[1-9][0-9]*) r2=(0|[1-9][0-9]*) r3=(0|[1-9][0-9]*)"
)


def _cpu_device(device) -> torch.device:
    result = torch.device(device)
    if result.type != "cpu":
        raise ValueError("native evaluation and reconstruction are CPU-only")
    return result


def _native_config(values) -> NativeConfig:
    if isinstance(values, NativeConfig):
        return values
    if not isinstance(values, Mapping):
        raise ValueError("checkpoint config must contain a native model mapping")
    model_values = values.get("model", values)
    if not isinstance(model_values, Mapping):
        raise ValueError("checkpoint model config must be a mapping")
    return NativeConfig.from_dict(dict(model_values))


def checkpoint_model_dtype(payload: Mapping) -> torch.dtype:
    dtypes = {
        value.dtype
        for value in payload["model"].values()
        if torch.is_tensor(value) and value.dtype.is_floating_point
    }
    if len(dtypes) != 1:
        raise ValueError(f"native checkpoint must have one floating model dtype, found {sorted(map(str, dtypes))}")
    return next(iter(dtypes))


def _check_restored_weights(model: NativeModel, expected: Mapping) -> None:
    actual = model.state_dict()
    if actual.keys() != expected.keys():
        raise RuntimeError("restored native state-dict keys differ from the checkpoint")
    for name, wanted in expected.items():
        restored = actual[name]
        if restored.dtype != wanted.dtype or not torch.equal(restored.cpu(), wanted.cpu()):
            raise RuntimeError(f"restored weight {name!r} differs from the checkpoint dtype or values")


def load_native_model(
    checkpoint_path,
    *,
    expected_sha256: str | None = None,
    device="cpu",
) -> tuple[NativeModel, dict]:
    """Validate checkpoint bytes, infer its dtype, and restore exact CPU weights."""
    device = _cpu_device(device)
    payload = load_checkpoint(checkpoint_path, expected_sha256=expected_sha256)
    config = _native_config(payload["config"])
    dtype = checkpoint_model_dtype(payload)
    model = NativeModel(config).to(device=device, dtype=dtype)
    restore_checkpoint(
        payload,
        model,
        restore_rng=False,
        expected_config=payload["config"],
    )
    model.eval()
    _check_restored_weights(model, payload["model"])
    return model, payload


class RoutingAccumulator:
    """Streaming route selection, entropy, starvation, load, and per-token union statistics."""

    def __init__(self, n_passes: int = 4, n_experts: int = 4) -> None:
        self.n_passes = n_passes
        self.n_experts = n_experts
        self.selection_counts = torch.zeros(n_passes, n_experts, dtype=torch.long)
        self.probability_sums = torch.zeros(n_passes, n_experts, dtype=torch.float64)
        self.entropy_sums = torch.zeros(n_passes, dtype=torch.float64)
        self.tokens = torch.zeros(n_passes, dtype=torch.long)
        self.distinct_histogram = torch.zeros(n_experts + 1, dtype=torch.long)
        self.applicable = False

    @torch.no_grad()
    def update(self, output, valid_tokens: torch.Tensor) -> None:
        if output.route_indices is None:
            return
        if output.route_probs is None:
            raise ValueError("routing diagnostics require return_aux=True route probabilities")
        indices = output.route_indices.detach().cpu()
        probabilities = output.route_probs.detach().float().cpu()
        valid = valid_tokens.detach().cpu().bool()
        if valid.shape != indices.shape[:2] or probabilities.shape != (*indices.shape, self.n_experts):
            raise ValueError("route outputs and valid-token mask have inconsistent shapes")
        if indices.shape[2] != self.n_passes:
            raise ValueError("route outputs have an unexpected pass dimension")
        self.applicable = True
        membership = F.one_hot(indices.long(), num_classes=self.n_experts).sum(dim=2).gt(0)
        distinct = membership.sum(dim=-1)
        self.distinct_histogram += torch.bincount(
            distinct[valid], minlength=self.n_experts + 1
        )
        for pass_index in range(self.n_passes):
            selected = indices[:, :, pass_index][valid]
            probs = probabilities[:, :, pass_index][valid]
            count = selected.numel()
            self.tokens[pass_index] += count
            if not count:
                continue
            self.selection_counts[pass_index] += torch.bincount(
                selected.long(), minlength=self.n_experts
            )
            self.probability_sums[pass_index] += probs.double().sum(dim=0)
            entropy = -(probs.double().clamp_min(torch.finfo(torch.float64).tiny).log() * probs.double()).sum(dim=-1)
            self.entropy_sums[pass_index] += entropy.sum()

    def _distinct_report(self) -> dict:
        count = int(self.distinct_histogram.sum())
        weighted = sum(
            distinct * int(self.distinct_histogram[distinct])
            for distinct in range(1, self.n_experts + 1)
        )
        used = [
            distinct for distinct in range(1, self.n_experts + 1)
            if self.distinct_histogram[distinct]
        ]
        return {
            "tokens": count,
            "histogram": {
                str(distinct): int(self.distinct_histogram[distinct])
                for distinct in range(1, self.n_experts + 1)
            },
            "fraction": {
                str(distinct): int(self.distinct_histogram[distinct]) / count if count else None
                for distinct in range(1, self.n_experts + 1)
            },
            "mean": weighted / count if count else None,
            "max": max(used) if used else None,
            "definition": "number of different hard-selected experts across the four passes of each live token",
        }

    def dataset_load_balance(self) -> float | None:
        decisions = int(self.selection_counts.sum())
        if not self.applicable or not decisions:
            return None
        counts = self.selection_counts.sum(dim=0)
        fractions = counts.double() / decisions
        means = self.probability_sums.sum(dim=0) / decisions
        return float(self.n_experts * (fractions * means).sum())

    def report(self, config: NativeConfig | None = None) -> dict:
        if not self.applicable:
            return {
                "status": "not_applicable",
                "reason": "dense and shared modes have no expert router",
            }
        per_pass = {}
        starved_cells = 0
        for pass_index in range(self.n_passes):
            count = int(self.tokens[pass_index])
            counts = self.selection_counts[pass_index]
            fractions = counts.double() / max(count, 1)
            means = self.probability_sums[pass_index] / max(count, 1)
            starved = [index for index, value in enumerate(counts.tolist()) if value == 0]
            starved_cells += len(starved)
            entropy = float(self.entropy_sums[pass_index] / max(count, 1))
            per_pass[str(pass_index)] = {
                "decisions_per_expert_bank": count,
                "selected_counts": {str(index): int(value) for index, value in enumerate(counts.tolist())},
                "selected_fractions": {str(index): float(value) for index, value in enumerate(fractions)},
                "mean_probabilities": {str(index): float(value) for index, value in enumerate(means)},
                "mean_entropy_nats": entropy,
                "mean_normalized_entropy": entropy / math.log(self.n_experts),
                "starved_experts": starved,
                "minimum_selected_fraction": float(fractions.min()),
                "maximum_selected_fraction": float(fractions.max()),
                "selected_fraction_std": float(fractions.std(unbiased=False)),
                "load_balance_4sum_fP": float(self.n_experts * (fractions * means).sum()),
            }
        cells = self.n_passes * self.n_experts
        distinct = self._distinct_report()
        result = {
            "status": "ok",
            "routing_decisions": int(self.selection_counts.sum()),
            "per_pass": per_pass,
            "starvation": {
                "pass_expert_cells": cells,
                "zero_selection_cells": starved_cells,
                "zero_selection_fraction": starved_cells / cells,
            },
            "distinct_experts_per_token": distinct,
            "dataset_load_balance_4sum_fP": self.dataset_load_balance(),
            "entropy_base": "natural logarithm; normalized entropy divides by log(4)",
            "load_definition": "dataset-wide and per-pass 4 * sum_e selected_fraction_e * mean_probability_e over live routes; these aggregations need not equal a mean of batch or pass balances",
        }
        if config is not None:
            result["active_parameter_estimates"] = _active_parameter_estimates(
                config, self.selection_counts, self.distinct_histogram
            )
        return result


def _union_active_parameters(config: NativeConfig, expert_count: int) -> tuple[int, str]:
    """Use the cost owner's route-union helper when present, with a formula fallback."""
    if not 1 <= expert_count <= 4:
        raise ValueError("expert_count must be in [1, 4]")
    helper = getattr(cost, "route_active_parameters", None)
    if callable(helper):
        representative_routes = torch.zeros(4, dtype=torch.long)
        representative_routes[:expert_count] = torch.arange(expert_count)
        return int(helper(config, representative_routes)), "cost.route_active_parameters"
    nominal = cost.parameter_counts(config)
    per_expert = cost.dense_transform_parameters(config)
    non_expert = nominal["active"] - per_expert
    return non_expert + expert_count * per_expert, "cost.parameter_counts + cost.dense_transform_parameters"


def _active_parameter_estimates(
    config: NativeConfig,
    selection_counts: torch.Tensor,
    distinct_histogram: torch.Tensor,
) -> dict:
    nominal = cost.parameter_counts(config)
    per_expert = cost.dense_transform_parameters(config)
    non_expert = nominal["active"] - per_expert
    used_experts = torch.nonzero(selection_counts.sum(dim=0) > 0, as_tuple=False).flatten().tolist()
    token_count = int(distinct_histogram.sum())
    per_token_values = []
    helper_names = set()
    for distinct in range(1, len(distinct_histogram)):
        count = int(distinct_histogram[distinct])
        if not count:
            continue
        estimate, helper_name = _union_active_parameters(config, distinct)
        helper_names.add(helper_name)
        per_token_values.append((distinct, count, estimate))
    if used_experts:
        evaluation_estimate, helper_name = _union_active_parameters(config, len(used_experts))
        helper_names.add(helper_name)
    else:
        evaluation_estimate = None
    mean_estimate = (
        sum(count * estimate for _, count, estimate in per_token_values) / token_count
        if token_count else None
    )
    return {
        "kind": "route-dependent estimates of unique stored parameters touched by observed hard-route unions",
        "nominal_one_expert_active_parameters": nominal["active"],
        "nominal_one_expert_is_observed_exact_cost": False,
        "non_expert_parameters": non_expert,
        "parameters_per_expert": per_expert,
        "evaluation_wide_route_union": {
            "experts": used_experts,
            "distinct_experts": len(used_experts),
            "active_parameters": evaluation_estimate,
        },
        "per_token_route_union": {
            "mean_active_parameters": mean_estimate,
            "minimum_active_parameters": min((estimate for _, _, estimate in per_token_values), default=None),
            "maximum_active_parameters": max((estimate for _, _, estimate in per_token_values), default=None),
        },
        "cost_helper": sorted(helper_names),
        "boundary": "estimates count unique stored parameters touched by routes, not exact MACs, memory traffic, latency, or dynamically loaded parameters",
    }


def _dataset_length(dataset) -> int:
    if isinstance(dataset, Mapping):
        return len(dataset["input_ids"])
    return len(dataset)


def _mapping_batch(dataset: Mapping, indices: Sequence[int], device) -> dict[str, torch.Tensor]:
    result = {}
    index = torch.as_tensor(indices, dtype=torch.long)
    for name, values in dataset.items():
        selected = values[index] if torch.is_tensor(values) else np.ascontiguousarray(values[list(indices)])
        tensor = selected if torch.is_tensor(selected) else torch.from_numpy(selected)
        if name in {"input_ids", "labels", "positions", "doc_ids", "route_targets", "state_targets", "source_ids"}:
            tensor = tensor.long()
        result[name] = tensor.to(device)
    return result


def _dataset_batch(dataset, indices: Sequence[int], device) -> dict[str, torch.Tensor]:
    if isinstance(dataset, Mapping):
        return _mapping_batch(dataset, indices, device)
    return dataset.batch(indices, as_torch=True, device=device)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@torch.no_grad()
def _compute_eval_losses(output, batch: Mapping[str, torch.Tensor]) -> tuple[dict, str]:
    parameters = inspect.signature(compute_losses).parameters
    accepts_mask = "key_padding_mask" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    routed = output.route_logits is not None
    kwargs = {
        "route_logits": output.route_logits,
        "route_indices": output.route_indices,
        "route_targets": batch.get("route_targets") if routed else None,
        "route_mask": batch.get("route_mask") if routed else None,
        "state_logits": output.state_logits,
        "state_targets": batch.get("state_targets"),
        "state_mask": batch.get("state_mask"),
    }
    if accepts_mask:
        return compute_losses(
            output.logits,
            batch["labels"],
            key_padding_mask=batch["key_padding_mask"],
            **kwargs,
        ), "key_padding_mask"
    if routed:
        live = ~batch["key_padding_mask"]
        kwargs["route_logits"] = output.route_logits[live].unsqueeze(1)
        kwargs["route_indices"] = output.route_indices[live].unsqueeze(1)
        if batch.get("route_targets") is not None:
            kwargs["route_targets"] = batch["route_targets"][live].unsqueeze(1)
            kwargs["route_mask"] = batch["route_mask"][live].unsqueeze(1)
    return compute_losses(output.logits, batch["labels"], **kwargs), "legacy_live_route_compaction"


@torch.no_grad()
def teacher_forced_evaluate(
    model: NativeModel,
    dataset,
    *,
    batch_size: int = 8,
    source_id_map: Mapping[str, int] = SOURCE_IDS,
    max_batches: int | None = None,
    device="cpu",
) -> dict:
    """Evaluate micro-averaged next-token loss independently for each packed source.

    ``source_ids`` is an external scoring array, never a model input. If it is
    absent this function fails rather than guessing source identity from labels.
    """
    device = _cpu_device(device)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = _dataset_length(dataset)
    total_batches = (rows + batch_size - 1) // batch_size
    batches = total_batches if max_batches is None else min(max_batches, total_batches)
    if batches < 0:
        raise ValueError("max_batches must be nonnegative")
    sums = {name: 0.0 for name in source_id_map}
    counts = {name: 0 for name in source_id_map}
    correct = {name: 0 for name in source_id_map}
    aux_sums = {"route": 0.0, "state": 0.0, "balance": 0.0}
    aux_counts = {"route": 0, "state": 0, "balance": 0}
    balance_mask_api = None
    routing = RoutingAccumulator()
    model.eval()
    for batch_index in range(batches):
        indices = list(range(batch_index * batch_size, min((batch_index + 1) * batch_size, rows)))
        batch = _dataset_batch(dataset, indices, device)
        if "source_ids" not in batch:
            raise ValueError(
                "packed arrays lack source_ids; Wikipedia/code/structured evaluation cannot be separated"
            )
        model_inputs = {name: batch[name] for name in MODEL_INPUT_ARRAYS}
        output = model(**model_inputs, return_aux=True)
        shifted_logits = output.logits[:, :-1]
        shifted_labels = batch["labels"][:, 1:]
        shifted_sources = batch["source_ids"][:, 1:]
        if shifted_labels.shape != shifted_sources.shape or shifted_logits.shape[:-1] != shifted_labels.shape:
            raise ValueError("labels, source_ids, and logits have inconsistent shifted shapes")
        supervised = shifted_labels.ne(IGNORE_INDEX)
        known_sources = torch.zeros_like(supervised)
        for name, source_id in source_id_map.items():
            mask = supervised & shifted_sources.eq(source_id)
            known_sources |= mask
            count = int(mask.sum())
            counts[name] += count
            if not count:
                continue
            selected_logits = shifted_logits[mask]
            selected_labels = shifted_labels[mask].long()
            sums[name] += float(F.cross_entropy(selected_logits.float(), selected_labels, reduction="sum"))
            correct[name] += int((selected_logits.argmax(dim=-1) == selected_labels).sum())
        unknown = supervised & ~known_sources
        if unknown.any():
            values = sorted(set(shifted_sources[unknown].cpu().tolist()))
            raise ValueError(f"supervised tokens have unknown source_ids {values}")
        routing.update(output, ~batch["key_padding_mask"])
        eval_losses, balance_mask_api = _compute_eval_losses(output, batch)
        for name in ("route", "state"):
            count = int(eval_losses[f"{name}_count"])
            aux_sums[name] += float(eval_losses[name]) * count
            aux_counts[name] += count
        live_route_decisions = int(eval_losses.get(
            "balance_count",
            (~batch["key_padding_mask"]).sum() * 4 if output.route_logits is not None else 0,
        ))
        aux_sums["balance"] += float(eval_losses["balance"]) * live_route_decisions
        aux_counts["balance"] += live_route_decisions
    losses = {
        name: {
            "loss": sums[name] / counts[name] if counts[name] else None,
            "perplexity": math.exp(sums[name] / counts[name]) if counts[name] and sums[name] / counts[name] < 709 else None,
            "loss_tokens": counts[name],
            "correct_next_tokens": correct[name],
            "next_token_accuracy": _rate(correct[name], counts[name]),
            "cross_entropy_sum": sums[name],
        }
        for name in source_id_map
    }
    total_count = sum(counts.values())
    total_sum = sum(sums.values())
    return {
        "evaluation": "checkpointed teacher forcing",
        "shift": "logits[:, :-1] -> labels[:, 1:], with source_ids[:, 1:] selecting the target source",
        "sources": losses,
        "overall": {
            "loss": total_sum / total_count if total_count else None,
            "loss_tokens": total_count,
            "cross_entropy_sum": total_sum,
        },
        "auxiliary_losses": {
            name: {
                "loss": (
                    routing.dataset_load_balance() if name == "balance"
                    else aux_sums[name] / aux_counts[name] if aux_counts[name] else None
                ),
                **({
                    "routing_decisions": aux_counts[name],
                    "key_padding_mask_applied": True,
                    "compute_losses_mask_api": balance_mask_api,
                    "aggregation": "dataset-wide live-route fractions and mean probabilities",
                    "per_batch_decision_weighted_mean": (
                        aux_sums[name] / aux_counts[name] if aux_counts[name] else None
                    ),
                } if name == "balance" else {
                    "supervised_values": aux_counts[name],
                }),
            }
            for name in aux_sums
        },
        "routing": routing.report(model.config),
        "sequences": min(rows, batches * batch_size),
        "batches": batches,
        "complete": batches == total_batches,
        "evidence_boundary": "teacher-forced pipeline measurements are not free-generation accuracy or evidence of learned composition",
    }


def parse_final_registers(text: str, *, modulus: int = 32) -> tuple[int, int, int, int] | None:
    """Parse only the complete canonical ``r0=.. r1=.. r2=.. r3=..`` target.

    Extra text, surrounding whitespace, reordered registers, signs, leading
    zeroes, missing registers, and values outside the machine modulus fail.
    """
    if not isinstance(text, str) or modulus <= 0:
        return None
    match = _TARGET_PATTERN.fullmatch(text)
    if match is None:
        return None
    values = tuple(int(value) for value in match.groups())
    return values if all(value < modulus for value in values) else None


def _prompt_ids(tokenizer, prompt: str) -> list[int]:
    ids, _ = encode_with_offsets(tokenizer, prompt)
    if not ids:
        raise ValueError("generation prompt tokenized to an empty sequence")
    return [int(value) for value in ids]


def _decode(tokenizer, ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(list(ids), skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(list(ids))


def _first_text_divergence(actual: str, expected: str) -> dict | None:
    for index, (left, right) in enumerate(zip(actual, expected)):
        if left != right:
            return {"character": index, "actual": left, "expected": right}
    if len(actual) != len(expected):
        index = min(len(actual), len(expected))
        return {
            "character": index,
            "actual": actual[index] if index < len(actual) else None,
            "expected": expected[index] if index < len(expected) else None,
        }
    return None


def _first_token_divergence(actual: Sequence[int], expected: Sequence[int]) -> dict | None:
    for index, (left, right) in enumerate(zip(actual, expected)):
        if left != right:
            return {"generated_token": index, "actual": int(left), "expected": int(right)}
    if len(actual) != len(expected):
        index = min(len(actual), len(expected))
        return {
            "generated_token": index,
            "actual": int(actual[index]) if index < len(actual) else None,
            "expected": int(expected[index]) if index < len(expected) else None,
        }
    return None


@torch.no_grad()
def generate_prompt(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    use_cache: bool = True,
    prompt_policy: str = "error",
) -> dict:
    """Greedily decode a bounded suffix using no gold or auxiliary inputs."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if prompt_policy not in {"error", "keep_last"}:
        raise ValueError("prompt_policy must be 'error' or 'keep_last'")
    prompt_ids = _prompt_ids(tokenizer, prompt)
    original_prompt_tokens = len(prompt_ids)
    context_length = model.config.context_length
    if len(prompt_ids) + max_new_tokens > context_length:
        if prompt_policy == "error":
            raise ValueError(
                f"prompt ({len(prompt_ids)}) plus generation bound ({max_new_tokens}) exceeds context_length {context_length}"
            )
        keep = context_length - max_new_tokens
        if keep <= 0:
            raise ValueError("max_new_tokens leaves no room for a prompt in context_length")
        prompt_ids = prompt_ids[-keep:]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    positions = torch.arange(len(prompt_ids), dtype=torch.long).unsqueeze(0)
    sequence = greedy_generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        positions=positions,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        use_cache=use_cache,
    )
    generated = sequence[0, len(prompt_ids):].cpu().tolist()
    completion = generated[: generated.index(eos_token_id)] if eos_token_id in generated else generated
    stopped = eos_token_id is not None and eos_token_id in generated
    return {
        "prompt_ids": prompt_ids,
        "positions": list(range(len(prompt_ids))),
        "original_prompt_tokens": original_prompt_tokens,
        "prompt_truncated": len(prompt_ids) != original_prompt_tokens,
        "prompt_policy": prompt_policy,
        "sequence_ids": sequence[0].cpu().tolist(),
        "generated_ids": generated,
        "completion_ids": completion,
        "completion_text": _decode(tokenizer, completion),
        "stopped_on_eos": stopped,
        "truncated": not stopped and len(generated) == max_new_tokens,
        "max_new_tokens": max_new_tokens,
        "use_cache": use_cache,
    }


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}: blank JSONL line {line_number}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            for field in ("example_id", "prompt", "target_text"):
                if not isinstance(record.get(field), str):
                    raise ValueError(f"{path}: line {line_number} needs string {field}")
            records.append(record)
    return records


def load_structured_records(directory) -> tuple[dict[str, list[dict]], dict]:
    directory = Path(directory)
    records = {}
    files = {}
    for split in SPLIT_NAMES:
        path = directory / f"{split}.jsonl"
        split_records = _load_jsonl(path)
        for record in split_records:
            if record.get("split") != split:
                raise ValueError(f"{path}: record split does not match its filename")
            if parse_final_registers(record["target_text"]) is None:
                raise ValueError(f"{path}: noncanonical target for {record['example_id']}")
        records[split] = split_records
        files[split] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "examples": len(split_records),
        }
    return records, files


def _instruction_length(record: Mapping) -> int:
    program = record.get("program")
    if not isinstance(program, (list, tuple)) or not program:
        raise ValueError(f"{record.get('example_id')}: structured scoring requires a nonempty saved program")
    return len(program)


def _generation_summary(
    results: Sequence[Mapping],
    examples: int,
    strict_registers: bool,
) -> dict:
    evaluated = len(results)
    correct = sum(record["correct"] for record in results)
    stopped = sum(record["stopped_on_eos"] for record in results)
    truncated = sum(record["truncated"] for record in results)
    prompt_truncated = sum(record["prompt_truncated"] for record in results)
    valid = sum(record["valid"] is True for record in results)
    return {
        "status": "complete" if evaluated == examples else "partial",
        "complete": evaluated == examples,
        "examples": examples,
        "evaluated": evaluated,
        "correct": correct,
        "exact_match": _rate(correct, evaluated),
        "valid": valid if strict_registers else None,
        "valid_rate": _rate(valid, evaluated) if strict_registers else None,
        "stopped_on_eos": stopped,
        "eos_rate": _rate(stopped, evaluated),
        "truncated": truncated,
        "truncation_rate": _rate(truncated, evaluated),
        "prompt_truncated": prompt_truncated,
        "prompt_truncation_rate": _rate(prompt_truncated, evaluated),
    }


def _generation_metrics(
    model,
    tokenizer,
    records: Sequence[Mapping],
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    strict_registers: bool,
    use_cache: bool,
    limit: int | None,
    prompt_policy: str = "error",
) -> dict:
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError("generation limit must be a nonnegative integer or None")
    selected = list(records if limit is None else records[:limit])
    results = []
    for record in selected:
        generated = generate_prompt(
            model,
            tokenizer,
            record["prompt"],
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            use_cache=use_cache,
            prompt_policy=prompt_policy,
        )
        prediction = generated["completion_text"]
        target = record["target_text"]
        parsed_prediction = parse_final_registers(prediction) if strict_registers else None
        parsed_target = parse_final_registers(target) if strict_registers else None
        if strict_registers and parsed_target is None:
            raise ValueError(f"{record['example_id']}: gold target is not canonical final-register text")
        correct = (
            parsed_prediction is not None and parsed_prediction == parsed_target and prediction == target
            if strict_registers
            else prediction == target
        )
        results.append({
            "example_id": record["example_id"],
            "completion_text": prediction,
            "target_text": target,
            "valid": parsed_prediction is not None if strict_registers else None,
            "correct": correct,
            "stopped_on_eos": generated["stopped_on_eos"],
            "truncated": generated["truncated"],
            "original_prompt_tokens": generated["original_prompt_tokens"],
            "prompt_truncated": generated["prompt_truncated"],
            "completion_tokens": len(generated["completion_ids"]),
            "first_divergence": _first_text_divergence(prediction, target),
            **({"instruction_length": _instruction_length(record)} if strict_registers else {}),
        })
    summary = _generation_summary(results, len(records), strict_registers)
    if strict_registers:
        summary["by_instruction_length"] = {
            str(length): _generation_summary(
                [record for record in results if record["instruction_length"] == length],
                sum(_instruction_length(record) == length for record in records),
                True,
            )
            for length in sorted({_instruction_length(record) for record in records})
        }
    summary.update({
        "prompt_policy": prompt_policy,
        "max_new_tokens": max_new_tokens,
        "target_length_used_for_decoding": False,
        "records": results,
    })
    return summary


def _composition_comparison(
    records: Sequence[Mapping],
    generation: Mapping,
) -> dict:
    def group(lengths: set[int]) -> dict:
        full = [record for record in records if _instruction_length(record) in lengths]
        scored = [
            record for record in generation["records"]
            if record["instruction_length"] in lengths
        ]
        return _generation_summary(scored, len(full), True)

    primary = group({3, 4})
    primary.update({
        "instruction_lengths": [3, 4],
        "role": "primary_composition_only_comparison",
    })
    secondary = group({5})
    secondary.update({
        "instruction_lengths": [5],
        "role": "secondary_composition_plus_length",
        "confound": "length 5 exceeds the ordinary train/dev length 2-4 range while also requiring the unseen composition",
    })
    return {
        "composition_only_lengths_3_4": primary,
        "secondary_length_5": secondary,
        "ordinary_train_dev_declared_length_range": [2, 4],
        "comparison_rule": "primary unseen-composition scoring uses actual instruction lengths 3-4; length 5 is secondary because it mixes composition and length generalization",
    }


def _study_metric(values: Mapping) -> dict:
    return {
        key: values.get(key)
        for key in ("complete", "examples", "evaluated", "correct", "exact_match", "valid_rate")
    }


def evaluate_structured_generation(
    model,
    tokenizer,
    structured_directory,
    *,
    max_new_tokens: int = 48,
    eos_token_id: int | None = 50256,
    use_cache: bool = True,
    limit_per_split: int | None = None,
) -> dict:
    records, files = load_structured_records(structured_directory)
    splits = {
        split: _generation_metrics(
            model,
            tokenizer,
            split_records,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            strict_registers=True,
            use_cache=use_cache,
            limit=limit_per_split,
        )
        for split, split_records in records.items()
    }
    comparison = _composition_comparison(records["unseen_composition"], splits["unseen_composition"])
    return {
        "metric": "complete canonical final-register target text, checked by native.evaluate.parse_final_registers",
        "model_inputs": ["tokenized prompt", "positions"],
        "forbidden_generation_channels": ["program", "execution trace", "route targets", "state targets", "target text"],
        "instruction_lengths_are_scoring_only": True,
        "files": files,
        "splits": splits,
        "composition_comparison": comparison,
        "study_visible_summary": {
            "ordinary_dev": _study_metric(splits["dev"]),
            "unseen_composition_overall": _study_metric(splits["unseen_composition"]),
            "composition_only_lengths_3_4": comparison["composition_only_lengths_3_4"],
            "secondary_length_5_composition_plus_length": comparison["secondary_length_5"],
            "longer_length": _study_metric(splits["longer_length"]),
            "comparison_rule": comparison["comparison_rule"],
        },
        "evidence_boundary": "tiny or overfit exact matches are pipeline checks; composition claims require held-out learning evidence",
    }


def _baseline_metrics(predictions: Sequence[str | None], records: Sequence[Mapping]) -> dict:
    correct = sum(prediction == record["target_text"] for prediction, record in zip(predictions, records))
    missing = sum(prediction is None for prediction in predictions)
    return {
        "examples": len(records),
        "correct": correct,
        "missing": missing,
        "exact_match": _rate(correct, len(records)),
    }


def structured_baseline_report(records: Mapping[str, Sequence[Mapping]]) -> dict:
    lookup = {}
    for record in records["train"]:
        previous = lookup.setdefault(record["prompt"], record["target_text"])
        if previous != record["target_text"]:
            raise ValueError("conflicting targets for one exact-lookup prompt")
    lookup_predictions = {
        split: [lookup.get(record["prompt"]) for record in split_records]
        for split, split_records in records.items()
    }
    executor_predictions = {
        split: [execute(record["program"], record["initial"]).target_text for record in split_records]
        for split, split_records in records.items()
    }
    return {
        "metric": "complete target-text exact match",
        "exact_lookup": {
            split: _baseline_metrics(predictions, records[split])
            for split, predictions in lookup_predictions.items()
        },
        "supplied_program_executor": {
            split: _baseline_metrics(predictions, records[split])
            for split, predictions in executor_predictions.items()
        },
        "boundary": "lookup is memorization, not learned composition; the executor receives reference programs and is not the neural model",
    }


def evaluate_structured_with_baselines(
    model,
    tokenizer,
    structured_directory,
    **generation_kwargs,
) -> dict:
    records, _ = load_structured_records(structured_directory)
    result = evaluate_structured_generation(model, tokenizer, structured_directory, **generation_kwargs)
    result["baselines"] = structured_baseline_report(records)
    return result


def evaluate_code_generation(
    model,
    tokenizer,
    path,
    *,
    max_new_tokens: int = 128,
    eos_token_id: int | None = 50256,
    use_cache: bool = True,
    limit: int | None = None,
) -> dict:
    path = Path(path)
    if not path.is_file():
        return {
            "status": "not_run",
            "reason": "code_generation.jsonl is absent",
            "path": str(path),
            "fabricated_results": False,
        }
    records = _load_jsonl(path)
    result = _generation_metrics(
        model,
        tokenizer,
        records,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        strict_registers=False,
        use_cache=use_cache,
        limit=limit,
        prompt_policy="keep_last",
    )
    return {
        **result,
        "status": "complete" if result["complete"] else "partial",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "metric": "bounded free-generated code-suffix exact text match",
        "separate_from_structured_metrics": True,
    }


@dataclass(frozen=True)
class InterventionSpec:
    """Raw slot-state control before a zero-indexed pass read and write."""

    pass_index: int
    slot_index: int
    kind: str
    source_slot: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.pass_index < 4 or not 0 <= self.slot_index < 4:
            raise ValueError("pass_index and slot_index must be in [0, 4)")
        if self.kind not in INTERVENTION_KINDS:
            raise ValueError(f"kind must be one of {INTERVENTION_KINDS}")
        if self.kind == "wrong_slot":
            if self.source_slot is None or not 0 <= self.source_slot < 4 or self.source_slot == self.slot_index:
                raise ValueError("wrong_slot requires a different source_slot in [0, 4)")
        elif self.source_slot is not None:
            raise ValueError("source_slot is only valid for wrong_slot")

    def to_dict(self) -> dict:
        return {
            "pass_index": self.pass_index,
            "slot_index": self.slot_index,
            "kind": self.kind,
            "source_slot": self.source_slot,
        }


def _replace_state(slots: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
    modified = slots.clone()
    if spec.kind == "zero":
        modified[:, :, spec.slot_index, :] = 0
    elif spec.kind == "wrong_slot":
        modified[:, :, spec.slot_index, :] = slots[:, :, spec.source_slot, :]
    else:
        values = slots[:, :, spec.slot_index, :]
        dimension = 0 if values.shape[0] > 1 else 1
        modified[:, :, spec.slot_index, :] = torch.roll(values, shifts=1, dims=dimension)
    return modified


@contextmanager
def structured_state_intervention(model: NativeModel, spec: InterventionSpec):
    """Apply a control to raw slots before the declared pass, then restore methods."""
    original_read = model._read_slots
    original_write = model._write_slots
    missing = object()
    previous_read = model.__dict__.get("_read_slots", missing)
    previous_write = model.__dict__.get("_write_slots", missing)
    state = {"next_pass": 0, "replacement": None}

    def read(slots):
        if state["next_pass"] == 4:
            state["next_pass"] = 0
        replacement = (
            _replace_state(slots, spec)
            if state["next_pass"] == spec.pass_index
            else slots
        )
        state["replacement"] = replacement
        return original_read(replacement)

    def write(slots, transformed, projected):
        replacement = state["replacement"]
        result = original_write(replacement, transformed, projected)
        state["next_pass"] += 1
        state["replacement"] = None
        return result

    model._read_slots = read
    model._write_slots = write
    try:
        yield
    finally:
        if previous_read is missing:
            del model._read_slots
        else:
            model._read_slots = previous_read
        if previous_write is missing:
            del model._write_slots
        else:
            model._write_slots = previous_write


def _logit_effect(actual: torch.Tensor, expected: torch.Tensor, valid: torch.Tensor | None) -> dict:
    difference = actual.float() - expected.float()
    if valid is not None:
        difference = difference[valid]
        actual = actual[valid]
        expected = expected[valid]
    unequal = difference.ne(0)
    first = None
    if unequal.any():
        flat_index = int(unequal.flatten().to(torch.int8).argmax())
        index = tuple(int(value) for value in np.unravel_index(flat_index, unequal.shape))
        first = {
            "index": list(index),
            "actual": float(actual[index]),
            "expected": float(expected[index]),
            "absolute_difference": float(difference[index].abs()),
        }
    return {
        "mean_absolute": float(difference.abs().mean()),
        "maximum_absolute": float(difference.abs().max()),
        "l2": float(difference.square().sum().sqrt()),
        "changed_values": int(unequal.sum()),
        "changed_value_fraction": float(unequal.float().mean()),
        "first_divergence": first,
    }


@torch.no_grad()
def evaluate_state_interventions(
    model,
    logit_inputs: Mapping[str, torch.Tensor],
    prompts: Sequence[Mapping],
    specifications: Sequence[InterventionSpec],
    tokenizer,
    *,
    max_new_tokens: int = 48,
    eos_token_id: int | None = 50256,
    use_cache: bool = True,
) -> dict:
    """Measure logit and prompt-only generation effects for declared controls."""
    model_inputs = {name: logit_inputs[name] for name in MODEL_INPUT_ARRAYS}
    baseline_logits = model(**model_inputs).logits
    valid = ~model_inputs["key_padding_mask"] if model_inputs.get("key_padding_mask") is not None else None
    baseline_generations = [
        generate_prompt(
            model,
            tokenizer,
            record["prompt"],
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            use_cache=use_cache,
        )
        for record in prompts
    ]
    interventions = []
    for spec in specifications:
        with structured_state_intervention(model, spec):
            intervened_logits = model(**model_inputs).logits
            generations = [
                generate_prompt(
                    model,
                    tokenizer,
                    record["prompt"],
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_token_id,
                    use_cache=use_cache,
                )
                for record in prompts
            ]
        records = []
        for prompt, baseline, intervened in zip(prompts, baseline_generations, generations):
            changed = baseline["sequence_ids"] != intervened["sequence_ids"]
            records.append({
                "example_id": prompt.get("example_id"),
                "changed": changed,
                "first_divergence": _first_token_divergence(
                    intervened["sequence_ids"][len(baseline["prompt_ids"]):],
                    baseline["sequence_ids"][len(baseline["prompt_ids"]):],
                ) if changed else None,
                "baseline_completion": baseline["completion_text"],
                "intervened_completion": intervened["completion_text"],
            })
        interventions.append({
            **spec.to_dict(),
            "logit_effect": _logit_effect(intervened_logits, baseline_logits, valid),
            "generation_effect": {
                "examples": len(records),
                "changed_examples": sum(record["changed"] for record in records),
                "changed_fraction": _rate(sum(record["changed"] for record in records), len(records)),
                "records": records,
            },
        })
    return {
        "coordinate_system": "zero-indexed pass and slot; control replaces raw slot input before that pass read and write",
        "interventions": interventions,
        "limits": list(INTERVENTION_LIMITS),
        "evidence_boundary": "intervention sensitivity is separate from probe accuracy, route traces, task accuracy, and learned composition",
    }


def _write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m native.evaluate", description=__doc__.split("\n\n")[0])
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--expected-sha256")
    result.add_argument("--packed-dev", help="prepared dev directory containing packed NPY arrays")
    result.add_argument("--structured", help="directory containing the four saved structured JSONL splits")
    result.add_argument("--tokenizer", help="hash-pinned tokenizer.json; required with --structured or --code-generation")
    result.add_argument("--code-generation", help="optional code_generation.jsonl; absence is reported as not_run")
    result.add_argument("--output", required=True)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--max-batches", type=int)
    result.add_argument("--max-new-tokens", type=int, default=48)
    result.add_argument("--code-max-new-tokens", type=int, default=128)
    result.add_argument("--eos-token-id", type=int, default=50256)
    result.add_argument("--limit-per-split", type=int, help="explicit partial smoke check; never presented as complete")
    result.add_argument("--code-limit", type=int, help="independent first-N code-generation sample; overrides --limit-per-split")
    result.add_argument("--no-cache", action="store_true")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    model, checkpoint = load_native_model(args.checkpoint, expected_sha256=args.expected_sha256)
    report = {
        "format": "native-evaluation-v1",
        "checkpoint": {
            "path": args.checkpoint,
            "sha256": checkpoint["sha256"],
            "step": checkpoint["step"],
            "config": checkpoint["config"],
            "dtype": str(checkpoint_model_dtype(checkpoint)),
            "device": "cpu",
        },
    }
    if args.packed_dev:
        report["teacher_forced"] = teacher_forced_evaluate(
            model,
            PackedDataset(args.packed_dev),
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )
    else:
        report["teacher_forced"] = {"status": "not_run", "reason": "--packed-dev was not supplied"}
    tokenizer = load_gpt2_tokenizer(args.tokenizer) if args.tokenizer else None
    if args.structured:
        if tokenizer is None:
            raise ValueError("--tokenizer is required for structured free generation")
        report["structured_generation"] = evaluate_structured_with_baselines(
            model,
            tokenizer,
            args.structured,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=args.eos_token_id,
            use_cache=not args.no_cache,
            limit_per_split=args.limit_per_split,
        )
    else:
        report["structured_generation"] = {"status": "not_run", "reason": "--structured was not supplied"}
    code_generation = args.code_generation
    if code_generation is None and args.structured:
        code_generation = str(Path(args.structured).parent / "code_generation.jsonl")
    elif code_generation is None and args.packed_dev:
        code_generation = str(Path(args.packed_dev).parent / "code_generation.jsonl")
    if code_generation is not None:
        if Path(code_generation).is_file() and tokenizer is None:
            raise ValueError("--tokenizer is required when code_generation.jsonl is present")
        report["code_generation"] = evaluate_code_generation(
            model,
            tokenizer,
            code_generation,
            max_new_tokens=args.code_max_new_tokens,
            eos_token_id=args.eos_token_id,
            use_cache=not args.no_cache,
            limit=args.code_limit if args.code_limit is not None else args.limit_per_split,
        )
    else:
        report["code_generation"] = {
            "status": "not_run",
            "reason": "code_generation.jsonl was not supplied or discoverable beside the evaluation data",
            "fabricated_results": False,
        }
    _write_json(args.output, report)
    print(json.dumps({
        "output": args.output,
        "teacher_forced": ("complete" if report["teacher_forced"]["complete"] else "partial")
        if "sources" in report["teacher_forced"] else report["teacher_forced"]["status"],
        "structured_generation": report["structured_generation"].get(
            "study_visible_summary", report["structured_generation"]["status"]
        ),
        "code_generation": report["code_generation"]["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
