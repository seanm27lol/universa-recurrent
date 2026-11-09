"""Deterministic teacher-forced training for the four native TI-TNN arms.

The token budget counts ``labels[:, 1:] != -100``, exactly the targets scored by
:func:`native.losses.next_token_cross_entropy`.  Batches are atomic: the last batch
may cross the requested budget, but padded positions and same-position final labels
never make a run stop early.

A tiny run checks the training and recovery pipeline, not learning.  Learning
curves keep language-model loss separate from the weighted auxiliary objective.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Mapping

import numpy as np
import torch

from . import checkpoint as C
from .config import MODES, NativeConfig
from .data import (
    ARRAY_NAMES,
    IGNORE_INDEX,
    MODEL_INPUT_ARRAYS,
    TARGET_ARRAYS,
    PackedDataset,
    sha256_file,
)
from .losses import compute_losses
from .model import NativeModel
from .sampler import PackedSequenceSampler, SamplerBatch


TRAIN_FORMAT = "native-train-v1"
JOURNAL_NAME = "checkpoint_journal.jsonl"
LEARNING_LOG_NAME = "learning_curve.jsonl"
RECOVERY_LOG_NAME = "recovery.jsonl"
LOSS_NAMES = ("total", "language", "route", "state", "balance")


def _json_bytes(payload) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _append_jsonl(path: Path, record: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write((json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    C.fsync_dir(path.parent)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _memory(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None}
    return {
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _seed_model(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def shifted_sequences(dataset: PackedDataset) -> list[dict]:
    """Expose full rows to the sampler, with supervision counted after causal shifting."""
    sequences = []
    for index in range(len(dataset)):
        row = {name: dataset.arrays[name][index] for name in ARRAY_NAMES}
        row["loss_mask"] = np.asarray(row["labels"][1:] != IGNORE_INDEX, dtype=np.bool_)
        sequences.append(row)
    if not sequences or not any(np.any(sequence["loss_mask"]) for sequence in sequences):
        raise ValueError("training data has no shifted labels eligible for language loss")
    return sequences


def make_sampler(dataset: PackedDataset, batch_size: int, data_seed: int) -> PackedSequenceSampler:
    return PackedSequenceSampler(
        shifted_sequences(dataset), batch_size=batch_size, seed=data_seed, shuffle=True, drop_last=False
    )


def _planned_budget(sampler: PackedSequenceSampler, max_loss_tokens: int) -> tuple[int, int]:
    planning = sampler.clone()
    while planning.consumed_loss_tokens < max_loss_tokens:
        planning.next_batch()
    return planning.batches_consumed, planning.consumed_loss_tokens


class _CheckpointAdamW(torch.optim.AdamW):
    """Adapt AdamW's scalar step tensor to the checkpoint hash's byte-view contract."""

    def state_dict(self):
        state = super().state_dict()
        for parameter_state in state["state"].values():
            step = parameter_state.get("step")
            if torch.is_tensor(step) and step.ndim == 0:
                parameter_state["step"] = step.reshape(1)
        return state

    def load_state_dict(self, state_dict):
        restored = dict(state_dict)
        restored["state"] = {}
        for key, parameter_state in state_dict["state"].items():
            copied = dict(parameter_state)
            step = copied.get("step")
            if torch.is_tensor(step) and step.ndim == 1 and step.numel() == 1:
                copied["step"] = step.reshape(())
            restored["state"][key] = copied
        super().load_state_dict(restored)


def build_optimizer(
    model: NativeModel,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Standard AdamW with decay for matrices and no decay for biases/norm vectors."""
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return _CheckpointAdamW(groups, lr=learning_rate, betas=betas, eps=eps)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_updates: int,
    *,
    warmup_fraction: float = 0.02,
    final_lr_fraction: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Two percent update warmup, then a cosine decay to ten percent of base LR."""
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    warmup = max(total_updates * warmup_fraction, 1e-12)
    remaining = max(total_updates - warmup, 1e-12)

    def multiplier(epoch: int) -> float:
        if epoch < warmup:
            return min(1.0, (epoch + 1) / warmup)
        progress = min(1.0, max(0.0, (epoch - warmup) / remaining))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_fraction + (1.0 - final_lr_fraction) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def forward_losses(
    model: NativeModel,
    batch: Mapping[str, torch.Tensor],
    *,
    lambda_balance: float = 0.01,
    lambda_route: float = 0.1,
    lambda_state: float = 0.1,
):
    """Run only model inputs through NativeModel and all targets through external losses."""
    inputs = {name: batch[name] for name in MODEL_INPUT_ARRAYS}
    with _autocast(inputs["input_ids"].device):
        output = model(**inputs, return_aux=True)
        routed = output.route_logits is not None
        losses = compute_losses(
            output.logits,
            batch["labels"],
            route_logits=output.route_logits,
            route_indices=output.route_indices,
            route_targets=batch["route_targets"] if routed else None,
            route_mask=batch["route_mask"] if routed else None,
            key_padding_mask=batch["key_padding_mask"],
            state_logits=output.state_logits,
            state_targets=batch["state_targets"],
            state_mask=batch["state_mask"],
            lambda_balance=lambda_balance,
            lambda_route=lambda_route,
            lambda_state=lambda_state,
        )
    for name in LOSS_NAMES:
        if losses[name].dtype != torch.float32:
            raise AssertionError(f"{name} loss must be float32, got {losses[name].dtype}")
    return output, losses


def _route_usage(output, key_padding_mask: torch.Tensor) -> list[dict]:
    if output.route_indices is None:
        return []
    indices = output.route_indices.detach()
    probabilities = output.route_probs.detach().float()
    live = ~key_padding_mask.detach().bool()
    result = []
    for pass_index in range(4):
        selected = indices[:, :, pass_index][live].long()
        selected_probabilities = probabilities[:, :, pass_index, :][live]
        decisions = int(selected.numel())
        counts = torch.bincount(selected, minlength=4).cpu()
        probability_sums = selected_probabilities.sum(dim=0).cpu()
        experts = []
        fractions = []
        means = []
        for expert in range(4):
            count = int(counts[expert])
            fraction = count / decisions if decisions else 0.0
            mean = float(probability_sums[expert]) / decisions if decisions else 0.0
            fractions.append(fraction)
            means.append(mean)
            experts.append({
                "expert": expert,
                "count": count,
                "selected_fraction": fraction,
                "mean_probability": mean,
            })
        result.append({
            "pass": pass_index,
            "decisions": decisions,
            "experts": experts,
            "starved_experts": [expert for expert, count in enumerate(counts.tolist()) if count == 0],
            "load_balance": 4.0 * sum(f * p for f, p in zip(fractions, means)),
        })
    return result


def _merge_route_usage(batches: list[list[dict]]) -> list[dict]:
    if not any(batches):
        return []
    merged = []
    for pass_index in range(4):
        decisions = sum(batch[pass_index]["decisions"] for batch in batches)
        counts = [sum(batch[pass_index]["experts"][expert]["count"] for batch in batches) for expert in range(4)]
        means = []
        for expert in range(4):
            probability_sum = sum(
                batch[pass_index]["experts"][expert]["mean_probability"] * batch[pass_index]["decisions"]
                for batch in batches
            )
            means.append(probability_sum / decisions if decisions else 0.0)
        fractions = [count / decisions if decisions else 0.0 for count in counts]
        merged.append({
            "pass": pass_index,
            "decisions": decisions,
            "experts": [
                {
                    "expert": expert,
                    "count": counts[expert],
                    "selected_fraction": fractions[expert],
                    "mean_probability": means[expert],
                }
                for expert in range(4)
            ],
            "starved_experts": [expert for expert, count in enumerate(counts) if count == 0],
            "load_balance": 4.0 * sum(f * p for f, p in zip(fractions, means)),
        })
    return merged


def train_batch(
    model: NativeModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batch: Mapping[str, torch.Tensor],
    *,
    grad_clip: float = 1.0,
    lambda_balance: float = 0.01,
    lambda_route: float = 0.1,
    lambda_state: float = 0.1,
    expected_loss_tokens: int | None = None,
) -> dict:
    """Apply one ordinary AdamW update and return separately timed loss diagnostics."""
    started = time.perf_counter()
    device = batch["input_ids"].device
    _sync(device)
    forward_started = time.perf_counter()
    output, losses = forward_losses(
        model,
        batch,
        lambda_balance=lambda_balance,
        lambda_route=lambda_route,
        lambda_state=lambda_state,
    )
    _sync(device)
    forward_seconds = time.perf_counter() - forward_started
    actual_tokens = int(losses["token_count"].item())
    if expected_loss_tokens is not None and actual_tokens != expected_loss_tokens:
        raise AssertionError(
            f"sampler accounted {expected_loss_tokens} shifted labels but loss received {actual_tokens}"
        )
    optimizer.zero_grad(set_to_none=True)
    backward_started = time.perf_counter()
    losses["total"].backward()
    _sync(device)
    backward_seconds = time.perf_counter() - backward_started
    optimizer_started = time.perf_counter()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True))
    learning_rate = float(optimizer.param_groups[0]["lr"])
    optimizer.step()
    scheduler.step()
    _sync(device)
    optimizer_seconds = time.perf_counter() - optimizer_started
    language = float(losses["language"].detach())
    objective = float(losses["total"].detach())
    wall_seconds = time.perf_counter() - started
    training_seconds = forward_seconds + backward_seconds + optimizer_seconds
    return {
        "loss": language,
        "language_loss": language,
        "route_loss": float(losses["route"].detach()),
        "state_loss": float(losses["state"].detach()),
        "balance_loss": float(losses["balance"].detach()),
        "weighted_auxiliary_loss": objective - language,
        "objective_loss": objective,
        "loss_tokens": actual_tokens,
        "route_targets": int(losses["route_count"].item()),
        "state_targets": int(losses["state_count"].item()),
        "balance_decisions": int(losses["balance_count"].item()),
        "grad_norm": grad_norm,
        "learning_rate": learning_rate,
        "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "data_loading_seconds": 0.0,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "checkpoint_seconds": 0.0,
        "wall_seconds": wall_seconds,
        "loss_tokens_per_second": actual_tokens / training_seconds if training_seconds > 0 else 0.0,
        "route_usage": _route_usage(output, batch["key_padding_mask"]),
    }


def _batch_hash(batch: Mapping[str, torch.Tensor], names) -> str:
    digest = hashlib.sha256()
    for name in names:
        tensor = batch[name].detach().to(device="cpu").contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(4, "big"))
        digest.update(header)
        raw = tensor.view(torch.uint8).numpy().tobytes() if tensor.numel() else b""
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _load_batch(dataset: PackedDataset, indices, device: torch.device) -> tuple[dict, dict, float]:
    started = time.perf_counter()
    cpu_batch = dataset.batch(list(indices), as_torch=True)
    identity = {
        "batch_content_sha256": _batch_hash(cpu_batch, ARRAY_NAMES),
        "model_inputs_sha256": _batch_hash(cpu_batch, MODEL_INPUT_ARRAYS),
        "targets_sha256": _batch_hash(cpu_batch, TARGET_ARRAYS),
    }
    batch = {name: tensor.to(device) for name, tensor in cpu_batch.items()}
    _sync(device)
    return batch, identity, time.perf_counter() - started


def evaluate_teacher_forced(
    model: NativeModel,
    dataset: PackedDataset,
    *,
    batch_size: int,
    device: torch.device,
    max_batches: int = 0,
    lambda_balance: float = 0.01,
    lambda_route: float = 0.1,
    lambda_state: float = 0.1,
) -> dict:
    """Evaluate shifted LM and auxiliary terms separately, without advancing training state."""
    was_training = model.training
    model.eval()
    sums = {name: 0.0 for name in ("language", "route", "state", "balance")}
    counts = {name: 0 for name in ("language", "route", "state")}
    route_usage = []
    data_seconds = 0.0
    forward_seconds = 0.0
    batches = 0
    try:
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                if max_batches and batches >= max_batches:
                    break
                indices = list(range(start, min(start + batch_size, len(dataset))))
                batch, _, loading = _load_batch(dataset, indices, device)
                data_seconds += loading
                _sync(device)
                started = time.perf_counter()
                output, losses = forward_losses(
                    model,
                    batch,
                    lambda_balance=lambda_balance,
                    lambda_route=lambda_route,
                    lambda_state=lambda_state,
                )
                _sync(device)
                forward_seconds += time.perf_counter() - started
                token_count = int(losses["token_count"].item())
                route_count = int(losses["route_count"].item())
                state_count = int(losses["state_count"].item())
                sums["language"] += float(losses["language"]) * token_count
                sums["route"] += float(losses["route"]) * route_count
                sums["state"] += float(losses["state"]) * state_count
                routing_decisions = int(losses["balance_count"].item())
                sums["balance"] += float(losses["balance"]) * routing_decisions
                counts["language"] += token_count
                counts["route"] += route_count
                counts["state"] += state_count
                counts["balance"] = counts.get("balance", 0) + routing_decisions
                route_usage.append(_route_usage(output, batch["key_padding_mask"]))
                batches += 1
    finally:
        model.train(was_training)
    averages = {
        name: sums[name] / counts[name] if counts[name] else 0.0
        for name in ("language", "route", "state", "balance")
    }
    auxiliary = (
        lambda_balance * averages["balance"]
        + lambda_route * averages["route"]
        + lambda_state * averages["state"]
    )
    timed = data_seconds + forward_seconds
    return {
        "teacher_forced": True,
        "split": "dev",
        "batches": batches,
        "loss": averages["language"],
        "language_loss": averages["language"],
        "route_loss": averages["route"],
        "state_loss": averages["state"],
        "balance_loss": averages["balance"],
        "weighted_auxiliary_loss": auxiliary,
        "objective_loss": averages["language"] + auxiliary,
        "loss_tokens": counts["language"],
        "route_targets": counts["route"],
        "state_targets": counts["state"],
        "balance_decisions": counts["balance"],
        "route_usage": _merge_route_usage(route_usage),
        "data_loading_seconds": data_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": 0.0,
        "optimizer_seconds": 0.0,
        "checkpoint_seconds": 0.0,
        "loss_tokens_per_second": counts["language"] / timed if timed > 0 else 0.0,
    }


def _code_identity() -> str:
    root = Path(__file__).resolve().parent
    files = sorted(root.glob("*.py"))
    scaling_ledger = root.parent / "scaling" / "ledger.py"
    if scaling_ledger.is_file():
        files.append(scaling_ledger)
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root.parent)
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("max_loss_tokens", "batch_size"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("checkpoint_every", "dev_every", "dev_batches"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if not 0 <= args.warmup_fraction <= 1:
        raise ValueError("warmup_fraction must be in [0, 1]")
    if not 0 < args.final_lr_fraction <= 1:
        raise ValueError("final_lr_fraction must be in (0, 1]")
    if args.grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    if (args.ledger_path is None) != (args.ledger_lock is None):
        raise ValueError("ledger_path and ledger_lock must be supplied together")


def _execute(
    args: argparse.Namespace,
    *,
    model_config: NativeConfig,
    train_dataset: PackedDataset,
    dev_dataset: PackedDataset,
    sampler: PackedSequenceSampler,
    dev_identity: str,
    planned_updates: int,
    planned_tokens: int,
    run_config: dict,
    code_identity: str,
    data_manifest_identity: str | None,
    device: torch.device,
    reservation,
    batch_hook=None,
) -> dict:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    encoded_config = _json_bytes(run_config)
    if config_path.exists() and config_path.read_bytes() != encoded_config:
        raise C.CheckpointIdentityMismatch(f"{config_path} belongs to a different training configuration")
    if not config_path.exists():
        C.atomic_write(config_path, encoded_config)

    _seed_model(args.model_seed)
    model = NativeModel(model_config)
    model.to(device)
    optimizer = build_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        eps=args.eps,
    )
    scheduler = build_scheduler(
        optimizer,
        planned_updates,
        warmup_fraction=args.warmup_fraction,
        final_lr_fraction=args.final_lr_fraction,
    )
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    journal = C.CheckpointJournal(output / JOURNAL_NAME)
    learning_log = output / LEARNING_LOG_NAME
    recovery_log = output / RECOVERY_LOG_NAME
    reservation_id = reservation.id if reservation is not None else None

    def common(event: str) -> dict:
        return {
            "event": event,
            "mode": args.mode,
            "device": str(device),
            "reservation_id": reservation_id,
            **_memory(device),
        }

    totals = {"data": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0, "checkpoint": 0.0}
    processed_tokens = 0
    processed_updates = 0
    replayed_updates = 0

    def save(kind: str, *, recovery_id: str | None = None, extra: Mapping | None = None) -> tuple[dict, float]:
        started = time.perf_counter()
        if kind == "initial":
            filename = "update-00000000.pt"
        elif kind == "periodic":
            filename = f"update-{sampler.batches_consumed:08d}.pt"
        elif kind == "recovery":
            filename = f"{recovery_id}-update-{sampler.batches_consumed:08d}.pt"
        elif kind == "final":
            filename = f"final-update-{sampler.batches_consumed:08d}.pt"
        else:
            raise ValueError(f"unknown checkpoint kind {kind}")
        descriptor = C.save_checkpoint(
            output / "checkpoints" / filename,
            model,
            optimizer,
            scheduler,
            sampler=sampler,
            config=run_config,
            code_identity=code_identity,
            data_identity=sampler.data_identity,
            order_identity=sampler.order_identity,
            identities={"dev_data": dev_identity, "data_manifest": data_manifest_identity},
            extra={
                "checkpoint_kind": kind,
                "mode": args.mode,
                "model_seed": args.model_seed,
                "data_seed": args.data_seed,
                "planned_updates": planned_updates,
                "planned_consumed_loss_tokens": planned_tokens,
                "reservation_id": reservation_id,
                **dict(extra or {}),
            },
        )
        journal.append_checkpoint(descriptor)
        elapsed = time.perf_counter() - started
        totals["checkpoint"] += elapsed
        return descriptor, elapsed

    recovered = None
    if journal.path.exists():
        plan = C.recover(
            journal.path,
            expected_config=run_config,
            expected_code_identity=code_identity,
            expected_data_identity=sampler.data_identity,
            expected_order_identity=sampler.order_identity,
            expected_identities={"dev_data": dev_identity, "data_manifest": data_manifest_identity},
        )
        plan.restore(
            model,
            optimizer,
            scheduler,
            sampler,
            strict_cuda=device.type == "cuda",
            expected_config=run_config,
            expected_code_identity=code_identity,
            expected_data_identity=sampler.data_identity,
            expected_order_identity=sampler.order_identity,
            expected_identities={"dev_data": dev_identity, "data_manifest": data_manifest_identity},
        )
        expected_replay = plan.validate_replay(sampler)
        previous_recoveries = 0
        if recovery_log.exists():
            previous_recoveries = sum(
                1 for line in recovery_log.read_text().splitlines() if line.strip() and json.loads(line).get("event") == "started"
            )
        recovery_id = f"recovery-{previous_recoveries + 1:04d}"
        removed = plan.truncate_incomplete_tail()
        recovered = {
            "recovery_id": recovery_id,
            "selected_checkpoint": plan.checkpoint_path.name,
            "selected_sha256": plan.checkpoint["sha256"],
            "replay_steps": [entry.step for entry in plan.replay_batches],
            "invalid_checkpoints": list(plan.invalid_checkpoints),
            "truncated_incomplete_bytes": removed,
        }
        _append_jsonl(recovery_log, {**common("started"), **recovered})
        for logged, expected in zip(plan.replay_batches, expected_replay):
            if reservation is not None:
                reservation.check()
            actual = sampler.next_batch()
            if actual != expected or actual != logged.batch or sampler.state_dict() != logged.sampler_state:
                raise C.ReplayMismatch(f"deterministic replay differs at logged step {logged.step}")
            batch, identities, loading = _load_batch(train_dataset, actual.indices, device)
            record = train_batch(
                model,
                optimizer,
                scheduler,
                batch,
                grad_clip=args.grad_clip,
                lambda_balance=args.lambda_balance,
                lambda_route=args.lambda_route,
                lambda_state=args.lambda_state,
                expected_loss_tokens=actual.loss_tokens,
            )
            record.update({
                **common("replay"),
                **identities,
                "recovery_id": recovery_id,
                "step": actual.step,
                "update": actual.step,
                "epoch": actual.epoch,
                "cursor": actual.cursor,
                "indices": list(actual.indices),
                "batch_identity": actual.batch_identity,
                "consumed_loss_tokens": actual.consumed_loss_tokens,
            })
            record["data_loading_seconds"] = loading
            totals["data"] += loading
            totals["forward"] += record["forward_seconds"]
            totals["backward"] += record["backward_seconds"]
            totals["optimizer"] += record["optimizer_seconds"]
            processed_tokens += actual.loss_tokens
            processed_updates += 1
            replayed_updates += 1
            _append_jsonl(learning_log, record)
        if plan.requires_replay or plan.invalid_checkpoints or removed:
            descriptor, checkpoint_seconds = save("recovery", recovery_id=recovery_id)
            recovered["checkpoint"] = descriptor
            recovered["checkpoint_seconds"] = checkpoint_seconds
        _append_jsonl(recovery_log, {
            **common("completed"),
            "recovery_id": recovery_id,
            "step": sampler.batches_consumed,
            "cursor": sampler.cursor,
            "consumed_loss_tokens": sampler.consumed_loss_tokens,
            "replayed_updates": replayed_updates,
            "checkpoint": recovered.get("checkpoint"),
        })
        model.train()
    else:
        descriptor, checkpoint_seconds = save("initial")
        _append_jsonl(learning_log, {
            **common("checkpoint"),
            "checkpoint_kind": "initial",
            "step": 0,
            "update": 0,
            "cursor": 0,
            "consumed_loss_tokens": 0,
            "loss_tokens": 0,
            "checkpoint_seconds": checkpoint_seconds,
            "checkpoint": descriptor,
        })

    new_updates = 0
    final_dev = None
    while sampler.consumed_loss_tokens < args.max_loss_tokens:
        if reservation is not None:
            reservation.check()
        sampler_batch: SamplerBatch = sampler.next_batch()
        batch, identities, loading = _load_batch(train_dataset, sampler_batch.indices, device)
        record = train_batch(
            model,
            optimizer,
            scheduler,
            batch,
            grad_clip=args.grad_clip,
            lambda_balance=args.lambda_balance,
            lambda_route=args.lambda_route,
            lambda_state=args.lambda_state,
            expected_loss_tokens=sampler_batch.loss_tokens,
        )
        journal.append_batch(sampler_batch, sampler)
        processed_tokens += sampler_batch.loss_tokens
        processed_updates += 1
        new_updates += 1
        totals["data"] += loading
        totals["forward"] += record["forward_seconds"]
        totals["backward"] += record["backward_seconds"]
        totals["optimizer"] += record["optimizer_seconds"]
        record.update({
            **common("train"),
            **identities,
            "step": sampler_batch.step,
            "update": sampler_batch.step,
            "epoch": sampler_batch.epoch,
            "cursor": sampler_batch.cursor,
            "indices": list(sampler_batch.indices),
            "dropped_indices": list(sampler_batch.dropped_indices),
            "batch_identity": sampler_batch.batch_identity,
            "consumed_loss_tokens": sampler_batch.consumed_loss_tokens,
        })
        record["data_loading_seconds"] = loading
        if batch_hook is not None:
            batch_hook(record)
        finished = sampler.consumed_loss_tokens >= args.max_loss_tokens
        periodic = not finished and args.checkpoint_every and sampler_batch.step % args.checkpoint_every == 0
        dev_due = not finished and args.dev_every and sampler_batch.step % args.dev_every == 0
        dev_record = None
        if dev_due:
            dev_record = evaluate_teacher_forced(
                model,
                dev_dataset,
                batch_size=args.batch_size,
                device=device,
                max_batches=args.dev_batches,
                lambda_balance=args.lambda_balance,
                lambda_route=args.lambda_route,
                lambda_state=args.lambda_state,
            )
        if periodic:
            descriptor, checkpoint_seconds = save("periodic")
            record["checkpoint"] = descriptor
            record["checkpoint_seconds"] = checkpoint_seconds
            if dev_record is not None:
                dev_record["checkpoint"] = descriptor
                dev_record["checkpoint_seconds"] = checkpoint_seconds
        _append_jsonl(learning_log, record)
        if dev_record is not None:
            _append_jsonl(learning_log, {
                **common("dev"),
                **dev_record,
                "step": sampler_batch.step,
                "update": sampler_batch.step,
                "cursor": sampler.cursor,
                "consumed_loss_tokens": sampler.consumed_loss_tokens,
            })

    if sampler.batches_consumed != planned_updates or sampler.consumed_loss_tokens != planned_tokens:
        raise C.ReplayMismatch(
            "recovered/new batch stream does not match the planned token-budget endpoint: "
            f"got step {sampler.batches_consumed}, tokens {sampler.consumed_loss_tokens}; "
            f"expected step {planned_updates}, tokens {planned_tokens}"
        )
    if reservation is not None:
        reservation.check()
    final_dev = evaluate_teacher_forced(
        model,
        dev_dataset,
        batch_size=args.batch_size,
        device=device,
        max_batches=args.dev_batches,
        lambda_balance=args.lambda_balance,
        lambda_route=args.lambda_route,
        lambda_state=args.lambda_state,
    )
    descriptor, checkpoint_seconds = save("final", extra={"dev": final_dev})
    final_dev.update({
        **common("dev"),
        **final_dev,
        "step": sampler.batches_consumed,
        "update": sampler.batches_consumed,
        "cursor": sampler.cursor,
        "consumed_loss_tokens": sampler.consumed_loss_tokens,
        "checkpoint_seconds": checkpoint_seconds,
        "checkpoint": descriptor,
    })
    _append_jsonl(learning_log, final_dev)
    _append_jsonl(learning_log, {
        **common("checkpoint"),
        "checkpoint_kind": "final",
        "step": sampler.batches_consumed,
        "update": sampler.batches_consumed,
        "cursor": sampler.cursor,
        "consumed_loss_tokens": sampler.consumed_loss_tokens,
        "loss_tokens": 0,
        "checkpoint_seconds": checkpoint_seconds,
        "checkpoint": descriptor,
    })
    training_seconds = totals["data"] + totals["forward"] + totals["backward"] + totals["optimizer"]
    return {
        "format": TRAIN_FORMAT,
        "mode": args.mode,
        "device": str(device),
        "step": sampler.batches_consumed,
        "cursor": sampler.cursor,
        "consumed_loss_tokens": sampler.consumed_loss_tokens,
        "max_loss_tokens": args.max_loss_tokens,
        "planned_updates": planned_updates,
        "new_updates": new_updates,
        "replayed_updates": replayed_updates,
        "processed_updates_this_invocation": processed_updates,
        "processed_loss_tokens_this_invocation": processed_tokens,
        "data_loading_seconds": totals["data"],
        "forward_seconds": totals["forward"],
        "backward_seconds": totals["backward"],
        "optimizer_seconds": totals["optimizer"],
        "checkpoint_seconds": totals["checkpoint"],
        "actual_loss_tokens_per_second": processed_tokens / training_seconds if training_seconds > 0 else 0.0,
        "reservation_id": reservation_id,
        **_memory(device),
        "final_checkpoint": descriptor,
        "dev": final_dev,
        "recovery": recovered,
        "learning_log": str(learning_log),
        "journal": str(journal.path),
        "evidence_scope": "tiny/pipeline checks are not learning evidence; inspect held-out curves separately",
    }


def run_training(args: argparse.Namespace, *, batch_hook=None) -> dict:
    """Build the actual native APIs and run one invocation, recovering an existing journal automatically."""
    _validate_args(args)
    if args.model_seed is None:
        args.model_seed = args.seed
    if args.data_seed is None:
        args.data_seed = args.seed
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or a CUDA device")
    if device.type == "cuda":
        from . import ledger as native_ledger

        ledger_path, ledger_lock = native_ledger.validate_shared_reservation_paths(
            args.ledger_path, args.ledger_lock
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
    base_config = NativeConfig.from_json(args.config)
    model_config = replace(base_config, mode=args.mode)
    data_root = Path(args.data)
    train_dataset = PackedDataset(data_root, "train")
    dev_dataset = PackedDataset(data_root, "dev")
    if train_dataset.arrays["input_ids"].shape[1] != model_config.context_length:
        raise ValueError("packed context length differs from NativeConfig.context_length")
    if int(train_dataset.arrays["input_ids"].max()) >= model_config.vocab_size:
        raise ValueError("packed token id exceeds NativeConfig.vocab_size")
    sampler = make_sampler(train_dataset, args.batch_size, args.data_seed)
    dev_sampler = make_sampler(dev_dataset, args.batch_size, args.data_seed)
    planned_updates, planned_tokens = _planned_budget(sampler, args.max_loss_tokens)
    code_identity = _code_identity()
    manifest_path = data_root / "manifest.json"
    data_manifest_identity = sha256_file(manifest_path) if manifest_path.is_file() else None
    run_config = {
        "format": TRAIN_FORMAT,
        "model": model_config.to_dict(),
        "training": {
            "max_loss_tokens": args.max_loss_tokens,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "model_seed": args.model_seed,
            "data_seed": args.data_seed,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "betas": list(args.betas),
            "eps": args.eps,
            "grad_clip": args.grad_clip,
            "lambda_balance": args.lambda_balance,
            "lambda_route": args.lambda_route,
            "lambda_state": args.lambda_state,
            "warmup_fraction": args.warmup_fraction,
            "final_lr_fraction": args.final_lr_fraction,
            "checkpoint_every": args.checkpoint_every,
            "dev_every": args.dev_every,
            "dev_batches": args.dev_batches,
            "device": str(device),
            "optimizer": "torch.optim.AdamW; matrices decay, biases and norm vectors do not",
            "scheduler": "2% update warmup then cosine to 10% of base learning rate",
            "autocast": "bfloat16 on CUDA only; all losses float32",
            "token_rule": "count labels[:,1:] != -100; atomic batches stop at the first count >= max_loss_tokens",
            "planned_updates": planned_updates,
            "planned_consumed_loss_tokens": planned_tokens,
        },
        "source_config_sha256": sha256_file(args.config),
        "data": str(data_root.resolve()),
        "train_data_identity": sampler.data_identity,
        "dev_data_identity": dev_sampler.data_identity,
        "order_identity": sampler.order_identity,
        "code_identity": code_identity,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    if device.type == "cpu":
        return _execute(
            args,
            model_config=model_config,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            sampler=sampler,
            dev_identity=dev_sampler.data_identity,
            planned_updates=planned_updates,
            planned_tokens=planned_tokens,
            run_config=run_config,
            code_identity=code_identity,
            data_manifest_identity=data_manifest_identity,
            device=device,
            reservation=None,
            batch_hook=batch_hook,
        )

    reservation = native_ledger.Reservation(
        command=f"native.train mode={args.mode} data={data_root} output={args.output}",
        projected_hours=args.projected_hours,
        path=ledger_path,
        lock=ledger_lock,
        extra={"mode": args.mode, "format": TRAIN_FORMAT},
    )
    with reservation as active:
        return _execute(
            args,
            model_config=model_config,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            sampler=sampler,
            dev_identity=dev_sampler.data_identity,
            planned_updates=planned_updates,
            planned_tokens=planned_tokens,
            run_config=run_config,
            code_identity=code_identity,
            data_manifest_identity=data_manifest_identity,
            device=device,
            reservation=active,
            batch_hook=batch_hook,
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m native.train", description=__doc__.split("\n\n")[0])
    result.add_argument("--config", required=True, help="NativeConfig JSON; its model section is used")
    result.add_argument("--mode", required=True, choices=MODES)
    result.add_argument("--data", required=True, help="PackedDataset root containing train/ and dev/")
    result.add_argument("--output", required=True, help="arm output directory; an existing journal is recovered")
    result.add_argument("--device", default="cpu")
    result.add_argument("--max-loss-tokens", type=int, default=4096)
    result.add_argument("--batch-size", type=int, default=2)
    result.add_argument("--seed", type=int, default=0, help="default for both independent model and data seeds")
    result.add_argument("--model-seed", type=int, default=None)
    result.add_argument("--data-seed", type=int, default=None)
    result.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=0.01)
    result.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    result.add_argument("--eps", type=float, default=1e-8)
    result.add_argument("--grad-clip", type=float, default=1.0)
    result.add_argument("--lambda-balance", type=float, default=0.01)
    result.add_argument("--lambda-route", type=float, default=0.1)
    result.add_argument("--lambda-state", type=float, default=0.1)
    result.add_argument("--warmup-fraction", type=float, default=0.02)
    result.add_argument("--final-lr-fraction", type=float, default=0.1)
    result.add_argument("--checkpoint-every", "--checkpoint-interval", dest="checkpoint_every", type=int, default=100)
    result.add_argument("--dev-every", "--dev-interval", dest="dev_every", type=int, default=0,
                        help="teacher-forced dev interval in updates; 0 means final only")
    result.add_argument("--dev-batches", type=int, default=0, help="limit each dev pass; 0 evaluates the full split")
    result.add_argument("--resume", action="store_true", help="accepted for clarity; journal recovery is always automatic")
    result.add_argument("--projected-hours", type=float, default=1.0)
    result.add_argument("--ledger-path", default=None, help="GPU test ledger override; CPU never opens a reservation")
    result.add_argument("--ledger-lock", default=None, help="GPU test lock override; CPU never opens a reservation")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    result = run_training(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


train = run_training


if __name__ == "__main__":
    raise SystemExit(main())
