"""Measure native training costs on packed batches before reserving a multi-arm run.

CUDA profiling opens one native reservation before probing or initializing CUDA and
keeps it through loading, warm-up, cache validation, training, evaluation, and a
durable checkpoint write. ``--dry-run --device cpu`` exercises the same data and
checkpoint path without a reservation; its timings are explicitly ineligible for
feasibility projections.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import statistics
import subprocess
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from scaling import ledger as scaling_ledger
from . import checkpoint as C
from . import evaluate as E
from . import reconstruct as R
from . import train as T
from .config import MODES, NativeConfig
from .data import ARRAY_NAMES, IGNORE_INDEX, MODEL_INPUT_ARRAYS, PackedDataset, sha256_file
from .ledger import Reservation
from .losses import compute_losses
from .model import NativeModel
from .sampler import identity_hash


PROFILE_FORMAT = "native-profile-v1"
DEFAULT_PLANNING_MARGIN = 0.20
MAX_NATIVE_HOURS = 12.0
EXECUTION_MODES = ("eager", "compile")
DTYPES = ("float32", "bfloat16", "float16")


class FeasibilityError(RuntimeError):
    """A measured plan does not fit the authorized hours with planning margin."""

    def __init__(self, message: str, projection: Mapping):
        super().__init__(message)
        self.projection = dict(projection)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _check_reservation(reservation) -> None:
    if reservation is not None:
        reservation.check()


def _torch_dtype(name: str) -> torch.dtype:
    try:
        return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]
    except KeyError as error:
        raise ValueError(f"dtype must be one of {DTYPES}, got {name!r}") from error


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype != torch.float32:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _forward_losses(model, batch: Mapping[str, torch.Tensor], args, dtype: torch.dtype):
    inputs = {name: batch[name] for name in MODEL_INPUT_ARRAYS}
    with _autocast(inputs["input_ids"].device, dtype):
        output = model(**inputs, return_aux=True)
        routed = output.route_logits is not None
        loss_kwargs = {
            "route_logits": output.route_logits,
            "route_indices": output.route_indices,
            "route_targets": batch["route_targets"] if routed else None,
            "route_mask": batch["route_mask"] if routed else None,
            "state_logits": output.state_logits,
            "state_targets": batch["state_targets"],
            "state_mask": batch["state_mask"],
            "lambda_balance": args.lambda_balance,
            "lambda_route": args.lambda_route,
            "lambda_state": args.lambda_state,
        }
        parameters = inspect.signature(compute_losses).parameters
        accepts_mask = "key_padding_mask" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_mask:
            loss_kwargs["key_padding_mask"] = batch["key_padding_mask"]
        losses = compute_losses(output.logits, batch["labels"], **loss_kwargs)
    for name in T.LOSS_NAMES:
        if losses[name].dtype != torch.float32:
            raise AssertionError(f"{name} loss must be float32, got {losses[name].dtype}")
    return output, losses


def _load_batch(
    dataset: PackedDataset,
    indices: Sequence[int],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, float]:
    started = time.perf_counter()
    cpu_batch = dataset.batch(list(indices), as_torch=True)
    loss_tokens = int(torch.count_nonzero(cpu_batch["labels"][:, 1:] != IGNORE_INDEX).item())
    batch = {name: tensor.to(device=device) for name, tensor in cpu_batch.items()}
    _synchronize(device)
    return batch, loss_tokens, time.perf_counter() - started


def _batch_indices(step: int, batch_size: int, rows: int) -> tuple[int, ...]:
    start = (step * batch_size) % rows
    return tuple((start + offset) % rows for offset in range(batch_size))


def _training_step(model, optimizer, scheduler, batch, args, dtype: torch.dtype) -> dict:
    device = batch["input_ids"].device
    model.train()
    _synchronize(device)
    started = time.perf_counter()
    output, losses = _forward_losses(model, batch, args, dtype)
    _synchronize(device)
    forward_seconds = time.perf_counter() - started
    actual_tokens = int(losses["token_count"].item())
    optimizer.zero_grad(set_to_none=True)
    _synchronize(device)
    started = time.perf_counter()
    losses["total"].backward()
    _synchronize(device)
    backward_seconds = time.perf_counter() - started
    _synchronize(device)
    started = time.perf_counter()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(
        model.parameters(), args.grad_clip, error_if_nonfinite=True
    ))
    optimizer.step()
    scheduler.step()
    _synchronize(device)
    optimizer_seconds = time.perf_counter() - started
    del output
    return {
        "actual_shifted_loss_tokens": actual_tokens,
        "language_loss": float(losses["language"].detach()),
        "objective_loss": float(losses["total"].detach()),
        "grad_norm": grad_norm,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "gradient_clipping_optimizer_seconds": optimizer_seconds,
        "optimizer_seconds": optimizer_seconds,
    }


def _run_training_steps(
    model,
    optimizer,
    scheduler,
    dataset: PackedDataset,
    args,
    dtype: torch.dtype,
    device: torch.device,
    reservation,
    *,
    first_step: int,
    count: int,
) -> list[dict]:
    records = []
    for local_step in range(count):
        _check_reservation(reservation)
        step = first_step + local_step
        indices = _batch_indices(step, args.batch_size, len(dataset))
        batch, expected_tokens, loading_seconds = _load_batch(dataset, indices, device)
        record = _training_step(model, optimizer, scheduler, batch, args, dtype)
        if record["actual_shifted_loss_tokens"] != expected_tokens:
            raise AssertionError(
                f"batch accounting differs: packed labels gave {expected_tokens}, "
                f"loss received {record['actual_shifted_loss_tokens']}"
            )
        record.update({
            "step": step + 1,
            "indices": list(indices),
            "data_loading_seconds": loading_seconds,
        })
        record["wall_seconds"] = (
            loading_seconds
            + record["forward_seconds"]
            + record["backward_seconds"]
            + record["gradient_clipping_optimizer_seconds"]
        )
        record["actual_shifted_loss_tokens_per_second"] = (
            expected_tokens / record["wall_seconds"] if record["wall_seconds"] > 0 else 0.0
        )
        records.append(record)
    return records


def _difference(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> dict:
    if actual.shape != expected.shape:
        raise AssertionError(f"cache comparison shapes differ: {actual.shape} versus {expected.shape}")
    if actual.device != expected.device or actual.dtype != expected.dtype:
        raise AssertionError("cache comparison must retain the actual output device and dtype")
    difference = actual.float() - expected.float()
    return {
        "allclose": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "maximum_absolute": float(difference.abs().max().item()) if difference.numel() else 0.0,
        "mean_absolute": float(difference.abs().mean().item()) if difference.numel() else 0.0,
        "dtype": str(actual.dtype),
        "device": str(actual.device),
    }


@torch.no_grad()
def check_full_vs_cached(
    model,
    batch: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict:
    """Compare full and prefill-plus-one cached execution at the final packed position."""
    inputs = {name: batch[name] for name in MODEL_INPUT_ARRAYS}
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] < 2:
        raise ValueError("cache validation needs at least two packed positions")
    was_training = model.training
    model.eval()
    try:
        with _autocast(input_ids.device, dtype):
            full = model(**inputs, return_aux=True)
            prefix_inputs = {name: value[:, :-1] for name, value in inputs.items()}
            prefix = model(**prefix_inputs, use_cache=True)
            cache = prefix.cache
            del prefix
            final_inputs = {name: value[:, -1:] for name, value in inputs.items()}
            cached = model(**final_inputs, cache=cache, use_cache=True, return_aux=True)
        _synchronize(input_ids.device)
        logits = _difference(cached.logits, full.logits[:, -1:], atol=atol, rtol=rtol)
        if (cached.route_logits is None) != (full.route_logits is None):
            raise AssertionError("full and cached execution disagree about router outputs")
        routes = None
        route_indices_equal = None
        if full.route_logits is not None:
            routes = _difference(
                cached.route_logits,
                full.route_logits[:, -1:],
                atol=atol,
                rtol=rtol,
            )
            route_indices_equal = bool(torch.equal(
                cached.route_indices,
                full.route_indices[:, -1:],
            ))
        valid = logits["allclose"] and (routes is None or routes["allclose"]) and route_indices_equal is not False
        return {
            "valid": bool(valid),
            "comparison": "full packed forward versus packed prefix cache plus its final original token",
            "compared_position": input_ids.shape[1] - 1,
            "next_token_logits": logits,
            "next_token_route_logits": routes,
            "next_token_route_indices_equal": route_indices_equal,
            "atol": atol,
            "rtol": rtol,
            "model_device": str(next(model.parameters()).device),
            "model_parameter_dtype": str(next(model.parameters()).dtype),
        }
    finally:
        model.train(was_training)


def _teacher_forced_evaluation(
    model,
    dataset: PackedDataset,
    args,
    dtype: torch.dtype,
    device: torch.device,
    reservation,
) -> dict:
    _check_reservation(reservation)
    _synchronize(device)
    wall_started = time.perf_counter()
    was_training = model.training
    model.eval()
    sums = {name: 0.0 for name in ("language", "route", "state", "balance")}
    counts = {name: 0 for name in ("language", "route", "state", "balance")}
    data_seconds = 0.0
    forward_seconds = 0.0
    batches = 0
    batch_size = args.evaluation_batch_size or args.batch_size
    try:
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                if args.evaluation_batches and batches >= args.evaluation_batches:
                    break
                _check_reservation(reservation)
                indices = tuple(range(start, min(start + batch_size, len(dataset))))
                batch, token_count, loading = _load_batch(dataset, indices, device)
                data_seconds += loading
                _synchronize(device)
                started = time.perf_counter()
                output, losses = _forward_losses(model, batch, args, dtype)
                _synchronize(device)
                forward_seconds += time.perf_counter() - started
                route_count = int(losses["route_count"].item())
                state_count = int(losses["state_count"].item())
                sums["language"] += float(losses["language"].detach()) * token_count
                sums["route"] += float(losses["route"].detach()) * route_count
                sums["state"] += float(losses["state"].detach()) * state_count
                default_balance_count = int(batch["input_ids"].numel() * 4) if output.route_logits is not None else 0
                routing_decisions = int(losses.get("balance_count", default_balance_count))
                sums["balance"] += float(losses["balance"].detach()) * routing_decisions
                counts["language"] += token_count
                counts["route"] += route_count
                counts["state"] += state_count
                counts["balance"] += routing_decisions
                batches += 1
                del output, losses, batch
    finally:
        model.train(was_training)
    _synchronize(device)
    wall_seconds = time.perf_counter() - wall_started
    averages = {
        name: sums[name] / counts[name] if counts[name] else 0.0
        for name in sums
    }
    auxiliary = (
        args.lambda_balance * averages["balance"]
        + args.lambda_route * averages["route"]
        + args.lambda_state * averages["state"]
    )
    return {
        "teacher_forced": True,
        "batches": batches,
        "batch_size": batch_size,
        "actual_shifted_loss_tokens": counts["language"],
        "language_loss": averages["language"],
        "route_loss": averages["route"],
        "state_loss": averages["state"],
        "balance_loss": averages["balance"],
        "balance_decisions": counts["balance"],
        "objective_loss": averages["language"] + auxiliary,
        "data_loading_seconds": data_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": 0.0,
        "gradient_clipping_optimizer_seconds": 0.0,
        "wall_seconds": wall_seconds,
        "actual_shifted_loss_tokens_per_second": (
            counts["language"] / wall_seconds if wall_seconds > 0 else 0.0
        ),
    }


def _discover_complete_inputs(args, data_root: Path) -> dict:
    structured = Path(args.structured) if args.structured else data_root / "structured"
    if args.structured and not structured.is_dir():
        raise FileNotFoundError(f"structured evaluation directory does not exist: {structured}")
    structured = structured if structured.is_dir() else None
    code = Path(args.code_generation) if args.code_generation else data_root / "code_generation.jsonl"
    if args.code_generation and not code.is_file():
        raise FileNotFoundError(f"code-generation input does not exist: {code}")
    code = code if code.is_file() else None
    replay_path = (
        Path(args.reconstruction_replay)
        if args.reconstruction_replay else data_root / "replay.json"
    )
    if args.reconstruction_replay and not replay_path.is_file():
        raise FileNotFoundError(f"reconstruction replay does not exist: {replay_path}")
    replay_path = replay_path if replay_path.is_file() else None
    if (structured is not None or code is not None) and replay_path is None:
        raise ValueError("generation inputs exist; --reconstruction-replay is required for interventions and replay")
    if not args.dry_run and (structured is None or code is None or replay_path is None):
        raise ValueError(
            "real profiling requires structured records, code_generation.jsonl, and a reconstruction replay; "
            "missing study inputs cannot become zero/not-run projections"
        )

    tokenizer = Path(args.tokenizer) if args.tokenizer else None
    if tokenizer is None:
        manifest_path = data_root / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            recorded = manifest.get("source_inventory", {}).get("tokenizer_file", {}).get("path")
            if recorded:
                tokenizer = Path(recorded)
        if tokenizer is None and (data_root / "tokenizer.json").is_file():
            tokenizer = data_root / "tokenizer.json"
    if tokenizer is None and (structured is not None or code is not None):
        raise ValueError("generation inputs exist; --tokenizer or a manifest tokenizer_file.path is required")
    replay = R.load_replay_spec(replay_path) if replay_path is not None else None
    return {
        "structured": structured,
        "code": code,
        "tokenizer": tokenizer,
        "replay_path": replay_path,
        "replay": replay,
    }


def _complete_teacher_forced(model, dataset, args, device, reservation) -> tuple[float, dict]:
    _check_reservation(reservation)
    cpu = torch.device("cpu")
    _synchronize(device)
    started = time.perf_counter()
    model.to(device=cpu)
    model.eval()
    _synchronize(device)
    transfer_seconds = time.perf_counter() - started
    _synchronize(cpu)
    started = time.perf_counter()
    measured = E.teacher_forced_evaluate(
        model,
        dataset,
        batch_size=args.complete_evaluation_batch_size,
        max_batches=None,
        device="cpu",
    )
    _synchronize(cpu)
    wall_seconds = time.perf_counter() - started
    if not measured.get("complete"):
        raise AssertionError("complete teacher-forced profiling did not consume the full dev split")
    return transfer_seconds + wall_seconds, {
        "status": "complete",
        "valid": True,
        "device": "cpu",
        "model_reused_from_profiling": True,
        "model_to_cpu_seconds": transfer_seconds,
        "measured_seconds": wall_seconds,
        "batches": measured["batches"],
        "sequences": measured["sequences"],
        "actual_shifted_loss_tokens": measured["overall"]["loss_tokens"],
        "batch_size": args.complete_evaluation_batch_size,
    }


def _structured_generation(model, tokenizer, directory: Path, args, reservation) -> tuple[dict, dict]:
    cpu = torch.device("cpu")
    _check_reservation(reservation)
    started = time.perf_counter()
    records, files = E.load_structured_records(directory)
    loading_seconds = time.perf_counter() - started
    limit = args.structured_limit_per_split or None
    split_reports = {}
    sample_seconds = 0.0
    projected_seconds = 0.0
    for split in E.SPLIT_NAMES:
        _check_reservation(reservation)
        _synchronize(cpu)
        started = time.perf_counter()
        measured = E._generation_metrics(
            model,
            tokenizer,
            records[split],
            max_new_tokens=args.max_new_tokens,
            eos_token_id=args.eos_token_id,
            strict_registers=True,
            use_cache=True,
            limit=limit,
            prompt_policy="error",
        )
        _synchronize(cpu)
        elapsed = time.perf_counter() - started
        examples = int(measured["examples"])
        evaluated = int(measured["evaluated"])
        if examples <= 0 or evaluated <= 0:
            raise ValueError(f"{split}: cannot project complete structured generation from an empty sample")
        projected = elapsed * examples / evaluated
        sample_seconds += elapsed
        projected_seconds += projected
        split_reports[split] = {
            "examples": examples,
            "evaluated": evaluated,
            "sample_complete": bool(measured["complete"]),
            "measured_seconds": elapsed,
            "projected_complete_seconds": projected,
            "file": files[split],
        }
    _check_reservation(reservation)
    started = time.perf_counter()
    E.structured_baseline_report(records)
    baseline_seconds = time.perf_counter() - started
    fixed_seconds = loading_seconds + baseline_seconds
    complete_seconds = fixed_seconds + projected_seconds
    return {
        "status": "complete" if all(item["sample_complete"] for item in split_reports.values())
                else "projected_from_fixed_sample",
        "valid": True,
        "directory": str(directory.resolve()),
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": args.eos_token_id,
        "use_cache": True,
        "limit_per_split": args.structured_limit_per_split,
        "loading_seconds": loading_seconds,
        "baseline_seconds": baseline_seconds,
        "sample_generation_seconds": sample_seconds,
        "measured_seconds": fixed_seconds + sample_seconds,
        "projected_complete_generation_seconds": projected_seconds,
        "seconds": complete_seconds,
        "splits": split_reports,
        "projection_rule": "per-split measured generation seconds times examples/evaluated, plus measured loading and baseline costs",
    }, records


def _code_generation(model, tokenizer, path: Path, args, reservation) -> dict:
    _check_reservation(reservation)
    started = time.perf_counter()
    records = E._load_jsonl(path)
    loading_seconds = time.perf_counter() - started
    if not records:
        raise ValueError(f"{path}: cannot project complete code generation from an empty input")
    _synchronize(torch.device("cpu"))
    started = time.perf_counter()
    measured = E._generation_metrics(
        model,
        tokenizer,
        records,
        max_new_tokens=args.code_max_new_tokens,
        eos_token_id=args.eos_token_id,
        strict_registers=False,
        use_cache=True,
        limit=args.code_generation_limit or None,
        prompt_policy="keep_last",
    )
    _synchronize(torch.device("cpu"))
    generation_seconds = time.perf_counter() - started
    examples = int(measured["examples"])
    evaluated = int(measured["evaluated"])
    if evaluated <= 0:
        raise ValueError(f"{path}: code-generation sample is empty")
    projected = generation_seconds * examples / evaluated
    return {
        "status": "complete" if measured["complete"] else "projected_from_fixed_sample",
        "valid": True,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "examples": examples,
        "evaluated": evaluated,
        "sample_complete": bool(measured["complete"]),
        "max_new_tokens": args.code_max_new_tokens,
        "eos_token_id": args.eos_token_id,
        "use_cache": True,
        "limit": args.code_generation_limit,
        "loading_seconds": loading_seconds,
        "sample_generation_seconds": generation_seconds,
        "measured_seconds": loading_seconds + generation_seconds,
        "projected_complete_generation_seconds": projected,
        "seconds": loading_seconds + projected,
        "projection_rule": "measured generation seconds times examples/evaluated, plus measured input-loading cost",
    }


def _state_interventions(model, tokenizer, records, replay, args, reservation) -> dict:
    if records is None or replay is None:
        return {
            "status": "not_run",
            "valid": False,
            "seconds": 0.0,
            "reason": "state interventions require both structured records and the reconstruction replay",
        }
    specifications = (
        E.InterventionSpec(0, 0, "zero"),
        E.InterventionSpec(0, 0, "shuffle"),
        E.InterventionSpec(0, 0, "wrong_slot", source_slot=1),
    )
    prompts = [records["dev"][0], records["unseen_composition"][0]]
    _check_reservation(reservation)
    _synchronize(torch.device("cpu"))
    started = time.perf_counter()
    E.evaluate_state_interventions(
        model,
        replay["teacher_forced"],
        prompts,
        specifications,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=args.eos_token_id,
        use_cache=True,
    )
    _synchronize(torch.device("cpu"))
    seconds = time.perf_counter() - started
    return {
        "status": "complete",
        "valid": True,
        "seconds": seconds,
        "measured_seconds": seconds,
        "specifications": [spec.to_dict() for spec in specifications],
        "samples": [
            {"split": split, "example_id": record["example_id"], "record_identity": identity_hash(record)}
            for split, record in zip(("dev", "unseen_composition"), prompts)
        ],
        "replay_spec_sha256": replay.get("spec_sha256"),
        "outcomes_retained": False,
    }


def _complete_evaluation(model, datasets, inputs, args, device, reservation) -> dict:
    teacher_seconds, teacher = _complete_teacher_forced(
        model, datasets["dev"], args, device, reservation
    )
    structured_records = None
    structured = {"status": "not_run", "valid": False, "seconds": 0.0, "reason": "no structured directory"}
    code = {"status": "not_run", "valid": False, "seconds": 0.0, "reason": "no code-generation file"}
    tokenizer = None
    tokenizer_seconds = 0.0
    if inputs["structured"] is not None or inputs["code"] is not None:
        _check_reservation(reservation)
        started = time.perf_counter()
        tokenizer = E.load_gpt2_tokenizer(inputs["tokenizer"])
        tokenizer_seconds = time.perf_counter() - started
    if inputs["structured"] is not None:
        structured, structured_records = _structured_generation(
            model, tokenizer, inputs["structured"], args, reservation
        )
        structured["tokenizer_loading_seconds"] = tokenizer_seconds
        structured["seconds"] += tokenizer_seconds
        structured["measured_seconds"] += tokenizer_seconds
    if inputs["code"] is not None:
        code = _code_generation(model, tokenizer, inputs["code"], args, reservation)
        if inputs["structured"] is None:
            code["tokenizer_loading_seconds"] = tokenizer_seconds
            code["seconds"] += tokenizer_seconds
            code["measured_seconds"] += tokenizer_seconds
    interventions = _state_interventions(
        model, tokenizer, structured_records, inputs["replay"], args, reservation
    )
    components = {
        "teacher_forced": teacher,
        "structured_generation": structured,
        "code_generation": code,
        "state_interventions": interventions,
    }
    seconds = {
        "teacher_forced_seconds": teacher_seconds,
        "structured_generation_seconds": structured["seconds"],
        "code_generation_seconds": code["seconds"],
        "state_interventions_seconds": interventions["seconds"],
    }
    valid = all(component.get("valid") is True for component in components.values())
    return {
        **seconds,
        "total_seconds": math.fsum(seconds.values()),
        "status": "complete" if valid else "incomplete",
        "valid": valid,
        "components": components,
        "device": "cpu",
        "child_processes": 0,
        "accounting": "final checkpoint evaluation, disjoint from training-dev passes and setup.fixed_seconds",
    }


def _reconstruction(descriptor, replay, args, reservation) -> dict:
    if replay is None:
        return {
            "status": "not_run",
            "valid": False,
            "seconds": 0.0,
            "reason": "no reconstruction replay specification",
        }
    _check_reservation(reservation)
    output = (
        Path(args.reconstruction_output)
        if args.reconstruction_output else Path(args.output).with_suffix(".reconstruction.json")
    )
    started = time.perf_counter()
    measured = R.reconstruct_checkpoint(
        descriptor["path"],
        replay,
        expected_sha256=descriptor["sha256"],
    )
    digest = R.write_reconstruction_report(output, measured)
    seconds = time.perf_counter() - started
    return {
        "status": "complete" if measured["agreement"] else "disagreement",
        "valid": bool(measured["agreement"]),
        "seconds": seconds,
        "measured_seconds": seconds,
        "agreement": bool(measured["agreement"]),
        "checkpoint_sha256": descriptor["sha256"],
        "replay_spec_sha256": replay.get("spec_sha256"),
        "report_path": str(output.resolve()),
        "report_sha256": digest,
        "checkpoint_dtype": measured["weights"]["dtype"],
        "device": "cpu",
        "child_processes": 0,
    }


def summarize_steps(records: Sequence[Mapping]) -> dict:
    """Sum real shifted labels and keep every timing component separate."""
    if not records:
        raise ValueError("at least one measured training step is required")
    names = (
        "data_loading_seconds",
        "forward_seconds",
        "backward_seconds",
        "gradient_clipping_optimizer_seconds",
        "wall_seconds",
    )
    totals = {name: math.fsum(float(record[name]) for record in records) for name in names}
    tokens = sum(int(record["actual_shifted_loss_tokens"]) for record in records)
    walls = [float(record["wall_seconds"]) for record in records]
    result = {
        "measured_steps": len(records),
        "actual_shifted_loss_tokens": tokens,
        **totals,
        "optimizer_seconds": totals["gradient_clipping_optimizer_seconds"],
        "actual_shifted_loss_tokens_per_second": (
            tokens / totals["wall_seconds"] if totals["wall_seconds"] > 0 else 0.0
        ),
        "mean_actual_shifted_loss_tokens_per_batch": tokens / len(records),
        "seconds_per_step_mean": statistics.fmean(walls),
        "seconds_per_step_median": statistics.median(walls),
        "seconds_per_step_p95": float(np.percentile(walls, 95)),
        "seconds_per_step_std": statistics.pstdev(walls),
        "steps": [dict(record) for record in records],
    }
    accounted = math.fsum(totals[name] for name in names[:-1])
    if not math.isclose(accounted, totals["wall_seconds"], rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError("training timing components do not sum to measured wall time")
    return result


def _data_identity(root: Path, datasets: Mapping[str, PackedDataset]) -> dict:
    manifest_path = root / "manifest.json"
    result = {
        "root": str(root.resolve()),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "splits": {},
    }
    for split, dataset in datasets.items():
        arrays = {}
        for name in ARRAY_NAMES:
            path = dataset.directory / f"{name}.npy"
            array = dataset.arrays[name]
            arrays[name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
        result["splits"][split] = {"arrays": arrays, "rows": len(dataset)}
    result["identity"] = identity_hash({
        "format": "native-profile-data-v1",
        "manifest_sha256": result["manifest_sha256"],
        "splits": result["splits"],
    })
    return result


def _git_commit() -> str | None:
    root = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _foreign_gpu_processes() -> list[dict]:
    return scaling_ledger.foreign_gpu_processes()


def _hardware(device: torch.device, foreign_start: list[dict] | None) -> dict:
    result = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "logical_cpu_count": os.cpu_count(),
        "device": str(device),
        "foreign_gpu_processes_start": foreign_start,
        "foreign_gpu_processes_end": None,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update({
            "cuda_runtime": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_device_capability": list(torch.cuda.get_device_capability(device)),
            "cuda_device_total_memory_bytes": int(properties.total_memory),
            "cuda_device_count": torch.cuda.device_count(),
        })
    return result


def _memory(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None}
    return {
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _cache_tolerances(args, dtype: torch.dtype) -> tuple[float, float]:
    default_atol = 1e-4 if dtype == torch.float32 else (5e-3 if dtype == torch.float16 else 3e-2)
    default_rtol = default_atol
    atol = default_atol if args.cache_atol is None else args.cache_atol
    rtol = default_rtol if args.cache_rtol is None else args.cache_rtol
    if atol < 0 or rtol < 0:
        raise ValueError("cache tolerances must be nonnegative")
    return atol, rtol


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "warmup_steps", "measured_steps"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if args.warmup_steps < 1:
        raise ValueError("warmup_steps must be at least 1")
    if args.measured_steps < 3:
        raise ValueError("measured_steps must be at least 3 for a stable projection sample")
    if args.evaluation_batch_size is not None and args.evaluation_batch_size <= 0:
        raise ValueError("evaluation_batch_size must be positive")
    if args.evaluation_batches < 0:
        raise ValueError("evaluation_batches must be nonnegative; zero evaluates the full dev split")
    if args.complete_evaluation_batch_size <= 0:
        raise ValueError("complete_evaluation_batch_size must be positive")
    for name in ("structured_limit_per_split", "code_generation_limit", "max_new_tokens", "code_max_new_tokens"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer; zero disables the sample limit where applicable")
    if args.grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    if not math.isfinite(args.projected_hours) or args.projected_hours <= 0:
        raise ValueError("projected_hours must be finite and positive")
    if (args.ledger_path is None) != (args.ledger_lock is None):
        raise ValueError("ledger_path and ledger_lock must be supplied together")
    output = Path(args.output).resolve()
    for name in ("checkpoint", "reconstruction_output"):
        value = getattr(args, name)
        if value is not None and Path(value).resolve() == output:
            raise ValueError(f"{name} must differ from the JSON profile output")
    device = torch.device(args.device)
    if args.dry_run:
        if device.type != "cpu":
            raise ValueError("CPU dry-run requires --device cpu")
    elif device.type != "cuda":
        raise ValueError("real profiling requires CUDA; use --dry-run --device cpu for the test path")
    _torch_dtype(args.dtype)


def _execute(args: argparse.Namespace, reservation, report: dict) -> None:
    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else _torch_dtype(args.dtype)
    dtype_name = str(dtype).removeprefix("torch.")
    if not args.dry_run and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for real profiling but is not available")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    report["hardware"] = _hardware(
        device,
        _foreign_gpu_processes() if device.type == "cuda" else None,
    )
    _check_reservation(reservation)

    preparation_started = time.perf_counter()
    base_config = NativeConfig.from_json(args.config)
    model_config = replace(base_config, mode=args.mode)
    data_root = Path(args.data)
    datasets = {
        "train": PackedDataset(data_root, "train"),
        "dev": PackedDataset(data_root, "dev"),
    }
    if datasets["train"].arrays["input_ids"].shape[1] != model_config.context_length:
        raise ValueError("packed context length differs from NativeConfig.context_length")
    if len(datasets["train"]) == 0 or len(datasets["dev"]) == 0:
        raise ValueError("profiling requires nonempty train and dev PackedDataset splits")
    if int(datasets["train"].arrays["input_ids"].max()) >= model_config.vocab_size:
        raise ValueError("packed token id exceeds NativeConfig.vocab_size")
    data_identity = _data_identity(data_root, datasets)
    complete_inputs = _discover_complete_inputs(args, data_root)
    code_identity = T._code_identity()
    order_identity = identity_hash({
        "format": "native-profile-order-v1",
        "data_identity": data_identity["identity"],
        "batch_size": args.batch_size,
        "rule": "cyclic contiguous packed rows",
    })
    source_config = json.loads(Path(args.config).read_text())
    run_config = {
        "format": PROFILE_FORMAT,
        "model": model_config.to_dict(),
        "profile": {
            "batch_size": args.batch_size,
            "evaluation_batch_size": args.evaluation_batch_size or args.batch_size,
            "evaluation_batches": args.evaluation_batches,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "betas": list(args.betas),
            "eps": args.eps,
            "grad_clip": args.grad_clip,
            "lambda_balance": args.lambda_balance,
            "lambda_route": args.lambda_route,
            "lambda_state": args.lambda_state,
            "requested_dtype": args.dtype,
            "dtype": dtype_name,
            "device": str(device),
            "execution_mode": args.execution,
            "compile_mode": args.compile_mode if args.execution == "compile" else None,
            "dry_run": args.dry_run,
            "complete_evaluation_batch_size": args.complete_evaluation_batch_size,
            "structured_limit_per_split": args.structured_limit_per_split,
            "code_generation_limit": args.code_generation_limit,
            "max_new_tokens": args.max_new_tokens,
            "code_max_new_tokens": args.code_max_new_tokens,
            "eos_token_id": args.eos_token_id,
            "structured_directory": str(complete_inputs["structured"]) if complete_inputs["structured"] else None,
            "code_generation": str(complete_inputs["code"]) if complete_inputs["code"] else None,
            "tokenizer": str(complete_inputs["tokenizer"]) if complete_inputs["tokenizer"] else None,
            "reconstruction_replay": str(complete_inputs["replay_path"]) if complete_inputs["replay_path"] else None,
            "reconstruction_replay_sha256": complete_inputs["replay"].get("spec_sha256") if complete_inputs["replay"] else None,
            "token_rule": "count labels[:, 1:] != -100 on actual PackedDataset batches",
        },
        "source_config_sha256": sha256_file(args.config),
        "data_identity": data_identity["identity"],
        "dev_data_identity": identity_hash(data_identity["splits"]["dev"]),
        "order_identity": order_identity,
        "code_identity": code_identity,
    }
    report.update({
        "config": {
            "name": source_config.get("name", Path(args.config).stem),
            "path": str(Path(args.config).resolve()),
            "sha256": run_config["source_config_sha256"],
            "identity": identity_hash(run_config),
            "model": model_config.to_dict(),
            "profile": run_config["profile"],
        },
        "data": data_identity,
        "code": {
            "identity": code_identity,
            "git_commit_at_attempt_start": _git_commit(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "order_identity": order_identity,
        "batch_size": args.batch_size,
        "context_length": model_config.context_length,
        "dtype": dtype_name,
        "execution": {
            "mode": args.execution,
            "compile_mode": run_config["profile"]["compile_mode"],
            "parameters_dtype": "float32",
            "autocast": dtype_name if device.type == "cuda" and dtype != torch.float32 else None,
        },
    })
    report["setup"] = {"data_preparation_seconds": time.perf_counter() - preparation_started}
    _check_reservation(reservation)

    _synchronize(device)
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = NativeModel(model_config)
    model.to(device=device)
    optimizer = T.build_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        eps=args.eps,
    )
    scheduler = T.build_scheduler(optimizer, args.warmup_steps + args.measured_steps)
    model.train()
    _synchronize(device)
    report["setup"]["model_loading_seconds"] = time.perf_counter() - started

    _synchronize(device)
    started = time.perf_counter()
    if args.execution == "compile":
        callable_model = torch.compile(model, mode=args.compile_mode)
    else:
        callable_model = model
    _synchronize(device)
    report["setup"]["compilation_seconds"] = time.perf_counter() - started
    report["setup"]["lazy_compilation_policy"] = (
        "torch.compile invocation is reported separately; lazy first-use compilation remains in warmup_seconds"
        if args.execution == "compile" else "eager: no compilation"
    )

    warmup_records = _run_training_steps(
        callable_model,
        optimizer,
        scheduler,
        datasets["train"],
        args,
        dtype,
        device,
        reservation,
        first_step=0,
        count=args.warmup_steps,
    )
    warmup = summarize_steps(warmup_records)
    report["setup"]["warmup_seconds"] = warmup["wall_seconds"]
    report["setup"]["warmup_steps"] = args.warmup_steps
    report["setup"]["first_warmup_step_seconds"] = warmup_records[0]["wall_seconds"]
    report["setup"]["warmup_actual_shifted_loss_tokens"] = warmup["actual_shifted_loss_tokens"]
    cache_batch, _, cache_data_seconds = _load_batch(
        datasets["train"],
        _batch_indices(args.warmup_steps, args.batch_size, len(datasets["train"])),
        device,
    )
    atol, rtol = _cache_tolerances(args, dtype)
    _synchronize(device)
    started = time.perf_counter()
    report["correctness"] = check_full_vs_cached(
        callable_model,
        cache_batch,
        dtype=dtype,
        atol=atol,
        rtol=rtol,
    )
    _synchronize(device)
    report["correctness"]["wall_seconds"] = time.perf_counter() - started
    report["correctness"]["data_loading_seconds"] = cache_data_seconds
    del cache_batch
    if not report["correctness"]["valid"]:
        raise AssertionError("full-forward and cached next-token logits/routes differ; timings are invalid")
    _check_reservation(reservation)

    measured_records = _run_training_steps(
        callable_model,
        optimizer,
        scheduler,
        datasets["train"],
        args,
        dtype,
        device,
        reservation,
        first_step=args.warmup_steps,
        count=args.measured_steps,
    )
    report["training"] = summarize_steps(measured_records)
    report["training"]["warmup_steps_excluded"] = args.warmup_steps
    report["evaluation"] = _teacher_forced_evaluation(
        callable_model,
        datasets["dev"],
        args,
        dtype,
        device,
        reservation,
    )
    _check_reservation(reservation)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(args.output).with_suffix(".checkpoint.pt")
    _synchronize(device)
    started = time.perf_counter()
    descriptor = C.save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        step=args.warmup_steps + args.measured_steps,
        cursor=(args.warmup_steps + args.measured_steps) * args.batch_size,
        consumed_loss_tokens=warmup["actual_shifted_loss_tokens"] + report["training"]["actual_shifted_loss_tokens"],
        config=run_config,
        code_identity=code_identity,
        data_identity=data_identity["identity"],
        order_identity=order_identity,
        identities={"dev_data": identity_hash(data_identity["splits"]["dev"])},
        extra={
            "profile": str(Path(args.output).resolve()),
            "mode": args.mode,
            "reservation_id": reservation.id if reservation is not None else None,
            "measurement_checkpoint": True,
        },
    )
    _synchronize(device)
    checkpoint_seconds = time.perf_counter() - started
    actual_bytes = checkpoint_path.stat().st_size
    if actual_bytes != descriptor["bytes"]:
        raise AssertionError("durable checkpoint bytes differ from its descriptor")
    report["checkpoint"] = {
        **descriptor,
        "wall_seconds": checkpoint_seconds,
        "bytes": actual_bytes,
        "real_write": True,
    }
    report["complete_evaluation"] = _complete_evaluation(
        model, datasets, complete_inputs, args, device, reservation
    )
    report["reconstruction"] = _reconstruction(
        descriptor, complete_inputs["replay"], args, reservation
    )
    if complete_inputs["replay"] is not None and not report["reconstruction"]["valid"]:
        raise AssertionError("fixed checkpoint reconstruction disagrees; profile timings are invalid")
    report["memory"] = _memory(device)
    report["hardware"]["foreign_gpu_processes_end"] = (
        _foreign_gpu_processes() if device.type == "cuda" else None
    )
    report["setup"]["fixed_seconds"] = math.fsum((
        report["setup"]["data_preparation_seconds"],
        report["setup"]["model_loading_seconds"],
        report["setup"]["compilation_seconds"],
        report["setup"]["warmup_seconds"],
        report["correctness"]["wall_seconds"],
        report["correctness"]["data_loading_seconds"],
    ))
    report["setup"]["complete_evaluation_seconds"] = report["complete_evaluation"]["total_seconds"]
    report["setup"]["reconstruction_seconds"] = report["reconstruction"]["seconds"]
    report["setup"]["post_checkpoint_seconds"] = math.fsum((
        report["setup"]["complete_evaluation_seconds"],
        report["setup"]["reconstruction_seconds"],
    ))
    report["setup"]["fixed_accounting"] = (
        "fixed_seconds covers data preparation/loading/warmup/cache correctness only; "
        "complete evaluation and reconstruction are disjoint post-checkpoint components"
    )
    report["checkpoint_seconds"] = checkpoint_seconds
    report["checkpoint_bytes"] = actual_bytes
    report["actual_shifted_loss_tokens_per_second"] = report["training"]["actual_shifted_loss_tokens_per_second"]
    report.update(report["memory"])
    report["status"] = "dry_run" if args.dry_run else "ok"
    report["timing_valid"] = (
        not args.dry_run
        and device.type == "cuda"
        and report["correctness"]["valid"]
        and report["complete_evaluation"]["valid"]
        and report["reconstruction"]["valid"]
    )
    report["valid_for_projection"] = report["timing_valid"] and report["config"]["name"] == "engineering"
    report["evidence_boundary"] = (
        "CPU dry-run checks the pipeline only; its timings and losses are not GPU performance or learning evidence"
        if args.dry_run else
        "Measured randomly initialized pipeline costs are not learning evidence or a quality comparison"
    )


def _reservation_report(reservation) -> dict:
    return {
        "id": reservation.id if reservation is not None else None,
        "hours": reservation.elapsed_hours() if reservation is not None else None,
        "projected_hours": reservation.projected_hours if reservation is not None else None,
    }


def _write_profile(path: Path, report: Mapping) -> None:
    encoded = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    C.atomic_write(path, encoded)


def _attempt(args: argparse.Namespace, reservation) -> dict:
    output = Path(args.output)
    report = {
        "format": PROFILE_FORMAT,
        "status": "started",
        "mode": args.mode,
        "dry_run": args.dry_run,
        "timing_valid": False,
        "valid_for_projection": False,
        "reservation": _reservation_report(reservation),
    }
    try:
        _execute(args, reservation, report)
        report["reservation"] = _reservation_report(reservation)
        report["reservation_id"] = report["reservation"]["id"]
        report["reservation_hours"] = report["reservation"]["hours"]
        _write_profile(output, report)
        return report
    except BaseException as error:
        report["status"] = "failed"
        report["timing_valid"] = False
        report["valid_for_projection"] = False
        report["error"] = f"{type(error).__name__}: {error}"[:4000]
        report["reservation"] = _reservation_report(reservation)
        report["reservation_id"] = report["reservation"]["id"]
        report["reservation_hours"] = report["reservation"]["hours"]
        _write_profile(output, report)
        raise


def run_profile(args: argparse.Namespace) -> dict:
    _validate_args(args)
    if args.dry_run:
        return _attempt(args, None)
    reservation = Reservation(
        command=f"native.profile mode={args.mode} config={args.config} data={args.data} output={args.output}",
        projected_hours=args.projected_hours,
        path=Path(args.ledger_path) if args.ledger_path is not None else None,
        lock=Path(args.ledger_lock) if args.ledger_lock is not None else None,
        extra={
            "format": PROFILE_FORMAT,
            "mode": args.mode,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "execution_mode": args.execution,
        },
    )
    with reservation as active:
        return _attempt(args, active)


def _positive_number(value, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (value == 0 and not allow_zero):
        comparison = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparison}")
    return float(value)


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _projection_profiles(profiles) -> dict[str, Mapping]:
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("profiles must be a nonempty mapping from arm names to measured profiles")
    result = {}
    for mode, profile in profiles.items():
        if mode not in MODES:
            raise ValueError(f"unknown native arm {mode!r}")
        if not isinstance(profile, Mapping):
            raise ValueError(f"{mode} profile must be a mapping")
        result[mode] = profile
    return result


def measured_feasibility(
    profiles,
    token_budget_per_arm: int,
    native_remaining_hours: float | Mapping,
    *,
    evaluation_tokens_per_arm: int | None = None,
    evaluation_passes_per_arm: int = 1,
    checkpoint_writes_per_arm: int = 1,
    planning_margin: float = DEFAULT_PLANNING_MARGIN,
    include_setup: bool = True,
    refuse: bool = True,
) -> dict:
    """Project equal shifted-token budgets from measured arm costs, or refuse the plan.

    The raw cost is multiplied by ``1 + planning_margin``. The default therefore
    reserves an additional 20% above the measured projection, and the padded total
    must fit within both the native remainder and the 12-hour work-order bound.
    """
    token_budget = _positive_integer(token_budget_per_arm, "token_budget_per_arm")
    if isinstance(native_remaining_hours, Mapping):
        native_remaining_hours = native_remaining_hours.get("native_remaining_hours")
    remaining = _positive_number(native_remaining_hours, "native_remaining_hours", allow_zero=True)
    margin = _positive_number(planning_margin, "planning_margin", allow_zero=True)
    if margin < DEFAULT_PLANNING_MARGIN:
        raise ValueError("planning_margin must be at least 0.20")
    evaluation_passes = _positive_integer(evaluation_passes_per_arm, "evaluation_passes_per_arm")
    checkpoint_writes = _positive_integer(checkpoint_writes_per_arm, "checkpoint_writes_per_arm")
    if evaluation_tokens_per_arm is not None:
        evaluation_tokens_per_arm = _positive_integer(evaluation_tokens_per_arm, "evaluation_tokens_per_arm")
    available_hours = min(remaining, MAX_NATIVE_HOURS)
    arms = {}
    total_seconds = 0.0
    total_checkpoint_bytes = 0
    for mode, profile in _projection_profiles(profiles).items():
        if profile.get("format") != PROFILE_FORMAT:
            raise ValueError(f"{mode}: unsupported or missing profile format")
        if profile.get("dry_run") or not profile.get("timing_valid") or profile.get("status") != "ok":
            raise ValueError(f"{mode}: only successful, cache-validated CUDA profiles can project a run")
        if not profile.get("correctness", {}).get("valid"):
            raise ValueError(f"{mode}: cache correctness is not valid")
        if profile.get("config", {}).get("name") != "engineering":
            raise ValueError(f"{mode}: feasibility requires engineering.json arm profiles")
        if not profile.get("valid_for_projection"):
            raise ValueError(f"{mode}: profile is not valid for feasibility projection")
        complete = profile.get("complete_evaluation", {})
        reconstruction = profile.get("reconstruction", {})
        if not complete.get("valid") or not reconstruction.get("valid"):
            raise ValueError(f"{mode}: complete evaluation and reconstruction must be valid")
        complete_components = {
            name: _positive_number(complete[f"{name}_seconds"], f"{mode} {name} seconds")
            for name in ("teacher_forced", "structured_generation", "code_generation", "state_interventions")
        }
        complete_seconds = math.fsum(complete_components.values())
        reconstruction_seconds = _positive_number(
            reconstruction["seconds"], f"{mode} reconstruction seconds"
        )
        training = profile["training"]
        measured_tokens = _positive_integer(training["actual_shifted_loss_tokens"], f"{mode} measured tokens")
        measured_training_seconds = _positive_number(training["wall_seconds"], f"{mode} training wall_seconds")
        throughput = measured_tokens / measured_training_seconds
        training_seconds = token_budget / throughput
        evaluation = profile["evaluation"]
        evaluation_tokens = evaluation_tokens_per_arm or _positive_integer(
            evaluation["actual_shifted_loss_tokens"], f"{mode} evaluation tokens"
        )
        measured_evaluation_tokens = _positive_integer(
            evaluation["actual_shifted_loss_tokens"], f"{mode} measured evaluation tokens"
        )
        evaluation_wall = _positive_number(evaluation["wall_seconds"], f"{mode} evaluation wall_seconds")
        evaluation_seconds = evaluation_passes * evaluation_tokens * evaluation_wall / measured_evaluation_tokens
        checkpoint = profile["checkpoint"]
        checkpoint_wall = _positive_number(checkpoint["wall_seconds"], f"{mode} checkpoint wall_seconds")
        checkpoint_seconds = checkpoint_writes * checkpoint_wall
        checkpoint_bytes = checkpoint_writes * _positive_integer(checkpoint["bytes"], f"{mode} checkpoint bytes")
        setup_seconds = _positive_number(
            profile["setup"]["fixed_seconds"], f"{mode} fixed setup seconds", allow_zero=True
        ) if include_setup else 0.0
        total = (
            setup_seconds + training_seconds + evaluation_seconds + complete_seconds
            + reconstruction_seconds + checkpoint_seconds
        )
        arms[mode] = {
            "equal_requested_shifted_loss_tokens": token_budget,
            "measured_actual_shifted_loss_tokens_per_second": throughput,
            "measured_training_seconds": measured_training_seconds,
            "measured_training_tokens": measured_tokens,
            "setup_seconds": setup_seconds,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "training_dev_evaluation_seconds": evaluation_seconds,
            "evaluation_tokens_per_pass": evaluation_tokens,
            "evaluation_passes": evaluation_passes,
            "complete_evaluation_seconds": complete_seconds,
            "complete_evaluation_components_seconds": complete_components,
            "reconstruction_seconds": reconstruction_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "checkpoint_writes": checkpoint_writes,
            "checkpoint_bytes": checkpoint_bytes,
            "total_seconds": total,
            "total_hours": total / 3600.0,
        }
        total_seconds += total
        total_checkpoint_bytes += checkpoint_bytes
    raw_hours = total_seconds / 3600.0
    projected_hours = raw_hours * (1.0 + margin)
    slack_hours = available_hours - projected_hours
    fits = slack_hours >= 0.0
    projection = {
        "format": "native-feasibility-v1",
        "basis": "measured per-arm shifted-token throughput plus setup, training-dev passes, complete generation/interventions, reconstruction, and durable checkpoints; no MAC-derived speed",
        "equal_token_budget_per_arm": token_budget,
        "arms": arms,
        "measured_total_seconds": total_seconds,
        "measured_total_hours": raw_hours,
        "planning_margin": margin,
        "planning_multiplier": 1.0 + margin,
        "projected_reservation_hours": projected_hours,
        "native_remaining_hours": remaining,
        "native_cap_hours": MAX_NATIVE_HOURS,
        "available_hours": available_hours,
        "slack_hours_after_margin": slack_hours,
        "total_checkpoint_bytes": total_checkpoint_bytes,
        "fits": fits,
    }
    if refuse and not fits:
        error = FeasibilityError(
            f"measured plan needs {projected_hours:.6f} reservation hours including {margin:.0%} planning margin, "
            f"but only {available_hours:.6f} hours are available",
            projection,
        )
        raise error
    return projection


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m native.profile", description=__doc__.split("\n\n")[0])
    result.add_argument("--config", required=True, help="engineering.json or serious.json")
    result.add_argument("--mode", required=True, choices=MODES)
    result.add_argument("--data", required=True, help="PackedDataset root containing train/ and dev/")
    result.add_argument("--output", required=True, help="atomic JSON profile path")
    result.add_argument("--checkpoint", help="checkpoint path; default replaces the JSON suffix with .checkpoint.pt")
    result.add_argument("--device", default="cuda")
    result.add_argument("--dry-run", action="store_true", help="CPU pipeline check; never opens a reservation or validates timing")
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--warmup-steps", type=int, default=5)
    result.add_argument("--measured-steps", type=int, default=20)
    result.add_argument("--evaluation-batches", type=int, default=1, help="0 evaluates the complete dev split")
    result.add_argument("--evaluation-batch-size", type=int, default=None)
    result.add_argument("--complete-evaluation-batch-size", type=int, default=8)
    result.add_argument("--structured", "--structured-directory", dest="structured", default=None,
                        help="four-split structured JSONL directory; defaults to DATA/structured when present")
    result.add_argument("--code-generation", default=None,
                        help="code-suffix JSONL; defaults to DATA/code_generation.jsonl when present")
    result.add_argument("--tokenizer", default=None, help="hash-pinned tokenizer.json required by generation inputs")
    result.add_argument("--structured-limit-per-split", type=int, default=1,
                        help="fixed first-N generation sample per split; 0 measures every record")
    result.add_argument("--code-generation-limit", type=int, default=1,
                        help="fixed first-N code-suffix sample; 0 measures every record")
    result.add_argument("--max-new-tokens", type=int, default=48)
    result.add_argument("--code-max-new-tokens", type=int, default=128)
    result.add_argument("--eos-token-id", type=int, default=50256)
    result.add_argument("--reconstruction-replay", "--replay-spec", dest="reconstruction_replay", default=None,
                        help="fixed native-reconstruction-replay-v1 JSON; defaults to DATA/replay.json when present")
    result.add_argument("--reconstruction-output", default=None,
                        help="replay report path; default replaces the profile suffix with .reconstruction.json")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=0.01)
    result.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    result.add_argument("--eps", type=float, default=1e-8)
    result.add_argument("--grad-clip", type=float, default=1.0)
    result.add_argument("--lambda-balance", type=float, default=0.01)
    result.add_argument("--lambda-route", type=float, default=0.1)
    result.add_argument("--lambda-state", type=float, default=0.1)
    result.add_argument("--dtype", choices=DTYPES, default="bfloat16", help="CUDA autocast and correctness-check dtype")
    result.add_argument("--execution", choices=EXECUTION_MODES, default="eager", help="eager is explicit; compile uses torch.compile")
    result.add_argument("--compile-mode", default="default", help="torch.compile mode when --execution compile is selected")
    result.add_argument("--cache-atol", type=float, default=None)
    result.add_argument("--cache-rtol", type=float, default=None)
    result.add_argument("--projected-hours", type=float, default=1.0)
    result.add_argument("--ledger-path", default=None, help="test ledger override; CPU dry-run never opens a reservation")
    result.add_argument("--ledger-lock", default=None, help="test lock override; CPU dry-run never opens a reservation")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    report = run_profile(args)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "status": report["status"],
        "mode": report["mode"],
        "timing_valid": report["timing_valid"],
        "actual_shifted_loss_tokens_per_second": report["actual_shifted_loss_tokens_per_second"],
        "checkpoint_seconds": report["checkpoint_seconds"],
        "checkpoint_bytes": report["checkpoint_bytes"],
        "reservation": report["reservation"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
