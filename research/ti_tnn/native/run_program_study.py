"""Prepare or execute the frozen varied-program study under one native reservation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import torch

from . import evaluate_program_study
from . import ledger
from . import reconstruct
from . import train
from .data import PackedDataset, sha256_file
from .independent import rescore_generations

ARMS = ("dense", "untyped", "typed")
SPLITS = ("train", "dev", "initial_values", "unseen_composition", "longer_length")
EVALUATION_COUNT = 128
INTERVENTION_COUNT = 32
LOSS_TOKENS_PER_EPOCH = 348165
UPDATES_PER_EPOCH = 212
EPOCHS = 16
SHUTDOWN_ALLOWANCE_HOURS = 0.05
WATCHDOG_MULTIPLIER = 1.5
HOUR_TOLERANCE = 1e-12
RUN_FORMAT = "native-program-study-run-status-v1"
LAUNCH_FORMAT = "native-program-study-launch-v1"

EXPECTED_PROTOCOL = {
    "arms": list(ARMS),
    "max_updates": 3392,
    "max_loss_tokens": 5570640,
    "batch_size": 8,
    "seed": 20261112,
    "model_seed": 20261112,
    "data_seed": 20261113,
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
    "dev_every": 500,
    "dev_batches": 0,
    "max_new_tokens": 48,
}
INTEGER_FIELDS = {
    "max_updates", "max_loss_tokens", "batch_size", "seed", "model_seed", "data_seed",
    "checkpoint_every", "dev_every", "dev_batches", "max_new_tokens",
}


def _write(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_bytes(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _positive_hours(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def validate_protocol(protocol):
    for name, expected in EXPECTED_PROTOCOL.items():
        if name not in protocol:
            raise ValueError(f"study protocol is missing {name}")
        actual = protocol[name]
        if name in INTEGER_FIELDS and (isinstance(actual, bool) or not isinstance(actual, int)):
            raise ValueError(f"study protocol {name} must be an integer")
        if actual != expected:
            raise ValueError(f"study protocol {name} must be {expected!r}, not {actual!r}")
    if protocol.get("position_policy", "document") != "document":
        raise ValueError("the varied-program study requires position_policy='document'")
    _positive_hours(protocol.get("projected_total_hours"), "projected_total_hours")
    _positive_hours(protocol.get("allocation_request_hours"), "allocation_request_hours")


def projection(protocol, before, endpoint=None):
    """Return the complete budget projection without writing or opening a reservation."""
    validate_protocol(protocol)
    remaining = before.get("native_remaining_hours")
    if isinstance(remaining, bool) or not isinstance(remaining, (int, float)) \
            or not math.isfinite(remaining) or remaining < 0:
        raise ValueError("live native_remaining_hours must be a finite nonnegative number")
    projected = float(protocol["projected_total_hours"])
    allocation = float(protocol["allocation_request_hours"])
    watchdog = projected * WATCHDOG_MULTIPLIER
    required = watchdog + SHUTDOWN_ALLOWANCE_HOURS
    fits_allocation = required <= allocation + HOUR_TOLERANCE
    fits_live = required <= remaining + HOUR_TOLERANCE
    result = {
        "arms": list(ARMS),
        "epochs_per_arm": EPOCHS,
        "loss_tokens_per_epoch": LOSS_TOKENS_PER_EPOCH,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "updates_per_arm": protocol["max_updates"],
        "loss_tokens_per_arm": protocol["max_loss_tokens"],
        "total_updates": protocol["max_updates"] * len(ARMS),
        "total_loss_tokens": protocol["max_loss_tokens"] * len(ARMS),
        "evaluation_splits": list(SPLITS),
        "evaluation_examples_per_split": EVALUATION_COUNT,
        "intervention_examples_per_split": INTERVENTION_COUNT,
        "projected_total_hours": projected,
        "allocation_request_hours": allocation,
        "maximum_watchdog_hours": watchdog,
        "shutdown_allowance_hours": SHUTDOWN_ALLOWANCE_HOURS,
        "required_native_hours": required,
        "live_native_remaining_hours": float(remaining),
        "allocation_headroom_hours": allocation - required,
        "live_native_headroom_hours": float(remaining) - required,
        "fits_allocation_request": fits_allocation,
        "fits_live_native_remaining": fits_live,
        "fits_budget": fits_allocation and fits_live,
    }
    if endpoint is not None:
        updates, tokens = endpoint
        result["sampler_planned_updates"] = updates
        result["sampler_planned_loss_tokens"] = tokens
        result["sampler_endpoint_matches"] = (
            updates == protocol["max_updates"] and tokens == protocol["max_loss_tokens"]
        )
    return result


def require_budget(plan):
    if not plan["fits_allocation_request"]:
        raise RuntimeError(
            "projected_total_hours * 1.5 plus the 0.05 h shutdown allowance exceeds "
            "allocation_request_hours"
        )
    if not plan["fits_live_native_remaining"]:
        raise RuntimeError(
            "projected_total_hours * 1.5 plus the 0.05 h shutdown allowance exceeds "
            "the live native remaining hours"
        )


def _authorization(protocol, execute, authorize_hours):
    if not execute:
        if authorize_hours is not None:
            raise RuntimeError("--authorize-hours requires --execute; the default dry-run takes neither flag")
        return None
    if authorize_hours is None:
        raise RuntimeError("execution requires both --execute and --authorize-hours")
    _positive_hours(authorize_hours, "--authorize-hours")
    if authorize_hours != protocol["allocation_request_hours"]:
        raise RuntimeError("--authorize-hours must exactly equal protocol allocation_request_hours")
    return float(authorize_hours)


def _validate_manifest(manifest):
    if manifest.get("position_policy") != "document":
        raise ValueError("the varied-program manifest requires position_policy='document'")
    expected_counts = {"train": 4096, "dev": 128, "initial_values": 128,
                       "unseen_composition": 128, "longer_length": 128}
    splits = manifest.get("splits", {})
    counts = manifest.get("counts", {})
    for split, expected in expected_counts.items():
        values = splits.get(split)
        if not isinstance(values, dict) or values.get("examples") != expected:
            raise ValueError(f"manifest {split} must contain exactly {expected} source examples")
        if values.get("position_policy", "document") != "document":
            raise ValueError(f"manifest {split} positions differ from the document policy")
        if split in counts and counts[split] != expected:
            raise ValueError(f"manifest counts disagree for {split}")
    epoch = manifest.get("train_epoch", {})
    if (epoch.get("loss_tokens_per_epoch") != LOSS_TOKENS_PER_EPOCH
            or epoch.get("updates_per_epoch") != UPDATES_PER_EPOCH
            or epoch.get("batch_size") != EXPECTED_PROTOCOL["batch_size"]):
        raise ValueError(
            "manifest train_epoch must declare 348165 loss tokens and 212 updates at batch size 8"
        )


def _validate_frozen_identity(protocol, protocol_bytes, root, repository, tokenizer):
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    sidecar = root / "study_protocol.sha256"
    if not sidecar.is_file() or sidecar.read_text().strip() != protocol_sha256:
        raise RuntimeError("study_protocol.sha256 does not match the actual protocol bytes")

    source_identities = protocol.get("source_files_sha256")
    if not isinstance(source_identities, dict) or not source_identities:
        raise ValueError("study protocol source_files_sha256 must be a nonempty object")
    native_root = (repository / "research/ti_tnn/native").resolve()
    for name, expected in source_identities.items():
        source_path = (native_root / name).resolve()
        try:
            source_path.relative_to(native_root)
        except ValueError as error:
            raise ValueError(f"frozen source path escapes the native source directory: {name}") from error
        if not source_path.is_file() or sha256_file(source_path) != expected:
            raise RuntimeError(f"frozen source file identity mismatch: {name}")

    data_identity = protocol.get("data")
    if not isinstance(data_identity, dict):
        raise ValueError("study protocol data must be an object")
    if Path(data_identity.get("root", "")).resolve() != root:
        raise RuntimeError("the supplied data root differs from the frozen protocol data root")
    for key, filename in (("manifest_sha256", "manifest.json"),
                          ("audit_sha256", "audit.json"),
                          ("selections_sha256", "selections.json")):
        path = root / filename
        if not path.is_file() or sha256_file(path) != data_identity.get(key):
            raise RuntimeError(f"frozen data identity mismatch: {filename}")

    expected_tokenizer = data_identity.get("tokenizer_path")
    if not isinstance(expected_tokenizer, str) or Path(expected_tokenizer).resolve() != tokenizer:
        raise RuntimeError("the supplied tokenizer path differs from the frozen protocol")
    tokenizer_sha256 = sha256_file(tokenizer)
    if tokenizer_sha256 != data_identity.get("tokenizer_file_sha256"):
        raise RuntimeError("the supplied tokenizer hash differs from the frozen protocol")
    nested_tokenizer = data_identity.get("tokenizer", {})
    if nested_tokenizer.get("path") and Path(nested_tokenizer["path"]).resolve() != tokenizer:
        raise RuntimeError("the supplied tokenizer path differs from the nested protocol identity")
    if nested_tokenizer.get("sha256") and nested_tokenizer["sha256"] != tokenizer_sha256:
        raise RuntimeError("the supplied tokenizer hash differs from the nested protocol identity")
    return protocol_sha256


def planned_endpoint(root, protocol):
    """Derive the atomic sampler endpoint from the packed training arrays."""
    sampler = train.make_sampler(
        PackedDataset(root, "train"), protocol["batch_size"], protocol["data_seed"]
    )
    endpoint = train._planned_budget(sampler, protocol["max_loss_tokens"])
    expected = (protocol["max_updates"], protocol["max_loss_tokens"])
    if endpoint != expected:
        raise RuntimeError(
            f"the packed sampler yields {endpoint}, not the required update/token endpoint {expected}"
        )
    return endpoint


def _arguments(protocol, root, arm, mode, projected_hours):
    repository = Path(train.__file__).resolve().parents[3]
    return [
        "--config", str(repository / "research/ti_tnn/native/configs/engineering.json"),
        "--mode", mode,
        "--data", str(root),
        "--output", str(arm / "train"),
        "--device", "cuda",
        "--max-loss-tokens", str(protocol["max_loss_tokens"]),
        "--batch-size", str(protocol["batch_size"]),
        "--seed", str(protocol["seed"]),
        "--model-seed", str(protocol["model_seed"]),
        "--data-seed", str(protocol["data_seed"]),
        "--learning-rate", str(protocol["learning_rate"]),
        "--weight-decay", str(protocol["weight_decay"]),
        "--betas", *map(str, protocol["betas"]),
        "--eps", str(protocol["eps"]),
        "--grad-clip", str(protocol["grad_clip"]),
        "--lambda-balance", str(protocol["lambda_balance"]),
        "--lambda-route", str(protocol["lambda_route"]),
        "--lambda-state", str(protocol["lambda_state"]),
        "--warmup-fraction", str(protocol["warmup_fraction"]),
        "--final-lr-fraction", str(protocol["final_lr_fraction"]),
        "--checkpoint-every", str(protocol["checkpoint_every"]),
        "--dev-every", str(protocol["dev_every"]),
        "--dev-batches", str(protocol["dev_batches"]),
        "--projected-hours", str(projected_hours),
    ]


def _training_checkpoint(protocol, arm, mode, payload):
    checkpoint = payload.get("final_checkpoint", {})
    expected_directory = (arm / "train" / "checkpoints").resolve()
    checkpoint_path = Path(checkpoint.get("path", "")).resolve()
    if (payload.get("format") != train.TRAIN_FORMAT
            or payload.get("mode") != mode
            or payload.get("step") != protocol["max_updates"]
            or payload.get("consumed_loss_tokens") != protocol["max_loss_tokens"]
            or payload.get("planned_updates") != protocol["max_updates"]
            or checkpoint.get("step") != protocol["max_updates"]
            or checkpoint_path.parent != expected_directory
            or checkpoint_path.name != f"final-update-{protocol['max_updates']:08d}.pt"):
        raise RuntimeError(f"{mode}: final training endpoint or checkpoint identity does not match the protocol")
    actual_sha256 = sha256_file(checkpoint_path)
    if checkpoint.get("sha256") != actual_sha256:
        raise RuntimeError(f"{mode}: final checkpoint SHA256 does not match its training descriptor")
    return {**checkpoint, "path": str(checkpoint_path), "sha256": actual_sha256}


def _validate_evaluation(evaluation, checkpoint):
    if evaluation.get("format") != evaluate_program_study.FORMAT:
        raise RuntimeError("program-study evaluation has an unexpected format")
    if evaluation.get("checkpoint", {}).get("sha256") != checkpoint["sha256"]:
        raise RuntimeError("program-study evaluation used a different checkpoint hash")
    if evaluation.get("checkpoint", {}).get("step") != checkpoint["step"]:
        raise RuntimeError("program-study evaluation used a different checkpoint endpoint")
    splits = evaluation.get("splits", {})
    if set(splits) != set(SPLITS):
        raise RuntimeError(f"program-study evaluation must contain exactly {list(SPLITS)}")
    exact = {}
    for split in SPLITS:
        values = splits[split]
        generation = values.get("generation", {})
        interventions = values.get("interventions", {})
        selection = values.get("selection", {})
        if (not generation.get("complete")
                or generation.get("evaluated") != EVALUATION_COUNT
                or selection.get("examples") != EVALUATION_COUNT):
            raise RuntimeError(f"{split}: incomplete 128-example program-study generation")
        if (interventions.get("selection_examples") != INTERVENTION_COUNT
                or len(interventions.get("records", [])) != INTERVENTION_COUNT
                or selection.get("interventions", {}).get("examples") != INTERVENTION_COUNT):
            raise RuntimeError(f"{split}: incomplete 32-example targeted intervention evaluation")
        if not values.get("state_path", {}).get("complete"):
            raise RuntimeError(f"{split}: incomplete state-path evaluation")
        exact[split] = generation.get("correct")
    return exact


def _validate_generation_diagnostics(diagnostics, evaluation, mode):
    if diagnostics.get("evaluation_format") != evaluate_program_study.FORMAT:
        raise RuntimeError(f"{mode}: independent rescore did not use the program-study evaluation schema")
    if set(diagnostics.get("arms", {})) != {mode}:
        raise RuntimeError(f"{mode}: independent rescore must contain only the current arm")
    arm = diagnostics["arms"][mode]
    structured = arm.get("structured", {})
    supplemental = arm.get("supplemental_baseline_generation", {})
    interventions = arm.get("interventions", {})
    if any(set(values) != set(SPLITS) for values in (structured, supplemental, interventions)):
        raise RuntimeError(f"{mode}: independent rescore must contain exactly {list(SPLITS)}")

    def validate_summary(summary, expected, kind, split, per_operation=None):
        if summary.get("examples") != expected or len(summary.get("records", [])) != expected:
            raise RuntimeError(f"{mode}: {split}: incomplete {expected}-example independent {kind} rescore")
        prerequisites = summary.get("prerequisites", {})
        if (prerequisites.get("complete") != expected
                or prerequisites.get("truncated_or_incomplete") != 0
                or any(not record.get("prerequisites_complete") for record in summary["records"])):
            raise RuntimeError(f"{mode}: {split}: incomplete prerequisites for independent {kind} rescore")
        if summary.get("score_changes"):
            raise RuntimeError(
                f"{mode}: {split}: independent rescore changed scores: {summary['score_changes'][:2]}"
            )
        if per_operation is not None:
            by_operation = summary.get("by_final_operation", {})
            if (set(by_operation) != set(evaluate_program_study.OPERATIONS)
                    or any(values.get("examples") != per_operation for values in by_operation.values())):
                raise RuntimeError(
                    f"{mode}: {split}: independent {kind} rescore requires {per_operation} examples per final operation"
                )

    for split in SPLITS:
        split_evaluation = evaluation["splits"][split]
        validate_summary(
            structured[split], EVALUATION_COUNT, "generation", split,
            per_operation=evaluate_program_study.PER_OPERATION_COUNT,
        )
        if structured[split].get("exact_count") != split_evaluation["generation"]["correct"]:
            raise RuntimeError(f"{mode}: {split}: independent exact count differs from the evaluation summary")

        intervention_evaluation = split_evaluation["interventions"]
        selection = split_evaluation["selection"]["interventions"]
        expected_supplemental = selection.get("additional_ordinary_baselines")
        if (isinstance(expected_supplemental, bool) or not isinstance(expected_supplemental, int)
                or not 0 <= expected_supplemental <= INTERVENTION_COUNT):
            raise RuntimeError(f"{mode}: {split}: invalid supplemental baseline evaluation count")
        supplemental_evaluation = intervention_evaluation.get("supplemental_baseline_generation", {})
        if (selection.get("reused_evaluation_baselines") != INTERVENTION_COUNT - expected_supplemental
                or supplemental_evaluation.get("examples") != expected_supplemental
                or supplemental_evaluation.get("evaluated", expected_supplemental) != expected_supplemental
                or (expected_supplemental and not supplemental_evaluation.get("complete"))):
            raise RuntimeError(f"{mode}: {split}: inconsistent supplemental baseline evaluation counts")
        validate_summary(supplemental[split], expected_supplemental, "supplemental baseline", split)

        intervention_diagnostics = interventions[split]
        if intervention_diagnostics.get("examples") != INTERVENTION_COUNT:
            raise RuntimeError(f"{mode}: {split}: incomplete 32-example independent intervention rescore")
        validate_summary(
            intervention_diagnostics.get("intervened", {}), INTERVENTION_COUNT,
            "intervened generation", split,
            per_operation=evaluate_program_study.INTERVENTIONS_PER_OPERATION,
        )
        if intervention_diagnostics.get("score_changes"):
            raise RuntimeError(
                f"{mode}: {split}: independent rescore changed scores: "
                f"{intervention_diagnostics['score_changes'][:2]}"
            )
        for key, label in (("changed_completions", "changed completion"),
                           ("changed_generated_suffixes", "changed generated suffix")):
            if intervention_diagnostics.get(key) != intervention_evaluation.get(key):
                raise RuntimeError(f"{mode}: {split}: independent {label} count differs from the evaluation")
        by_operation = intervention_diagnostics.get("by_final_operation", {})
        reported_by_operation = intervention_evaluation.get("by_final_operation", {})
        if set(by_operation) != set(evaluate_program_study.OPERATIONS):
            raise RuntimeError(f"{mode}: {split}: incomplete independent intervention operation groups")
        for operation in evaluate_program_study.OPERATIONS:
            independent_operation = by_operation[operation]
            reported_operation = reported_by_operation.get(operation, {})
            if independent_operation.get("examples") != evaluate_program_study.INTERVENTIONS_PER_OPERATION:
                raise RuntimeError(f"{mode}: {split}: interventions require 8 examples per final operation")
            for key in ("changed_completions", "changed_generated_suffixes"):
                if independent_operation.get(key) != reported_operation.get(key):
                    raise RuntimeError(
                        f"{mode}: {split}: independent {operation} {key} count differs from the evaluation"
                    )


def execute_arm(protocol, root, arm, mode, tokenizer, replay, *, on_stage=None):
    """Train, evaluate, independently rescore, and reconstruct one arm."""
    arm = Path(arm)

    def stage(name, **fields):
        if on_stage is not None:
            on_stage(name, **fields)

    payload = train.run_training(train.parser().parse_args(
        _arguments(protocol, root, arm, mode, protocol["projected_total_hours"])
    ))
    _write(arm / "train-result.json", payload)
    checkpoint = _training_checkpoint(protocol, arm, mode, payload)
    stage("evaluating", updates=payload["step"],
          consumed_loss_tokens=payload["consumed_loss_tokens"], checkpoint=checkpoint)
    evaluation_path = arm / "evaluation.json"
    evaluation = evaluate_program_study.evaluate(
        checkpoint=checkpoint["path"],
        expected_sha256=checkpoint["sha256"],
        data=root,
        tokenizer=tokenizer,
        output=evaluation_path,
        max_new_tokens=protocol["max_new_tokens"],
    )
    exact = _validate_evaluation(evaluation, checkpoint)
    if not evaluation_path.is_file():
        raise RuntimeError(f"{mode}: program-study evaluation did not create evaluation.json")
    evaluation_sha256 = sha256_file(evaluation_path)
    stage("rescoring", exact=exact, evaluation_sha256=evaluation_sha256)
    diagnostics_path = arm / "generation_diagnostics.json"
    diagnostics = rescore_generations.rescore(
        study=arm.parent.parent, data=root, output=diagnostics_path, modes=(mode,),
    )
    _validate_generation_diagnostics(diagnostics, evaluation, mode)
    generation_diagnostics_sha256 = sha256_file(diagnostics_path)
    stage("reconstructing", exact=exact, evaluation_sha256=evaluation_sha256,
          generation_diagnostics_sha256=generation_diagnostics_sha256)
    replay_spec = reconstruct.load_replay_spec(replay) if isinstance(replay, (str, os.PathLike)) else replay
    report = reconstruct.reconstruct_checkpoint(
        checkpoint["path"], replay_spec,
        expected_sha256=checkpoint["sha256"], device="cpu", autocast_dtype=None,
    )
    reconstruction_sha256 = reconstruct.write_reconstruction_report(arm / "reconstruction.json", report)
    execution = report.get("execution", {})
    if (execution.get("device") != "cpu"
            or execution.get("parameter_dtype") != "torch.float32"
            or execution.get("autocast_dtype") is not None):
        raise RuntimeError(f"{mode}: reconstruction was not an unautocast CPU FP32 replay")
    if not report.get("agreement"):
        raise RuntimeError(f"{mode}: FP32 full/cached reconstruction disagrees; see reconstruction.json")
    return {
        "checkpoint": checkpoint,
        "updates": payload["step"],
        "consumed_loss_tokens": payload["consumed_loss_tokens"],
        "train_seconds": payload["training_seconds"],
        "exact": exact,
        "train_exact": exact["train"],
        "dev_exact": exact["dev"],
        "evaluation_sha256": evaluation_sha256,
        "generation_diagnostics_sha256": generation_diagnostics_sha256,
        "reconstruction_sha256": reconstruction_sha256,
    }


def _recover_statuses(run_dir, statuses):
    return {
        mode: json.loads((run_dir / "arms" / mode / "status.json").read_text())
        if (run_dir / "arms" / mode / "status.json").is_file() else values
        for mode, values in statuses.items()
    }


def _execute(protocol, root, run_dir, tokenizer, replay, *, reservation=None):
    """Run arms sequentially, continuing only after ordinary per-arm failures."""
    statuses = {
        mode: {"status": "not_run", "required_updates": protocol["max_updates"],
               "required_loss_tokens": protocol["max_loss_tokens"]}
        for mode in protocol["arms"]
    }
    for mode in protocol["arms"]:
        if reservation is not None:
            reservation.check()
        started = time.perf_counter()
        arm = run_dir / "arms" / mode
        arm.mkdir(parents=True, exist_ok=False)
        statuses[mode]["status"] = "training"
        _write(arm / "status.json", statuses[mode])

        def stage(name, **fields):
            statuses[mode].update(fields)
            statuses[mode]["status"] = name
            _write(arm / "status.json", statuses[mode])

        try:
            result = execute_arm(protocol, root, arm, mode, tokenizer, replay, on_stage=stage)
            statuses[mode] = {
                **statuses[mode], **result, "status": "completed",
                "wall_seconds": time.perf_counter() - started,
            }
        except Exception as error:
            statuses[mode] = {
                **statuses[mode], "status": "failed", "stage": statuses[mode]["status"],
                "error": str(error), "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            }
        except BaseException as error:
            statuses[mode] = {
                **statuses[mode], "status": "interrupted", "stage": statuses[mode]["status"],
                "error": str(error), "kind": type(error).__name__,
                "traceback": traceback.format_exc(), "wall_seconds": time.perf_counter() - started,
            }
            _write(arm / "status.json", statuses[mode])
            raise
        finally:
            _write(arm / "status.json", statuses[mode])
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return statuses


def _new_run_id():
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _validate_run_id(run_id):
    if not isinstance(run_id, str) or not run_id or run_id in {".", ".."} \
            or Path(run_id).name != run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("--run-id must be one nonempty path component")
    return run_id


def _validate_runtime(repository, python):
    pinned = repository.parent / "universa-recurrent-native/.venv-native/bin/python"
    if Path(sys.executable).resolve() != pinned.resolve():
        raise RuntimeError(f"use the pinned native environment, not {sys.executable}")
    if python is not None and Path(python).resolve() != Path(sys.executable).resolve():
        raise RuntimeError("--python must be the interpreter running this command")
    expected_directory = repository / "research/ti_tnn/native"
    for module, name in ((train, "train.py"), (reconstruct, "reconstruct.py"),
                         (evaluate_program_study, "evaluate_program_study.py"),
                         (rescore_generations, "independent/rescore_generations.py")):
        imported = Path(module.__file__).resolve()
        if imported != expected_directory / name:
            raise RuntimeError(f"PYTHONPATH selected {imported}, not the requested source worktree")


def _source_state(repository):
    source = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository, capture_output=True, text=True, check=True,
    ).stdout
    if dirty:
        raise RuntimeError("the source-bound program study requires a clean, committed worktree")
    return source, train._code_identity()


def run(data, repository, tokenizer, run_id=None, *, execute=False, authorize_hours=None, python=None):
    root = Path(data).resolve()
    repository = Path(repository).resolve()
    tokenizer = Path(tokenizer).resolve()
    protocol_bytes = (root / "study_protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    validate_protocol(protocol)
    authorized_hours = _authorization(protocol, execute, authorize_hours)
    protocol_sha256 = _validate_frozen_identity(protocol, protocol_bytes, root, repository, tokenizer)
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    _validate_manifest(manifest)
    path, lock = ledger.validate_shared_reservation_paths()
    before = ledger.usage(path)
    endpoint = planned_endpoint(root, protocol)
    plan = projection(protocol, before, endpoint)
    data_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if not execute:
        print(json.dumps({
            "status": "dry_run", "run_id": run_id, "output_root": str(root / "runs"),
            "run_dir": str(root / "runs" / run_id) if run_id else None,
            "protocol_sha256": protocol_sha256, "data_manifest_sha256": data_manifest_sha256,
            "projection": plan, "artifacts_created": False, "reservation_created": False,
        }, sort_keys=True, allow_nan=False))
        return 0

    require_budget(plan)
    if ledger.active() is not None:
        raise RuntimeError("an active native reservation already owns this process; refusing to nest")
    _validate_runtime(repository, python)
    replay = root.parent / "ti_tnn_native" / "replay.json"
    config = repository / "research/ti_tnn/native/configs/engineering.json"
    for required in (tokenizer, replay, config):
        if not required.is_file():
            raise FileNotFoundError(required)
    tokenizer_identity = manifest.get("tokenizer", {})
    if tokenizer_identity.get("path") and Path(tokenizer_identity["path"]).resolve() != tokenizer:
        raise RuntimeError("the supplied tokenizer differs from the program-study manifest")
    if tokenizer_identity.get("sha256") and sha256_file(tokenizer) != tokenizer_identity["sha256"]:
        raise RuntimeError("the supplied tokenizer hash differs from the program-study manifest")
    replay_spec = reconstruct.load_replay_spec(replay)
    source, code_identity = _source_state(repository)
    run_id = _validate_run_id(run_id or _new_run_id())
    run_dir = root / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse an existing program-study run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_bytes(run_dir / "protocol.json", protocol_bytes)
    (run_dir / "protocol.sha256").write_text(protocol_sha256 + "\n")
    _write(run_dir / "launch.json", {
        "format": LAUNCH_FORMAT,
        "run_id": run_id,
        "source_commit": source,
        "code_identity": code_identity,
        "data": str(root),
        "protocol_sha256": protocol_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "tokenizer": str(tokenizer),
        "tokenizer_sha256": sha256_file(tokenizer),
        "replay": str(replay),
        "replay_sha256": sha256_file(replay),
        "engineering_config": str(config),
        "engineering_config_sha256": sha256_file(config),
        "position_policy": "document",
        "authorized_hours": authorized_hours,
        "ledger_before": before,
        "projection": plan,
    })
    statuses = {mode: {"status": "not_run"} for mode in protocol["arms"]}
    reservation_id = None
    failure = None
    started = time.perf_counter()
    old_run_id = os.environ.get("TI_TNN_NATIVE_ACQUISITION_RUN_ID")
    os.environ["TI_TNN_NATIVE_ACQUISITION_RUN_ID"] = run_id
    try:
        with ledger.Reservation(
                f"native.program_study run={run_id}",
                projected_hours=protocol["projected_total_hours"], path=path, lock=lock,
                extra={"native_acquisition_run_id": run_id, "source_commit": source,
                       "code_identity": code_identity, "protocol_sha256": protocol_sha256,
                       "data_manifest_sha256": data_manifest_sha256}) as active:
            reservation_id = active.id
            statuses = _execute(protocol, root, run_dir, tokenizer, replay_spec, reservation=active)
    except BaseException as error:
        failure = {"kind": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        statuses = _recover_statuses(run_dir, statuses)
    finally:
        if old_run_id is None:
            os.environ.pop("TI_TNN_NATIVE_ACQUISITION_RUN_ID", None)
        else:
            os.environ["TI_TNN_NATIVE_ACQUISITION_RUN_ID"] = old_run_id
    after = ledger.usage(path)
    charged = after["used_hours"] - before["used_hours"]
    allocation_limit_respected = charged <= authorized_hours + 1e-12
    complete = (failure is None and set(statuses) == set(protocol["arms"])
                and all(item.get("status") == "completed"
                        and item.get("updates") == protocol["max_updates"]
                        and item.get("consumed_loss_tokens") == protocol["max_loss_tokens"]
                        for item in statuses.values())
                and allocation_limit_respected)
    result = {
        "format": RUN_FORMAT,
        "run_id": run_id,
        "source_commit": source,
        "code_identity": code_identity,
        "protocol_sha256": protocol_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "complete": complete,
        "arms": statuses,
        "checkpoints": {mode: values["checkpoint"] for mode, values in statuses.items()
                        if values.get("checkpoint") is not None},
        "failure": failure,
        "reservation_id": reservation_id,
        "wall_seconds": time.perf_counter() - started,
        "charged_hours": charged,
        "authorized_hours": authorized_hours,
        "allocation_limit_respected": allocation_limit_respected,
        "projection": plan,
        "ledger_before": before,
        "ledger_after": after,
        "evidence_boundary": "execution status only; inspect all five evaluation splits, generation diagnostics, and reconstruction reports",
    }
    _write(run_dir / "run_status.json", result)
    print(json.dumps({
        "run_dir": str(run_dir), "complete": complete, "charged_hours": charged,
        "reservation_id": reservation_id, "arms": statuses, "failure": failure,
    }, sort_keys=True, allow_nan=False))
    return 0 if complete else 3


def parser():
    result = argparse.ArgumentParser(prog="python -m native.run_program_study", description=__doc__)
    result.add_argument("--data", required=True, help="root containing study_protocol.json, manifest.json, and packed splits")
    result.add_argument("--repository", required=True, help="clean source worktree used for execution")
    result.add_argument("--python", required=True, help="pinned native interpreter; must be the running interpreter")
    result.add_argument("--tokenizer", required=True)
    result.add_argument("--run-id", help="new immutable attempt identifier; existing directories are refused")
    result.add_argument("--execute", action="store_true", help="execute only with exact explicit hour authorization")
    result.add_argument("--authorize-hours", type=float)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    return run(
        args.data, args.repository, args.tokenizer, args.run_id,
        execute=args.execute, authorize_hours=args.authorize_hours, python=args.python,
    )


if __name__ == "__main__":
    raise SystemExit(main())
