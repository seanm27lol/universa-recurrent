"""Dry-run, prepare, execute, or explicitly recover the six #78 continuation children.

No mode implicitly launches training. Execution needs exact 3.0-hour owner authorization;
preparation creates forks on CPU without optimizer updates or a reservation. Each execution
uses one bounded reservation, and recovery retains prior results and charges them to the same cap.
"""
from __future__ import annotations

import argparse
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
from typing import Mapping

import torch

from . import checkpoint as C
from . import continuation
from . import diagnose_program_results
from . import evaluate_program_study
from . import ledger
from . import prepare_continuation_protocol as P
from . import reconstruct
from . import run_program_study as program_runner
from . import train
from .data import sha256_file
from .independent import rescore_generations
from .sampler import identity_hash

OWNER_ENV = "TI_TNN_NATIVE_ACQUISITION_RUN_ID"
FORK_MANIFEST_NAME = "fork_manifest.json"


def _encoded(payload) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write(path, payload) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(_encoded(payload))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_new(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_encoded(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _recovery_path(path, index: int) -> Path:
    path = Path(path)
    if isinstance(index, bool) or not isinstance(index, int) or index <= 0:
        raise ValueError("recovery index must be a positive integer")
    return path.with_name(f"{path.stem}.recovery-{index:04d}{path.suffix}")


def _next_recovery_index(output_root: Path) -> int:
    indices = []
    for name in ("launch.json", "run_status.json"):
        canonical = output_root / name
        for path in canonical.parent.glob(f"{canonical.stem}.recovery-*{canonical.suffix}"):
            try:
                indices.append(int(path.stem.rsplit("-", 1)[1]))
            except (ValueError, IndexError):
                continue
    return max(indices, default=0) + 1


def _result_candidates(path) -> list[Path]:
    path = Path(path)
    candidates = sorted(
        path.parent.glob(f"{path.stem}.recovery-*{path.suffix}"),
        key=lambda item: item.name,
        reverse=True,
    )
    if path.is_file():
        candidates.append(path)
    return candidates


def _nonnegative_integer(value, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 \
            or (maximum is not None and value > maximum):
        limit = "" if maximum is None else f" no greater than {maximum}"
        raise RuntimeError(f"{name} must be a nonnegative integer{limit}")
    return value


def _read_json(path) -> dict:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _finite_nonnegative(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def projection(protocol: Mapping, before: Mapping) -> dict:
    """Apply allocation, watchdog, shutdown, and live-ledger guards without side effects."""
    P.validate_protocol(protocol)
    projected = _finite_nonnegative(protocol.get("projected_total_hours"), "projected_total_hours")
    if projected <= 0:
        raise ValueError("projected_total_hours must be positive")
    allocation = protocol.get("allocation_request_hours")
    if allocation != P.ALLOCATION_HOURS:
        raise ValueError("continuation allocation must be exactly 3.0 hours")
    remaining = _finite_nonnegative(before.get("native_remaining_hours"), "native_remaining_hours")
    watchdog = projected * P.WATCHDOG_MULTIPLIER
    required = watchdog + P.SHUTDOWN_ALLOWANCE_HOURS
    fits_watchdog = required <= allocation + P.HOUR_TOLERANCE
    fits_planning = projected <= P.PLANNING_FRACTION * allocation + P.HOUR_TOLERANCE
    fits_live = (required <= remaining + P.HOUR_TOLERANCE
                 and allocation <= remaining + P.HOUR_TOLERANCE)
    return {
        "children": list(protocol["run_ids"]),
        "additional_updates_per_child": continuation.ADDITIONAL_UPDATES,
        "additional_exposure_tokens_per_child": continuation.ADDITIONAL_EXPOSURE_TOKENS,
        "total_additional_updates": 6 * continuation.ADDITIONAL_UPDATES,
        "total_additional_exposure_tokens": 6 * continuation.ADDITIONAL_EXPOSURE_TOKENS,
        "additional_main_loss_tokens": dict(P.ADDITIONAL_MAIN_LOSS),
        "target_updates_per_child": continuation.TARGET_UPDATES,
        "target_exposure_tokens_per_child": continuation.TARGET_EXPOSURE_TOKENS,
        "target_cursor_per_child": continuation.TARGET_CURSOR,
        "projected_total_hours": projected,
        "allocation_request_hours": allocation,
        "maximum_watchdog_hours": watchdog,
        "shutdown_allowance_hours": P.SHUTDOWN_ALLOWANCE_HOURS,
        "required_native_hours": required,
        "planning_limit_hours": P.PLANNING_FRACTION * allocation,
        "allocation_headroom_hours": allocation - required,
        "planning_headroom_hours": P.PLANNING_FRACTION * allocation - projected,
        "live_native_remaining_hours": remaining,
        "live_native_headroom_hours": remaining - required,
        "fits_watchdog_and_shutdown": fits_watchdog,
        "fits_80_percent_planning_margin": fits_planning,
        "fits_live_native_ledger": fits_live,
        "fits_budget": fits_watchdog and fits_planning and fits_live,
    }


def recovery_projection(protocol: Mapping, before: Mapping, *, prior_hours: float,
                        remaining_projected_hours: float) -> dict:
    """Budget the whole remaining comparison, never a truncated subset of its controls."""
    P.validate_protocol(protocol)
    prior = _finite_nonnegative(prior_hours, "prior continuation hours")
    projected = _finite_nonnegative(remaining_projected_hours, "remaining_projected_hours")
    remaining = _finite_nonnegative(before.get("native_remaining_hours"), "native_remaining_hours")
    if projected <= 0:
        raise ValueError("remaining recovery work must have a positive projection")
    allocation = P.ALLOCATION_HOURS
    available = allocation - prior
    watchdog = projected * P.WATCHDOG_MULTIPLIER
    new_required = watchdog + P.SHUTDOWN_ALLOWANCE_HOURS
    cumulative_required = prior + new_required
    fits_watchdog = cumulative_required <= allocation + P.HOUR_TOLERANCE
    fits_planning = prior + projected <= P.PLANNING_FRACTION * allocation + P.HOUR_TOLERANCE
    fits_live = new_required <= remaining + P.HOUR_TOLERANCE \
        and projected <= remaining + P.HOUR_TOLERANCE
    return {
        "children": list(protocol["run_ids"]),
        "projected_total_hours": projected,
        "remaining_projected_hours": projected,
        "reservation_projected_hours": projected,
        "prior_continuation_charged_hours": prior,
        "cumulative_projected_hours": prior + projected,
        "allocation_request_hours": allocation,
        "remaining_allocation_hours": available,
        "maximum_watchdog_hours": watchdog,
        "shutdown_allowance_hours": P.SHUTDOWN_ALLOWANCE_HOURS,
        "required_native_hours": cumulative_required,
        "new_reservation_required_native_hours": new_required,
        "planning_limit_hours": P.PLANNING_FRACTION * allocation,
        "allocation_headroom_hours": allocation - cumulative_required,
        "planning_headroom_hours": P.PLANNING_FRACTION * allocation - prior - projected,
        "live_native_remaining_hours": remaining,
        "live_native_headroom_hours": remaining - new_required,
        "fits_watchdog_and_shutdown": fits_watchdog,
        "fits_80_percent_planning_margin": fits_planning,
        "fits_live_native_ledger": fits_live,
        "fits_budget": fits_watchdog and fits_planning and fits_live,
        "target_updates_per_child": continuation.TARGET_UPDATES,
        "target_exposure_tokens_per_child": continuation.TARGET_EXPOSURE_TOKENS,
        "target_cursor_per_child": continuation.TARGET_CURSOR,
    }


def _remaining_projected_hours(protocol: Mapping, statuses: Mapping) -> float:
    per_child = protocol.get("projection", {}).get("per_child_seconds", {})
    total_seconds = 0.0
    synthetic_share = protocol["projected_total_hours"] / len(protocol["children"])
    for child in protocol["children"]:
        run_id = child["run_id"]
        if statuses.get(run_id, {}).get("status") == "completed":
            continue
        components = per_child.get(run_id)
        if not isinstance(components, Mapping) or "total" not in components:
            total_seconds += synthetic_share * 3600.0
            continue
        values = dict(components)
        training_complete = Path(child["final_checkpoint"]).is_file()
        if training_complete:
            for name in ("training", "checkpoint", "periodic_and_final_dev", "fork_initialization"):
                values.pop(name, None)
        total_seconds += sum(
            value for name, value in values.items()
            if name != "total" and isinstance(value, (int, float)) and not isinstance(value, bool)
        )
    return total_seconds / 3600.0


def require_budget(plan: Mapping) -> None:
    if not plan["fits_watchdog_and_shutdown"]:
        raise RuntimeError("prior charges plus remaining projection * 1.5 + 0.05 exceed the exact 3.0-hour allocation")
    if not plan["fits_80_percent_planning_margin"]:
        raise RuntimeError("prior charges plus remaining projection exceed 0.8 * 3.0 hours")
    if not plan["fits_live_native_ledger"]:
        raise RuntimeError("the remaining allocation and watchdog/shutdown requirement do not fit the live native ledger")


def _authorization(protocol: Mapping, execute: bool, authorize_hours=None, *, prepare_only: bool = False) -> float | None:
    if execute and prepare_only:
        raise RuntimeError("--execute and --prepare-only are mutually exclusive")
    if prepare_only:
        if authorize_hours is not None:
            raise RuntimeError("--prepare-only takes no --authorize-hours and never opens a reservation")
        return None
    if not execute:
        if authorize_hours is not None:
            raise RuntimeError("--authorize-hours requires --execute; the default dry-run takes neither flag")
        return None
    if authorize_hours is None:
        raise RuntimeError("execution requires both --execute and --authorize-hours exactly 3.0")
    if isinstance(authorize_hours, bool) or not isinstance(authorize_hours, (int, float)) \
            or not math.isfinite(authorize_hours):
        raise RuntimeError("--authorize-hours must be exactly 3.0")
    if float(authorize_hours) != P.ALLOCATION_HOURS \
            or protocol.get("allocation_request_hours") != P.ALLOCATION_HOURS:
        raise RuntimeError("--authorize-hours must exactly equal the proposed 3.0-hour allocation")
    return float(authorize_hours)


def _validate_frozen_identity(protocol: Mapping, protocol_bytes: bytes, protocol_path: Path,
                              repository: Path, output_root: Path) -> str:
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    sidecar = protocol_path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text().strip() != protocol_sha256:
        raise RuntimeError("continuation protocol SHA sidecar does not match the actual protocol bytes")
    if Path(protocol["output_root"]).resolve() != output_root:
        raise RuntimeError("the supplied output root differs from the frozen continuation protocol")
    native_root = (repository / "research" / "ti_tnn" / "native").resolve()
    sources = protocol.get("source_files_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("continuation source_files_sha256 must be a nonempty object")
    for name, expected in sources.items():
        source = (native_root / name).resolve()
        try:
            source.relative_to(native_root)
        except ValueError as error:
            raise ValueError(f"frozen source path escapes the native directory: {name}") from error
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"frozen source file identity mismatch: {name}")
    parent_run = Path(protocol["parent_run"]["path"]).resolve()
    for key, name in (("launch_sha256", "launch.json"), ("run_status_sha256", "run_status.json"),
                      ("diagnostic_manifest_sha256", "diagnostic_manifest.json")):
        if sha256_file(parent_run / name) != protocol["parent_run"][key]:
            raise RuntimeError(f"immutable #78 parent identity mismatch: {name}")
    for mode, parent in protocol["parents"].items():
        path = Path(parent["checkpoint_path"])
        if parent_run not in path.resolve().parents:
            raise RuntimeError(f"{mode}: parent checkpoint escapes the immutable #78 run")
        if sha256_file(path) != parent["expected_parent_sha256"]:
            raise RuntimeError(f"{mode}: immutable parent checkpoint SHA256 differs from the protocol")
    data = protocol["data"]
    root = Path(data["root"]).resolve()
    if sha256_file(root / "manifest.json") != data["manifest_sha256"] \
            or sha256_file(data["tokenizer_path"]) != data["tokenizer_sha256"] \
            or sha256_file(data["replay"]) != data["replay_sha256"]:
        raise RuntimeError("frozen data, tokenizer, or replay identity mismatch")
    masks = protocol["masks"]
    if Path(masks["root"]).resolve() == root or root in Path(masks["root"]).resolve().parents:
        if sha256_file(masks["manifest_path"]) != masks["manifest_sha256"]:
            raise RuntimeError("answer-mask manifest identity mismatch")
    if continuation._path_identity(masks["root"]) != masks["identity"]:
        raise RuntimeError("answer-mask directory identity mismatch")
    return protocol_sha256


def _optimizer_inheritance(child: Mapping, parent: Mapping) -> None:
    child_optimizer = child.get("optimizer")
    parent_optimizer = parent.get("optimizer")
    if not isinstance(child_optimizer, dict) or not isinstance(parent_optimizer, dict):
        raise C.CheckpointValidationError("fork is missing complete optimizer state")
    if identity_hash(child_optimizer.get("state")) != identity_hash(parent_optimizer.get("state")):
        raise C.CheckpointIdentityMismatch("fork optimizer tensors/state differ from the parent")
    child_groups = child_optimizer.get("param_groups")
    parent_groups = parent_optimizer.get("param_groups")
    if not isinstance(child_groups, list) or not isinstance(parent_groups, list) \
            or len(child_groups) != len(parent_groups):
        raise C.CheckpointIdentityMismatch("fork optimizer param groups differ from the parent")
    for child_group, parent_group in zip(child_groups, parent_groups):
        if child_group.get("lr") != continuation.CONTINUATION_LEARNING_RATE:
            raise C.CheckpointIdentityMismatch("fork optimizer does not use the new constant 3e-5 LR")
        if set(child_group) != set(parent_group):
            raise C.CheckpointIdentityMismatch("fork optimizer param-group fields differ from the parent")
        for key in parent_group.keys() - {"lr"}:
            if identity_hash(child_group[key]) != identity_hash(parent_group[key]):
                raise C.CheckpointIdentityMismatch(f"fork optimizer param-group {key} differs from the parent")


def _validate_fork_payload(child_spec: Mapping, initial_path: Path, expected_initial_sha256: str) -> dict:
    initial = C.load_checkpoint(initial_path, expected_sha256=expected_initial_sha256)
    parent = C.load_checkpoint(
        child_spec["parent_checkpoint"], expected_sha256=child_spec["expected_parent_sha256"]
    )
    if initial.get("extra", {}).get("checkpoint_kind") != "fork" \
            or initial.get("step") != continuation.PARENT_UPDATES \
            or initial.get("cursor") != continuation.PARENT_CURSOR \
            or initial.get("consumed_loss_tokens") != continuation.PARENT_EXPOSURE_TOKENS:
        raise C.CheckpointValidationError("initial fork is not the exact #78 parent endpoint")
    if initial["state_hashes"]["model"] != parent["state_hashes"]["model"] \
            or initial["state_hashes"]["sampler"] != parent["state_hashes"]["sampler"] \
            or identity_hash(initial["rng"]) != identity_hash(parent["rng"]):
        raise C.CheckpointIdentityMismatch("initial fork model, sampler, or RNG differs from its parent")
    _optimizer_inheritance(initial, parent)
    config = initial["config"]
    fork = config.get("fork", {})
    if config.get("run_id") != child_spec["run_id"] or fork.get("run_id") != child_spec["run_id"] \
            or fork.get("loss_mode") != child_spec["loss_mode"] \
            or fork.get("continuation_study_id") != child_spec.get(
                "continuation_study_id", P.CONTINUATION_STUDY_ID
            ) \
            or config.get("model", {}).get("mode") != child_spec["mode"]:
        raise C.CheckpointIdentityMismatch("initial fork child, study, mode, or loss identity differs from the protocol")
    if fork.get("parent", {}).get("checkpoint_sha256") != child_spec["expected_parent_sha256"] \
            or initial["identities"].get("parent_checkpoint_sha256") != child_spec["expected_parent_sha256"]:
        raise C.CheckpointIdentityMismatch("initial fork parent SHA identity differs from the protocol")
    if initial["identities"].get("parent_model") != parent["state_hashes"]["model"] \
            or initial["identities"].get("parent_optimizer") != parent["state_hashes"]["optimizer"] \
            or initial["identities"].get("parent_sampler") != parent["state_hashes"]["sampler"] \
            or initial["identities"].get("parent_rng") != identity_hash(parent["rng"]):
        raise C.CheckpointIdentityMismatch("initial fork recorded parent inheritance identities are inconsistent")
    if initial["identities"].get("data") != parent["identities"]["data"] \
            or initial["identities"].get("order") != parent["identities"]["order"]:
        raise C.CheckpointIdentityMismatch("initial fork data or ordered sampler identity differs from the parent")
    schedule = fork.get("schedule", {})
    expected_schedule = {
        name: child_spec["schedule"][name]
        for name in ("learning_rate", "warmup_fraction", "scheduler", "inherits_parent_scheduler")
    }
    training = config.get("training", {})
    if schedule != expected_schedule \
            or fork.get("additional_epochs") != continuation.CONTINUATION_EPOCHS \
            or fork.get("additional_updates") != continuation.ADDITIONAL_UPDATES \
            or fork.get("target_updates") != continuation.TARGET_UPDATES \
            or fork.get("target_exposure_tokens") != continuation.TARGET_EXPOSURE_TOKENS \
            or fork.get("target_cursor") != continuation.TARGET_CURSOR \
            or training.get("learning_rate") != continuation.CONTINUATION_LEARNING_RATE \
            or training.get("warmup_fraction") != 0.0 \
            or training.get("final_lr_fraction") != 1.0 \
            or training.get("max_loss_tokens") != continuation.TARGET_EXPOSURE_TOKENS:
        raise C.CheckpointIdentityMismatch("initial fork does not carry the new constant-LR endpoint contract")
    if fork.get("answer_masks_identity") != child_spec["answer_masks_identity"]:
        raise C.CheckpointIdentityMismatch("initial fork answer-mask identity differs from the protocol")
    return {
        "model_identity": initial["state_hashes"]["model"],
        "parent_optimizer_identity": parent["state_hashes"]["optimizer"],
        "optimizer_state_identity": identity_hash(initial["optimizer"]["state"]),
        "sampler_identity": initial["state_hashes"]["sampler"],
        "order_identity": initial["identities"]["order"],
        "rng_identity": identity_hash(initial["rng"]),
        "parent_scheduler_identity": parent["state_hashes"]["scheduler"],
        "child_scheduler_identity": initial["state_hashes"]["scheduler"],
        "scheduler_inherited": False,
    }


def _manifest_record(manifest: Mapping, run_id: str) -> dict:
    record = manifest.get("children", {}).get(run_id)
    if not isinstance(record, dict):
        raise RuntimeError(f"fork manifest has no child record for {run_id}")
    return record


def validate_fork_manifest(protocol: Mapping, manifest_path, *, protocol_sha256: str,
                           code_identity: str | None = None, validate_payloads: bool = True) -> dict:
    path = Path(manifest_path).resolve()
    manifest = _read_json(path)
    if manifest.get("format") != P.FORK_MANIFEST_FORMAT or not manifest.get("complete") \
            or manifest.get("status") != "completed_validated":
        raise RuntimeError("execution requires a completed validated six-fork manifest")
    if manifest.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError("fork manifest belongs to a different continuation protocol")
    if code_identity is not None and manifest.get("code_identity") != code_identity:
        raise RuntimeError("fork manifest code identity differs from the clean execution source")
    if set(manifest.get("children", {})) != set(protocol["run_ids"]):
        raise RuntimeError("fork manifest does not contain exactly the six protocol children")
    validated = {}
    for child in protocol["children"]:
        record = _manifest_record(manifest, child["run_id"])
        descriptor = record.get("initial_checkpoint", {})
        initial_path = Path(descriptor.get("path", "")).resolve()
        if record.get("status") != "prepared" or initial_path != Path(child["initial_checkpoint"]).resolve():
            raise RuntimeError(f"{child['run_id']}: fork manifest initial checkpoint is incomplete or misplaced")
        if sha256_file(initial_path) != descriptor.get("sha256"):
            raise RuntimeError(f"{child['run_id']}: initial fork checkpoint hash differs from its manifest")
        if validate_payloads:
            inheritance = _validate_fork_payload(child, initial_path, descriptor["sha256"])
            if inheritance != record.get("inheritance"):
                raise RuntimeError(f"{child['run_id']}: recorded fork inheritance differs from checkpoint payloads")
        validated[child["run_id"]] = record
    return {**manifest, "children": validated, "path": str(path), "sha256": sha256_file(path)}


def prepare_forks(protocol: Mapping, protocol_sha256: str, *, code_identity: str | None = None,
                  validate_payloads: bool = True) -> dict:
    """Create six explicit CPU-loaded forks, preserving every preparation failure."""
    P.validate_protocol(protocol)
    output_root = Path(protocol["output_root"]).resolve()
    manifest_path = output_root / FORK_MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing fork manifest: {manifest_path}")
    code_identity = train._code_identity() if code_identity is None else code_identity
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": P.FORK_MANIFEST_FORMAT,
        "status": "preparing",
        "complete": False,
        "protocol_sha256": protocol_sha256,
        "code_identity": code_identity,
        "device_policy": "create_fork(device='cuda', load_device='cpu'); no optimizer step or reservation",
        "optimizer_updates_performed": 0,
        "reservation_created": False,
        "children": {
            run_id: {"status": "not_run"} for run_id in protocol["run_ids"]
        },
    }
    _write_new(manifest_path, manifest)
    for child in protocol["children"]:
        run_id = child["run_id"]
        started = time.perf_counter()
        manifest["children"][run_id] = {"status": "creating"}
        _write(manifest_path, manifest)
        try:
            result = continuation.create_fork(
                child["parent_checkpoint"], child["output"], run_id=run_id,
                code_identity=code_identity, continuation_study_id=protocol["continuation_study_id"],
                loss_mode=child["loss_mode"],
                answer_masks=child["answer_masks"], expected_parent_sha256=child["expected_parent_sha256"],
                additional_epochs=continuation.CONTINUATION_EPOCHS,
                learning_rate=continuation.CONTINUATION_LEARNING_RATE,
                device="cuda", load_device="cpu", validate_completed_78=True,
            )
            descriptor = result["initial_checkpoint"]
            if Path(result["config_path"]).resolve() != Path(child["config_path"]).resolve() \
                    or Path(descriptor["path"]).resolve() != Path(child["initial_checkpoint"]).resolve() \
                    or descriptor.get("step") != continuation.PARENT_UPDATES:
                raise RuntimeError(f"{run_id}: create_fork returned a different child or initial endpoint")
            inheritance = _validate_fork_payload(
                child, Path(descriptor["path"]), descriptor["sha256"]
            ) if validate_payloads else {
                "payload_validation_deferred": True,
            }
            manifest["children"][run_id] = {
                "status": "prepared",
                "run_id": run_id,
                "mode": child["mode"],
                "loss_mode": child["loss_mode"],
                "initial_checkpoint": descriptor,
                "config_path": result["config_path"],
                "journal": result["journal"],
                "parent": result["parent"],
                "schedule": result["schedule"],
                "continuation": result["continuation"],
                "inheritance": inheritance,
                "wall_seconds": time.perf_counter() - started,
            }
            _write(manifest_path, manifest)
        except Exception as error:
            manifest["children"][run_id] = {
                "status": "failed", "stage": "creating", "error": str(error),
                "traceback": traceback.format_exc(), "wall_seconds": time.perf_counter() - started,
            }
            manifest["status"] = "failed_partial_preserved"
            _write(manifest_path, manifest)
            raise
        except BaseException as error:
            manifest["children"][run_id] = {
                "status": "interrupted", "stage": "creating", "kind": type(error).__name__,
                "error": str(error), "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            }
            manifest["status"] = "interrupted_partial_preserved"
            _write(manifest_path, manifest)
            raise
        finally:
            gc.collect()
    manifest["status"] = "completed_validated" if validate_payloads else "created_unvalidated"
    manifest["complete"] = validate_payloads
    _write(manifest_path, manifest)
    return manifest


def _arguments(protocol: Mapping, child: Mapping) -> list[str]:
    arguments = [
        "--config", child["config_path"], "--mode", child["mode"],
        "--data", protocol["data"]["root"], "--output", child["output"],
        "--device", "cuda", "--fork", child["run_id"], "--loss-mode", child["loss_mode"],
        "--projected-hours", str(protocol["projected_total_hours"]),
    ]
    if child["answer_masks"] is not None:
        arguments.extend(["--answer-masks", child["answer_masks"]])
    return arguments


def _journal_path(journal: Path, reference: str) -> Path:
    path = Path(reference)
    return path.resolve() if path.is_absolute() else (journal.parent / path).resolve()


def _validate_fork_to_final_journal(protocol: Mapping, child: Mapping, descriptor: Mapping,
                                    fork_record: Mapping | None = None) -> dict:
    """Verify every acknowledged update between the explicit fork and this final checkpoint."""
    run_id = child["run_id"]
    journal_path = Path(child["output"]) / train.JOURNAL_NAME
    entries, _, incomplete, _ = C.CheckpointJournal(journal_path).entries()
    if incomplete or not entries:
        raise RuntimeError(f"{run_id}: final checkpoint has no complete fork-to-final journal")
    initial_path = Path(child["initial_checkpoint"]).resolve()
    final_path = Path(child["final_checkpoint"]).resolve()
    initial = entries[0].record
    if initial.get("event") != "checkpoint" \
            or _journal_path(journal_path, initial.get("path", "")) != initial_path \
            or initial.get("step") != continuation.PARENT_UPDATES \
            or initial.get("cursor") != continuation.PARENT_CURSOR \
            or initial.get("consumed_loss_tokens") != continuation.PARENT_EXPOSURE_TOKENS:
        raise RuntimeError(f"{run_id}: journal does not start at the declared initial fork")
    initial_descriptor = (fork_record or {}).get("initial_checkpoint", {})
    if initial_descriptor and initial.get("sha256") != initial_descriptor.get("sha256"):
        raise RuntimeError(f"{run_id}: journal initial-fork hash differs from the fork manifest")
    parent = protocol["parents"][child["mode"]]
    step = continuation.PARENT_UPDATES
    cursor = continuation.PARENT_CURSOR
    exposure = continuation.PARENT_EXPOSURE_TOKENS
    batches = {}
    final_records = []
    for entry in entries:
        record = entry.record
        if record.get("data_identity") not in (None, parent["data_identity"]) \
                or record.get("order_identity") not in (None, parent["order_identity"]):
            raise RuntimeError(f"{run_id}: journal data or order identity differs from the parent")
        if record.get("event") == "checkpoint":
            if (record.get("step"), record.get("cursor"), record.get("consumed_loss_tokens")) \
                    != (step, cursor, exposure):
                raise RuntimeError(f"{run_id}: checkpoint journal counters are not at the acknowledged state")
            path = _journal_path(journal_path, record.get("path", ""))
            if path == final_path:
                final_records.append((entry, record))
            continue
        step += 1
        if record.get("step") != step or record.get("batches_consumed") != step:
            raise RuntimeError(f"{run_id}: acknowledged batch steps are not contiguous from the fork")
        loss_tokens = record.get("loss_tokens")
        if isinstance(loss_tokens, bool) or not isinstance(loss_tokens, int) or loss_tokens <= 0:
            raise RuntimeError(f"{run_id}: acknowledged batch has invalid exposure")
        exposure += loss_tokens
        previous_cursor = cursor
        cursor = record.get("cursor")
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor <= previous_cursor:
            raise RuntimeError(f"{run_id}: acknowledged batch has a non-increasing cursor")
        if record.get("consumed_loss_tokens") != exposure \
                or record.get("sampler", {}).get("cursor") != cursor \
                or record.get("sampler", {}).get("consumed_loss_tokens") != exposure \
                or record.get("sampler", {}).get("batches_consumed") != step:
            raise RuntimeError(f"{run_id}: batch and sampler cumulative counters disagree")
        batches[step] = {
            "exposure_tokens": loss_tokens, "cursor": cursor,
            "consumed_exposure_tokens": exposure,
            "batch_identity": record.get("batch_identity"),
        }
    if len(final_records) != 1 or entries[-1] is not final_records[0][0]:
        raise RuntimeError(f"{run_id}: journal must end with exactly one declared final checkpoint")
    final_record = final_records[0][1]
    if (step, cursor, exposure) != (
            continuation.TARGET_UPDATES, continuation.TARGET_CURSOR, continuation.TARGET_EXPOSURE_TOKENS
    ) or len(batches) != continuation.ADDITIONAL_UPDATES:
        raise RuntimeError(f"{run_id}: fork-to-final journal does not reach the cumulative 64-epoch endpoint")
    if (descriptor.get("sha256") is not None and final_record.get("sha256") != descriptor["sha256"]) \
            or (descriptor.get("bytes") is not None and final_record.get("bytes") != descriptor["bytes"]):
        raise RuntimeError(f"{run_id}: final checkpoint descriptor differs from its journal record")
    return {
        "path": str(journal_path), "sha256": sha256_file(journal_path), "batches": batches,
        "additional_exposure_tokens": exposure - continuation.PARENT_EXPOSURE_TOKENS,
        "final_record": {**final_record, "path": str(final_path)},
    }


def _complete_jsonl(path: Path):
    data = path.read_bytes()
    offset = 0
    while offset < len(data):
        newline = data.find(b"\n", offset)
        if newline < 0:
            break
        raw = data[offset:newline].rstrip(b"\r")
        offset = newline + 1
        try:
            record = json.loads(raw)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"corrupt complete durable training-log record: {path}") from error
        if not isinstance(record, dict):
            raise RuntimeError(f"durable training log contains a non-object record: {path}")
        yield record


def _training_accounting(protocol: Mapping, child: Mapping, payload: Mapping,
                         fork_record: Mapping | None = None) -> dict:
    journal = _validate_fork_to_final_journal(
        protocol, child, payload["final_checkpoint"], fork_record
    )
    log_path = Path(payload.get("learning_log") or Path(child["output"]) / train.LEARNING_LOG_NAME)
    records = {}
    log_available = log_path.is_file()
    if log_available:
        for record in _complete_jsonl(log_path):
            if record.get("event") not in ("train", "replay"):
                continue
            step = record.get("step")
            if step not in journal["batches"]:
                raise RuntimeError(f"{child['run_id']}: durable training log has an out-of-contract step")
            expected = journal["batches"][step]
            values = {
                "exposure_tokens": record.get("exposure_tokens"),
                "main_loss_tokens": record.get("main_loss_tokens"),
                "cursor": record.get("cursor"),
                "consumed_exposure_tokens": record.get(
                    "consumed_exposure_tokens", record.get("consumed_loss_tokens")
                ),
                "batch_identity": record.get("batch_identity"),
            }
            if any(values[key] != expected[key] for key in (
                    "exposure_tokens", "cursor", "consumed_exposure_tokens", "batch_identity"
            )):
                raise RuntimeError(f"{child['run_id']}: durable training log differs from the journal at step {step}")
            main = values["main_loss_tokens"]
            if isinstance(main, bool) or not isinstance(main, int) or main < 0 \
                    or (child["loss_mode"] == "full_sequence" and main != expected["exposure_tokens"]):
                raise RuntimeError(f"{child['run_id']}: invalid main-loss count at durable step {step}")
            if step in records and records[step] != values:
                raise RuntimeError(f"{child['run_id']}: replay and original durable records differ at step {step}")
            records[step] = values
    complete_log = len(records) == continuation.ADDITIONAL_UPDATES
    if complete_log:
        exposure = sum(record["exposure_tokens"] for record in records.values())
        main = sum(record["main_loss_tokens"] for record in records.values())
        if exposure != continuation.ADDITIONAL_EXPOSURE_TOKENS \
                or main != child["additional_main_loss_tokens"]:
            raise RuntimeError(f"{child['run_id']}: durable additional exposure/main-loss totals differ by mode")
        exposure_source = main_source = "complete_durable_training_log"
    else:
        exposure = journal["additional_exposure_tokens"]
        main = child["additional_main_loss_tokens"]
        exposure_source = "fork_to_final_journal"
        main_source = "mode_contract_after_journal_validation"
    return {
        "additional_exposure_tokens": exposure,
        "additional_main_loss_tokens": main,
        "additional_exposure_source": exposure_source,
        "additional_main_loss_source": main_source,
        "durable_training_log": str(log_path) if log_available else None,
        "durable_training_log_complete": complete_log,
        "durable_training_steps": len(records),
        "fork_to_final_journal": journal["path"],
        "fork_to_final_journal_sha256": journal["sha256"],
    }


def _validate_invocation_counts(child: Mapping, payload: Mapping) -> None:
    maximum = continuation.ADDITIONAL_UPDATES
    new = payload.get("new_updates")
    replayed = payload.get("replayed_updates")
    processed = payload.get("processed_updates_this_invocation")
    if new is not None:
        _nonnegative_integer(new, f"{child['run_id']} new_updates", maximum=maximum)
    if replayed is not None:
        _nonnegative_integer(replayed, f"{child['run_id']} replayed_updates", maximum=maximum)
    if processed is not None:
        _nonnegative_integer(
            processed, f"{child['run_id']} processed_updates_this_invocation", maximum=maximum
        )
    if new is not None and replayed is not None and processed is not None \
            and new + replayed != processed:
        raise RuntimeError(f"{child['run_id']}: new plus replayed updates differ from invocation updates")
    for name, expected in (
            ("processed_exposure_tokens_this_invocation", continuation.ADDITIONAL_EXPOSURE_TOKENS),
            ("processed_main_loss_tokens_this_invocation", child["additional_main_loss_tokens"]),
    ):
        if payload.get(name) is not None:
            _nonnegative_integer(payload[name], f"{child['run_id']} {name}", maximum=expected)


def _training_checkpoint(protocol: Mapping, child: Mapping, payload: Mapping,
                         fork_record: Mapping | None = None) -> dict:
    descriptor = payload.get("final_checkpoint", {})
    path = Path(descriptor.get("path", "")).resolve()
    if (payload.get("format") != train.TRAIN_FORMAT
            or payload.get("mode") != child["mode"]
            or payload.get("loss_mode") != child["loss_mode"]
            or payload.get("step") != continuation.TARGET_UPDATES
            or payload.get("cursor") != continuation.TARGET_CURSOR
            or payload.get("consumed_loss_tokens") != continuation.TARGET_EXPOSURE_TOKENS
            or payload.get("consumed_exposure_tokens") != continuation.TARGET_EXPOSURE_TOKENS
            or payload.get("planned_updates") != continuation.TARGET_UPDATES
            or descriptor.get("step") != continuation.TARGET_UPDATES
            or descriptor.get("cursor") != continuation.TARGET_CURSOR
            or descriptor.get("consumed_loss_tokens") != continuation.TARGET_EXPOSURE_TOKENS
            or path != Path(child["final_checkpoint"]).resolve()
            or path.name != "final-update-00013568.pt"):
        raise RuntimeError(f"{child['run_id']}: final training endpoint or checkpoint identity differs from the protocol")
    _validate_invocation_counts(child, payload)
    if sha256_file(path) != descriptor.get("sha256"):
        raise RuntimeError(f"{child['run_id']}: final checkpoint SHA256 differs from its descriptor")
    final = C.load_checkpoint(path, expected_sha256=descriptor["sha256"])
    if final.get("extra", {}).get("checkpoint_kind") != "final" \
            or final.get("step") != continuation.TARGET_UPDATES \
            or final.get("cursor") != continuation.TARGET_CURSOR \
            or final.get("consumed_loss_tokens") != continuation.TARGET_EXPOSURE_TOKENS \
            or final["config"].get("run_id") != child["run_id"] \
            or final["config"].get("fork", {}).get("run_id") != child["run_id"]:
        raise RuntimeError(f"{child['run_id']}: final checkpoint is not the declared child's final state")
    if any(group.get("lr") != continuation.CONTINUATION_LEARNING_RATE
           for group in final["optimizer"]["param_groups"]):
        raise RuntimeError(f"{child['run_id']}: final optimizer does not retain constant 3e-5")
    parent = protocol["parents"][child["mode"]]
    for key, expected in (("parent_model", parent["model_identity"]),
                          ("parent_optimizer", parent["optimizer_identity"]),
                          ("parent_sampler", parent["sampler_identity"]),
                          ("parent_checkpoint_sha256", parent["expected_parent_sha256"])):
        if final["identities"].get(key) != expected:
            raise RuntimeError(f"{child['run_id']}: final checkpoint lost parent identity {key}")
    if fork_record is not None \
            and final["identities"].get("parent_rng") != fork_record["inheritance"]["rng_identity"]:
        raise RuntimeError(f"{child['run_id']}: final checkpoint lost the parent's RNG identity")
    _training_accounting(protocol, child, payload, fork_record)
    return {**descriptor, "path": str(path), "sha256": sha256_file(path)}


def _recovered_training_payload(protocol: Mapping, child: Mapping, fork_record: Mapping,
                                previous: Mapping | None = None) -> dict:
    journal = _validate_fork_to_final_journal(
        protocol, child, {"path": child["final_checkpoint"]}, fork_record
    )
    descriptor = journal["final_record"]
    payload = {
        "format": train.TRAIN_FORMAT, "mode": child["mode"], "loss_mode": child["loss_mode"],
        "step": continuation.TARGET_UPDATES, "cursor": continuation.TARGET_CURSOR,
        "consumed_loss_tokens": continuation.TARGET_EXPOSURE_TOKENS,
        "consumed_exposure_tokens": continuation.TARGET_EXPOSURE_TOKENS,
        "planned_updates": continuation.TARGET_UPDATES,
        "new_updates": 0, "replayed_updates": 0, "processed_updates_this_invocation": 0,
        "processed_exposure_tokens_this_invocation": 0,
        "processed_main_loss_tokens_this_invocation": 0,
        "training_seconds": (previous or {}).get("training_seconds", 0.0),
        "checkpoint_seconds": (previous or {}).get("checkpoint_seconds", 0.0),
        "learning_log": str(Path(child["output"]) / train.LEARNING_LOG_NAME),
        "journal": journal["path"], "final_checkpoint": descriptor,
        "recovered_from_completed_checkpoint": True,
    }
    _training_checkpoint(protocol, child, payload, fork_record)
    return payload


def _existing_training(protocol: Mapping, child: Mapping, fork_record: Mapping):
    errors = []
    for path in _result_candidates(Path(child["output"]) / "train-result.json"):
        try:
            payload = _read_json(path)
            checkpoint = _training_checkpoint(protocol, child, payload, fork_record)
            return payload, checkpoint, path
        except Exception as error:
            errors.append(f"{path}: {error}")
    final_path = Path(child["final_checkpoint"])
    if final_path.exists():
        previous = None
        for path in _result_candidates(Path(child["output"]) / "train-result.json"):
            try:
                previous = _read_json(path)
                break
            except Exception:
                continue
        try:
            payload = _recovered_training_payload(protocol, child, fork_record, previous)
            return payload, payload["final_checkpoint"], None
        except Exception as error:
            errors.append(f"completed checkpoint recovery: {error}")
        raise RuntimeError(
            f"{child['run_id']}: a final checkpoint exists but is not valid; refusing to retrain or overwrite it: "
            + " | ".join(errors)
        )
    return None


def _stage_output(path: Path, *, resume: bool, recovery_index: int | None) -> Path:
    if not resume:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite an existing stage result without --resume: {path}")
        return path
    if recovery_index is None:
        raise ValueError("resumed stage output needs a recovery index")
    return _recovery_path(path, recovery_index)


def _existing_evaluation(output: Path, checkpoint: Mapping):
    for path in _result_candidates(output / "evaluation.json"):
        try:
            evaluation = _read_json(path)
            exact = program_runner._validate_evaluation(evaluation, checkpoint)
            return path, evaluation, exact
        except Exception:
            continue
    return None


def _existing_rescore(output: Path, evaluation_path: Path, evaluation: Mapping, mode: str):
    for path in _result_candidates(output / "generation_diagnostics.json"):
        try:
            diagnostics = _read_json(path)
            recorded = Path(diagnostics.get("evaluation_paths", {}).get(mode, "")).resolve()
            if recorded != evaluation_path.resolve():
                continue
            program_runner._validate_generation_diagnostics(diagnostics, evaluation, mode)
            return path, diagnostics
        except Exception:
            continue
    return None


def _validate_reconstruction(report: Mapping, checkpoint: Mapping, run_id: str) -> None:
    execution = report.get("execution", {})
    if report.get("checkpoint", {}).get("sha256") != checkpoint["sha256"] \
            or report.get("checkpoint", {}).get("step") != checkpoint["step"] \
            or execution.get("device") != "cpu" \
            or execution.get("parameter_dtype") != "torch.float32" \
            or execution.get("autocast_dtype") is not None:
        raise RuntimeError(f"{run_id}: reconstruction was not unautocast CPU FP32 for this checkpoint")
    if not report.get("agreement"):
        raise RuntimeError(f"{run_id}: FP32 full/cached reconstruction disagrees")


def _existing_reconstruction(output: Path, checkpoint: Mapping, run_id: str):
    for path in _result_candidates(output / "reconstruction.json"):
        try:
            report = _read_json(path)
            _validate_reconstruction(report, checkpoint, run_id)
            return path, report
        except Exception:
            continue
    return None


def _answer_manifest_values(manifest: Mapping, child: Mapping, checkpoint: Mapping,
                            train_result_path: Path, evaluation_path: Path) -> dict:
    answer = manifest.get("arms", {}).get(child["run_id"], {})
    if not answer.get("teacher_forced_complete") \
            or answer.get("checkpoint", {}).get("expected_sha256") != checkpoint["sha256"]:
        raise RuntimeError(f"{child['run_id']}: answer diagnostics are incomplete or used another checkpoint")
    answer_diagnostics = answer.get("answer_diagnostics", {})
    if not Path(answer_diagnostics.get("path", "")).is_file() \
            or sha256_file(answer_diagnostics["path"]) != answer_diagnostics.get("sha256"):
        raise RuntimeError(f"{child['run_id']}: answer diagnostics artifact identity is invalid")
    if answer.get("train_result", {}).get("sha256") != sha256_file(train_result_path) \
            or answer.get("official_evaluation_source", {}).get("sha256") != sha256_file(evaluation_path):
        raise RuntimeError(f"{child['run_id']}: answer diagnostics used different training/evaluation results")
    return answer


def _existing_answer(output: Path, child: Mapping, checkpoint: Mapping,
                     train_result_path: Path, evaluation_path: Path):
    for path in _result_candidates(output / "answer_diagnostic_manifest.json"):
        try:
            manifest = _read_json(path)
            answer = _answer_manifest_values(
                manifest, child, checkpoint, train_result_path, evaluation_path
            )
            return path, manifest, answer
        except Exception:
            continue
    return None


def execute_child(protocol: Mapping, child: Mapping, fork_record: Mapping, *, on_stage=None,
                  resume: bool = False, recovery_index: int | None = None) -> dict:
    """Run missing stages, retaining every prior result and every valid completed stage."""
    output = Path(child["output"])
    timings = {}
    if not resume:
        for name in ("train-result.json", "evaluation.json", "generation_diagnostics.json",
                     "reconstruction.json", "answer_diagnostic_manifest.json", "answer_diagnostics.json"):
            if (output / name).exists() or Path(child["final_checkpoint"]).exists():
                raise FileExistsError(f"{child['run_id']}: existing child results require explicit --resume")

    def stage(name: str, **fields) -> None:
        if on_stage is not None:
            on_stage(name, **fields)

    started = time.perf_counter()
    existing = _existing_training(protocol, child, fork_record) if resume else None
    if existing is None:
        arguments = _arguments(protocol, child)
        if resume:
            arguments.append("--resume")
        payload = train.run_training(train.parser().parse_args(arguments))
        train_result_path = _stage_output(
            output / "train-result.json", resume=resume, recovery_index=recovery_index
        )
        _write_new(train_result_path, payload)
        checkpoint = _training_checkpoint(protocol, child, payload, fork_record)
    else:
        payload, checkpoint, train_result_path = existing
        if train_result_path is None:
            train_result_path = _stage_output(
                output / "train-result.json", resume=resume, recovery_index=recovery_index
            )
            _write_new(train_result_path, payload)
    timings["training_invocation_wall"] = time.perf_counter() - started
    accounting = _training_accounting(protocol, child, payload, fork_record)
    stage("evaluating", updates=payload["step"], cursor=payload["cursor"],
          consumed_exposure_tokens=payload["consumed_exposure_tokens"], checkpoint=checkpoint,
          train_result_path=str(train_result_path), **accounting)
    started = time.perf_counter()
    reused = _existing_evaluation(output, checkpoint) if resume else None
    if reused is None:
        evaluation_path = _stage_output(
            output / "evaluation.json", resume=resume, recovery_index=recovery_index
        )
        evaluation = evaluate_program_study.evaluate(
            checkpoint=checkpoint["path"], expected_sha256=checkpoint["sha256"],
            data=Path(protocol["data"]["root"]), tokenizer=Path(protocol["data"]["tokenizer_path"]),
            output=evaluation_path, max_new_tokens=48,
        )
        if not evaluation_path.is_file() or _read_json(evaluation_path) != evaluation:
            raise RuntimeError(f"{child['run_id']}: official evaluation did not create its declared JSON result")
        exact = program_runner._validate_evaluation(evaluation, checkpoint)
    else:
        evaluation_path, evaluation, exact = reused
    evaluation_sha256 = sha256_file(evaluation_path)
    timings["official_evaluation"] = time.perf_counter() - started
    stage("rescoring", exact=exact, evaluation_sha256=evaluation_sha256,
          evaluation_path=str(evaluation_path))
    started = time.perf_counter()
    reused = _existing_rescore(output, evaluation_path, evaluation, child["mode"]) if resume else None
    if reused is None:
        diagnostics_path = _stage_output(
            output / "generation_diagnostics.json", resume=resume, recovery_index=recovery_index
        )
        diagnostics = rescore_generations.rescore(
            study=Path(protocol["output_root"]), data=Path(protocol["data"]["root"]),
            output=diagnostics_path, evaluation_paths={child["mode"]: evaluation_path},
        )
        program_runner._validate_generation_diagnostics(diagnostics, evaluation, child["mode"])
    else:
        diagnostics_path, diagnostics = reused
    generation_diagnostics_sha256 = sha256_file(diagnostics_path)
    timings["independent_rescore"] = time.perf_counter() - started
    stage("reconstructing", generation_diagnostics_sha256=generation_diagnostics_sha256,
          generation_diagnostics_path=str(diagnostics_path))
    started = time.perf_counter()
    reused = _existing_reconstruction(output, checkpoint, child["run_id"]) if resume else None
    if reused is None:
        reconstruction_path = _stage_output(
            output / "reconstruction.json", resume=resume, recovery_index=recovery_index
        )
        replay_spec = reconstruct.load_replay_spec(protocol["data"]["replay"])
        report = reconstruct.reconstruct_checkpoint(
            checkpoint["path"], replay_spec, expected_sha256=checkpoint["sha256"],
            device="cpu", autocast_dtype=None,
        )
        _validate_reconstruction(report, checkpoint, child["run_id"])
        reconstruction_sha256 = reconstruct.write_reconstruction_report(reconstruction_path, report)
    else:
        reconstruction_path, report = reused
        reconstruction_sha256 = sha256_file(reconstruction_path)
    timings["fp32_reconstruction"] = time.perf_counter() - started
    stage("answer_diagnostics", reconstruction_sha256=reconstruction_sha256,
          reconstruction_path=str(reconstruction_path))
    started = time.perf_counter()
    reused = _existing_answer(
        output, child, checkpoint, train_result_path, evaluation_path
    ) if resume else None
    if reused is None:
        answer_manifest_path = _stage_output(
            output / "answer_diagnostic_manifest.json", resume=resume, recovery_index=recovery_index
        )
        answer_diagnostics_path = _stage_output(
            output / "answer_diagnostics.json", resume=resume, recovery_index=recovery_index
        )
        answer_manifest = diagnose_program_results.diagnose(
            protocol["output_root"], protocol["data"]["root"], protocol["masks"]["root"],
            child_mappings={child["run_id"]: {
                "mode": child["mode"], "train_result": train_result_path,
                "evaluation": evaluation_path, "output": answer_diagnostics_path,
            }},
            manifest_output=answer_manifest_path,
        )
        answer = _answer_manifest_values(
            answer_manifest, child, checkpoint, train_result_path, evaluation_path
        )
    else:
        answer_manifest_path, answer_manifest, answer = reused
    answer_diagnostics = answer["answer_diagnostics"]
    timings["answer_diagnostics"] = time.perf_counter() - started
    return {
        "checkpoint": checkpoint,
        "updates": payload["step"],
        "cursor": payload["cursor"],
        "consumed_exposure_tokens": payload["consumed_exposure_tokens"],
        **accounting,
        "new_updates": payload.get("new_updates"),
        "replayed_updates": payload.get("replayed_updates"),
        "processed_exposure_tokens_this_invocation": payload.get(
            "processed_exposure_tokens_this_invocation"
        ),
        "processed_main_loss_tokens_this_invocation": payload.get(
            "processed_main_loss_tokens_this_invocation"
        ),
        "train_seconds": payload["training_seconds"],
        "checkpoint_seconds": payload["checkpoint_seconds"],
        "train_result_path": str(train_result_path),
        "exact": exact,
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": evaluation_sha256,
        "generation_diagnostics_path": str(diagnostics_path),
        "generation_diagnostics_sha256": generation_diagnostics_sha256,
        "reconstruction_path": str(reconstruction_path),
        "reconstruction_sha256": reconstruction_sha256,
        "answer_diagnostics_path": answer_diagnostics["path"],
        "answer_diagnostics_sha256": answer_diagnostics["sha256"],
        "answer_diagnostic_manifest_path": str(answer_manifest_path),
        "answer_diagnostic_manifest_sha256": sha256_file(answer_manifest_path),
        "measured_stage_seconds": timings,
    }


def _status_history(output: Path) -> list[dict]:
    canonical = output / "status.json"
    paths = ([canonical] if canonical.is_file() else []) + sorted(
        output.glob("status.recovery-*.json"), key=lambda path: path.name
    )
    return [_read_json(path) for path in paths]


def _recover_statuses(protocol: Mapping, statuses: Mapping, *, recovery_index: int | None = None) -> dict:
    recovered = {}
    for child in protocol["children"]:
        canonical = Path(child["output"]) / "status.json"
        path = canonical if recovery_index is None else _recovery_path(canonical, recovery_index)
        recovered[child["run_id"]] = _read_json(path) if path.is_file() else dict(statuses[child["run_id"]])
    return recovered


def _execute(protocol: Mapping, fork_manifest: Mapping, *, reservation=None, resume: bool = False,
             recovery_index: int | None = None, previous_statuses: Mapping | None = None) -> dict:
    """Run incomplete children sequentially without erasing prior status or failure history."""
    previous_statuses = previous_statuses or {}
    statuses = {
        child["run_id"]: {
            "status": "not_run", "mode": child["mode"], "loss_mode": child["loss_mode"],
            "required_updates": continuation.TARGET_UPDATES,
            "required_exposure_tokens": continuation.TARGET_EXPOSURE_TOKENS,
        }
        for child in protocol["children"]
    }
    for child in protocol["children"]:
        run_id = child["run_id"]
        output = Path(child["output"])
        previous = previous_statuses.get(run_id, {})
        if resume and previous.get("status") == "completed":
            if _existing_training(protocol, child, _manifest_record(fork_manifest, run_id)) is None:
                raise RuntimeError(f"{run_id}: completed child has no valid final training state")
            statuses[run_id] = {**previous, "recovery_skipped": True}
            continue
        history = _status_history(output) if resume else []
        if resume and not history and previous:
            history = [dict(previous)]
        if resume:
            statuses[run_id]["previous_status_history"] = history
            statuses[run_id]["failure_history"] = [
                {key: item[key] for key in ("status", "stage", "kind", "error", "traceback") if key in item}
                for item in history if item.get("error") or item.get("status") in ("failed", "interrupted")
            ]
            statuses[run_id]["recovery_index"] = recovery_index
        status_path = output / "status.json" if not resume else _recovery_path(
            output / "status.json", recovery_index
        )
        if reservation is not None:
            reservation.check()
            reservation.extra["native_acquisition_run_id"] = run_id
        previous_owner = os.environ.get(OWNER_ENV)
        os.environ[OWNER_ENV] = run_id
        started = time.perf_counter()
        statuses[run_id]["status"] = "training"
        _write(status_path, statuses[run_id])

        def stage(name: str, **fields) -> None:
            statuses[run_id].update(fields)
            statuses[run_id]["status"] = name
            _write(status_path, statuses[run_id])

        try:
            if resume:
                result = execute_child(
                    protocol, child, _manifest_record(fork_manifest, run_id), on_stage=stage,
                    resume=True, recovery_index=recovery_index,
                )
            else:
                result = execute_child(
                    protocol, child, _manifest_record(fork_manifest, run_id), on_stage=stage
                )
            statuses[run_id] = {
                **statuses[run_id], **result, "status": "completed",
                "wall_seconds": time.perf_counter() - started,
            }
        except Exception as error:
            statuses[run_id] = {
                **statuses[run_id], "status": "failed", "stage": statuses[run_id]["status"],
                "error": str(error), "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            }
        except BaseException as error:
            statuses[run_id] = {
                **statuses[run_id], "status": "interrupted", "stage": statuses[run_id]["status"],
                "kind": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            }
            _write(status_path, statuses[run_id])
            raise
        finally:
            _write(status_path, statuses[run_id])
            if previous_owner is None:
                os.environ.pop(OWNER_ENV, None)
            else:
                os.environ[OWNER_ENV] = previous_owner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return statuses


def _validate_runtime(repository: Path, python) -> None:
    pinned = repository.parent / "universa-recurrent-native" / ".venv-native" / "bin" / "python"
    if Path(sys.executable).resolve() != pinned.resolve():
        raise RuntimeError(f"use the pinned native environment, not {sys.executable}")
    if python is not None and Path(python).resolve() != Path(sys.executable).resolve():
        raise RuntimeError("--python must be the interpreter running this command")
    native = repository / "research" / "ti_tnn" / "native"
    for module, name in (
        (train, "train.py"), (continuation, "continuation.py"),
        (evaluate_program_study, "evaluate_program_study.py"),
        (diagnose_program_results, "diagnose_program_results.py"),
        (reconstruct, "reconstruct.py"), (P, "prepare_continuation_protocol.py"),
    ):
        imported = Path(module.__file__).resolve()
        if imported != native / name:
            raise RuntimeError(f"PYTHONPATH selected {imported}, not the requested source worktree")
    imported_rescore = Path(rescore_generations.__file__).resolve()
    if imported_rescore != native / "independent" / "rescore_generations.py":
        raise RuntimeError("PYTHONPATH selected a different independent rescore implementation")


def _source_state(repository: Path, *, require_clean: bool) -> tuple[str, str]:
    source = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository, capture_output=True, text=True, check=True,
    ).stdout
    if require_clean and dirty:
        raise RuntimeError("continuation execution requires a clean, committed source worktree")
    return source, train._code_identity()


def _run_status_history(output_root: Path) -> list[tuple[Path, dict]]:
    canonical = output_root / "run_status.json"
    paths = ([canonical] if canonical.is_file() else []) + sorted(
        output_root.glob("run_status.recovery-*.json"), key=lambda path: path.name
    )
    return [(path, _read_json(path)) for path in paths]


def _continuation_study_id(protocol: Mapping, fork_manifest: Mapping) -> str:
    declared = protocol.get("continuation_study_id")
    discovered = []
    for child in protocol["children"]:
        config_path = Path(child["config_path"])
        if config_path.is_file():
            fork = _read_json(config_path).get("fork", {})
            if fork.get("continuation_study_id") is None:
                raise RuntimeError(f"{child['run_id']}: fork config has no continuation_study_id")
            discovered.append(fork["continuation_study_id"])
        discovered.append(
            _manifest_record(fork_manifest, child["run_id"]).get("continuation_study_id")
        )
    present = {value for value in discovered if value is not None}
    if len(present) > 1:
        raise RuntimeError("continuation forks do not share one continuation_study_id")
    study_id = next(iter(present), declared)
    if not isinstance(study_id, str) or not study_id.strip():
        raise RuntimeError("continuation forks do not declare a continuation_study_id")
    if declared is not None and study_id != declared:
        raise RuntimeError("fork continuation_study_id differs from the frozen protocol")
    return study_id


def _validate_resume_identity(protocol: Mapping, launch: Mapping, previous_status: Mapping | None,
                              *, protocol_path: Path, protocol_sha256: str, output_root: Path,
                              source: str, code_identity: str, authorized_hours: float,
                              fork_manifest: Mapping, study_id: str) -> None:
    expected = {
        "format": P.LAUNCH_FORMAT, "source_commit": source, "code_identity": code_identity,
        "protocol_sha256": protocol_sha256, "output_root": str(output_root),
        "run_ids": list(protocol["run_ids"]), "children": list(protocol["children"]),
        "parents": dict(protocol["parents"]), "sampler_order_identity": protocol["data"]["order_identity"],
        "masks": dict(protocol["masks"]), "endpoint": dict(protocol["endpoint"]),
        "exposure_accounting": dict(protocol["exposure_accounting"]),
        "data": dict(protocol["data"]), "authorized_hours": authorized_hours,
        "authorization_status": "explicitly_authorized_for_this_execution",
        "fork_manifest_sha256": fork_manifest["sha256"],
        "continuation_study_id": study_id,
    }
    for key, value in expected.items():
        if key == "continuation_study_id" and key not in launch:
            continue
        if launch.get(key) != value:
            raise RuntimeError(f"previous continuation launch identity mismatch: {key}")
    if Path(launch.get("protocol", "")).resolve() != protocol_path:
        raise RuntimeError("previous continuation launch identity mismatch: protocol path")
    if previous_status is not None:
        if previous_status.get("format") != P.RUN_FORMAT \
                or previous_status.get("protocol_sha256") != protocol_sha256 \
                or previous_status.get("source_commit") != source \
                or previous_status.get("code_identity") != code_identity \
                or previous_status.get("authorized_hours") != authorized_hours:
            raise RuntimeError("previous continuation status identity or authorization differs")


def _prior_continuation_charges(path: Path, before: Mapping, study_id: str,
                                history: list[tuple[Path, dict]]) -> dict:
    reservation_ids = set()
    if path.is_file():
        for entry in ledger.read(path):
            if entry.get("event") != "start":
                continue
            extra = entry.get("extra", {})
            if extra.get("native_continuation_run_id") == study_id:
                if extra.get("native_continuation_authorized_hours") != P.ALLOCATION_HOURS:
                    raise RuntimeError("a prior continuation reservation has the wrong authorization metadata")
                reservation_ids.add(entry["reservation"])
    for _, status in history:
        reservation_id = status.get("reservation_id")
        if reservation_id is not None:
            reservation_ids.add(reservation_id)
    reservations = {item.get("reservation"): item for item in before.get("reservations", [])}
    charges = {}
    for reservation_id in sorted(reservation_ids):
        if reservation_id not in reservations:
            raise RuntimeError(f"the live ledger is missing continuation reservation {reservation_id}")
        charges[reservation_id] = _finite_nonnegative(
            reservations[reservation_id].get("hours"), f"charge for {reservation_id}"
        )
    for status_path, status in history:
        reservation_id = status.get("reservation_id")
        if reservation_id is not None and reservation_id in charges \
                and abs(charges[reservation_id] - _finite_nonnegative(
                    status.get("charged_hours"), f"charged_hours in {status_path}"
                )) > 1e-9:
            raise RuntimeError(f"recorded and live continuation charges differ for {reservation_id}")
    return {
        "continuation_study_id": study_id,
        "prior_continuation_charged_hours": sum(charges.values()),
        "prior_continuation_reservations": charges,
        "prior_continuation_reservation_count": len(charges),
    }


def run(protocol_path=P.DEFAULT_PROTOCOL_PATH, repository=None, *, output_root=None,
        execute=False, prepare_only=False, authorize_hours=None, python=None, resume=False) -> int:
    protocol_path = Path(protocol_path).resolve()
    repository = Path(repository).resolve() if repository is not None else Path(__file__).resolve().parents[3]
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    P.validate_protocol(protocol)
    authorized_hours = _authorization(
        protocol, execute, authorize_hours, prepare_only=prepare_only
    )
    if prepare_only and resume:
        raise RuntimeError("--prepare-only and --resume are mutually exclusive")
    output_root = Path(output_root).resolve() if output_root is not None else Path(protocol["output_root"]).resolve()
    protocol_sha256 = _validate_frozen_identity(protocol, protocol_bytes, protocol_path, repository, output_root)
    path, lock = ledger.validate_shared_reservation_paths()
    before = ledger.usage(path)
    plan = projection(protocol, before)
    if not execute and not prepare_only:
        print(json.dumps({
            "status": "dry_run", "protocol": str(protocol_path), "protocol_sha256": protocol_sha256,
            "output_root": str(output_root), "projection": plan, "resume_requested": resume,
            "fork_manifest_present": (output_root / FORK_MANIFEST_NAME).is_file(),
            "artifacts_created": False, "reservation_created": False,
            "optimizer_updates_performed": 0, "training_launched": False,
        }, sort_keys=True, allow_nan=False))
        return 0
    if prepare_only:
        require_budget(plan)
        manifest = prepare_forks(protocol, protocol_sha256)
        print(json.dumps({
            "status": manifest["status"], "fork_manifest": str(output_root / FORK_MANIFEST_NAME),
            "children": protocol["run_ids"], "optimizer_updates_performed": 0,
            "reservation_created": False, "training_launched": False,
        }, sort_keys=True, allow_nan=False))
        return 0
    if ledger.active() is not None:
        raise RuntimeError("an active native reservation already owns this process; refusing to nest")
    _validate_runtime(repository, python)
    source, code_identity = _source_state(repository, require_clean=True)
    fork_manifest = validate_fork_manifest(
        protocol, output_root / FORK_MANIFEST_NAME,
        protocol_sha256=protocol_sha256, code_identity=code_identity,
    )
    launch_path = output_root / "launch.json"
    status_path = output_root / "run_status.json"
    if not resume and (launch_path.exists() or status_path.exists()):
        raise FileExistsError("refusing to reuse an existing continuation run directory without --resume")
    study_id = _continuation_study_id(protocol, fork_manifest)
    history = _run_status_history(output_root) if resume else []
    previous_launch = _read_json(launch_path) if resume else None
    if resume and previous_launch is None:
        raise FileNotFoundError(f"--resume requires the previous continuation launch: {launch_path}")
    previous_status = history[-1][1] if history else None
    recovery_index = _next_recovery_index(output_root) if resume else 0
    previous_statuses = {run_id: {"status": "not_run"} for run_id in protocol["run_ids"]}
    if resume:
        _validate_resume_identity(
            protocol, previous_launch, previous_status, protocol_path=protocol_path,
            protocol_sha256=protocol_sha256, output_root=output_root, source=source,
            code_identity=code_identity, authorized_hours=authorized_hours,
            fork_manifest=fork_manifest, study_id=study_id,
        )
        for child in protocol["children"]:
            child_history = _status_history(Path(child["output"]))
            if child_history:
                previous_statuses[child["run_id"]] = child_history[-1]
            elif previous_status is not None:
                previous_statuses[child["run_id"]] = previous_status.get(
                    "children", {}
                ).get(child["run_id"], {"status": "not_run"})
        all_complete = all(
            item.get("status") == "completed"
            and item.get("updates") == continuation.TARGET_UPDATES
            and item.get("cursor") == continuation.TARGET_CURSOR
            and item.get("consumed_exposure_tokens") == continuation.TARGET_EXPOSURE_TOKENS
            for item in previous_statuses.values()
        )
        if all_complete:
            for child in protocol["children"]:
                if _existing_training(
                        protocol, child, _manifest_record(fork_manifest, child["run_id"])
                ) is None:
                    raise RuntimeError(f"{child['run_id']}: completed child has no recoverable final state")
            print(json.dumps({
                "status": "already_complete", "complete": True, "output_root": str(output_root),
                "run_status": str(history[-1][0]) if history else str(status_path),
                "reservation_created": False, "training_launched": False,
            }, sort_keys=True, allow_nan=False))
            return 0
        prior = _prior_continuation_charges(path, before, study_id, history)
        remaining_hours = _remaining_projected_hours(protocol, previous_statuses)
        plan = recovery_projection(
            protocol, before, prior_hours=prior["prior_continuation_charged_hours"],
            remaining_projected_hours=remaining_hours,
        )
        require_budget(plan)
        launch_path = _recovery_path(launch_path, recovery_index)
        status_path = _recovery_path(status_path, recovery_index)
    else:
        require_budget(plan)
        prior = {
            "continuation_study_id": study_id, "prior_continuation_charged_hours": 0.0,
            "prior_continuation_reservations": {}, "prior_continuation_reservation_count": 0,
        }
    launch = {
        "format": P.LAUNCH_FORMAT,
        "source_commit": source,
        "code_identity": code_identity,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "output_root": str(output_root),
        "run_ids": list(protocol["run_ids"]),
        "children": list(protocol["children"]),
        "parents": dict(protocol["parents"]),
        "parent_inheritance": {
            run_id: record["inheritance"] for run_id, record in fork_manifest["children"].items()
        },
        "sampler_order_identity": protocol["data"]["order_identity"],
        "masks": dict(protocol["masks"]),
        "endpoint": dict(protocol["endpoint"]),
        "exposure_accounting": dict(protocol["exposure_accounting"]),
        "final_checkpoints": {child["run_id"]: child["final_checkpoint"] for child in protocol["children"]},
        "fork_manifest": fork_manifest["path"],
        "fork_manifest_sha256": fork_manifest["sha256"],
        "data": dict(protocol["data"]),
        "projection": plan,
        "measured_costs": dict(protocol["projection"]),
        "authorized_hours": authorized_hours,
        "authorization_status": "explicitly_authorized_for_this_execution",
        "continuation_study_id": study_id,
        "ledger_before": before,
        "reservation_count": prior["prior_continuation_reservation_count"] + 1,
        "recovery_index": recovery_index,
        "recovery_of": None if not resume else {
            "path": str(output_root / "launch.json"),
            "sha256": sha256_file(output_root / "launch.json"),
        },
        "previous_status_history": [
            {"path": str(path), "sha256": sha256_file(path), "complete": status.get("complete"),
             "failure": status.get("failure")}
            for path, status in history
        ],
        **prior,
        "intermediate_checkpoint_selection": False,
    }
    _write_new(launch_path, launch)
    statuses = dict(previous_statuses)
    reservation_id = None
    failure = None
    started = time.perf_counter()
    try:
        with ledger.Reservation(
                f"native.continuation issue=78 study={study_id} recovery={recovery_index}",
                projected_hours=plan["projected_total_hours"], path=path, lock=lock,
                extra={
                    "native_acquisition_run_id": protocol["run_ids"][0],
                    "native_continuation_run_id": study_id,
                    "native_continuation_authorized_hours": P.ALLOCATION_HOURS,
                    "native_continuation_run_ids": list(protocol["run_ids"]),
                    "source_commit": source, "code_identity": code_identity,
                    "protocol_sha256": protocol_sha256,
                    "fork_manifest_sha256": fork_manifest["sha256"],
                    "prior_continuation_charged_hours": prior["prior_continuation_charged_hours"],
                    "recovery_index": recovery_index,
                }) as active:
            reservation_id = active.id
            if resume:
                statuses = _execute(
                    protocol, fork_manifest, reservation=active, resume=True,
                    recovery_index=recovery_index, previous_statuses=previous_statuses,
                )
            else:
                statuses = _execute(protocol, fork_manifest, reservation=active)
    except BaseException as error:
        failure = {"kind": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        statuses = _recover_statuses(protocol, statuses, recovery_index=recovery_index if resume else None)
    after = ledger.usage(path)
    charged = after["used_hours"] - before["used_hours"]
    cumulative_charged = prior["prior_continuation_charged_hours"] + charged
    allocation_limit_respected = cumulative_charged <= authorized_hours + P.HOUR_TOLERANCE
    complete = (failure is None and set(statuses) == set(protocol["run_ids"])
                and all(item.get("status") == "completed"
                        and item.get("updates") == continuation.TARGET_UPDATES
                        and item.get("cursor") == continuation.TARGET_CURSOR
                        and item.get("consumed_exposure_tokens") == continuation.TARGET_EXPOSURE_TOKENS
                        for item in statuses.values())
                and allocation_limit_respected)
    result = {
        "format": P.RUN_FORMAT,
        "source_commit": source,
        "code_identity": code_identity,
        "protocol_sha256": protocol_sha256,
        "complete": complete,
        "children": statuses,
        "checkpoints": {run_id: values["checkpoint"] for run_id, values in statuses.items()
                        if values.get("checkpoint") is not None},
        "final_checkpoints": {child["run_id"]: child["final_checkpoint"] for child in protocol["children"]},
        "parent_inheritance": launch["parent_inheritance"],
        "sampler_order_identity": protocol["data"]["order_identity"],
        "mask_identity": protocol["masks"]["identity"],
        "exposure_accounting": dict(protocol["exposure_accounting"]),
        "failure": failure,
        "failure_history": [
            {"path": str(path), "failure": status.get("failure")}
            for path, status in history if status.get("failure") is not None
        ] + ([{"path": str(status_path), "failure": failure}] if failure is not None else []),
        "previous_status_history": launch["previous_status_history"],
        "reservation_id": reservation_id,
        "reservation_count": prior["prior_continuation_reservation_count"] + (
            1 if reservation_id is not None else 0
        ),
        "wall_seconds": time.perf_counter() - started,
        "charged_hours": charged,
        "cumulative_continuation_charged_hours": cumulative_charged,
        "authorized_hours": authorized_hours,
        "authorization_status": launch["authorization_status"],
        "continuation_study_id": study_id,
        "recovery_index": recovery_index,
        **prior,
        "allocation_limit_respected": allocation_limit_respected,
        "projection": plan,
        "measured_costs": dict(protocol["projection"]),
        "ledger_before": before,
        "ledger_after": after,
        "official_evaluation_policy": dict(protocol["evaluation_policy"]),
        "intervention_policy": dict(protocol["intervention_policy"]),
        "evidence_boundary": "all six fixed endpoints are reported; no intermediate checkpoint or winner selection",
    }
    _write_new(status_path, result)
    print(json.dumps({
        "status": "recovered" if resume and complete else "incomplete" if not complete else "completed",
        "output_root": str(output_root), "run_status": str(status_path), "complete": complete,
        "charged_hours": charged, "cumulative_continuation_charged_hours": cumulative_charged,
        "reservation_id": reservation_id, "children": statuses, "failure": failure,
    }, sort_keys=True, allow_nan=False))
    return 0 if complete else 3


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m native.run_continuation", description=__doc__)
    result.add_argument("--protocol", default=str(P.DEFAULT_PROTOCOL_PATH))
    result.add_argument("--repository", default=str(Path(__file__).resolve().parents[3]))
    result.add_argument("--output-root", help="must match the frozen protocol output root")
    result.add_argument("--python", help="pinned native interpreter; required to match when executing")
    result.add_argument("--prepare-only", action="store_true", help="create and validate CPU-loaded forks only")
    result.add_argument("--execute", action="store_true", help="train only with exact explicit owner authorization")
    result.add_argument("--authorize-hours", type=float)
    result.add_argument("--resume", action="store_true",
                        help="recover the existing launch into versioned results; never overwrite prior history")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    return run(
        args.protocol, args.repository, output_root=args.output_root,
        execute=args.execute, prepare_only=args.prepare_only,
        authorize_hours=args.authorize_hours, python=args.python, resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
