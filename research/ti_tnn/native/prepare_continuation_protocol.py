"""Build the proposed six-child #78 continuation protocol without training.

The children retain ordered full-label exposure as their stopping rule. Answer masks
change only main-loss labels; they never remove prompt inputs or auxiliary targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

from . import answer_analysis
from . import continuation
from . import ledger
from .data import sha256_file

FORMAT = "native-six-child-continuation-protocol-v1"
FORK_MANIFEST_FORMAT = "native-six-child-continuation-forks-v1"
LAUNCH_FORMAT = "native-six-child-continuation-launch-v1"
RUN_FORMAT = "native-six-child-continuation-run-status-v1"
STATUS = "proposed_not_authorized_or_launched"
MODES = ("dense", "untyped", "typed")
ARMS = ("A", "B")
LOSS_MODES = continuation.LOSS_MODES
SPLITS = answer_analysis.PROGRAM_SPLITS
ALLOCATION_HOURS = 3.0
CONTINUATION_STUDY_ID = "78-native-continuation"
WATCHDOG_MULTIPLIER = 1.5
SHUTDOWN_ALLOWANCE_HOURS = 0.05
PLANNING_FRACTION = 0.8
HOUR_TOLERANCE = 1e-12
PARENT_ANSWER_DIAGNOSTICS_SECONDS = 203.0
ANSWER_DIAGNOSTICS_SECONDS_PER_CHILD = PARENT_ANSWER_DIAGNOSTICS_SECONDS / 3.0
DEFAULT_PARENT_RUN = Path(
    "/home/seanjazm27/projects/universa-recurrent-scaling/runs/"
    "ti_tnn_native_program_generalization/runs/20261110T210743-921461de"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_continuation"
)
DEFAULT_PROTOCOL_PATH = DEFAULT_OUTPUT_ROOT / "continuation_protocol.json"
SOURCE_FILES = (
    "prepare_continuation_protocol.py",
    "run_continuation.py",
    "continuation.py",
    "train.py",
    "checkpoint.py",
    "answer_analysis.py",
    "diagnose_program_results.py",
    "evaluate_program_study.py",
    "evaluate_state_path.py",
    "diagnose_learned_state.py",
    "reconstruct.py",
    "model.py",
    "data.py",
    "sampler.py",
    "independent/rescore_generations.py",
    "configs/engineering.json",
)

PARENT_ENDPOINT = {
    "epochs": continuation.PARENT_EPOCHS,
    "updates": continuation.PARENT_UPDATES,
    "exposure_tokens": continuation.PARENT_EXPOSURE_TOKENS,
    "cursor": continuation.PARENT_CURSOR,
}
ADDITIONAL_ENDPOINT = {
    "epochs": continuation.CONTINUATION_EPOCHS,
    "updates": continuation.ADDITIONAL_UPDATES,
    "exposure_tokens": continuation.ADDITIONAL_EXPOSURE_TOKENS,
}
TARGET_ENDPOINT = {
    "epochs": continuation.TARGET_EPOCHS,
    "updates": continuation.TARGET_UPDATES,
    "exposure_tokens": continuation.TARGET_EXPOSURE_TOKENS,
    "cursor": continuation.TARGET_CURSOR,
}
PER_EPOCH_ACCOUNTING = dict(answer_analysis.FULL_TRAIN_EPOCH)
ADDITIONAL_MAIN_LOSS = {
    "full_sequence": continuation.ADDITIONAL_EXPOSURE_TOKENS,
    "answer_eos": PER_EPOCH_ACCOUNTING["answer_eos_tokens"] * continuation.CONTINUATION_EPOCHS,
}
ADDITIONAL_ANSWER_PARTITIONS = {
    "answer_value_tokens": PER_EPOCH_ACCOUNTING["answer_value_tokens"] * continuation.CONTINUATION_EPOCHS,
    "formatting_eos_tokens": PER_EPOCH_ACCOUNTING["formatting_eos_tokens"] * continuation.CONTINUATION_EPOCHS,
}
EXPECTED_ANSWER_ADDITIONAL = 3342336
assert ADDITIONAL_MAIN_LOSS["answer_eos"] == EXPECTED_ANSWER_ADDITIONAL
assert sum(ADDITIONAL_ANSWER_PARTITIONS.values()) == EXPECTED_ANSWER_ADDITIONAL


def child_run_id(mode: str, loss_mode: str) -> str:
    if mode not in MODES or loss_mode not in LOSS_MODES:
        raise ValueError(f"unknown continuation child: {mode}/{loss_mode}")
    return f"78-{mode}-{loss_mode}"


def child_arm(loss_mode: str) -> str:
    if loss_mode == "full_sequence":
        return "A"
    if loss_mode == "answer_eos":
        return "B"
    raise ValueError(f"unknown loss mode: {loss_mode}")


def final_checkpoint_path(output_root, run_id: str) -> Path:
    return Path(output_root) / run_id / "checkpoints" / (
        f"final-update-{continuation.TARGET_UPDATES:08d}.pt"
    )


def initial_checkpoint_path(output_root, run_id: str) -> Path:
    return Path(output_root) / run_id / "checkpoints" / (
        f"fork-update-{continuation.PARENT_UPDATES:08d}.pt"
    )


def source_identities(repository) -> dict[str, str]:
    root = Path(repository).resolve() / "research" / "ti_tnn" / "native"
    return {name: sha256_file(root / name) for name in SOURCE_FILES}


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _finite_nonnegative(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _validate_parent_descriptor(mode: str, descriptor: Mapping, result: Mapping) -> None:
    path = Path(descriptor.get("path", ""))
    if (descriptor.get("step") != continuation.PARENT_UPDATES
            or descriptor.get("cursor") != continuation.PARENT_CURSOR
            or descriptor.get("consumed_loss_tokens") != continuation.PARENT_EXPOSURE_TOKENS
            or path.name != f"final-update-{continuation.PARENT_UPDATES:08d}.pt"):
        raise ValueError(f"{mode}: parent is not the completed #78 16-epoch endpoint")
    if (result.get("format") != "native-train-v1" or result.get("mode") != mode
            or result.get("step") != continuation.PARENT_UPDATES
            or result.get("cursor") != continuation.PARENT_CURSOR
            or result.get("consumed_loss_tokens") != continuation.PARENT_EXPOSURE_TOKENS
            or result.get("planned_updates") != continuation.PARENT_UPDATES):
        raise ValueError(f"{mode}: parent train result does not match the #78 endpoint")
    if result.get("final_checkpoint", {}).get("sha256") != descriptor.get("sha256"):
        raise ValueError(f"{mode}: parent checkpoint descriptors disagree")
    if sha256_file(path) != descriptor.get("sha256"):
        raise ValueError(f"{mode}: parent checkpoint bytes differ from the frozen descriptor")


def measure_parent_costs(parent_run=DEFAULT_PARENT_RUN) -> dict:
    """Read actual #78 train/checkpoint/wall costs and pin all referenced evidence."""
    run = Path(parent_run).resolve()
    launch_path = run / "launch.json"
    status_path = run / "run_status.json"
    diagnostic_path = run / "diagnostic_manifest.json"
    launch = _read_json(launch_path)
    status = _read_json(status_path)
    if not status.get("complete") or status.get("failure") is not None:
        raise ValueError("#78 parent run must be complete with no run-level failure")
    if launch.get("format") != "native-program-study-launch-v1":
        raise ValueError("unexpected #78 parent launch format")
    diagnostic = _read_json(diagnostic_path)
    if set(diagnostic.get("arms", {})) != set(MODES):
        raise ValueError("#78 answer diagnostics must cover all three parents")
    arms = {}
    order_identities = set()
    objective = None
    for mode in MODES:
        arm = run / "arms" / mode
        train_result_path = arm / "train-result.json"
        arm_status_path = arm / "status.json"
        result = _read_json(train_result_path)
        arm_status = _read_json(arm_status_path)
        if arm_status.get("status") != "completed":
            raise ValueError(f"{mode}: parent arm is not completed")
        descriptor = status.get("arms", {}).get(mode, {}).get("checkpoint")
        if not isinstance(descriptor, dict):
            raise ValueError(f"{mode}: parent run status has no final checkpoint")
        _validate_parent_descriptor(mode, descriptor, result)
        for name in ("evaluation.json", "generation_diagnostics.json", "reconstruction.json",
                     "answer_diagnostics.json"):
            if not (arm / name).is_file():
                raise FileNotFoundError(arm / name)
        training_seconds = _finite_nonnegative(result.get("training_seconds"), f"{mode} training_seconds")
        checkpoint_seconds = _finite_nonnegative(result.get("checkpoint_seconds"), f"{mode} checkpoint_seconds")
        wall_seconds = _finite_nonnegative(arm_status.get("wall_seconds"), f"{mode} wall_seconds")
        if training_seconds <= 0 or checkpoint_seconds <= 0 or wall_seconds <= training_seconds + checkpoint_seconds:
            raise ValueError(f"{mode}: insufficient positive parent timing measurements")
        dev = result.get("dev", {})
        dev_seconds = (
            _finite_nonnegative(dev.get("forward_seconds"), f"{mode} dev forward")
            + _finite_nonnegative(dev.get("data_loading_seconds"), f"{mode} dev loading")
        )
        if dev_seconds <= 0:
            raise ValueError(f"{mode}: parent has no measured full-dev cost")
        run_config = _read_json(arm / "train" / "run_config.json")
        training = run_config["training"]
        coefficients = {
            "lambda_balance": training["lambda_balance"],
            "lambda_route": training["lambda_route"],
            "lambda_state": training["lambda_state"],
        }
        if objective is None:
            objective = coefficients
        elif coefficients != objective:
            raise ValueError("parent arms do not share auxiliary coefficients")
        order_identities.add(descriptor["identities"]["order"])
        parent_dev_evaluations = continuation.PARENT_UPDATES // training["dev_every"] + 1
        child_dev_evaluations = sum(
            step % training["dev_every"] == 0
            for step in range(continuation.PARENT_UPDATES + 1, continuation.TARGET_UPDATES)
        ) + 1
        post_training_wall = wall_seconds - training_seconds - checkpoint_seconds
        parent_dev_wall = parent_dev_evaluations * dev_seconds
        fixed_endpoint_wall = post_training_wall - parent_dev_wall
        if fixed_endpoint_wall <= 0:
            raise ValueError(f"{mode}: parent wall cost does not cover fixed-endpoint evaluation")
        arms[mode] = {
            "parent_checkpoint": str(Path(descriptor["path"]).resolve()),
            "expected_parent_sha256": descriptor["sha256"],
            "parent_checkpoint_descriptor": dict(descriptor),
            "model_identity": descriptor["state_hashes"]["model"],
            "optimizer_identity": descriptor["state_hashes"]["optimizer"],
            "sampler_identity": descriptor["state_hashes"]["sampler"],
            "parent_scheduler_identity": descriptor["state_hashes"]["scheduler"],
            "data_identity": descriptor["identities"]["data"],
            "order_identity": descriptor["identities"]["order"],
            "train_result_sha256": sha256_file(train_result_path),
            "arm_status_sha256": sha256_file(arm_status_path),
            "evaluation_sha256": sha256_file(arm / "evaluation.json"),
            "generation_diagnostics_sha256": sha256_file(arm / "generation_diagnostics.json"),
            "reconstruction_sha256": sha256_file(arm / "reconstruction.json"),
            "answer_diagnostics_sha256": sha256_file(arm / "answer_diagnostics.json"),
            "training_seconds": training_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "wall_seconds": wall_seconds,
            "dev_seconds_per_evaluation": dev_seconds,
            "parent_dev_evaluations": parent_dev_evaluations,
            "child_dev_evaluations": child_dev_evaluations,
            "fixed_endpoint_evaluation_rescore_fp32_seconds": fixed_endpoint_wall,
            "run_config": run_config,
        }
    if len(order_identities) != 1:
        raise ValueError("#78 parents do not share one ordered sampler identity")
    return {
        "parent_run": str(run),
        "parent_run_id": launch["run_id"],
        "launch_sha256": sha256_file(launch_path),
        "run_status_sha256": sha256_file(status_path),
        "diagnostic_manifest_sha256": sha256_file(diagnostic_path),
        "data_root": str(Path(launch["data"]).resolve()),
        "data_manifest_sha256": launch["data_manifest_sha256"],
        "tokenizer": str(Path(launch["tokenizer"]).resolve()),
        "tokenizer_sha256": launch["tokenizer_sha256"],
        "replay": str(Path(launch["replay"]).resolve()),
        "replay_sha256": launch["replay_sha256"],
        "objective": objective,
        "arms": arms,
    }


def _projection_from_measurements(measurements: Mapping, before: Mapping | None,
                                  answer_diagnostics_seconds: float) -> dict:
    measured_total = _finite_nonnegative(
        answer_diagnostics_seconds, "parent answer-diagnostics seconds"
    )
    if measured_total < PARENT_ANSWER_DIAGNOSTICS_SECONDS:
        raise ValueError("answer-diagnostics measurement must be at least the observed 203 seconds")
    per_child = {}
    per_parent = {}
    for mode in MODES:
        measured = measurements["arms"][mode]
        for loss_mode in LOSS_MODES:
            run_id = child_run_id(mode, loss_mode)
            components = {
                "training": 3.0 * measured["training_seconds"],
                "checkpoint": 3.0 * measured["checkpoint_seconds"],
                "periodic_and_final_dev": measured["child_dev_evaluations"] * measured["dev_seconds_per_evaluation"],
                "official_evaluation_rescore_fp32_replay": measured[
                    "fixed_endpoint_evaluation_rescore_fp32_seconds"
                ],
                "answer_diagnostics": measured_total / 3.0,
                "fork_initialization": measured["checkpoint_seconds"],
            }
            components["total"] = sum(components.values())
            per_child[run_id] = components
        per_parent[mode] = {
            key: measured[key]
            for key in (
                "train_result_sha256", "arm_status_sha256", "training_seconds",
                "checkpoint_seconds", "wall_seconds", "dev_seconds_per_evaluation",
                "parent_dev_evaluations", "child_dev_evaluations",
                "fixed_endpoint_evaluation_rescore_fp32_seconds",
            )
        }
    total_seconds = sum(values["total"] for values in per_child.values())
    projected_hours = total_seconds / 3600.0
    watchdog = projected_hours * WATCHDOG_MULTIPLIER
    required = watchdog + SHUTDOWN_ALLOWANCE_HOURS
    remaining = None if before is None else _finite_nonnegative(
        before.get("native_remaining_hours"), "live native_remaining_hours"
    )
    fits_watchdog = required <= ALLOCATION_HOURS + HOUR_TOLERANCE
    fits_planning = projected_hours <= PLANNING_FRACTION * ALLOCATION_HOURS + HOUR_TOLERANCE
    fits_live = remaining is not None and (
        required <= remaining + HOUR_TOLERANCE
        and ALLOCATION_HOURS <= remaining + HOUR_TOLERANCE
    )
    return {
        "measurement_basis": (
            "actual completed #78 per-mode train/checkpoint/arm-wall costs; 48/16 training and "
            "checkpoint scaling; fixed official evaluation, independent rescore and FP32 replay once per child"
        ),
        "fixed_cost_boundary": (
            "#78 did not separately time official evaluation, independent rescore, and FP32 replay; "
            "their measured combined wall cost is retained without fabricated splits"
        ),
        "included_fixed_endpoint_components": [
            "ordinary_full_official_evaluation", "independent_per_child_rescore", "cpu_fp32_replay",
        ],
        "parent_answer_diagnostics_seconds_all_three": measured_total,
        "answer_diagnostics_seconds_per_child": measured_total / 3.0,
        "answer_diagnostics_minimum_seconds_per_child": ANSWER_DIAGNOSTICS_SECONDS_PER_CHILD,
        "fork_initialization_budget": "one full measured parent checkpoint interval per child",
        "shutdown_allowance_hours": SHUTDOWN_ALLOWANCE_HOURS,
        "per_parent_measured_seconds": per_parent,
        "per_child_seconds": per_child,
        "total_seconds_all_six": total_seconds,
        "projected_total_hours": projected_hours,
        "allocation_request_hours": ALLOCATION_HOURS,
        "maximum_watchdog_hours": watchdog,
        "required_native_hours": required,
        "planning_limit_hours": PLANNING_FRACTION * ALLOCATION_HOURS,
        "planning_headroom_hours": PLANNING_FRACTION * ALLOCATION_HOURS - projected_hours,
        "allocation_headroom_hours": ALLOCATION_HOURS - required,
        "live_native_remaining_hours_at_proposal": remaining,
        "live_native_headroom_hours": None if remaining is None else remaining - required,
        "fits_watchdog_and_shutdown": fits_watchdog,
        "fits_80_percent_planning_margin": fits_planning,
        "fits_live_native_ledger": fits_live,
        "fits_budget": fits_watchdog and fits_planning and fits_live,
    }


def compute_projection(parent_run=DEFAULT_PARENT_RUN, *, before=None,
                       answer_diagnostics_seconds=PARENT_ANSWER_DIAGNOSTICS_SECONDS) -> dict:
    return _projection_from_measurements(
        measure_parent_costs(parent_run), before, answer_diagnostics_seconds
    )


def _validate_mask_identity(data_root: Path, data_manifest_sha256: str, mask_root: Path) -> dict:
    manifest_path = mask_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("format") != answer_analysis.FORMAT:
        raise ValueError("unexpected answer-mask companion format")
    if manifest.get("source_manifest", {}).get("sha256") != data_manifest_sha256:
        raise ValueError("answer masks belong to a different source data manifest")
    train = manifest.get("splits", {}).get("train", {})
    actual = {name: train.get(name) for name in PER_EPOCH_ACCOUNTING}
    if actual != PER_EPOCH_ACCOUNTING:
        raise ValueError(f"answer-mask train counts differ from the required contract: {actual}")
    for split in SPLITS:
        for name in answer_analysis.MASK_ARRAY_NAMES:
            entry = manifest["splits"][split]["arrays"][name]
            path = Path(entry["path"])
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"{split}/{name}: answer-mask bytes differ from their manifest")
    return {
        "root": str(mask_root),
        "identity": continuation._path_identity(mask_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "arrays": list(answer_analysis.MASK_ARRAY_NAMES),
        "train_per_epoch": dict(PER_EPOCH_ACCOUNTING),
        "model_input_arrays_unchanged": list(answer_analysis.MODEL_INPUT_ARRAYS),
        "auxiliary_targets_unchanged": ["route_targets", "route_mask", "state_targets", "state_mask"],
        "mask_use": "main-loss label selection only; never model input or ordered-exposure stopping",
    }


def _assert_separate_output(output_root: Path, parent_run: Path, data_root: Path) -> None:
    for immutable in (parent_run, data_root):
        if output_root == immutable or immutable in output_root.parents:
            raise ValueError(f"continuation output must be outside immutable directory: {immutable}")


def build_protocol(parent_run=DEFAULT_PARENT_RUN, *, output_root=DEFAULT_OUTPUT_ROOT,
                   repository=None, before=None, answer_diagnostics_seconds=PARENT_ANSWER_DIAGNOSTICS_SECONDS,
                   mask_directory=None) -> dict:
    if before is None:
        before = ledger.usage(ledger.validate_shared_reservation_paths()[0])
    measurements = measure_parent_costs(parent_run)
    parent_run = Path(parent_run).resolve()
    output_root = Path(output_root).resolve()
    data_root = Path(measurements["data_root"]).resolve()
    _assert_separate_output(output_root, parent_run, data_root)
    if sha256_file(data_root / "manifest.json") != measurements["data_manifest_sha256"]:
        raise ValueError("parent data manifest differs from the completed #78 launch")
    if sha256_file(measurements["tokenizer"]) != measurements["tokenizer_sha256"]:
        raise ValueError("parent tokenizer differs from the completed #78 launch")
    if sha256_file(measurements["replay"]) != measurements["replay_sha256"]:
        raise ValueError("parent replay spec differs from the completed #78 launch")
    mask_root = Path(mask_directory).resolve() if mask_directory is not None else data_root / "answer_masks"
    masks = _validate_mask_identity(data_root, measurements["data_manifest_sha256"], mask_root)
    projection = _projection_from_measurements(
        measurements, before, answer_diagnostics_seconds
    )
    parents = {}
    children = []
    for mode in MODES:
        measured = measurements["arms"][mode]
        parent = {
            "mode": mode,
            "parent_run_id": measurements["parent_run_id"],
            "parent_run": str(parent_run),
            "checkpoint_path": measured["parent_checkpoint"],
            "expected_parent_sha256": measured["expected_parent_sha256"],
            "model_identity": measured["model_identity"],
            "optimizer_identity": measured["optimizer_identity"],
            "sampler_identity": measured["sampler_identity"],
            "parent_scheduler_identity": measured["parent_scheduler_identity"],
            "data_identity": measured["data_identity"],
            "order_identity": measured["order_identity"],
            "endpoint": dict(PARENT_ENDPOINT),
            "inheritance": {
                "model": "exact tensors from expected_parent_sha256",
                "optimizer": "exact state and param groups except the new common learning rate",
                "sampler": "exact ordered sampler state, cursor, data seed and order identity",
                "rng": "exact Python, NumPy, Torch and saved CUDA RNG states",
                "scheduler": "not inherited; a new constant schedule is constructed",
            },
        }
        parents[mode] = parent
        for loss_mode in LOSS_MODES:
            run_id = child_run_id(mode, loss_mode)
            arm = child_arm(loss_mode)
            child_output = output_root / run_id
            children.append({
                "child_id": f"{mode}-{arm}",
                "pair_id": mode,
                "arm": arm,
                "mode": mode,
                "run_id": run_id,
                "loss_mode": loss_mode,
                "parent_mode": mode,
                "parent_checkpoint": parent["checkpoint_path"],
                "expected_parent_sha256": parent["expected_parent_sha256"],
                "output": str(child_output),
                "config_path": str(child_output / "run_config.json"),
                "initial_checkpoint": str(initial_checkpoint_path(output_root, run_id)),
                "final_checkpoint": str(final_checkpoint_path(output_root, run_id)),
                "answer_masks": masks["root"] if loss_mode == "answer_eos" else None,
                "answer_masks_identity": masks["identity"] if loss_mode == "answer_eos" else None,
                "schedule": {
                    "learning_rate": continuation.CONTINUATION_LEARNING_RATE,
                    "warmup_fraction": 0.0,
                    "final_lr_fraction": 1.0,
                    "inherits_parent_scheduler": False,
                    "scheduler": continuation.CONTINUATION_SCHEDULER,
                    "additional_epochs": continuation.CONTINUATION_EPOCHS,
                    "additional_updates": continuation.ADDITIONAL_UPDATES,
                },
                "additional_exposure_tokens": continuation.ADDITIONAL_EXPOSURE_TOKENS,
                "additional_main_loss_tokens": ADDITIONAL_MAIN_LOSS[loss_mode],
                "target": dict(TARGET_ENDPOINT),
                "projected_seconds": projection["per_child_seconds"][run_id],
            })
    repository = Path(repository).resolve() if repository is not None else Path(__file__).resolve().parents[3]
    protocol = {
        "format": FORMAT,
        "status": STATUS,
        "issue": 78,
        "continuation_study_id": CONTINUATION_STUDY_ID,
        "purpose": "three paired full-sequence versus answer+EOS main-loss continuations; preparation only until authorized",
        "parent_run": {
            "path": str(parent_run),
            "run_id": measurements["parent_run_id"],
            "launch_sha256": measurements["launch_sha256"],
            "run_status_sha256": measurements["run_status_sha256"],
            "diagnostic_manifest_sha256": measurements["diagnostic_manifest_sha256"],
            "immutable": True,
        },
        "parents": parents,
        "children": children,
        "run_ids": [child["run_id"] for child in children],
        "output_root": str(output_root),
        "objective": {
            "full_sequence": "existing full-sequence LM plus unchanged load-balance, route and state objectives",
            "answer_eos": "answer+EOS-selected main LM loss plus unchanged load-balance, route and state objectives",
            "main_loss_rule": "mean selected-token cross entropy per batch",
            "auxiliary_coefficients": dict(measurements["objective"]),
            "auxiliary_targets_identical_within_pair": True,
            "prompt_inputs_identical_within_pair": True,
        },
        "auxiliary_coefficients": dict(measurements["objective"]),
        "data": {
            "root": str(data_root),
            "manifest_sha256": measurements["data_manifest_sha256"],
            "tokenizer_path": measurements["tokenizer"],
            "tokenizer_sha256": measurements["tokenizer_sha256"],
            "replay": measurements["replay"],
            "replay_sha256": measurements["replay_sha256"],
            "order_identity": next(iter({parent["order_identity"] for parent in parents.values()})),
            "immutable": True,
        },
        "masks": masks,
        "endpoint": {
            "parent": dict(PARENT_ENDPOINT),
            "additional": dict(ADDITIONAL_ENDPOINT),
            "target": dict(TARGET_ENDPOINT),
            "updates_per_epoch": continuation.UPDATES_PER_EPOCH,
            "sequences_per_epoch": continuation.SEQUENCES_PER_EPOCH,
            "full_label_exposure_tokens_per_epoch": continuation.EXPOSURE_TOKENS_PER_EPOCH,
            "stopping_rule": "ordered full-label labels[:,1:] != -100 exposure, never selected main-loss tokens",
            "paired_matching": "ordered input exposure and complete epochs, not masked-token counts",
        },
        "exposure_accounting": {
            "per_epoch": dict(PER_EPOCH_ACCOUNTING),
            "additional_main_loss_tokens": dict(ADDITIONAL_MAIN_LOSS),
            "additional_answer_eos_tokens": EXPECTED_ANSWER_ADDITIONAL,
            "additional_answer_value_tokens": ADDITIONAL_ANSWER_PARTITIONS["answer_value_tokens"],
            "additional_formatting_eos_tokens": ADDITIONAL_ANSWER_PARTITIONS["formatting_eos_tokens"],
            "full_sequence_additional_exposure_tokens": continuation.ADDITIONAL_EXPOSURE_TOKENS,
            "answer_eos_additional_exposure_tokens": continuation.ADDITIONAL_EXPOSURE_TOKENS,
            "full_sequence_target_main_loss_tokens": continuation.TARGET_EXPOSURE_TOKENS,
            "answer_eos_target_exposure_tokens": continuation.TARGET_EXPOSURE_TOKENS,
            "answer_eos_whole_history_main_loss_tokens": (
                continuation.PARENT_EXPOSURE_TOKENS + EXPECTED_ANSWER_ADDITIONAL
            ),
        },
        "constant_lr_policy": {
            "learning_rate": continuation.CONTINUATION_LEARNING_RATE,
            "schedule": "new constant LR for all six children",
            "warmup": False,
            "warmup_fraction": 0.0,
            "inherits_parent_cosine_scheduler": False,
            "additional_epochs": continuation.CONTINUATION_EPOCHS,
        },
        "evaluation_policy": {
            "checkpoint": "only each child's final-update-00013568.pt",
            "intermediate_checkpoint_selection": False,
            "winner_selection": "none; all six fixed endpoints are evaluated and retained",
            "official": "ordinary full native.evaluate_program_study.evaluate on all five fixed splits",
            "evaluation_examples_per_split": 128,
            "independent_rescore": "native.independent.rescore_generations with an explicit per-child evaluation path",
            "reconstruction": "unautocast CPU FP32 full/cached replay with fixed replay tokens",
            "answer_diagnostics": "native.diagnose_program_results with explicit child mappings",
            "development_diagnostics": "reuse existing populations and fixed selections; no new population",
        },
        "intervention_policy": {
            "operation": "existing pass-two final-instruction writer-update suppression",
            "examples_per_split": 32,
            "selection": "existing frozen selections; reused development diagnostics only",
            "intermediate_outcome_selection": False,
        },
        "projection": projection,
        "projected_total_hours": projection["projected_total_hours"],
        "allocation_request_hours": ALLOCATION_HOURS,
        "authorization": {
            "status": STATUS,
            "proposal_only": True,
            "owner_approved": False,
            "required_future_flag": "--execute --authorize-hours 3.0",
            "native_cap": "existing cumulative native v3 cap; no new cap or WSP",
        },
        "execution_policy": {
            "default": "read-only dry-run of protocol and live ledger; existing run directories are refused without --resume",
            "prepare_only": "CPU restore/save with create_fork(device='cuda', load_device='cpu'); no optimizer step or reservation",
            "execute": "exact 3.0 h authorization, clean committed source and completed validated fork manifest required",
            "resume": "verify launch/protocol/source/data/authorization; reuse valid stages and write versioned recovery results only",
            "reservations": "one per execution; completed runs open none; all study charges count toward the same 3.0 h cap",
            "recovery_budget": "prior charges + 1.5 * remaining complete-comparison projection + 0.05 shutdown <= 3.0 h; never truncate a control",
            "failure_policy": "preserve all child and run-level partial/failure records; continue ordinary child failures",
        },
        "source_files_sha256": source_identities(repository),
    }
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: Mapping) -> None:
    if protocol.get("format") != FORMAT or protocol.get("status") != STATUS:
        raise ValueError("unexpected continuation protocol format or status")
    if protocol.get("allocation_request_hours") != ALLOCATION_HOURS:
        raise ValueError("continuation allocation request must be exactly 3.0 hours")
    if protocol.get("continuation_study_id") != CONTINUATION_STUDY_ID:
        raise ValueError("continuation study identity differs from the frozen six-child contract")
    endpoint = protocol.get("endpoint", {})
    for name, expected in (("parent", PARENT_ENDPOINT), ("additional", ADDITIONAL_ENDPOINT),
                           ("target", TARGET_ENDPOINT)):
        if endpoint.get(name) != expected:
            raise ValueError(f"continuation {name} endpoint differs from the six-child contract")
    accounting = protocol.get("exposure_accounting", {})
    if (accounting.get("per_epoch") != PER_EPOCH_ACCOUNTING
            or accounting.get("additional_main_loss_tokens") != ADDITIONAL_MAIN_LOSS
            or accounting.get("additional_answer_eos_tokens") != EXPECTED_ANSWER_ADDITIONAL
            or accounting.get("additional_answer_value_tokens") != 786432
            or accounting.get("additional_formatting_eos_tokens") != 2555904):
        raise ValueError("continuation exposure and selected main-loss accounting are inconsistent")
    children = protocol.get("children")
    if not isinstance(children, list) or len(children) != 6:
        raise ValueError("continuation protocol must declare exactly six children")
    expected = {(mode, loss_mode) for mode in MODES for loss_mode in LOSS_MODES}
    actual = {(child.get("mode"), child.get("loss_mode")) for child in children}
    if actual != expected:
        raise ValueError("continuation children do not form the three required A/B pairs")
    if len({child.get("run_id") for child in children}) != 6 \
            or len({child.get("output") for child in children}) != 6 \
            or len({child.get("final_checkpoint") for child in children}) != 6:
        raise ValueError("every continuation child needs independent run/output/checkpoint identities")
    if protocol.get("run_ids") != [child.get("run_id") for child in children]:
        raise ValueError("protocol run_ids must exactly match the ordered six child declarations")
    if set(protocol.get("parents", {})) != set(MODES):
        raise ValueError("continuation protocol must declare exactly the three #78 parents")
    coefficients = protocol.get("auxiliary_coefficients")
    if protocol.get("objective", {}).get("auxiliary_coefficients") != coefficients \
            or set(coefficients or {}) != {"lambda_balance", "lambda_route", "lambda_state"}:
        raise ValueError("objective and top-level auxiliary coefficients are inconsistent")
    lr_policy = protocol.get("constant_lr_policy", {})
    if lr_policy.get("learning_rate") != continuation.CONTINUATION_LEARNING_RATE \
            or lr_policy.get("warmup") is not False \
            or lr_policy.get("inherits_parent_cosine_scheduler") is not False \
            or lr_policy.get("additional_epochs") != continuation.CONTINUATION_EPOCHS:
        raise ValueError("constant_lr_policy must be new 3e-5, no warmup, for 48 epochs")
    output_root = Path(protocol.get("output_root", "")).resolve()
    for child in children:
        run_id = child.get("run_id")
        if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id \
                or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
            raise ValueError("child run_id must be one nonempty path component")
        if child.get("arm") != child_arm(child["loss_mode"]) \
                or child.get("pair_id") != child.get("mode") \
                or child.get("parent_mode") != child.get("mode"):
            raise ValueError(f"{run_id}: invalid pair or arm identity")
        if Path(child["output"]).resolve() != output_root / run_id:
            raise ValueError(f"{run_id}: output differs from its independent run ID")
        if Path(child["final_checkpoint"]).resolve() != final_checkpoint_path(output_root, run_id):
            raise ValueError(f"{run_id}: final checkpoint is not final-update-00013568.pt")
        if Path(child["initial_checkpoint"]).resolve() != initial_checkpoint_path(output_root, run_id):
            raise ValueError(f"{run_id}: unexpected initial fork checkpoint")
        schedule = child.get("schedule", {})
        if (schedule.get("learning_rate") != continuation.CONTINUATION_LEARNING_RATE
                or schedule.get("warmup_fraction") != 0.0
                or schedule.get("inherits_parent_scheduler") is not False
                or schedule.get("additional_epochs") != continuation.CONTINUATION_EPOCHS
                or schedule.get("additional_updates") != continuation.ADDITIONAL_UPDATES):
            raise ValueError(f"{run_id}: continuation schedule is not new constant 3e-5 for 48 epochs")
        if child.get("additional_exposure_tokens") != continuation.ADDITIONAL_EXPOSURE_TOKENS \
                or child.get("additional_main_loss_tokens") != ADDITIONAL_MAIN_LOSS[child["loss_mode"]]:
            raise ValueError(f"{run_id}: exposure versus main-loss accounting differs")
        has_masks = child.get("answer_masks") is not None
        if has_masks != (child["loss_mode"] == "answer_eos") \
                or has_masks != (child.get("answer_masks_identity") is not None):
            raise ValueError(f"{run_id}: answer-mask identity does not match its loss mode")
        if has_masks and (Path(child["answer_masks"]).resolve() != Path(protocol["masks"]["root"]).resolve()
                          or child["answer_masks_identity"] != protocol["masks"]["identity"]):
            raise ValueError(f"{run_id}: child is not bound to the frozen answer-mask identity")
        parent = protocol.get("parents", {}).get(child["mode"], {})
        if child.get("expected_parent_sha256") != parent.get("expected_parent_sha256") \
                or child.get("parent_checkpoint") != parent.get("checkpoint_path"):
            raise ValueError(f"{run_id}: child does not reference its own parent and expected SHA")
    projected = protocol.get("projected_total_hours")
    if isinstance(projected, bool) or not isinstance(projected, (int, float)) \
            or not math.isfinite(projected) or projected <= 0:
        raise ValueError("projected_total_hours must be finite and positive")
    if protocol.get("projection", {}).get("projected_total_hours") != projected:
        raise ValueError("top-level and detailed projections disagree")
    if protocol.get("projection", {}).get("answer_diagnostics_seconds_per_child", 0) \
            < ANSWER_DIAGNOSTICS_SECONDS_PER_CHILD:
        raise ValueError("answer diagnostics are under-budgeted")
    if protocol.get("evaluation_policy", {}).get("intermediate_checkpoint_selection") is not False \
            or protocol.get("intervention_policy", {}).get("intermediate_outcome_selection") is not False:
        raise ValueError("continuation must not select intermediate outcomes")


def prepare(parent_run=DEFAULT_PARENT_RUN, *, output_root=DEFAULT_OUTPUT_ROOT, repository=None,
            output=None, before=None, mask_directory=None) -> dict:
    """Write a new protocol and SHA sidecar only when explicitly requested by the caller."""
    if before is None:
        before = ledger.usage(ledger.validate_shared_reservation_paths()[0])
    protocol = build_protocol(
        parent_run, output_root=output_root, repository=repository, before=before,
        mask_directory=mask_directory,
    )
    if not protocol["projection"]["fits_budget"]:
        raise RuntimeError("measured continuation projection does not fit its allocation and live ledger guards")
    output = Path(output).resolve() if output is not None else Path(output_root).resolve() / "continuation_protocol.json"
    sidecar = output.with_suffix(".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite continuation protocol or sidecar: {output}")
    encoded = (json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    sidecar.write_text(hashlib.sha256(encoded).hexdigest() + "\n")
    return protocol


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m native.prepare_continuation_protocol", description=__doc__)
    result.add_argument("--parent-run", default=str(DEFAULT_PARENT_RUN))
    result.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    result.add_argument("--repository", default=str(Path(__file__).resolve().parents[3]))
    result.add_argument("--mask-directory")
    result.add_argument("--output", help="new protocol JSON; defaults to OUTPUT_ROOT/continuation_protocol.json")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    protocol = prepare(
        args.parent_run, output_root=args.output_root, repository=args.repository,
        output=args.output, mask_directory=args.mask_directory,
    )
    print(json.dumps({
        "status": protocol["status"], "children": protocol["run_ids"],
        "projected_total_hours": protocol["projected_total_hours"],
        "required_native_hours": protocol["projection"]["required_native_hours"],
        "protocol": str(Path(args.output).resolve()) if args.output else str(
            Path(args.output_root).resolve() / "continuation_protocol.json"
        ),
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
