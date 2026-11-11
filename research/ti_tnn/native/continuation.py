"""Explicit checkpoint forks for paired native continuation training.

A fork copies a validated final parent into a new child directory.  Model,
optimizer, sampler, and RNG state survive the copy; the parent's cosine
scheduler does not.  Children share a constant 3e-5 learning rate for 48
complete additional epochs and stop on ordered full-label exposure.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from . import checkpoint as C
from .config import NativeConfig
from .data import PackedDataset
from .sampler import PackedSequenceSampler, identity_hash


CONTINUATION_FORMAT = "native-checkpoint-fork-v1"
LOSS_MODES = ("full_sequence", "answer_eos")
CONTINUATION_LEARNING_RATE = 3e-5
CONTINUATION_EPOCHS = 48
CONTINUATION_SCHEDULER = (
    "constant learning rate 3e-5 for 48 complete additional epochs; "
    "no warmup; parent cosine scheduler state is not inherited"
)

PARENT_EPOCHS = 16
PARENT_UPDATES = 3392
PARENT_EXPOSURE_TOKENS = 5570640
PARENT_CURSOR = 27040
UPDATES_PER_EPOCH = 212
EXPOSURE_TOKENS_PER_EPOCH = 348165
SEQUENCES_PER_EPOCH = 1690
ADDITIONAL_UPDATES = 10176
ADDITIONAL_EXPOSURE_TOKENS = 16711920
TARGET_EPOCHS = 64
TARGET_UPDATES = 13568
TARGET_EXPOSURE_TOKENS = 22282560
TARGET_CURSOR = 108160

PROGRAM_GENERALIZATION_CONTRACT = {
    "parent_epochs": PARENT_EPOCHS,
    "parent_updates": PARENT_UPDATES,
    "parent_exposure_tokens": PARENT_EXPOSURE_TOKENS,
    "parent_cursor": PARENT_CURSOR,
    "updates_per_epoch": UPDATES_PER_EPOCH,
    "exposure_tokens_per_epoch": EXPOSURE_TOKENS_PER_EPOCH,
    "sequences_per_epoch": SEQUENCES_PER_EPOCH,
    "additional_epochs": CONTINUATION_EPOCHS,
    "additional_updates": ADDITIONAL_UPDATES,
    "additional_exposure_tokens": ADDITIONAL_EXPOSURE_TOKENS,
    "target_epochs": TARGET_EPOCHS,
    "target_updates": TARGET_UPDATES,
    "target_exposure_tokens": TARGET_EXPOSURE_TOKENS,
    "target_cursor": TARGET_CURSOR,
}


def _json_bytes(payload) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _path_identity(path) -> str:
    path = Path(path)
    if path.is_file():
        return C.sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"answer-mask directory is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(C.sha256_file(item)))
    return digest.hexdigest()


def build_constant_scheduler(
    optimizer: torch.optim.Optimizer,
    total_updates: int | None = None,
    *,
    learning_rate: float = CONTINUATION_LEARNING_RATE,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Start a new no-warmup constant schedule without reading parent scheduler state."""
    del total_updates
    if learning_rate != CONTINUATION_LEARNING_RATE:
        raise ValueError(f"continuation learning rate must be {CONTINUATION_LEARNING_RATE}")
    inherited_initial_lrs = [group.get("initial_lr") for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
        group["initial_lr"] = learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    for group, inherited in zip(optimizer.param_groups, inherited_initial_lrs):
        if inherited is not None:
            group["initial_lr"] = inherited
    return scheduler


def continuation_plan(
    parent: Mapping,
    sampler: PackedSequenceSampler,
    *,
    additional_epochs: int = CONTINUATION_EPOCHS,
) -> dict:
    """Derive an epoch-aligned ordered-exposure endpoint from restored parent state."""
    if isinstance(additional_epochs, bool) or not isinstance(additional_epochs, int) \
            or additional_epochs <= 0:
        raise ValueError("additional_epochs must be a positive integer")
    state = parent.get("sampler")
    if state is None:
        raise C.CheckpointValidationError("parent checkpoint has no sampler state")
    parent_updates = parent.get("step")
    parent_cursor = parent.get("cursor")
    parent_exposure = parent.get("consumed_loss_tokens")
    for name, value in (
        ("parent step", parent_updates),
        ("parent cursor", parent_cursor),
        ("parent exposure", parent_exposure),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise C.CheckpointValidationError(f"{name} must be a positive integer")
    if state.get("batches_consumed") != parent_updates or state.get("cursor") != parent_cursor \
            or state.get("consumed_loss_tokens") != parent_exposure:
        raise C.CheckpointValidationError("parent counters differ from its sampler state")
    if sampler.cursor != parent_cursor or sampler.batches_consumed != parent_updates \
            or sampler.consumed_loss_tokens != parent_exposure:
        raise C.CheckpointIdentityMismatch("live sampler does not match the restored parent")
    if sampler.offset != 0 or sampler.epoch <= 0:
        raise C.CheckpointValidationError("parent must end at a complete epoch boundary")
    parent_epochs = sampler.epoch
    updates_per_epoch = len(sampler)
    sequences_per_epoch = len(sampler._sequences)
    if parent_updates != parent_epochs * updates_per_epoch:
        raise C.CheckpointValidationError("parent updates do not form complete epochs")
    if parent_exposure % parent_epochs:
        raise C.CheckpointValidationError("parent exposure does not form complete epochs")
    exposure_per_epoch = parent_exposure // parent_epochs
    additional_updates = additional_epochs * updates_per_epoch
    additional_exposure = additional_epochs * exposure_per_epoch
    return {
        "parent_epochs": parent_epochs,
        "parent_updates": parent_updates,
        "parent_exposure_tokens": parent_exposure,
        "parent_cursor": parent_cursor,
        "updates_per_epoch": updates_per_epoch,
        "exposure_tokens_per_epoch": exposure_per_epoch,
        "sequences_per_epoch": sequences_per_epoch,
        "additional_epochs": additional_epochs,
        "additional_updates": additional_updates,
        "additional_exposure_tokens": additional_exposure,
        "target_epochs": parent_epochs + additional_epochs,
        "target_updates": parent_updates + additional_updates,
        "target_exposure_tokens": parent_exposure + additional_exposure,
        "target_cursor": parent_cursor + additional_epochs * sequences_per_epoch,
    }


def validate_program_generalization_contract(plan: Mapping) -> None:
    """Pin the completed #78 16-to-64 epoch endpoints when that contract is requested."""
    mismatches = {
        name: (plan.get(name), expected)
        for name, expected in PROGRAM_GENERALIZATION_CONTRACT.items()
        if plan.get(name) != expected
    }
    if mismatches:
        raise C.CheckpointValidationError(
            f"continuation does not match completed #78 endpoints: {mismatches}"
        )


def _parent_record(parent: Mapping) -> dict:
    config = parent["config"]
    identities = parent["identities"]
    hashes = parent["state_hashes"]
    return {
        "checkpoint_path": str(Path(parent["path"]).resolve()),
        "checkpoint_sha256": parent["sha256"],
        "run_id": config.get("run_id", config.get("acquisition_run_id")),
        "parent_acquisition_run_id": config.get("acquisition_run_id"),
        "parent_reservation_id": config.get("reservation_id"),
        "config_identity": identities["config"],
        "code_identity": identities["code"],
        "data_identity": identities["data"],
        "order_identity": identities["order"],
        "model_identity": hashes["model"],
        "optimizer_identity": hashes["optimizer"],
        "scheduler_identity": hashes["scheduler"],
        "sampler_identity": hashes["sampler"],
        "rng_identity": identity_hash(parent["rng"]),
        "step": parent["step"],
        "cursor": parent["cursor"],
        "exposure_tokens": parent["consumed_loss_tokens"],
    }


def fork_checkpoint_identities(parent_record: Mapping, run_id: str, loss_mode: str) -> dict:
    """Identities retained in every child checkpoint, not just its JSON configuration."""
    return {
        "fork_run_id": run_id,
        "loss_mode": loss_mode,
        "parent_checkpoint_sha256": parent_record["checkpoint_sha256"],
        "parent_config": parent_record["config_identity"],
        "parent_code": parent_record["code_identity"],
        "parent_data": parent_record["data_identity"],
        "parent_order": parent_record["order_identity"],
        "parent_model": parent_record["model_identity"],
        "parent_optimizer": parent_record["optimizer_identity"],
        "parent_sampler": parent_record["sampler_identity"],
        "parent_rng": parent_record["rng_identity"],
    }


def create_fork(
    parent_checkpoint,
    output,
    *,
    run_id: str,
    code_identity: str,
    continuation_study_id: str | None = None,
    loss_mode: str = "full_sequence",
    answer_masks=None,
    expected_parent_sha256: str | None = None,
    additional_epochs: int = CONTINUATION_EPOCHS,
    learning_rate: float = CONTINUATION_LEARNING_RATE,
    device="cpu",
    load_device=None,
    validate_completed_78: bool = False,
) -> dict:
    """Validate and explicitly fork a final parent without writing to its run directory."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a nonempty string")
    if continuation_study_id is not None \
            and (not isinstance(continuation_study_id, str) or not continuation_study_id.strip()):
        raise ValueError("continuation_study_id must be a nonempty string or None")
    if loss_mode not in LOSS_MODES:
        raise ValueError(f"loss_mode must be one of {LOSS_MODES}")
    if learning_rate != CONTINUATION_LEARNING_RATE:
        raise ValueError(f"continuation learning rate must be {CONTINUATION_LEARNING_RATE}")
    if (loss_mode == "answer_eos") != (answer_masks is not None):
        raise ValueError("answer_eos requires answer_masks and full_sequence forbids them")
    parent_path = Path(parent_checkpoint).resolve()
    output = Path(output).resolve()
    if parent_path.parent.name == "checkpoints":
        parent_run = parent_path.parent.parent
        if output == parent_run or parent_run in output.parents:
            raise ValueError("a fork must be created outside the immutable parent run")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"fork output must be a new or empty child directory: {output}")
    parent = C.load_checkpoint(parent_path, expected_sha256=expected_parent_sha256)
    if parent["extra"].get("checkpoint_kind") != "final":
        raise C.CheckpointValidationError("continuation parent must be a final checkpoint")
    if parent["optimizer"] is None:
        raise C.CheckpointValidationError("continuation parent has no optimizer state")
    parent_config = parent["config"]
    if parent_config.get("format") != "native-train-v1":
        raise C.CheckpointValidationError("unsupported parent training configuration")

    from . import train as T

    training = parent_config["training"]
    model_config = NativeConfig.from_dict(parent_config["model"])
    data_root = Path(parent_config["data"])
    train_dataset = PackedDataset(data_root, "train")
    sampler = T.make_sampler(train_dataset, training["batch_size"], training["data_seed"])
    target_device = torch.device(device)
    restore_device = torch.device(load_device or target_device)
    T._seed_model(training["model_seed"])
    model = T.NativeModel(model_config)
    model.to(restore_device)
    optimizer = T.build_optimizer(
        model,
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        betas=tuple(training["betas"]),
        eps=training["eps"],
    )
    C.restore_checkpoint(
        parent,
        model,
        optimizer,
        None,
        sampler,
        restore_rng=True,
        strict_cuda=restore_device.type == "cuda",
        expected_config=parent_config,
        expected_code_identity=parent["identities"]["code"],
        expected_data_identity=parent["identities"]["data"],
        expected_order_identity=parent["identities"]["order"],
    )
    plan = continuation_plan(parent, sampler, additional_epochs=additional_epochs)
    completed_78_parent = all(plan[name] == PROGRAM_GENERALIZATION_CONTRACT[name] for name in (
        "parent_epochs", "parent_updates", "parent_exposure_tokens", "parent_cursor",
        "updates_per_epoch", "exposure_tokens_per_epoch", "sequences_per_epoch", "additional_epochs",
    ))
    if validate_completed_78 or completed_78_parent:
        validate_program_generalization_contract(plan)
    scheduler = build_constant_scheduler(optimizer, plan["additional_updates"], learning_rate=learning_rate)
    model.train()

    parent_record = _parent_record(parent)
    masks_path = str(Path(answer_masks).resolve()) if answer_masks is not None else None
    masks_identity = _path_identity(answer_masks) if answer_masks is not None else None
    schedule = {
        "learning_rate": learning_rate,
        "warmup_fraction": 0.0,
        "scheduler": CONTINUATION_SCHEDULER if additional_epochs == CONTINUATION_EPOCHS else
                     f"constant learning rate {learning_rate} for {additional_epochs} complete additional epochs; "
                     "no warmup; parent cosine scheduler state is not inherited",
        "inherits_parent_scheduler": False,
    }
    fork = {
        "format": CONTINUATION_FORMAT,
        "run_id": run_id,
        "continuation_study_id": continuation_study_id,
        "loss_mode": loss_mode,
        "answer_masks": masks_path,
        "answer_masks_identity": masks_identity,
        "parent": parent_record,
        "schedule": schedule,
        **plan,
    }
    run_config = copy.deepcopy(parent_config)
    run_config["run_id"] = run_id
    run_config["parent_run_id"] = parent_record["run_id"]
    run_config["acquisition_run_id"] = continuation_study_id or run_id
    run_config["reservation_id"] = None
    run_config["code_identity"] = code_identity
    run_config["fork"] = fork
    child_training = run_config["training"]
    child_training.update({
        "max_loss_tokens": plan["target_exposure_tokens"],
        "learning_rate": learning_rate,
        "warmup_fraction": 0.0,
        "final_lr_fraction": 1.0,
        "scheduler": schedule["scheduler"],
        "token_rule": "ordered full-label labels[:,1:] != -100 exposure; answer masks never change stopping",
        "main_loss_rule": "mean selected-token cross entropy per batch",
        "loss_mode": loss_mode,
        "answer_masks": masks_path,
        "answer_masks_identity": masks_identity,
        "planned_updates": plan["target_updates"],
        "planned_consumed_loss_tokens": plan["target_exposure_tokens"],
        "continuation_updates": plan["additional_updates"],
        "continuation_exposure_tokens": plan["additional_exposure_tokens"],
        "device": str(target_device),
    })

    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    C.atomic_write(config_path, _json_bytes(run_config))
    identities = {
        "dev_data": parent["identities"].get("dev_data"),
        "data_manifest": parent["identities"].get("data_manifest"),
        **fork_checkpoint_identities(parent_record, run_id, loss_mode),
    }
    descriptor = C.save_checkpoint(
        output / "checkpoints" / f"fork-update-{parent['step']:08d}.pt",
        model,
        optimizer,
        scheduler,
        sampler=sampler,
        config=run_config,
        code_identity=code_identity,
        data_identity=sampler.data_identity,
        order_identity=sampler.order_identity,
        identities=identities,
        rng_state=parent["rng"],
        extra={
            "checkpoint_kind": "fork",
            "mode": model_config.mode,
            "model_seed": training["model_seed"],
            "data_seed": training["data_seed"],
            "planned_updates": plan["target_updates"],
            "planned_consumed_loss_tokens": plan["target_exposure_tokens"],
            "reservation_id": None,
            "fork": fork,
            "parent": parent_record,
        },
    )
    journal = C.CheckpointJournal(output / T.JOURNAL_NAME)
    journal.append_checkpoint(descriptor)
    return {
        "format": CONTINUATION_FORMAT,
        "run_id": run_id,
        "loss_mode": loss_mode,
        "config_path": str(config_path),
        "initial_checkpoint": descriptor,
        "journal": str(journal.path),
        "parent": parent_record,
        "schedule": schedule,
        "continuation": plan,
    }


fork_checkpoint = create_fork


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="python -m native.continuation",
        description=__doc__.split("\n\n")[0],
    )
    result.add_argument("--parent-checkpoint", required=True)
    result.add_argument("--expected-parent-sha256", default=None)
    result.add_argument("--output", required=True, help="new child run directory, never the parent run")
    result.add_argument("--run-id", required=True)
    result.add_argument("--code-identity", default=None, help="defaults to the current native source identity")
    result.add_argument("--loss-mode", choices=LOSS_MODES, default="full_sequence")
    result.add_argument("--answer-masks", default=None)
    result.add_argument("--device", default="cpu")
    result.add_argument("--load-device", choices=("cpu", "cuda"), default=None,
                        help="device for restoring the parent, independent of the declared training device")
    result.add_argument("--validate-completed-78", action="store_true")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    code_identity = args.code_identity
    if code_identity is None:
        from .train import _code_identity

        code_identity = _code_identity()
    result = create_fork(
        args.parent_checkpoint,
        args.output,
        run_id=args.run_id,
        code_identity=code_identity,
        loss_mode=args.loss_mode,
        answer_masks=args.answer_masks,
        expected_parent_sha256=args.expected_parent_sha256,
        device=args.device,
        load_device=args.load_device,
        validate_completed_78=args.validate_completed_78,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
