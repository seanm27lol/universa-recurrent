"""Mocked, CPU-only checks for guarded six-child continuation orchestration."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
from native import continuation as C  # noqa: E402
from native import prepare_continuation_protocol as P  # noqa: E402
from native import run_continuation as R  # noqa: E402
from native.independent import rescore_generations  # noqa: E402
from native.sampler import identity_hash  # noqa: E402


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def synthetic_protocol(tmp_path: Path, *, projected_hours=1.0):
    output = tmp_path / "continuation-output"
    data = tmp_path / "immutable-data"
    masks = data / "answer_masks"
    parent_run = tmp_path / "immutable-parent"
    parents = {}
    children = []
    for mode in P.MODES:
        checkpoint = parent_run / "arms" / mode / "train" / "checkpoints" / "final-update-00003392.pt"
        parents[mode] = {
            "mode": mode,
            "checkpoint_path": str(checkpoint),
            "expected_parent_sha256": hashlib.sha256(mode.encode()).hexdigest(),
            "model_identity": f"model-{mode}",
            "optimizer_identity": f"optimizer-{mode}",
            "sampler_identity": "shared-sampler",
            "parent_scheduler_identity": f"scheduler-{mode}",
            "data_identity": "shared-data",
            "order_identity": "shared-order",
            "inheritance": {"rng": "exact parent RNG"},
        }
        for loss_mode in P.LOSS_MODES:
            run_id = P.child_run_id(mode, loss_mode)
            child_output = output / run_id
            children.append({
                "child_id": f"{mode}-{P.child_arm(loss_mode)}",
                "pair_id": mode, "arm": P.child_arm(loss_mode),
                "mode": mode, "run_id": run_id, "loss_mode": loss_mode,
                "parent_mode": mode, "parent_checkpoint": str(checkpoint),
                "expected_parent_sha256": parents[mode]["expected_parent_sha256"],
                "output": str(child_output),
                "config_path": str(child_output / "run_config.json"),
                "initial_checkpoint": str(P.initial_checkpoint_path(output, run_id)),
                "final_checkpoint": str(P.final_checkpoint_path(output, run_id)),
                "answer_masks": str(masks) if loss_mode == "answer_eos" else None,
                "answer_masks_identity": "mask-directory-identity" if loss_mode == "answer_eos" else None,
                "schedule": {
                    "learning_rate": 3e-5, "warmup_fraction": 0.0, "final_lr_fraction": 1.0,
                    "inherits_parent_scheduler": False, "scheduler": C.CONTINUATION_SCHEDULER,
                    "additional_epochs": 48, "additional_updates": 10176,
                },
                "additional_exposure_tokens": 16711920,
                "additional_main_loss_tokens": P.ADDITIONAL_MAIN_LOSS[loss_mode],
                "target": dict(P.TARGET_ENDPOINT),
                "projected_seconds": {"total": 1.0},
            })
    return {
        "format": P.FORMAT, "status": P.STATUS, "issue": 78,
        "continuation_study_id": P.CONTINUATION_STUDY_ID,
        "allocation_request_hours": 3.0,
        "projected_total_hours": projected_hours,
        "projection": {
            "projected_total_hours": projected_hours,
            "answer_diagnostics_seconds_per_child": 203.0 / 3.0,
            "per_parent_measured_seconds": {"synthetic": True},
        },
        "output_root": str(output), "run_ids": [child["run_id"] for child in children],
        "parents": parents, "children": children,
        "parent_run": {
            "path": str(parent_run), "run_id": "synthetic-78",
            "launch_sha256": "a" * 64, "run_status_sha256": "b" * 64,
            "diagnostic_manifest_sha256": "c" * 64, "immutable": True,
        },
        "objective": {
            "auxiliary_coefficients": {"lambda_balance": 0.01, "lambda_route": 0.1, "lambda_state": 0.1},
            "prompt_inputs_identical_within_pair": True,
            "auxiliary_targets_identical_within_pair": True,
        },
        "auxiliary_coefficients": {"lambda_balance": 0.01, "lambda_route": 0.1, "lambda_state": 0.1},
        "data": {
            "root": str(data), "manifest_sha256": "d" * 64,
            "tokenizer_path": str(tmp_path / "tokenizer.json"), "tokenizer_sha256": "e" * 64,
            "replay": str(tmp_path / "replay.json"), "replay_sha256": "f" * 64,
            "order_identity": "shared-order", "immutable": True,
        },
        "masks": {
            "root": str(masks), "identity": "mask-directory-identity",
            "manifest_path": str(masks / "manifest.json"), "manifest_sha256": "0" * 64,
            "arrays": ["answer_mask", "answer_value_mask", "formatting_eos_mask"],
        },
        "endpoint": {
            "parent": dict(P.PARENT_ENDPOINT), "additional": dict(P.ADDITIONAL_ENDPOINT),
            "target": dict(P.TARGET_ENDPOINT),
            "stopping_rule": "ordered full-label exposure", "paired_matching": "ordered input exposure",
        },
        "exposure_accounting": {
            "per_epoch": dict(P.PER_EPOCH_ACCOUNTING),
            "additional_main_loss_tokens": dict(P.ADDITIONAL_MAIN_LOSS),
            "additional_answer_eos_tokens": 3342336,
            "additional_answer_value_tokens": 786432,
            "additional_formatting_eos_tokens": 2555904,
            "full_sequence_additional_exposure_tokens": 16711920,
            "answer_eos_additional_exposure_tokens": 16711920,
        },
        "constant_lr_policy": {
            "learning_rate": 3e-5, "warmup": False,
            "inherits_parent_cosine_scheduler": False, "additional_epochs": 48,
        },
        "evaluation_policy": {
            "checkpoint": "only final-update-00013568.pt", "intermediate_checkpoint_selection": False,
            "official": "ordinary full evaluation", "independent_rescore": "explicit child paths",
            "reconstruction": "CPU FP32", "answer_diagnostics": "explicit child mappings",
        },
        "intervention_policy": {"examples_per_split": 32, "intermediate_outcome_selection": False},
        "authorization": {"status": P.STATUS, "owner_approved": False},
        "execution_policy": {"reservations": 1},
        "source_files_sha256": {"run_continuation.py": "1" * 64},
    }


def protocol_file(tmp_path: Path, protocol) -> Path:
    path = tmp_path / "input" / "continuation_protocol.json"
    encoded = (json.dumps(protocol, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    path.with_suffix(".sha256").write_text(hashlib.sha256(encoded).hexdigest() + "\n")
    return path


def bypass_identity(monkeypatch):
    monkeypatch.setattr(R, "_validate_frozen_identity",
                        lambda protocol, raw, path, repository, output: hashlib.sha256(raw).hexdigest())


def mock_ledger_paths(monkeypatch, tmp_path, usage):
    monkeypatch.setattr(R.ledger, "validate_shared_reservation_paths", lambda: (
        tmp_path / "live-ledger.jsonl", tmp_path / "live-gpu.lock"
    ))
    monkeypatch.setattr(R.ledger, "usage", lambda path: usage)


def test_default_dry_run_reads_protocol_and_live_ledger_without_artifacts(tmp_path, monkeypatch, capsys):
    protocol = synthetic_protocol(tmp_path)
    path = protocol_file(tmp_path, protocol)
    bypass_identity(monkeypatch)
    calls = []
    monkeypatch.setattr(R.ledger, "validate_shared_reservation_paths", lambda: (
        tmp_path / "live-ledger.jsonl", tmp_path / "live-gpu.lock"
    ))
    monkeypatch.setattr(R.ledger, "usage", lambda ledger: calls.append(ledger) or {
        "used_hours": 28.0, "native_remaining_hours": 7.7003634793909335,
    })
    monkeypatch.setattr(R.ledger, "Reservation", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("dry-run constructed a reservation")
    ))
    monkeypatch.setattr(R.continuation, "create_fork", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("dry-run created a fork")
    ))
    monkeypatch.setattr(R.train, "run_training", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("dry-run launched training")
    ))
    assert R.main(["--protocol", str(path), "--repository", str(REPO)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "dry_run"
    assert result["artifacts_created"] is False and result["reservation_created"] is False
    assert result["optimizer_updates_performed"] == 0 and result["training_launched"] is False
    assert result["projection"]["target_updates_per_child"] == 13568
    assert result["projection"]["target_exposure_tokens_per_child"] == 22282560
    assert result["projection"]["target_cursor_per_child"] == 108160
    assert len(calls) == 1
    assert not Path(protocol["output_root"]).exists()
    assert not (tmp_path / "live-ledger.jsonl").exists()
    assert not (tmp_path / "live-gpu.lock").exists()


@pytest.mark.parametrize(("execute", "prepare_only", "hours", "match"), [
    (True, False, None, "requires both"),
    (True, False, 2.999, "exactly equal"),
    (True, False, 3.001, "exactly equal"),
    (False, False, 3.0, "requires --execute"),
    (False, True, 3.0, "takes no --authorize-hours"),
    (True, True, 3.0, "mutually exclusive"),
])
def test_authorization_gates_precede_identity_ledger_and_artifacts(
        tmp_path, monkeypatch, execute, prepare_only, hours, match):
    protocol = synthetic_protocol(tmp_path)
    path = protocol_file(tmp_path, protocol)
    monkeypatch.setattr(R, "_validate_frozen_identity", lambda *args: (_ for _ in ()).throw(
        AssertionError("authorization refusal reached identity validation")
    ))
    monkeypatch.setattr(R.ledger, "validate_shared_reservation_paths", lambda: (_ for _ in ()).throw(
        AssertionError("authorization refusal reached the ledger")
    ))
    with pytest.raises(RuntimeError, match=match):
        R.run(path, REPO, execute=execute, prepare_only=prepare_only, authorize_hours=hours)
    assert not Path(protocol["output_root"]).exists()


def test_projection_enforces_both_numeric_limits_and_live_allocation(tmp_path):
    protocol = synthetic_protocol(tmp_path, projected_hours=1.9)
    plan = R.projection(protocol, {"native_remaining_hours": 7.7003634793909335})
    assert plan["maximum_watchdog_hours"] == pytest.approx(2.85)
    assert plan["required_native_hours"] == pytest.approx(2.9)
    assert plan["fits_budget"]
    R.require_budget(plan)
    too_slow = R.projection(
        synthetic_protocol(tmp_path, projected_hours=2.0), {"native_remaining_hours": 8.0}
    )
    assert not too_slow["fits_watchdog_and_shutdown"]
    with pytest.raises(RuntimeError, match="1.5"):
        R.require_budget(too_slow)
    too_large = R.projection(
        synthetic_protocol(tmp_path, projected_hours=2.41), {"native_remaining_hours": 8.0}
    )
    assert not too_large["fits_80_percent_planning_margin"]
    short_live = R.projection(protocol, {"native_remaining_hours": 2.95})
    assert not short_live["fits_live_native_ledger"]
    with pytest.raises(RuntimeError, match="live native ledger"):
        R.require_budget(short_live)


def test_execute_requires_clean_committed_source_before_launch_or_reservation(tmp_path, monkeypatch):
    protocol = synthetic_protocol(tmp_path)
    path = protocol_file(tmp_path, protocol)
    bypass_identity(monkeypatch)
    mock_ledger_paths(monkeypatch, tmp_path, {"used_hours": 28.0, "native_remaining_hours": 8.0})
    monkeypatch.setattr(R, "_validate_runtime", lambda repository, python: None)
    monkeypatch.setattr(R, "_source_state", lambda repository, require_clean: (_ for _ in ()).throw(
        RuntimeError("clean, committed source required")
    ))
    monkeypatch.setattr(R.ledger, "Reservation", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("dirty source opened a reservation")
    ))
    with pytest.raises(RuntimeError, match="clean, committed"):
        R.run(path, REPO, execute=True, authorize_hours=3.0)
    assert not (Path(protocol["output_root"]) / "launch.json").exists()


def test_execute_requires_completed_validated_fork_manifest(tmp_path, monkeypatch):
    protocol = synthetic_protocol(tmp_path)
    path = protocol_file(tmp_path, protocol)
    bypass_identity(monkeypatch)
    mock_ledger_paths(monkeypatch, tmp_path, {"used_hours": 28.0, "native_remaining_hours": 8.0})
    monkeypatch.setattr(R, "_validate_runtime", lambda repository, python: None)
    monkeypatch.setattr(R, "_source_state", lambda repository, require_clean: ("a" * 40, "code"))
    monkeypatch.setattr(R.ledger, "Reservation", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("missing forks opened a reservation")
    ))
    with pytest.raises(FileNotFoundError):
        R.run(path, REPO, execute=True, authorize_hours=3.0)
    assert not (Path(protocol["output_root"]) / "launch.json").exists()


def fake_fork_factory(calls):
    def create(parent, output, **kwargs):
        calls.append((parent, output, kwargs))
        output = Path(output)
        initial = P.initial_checkpoint_path(output.parent, kwargs["run_id"])
        initial.parent.mkdir(parents=True, exist_ok=True)
        initial.write_bytes(f"fork {kwargs['run_id']}\n".encode())
        (output / "run_config.json").write_text("{}\n")
        (output / "checkpoint_journal.jsonl").write_text("synthetic journal\n")
        descriptor = {
            "path": str(initial), "sha256": hashlib.sha256(initial.read_bytes()).hexdigest(),
            "step": 3392, "cursor": 27040, "consumed_loss_tokens": 5570640,
        }
        return {
            "initial_checkpoint": descriptor, "config_path": str(output / "run_config.json"),
            "journal": str(output / "checkpoint_journal.jsonl"),
            "parent": {"checkpoint_sha256": kwargs["expected_parent_sha256"]},
            "schedule": {
                "learning_rate": 3e-5, "warmup_fraction": 0.0,
                "scheduler": C.CONTINUATION_SCHEDULER, "inherits_parent_scheduler": False,
            },
            "continuation": {**P.PARENT_ENDPOINT, **P.TARGET_ENDPOINT},
        }
    return create


def test_prepare_only_creates_cpu_loaded_forks_without_optimizer_or_reservation(
        tmp_path, monkeypatch, capsys):
    protocol = synthetic_protocol(tmp_path)
    path = protocol_file(tmp_path, protocol)
    bypass_identity(monkeypatch)
    mock_ledger_paths(monkeypatch, tmp_path, {"used_hours": 28.0, "native_remaining_hours": 8.0})
    calls = []
    monkeypatch.setattr(R.continuation, "create_fork", fake_fork_factory(calls))
    monkeypatch.setattr(R, "_validate_fork_payload", lambda child, path, sha: {
        "model_identity": "inherited-model", "optimizer_state_identity": "inherited-optimizer",
        "sampler_identity": "inherited-sampler", "order_identity": "shared-order",
        "rng_identity": f"rng-{child['run_id']}", "scheduler_inherited": False,
    })
    monkeypatch.setattr(R.train, "run_training", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("prepare-only launched training")
    ))
    monkeypatch.setattr(R.ledger, "Reservation", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("prepare-only opened a reservation")
    ))
    monkeypatch.setattr(torch.optim.Optimizer, "step", lambda self: (_ for _ in ()).throw(
        AssertionError("prepare-only performed an optimizer update")
    ))
    assert R.run(path, REPO, prepare_only=True) == 0
    assert len(calls) == 6
    for (parent, output, kwargs), child in zip(calls, protocol["children"]):
        assert parent == child["parent_checkpoint"]
        assert Path(output) == Path(child["output"])
        assert kwargs["device"] == "cuda" and kwargs["load_device"] == "cpu"
        assert kwargs["expected_parent_sha256"] == child["expected_parent_sha256"]
        assert kwargs["run_id"] == child["run_id"]
        assert kwargs["continuation_study_id"] == P.CONTINUATION_STUDY_ID
        assert kwargs["loss_mode"] == child["loss_mode"]
        assert kwargs["additional_epochs"] == 48 and kwargs["learning_rate"] == 3e-5
        assert kwargs["validate_completed_78"] is True
    manifest = json.loads((Path(protocol["output_root"]) / "fork_manifest.json").read_text())
    assert manifest["complete"] and manifest["status"] == "completed_validated"
    assert manifest["optimizer_updates_performed"] == 0 and manifest["reservation_created"] is False
    assert set(manifest["children"]) == set(protocol["run_ids"])
    assert all(values["status"] == "prepared" for values in manifest["children"].values())
    assert not (Path(protocol["output_root"]) / "launch.json").exists()
    assert not (Path(protocol["output_root"]) / "run_status.json").exists()
    result = json.loads(capsys.readouterr().out)
    assert result["training_launched"] is False and result["reservation_created"] is False


def test_prepare_failure_preserves_partial_manifest_and_never_trains(tmp_path, monkeypatch):
    protocol = synthetic_protocol(tmp_path)
    calls = []

    def create(parent, output, **kwargs):
        if len(calls) == 2:
            calls.append((parent, output, kwargs))
            raise RuntimeError("synthetic fork failure")
        return fake_fork_factory(calls)(parent, output, **kwargs)

    monkeypatch.setattr(R.continuation, "create_fork", create)
    monkeypatch.setattr(R, "_validate_fork_payload", lambda *args: {"validated": True})
    monkeypatch.setattr(R.train, "run_training", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("failed preparation launched training")
    ))
    with pytest.raises(RuntimeError, match="synthetic fork failure"):
        R.prepare_forks(protocol, "9" * 64, code_identity="code")
    manifest = json.loads((Path(protocol["output_root"]) / "fork_manifest.json").read_text())
    statuses = [manifest["children"][run_id]["status"] for run_id in protocol["run_ids"]]
    assert statuses == ["prepared", "prepared", "failed", "not_run", "not_run", "not_run"]
    assert manifest["status"] == "failed_partial_preserved" and not manifest["complete"]
    assert "synthetic fork failure" in manifest["children"][protocol["run_ids"][2]]["error"]


def fork_payloads(child):
    parent = {
        "model": {"weight": [1.0, 2.0]}, "sampler": {"cursor": 27040, "order": [3, 1, 2]},
        "rng": {"python": [1, 2], "torch": [3, 4]},
        "optimizer": {
            "state": {"weight": {"exp_avg": [0.1, 0.2]}},
            "param_groups": [{"params": [0], "lr": 3e-4, "initial_lr": 3e-4, "weight_decay": 0.01}],
        },
        "state_hashes": {
            "model": child["mode"] + "-model", "optimizer": child["mode"] + "-optimizer",
            "sampler": "shared-sampler", "scheduler": child["mode"] + "-old-scheduler",
        },
        "identities": {"data": "shared-data", "order": "shared-order"},
    }
    initial = {
        "extra": {"checkpoint_kind": "fork"}, "step": 3392, "cursor": 27040,
        "consumed_loss_tokens": 5570640,
        "model": copy.deepcopy(parent["model"]), "sampler": copy.deepcopy(parent["sampler"]),
        "rng": copy.deepcopy(parent["rng"]),
        "optimizer": copy.deepcopy(parent["optimizer"]),
        "state_hashes": {
            "model": parent["state_hashes"]["model"], "optimizer": "new-optimizer-hash",
            "sampler": parent["state_hashes"]["sampler"], "scheduler": "new-constant-scheduler",
        },
        "identities": {
            "parent_checkpoint_sha256": child["expected_parent_sha256"],
            "parent_model": parent["state_hashes"]["model"],
            "parent_optimizer": parent["state_hashes"]["optimizer"],
            "parent_sampler": parent["state_hashes"]["sampler"],
            "parent_rng": identity_hash(parent["rng"]),
            "data": "shared-data", "order": "shared-order",
        },
        "config": {
            "run_id": child["run_id"], "model": {"mode": child["mode"]},
            "training": {
                "learning_rate": 3e-5, "warmup_fraction": 0.0, "final_lr_fraction": 1.0,
                "max_loss_tokens": 22282560,
            },
            "fork": {
                "run_id": child["run_id"], "loss_mode": child["loss_mode"],
                "continuation_study_id": P.CONTINUATION_STUDY_ID,
                "parent": {"checkpoint_sha256": child["expected_parent_sha256"]},
                "answer_masks_identity": child["answer_masks_identity"],
                "schedule": {
                    "learning_rate": 3e-5, "warmup_fraction": 0.0,
                    "scheduler": C.CONTINUATION_SCHEDULER, "inherits_parent_scheduler": False,
                },
                "additional_epochs": 48, "additional_updates": 10176,
                "target_updates": 13568, "target_exposure_tokens": 22282560, "target_cursor": 108160,
            },
        },
    }
    initial["optimizer"]["param_groups"][0]["lr"] = 3e-5
    return initial, parent


def test_pair_fork_payloads_preserve_model_optimizer_sampler_rng_but_replace_schedule(tmp_path, monkeypatch):
    protocol = synthetic_protocol(tmp_path)
    for child in protocol["children"][:2]:
        initial, parent = fork_payloads(child)
        monkeypatch.setattr(R.C, "load_checkpoint", lambda path, **kwargs: initial if "fork-update" in str(path) else parent)
        inheritance = R._validate_fork_payload(child, Path(child["initial_checkpoint"]), "f" * 64)
        assert inheritance["model_identity"] == parent["state_hashes"]["model"]
        assert inheritance["optimizer_state_identity"] == identity_hash(initial["optimizer"]["state"])
        assert inheritance["sampler_identity"] == "shared-sampler"
        assert inheritance["order_identity"] == "shared-order"
        assert inheritance["rng_identity"] == identity_hash(parent["rng"])
        assert inheritance["scheduler_inherited"] is False
        assert inheritance["child_scheduler_identity"] != inheritance["parent_scheduler_identity"]
        initial["optimizer"]["state"]["weight"]["exp_avg"][0] = 99.0
        with pytest.raises(R.C.CheckpointIdentityMismatch, match="optimizer"):
            R._validate_fork_payload(child, Path(child["initial_checkpoint"]), "f" * 64)


def training_payload(child, descriptor):
    path = Path(descriptor["path"])
    descriptor = {
        **descriptor, "bytes": path.stat().st_size, "step": 13568,
        "cursor": 108160, "consumed_loss_tokens": 22282560,
    }
    return {
        "format": "native-train-v1", "mode": child["mode"], "loss_mode": child["loss_mode"],
        "step": 13568, "cursor": 108160, "consumed_loss_tokens": 22282560,
        "consumed_exposure_tokens": 22282560, "planned_updates": 13568, "new_updates": 10176,
        "replayed_updates": 0, "processed_updates_this_invocation": 10176,
        "processed_exposure_tokens_this_invocation": 16711920,
        "processed_main_loss_tokens_this_invocation": child["additional_main_loss_tokens"],
        "training_seconds": 10.0, "checkpoint_seconds": 2.0, "final_checkpoint": descriptor,
    }


def write_final_journal(protocol, child, descriptor, *, learning_log=False):
    output = Path(child["output"])
    output.mkdir(parents=True, exist_ok=True)
    records = [{
        "journal_format": "native-recovery-journal-v1", "event": "checkpoint",
        "path": str(Path(child["initial_checkpoint"]).resolve()), "sha256": "initial-sha", "bytes": 1,
        "step": 3392, "cursor": 27040, "consumed_loss_tokens": 5570640,
        "data_identity": "shared-data", "order_identity": "shared-order",
    }]
    logs = []
    exposure_base, exposure_extra = divmod(16711920, 10176)
    cursor_base, cursor_extra = divmod(81120, 10176)
    main_base, main_extra = divmod(child["additional_main_loss_tokens"], 10176)
    exposure = 5570640
    cursor = 27040
    for offset in range(1, 10177):
        step = 3392 + offset
        exposure_delta = exposure_base + (offset <= exposure_extra)
        main_tokens = main_base + (offset <= main_extra)
        exposure += exposure_delta
        cursor += cursor_base + (offset <= cursor_extra)
        batch_identity = f"batch-{step}"
        records.append({
            "journal_format": "native-recovery-journal-v1", "event": "batch",
            "step": step, "batches_consumed": step, "loss_tokens": exposure_delta,
            "cursor": cursor, "consumed_loss_tokens": exposure, "batch_identity": batch_identity,
            "data_identity": "shared-data", "order_identity": "shared-order",
            "sampler": {"batches_consumed": step, "cursor": cursor, "consumed_loss_tokens": exposure},
        })
        if learning_log:
            log = {
                "event": "train", "step": step, "exposure_tokens": exposure_delta,
                "main_loss_tokens": main_tokens, "cursor": cursor,
                "consumed_exposure_tokens": exposure, "batch_identity": batch_identity,
            }
            logs.append(log)
            if offset == 1:
                logs.append({**log, "event": "replay"})
    records.append({
        "journal_format": "native-recovery-journal-v1", "event": "checkpoint",
        "path": str(Path(child["final_checkpoint"]).resolve()), "sha256": descriptor["sha256"],
        "bytes": descriptor["bytes"], "step": 13568, "cursor": 108160,
        "consumed_loss_tokens": 22282560, "data_identity": "shared-data", "order_identity": "shared-order",
    })
    (output / "checkpoint_journal.jsonl").write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    )
    if learning_log:
        (output / "learning_curve.jsonl").write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in logs)
        )


def test_final_checkpoint_cumulative_state_allows_shorter_recovery_invocations(tmp_path, monkeypatch):
    protocol = synthetic_protocol(tmp_path)
    for child in protocol["children"][:2]:
        path = Path(child["final_checkpoint"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"final {child['run_id']}\n".encode())
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        payload = training_payload(child, {"path": str(path), "sha256": sha})
        write_final_journal(protocol, child, payload["final_checkpoint"], learning_log=True)
        final = {
            "extra": {"checkpoint_kind": "final"}, "step": 13568,
            "cursor": 108160, "consumed_loss_tokens": 22282560,
            "config": {"run_id": child["run_id"], "fork": {"run_id": child["run_id"]}},
            "optimizer": {"param_groups": [{"lr": 3e-5}]},
            "identities": {
                "parent_model": protocol["parents"][child["mode"]]["model_identity"],
                "parent_optimizer": protocol["parents"][child["mode"]]["optimizer_identity"],
                "parent_sampler": "shared-sampler",
                "parent_checkpoint_sha256": child["expected_parent_sha256"],
                "parent_rng": f"rng-{child['mode']}",
            },
        }
        monkeypatch.setattr(R.C, "load_checkpoint", lambda *args, **kwargs: final)
        fork_record = {"inheritance": {"rng_identity": f"rng-{child['mode']}"}}
        payload.update({
            "new_updates": 10, "replayed_updates": 20, "processed_updates_this_invocation": 30,
            "processed_exposure_tokens_this_invocation": 12345,
            "processed_main_loss_tokens_this_invocation": 1,
        })
        result = R._training_checkpoint(protocol, child, payload, fork_record)
        assert result["path"] == str(path) and result["step"] == 13568
        accounting = R._training_accounting(protocol, child, payload, fork_record)
        assert accounting["additional_exposure_tokens"] == 16711920
        assert accounting["additional_main_loss_tokens"] == child["additional_main_loss_tokens"]
        assert accounting["additional_exposure_source"] == "complete_durable_training_log"
        wrong = {**payload, "cursor": 108159}
        with pytest.raises(RuntimeError, match="endpoint"):
            R._training_checkpoint(protocol, child, wrong, fork_record)


def test_execute_child_runs_full_evaluation_rescore_fp32_and_answer_diagnostics(tmp_path, monkeypatch):
    protocol = synthetic_protocol(tmp_path)
    child = protocol["children"][1]
    output = Path(child["output"])
    output.mkdir(parents=True)
    checkpoint = Path(child["final_checkpoint"])
    data = b"synthetic final child"
    sha = hashlib.sha256(data).hexdigest()
    descriptor = {"path": str(checkpoint), "sha256": sha}
    calls = {"stages": []}

    def run_training(args):
        assert args.device == "cuda" and args.fork == child["run_id"]
        assert args.loss_mode == "answer_eos" and args.answer_masks == child["answer_masks"]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(data)
        payload = training_payload(child, descriptor)
        descriptor.update(payload["final_checkpoint"])
        write_final_journal(protocol, child, descriptor)
        return payload

    monkeypatch.setattr(R.train, "run_training", run_training)
    monkeypatch.setattr(R, "_training_checkpoint", lambda *args: descriptor)

    def evaluate(**kwargs):
        assert kwargs["checkpoint"] == str(checkpoint) and kwargs["expected_sha256"] == sha
        evaluation = {"format": "native-program-study-evaluation-v1", "checkpoint": descriptor}
        write_json(kwargs["output"], evaluation)
        return evaluation

    monkeypatch.setattr(R.evaluate_program_study, "evaluate", evaluate)
    monkeypatch.setattr(R.program_runner, "_validate_evaluation", lambda evaluation, checkpoint: {"dev": 1})

    def rescore(**kwargs):
        assert kwargs["evaluation_paths"] == {child["mode"]: output / "evaluation.json"}
        result = {
            "arms": {child["mode"]: {}},
            "evaluation_paths": {child["mode"]: str((output / "evaluation.json").resolve())},
        }
        write_json(kwargs["output"], result)
        return result

    monkeypatch.setattr(R.rescore_generations, "rescore", rescore)
    monkeypatch.setattr(R.program_runner, "_validate_generation_diagnostics", lambda *args: None)
    monkeypatch.setattr(R.reconstruct, "load_replay_spec", lambda path: {"replay": str(path)})
    monkeypatch.setattr(R.reconstruct, "reconstruct_checkpoint", lambda *args, **kwargs: {
        "agreement": True, "checkpoint": {"sha256": sha, "step": 13568},
        "execution": {"device": "cpu", "parameter_dtype": "torch.float32", "autocast_dtype": None},
    })

    def write_reconstruction(path, report):
        write_json(path, report)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(R.reconstruct, "write_reconstruction_report", write_reconstruction)

    def diagnose(*args, **kwargs):
        mapping = kwargs["child_mappings"][child["run_id"]]
        assert mapping["mode"] == child["mode"]
        assert Path(mapping["train_result"]) == output / "train-result.json"
        assert Path(mapping["evaluation"]) == output / "evaluation.json"
        answer_path = Path(mapping["output"])
        write_json(answer_path, {"diagnostic": True})
        answer_sha = hashlib.sha256(answer_path.read_bytes()).hexdigest()
        result = {"arms": {child["run_id"]: {
            "teacher_forced_complete": True, "checkpoint": {"expected_sha256": sha},
            "train_result": {"sha256": hashlib.sha256(Path(mapping["train_result"]).read_bytes()).hexdigest()},
            "official_evaluation_source": {
                "sha256": hashlib.sha256(Path(mapping["evaluation"]).read_bytes()).hexdigest()
            },
            "answer_diagnostics": {"path": str(answer_path), "sha256": answer_sha},
        }}}
        write_json(kwargs["manifest_output"], result)
        return result

    monkeypatch.setattr(R.diagnose_program_results, "diagnose", diagnose)
    result = R.execute_child(
        protocol, child, {"inheritance": {"rng_identity": "rng"}},
        on_stage=lambda name, **fields: calls["stages"].append(name),
    )
    assert calls["stages"] == ["evaluating", "rescoring", "reconstructing", "answer_diagnostics"]
    assert result["updates"] == 13568 and result["cursor"] == 108160
    assert result["additional_exposure_tokens"] == 16711920
    assert result["additional_main_loss_tokens"] == 3342336
    assert set(result["measured_stage_seconds"]) == {
        "training_invocation_wall", "official_evaluation", "independent_rescore",
        "fp32_reconstruction", "answer_diagnostics",
    }
    assert (output / "answer_diagnostics.json").is_file()


def completed_child(child):
    return {
        "updates": 13568, "cursor": 108160, "consumed_exposure_tokens": 22282560,
        "additional_exposure_tokens": 16711920,
        "additional_main_loss_tokens": child["additional_main_loss_tokens"],
        "checkpoint": {"path": child["final_checkpoint"], "step": 13568},
    }


def mock_endpoint_stages(monkeypatch, protocol, child):
    calls = []

    def evaluate(**kwargs):
        calls.append("evaluation")
        evaluation = {
            "format": "native-program-study-evaluation-v1",
            "checkpoint": {"sha256": kwargs["expected_sha256"], "step": 13568},
        }
        write_json(kwargs["output"], evaluation)
        return evaluation

    def rescore(**kwargs):
        calls.append("rescore")
        result = {
            "arms": {child["mode"]: {}},
            "evaluation_paths": {
                child["mode"]: str(kwargs["evaluation_paths"][child["mode"]].resolve())
            },
        }
        write_json(kwargs["output"], result)
        return result

    def reconstruct(*args, **kwargs):
        calls.append("reconstruction")
        return {
            "agreement": True,
            "checkpoint": {"sha256": kwargs["expected_sha256"], "step": 13568},
            "execution": {"device": "cpu", "parameter_dtype": "torch.float32", "autocast_dtype": None},
        }

    def diagnose(*args, **kwargs):
        calls.append("answer")
        mapping = kwargs["child_mappings"][child["run_id"]]
        answer_path = Path(mapping["output"])
        write_json(answer_path, {"diagnostic": True})
        train_result_path = Path(mapping["train_result"])
        evaluation_path = Path(mapping["evaluation"])
        result = {"arms": {child["run_id"]: {
            "teacher_forced_complete": True,
            "checkpoint": {"expected_sha256": json.loads(train_result_path.read_text())[
                "final_checkpoint"
            ]["sha256"]},
            "train_result": {"sha256": hashlib.sha256(train_result_path.read_bytes()).hexdigest()},
            "official_evaluation_source": {
                "sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
            },
            "answer_diagnostics": {
                "path": str(answer_path), "sha256": hashlib.sha256(answer_path.read_bytes()).hexdigest()
            },
        }}}
        write_json(kwargs["manifest_output"], result)
        return result

    def validate_evaluation(evaluation, checkpoint):
        if evaluation.get("format") != "native-program-study-evaluation-v1" \
                or evaluation.get("checkpoint", {}).get("sha256") != checkpoint["sha256"]:
            raise RuntimeError("synthetic evaluation belongs to another checkpoint")
        return {"dev": 1}

    monkeypatch.setattr(R.evaluate_program_study, "evaluate", evaluate)
    monkeypatch.setattr(R.program_runner, "_validate_evaluation", validate_evaluation)
    monkeypatch.setattr(R.rescore_generations, "rescore", rescore)
    monkeypatch.setattr(R.program_runner, "_validate_generation_diagnostics", lambda *args: None)
    monkeypatch.setattr(R.reconstruct, "load_replay_spec", lambda path: {"replay": str(path)})
    monkeypatch.setattr(R.reconstruct, "reconstruct_checkpoint", reconstruct)

    def write_reconstruction(path, report):
        write_json(path, report)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(R.reconstruct, "write_reconstruction_report", write_reconstruction)
    monkeypatch.setattr(R.diagnose_program_results, "diagnose", diagnose)
    return calls


def final_payload(protocol, child):
    return {
        "extra": {"checkpoint_kind": "final"}, "step": 13568,
        "cursor": 108160, "consumed_loss_tokens": 22282560,
        "config": {"run_id": child["run_id"], "fork": {"run_id": child["run_id"]}},
        "optimizer": {"param_groups": [{"lr": 3e-5}]},
        "identities": {
            "parent_model": protocol["parents"][child["mode"]]["model_identity"],
            "parent_optimizer": protocol["parents"][child["mode"]]["optimizer_identity"],
            "parent_sampler": "shared-sampler",
            "parent_checkpoint_sha256": child["expected_parent_sha256"],
            "parent_rng": f"rng-{child['mode']}",
        },
    }


def test_interrupted_training_recovery_uses_cumulative_state_not_fresh_budget(tmp_path, monkeypatch):
    protocol = synthetic_protocol(tmp_path)
    child = protocol["children"][0]
    output = Path(child["output"])
    output.mkdir(parents=True)
    (output / "checkpoint_journal.jsonl").write_text("interrupted partial journal")
    invocations = []

    def run_training(args):
        invocations.append(args)
        assert args.resume is True
        checkpoint = Path(child["final_checkpoint"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"recovered final")
        payload = training_payload(child, {
            "path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        })
        payload.update({
            "new_updates": 10, "replayed_updates": 20, "processed_updates_this_invocation": 30,
            "processed_exposure_tokens_this_invocation": 12345,
            "processed_main_loss_tokens_this_invocation": 12345,
        })
        write_final_journal(protocol, child, payload["final_checkpoint"])
        return payload

    monkeypatch.setattr(R.train, "run_training", run_training)
    monkeypatch.setattr(R.C, "load_checkpoint", lambda *args, **kwargs: final_payload(protocol, child))
    calls = mock_endpoint_stages(monkeypatch, protocol, child)
    result = R.execute_child(
        protocol, child, {"inheritance": {"rng_identity": f"rng-{child['mode']}"}},
        resume=True, recovery_index=1,
    )
    assert len(invocations) == 1
    assert result["updates"] == 13568 and result["cursor"] == 108160
    assert result["consumed_exposure_tokens"] == 22282560
    assert result["additional_exposure_tokens"] == 16711920
    assert result["new_updates"] == 10 and result["replayed_updates"] == 20
    assert result["processed_exposure_tokens_this_invocation"] == 12345
    assert (output / "train-result.recovery-0001.json").is_file()
    assert not (output / "train-result.json").exists()
    assert calls == ["evaluation", "rescore", "reconstruction", "answer"]
    journal = (output / "checkpoint_journal.jsonl").read_text().splitlines()
    del journal[5]
    (output / "checkpoint_journal.jsonl").write_text("\n".join(journal) + "\n")
    with pytest.raises(RuntimeError, match="contiguous|counters"):
        R._validate_fork_to_final_journal(protocol, child, result["checkpoint"])


def test_evaluation_only_recovery_never_retrains_or_overwrites_and_reuses_stages(tmp_path, monkeypatch):
    protocol = synthetic_protocol(tmp_path)
    child = protocol["children"][1]
    output = Path(child["output"])
    checkpoint = Path(child["final_checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"completed before evaluation failed")
    payload = training_payload(child, {
        "path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    })
    payload.update({
        "new_updates": 100, "replayed_updates": 25, "processed_updates_this_invocation": 125,
        "processed_exposure_tokens_this_invocation": 5000,
        "processed_main_loss_tokens_this_invocation": 1000,
    })
    write_final_journal(protocol, child, payload["final_checkpoint"])
    write_json(output / "train-result.json", payload)
    write_json(output / "evaluation.json", {"stale": "preserve the failed evaluation attempt"})
    checkpoint_bytes = checkpoint.read_bytes()
    train_result_bytes = (output / "train-result.json").read_bytes()
    failed_evaluation_bytes = (output / "evaluation.json").read_bytes()
    monkeypatch.setattr(R.C, "load_checkpoint", lambda *args, **kwargs: final_payload(protocol, child))
    monkeypatch.setattr(R.train, "run_training", lambda *args: (_ for _ in ()).throw(
        AssertionError("evaluation-only recovery retrained a completed checkpoint")
    ))
    calls = mock_endpoint_stages(monkeypatch, protocol, child)
    result = R.execute_child(
        protocol, child, {"inheritance": {"rng_identity": f"rng-{child['mode']}"}},
        resume=True, recovery_index=1,
    )
    assert result["train_result_path"] == str(output / "train-result.json")
    assert result["updates"] == 13568 and result["cursor"] == 108160
    assert result["additional_main_loss_tokens"] == 3342336
    assert (output / "evaluation.recovery-0001.json").is_file()
    assert not (output / "train-result.recovery-0001.json").exists()
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert (output / "train-result.json").read_bytes() == train_result_bytes
    assert (output / "evaluation.json").read_bytes() == failed_evaluation_bytes
    first_calls = list(calls)
    second = R.execute_child(
        protocol, child, {"inheritance": {"rng_identity": f"rng-{child['mode']}"}},
        resume=True, recovery_index=2,
    )
    assert calls == first_calls, "valid completed stages were executed again"
    assert second["evaluation_path"] == result["evaluation_path"]
    assert not list(output.glob("*.recovery-0002.json"))
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert (output / "train-result.json").read_bytes() == train_result_bytes


def test_ordinary_child_failure_is_preserved_and_remaining_children_continue(tmp_path, monkeypatch):
    protocol = synthetic_protocol(tmp_path)
    for child in protocol["children"]:
        Path(child["output"]).mkdir(parents=True)
    fork_manifest = {"children": {
        child["run_id"]: {"inheritance": {"rng_identity": "rng"}} for child in protocol["children"]
    }}

    class Reservation:
        def __init__(self):
            self.extra = {"native_acquisition_run_id": None}
            self.checks = 0

        def check(self):
            self.checks += 1

    reservation = Reservation()

    def execute(protocol, child, record, on_stage=None):
        assert reservation.extra["native_acquisition_run_id"] == child["run_id"]
        assert os.environ[R.OWNER_ENV] == child["run_id"]
        if child["run_id"] == protocol["run_ids"][2]:
            raise RuntimeError("synthetic child failure")
        return completed_child(child)

    monkeypatch.setattr(R, "execute_child", execute)
    statuses = R._execute(protocol, fork_manifest, reservation=reservation)
    assert reservation.checks == 6
    assert statuses[protocol["run_ids"][2]]["status"] == "failed"
    assert statuses[protocol["run_ids"][2]]["stage"] == "training"
    assert all(statuses[run_id]["status"] == "completed" for run_id in protocol["run_ids"][3:])
    assert json.loads((Path(protocol["children"][2]["output"]) / "status.json").read_text()) == statuses[
        protocol["run_ids"][2]
    ]
    assert os.environ.get(R.OWNER_ENV) is None


def test_interruption_preserves_completed_and_partial_states_and_aborts_remaining(tmp_path, monkeypatch):
    protocol = synthetic_protocol(tmp_path)
    for child in protocol["children"]:
        Path(child["output"]).mkdir(parents=True)
    fork_manifest = {"children": {
        child["run_id"]: {"inheritance": {"rng_identity": "rng"}} for child in protocol["children"]
    }}

    class Stop(BaseException):
        pass

    def execute(protocol, child, record, on_stage=None):
        if child["run_id"] == protocol["run_ids"][1]:
            raise Stop("synthetic watchdog")
        return completed_child(child)

    monkeypatch.setattr(R, "execute_child", execute)
    with pytest.raises(Stop):
        R._execute(protocol, fork_manifest)
    statuses = R._recover_statuses(protocol, {
        run_id: {"status": "not_run"} for run_id in protocol["run_ids"]
    })
    assert statuses[protocol["run_ids"][0]]["status"] == "completed"
    assert statuses[protocol["run_ids"][1]]["status"] == "interrupted"
    assert statuses[protocol["run_ids"][1]]["stage"] == "training"
    assert all(statuses[run_id]["status"] == "not_run" for run_id in protocol["run_ids"][2:])


def test_execute_uses_one_reservation_and_records_inheritance_costs_and_authorization(
        tmp_path, monkeypatch, capsys):
    protocol = synthetic_protocol(tmp_path)
    path = protocol_file(tmp_path, protocol)
    original_protocol = path.read_bytes()
    bypass_identity(monkeypatch)
    before = {"used_hours": 28.0, "native_remaining_hours": 8.0}
    after = {"used_hours": 28.5, "native_remaining_hours": 7.5}
    usage = iter((before, after))
    monkeypatch.setattr(R.ledger, "validate_shared_reservation_paths", lambda: (
        tmp_path / "live-ledger.jsonl", tmp_path / "live-gpu.lock"
    ))
    monkeypatch.setattr(R.ledger, "usage", lambda path: next(usage))
    monkeypatch.setattr(R, "_validate_runtime", lambda repository, python: None)
    monkeypatch.setattr(R, "_source_state", lambda repository, require_clean: ("a" * 40, "code"))
    manifest = {
        "path": str(Path(protocol["output_root"]) / "fork_manifest.json"), "sha256": "f" * 64,
        "children": {
            child["run_id"]: {"inheritance": {
                "model_identity": protocol["parents"][child["mode"]]["model_identity"],
                "optimizer_state_identity": "inherited-optimizer",
                "sampler_identity": "shared-sampler", "order_identity": "shared-order",
                "rng_identity": f"rng-{child['run_id']}", "scheduler_inherited": False,
            }} for child in protocol["children"]
        },
    }
    monkeypatch.setattr(R, "validate_fork_manifest", lambda *args, **kwargs: manifest)
    for child in protocol["children"]:
        write_json(Path(child["config_path"]), {"fork": {"continuation_study_id": P.CONTINUATION_STUDY_ID}})
    owner = {"active": None}
    monkeypatch.setattr(R.ledger, "active", lambda: owner["active"])
    reservations = []

    class FakeReservation:
        def __init__(self, command, **kwargs):
            self.command, self.kwargs = command, kwargs
            self.id = "one-reservation"
            reservations.append(self)

        def __enter__(self):
            owner["active"] = self
            return self

        def __exit__(self, *args):
            owner["active"] = None
            return False

    monkeypatch.setattr(R.ledger, "Reservation", FakeReservation)

    def execute(protocol, fork_manifest, reservation=None):
        assert reservation is owner["active"]
        return {child["run_id"]: {"status": "completed", **completed_child(child)} for child in protocol["children"]}

    monkeypatch.setattr(R, "_execute", execute)
    assert R.run(path, REPO, execute=True, authorize_hours=3.0) == 0
    assert len(reservations) == 1
    reservation = reservations[0]
    assert reservation.kwargs["projected_hours"] == protocol["projected_total_hours"]
    assert reservation.kwargs["extra"]["native_continuation_run_ids"] == protocol["run_ids"]
    assert reservation.kwargs["extra"]["native_continuation_run_id"] == P.CONTINUATION_STUDY_ID
    assert reservation.kwargs["extra"]["native_continuation_authorized_hours"] == 3.0
    launch = json.loads((Path(protocol["output_root"]) / "launch.json").read_text())
    status = json.loads((Path(protocol["output_root"]) / "run_status.json").read_text())
    assert launch["authorized_hours"] == status["authorized_hours"] == 3.0
    assert launch["authorization_status"] == "explicitly_authorized_for_this_execution"
    assert launch["sampler_order_identity"] == status["sampler_order_identity"] == "shared-order"
    assert launch["masks"]["identity"] == status["mask_identity"] == "mask-directory-identity"
    assert launch["endpoint"]["target"] == P.TARGET_ENDPOINT
    assert launch["exposure_accounting"]["additional_main_loss_tokens"]["answer_eos"] == 3342336
    assert launch["measured_costs"] == protocol["projection"]
    assert launch["parent_inheritance"] == status["parent_inheritance"]
    assert status["complete"] and status["reservation_count"] == 1
    assert status["charged_hours"] == pytest.approx(0.5)
    assert status["evaluation_policy"] if "evaluation_policy" in status else status["official_evaluation_policy"]
    assert all(name.endswith("final-update-00013568.pt") for name in status["final_checkpoints"].values())
    assert path.read_bytes() == original_protocol
    assert json.loads(capsys.readouterr().out)["complete"] is True


def previous_execution(tmp_path, monkeypatch, child_statuses):
    protocol = synthetic_protocol(tmp_path, projected_hours=0.8)
    protocol_path = protocol_file(tmp_path, protocol)
    bypass_identity(monkeypatch)
    monkeypatch.setattr(R, "_validate_runtime", lambda repository, python: None)
    monkeypatch.setattr(R, "_source_state", lambda repository, require_clean: ("a" * 40, "code"))
    manifest = {
        "path": str(Path(protocol["output_root"]) / "fork_manifest.json"), "sha256": "f" * 64,
        "children": {
            child["run_id"]: {"inheritance": {"rng_identity": f"rng-{child['run_id']}"}}
            for child in protocol["children"]
        },
    }
    monkeypatch.setattr(R, "validate_fork_manifest", lambda *args, **kwargs: manifest)
    before = {
        "used_hours": 28.0, "native_remaining_hours": 7.7003634793909335, "reservations": [],
    }
    after = {
        "used_hours": 28.4, "native_remaining_hours": 7.3003634793909335,
        "reservations": [{"reservation": "prior-reservation", "hours": 0.4}],
    }
    usage = iter((before, after))
    monkeypatch.setattr(R.ledger, "validate_shared_reservation_paths", lambda: (
        tmp_path / "live-ledger.jsonl", tmp_path / "live-gpu.lock"
    ))
    monkeypatch.setattr(R.ledger, "usage", lambda path: next(usage))
    monkeypatch.setattr(R.ledger, "active", lambda: None)

    class InitialReservation:
        def __init__(self, command, **kwargs):
            self.id = "prior-reservation"
            self.extra = kwargs["extra"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(R.ledger, "Reservation", InitialReservation)
    monkeypatch.setattr(R, "_execute", lambda *args, **kwargs: child_statuses)
    complete = all(item.get("status") == "completed" for item in child_statuses.values())
    assert R.run(protocol_path, REPO, execute=True, authorize_hours=3.0) == (0 if complete else 3)
    return protocol, protocol_path, manifest


def test_recovery_sums_prior_charges_and_preserves_run_and_child_failures(
        tmp_path, monkeypatch, capsys):
    execute_impl = R._execute
    protocol = synthetic_protocol(tmp_path, projected_hours=0.8)
    child_statuses = {}
    for index, child in enumerate(protocol["children"]):
        if index == 0:
            child_statuses[child["run_id"]] = {
                **completed_child(child), "status": "failed", "stage": "evaluating",
                "error": "previous evaluation failure", "traceback": "previous traceback",
            }
        else:
            child_statuses[child["run_id"]] = {"status": "not_run"}
    protocol_path = protocol_file(tmp_path, protocol)
    bypass_identity(monkeypatch)
    monkeypatch.setattr(R, "_validate_runtime", lambda repository, python: None)
    monkeypatch.setattr(R, "_source_state", lambda repository, require_clean: ("a" * 40, "code"))
    manifest = {
        "path": str(Path(protocol["output_root"]) / "fork_manifest.json"), "sha256": "f" * 64,
        "children": {
            child["run_id"]: {"inheritance": {"rng_identity": f"rng-{child['run_id']}"}}
            for child in protocol["children"]
        },
    }
    monkeypatch.setattr(R, "validate_fork_manifest", lambda *args, **kwargs: manifest)
    usage = iter((
        {"used_hours": 28.0, "native_remaining_hours": 7.7003634793909335, "reservations": []},
        {"used_hours": 28.4, "native_remaining_hours": 7.3003634793909335,
         "reservations": [{"reservation": "prior-reservation", "hours": 0.4}]},
    ))
    monkeypatch.setattr(R.ledger, "validate_shared_reservation_paths", lambda: (
        tmp_path / "live-ledger.jsonl", tmp_path / "live-gpu.lock"
    ))
    monkeypatch.setattr(R.ledger, "usage", lambda path: next(usage))
    monkeypatch.setattr(R.ledger, "active", lambda: None)

    class InitialReservation:
        def __init__(self, command, **kwargs):
            self.id = "prior-reservation"
            self.extra = kwargs["extra"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(R.ledger, "Reservation", InitialReservation)
    monkeypatch.setattr(R, "_execute", lambda *args, **kwargs: child_statuses)
    assert R.run(protocol_path, REPO, execute=True, authorize_hours=3.0) == 3
    capsys.readouterr()
    output_root = Path(protocol["output_root"])
    failed = dict(child_statuses[protocol["run_ids"][0]])
    write_json(Path(protocol["children"][0]["output"]) / "status.json", failed)
    root_status_path = output_root / "run_status.json"
    root_status = json.loads(root_status_path.read_text())
    root_status["failure"] = {"kind": "RuntimeError", "error": "previous run-level failure"}
    write_json(root_status_path, root_status)
    launch_bytes = (output_root / "launch.json").read_bytes()
    status_bytes = root_status_path.read_bytes()
    child_status_bytes = (Path(protocol["children"][0]["output"]) / "status.json").read_bytes()
    before = {
        "used_hours": 28.4, "native_remaining_hours": 7.3003634793909335,
        "reservations": [
            {"reservation": "prior-reservation", "hours": 0.4},
            {"reservation": "unrelated", "hours": 4.0},
        ],
    }
    after = {
        "used_hours": 28.55, "native_remaining_hours": 7.1503634793909335,
        "reservations": before["reservations"] + [{"reservation": "recovery-reservation", "hours": 0.15}],
    }
    usage = iter((before, after))
    monkeypatch.setattr(R.ledger, "usage", lambda path: next(usage))
    reservations = []

    class RecoveryReservation:
        def __init__(self, command, **kwargs):
            self.kwargs = kwargs
            self.id = "recovery-reservation"
            self.extra = kwargs["extra"]
            reservations.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def check(self):
            return None

    monkeypatch.setattr(R.ledger, "Reservation", RecoveryReservation)
    monkeypatch.setattr(R, "_execute", execute_impl)
    monkeypatch.setattr(R, "execute_child", lambda *args, **kwargs: completed_child(args[1]))
    assert R.run(protocol_path, REPO, execute=True, authorize_hours=3.0, resume=True) == 0
    assert len(reservations) == 1
    projected = reservations[0].kwargs["projected_hours"]
    assert 0.4 + 1.5 * projected + 0.05 <= 3.0
    assert reservations[0].kwargs["extra"]["native_continuation_run_id"] == P.CONTINUATION_STUDY_ID
    assert reservations[0].kwargs["extra"]["native_continuation_authorized_hours"] == 3.0
    assert (output_root / "launch.json").read_bytes() == launch_bytes
    assert root_status_path.read_bytes() == status_bytes
    assert (Path(protocol["children"][0]["output"]) / "status.json").read_bytes() == child_status_bytes
    recovered = json.loads((output_root / "run_status.recovery-0001.json").read_text())
    assert recovered["complete"] and recovered["reservation_count"] == 2
    assert recovered["prior_continuation_charged_hours"] == pytest.approx(0.4)
    assert recovered["charged_hours"] == pytest.approx(0.15)
    assert recovered["cumulative_continuation_charged_hours"] == pytest.approx(0.55)
    assert recovered["failure_history"][0]["failure"]["error"] == "previous run-level failure"
    child_recovered = json.loads(
        (Path(protocol["children"][0]["output"]) / "status.recovery-0001.json").read_text()
    )
    assert child_recovered["previous_status_history"][0]["error"] == "previous evaluation failure"
    assert child_recovered["failure_history"][0]["error"] == "previous evaluation failure"
    assert child_recovered["status"] == "completed"


def test_completed_resume_returns_already_complete_and_default_reuse_is_refused(
        tmp_path, monkeypatch, capsys):
    protocol = synthetic_protocol(tmp_path, projected_hours=0.8)
    statuses = {
        child["run_id"]: {"status": "completed", **completed_child(child)}
        for child in protocol["children"]
    }
    protocol, protocol_path, _ = previous_execution(tmp_path, monkeypatch, statuses)
    capsys.readouterr()
    output_root = Path(protocol["output_root"])
    launch_bytes = (output_root / "launch.json").read_bytes()
    status_bytes = (output_root / "run_status.json").read_bytes()
    before = {
        "used_hours": 28.4, "native_remaining_hours": 7.3003634793909335,
        "reservations": [{"reservation": "prior-reservation", "hours": 0.4}],
    }
    monkeypatch.setattr(R.ledger, "usage", lambda path: before)
    monkeypatch.setattr(R.ledger, "Reservation", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("completed or refused recovery opened a reservation")
    ))
    with pytest.raises(FileExistsError, match="without --resume"):
        R.run(protocol_path, REPO, execute=True, authorize_hours=3.0)
    monkeypatch.setattr(R, "_existing_training", lambda *args: ({}, {}, "synthetic-train-result"))
    assert R.run(protocol_path, REPO, execute=True, authorize_hours=3.0, resume=True) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "already_complete" and result["reservation_created"] is False
    assert (output_root / "launch.json").read_bytes() == launch_bytes
    assert (output_root / "run_status.json").read_bytes() == status_bytes
    assert not list(output_root.glob("*.recovery-*.json"))


def test_resume_rejects_changed_previous_authorization_without_new_output(tmp_path, monkeypatch):
    protocol = synthetic_protocol(tmp_path, projected_hours=0.8)
    statuses = {child["run_id"]: {"status": "not_run"} for child in protocol["children"]}
    protocol, protocol_path, _ = previous_execution(tmp_path, monkeypatch, statuses)
    output_root = Path(protocol["output_root"])
    launch_path = output_root / "launch.json"
    launch = json.loads(launch_path.read_text())
    launch["authorized_hours"] = 2.0
    write_json(launch_path, launch)
    launch_bytes = launch_path.read_bytes()
    monkeypatch.setattr(R.ledger, "usage", lambda path: {
        "used_hours": 28.4, "native_remaining_hours": 7.3003634793909335,
        "reservations": [{"reservation": "prior-reservation", "hours": 0.4}],
    })
    monkeypatch.setattr(R.ledger, "Reservation", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("identity-mismatched recovery opened a reservation")
    ))
    with pytest.raises(RuntimeError, match="authorized_hours"):
        R.run(protocol_path, REPO, execute=True, authorize_hours=3.0, resume=True)
    assert launch_path.read_bytes() == launch_bytes
    assert not list(output_root.glob("*.recovery-*.json"))


def test_prior_reservations_are_summed_and_unfittable_comparison_is_not_truncated(tmp_path):
    protocol = synthetic_protocol(tmp_path, projected_hours=1.0)
    ledger_path = tmp_path / "ledger.jsonl"
    entries = [
        {"event": "start", "reservation": reservation, "extra": {
            "native_continuation_run_id": P.CONTINUATION_STUDY_ID,
            "native_continuation_authorized_hours": 3.0,
        }}
        for reservation in ("prior-a", "prior-b")
    ]
    ledger_path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    before = {
        "native_remaining_hours": 7.7003634793909335,
        "reservations": [
            {"reservation": "prior-a", "hours": 0.4},
            {"reservation": "prior-b", "hours": 0.2},
            {"reservation": "unrelated", "hours": 9.0},
        ],
    }
    charges = R._prior_continuation_charges(ledger_path, before, P.CONTINUATION_STUDY_ID, [])
    assert charges["prior_continuation_charged_hours"] == pytest.approx(0.6)
    assert charges["prior_continuation_reservation_count"] == 2
    plan = R.recovery_projection(protocol, before, prior_hours=0.6, remaining_projected_hours=0.5)
    assert plan["projected_total_hours"] == 0.5
    assert plan["required_native_hours"] == pytest.approx(1.4)
    R.require_budget(plan)
    statuses = {run_id: {"status": "not_run"} for run_id in protocol["run_ids"]}
    remaining = R._remaining_projected_hours(protocol, statuses)
    assert remaining == pytest.approx(protocol["projected_total_hours"])
    impossible = R.recovery_projection(protocol, before, prior_hours=2.1, remaining_projected_hours=remaining)
    assert not impossible["fits_watchdog_and_shutdown"]
    with pytest.raises(RuntimeError, match="prior charges"):
        R.require_budget(impossible)


def test_independent_rescore_accepts_explicit_path_and_keeps_default_discovery(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    evaluation = {
        "format": "synthetic-evaluation",
        "structured_generation": {"splits": {
            split: {"records": []} for split in rescore_generations.SPLITS
        }},
        "code_generation": {"records": []},
    }
    default = tmp_path / "default-study"
    write_json(default / "arms" / "dense" / "evaluation.json", evaluation)
    default_result = rescore_generations.rescore(
        default, data, tmp_path / "default-rescore.json", modes=("dense",)
    )
    assert "evaluation_paths" not in default_result
    assert set(default_result["arms"]) == {"dense"}
    explicit = tmp_path / "children" / "dense-A" / "evaluation.json"
    write_json(explicit, evaluation)
    result = rescore_generations.rescore(
        tmp_path / "continuation", data, tmp_path / "explicit-rescore.json",
        evaluation_paths={"dense": explicit},
    )
    assert result["evaluation_paths"] == {"dense": str(explicit.resolve())}
    assert result["arms"] == default_result["arms"]
    with pytest.raises(ValueError, match="mutually exclusive"):
        rescore_generations.rescore(
            default, data, tmp_path / "forbidden.json", modes=("dense",),
            evaluation_paths={"dense": explicit},
        )
