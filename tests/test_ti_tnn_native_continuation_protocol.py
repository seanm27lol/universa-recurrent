"""Self-contained six-child protocol tests using only synthetic parent records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
from native import continuation as C  # noqa: E402
from native import prepare_continuation_protocol as P  # noqa: E402
from native import train as T  # noqa: E402
from native.data import IGNORE_INDEX  # noqa: E402


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    parent_run = tmp_path / "immutable-parent"
    data = tmp_path / "immutable-data"
    masks = data / "answer_masks"
    data.mkdir()
    tokenizer = tmp_path / "tokenizer.json"
    replay = tmp_path / "replay.json"
    tokenizer.write_text("synthetic tokenizer\n")
    replay.write_text("synthetic replay\n")
    write_json(data / "manifest.json", {"synthetic": True})
    data_sha = digest(data / "manifest.json")
    launch = {
        "format": "native-program-study-launch-v1",
        "run_id": "synthetic-78",
        "data": str(data),
        "data_manifest_sha256": data_sha,
        "tokenizer": str(tokenizer),
        "tokenizer_sha256": digest(tokenizer),
        "replay": str(replay),
        "replay_sha256": digest(replay),
    }
    write_json(parent_run / "launch.json", launch)
    status = {"complete": True, "failure": None, "arms": {}}
    measured = {
        "dense": (100.0, 4.0, 2.0, 300.0),
        "untyped": (120.0, 5.0, 1.0, 320.0),
        "typed": (110.0, 6.0, 1.5, 330.0),
    }
    for mode, (training, checkpoint, dev, fixed) in measured.items():
        arm = parent_run / "arms" / mode
        checkpoint_path = arm / "train" / "checkpoints" / "final-update-00003392.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(f"synthetic final parent {mode}\n".encode())
        descriptor = {
            "path": str(checkpoint_path),
            "sha256": digest(checkpoint_path),
            "step": C.PARENT_UPDATES,
            "cursor": C.PARENT_CURSOR,
            "consumed_loss_tokens": C.PARENT_EXPOSURE_TOKENS,
            "state_hashes": {
                "model": f"model-{mode}", "optimizer": f"optimizer-{mode}",
                "sampler": "shared-sampler", "scheduler": f"parent-scheduler-{mode}",
            },
            "identities": {"data": "shared-data", "order": "shared-order"},
        }
        result = {
            "format": "native-train-v1", "mode": mode,
            "step": C.PARENT_UPDATES, "cursor": C.PARENT_CURSOR,
            "planned_updates": C.PARENT_UPDATES,
            "consumed_loss_tokens": C.PARENT_EXPOSURE_TOKENS,
            "final_checkpoint": descriptor,
            "training_seconds": training, "checkpoint_seconds": checkpoint,
            "dev": {"forward_seconds": dev * 0.75, "data_loading_seconds": dev * 0.25},
        }
        write_json(arm / "train-result.json", result)
        write_json(arm / "train" / "run_config.json", {
            "training": {
                "dev_every": 500, "lambda_balance": 0.01,
                "lambda_route": 0.1, "lambda_state": 0.1,
            },
        })
        wall = training + checkpoint + 7 * dev + fixed
        write_json(arm / "status.json", {"status": "completed", "wall_seconds": wall})
        for name in ("evaluation.json", "generation_diagnostics.json", "reconstruction.json",
                     "answer_diagnostics.json"):
            write_json(arm / name, {"synthetic": name, "mode": mode})
        status["arms"][mode] = {"checkpoint": descriptor}
    write_json(parent_run / "run_status.json", status)
    write_json(parent_run / "diagnostic_manifest.json", {
        "arms": {mode: {"teacher_forced_complete": True} for mode in P.MODES},
    })
    mask_manifest = {
        "format": "native-program-answer-mask-companions-v1",
        "source_manifest": {"sha256": data_sha},
        "splits": {},
    }
    for split in P.SPLITS:
        arrays = {}
        for name in ("answer_mask", "answer_value_mask", "formatting_eos_mask"):
            path = masks / split / f"{name}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic {split} {name}\n".encode())
            arrays[name] = {"path": str(path), "sha256": digest(path)}
        values = {"arrays": arrays}
        if split == "train":
            values.update(P.PER_EPOCH_ACCOUNTING)
        mask_manifest["splits"][split] = values
    write_json(masks / "manifest.json", mask_manifest)
    return parent_run, data, masks


def build_fixture(tmp_path: Path, *, before=None):
    parent_run, data, masks = parent_fixture(tmp_path)
    output = tmp_path / "continuation-output"
    before = before or {"native_remaining_hours": 7.7003634793909335}
    protocol = P.build_protocol(
        parent_run, output_root=output, repository=REPO, before=before,
        mask_directory=masks,
    )
    return protocol, parent_run, data, output


def test_protocol_pins_six_pairs_parent_inheritance_and_fixed_endpoint(tmp_path):
    protocol, _, _, output = build_fixture(tmp_path)
    assert protocol["status"] == "proposed_not_authorized_or_launched"
    assert protocol["allocation_request_hours"] == 3.0
    assert protocol["continuation_study_id"] == P.CONTINUATION_STUDY_ID
    assert P.child_run_id("dense", "full_sequence") == "78-dense-full_sequence"
    assert protocol["run_ids"][0] == "78-dense-full_sequence"
    assert "versioned recovery" in protocol["execution_policy"]["resume"]
    assert "never truncate a control" in protocol["execution_policy"]["recovery_budget"]
    assert protocol["endpoint"]["parent"] == {
        "epochs": 16, "updates": 3392, "exposure_tokens": 5570640, "cursor": 27040,
    }
    assert protocol["endpoint"]["additional"] == {
        "epochs": 48, "updates": 10176, "exposure_tokens": 16711920,
    }
    assert protocol["endpoint"]["target"] == {
        "epochs": 64, "updates": 13568, "exposure_tokens": 22282560, "cursor": 108160,
    }
    assert len(protocol["run_ids"]) == len(set(protocol["run_ids"])) == 6
    assert not output.exists(), "building an in-memory protocol must not create output artifacts"
    for mode in P.MODES:
        pair = [child for child in protocol["children"] if child["mode"] == mode]
        assert [child["arm"] for child in pair] == ["A", "B"]
        full, answer = pair
        assert full["loss_mode"] == "full_sequence" and answer["loss_mode"] == "answer_eos"
        assert full["parent_checkpoint"] == answer["parent_checkpoint"]
        assert full["expected_parent_sha256"] == answer["expected_parent_sha256"]
        assert len(full["expected_parent_sha256"]) == 64
        assert full["run_id"] != answer["run_id"]
        assert full["output"] != answer["output"]
        assert full["final_checkpoint"] != answer["final_checkpoint"]
        assert full["final_checkpoint"].endswith("final-update-00013568.pt")
        assert answer["final_checkpoint"].endswith("final-update-00013568.pt")
        assert full["answer_masks"] is None and full["answer_masks_identity"] is None
        assert answer["answer_masks_identity"] == protocol["masks"]["identity"]
        assert full["schedule"] == answer["schedule"]
        assert full["schedule"]["learning_rate"] == 3e-5
        assert full["schedule"]["warmup_fraction"] == 0.0
        assert full["schedule"]["inherits_parent_scheduler"] is False
        assert protocol["parents"][mode]["inheritance"]["rng"].startswith("exact Python")
    assert protocol["objective"]["auxiliary_coefficients"] == {
        "lambda_balance": 0.01, "lambda_route": 0.1, "lambda_state": 0.1,
    }
    assert protocol["objective"]["prompt_inputs_identical_within_pair"]
    assert protocol["objective"]["auxiliary_targets_identical_within_pair"]
    assert protocol["data"]["order_identity"] == "shared-order"
    assert protocol["evaluation_policy"]["intermediate_checkpoint_selection"] is False
    assert protocol["intervention_policy"]["examples_per_split"] == 32


def test_exposure_and_main_loss_accounting_never_conflates_answer_masks(tmp_path):
    protocol, _, _, _ = build_fixture(tmp_path)
    accounting = protocol["exposure_accounting"]
    assert accounting["per_epoch"] == {
        "full_label_exposure_tokens": 348165,
        "answer_eos_tokens": 69632,
        "answer_value_tokens": 16384,
        "formatting_eos_tokens": 53248,
    }
    assert accounting["additional_main_loss_tokens"] == {
        "full_sequence": 16711920, "answer_eos": 3342336,
    }
    assert accounting["additional_answer_value_tokens"] == 16384 * 48 == 786432
    assert accounting["additional_formatting_eos_tokens"] == 53248 * 48 == 2555904
    assert accounting["additional_answer_value_tokens"] + accounting[
        "additional_formatting_eos_tokens"
    ] == 3342336
    for child in protocol["children"]:
        assert child["additional_exposure_tokens"] == 16711920
        expected_main = 16711920 if child["arm"] == "A" else 3342336
        assert child["additional_main_loss_tokens"] == expected_main

    labels = torch.tensor([[20, 21, 22, 23, 24]])
    answer_mask = torch.tensor([[False, True, False, True, True]])
    batch = {
        "labels": labels, "answer_mask": answer_mask,
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "route_targets": torch.tensor([[[1], [2], [3], [4], [5]]]),
        "state_targets": torch.tensor([[[5], [4], [3], [2], [1]]]),
    }
    original = {name: value.clone() for name, value in batch.items()}
    selected = T.main_loss_labels(batch, "answer_eos")
    assert selected.tolist() == [[IGNORE_INDEX, 21, IGNORE_INDEX, 23, 24]]
    assert T.main_loss_labels(batch, "full_sequence") is labels
    assert T.exposure_token_count(batch) == 4
    assert T.main_loss_token_count(batch, "answer_eos") == 3
    for name, value in batch.items():
        assert torch.equal(value, original[name])


def test_projection_uses_actual_mode_costs_and_contains_every_required_stage(tmp_path):
    protocol, _, _, _ = build_fixture(tmp_path)
    projection = protocol["projection"]
    dense = projection["per_child_seconds"]["78-dense-full_sequence"]
    assert dense["training"] == pytest.approx(3 * 100.0)
    assert dense["checkpoint"] == pytest.approx(3 * 4.0)
    assert dense["periodic_and_final_dev"] == pytest.approx(22 * 2.0)
    assert dense["official_evaluation_rescore_fp32_replay"] == pytest.approx(300.0)
    assert dense["answer_diagnostics"] == pytest.approx(203.0 / 3.0)
    assert dense["fork_initialization"] == pytest.approx(4.0)
    assert dense["total"] == pytest.approx(sum(dense[key] for key in dense if key != "total"))
    assert projection["per_parent_measured_seconds"]["dense"]["wall_seconds"] == pytest.approx(418.0)
    assert projection["included_fixed_endpoint_components"] == [
        "ordinary_full_official_evaluation", "independent_per_child_rescore", "cpu_fp32_replay",
    ]
    assert projection["answer_diagnostics_seconds_per_child"] >= 203.0 / 3.0
    assert projection["shutdown_allowance_hours"] == 0.05
    assert protocol["projected_total_hours"] == pytest.approx(
        projection["total_seconds_all_six"] / 3600.0
    )
    assert protocol["projected_total_hours"] <= 0.8 * 3.0
    assert projection["required_native_hours"] == pytest.approx(
        protocol["projected_total_hours"] * 1.5 + 0.05
    )
    assert projection["required_native_hours"] <= 3.0
    assert projection["live_native_remaining_hours_at_proposal"] == pytest.approx(7.7003634793909335)
    assert projection["fits_budget"]


def test_builder_reads_live_ledger_when_no_snapshot_is_supplied(tmp_path, monkeypatch):
    parent_run, data, masks = parent_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(P.ledger, "validate_shared_reservation_paths", lambda: (
        tmp_path / "ledger.jsonl", tmp_path / "gpu.lock"
    ))
    monkeypatch.setattr(P.ledger, "usage", lambda path: calls.append(path) or {
        "native_remaining_hours": 7.7003634793909335,
    })
    protocol = P.build_protocol(
        parent_run, output_root=tmp_path / "output", repository=REPO, mask_directory=masks,
    )
    assert calls == [tmp_path / "ledger.jsonl"]
    assert protocol["projection"]["live_native_remaining_hours_at_proposal"] == pytest.approx(
        7.7003634793909335
    )
    assert not (tmp_path / "ledger.jsonl").exists()
    assert not (tmp_path / "gpu.lock").exists()


def test_prepare_writes_only_protocol_and_sidecar_without_touching_inputs(tmp_path):
    parent_run, data, masks = parent_fixture(tmp_path)
    output = tmp_path / "continuation-output"
    before = {
        path: digest(path)
        for root in (parent_run, data)
        for path in root.rglob("*") if path.is_file()
    }
    protocol = P.prepare(
        parent_run, output_root=output, repository=REPO,
        before={"native_remaining_hours": 7.7003634793909335}, mask_directory=masks,
    )
    assert protocol["status"] == "proposed_not_authorized_or_launched"
    assert sorted(path.name for path in output.iterdir()) == [
        "continuation_protocol.json", "continuation_protocol.sha256",
    ]
    encoded = (output / "continuation_protocol.json").read_bytes()
    assert json.loads(encoded) == protocol
    assert (output / "continuation_protocol.sha256").read_text().strip() == hashlib.sha256(encoded).hexdigest()
    assert not (output / "fork_manifest.json").exists()
    assert not list(output.rglob("*.pt"))
    after = {
        path: digest(path)
        for root in (parent_run, data)
        for path in root.rglob("*") if path.is_file()
    }
    assert after == before
    with pytest.raises(FileExistsError):
        P.prepare(
            parent_run, output_root=output, repository=REPO,
            before={"native_remaining_hours": 7.7003634793909335}, mask_directory=masks,
        )


def test_protocol_rejects_intermediate_selection_or_wrong_accounting(tmp_path):
    protocol, _, _, _ = build_fixture(tmp_path)
    wrong = json.loads(json.dumps(protocol))
    wrong["evaluation_policy"]["intermediate_checkpoint_selection"] = True
    with pytest.raises(ValueError, match="must not select intermediate"):
        P.validate_protocol(wrong)
    wrong = json.loads(json.dumps(protocol))
    wrong["children"][1]["additional_main_loss_tokens"] = 16711920
    with pytest.raises(ValueError, match="exposure versus main-loss"):
        P.validate_protocol(wrong)
    wrong = json.loads(json.dumps(protocol))
    wrong["children"][0]["final_checkpoint"] = wrong["children"][0]["final_checkpoint"].replace(
        "final-update-00013568.pt", "update-00010000.pt"
    )
    with pytest.raises(ValueError, match="final-update-00013568"):
        P.validate_protocol(wrong)
