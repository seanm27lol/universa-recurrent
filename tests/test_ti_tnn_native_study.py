"""Offline orchestration checks with fake profiles, subprocesses, and GPU accounting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "ti_tnn"))
from native import ledger as native_ledger  # noqa: E402
from native import study as S  # noqa: E402
from native.config import MODES  # noqa: E402
from native.data import ARRAY_DTYPES, ARRAY_NAMES, IGNORE_INDEX  # noqa: E402
from native.reconstruct import REPLAY_FORMAT  # noqa: E402


torch.set_num_threads(1)
COMMIT = "a" * 40


def clean_snapshot(repository):
    repository = Path(repository)
    tracked = [
        str(path.relative_to(repository))
        for path in (repository / "research" / "ti_tnn" / "native").glob("*.py")
    ]
    tracked.append("research/ti_tnn/scaling/ledger.py")
    return {"clean": True, "commit": COMMIT, "tracked_files": tracked}


def write_packed(root: Path) -> Path:
    context = 8
    for split, rows in (("train", 4), ("dev", 2)):
        directory = root / split
        directory.mkdir(parents=True)
        arrays = {
            "input_ids": np.zeros((rows, context), dtype=np.int32),
            "labels": np.full((rows, context), IGNORE_INDEX, dtype=np.int32),
            "source_ids": np.full((rows, context), -1, dtype=np.int8),
            "positions": np.zeros((rows, context), dtype=np.int32),
            "doc_ids": np.full((rows, context), -1, dtype=np.int32),
            "key_padding_mask": np.ones((rows, context), dtype=np.bool_),
            "route_targets": np.zeros((rows, context, 4), dtype=np.int8),
            "route_mask": np.zeros((rows, context, 4), dtype=np.bool_),
            "state_targets": np.zeros((rows, context, 4, 4), dtype=np.int16),
            "state_mask": np.zeros((rows, context, 4, 4), dtype=np.bool_),
        }
        for row in range(rows):
            ids = (row * 3 + np.arange(context) + 1) % 31
            arrays["input_ids"][row] = ids
            arrays["labels"][row] = ids
            arrays["positions"][row] = np.arange(context)
            arrays["doc_ids"][row] = row + (100 if split == "dev" else 0)
            arrays["key_padding_mask"][row] = False
            arrays["labels"][row, 0] = IGNORE_INDEX
            if row % 2:
                arrays["doc_ids"][row, 4:] = row + 1000
                arrays["labels"][row, 4] = IGNORE_INDEX
            if row == 1:
                arrays["key_padding_mask"][row, -1] = True
                arrays["labels"][row, -1] = IGNORE_INDEX
                arrays["positions"][row, -1] = 0
                arrays["doc_ids"][row, -1] = -1
            live = ~arrays["key_padding_mask"][row]
            arrays["source_ids"][row, live] = row % 3
            arrays["route_targets"][row] = np.arange(context * 4).reshape(context, 4) % 4
            arrays["route_mask"][row] = live[:, None]
            arrays["state_targets"][row] = np.arange(context * 4 * 4).reshape(context, 4, 4) % 5
            arrays["state_mask"][row] = live[:, None, None]
        for name in ARRAY_NAMES:
            assert arrays[name].dtype == ARRAY_DTYPES[name]
            np.save(directory / f"{name}.npy", arrays[name], allow_pickle=False)
    structured = root / "structured"
    structured.mkdir()
    for split in S.STRUCTURED_SPLITS:
        record = {
            "example_id": f"{split}-0",
            "split": split,
            "prompt": f"prompt for {split}",
            "target_text": "r0=0 r1=0 r2=0 r3=0",
            "program": [],
            "initial": [0, 0, 0, 0],
        }
        (structured / f"{split}.jsonl").write_text(json.dumps(record) + "\n")
    code = {"example_id": "code-0", "prompt": "def f():\n    ", "target_text": "return 1"}
    (root / "code_generation.jsonl").write_text(json.dumps(code) + "\n")
    (root / "manifest.json").write_text(json.dumps({"test_fixture": True}) + "\n")
    return root


def write_replay(path: Path) -> Path:
    values = {
        "format": REPLAY_FORMAT,
        "teacher_forced": {
            "input_ids": [[1, 2, 3, 4]],
            "positions": [[0, 1, 0, 1]],
            "doc_ids": [[5, 5, 9, 9]],
            "key_padding_mask": [[False, False, False, False]],
            "labels": [[-100, 2, -100, 4]],
            "source_ids": [[0, 0, 2, 2]],
        },
        "generation_prompts": [
            {"example_id": "fixed-a", "input_ids": [1, 2], "positions": [0, 1]},
            {"example_id": "fixed-b", "input_ids": [3, 4, 5], "positions": [0, 1, 2]},
        ],
        "max_new_tokens": 3,
        "eos_token_id": 7,
    }
    path.write_text(json.dumps(values, sort_keys=True) + "\n")
    return path


def measured_profile(mode: str, *, throughput: float = 1000.0) -> dict:
    return {
        "mode": mode,
        "measured": {
            "training": {
                "fixed_seconds": 1.0,
                "loss_tokens_per_second": throughput,
                "seconds_per_update": 0.1,
                "seconds_per_checkpoint": 0.2,
                "checkpoint_bytes_each": 100,
                "seconds_per_complete_dev_pass": 0.5,
                "language_loss": 123.0,
                "steps": [{"language_loss": 123.0}],
            },
            "complete_evaluation": {
                "teacher_forced_seconds": 0.5,
                "structured_generation_seconds": 1.0,
                "code_generation_seconds": 0.75,
                "state_interventions_seconds": 0.25,
            },
            "reconstruction": {"seconds": 0.25},
        },
    }


@pytest.fixture
def fixture(tmp_path):
    data = write_packed(tmp_path / "data")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(b"fake tokenizer; subprocesses never load it")
    replay = write_replay(tmp_path / "replay.json")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    for index, mode in enumerate(MODES):
        (profiles / f"{mode}.json").write_text(json.dumps(
            measured_profile(mode, throughput=1000.0 + index * 100.0)
        ))
    ledger = tmp_path / "gpu_ledger.jsonl"
    ledger.write_bytes(b"sentinel ledger bytes; fakes never append\n")
    lock = tmp_path / "gpu.lock"
    return {
        "tmp": tmp_path,
        "data": data,
        "tokenizer": tokenizer,
        "replay": replay,
        "profiles": sorted(profiles.glob("*.json")),
        "engineering": ROOT / "research" / "ti_tnn" / "native" / "configs" / "engineering.json",
        "serious": ROOT / "research" / "ti_tnn" / "native" / "configs" / "serious.json",
        "ledger": ledger,
        "lock": lock,
    }


def freeze(fixture, output: Path, *, available=12.0, provisional=100, snapshot=clean_snapshot):
    return S.freeze_study(
        output,
        data=fixture["data"],
        tokenizer=fixture["tokenizer"],
        replay_spec=fixture["replay"],
        engineering_profiles=fixture["profiles"],
        engineering_config=fixture["engineering"],
        serious_config=fixture["serious"],
        repository=ROOT,
        available_gpu_hours=available,
        ledger_path=fixture["ledger"],
        ledger_lock=fixture["lock"],
        provisional_loss_tokens=provisional,
        train_hyperparameters={"batch_size": 2, "checkpoint_every": 2, "dev_every": 2},
        git_snapshot_fn=snapshot,
    )


def arguments(command):
    return {command[index]: command[index + 1] for index in range(len(command) - 1) if command[index].startswith("--")}


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def fake_train(protocol, mode: str, output: Path, reservation: str) -> dict:
    output.mkdir(parents=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    budget = protocol["budget"]
    steps = budget["planned_updates"]
    total = budget["actual_loss_tokens"]
    base, remainder = divmod(total, steps)
    increments = [base + (1 if index < remainder else 0) for index in range(steps)]
    cumulative = np.cumsum(increments).tolist()
    checkpoint_records = []
    kinds = [(0, "initial", "update-00000000.pt")]
    kinds.extend(
        (step, "periodic", f"update-{step:08d}.pt")
        for step in range(1, steps)
        if step % protocol["train"]["checkpoint_every"] == 0
    )
    kinds.append((steps, "final", f"final-update-{steps:08d}.pt"))
    final_descriptor = None
    for step, kind, filename in kinds:
        path = checkpoints / filename
        path.write_bytes(f"{mode}:{kind}:{step}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        consumed = 0 if step == 0 else cumulative[step - 1]
        record = {
            "event": "checkpoint",
            "step": step,
            "path": f"checkpoints/{filename}",
            "sha256": digest,
            "bytes": path.stat().st_size,
            "consumed_loss_tokens": consumed,
        }
        checkpoint_records.append(record)
        if kind == "final":
            manifest_path = path.with_name(f"{path.name}.{digest}.manifest.json")
            manifest = {
                "format": "native-checkpoint-v1",
                "checkpoint_name": path.name,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "step": step,
                "consumed_loss_tokens": consumed,
            }
            write_json(manifest_path, manifest)
            final_descriptor = {
                "path": str(path),
                "manifest_path": str(manifest_path),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "step": step,
                "consumed_loss_tokens": consumed,
            }
    learning = []
    dev_records = []
    for step in range(1, steps + 1):
        consumed = cumulative[step - 1]
        learning.append({
            "event": "train",
            "step": step,
            "update": step,
            "consumed_loss_tokens": consumed,
            "loss_tokens": increments[step - 1],
            "language_loss": 10.0 / step,
            "objective_loss": 10.0 / step,
            "wall_seconds": 0.01,
        })
        dev_due = step == steps or step % protocol["train"]["dev_every"] == 0
        if dev_due:
            dev = {
                "event": "dev",
                "step": step,
                "consumed_loss_tokens": consumed,
                "loss_tokens": protocol["data"]["dev_loss_tokens"],
                "language_loss": 8.0 / step,
                "route_usage": [{"pass": index, "decisions": 2} for index in range(4)]
                if mode in ("untyped", "typed") else [],
            }
            learning.append(dev)
            dev_records.append(dev)
    learning_path = output / "learning_curve.jsonl"
    learning_path.write_text("".join(json.dumps(record) + "\n" for record in learning))
    journal_path = output / "checkpoint_journal.jsonl"
    batches = [{
        "event": "batch",
        "step": step,
        "consumed_loss_tokens": cumulative[step - 1],
        "loss_tokens": increments[step - 1],
    } for step in range(1, steps + 1)]
    journal_path.write_text("".join(json.dumps(record) + "\n" for record in checkpoint_records + batches))
    training = {
        **protocol["train"],
        "max_loss_tokens": budget["requested_loss_tokens"],
        "seed": protocol["development"]["seed"],
        "model_seed": protocol["development"]["model_seed"],
        "data_seed": protocol["development"]["data_seed"],
        "device": "cuda",
        "planned_updates": steps,
        "planned_consumed_loss_tokens": total,
    }
    run_config = {
        "format": "native-train-v1",
        "model": protocol["configs"][mode]["model"],
        "training": training,
        "source_config_sha256": protocol["configs"][mode]["sha256"],
        "train_data_identity": protocol["data"]["train_data_identity"],
        "dev_data_identity": protocol["data"]["dev_data_identity"],
        "order_identity": protocol["data"]["order_identity"],
        "code_identity": protocol["repository"]["native_code_identity"],
    }
    write_json(output / "run_config.json", run_config)
    final_dev = dict(dev_records[-1])
    final_dev["checkpoint"] = final_descriptor
    return {
        "format": "native-train-v1",
        "mode": mode,
        "device": "cuda",
        "step": steps,
        "planned_updates": steps,
        "consumed_loss_tokens": total,
        "max_loss_tokens": budget["requested_loss_tokens"],
        "reservation_id": reservation,
        "final_checkpoint": final_descriptor,
        "dev": final_dev,
        "learning_log": str(learning_path),
        "journal": str(journal_path),
        "data_loading_seconds": 0.1,
        "forward_seconds": 0.2,
        "backward_seconds": 0.3,
        "optimizer_seconds": 0.1,
        "checkpoint_seconds": 0.2,
        "actual_loss_tokens_per_second": 1000.0,
    }


def generation_report(frozen, *, partial=False):
    records = [{
        "example_id": sample["example_id"],
        "completion_text": "fake completion",
        "target_text": "fake target",
        "correct": False,
    } for sample in frozen["samples"]]
    return {
        "status": "partial" if partial else "complete",
        "complete": not partial,
        "examples": len(records),
        "evaluated": len(records),
        "correct": 0,
        "exact_match": 0.0,
        "max_new_tokens": 48,
        "records": records,
    }


def fake_evaluation(protocol, mode: str, checkpoint, output: Path, *, partial=False):
    structured = {"files": {}, "splits": {}, "baselines": {"boundary": "fake baseline"}}
    for split in S.STRUCTURED_SPLITS:
        frozen = protocol["evaluation"]["samples"]["structured"][split]
        structured["files"][split] = {"path": frozen["path"], "sha256": frozen["sha256"]}
        structured["splits"][split] = generation_report(frozen, partial=partial and split == "dev")
    code_frozen = protocol["evaluation"]["samples"]["code"]
    code = generation_report(code_frozen)
    code.update({
        "path": code_frozen["path"],
        "sha256": code_frozen["sha256"],
        "max_new_tokens": protocol["evaluation"]["code_max_new_tokens"],
    })
    report = {
        "format": "native-evaluation-v1",
        "checkpoint": {"path": checkpoint["path"], "sha256": checkpoint["sha256"], "config": {}},
        "teacher_forced": {
            "complete": True,
            "sequences": len(protocol["data"]["dev_sequences"]),
            "overall": {"loss": 3.0, "loss_tokens": protocol["data"]["dev_loss_tokens"]},
            "sources": {
                source: {"loss": 3.0, "loss_tokens": 1, "next_token_accuracy": 0.0}
                for source in ("wikipedia", "code", "structured")
            },
            "routing": {
                "status": "ok" if mode in ("untyped", "typed") else "not_applicable",
                "per_pass": {},
            },
        },
        "structured_generation": structured,
        "code_generation": code,
    }
    write_json(output, report)


def fake_interventions(protocol, checkpoint, output: Path):
    interventions = []
    for specification in protocol["interventions"]["specifications"]:
        interventions.append({
            **specification,
            "logit_effect": {"mean_absolute": 0.25, "changed_values": 2},
            "generation_effect": {
                "examples": len(protocol["interventions"]["samples"]),
                "changed_examples": 1,
                "records": [{"example_id": sample["example_id"], "changed": True}
                            for sample in protocol["interventions"]["samples"]],
            },
        })
    write_json(output, {
        "format": "native-state-interventions-v1",
        "checkpoint": {"path": checkpoint["path"], "sha256": checkpoint["sha256"]},
        "replay_spec_sha256": protocol["evaluation"]["replay"]["sha256"],
        "specifications": protocol["interventions"]["specifications"],
        "samples": protocol["interventions"]["samples"],
        "interventions": interventions,
        "limits": ["fake control limits"],
    })


def fake_reconstruction(protocol, checkpoint, output: Path):
    write_json(output, {
        "format": "native-reconstruction-report-v1",
        "checkpoint": {
            "path": checkpoint["path"],
            "sha256": checkpoint["sha256"],
            "sha256_verified": True,
        },
        "replay_spec_sha256": protocol["evaluation"]["replay"]["sha256"],
        "teacher_forced": {"agrees": True, "comparisons": {}},
        "generation": {"agrees": True, "records": []},
        "agreement": True,
    })


class FakeSubprocesses:
    def __init__(self, protocol, *, fail_mode=None, truncate_mode=None, partial_evaluation=False):
        self.protocol = protocol
        self.fail_mode = fail_mode
        self.truncate_mode = truncate_mode
        self.partial_evaluation = partial_evaluation
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append(list(command))
        module = command[2]
        if module == "native.train":
            values = arguments(command)
            mode = values["--mode"]
            reservation = f"reservation-{mode}-train"
            if mode == self.fail_mode:
                return SimpleNamespace(returncode=2, stdout=json.dumps({"reservation_id": reservation}), stderr="failed")
            result = fake_train(self.protocol, mode, Path(values["--output"]), reservation)
            if mode == self.truncate_mode:
                result["consumed_loss_tokens"] -= 1
            return SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr="")
        assert module == "native.study" and command[3] == "_reserved"
        values = arguments(command)
        mode, stage = values["--mode"], values["--stage"]
        arm = Path(values["--child-stdout"]).parent
        checkpoint = json.loads((arm / "train-result.json").read_text())["final_checkpoint"]
        reservation = f"reservation-{mode}-{stage}"
        if stage == "evaluate":
            fake_evaluation(
                self.protocol, mode, checkpoint, arm / "evaluation.json", partial=self.partial_evaluation
            )
            fake_interventions(self.protocol, checkpoint, arm / "state_interventions.json")
        else:
            fake_reconstruction(self.protocol, checkpoint, arm / "reconstruction.json")
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "reservation_id": reservation,
            "returncode": 0,
            "exact_native_command": [command[0], "-m", f"native.{stage}", *command[command.index("--") + 1:]],
        }), stderr="")


def fake_accounting(path, *, reservation_id, mode, stage, started, output):
    reservation_id = reservation_id or f"reservation-{mode}-{stage}"
    return {
        "ledger_path": str(path),
        "ledger_sha256": "sentinel",
        "reservation_id": reservation_id,
        "reservation": {
            "reservation": reservation_id,
            "hours": 0.01,
            "open": False,
            "ended": True,
            "status": "ok",
            "command": f"fake {mode} {stage}",
        },
        "events": [
            {"event": "start", "reservation": reservation_id, "elapsed": 0.0},
            {"event": "heartbeat", "reservation": reservation_id, "elapsed": 18.0},
            {"event": "end", "reservation": reservation_id, "elapsed": 36.0, "status": "ok"},
        ],
        "aggregate": {"used_hours": 24.0, "native_remaining_hours": 11.0},
        "status": "accounted",
    }


def run(study_path, protocol, fake=None, **kwargs):
    fake = fake or FakeSubprocesses(protocol)
    status = S.run_study(
        study_path,
        runner=fake,
        accounting_reader=fake_accounting,
        git_snapshot_fn=clean_snapshot,
        **kwargs,
    )
    return status, fake


def test_freeze_precedes_run_and_complete_summary_has_all_required_evidence(fixture, tmp_path, monkeypatch):
    before = tmp_path / "not-frozen"
    assert S.run_study(before, runner=FakeSubprocesses({}), accounting_reader=fake_accounting) == 3
    assert json.loads((before / "run-status.json").read_text())["status"] == "incomplete"

    forbidden_calls = []
    monkeypatch.setattr(native_ledger, "Reservation", lambda **kwargs: forbidden_calls.append(kwargs))
    ledger_before = fixture["ledger"].read_bytes()
    frozen = freeze(fixture, tmp_path / "study")
    assert frozen["status"] == "frozen"
    protocol = frozen["protocol"]
    assert protocol["development"]["seed"] == S.DEVELOPMENT_SEED
    assert protocol["repository"]["commit"] == COMMIT and protocol["repository"]["clean"]
    assert "language_loss" not in json.dumps(protocol["profiles"])
    assert protocol["budget"]["equal_across_arms"]
    assert protocol["budget"]["actual_loss_tokens"] >= protocol["budget"]["requested_loss_tokens"]
    assert protocol["budget"]["actual_loss_tokens"] == protocol["feasibility"]["provisional_plan"]["actual_loss_tokens"]
    assert len(protocol["data"]["dev_sequences"]) == 2
    assert all(len(values["samples"]) == 1 for values in protocol["evaluation"]["samples"]["structured"].values())
    assert protocol["evaluation"]["samples"]["code"]["samples"][0]["example_id"] == "code-0"
    assert protocol["evaluation"]["scoring_rules"]["structured"]["limit_per_split"] is None
    assert protocol["evaluation"]["scoring_rules"]["code"]["limit"] is None
    assert [protocol["configs"][mode]["model"]["mode"] for mode in MODES] == list(MODES)

    status, fake = run(tmp_path / "study", protocol)
    assert status == 0
    assert len(fake.calls) == 12
    assert [call[2] for call in fake.calls] == ["native.train", "native.study", "native.study"] * 4
    assert not forbidden_calls
    assert fixture["ledger"].read_bytes() == ledger_before
    assert not fixture["lock"].exists()
    summary = json.loads((tmp_path / "study" / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert set(summary["arms"]) == set(MODES)
    assert summary["equal_actual_loss_tokens_per_arm"] == protocol["budget"]["actual_loss_tokens"]
    for mode, arm in summary["arms"].items():
        assert arm["actual_loss_tokens"] == summary["equal_actual_loss_tokens_per_arm"]
        assert arm["loss_vs_token"]["train"] and arm["loss_vs_token"]["dev"]
        assert set(arm["per_source_metrics"]) == {"wikipedia", "code", "structured"}
        assert arm["generation"]["structured"]["splits"]["dev"]["complete"]
        assert arm["generation"]["code"]["complete"]
        assert arm["routing"]["checkpointed_evaluation"]["status"] == (
            "ok" if mode in ("untyped", "typed") else "not_applicable"
        )
        assert len(arm["state_interventions"]["interventions"]) == 3
        assert arm["checkpoints"]["all_journal_checkpoints"]
        assert len(arm["reservations"]) == 3
        assert all(stage["gpu_accounting"]["events"] for stage in arm["reservations"])
    proposal = summary["future_serious_125m_proposal"]
    assert proposal["status"] == "measured_proposal_only_not_authorization"
    assert proposal["authorization"] is False and proposal["opens_reservation"] is False
    assert proposal["target_loss_tokens_per_arm"] == 125_000_000
    assert proposal["typed_profile"] == protocol["serious_config"]["modes"]["typed"]
    assert proposal["required_available_gpu_hours_to_preserve_reserve"] == pytest.approx(
        proposal["total_four_arm_gpu_hours_before_reserve"] * 1.2
    )
    train_stage = json.loads((tmp_path / "study" / "arms" / "typed" / "stage-evaluate.json").read_text())
    assert train_stage["exact_native_command"][2] == "native.evaluate"
    assert "--limit-per-split" not in train_stage["exact_native_command"]
    assert "--max-batches" not in train_stage["exact_native_command"]
    reconstruct = json.loads((tmp_path / "study" / "arms" / "typed" / "stage-reconstruct.json").read_text())
    assert reconstruct["exact_native_command"][2] == "native.reconstruct"


def test_budget_is_reduced_from_measured_profiles_before_any_outcome(fixture, tmp_path):
    reference = freeze(fixture, tmp_path / "reference")
    feasibility = reference["protocol"]["feasibility"]
    provisional_hours = feasibility["provisional_projection"]["projected_reservation_hours"]
    minimum_hours = feasibility["minimum_projection"]["projected_reservation_hours"]
    assert provisional_hours > minimum_hours
    reduced = freeze(
        fixture,
        tmp_path / "reduced",
        available=(provisional_hours + minimum_hours) / 2,
    )
    protocol = reduced["protocol"]
    assert protocol["budget"]["reduction"]["selected_before_comparative_outcomes"]
    assert protocol["budget"]["actual_loss_tokens"] < feasibility["provisional_plan"]["actual_loss_tokens"]
    assert protocol["budget"]["actual_loss_tokens"] >= feasibility["minimum_plan"]["actual_loss_tokens"]
    assert protocol["budget"]["projection"]["projected_reservation_hours"] <= (
        provisional_hours + minimum_hours
    ) / 2
    assert protocol["budget"]["order_prefix_sha256"] != feasibility["provisional_plan"]["order_prefix_sha256"]
    assert not list((tmp_path / "reduced").glob("arms/*"))


@pytest.mark.parametrize("available", [0.0, 1e-9])
def test_measured_infeasibility_is_reported_and_cannot_run(fixture, tmp_path, available):
    result = freeze(fixture, tmp_path / "infeasible", available=available)
    assert result["status"] == "infeasible"
    assert "minimum complete curve" in result["reason"]
    assert (tmp_path / "infeasible" / "infeasible.json").is_file()
    assert not (tmp_path / "infeasible" / "protocol.json").exists()
    fake = FakeSubprocesses({})
    assert S.run_study(tmp_path / "infeasible", runner=fake, accounting_reader=fake_accounting) == 3
    assert not fake.calls


def test_truncated_dense_baseline_stops_before_other_arms_and_summary(fixture, tmp_path):
    protocol = freeze(fixture, tmp_path / "study")["protocol"]
    fake = FakeSubprocesses(protocol, truncate_mode="dense")
    status, fake = run(tmp_path / "study", protocol, fake)
    assert status == 3
    assert len(fake.calls) == 1
    assert json.loads((tmp_path / "study" / "arms" / "dense" / "status.json").read_text())["status"] == "failed"
    assert not (tmp_path / "study" / "arms" / "shared").exists()
    assert not (tmp_path / "study" / "arms" / "typed").exists()
    incomplete = S.summarize_study(tmp_path / "study", git_snapshot_fn=clean_snapshot)
    assert incomplete["status"] == "incomplete"
    assert not (tmp_path / "study" / "summary.json").exists()


def test_failed_shared_baseline_is_preserved_without_retry_or_reallocation(fixture, tmp_path):
    protocol = freeze(fixture, tmp_path / "study")["protocol"]
    original_budget = protocol["budget"]["actual_loss_tokens"]
    fake = FakeSubprocesses(protocol, fail_mode="shared")
    status, fake = run(tmp_path / "study", protocol, fake)
    assert status == 3
    assert len(fake.calls) == 4
    assert json.loads((tmp_path / "study" / "arms" / "dense" / "status.json").read_text())["status"] == "completed"
    assert json.loads((tmp_path / "study" / "arms" / "shared" / "status.json").read_text())["status"] == "failed"
    assert not (tmp_path / "study" / "arms" / "untyped").exists()
    before = len(fake.calls)
    assert run(tmp_path / "study", protocol, fake)[0] == 3
    assert len(fake.calls) == before
    assert json.loads((tmp_path / "study" / "protocol.json").read_text())["budget"]["actual_loss_tokens"] == original_budget
    incomplete = S.summarize_study(tmp_path / "study", git_snapshot_fn=clean_snapshot)
    assert not incomplete["comparative_outcomes_emitted"]
    shared_accounting = incomplete["preserved_arm_status_and_accounting"]["shared"]["gpu_accounting"]
    assert shared_accounting["train"]["events"][-1]["event"] == "end"


def test_partial_frozen_generation_is_refused(fixture, tmp_path):
    protocol = freeze(fixture, tmp_path / "study")["protocol"]
    fake = FakeSubprocesses(protocol, partial_evaluation=True)
    status, fake = run(tmp_path / "study", protocol, fake)
    assert status == 3
    assert len(fake.calls) == 2
    assert "generation is not complete" in json.loads(
        (tmp_path / "study" / "arms" / "dense" / "status.json").read_text()
    )["error"]
    assert S.summarize_study(tmp_path / "study", git_snapshot_fn=clean_snapshot)["status"] == "incomplete"


def test_clean_commit_and_active_producer_hashes_are_enforced(fixture, tmp_path):
    def dirty(repository):
        return {**clean_snapshot(repository), "clean": False}

    with pytest.raises(S.StudyError, match="clean commit"):
        freeze(fixture, tmp_path / "dirty", snapshot=dirty)

    protocol = freeze(fixture, tmp_path / "study")["protocol"]
    changed_head = lambda repository: {**clean_snapshot(repository), "commit": "b" * 40}
    fake = FakeSubprocesses(protocol)
    assert S.run_study(
        tmp_path / "study", runner=fake, accounting_reader=fake_accounting, git_snapshot_fn=changed_head
    ) == 3
    assert not fake.calls

    profile = fixture["profiles"][0]
    original = profile.read_bytes()
    try:
        profile.write_bytes(original + b"\n")
        assert S.run_study(
            tmp_path / "study", runner=fake, accounting_reader=fake_accounting, git_snapshot_fn=clean_snapshot
        ) == 3
        assert not fake.calls
    finally:
        profile.write_bytes(original)

    class MutatingSubprocesses(FakeSubprocesses):
        def __call__(self, command, **kwargs):
            result = super().__call__(command, **kwargs)
            if len(self.calls) == 1:
                profile.write_bytes(original + b"changed during native.train")
            return result

    mutating = MutatingSubprocesses(protocol)
    try:
        assert run(tmp_path / "study", protocol, mutating)[0] == 3
        assert len(mutating.calls) == 1
        assert "changed" in json.loads((tmp_path / "study" / "run-status.json").read_text())["reason"]
    finally:
        profile.write_bytes(original)


def test_native_profile_projection_scales_complete_dev_and_checkpoint_counts():
    raw = {
        "format": "native-profile-v1",
        "status": "ok",
        "mode": "dense",
        "dry_run": False,
        "timing_valid": True,
        "valid_for_projection": True,
        "config": {"name": "engineering"},
        "correctness": {"valid": True},
        "setup": {"fixed_seconds": 5.0},
        "training": {"actual_shifted_loss_tokens": 1000, "wall_seconds": 10.0},
        "evaluation": {"actual_shifted_loss_tokens": 200, "wall_seconds": 4.0},
        "checkpoint": {"wall_seconds": 2.0, "bytes": 1024},
        "complete_evaluation": {
            "teacher_forced_seconds": 8.0,
            "structured_generation_seconds": 6.0,
            "code_generation_seconds": 3.0,
            "state_interventions_seconds": 1.0,
        },
        "reconstruction": {"seconds": 2.0},
    }
    profile = {"mode": "dense", "projection": S._profile_projection(raw)}
    projection = S.project_arm(
        profile,
        {"planned_updates": 101, "actual_loss_tokens": 20_000},
        {"checkpoint_every": 100, "dev_every": 100},
        dev_loss_tokens=400,
        reserve_fraction=0.2,
    )
    assert projection["train_seconds"] == pytest.approx(5 + 200 + 6 + 16)
    assert projection["complete_evaluation_seconds"] == pytest.approx(8 + 6 + 3 + 1)
    assert projection["total_seconds"] == pytest.approx(247)
    assert projection["projected_reservation_gpu_hours"] == pytest.approx(247 * 1.2 / 3600)
    assert projection["checkpoints"]["total"] == 3
    assert projection["complete_train_dev_passes"] == 2
    raw["dry_run"] = True
    with pytest.raises(S.StudyError, match="cache-validated CUDA"):
        S._profile_projection(raw)
