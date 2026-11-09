"""CPU checks for measured native profiling and reservation-hour projections."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from native import checkpoint as C  # noqa: E402
from native import profile as P  # noqa: E402
from native.config import MODES, NativeConfig  # noqa: E402
from native.data import ARRAY_DTYPES, ARRAY_NAMES, IGNORE_INDEX, PackedDataset  # noqa: E402


torch.set_num_threads(1)
VOCAB_SIZE = 32
CONTEXT = 8


def tiny_config(mode="dense"):
    return NativeConfig(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT,
        d_model=8,
        n_heads=2,
        d_slot=4,
        d_ff=16,
        mode=mode,
        n_state_classes=5,
    )


def write_config(path: Path, *, name="engineering", mode="dense") -> Path:
    path.write_text(json.dumps({"name": name, "model": tiny_config(mode).to_dict()}))
    return path


def write_packed(root: Path) -> Path:
    for split, rows in (("train", 4), ("dev", 2)):
        directory = root / split
        directory.mkdir(parents=True)
        arrays = {
            "input_ids": np.zeros((rows, CONTEXT), dtype=np.int32),
            "labels": np.full((rows, CONTEXT), IGNORE_INDEX, dtype=np.int32),
            "source_ids": np.full((rows, CONTEXT), -1, dtype=np.int8),
            "positions": np.zeros((rows, CONTEXT), dtype=np.int32),
            "doc_ids": np.full((rows, CONTEXT), -1, dtype=np.int32),
            "key_padding_mask": np.ones((rows, CONTEXT), dtype=np.bool_),
            "route_targets": np.zeros((rows, CONTEXT, 4), dtype=np.int8),
            "route_mask": np.zeros((rows, CONTEXT, 4), dtype=np.bool_),
            "state_targets": np.zeros((rows, CONTEXT, 4, 4), dtype=np.int16),
            "state_mask": np.zeros((rows, CONTEXT, 4, 4), dtype=np.bool_),
        }
        for row in range(rows):
            ids = (row * 5 + np.arange(CONTEXT) + 1) % VOCAB_SIZE
            arrays["input_ids"][row] = ids
            arrays["labels"][row] = ids
            arrays["positions"][row] = np.arange(CONTEXT)
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
            arrays["route_targets"][row] = (np.arange(CONTEXT * 4).reshape(CONTEXT, 4) + row) % 4
            arrays["route_mask"][row] = live[:, None]
            arrays["state_targets"][row] = (
                np.arange(CONTEXT * 4 * 4).reshape(CONTEXT, 4, 4) + row
            ) % 5
            arrays["state_mask"][row] = live[:, None, None]
        for name in ARRAY_NAMES:
            assert arrays[name].dtype == ARRAY_DTYPES[name]
            np.save(directory / f"{name}.npy", arrays[name], allow_pickle=False)
    (root / "manifest.json").write_text(json.dumps({"test_fixture": True}) + "\n")
    return root


def profile_argv(config: Path, data: Path, output: Path, mode: str):
    return [
        "--config", str(config),
        "--mode", mode,
        "--data", str(data),
        "--output", str(output),
        "--device", "cpu",
        "--dry-run",
        "--batch-size", "2",
        "--warmup-steps", "1",
        "--measured-steps", "3",
        "--evaluation-batches", "1",
    ]


@pytest.mark.parametrize("mode", MODES)
def test_cpu_dry_run_uses_packed_batches_and_writes_real_checkpoint(tmp_path, monkeypatch, capsys, mode):
    data = write_packed(tmp_path / "data")
    config = write_config(tmp_path / "config.json")
    output = tmp_path / f"{mode}.json"
    synchronizations = []

    def synchronize(device):
        assert device.type == "cpu"
        synchronizations.append(str(device))

    def forbidden_reservation(**kwargs):
        raise AssertionError("CPU dry-run attempted to open a native Reservation")

    monkeypatch.setattr(P, "_synchronize", synchronize)
    monkeypatch.setattr(P, "Reservation", forbidden_reservation)
    assert P.main(profile_argv(config, data, output, mode)) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "dry_run" and not summary["timing_valid"]
    report = json.loads(output.read_text())

    assert report["format"] == P.PROFILE_FORMAT
    assert report["status"] == "dry_run"
    assert report["dry_run"] and not report["timing_valid"] and not report["valid_for_projection"]
    assert report["reservation"] == {"id": None, "hours": None, "projected_hours": None}
    assert report["correctness"]["valid"]
    assert report["correctness"]["next_token_logits"]["device"] == "cpu"
    if mode in ("untyped", "typed"):
        assert report["correctness"]["next_token_route_indices_equal"] is True
        assert report["correctness"]["next_token_route_logits"]["allclose"]
    else:
        assert report["correctness"]["next_token_route_logits"] is None
    assert synchronizations
    assert report["execution"]["mode"] == "eager"
    assert report["execution"]["autocast"] is None
    assert report["dtype"] == "float32"
    assert report["config"]["profile"]["requested_dtype"] == "bfloat16"
    assert report["setup"]["warmup_steps"] == 1
    assert report["training"]["measured_steps"] == 3
    assert report["training"]["warmup_steps_excluded"] == 1

    dataset = PackedDataset(data, "train")
    expected_tokens = 0
    for record in report["training"]["steps"]:
        labels = dataset.batch(record["indices"])["labels"]
        actual = int(torch.count_nonzero(labels[:, 1:] != IGNORE_INDEX))
        assert record["actual_shifted_loss_tokens"] == actual
        assert record["wall_seconds"] == pytest.approx(sum(record[name] for name in (
            "data_loading_seconds",
            "forward_seconds",
            "backward_seconds",
            "gradient_clipping_optimizer_seconds",
        )))
        expected_tokens += actual
    training = report["training"]
    assert training["actual_shifted_loss_tokens"] == expected_tokens
    assert training["actual_shifted_loss_tokens_per_second"] == pytest.approx(
        expected_tokens / training["wall_seconds"]
    )
    assert training["wall_seconds"] == pytest.approx(sum(
        training[name] for name in (
            "data_loading_seconds",
            "forward_seconds",
            "backward_seconds",
            "gradient_clipping_optimizer_seconds",
        )
    ))
    assert report["evaluation"]["teacher_forced"]
    assert report["evaluation"]["actual_shifted_loss_tokens"] > 0
    assert report["evaluation"]["wall_seconds"] > 0
    expected_balance_decisions = 15 * 4 if mode in ("untyped", "typed") else 0
    assert report["evaluation"]["balance_decisions"] == expected_balance_decisions

    checkpoint_path = Path(report["checkpoint"]["path"])
    assert checkpoint_path.is_file()
    assert checkpoint_path.stat().st_size == report["checkpoint_bytes"] == report["checkpoint"]["bytes"]
    restored = C.load_checkpoint(checkpoint_path)
    assert restored["step"] == 4
    assert restored["consumed_loss_tokens"] == (
        report["setup"]["warmup_actual_shifted_loss_tokens"] + expected_tokens
    )
    assert restored["optimizer"] is not None and restored["scheduler"] is not None
    assert restored["extra"]["measurement_checkpoint"] is True
    assert report["checkpoint"]["real_write"]
    assert report["checkpoint_seconds"] > 0
    assert report["memory"] == {
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
    }
    assert len(report["data"]["identity"]) == 64
    assert len(report["config"]["identity"]) == 64
    assert len(report["code"]["identity"]) == 64
    assert json.loads(output.read_text()) == report
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_cuda_attempt_enters_reservation_before_cuda_probe(tmp_path, monkeypatch):
    events = []

    class FakeReservation:
        id = "fake-attempt"
        projected_hours = 0.25

        def __init__(self, **kwargs):
            events.append(("constructed", kwargs))

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, traceback):
            events.append(("exit", exc_type))
            return False

        def elapsed_hours(self):
            return 0.001

        def check(self):
            events.append("check")

    monkeypatch.setattr(P, "Reservation", FakeReservation)
    monkeypatch.setattr(P.torch.cuda, "is_available", lambda: events.append("cuda_probe") or False)
    output = tmp_path / "failed.json"
    args = P.parser().parse_args([
        "--config", str(tmp_path / "not-read.json"),
        "--mode", "dense",
        "--data", str(tmp_path / "not-read-data"),
        "--output", str(output),
        "--device", "cuda",
        "--warmup-steps", "1",
        "--measured-steps", "3",
    ])
    with pytest.raises(RuntimeError, match="CUDA.*not available"):
        P.run_profile(args)
    assert events[0][0] == "constructed"
    assert events[1:3] == ["enter", "cuda_probe"]
    assert events[-1] == ("exit", RuntimeError)
    assert not (tmp_path / "not-read.json").exists()
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["reservation_id"] == "fake-attempt"
    assert report["reservation_hours"] == 0.001
    assert not report["timing_valid"]


def test_discovered_complete_inputs_are_measured_and_study_compatible(tmp_path, monkeypatch, capsys):
    data = write_packed(tmp_path / "data")
    structured = data / "structured"
    structured.mkdir()
    for split in P.E.SPLIT_NAMES:
        (structured / f"{split}.jsonl").write_text("{}\n")
    (data / "code_generation.jsonl").write_text("{}\n")
    (data / "replay.json").write_text("{}\n")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("mock tokenizer")
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "complete.json"
    records = {
        split: [
            {"split": split, "example_id": f"{split}-{index}", "prompt": f"prompt-{split}-{index}",
             "target_text": "r0=0 r1=0 r2=0 r3=0"}
            for index in range(130)
        ]
        for split in P.E.SPLIT_NAMES
    }
    code_records = [
        {"example_id": f"code-{index}", "prompt": f"code-prompt-{index}", "target_text": "return 0\n"}
        for index in range(65)
    ]
    files = {
        split: {"path": str(structured / f"{split}.jsonl"), "sha256": "a" * 64, "examples": 130}
        for split in P.E.SPLIT_NAMES
    }
    replay = {
        "teacher_forced": PackedDataset(data, "dev").batch([0]),
        "generation_prompts": [],
        "max_new_tokens": 1,
        "eos_token_id": 3,
        "spec_sha256": "b" * 64,
    }
    model_calls = []
    generation_calls = []
    intervention_calls = []
    reconstruction_calls = []

    def forbidden_reservation(**kwargs):
        raise AssertionError("CPU dry-run attempted to open a native Reservation")

    def forbidden_subprocess(*args, **kwargs):
        raise AssertionError("profiling spawned a child process")

    def synchronize(device):
        assert device.type == "cpu"

    def teacher(model, dataset, **kwargs):
        model_calls.append(model)
        assert kwargs["max_batches"] == 16 and kwargs["batch_size"] == 8
        assert kwargs["device"] == "cpu"
        return {
            "complete": False,
            "batches": 16,
            "sequences": 128,
            "overall": {"loss_tokens": 1000},
        }

    def generation(model, tokenizer, selected, **kwargs):
        generation_calls.append((model, kwargs))
        evaluated = len(selected) if kwargs["limit"] is None else min(kwargs["limit"], len(selected))
        return {
            "examples": len(selected),
            "evaluated": evaluated,
            "complete": evaluated == len(selected),
        }

    def interventions(model, logit_inputs, prompts, specifications, tokenizer, **kwargs):
        intervention_calls.append((model, logit_inputs, prompts, specifications, kwargs))
        return {"interventions": []}

    def reconstruct(path, spec, *, expected_sha256):
        reconstruction_calls.append((path, spec, expected_sha256))
        return {
            "format": P.R.REPORT_FORMAT,
            "agreement": True,
            "weights": {"dtype": "torch.float32"},
            "checkpoint": {"sha256": expected_sha256},
        }

    clock = iter(value / 1000 for value in range(1, 1_000_000))
    monkeypatch.setattr(P.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(P, "_synchronize", synchronize)
    monkeypatch.setattr(P, "Reservation", forbidden_reservation)
    monkeypatch.setattr(P, "_git_commit", lambda: "c" * 40)
    monkeypatch.setattr(P.subprocess, "run", forbidden_subprocess)
    monkeypatch.setattr(P.E, "load_gpt2_tokenizer", lambda path: object())
    monkeypatch.setattr(P.E, "teacher_forced_evaluate", teacher)
    monkeypatch.setattr(P.E, "load_structured_records", lambda path: (records, files))
    monkeypatch.setattr(P.E, "_generation_metrics", generation)
    monkeypatch.setattr(P.E, "structured_baseline_report", lambda selected: {"mock": True})
    monkeypatch.setattr(P.E, "_load_jsonl", lambda path: code_records)
    monkeypatch.setattr(P.E, "evaluate_state_interventions", interventions)
    monkeypatch.setattr(P.R, "load_replay_spec", lambda path: replay)
    monkeypatch.setattr(P.R, "reconstruct_checkpoint", reconstruct)

    argv = profile_argv(config, data, output, "typed") + [
        "--tokenizer", str(tokenizer_path),
        "--max-new-tokens", "2",
        "--code-max-new-tokens", "3",
    ]
    assert P.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "dry_run"
    report = json.loads(output.read_text())
    complete = report["complete_evaluation"]
    assert complete["valid"] and complete["status"] == "bounded_frozen_sample"
    assert complete["bounded_frozen_sample"]
    assert complete["limits"] == {
        "complete_evaluation_max_batches": 16,
        "complete_evaluation_batch_size": 8,
        "teacher_forced_max_sequences": 128,
        "structured_limit_per_split": 128,
        "code_generation_limit": 64,
        "max_new_tokens": 2,
        "code_max_new_tokens": 3,
    }
    assert report["config"]["profile"]["complete_evaluation_max_batches"] == 16
    assert report["execution"]["complete_evaluation_limits"] == complete["limits"]
    assert complete["child_processes"] == 0
    for name in (
        "teacher_forced_seconds", "structured_generation_seconds",
        "code_generation_seconds", "state_interventions_seconds",
    ):
        assert complete[name] > 0
    teacher_report = complete["components"]["teacher_forced"]
    assert teacher_report["status"] == "bounded_frozen_sample"
    assert teacher_report["batches"] == 16 and teacher_report["sequences"] == 128
    assert teacher_report["max_batches"] == teacher_report["complete_evaluation_max_batches"] == 16
    assert teacher_report["max_sequences"] == 128
    assert teacher_report["actual_shifted_loss_tokens"] == 1000
    structured_report = complete["components"]["structured_generation"]
    assert structured_report["status"] == "bounded_frozen_sample"
    assert structured_report["limit_per_split"] == structured_report["effective_limit_per_split"] == 128
    assert structured_report["seconds"] == structured_report["measured_seconds"]
    assert structured_report["seconds"] == structured_report["bounded_evaluation_seconds"]
    assert structured_report["full_source_examples"] == 4 * 130
    assert structured_report["evaluated_examples"] == 4 * 128
    for split in P.E.SPLIT_NAMES:
        values = structured_report["splits"][split]
        assert values["full_source_examples"] == 130 and values["evaluated"] == 128
        assert values["bounded_evaluation_seconds"] == values["measured_seconds"]
        assert "projected_complete_seconds" not in values
    code = complete["components"]["code_generation"]
    assert code["status"] == "bounded_frozen_sample"
    assert code["limit"] == code["effective_limit"] == 64
    assert code["full_source_examples"] == 65 and code["evaluated"] == 64
    assert code["seconds"] == code["measured_seconds"] == code["bounded_evaluation_seconds"]
    assert "projected_complete_generation_seconds" not in structured_report
    assert "projected_complete_generation_seconds" not in code
    assert complete["components"]["state_interventions"]["valid"]
    assert "correct" not in structured_report and "exact_match" not in code
    assert len({id(model) for model in model_calls + [call[0] for call in generation_calls]}) == 1
    assert intervention_calls[0][0] is model_calls[0]
    assert [record["example_id"] for record in intervention_calls[0][2]] == ["dev-0", "unseen_composition-0"]
    assert [spec.kind for spec in intervention_calls[0][3]] == ["zero", "shuffle", "wrong_slot"]
    assert reconstruction_calls[0][1] is replay
    assert reconstruction_calls[0][2] == report["checkpoint"]["sha256"]
    assert report["reconstruction"]["valid"] and report["reconstruction"]["seconds"] > 0
    assert Path(report["reconstruction"]["report_path"]).is_file()
    assert report["setup"]["post_checkpoint_seconds"] == pytest.approx(
        complete["total_seconds"] + report["reconstruction"]["seconds"]
    )

    from native import study as S
    real = {
        **report,
        "status": "ok",
        "dry_run": False,
        "timing_valid": True,
        "valid_for_projection": True,
    }
    projection = S._profile_projection(real)
    assert projection["evaluation"] == {
        name: complete[f"{name}_seconds"]
        for name in ("teacher_forced", "structured_generation", "code_generation", "state_interventions")
    }
    assert projection["reconstruction_seconds"] == report["reconstruction"]["seconds"]
    feasibility = P.measured_feasibility({"typed": real}, 100, 12.0)
    arm = feasibility["arms"]["typed"]
    assert arm["complete_evaluation_seconds"] == pytest.approx(complete["total_seconds"])
    assert arm["reconstruction_seconds"] == pytest.approx(report["reconstruction"]["seconds"])


def test_unlimited_complete_evaluation_requires_explicit_zero(tmp_path, monkeypatch):
    data = write_packed(tmp_path / "data")
    code_path = data / "code_generation.jsonl"
    code_path.write_text("{}\n")
    config = write_config(tmp_path / "config.json")
    args = P.parser().parse_args(profile_argv(config, data, tmp_path / "unused.json", "dense"))
    assert args.complete_evaluation_max_batches == 16
    assert args.complete_evaluation_batch_size == 8
    assert args.structured_limit_per_split == 128
    assert args.code_generation_limit == 64
    args.complete_evaluation_max_batches = 0
    args.structured_limit_per_split = 0
    args.code_generation_limit = 0
    records = {
        split: [
            {"split": split, "example_id": f"{split}-{index}", "prompt": "p", "target_text": "r0=0 r1=0 r2=0 r3=0"}
            for index in range(2)
        ]
        for split in P.E.SPLIT_NAMES
    }
    files = {
        split: {"path": str(data / "structured" / f"{split}.jsonl"), "sha256": "a" * 64, "examples": 2}
        for split in P.E.SPLIT_NAMES
    }
    limits = []

    def teacher(model, dataset, **kwargs):
        limits.append(kwargs["max_batches"])
        return {"complete": True, "batches": 1, "sequences": 2, "overall": {"loss_tokens": 14}}

    def generation(model, tokenizer, selected, **kwargs):
        limits.append(kwargs["limit"])
        return {"examples": len(selected), "evaluated": len(selected), "complete": True}

    monkeypatch.setattr(P, "_synchronize", lambda device: None)
    monkeypatch.setattr(P.E, "teacher_forced_evaluate", teacher)
    monkeypatch.setattr(P.E, "load_structured_records", lambda path: (records, files))
    monkeypatch.setattr(P.E, "_generation_metrics", generation)
    monkeypatch.setattr(P.E, "structured_baseline_report", lambda selected: {})
    monkeypatch.setattr(P.E, "_load_jsonl", lambda path: records["train"])
    model = P.NativeModel(tiny_config())
    _, teacher_report = P._complete_teacher_forced(
        model, PackedDataset(data, "dev"), args, torch.device("cpu"), None
    )
    assert teacher_report["status"] == "complete" and teacher_report["source_complete"]
    assert teacher_report["max_batches"] is None and teacher_report["max_sequences"] is None
    structured_report, _ = P._structured_generation(
        model, object(), data / "structured", args, None
    )
    code_report = P._code_generation(model, object(), code_path, args, None)
    assert structured_report["status"] == "complete"
    assert structured_report["bounded_evaluation_seconds"] is None
    assert structured_report["seconds"] == structured_report["measured_seconds"]
    assert structured_report["full_source_examples"] == structured_report["evaluated_examples"] == 8
    assert code_report["status"] == "complete" and code_report["bounded_evaluation_seconds"] is None
    assert code_report["full_source_examples"] == code_report["evaluated"] == 2
    assert code_report["seconds"] == code_report["measured_seconds"]
    assert limits == [None] * 6


def test_profiling_losses_forward_padding_mask_when_api_supports_it(tmp_path, monkeypatch):
    data = write_packed(tmp_path / "data")
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "unused.json"
    args = P.parser().parse_args(profile_argv(config, data, output, "dense"))
    batch = PackedDataset(data, "train").batch([0, 1])
    model = P.NativeModel(tiny_config())
    original = P.compute_losses
    received = []

    def with_padding(logits, labels, *, key_padding_mask=None, **kwargs):
        received.append(key_padding_mask)
        return original(logits, labels, **kwargs)

    monkeypatch.setattr(P, "compute_losses", with_padding)
    _, losses = P._forward_losses(model, batch, args, torch.float32)
    assert received == [batch["key_padding_mask"]]
    assert int(losses["token_count"]) > 0


def test_existing_generation_input_cannot_fall_through_to_zero_projection(tmp_path):
    data = write_packed(tmp_path / "data")
    (data / "code_generation.jsonl").write_text("{}\n")
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "missing-replay.json"
    args = P.parser().parse_args(profile_argv(config, data, output, "dense"))
    with pytest.raises(ValueError, match="reconstruction-replay is required"):
        P.run_profile(args)
    failure = json.loads(output.read_text())
    assert failure["status"] == "failed" and not failure["valid_for_projection"]
    assert "complete_evaluation" not in failure


def measured_arm(mode="dense", *, throughput=100.0):
    return {
        "format": P.PROFILE_FORMAT,
        "status": "ok",
        "dry_run": False,
        "timing_valid": True,
        "valid_for_projection": True,
        "config": {"name": "engineering"},
        "correctness": {"valid": True},
        "setup": {"fixed_seconds": 5.0},
        "complete_evaluation": {
            "valid": True,
            "teacher_forced_seconds": 8.0,
            "structured_generation_seconds": 6.0,
            "code_generation_seconds": 3.0,
            "state_interventions_seconds": 1.0,
        },
        "reconstruction": {"valid": True, "seconds": 2.0},
        "training": {
            "actual_shifted_loss_tokens": 1000,
            "wall_seconds": 1000.0 / throughput,
        },
        "evaluation": {
            "actual_shifted_loss_tokens": 200,
            "wall_seconds": 4.0,
        },
        "checkpoint": {
            "wall_seconds": 2.0,
            "bytes": 1024,
        },
        "macs_are_not_a_timing_basis": 10**30,
    }


def all_arms():
    return {mode: measured_arm(mode) for mode in MODES}


def test_measured_feasibility_projects_equal_budgets_and_all_costs():
    projection = P.measured_feasibility(
        all_arms(),
        token_budget_per_arm=1000,
        native_remaining_hours=20.0,
        evaluation_tokens_per_arm=400,
        evaluation_passes_per_arm=2,
        checkpoint_writes_per_arm=2,
    )
    arm = projection["arms"]["dense"]
    assert arm["training_seconds"] == pytest.approx(10.0)
    assert arm["evaluation_seconds"] == pytest.approx(16.0)
    assert arm["checkpoint_seconds"] == pytest.approx(4.0)
    assert arm["setup_seconds"] == pytest.approx(5.0)
    assert arm["complete_evaluation_seconds"] == pytest.approx(18.0)
    assert arm["reconstruction_seconds"] == pytest.approx(2.0)
    assert arm["total_seconds"] == pytest.approx(55.0)
    assert projection["measured_total_seconds"] == pytest.approx(220.0)
    assert projection["projected_reservation_hours"] == pytest.approx(264.0 / 3600.0)
    assert projection["total_checkpoint_bytes"] == 4 * 2 * 1024
    assert projection["native_cap_hours"] == 12.0
    assert projection["available_hours"] == 12.0
    assert projection["planning_margin"] == 0.20
    assert projection["fits"]


def test_measured_feasibility_refuses_without_twenty_percent_margin():
    with pytest.raises(P.FeasibilityError, match="including 20% planning margin") as caught:
        P.measured_feasibility(all_arms(), 1000, 0.02)
    projection = caught.value.projection
    assert not projection["fits"]
    assert projection["available_hours"] == 0.02
    assert projection["projected_reservation_hours"] > 0.02
    with pytest.raises(ValueError, match="at least 0.20"):
        P.measured_feasibility(all_arms(), 1000, 12.0, planning_margin=0.19)
    refused = P.measured_feasibility(all_arms(), 1000, 0.01, refuse=False)
    assert not refused["fits"]


def test_feasibility_rejects_dry_run_wrong_config_and_unvalidated_cache():
    profiles = all_arms()
    profiles["dense"]["dry_run"] = True
    with pytest.raises(ValueError, match="cache-validated CUDA"):
        P.measured_feasibility(profiles, 1000, 12.0)
    profiles = all_arms()
    profiles["typed"]["config"]["name"] = "serious"
    with pytest.raises(ValueError, match="engineering.json"):
        P.measured_feasibility(profiles, 1000, 12.0)
    profiles = all_arms()
    profiles["shared"]["correctness"]["valid"] = False
    with pytest.raises(ValueError, match="cache correctness"):
        P.measured_feasibility(profiles, 1000, 12.0)
    profiles = all_arms()
    profiles["dense"]["complete_evaluation"]["valid"] = False
    with pytest.raises(ValueError, match="complete evaluation and reconstruction"):
        P.measured_feasibility(profiles, 1000, 12.0)


def test_minimum_measured_steps_and_cpu_real_profile_are_refused(tmp_path):
    args = P.parser().parse_args([
        "--config", "engineering.json", "--mode", "dense", "--data", "data", "--output", "out.json",
        "--device", "cpu", "--dry-run", "--measured-steps", "2",
    ])
    with pytest.raises(ValueError, match="at least 3"):
        P.run_profile(args)
    args.measured_steps = 3
    args.dry_run = False
    with pytest.raises(ValueError, match="real profiling requires CUDA"):
        P.run_profile(args)
