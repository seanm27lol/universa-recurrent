"""Tiny CPU trainer checks; an overfit smoke run is not learning evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from native import checkpoint as C  # noqa: E402
from native import ledger as native_ledger  # noqa: E402
from native import train as T  # noqa: E402
from native.config import MODES, NativeConfig  # noqa: E402
from native.data import ARRAY_DTYPES, ARRAY_NAMES, IGNORE_INDEX, PackedDataset  # noqa: E402
from native.model import NativeModel  # noqa: E402


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


def write_config(path: Path, mode="dense") -> Path:
    path.write_text(json.dumps({"model": tiny_config(mode).to_dict()}))
    return path


def write_packed(root: Path, train_rows=4, dev_rows=2) -> Path:
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        directory = root / split
        directory.mkdir(parents=True)
        shape = (rows, CONTEXT)
        arrays = {
            "input_ids": np.zeros(shape, dtype=np.int32),
            "labels": np.full(shape, IGNORE_INDEX, dtype=np.int32),
            "source_ids": np.full(shape, -1, dtype=np.int8),
            "positions": np.zeros(shape, dtype=np.int32),
            "doc_ids": np.full(shape, -1, dtype=np.int32),
            "key_padding_mask": np.ones(shape, dtype=np.bool_),
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
            padding = (row % 3 == 1) + (row % 3 == 2) * 2
            if padding:
                arrays["key_padding_mask"][row, -padding:] = True
                arrays["labels"][row, -padding:] = IGNORE_INDEX
                arrays["positions"][row, -padding:] = 0
                arrays["doc_ids"][row, -padding:] = -1
            live = ~arrays["key_padding_mask"][row]
            arrays["source_ids"][row, live] = row % 3
            route = (np.arange(CONTEXT * 4).reshape(CONTEXT, 4) + row) % 4
            state = (np.arange(CONTEXT * 4 * 4).reshape(CONTEXT, 4, 4) + row) % 5
            arrays["route_targets"][row] = route
            arrays["route_mask"][row] = live[:, None]
            arrays["state_targets"][row] = state
            arrays["state_mask"][row] = live[:, None, None]
        for name in ARRAY_NAMES:
            assert arrays[name].dtype == ARRAY_DTYPES[name]
            np.save(directory / f"{name}.npy", arrays[name], allow_pickle=False)
    (root / "manifest.json").write_text(json.dumps({"test_fixture": True}) + "\n")
    return root


def args_for(config: Path, data: Path, output: Path, *, mode="dense", budget=24, **overrides):
    args = T.parser().parse_args([
        "--config", str(config),
        "--mode", mode,
        "--data", str(data),
        "--output", str(output),
        "--device", "cpu",
        "--max-loss-tokens", str(budget),
        "--batch-size", "2",
        "--seed", "0",
        "--checkpoint-every", "2",
        "--dev-every", "2",
        "--dev-batches", "1",
    ])
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def records(output: Path, event=None):
    result = [json.loads(line) for line in (output / T.LEARNING_LOG_NAME).read_text().splitlines()]
    return [record for record in result if event is None or record["event"] == event]


def journal_steps(output: Path):
    journal = C.CheckpointJournal(output / T.JOURNAL_NAME)
    entries, _, incomplete, _ = journal.entries()
    assert not incomplete
    return [entry.record["step"] for entry in entries if entry.record["event"] == "batch"]


def assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_nested_equal(left, right)
    else:
        assert actual == expected


def test_cli_all_modes_defaults_diagnostics_and_byte_identical_batches(tmp_path, capsys):
    data = write_packed(tmp_path / "data")
    config = write_config(tmp_path / "config.json")
    outputs = {}
    summaries = {}
    for mode in MODES:
        output = tmp_path / mode
        argv = [
            "--config", str(config), "--mode", mode, "--data", str(data), "--output", str(output),
            "--device", "cpu", "--max-loss-tokens", "1", "--batch-size", "2", "--seed", "0",
            "--dev-batches", "1",
        ]
        assert T.main(argv) == 0
        outputs[mode] = output
        summaries[mode] = json.loads(capsys.readouterr().out.strip())
        assert summaries[mode]["step"] == 1
        assert summaries[mode]["reservation_id"] is None
        initial = C.load_checkpoint(output / "checkpoints" / "update-00000000.pt")
        final = C.load_checkpoint(summaries[mode]["final_checkpoint"]["path"])
        assert initial["step"] == 0 and initial["optimizer"] is not None and initial["scheduler"] is not None
        assert final["optimizer"] is not None and final["scheduler"] is not None
        assert final["sampler"]["consumed_loss_tokens"] == final["consumed_loss_tokens"]
        assert Path(final["path"]).is_file() and Path(final["manifest"]["manifest_path"] if "manifest_path" in final["manifest"] else summaries[mode]["final_checkpoint"]["manifest_path"]).is_file()
        train_record = records(output, "train")[0]
        assert train_record["loss"] == train_record["language_loss"]
        assert train_record["objective_loss"] != train_record["language_loss"]
        assert len(train_record["route_usage"]) == (4 if mode in ("untyped", "typed") else 0)
        if mode in ("untyped", "typed"):
            for pass_usage in train_record["route_usage"]:
                assert len(pass_usage["experts"]) == 4
                assert "starved_experts" in pass_usage and "load_balance" in pass_usage
        run_config = json.loads((output / "run_config.json").read_text())
        assert run_config["training"]["lambda_balance"] == 0.01
        assert run_config["training"]["lambda_route"] == 0.1
        assert run_config["training"]["lambda_state"] == 0.1
    reference = records(outputs["dense"], "train")[0]
    for mode in MODES[1:]:
        current = records(outputs[mode], "train")[0]
        for key in ("indices", "batch_identity", "batch_content_sha256", "model_inputs_sha256", "targets_sha256"):
            assert current[key] == reference[key]


@pytest.mark.parametrize("mode", MODES)
def test_gradients_reach_native_modules_and_adamw_groups(tmp_path, mode):
    data = write_packed(tmp_path / "data")
    dataset = PackedDataset(data, "train")
    torch.manual_seed(7)
    model = NativeModel(tiny_config(mode)).train()
    optimizer = T.build_optimizer(model, learning_rate=0.003, weight_decay=0.01)
    scheduler = T.build_scheduler(optimizer, 1)
    batch = dataset.batch([0, 1])
    expected = int(torch.count_nonzero(batch["labels"][:, 1:] != IGNORE_INDEX))
    result = T.train_batch(model, optimizer, scheduler, batch, expected_loss_tokens=expected)
    assert result["loss_tokens"] == expected
    assert np.isfinite(result["objective_loss"])
    for group in optimizer.param_groups:
        if group["weight_decay"]:
            assert all(parameter.ndim >= 2 for parameter in group["params"])
        else:
            assert all(parameter.ndim < 2 for parameter in group["params"])
    core = [
        model.token_embedding.weight,
        model.position_embedding.weight,
        model.output_head.weight,
        model.slot_initializers,
        model.h_reader.weight,
        model.h_writer.weight,
    ]
    for parameter in core:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert all(probe.weight.grad is not None and probe.weight.grad.abs().sum() > 0 for probe in model.state_probes)
    if mode == "dense":
        transforms = list(model.transforms)
        attentions = list(model.attentions)
    elif mode == "shared":
        transforms = [model.transform]
        attentions = [model.attention]
    else:
        transforms = list(model.experts)
        attentions = [model.attention]
        assert model.router.weight.grad is not None and model.router.weight.grad.abs().sum() > 0
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for transform in transforms for parameter in transform.parameters())
    assert all(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in attention.parameters()) for attention in attentions)
    if mode == "typed":
        assert all(reader.weight.grad is not None and reader.weight.grad.abs().sum() > 0 for reader in model.slot_readers)
        assert all(writer.weight.grad is not None and writer.weight.grad.abs().sum() > 0 for writer in model.slot_writers)


def test_right_padding_preserves_total_objective_gradients_and_live_route_usage(tmp_path):
    data = write_packed(tmp_path / "data")
    dataset = PackedDataset(data, "train")
    full = dataset.batch([0, 3])
    live_length = 4
    short = {name: tensor[:, :live_length].clone() for name, tensor in full.items()}
    padded = {name: tensor.clone() for name, tensor in full.items()}
    padded["input_ids"][:, live_length:] = 0
    padded["labels"][:, live_length:] = IGNORE_INDEX
    padded["source_ids"][:, live_length:] = -1
    padded["positions"][:, live_length:] = 0
    padded["doc_ids"][:, live_length:] = -1
    padded["key_padding_mask"][:, live_length:] = True
    padded["route_targets"][:, live_length:] = 3
    padded["route_mask"][:, live_length:] = False
    padded["state_targets"][:, live_length:] = 4
    padded["state_mask"][:, live_length:] = False
    assert not short["key_padding_mask"].any()
    assert short["route_mask"].all() and short["state_mask"].all()

    torch.manual_seed(23)
    short_model = NativeModel(tiny_config("typed")).train()
    padded_model = NativeModel(tiny_config("typed")).train()
    padded_model.load_state_dict(short_model.state_dict())

    def update(model, batch):
        optimizer = T.build_optimizer(model, learning_rate=0.001, weight_decay=0.01)
        scheduler = T.build_scheduler(optimizer, 1)
        expected = int(torch.count_nonzero(batch["labels"][:, 1:] != IGNORE_INDEX))
        record = T.train_batch(
            model,
            optimizer,
            scheduler,
            batch,
            grad_clip=1e9,
            lambda_balance=0.2,
            lambda_route=0.3,
            lambda_state=0.4,
            expected_loss_tokens=expected,
        )
        gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        return record, gradients

    short_record, short_gradients = update(short_model, short)
    padded_record, padded_gradients = update(padded_model, padded)
    for name in ("language_loss", "route_loss", "state_loss", "balance_loss", "objective_loss"):
        assert short_record[name] > 0
        assert padded_record[name] == pytest.approx(short_record[name], rel=2e-6, abs=2e-7)
    for name in ("loss_tokens", "route_targets", "state_targets", "balance_decisions"):
        assert padded_record[name] == short_record[name]
    live_tokens = short["input_ids"].numel()
    assert short_record["balance_decisions"] == live_tokens * 4
    assert short_record["loss_tokens"] == int(torch.count_nonzero(short["labels"][:, 1:] != IGNORE_INDEX))

    assert short_gradients.keys() == padded_gradients.keys()
    for name, short_gradient in short_gradients.items():
        padded_gradient = padded_gradients[name]
        assert (short_gradient is None) == (padded_gradient is None), name
        if short_gradient is not None:
            torch.testing.assert_close(padded_gradient, short_gradient, rtol=2e-5, atol=2e-6, msg=name)

    assert len(short_record["route_usage"]) == len(padded_record["route_usage"]) == 4
    for short_pass, padded_pass in zip(short_record["route_usage"], padded_record["route_usage"]):
        assert short_pass["decisions"] == padded_pass["decisions"] == live_tokens
        assert sum(expert["count"] for expert in padded_pass["experts"]) == live_tokens
        assert padded_pass["starved_experts"] == short_pass["starved_experts"]
        assert padded_pass["load_balance"] == pytest.approx(short_pass["load_balance"], rel=2e-6, abs=2e-7)
        for short_expert, padded_expert in zip(short_pass["experts"], padded_pass["experts"]):
            assert padded_expert["count"] == short_expert["count"]
            assert padded_expert["selected_fraction"] == short_expert["selected_fraction"]
            assert padded_expert["mean_probability"] == pytest.approx(
                short_expert["mean_probability"], rel=2e-6, abs=2e-7
            )


def test_two_percent_warmup_then_cosine_to_ten_percent():
    model = NativeModel(tiny_config())
    optimizer = T.build_optimizer(model, learning_rate=0.01, weight_decay=0.01)
    scheduler = T.build_scheduler(optimizer, 100)
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == pytest.approx(0.005)
    assert rates[1] == pytest.approx(0.01)
    assert rates[2:] == sorted(rates[2:], reverse=True)
    assert rates[50] > 0.001
    assert rates[-1] == pytest.approx(0.001)


def test_tiny_repeated_batch_loss_decreases(tmp_path):
    data = write_packed(tmp_path / "data")
    config = write_config(tmp_path / "config.json")
    output = tmp_path / "overfit"
    args = args_for(
        config, data, output, mode="shared", budget=180,
        learning_rate=0.01, lambda_balance=0.0, lambda_route=0.0, lambda_state=0.0,
    )
    result = T.run_training(args)
    curve = records(output, "train")
    assert len(curve) == result["step"] >= 8
    first = np.mean([record["language_loss"] for record in curve[:3]])
    last = np.mean([record["language_loss"] for record in curve[-3:]])
    assert last < first * 0.95
    assert all(record["objective_loss"] == pytest.approx(record["language_loss"]) for record in curve)


def test_interrupted_resume_replays_exactly_without_duplicate_acknowledgments(tmp_path):
    data = write_packed(tmp_path / "data")
    config = write_config(tmp_path / "config.json")
    reference_args = args_for(config, data, tmp_path / "reference", mode="typed", budget=72,
                              learning_rate=0.003, checkpoint_every=1)
    reference = T.run_training(reference_args)
    interrupted_args = args_for(config, data, tmp_path / "interrupted", mode="typed", budget=72,
                                learning_rate=0.003, checkpoint_every=1)

    def interrupt(record):
        if record["step"] == 2:
            raise RuntimeError("simulated interruption after batch acknowledgment")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        T.run_training(interrupted_args, batch_hook=interrupt)
    plan = C.recover(interrupted_args.output / T.JOURNAL_NAME if isinstance(interrupted_args.output, Path) else Path(interrupted_args.output) / T.JOURNAL_NAME)
    assert [entry.step for entry in plan.replay_batches] == [2]
    resumed = T.run_training(interrupted_args)
    assert resumed["replayed_updates"] == 1
    assert resumed["step"] == reference["step"]
    expected = C.load_checkpoint(reference["final_checkpoint"]["path"])
    actual = C.load_checkpoint(resumed["final_checkpoint"]["path"])
    assert actual["state_hashes"] == expected["state_hashes"]
    assert_nested_equal(actual["model"], expected["model"])
    assert_nested_equal(actual["optimizer"], expected["optimizer"])
    assert_nested_equal(actual["scheduler"], expected["scheduler"])
    assert actual["sampler"] == expected["sampler"]
    assert_nested_equal(actual["rng"], expected["rng"])
    output = Path(interrupted_args.output)
    assert journal_steps(output) == list(range(1, reference["step"] + 1))
    replay = records(output, "replay")
    assert [record["step"] for record in replay] == [2]
    reference_by_step = {record["step"]: record for record in records(Path(reference_args.output), "train")}
    for record in records(output, "replay") + records(output, "train"):
        assert record["batch_content_sha256"] == reference_by_step[record["step"]]["batch_content_sha256"]


def test_corrupt_final_and_incomplete_tail_fall_back_replay_and_record(tmp_path):
    data = write_packed(tmp_path / "data")
    config = write_config(tmp_path / "config.json")
    args = args_for(config, data, tmp_path / "damaged", mode="dense", budget=48, checkpoint_every=1)
    first = T.run_training(args)
    Path(first["final_checkpoint"]["path"]).write_bytes(b"corrupt final checkpoint")
    with (tmp_path / "damaged" / T.JOURNAL_NAME).open("ab") as handle:
        handle.write(b'{"event":"batch","step":')
    plan = C.recover(tmp_path / "damaged" / T.JOURNAL_NAME)
    assert plan.incomplete_tail and len(plan.invalid_checkpoints) == 1
    assert plan.requires_replay
    resumed = T.run_training(args)
    assert resumed["replayed_updates"] >= 1
    assert resumed["recovery"]["truncated_incomplete_bytes"] > 0
    assert len(resumed["recovery"]["invalid_checkpoints"]) == 1
    loaded = C.load_checkpoint(resumed["final_checkpoint"]["path"])
    assert loaded["step"] == first["step"]
    assert loaded["consumed_loss_tokens"] == first["consumed_loss_tokens"]
    assert journal_steps(tmp_path / "damaged") == list(range(1, first["step"] + 1))
    recovery_events = [json.loads(line) for line in (tmp_path / "damaged" / T.RECOVERY_LOG_NAME).read_text().splitlines()]
    assert [event["event"] for event in recovery_events] == ["started", "completed"]


def test_stopping_uses_shifted_labels_not_unshifted_or_padded_tokens(tmp_path):
    data = write_packed(tmp_path / "data")
    config = write_config(tmp_path / "config.json")
    sampler = T.make_sampler(PackedDataset(data, "train"), batch_size=2, data_seed=0)
    first = sampler.next_batch()
    second = sampler.next_batch()
    exact = T.run_training(args_for(config, data, tmp_path / "exact", budget=first.loss_tokens))
    crossed = T.run_training(args_for(config, data, tmp_path / "crossed", budget=first.loss_tokens + 1))
    assert exact["step"] == 1 and exact["consumed_loss_tokens"] == first.loss_tokens
    assert crossed["step"] == 2
    assert crossed["consumed_loss_tokens"] == first.consumed_loss_tokens + second.loss_tokens
    assert crossed["consumed_loss_tokens"] > first.loss_tokens + 1
    for output in (tmp_path / "exact", tmp_path / "crossed"):
        for record in records(output, "train"):
            labels = PackedDataset(data, "train").batch(record["indices"])["labels"]
            actual = int(torch.count_nonzero(labels[:, 1:] != IGNORE_INDEX))
            assert record["loss_tokens"] == actual
            assert record["loss_tokens"] < record["indices"].__len__() * CONTEXT


def test_cpu_never_writes_historical_ledger_or_opens_lock(tmp_path, monkeypatch):
    data = write_packed(tmp_path / "data")
    config = write_config(tmp_path / "config.json")
    shared = tmp_path / "shared"
    shared.mkdir()
    ledger = shared / "gpu_ledger.jsonl"
    historical = b"historical-ledger-bytes-must-stay-unchanged\n"
    ledger.write_bytes(historical)
    monkeypatch.setenv("TI_TNN_SCALING_RUNS", str(shared))

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU training attempted to open a native Reservation")

    monkeypatch.setattr(native_ledger, "Reservation", forbidden)
    result = T.run_training(args_for(config, data, tmp_path / "cpu", budget=1))
    assert result["reservation_id"] is None
    assert ledger.read_bytes() == historical
    assert not (shared / "gpu.lock").exists()


def test_cuda_override_validation_precedes_reservation_and_model(tmp_path, monkeypatch):
    args = args_for(
        tmp_path / "missing-config.json",
        tmp_path / "missing-data",
        tmp_path / "cuda",
        device="cuda",
        ledger_path=str(tmp_path / "unanchored" / "gpu_ledger.jsonl"),
        ledger_lock=str(tmp_path / "unanchored" / "gpu.lock"),
    )
    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: True)
    calls = []

    def refuse_unanchored(path, lock):
        calls.append((path, lock))
        raise RuntimeError("unanchored production override")

    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA setup passed the reservation-path validation boundary")

    monkeypatch.setattr(native_ledger, "validate_shared_reservation_paths", refuse_unanchored)
    monkeypatch.setattr(native_ledger, "Reservation", forbidden)
    monkeypatch.setattr(T, "NativeModel", forbidden)
    monkeypatch.setattr(T, "PackedDataset", forbidden)
    with pytest.raises(RuntimeError, match="unanchored production override"):
        T.run_training(args)
    assert calls == [(args.ledger_path, args.ledger_lock)]


def test_cuda_reservation_uses_validated_temporary_pair(tmp_path, monkeypatch):
    data = write_packed(tmp_path / "data")
    config = write_config(tmp_path / "config.json")
    args = args_for(
        config,
        data,
        tmp_path / "cuda",
        budget=1,
        device="cuda",
        ledger_path=str(tmp_path / "requested-ledger.jsonl"),
        ledger_lock=str(tmp_path / "requested-lock"),
    )
    anchored = (tmp_path / "anchored" / "gpu_ledger.jsonl", tmp_path / "anchored" / "gpu.lock")
    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(native_ledger, "validate_shared_reservation_paths", lambda path, lock: anchored)
    captured = {}

    class CaptureReservation:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop before opening reservation context")

    def forbidden_model(*args, **kwargs):
        raise AssertionError("model initialization occurred before the reservation boundary")

    monkeypatch.setattr(native_ledger, "Reservation", CaptureReservation)
    monkeypatch.setattr(T, "NativeModel", forbidden_model)
    with pytest.raises(RuntimeError, match="stop before opening reservation context"):
        T.run_training(args)
    assert captured["path"] == anchored[0]
    assert captured["lock"] == anchored[1]
    assert not anchored[0].exists()
    assert not anchored[1].exists()
