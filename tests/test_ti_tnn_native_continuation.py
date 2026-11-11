"""Tiny CPU-only checks for explicit native checkpoint continuations."""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from native import checkpoint as C  # noqa: E402
from native import continuation as N  # noqa: E402
from native import ledger as native_ledger  # noqa: E402
from native import train as T  # noqa: E402
from native.config import NativeConfig  # noqa: E402
from native.data import ARRAY_DTYPES, ARRAY_NAMES, IGNORE_INDEX, PackedDataset  # noqa: E402
from native.sampler import PackedSequenceSampler, identity_hash  # noqa: E402


torch.set_num_threads(1)
VOCAB_SIZE = 32
CONTEXT = 8


def tiny_config() -> NativeConfig:
    return NativeConfig(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT,
        d_model=8,
        n_heads=2,
        d_slot=4,
        d_ff=16,
        mode="dense",
        n_state_classes=5,
    )


def write_packed(root: Path) -> Path:
    for split, rows in (("train", 4), ("dev", 2)):
        directory = root / split
        directory.mkdir(parents=True)
        arrays = {
            "input_ids": np.zeros((rows, CONTEXT), dtype=np.int32),
            "labels": np.full((rows, CONTEXT), IGNORE_INDEX, dtype=np.int32),
            "source_ids": np.zeros((rows, CONTEXT), dtype=np.int8),
            "positions": np.tile(np.arange(CONTEXT, dtype=np.int32), (rows, 1)),
            "doc_ids": np.repeat(np.arange(rows, dtype=np.int32)[:, None], CONTEXT, axis=1),
            "key_padding_mask": np.zeros((rows, CONTEXT), dtype=np.bool_),
            "route_targets": np.zeros((rows, CONTEXT, 4), dtype=np.int8),
            "route_mask": np.ones((rows, CONTEXT, 4), dtype=np.bool_),
            "state_targets": np.zeros((rows, CONTEXT, 4, 4), dtype=np.int16),
            "state_mask": np.ones((rows, CONTEXT, 4, 4), dtype=np.bool_),
        }
        for row in range(rows):
            ids = (row * 5 + np.arange(CONTEXT) + 1) % VOCAB_SIZE
            arrays["input_ids"][row] = ids
            arrays["labels"][row] = ids
            arrays["labels"][row, 0] = IGNORE_INDEX
        for name in ARRAY_NAMES:
            assert arrays[name].dtype == ARRAY_DTYPES[name]
            np.save(directory / f"{name}.npy", arrays[name], allow_pickle=False)
    (root / "manifest.json").write_text(json.dumps({"tiny_continuation_fixture": True}) + "\n")
    return root


def write_answer_masks(root: Path) -> Path:
    for split, rows in (("train", 4), ("dev", 2)):
        directory = root / split
        directory.mkdir(parents=True)
        answer = np.zeros((rows, CONTEXT), dtype=np.bool_)
        answer[:, 3] = True
        answer[:, 4] = True
        value = np.zeros_like(answer)
        value[:, 3] = True
        formatting = answer & ~value
        for name, mask in (
            ("answer_mask", answer), ("answer_value_mask", value), ("formatting_eos_mask", formatting)
        ):
            np.save(directory / f"{name}.npy", mask, allow_pickle=False)
    return root


def training_args(config: Path, data: Path, output: Path, budget: int) -> object:
    return T.parser().parse_args([
        "--config", str(config), "--mode", "dense", "--data", str(data),
        "--output", str(output), "--device", "cpu", "--max-loss-tokens", str(budget),
        "--batch-size", "2", "--seed", "19", "--data-seed", "23", "--dev-batches", "1",
        "--checkpoint-every", "2",
    ])


def make_parent(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    data = write_packed(tmp_path / "data")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": tiny_config().to_dict()}))
    sampler = T.make_sampler(PackedDataset(data, "train"), batch_size=2, data_seed=23)
    epoch_tokens = sum(sampler.next_batch().loss_tokens for _ in range(len(sampler)))
    output = tmp_path / "parent"
    result = T.run_training(training_args(config, data, output, epoch_tokens))
    payload = C.load_checkpoint(result["final_checkpoint"]["path"])
    assert payload["step"] == 2 and payload["cursor"] == 4
    assert payload["extra"]["checkpoint_kind"] == "final"
    return data, output, result, payload


def fork_args(
    data: Path, child: Path, run_id: str, loss_mode: str, masks: Path | None, *, device: str = "cpu"
) -> object:
    argv = [
        "--config", str(child / "run_config.json"), "--mode", "dense", "--data", str(data),
        "--output", str(child), "--device", device, "--fork", run_id, "--loss-mode", loss_mode,
    ]
    if masks is not None:
        argv.extend(["--answer-masks", str(masks)])
    return T.parser().parse_args(argv)


def assert_nested_equal(actual, expected) -> None:
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_nested_equal(left, right)
    else:
        assert actual == expected


def assert_constant_lr_optimizer_inheritance(initial: dict, parent: dict) -> None:
    child_optimizer = initial["optimizer"]
    parent_optimizer = parent["optimizer"]
    assert child_optimizer.keys() == parent_optimizer.keys()
    assert_nested_equal(child_optimizer["state"], parent_optimizer["state"])
    assert len(child_optimizer["param_groups"]) == len(parent_optimizer["param_groups"])
    for child_group, parent_group in zip(child_optimizer["param_groups"], parent_optimizer["param_groups"]):
        assert child_group["lr"] == N.CONTINUATION_LEARNING_RATE
        assert child_group.keys() == parent_group.keys()
        for key in parent_group.keys() - {"lr"}:
            assert_nested_equal(child_group[key], parent_group[key])


def learning_records(output: Path, event: str = "train") -> list[dict]:
    return [
        json.loads(line) for line in (output / T.LEARNING_LOG_NAME).read_text().splitlines()
        if json.loads(line)["event"] == event
    ]


def test_completed_78_exposure_matches_complete_ordered_epochs() -> None:
    sequences = [{"loss_tokens": 206 + (index < 25)} for index in range(N.SEQUENCES_PER_EPOCH)]
    sampler = PackedSequenceSampler(sequences, batch_size=8, seed=20261113)
    state = {
        **sampler.state_dict(),
        "cursor": N.PARENT_CURSOR,
        "consumed_loss_tokens": N.PARENT_EXPOSURE_TOKENS,
        "batches_consumed": N.PARENT_UPDATES,
    }
    sampler.load_state_dict(state)
    parent = {
        "step": N.PARENT_UPDATES,
        "cursor": N.PARENT_CURSOR,
        "consumed_loss_tokens": N.PARENT_EXPOSURE_TOKENS,
        "sampler": state,
    }
    plan = N.continuation_plan(parent, sampler)
    assert plan["additional_updates"] == 10176
    assert plan["additional_exposure_tokens"] == 16711920
    assert (plan["target_updates"], plan["target_exposure_tokens"], plan["target_cursor"]) == (
        13568, 22282560, 108160
    )
    assert N.ADDITIONAL_EXPOSURE_TOKENS == 16711920
    assert N.TARGET_EXPOSURE_TOKENS == 22282560
    N.validate_program_generalization_contract(plan)


def test_fork_preserves_parent_optimizer_sampler_model_and_exact_rng(tmp_path: Path) -> None:
    data, parent_run, parent_result, parent = make_parent(tmp_path)
    del data
    parent_path = Path(parent_result["final_checkpoint"]["path"])
    parent_bytes = parent_path.read_bytes()
    parent_manifest = C.manifest_path(parent_path).read_bytes()
    random.random()
    np.random.randint(0, 100000)
    torch.rand(1)
    child_run = tmp_path / "child"
    fork = N.create_fork(
        parent_path,
        child_run,
        run_id="child-full",
        code_identity=T._code_identity(),
        loss_mode="full_sequence",
    )
    initial = C.load_checkpoint(fork["initial_checkpoint"]["path"])
    assert initial["step"] == parent["step"] == 2
    assert initial["cursor"] == parent["cursor"] == 4
    assert initial["consumed_loss_tokens"] == parent["consumed_loss_tokens"]
    assert_nested_equal(initial["model"], parent["model"])
    assert_nested_equal(initial["sampler"], parent["sampler"])
    assert_nested_equal(initial["rng"], parent["rng"])
    assert_constant_lr_optimizer_inheritance(initial, parent)
    assert initial["state_hashes"]["model"] == parent["state_hashes"]["model"]
    assert initial["state_hashes"]["sampler"] == parent["state_hashes"]["sampler"]
    assert initial["state_hashes"]["scheduler"] != parent["state_hashes"]["scheduler"]
    assert initial["identities"]["parent_model"] == parent["state_hashes"]["model"]
    assert initial["identities"]["parent_optimizer"] == parent["state_hashes"]["optimizer"]
    assert initial["identities"]["parent_sampler"] == parent["state_hashes"]["sampler"]
    assert initial["identities"]["parent_rng"] == identity_hash(parent["rng"])
    assert initial["identities"]["parent_checkpoint_sha256"] == parent["sha256"]
    assert initial["identities"]["parent_config"] == parent["identities"]["config"]
    assert initial["identities"]["parent_code"] == parent["identities"]["code"]
    assert initial["identities"]["parent_data"] == parent["identities"]["data"]
    assert initial["identities"]["parent_order"] == parent["identities"]["order"]
    assert initial["identities"]["data"] == parent["identities"]["data"]
    assert initial["identities"]["order"] == parent["identities"]["order"]
    assert initial["identities"]["code"] == T._code_identity()
    config = initial["config"]
    assert config["acquisition_run_id"] == "child-full"
    assert config["reservation_id"] is None
    assert initial["extra"]["reservation_id"] is None
    assert config["fork"]["continuation_study_id"] is None
    assert config["fork"]["parent"]["parent_acquisition_run_id"] is None
    assert config["fork"]["parent"]["parent_reservation_id"] is None
    assert initial["config"]["fork"]["schedule"] == {
        "learning_rate": 3e-5,
        "warmup_fraction": 0.0,
        "scheduler": N.CONTINUATION_SCHEDULER,
        "inherits_parent_scheduler": False,
    }
    assert initial["scheduler"]["last_epoch"] == 0
    rates = [group["lr"] for group in initial["optimizer"]["param_groups"]]
    assert rates == [3e-5] * len(rates)
    assert Path(fork["journal"]).parent == child_run
    assert parent_path.read_bytes() == parent_bytes
    assert C.manifest_path(parent_path).read_bytes() == parent_manifest
    assert not (parent_run / "fork-update-00000002.pt").exists()


def test_fork_separates_parent_owner_from_child_lineage(tmp_path: Path, monkeypatch) -> None:
    original_execute = T._execute

    def owned_parent_execute(args, **kwargs):
        kwargs["run_config"]["acquisition_run_id"] = "parent-acquisition"
        kwargs["run_config"]["reservation_id"] = "parent-reservation"
        return original_execute(args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(T, "_execute", owned_parent_execute)
        _, _, parent_result, parent = make_parent(tmp_path)
    parent_path = Path(parent_result["final_checkpoint"]["path"])
    parent_bytes = parent_path.read_bytes()
    result = N.create_fork(
        parent_path,
        tmp_path / "owned-child",
        run_id="child-owned",
        code_identity=T._code_identity(),
        continuation_study_id="continuation-study",
    )
    initial = C.load_checkpoint(result["initial_checkpoint"]["path"])
    config = initial["config"]
    parent_record = config["fork"]["parent"]
    assert parent["config"]["acquisition_run_id"] == "parent-acquisition"
    assert parent["config"]["reservation_id"] == "parent-reservation"
    assert parent_record["parent_acquisition_run_id"] == "parent-acquisition"
    assert parent_record["parent_reservation_id"] == "parent-reservation"
    assert parent_record["config_identity"] == parent["identities"]["config"]
    assert parent_record["code_identity"] == parent["identities"]["code"]
    assert parent_record["data_identity"] == parent["identities"]["data"]
    assert parent_record["order_identity"] == parent["identities"]["order"]
    assert config["run_id"] == config["fork"]["run_id"] == "child-owned"
    assert config["fork"]["continuation_study_id"] == "continuation-study"
    assert config["acquisition_run_id"] == "continuation-study"
    assert config["reservation_id"] is None
    assert initial["extra"]["reservation_id"] is None
    assert parent_path.read_bytes() == parent_bytes


def test_explicit_fork_run_loss_and_journal_validation(tmp_path: Path) -> None:
    data, parent_run, parent_result, _ = make_parent(tmp_path)
    masks = write_answer_masks(tmp_path / "answer_masks")
    full = N.create_fork(
        parent_result["final_checkpoint"]["path"], tmp_path / "full",
        run_id="child-full", code_identity=T._code_identity(),
    )
    answer = N.create_fork(
        parent_result["final_checkpoint"]["path"], tmp_path / "answer",
        run_id="child-answer", code_identity=T._code_identity(),
        loss_mode="answer_eos", answer_masks=masks,
    )
    del full, answer
    args = fork_args(data, tmp_path / "full", "child-full", "full_sequence", None)
    args.fork = None
    with pytest.raises(ValueError, match="explicit --fork"):
        T.run_training(args)
    args = fork_args(data, tmp_path / "full", "wrong-child", "full_sequence", None)
    with pytest.raises(C.CheckpointIdentityMismatch, match="run ID"):
        T.run_training(args)
    args = T.parser().parse_args([
        "--config", str(parent_run / "run_config.json"), "--mode", "dense", "--data", str(data),
        "--output", str(tmp_path / "implicit"), "--device", "cpu", "--fork", "child-full",
    ])
    with pytest.raises(ValueError, match="requires a config containing fork.run_id"):
        T.run_training(args)
    args = fork_args(data, tmp_path / "answer", "child-answer", "full_sequence", None)
    with pytest.raises(C.CheckpointIdentityMismatch, match="declared --loss-mode answer_eos"):
        T.run_training(args)
    args = fork_args(data, tmp_path / "answer", "child-answer", "answer_eos", masks)
    args.output = str(tmp_path / "missing-journal")
    with pytest.raises(C.RecoveryError, match="missing explicit initial-fork journal"):
        T.run_training(args)


def test_answer_eos_masks_only_main_targets_and_counts_selected_tokens(tmp_path: Path) -> None:
    data = write_packed(tmp_path / "data")
    batch = PackedDataset(data, "train").batch([0, 1])
    batch["answer_mask"] = torch.zeros_like(batch["labels"], dtype=torch.bool)
    batch["answer_mask"][:, 3] = True
    batch["answer_mask"][:, 4] = True
    original = {name: tensor.clone() for name, tensor in batch.items()}
    labels = T.main_loss_labels(batch, "answer_eos")
    assert labels[:, 3].equal(batch["labels"][:, 3])
    assert labels[:, 4].equal(batch["labels"][:, 4])
    assert torch.all(labels[:, :3] == IGNORE_INDEX)
    assert torch.all(labels[:, 5:] == IGNORE_INDEX)
    torch.manual_seed(71)
    model = T.NativeModel(tiny_config()).train()
    output, losses = T.forward_losses(model, batch, loss_mode="answer_eos")
    shifted_labels = original["labels"][:, 1:]
    selected = original["answer_mask"][:, 1:] & shifted_labels.ne(IGNORE_INDEX)
    expected = F.cross_entropy(
        output.logits[:, :-1][selected].float(), shifted_labels[selected], reduction="mean"
    )
    assert float(losses["language"].detach()) == pytest.approx(float(expected.detach()), rel=1e-6, abs=1e-7)
    assert int(losses["token_count"]) == 4
    assert int(losses["exposure_token_count"]) == 14
    for name in (
        "input_ids", "positions", "doc_ids", "key_padding_mask", "labels",
        "route_targets", "route_mask", "state_targets", "state_mask",
    ):
        assert torch.equal(batch[name], original[name])
    optimizer = T.build_optimizer(model, learning_rate=3e-5, weight_decay=0.01)
    scheduler = N.build_constant_scheduler(optimizer)
    record = T.train_batch(
        model, optimizer, scheduler, batch,
        expected_answer_tokens=4, expected_exposure_tokens=14, loss_mode="answer_eos",
    )
    assert record["main_loss_tokens"] == record["loss_tokens"] == 4
    assert record["exposure_tokens"] == 14
    assert record["learning_rate"] == record["next_learning_rate"] == 3e-5
    assert record["main_loss_tokens_per_second"] >= 0
    assert record["exposure_tokens_per_second"] >= 0
    with pytest.raises(AssertionError, match="answer mask selected 3"):
        T.train_batch(
            model, optimizer, scheduler, batch,
            expected_answer_tokens=3, expected_exposure_tokens=14, loss_mode="answer_eos",
        )
    with pytest.raises(AssertionError, match="sampler accounted 13 shifted labels"):
        T.train_batch(model, optimizer, scheduler, original, expected_loss_tokens=13)


def test_paired_children_share_order_exposure_schedule_and_complete_endpoint(tmp_path: Path) -> None:
    data, _, parent_result, _ = make_parent(tmp_path)
    masks = write_answer_masks(tmp_path / "answer_masks")
    children = {}
    for loss_mode in N.LOSS_MODES:
        run_id = f"child-{loss_mode}"
        child = tmp_path / loss_mode
        N.create_fork(
            parent_result["final_checkpoint"]["path"], child,
            run_id=run_id, code_identity=T._code_identity(), loss_mode=loss_mode,
            answer_masks=masks if loss_mode == "answer_eos" else None,
            additional_epochs=2,
        )
        result = T.run_training(fork_args(
            data, child, run_id, loss_mode, masks if loss_mode == "answer_eos" else None
        ))
        children[loss_mode] = (child, result, learning_records(child))
        assert result["step"] == 6
        assert result["cursor"] == 12
        assert result["consumed_exposure_tokens"] == 84
        assert result["new_updates"] == 4
        assert result["processed_exposure_tokens_this_invocation"] == 56
        assert [record["step"] for record in children[loss_mode][2]] == [3, 4, 5, 6]
        assert all(record["learning_rate"] == record["next_learning_rate"] == 3e-5
                   for record in children[loss_mode][2])
    full_records = children["full_sequence"][2]
    answer_records = children["answer_eos"][2]
    assert [record["indices"] for record in answer_records] == [record["indices"] for record in full_records]
    assert [record["batch_identity"] for record in answer_records] == [record["batch_identity"] for record in full_records]
    assert [record["model_inputs_sha256"] for record in answer_records] == [
        record["model_inputs_sha256"] for record in full_records
    ]
    for answer, full in zip(answer_records, full_records):
        assert answer["exposure_tokens"] == full["exposure_tokens"] == full["main_loss_tokens"] == 14
        assert answer["main_loss_tokens"] == 4
        assert answer["consumed_exposure_tokens"] == full["consumed_exposure_tokens"]
        assert answer["exposure_tokens_per_second"] >= 0
        assert answer["main_loss_tokens_per_second"] >= 0
    assert children["full_sequence"][1]["processed_main_loss_tokens_this_invocation"] == 56
    assert children["answer_eos"][1]["processed_main_loss_tokens_this_invocation"] == 16
    assert children["answer_eos"][1]["dev"]["main_loss_tokens"] == 4
    assert children["answer_eos"][1]["dev"]["exposure_tokens"] == 14


def prepared_cuda_fork(tmp_path: Path, *, continuation_study_id="continuation-study"):
    data, _, parent_result, _ = make_parent(tmp_path)
    child = tmp_path / "child-cuda"
    N.create_fork(
        parent_result["final_checkpoint"]["path"],
        child,
        run_id="child-cuda",
        code_identity=T._code_identity(),
        continuation_study_id=continuation_study_id,
        device="cuda",
        load_device="cpu",
    )
    args = fork_args(data, child, "child-cuda", "full_sequence", None, device="cuda")
    return args


def forbid_model_and_new_reservation(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("fork validation reached model allocation or a fresh reservation")

    monkeypatch.setattr(T, "NativeModel", forbidden)
    monkeypatch.setattr(native_ledger, "Reservation", forbidden)
    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(native_ledger, "validate_shared_reservation_paths", lambda path, lock: (path, lock))


def test_declared_cuda_fork_can_be_created_on_cpu(tmp_path: Path) -> None:
    _, _, parent_result, parent = make_parent(tmp_path)
    parent_path = Path(parent_result["final_checkpoint"]["path"])
    result = N.create_fork(
        parent_path,
        tmp_path / "child-cuda",
        run_id="child-cuda",
        code_identity=T._code_identity(),
        loss_mode="full_sequence",
        device="cuda",
        load_device="cpu",
    )
    initial = C.load_checkpoint(result["initial_checkpoint"]["path"])
    assert initial["config"]["training"]["device"] == "cuda"
    assert initial["config"]["training"]["max_loss_tokens"] == result["continuation"]["target_exposure_tokens"]
    assert initial["config"]["acquisition_run_id"] == "child-cuda"
    assert initial["config"]["reservation_id"] is None
    assert initial["extra"]["reservation_id"] is None
    assert initial["config"]["fork"]["continuation_study_id"] is None
    assert initial["step"] == parent["step"]
    assert initial["consumed_loss_tokens"] == parent["consumed_loss_tokens"]
    assert_constant_lr_optimizer_inheritance(initial, parent)
    assert initial["config"]["fork"]["schedule"]["learning_rate"] == N.CONTINUATION_LEARNING_RATE
    assert initial["config"]["fork"]["schedule"]["inherits_parent_scheduler"] is False
    assert_nested_equal(initial["rng"], parent["rng"])


def test_cuda_fork_without_inherited_reservation_is_rejected_before_allocation(tmp_path: Path, monkeypatch) -> None:
    args = prepared_cuda_fork(tmp_path)
    forbid_model_and_new_reservation(monkeypatch)
    monkeypatch.setattr(native_ledger, "active", lambda: None)
    with pytest.raises(RuntimeError, match="requires an inherited continuation reservation"):
        T.run_training(args)


@pytest.mark.parametrize("extra", [
    {"native_continuation_run_id": "wrong-study", "native_continuation_authorized_hours": 3.0},
    {"native_continuation_run_id": "continuation-study", "native_continuation_authorized_hours": 2.0},
])
def test_cuda_fork_rejects_wrong_authorized_owner_before_allocation(
    tmp_path: Path, monkeypatch, extra: dict
) -> None:
    args = prepared_cuda_fork(tmp_path)
    forbid_model_and_new_reservation(monkeypatch)
    owner = SimpleNamespace(id="reservation-1", extra=extra)
    monkeypatch.setattr(native_ledger, "active", lambda: owner)
    with pytest.raises(RuntimeError, match="not the authorized continuation owner"):
        T.run_training(args)


def test_manual_cuda_fork_without_continuation_study_is_forbidden(tmp_path: Path, monkeypatch) -> None:
    args = prepared_cuda_fork(tmp_path, continuation_study_id=None)
    forbid_model_and_new_reservation(monkeypatch)
    owner = SimpleNamespace(id="reservation-1", extra={
        "native_acquisition_run_id": "child-cuda",
        "native_continuation_run_id": None,
        "native_continuation_authorized_hours": 3.0,
    })
    monkeypatch.setattr(native_ledger, "active", lambda: owner)
    monkeypatch.setenv("TI_TNN_NATIVE_ACQUISITION_RUN_ID", "child-cuda")
    with pytest.raises(RuntimeError, match="not the authorized continuation owner"):
        T.run_training(args)


def test_cuda_fork_records_authorized_study_and_inherited_reservation(tmp_path: Path, monkeypatch) -> None:
    args = prepared_cuda_fork(tmp_path)
    forbid_model_and_new_reservation(monkeypatch)
    owner = SimpleNamespace(id="reservation-1", extra={
        "native_continuation_run_id": "continuation-study",
        "native_continuation_authorized_hours": 3.0,
    })
    monkeypatch.setattr(native_ledger, "active", lambda: owner)
    monkeypatch.delenv("TI_TNN_NATIVE_ACQUISITION_RUN_ID", raising=False)
    captured = {}

    def execute(args, **kwargs):
        captured.update(kwargs)
        return {"reservation_id": kwargs["reservation"].id}

    monkeypatch.setattr(T, "_execute", execute)
    assert T.run_training(args)["reservation_id"] == "reservation-1"
    assert captured["reservation"] is owner
    assert captured["run_config"]["acquisition_run_id"] == "continuation-study"
    assert captured["run_config"]["reservation_id"] == "reservation-1"


def test_active_owner_identity_preserves_initial_checkpoint_recovery(tmp_path: Path, monkeypatch) -> None:
    data, _, parent_result, _ = make_parent(tmp_path)
    child = tmp_path / "child-cuda"
    fork = N.create_fork(
        parent_result["final_checkpoint"]["path"],
        child,
        run_id="child-cuda",
        code_identity=T._code_identity(),
        continuation_study_id="continuation-study",
        additional_epochs=1,
        device="cuda",
        load_device="cpu",
    )
    initial_path = Path(fork["initial_checkpoint"]["path"])
    initial_bytes = initial_path.read_bytes()
    args = fork_args(data, child, "child-cuda", "full_sequence", None, device="cuda")
    owner = SimpleNamespace(id="reservation-1", extra={
        "native_continuation_run_id": "continuation-study",
        "native_continuation_authorized_hours": 3.0,
    }, check=lambda: None)
    monkeypatch.setattr(T.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(T.torch.accelerator, "current_stream", lambda: SimpleNamespace(is_capturing=lambda: False))
    monkeypatch.setattr(native_ledger, "validate_shared_reservation_paths", lambda path, lock: (path, lock))
    monkeypatch.setattr(native_ledger, "active", lambda: owner)
    monkeypatch.setattr(native_ledger, "Reservation", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("an authorized fork must not open a fresh reservation")
    ))
    original_execute = T._execute

    def cpu_execute(args, **kwargs):
        kwargs["device"] = torch.device("cpu")
        return original_execute(args, **kwargs)

    monkeypatch.setattr(T, "_execute", cpu_execute)
    result = T.run_training(args)
    assert result["new_updates"] == 2
    assert result["reservation_id"] == "reservation-1"
    final = C.load_checkpoint(result["final_checkpoint"]["path"])
    assert final["config"]["acquisition_run_id"] == "continuation-study"
    assert final["config"]["reservation_id"] == "reservation-1"
    initial = C.load_checkpoint(initial_path)
    assert initial["config"]["acquisition_run_id"] == "continuation-study"
    assert initial["config"]["reservation_id"] is None
    assert initial_path.read_bytes() == initial_bytes
