import json
import random

import numpy as np
import pytest
import torch

from native import checkpoint as C
from native.sampler import PackedSequenceSampler


CONFIG = {"model": "tiny-linear", "recipe": [1, 2, 3]}
CODE_IDENTITY = "native-test-code-v1"


def packed_sequences():
    return [
        {
            "sequence_id": f"packed-{index}",
            "input_ids": torch.tensor([index, index + 1, index + 2], dtype=torch.long),
            "doc_ids": torch.tensor([index, index, index + 100], dtype=torch.long),
            "loss_mask": torch.tensor([1, 1, index % 2], dtype=torch.bool),
        }
        for index in range(6)
    ]


def training_state():
    torch.manual_seed(1234)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    model.train()
    optimizer.zero_grad()
    model(torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, -1.0]])).square().mean().backward()
    optimizer.step()
    scheduler.step()
    return model, optimizer, scheduler


def assert_nested_equal(actual, expected):
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


def save(path, model, optimizer, scheduler, sampler, **kwargs):
    return C.save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        sampler=sampler,
        config=CONFIG,
        code_identity=CODE_IDENTITY,
        **kwargs,
    )


def test_complete_checkpoint_restores_model_optimizer_scheduler_sampler_and_rng(tmp_path):
    random.seed(81)
    np.random.seed(81)
    torch.manual_seed(81)
    model, optimizer, scheduler = training_state()
    assert sum(parameter.numel() for parameter in model.parameters()) == 8
    sampler = PackedSequenceSampler(packed_sequences(), batch_size=2, seed=44)
    sampler.next_batch()
    descriptor = save(tmp_path / "complete.pt", model, optimizer, scheduler, sampler)
    assert descriptor["step"] == 1
    assert descriptor["cursor"] == 2
    assert descriptor["consumed_loss_tokens"] == 5
    assert C.sha256_file(descriptor["path"]) == descriptor["sha256"]
    assert json.loads(open(descriptor["manifest_path"]).read())["sha256"] == descriptor["sha256"]

    expected_rng = (random.random(), np.random.randint(0, 1_000_000), torch.rand(4))
    random.random()
    np.random.randint(0, 1_000_000)
    torch.rand(4)
    expected_batch = sampler.next_batch()

    resumed_model, resumed_optimizer, resumed_scheduler = training_state()
    with torch.no_grad():
        for parameter in resumed_model.parameters():
            parameter.add_(100)
    resumed_sampler = PackedSequenceSampler(packed_sequences(), batch_size=2, seed=44)
    result = C.restore_checkpoint(
        descriptor["path"],
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        resumed_sampler,
        expected_config=CONFIG,
        expected_code_identity=CODE_IDENTITY,
        expected_data_identity=sampler.data_identity,
        expected_order_identity=sampler.order_identity,
    )
    assert result.step == 1 and result.cursor == 2 and result.consumed_loss_tokens == 5
    assert_nested_equal(resumed_model.state_dict(), model.state_dict())
    assert_nested_equal(resumed_optimizer.state_dict(), optimizer.state_dict())
    assert_nested_equal(resumed_scheduler.state_dict(), scheduler.state_dict())
    assert (random.random(), np.random.randint(0, 1_000_000)) == expected_rng[:2]
    assert torch.equal(torch.rand(4), expected_rng[2])
    assert resumed_sampler.next_batch() == expected_batch


def test_checkpoint_refuses_identities_and_detects_tampering(tmp_path):
    model, optimizer, scheduler = training_state()
    sampler = PackedSequenceSampler(packed_sequences(), batch_size=2, seed=7)
    descriptor = save(tmp_path / "identity.pt", model, optimizer, scheduler, sampler)
    with pytest.raises(C.CheckpointIdentityMismatch, match="config identity"):
        C.load_checkpoint(descriptor["path"], expected_config={"model": "different"})
    with pytest.raises(C.CheckpointIdentityMismatch, match="code identity"):
        C.load_checkpoint(descriptor["path"], expected_code_identity="different-code")
    with pytest.raises(C.CheckpointIdentityMismatch, match="data identity"):
        C.load_checkpoint(descriptor["path"], expected_data_identity="different-data")
    with pytest.raises(C.CheckpointIdentityMismatch, match="order identity"):
        C.load_checkpoint(descriptor["path"], expected_order_identity="different-order")

    path = tmp_path / "identity.pt"
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(C.CheckpointValidationError, match="hash manifest"):
        C.load_checkpoint(path)


@pytest.mark.parametrize("fail_at", [1, 2])
def test_interrupted_atomic_replace_preserves_previous_checkpoint_and_manifest(tmp_path, monkeypatch, fail_at):
    model, optimizer, scheduler = training_state()
    sampler = PackedSequenceSampler(packed_sequences(), batch_size=2, seed=17)
    sampler.next_batch()
    path = tmp_path / "latest.pt"
    first = save(path, model, optimizer, scheduler, sampler)
    first_bytes = path.read_bytes()
    first_manifest = C.manifest_path(path).read_bytes()
    sampler.next_batch()
    with torch.no_grad():
        model.weight.add_(1)
    replacements = 0

    def interrupt_replace(source, destination):
        nonlocal replacements
        replacements += 1
        if replacements == fail_at:
            raise OSError("simulated interruption before atomic replace")
        return original_replace(source, destination)

    original_replace = C.os.replace
    monkeypatch.setattr(C.os, "replace", interrupt_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        save(path, model, optimizer, scheduler, sampler)
    assert path.read_bytes() == first_bytes
    assert C.manifest_path(path).read_bytes() == first_manifest
    assert C.load_checkpoint(path)["step"] == first["step"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def logged_run(tmp_path):
    model, optimizer, scheduler = training_state()
    sampler = PackedSequenceSampler(packed_sequences(), batch_size=2, seed=29)
    journal = C.CheckpointJournal(tmp_path / "journal.jsonl")
    checkpoints = []
    batches = []
    for step in range(1, 4):
        batch = sampler.next_batch()
        batches.append(batch)
        optimizer.zero_grad()
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        scheduler.step()
        journal.append_batch(batch, sampler)
        descriptor = save(tmp_path / f"step-{step}.pt", model, optimizer, scheduler, sampler)
        journal.append_checkpoint(descriptor)
        checkpoints.append(descriptor)
    final_batch = sampler.next_batch()
    batches.append(final_batch)
    journal.append_batch(final_batch, sampler)
    return model, optimizer, scheduler, sampler, journal, checkpoints, batches


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_corrupt_or_missing_final_checkpoint_falls_back_and_plans_all_logged_replay(tmp_path, damage):
    model, optimizer, scheduler, sampler, journal, checkpoints, batches = logged_run(tmp_path)
    del model, optimizer, scheduler
    final_path = tmp_path / "step-3.pt"
    if damage == "corrupt":
        final_path.write_bytes(b"corrupt final checkpoint")
    else:
        final_path.unlink()

    sampler_before = sampler.state_dict()
    torch_before = torch.get_rng_state()
    plan = C.recover(
        journal.path,
        expected_config=CONFIG,
        expected_code_identity=CODE_IDENTITY,
        expected_data_identity=sampler.data_identity,
        expected_order_identity=sampler.order_identity,
    )
    assert sampler.state_dict() == sampler_before
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert plan.checkpoint_path == tmp_path / "step-2.pt"
    assert plan.checkpoint["step"] == 2
    assert plan.requires_replay
    assert [record.step for record in plan.replay_batches] == [3, 4]
    assert len(plan.invalid_checkpoints) == 1
    assert plan.acknowledged_cursor == batches[-1].cursor
    assert plan.acknowledged_consumed_loss_tokens == batches[-1].consumed_loss_tokens

    resumed_model, resumed_optimizer, resumed_scheduler = training_state()
    resumed_sampler = PackedSequenceSampler(packed_sequences(), batch_size=2, seed=29)
    restored = plan.restore(resumed_model, resumed_optimizer, resumed_scheduler, resumed_sampler)
    assert restored.step == 2
    replayed = plan.validate_replay(resumed_sampler)
    assert [batch.indices for batch in replayed] == [batches[2].indices, batches[3].indices]
    assert resumed_sampler.cursor == restored.cursor
    for expected in plan.replay_batches:
        assert resumed_sampler.next_batch() == expected.batch
    assert resumed_sampler.cursor == batches[-1].cursor
    assert resumed_sampler.consumed_loss_tokens == batches[-1].consumed_loss_tokens


def test_newest_valid_checkpoint_needs_no_replay_and_recovery_does_not_move_sampler(tmp_path):
    model, optimizer, scheduler = training_state()
    sampler = PackedSequenceSampler(packed_sequences(), batch_size=2, seed=3)
    journal = C.CheckpointJournal(tmp_path / "journal.jsonl")
    batch = sampler.next_batch()
    journal.append_batch(batch, sampler)
    descriptor = save(tmp_path / "valid.pt", model, optimizer, scheduler, sampler)
    journal.append_checkpoint(descriptor)
    before = sampler.state_dict()
    plan = C.recover(journal.path)
    assert not plan.requires_replay
    assert plan.acknowledged_cursor == batch.cursor
    assert plan.validate_replay(sampler) == ()
    assert sampler.state_dict() == before


def test_incomplete_journal_tail_is_explicitly_truncated_not_acknowledged(tmp_path):
    model, optimizer, scheduler = training_state()
    sampler = PackedSequenceSampler(packed_sequences(), batch_size=2, seed=12)
    journal = C.CheckpointJournal(tmp_path / "journal.jsonl")
    descriptor = save(tmp_path / "base.pt", model, optimizer, scheduler, sampler)
    journal.append_checkpoint(tmp_path / "base.pt")
    complete_bytes = journal.path.read_bytes()
    with open(journal.path, "ab") as handle:
        handle.write(b'{"event":"batch","step":1')

    plan = C.recover(journal.path)
    assert plan.incomplete_tail
    assert not plan.requires_replay
    assert plan.valid_journal_bytes == len(complete_bytes)
    assert plan.journal_bytes > len(complete_bytes)
    with pytest.raises(C.JournalCorrupt, match="refusing to append"):
        journal.append_checkpoint(descriptor)
    removed = plan.truncate_incomplete_tail()
    assert removed == plan.journal_bytes - plan.valid_journal_bytes
    assert journal.path.read_bytes() == complete_bytes
    assert plan.truncate_incomplete_tail() == 0
    assert not C.recover(journal.path).incomplete_tail


def test_valid_checkpoint_with_wrong_identity_is_not_bypassed_for_an_older_one(tmp_path):
    model, optimizer, scheduler = training_state()
    sampler = PackedSequenceSampler(packed_sequences(), batch_size=2, seed=37)
    journal = C.CheckpointJournal(tmp_path / "journal.jsonl")
    for step in range(2):
        batch = sampler.next_batch()
        journal.append_batch(batch, sampler)
        descriptor = save(tmp_path / f"identity-{step}.pt", model, optimizer, scheduler, sampler)
        journal.append_checkpoint(descriptor)
    with pytest.raises(C.CheckpointIdentityMismatch, match="code identity"):
        C.recover(journal.path, expected_code_identity="not-the-code")
