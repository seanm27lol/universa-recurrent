import json
import random

import numpy as np
import pytest
import torch

from native.sampler import PackedSequenceSampler, SamplerIdentityError, SamplerStateError


def packed_sequences():
    return [
        {
            "sequence_id": f"packed-{index}",
            "input_ids": torch.tensor([index, index + 1, index + 2], dtype=torch.long),
            "doc_ids": torch.tensor([index, index, index + 100], dtype=torch.long),
            "loss_mask": torch.tensor([1, index % 2, 1], dtype=torch.bool),
        }
        for index in range(5)
    ]


def test_order_is_seeded_and_independent_of_all_global_rngs():
    data = packed_sequences()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state()
    sampler = PackedSequenceSampler(data, batch_size=2, seed=91)
    first_epoch = [sampler.next_batch() for _ in range(3)]

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert sorted(index for batch in first_epoch for index in batch.indices) == list(range(5))

    random.seed(712)
    np.random.seed(712)
    torch.manual_seed(712)
    random.random()
    np.random.random(3)
    torch.nn.Linear(9, 7)
    repeated = PackedSequenceSampler(data, batch_size=2, seed=91)
    assert [repeated.next_batch().indices for _ in range(3)] == [batch.indices for batch in first_epoch]
    assert repeated.epoch_order(0) == sampler.epoch_order(0)
    assert repeated.epoch_order(1) != repeated.epoch_order(0)


def test_cursor_and_consumed_loss_tokens_are_exact_across_epoch_boundaries():
    sampler = PackedSequenceSampler(packed_sequences(), batch_size=2, seed=4, shuffle=False)
    batches = [sampler.next_batch() for _ in range(5)]
    assert [batch.indices for batch in batches] == [(0, 1), (2, 3), (4,), (0, 1), (2, 3)]
    assert [batch.cursor for batch in batches] == [2, 4, 5, 7, 9]
    per_sequence = [2, 3, 2, 3, 2]
    expected_tokens = []
    total = 0
    for batch in batches:
        total += sum(per_sequence[index] for index in batch.indices)
        expected_tokens.append(total)
    assert [batch.consumed_loss_tokens for batch in batches] == expected_tokens
    assert sampler.cursor == batches[-1].cursor
    assert sampler.consumed_loss_tokens == expected_tokens[-1]
    assert sampler.batches_consumed == 5
    assert batches[-1].batch_identity == batches[-1].batch_identity


def test_drop_last_reports_the_dropped_suffix_and_does_not_count_its_loss_tokens():
    sampler = PackedSequenceSampler(packed_sequences(), batch_size=2, seed=4, shuffle=False, drop_last=True)
    first = sampler.next_batch()
    second = sampler.next_batch()
    assert first.indices == (0, 1)
    assert second.indices == (2, 3)
    assert second.dropped_indices == (4,)
    assert second.cursor == 5
    assert second.consumed_loss_tokens == 10
    assert len(sampler) == 2
    following = sampler.next_batch()
    assert following.epoch == 1 and following.indices == (0, 1)


def test_state_json_round_trip_restores_order_cursor_and_token_accounting():
    data = packed_sequences()
    sampler = PackedSequenceSampler(data, batch_size=2, seed=19)
    sampler.next_batch()
    state = sampler.state_dict()
    json.dumps(state)
    restored = PackedSequenceSampler.from_json(data, sampler.to_json())
    assert restored.state_dict() == state
    expected = [sampler.next_batch() for _ in range(4)]
    actual = [restored.next_batch() for _ in range(4)]
    assert actual == expected
    assert restored.state_dict() == sampler.state_dict()
    assert restored.data_hash == sampler.data_identity
    assert restored.order_hash == sampler.order_identity


def test_resume_refuses_changed_data_or_order_without_changing_counters():
    data = packed_sequences()
    sampler = PackedSequenceSampler(data, batch_size=2, seed=19)
    sampler.next_batch()
    state = sampler.state_dict()

    changed_data = packed_sequences()
    changed_data[0]["input_ids"][0] = 99
    wrong_data = PackedSequenceSampler(changed_data, batch_size=2, seed=19)
    before = wrong_data.state_dict()
    with pytest.raises(SamplerIdentityError, match="data identity"):
        wrong_data.load_state_dict(state)
    assert wrong_data.state_dict() == before

    wrong_order = PackedSequenceSampler(data, batch_size=2, seed=20)
    assert wrong_order.data_identity == sampler.data_identity
    assert wrong_order.order_identity != sampler.order_identity
    before = wrong_order.state_dict()
    with pytest.raises(SamplerIdentityError, match="order identity"):
        wrong_order.load_state_dict(state)
    assert wrong_order.state_dict() == before

    wrong_batch_size = PackedSequenceSampler(data, batch_size=3, seed=19)
    before = wrong_batch_size.state_dict()
    with pytest.raises(SamplerIdentityError, match="order identity|batch_size"):
        wrong_batch_size.load_state_dict(state)
    assert wrong_batch_size.state_dict() == before


def test_malformed_state_is_rejected_without_a_partial_restore():
    sampler = PackedSequenceSampler(packed_sequences(), batch_size=2, seed=5)
    sampler.next_batch()
    before = sampler.state_dict()
    malformed = dict(before, cursor=-1)
    with pytest.raises(SamplerStateError, match="cursor"):
        sampler.load_state_dict(malformed)
    assert sampler.state_dict() == before


def test_labels_or_explicit_counts_support_loss_accounting_and_masks_are_validated():
    sequences = [
        {"labels": torch.tensor([4, -100, 8]), "input_ids": torch.tensor([3, 4, 8])},
        {"input_ids": np.array([1, 2]), "loss_tokens": 1},
    ]
    sampler = PackedSequenceSampler(sequences, batch_size=2, seed=1, shuffle=False)
    assert sampler.next_batch().loss_tokens == 3
    with pytest.raises(ValueError, match="loss_mask"):
        PackedSequenceSampler([{"loss_mask": [0.0, 0.5]}], batch_size=1, seed=1)
    with pytest.raises(ValueError, match="needs loss_mask"):
        PackedSequenceSampler([{"input_ids": [1, 2]}], batch_size=1, seed=1)
