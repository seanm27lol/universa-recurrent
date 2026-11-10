"""CPU checks for causal masks and standalone/packed loss alignment."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
from native import diagnose_loss_gradients as D  # noqa: E402
from native.config import NativeConfig  # noqa: E402
from native.model import NativeModel  # noqa: E402


def test_shifted_masks_exclude_document_starts_and_padding():
    labels = torch.tensor([[4, 5, 6, -100, 8, -100, -100]])
    answer_targets = torch.tensor([[False, False, True, False, True, True, True]])
    padding = torch.tensor([[False, False, False, False, False, True, True]])

    full, answer = D.shifted_masks(labels, answer_targets, padding)

    assert full.tolist() == [[True, True, False, True, False, False]]
    assert answer.tolist() == [[False, True, False, True, False, False]]


def test_masked_ce_uses_preceding_logits_and_ignores_unselected_positions():
    torch.manual_seed(3)
    logits = torch.randn(1, 7, 12)
    labels = torch.tensor([[4, 5, 6, -100, 8, -100, -100]])
    answer_targets = torch.tensor([[False, False, True, False, True, True, True]])
    padding = torch.tensor([[False, False, False, False, False, True, True]])
    full, answer = D.shifted_masks(labels, answer_targets, padding)

    expected_full = F.cross_entropy(
        logits[0, torch.tensor([0, 1, 3])].float(), labels[0, torch.tensor([1, 2, 4])], reduction="sum",
    )
    expected_answer = F.cross_entropy(
        logits[0, torch.tensor([1, 3])].float(), labels[0, torch.tensor([2, 4])], reduction="sum",
    )
    assert D.masked_ce_sum(logits[:, :-1], labels[:, 1:], full) == pytest.approx(expected_full)
    assert D.masked_ce_sum(logits[:, :-1], labels[:, 1:], answer) == pytest.approx(expected_answer)

    changed = logits.clone()
    changed[0, torch.tensor([0, 2, 4, 5, 6])] = 1000.0
    assert D.masked_ce_sum(changed[:, :-1], labels[:, 1:], answer) == pytest.approx(expected_answer)


def _example(example_id, doc_id, start, token_ids, prompt_tokens):
    return D.Example(
        example_id=example_id,
        doc_id=doc_id,
        row=0,
        start=start,
        end=start + len(token_ids),
        actual_starting_position=start,
        prompt_tokens=prompt_tokens,
        answer_tokens=len(token_ids) - prompt_tokens,
        token_ids=tuple(token_ids),
    )


def test_zero_based_standalone_matches_document_isolated_packed_rows():
    torch.manual_seed(7)
    model = NativeModel(NativeConfig(
        vocab_size=16, context_length=16, d_model=8, n_heads=2, d_slot=4, d_ff=16, mode="dense",
    )).cpu().eval()
    examples = [
        _example("first", 0, 0, (2, 3, 4, 5), 2),
        _example("second", 1, 4, (6, 7, 8), 1),
    ]
    zero, zero_examples = D._position_pass(model, D._zero_units(examples), examples, 5, 4)

    input_ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 0, 0, 0]])
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 0, 0, 0]])
    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, -1, -1, -1]])
    padding = torch.tensor([[False, False, False, False, False, False, False, True, True, True]])
    labels = input_ids.clone()
    labels[0, 0] = -100
    labels[0, 4] = -100
    labels[padding] = -100
    packed_units = [(
        {"input_ids": input_ids, "positions": positions, "doc_ids": doc_ids, "key_padding_mask": padding},
        labels,
        [D.UnitExample(examples[0], 0, 4), D.UnitExample(examples[1], 4, 7)],
    )]
    packed, packed_examples = D._position_pass(model, packed_units, examples, 5, 4)

    for objective in ("full_lm", "answer"):
        assert zero["losses"][objective]["count"] == packed["losses"][objective]["count"]
        assert zero["losses"][objective]["sum"] == pytest.approx(
            packed["losses"][objective]["sum"], rel=2e-6, abs=2e-6,
        )
    for example in examples:
        identity = example.example_id
        for objective in ("full_lm", "answer"):
            zero_loss = zero_examples[identity][objective]
            packed_loss = packed_examples[identity][objective]
            assert zero_loss["count"] == packed_loss["count"]
            assert zero_loss["sum"] == pytest.approx(packed_loss["sum"], rel=2e-6, abs=2e-6)
        assert zero_examples[identity]["full_lm"]["count"] == example.token_count - 1
        assert zero_examples[identity]["answer"]["count"] == example.answer_tokens


@pytest.mark.parametrize("mode", ["dense", "shared", "untyped", "typed"])
def test_parameter_groups_cover_every_trainable_parameter_once(mode):
    model = NativeModel(NativeConfig(
        vocab_size=16, context_length=16, d_model=8, n_heads=2, d_slot=4, d_ff=16, mode=mode,
    ))
    groups = D.parameter_groups(model)
    grouped = [name for values in groups.values() for name, _ in values]

    assert set(groups) == set(D.GROUP_NAMES)
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert {name for name, _ in groups["position_embedding"]} == {"position_embedding.weight"}
    assert {name for name, _ in groups["token_embedding"]} == {"token_embedding.weight"}
    assert "output_norm.weight" in {name for name, _ in groups["output_head"]}
    assert "output_norm.bias" in {name for name, _ in groups["output_head"]}


def test_gradient_accumulator_sums_vectors_before_norms_and_cosines():
    model = NativeModel(NativeConfig(
        vocab_size=16, context_length=16, d_model=8, n_heads=2, d_slot=4, d_ff=16, mode="dense",
    ))
    parameter = model.position_embedding.weight
    accumulator = D.GradientAccumulator(model)
    accumulator.update(parameter.sum(), parameter.sum())
    accumulator.update(parameter.sum(), -parameter.sum())
    metrics = accumulator.summary()["position_embedding"]
    assert metrics["answer_only_gradient_l2"] == pytest.approx(2 * parameter.numel() ** 0.5)
    assert metrics["full_lm_gradient_l2"] == 0.0
    assert metrics["answer_only_full_lm_cosine"] is None
    assert parameter.grad is None
