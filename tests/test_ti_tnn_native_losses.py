"""Tiny CPU construction checks for native losses; these are not learning evidence."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from native import losses  # noqa: E402


def test_shifted_language_loss_excludes_document_starts_padding_and_final_logits():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 4)
    logits[0, 0] = torch.tensor([10000.0, -10000.0, 3.0, -2.0])
    labels = torch.tensor([[-100, 2, -100, 3], [-100, -100, 1, -100]])
    loss, count = losses.next_token_cross_entropy(logits, labels)
    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    expected = F.cross_entropy(
        shifted_logits.reshape(-1, 4), shifted_labels.reshape(-1), ignore_index=-100, reduction="mean"
    )
    assert loss.dtype == torch.float32
    assert count.dtype == torch.int64
    assert count.item() == shifted_labels.ne(-100).sum().item() == 3
    assert torch.isfinite(loss) and torch.allclose(loss, expected)

    changed = logits.clone()
    changed[:, -1] = 1e20
    changed[:, :-1][shifted_labels.eq(-100)] = 1e20
    changed_loss, changed_count = losses.next_token_cross_entropy(changed, labels)
    assert changed_count.item() == 3
    assert torch.allclose(changed_loss, loss)


@pytest.mark.parametrize("length", [0, 1])
def test_language_loss_rejects_sequences_too_short_to_shift(length):
    logits = torch.randn(1, length, 5)
    labels = torch.full((1, length), -100)
    with pytest.raises(ValueError, match="T >= 2"):
        losses.next_token_cross_entropy(logits, labels)


def test_all_ignored_shifted_language_labels_return_differentiable_zero():
    logits = torch.randn(1, 3, 5, requires_grad=True)
    loss, count = losses.next_token_cross_entropy(logits, torch.full((1, 3), -100))
    assert count.item() == 0
    assert loss.item() == 0.0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == 0


def test_unlabelled_text_has_active_switch_balance_and_router_gradient():
    torch.manual_seed(1)
    language_logits = torch.randn(2, 2, 7, requires_grad=True)
    route_logits = torch.randn(2, 2, 4, 4, requires_grad=True)
    route_indices = (torch.arange(16).reshape(2, 2, 4) + 1) % 4
    result = losses.compute_losses(
        language_logits,
        torch.full((2, 2), -100),
        route_logits=route_logits,
        route_indices=route_indices,
    )

    probabilities = torch.softmax(route_logits.detach().float(), dim=-1)
    fractions = F.one_hot(route_indices, num_classes=4).float().mean(dim=(0, 1, 2))
    expected = 4 * (fractions * probabilities.mean(dim=(0, 1, 2))).sum()
    assert result["balance"].dtype == torch.float32
    assert torch.allclose(result["balance"], expected)
    assert torch.allclose(result["total"], result["balance"])
    assert result["route"].item() == 0.0 and result["state"].item() == 0.0
    assert result["token_count"].item() == 0
    assert result["route_count"].item() == 0 and result["state_count"].item() == 0
    assert result["balance_count"].item() == 16

    result["total"].backward()
    assert torch.isfinite(route_logits.grad).all()
    assert route_logits.grad.abs().sum() > 0


def test_right_padding_preserves_live_balance_counts_and_gradients():
    torch.manual_seed(6)
    compact_route_logits = torch.randn(2, 2, 4, 4, requires_grad=True)
    compact_route_indices = (torch.arange(16).reshape(2, 2, 4) + 2) % 4
    padded_route_logits = torch.randn(2, 4, 4, 4)
    padded_route_logits[:, :2] = compact_route_logits.detach()
    padded_route_logits[:, 2:] = 100 * torch.randn(2, 2, 4, 4)
    padded_route_logits.requires_grad_()
    padded_route_indices = torch.zeros(2, 4, 4, dtype=torch.long)
    padded_route_indices[:, :2] = compact_route_indices
    padded_route_indices[:, 2:] = 3
    key_padding_mask = torch.tensor([[False, False, True, True], [False, False, True, True]])

    compact_balance = losses.switch_load_balance(compact_route_logits, compact_route_indices)
    padded_balance = losses.switch_load_balance(
        padded_route_logits, padded_route_indices, key_padding_mask
    )
    assert compact_balance.dtype == padded_balance.dtype == torch.float32
    assert torch.allclose(padded_balance, compact_balance)

    compact_language_logits = torch.randn(2, 2, 6)
    padded_language_logits = torch.randn(2, 4, 6)
    padded_language_logits[:, :2] = compact_language_logits
    compact_labels = torch.tensor([[-100, 1], [-100, 2]])
    padded_labels = torch.tensor([[-100, 1, -100, -100], [-100, 2, -100, -100]])
    compact_result = losses.compute_losses(
        compact_language_logits,
        compact_labels,
        route_logits=compact_route_logits,
        route_indices=compact_route_indices,
    )
    padded_result = losses.compute_losses(
        padded_language_logits,
        padded_labels,
        key_padding_mask=key_padding_mask,
        route_logits=padded_route_logits,
        route_indices=padded_route_indices,
    )
    unmasked_result = losses.compute_losses(
        padded_language_logits,
        padded_labels,
        route_logits=padded_route_logits,
        route_indices=padded_route_indices,
    )
    assert torch.allclose(padded_result["balance"], compact_result["balance"])
    assert torch.allclose(padded_result["language"], compact_result["language"])
    assert not torch.allclose(unmasked_result["balance"], compact_result["balance"])
    assert compact_result["balance_count"].item() == padded_result["balance_count"].item() == 16
    assert unmasked_result["balance_count"].item() == 32
    assert compact_result["token_count"].item() == padded_result["token_count"].item() == 2
    assert padded_result["route_count"].item() == 0 and padded_result["state_count"].item() == 0

    compact_balance.backward()
    padded_balance.backward()
    assert torch.isfinite(compact_route_logits.grad).all()
    assert compact_route_logits.grad.abs().sum() > 0
    assert torch.isfinite(padded_route_logits.grad).all()
    assert torch.count_nonzero(padded_route_logits.grad[:, 2:]) == 0
    assert torch.allclose(padded_route_logits.grad[:, :2], compact_route_logits.grad)


def test_all_padded_balance_is_finite_zero_with_zero_count_and_gradients():
    torch.manual_seed(7)
    language_logits = torch.randn(1, 3, 5, requires_grad=True)
    route_logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    route_indices = torch.zeros(1, 3, 4, dtype=torch.long)
    key_padding_mask = torch.ones(1, 3, dtype=torch.bool)
    balance = losses.switch_load_balance(route_logits, route_indices, key_padding_mask)
    assert balance.dtype == torch.float32
    assert balance.item() == 0.0 and torch.isfinite(balance)
    balance.backward()
    assert torch.isfinite(route_logits.grad).all()
    assert torch.count_nonzero(route_logits.grad) == 0

    result = losses.compute_losses(
        language_logits,
        torch.full((1, 3), -100),
        key_padding_mask=key_padding_mask,
        route_logits=route_logits,
        route_indices=route_indices,
    )
    assert result["balance"].item() == 0.0 and torch.isfinite(result["balance"])
    assert result["total"].item() == 0.0 and torch.isfinite(result["total"])
    assert result["balance_count"].item() == 0
    assert result["token_count"].item() == 0
    assert result["route_count"].item() == 0 and result["state_count"].item() == 0
    result["total"].backward()
    assert torch.isfinite(language_logits.grad).all() and torch.count_nonzero(language_logits.grad) == 0
    assert torch.isfinite(route_logits.grad).all() and torch.count_nonzero(route_logits.grad) == 0
    with pytest.raises(ValueError, match="key padding mask"):
        losses.switch_load_balance(route_logits, route_indices, torch.ones(1, 2, dtype=torch.bool))


def test_unbatched_auxiliaries_stay_unshifted_and_optional_tensors_are_unchanged():
    torch.manual_seed(2)
    language_logits = torch.randn(2, 2, 6, requires_grad=True)
    labels = torch.tensor([[0, 1], [2, -100]])
    route_logits = torch.randn(2, 2, 4, 4, requires_grad=True)
    route_indices = route_logits.detach().argmax(dim=-1)
    state_logits = torch.randn(2, 2, 4, 4, 5, requires_grad=True)
    route_targets = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    route_mask = torch.tensor([[True, False, True, False], [False, True, False, False]])
    state_targets = torch.arange(2 * 4 * 4).reshape(2, 4, 4) % 5
    state_mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    state_mask[0, 0, 0] = True
    state_mask[0, 2, 3] = True
    state_mask[1, 1, 2] = True
    route_targets = route_targets.masked_fill(~route_mask, 12345)
    state_targets = state_targets.masked_fill(~state_mask, 12345)
    optional_before = tuple(t.clone() for t in (route_targets, route_mask, state_targets, state_mask))

    result = losses.compute_losses(
        language_logits,
        labels,
        route_logits=route_logits,
        route_indices=route_indices,
        route_targets=route_targets,
        route_mask=route_mask,
        state_logits=state_logits,
        state_targets=state_targets,
        state_mask=state_mask,
        lambda_balance=0.25,
        lambda_route=0.5,
        lambda_state=2.0,
    )
    expanded_route_targets = route_targets.unsqueeze(0).expand(2, -1, -1)
    expanded_route_mask = route_mask.unsqueeze(0).expand(2, -1, -1)
    expected_route = F.cross_entropy(
        route_logits[expanded_route_mask], expanded_route_targets[expanded_route_mask]
    )
    expanded_state_targets = state_targets.unsqueeze(0).expand(2, -1, -1, -1)
    expanded_state_mask = state_mask.unsqueeze(0).expand(2, -1, -1, -1)
    expected_state = F.cross_entropy(
        state_logits[expanded_state_mask], expanded_state_targets[expanded_state_mask]
    )
    direct_language, direct_count = losses.next_token_cross_entropy(language_logits, labels)
    assert torch.allclose(result["language"], direct_language)
    assert torch.allclose(result["route"], expected_route)
    assert torch.allclose(result["state"], expected_state)
    assert result["token_count"].item() == direct_count.item() == 1
    assert result["route_count"].item() == 2 * route_mask.sum().item()
    assert result["state_count"].item() == 2 * state_mask.sum().item()
    expected_total = (
        result["language"]
        + 0.25 * result["balance"]
        + 0.5 * result["route"]
        + 2.0 * result["state"]
    )
    assert torch.allclose(result["total"], expected_total)
    for tensor, before in zip((route_targets, route_mask, state_targets, state_mask), optional_before):
        assert torch.equal(tensor, before)

    result["total"].backward()
    assert torch.isfinite(route_logits.grad).all() and route_logits.grad.abs().sum() > 0
    assert torch.isfinite(state_logits.grad).all() and state_logits.grad.abs().sum() > 0


def test_batched_auxiliaries_equal_count_weighted_per_example_calls():
    torch.manual_seed(3)
    route_logits = torch.randn(2, 3, 4, 4)
    state_logits = torch.randn(2, 3, 4, 4, 5)
    route_targets = torch.randint(0, 4, (2, 3, 4))
    state_targets = torch.randint(0, 5, (2, 3, 4, 4))
    route_mask = torch.tensor(
        [
            [[True, False, False, False], [False, True, False, False], [False] * 4],
            [[False] * 4, [True, True, False, False], [False, False, True, False]],
        ]
    )
    state_mask = torch.zeros(2, 3, 4, 4, dtype=torch.bool)
    state_mask[0, 0, 0, 0] = True
    state_mask[0, 2, 3, 2] = True
    state_mask[1, 1, 2, 1] = True

    batched_route, batched_route_count = losses.masked_route_cross_entropy(
        route_logits, route_targets, route_mask
    )
    batched_state, batched_state_count = losses.masked_state_cross_entropy(
        state_logits, state_targets, state_mask
    )
    per_route = [
        losses.masked_route_cross_entropy(route_logits[i : i + 1], route_targets[i], route_mask[i])
        for i in range(2)
    ]
    per_state = [
        losses.masked_state_cross_entropy(state_logits[i : i + 1], state_targets[i], state_mask[i])
        for i in range(2)
    ]
    assert batched_route_count.item() == sum(count.item() for _, count in per_route) == 5
    assert batched_state_count.item() == sum(count.item() for _, count in per_state) == 3
    expected_route = sum(loss * count for loss, count in per_route) / batched_route_count
    expected_state = sum(loss * count for loss, count in per_state) / batched_state_count
    assert torch.allclose(batched_route, expected_route)
    assert torch.allclose(batched_state, expected_state)


def test_auxiliary_shapes_are_strictly_validated():
    route_logits = torch.randn(2, 3, 4, 4)
    invalid_route = [
        ((1, 3, 4), (1, 3, 4)),
        ((2, 2, 4), (2, 2, 4)),
        ((2, 3, 4), (3, 4)),
        ((3, 4, 1), (3, 4, 1)),
        ((2, 3, 4, 1), (2, 3, 4, 1)),
    ]
    for target_shape, mask_shape in invalid_route:
        targets = torch.zeros(target_shape, dtype=torch.long)
        mask = torch.zeros(mask_shape, dtype=torch.bool)
        with pytest.raises(ValueError, match="route targets and mask"):
            losses.masked_route_cross_entropy(route_logits, targets, mask)

    state_logits = torch.randn(2, 3, 4, 4, 5)
    invalid_state = [
        ((1, 3, 4, 4), (1, 3, 4, 4)),
        ((2, 2, 4, 4), (2, 2, 4, 4)),
        ((2, 3, 4, 4), (3, 4, 4)),
        ((3, 4, 4, 1), (3, 4, 4, 1)),
        ((2, 3, 4, 4, 1), (2, 3, 4, 4, 1)),
    ]
    for target_shape, mask_shape in invalid_state:
        targets = torch.zeros(target_shape, dtype=torch.long)
        mask = torch.zeros(mask_shape, dtype=torch.bool)
        with pytest.raises(ValueError, match="state targets and mask"):
            losses.masked_state_cross_entropy(state_logits, targets, mask)


def test_all_masked_auxiliaries_are_finite_zero_with_zero_gradients():
    torch.manual_seed(4)
    language_logits = torch.randn(1, 2, 5, requires_grad=True)
    route_logits = torch.randn(1, 2, 4, 4, requires_grad=True)
    state_logits = torch.randn(1, 2, 4, 4, 3, requires_grad=True)
    result = losses.compute_losses(
        language_logits,
        torch.full((1, 2), -100),
        route_logits=route_logits,
        route_indices=torch.zeros(1, 2, 4, dtype=torch.long),
        route_targets=torch.zeros(1, 2, 4, dtype=torch.long),
        route_mask=torch.zeros(1, 2, 4, dtype=torch.bool),
        state_logits=state_logits,
        state_targets=torch.zeros(1, 2, 4, 4, dtype=torch.long),
        state_mask=torch.zeros(1, 2, 4, 4, dtype=torch.bool),
        lambda_balance=0.0,
    )
    for name in ("total", "language", "route", "state"):
        assert torch.isfinite(result[name]) and result[name].item() == 0.0
    assert torch.isfinite(result["balance"])
    assert result["route_count"].item() == 0 and result["state_count"].item() == 0

    result["total"].backward()
    for tensor in (language_logits, route_logits, state_logits):
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) == 0


def test_all_loss_terms_are_float32_under_cpu_autocast():
    torch.manual_seed(5)
    language_logits = torch.randn(1, 2, 6, requires_grad=True)
    route_logits = torch.randn(1, 2, 4, 4, requires_grad=True)
    state_logits = torch.randn(1, 2, 4, 4, 3, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = losses.compute_losses(
            language_logits.bfloat16(),
            torch.tensor([[0, 2]]),
            route_logits=route_logits.bfloat16(),
            route_indices=torch.tensor([[[0, 1, 2, 3], [3, 2, 1, 0]]]),
            route_targets=torch.tensor([[[0, 1, 2, 3], [3, 2, 1, 0]]]),
            route_mask=torch.ones(1, 2, 4, dtype=torch.bool),
            state_logits=state_logits.bfloat16(),
            state_targets=torch.zeros(1, 2, 4, 4, dtype=torch.long),
            state_mask=torch.ones(1, 2, 4, 4, dtype=torch.bool),
        )
    for name in ("total", "language", "route", "state", "balance"):
        assert result[name].dtype == torch.float32
        assert torch.isfinite(result[name])
    result["total"].backward()
    for tensor in (language_logits, route_logits, state_logits):
        assert torch.isfinite(tensor.grad).all()
