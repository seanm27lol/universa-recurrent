import pytest
import torch

from native.config import NativeConfig
from native.model import NativeModel


def make_model(mode):
    torch.manual_seed(31)
    return NativeModel(
        NativeConfig(
            vocab_size=48,
            context_length=24,
            d_model=16,
            n_heads=4,
            d_slot=8,
            d_ff=32,
            mode=mode,
            n_state_classes=5,
        )
    )


@pytest.mark.parametrize("mode", ["untyped", "typed"])
def test_selected_only_grouped_calls_cover_every_decision(mode):
    model = make_model(mode)
    calls = []
    for expert_index, expert in enumerate(model.experts):
        def record(module, inputs, expert_index=expert_index):
            calls.append((expert_index, inputs[0].shape[0]))
        expert.register_forward_pre_hook(record)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    output = model(ids, return_aux=True)
    cursor = 0
    for pass_index in range(4):
        covered = 0
        for expert_index in range(4):
            expected = (output.route_indices[:, :, pass_index] == expert_index).sum().item()
            covered += expected
            if expected:
                assert calls[cursor] == (expert_index, expected)
                cursor += 1
        assert covered == ids.numel()
    assert cursor == len(calls)
    assert torch.equal(output.route_weights.argmax(dim=-1), output.route_indices)
    assert torch.equal(output.route_weights.sum(dim=-1), torch.ones_like(output.route_indices, dtype=torch.float32))
    assert set(output.route_weights.unique().tolist()) <= {0.0, 1.0}


@pytest.mark.parametrize("mode", ["untyped", "typed"])
def test_isolated_decision_gives_no_gradient_to_unselected_experts(mode):
    model = make_model(mode)
    with torch.no_grad():
        model.router.weight.zero_()
        model.router.bias.copy_(torch.tensor([4.0, 0.0, 0.0, 0.0]))
    output = model(torch.tensor([[1, 2]]), return_aux=True)
    assert output.route_indices.eq(0).all()
    output.logits.square().mean().backward()
    assert model.experts[0].up.weight.grad.abs().sum() > 0
    for expert in list(model.experts)[1:]:
        for parameter in expert.parameters():
            assert parameter.grad is None or parameter.grad.abs().sum() == 0


@pytest.mark.parametrize("mode", ["untyped", "typed"])
def test_unlabeled_task_and_balance_signals_reach_router(mode):
    model = make_model(mode)
    output = model(torch.tensor([[1, 2, 3]]), return_aux=True)
    stats = output.load_stats
    assert stats["selected_fraction"].grad_fn is None
    assert torch.isclose(stats["selected_fraction"].sum(), torch.tensor(1.0))
    assert torch.isclose(stats["mean_probability"].sum(), torch.tensor(1.0))
    balance = 4 * (stats["selected_fraction"] * stats["mean_probability"]).sum()
    task = output.logits.square().mean()
    (task + balance).backward()
    assert model.router.weight.grad is not None
    assert torch.isfinite(model.router.weight.grad).all()
    assert model.router.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["untyped", "typed"])
def test_router_softmax_is_float32_under_cpu_autocast(mode):
    model = make_model(mode)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(torch.tensor([[1, 2]]), return_aux=True)
    assert output.route_probs.dtype == torch.float32
    assert output.route_weights.dtype == torch.float32
    output.logits.float().square().mean().backward()
    assert torch.isfinite(model.router.weight.grad).all()


@pytest.mark.parametrize("mode", ["untyped", "typed"])
def test_router_ties_choose_lowest_expert(mode):
    model = make_model(mode)
    with torch.no_grad():
        model.router.weight.zero_()
        model.router.bias.zero_()
    output = model(torch.tensor([[1, 2, 3]]), return_aux=True)
    assert output.route_indices.eq(0).all()
    assert output.route_probs.eq(0.25).all()


def test_router_inputs_follow_generic_and_typed_contract():
    seen = []
    for mode, expected_width in (("untyped", 16), ("typed", 32)):
        model = make_model(mode)
        model.router.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].shape[-1]))
        model(torch.tensor([[1, 2]]))
        assert seen[-1] == expected_width
    assert seen == [16, 16, 16, 16, 32, 32, 32, 32]
