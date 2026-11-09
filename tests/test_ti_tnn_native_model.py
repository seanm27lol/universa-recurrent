from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from native.config import NativeConfig
from native.cost import (
    active_parameter_bounds,
    active_parameter_report,
    active_parameters,
    count_model_parameters,
    counts_for_all_modes,
    dense_transform_parameters,
    executed_macs,
    mac_breakdown,
    nominal_one_expert_active_parameters,
    parameter_counts,
    route_active_parameters,
    token_macs,
    total_parameters,
)
from native.model import NativeModel


MODES = ("dense", "shared", "untyped", "typed")


def tiny_config(mode="dense"):
    return NativeConfig(
        vocab_size=64,
        context_length=32,
        d_model=16,
        n_heads=4,
        d_slot=8,
        d_ff=32,
        mode=mode,
        n_state_classes=5,
    )


def make_model(mode="dense", seed=17):
    torch.manual_seed(seed)
    return NativeModel(tiny_config(mode)).eval()


@pytest.mark.parametrize("mode", MODES)
def test_shape_pass_cache_and_probe_contract(mode):
    model = make_model(mode)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    output = model(ids, use_cache=True, return_aux=True, return_trace=True)
    assert output.logits.shape == (2, 3, 64)
    assert output.state_logits.shape == (2, 3, 4, 4, 5)
    assert output.h.shape == (2, 3, 4, 16)
    assert output.slots.shape == (2, 3, 4, 4, 8)
    assert len(output.cache.keys) == len(output.cache.values) == 4
    assert all(key.shape == (2, 4, 3, 4) for key in output.cache.keys)
    assert model.token_embedding.weight.data_ptr() != model.output_head.weight.data_ptr()
    assert len(model.state_probes) == 4
    if mode in ("untyped", "typed"):
        assert output.route_logits.shape == (2, 3, 4, 4)
        assert output.route_indices.shape == (2, 3, 4)
    else:
        assert output.route_logits is None
        assert output.route_indices is None


def test_core_weight_sharing_and_typed_interfaces():
    dense = make_model("dense")
    assert len(dense.attentions) == len(dense.transforms) == 4
    assert len({id(cell) for cell in dense.attentions}) == 4
    assert len({id(cell) for cell in dense.transforms}) == 4
    for mode in ("shared", "untyped", "typed"):
        model = make_model(mode)
        assert all(model._attention(pass_index) is model.attention for pass_index in range(4))
    assert hasattr(make_model("shared"), "transform")
    for mode in ("untyped", "typed"):
        assert len(make_model(mode).experts) == 4
    typed = make_model("typed")
    assert len({parameter.data_ptr() for parameter in typed.slot_readers.parameters()}) == 4
    assert typed.router.in_features == 32
    assert make_model("untyped").router.in_features == 16


def test_corresponding_random_initial_draws_match():
    models = {mode: make_model(mode, seed=23) for mode in MODES}
    reference = models["dense"]
    for model in models.values():
        for name in ("token_embedding", "position_embedding", "output_head"):
            assert torch.equal(getattr(reference, name).weight, getattr(model, name).weight)
        assert torch.equal(reference.slot_initializers, model.slot_initializers)
        for reference_probe, probe in zip(reference.state_probes, model.state_probes):
            assert torch.equal(reference_probe.weight, probe.weight)
        first_attention = reference.attentions[0]
        model_attention = model.attentions[0] if model.config.mode == "dense" else model.attention
        assert torch.equal(first_attention.qkv.weight, model_attention.qkv.weight)
        reference_transform = reference.transforms[0]
        if model.config.mode == "dense":
            model_transform = model.transforms[0]
        elif model.config.mode == "shared":
            model_transform = model.transform
        else:
            model_transform = model.experts[0]
        assert torch.equal(reference_transform.up.weight, model_transform.up.weight)
        assert torch.equal(reference.h_reader.weight, model.h_reader.weight)
    for untyped_expert, typed_expert in zip(models["untyped"].experts, models["typed"].experts):
        assert torch.equal(untyped_expert.up.weight, typed_expert.up.weight)
        assert torch.equal(untyped_expert.down.weight, typed_expert.down.weight)


@pytest.mark.parametrize("mode", MODES)
def test_causal_prefix_invariance_through_all_passes(mode):
    model = make_model(mode)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        full = model(ids, return_aux=True, return_trace=True)
        prefix = model(ids[:, :3], return_aux=True, return_trace=True)
    for name in ("logits", "state_logits", "h", "slots", "route_logits"):
        full_value = getattr(full, name)
        if full_value is not None:
            assert torch.allclose(full_value[:, :3], getattr(prefix, name), atol=2e-6, rtol=2e-6)
    if full.route_indices is not None:
        assert torch.equal(full.route_indices[:, :3], prefix.route_indices)


@pytest.mark.parametrize("mode", MODES)
def test_cross_example_isolation(mode):
    model = make_model(mode)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    with torch.no_grad():
        batch = model(ids, return_trace=True)
        singles = [model(ids[index : index + 1], return_trace=True) for index in range(2)]
    for index, single in enumerate(singles):
        assert torch.allclose(batch.logits[index : index + 1], single.logits, atol=2e-6, rtol=2e-6)
        assert torch.allclose(batch.h[index : index + 1], single.h, atol=2e-6, rtol=2e-6)
        assert torch.allclose(batch.slots[index : index + 1], single.slots, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("mode", MODES)
def test_block_diagonal_document_mask(mode):
    model = make_model(mode)
    packed_ids = torch.tensor([[1, 2, 3, 4]])
    packed_positions = torch.tensor([[0, 1, 0, 1]])
    packed_docs = torch.tensor([[7, 7, 9, 9]])
    with torch.no_grad():
        packed = model(
            packed_ids,
            positions=packed_positions,
            doc_ids=packed_docs,
            return_aux=True,
            return_trace=True,
        )
        isolated = model(
            packed_ids[:, 2:],
            positions=packed_positions[:, 2:],
            doc_ids=torch.ones_like(packed_ids[:, 2:]),
            return_aux=True,
            return_trace=True,
        )
    for name in ("logits", "state_logits", "h", "slots", "route_logits"):
        packed_value = getattr(packed, name)
        if packed_value is not None:
            assert torch.allclose(packed_value[:, 2:], getattr(isolated, name), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("mode", MODES)
def test_padding_cannot_change_live_tokens(mode):
    model = make_model(mode)
    ids = torch.tensor([[2, 3, 4]])
    padded_ids = torch.tensor([[0, 2, 3, 4, 0]])
    mask = torch.tensor([[True, False, False, False, True]])
    positions = torch.tensor([[0, 0, 1, 2, 0]])
    with torch.no_grad():
        plain = model(ids, return_aux=True, return_trace=True)
        padded = model(
            padded_ids,
            positions=positions,
            key_padding_mask=mask,
            return_aux=True,
            return_trace=True,
        )
    for name in ("logits", "state_logits", "h", "slots", "route_logits"):
        value = getattr(plain, name)
        if value is not None:
            assert torch.allclose(value, getattr(padded, name)[:, 1:4], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("mode", MODES)
def test_external_lm_and_state_losses_deliver_gradients(mode):
    model = make_model(mode)
    model.train()
    output = model(torch.tensor([[1, 2, 3]]), return_aux=True)
    loss = output.logits.square().mean() + output.state_logits.square().mean()
    loss.backward()
    for parameter in (
        model.token_embedding.weight,
        model.output_head.weight,
        model.slot_initializers,
        model.h_reader.weight,
        model.h_writer.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert all(probe.weight.grad.abs().sum() > 0 for probe in model.state_probes)
    if mode in ("untyped", "typed"):
        assert model.router.weight.grad is not None
        assert torch.isfinite(model.router.weight.grad).all()


def test_structured_slot_intervention_and_direct_h_bypass():
    model = make_model("typed")
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        baseline = model(ids).logits
    calls = 0

    def zero_second_pass(module, inputs, output):
        nonlocal calls
        calls += 1
        if calls == 2:
            return torch.zeros_like(output)
        return output

    handle = model.slot_readers[1].register_forward_hook(zero_second_pass)
    with torch.no_grad():
        intervened = model(ids).logits
    handle.remove()
    assert calls == 4
    assert not torch.allclose(baseline, intervened)

    with torch.no_grad():
        model.slot_initializers.zero_()
        model.slot_codes.zero_()
        for module in (*model.slot_readers, *model.slot_writers):
            module.weight.zero_()
    blocked = model(ids, return_trace=True)
    with torch.no_grad():
        other = model(torch.tensor([[4, 5, 6]])).logits
    assert blocked.slots.abs().sum() == 0
    assert blocked.h.abs().sum() > 0
    assert blocked.logits.abs().sum() > 0
    assert not torch.allclose(blocked.logits, other)
    blocked.logits.square().mean().backward()
    assert model.token_embedding.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", MODES)
def test_parameter_and_mac_formulas_match_modules(mode):
    config = tiny_config(mode)
    model = make_model(mode)
    counts = parameter_counts(config)
    assert count_model_parameters(model) == counts["total"] == total_parameters(config)
    nominal = counts["total"]
    if mode in ("untyped", "typed"):
        nominal -= 3 * dense_transform_parameters(config)
    assert active_parameters(config) == counts["active"] == nominal
    assert nominal_one_expert_active_parameters(config) == counts["nominal_one_expert_active"] == nominal
    assert counts["active_min"] == nominal
    assert counts["active_max"] == counts["total"]
    assert active_parameter_bounds(config) == {
        "active_min": nominal,
        "active_max": counts["total"],
    }

    d = config.d_model
    d_s = config.d_slot
    expected = 4 * d * d_s + 4 * (4 * d * d + 2 * d)
    expected += 4 * 3 * d * config.d_ff + 4 * 2 * d * d
    expected += 4 * 12 * d * d_s + d * config.vocab_size
    if mode == "untyped":
        expected += 4 * 4 * d
    elif mode == "typed":
        expected += 4 * 4 * 2 * d
    assert token_macs(config, 1) == expected == executed_macs(config)
    assert mac_breakdown(config)["slot_readers_writers"] == 4 * 12 * d * d_s
    assert token_macs(config, 2) - token_macs(config, 1) == 4 * 2 * d
    assert token_macs(config, 1, include_probes=True) - token_macs(config, 1) == 16 * d_s * 5
    full = executed_macs(config, sequence_length=3, cache_length=2)
    incremental = executed_macs(config, sequence_length=3, cache_length=2, incremental=True)
    assert full - incremental == 4 * 2 * d * (15 - 12)


@pytest.mark.parametrize("mode", ["untyped", "typed"])
def test_route_union_active_parameters_and_report(mode):
    config = tiny_config(mode)
    counts = parameter_counts(config)
    expert = dense_transform_parameters(config)
    non_expert = counts["total"] - 4 * expert
    routes = torch.tensor([[[0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 2, 2], [0, 1, 2, 3]]])
    actual = route_active_parameters(config, routes)
    expected = non_expert + torch.tensor([[1, 2, 3, 4]]) * expert
    assert torch.equal(actual, expected)
    assert actual[0, 0].item() == counts["active_min"]
    assert actual[0, -1].item() == counts["active_max"]
    report = active_parameter_report(config, routes)
    assert report["nominal_one_expert_active"] == counts["active"]
    assert report["observed_active_min"] == counts["active_min"]
    assert report["observed_active_max"] == counts["active_max"]
    assert report["observed_active_mean"] == expected.float().mean().item()
    assert report["observed_token_count"] == 4


EXPECTED_CONFIG_MACS = {
    "engineering.json": {
        "dense": (13176000, 13200576),
        "shared": (13176000, 13200576),
        "untyped": (13179072, 13203648),
        "typed": (13182144, 13206720),
    },
    "serious.json": {
        "dense": (102356800, 102463296),
        "shared": (102356800, 102463296),
        "untyped": (102370112, 102476608),
        "typed": (102383424, 102489920),
    },
}


@pytest.mark.parametrize("filename,minimum,maximum", [
    ("engineering.json", 20_000_000, 40_000_000),
    ("serious.json", 120_000_000, 130_000_000),
])
def test_shipped_config_counts(filename, minimum, maximum):
    path = Path(__file__).parents[1] / "research" / "ti_tnn" / "native" / "configs" / filename
    values = json.loads(path.read_text())
    config = NativeConfig.from_json(path)
    expected = counts_for_all_modes(config)
    assert values["modes"] == expected
    for mode, expected_macs in EXPECTED_CONFIG_MACS[filename].items():
        actual_macs = expected[mode]
        assert (actual_macs["decoded_token_macs"], actual_macs["decoded_token_macs_with_probes"]) == expected_macs
        assert actual_macs["active"] == actual_macs["nominal_one_expert_active"]
        assert actual_macs["active_min"] <= actual_macs["active_max"] == actual_macs["total"]
    assert "nominal_one_expert_active" in values["active_parameter_rule"]
    assert "union" in values["active_parameter_rule"]
    assert "four slot readers and writers" in values["mac_rule"]
    assert minimum <= expected["typed"]["total"] <= maximum
    assert config.d_ff == 4 * config.d_model
    assert config.d_slot == config.d_model // 4
    assert replace(config, mode="typed").n_passes == 4
