from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from universa_recurrent.neural.baselines import (
    fit_each_structure,
    gaussian_generator_reference,
)
from universa_recurrent.neural.data import StructuredFlowDataset, candidate_bases
from universa_recurrent.neural.lingua import neural_record
from universa_recurrent.neural.model import (
    NeuralConfig,
    StructuredRecurrentNet,
    future_regret_targets,
    training_loss,
)
from universa_recurrent.neural.train import (
    load_checkpoint,
    train_checkpoint,
)
from universa_recurrent.neural.verification import verify_neural_record


def _batch(dataset: StructuredFlowDataset, n: int) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([dataset[index][key] for index in range(n)])
        for key in ("observed", "mask", "truth", "label")
    }


def _model(*, steps: int = 4) -> StructuredRecurrentNet:
    _, bases, _ = candidate_bases()
    return StructuredRecurrentNet(
        torch.from_numpy(bases),
        NeuralConfig(5, 2, 2, 24, steps),
    )


def test_dataset_is_deterministic_and_masks_are_nonempty():
    first = StructuredFlowDataset(32, seed=123)
    second = StructuredFlowDataset(32, seed=123)
    assert torch.equal(first.observed, second.observed)
    assert torch.equal(first.label, second.label)
    assert torch.all(first.mask.sum(dim=-1) >= 2)


def test_config_and_input_validation_fail_closed():
    with pytest.raises(ValueError):
        NeuralConfig(hidden_dim=0)
    model = _model()
    with pytest.raises(ValueError):
        model.infer(torch.zeros(2, 4), torch.ones(2, 4))
    with pytest.raises(ValueError):
        model.infer(
            torch.zeros(2, 5),
            torch.full((2, 5), 0.5),
            validate_values=True,
        )


def test_masked_values_cannot_leak_through_encoder():
    dataset = StructuredFlowDataset(6, seed=909)
    model = _model(steps=3).eval()
    observed = dataset.observed.clone()
    changed = observed.clone()
    changed[dataset.mask == 0] = 12345.0
    first = model.infer(
        observed,
        dataset.mask,
        max_steps=3,
        fixed_depth=True,
        execution_mode="dense",
    )
    second = model.infer(
        changed,
        dataset.mask,
        max_steps=3,
        fixed_depth=True,
        execution_mode="dense",
    )
    assert torch.equal(first["routes"], second["routes"])
    assert torch.allclose(first["state"], second["state"], atol=0, rtol=0)


def test_forward_loss_backward_are_finite():
    dataset = StructuredFlowDataset(16, seed=1)
    batch = _batch(dataset, 16)
    model = _model(steps=3)
    output = model.forward_train(batch["observed"], batch["mask"], validate_values=True)
    loss, metrics = training_loss(output, batch["truth"], batch["label"])
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_future_regret_teacher_marks_last_step_ready():
    truth = torch.zeros(1, 1)
    states = [
        torch.tensor([[[2.0], [3.0]]]),
        torch.tensor([[[1.0], [3.0]]]),
        torch.tensor([[[1.0], [2.0]]]),
    ]
    target = future_regret_targets(states, truth, tolerance=0.0)
    assert target.shape == (3, 1, 2)
    assert torch.all(target[-1] == 1)
    assert target[0, 0, 0] == 0  # a later step improves candidate 0


def test_dense_and_compact_are_numerically_equivalent_but_count_work_differently():
    dataset = StructuredFlowDataset(20, seed=2)
    batch = _batch(dataset, 20)
    model = _model(steps=5).eval()
    # Zero halt weights and set a bias so every sample crosses after min_steps.
    with torch.no_grad():
        for parameter in model.halt.parameters():
            parameter.zero_()
        model.halt[-1].bias.fill_(10.0)
    dense = model.infer(
        batch["observed"],
        batch["mask"],
        max_steps=5,
        min_steps=2,
        halt_threshold=0.5,
        execution_mode="dense",
    )
    compact = model.infer(
        batch["observed"],
        batch["mask"],
        max_steps=5,
        min_steps=2,
        halt_threshold=0.5,
        execution_mode="compact",
    )
    assert torch.allclose(dense["state"], compact["state"], atol=1e-6, rtol=1e-6)
    assert torch.equal(dense["steps_taken"], compact["steps_taken"])
    assert torch.all(dense["steps_taken"] == 2)
    assert dense["update_examples"] == 20 * 5
    assert compact["update_examples"] == 20 * 2


def test_fixed_depth_executes_declared_number_of_steps():
    dataset = StructuredFlowDataset(7, seed=3)
    batch = _batch(dataset, 7)
    model = _model(steps=4).eval()
    output = model.infer(
        batch["observed"],
        batch["mask"],
        max_steps=3,
        fixed_depth=True,
        execution_mode="compact",
    )
    assert torch.all(output["steps_taken"] == 3)
    assert output["update_examples"] == 21
    assert output["recurrent_rounds"] == 3


def test_inference_state_satisfies_selected_boundary():
    dataset = StructuredFlowDataset(12, seed=4)
    batch = _batch(dataset, 12)
    names, _, boundaries = candidate_bases()
    assert len(names) == 2
    model = _model(steps=3).eval()
    output = model.infer(
        batch["observed"],
        batch["mask"],
        fixed_depth=True,
        max_steps=3,
    )
    for index in range(12):
        boundary = torch.from_numpy(boundaries[int(output["routes"][index])])
        residual = torch.linalg.vector_norm(boundary @ output["state"][index])
        assert residual.item() < 1e-5


def test_dense_one_example_event_trace_stops_cleanly():
    dataset = StructuredFlowDataset(1, seed=44)
    model = _model(steps=5).eval()
    with torch.no_grad():
        for parameter in model.halt.parameters():
            parameter.zero_()
        model.halt[-1].bias.fill_(10.0)
    output = model.infer(
        dataset.observed,
        dataset.mask,
        max_steps=5,
        min_steps=2,
        halt_threshold=0.5,
        execution_mode="dense",
        record_events=True,
    )
    assert int(output["steps_taken"][0]) == 2
    assert len(output["events"]) == 2
    assert output["recurrent_rounds"] == 2


def test_transparent_and_gaussian_references_have_valid_shapes():
    dataset = StructuredFlowDataset(10, seed=5)
    batch = _batch(dataset, 10)
    _, bases, _ = candidate_bases()
    basis_tensor = torch.from_numpy(bases)
    transparent = fit_each_structure(batch["observed"], batch["mask"], basis_tensor)
    gaussian = gaussian_generator_reference(
        batch["observed"], batch["mask"], basis_tensor, noise_std=0.05
    )
    assert transparent["state"].shape == batch["truth"].shape
    assert gaussian["mixture_state"].shape == batch["truth"].shape
    assert torch.allclose(
        gaussian["route_probabilities"].sum(dim=-1),
        torch.ones(10),
        atol=1e-6,
    )


def test_references_are_finite_for_every_nonempty_mask_pattern():
    _, bases, _ = candidate_bases()
    bases = torch.from_numpy(bases)
    observed = torch.linspace(-1.0, 1.0, 5).repeat(31, 1)
    masks = []
    for bits in range(1, 32):
        masks.append([(bits >> index) & 1 for index in range(5)])
    mask = torch.tensor(masks, dtype=torch.float32)
    fit = fit_each_structure(observed, mask, bases)
    gaussian = gaussian_generator_reference(observed, mask, bases, noise_std=0.05)
    for result in (fit["state"], gaussian["hard_state"], gaussian["mixture_state"]):
        assert torch.isfinite(result).all()


def test_neural_record_and_checkpoint_binding_detect_tampering(tmp_path: Path):
    checkpoint = tmp_path / "tiny.pt"
    train_checkpoint(
        checkpoint,
        device_name="cpu",
        train_size=64,
        val_size=32,
        epochs=1,
        batch_size=32,
        hidden_dim=16,
        steps=2,
        seed=77,
    )
    model, metadata, device = load_checkpoint(checkpoint, device_name="cpu")
    dataset = StructuredFlowDataset(1, seed=9001)
    observed = dataset.observed.to(device)
    mask = dataset.mask.to(device)
    output = model.infer(
        observed,
        mask,
        max_steps=2,
        fixed_depth=True,
        execution_mode="compact",
        record_events=True,
    )
    record = neural_record(
        model,
        output,
        observed,
        mask,
        metadata["names"],
        metadata["boundaries"],
        checkpoint_sha256=metadata["checkpoint_sha256"],
        checkpoint_format=metadata["format"],
    )
    assert verify_neural_record(record).accepted
    assert verify_neural_record(record, checkpoint=checkpoint).accepted

    changed_state = copy.deepcopy(record)
    changed_state["final"]["state"][0] += 0.5
    assert not verify_neural_record(changed_state).accepted

    changed_hash = copy.deepcopy(record)
    changed_hash["checkpoint"]["sha256"] = "0" * 64
    assert verify_neural_record(changed_hash).accepted  # standalone hash is descriptive
    assert not verify_neural_record(changed_hash, checkpoint=checkpoint).accepted

    changed_event = copy.deepcopy(record)
    changed_event["events"][-1]["state_after"][0] += 0.5
    assert not verify_neural_record(changed_event).accepted

    changed_counter = copy.deepcopy(record)
    changed_counter["execution"]["update_examples"] += 1
    assert not verify_neural_record(changed_counter).accepted


def test_legacy_checkpoint_loads_with_ambient_normalization(tmp_path: Path):
    model = _model(steps=2)
    names, _, boundaries = candidate_bases()
    legacy_config = asdict(model.config)
    legacy_config.pop("residual_normalization")
    legacy_config.pop("halt_timing")
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": legacy_config,
            "names": names,
            "boundaries": torch.from_numpy(boundaries),
            "torch_version": torch.__version__,
        },
        path,
    )
    loaded, metadata, _ = load_checkpoint(path, device_name="cpu")
    assert loaded.config.residual_normalization == "ambient"
    assert loaded.config.halt_timing == "before_update"
    assert metadata["format"].endswith("legacy")


def test_checkpoint_with_nonfinite_weight_is_rejected(tmp_path: Path):
    model = _model(steps=2)
    names, bases, boundaries = candidate_bases()
    state = model.state_dict()
    key = next(key for key, value in state.items() if value.is_floating_point() and key != "bases")
    state[key] = state[key].clone()
    state[key].view(-1)[0] = float("nan")
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "format": "universa-recurrent.neural-checkpoint.v1",
            "model_state": state,
            "config": asdict(model.config),
            "library": {
                "names": names,
                "bases": torch.from_numpy(bases),
                "boundaries": torch.from_numpy(boundaries),
            },
        },
        path,
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_checkpoint(path, device_name="cpu")
