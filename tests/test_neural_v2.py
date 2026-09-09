from __future__ import annotations

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from universa_recurrent.neural.data import StructuredFlowDataset, candidate_bases
from universa_recurrent.neural.v2 import (
    CalibrationConfig,
    DecisionPolicy,
    MultiHypothesisRecurrentNet,
    V2Config,
    parameter_count,
    v2_training_loss,
)
from universa_recurrent.neural.v2_lingua import v2_record
from universa_recurrent.neural.v2_train import (
    build_v2_models,
    evaluate_v2_checkpoint,
    load_v2_checkpoint,
    train_v2_checkpoint,
)
from universa_recurrent.neural.v2_verification import verify_v2_record


def _model(*, steps: int = 4) -> MultiHypothesisRecurrentNet:
    _, bases, _ = candidate_bases()
    return MultiHypothesisRecurrentNet(
        torch.from_numpy(bases),
        V2Config(
            ambient_dim=5,
            latent_dim=2,
            num_structures=2,
            hidden_dim=20,
            candidate_embedding_dim=4,
            steps=steps,
        ),
    )


def test_v2_config_and_policy_fail_closed():
    with pytest.raises(ValueError):
        V2Config(num_structures=1)
    with pytest.raises(ValueError):
        DecisionPolicy(final_probability=0.9, commit_probability=0.8)
    with pytest.raises(ValueError):
        CalibrationConfig(min_coverage=1.1)


def test_v2_rollout_updates_every_hypothesis_and_backpropagates():
    dataset = StructuredFlowDataset(12, seed=201)
    model = _model(steps=3)
    output = model.forward_train(dataset.observed, dataset.mask, validate_values=True)
    assert output["states"].shape == (3, 12, 2, 5)
    assert output["route_probabilities"].shape == (3, 12, 2)
    assert torch.allclose(
        output["route_probabilities"].sum(dim=-1),
        torch.ones(3, 12),
        atol=1e-6,
    )
    loss, metrics = v2_training_loss(output, dataset.truth, dataset.label)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_v2_dense_and_compact_decisions_match_but_work_counts_differ():
    dataset = StructuredFlowDataset(10, seed=202)
    model = _model(steps=4).eval()
    with torch.no_grad():
        model.prior_router.weight.zero_()
        model.prior_router.bias.copy_(torch.tensor([10.0, -10.0]))
        for parameter in model.evidence.parameters():
            parameter.zero_()
    policy = DecisionPolicy(
        commit_probability=0.8,
        commit_margin=0.5,
        final_probability=0.6,
        final_margin=0.1,
        min_steps=1,
    )
    dense = model.infer(
        dataset.observed,
        dataset.mask,
        policy=policy,
        execution_mode="dense",
    )
    compact = model.infer(
        dataset.observed,
        dataset.mask,
        policy=policy,
        execution_mode="compact",
    )
    assert torch.equal(dense["routes"], compact["routes"])
    assert torch.equal(dense["decision_steps"], compact["decision_steps"])
    assert torch.all(dense["decision_steps"] == 1)
    assert torch.allclose(dense["state"], compact["state"], atol=1e-6, rtol=1e-6)
    assert dense["candidate_update_examples"] == 10 * 2 * 4
    assert compact["candidate_update_examples"] == 10 * 2


def test_v2_abstention_returns_mixture_without_single_structure_route():
    dataset = StructuredFlowDataset(7, seed=203)
    model = _model(steps=2).eval()
    with torch.no_grad():
        model.prior_router.weight.zero_()
        model.prior_router.bias.zero_()
        for parameter in model.evidence.parameters():
            parameter.zero_()
    policy = DecisionPolicy(
        commit_probability=0.99,
        commit_margin=0.9,
        final_probability=0.9,
        final_margin=0.5,
        min_steps=1,
    )
    output = model.infer(
        dataset.observed,
        dataset.mask,
        policy=policy,
        execution_mode="compact",
    )
    assert torch.all(output["abstained"])
    assert torch.all(output["routes"] == -1)
    mixture = torch.einsum(
        "bk,bkn->bn", output["route_probabilities"], output["candidate_states"]
    )
    assert torch.allclose(output["state"], mixture, atol=1e-6, rtol=1e-6)


def test_v2_controls_are_approximately_parameter_matched():
    main, direct, untied, metadata = build_v2_models(
        torch.device("cpu"),
        hidden_dim=20,
        candidate_embedding_dim=4,
        steps=3,
        include_controls=True,
    )
    assert direct is not None and untied is not None
    main_count = parameter_count(main)
    assert abs(parameter_count(direct) - main_count) / main_count < 0.15
    assert abs(parameter_count(untied) - main_count) / main_count < 0.15
    assert metadata["main_parameters"] == main_count


def test_v2_checkpoint_calibrates_on_disjoint_split_and_loads_controls(tmp_path: Path):
    path = tmp_path / "v2.pt"
    result = train_v2_checkpoint(
        path,
        device_name="cpu",
        seed=300,
        train_size=48,
        calibration_size=24,
        epochs=1,
        batch_size=16,
        hidden_dim=12,
        candidate_embedding_dim=4,
        steps=2,
        include_controls=True,
    )
    assert path.exists() and path.with_name(path.name + ".sha256").exists()
    assert result["note"].startswith("Calibration metrics")
    model, controls, metadata, device = load_v2_checkpoint(path, device_name="cpu")
    assert device.type == "cpu"
    assert set(controls) == {"direct", "untied", "ambient", "fixed_depth_1"}
    assert model.config.steps == 2
    training = metadata["training"]
    assert training["train_seed"] != training["calibration_seed"]
    assert metadata["calibration"]["policy"]["min_steps"] in (1, 2)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        train_v2_checkpoint(
            path,
            device_name="cpu",
            train_size=8,
            calibration_size=8,
            epochs=1,
            batch_size=8,
            hidden_dim=8,
            candidate_embedding_dim=2,
            steps=1,
            include_controls=False,
        )


def test_v2_evaluation_marks_test_seed_as_distinct(tmp_path: Path):
    path = tmp_path / "v2.pt"
    train_v2_checkpoint(
        path,
        device_name="cpu",
        seed=400,
        train_size=32,
        calibration_size=16,
        epochs=1,
        batch_size=16,
        hidden_dim=10,
        candidate_embedding_dim=3,
        steps=2,
        include_controls=False,
    )
    result = evaluate_v2_checkpoint(
        path,
        device_name="cpu",
        seed=999,
        n=16,
        batch_size=8,
    )
    assert result["test_seed_distinct_from_training"] is True
    assert result["rows"][0]["method"] == "v2_calibrated_policy"
    assert "coverage" in result["rows"][0]
    assert "route_brier" in result["rows"][0]
    assert "route_ece_10bin" in result["rows"][0]


def test_v2_lingua_verifies_and_rejects_tampering(tmp_path: Path):
    path = tmp_path / "v2.pt"
    train_v2_checkpoint(
        path,
        device_name="cpu",
        seed=500,
        train_size=48,
        calibration_size=24,
        epochs=1,
        batch_size=16,
        hidden_dim=12,
        candidate_embedding_dim=4,
        steps=2,
        include_controls=False,
    )
    model, _, metadata, _ = load_v2_checkpoint(path, device_name="cpu")
    policy = DecisionPolicy(**metadata["calibration"]["policy"])
    dataset = StructuredFlowDataset(1, seed=12001)
    output = model.infer(
        dataset.observed,
        dataset.mask,
        policy=policy,
        execution_mode="compact",
        record_events=True,
        validate_values=True,
    )
    record = v2_record(
        model,
        output,
        dataset.observed,
        dataset.mask,
        metadata["names"],
        metadata["boundaries"],
        checkpoint_sha256=metadata["checkpoint_sha256"],
        checkpoint_format=metadata["format"],
        policy=policy,
    )
    assert verify_v2_record(record).accepted
    assert verify_v2_record(record, checkpoint=path).accepted

    changed_probability = copy.deepcopy(record)
    changed_probability["events"][-1]["route_probabilities"][0] += 0.2
    assert not verify_v2_record(changed_probability).accepted

    changed_decision = copy.deepcopy(record)
    changed_decision["final"]["decision"] = (
        "abstain" if record["final"]["decision"] == "commit" else "commit"
    )
    assert not verify_v2_record(changed_decision).accepted

    changed_checkpoint = copy.deepcopy(record)
    changed_checkpoint["checkpoint"]["sha256"] = "0" * 64
    assert verify_v2_record(changed_checkpoint).accepted
    assert not verify_v2_record(changed_checkpoint, checkpoint=path).accepted
