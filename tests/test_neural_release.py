"""Regression cases found while auditing the GitHub-native release.

An empty measurement and an exactly tied decision are legitimate inputs, not
reasons for gradients or the independent record checker to fail.
"""
from pathlib import Path
import copy

import pytest

torch = pytest.importorskip("torch")
from universa_recurrent.neural.data import candidate_bases, StructuredFlowDataset
from universa_recurrent.neural.v2 import (
    AmbientRecurrentControl, DecisionPolicy, DirectMultiHypothesisNet,
    MultiHypothesisRecurrentNet, V2Config, ambient_training_loss,
    direct_training_loss, v2_training_loss,
)
from universa_recurrent.neural.v2_lingua import v2_record
from universa_recurrent.neural.v2_verification import verify_v2_record
from universa_recurrent.neural.v2_train import (
    train_v2_checkpoint, load_v2_checkpoint, evaluate_v2_checkpoint,
    benchmark_v2_checkpoint,
)


def make_model():
    _, bases, _ = candidate_bases()
    return MultiHypothesisRecurrentNet(torch.tensor(bases), V2Config(
        hidden_dim=12, candidate_embedding_dim=4, steps=2))


@pytest.mark.parametrize("kind", ["shared", "direct", "ambient"])
def test_exact_zero_residual_has_finite_gradients(kind):
    _, bases, _ = candidate_bases()
    cfg = V2Config(hidden_dim=12, candidate_embedding_dim=4, steps=2)
    if kind == "shared":
        model = make_model()
    elif kind == "direct":
        model = DirectMultiHypothesisNet(torch.tensor(bases), cfg)
    else:
        model = AmbientRecurrentControl(5, 12, 2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    observed = torch.zeros(3, 5)
    mask = torch.ones_like(observed)
    labels = torch.zeros(3, dtype=torch.long)
    if kind == "shared":
        loss, _ = v2_training_loss(model.forward_train(observed, mask), observed, labels)
    elif kind == "direct":
        loss, _ = direct_training_loss(model(observed, mask), observed, labels)
    else:
        loss, _ = ambient_training_loss(model.rollout(observed, mask), observed)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


@pytest.mark.parametrize("mode", ["dense", "compact"])
def test_exact_route_tie_uses_first_candidate_and_validates(mode):
    model = make_model().eval()
    with torch.no_grad():
        for parameter in model.prior_router.parameters(): parameter.zero_()
        for parameter in model.evidence.parameters(): parameter.zero_()
    data = StructuredFlowDataset(1, seed=8744)
    policy = DecisionPolicy(min_steps=1)
    result = model.infer(data.observed, data.mask, policy=policy,
                         execution_mode=mode, record_events=True)
    assert result["top_routes"].item() == 0
    assert torch.is_tensor(result["logical_hypothesis_updates"])
    names, _, boundaries = candidate_bases()
    record = v2_record(model, result, data.observed, data.mask, names, boundaries,
                       checkpoint_sha256="0" * 64,
                       checkpoint_format="universa-recurrent.neural-checkpoint.v2",
                       policy=policy)
    checked = verify_v2_record(record)
    assert checked.accepted, checked.reason


def test_model_owns_its_basis_buffer():
    _, bases, _ = candidate_bases()
    tensor = torch.tensor(bases)
    model = MultiHypothesisRecurrentNet(tensor)
    saved = model.bases.clone()
    tensor.zero_()
    assert torch.equal(model.bases, saved)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("release-regression") / "model.pt"
    train_v2_checkpoint(path, device_name="cpu", seed=700,
                        train_size=24, calibration_size=12, epochs=1,
                        batch_size=12, hidden_dim=8, candidate_embedding_dim=2,
                        steps=2, include_controls=True)
    return path


@pytest.mark.parametrize("component", ["main", "direct", "untied", "fixed_depth_1"])
def test_checkpoint_cannot_override_certified_library(checkpoint, tmp_path, component):
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    state = payload["model_state"] if component == "main" else payload["controls"][component]["model_state"]
    state["bases"] = state["bases"].clone() + 0.125
    path = tmp_path / "tampered.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="model-state bases"):
        load_v2_checkpoint(path, device_name="cpu")


@pytest.mark.parametrize("seed", [701, 702])
@pytest.mark.parametrize("operation", [evaluate_v2_checkpoint, benchmark_v2_checkpoint])
def test_eval_and_benchmark_refuse_training_or_calibration_seeds(checkpoint, seed, operation):
    with pytest.raises(ValueError, match="overlaps"):
        operation(checkpoint, device_name="cpu", seed=seed, n=8, batch_size=4)


def test_zero_noise_v2_training_is_rejected_before_work(tmp_path):
    with pytest.raises(ValueError, match="noise_std must be positive"):
        train_v2_checkpoint(tmp_path / "zero.pt", noise_std=0)


def test_commit_reason_is_not_just_a_free_text_label():
    model = make_model().eval()
    with torch.no_grad():
        model.prior_router.weight.zero_()
        model.prior_router.bias.copy_(torch.tensor([20., -20.]))
        for p in model.evidence.parameters(): p.zero_()
    data = StructuredFlowDataset(1, seed=8044)
    policy = DecisionPolicy(min_steps=1)
    out = model.infer(data.observed, data.mask, policy=policy, record_events=True)
    names, _, boundaries = candidate_bases()
    record = v2_record(model, out, data.observed, data.mask, names, boundaries,
                       checkpoint_sha256="0" * 64,
                       checkpoint_format="universa-recurrent.neural-checkpoint.v2", policy=policy)
    assert verify_v2_record(record).accepted
    record["final"]["decision_reason"] = "forced_commit"
    assert not verify_v2_record(record).accepted
