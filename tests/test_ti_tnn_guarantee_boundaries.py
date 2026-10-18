"""Guarantee boundaries for the CDL integration (research/ti_tnn/integration_plan.md, section 3).

Two counterexamples kept in the suite on purpose:
1. every categorical law holds and the router's Brier score is unchanged, yet the answer flips;
2. in the real TI-TNN model, a branch's output layer changes while the route probabilities and every
   categorical defect stay identical, and the answers change.

The right classification is: structural checks pass, calibration is unchanged, the behavioral check
fails, and the candidate is rejected. Neither the laws nor the Brier score can be asked to catch it.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn" / "ti_tnn_train.py"
spec = importlib.util.spec_from_file_location("ti_tnn_train_boundaries", SCRIPT)
ti = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ti)
cdl = ti._cdl()


def brier(probs: torch.Tensor, target: torch.Tensor) -> float:
    """Multiclass Brier score: squared distance between the probabilities and the one-hot target."""
    return float(((probs - target) ** 2).sum(-1).mean())


def behavioral_check(answers: torch.Tensor, expected: torch.Tensor) -> bool:
    """Pass only if every registered answer is unchanged."""
    return bool(torch.equal(answers, expected))


def activate(structural_ok: bool, behavioral_ok: bool) -> bool:
    """A candidate is activated only if every mandatory check passes; nothing offsets a failure."""
    return structural_ok and behavioral_ok


def inverse_pair_branch(sign: float):
    """A branch over the category presented by an inverse pair f: A -> B, g = f^-1, with M_f = sign * I_2."""
    cat = cdl.QueryCategory(objects=("A", "B"), generators={"f": ("A", "B"), "g": ("B", "A")},
                            inverse_pairs=(("f", "g"),))
    branch = cdl.CDLSpecialistBranch(4, 2, 1.0, cat)
    with torch.no_grad():
        branch.morphisms["f"].weight = sign * torch.eye(2)
    return branch


def test_lawful_rotation_keeps_laws_and_brier_but_flips_the_answer():
    x = torch.tensor([1.0, 0.0])
    decoder = torch.stack([x, -x])  # class 0 reads (1, 0), class 1 reads (-1, 0)
    route_probs, route_target = torch.tensor([[0.9, 0.1]]), torch.tensor([[1.0, 0.0]])
    answers = {}
    for sign in (1.0, -1.0):
        b = inverse_pair_branch(sign)
        with torch.no_grad():
            f, g = b.matrix("f"), b.matrix("g")
            eye = torch.eye(2)
            # structural checks: orthogonal, determinant +1, inverse laws, functorial paths
            assert torch.allclose(f.T @ f, eye, atol=1e-6) and torch.linalg.det(f) == pytest.approx(1.0, abs=1e-6)
            assert torch.allclose(g @ f, eye, atol=1e-6) and torch.allclose(f @ g, eye, atol=1e-6)
            assert torch.allclose(b.path_matrix(["f", "g"]), eye, atol=1e-6)
            answers[sign] = int((decoder @ (f @ x)).argmax())
        assert brier(route_probs, route_target) == pytest.approx(0.02)  # calibration measurement unchanged
    assert answers[1.0] == 0 and answers[-1.0] == 1  # the answer flips
    assert not activate(structural_ok=True, behavioral_ok=answers[-1.0] == answers[1.0])


def test_changing_a_branch_output_layer_leaves_routing_and_laws_identical():
    cat = cdl.routing_world_category()
    world = ti.ToyWorld(n_countries=8, n_fillers=4)
    torch.manual_seed(0)
    model = ti.TypeInformedTreeNN(world.vocab_size, d_model=16, d_stalk=8, d_edge=4, branch_kind="cdl",
                                  bridge_kind="functorial", category=cat).eval()
    batch = world.sample(64, 6, torch.Generator().manual_seed(1), "cpu")

    def observe():
        with torch.no_grad():
            out = model(batch["ids"])
            probs = out["route_info"]["choice_probs"]
            defect = cdl.naturality_loss(model.branches, model.sheaf_bridge, probs, floor=1.0)
            laws = torch.stack([b.matrices() for b in model.branches])
            return probs.clone(), float(defect), laws.clone(), out["logits"][:, -1].argmax(-1), out["top1"]

    probs0, defect0, laws0, answers0, top1 = observe()
    used = int(top1.mode().values)  # change the branch the router actually uses most
    with torch.no_grad():
        model.branches[used].up_proj.weight.mul_(-3.0)
    probs1, defect1, laws1, answers1, _ = observe()
    assert torch.equal(probs0, probs1)  # Jev reads the input before any specialist runs
    assert defect1 == defect0 and torch.equal(laws0, laws1)  # every categorical defect is unchanged
    assert not torch.equal(answers0, answers1)  # but the answers changed
    assert not activate(structural_ok=True, behavioral_ok=behavioral_check(answers1, answers0))
