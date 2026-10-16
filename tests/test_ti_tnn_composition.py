"""Properties the functorial-composition experiment depends on (research/ti_tnn/composition.py).

Software checks on tiny models, not experiment results.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn" / "composition.py"
spec = importlib.util.spec_from_file_location("ti_tnn_composition", SCRIPT)
comp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comp)


def world():
    return comp.World(n_countries=32, n_regions=4, n_sizes=3)


def test_world_relations_are_consistent():
    w = world()
    c = torch.arange(32)
    assert torch.equal(w.country_of[w.capital_of[c]], c) and torch.equal(w.capital_of[w.country_of[c]], c)
    assert torch.equal(w.answer(["capital_of", "size_of"], c), w.size_of[w.capital_of[c]])
    assert torch.equal(w.answer(["country_of", "capital_of", "size_of"], c), w.size_of[c])  # = size_of on capitals
    assert len(w.training_facts()) == 4


def test_paths_must_type_check():
    assert comp.check_path(["capital_of", "size_of"]) == ("country", "size")
    with pytest.raises(ValueError, match="cannot follow"):
        comp.check_path(["region_of", "size_of"])
    for paths in comp.FAMILIES.values():
        for path in paths:
            comp.check_path(path)


def test_training_never_shows_a_chain():
    """Only single-relation facts are trained; every scored chain is new."""
    assert all(len([r]) == 1 for r, _, _ in world().training_facts())
    assert all(len(p) >= 2 for fam in ("two_hop", "three_hop", "round_trip") for p in comp.FAMILIES[fam])


def test_functorial_relations_satisfy_the_laws_exactly():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    try:
        m = comp.build(world(), "functorial", 16)
    finally:
        torch.set_default_dtype(torch.float32)
    with torch.no_grad():  # heavy perturbation: the laws come from the parametrization
        for p in m.bridge.parameters():
            p.add_(10 * torch.randn_like(p))
    eye = torch.eye(16, dtype=torch.float64)
    A, B = m.relation_matrix("capital_of"), m.relation_matrix("country_of")
    assert torch.allclose(B @ A, eye, atol=1e-9) and torch.allclose(A.T @ A, eye, atol=1e-9)
    two_step = m.relation_matrix("size_of") @ m.relation_matrix("capital_of")
    direct = m.bridge.transport(comp.KINDS.index("country"), comp.KINDS.index("size"))
    assert torch.allclose(two_step, direct, atol=1e-9)  # path independence: a flat connection


def test_soft_penalty_vanishes_only_at_inverse_maps():
    m = comp.build(world(), "soft", 8)
    with torch.no_grad():
        q, _ = torch.linalg.qr(torch.randn(8, 8))
        m.maps["capital_of"].copy_(2 * q)
        m.maps["country_of"].copy_(0.5 * q.T)
    assert m.penalty().item() < 1e-10
    with torch.no_grad():
        m.maps["country_of"].add_(0.1)
    assert m.penalty().item() > 1e-3
    assert comp.build(world(), "linear", 8).penalty().item() == 0.0


@pytest.mark.parametrize("variant", comp.VARIANTS)
def test_every_variant_scores_chains_of_any_length(variant):
    torch.manual_seed(1)
    m = comp.build(world(), variant, 16)
    for paths in comp.FAMILIES.values():
        for path in paths:
            x, y = world().queries(path)
            logits = m(path, x)
            assert logits.shape == (len(x), world().sizes[comp.RELATIONS[path[-1]][1]])


@pytest.mark.parametrize("variant", comp.VARIANTS)
def test_training_fits_single_relations(variant):
    args = comp.parse_args(["--steps", "300", "--d", "32", "--log-every", "100"])
    torch.manual_seed(2)
    w = world()
    m = comp.build(w, variant, 32)
    log = comp.train(m, w, args)
    assert log[-1]["loss"] < log[0]["loss"]
    res = comp.evaluate(m, w)
    assert set(res) >= set(comp.FAMILIES)
    assert all(0 <= res[f]["accuracy"] <= 1 for f in comp.FAMILIES)
    assert res["one_hop"]["paths"]["region_of"]["chance"] == pytest.approx(1 / 4)


def test_main_writes_results(tmp_path):
    comp.main(["--variants", "functorial", "linear", "--seeds", "0", "--countries", "16", "--regions", "4",
               "--sizes", "3", "--d", "16", "--steps", "20", "--out", str(tmp_path)])
    files = list(tmp_path.glob("*/composition.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert {r["variant"] for r in data["runs"]} == {"functorial", "linear"}
    assert "inverse_defect" in data["runs"][0]["eval"]
