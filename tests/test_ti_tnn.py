"""Properties the TI-TNN benchmarks' claims depend on (research/ti_tnn/ti_tnn_train.py).

These are software checks on tiny untrained models, not benchmark results.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn" / "ti_tnn_train.py"
spec = importlib.util.spec_from_file_location("ti_tnn_train", SCRIPT)
ti = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ti)


def comp_model(world, **kw):
    torch.manual_seed(0)
    defaults = dict(d_model=16, d_stalk=8, d_edge=4, branch_embeddings="private", router_context="context_only")
    defaults.update(kw)
    return ti.TypeInformedTreeNN(world.vocab_size, **defaults)


@pytest.fixture(scope="module")
def world():
    return ti.CompositionalWorld(n_countries=64, n_fillers=8)


def test_splits_are_disjoint_balanced_and_cover_every_country(world):
    sets = [set(world.countries_in[s].tolist()) for s in world.SPLITS]
    assert sum(len(s) for s in sets) == world.n and len(set().union(*sets)) == world.n
    for name in world.SPLITS:
        c = world.countries_in[name]
        combos = (world.trait_a[c] * 2 + world.trait_b[c]).bincount(minlength=4)
        assert combos.min() == combos.max(), f"{name} split is not balanced over (A, B)"
        assert world.status[c].float().mean().item() == 0.5


def test_training_never_shows_combined_queries_for_held_out_countries(world):
    gen = torch.Generator().manual_seed(1)
    held_out = set(world.countries_in["val"].tolist()) | set(world.countries_in["test"].tolist())
    seen_single = set()
    for _ in range(50):
        b = world.sample(256, 6, gen, "cpu")
        combined = b["qtype"] == world.COMBINED
        assert not (set(b["country"][combined].tolist()) & held_out)
        seen_single |= set(b["country"][b["qtype"] == 0].tolist())
    assert held_out <= seen_single, "trait queries should still cover held-out countries"


def test_status_is_xor_and_independent_of_each_trait(world):
    assert torch.equal(world.status, world.trait_a ^ world.trait_b)
    for trait in (world.trait_a, world.trait_b):
        for value in (0, 1):
            assert world.status[trait == value].float().mean().item() == 0.5


def test_gradient_masking_keeps_trait_b_supervision_out_of_branch_0(world):
    model = comp_model(world).train()
    b = world.sample(0, 6, torch.Generator().manual_seed(2), "cpu", qtype=1,
                     countries=torch.arange(world.n))  # traitB_of queries only
    out = model(b["ids"], tau=1.0, grad_allow=ti.grad_allow_mask(b, combined_trains_branches=False))
    model.compute_calibrated_loss(out, b["answer"], b["route"], b["bridge"])["total_loss"].backward()
    d = model.d_model
    emb_grad = model.branch_vectorizer.weight.grad
    assert emb_grad[:, 0:d].abs().sum() == 0, "Branch 0's embedding subspace received Trait-B gradient"
    assert emb_grad[:, d:2 * d].abs().sum() > 0
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.branches[0].parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.branches[1].parameters())


def test_strict_isolation_trains_only_the_bridge_on_combined_queries(world):
    model = comp_model(world).train()
    b = world.sample(0, 6, torch.Generator().manual_seed(3), "cpu", qtype=world.COMBINED,
                     countries=world.countries_in["train"])
    with torch.no_grad():  # make the gate fire so the bridge is in the graph
        model.router.noul_bridge_head.bias.fill_(5.0)
    out = model(b["ids"], tau=1.0, grad_allow=ti.grad_allow_mask(b, combined_trains_branches=False))
    model.compute_calibrated_loss(out, b["answer"], b["route"], b["bridge"])["total_loss"].backward()
    assert model.branch_vectorizer.weight.grad is None or model.branch_vectorizer.weight.grad.abs().sum() == 0
    for branch in model.branches:
        assert all(p.grad is None or p.grad.abs().sum() == 0 for p in branch.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.sheaf_bridge.parameters())


def test_router_ignores_the_entity_token(world):
    model = comp_model(world).eval()
    b = world.sample(0, 6, torch.Generator().manual_seed(4), "cpu", qtype=world.COMBINED,
                     countries=world.countries_in["test"])
    swapped = b["ids"].clone()
    swapped[:, -1] = world.country_tok[0]
    with torch.no_grad():
        p1 = model(b["ids"])["route_info"]["choice_probs"]
        p2 = model(swapped)["route_info"]["choice_probs"]
    assert torch.equal(p1, p2)


@pytest.mark.parametrize("private", [False, True])
def test_conditional_execution_matches_running_every_branch(world, private):
    model = comp_model(world, branch_embeddings="private" if private else "shared").eval()
    b = world.sample(512, 6, torch.Generator().manual_seed(5), "cpu")
    with torch.no_grad():
        model.router.noul_bridge_head.bias.fill_(0.3)  # some queries fire the bridge, some do not
        out, ref = model(b["ids"]), model(b["ids"], reference=True)
    assert out["branch_execs"] < ref["branch_execs"]
    assert torch.allclose(out["logits"], ref["logits"], atol=1e-5, rtol=0)


def test_bridge_edge_has_a_fixed_orientation(world):
    model = comp_model(world).eval()
    b = world.sample(0, 6, torch.Generator().manual_seed(6), "cpu", qtype=world.COMBINED,
                     countries=world.countries_in["test"])
    with torch.no_grad():
        model.router.choice_head.bias.copy_(torch.tensor([3.0, 2.0, -5.0, -5.0]))  # Branch 0 first, 1 second
        model.router.noul_bridge_head.bias.fill_(5.0)
    seen = []
    original = model.sheaf_bridge.forward
    model.sheaf_bridge.forward = lambda u, v: (seen.append((u.clone(), v.clone())), original(u, v))[1]
    with torch.no_grad():
        model(b["ids"], force_route=torch.zeros(len(b["ids"]), dtype=torch.long))
        model(b["ids"], force_route=torch.ones(len(b["ids"]), dtype=torch.long))
    (u0, v0), (u1, v1) = seen
    assert torch.equal(u0, u1) and torch.equal(v0, v1), "executing Branch 1 first flipped the edge"


def test_mac_counts_follow_layer_shapes():
    model = ti.TypeInformedTreeNN(100, d_model=128, d_stalk=64, d_edge=32)
    assert model.branches[0].macs_per_token() == 32768
    assert model.sheaf_bridge.macs_per_token() == 8192
    assert model.macs_per_query(6, branch_execs=2, bridge_execs=1) == 6 * (2 * 32768 + 8192)
