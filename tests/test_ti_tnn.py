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


@pytest.mark.parametrize("use_bridge", [True, False])
def test_branch_1_state_never_shapes_branch_0_training(world, use_bridge):
    """Routed training: a Trait-A query's update to Branch 0 must not depend on Branch 1's state,
    even when the bridge gate fires. (The dense variant mixes every branch and is not isolated.)"""

    def branch0_grad(shift_branch_1):
        model = comp_model(world, use_bridge=use_bridge).train()
        with torch.no_grad():
            model.router.noul_bridge_head.bias.fill_(5.0)  # the gate fires, as it can early in training
            model.router.choice_head.bias.copy_(torch.tensor([3.0, 2.0, -5.0, -5.0]))
            if shift_branch_1:  # change only Branch 1's embedding subspace
                d = model.d_model
                model.branch_vectorizer.weight[:, d:2 * d] += torch.randn_like(model.branch_vectorizer.weight[:, d:2 * d])
        b = world.sample(0, 6, torch.Generator().manual_seed(7), "cpu", qtype=0, countries=torch.arange(world.n))
        torch.manual_seed(1)  # same Gumbel noise in both calls
        out = model(b["ids"], tau=1.0, grad_allow=ti.grad_allow_mask(b, combined_trains_branches=False))
        assert out["bridge_fired"].all() == use_bridge
        model.compute_calibrated_loss(out, b["answer"], b["route"], b["bridge"])["total_loss"].backward()
        return model.branch_vectorizer.weight.grad[:, :model.d_model].clone()

    assert torch.equal(branch0_grad(False), branch0_grad(True))


def test_shuffled_partner_control_reads_the_partners_computed_stalk(world):
    model = comp_model(world).eval()
    b = world.sample(512, 6, torch.Generator().manual_seed(8), "cpu")
    perm = torch.roll(torch.arange(len(b["ids"])), shifts=1)
    with torch.no_grad():
        model.router.noul_bridge_head.bias.fill_(0.3)  # some queries fire the bridge, some do not
        out = model(b["ids"], partner_perm=perm)
        ref = model(b["ids"], partner_perm=perm, reference=True)  # every stalk computed
    assert 0 < out["bridge_execs"] < len(b["ids"])
    assert torch.allclose(out["logits"], ref["logits"], atol=1e-5, rtol=0)


def test_routing_evaluation_reuses_training_facts():
    """The routing benchmark scores facts also seen in training, in fresh contexts: not held-out facts."""
    w = ti.ToyWorld()
    facts = lambda b: set(zip(b["qtype"].tolist(), b["ids"][:, -1].tolist()))  # noqa: E731
    train, gen = set(), torch.Generator().manual_seed(10_000)
    for _ in range(50):
        train |= facts(w.sample(512, 6, gen, "cpu"))
    assert facts(w.sample(2048, 6, torch.Generator().manual_seed(999_001), "cpu")) <= train


SMOKE = ["--device", "cpu", "--seeds", "0", "--steps", "3", "--batch", "16", "--eval-size", "64",
         "--comp-countries", "64", "--n-countries", "16", "--n-fillers", "8", "--queries-per-country", "2",
         "--d-model", "16", "--d-stalk", "8", "--d-edge", "4", "--log-every", "100"]


def test_routing_benchmark_runs_end_to_end():
    result = ti.run_routing(ti.parse_args(SMOKE), torch.device("cpu"))
    assert {r["variant"] for r in result["runs"]} == set(ti.ROUTING_VARIANTS)
    routed = next(r for r in result["runs"] if r["variant"] == "sparse")["eval"]
    assert 0 <= routed["task_acc"] <= 1 and set(routed["controls"]) == {"oracle_routing", "random_routing_bridge_off", "bridge_off"}
    assert routed["exactness"]["max_abs_logit_diff_vs_run_everything"] < 1e-5


def test_compositional_benchmark_runs_end_to_end():
    result = ti.run_compositional(ti.parse_args(SMOKE), torch.device("cpu"))
    assert {r["variant"] for r in result["runs"]} == set(ti.COMP_VARIANTS)
    routed = next(r for r in result["runs"] if r["variant"] == "routed_bridge")["eval"]
    assert routed["combined_test"]["chance"] == 0.5 and routed["combined_test"]["n_countries"] == 16
    assert set(routed["controls"]) == {"combined_test_bridge_off", "combined_test_shuffled_partner"}
    assert routed["exactness"]["max_abs_logit_diff_vs_run_everything"] < 1e-5


def test_normalized_mutual_information_bounds():
    a = torch.tensor([0, 0, 1, 1, 2, 2])
    assert ti.normalized_mutual_information(a, torch.tensor([3, 3, 0, 0, 1, 1])) == pytest.approx(1.0)
    assert ti.normalized_mutual_information(a, torch.zeros(6, dtype=torch.long)) == 0.0
    assert ti.normalized_mutual_information(torch.tensor([0, 1, 0, 1]), torch.tensor([0, 0, 1, 1])) == pytest.approx(0.0)


def test_unsupervised_loss_ignores_labels_and_charges_the_bridge():
    w = ti.ToyWorld(n_countries=8, n_fillers=4)
    b = w.sample(64, 6, torch.Generator().manual_seed(9), "cpu")
    torch.manual_seed(0)
    model = ti.TypeInformedTreeNN(w.vocab_size, d_model=16, d_stalk=8, d_edge=4).eval()
    with torch.no_grad():
        out = model(b["ids"])
        base = model.compute_unsupervised_loss(out, b["answer"], lambda_flops=0.05, bridge_cost=0.0)
        charged = model.compute_unsupervised_loss(out, b["answer"], lambda_flops=0.05, bridge_cost=1.0)
        labelled = model.compute_calibrated_loss(out, b["answer"], b["route"], b["bridge"], 0.5, 0.05)
        relabelled = model.compute_calibrated_loss(out, b["answer"], (b["route"] + 1) % 4, b["bridge"], 0.5, 0.05)
    assert charged["total_loss"] > base["total_loss"]
    assert charged["expected_cost"] - base["expected_cost"] == pytest.approx(out["route_info"]["noul_prob"].mean().item(), abs=1e-6)
    assert labelled["total_loss"] != relabelled["total_loss"]  # the labelled loss does depend on labels


def test_must_separate_pairs_share_an_entity_kind():
    kinds = {rel: kind for rel, kind, _, _ in ti.ToyWorld.TYPES}
    assert all(kinds[x] == kinds[y] for x, y in ti.MUST_SEPARATE)


def test_unsupervised_benchmark_runs_end_to_end():
    result = ti.run_unsupervised(ti.parse_args(SMOKE + ["--benchmark", "unsupervised"]), torch.device("cpu"))
    assert {r["variant"] for r in result["runs"]} == set(ti.UNSUP_VARIANTS)
    st = next(r for r in result["runs"] if r["variant"] == "unsup_bridge")["structure"]
    assert 0 <= st["purity"] <= 1 and 0 <= st["nmi_type_branch"] <= 1 + 1e-9
    assert set(st["majority_branch"]) == {t[0] for t in ti.ToyWorld.TYPES}
    assert abs(sum(st["branch_share_by_type"]["echo"]) - 1) < 1e-6
    nobridge = next(r for r in result["runs"] if r["variant"] == "unsup_nobridge")["structure"]
    assert nobridge["bridge_fire_rate"] == 0.0


# =========================================================================== CDL: functorial branches and bridge
cdl = ti._cdl()


def perturb_(module, scale=10.0, seed=0):
    """Heavy random perturbation of every underlying parameter (parametrization originals included)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))


def eye(d):
    return torch.eye(d, dtype=torch.float64)


EXACT = dict(atol=1e-9, rtol=1e-9)  # the laws are exact; float64 leaves only rounding error


def build64(make):
    """Build a module under float64, so the orthogonal parametrization's base is float64 too."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return make()
    finally:
        torch.set_default_dtype(previous)


def test_query_category_rejects_malformed_presentations():
    with pytest.raises(ValueError, match="unknown object"):
        cdl.QueryCategory(("a",), {"f": ("a", "b")})
    with pytest.raises(ValueError, match="not opposite"):
        cdl.QueryCategory(("a", "b"), {"f": ("a", "b"), "g": ("a", "b")}, (("f", "g"),))
    cat = cdl.routing_world_category()
    assert cat.check_path(["capital_of", "size_of"]) == ("country", "size")
    with pytest.raises(ValueError, match="cannot follow"):
        cat.check_path(["region_of", "size_of"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cdl_branch_composition_laws_hold_exactly_under_heavy_perturbation(seed):
    cat = cdl.routing_world_category()
    torch.manual_seed(seed)
    branch = build64(lambda: cdl.CDLSpecialistBranch(16, 8, 1.0, cat))
    perturb_(branch, seed=seed)
    d = 8
    for name in cat.names:  # every generator is an isomorphism (orthogonal)
        M = branch.matrix(name)
        assert torch.allclose(M.T @ M, eye(d), **EXACT)
    # Inverse pair: country_of o capital_of = id_country and capital_of o country_of = id_capital.
    assert torch.allclose(branch.path_matrix(["capital_of", "country_of"]), eye(d), **EXACT)
    assert torch.allclose(branch.path_matrix(["country_of", "capital_of"]), eye(d), **EXACT)
    # Functoriality: applying generators step by step equals applying the path's image.
    x = torch.randn(5, 3, 16, dtype=torch.float64)
    hot = lambda n: torch.nn.functional.one_hot(torch.full((5,), cat.index(n)), len(cat.names)).double()  # noqa: E731
    _, stepped = branch(x, steps=[hot("capital_of"), hot("size_of")])
    direct = branch.encode(x) @ branch.path_matrix(["capital_of", "size_of"]).T
    assert torch.allclose(stepped, direct, **EXACT)
    # Associativity of a longer path, by construction.
    p3 = branch.path_matrix(["country_of", "capital_of", "size_of"])
    assert torch.allclose(p3, branch.matrix("size_of") @ branch.path_matrix(["country_of", "capital_of"]), **EXACT)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_functorial_bridge_transports_compose_exactly_under_heavy_perturbation(seed):
    K, d = 4, 8
    torch.manual_seed(seed)
    bridge = build64(lambda: cdl.FunctorialSheafBridge(K, d, 16))
    perturb_(bridge, seed=seed)
    T = lambda u, v: bridge.transport(u, v)  # noqa: E731
    for u in range(K):
        assert torch.allclose(T(u, u), eye(d), **EXACT)  # identity
        for v in range(K):
            assert torch.allclose(T(v, u), T(u, v).T, **EXACT)  # inverse
            assert torch.allclose(T(v, u) @ T(u, v), eye(d), **EXACT)
            for w in range(K):
                assert torch.allclose(T(v, w) @ T(u, v), T(u, w), **EXACT)  # composition
    holonomy = T(3, 0) @ T(2, 3) @ T(1, 2) @ T(0, 1)  # around a cycle of the branch graph
    assert torch.allclose(holonomy, eye(d), **EXACT)


@pytest.mark.parametrize("seed", [0, 1])
def test_coboundary_vanishes_exactly_on_transported_sections(seed):
    K, d = 4, 8
    torch.manual_seed(seed)
    bridge = build64(lambda: cdl.FunctorialSheafBridge(K, d, 16))
    perturb_(bridge, seed=seed)
    s_u = 5 * torch.randn(6, 3, d, dtype=torch.float64)
    u, v = torch.tensor([0, 1, 2, 3, 0, 2]), torch.tensor([1, 2, 3, 0, 3, 1])
    s_v = torch.stack([s_u[i] @ bridge.transport(int(u[i]), int(v[i])).T for i in range(6)])
    assert bridge.coboundary(s_u, s_v, u, v).abs().max() < 1e-9
    assert torch.allclose(bridge(s_u, s_v, u, v), bridge.edge_to_model.bias.expand(6, 3, 16), **EXACT)
    assert bridge.coboundary(s_u, s_v + 1.0, u, v).abs().max() > 0.1  # an inconsistent section is detected


def _natural_family(K=4, d=8, seed=0):
    """Branch functors M^k_g = Q_k^T C_g Q_k for shared C_g: the bridge's transports are natural."""
    cat = cdl.routing_world_category()
    torch.manual_seed(seed)
    branches = [cdl.CDLSpecialistBranch(16, d, 1.0, cat) for _ in range(K)]
    bridge = cdl.FunctorialSheafBridge(K, d, 16)
    perturb_(bridge, seed=seed)
    Q = bridge.maps().detach()
    C = {g: torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(seed + 7 + i)))[0]
         for i, g in enumerate(branches[0].morphisms)}
    with torch.no_grad():
        for k, b in enumerate(branches):
            for g, mod in b.morphisms.items():
                mod.weight = Q[k].T @ C[g] @ Q[k]
    return cat, branches, bridge


def test_naturality_loss_vanishes_on_natural_transports_and_detects_defects():
    cat, branches, bridge = _natural_family()
    uniform = torch.full((32, 4), 0.25)
    assert cdl.naturality_loss(branches, bridge, uniform, floor=0.0).item() < 1e-6
    perturb_(branches[2].morphisms, scale=1.0, seed=3)
    assert cdl.naturality_loss(branches, bridge, uniform, floor=0.0).item() > 1e-2


def test_naturality_loss_follows_the_router():
    cat, branches, bridge = _natural_family()
    perturb_(branches[3].morphisms, scale=1.0, seed=4)  # break only branch 3's naturality
    pair01 = torch.zeros(32, 4)
    pair01[:, :2] = 0.5  # the router only ever uses branches 0 and 1
    assert cdl.naturality_loss(branches, bridge, pair01, floor=0.0).item() < 1e-6
    assert cdl.naturality_loss(branches, bridge, pair01, floor=0.05).item() > 1e-3  # the floor still constrains idle pairs
    w = cdl.pair_weights(pair01, floor=0.0)
    assert torch.allclose(w, w.T) and w[0, 1] == pytest.approx(0.5) and w[2, 3] == 0


def test_morphisms_cannot_collapse_to_zero():
    cat = cdl.routing_world_category()
    branch = cdl.CDLSpecialistBranch(16, 8, 1.0, cat)
    with torch.no_grad():
        for p in branch.morphisms.parameters():
            p.zero_()
    for name in cat.names:  # zeroed parameters still give orthogonal (here identity) morphisms, never 0
        M = branch.matrix(name)
        assert torch.allclose(M.T @ M, torch.eye(8), atol=1e-5) and M.norm() > 1


def test_cdl_model_trains_and_executes_conditionally_exactly():
    w = ti.ToyWorld(n_countries=8, n_fillers=4)
    torch.manual_seed(0)
    model = ti.TypeInformedTreeNN(w.vocab_size, d_model=16, d_stalk=8, d_edge=4, branch_kind="cdl",
                                  bridge_kind="functorial", category=cdl.routing_world_category())
    b = w.sample(64, 6, torch.Generator().manual_seed(9), "cpu")
    out = model.train()(b["ids"])
    losses = model.compute_calibrated_loss(out, b["answer"], b["route"], b["bridge"], 0.5, 0.05)
    assert losses["functorial_loss"].item() > 0
    losses["total_loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.branches[0].morphisms.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.sheaf_bridge.restrict.parameters())
    model.eval()
    with torch.no_grad():
        model.router.noul_bridge_head.bias.fill_(0.3)
        cond, ref = model(b["ids"]), model(b["ids"], reference=True)
    assert cond["branch_execs"] < ref["branch_execs"]
    assert torch.allclose(cond["logits"], ref["logits"], atol=1e-5)


def test_isolation_holds_with_cdl_branches_and_functorial_bridge(world):
    """The routed-training isolation property survives the CDL options: Branch 1's state never shapes
    Branch 0's update, with the naturality loss included."""

    def branch0_grad(shift):
        model = comp_model(world, branch_kind="cdl", bridge_kind="functorial",
                           category=cdl.routing_world_category()).train()
        with torch.no_grad():
            model.router.noul_bridge_head.bias.fill_(5.0)
            model.router.choice_head.bias.copy_(torch.tensor([3.0, 2.0, -5.0, -5.0]))
            if shift:
                d = model.d_model
                model.branch_vectorizer.weight[:, d:2 * d] += torch.randn_like(model.branch_vectorizer.weight[:, d:2 * d])
        b = world.sample(0, 6, torch.Generator().manual_seed(7), "cpu", qtype=0, countries=torch.arange(world.n))
        torch.manual_seed(1)
        out = model(b["ids"], tau=1.0, grad_allow=ti.grad_allow_mask(b, combined_trains_branches=False))
        losses = model.compute_calibrated_loss(out, b["answer"], b["route"], b["bridge"])
        assert "functorial_loss" in losses
        losses["total_loss"].backward()
        return model.branch_vectorizer.weight.grad[:, :model.d_model].clone()

    assert torch.equal(branch0_grad(False), branch0_grad(True))


def test_default_model_has_no_cdl_parts():
    model = ti.TypeInformedTreeNN(50, d_model=16, d_stalk=8, d_edge=4)
    assert not hasattr(model, "morphism_head") and isinstance(model.sheaf_bridge, ti.SheafRestrictionBridge)
    with pytest.raises(ValueError, match="QueryCategory"):
        ti.TypeInformedTreeNN(50, branch_kind="cdl")
