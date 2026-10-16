"""Properties the TI-TNN language-modelling benchmark depends on (research/ti_tnn/bench).

Software checks on tiny random models, not benchmark results.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

BENCH = Path(__file__).resolve().parents[1] / "research" / "ti_tnn" / "bench"
sys.path.insert(0, str(BENCH))
models = importlib.import_module("ti_tnn_models")
spec = importlib.util.spec_from_file_location("ti_tnn_lm", BENCH / "lm.py")
lm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lm)


def cfg(ffn, **kw):
    base = dict(vocab_size=50, context=8, d_model=16, n_layers=2, n_heads=2, ffn=ffn, n_experts=4, d_edge=4)
    base.update(kw)
    return models.ModelConfig(**base)


def expert_out(ffn, x, e):
    return torch.nn.functional.gelu(x @ ffn.experts.w_up[e]) @ ffn.experts.w_down[e]


def test_moe_top1_and_top2_match_a_per_token_reference():
    torch.manual_seed(0)
    x = torch.randn(10, 16)
    for k in (1, 2):
        ffn = models.MoEFFN(cfg(f"moe_top{k}"), k).eval()
        y, stats = ffn(x)
        probs = torch.softmax(ffn.router(x), -1)
        p, e = probs.topk(k, -1)
        p = p / p.sum(-1, keepdim=True) if k > 1 else p
        ref = torch.stack([sum(p[i, j] * expert_out(ffn, x[i], e[i, j]) for j in range(k)) for i in range(10)])
        assert torch.allclose(y, ref, atol=1e-5)
        assert float(stats["expert_runs"]) == k


def test_ti_tnn_without_firing_is_moe_top1():
    torch.manual_seed(1)
    x = torch.randn(12, 16)
    ti = models.TITNNFFN(cfg("ti_tnn")).eval()
    with torch.no_grad():
        ti.noul.bias.fill_(-20.0)  # the gate never fires
    moe = models.MoEFFN(cfg("moe_top1"), 1).eval()
    moe.router.load_state_dict(ti.router.state_dict())
    moe.experts.load_state_dict(ti.experts.state_dict())
    y_ti, stats = ti(x)
    assert torch.equal(y_ti, moe(x)[0]) and float(stats["fire_rate"]) == 0.0


def test_ti_tnn_conditional_inference_matches_a_reference():
    torch.manual_seed(2)
    x = torch.randn(64, 16)
    ti = models.TITNNFFN(cfg("ti_tnn")).eval()
    with torch.no_grad():
        ti.noul.weight.normal_(0, 2.0)  # some tokens fire, some do not
        y, stats = ti(x)
        probs = torch.softmax(ti.router(x), -1)
        p, e = probs.topk(2, -1)
        fire = torch.sigmoid(ti.noul(x)).squeeze(-1) >= 0.5
        ref = []
        for i in range(64):
            o1, o2 = expert_out(ti, x[i], e[i, 0]), expert_out(ti, x[i], e[i, 1])
            out = p[i, 0] * o1
            if fire[i]:
                u, v = (o1, o2) if e[i, 0] < e[i, 1] else (o2, o1)
                out = out + ti.bridge(u, v)
            ref.append(out)
    assert 0 < float(stats["fire_rate"]) < 1
    assert torch.allclose(y, torch.stack(ref), atol=1e-5)
    assert float(stats["expert_runs"]) == pytest.approx(1 + fire.float().mean().item())


def test_ti_tnn_gate_learns_from_tokens_where_it_did_not_fire():
    torch.manual_seed(3)
    x = torch.randn(32, 16)
    ti = models.TITNNFFN(cfg("ti_tnn", bridge_cost=0.0, balance_coef=0.0)).train()
    with torch.no_grad():
        ti.noul.bias.fill_(-20.0)  # nothing fires
    y, stats = ti(x)
    (y ** 2).sum().backward()
    assert float(stats["fire_rate"]) == 0.0 and ti.noul.weight.grad.abs().sum() > 0


def test_bridge_orientation_is_fixed():
    torch.manual_seed(4)
    bridge = models.SheafBridge(16, 4)
    a, b = torch.randn(3, 16), torch.randn(3, 16)
    assert not torch.allclose(bridge(a, b), bridge(b, a))  # the edge has a direction


def test_balance_loss_is_one_when_routing_is_uniform():
    probs = torch.full((8, 4), 0.25)
    top1 = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    assert models.balance_loss(probs, top1, 4).item() == pytest.approx(1.0)


def test_parameter_and_compute_counts():
    for ffn in models.FFN_KINDS:
        model = models.LanguageModel(cfg(ffn))
        counts = models.parameter_counts(model)
        assert counts["active_top1"] <= counts["total"]
        if ffn == "dense":
            assert counts["active_top1"] == counts["total"]
    assert models.ffn_macs_per_token(cfg("dense"), 1) == 8 * 16 ** 2
    assert models.ffn_macs_per_token(cfg("moe_top1"), 1) == 8 * 16 ** 2 + 16 * 4
    assert models.ffn_macs_per_token(cfg("ti_tnn"), 1.5) == 1.5 * 8 * 16 ** 2 + 16 * 4 + 16 + 0.5 * 3 * 16 * 4


def test_lm_trains_and_evaluates_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 4000), ("validation", 600), ("test", 600)):
        rng.integers(0, 32, n).astype(np.uint16).tofile(tmp_path / f"{split}.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"vocab_size": 32}))
    for ffn in models.FFN_KINDS:
        args = lm.parse_args(["--ffn", ffn, "--device", "cpu", "--data", str(tmp_path), "--d-model", "16",
                              "--layers", "1", "--heads", "2", "--context", "16", "--batch", "4", "--tokens", "640",
                              "--experts", "4", "--d-edge", "4", "--log-every", "1000"])
        result = lm.train(args)
        assert result["steps"] == 10 and result["validation"]["windows"] == (600 - 1) // 16
        assert result["validation"]["perplexity"] > 1
        if ffn in models.ADAPTIVE_KINDS:
            assert len(result["validation"]["option_share_by_layer"]) == 1
            assert 0 <= result["validation"]["realized_compute"] <= 1
        elif ffn != "dense":
            assert len(result["validation"]["expert_share_by_layer"]) == 1
        assert result["validation"]["non_embedding_macs_per_token"] > 0


scan_spec = importlib.util.spec_from_file_location("ti_tnn_scan", BENCH / "scan.py")
scan = importlib.util.module_from_spec(scan_spec)
scan_spec.loader.exec_module(scan)


def test_scan_batches_train_only_the_actions():
    vocab = scan.build_vocab(["IN: jump twice OUT: I_JUMP I_JUMP"])
    pairs = scan.encode(["IN: jump twice OUT: I_JUMP I_JUMP", "IN: jump OUT: I_JUMP"], vocab)
    ids, target = scan.make_batch(pairs, "cpu")
    assert ids[0].tolist()[:6] == pairs[0][0] + [scan.SEP] + pairs[0][1] + [scan.EOS]
    assert target[0].tolist()[:2] == [-100, -100] and target[0, 2:5].tolist() == pairs[0][1] + [scan.EOS]
    assert target[1, 1:3].tolist() == pairs[1][1] + [scan.EOS] and (target[1, 3:] == -100).all()


def test_scan_greedy_scoring_counts_only_exact_complete_answers():
    class Echo(torch.nn.Module):
        """Predicts a fixed continuation: the first action, then <eos>."""

        cfg = type("C", (), {"context": 16})()

        def __init__(self, first):
            super().__init__()
            self.first = first

        def forward(self, ids):
            logits = torch.full((ids.shape[0], ids.shape[1], 8), -1e9)
            nxt = torch.where(ids[:, -1] == scan.SEP, torch.tensor(self.first), torch.tensor(scan.EOS))
            logits[torch.arange(len(ids)), -1, nxt] = 0
            return logits, []

    pairs = [([3, 4], [5]), ([3], [5, 5]), ([4, 4], [6])]
    res = scan.greedy_accuracy(Echo(5), pairs, "cpu", batch=2, max_new=4)
    assert res["exact_match"] == pytest.approx(1 / 3) and res["n"] == 3


@pytest.mark.parametrize("ffn", models.FFN_KINDS)
def test_every_variant_trains_under_bf16_autocast(ffn):
    """GPU runs use bf16 autocast; CPU autocast catches dtype mismatches in the expert code."""
    torch.manual_seed(5)
    model = models.LanguageModel(cfg(ffn)).train()
    idx = torch.randint(0, 50, (2, 8))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits, stats = model(idx)
        loss = logits.float().mean() + sum((s["aux_loss"] for s in stats if "aux_loss" in s), torch.zeros(()))
    loss.backward()
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert model(idx)[0].shape == (2, 8, 50)


def test_variant_names_distinguish_the_bridge_charge():
    assert models.variant_name("ti_tnn", 0.0) == "ti_tnn_free"
    assert models.variant_name("ti_tnn", 0.01) == "ti_tnn_charged"
    assert models.variant_name("moe_top2", 0.01) == "moe_top2"


def test_summary_groups_runs_by_variant_and_seed(tmp_path):
    summ_spec = importlib.util.spec_from_file_location("ti_tnn_summarize", BENCH / "summarize.py")
    summ = importlib.util.module_from_spec(summ_spec)
    summ_spec.loader.exec_module(summ)
    for seed, ppl in ((0, 30.0), (1, 32.0)):
        d = tmp_path / "lm" / f"ti_tnn_free-seed{seed}"
        d.mkdir(parents=True)
        d.joinpath("result.json").write_text(json.dumps({
            "config": {"ffn": "ti_tnn", "bridge_cost": 0.0, "seed": seed},
            "validation": {"perplexity": ppl, "expert_runs_per_token": 1.2, "fire_rate": 0.2}, "test": {"perplexity": ppl + 1},
            "compute": {"ffn_macs_per_token_per_layer_inference": 1.0, "ffn_macs_per_token_per_layer_training": 2.0},
            "timing": {"train_tokens_per_second": 100.0}, "parameters": {"total": 10}}))
    s = summ.summarize(*summ.load([tmp_path / "lm"]))
    assert s["lm"]["ti_tnn_free"]["validation_perplexity"]["mean"] == pytest.approx(31.0)
    assert s["lm"]["ti_tnn_free"]["seeds"] == [0, 1]


decode = importlib.import_module("decode")


@pytest.mark.parametrize("ffn", ["dense", "moe_top1", "moe_top2", "mod", "ti_tnn_adaptive"])
def test_token_by_token_decoding_matches_the_full_forward_pass(ffn):
    """Equal logits mean no routing decision or attention read depends on a later token."""
    torch.manual_seed(3)
    model = models.LanguageModel(cfg(ffn, context=12, n_layers=3)).eval()
    if ffn in models.ADAPTIVE_KINDS:  # make every option common
        for block in model.blocks:
            torch.nn.init.normal_(block.router.weight, std=1.0)
    idx = torch.randint(0, 50, (1, 12))
    with torch.no_grad():
        full, stats = model(idx)
    logits, _, taken = decode.decode(model, idx)
    assert torch.allclose(logits, full[0], atol=1e-5)
    if ffn in models.ADAPTIVE_KINDS:
        assert torch.equal(taken, torch.stack([s["choice"][0] for s in stats], 1))
        assert len(taken.unique()) == len(models.option_costs(model.cfg))


@pytest.mark.parametrize("ffn", models.ADAPTIVE_KINDS)
def test_a_skipped_token_is_unchanged_and_invisible_to_later_tokens(ffn):
    torch.manual_seed(4)
    c = cfg(ffn, n_layers=1)
    block = models.AdaptiveBlock(c).eval()
    x = torch.randn(1, 6, 16)
    x[..., 0] = 0.0
    x2 = x.clone()
    x2[0, 2, 0] = 5.0  # only token 2 differs, and only in the feature the skip logit reads
    with torch.no_grad():
        block.router.weight.zero_()
        block.router.bias.zero_()
        block.router.bias[-1] = 1.0   # the full branch, unless ...
        block.router.weight[0, 0] = 1.0  # ... the skip logit x[0] beats it
        y_all, s_all = block(x)
        y, s = block(x2)
    assert (s_all["choice"] == len(models.option_costs(c)) - 1).all()
    assert s["choice"][0].tolist() == [s_all["choice"][0, 0].item()] * 2 + [0] + [s_all["choice"][0, 0].item()] * 3
    assert torch.equal(y[0, 2], x2[0, 2])  # skipped: the residual passes through
    assert torch.allclose(y[0, :2], y_all[0, :2], atol=1e-6)  # earlier tokens never see later ones
    assert not torch.allclose(y[0, 3:], y_all[0, 3:])  # later tokens lost token 2 as a key


@pytest.mark.parametrize("ffn", models.ADAPTIVE_KINDS)
def test_straight_through_router_learns_from_every_option(ffn):
    """Even tokens that skipped send the router a gradient (no absorbing skip)."""
    torch.manual_seed(5)
    block = models.AdaptiveBlock(cfg(ffn, n_layers=1)).train()
    with torch.no_grad():
        block.router.weight.zero_()
        block.router.bias.zero_()
        block.router.bias[0] = 3.0  # everyone skips
    y, s = block(torch.randn(2, 5, 16))
    assert (s["choice"] == 0).all()
    (y ** 2).sum().backward()
    assert block.router.weight.grad.abs().sum() > 0
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in block.branches.parameters())  # unused branches learn nothing


def test_compute_accounting_matches_the_analytic_count():
    c = cfg("dense", context=8)
    model = models.LanguageModel(c).eval()
    idx = torch.randint(0, 50, (2, 8))
    with torch.no_grad():
        _, stats = model(idx)
    got = models.macs_per_token(c, stats, (2, 8))["non_embedding"]
    assert got == pytest.approx(models.full_block_macs_per_token(c, 8))
    a = cfg("ti_tnn_adaptive", context=8)
    assert models.option_costs(a) == [0.0, 0.5, 1.0] and models.option_costs(cfg("mod")) == [0.0, 1.0]
    stats = [{"choice": torch.zeros(2, 8, dtype=torch.long)}] * 2  # all skip: only the compute routers run
    assert models.macs_per_token(a, stats, (2, 8))["non_embedding"] == 2 * 16 * 3
    full = [{"choice": torch.full((2, 8), 2)}] * 2  # all full: dense cost plus the routers
    assert models.macs_per_token(a, full, (2, 8))["non_embedding"] == pytest.approx(
        models.full_block_macs_per_token(a, 8) + 2 * (16 * 3 + 16 * 4))


def test_dual_ascent_prices_compute_only_while_the_cap_binds():
    model = models.LanguageModel(cfg("mod", target_compute=0.5, dual_lr=0.1))
    over = [{"realized_cost": torch.tensor(0.9)}] * 2
    under = [{"realized_cost": torch.tensor(0.1)}] * 2
    model.dual_update(over)
    assert float(model.cost_price) == pytest.approx(0.04)
    for _ in range(3):
        model.dual_update(under)
    assert float(model.cost_price) == 0.0
    model.dual_update(over)
    assert float(model.cost_price) > 0


def test_default_sizes_are_unchanged():
    """The reference variants keep their shapes, so earlier runs remain valid baselines."""
    big = dict(vocab_size=50257, context=512, d_model=512, n_layers=8, n_heads=8)
    for ffn, total in (("dense", 51_176_960), ("moe_top1", 168_650_240), ("moe_top2", 168_650_240)):  # PR #33 runs
        assert models.parameter_counts(models.LanguageModel(models.ModelConfig(ffn=ffn, **big)))["total"] == total


hops_spec = importlib.util.spec_from_file_location("ti_tnn_hops", BENCH / "hops.py")
hops = importlib.util.module_from_spec(hops_spec)
sys.modules["ti_tnn_hops"] = hops  # dataclasses resolve their module through sys.modules
hops_spec.loader.exec_module(hops)


def follow(seq, x, k):
    """Answer a query by reading the facts stated in this sequence."""
    task = hops.Task()
    f = {int(seq[1 + 2 * i]): int(seq[2 + 2 * i]) for i in range(task.facts)}
    for _ in range(k):
        x = f[x]
    return x


def test_hop_sequences_are_answerable_from_their_own_facts():
    task = hops.Task()
    b = hops.make_batch(16, task, torch.Generator().manual_seed(0))
    assert b["tokens"].shape == (16, task.length) and int(b["tokens"].max()) < task.vocab
    q = hops.KIND_NAMES.index("query")
    easy = hops.KIND_NAMES.index("easy")
    for i in range(16):
        seq, tgt, kind, k = b["tokens"][i], b["targets"][i], b["kind"][i], b["hops"][i]
        assert int((kind == q).sum()) == task.queries
        for p in (kind == q).nonzero().flatten().tolist():
            hop = int(seq[p - 1]) - (1 + task.symbols) + 1  # the H_k token just before
            assert hop == int(k[p]) and 1 <= hop <= task.max_hops
            assert int(tgt[p]) == follow(seq, int(seq[p]), hop) != int(seq[p])
        for p in (kind == easy).nonzero().flatten().tolist():  # easy: the next count, from this token alone
            c = int(seq[p]) - (1 + task.symbols + task.max_hops)
            assert int(tgt[p]) - (1 + task.symbols + task.max_hops) == (c + 1) % task.counts
        assert (tgt[kind != q][kind[kind != q] != easy] == -100).all()  # facts and hop tokens are not scored


def test_hop_eval_set_is_fixed_and_counts_compute_by_kind():
    args = hops.parse_args(["--ffn", "ti_tnn_adaptive", "--device", "cpu", "--d-model", "16", "--layers", "2",
                            "--heads", "2", "--experts", "2", "--steps", "3", "--batch", "4", "--eval-sequences", "8"])
    r1 = hops.train(args)
    r2 = hops.train(args)
    assert r1["eval"] == r2["eval"]  # same seed, same fixed evaluation sequences
    e = r1["eval"]
    assert set(e["accuracy_by_hops"]) == {"1", "2", "3"} and 0 <= e["hard_accuracy"] <= 1
    assert set(e["relative_compute_by_kind"]) == set(hops.KIND_NAMES)
    dense = hops.train(hops.parse_args(["--ffn", "dense", "--device", "cpu", "--d-model", "16", "--layers", "2",
                                        "--heads", "2", "--steps", "2", "--batch", "4", "--eval-sequences", "8"]))
    assert dense["eval"]["relative_compute"] == 1.0
    assert dense["eval"]["non_embedding_macs_per_token"] == pytest.approx(dense["same_size_dense_non_embedding_macs_per_token"])


frontier = importlib.import_module("frontier")


def point(kind, quality, compute):
    return {"kind": kind, "quality": {"mean": quality}, "compute": {"mean": compute}}


def test_pass_bar_requires_half_the_compute_of_the_cheapest_baseline_as_good():
    lower_is_better = False  # perplexity
    base = {"dense": point("dense", 40.0, 100.0), "dense_d256_l8": point("dense", 44.0, 30.0),
            "mod_c30": point("mod", 41.5, 35.0)}
    # 40.5 is within 2% of dense; the cheapest baseline at least as good (perplexity <= 40.5) is dense at 100
    win = dict(base, ti_tnn_adaptive_c30=point("ti_tnn_adaptive", 40.5, 45.0))
    j = frontier.judge(win, "dense", lower_is_better, 0.02, True)
    assert j["efficiency"]["holds"] and j["matched_cap"]["holds"] and j["equal_compute"]["holds"]
    # the same quality at 60 is only 1.7x cheaper: not drastic
    j = frontier.judge(dict(base, ti_tnn_adaptive_c30=point("ti_tnn_adaptive", 40.5, 60.0)), "dense", lower_is_better, 0.02, True)
    assert not j["efficiency"]["holds"] and j["equal_compute"]["holds"]
    # cheap but not close: 42.0 misses the 2% band, and loses to mod at the same cap
    j = frontier.judge(dict(base, ti_tnn_adaptive_c30=point("ti_tnn_adaptive", 42.0, 33.0)), "dense", lower_is_better, 0.02, True)
    assert not j["efficiency"]["holds"] and not j["matched_cap"]["holds"]
    assert j["equal_compute"]["per_candidate"]["ti_tnn_adaptive_c30"]["holds"]  # beats the d256 model at <= 33
    # below every baseline's compute: equal-compute is untestable, not passed
    j = frontier.judge(dict(base, ti_tnn_adaptive_c30=point("ti_tnn_adaptive", 45.0, 10.0)), "dense", lower_is_better, 0.02, True)
    assert j["equal_compute"]["per_candidate"]["ti_tnn_adaptive_c30"]["holds"] is None and j["equal_compute"]["holds"] is None


def test_pass_bar_for_accuracy_uses_an_absolute_band():
    v = {"dense": point("dense", 0.99, 100.0), "dense_d128_l4": point("dense", 0.60, 50.0),
         "ti_tnn_adaptive_c30": point("ti_tnn_adaptive", 0.975, 20.0)}
    j = frontier.judge(v, "dense", True, 0.02, False)
    assert j["close_threshold"] == pytest.approx(0.97) and j["efficiency"]["holds"]
    assert j["efficiency"]["per_candidate"]["ti_tnn_adaptive_c30"]["cheapest_baseline_as_good"] == 100.0


def test_untestable_comparisons_never_count_as_passed():
    """Review fixture: candidate A beats the small baseline, B is cheaper than every baseline."""
    v = {"dense": point("dense", 40.0, 100.0), "small": point("dense", 50.0, 20.0),
         "ti_tnn_adaptive_c50": point("ti_tnn_adaptive", 45.0, 30.0),
         "ti_tnn_adaptive_c30": point("ti_tnn_adaptive", 39.0, 10.0)}
    j = frontier.judge(v, "dense", False, 0.02, True)
    eq = j["equal_compute"]["per_candidate"]
    assert eq["ti_tnn_adaptive_c50"]["holds"] is True and eq["ti_tnn_adaptive_c30"]["holds"] is None
    assert j["equal_compute"]["holds"] is None  # one untestable candidate: "every candidate" is untestable
    assert j["matched_cap"]["holds"] is None  # no mod twins at all
    assert frontier.all_of([True, False, None]) is False and frontier.all_of([True, None]) is None
    assert frontier.all_of([True, True]) is True and frontier.all_of([]) is None


def test_inventory_requires_every_variant_with_three_distinct_seeds():
    full = {n: {"seeds": [0, 1, 2]} for n in frontier.EXPECTED["lm"]}
    assert frontier.inventory(full, "lm")["complete"]
    dup = dict(full, mod_c30={"seeds": [0, 1, 1]})
    assert not frontier.inventory(dup, "lm")["complete"]
    short = {n: v for n, v in full.items() if n != "ti_tnn_adaptive_c30"}
    assert frontier.inventory(short, "lm")["missing"] == ["ti_tnn_adaptive_c30"]


@pytest.mark.parametrize("ffn", ["moe_top1", "moe_top2", "ti_tnn_adaptive"])
def test_cached_decoding_matches_the_full_pass_under_bf16(ffn):
    """The precision latency is measured in: routing decisions agree and logits stay close."""
    torch.manual_seed(6)
    model = models.LanguageModel(cfg(ffn, context=12, n_layers=3)).eval()
    if ffn in models.ADAPTIVE_KINDS:
        for block in model.blocks:
            torch.nn.init.normal_(block.router.weight, std=1.0)
    idx = torch.randint(0, 50, (1, 12))
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        full, stats = model(idx)
    logits, _, taken = decode.decode(model, idx, autocast=True)
    assert (full[0].float() - logits).abs().max() < 0.05
    if ffn in models.ADAPTIVE_KINDS:
        assert torch.equal(taken, torch.stack([s["choice"][0] for s in stats], 1))


def test_moe_weight_is_rounded_like_the_full_pass():
    """Multiplying a bf16 output by a Python float rounds differently from the full pass's bf16 weight."""
    torch.manual_seed(7)
    ffn = models.MoEFFN(cfg("moe_top2"), 2).eval()
    x = torch.randn(1, 16)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y_full, _ = ffn(x)
        y_step = decode.moe_step(ffn, x)
    assert y_full.dtype == y_step.dtype and torch.equal(y_full, y_step)
