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
        if ffn != "dense":
            assert len(result["validation"]["expert_share_by_layer"]) == 1


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
