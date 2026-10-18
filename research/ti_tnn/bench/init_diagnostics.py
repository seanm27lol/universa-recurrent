"""Initial residual-stream RMS and routing probabilities under each expert initialization (init control).

Each model is built exactly as lm.py builds it for that seed (torch.manual_seed(seed), then the model),
and fed lm.py's first training batch for that seed. So this reconstructs the initial state of runs that
predate lm.py's own init_diagnostics record (the legacy-initialization runs), and checks it for the
control runs. No training happens here.

    .venv-phase2/bin/python research/ti_tnn/bench/init_diagnostics.py --out runs/titnn_efficiency/init_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lm  # noqa: E402
import ti_tnn_models as models  # noqa: E402

FAMILIES = {"dense": ("dense", 1.0), "mod_c30": ("mod", 0.3), "moe_top1": ("moe_top1", 1.0),
            "ti_tnn_adaptive_c30": ("ti_tnn_adaptive", 0.3)}


def diagnose(ffn: str, target: float, init: str, seed: int, args, device) -> dict:
    meta = json.loads((args.data / "meta.json").read_text())
    a = lm.parse_args(["--ffn", ffn, "--target-compute", str(target), "--seed", str(seed), "--expert-init", init,
                       "--data", str(args.data)])
    torch.manual_seed(a.seed)
    cfg = models.ModelConfig(vocab_size=meta["vocab_size"], context=a.context, d_model=a.d_model, n_layers=a.layers,
                             n_heads=a.heads, ffn=a.ffn, n_experts=a.experts, d_edge=a.d_edge,
                             balance_coef=a.balance_coef, bridge_cost=a.bridge_cost, ffn_mult=a.ffn_mult,
                             small_mult=a.small_mult, target_compute=a.target_compute, dual_lr=a.dual_lr,
                             expert_init=a.expert_init)
    model = models.LanguageModel(cfg).to(device)
    gen = torch.Generator().manual_seed(1_000 + seed)  # lm.py's data order
    x, _ = lm.sample_batch(lm.load_split(args.data, "train"), a.batch, a.context, gen, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return models.init_diagnostics(model, x)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=Path("runs/titnn_data/wikitext103"))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out = []
    for name, (ffn, target) in FAMILIES.items():
        inits = ("legacy",) if ffn in ("dense", "mod") else ("legacy", "dense")  # no experts, nothing to vary
        for init in inits:
            for seed in args.seeds:
                d = diagnose(ffn, target, init, seed, args, device)
                out.append({"family": name, "expert_init": init, "seed": seed, **d})
                print(f"{name:20s} {init:6s} seed {seed}: final residual RMS {d['residual_rms_by_layer'][-1]:.3f}"
                      + (f", mean top probability {sum(d['mean_top_probability_by_layer']) / len(d['mean_top_probability_by_layer']):.3f}"
                         if "mean_top_probability_by_layer" in d else ""), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"device": str(device), "rows": out}, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
