"""Measure batch-1 decoding latency of trained WikiText-103 runs (TI-TNN adaptive-compute benchmark).

Every model decodes the same validation windows one token at a time (decode.py), so a skipped block
really is skipped. Variants are measured in interleaved rounds, so drift in the GPU's state affects all
of them alike. Mixture and dense runs keep no weights; their latency does not depend on the weights
(top-k picks some expert either way), so they are timed at a random initialization of the same
architecture. Adaptive runs load their trained weights (model.pt), because which blocks they skip
depends on what they learned.

    .venv-phase2/bin/python research/ti_tnn/bench/latency.py runs/titnn_bench/lm runs/titnn_efficiency/lm
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import decode
import ti_tnn_models as models  # research/ti_tnn/bench is on sys.path when this file runs as a script

VARIANT_KINDS = ("dense", "moe_top1", "moe_top2", "mod", "ti_tnn_adaptive")


def run_name(r: dict) -> str:
    c, m = r["config"], r["model_config"]
    return models.variant_name(c["ffn"], c.get("bridge_cost", 0.01), m.get("target_compute", 1.0),
                               models.size_tag(m["d_model"], m["n_layers"], m.get("ffn_mult", 4)))


def collect(roots) -> list:
    """One entry per adaptive run (its own weights) and one per other variant (weights do not matter)."""
    entries, seen = [], set()
    for root in roots:
        for path in sorted(Path(root).glob("*/result.json")):
            r = json.loads(path.read_text())
            if r["config"]["ffn"] not in VARIANT_KINDS:
                continue
            name = run_name(r)
            weights = path.parent / "model.pt"
            if r["config"]["ffn"] in models.ADAPTIVE_KINDS:
                if not weights.exists():
                    raise FileNotFoundError(f"{weights}: adaptive runs need their trained weights")
                entries.append({"name": name, "seed": r["config"]["seed"], "dir": str(path.parent), "result": r})
            elif name not in seen:
                seen.add(name)
                entries.append({"name": name, "seed": None, "dir": str(path.parent), "result": r})
    return entries


def build(entry: dict, device) -> models.LanguageModel:
    cfg = models.ModelConfig(**entry["result"]["model_config"])
    torch.manual_seed(0)
    model = models.LanguageModel(cfg)
    weights = Path(entry["dir"]) / "model.pt"
    if cfg.ffn in models.ADAPTIVE_KINDS:
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.to(device).eval()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--data", type=Path, default=Path("runs/titnn_data/wikitext103"))
    p.add_argument("--windows", type=int, default=2, help="validation windows decoded per round")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup-tokens", type=int, default=8, help="tokens per window excluded from timing")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("runs/titnn_efficiency/latency.json"))
    args = p.parse_args(argv)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    entries = collect(args.roots)
    tokens = np.memmap(args.data / "validation.bin", dtype=np.uint16, mode="r")
    ctx = entries[0]["result"]["model_config"]["context"]
    windows = [torch.from_numpy(tokens[i * ctx:(i + 1) * ctx].astype(np.int64))[None].to(device)
               for i in range(args.windows)]
    times = {i: [] for i in range(len(entries))}
    agreement = {}
    for rnd in range(args.rounds):
        for i, entry in enumerate(entries):
            model = build(entry, device)
            for w in windows:
                logits, seconds, taken = decode.decode(model, w, timed=True, autocast=device.type == "cuda")
                times[i].extend(seconds[args.warmup_tokens:])
                if rnd == 0 and w is windows[0] and model.cfg.ffn in models.ADAPTIVE_KINDS:
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                         enabled=device.type == "cuda"):
                        _, stats = model(w)
                    full = torch.stack([s["choice"][0] for s in stats], 1).cpu()
                    agreement[i] = {"decision_agreement": float((full == taken).float().mean()),
                                    "decoded_option_share": torch.bincount(taken.reshape(-1), minlength=len(
                                        models.option_costs(model.cfg))).div(taken.numel()).tolist()}
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"round {rnd + 1}/{args.rounds} done", flush=True)
    out = []
    for i, entry in enumerate(entries):
        ms = [1e3 * s for s in times[i]]
        row = {"name": entry["name"], "seed": entry["seed"], "dir": entry["dir"],
               "ms_per_token_mean": statistics.fmean(ms), "ms_per_token_median": statistics.median(ms),
               "timed_tokens": len(ms)}
        row.update(agreement.get(i, {}))
        out.append(row)
        print(f"{entry['name']:>28} seed {entry['seed']}: {row['ms_per_token_median']:.3f} ms/token (median)")
    env = {"torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": {k: str(v) for k, v in vars(args).items()}, "environment": env,
                                    "runs": out}, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
