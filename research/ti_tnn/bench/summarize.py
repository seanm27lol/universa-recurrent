"""Summarize the TI-TNN real-benchmark runs (result.json files) into tables and a JSON summary.

    python research/ti_tnn/bench/summarize.py runs/titnn_bench/lm runs/titnn_bench/scan --out summary.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import ti_tnn_models as models  # research/ti_tnn/bench is on sys.path when this file runs as a script

ORDER = ["dense", "moe_top1", "moe_top2", "ti_tnn_charged", "ti_tnn_free"]


def stat(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values), "values": values}


def load(roots):
    lm, scan = defaultdict(list), defaultdict(list)
    for root in roots:
        for path in sorted(Path(root).glob("*/result.json")):
            r = json.loads(path.read_text())
            c = r["config"]
            name = models.variant_name(c["ffn"], c["bridge_cost"])
            r["_path"] = str(path)
            (scan[(c["split"], name)] if "split" in c else lm[name]).append(r)
    return lm, scan


def summarize(lm, scan) -> dict:
    out = {"lm": {}, "scan": {}}
    for name, runs in lm.items():
        runs = sorted(runs, key=lambda r: r["config"]["seed"])
        out["lm"][name] = {
            "seeds": [r["config"]["seed"] for r in runs],
            "validation_perplexity": stat([r["validation"]["perplexity"] for r in runs]),
            "test_perplexity": stat([r["test"]["perplexity"] for r in runs]),
            "inference_expert_runs_per_token": stat([r["validation"].get("expert_runs_per_token", 1.0) for r in runs]),
            "fire_rate": stat([r["validation"].get("fire_rate") for r in runs]),
            "ffn_macs_inference": stat([r["compute"]["ffn_macs_per_token_per_layer_inference"] for r in runs]),
            "ffn_macs_training": stat([r["compute"]["ffn_macs_per_token_per_layer_training"] for r in runs]),
            "train_tokens_per_second": stat([r["timing"]["train_tokens_per_second"] for r in runs]),
            "parameters_total": runs[0]["parameters"]["total"],
            "runs": [r["_path"] for r in runs],
        }
    for (split, name), runs in scan.items():
        runs = sorted(runs, key=lambda r: r["config"]["seed"])
        out["scan"].setdefault(split, {})[name] = {
            "seeds": [r["config"]["seed"] for r in runs],
            "test_exact_match": stat([r["test"]["exact_match"] for r in runs]),
            "train_sample_exact_match": stat([r["train_sample"]["exact_match"] for r in runs]),
            "fire_rate": stat([r["test"].get("fire_rate") for r in runs]),
            "runs": [r["_path"] for r in runs],
        }
    return out


def fmt(s, digits=2):
    return "n/a" if s is None else (f"{s['mean']:.{digits}f} ± {s['sd']:.{digits}f} (n={s['n']})")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("roots", nargs="+")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    summary = summarize(*load(args.roots))
    if summary["lm"]:
        print("WikiText-103 (lower perplexity is better)")
        for name in [n for n in ORDER if n in summary["lm"]]:
            s = summary["lm"][name]
            print(f"  {name:15s} val ppl {fmt(s['validation_perplexity'])}  test ppl {fmt(s['test_perplexity'])}  "
                  f"expert runs/token {fmt(s['inference_expert_runs_per_token'], 3)}  "
                  f"tokens/s {fmt(s['train_tokens_per_second'], 0)}")
    for split, by_variant in sorted(summary["scan"].items()):
        print(f"SCAN {split} (test exact match)")
        for name in [n for n in ORDER if n in by_variant]:
            s = by_variant[name]
            print(f"  {name:15s} {fmt(s['test_exact_match'], 4)}  train sample {fmt(s['train_sample_exact_match'], 4)}")
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
