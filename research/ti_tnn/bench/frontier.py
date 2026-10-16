"""Quality-against-compute frontier and the frozen pass bar (TI-TNN adaptive-compute benchmark).

Reads part A (hops.py) and part B (lm.py) results, averages each variant over seeds and applies the
predictions of research/ti_tnn/protocols/adaptive_compute.md exactly as written there. Compute is
non-embedding forward multiply-accumulates per token, from the routing decisions each model actually
made on the evaluation data (runs from the earlier benchmark are counted with the same formula).

    .venv-phase2/bin/python research/ti_tnn/bench/frontier.py --hops runs/titnn_efficiency/hops \\
        --lm runs/titnn_bench/lm runs/titnn_efficiency/lm --latency runs/titnn_efficiency/latency.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch

import ti_tnn_models as models  # research/ti_tnn/bench is on sys.path when this file runs as a script

CANDIDATE = "ti_tnn_adaptive"
LM_KINDS = ("dense", "moe_top1", "moe_top2", "mod", "ti_tnn_adaptive")  # the earlier ti_tnn bridge runs are not in the frontier


def stat(values):
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values), "values": values}


def lm_compute(r: dict) -> float:
    """Validation non-embedding MACs per token; recomputed for runs that predate the field."""
    v = r["validation"]
    if "non_embedding_macs_per_token" in v:
        return v["non_embedding_macs_per_token"]
    cfg = models.ModelConfig(**r["model_config"])
    runs = v.get("expert_runs_per_token", 1.0)
    full = [{"expert_runs": torch.tensor(runs)} if cfg.ffn != "dense" else {}] * cfg.n_layers
    return models.macs_per_token(cfg, full, (1, cfg.context))["non_embedding"]


def name_of(r: dict, reference) -> str:
    c, m = r["config"], r["model_config"]
    return models.variant_name(c["ffn"], c.get("bridge_cost", 0.01), m.get("target_compute", 1.0),
                               models.size_tag(m["d_model"], m["n_layers"], m.get("ffn_mult", 4), reference))


def load(roots, part: str) -> dict:
    groups = defaultdict(list)
    for root in roots:
        for path in sorted(Path(root).glob("*/result.json")):
            r = json.loads(path.read_text())
            if part == "lm" and r["config"]["ffn"] not in LM_KINDS:
                continue
            ref = (128, 8, 4) if part == "hops" else (512, 8, 4)
            groups[name_of(r, ref)].append(r)
    out = {}
    for name, runs in groups.items():
        runs.sort(key=lambda r: r["config"]["seed"])
        if part == "hops":
            quality = [r["eval"]["hard_accuracy"] for r in runs]
            compute = [r["eval"]["non_embedding_macs_per_token"] for r in runs]
            hop_keys = sorted(runs[0]["eval"]["accuracy_by_hops"], key=int)
            extra = {"by_hops": {h: stat([r["eval"]["accuracy_by_hops"][h] for r in runs]) for h in hop_keys},
                     "easy_accuracy": stat([r["eval"]["easy_accuracy"] for r in runs]),
                     "relative_compute_by_hops": {h: stat([r["eval"]["relative_compute_by_hops"][h] for r in runs])
                                                  for h in hop_keys},
                     "relative_compute_by_kind": {k: stat([r["eval"]["relative_compute_by_kind"][k] for r in runs])
                                                  for k in runs[0]["eval"]["relative_compute_by_kind"]}}
        else:
            quality = [r["validation"]["perplexity"] for r in runs]
            compute = [lm_compute(r) for r in runs]
            extra = {"test_perplexity": stat([r["test"]["perplexity"] for r in runs]),
                     "realized_compute": stat([r["validation"].get("realized_compute", 1.0) for r in runs]),
                     "parameters": runs[0]["parameters"]["total"]}
        out[name] = {"kind": runs[0]["config"]["ffn"], "seeds": [r["config"]["seed"] for r in runs],
                     "quality": stat(quality), "compute": stat(compute), **extra}
    return out


def better(a: float, b: float, higher_is_better: bool) -> bool:
    return a > b if higher_is_better else a < b


def at_least_as_good(a: float, b: float, higher_is_better: bool) -> bool:
    return a >= b if higher_is_better else a <= b


def judge(variants: dict, reference: str, higher_is_better: bool, close: float, close_is_ratio: bool) -> dict:
    """The protocol's efficiency (P-eff), equal-compute (P-eq) and matched-cap (P-mod) predictions."""
    base = {n: v for n, v in variants.items() if v["kind"] != CANDIDATE}
    cands = {n: v for n, v in variants.items() if v["kind"] == CANDIDATE}
    q = lambda v: v["quality"]["mean"]
    c = lambda v: v["compute"]["mean"]
    ref_q = q(variants[reference])
    threshold = ref_q * (1 + close) if close_is_ratio else ref_q - close
    eff, eq, cap = {}, {}, {}
    for n, v in cands.items():
        reach = [b for b in base.values() if at_least_as_good(q(b), q(v), higher_is_better)]
        cheapest = min((c(b) for b in reach), default=None)
        if cheapest is None:  # no baseline is as good: compare with the best baseline's compute
            best = max(base.values(), key=lambda b: q(b) if higher_is_better else -q(b))
            cheapest = c(best)
        is_close = at_least_as_good(q(v), threshold, higher_is_better)
        eff[n] = {"quality": q(v), "compute": c(v), "close_to_reference": is_close,
                  "cheapest_baseline_as_good": cheapest, "compute_ratio": c(v) / cheapest,
                  "holds": bool(is_close and c(v) <= 0.5 * cheapest)}
        cheaper = [b for b in base.values() if c(b) <= c(v)]
        if not cheaper:
            eq[n] = {"holds": None, "note": "no baseline uses this little compute"}
        else:
            best = max(cheaper, key=lambda b: q(b) if higher_is_better else -q(b))
            eq[n] = {"holds": bool(better(q(v), q(best), higher_is_better)), "best_baseline_at_or_below": q(best)}
        twin = n.replace(CANDIDATE, "mod")
        if twin in base:
            cap[n] = {"mod": q(base[twin]), "holds": bool(better(q(v), q(base[twin]), higher_is_better))}
    return {"reference": reference, "reference_quality": ref_q, "close_threshold": threshold,
            "efficiency": {"holds": any(e["holds"] for e in eff.values()), "per_candidate": eff},
            "equal_compute": {"holds": None if all(e["holds"] is None for e in eq.values())
                              else all(e["holds"] is not False for e in eq.values()), "per_candidate": eq},
            "matched_cap": {"holds": all(e["holds"] for e in cap.values()) if cap else None, "per_candidate": cap}}


def difficulty_tracking(variants: dict) -> dict:
    """Part A: in every candidate, the longest queries get more compute than 1-hop ones, and easy tokens less than any query."""
    out = {}
    for n, v in variants.items():
        if v["kind"] != CANDIDATE:
            continue
        h = {k: s["mean"] for k, s in v["relative_compute_by_hops"].items()}
        easy = v["relative_compute_by_kind"]["easy"]["mean"]
        longest = max(h, key=int)
        out[n] = {"compute_by_hops": h, "easy": easy, "holds": bool(h[longest] > h["1"] and easy < min(h.values()))}
    return {"holds": all(o["holds"] for o in out.values()) if out else None, "per_candidate": out}


def table(variants: dict, quality: str, fmt: str) -> str:
    rows = sorted(variants.items(), key=lambda kv: kv[1]["compute"]["mean"])
    lines = [f"| Variant | {quality} | Non-embedding MACs per token | Seeds |", "|---|---:|---:|---:|"]
    for n, v in rows:
        lines.append(f"| `{n}` | {v['quality']['mean']:{fmt}} ± {v['quality']['sd']:{fmt}} | "
                     f"{v['compute']['mean'] / 1e6:.2f}M | {len(v['seeds'])} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hops", nargs="*", type=Path, default=[])
    p.add_argument("--lm", nargs="*", type=Path, default=[])
    p.add_argument("--latency", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    summary = {}
    if args.hops:
        A = load(args.hops, "hops")
        summary["hops"] = {"variants": A, "predictions": judge(A, "dense", True, 0.02, False),
                           "difficulty_tracking": difficulty_tracking(A)}
        print("## Part A: variable-hop lookups (hard-query accuracy)\n\n" + table(A, "Hard accuracy", ".3f") + "\n")
    if args.lm:
        B = load(args.lm, "lm")
        summary["lm"] = {"variants": B, "predictions": judge(B, "dense", False, 0.02, True)}
        if args.latency and args.latency.exists():
            lat = json.loads(args.latency.read_text())["runs"]
            by = defaultdict(list)
            for row in lat:
                by[row["name"]].append(row["ms_per_token_median"])
            summary["lm"]["latency_ms_per_token"] = {n: stat(v) for n, v in by.items()}
            dense = statistics.fmean(by["dense"]) if by.get("dense") else None
            c30 = by.get(f"{CANDIDATE}_c30")
            summary["lm"]["predictions"]["latency"] = {
                "holds": bool(dense and c30 and statistics.fmean(c30) <= 0.6 * dense),
                "candidate_ms": statistics.fmean(c30) if c30 else None, "dense_ms": dense}
        print("## Part B: WikiText-103 (validation perplexity)\n\n" + table(B, "Perplexity", ".2f") + "\n")
    for part, s in summary.items():
        print(f"{part}: " + json.dumps({k: (v["holds"] if isinstance(v, dict) and "holds" in v else v)
                                         for k, v in s["predictions"].items() if k not in ("reference", "reference_quality",
                                                                                           "close_threshold")}))
        if "difficulty_tracking" in s:
            print(f"{part}: difficulty tracking holds = {s['difficulty_tracking']['holds']}")
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
