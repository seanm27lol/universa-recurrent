"""Rebuild the TI-TNN reports' tables from this evidence bundle alone.

Checks every file against manifest.json first (sha256), then prints:
- the toy benchmarks' held-out results;
- the real benchmarks, recomputed with research/ti_tnn/bench/summarize.py and checked against the
  bundled summary.json;
- both composition runs, as exact correct/total counts per path;
- the adaptive-compute verdict, recomputed with research/ti_tnn/bench/frontier.py from the bundled runs
  and checked against the bundled frontier.json. Every hash in that file's run manifest must match a
  bundled run, so the bundle holds exactly the files that were evaluated.

    python experiments/results/ti_tnn/make_tables.py            # from the repository root
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[2] / "research" / "ti_tnn" / "bench"


def verify(bundle: Path = HERE) -> dict:
    """Every listed file exists with its recorded hash, and no JSON file is unlisted."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    listed = {f["path"] for f in manifest["files"]}
    for f in manifest["files"]:
        digest = hashlib.sha256((bundle / f["path"]).read_bytes()).hexdigest()
        if digest != f["sha256"]:
            raise ValueError(f"{f['path']}: sha256 {digest} does not match the manifest")
    stray = {str(p.relative_to(bundle)) for p in bundle.rglob("*.json")} - listed - {"manifest.json"}
    if stray:
        raise ValueError(f"files not in the manifest: {sorted(stray)}")
    return manifest


def mean_sd(values):
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def toy_tables(bundle: Path = HERE) -> dict:
    comp = json.loads((bundle / "toy_benchmarks/20261015T002259/compositional.json").read_text())
    route = json.loads((bundle / "toy_benchmarks/20261015T002259/routing.json").read_text())
    unsup = json.loads((bundle / "unsupervised_routing/20261015T152306/unsupervised.json").read_text())
    by = lambda doc: defaultdict(list, {})
    out = {"compositional_heldout_xor": {}, "routing_task_accuracy": {}, "unsupervised": {}}
    groups = defaultdict(list)
    for r in comp["runs"]:
        groups[r["variant"]].append(r["eval"]["combined_test"]["acc"])
    out["compositional_heldout_xor"] = {v: mean_sd(a) for v, a in groups.items()}
    groups = defaultdict(list)
    for r in route["runs"]:
        groups[r["variant"]].append(r["eval"]["task_acc"])
    out["routing_task_accuracy"] = {v: mean_sd(a) for v, a in groups.items()}
    groups = defaultdict(list)
    for r in unsup["runs"]:
        groups[r["variant"]].append((r["structure"]["task_acc"], r["structure"]["purity"], r["structure"]["bridge_fire_rate"]))
    out["unsupervised"] = {v: {"task_accuracy": mean_sd([x[0] for x in a]), "purity": mean_sd([x[1] for x in a]),
                               "bridge_fire_rate": mean_sd([x[2] for x in a])} for v, a in groups.items()}
    return out


def strip_paths(summary):
    """summarize.py records each run's own path; compare everything else."""
    if isinstance(summary, dict):
        return {k: strip_paths(v) for k, v in summary.items() if k != "runs"}
    return summary


def real_benchmark_tables(bundle: Path = HERE) -> dict:
    sys.path.insert(0, str(BENCH))
    import summarize  # research/ti_tnn/bench/summarize.py
    lm, scan = summarize.load([bundle / "real_benchmarks/lm", bundle / "real_benchmarks/scan"])
    rebuilt = summarize.summarize(lm, scan)
    recorded = json.loads((bundle / "real_benchmarks/summary.json").read_text())
    if strip_paths(rebuilt) != strip_paths(recorded):
        raise ValueError("the rebuilt real-benchmark summary differs from the bundled summary.json")
    return rebuilt


def composition_tables(bundle: Path = HERE) -> dict:
    out = {}
    for name in ("functorial_composition/20261016T152240", "functorial_composition_controls/20261016T173618"):
        doc = json.loads((bundle / name / "composition.json").read_text())
        counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        acc = defaultdict(lambda: defaultdict(list))
        for r in doc["runs"]:
            for family, res in r["eval"].items():
                if not isinstance(res, dict) or "paths" not in res:
                    continue
                acc[r["variant"]][family].append(res["accuracy"])
                for path, p in res["paths"].items():
                    correct = round(p["accuracy"] * p["n"])
                    counts[r["variant"]][f"{family}:{path}"][0] += correct
                    counts[r["variant"]][f"{family}:{path}"][1] += p["n"]
        out[name.split("/")[0]] = {
            v: {"mean_sd": {f: mean_sd(a) for f, a in acc[v].items()},
                "correct_total": {k: tuple(c) for k, c in counts[v].items()},
                "two_hop_correct_total": tuple(map(sum, zip(*[c for k, c in counts[v].items() if k.startswith("two_hop:")])))}
            for v in acc}
    return out


def adaptive_tables(bundle: Path = HERE) -> dict:
    """Rebuild the frozen adaptive-compute verdict from the bundle and check it against frontier.json."""
    sys.path.insert(0, str(BENCH))
    import frontier  # research/ti_tnn/bench/frontier.py
    manifest = json.loads((bundle / "manifest.json").read_text())
    original = {str(bundle / f["path"]): f["original_path"] for f in manifest["files"]}
    by_original = {f["original_path"]: f.get("original_sha256", f["sha256"]) for f in manifest["files"]}
    recorded = json.loads((bundle / "adaptive_compute/frontier.json").read_text())
    for part in ("hops", "lm"):  # the evaluated files are exactly the bundled ones
        for row in recorded[part]["manifest"]:
            if by_original.get(row["path"]) != row["sha256"]:
                raise ValueError(f"{row['path']}: evaluated file is not the bundled one")
    sources = {str(bundle / "real_benchmarks/lm"): "316c18f", str(bundle / "adaptive_compute"): "64ecc86"}
    A, mA = frontier.load([bundle / "adaptive_compute/hops"], "hops", sources)
    B, mB = frontier.load([bundle / "real_benchmarks/lm", bundle / "adaptive_compute/lm"], "lm", sources)
    for m in mA + mB:  # judge with the paths the evaluator saw
        m["path"] = original[str(Path(m["path"]))]
    out = {}
    for part, V, man, ref, higher, close, ratio in (("hops", A, mA, "dense", True, 0.02, False),
                                                   ("lm", B, mB, "dense", False, 0.02, True)):
        inv = frontier.inventory(V, part, man)
        P = frontier.planned_only(V, part)
        pred = frontier.judge(P, ref, higher, close, ratio)
        if part == "lm":
            pred["latency"] = frontier.latency_prediction(
                json.loads((bundle / "adaptive_compute/latency.json").read_text()), man)
        for k in ("efficiency", "equal_compute", "matched_cap") + (("latency",) if part == "lm" else ()):
            if pred[k]["holds"] != recorded[part]["predictions"][k]["holds"]:
                raise ValueError(f"{part} {k}: rebuilt {pred[k]['holds']}, recorded {recorded[part]['predictions'][k]['holds']}")
        for n, v in P.items():
            if abs(v["quality"]["mean"] - recorded[part]["variants"][n]["quality"]["mean"]) > 1e-12:
                raise ValueError(f"{part} {n}: rebuilt quality differs")
        out[part] = {"complete": inv["complete"], "variants": P,
                     "predictions": {k: pred[k]["holds"] for k in pred if isinstance(pred[k], dict) and "holds" in pred[k]}}
    out["lm"]["validated"] = frontier.all_of([out["lm"]["predictions"]["efficiency"], out["lm"]["predictions"]["equal_compute"]])
    if out["lm"]["validated"] != recorded["lm"]["validated"]:
        raise ValueError("the rebuilt decision differs from the recorded one")
    return out


def init_control_tables(bundle: Path = HERE) -> dict:
    """Rebuild the WikiText-103 initialization control's verdict from the bundle and check it against the
    recorded analysis (frontier_ic_v2_init_control.json)."""
    sys.path.insert(0, str(BENCH))
    import frontier  # research/ti_tnn/bench/frontier.py
    manifest = json.loads((bundle / "manifest.json").read_text())
    original = {str(bundle / f["path"]): f["original_path"] for f in manifest["files"]}
    V, rows = frontier.load([bundle / "real_benchmarks/lm", bundle / "adaptive_compute/lm",
                             bundle / "adaptive_compute_init_control/lm"], "lm")
    for m in rows:
        m["path"] = original[str(Path(m["path"]))]
    doc = json.loads((bundle / "adaptive_compute_init_control/latency_init_control.json").read_text())
    rebuilt = frontier.init_control(V, rows, doc)
    recorded = json.loads((bundle / "adaptive_compute_init_control/frontier_ic_v2_init_control.json").read_text())
    for n, f in recorded["families"].items():
        if rebuilt["families"][n]["holds"] != f["holds"]:
            raise ValueError(f"{n}: rebuilt {rebuilt['families'][n]['holds']}, recorded {f['holds']}")
    if rebuilt["latency"]["holds"] != recorded["latency"]["holds"]:
        raise ValueError("the rebuilt latency verdict differs from the recorded one")
    return rebuilt


def learning_tables(bundle: Path = HERE) -> dict:
    """The lookup-task learning work: the frozen initialization check, every exploratory diagnostic run,
    and part A v2's verdict rebuilt with frontier.py --hops-v2 and checked against the recorded one."""
    sys.path.insert(0, str(BENCH))
    import frontier  # research/ti_tnn/bench/frontier.py
    manifest = json.loads((bundle / "manifest.json").read_text())

    def runs(root):
        out = []
        for f in sorted((bundle / root).rglob("result.json")):
            r = json.loads(f.read_text())
            c = r["config"]
            knobs = "".join(f" {k}={c[k]}" for k, d in (("lr", 1e-3), ("balance_coef", 0.01)) if c.get(k, d) != d)
            out.append({"round": f.parent.parent.name, "name": frontier.v2_name(r) + knobs, "seed": c["seed"],
                        "hard_accuracy": r["eval"]["hard_accuracy"], "by_hops": r["eval"]["accuracy_by_hops"]})
        return out
    by_original = {f["original_path"]: f.get("original_sha256", f["sha256"]) for f in manifest["files"]}
    recorded = json.loads((bundle / "adaptive_compute_part_a_v2/frontier_part_a_v2.json").read_text())
    for row in recorded["manifest"]:  # the evaluated files are exactly the bundled ones
        if by_original.get(row["path"]) != row["sha256"]:
            raise ValueError(f"{row['path']}: evaluated file is not the bundled one")
    V, rows = frontier.load([bundle / "adaptive_compute_part_a_v2/hops_v2"], "hops", namer=frontier.v2_name,
                            spec=frontier.V2_PROTOCOL)
    rebuilt = frontier.evaluate_v2(V, rows)
    for key in ("learning_fixes_work", "ti_tnn_passes"):
        if rebuilt[key] != recorded[key]:
            raise ValueError(f"part A v2 {key}: rebuilt {rebuilt[key]}, recorded {recorded[key]}")
    for key in ("validity", "fixed_moe"):
        if rebuilt[key]["holds"] != recorded[key]["holds"]:
            raise ValueError(f"part A v2 {key}: rebuilt differs from recorded")
    for key in ("efficiency", "equal_compute", "matched_cap"):
        if rebuilt["predictions"][key]["holds"] != recorded["predictions"][key]["holds"]:
            raise ValueError(f"part A v2 prediction {key}: rebuilt differs from recorded")
    return {"init_check": runs("lookup_init_check"), "diagnostics": runs("lookup_learning_diagnostics"),
            "part_a_v2": rebuilt, "part_a_v2_variants": V}


def frontier_names(learning: dict) -> list:
    sys.path.insert(0, str(BENCH))
    import frontier
    return [n for n in frontier.V2_EXPECTED if n in learning["part_a_v2_variants"]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", type=Path, help="also write every table as JSON")
    args = p.parse_args(argv)
    manifest = verify()
    print(f"{len(manifest['files'])} files match manifest.json\n")
    toys = toy_tables()
    print("Toy benchmarks (3 seeds; mean ± sd)")
    for v, (m, s) in toys["compositional_heldout_xor"].items():
        print(f"  held-out XOR, {v:20s} {m:.3f} ± {s:.3f}")
    for v, (m, s) in toys["routing_task_accuracy"].items():
        print(f"  routing task accuracy, {v:15s} {m:.3f} ± {s:.3f}")
    for v, d in toys["unsupervised"].items():
        print(f"  unsupervised, {v:15s} accuracy {d['task_accuracy'][0]:.3f}, purity {d['purity'][0]:.3f}, "
              f"bridge fires {d['bridge_fire_rate'][0]:.3f}")
    real = real_benchmark_tables()
    print("\nWikiText-103 (3 seeds; validation and test perplexity), rebuilt and matching summary.json")
    for v, s in real["lm"].items():
        print(f"  {v:15s} {s['validation_perplexity']['mean']:.2f} ± {s['validation_perplexity']['sd']:.2f}   "
              f"{s['test_perplexity']['mean']:.2f} ± {s['test_perplexity']['sd']:.2f}")
    print("SCAN (3 seeds; test exact match)")
    for split, variants in real["scan"].items():
        print(f"  {split:18s} " + "  ".join(f"{v} {s['test_exact_match']['mean']:.4f}" for v, s in variants.items()))
    comp = composition_tables()
    for run, variants in comp.items():
        print(f"\n{run} (5 seeds; two-hop correct / total)")
        for v, d in variants.items():
            c, n = d["two_hop_correct_total"]
            paths = "  ".join(f"{k.split(':')[1]} {a}/{b}" for k, (a, b) in d["correct_total"].items() if k.startswith("two_hop:"))
            print(f"  {v:12s} {c}/{n}   ({paths})")
    adaptive = adaptive_tables()
    print("\nAdaptive compute (3 seeds), rebuilt and matching frontier.json")
    for part, label, fmt in (("lm", "WikiText-103 validation perplexity", ".2f"), ("hops", "lookup hard accuracy", ".3f")):
        print(f"  {label}; inventory complete = {adaptive[part]['complete']}")
        for n, v in sorted(adaptive[part]["variants"].items(), key=lambda kv: kv[1]["compute"]["mean"]):
            print(f"    {n:22s} {v['quality']['mean']:{fmt}}  decoder MACs {v['compute']['mean'] / 1e6:.2f}M")
        print(f"    predictions: {adaptive[part]['predictions']}")
    print(f"  efficiency validated (predictions 5 and 6): {adaptive['lm']['validated']}")
    learning = learning_tables()
    print("\nLookup task: the frozen initialization check (moe_top1, dense-matched experts)")
    for r in learning["init_check"]:
        print(f"  seed {r['seed']}: {r['hard_accuracy']:.3f}  by hops {r['by_hops']}")
    print("Lookup task: exploratory diagnostics on development seeds (every run)")
    for r in learning["diagnostics"]:
        hops = " / ".join(f"{v:.2f}" for v in r["by_hops"].values())
        print(f"  {r['round']:10s} {r['name']:58s} seed {r['seed']}: {r['hard_accuracy']:.3f}  ({hops})")
    v2 = learning["part_a_v2"]
    print("Part A v2, rebuilt and matching the recorded verdict")
    for n in frontier_names(learning):
        v = learning["part_a_v2_variants"][n]
        print(f"  {n:58s} {v['quality']['mean']:.3f}  learned {v2['learned_by_variant'][n]}/5")
    print(f"  validity {v2['validity']['holds']}; fixed mixture {v2['fixed_moe']['holds']}; "
          f"learning fixes work {v2['learning_fixes_work']}; TI-TNN passes {v2['ti_tnn_passes']}")
    control = init_control_tables()
    print("\nWikiText-103 initialization control, rebuilt and matching the recorded analysis")
    for n, f in control["families"].items():
        print(f"  {n:32s} comparison keeps its sign = {f['holds']}; perplexity "
              + ", ".join(f"{k} {v:.2f}" for k, v in f["mean_perplexity"].items()))
    print(f"  latency: {control['latency']['holds']}")
    if args.json:
        args.json.write_text(json.dumps({"toys": toys, "real_benchmarks": real, "composition": comp,
                                         "adaptive_compute": adaptive}, indent=2, default=list))


if __name__ == "__main__":
    main()
