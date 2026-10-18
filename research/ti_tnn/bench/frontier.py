"""Quality-against-compute frontier and the frozen pass bar (TI-TNN adaptive-compute benchmark).

Reads part A (hops.py) and part B (lm.py) results and applies the predictions of
research/ti_tnn/protocols/adaptive_compute.md exactly as written there. Compute is non-embedding forward
multiply-accumulates per token, from the routing decisions each model actually made on the evaluation
data (runs from the earlier benchmark are counted with the same formula).

The verdict comes from an explicit run manifest. Only the planned variants enter it. Every planned
variant needs seeds 0, 1 and 2 exactly once, one configuration shared by all its seeds, the protocol's
fixed settings and the legacy expert initialization. Unplanned runs (for example a later control) are
listed and excluded, so they cannot change the frozen verdict. Prediction 8 (latency) additionally
needs a valid execution check for every measured model (LATENCY_RULE); otherwise it is untestable.

    .venv-phase2/bin/python research/ti_tnn/bench/frontier.py --hops runs/titnn_efficiency/hops \\
        --lm runs/titnn_bench/lm runs/titnn_efficiency/lm --latency runs/titnn_efficiency/latency.json \\
        --source runs/titnn_bench/lm=316c18f --source runs/titnn_efficiency=64ecc86 --out frontier.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch

import ti_tnn_models as models  # research/ti_tnn/bench is on sys.path when this file runs as a script

CANDIDATE = "ti_tnn_adaptive"
LM_KINDS = ("dense", "moe_top1", "moe_top2", "mod", "ti_tnn_adaptive")  # the earlier ti_tnn bridge runs are not in the frontier
SEEDS = (0, 1, 2)
# The planned run inventory (protocol): every variant, each with exactly these seeds.
EXPECTED = {
    "hops": ["dense", "dense_d128_l2", "dense_d128_l3", "dense_d128_l4", "dense_d128_l6", "moe_top1",
             "mod_c50", "mod_c30", "mod_c15", "ti_tnn_adaptive_c50", "ti_tnn_adaptive_c30", "ti_tnn_adaptive_c15"],
    "lm": ["dense", "moe_top1", "moe_top2", "dense_d512_l4", "dense_d384_l8", "dense_d256_l8",
           "mod_c50", "mod_c30", "ti_tnn_adaptive_c50", "ti_tnn_adaptive_c30"],
}
# Settings the protocol fixes for every run of a part (checked against each run's config).
PROTOCOL = {
    "lm": {"config": {"batch": 32, "context": 512, "lr": 6e-4, "min_lr_frac": 0.1, "warmup_frac": 0.02,
                      "weight_decay": 0.1, "experts": 8, "heads": 8, "balance_coef": 0.01},
           "tokens_trained": 6103 * 32 * 512},
    "hops": {"config": {"steps": 6000, "batch": 64, "lr": 1e-3, "min_lr_frac": 0.1, "warmup_frac": 0.02,
                        "weight_decay": 0.1, "experts": 8, "heads": 4, "balance_coef": 0.01, "d_model": 128,
                        "eval_sequences": 2048},
             "task": {"symbols": 32, "facts": 8, "max_hops": 3, "counts": 16, "queries": 12, "length": 128,
                      "one_hop_share": 50}},
}
ADAPTIVE_SETTINGS = {"dual_lr": 0.01, "small_mult": 1}
FROZEN_INIT = "legacy"  # the expert initialization every frozen run used
A_VALIDITY = 0.90  # part A is read only if the 8-layer dense reference reaches this hard accuracy
# Prediction 8 is testable only if every measured model passes these checks on every checked window
# (fixed 2026-10-16, before any latency result existed; see the protocol's execution record).
LATENCY_RULE = {
    "relative_logit_tolerance": 0.05,  # max |decoded - full| logit, over positions where routing agrees, / max |full| logit
    "decision_agreement_min": 0.99,    # trained adaptive models: share of (token, layer) routing decisions that agree
    "token_agreement_min": 0.99,       # trained models: share of positions whose top-1 token agrees
}
IGNORED_CONFIG_KEYS = ("seed", "out", "save_model")


def all_of(outcomes) -> bool | None:
    """Three-valued AND: any failure fails; otherwise any untestable outcome is untestable; else it holds."""
    outcomes = list(outcomes)
    if any(o is False for o in outcomes):
        return False
    if not outcomes or any(o is None for o in outcomes):
        return None
    return True


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
    name = models.variant_name(c["ffn"], c.get("bridge_cost", 0.01), m.get("target_compute", 1.0),
                               models.size_tag(m["d_model"], m["n_layers"], m.get("ffn_mult", 4), reference))
    init = m.get("expert_init", FROZEN_INIT)
    return name if init == FROZEN_INIT else f"{name}_init{init}"


def fingerprint(config: dict) -> str:
    kept = {k: v for k, v in sorted(config.items()) if k not in IGNORED_CONFIG_KEYS}
    return hashlib.sha256(json.dumps(kept, sort_keys=True).encode()).hexdigest()[:16]


def deviations(r: dict, part: str, spec: dict | None = None) -> list:
    """The protocol settings this run does not match."""
    spec, c, out = spec or PROTOCOL[part], r["config"], []
    for k, v in spec["config"].items():
        if c.get(k) != v:
            out.append(f"{k}={c.get(k)!r} (protocol {v!r})")
    if part == "lm" and r.get("tokens_trained") != spec["tokens_trained"]:
        out.append(f"tokens_trained={r.get('tokens_trained')} (protocol {spec['tokens_trained']})")
    if part == "hops" and r.get("task") != spec["task"]:
        out.append(f"task={r.get('task')} (protocol {spec['task']})")
    if c["ffn"] in models.ADAPTIVE_KINDS:
        for k, v in ADAPTIVE_SETTINGS.items():
            if c.get(k) != v:
                out.append(f"{k}={c.get(k)!r} (protocol {v!r})")
    return out


def source_of(path: Path, sources: dict) -> str:
    """The source version declared for the longest matching root (run files do not record one)."""
    best = max((root for root in sources if str(path).startswith(str(root))), key=lambda r: len(str(r)), default=None)
    return sources[best] if best is not None else "unrecorded"


def load(roots, part: str, sources: dict | None = None, namer=None, spec: dict | None = None) -> tuple[dict, list]:
    """Variants averaged over seeds, and the run manifest (one row per result file)."""
    groups, manifest = defaultdict(list), []
    for root in roots:
        for path in sorted(Path(root).glob("*/result.json")):
            raw = path.read_bytes()
            r = json.loads(raw)
            if part == "lm" and r["config"]["ffn"] not in LM_KINDS:
                continue
            ref = (128, 8, 4) if part == "hops" else (512, 8, 4)
            name = namer(r) if namer else name_of(r, ref)
            groups[name].append(r)
            manifest.append({"variant": name, "seed": r["config"]["seed"], "path": str(path),
                             "sha256": hashlib.sha256(raw).hexdigest(), "config_fingerprint": fingerprint(r["config"]),
                             "init_policy": r["model_config"].get("expert_init", FROZEN_INIT),
                             "training": ({"tokens": r.get("tokens_trained")} if part == "lm"
                                          else {"steps": r["config"].get("steps")}),
                             "source": source_of(path, sources or {}), "deviations": deviations(r, part, spec)})
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
                     "head_macs_per_token": float(runs[0]["model_config"]["vocab_size"] * runs[0]["model_config"]["d_model"]),
                     "parameters": runs[0]["parameters"]["total"]}
        out[name] = {"kind": runs[0]["config"]["ffn"], "seeds": [r["config"]["seed"] for r in runs],
                     "quality": stat(quality), "compute": stat(compute), **extra}
    return out, manifest


def inventory(variants: dict, part: str, manifest: list | None = None) -> dict:
    """Complete only if every planned variant has seeds 0-2 exactly once, one configuration across its
    seeds, the protocol's settings and the frozen initialization. Unplanned variants are listed; they
    never enter the verdict."""
    planned = EXPECTED[part]
    missing = [n for n in planned if n not in variants]
    bad_seeds = {n: v["seeds"] for n, v in variants.items() if n in planned and sorted(v["seeds"]) != list(SEEDS)}
    rows = [m for m in (manifest or []) if m["variant"] in planned]
    prints = defaultdict(set)
    for m in rows:
        prints[m["variant"]].add(m["config_fingerprint"])
    inconsistent = sorted(n for n, fp in prints.items() if len(fp) > 1)
    off_protocol = {f"{m['variant']} seed {m['seed']}": m["deviations"] for m in rows if m["deviations"]}
    wrong_init = sorted({m["variant"] for m in rows if m["init_policy"] != FROZEN_INIT})
    return {"complete": not (missing or bad_seeds or inconsistent or off_protocol or wrong_init),
            "missing": missing, "bad_seeds": bad_seeds, "inconsistent_configs": inconsistent,
            "protocol_deviations": off_protocol, "wrong_init": wrong_init,
            "unexpected_excluded": sorted(n for n in variants if n not in planned)}


def planned_only(variants: dict, part: str) -> dict:
    return {n: v for n, v in variants.items() if n in EXPECTED[part]}


def better(a: float, b: float, higher_is_better: bool) -> bool:
    return a > b if higher_is_better else a < b


def at_least_as_good(a: float, b: float, higher_is_better: bool) -> bool:
    return a >= b if higher_is_better else a <= b


def judge(variants: dict, reference: str, higher_is_better: bool, close: float, close_is_ratio: bool,
          twin: dict | None = None) -> dict:
    """The protocol's efficiency (P-eff), equal-compute (P-eq) and matched-cap (P-mod) predictions,
    over exactly the variants given (pass planned_only(...) for the frozen verdict)."""
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
        mate = twin.get(n) if twin else n.replace(CANDIDATE, "mod")
        if mate in base:
            cap[n] = {"mod": q(base[mate]), "holds": bool(better(q(v), q(base[mate]), higher_is_better))}
        else:  # a missing matched-cap baseline makes the comparison untestable, not passed
            cap[n] = {"mod": None, "holds": None}
    return {"reference": reference, "reference_quality": ref_q, "close_threshold": threshold,
            "efficiency": {"holds": any(e["holds"] for e in eff.values()) if eff else None, "per_candidate": eff},
            "equal_compute": {"holds": all_of(e["holds"] for e in eq.values()), "per_candidate": eq},
            "matched_cap": {"holds": all_of(e["holds"] for e in cap.values()), "per_candidate": cap}}


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
    return {"holds": all_of(o["holds"] for o in out.values()), "per_candidate": out}


def run_key(path: str) -> str:
    """A run directory from its "runs/" part on, so absolute and relative spellings of one run match."""
    i = path.find("runs/")
    return path[i:] if i >= 0 else path


def latency_row_problems(row: dict, cfg: dict) -> list:
    """Why a latency measurement is not a valid record of the evaluated model's execution (empty if valid)."""
    problems = []
    windows, rounds = int(cfg["windows"]), int(cfg["rounds"])
    expected_tokens = rounds * windows * (int(cfg["context"]) - int(cfg["warmup_tokens"]))
    checks = row.get("checks", [])
    if len(checks) != windows:
        problems.append(f"{len(checks)} checked windows (expected {windows})")
    if row.get("timed_tokens") != expected_tokens:
        problems.append(f"{row.get('timed_tokens')} timed tokens (expected {expected_tokens})")
    trained = row.get("weights") == "trained"
    for i, c in enumerate(checks):
        diff = c.get("max_abs_logit_difference_where_routing_agrees")
        scale = c.get("max_abs_full_logit")
        if diff is None or not scale or diff > LATENCY_RULE["relative_logit_tolerance"] * scale:
            problems.append(f"window {i}: logit difference {diff} exceeds tolerance on scale {scale}")
        if trained and c.get("top1_token_agreement", 0.0) < LATENCY_RULE["token_agreement_min"]:
            problems.append(f"window {i}: top-1 token agreement {c.get('top1_token_agreement')}")
        if trained and "decision_agreement" in c and c["decision_agreement"] < LATENCY_RULE["decision_agreement_min"]:
            problems.append(f"window {i}: routing-decision agreement {c['decision_agreement']}")
    return problems


def latency_prediction(doc: dict, manifest: list, candidate: str = f"{CANDIDATE}_c30", reference: str = "dense") -> dict:
    """Prediction 8: the candidate's mean (over seeds 0-2) median ms/token <= 0.6 x the reference's.
    Untestable unless the candidate's three planned seeds and the reference appear exactly once each,
    each candidate row is the planned run's own checkpoint, and every one of those rows is valid."""
    rows, cfg = doc["runs"], doc["config"]
    cand = [r for r in rows if r["name"] == candidate]
    ref = [r for r in rows if r["name"] == reference]
    planned_dirs = {run_key(str(Path(m["path"]).parent)) for m in manifest if m["variant"] == candidate}
    reasons = []
    if sorted(r["seed"] for r in cand) != list(SEEDS):
        reasons.append(f"{candidate} seeds {sorted(r['seed'] for r in cand)} (expected {list(SEEDS)} once each)")
    if len(ref) != 1:
        reasons.append(f"{len(ref)} {reference} rows (expected 1)")
    stray = [r["dir"] for r in cand if run_key(str(Path(r["dir"]))) not in planned_dirs]
    if stray:
        reasons.append(f"{candidate} rows not from the planned runs: {stray}")
    validity = {f"{r['name']} seed {r['seed']}": latency_row_problems(r, cfg) for r in cand + ref}
    reasons += [f"{k}: {p}" for k, ps in validity.items() for p in ps]
    cand_ms = statistics.fmean(r["ms_per_token_median"] for r in cand) if cand else None
    ref_ms = statistics.fmean(r["ms_per_token_median"] for r in ref) if ref else None
    holds = None if reasons else bool(cand_ms <= 0.6 * ref_ms)
    return {"holds": holds, "candidate_ms": cand_ms, "reference_ms": ref_ms, "untestable_because": reasons,
            "rule": LATENCY_RULE, "validity": validity}


# ---------------------------------------------------------------- the initialization control (separate analysis)
# protocols/adaptive_compute_init_control.md. Its runs never enter the frozen verdict above.
CONTROL = {"ti_tnn_adaptive_c30_initdense": "ti_tnn_adaptive_c30", "moe_top1_initdense": "moe_top1"}
CONTROL_COMPARISONS = {"ti_tnn_adaptive_c30": "mod_c30", "moe_top1": "dense"}  # the comparison each must survive


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def init_control(variants: dict, manifest: list, latency_doc: dict | None = None) -> dict:
    """Paired by seed: each control family under dense-matched expert initialization against the same
    family under the legacy initialization, and whether its comparison keeps its sign."""
    rows = {(m["variant"], m["seed"]): m for m in manifest}
    problems = []
    for ctrl, orig in CONTROL.items():
        for name, init in ((ctrl, "dense"), (orig, FROZEN_INIT), (CONTROL_COMPARISONS[orig], FROZEN_INIT)):
            seeds = variants.get(name, {}).get("seeds", [])
            if sorted(seeds) != list(SEEDS):
                problems.append(f"{name}: seeds {sorted(seeds)} (expected {list(SEEDS)} once each)")
            for s in SEEDS:
                m = rows.get((name, s))
                if m and (m["deviations"] or m["init_policy"] != init):
                    problems.append(f"{name} seed {s}: init {m['init_policy']}, deviations {m['deviations']}")
    q = lambda n: variants[n]["quality"]["mean"]
    out = {"complete": not problems, "problems": problems, "families": {}}
    for ctrl, orig in CONTROL.items():
        other = CONTROL_COMPARISONS[orig]
        fam = {"control": ctrl, "original": orig, "compared_with": other}
        if not problems:
            per_seed = {}
            for i, s in enumerate(SEEDS):
                a, b = variants[ctrl], variants[orig]
                per_seed[s] = {"perplexity": {"legacy": b["quality"]["values"][i], "dense_init": a["quality"]["values"][i]},
                               "decoder_macs": {"legacy": b["compute"]["values"][i], "dense_init": a["compute"]["values"][i]},
                               "realized_compute": {"legacy": b["realized_compute"]["values"][i],
                                                    "dense_init": a["realized_compute"]["values"][i]}}
            gap_legacy, gap_dense = q(orig) - q(other), q(ctrl) - q(other)
            fam.update({"per_seed": per_seed, "mean_perplexity": {"legacy": q(orig), "dense_init": q(ctrl), other: q(other)},
                        "total_macs": {"legacy": variants[orig]["compute"]["mean"] + variants[orig]["head_macs_per_token"],
                                       "dense_init": variants[ctrl]["compute"]["mean"] + variants[ctrl]["head_macs_per_token"]},
                        "gap_to_comparison": {"legacy": gap_legacy, "dense_init": gap_dense},
                        "holds": sign(gap_legacy) == sign(gap_dense) != 0})
        else:
            fam["holds"] = None
        out["families"][ctrl] = fam
    if latency_doc is not None:
        lat_manifest = [m for m in manifest if m["variant"] == "ti_tnn_adaptive_c30_initdense"]
        out["latency"] = latency_prediction(latency_doc, lat_manifest, candidate="ti_tnn_adaptive_c30_initdense")
    out["holds"] = all_of(f["holds"] for f in out["families"].values())
    return out


# ---------------------------------------------------------------- part A v2 (protocols/adaptive_compute_part_a_v2.md)
# Written after the v2 queue started and before any v2 result was examined.
V2_SEEDS = (200, 201, 202, 203, 204)
V2_REFERENCE = "dense_s18000"
V2_FIXED_MOE = "moe_top1_initdense_sh50_s18000"
V2_EXPECTED = ["dense_d128_l2_s18000", "dense_d128_l3_s18000", "dense_d128_l4_s18000", "dense_d128_l6_s18000",
               V2_REFERENCE, V2_FIXED_MOE, "mod_c50_warm50_kv_s18000", "mod_c30_warm50_kv_s18000",
               "ti_tnn_adaptive_c50_initdense_warm50_kv_sh50_s18000", "ti_tnn_adaptive_c30_initdense_warm50_kv_sh50_s18000"]
V2_TWIN = {"ti_tnn_adaptive_c50_initdense_warm50_kv_sh50_s18000": "mod_c50_warm50_kv_s18000",
           "ti_tnn_adaptive_c30_initdense_warm50_kv_sh50_s18000": "mod_c30_warm50_kv_s18000"}
V2_PROTOCOL = {"config": dict(PROTOCOL["hops"]["config"], steps=18000), "task": PROTOCOL["hops"]["task"]}
V2_LEARNS = 0.9  # a run learns the task if its 3-hop accuracy reaches this


def v2_name(r: dict) -> str:
    """The run's name as hops.py gives it (the name records every fix applied)."""
    import argparse as _argparse
    import hops
    config = vars(hops.parse_args(["--ffn", r["config"]["ffn"]]))  # defaults for options newer than the run
    config.update(r["config"])
    return hops.variant_name(_argparse.Namespace(**config))


def learned(v: dict) -> int:
    """How many seeds learned the task (3-hop accuracy >= V2_LEARNS)."""
    return sum(x >= V2_LEARNS for x in v["by_hops"]["3"]["values"])


def evaluate_v2(variants: dict, manifest: list) -> dict:
    """Part A v2's validity condition, predictions 1-5 and both decision rules, on the planned variants."""
    missing = [n for n in V2_EXPECTED if n not in variants]
    bad_seeds = {n: v["seeds"] for n, v in variants.items() if n in V2_EXPECTED and sorted(v["seeds"]) != list(V2_SEEDS)}
    prints = defaultdict(set)
    for m in manifest:
        if m["variant"] in V2_EXPECTED:
            prints[m["variant"]].add(m["config_fingerprint"])
    inconsistent = sorted(n for n, fp in prints.items() if len(fp) > 1)
    off = {f"{m['variant']} seed {m['seed']}": m["deviations"] for m in manifest if m["variant"] in V2_EXPECTED and m["deviations"]}
    complete = not (missing or bad_seeds or inconsistent or off)
    inv = {"complete": complete, "missing": missing, "bad_seeds": bad_seeds, "inconsistent_configs": inconsistent,
           "protocol_deviations": off, "unexpected_excluded": sorted(n for n in variants if n not in V2_EXPECTED)}
    out = {"inventory": inv}
    if not complete:
        return dict(out, learning_fixes_work=None, ti_tnn_passes=None)
    P = planned = {n: variants[n] for n in V2_EXPECTED}
    ref = P[V2_REFERENCE]
    validity = bool(ref["quality"]["mean"] >= 0.90 and learned(ref) >= 4)
    p1 = bool(P[V2_FIXED_MOE]["quality"]["mean"] >= 0.90)
    pred = judge(planned, V2_REFERENCE, True, 0.02, False, twin=V2_TWIN)
    out.update({"validity": {"holds": validity, "reference_hard_accuracy": ref["quality"]["mean"], "reference_learned": learned(ref)},
                "fixed_moe": {"holds": p1, "hard_accuracy": P[V2_FIXED_MOE]["quality"]["mean"], "learned": learned(P[V2_FIXED_MOE])},
                "predictions": pred, "difficulty_tracking": difficulty_tracking(planned),
                "learned_by_variant": {n: learned(v) for n, v in P.items()}})
    out["learning_fixes_work"] = all_of([validity, p1])
    out["ti_tnn_passes"] = all_of([pred["efficiency"]["holds"], pred["equal_compute"]["holds"]]) if validity else None
    return out


# ---------------------------------------------------------------- WikiText-103 with the lookup-task fixes
# protocols/adaptive_compute_wikitext_fixed.md. Written before any run of it.
WF_CANDIDATE = "ti_tnn_adaptive_c30_initdense_warm50_kv_sh50"
WF_INIT_ONLY = "ti_tnn_adaptive_c30_initdense"  # the initialization control's run of the same cap
WF_MOD = "mod_c30"


def lm_name(r: dict) -> str:
    """The run's name as lm.py gives it, with every option's default filled in for older runs."""
    import argparse as _argparse
    import lm
    config = vars(lm.parse_args(["--ffn", r["config"]["ffn"]]))
    config.update(r["config"])
    return lm.variant_name(_argparse.Namespace(**config))


def evaluate_wikitext_fixed(variants: dict, manifest: list, latency_doc: dict | None = None) -> dict:
    """Predictions 1-4 and the latency check for the fixed TI-TNN at the 30% cap, against the frozen
    benchmark's planned baselines and the initialization control's run."""
    rows = [m for m in manifest if m["variant"] == WF_CANDIDATE]
    problems = []
    if sorted(m["seed"] for m in rows) != list(SEEDS):
        problems.append(f"{WF_CANDIDATE}: seeds {sorted(m['seed'] for m in rows)}")
    problems += [f"{WF_CANDIDATE} seed {m['seed']}: {m['deviations']}" for m in rows if m["deviations"]]
    problems += [f"{WF_CANDIDATE} seed {m['seed']}: init {m['init_policy']}" for m in rows if m["init_policy"] != "dense"]
    for n in [WF_INIT_ONLY] + EXPECTED["lm"]:
        if sorted(variants.get(n, {}).get("seeds", [])) != list(SEEDS):
            problems.append(f"{n}: missing or incomplete")
    out = {"complete": not problems, "problems": problems}
    if problems:
        return out
    q = lambda n: variants[n]["quality"]["mean"]
    baselines = {n: variants[n] for n in EXPECTED["lm"] if variants[n]["kind"] != CANDIDATE}
    pred = judge(dict(baselines, **{WF_CANDIDATE: variants[WF_CANDIDATE]}), "dense", False, 0.02, True,
                 twin={WF_CANDIDATE: WF_MOD})
    out.update({
        "fixes_help_on_text": {"holds": q(WF_CANDIDATE) < q(WF_INIT_ONLY), "fixed": q(WF_CANDIDATE), "init_only": q(WF_INIT_ONLY)},
        "beats_plain_skipping": pred["matched_cap"],
        "efficiency": pred["efficiency"], "equal_compute": pred["equal_compute"],
        "candidate": variants[WF_CANDIDATE]})
    if latency_doc is not None:
        out["latency"] = latency_prediction(latency_doc, manifest, candidate=WF_CANDIDATE)
    return out


def table(variants: dict, quality: str, fmt: str) -> str:
    rows = sorted(variants.items(), key=lambda kv: kv[1]["compute"]["mean"])
    lines = [f"| Variant | {quality} | Non-embedding MACs per token | Seeds |", "|---|---:|---:|---:|"]
    for n, v in rows:
        lines.append(f"| `{n}` | {v['quality']['mean']:{fmt}} ± {v['quality']['sd']:{fmt}} | "
                     f"{v['compute']['mean'] / 1e6:.2f}M | {len(v['seeds'])} |")
    return "\n".join(lines)


def parse_sources(items) -> dict:
    out = {}
    for item in items or []:
        root, _, label = item.partition("=")
        out[str(Path(root))] = label
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hops", nargs="*", type=Path, default=[])
    p.add_argument("--lm", nargs="*", type=Path, default=[])
    p.add_argument("--latency", type=Path)
    p.add_argument("--source", action="append", default=[], help="ROOT=VERSION: the source version runs under ROOT came from")
    p.add_argument("--hops-v2", nargs="*", type=Path, default=[], help="part A v2 run roots (separate analysis)")
    p.add_argument("--wikitext-fixed", nargs="*", type=Path, default=[],
                   help="roots holding the fixed TI-TNN WikiText-103 runs (separate analysis; --lm gives the baselines)")
    p.add_argument("--init-control", nargs="*", type=Path, default=[],
                   help="roots holding the initialization-control runs (separate analysis)")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    sources = parse_sources(args.source)
    summary = {}
    if args.hops:
        A, manifest = load(args.hops, "hops", sources)
        inv = inventory(A, "hops", manifest)
        P = planned_only(A, "hops")
        ref = P.get("dense", {}).get("quality", {}).get("mean")
        valid = ref is not None and ref >= A_VALIDITY
        summary["hops"] = {"variants": A, "manifest": manifest, "inventory": inv,
                           "validity": {"reference_hard_accuracy": ref, "holds": valid},
                           "predictions": judge(P, "dense", True, 0.02, False) if "dense" in P else {},
                           "difficulty_tracking": difficulty_tracking(P)}
        summary["hops"]["readable"] = bool(inv["complete"] and valid)
        print("## Part A: variable-hop lookups (hard-query accuracy)\n\n" + table(P, "Hard accuracy", ".3f") + "\n")
    if args.lm:
        B, manifest = load(args.lm, "lm", sources)
        inv = inventory(B, "lm", manifest)
        P = planned_only(B, "lm")
        summary["lm"] = {"variants": B, "manifest": manifest, "inventory": inv,
                         "predictions": judge(P, "dense", False, 0.02, True) if "dense" in P else {}}
        pr = summary["lm"]["predictions"]
        # The decision rule: validated only if predictions 5 (efficiency) and 6 (equal compute) both hold,
        # on the complete run inventory.
        summary["lm"]["validated"] = (None if not inv["complete"] or not pr
                                      else all_of([pr["efficiency"]["holds"], pr["equal_compute"]["holds"]]))
        if args.latency and args.latency.exists() and pr:
            doc = json.loads(args.latency.read_text())
            by = defaultdict(list)
            for row in doc["runs"]:
                by[row["name"]].append(row["ms_per_token_median"])
            summary["lm"]["latency_ms_per_token"] = {n: stat(v) for n, v in by.items()}
            pr["latency"] = latency_prediction(doc, manifest)
        print("## Part B: WikiText-103 (validation perplexity)\n\n" + table(P, "Perplexity", ".2f") + "\n")
    if args.wikitext_fixed:
        V, manifest = load(list(args.lm) + list(args.wikitext_fixed), "lm", sources, namer=lm_name)
        doc = json.loads(args.latency.read_text()) if args.latency and args.latency.exists() else None
        res = evaluate_wikitext_fixed(V, manifest, doc)
        if res["complete"]:
            c = res["candidate"]
            print(f"wikitext fixed: {WF_CANDIDATE} {c['quality']['mean']:.2f} (init only {res['fixes_help_on_text']['init_only']:.2f}); "
                  f"fixes help = {res['fixes_help_on_text']['holds']}; beats mod = {res['beats_plain_skipping']['holds']}; "
                  f"efficiency = {res['efficiency']['holds']}; equal compute = {res['equal_compute']['holds']}"
                  + (f"; latency = {res['latency']['holds']}" if "latency" in res else ""))
        else:
            print(f"wikitext fixed: INCOMPLETE {res['problems']}")
        if args.out:
            args.out.with_name(args.out.stem + "_wikitext_fixed.json").write_text(json.dumps(res, indent=2, default=str))
        return
    if args.hops_v2:
        V, manifest = load(args.hops_v2, "hops", sources, namer=v2_name, spec=V2_PROTOCOL)
        res = evaluate_v2(V, manifest)
        print("## Part A v2 (hard accuracy; seeds learning the task)\n")
        for n in V2_EXPECTED:
            if n in V:
                print(f"  {n:52s} {V[n]['quality']['mean']:.3f}  learned {learned(V[n])}/{len(V[n]['seeds'])}  "
                      f"{V[n]['compute']['mean'] / 1e6:.2f}M MACs")
        print(f"part A v2: complete = {res['inventory']['complete']}; learning fixes work = {res['learning_fixes_work']}; "
              f"TI-TNN passes = {res['ti_tnn_passes']}")
        if args.out:
            args.out.with_name(args.out.stem + "_part_a_v2.json").write_text(
                json.dumps(dict(res, variants=V, manifest=manifest), indent=2))
        if not summary and not args.init_control:
            return
    if args.init_control:
        V, manifest = load(list(args.lm) + list(args.init_control), "lm", sources)
        doc = json.loads(args.latency.read_text()) if args.latency and args.latency.exists() else None
        ctrl = init_control(V, manifest, doc)
        print(f"init control: complete = {ctrl['complete']}; comparisons keep their sign = {ctrl['holds']}"
              + (f"; latency = {ctrl['latency']['holds']}" if "latency" in ctrl else ""))
        if args.out:
            args.out.with_name(args.out.stem + "_init_control.json").write_text(json.dumps(ctrl, indent=2))
        if not summary:
            return
    for part, s in summary.items():
        if not s["inventory"]["complete"]:
            print(f"{part}: INCOMPLETE run inventory, not a valid test: {json.dumps(s['inventory'])}")
        elif s["inventory"]["unexpected_excluded"]:
            print(f"{part}: unplanned variants excluded from the verdict: {s['inventory']['unexpected_excluded']}")
        if part == "hops":
            print(f"hops: reference validity (>= {A_VALIDITY}) holds = {s['validity']['holds']}; readable = {s['readable']}")
        else:
            print(f"lm: TI-TNN efficiency validated (predictions 5 and 6) = {s['validated']}")
        print(f"{part}: " + json.dumps({k: (v["holds"] if isinstance(v, dict) and "holds" in v else v)
                                         for k, v in s["predictions"].items() if k not in ("reference", "reference_quality",
                                                                                           "close_threshold")}))
        if "difficulty_tracking" in s:
            print(f"{part}: difficulty tracking holds = {s['difficulty_tracking']['holds']}")
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
