"""Phase Sixteen confirmatory: with the question first, the program carries the asked variable, on a sixth fresh split.

protocols/phase_sixteen_confirmatory_fresh.md is the frozen design. Example:

    What is x? ... / x = 7 / y = 16 / y = y + 1 / y = y - 3      -> 7  (y ends at 14)

Exploratory Phase Sixteen found that when the program tokens cannot attend to
the question, the late read falls back to the most recent assignment of either
variable: programs ending on the other variable are answered with its value,
programs ending on the asked variable stay mostly right. This module generates
768 new groups (seed 211001, excluding Phase Two and the five earlier fresh
splits), runs every question-first prompt once, and tests six pre-registered
claims. It also records, descriptively, a reader block limited to `What is x?`
that leaves the answer-format instruction visible.

    run       one family on the GPU: generate, record hashes, block
    evaluate  CPU: paired measures, Bonferroni bootstrap, verdicts
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback
import uuid

import numpy as np
import torch

from .answer_knockout import knockout, logits_at, program_span, score
from .artifacts import RunDirectory, sha256_file, write_json
from .edit_diagnostics import PROJECT, _git_head
from .order_confirm import FRESH_INVENTORIES as EARLIER_INVENTORIES
from .order_confirm import fresh_groups as fifth_fresh_groups
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .question_route_knockout import route_row
from .runner import release_models, source_identity
from .scale_confirm import fresh_groups as fourth_fresh_groups
from .selection_confirm import fresh_groups as third_fresh_groups
from .span_knockout_confirm import fresh_groups as first_fresh_groups
from .splits import SPLIT_ORDER, build_plan
from .supersession_confirm import fresh_groups as second_fresh_groups
from .tasks import canonical, generate_groups

PROTOCOL = PROJECT / "protocols" / "phase_sixteen_confirmatory_fresh.md"
FORMAT = "open_weight_lingua.route_confirm.v1"
PLAN_HASH = "ff060012d35d92c480a51400a02597bea7421a525c44d97043a56dd3f3de667f"
SPLIT = "fresh_2026_10f"
GROUPS = 768
FRESH_INVENTORIES = {**EARLIER_INVENTORIES,
                     "fresh_2026_10e": "4f744c7c0f1b7e8dbd19ed440789f93437fe58a86e9132eac1c6f7e98c00b33c"}
CONDITIONS = ("baseline", "program_blind", "reader_blind_clause")
RESAMPLES = 10_000
BOOTSTRAP_SEEDS = {"gemma3-12b": 808100, "qwen2.5-7b": 808101}
COVERAGE = 1 - 0.05 / 6
HYPOTHESES = (  # (id, family, measure, direction, threshold)
    ("G1", "gemma3-12b", "other_rise", "above", 0.05),
    ("G2", "gemma3-12b", "loss", "above", 0.10),
    ("G3", "gemma3-12b", "stratum_contrast", "above", 0.0),
    ("Q1", "qwen2.5-7b", "other_rise", "above", 0.05),
    ("Q2", "qwen2.5-7b", "loss", "above", 0.10),
    ("Q3", "qwen2.5-7b", "stratum_contrast", "above", 0.0),
)


def fresh_groups(plan, count: int = GROUPS):
    """New groups excluding Phase Two and the five earlier fresh splits (each verified against its inventory)."""
    if plan.plan_hash != PLAN_HASH:
        raise ValueError("split plan differs from the frozen Phase Two plan")
    excluded = {canonical(p) for name in SPLIT_ORDER for g in plan.groups[name] for p in (g.source, g.counterfactual)}
    for name, generate in (("fresh_2026_10", first_fresh_groups), ("fresh_2026_10b", second_fresh_groups),
                           ("fresh_2026_10c", third_fresh_groups), ("fresh_2026_10d", fourth_fresh_groups),
                           ("fresh_2026_10e", fifth_fresh_groups)):
        groups, stats = generate(plan)
        if stats["inventory_sha256"] != FRESH_INVENTORIES[name]:
            raise ValueError(f"{name} does not regenerate to its recorded inventory")
        excluded |= {canonical(p) for g in groups for p in (g.source, g.counterfactual)}
    groups, stats = generate_groups(SPLIT, count, excluded_programs=excluded)
    if {canonical(p) for g in groups for p in (g.source, g.counterfactual)} & excluded:
        raise ValueError("a fresh program repeats an earlier program")
    return groups, stats


def _ids(tokenizer, prompt: str) -> list[int]:
    return list(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True))


def fresh_rows(groups, tokenizer) -> list[dict]:
    rows = []
    for group in groups:
        for base in group.variants():
            row = route_row({**base, "input_ids": _ids(tokenizer, base["prompt"])}, SPLIT, tokenizer)
            clause = program_span(tokenizer, row["input_ids"], f"What is {base['variable']}?")
            if not row["question"][0] <= clause[0] <= clause[1] <= row["question"][1]:
                raise ValueError(f"{base['id']}: the clause is not inside the question")
            rows.append({**row, "side": base["side"], "clause": list(clause)})
    return rows


def main_stratum(row: dict) -> bool:
    """Programs ending on the other variable, whose answer does not depend on a copy of it."""
    return not row["last_line_assigns_asked"] and not row["depends_on_other"]


# ---------------------------------------------------------------- run (GPU)


def blocks(row: dict, length: int, condition: str) -> list[tuple[list[int], list[int]]]:
    if condition == "baseline":
        return []
    if condition == "program_blind":
        return [(list(range(row["span"][0], row["span"][1] + 1)), list(range(row["question"][0], row["question"][1] + 1)))]
    return [(list(range(row["span"][1] + 1, length)), list(range(row["clause"][0], row["clause"][1] + 1)))]


def evaluate_row(model, row: dict, device: str) -> dict:
    answer = row["sequences"]["answer"]
    baseline = logits_at(model, answer["ids"], answer["positions"], device)
    with knockout(model, 0, answer["positions"], []):
        if not torch.equal(logits_at(model, answer["ids"], answer["positions"], device), baseline):
            raise RuntimeError(f"{row['id']}: the no-op hook changed the logits")
    out = {}
    for condition in CONDITIONS:
        scored = {}
        for name, q in row["sequences"].items():
            with ExitStack() as stack:
                for queries, keys in blocks(row, len(q["ids"]), condition):
                    stack.enter_context(knockout(model, 0, queries, keys))
                scored[name] = score(logits_at(model, q["ids"], q["positions"], device), q["targets"])
        out[condition] = scored
    return out


def _family(lock: Path) -> str:
    return lock.stem.replace("model-lock-", "") if lock.stem != "model-lock" else "qwen2.5-7b"


def run(args, run_dir: RunDirectory, timings: dict) -> dict:
    family = _family(args.lock)
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models({**lock, "models": {"target": lock["models"]["target"]}}, {"target": paths["target"]})
    timings["hash_verification"] = time.perf_counter() - start
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(paths["target"], local_files_only=True)
    groups, stats = fresh_groups(build_plan())
    rows = fresh_rows(groups, tokenizer)
    stimuli = run_dir.path / "stimuli.json"
    write_json(stimuli, rows)
    write_json(run_dir.path / "manifest.json", {
        "format": FORMAT, "family": family, "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(), "protocol_path": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL),
        "source_file_sha256": source_identity(), "model_lock_sha256": sha256_file(args.lock),
        "plan_hash": PLAN_HASH, "split": SPLIT, "generation": stats, "excluded_fresh_inventories": FRESH_INVENTORIES,
        "stimuli_sha256": sha256_file(stimuli), "prompts": len(rows), "main_stratum_prompts": sum(main_stratum(r) for r in rows),
        "verified_artifact_bytes": verified, "software": compatibility(args.device),
    })
    start = time.perf_counter()
    model = load_model(paths["target"], "target", args.device)
    timings["loading"] = time.perf_counter() - start
    start = time.perf_counter()
    with (run_dir.path / "results.jsonl").open("x") as stream:
        for index, row in enumerate(rows):
            stream.write(json.dumps({"id": row["id"], **evaluate_row(model, row, args.device)}) + "\n")
            stream.flush()
            if index % 128 == 0:
                print(f"  prompts {index}/{len(rows)}", flush=True)
    timings["knockouts"] = time.perf_counter() - start
    del model
    release_models()
    return {"prompts": len(rows)}


# ---------------------------------------------------------------- evaluate (CPU)


def _loss(result: dict) -> int:
    return int(result["baseline"]["answer"]["correct"]) - int(result["program_blind"]["answer"]["correct"])


def units(stimuli: list[dict], results: list[dict], name: str) -> list[tuple[str, int]]:
    """(group id, value) per unit: main-stratum prompts, or programs with both strata for the contrast."""
    if name == "other_rise":
        return [(s["group_id"], int(r["program_blind"]["other"]["correct"]) - int(r["baseline"]["other"]["correct"]))
                for s, r in zip(stimuli, results) if main_stratum(s) and "other" in r["baseline"]]
    if name == "loss":
        return [(s["group_id"], _loss(r)) for s, r in zip(stimuli, results) if main_stratum(s)]
    programs = {}
    for s, r in zip(stimuli, results):
        kind = "asked" if s["last_line_assigns_asked"] else ("main" if main_stratum(s) else None)
        if kind:
            programs.setdefault((s["group_id"], s["side"]), {})[kind] = r
    return [(g, _loss(p["main"]) - _loss(p["asked"])) for (g, _), p in sorted(programs.items()) if len(p) == 2]


def load_family(run: Path):
    manifest = json.loads((run / "manifest.json").read_text())
    completion = json.loads((run / "completion.json").read_text())
    results = [json.loads(line) for line in (run / "results.jsonl").open()]
    stimuli = json.loads((run / "stimuli.json").read_text())
    if completion["status"] != "COMPLETE" or len(results) != manifest["prompts"]:
        raise ValueError(f"{run}: incomplete runs never enter the confirmatory analysis")
    if manifest["plan_hash"] != PLAN_HASH or manifest["protocol_sha256"] != sha256_file(PROTOCOL):
        raise ValueError(f"{run}: plan or protocol differs from the frozen ones")
    if manifest["stimuli_sha256"] != sha256_file(run / "stimuli.json") or [s["id"] for s in stimuli] != [r["id"] for r in results]:
        raise ValueError(f"{run}: stimuli and results disagree")
    return manifest, stimuli, results


def bootstrap(pairs: list[tuple[str, int]], seed: int) -> dict:
    groups = sorted({g for g, _ in pairs})
    index = {g: i for i, g in enumerate(groups)}
    sums, counts = np.zeros(len(groups)), np.zeros(len(groups))
    for g, value in pairs:
        sums[index[g]] += value
        counts[index[g]] += 1
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(RESAMPLES, len(groups)))
    resampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1 - COVERAGE) / 2
    return {"mean": float(sums.sum() / counts.sum()), "units": int(counts.sum()), "groups": len(groups),
            "interval": [float(np.quantile(resampled, tail)), float(np.quantile(resampled, 1 - tail))]}


def describe(stimuli, results) -> dict:
    out = {}
    for stratum, keep in (("all", lambda s: True), ("main", main_stratum), ("asked_last", lambda s: s["last_line_assigns_asked"])):
        chosen = [(s, r) for s, r in zip(stimuli, results) if keep(s)]
        said = [r for _, r in chosen if "other" in r["baseline"]]
        out[stratum] = {"n": len(chosen), **{c: {"accuracy": float(np.mean([r[c]["answer"]["correct"] for _, r in chosen])),
                                                  "says_other": float(np.mean([r[c]["other"]["correct"] for r in said])) if said else None}
                                              for c in CONDITIONS}}
    return out


def evaluate(runs: list[Path]) -> dict:
    families = {}
    for run in runs:
        manifest, stimuli, results = load_family(run)
        families[manifest["family"]] = (run, manifest, stimuli, results)
    inventories = {m["generation"]["inventory_sha256"] for _, m, _, _ in families.values()}
    if len(inventories) > 1:
        raise ValueError("the families were given different fresh programs")
    verdicts = []
    for test_id, family, name, direction, threshold in HYPOTHESES:
        if family not in families:
            verdicts.append({"id": test_id, "family": family, "measure": name, "status": "NOT_RUN"})
            continue
        _, _, stimuli, results = families[family]
        estimate = bootstrap(units(stimuli, results, name), BOOTSTRAP_SEEDS[family])
        low, high = estimate["interval"]
        supported = low > threshold if direction == "above" else high < threshold
        verdicts.append({"id": test_id, "family": family, "measure": name, "direction": direction, "threshold": threshold,
                         **estimate, "coverage": COVERAGE, "status": "SUPPORTED" if supported else "NOT_SUPPORTED"})
    return {"format": FORMAT + ".evaluation", "protocol_sha256": sha256_file(PROTOCOL), "plan_hash": PLAN_HASH,
            "fresh_inventory_sha256": next(iter(inventories)), "hypotheses": verdicts,
            "families": {f: {"run": str(run), "prompts": len(results), "describe": describe(stimuli, results)}
                         for f, (run, _, stimuli, results) in families.items()}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--lock", type=Path, required=True)
    go.add_argument("--cache", type=Path, default=PROJECT / "model-cache")
    go.add_argument("--device", default="cuda")
    go.add_argument("--run-dir", type=Path)
    check = sub.add_parser("evaluate")
    check.add_argument("--runs", type=Path, nargs="+", required=True)
    check.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "evaluate":
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        result = evaluate(args.runs)
        write_json(args.output, result)
        for h in result["hypotheses"]:
            print(json.dumps({k: h.get(k) for k in ("id", "measure", "mean", "interval", "status")}))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p16-confirm-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timings, started = {}, time.perf_counter()
    status, error, trace = "FAILED", None, None
    try:
        run(args, run_dir, timings)
        status = "COMPLETE"
    except BaseException as failure:  # recorded in completion.json, then re-raised
        error, trace = f"{type(failure).__name__}: {failure}", traceback.format_exc()
        raise
    finally:
        results = run_dir.path / "results.jsonl"
        write_json(run_dir.path / "completion.json", {
            "status": status, "error": error, "traceback": trace,
            "rows_written": sum(1 for _ in results.open()) if results.exists() else 0,
            "results_sha256": sha256_file(results) if results.exists() else None,
            "timings_seconds": timings, "wall_seconds": time.perf_counter() - started,
            "setup_seconds": float(os.environ["OWL_SETUP_SECONDS"]) if "OWL_SETUP_SECONDS" in os.environ else None,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        })
        print(f"{status}: {run_dir.path}", flush=True)


if __name__ == "__main__":
    main()
