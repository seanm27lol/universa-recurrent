"""Phase Eleven confirmatory: a disrupted final assignment, on fresh programs.

protocols/phase_eleven_confirmatory_fresh.md is the frozen design. Example:

    x = 4 / y = 3 / x = 7 / y = 5 / What is x?      -> 7 (superseded value 4)

Exploratory Phases Ten and Eleven found that blocking the final statement
`x = 7` from earlier lines (of either variable) damages the answer, and
that the model then mostly answers the superseded value 4. This module
generates 512 new groups (seed 207001, excluding every Phase Two program and
Phase Nine's fresh split), runs the literal-final prompts once under both
blocks, and tests six pre-registered claims.

    run       one family on the GPU: generate, record hashes, knock out
    evaluate  CPU: losses, superseded gains, Bonferroni bootstrap, verdicts
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback
import uuid

import numpy as np
import torch

from .artifacts import RunDirectory, sha256_file, write_json
from .edit_diagnostics import PROJECT, _git_head
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .runner import release_models, source_identity
from .span_knockout_confirm import fresh_groups as previous_fresh_groups
from .splits import SPLIT_ORDER, build_plan
from .supersession_knockout import evaluate_row, supersession_row
from .target import TARGET_BUCKET
from .tasks import canonical, generate_groups

PROTOCOL = PROJECT / "protocols" / "phase_eleven_confirmatory_fresh.md"
FORMAT = "open_weight_lingua.supersession_confirm.v1"
PLAN_HASH = "ff060012d35d92c480a51400a02597bea7421a525c44d97043a56dd3f3de667f"
PREVIOUS_FRESH_INVENTORY = "8e0d30ab8915e08f465ed29c2861fdd194f4a02c7e73afaabc6676cb9efff963"
SPLIT = "fresh_2026_10b"
GROUPS = 512
RESAMPLES = 10_000
BOOTSTRAP_SEEDS = {"gemma3-12b": 804100, "qwen2.5-7b": 804101}
COVERAGE = 1 - 0.05 / 6
HYPOTHESES = (
    ("G1", "gemma3-12b", "loss_same", 0.05),
    ("G2", "gemma3-12b", "loss_other", 0.05),
    ("G3", "gemma3-12b", "superseded_gain_same", 0.10),
    ("Q1", "qwen2.5-7b", "loss_same", 0.05),
    ("Q2", "qwen2.5-7b", "loss_other", 0.05),
    ("Q3", "qwen2.5-7b", "superseded_gain_same", 0.10),
)


# ---------------------------------------------------------------- fresh programs


def fresh_groups(plan, count: int = GROUPS):
    """New groups excluding the five Phase Two splits and Phase Nine's verified fresh split."""
    if plan.plan_hash != PLAN_HASH:
        raise ValueError("split plan differs from the frozen Phase Two plan")
    previous, previous_stats = previous_fresh_groups(plan)
    if previous_stats["inventory_sha256"] != PREVIOUS_FRESH_INVENTORY:
        raise ValueError("Phase Nine's fresh split does not regenerate to its recorded inventory")
    excluded = {canonical(p) for name in SPLIT_ORDER for g in plan.groups[name] for p in (g.source, g.counterfactual)}
    excluded |= {canonical(p) for g in previous for p in (g.source, g.counterfactual)}
    groups, stats = generate_groups(SPLIT, count, excluded_programs=excluded)
    if {canonical(p) for g in groups for p in (g.source, g.counterfactual)} & excluded:
        raise ValueError("a fresh program repeats an earlier program")
    return groups, stats


def fresh_rows(groups, tokenizer) -> list[dict]:
    """Literal-final prompts with earlier assignments and a superseded value different from the answer."""
    rows = []
    for group in groups:
        for base in group.variants():
            ids = list(tokenizer.apply_chat_template([{"role": "user", "content": base["prompt"]}], tokenize=True, add_generation_prompt=True))
            row = supersession_row({**base, "input_ids": ids}, SPLIT, tokenizer)
            if row is None or row["kind"] != "literal" or "superseded_sequence" not in row:
                continue
            if max(len(row["sequence"]["ids"]), len(row["superseded_sequence"]["ids"])) > TARGET_BUCKET:
                raise ValueError(f"{row['id']}: prompt plus answer does not fit the pinned bucket")
            rows.append({**row, "prompt": base["prompt"]})
    return rows


# ---------------------------------------------------------------- run (GPU)


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
        "git_head": _git_head(), "protocol_sha256": sha256_file(PROTOCOL), "source_file_sha256": source_identity(),
        "model_lock_sha256": sha256_file(args.lock), "plan_hash": PLAN_HASH, "split": SPLIT, "generation": stats,
        "previous_fresh_inventory": PREVIOUS_FRESH_INVENTORY, "stimuli_sha256": sha256_file(stimuli),
        "prompts": len(rows), "groups_with_prompts": len({r["group_id"] for r in rows}),
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
            if index % 64 == 0:
                print(f"  prompts {index}/{len(rows)}", flush=True)
    timings["knockouts"] = time.perf_counter() - start
    del model
    release_models()
    return {"prompts": len(rows)}


# ---------------------------------------------------------------- evaluate (CPU)


def measures(result: dict) -> dict:
    base = result["baseline"]
    return {"loss_same": int(base["correct"]) - int(result["same_variable"]["correct"]),
            "loss_other": int(base["correct"]) - int(result["other_variable"]["correct"]),
            "superseded_gain_same": int(result["same_variable"]["superseded_correct"]) - int(base["superseded_correct"])}


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


def bootstrap(stimuli, results, name: str, seed: int) -> dict:
    groups = sorted({s["group_id"] for s in stimuli})
    index = {g: i for i, g in enumerate(groups)}
    sums, counts = np.zeros(len(groups)), np.zeros(len(groups))
    for s, r in zip(stimuli, results):
        sums[index[s["group_id"]]] += measures(r)[name]
        counts[index[s["group_id"]]] += 1
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(RESAMPLES, len(groups)))
    resampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1 - COVERAGE) / 2
    return {"mean": float(sums.sum() / counts.sum()), "prompts": int(counts.sum()), "groups": len(groups),
            "interval": [float(np.quantile(resampled, tail)), float(np.quantile(resampled, 1 - tail))]}


def describe(results) -> dict:
    out = {}
    for condition in ("baseline", "same_variable", "other_variable"):
        scored = [r[condition] for r in results]
        wrong = [s for s in scored if not s["correct"]]
        out[condition] = {"accuracy": float(np.mean([s["correct"] for s in scored])),
                          "superseded_rate": float(np.mean([s["superseded_correct"] for s in scored])),
                          "mean_preference": float(np.mean([s["preference"] for s in scored])),
                          "superseded_share_of_wrong": float(np.mean([s["superseded_correct"] for s in wrong])) if wrong else None,
                          "wrong": len(wrong)}
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
    for test_id, family, name, threshold in HYPOTHESES:
        if family not in families:
            verdicts.append({"id": test_id, "family": family, "measure": name, "status": "NOT_RUN"})
            continue
        _, _, stimuli, results = families[family]
        estimate = bootstrap(stimuli, results, name, BOOTSTRAP_SEEDS[family])
        verdicts.append({"id": test_id, "family": family, "measure": name, "threshold": threshold, **estimate,
                         "coverage": COVERAGE, "status": "SUPPORTED" if estimate["interval"][0] > threshold else "NOT_SUPPORTED"})
    return {"format": FORMAT + ".evaluation", "protocol_sha256": sha256_file(PROTOCOL), "plan_hash": PLAN_HASH,
            "fresh_inventory_sha256": next(iter(inventories)), "hypotheses": verdicts,
            "families": {f: {"run": str(run), "prompts": len(results), **describe(results)} for f, (run, _, _, results) in families.items()}}


# ---------------------------------------------------------------- CLI


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
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p11-confirm-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
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
