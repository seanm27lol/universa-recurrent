"""Phase Thirteen confirmatory: the account at Gemma-3-27B, on a fourth fresh split.

protocols/phase_thirteen_confirmatory_fresh.md is the frozen design. Example:

    x = 2 / y = 3 / x = x + 9 / y = 5 / x = 7 / What is x?      -> 7  (v1 = 11, v2 = 2)

Exploratory Phase Thirteen found that Gemma-3-27B routes the late read like
Gemma-3-12B (its answer positions never need the program; every post-program
position needs it until layer 46 of 62), and that it takes the most recent
intact assignment. This module generates 1,536 new groups (seed 209001,
excluding Phase Two and the three earlier fresh splits), runs the routing
knockouts on the first 512 groups and the selection degradations on every
eligible prompt, once, and tests six pre-registered claims.

    run       Gemma-3-27B on the GPU: generate, record hashes, knock out
    evaluate  CPU: paired losses and gains, Bonferroni bootstrap, verdicts
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

from .answer_knockout import knockout, logits_at, program_span, score, teacher_forced
from .architectures import decoder_layers
from .artifacts import RunDirectory, sha256_file, write_json
from .description_census import program, program_text
from .edit_diagnostics import PROJECT, _git_head
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .runner import release_models, source_identity
from .selection_confirm import fresh_groups as third_fresh_groups
from .selection_confirm import measures as selection_measures
from .selection_knockout import CONDITIONS as SELECTION_CONDITIONS
from .selection_knockout import FRESH_INVENTORIES as EARLIER_INVENTORIES
from .selection_knockout import VALUES, selection_row
from .selection_knockout import evaluate_row as evaluate_selection
from .span_knockout_confirm import fresh_groups as first_fresh_groups
from .splits import SPLIT_ORDER, build_plan
from .supersession_confirm import fresh_groups as second_fresh_groups
from .target import TARGET_BUCKET
from .tasks import canonical, generate_groups, interpret

PROTOCOL = PROJECT / "protocols" / "phase_thirteen_confirmatory_fresh.md"
FORMAT = "open_weight_lingua.scale_confirm.v1"
PLAN_HASH = "ff060012d35d92c480a51400a02597bea7421a525c44d97043a56dd3f3de667f"
FAMILY = "gemma3-27b"
LAYERS = 62
SPLIT = "fresh_2026_10d"
GROUPS = 1536
ROUTING_GROUPS = 512
FRESH_INVENTORIES = {**EARLIER_INVENTORIES,
                     "fresh_2026_10c": "c56dfba4c83d36200c18192a1ae2bad31807996b8f852e8914621592a9a1a254"}
ROUTES = {"answer@0": ("answer", 0), "after_program@38": ("after_program", 38), "after_program@46": ("after_program", 46)}
RESAMPLES = 10_000
BOOTSTRAP_SEED = 806100
COVERAGE = 1 - 0.05 / 6
HYPOTHESES = (
    ("R1", "routing", "answer@0", "below", 0.05),
    ("R2", "routing", "after_program@46", "below", 0.05),
    ("R3", "routing", "after_program@38", "above", 0.05),
    ("S1", "selection", "v1_gain", "above", 0.10),
    ("S2", "selection", "v2_gain", "above", 0.10),
    ("S3", "selection", "loss_previous", "below", 0.05),
)


# ---------------------------------------------------------------- fresh programs


def fresh_groups(plan, count: int = GROUPS):
    """New groups excluding Phase Two and the three earlier fresh splits (each verified against its inventory)."""
    if plan.plan_hash != PLAN_HASH:
        raise ValueError("split plan differs from the frozen Phase Two plan")
    excluded = {canonical(p) for name in SPLIT_ORDER for g in plan.groups[name] for p in (g.source, g.counterfactual)}
    for name, generate in (("fresh_2026_10", first_fresh_groups), ("fresh_2026_10b", second_fresh_groups),
                           ("fresh_2026_10c", third_fresh_groups)):
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


def routing_rows(groups, tokenizer) -> list[dict]:
    """Every prompt of the first ROUTING_GROUPS groups, with its program span and teacher-forced answer."""
    rows = []
    for group in groups[:ROUTING_GROUPS]:
        for base in group.variants():
            if str(interpret(program(base["prompt"]))[base["variable"]]) != base["answer"]:
                raise ValueError("reference answer disagrees with the interpreter")
            ids = _ids(tokenizer, base["prompt"])
            row = {"id": base["id"], "group_id": base["group_id"], "variable": base["variable"], "prompt": base["prompt"],
                   "answer": base["answer"], "input_ids": ids, "span": list(program_span(tokenizer, ids, program_text(base["prompt"])))}
            row["sequence"] = teacher_forced(row, tokenizer)
            if len(row["sequence"]["ids"]) > TARGET_BUCKET:
                raise ValueError(f"{row['id']}: prompt plus answer does not fit the pinned bucket")
            rows.append(row)
    return rows


def selection_rows(groups, tokenizer) -> list[dict]:
    """Every eligible prompt of all groups, by Phase Twelve's rule."""
    rows = []
    for group in groups:
        for base in group.variants():
            row = selection_row({**base, "input_ids": _ids(tokenizer, base["prompt"])}, SPLIT, tokenizer)
            if row is not None:
                rows.append({**row, "prompt": base["prompt"]})
    return rows


# ---------------------------------------------------------------- run (GPU)


def evaluate_routing(model, row: dict, device: str) -> dict:
    ids, positions, targets = row["sequence"]["ids"], row["sequence"]["positions"], row["sequence"]["targets"]
    baseline = logits_at(model, ids, positions, device)
    with knockout(model, 0, positions, []):
        if not torch.equal(logits_at(model, ids, positions, device), baseline):
            raise RuntimeError(f"{row['id']}: the no-op hook changed the logits")
    keys = list(range(row["span"][0], row["span"][1] + 1))
    queries = {"answer": positions, "after_program": list(range(row["span"][1] + 1, len(ids)))}
    out = {"baseline": score(baseline, targets)}
    for name, (variant, layer) in ROUTES.items():
        with knockout(model, layer, queries[variant], keys):
            out[name] = score(logits_at(model, ids, positions, device), targets)
    return out


def _family(lock: Path) -> str:
    return lock.stem.replace("model-lock-", "") if lock.stem != "model-lock" else "qwen2.5-7b"


def run(args, run_dir: RunDirectory, timings: dict) -> dict:
    family = _family(args.lock)
    if family != FAMILY:
        raise ValueError(f"this protocol is for {FAMILY}, not {family}")
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models({**lock, "models": {"target": lock["models"]["target"]}}, {"target": paths["target"]})
    timings["hash_verification"] = time.perf_counter() - start
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(paths["target"], local_files_only=True)
    groups, stats = fresh_groups(build_plan())
    routing, selection = routing_rows(groups, tokenizer), selection_rows(groups, tokenizer)
    stimuli = run_dir.path / "stimuli.json"
    write_json(stimuli, {"routing": routing, "selection": selection})
    write_json(run_dir.path / "manifest.json", {
        "format": FORMAT, "family": family, "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(), "protocol_path": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL),
        "source_file_sha256": source_identity(), "model_lock_sha256": sha256_file(args.lock),
        "plan_hash": PLAN_HASH, "split": SPLIT, "generation": stats, "excluded_fresh_inventories": FRESH_INVENTORIES,
        "stimuli_sha256": sha256_file(stimuli), "routes": ROUTES,
        "prompts": {"routing": len(routing), "selection": len(selection)},
        "groups_with_prompts": {"routing": len({r["group_id"] for r in routing}), "selection": len({r["group_id"] for r in selection})},
        "verified_artifact_bytes": verified, "software": compatibility(args.device),
    })
    start = time.perf_counter()
    model = load_model(paths["target"], "target", args.device)
    if len(decoder_layers(model)) != LAYERS:
        raise ValueError(f"expected {LAYERS} decoder layers")
    timings["loading"] = time.perf_counter() - start
    start = time.perf_counter()
    with (run_dir.path / "results.jsonl").open("x") as stream:
        for part, rows, evaluate_one in (("routing", routing, evaluate_routing), ("selection", selection, evaluate_selection)):
            for index, row in enumerate(rows):
                stream.write(json.dumps({"part": part, "id": row["id"], **evaluate_one(model, row, args.device)}) + "\n")
                stream.flush()
                if index % 128 == 0:
                    print(f"  {part} {index}/{len(rows)}", flush=True)
            timings[part] = time.perf_counter() - start
            start = time.perf_counter()
    del model
    release_models()
    return {"prompts": len(routing) + len(selection)}


# ---------------------------------------------------------------- evaluate (CPU)


def measure(part: str, name: str, result: dict) -> int:
    if part == "routing":
        return int(result["baseline"]["correct"]) - int(result[name]["correct"])
    return selection_measures(result)[name]


def load_run(run: Path):
    manifest = json.loads((run / "manifest.json").read_text())
    completion = json.loads((run / "completion.json").read_text())
    lines = [json.loads(line) for line in (run / "results.jsonl").open()]
    stimuli = json.loads((run / "stimuli.json").read_text())
    if completion["status"] != "COMPLETE" or len(lines) != sum(manifest["prompts"].values()):
        raise ValueError(f"{run}: incomplete runs never enter the confirmatory analysis")
    if manifest["family"] != FAMILY or manifest["plan_hash"] != PLAN_HASH or manifest["protocol_sha256"] != sha256_file(PROTOCOL):
        raise ValueError(f"{run}: family, plan or protocol differs from the frozen ones")
    if manifest["stimuli_sha256"] != sha256_file(run / "stimuli.json"):
        raise ValueError(f"{run}: stimuli differ from the recorded hash")
    results = {}
    for part in ("routing", "selection"):
        results[part] = [r for r in lines if r["part"] == part]
        if [s["id"] for s in stimuli[part]] != [r["id"] for r in results[part]]:
            raise ValueError(f"{run}: {part} stimuli and results disagree")
    return manifest, stimuli, results


def bootstrap(stimuli, results, part: str, name: str) -> dict:
    groups = sorted({s["group_id"] for s in stimuli})
    index = {g: i for i, g in enumerate(groups)}
    sums, counts = np.zeros(len(groups)), np.zeros(len(groups))
    for s, r in zip(stimuli, results):
        sums[index[s["group_id"]]] += measure(part, name, r)
        counts[index[s["group_id"]]] += 1
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(groups), size=(RESAMPLES, len(groups)))
    resampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1 - COVERAGE) / 2
    return {"mean": float(sums.sum() / counts.sum()), "prompts": int(counts.sum()), "groups": len(groups),
            "interval": [float(np.quantile(resampled, tail)), float(np.quantile(resampled, 1 - tail))]}


def evaluate(run: Path) -> dict:
    manifest, stimuli, results = load_run(run)
    verdicts = []
    for test_id, part, name, direction, threshold in HYPOTHESES:
        estimate = bootstrap(stimuli[part], results[part], part, name)
        low, high = estimate["interval"]
        supported = low > threshold if direction == "above" else high < threshold
        verdicts.append({"id": test_id, "part": part, "measure": name, "direction": direction, "threshold": threshold,
                         **estimate, "coverage": COVERAGE, "status": "SUPPORTED" if supported else "NOT_SUPPORTED"})
    accuracy = {c: float(np.mean([r[c]["correct"] for r in results["routing"]])) for c in ("baseline", *ROUTES)}
    says = {c: {v: float(np.mean([r[c][v]["correct"] for r in results["selection"]])) for v in VALUES} for c in SELECTION_CONDITIONS}
    return {"format": FORMAT + ".evaluation", "protocol_sha256": sha256_file(PROTOCOL), "plan_hash": PLAN_HASH,
            "run": str(run), "fresh_inventory_sha256": manifest["generation"]["inventory_sha256"], "hypotheses": verdicts,
            "routing_accuracy": accuracy, "selection_says": says,
            "prompts": {part: len(results[part]) for part in results}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--lock", type=Path, required=True)
    go.add_argument("--cache", type=Path, default=PROJECT / "model-cache")
    go.add_argument("--device", default="cuda")
    go.add_argument("--run-dir", type=Path)
    check = sub.add_parser("evaluate")
    check.add_argument("--run", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "evaluate":
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        result = evaluate(args.run)
        write_json(args.output, result)
        for h in result["hypotheses"]:
            print(json.dumps({k: h.get(k) for k in ("id", "measure", "mean", "interval", "status")}))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p13-confirm-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
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
