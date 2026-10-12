"""Phase Nine confirmatory: which statements the late read uses, on freshly generated programs.

protocols/phase_nine_confirmatory_fresh.md is the frozen design. Example: for
`x = 7 / y = 16 / y = y + 1 / y = y - 3 / What is x?`, the exploratory run
found that blocking the asked variable's lines inside each model's late route
is as damaging as blocking the whole program, while blocking the other
variable's lines, or (when the program ends on the other variable) the last
token, costs nothing. This module generates 256 new groups (seed 206001,
excluding every Phase Two program), asks their 1,024 prompts once under those
three knockouts, and tests six pre-registered claims.

    run       one family on the GPU: generate, record hashes, knock out
    evaluate  CPU: losses, Bonferroni bootstrap intervals, verdicts, breakdowns
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

from .answer_knockout import KIND, knockout, logits_at, program_span, score, teacher_forced
from .architectures import decoder_layers
from .artifacts import RunDirectory, sha256_file, write_json
from .description_census import program, program_text
from .edit_diagnostics import PROJECT, _git_head
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .runner import release_models, source_identity
from .span_knockout import ROUTES, spans_for, token_lines
from .splits import SPLIT_ORDER, build_plan
from .target import TARGET_BUCKET
from .tasks import canonical, generate_groups, interpret

PROTOCOL = PROJECT / "protocols" / "phase_nine_confirmatory_fresh.md"
FORMAT = "open_weight_lingua.span_knockout_confirm.v1"
PLAN_HASH = "ff060012d35d92c480a51400a02597bea7421a525c44d97043a56dd3f3de667f"
SPLIT = "fresh_2026_10"
GROUPS = 256
CONDITIONS = ("asked_lines_minus_last", "other_lines_minus_last", "last_token")
RESAMPLES = 10_000
BOOTSTRAP_SEEDS = {"gemma3-12b": 803100, "qwen2.5-7b": 803101}
COVERAGE = 1 - 0.05 / 6
MARGIN = 0.05
HYPOTHESES = (
    ("G1", "gemma3-12b", "asked_lines_minus_last", "all", "damaged"),
    ("G2", "gemma3-12b", "other_lines_minus_last", "all", "survives"),
    ("G3", "gemma3-12b", "last_token", "other_last", "survives"),
    ("Q1", "qwen2.5-7b", "asked_lines_minus_last", "all", "damaged"),
    ("Q2", "qwen2.5-7b", "other_lines_minus_last", "all", "survives"),
    ("Q3", "qwen2.5-7b", "last_token", "other_last", "survives"),
)


# ---------------------------------------------------------------- fresh programs


def fresh_groups(plan, count: int = GROUPS):
    """New groups under SPLIT, excluding every canonical program of the frozen five-split plan."""
    if plan.plan_hash != PLAN_HASH:
        raise ValueError("split plan differs from the frozen Phase Two plan")
    excluded = {canonical(program_) for name in SPLIT_ORDER for group in plan.groups[name]
                for program_ in (group.source, group.counterfactual)}
    groups, stats = generate_groups(SPLIT, count, excluded_programs=excluded)
    fresh = {canonical(p) for group in groups for p in (group.source, group.counterfactual)}
    if fresh & excluded:
        raise ValueError("a fresh program repeats a Phase Two program")
    return groups, stats


def fresh_rows(groups, tokenizer) -> list[dict]:
    rows = []
    for group in groups:
        for base in group.variants():
            statements = program(base["prompt"])
            variable = base["variable"]
            if str(interpret(statements)[variable]) != base["answer"]:
                raise ValueError("reference answer disagrees with the interpreter")
            ids = list(tokenizer.apply_chat_template([{"role": "user", "content": base["prompt"]}], tokenize=True, add_generation_prompt=True))
            body = program_text(base["prompt"])
            span = program_span(tokenizer, ids, body)
            lines = token_lines(tokenizer, ids, span, body)
            last_update = next(s for s in reversed(statements) if s.variable == variable)
            row = {"id": base["id"], "group_id": base["group_id"], "variable": variable, "prompt": base["prompt"],
                   "answer": base["answer"], "input_ids": ids, "span": list(span),
                   "asked_last_update_kind": KIND[last_update.operation],
                   "last_line_assigns_asked": statements[-1].variable == variable,
                   "asked_copies_other": any(s.variable == variable and s.operation == "copy" for s in statements),
                   "spans": spans_for(statements, variable, span, lines)}
            row["sequence"] = teacher_forced(row, tokenizer)
            if len(row["sequence"]["ids"]) > TARGET_BUCKET:
                raise ValueError(f"{row['id']}: prompt plus answer does not fit the pinned bucket")
            rows.append(row)
    return rows


# ---------------------------------------------------------------- run (GPU)


def _family(lock: Path) -> str:
    return lock.stem.replace("model-lock-", "") if lock.stem != "model-lock" else "qwen2.5-7b"


def evaluate_row(model, row: dict, route: tuple[str, int], device: str) -> dict:
    ids, positions, targets = row["sequence"]["ids"], row["sequence"]["positions"], row["sequence"]["targets"]
    baseline = logits_at(model, ids, positions, device)
    with knockout(model, 0, positions, []):
        if not torch.equal(logits_at(model, ids, positions, device), baseline):
            raise RuntimeError(f"{row['id']}: the no-op hook changed the logits")
    variant, layer = route
    queries = positions if variant == "answer" else list(range(row["span"][1] + 1, len(ids)))
    out = {"baseline": score(baseline, targets)}
    for name in CONDITIONS:
        with knockout(model, layer, queries, row["spans"][name]):
            out[name] = score(logits_at(model, ids, positions, device), targets)
    return out


def run(args, run_dir: RunDirectory, timings: dict) -> dict:
    family = _family(args.lock)
    route = ROUTES[family]
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
        "model_lock_sha256": sha256_file(args.lock), "plan_hash": PLAN_HASH, "split": SPLIT,
        "generation": stats, "stimuli_sha256": sha256_file(stimuli), "prompts": len(rows), "groups": len(groups),
        "route": route, "conditions": CONDITIONS,
        "verified_artifact_bytes": verified, "software": compatibility(args.device),
    })
    start = time.perf_counter()
    model = load_model(paths["target"], "target", args.device)
    timings["loading"] = time.perf_counter() - start
    if route[1] >= len(decoder_layers(model)):
        raise ValueError("the frozen route layer exceeds the model's depth")
    start = time.perf_counter()
    with (run_dir.path / "results.jsonl").open("x") as stream:
        for index, row in enumerate(rows):
            stream.write(json.dumps({"id": row["id"], **evaluate_row(model, row, route, args.device)}) + "\n")
            stream.flush()
            if index % 128 == 0:
                print(f"  prompts {index}/{len(rows)}", flush=True)
    timings["knockouts"] = time.perf_counter() - start
    del model
    release_models()
    return {"prompts": len(rows)}


# ---------------------------------------------------------------- evaluate (CPU)


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


def bootstrap_loss(stimuli, results, condition: str, keep, seed: int) -> dict:
    groups = sorted({s["group_id"] for s in stimuli})
    index = {g: i for i, g in enumerate(groups)}
    sums, counts = np.zeros(len(groups)), np.zeros(len(groups))
    for s, r in zip(stimuli, results):
        if keep(s):
            sums[index[s["group_id"]]] += int(r["baseline"]["correct"]) - int(r[condition]["correct"])
            counts[index[s["group_id"]]] += 1
    used = counts > 0
    sums, counts = sums[used], counts[used]
    draws = np.random.default_rng(seed).integers(0, len(sums), size=(RESAMPLES, len(sums)))
    resampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1 - COVERAGE) / 2
    return {"mean_loss": float(sums.sum() / counts.sum()), "prompts": int(counts.sum()), "groups": int(used.sum()),
            "interval": [float(np.quantile(resampled, tail)), float(np.quantile(resampled, 1 - tail))]}


SUBSETS = {"all": lambda s: True, "other_last": lambda s: not s["last_line_assigns_asked"]}


def describe(stimuli, results) -> dict:
    def table(keep):
        chosen = [r for s, r in zip(stimuli, results) if keep(s)]
        return {"n": len(chosen), "baseline": float(np.mean([r["baseline"]["correct"] for r in chosen])) if chosen else None,
                **{c: float(np.mean([r[c]["correct"] for r in chosen])) if chosen else None for c in CONDITIONS}}

    strata = {name: {str(level): table(lambda s, n=name, v=level: s[n] == v) for level in levels}
              for name, levels in (("asked_last_update_kind", ("literal", "arithmetic", "copy")),
                                   ("last_line_assigns_asked", (True, False)), ("asked_copies_other", (True, False)))}
    return {"all": table(lambda s: True), "strata": strata}


def evaluate(runs: list[Path]) -> dict:
    families = {}
    for run in runs:
        manifest, stimuli, results = load_family(run)
        families[manifest["family"]] = (run, manifest, stimuli, results)
    inventories = {f: m["generation"]["inventory_sha256"] for f, (_, m, _, _) in families.items()}
    if len(set(inventories.values())) > 1:
        raise ValueError("the families were given different fresh programs")
    verdicts = []
    for test_id, family, condition, subset, claim in HYPOTHESES:
        if family not in families:
            verdicts.append({"id": test_id, "family": family, "condition": condition, "subset": subset, "claim": claim, "status": "NOT_RUN"})
            continue
        _, _, stimuli, results = families[family]
        loss = bootstrap_loss(stimuli, results, condition, SUBSETS[subset], BOOTSTRAP_SEEDS[family])
        low, high = loss["interval"]
        supported = low > MARGIN if claim == "damaged" else high < MARGIN
        verdicts.append({"id": test_id, "family": family, "condition": condition, "subset": subset, "claim": claim, **loss,
                         "coverage": COVERAGE, "status": "SUPPORTED" if supported else "NOT_SUPPORTED"})
    return {"format": FORMAT + ".evaluation", "protocol_sha256": sha256_file(PROTOCOL), "plan_hash": PLAN_HASH,
            "fresh_inventory_sha256": next(iter(inventories.values())), "hypotheses": verdicts,
            "families": {f: {"run": str(run), **describe(stimuli, results)} for f, (run, _, stimuli, results) in families.items()}}


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
            print(json.dumps({k: h.get(k) for k in ("id", "condition", "subset", "claim", "mean_loss", "interval", "status")}))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p9-confirm-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
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
