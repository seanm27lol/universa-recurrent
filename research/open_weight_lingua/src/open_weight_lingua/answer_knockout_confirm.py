"""Phase Eight confirmatory: answer routing on validation_b, by attention knockout.

protocols/phase_eight_confirmatory_validation_b.md is the frozen design.
Example: for `x = 7 / y = 16 / y = y + 1 / y = y - 3 / What is x?`, the
exploratory run found that Qwen's answer position must read the program in
layers 21-23, while Gemma's answer positions never need it directly and its
question tokens read it in layers 24-29. This module asks the 1,024
never-opened validation_b prompts once, under only the knockouts the six
pre-registered hypotheses need, and tests those hypotheses.

    run       one family on the GPU: build validation_b, record hashes, knock out
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
from .splits import build_plan
from .target import TARGET_BUCKET
from .tasks import interpret

PROTOCOL = PROJECT / "protocols" / "phase_eight_confirmatory_validation_b.md"
FORMAT = "open_weight_lingua.answer_knockout_confirm.v1"
PLAN_HASH = "ff060012d35d92c480a51400a02597bea7421a525c44d97043a56dd3f3de667f"
SPLIT = "validation_b"
KNOCKOUTS = {
    "gemma3-12b": (("after_program", 24), ("after_program", 30), ("answer", 0)),
    "qwen2.5-7b": (("answer", 21), ("answer", 24), ("after_program", 24)),
}
RESAMPLES = 10_000
BOOTSTRAP_SEEDS = {"gemma3-12b": 802100, "qwen2.5-7b": 802101}
COVERAGE = 1 - 0.05 / 6
MARGIN = 0.05
HYPOTHESES = (
    ("G1", "gemma3-12b", "after_program@30", "survives"),
    ("G2", "gemma3-12b", "after_program@24", "damaged"),
    ("G3", "gemma3-12b", "answer@0", "survives"),
    ("Q1", "qwen2.5-7b", "answer@24", "survives"),
    ("Q2", "qwen2.5-7b", "answer@21", "damaged"),
    ("Q3", "qwen2.5-7b", "after_program@24", "survives"),
)


# ---------------------------------------------------------------- stimuli


def validation_rows(plan, tokenizer) -> list[dict]:
    """Original-format validation_b prompts, tokenized, with spans and breakdown labels."""
    if plan.plan_hash != PLAN_HASH:
        raise ValueError("split plan differs from the frozen Phase Two plan")
    rows = []
    for group in plan.groups[SPLIT]:
        for base in group.variants():
            statements = program(base["prompt"])
            variable = base["variable"]
            if str(interpret(statements)[variable]) != base["answer"]:
                raise ValueError("reference answer disagrees with the interpreter")
            ids = list(tokenizer.apply_chat_template([{"role": "user", "content": base["prompt"]}], tokenize=True, add_generation_prompt=True))
            last_update = next(s for s in reversed(statements) if s.variable == variable)
            row = {"id": base["id"], "group_id": base["group_id"], "variable": variable, "prompt": base["prompt"],
                   "answer": base["answer"], "input_ids": ids,
                   "asked_last_update_kind": KIND[last_update.operation],
                   "last_line_assigns_asked": statements[-1].variable == variable}
            row["span"] = list(program_span(tokenizer, ids, program_text(base["prompt"])))
            sequence = teacher_forced(row, tokenizer)
            if len(sequence["ids"]) > TARGET_BUCKET:
                raise ValueError(f"{row['id']}: prompt plus answer does not fit the pinned bucket")
            row.update(sequence=sequence)
            rows.append(row)
    return rows


# ---------------------------------------------------------------- run (GPU)


def _family(lock: Path) -> str:
    return lock.stem.replace("model-lock-", "") if lock.stem != "model-lock" else "qwen2.5-7b"


def evaluate_row(model, row: dict, family: str, device: str) -> dict:
    ids, positions, targets = row["sequence"]["ids"], row["sequence"]["positions"], row["sequence"]["targets"]
    baseline = logits_at(model, ids, positions, device)
    with knockout(model, 0, positions, []):
        if not torch.equal(logits_at(model, ids, positions, device), baseline):
            raise RuntimeError(f"{row['id']}: the no-op hook changed the logits")
    first, last = row["span"]
    keys = list(range(first, last + 1))
    queries = {"answer": positions, "after_program": list(range(last + 1, len(ids)))}
    out = {"baseline": score(baseline, targets)}
    for variant, layer in KNOCKOUTS[family]:
        with knockout(model, layer, queries[variant], keys):
            out[f"{variant}@{layer}"] = score(logits_at(model, ids, positions, device), targets)
    return out


def run(args, run_dir: RunDirectory, timings: dict) -> dict:
    family = _family(args.lock)
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models({**lock, "models": {"target": lock["models"]["target"]}}, {"target": paths["target"]})
    timings["hash_verification"] = time.perf_counter() - start
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(paths["target"], local_files_only=True)
    rows = validation_rows(build_plan(), tokenizer)
    stimuli = run_dir.path / "stimuli.json"
    write_json(stimuli, rows)
    write_json(run_dir.path / "manifest.json", {
        "format": FORMAT, "family": family, "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(), "protocol_sha256": sha256_file(PROTOCOL), "source_file_sha256": source_identity(),
        "model_lock_sha256": sha256_file(args.lock), "plan_hash": PLAN_HASH, "split": SPLIT,
        "stimuli_sha256": sha256_file(stimuli), "prompts": len(rows), "groups": len({r["group_id"] for r in rows}),
        "knockouts": [f"{v}@{k}" for v, k in KNOCKOUTS[family]],
        "verified_artifact_bytes": verified, "software": compatibility(args.device),
    })
    start = time.perf_counter()
    model = load_model(paths["target"], "target", args.device)
    timings["loading"] = time.perf_counter() - start
    if family in KNOCKOUTS and max(k for _, k in KNOCKOUTS[family]) >= len(decoder_layers(model)):
        raise ValueError("a frozen knockout layer exceeds the model's depth")
    start = time.perf_counter()
    with (run_dir.path / "results.jsonl").open("x") as stream:
        for index, row in enumerate(rows):
            stream.write(json.dumps({"id": row["id"], **evaluate_row(model, row, family, args.device)}) + "\n")
            stream.flush()
            if index % 128 == 0:
                print(f"  prompts {index}/{len(rows)}", flush=True)
    timings["knockouts"] = time.perf_counter() - start
    del model
    release_models()
    return {"prompts": len(rows)}


# ---------------------------------------------------------------- evaluate (CPU)


def load_family(run: Path) -> tuple[str, list[dict], list[dict]]:
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
    return manifest["family"], stimuli, results


def bootstrap_losses(stimuli, results, condition: str, seed: int) -> dict:
    groups = sorted({s["group_id"] for s in stimuli})
    index = {g: i for i, g in enumerate(groups)}
    sums, counts = np.zeros(len(groups)), np.zeros(len(groups))
    for s, r in zip(stimuli, results):
        sums[index[s["group_id"]]] += int(r["baseline"]["correct"]) - int(r[condition]["correct"])
        counts[index[s["group_id"]]] += 1
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(RESAMPLES, len(groups)))
    resampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1 - COVERAGE) / 2
    return {"mean_loss": float(sums.sum() / counts.sum()),
            "interval": [float(np.quantile(resampled, tail)), float(np.quantile(resampled, 1 - tail))]}


def accuracy_table(stimuli, results, conditions, keep=lambda s: True) -> dict:
    chosen = [r for s, r in zip(stimuli, results) if keep(s)]
    return {c: {"n": len(chosen), "accuracy": float(np.mean([r[c]["correct"] for r in chosen])) if chosen else None}
            for c in conditions}


def evaluate(runs: list[Path]) -> dict:
    families = {}
    for run in runs:
        family, stimuli, results = load_family(run)
        families[family] = (run, stimuli, results)
    verdicts, summaries = [], {}
    for family, (run, stimuli, results) in families.items():
        conditions = ["baseline"] + [f"{v}@{k}" for v, k in KNOCKOUTS[family]]
        strata = {name: {str(level): accuracy_table(stimuli, results, conditions, lambda s, n=name, v=level: s[n] == v) for level in levels}
                  for name, levels in (("asked_last_update_kind", ("literal", "arithmetic", "copy")),
                                       ("last_line_assigns_asked", (True, False)))}
        summaries[family] = {"run": str(run), "prompts": len(results), "accuracy": accuracy_table(stimuli, results, conditions),
                             "strata": strata,
                             "bootstrap": {"resamples": RESAMPLES, "seed": BOOTSTRAP_SEEDS[family], "coverage": COVERAGE}}
    for test_id, family, condition, claim in HYPOTHESES:
        if family not in families:
            verdicts.append({"id": test_id, "family": family, "condition": condition, "claim": claim, "status": "NOT_RUN"})
            continue
        _, stimuli, results = families[family]
        loss = bootstrap_losses(stimuli, results, condition, BOOTSTRAP_SEEDS[family])
        low, high = loss["interval"]
        supported = high < MARGIN if claim == "survives" else low > MARGIN
        verdicts.append({"id": test_id, "family": family, "condition": condition, "claim": claim, **loss,
                         "coverage": COVERAGE, "status": "SUPPORTED" if supported else "NOT_SUPPORTED"})
    return {"format": FORMAT + ".evaluation", "protocol_sha256": sha256_file(PROTOCOL), "plan_hash": PLAN_HASH,
            "hypotheses": verdicts, "families": summaries}


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
            print(json.dumps({k: h.get(k) for k in ("id", "condition", "claim", "mean_loss", "interval", "status")}))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p8-confirm-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
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
