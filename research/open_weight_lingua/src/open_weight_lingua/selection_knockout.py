"""Phase Twelve: does the late read take the most recent intact assignment? Exploratory.

protocols/phase_twelve_selection_brief.md is the frozen design. Example:

    x = 2 / y = 3 / x = x + 9 / y = 5 / x = 7 / What is x?      -> 7  (v1 = 11, v2 = 2)

Phase Eleven confirmed that degrading the final assignment `x = 7` makes the
model answer the previous value 11. If the late read takes the most recent
intact assignment, degrading both `x = 7` and `x = x + 9` should push the
answer back to 2. A statement is degraded by blocking its tokens, from layer
0, from every earlier program token.
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

from .answer_knockout import knockout, logits_at, program_span, score, teacher_forced
from .artifacts import RunDirectory, sha256_file, write_json
from .description_census import program, program_text
from .edit_diagnostics import PROJECT, _git_head
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .runner import release_models, source_identity
from .span_knockout import token_lines
from .span_knockout_confirm import fresh_groups as first_fresh_groups
from .splits import build_plan
from .supersession_confirm import fresh_groups as second_fresh_groups
from .supersession_knockout import value_before
from .target import TARGET_BUCKET
from .tasks import interpret

PROTOCOL = PROJECT / "protocols" / "phase_twelve_selection_brief.md"
FORMAT = "open_weight_lingua.selection_knockout.v1"
FRESH_INVENTORIES = {"fresh_2026_10": "8e0d30ab8915e08f465ed29c2861fdd194f4a02c7e73afaabc6676cb9efff963",
                     "fresh_2026_10b": "67dc17bb65b90d78f8453f25193f713b15313be011a984302367a45035e92a39"}
CONDITIONS = ("baseline", "final", "final_and_previous", "previous")
VALUES = ("v0", "v1", "v2")


def selection_row(item: dict, source: str, tokenizer) -> dict | None:
    """Eligible prompts: >= 3 asked assignments, a literal final one, v0/v1/v2 pairwise distinct."""
    statements = program(item["prompt"])
    variable = item["variable"]
    if str(interpret(statements)[variable]) != item["answer"]:
        raise ValueError(f"{item['id']}: answer disagrees with the interpreter")
    assigned = [i for i, s in enumerate(statements) if s.variable == variable]
    if len(assigned) < 3 or statements[assigned[-1]].operation != "assign":
        return None
    final, previous = assigned[-1], assigned[-2]
    values = {"v0": int(item["answer"]), "v1": value_before(statements, final, variable),
              "v2": value_before(statements, previous, variable)}
    if len(set(values.values())) != 3:
        return None
    body = program_text(item["prompt"])
    span = program_span(tokenizer, item["input_ids"], body)
    lines = token_lines(tokenizer, item["input_ids"], span, body)
    tokens = list(range(span[0], span[1] + 1))

    def statement(line):
        return {"queries": [t for t, ln in zip(tokens, lines) if ln == line],
                "keys": [t for t, ln in zip(tokens, lines) if ln < line]}

    row = {"id": item["id"], "group_id": item["group_id"], "source": source, "variable": variable,
           "input_ids": item["input_ids"], "answer": item["answer"], "values": values,
           "final": statement(final), "previous": statement(previous)}
    row["sequences"] = {name: teacher_forced({**row, "answer": str(value)}, tokenizer) for name, value in values.items()}
    if max(len(q["ids"]) for q in row["sequences"].values()) > TARGET_BUCKET:
        raise ValueError(f"{item['id']}: prompt plus answer does not fit the pinned bucket")
    return row


def degraded(model, row: dict, condition: str) -> ExitStack:
    stack = ExitStack()
    for name in {"baseline": (), "final": ("final",), "final_and_previous": ("final", "previous"), "previous": ("previous",)}[condition]:
        stack.enter_context(knockout(model, 0, row[name]["queries"], row[name]["keys"]))
    return stack


def evaluate_row(model, row: dict, device: str) -> dict:
    v0 = row["sequences"]["v0"]
    baseline = logits_at(model, v0["ids"], v0["positions"], device)
    with knockout(model, 0, v0["positions"], []):
        if not torch.equal(logits_at(model, v0["ids"], v0["positions"], device), baseline):
            raise RuntimeError(f"{row['id']}: the no-op hook changed the logits")
    out = {}
    for condition in CONDITIONS:
        with degraded(model, row, condition):
            out[condition] = {name: score(logits_at(model, q["ids"], q["positions"], device), q["targets"])
                              for name, q in row["sequences"].items()}
    return out


def summarize(rows: list[dict], results: list[dict]) -> dict:
    out = {}
    for source in ("all", "reused", "fresh"):
        chosen = [r for row, r in zip(rows, results) if source == "all" or (row["source"] == "reused") == (source == "reused")]
        if not chosen:
            continue
        entry = {"n": len(chosen)}
        for condition in CONDITIONS:
            says = {name: float(np.mean([r[condition][name]["correct"] for r in chosen])) for name in VALUES}
            wrong = [r for r in chosen if not r[condition]["v0"]["correct"]]
            entry[condition] = {"says": says, "wrong": len(wrong),
                                "among_wrong": {name: float(np.mean([r[condition][name]["correct"] for r in wrong])) if wrong else None
                                                for name in ("v1", "v2")},
                                "mean_logprob": {name: float(np.mean([r[condition][name]["logprob"] for r in chosen])) for name in VALUES}}
        out[source] = entry
    return out


# ---------------------------------------------------------------- execution


def _family(lock: Path) -> str:
    return lock.stem.replace("model-lock-", "") if lock.stem != "model-lock" else "qwen2.5-7b"


def pool(args, tokenizer) -> tuple[list[dict], dict]:
    rows, stats = [], {}
    for split, run in (("calibration", args.calibration_run), ("pilot", args.pilot_run)):
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest["model_lock_sha256"] != sha256_file(args.lock):
            raise ValueError("lock differs from the input runs' lock")
        rows += [r for r in (selection_row(item, "reused", tokenizer) for item in manifest["inputs"]) if r]
    plan = build_plan()
    for name, generate in (("fresh_2026_10", first_fresh_groups), ("fresh_2026_10b", second_fresh_groups)):
        groups, generation = generate(plan)
        if generation["inventory_sha256"] != FRESH_INVENTORIES[name]:
            raise ValueError(f"{name} does not regenerate to its recorded inventory")
        stats[name] = generation
        for group in groups:
            for base in group.variants():
                ids = list(tokenizer.apply_chat_template([{"role": "user", "content": base["prompt"]}], tokenize=True, add_generation_prompt=True))
                row = selection_row({**base, "input_ids": ids}, "fresh", tokenizer)
                if row:
                    rows.append(row)
    return rows, stats


def run(args, run_dir: RunDirectory, timings: dict) -> dict:
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models({**lock, "models": {"target": lock["models"]["target"]}}, {"target": paths["target"]})
    timings["hash_verification"] = time.perf_counter() - start
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(paths["target"], local_files_only=True)
    rows, generation = pool(args, tokenizer)
    write_json(run_dir.path / "manifest.json", {
        "format": FORMAT, "family": _family(args.lock), "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(), "protocol_path": str(args.protocol), "protocol_sha256": sha256_file(args.protocol), "source_file_sha256": source_identity(),
        "model_lock_sha256": sha256_file(args.lock),
        "inputs": {"calibration_run": args.calibration_run.name, "pilot_run": args.pilot_run.name, "fresh": generation},
        "prompts": len(rows), "by_source": {s: sum(r["source"] == s for r in rows) for s in ("reused", "fresh")},
        "rows": [{k: r[k] for k in ("id", "source", "values", "final", "previous")} for r in rows],
        "verified_artifact_bytes": verified, "software": compatibility(args.device),
    })
    start = time.perf_counter()
    model = load_model(paths["target"], "target", args.device)
    timings["loading"] = time.perf_counter() - start
    start = time.perf_counter()
    results = []
    with (run_dir.path / "results.jsonl").open("x") as stream:
        for index, row in enumerate(rows):
            result = evaluate_row(model, row, args.device)
            results.append(result)
            stream.write(json.dumps({"id": row["id"], **result}) + "\n")
            stream.flush()
            if index % 64 == 0:
                print(f"  prompts {index}/{len(rows)}", flush=True)
    timings["knockouts"] = time.perf_counter() - start
    del model
    release_models()
    summary = summarize(rows, results)
    write_json(run_dir.path / "summary.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--calibration-run", type=Path, required=True)
    parser.add_argument("--pilot-run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=PROJECT / "model-cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL,
                        help="the frozen brief this run executes under (default: this module's own brief)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p12-selection-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
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
            "timings_seconds": timings, "wall_seconds": time.perf_counter() - started,
            "setup_seconds": float(os.environ["OWL_SETUP_SECONDS"]) if "OWL_SETUP_SECONDS" in os.environ else None,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        })
        print(f"{status}: {run_dir.path}", flush=True)


if __name__ == "__main__":
    main()
