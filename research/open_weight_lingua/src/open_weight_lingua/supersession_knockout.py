"""Phase Eleven: does the final statement mark itself as current? Supersession knockout, exploratory.

protocols/phase_eleven_supersession_brief.md is the frozen design. Example:

    x = 4 / y = 3 / x = 7 / y = 5 / What is x?      -> 7 (superseded value 4)

Phase Ten found that blocking the final statement `x = 7` from the earlier
assignment `x = 4` damages the answer although a literal needs nothing
earlier. Here that block is compared with a control block (the final
statement blocked from the other variable's earlier lines), and the
superseded value 4 is scored alongside the answer 7: does the block make the
model fall back to the old value?
"""

import argparse
from contextlib import nullcontext
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
from .dependency_knockout import dependency
from .description_census import program, program_text
from .edit_diagnostics import PROJECT, _git_head
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .runner import release_models, source_identity
from .span_knockout import token_lines
from .tasks import interpret

PROTOCOL = PROJECT / "protocols" / "phase_eleven_supersession_brief.md"
FORMAT = "open_weight_lingua.supersession_knockout.v1"
KINDS = ("literal", "arithmetic")
CONDITIONS = ("same_variable", "other_variable")


def value_before(statements, final: int, variable: str) -> int | None:
    """The asked variable's value just before line `final` (None if it was not yet assigned)."""
    state: dict[str, int] = {}
    for s in statements[:final]:
        if s.operation == "assign":
            state[s.variable] = s.operand
        elif s.operation == "copy":
            state[s.variable] = state[s.operand]
        else:
            state[s.variable] = state[s.variable] + (s.operand if s.operation == "add" else -s.operand)
    return state.get(variable)


def supersession_row(item: dict, split: str, tokenizer) -> dict | None:
    """Final-statement queries, same- and other-variable keys, and the superseded value; None if out of scope."""
    statements = program(item["prompt"])
    variable = item["variable"]
    if str(interpret(statements)[variable]) != item["answer"]:
        raise ValueError(f"{item['id']}: answer disagrees with the interpreter")
    body = program_text(item["prompt"])
    span = program_span(tokenizer, item["input_ids"], body)
    lines = token_lines(tokenizer, item["input_ids"], span, body)
    dep = dependency(statements, variable, span, lines)
    if dep["kind"] not in KINDS or not dep["keys"]:
        return None
    final = dep["final_line"]
    other_lines = {i for i in range(final) if statements[i].variable != variable}
    tokens = list(range(span[0], span[1] + 1))
    before = value_before(statements, final, variable)
    superseded = None if before is None else str(before)
    row = {"id": item["id"], "group_id": item["group_id"], "split": split, "variable": variable, "kind": dep["kind"],
           "input_ids": item["input_ids"], "answer": item["answer"], "superseded": superseded,
           "queries": dep["queries"],
           "keys": {"same_variable": dep["keys"],
                    "other_variable": [t for t, line in zip(tokens, lines) if line in other_lines]}}
    row["sequence"] = teacher_forced(row, tokenizer)
    if superseded is not None and superseded != item["answer"]:
        row["superseded_sequence"] = teacher_forced({**row, "answer": superseded}, tokenizer)
    return row


def evaluate_row(model, row: dict, device: str) -> dict:
    def both(context):
        with context:
            main = score(logits_at(model, row["sequence"]["ids"], row["sequence"]["positions"], device), row["sequence"]["targets"])
            old = None
            if "superseded_sequence" in row:
                q = row["superseded_sequence"]
                old = score(logits_at(model, q["ids"], q["positions"], device), q["targets"])
        return {**main, "superseded_correct": old["correct"] if old else None,
                "preference": old["logprob"] - main["logprob"] if old else None}

    ids, positions = row["sequence"]["ids"], row["sequence"]["positions"]
    baseline_logits = logits_at(model, ids, positions, device)
    with knockout(model, 0, positions, []):
        if not torch.equal(logits_at(model, ids, positions, device), baseline_logits):
            raise RuntimeError(f"{row['id']}: the no-op hook changed the logits")
    out = {"baseline": both(nullcontext())}
    for name in CONDITIONS:
        keys = row["keys"][name]
        out[name] = both(knockout(model, 0, row["queries"], keys)) if keys else None
    return out


def summarize(rows: list[dict], results: list[dict]) -> dict:
    out = {}
    for kind in KINDS:
        for split in ("calibration", "pilot", "all"):
            chosen = [(row, r) for row, r in zip(rows, results) if row["kind"] == kind and (split == "all" or row["split"] == split)]
            if not chosen:
                continue
            entry = {"n": len(chosen)}
            for condition in ("baseline", *CONDITIONS):
                scored = [r[condition] for _, r in chosen if r[condition] is not None]
                with_old = [s for s in scored if s["preference"] is not None]
                entry[condition] = {
                    "n": len(scored),
                    "accuracy": float(np.mean([s["correct"] for s in scored])) if scored else None,
                    "superseded_rate": float(np.mean([s["superseded_correct"] for s in with_old])) if with_old else None,
                    "mean_preference": float(np.mean([s["preference"] for s in with_old])) if with_old else None,
                    "n_with_superseded": len(with_old),
                }
            entry["mean_keys"] = {c: float(np.mean([len(row["keys"][c]) for row, _ in chosen])) for c in CONDITIONS}
            out[f"{kind}/{split}"] = entry
    return out


def _family(lock: Path) -> str:
    return lock.stem.replace("model-lock-", "") if lock.stem != "model-lock" else "qwen2.5-7b"


def run(args, run_dir: RunDirectory, timings: dict) -> dict:
    calibration = json.loads((args.calibration_run / "manifest.json").read_text())
    pilot = json.loads((args.pilot_run / "manifest.json").read_text())
    for manifest in (calibration, pilot):
        if manifest["model_lock_sha256"] != sha256_file(args.lock):
            raise ValueError("lock differs from the input runs' lock")
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models({**lock, "models": {"target": lock["models"]["target"]}}, {"target": paths["target"]})
    timings["hash_verification"] = time.perf_counter() - start
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(paths["target"], local_files_only=True)
    rows = [r for split, manifest in (("calibration", calibration), ("pilot", pilot))
            for r in (supersession_row(item, split, tokenizer) for item in manifest["inputs"]) if r is not None]
    write_json(run_dir.path / "manifest.json", {
        "format": FORMAT, "family": _family(args.lock), "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(), "protocol_sha256": sha256_file(PROTOCOL), "source_file_sha256": source_identity(),
        "model_lock_sha256": sha256_file(args.lock),
        "inputs": {"calibration_run": args.calibration_run.name, "calibration_manifest_sha256": sha256_file(args.calibration_run / "manifest.json"),
                   "pilot_run": args.pilot_run.name, "pilot_manifest_sha256": sha256_file(args.pilot_run / "manifest.json")},
        "prompts": len(rows),
        "rows": [{k: r[k] for k in ("id", "split", "kind", "answer", "superseded", "queries", "keys")} for r in rows],
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
            if index % 128 == 0:
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p11-supersession-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
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
