"""Phase Ten: when does a statement collect what it depends on? Dependency knockout, exploratory.

protocols/phase_ten_dependency_knockout_brief.md is the frozen design. Example:

    y = 16 / x = 7 / x = y / y = y - 3 / What is x?      -> 16

Phase Nine showed that the late read needs only the asked variable's own
statements, so by then the copied 16 must sit at `x = y`. Here, from layer k
onward, the asked variable's final statement is blocked from the lines it
depends on: the copied variable's earlier lines for a cross-variable copy
(x = y), the asked variable's earlier lines for arithmetic and self-copies
(x = x), and the same for a literal as a placebo. If the answer survives, the statement had already read its
dependency before layer k.
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

from .answer_knockout import KIND, knockout, layer_grid, logits_at, program_span, score, teacher_forced
from .architectures import decoder_layers
from .artifacts import RunDirectory, sha256_file, write_json
from .description_census import program, program_text
from .edit_diagnostics import PROJECT, _git_head
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .runner import release_models, source_identity
from .span_knockout import token_lines
from .tasks import interpret

PROTOCOL = PROJECT / "protocols" / "phase_ten_dependency_knockout_brief.md"
FORMAT = "open_weight_lingua.dependency_knockout.v1"
KINDS = ("arithmetic", "self_copy", "cross_copy", "literal")
RELEASE_TOLERANCE = 0.05


def dependency(statements, variable: str, span: tuple[int, int], lines: list[int]) -> dict:
    """The asked variable's final statement (queries) and the lines it depends on (keys)."""
    final = max(i for i, s in enumerate(statements) if s.variable == variable)
    kind = KIND[statements[final].operation]
    if kind == "copy":  # x = x copies itself; only x = y reads the other variable
        kind = "self_copy" if statements[final].operand == variable else "cross_copy"
    source = statements[final].operand if kind == "cross_copy" else variable
    depends = {i for i in range(final) if statements[i].variable == source}
    tokens = list(range(span[0], span[1] + 1))
    return {"kind": kind, "final_line": final, "dependency_lines": sorted(depends),
            "queries": [t for t, line in zip(tokens, lines) if line == final],
            "keys": [t for t, line in zip(tokens, lines) if line in depends]}


def prompt_rows(manifest: dict, split: str, tokenizer) -> list[dict]:
    rows = []
    for item in manifest["inputs"]:
        statements = program(item["prompt"])
        variable = item["variable"]
        if str(interpret(statements)[variable]) != item["answer"]:
            raise ValueError(f"{item['id']}: answer disagrees with the interpreter")
        body = program_text(item["prompt"])
        span = program_span(tokenizer, item["input_ids"], body)
        lines = token_lines(tokenizer, item["input_ids"], span, body)
        row = {"id": item["id"], "group_id": item["group_id"], "split": split, "variable": variable,
               "input_ids": item["input_ids"], "answer": item["answer"], "span": list(span),
               **dependency(statements, variable, span, lines)}
        row["sequence"] = teacher_forced(row, tokenizer)
        rows.append(row)
    return rows


def evaluate_row(model, row: dict, layers: int, device: str) -> dict | None:
    """Baseline, the no-op control, and the dependency knockout from every grid layer; None if no keys."""
    if not row["keys"]:
        return None
    ids, positions, targets = row["sequence"]["ids"], row["sequence"]["positions"], row["sequence"]["targets"]
    baseline = logits_at(model, ids, positions, device)
    with knockout(model, 0, positions, []):
        if not torch.equal(logits_at(model, ids, positions, device), baseline):
            raise RuntimeError(f"{row['id']}: the no-op hook changed the logits")
    out = {"baseline": score(baseline, targets)}
    for k in layer_grid(layers):
        with knockout(model, k, row["queries"], row["keys"]):
            out[f"dependency@{k}"] = score(logits_at(model, ids, positions, device), targets)
    return out


def summarize(rows: list[dict], results: list, layers: int) -> dict:
    grid = layer_grid(layers)
    out = {"grid": grid, "by_kind": {}, "release_layer": {}}
    for kind in KINDS:
        kind_entry, release = {}, {}
        for split in ("calibration", "pilot"):
            chosen = [r for row, r in zip(rows, results) if r is not None and row["kind"] == kind and row["split"] == split]
            if not chosen:
                continue
            base = float(np.mean([r["baseline"]["correct"] for r in chosen]))
            accuracy = {k: float(np.mean([r[f"dependency@{k}"]["correct"] for r in chosen])) for k in grid}
            kind_entry[split] = {"n": len(chosen), "baseline": base, "accuracy": accuracy,
                                 "loss": {k: base - a for k, a in accuracy.items()}}
            ok = [accuracy[k] >= base - RELEASE_TOLERANCE for k in grid]
            qualifying = [k for i, k in enumerate(grid) if all(ok[i:])]
            release[split] = qualifying[0] if qualifying else None
        out["by_kind"][kind] = kind_entry
        out["release_layer"][kind] = release
    out["skipped_no_dependency"] = sum(r is None for r in results)
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
    rows = prompt_rows(calibration, "calibration", tokenizer) + prompt_rows(pilot, "pilot", tokenizer)
    write_json(run_dir.path / "manifest.json", {
        "format": FORMAT, "family": _family(args.lock), "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(), "protocol_sha256": sha256_file(PROTOCOL), "source_file_sha256": source_identity(),
        "model_lock_sha256": sha256_file(args.lock),
        "inputs": {"calibration_run": args.calibration_run.name, "calibration_manifest_sha256": sha256_file(args.calibration_run / "manifest.json"),
                   "pilot_run": args.pilot_run.name, "pilot_manifest_sha256": sha256_file(args.pilot_run / "manifest.json")},
        "prompts": len(rows),
        "rows": [{k: r[k] for k in ("id", "split", "kind", "final_line", "dependency_lines", "queries", "keys")} for r in rows],
        "verified_artifact_bytes": verified, "software": compatibility(args.device),
    })
    start = time.perf_counter()
    model = load_model(paths["target"], "target", args.device)
    layers = len(decoder_layers(model))
    timings["loading"] = time.perf_counter() - start
    start = time.perf_counter()
    results = []
    with (run_dir.path / "results.jsonl").open("x") as stream:
        for index, row in enumerate(rows):
            result = evaluate_row(model, row, layers, args.device)
            results.append(result)
            stream.write(json.dumps({"id": row["id"], "result": result}) + "\n")
            stream.flush()
            if index % 128 == 0:
                print(f"  prompts {index}/{len(rows)}", flush=True)
    timings["knockouts"] = time.perf_counter() - start
    del model
    release_models()
    summary = summarize(rows, results, layers)
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
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p10-dependency-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
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
