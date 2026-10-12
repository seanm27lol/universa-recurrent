"""Phase Nine: which program tokens does the late read use? Partial-span knockout, exploratory.

protocols/phase_nine_span_knockout_brief.md is the frozen design. Example: for

    x = 7 / y = 16 / y = y + 1 / y = y - 3 / What is x?

Phase Eight showed that the answer needs a late look back at the program
(Qwen: answer positions, layers 21+; Gemma: every post-program position,
layers 24+). Here that same route is blocked from only part of the program:
the last token alone, everything but it, the asked variable's lines, the
other variable's lines, or the asked variable's final line. Reused
calibration and pilot prompts; teacher-forced digit accuracy.
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
from .tasks import interpret

PROTOCOL = PROJECT / "protocols" / "phase_nine_span_knockout_brief.md"
FORMAT = "open_weight_lingua.span_knockout.v1"
ROUTES = {"gemma3-12b": ("after_program", 24), "qwen2.5-7b": ("answer", 21)}  # Phase Eight's confirmed damaging conditions
SPANS = ("full", "last_token", "all_minus_last", "asked_lines_minus_last", "other_lines_minus_last", "asked_final_line_minus_last")


# ---------------------------------------------------------------- lines and spans


def token_lines(tokenizer, ids: list[int], span: tuple[int, int], body: str) -> list[int]:
    """Line index of each program token: the line holding its first character; a newline ends its line."""
    text = tokenizer.decode(ids, skip_special_tokens=False)
    at = text.find(body)
    out = []
    for p in range(span[0], span[1] + 1):
        start = len(tokenizer.decode(ids[:p], skip_special_tokens=False)) - at
        end = len(tokenizer.decode(ids[: p + 1], skip_special_tokens=False)) - at
        piece = body[max(start, 0) : end]
        if "\n" in piece.strip("\n"):
            raise ValueError("a token spans two program lines")
        out.append(body[: max(start, 0)].count("\n"))
    return out


def spans_for(statements, variable: str, span: tuple[int, int], lines: list[int]) -> dict[str, list[int]]:
    first, last = span
    tokens = list(range(first, last + 1))
    asked = {i for i, s in enumerate(statements) if s.variable == variable}
    final = max(asked)
    without_last = [t for t in tokens if t != last]
    return {
        "full": tokens,
        "last_token": [last],
        "all_minus_last": without_last,
        "asked_lines_minus_last": [t for t, line in zip(tokens, lines) if line in asked and t != last],
        "other_lines_minus_last": [t for t, line in zip(tokens, lines) if line not in asked and t != last],
        "asked_final_line_minus_last": [t for t, line in zip(tokens, lines) if line == final and t != last],
    }


# ---------------------------------------------------------------- evaluation


def evaluate_row(model, row: dict, route: tuple[str, int], device: str) -> dict:
    ids, positions, targets = row["sequence"]["ids"], row["sequence"]["positions"], row["sequence"]["targets"]
    baseline = logits_at(model, ids, positions, device)
    with knockout(model, 0, positions, []):
        if not torch.equal(logits_at(model, ids, positions, device), baseline):
            raise RuntimeError(f"{row['id']}: the no-op hook changed the logits")
    variant, layer = route
    queries = positions if variant == "answer" else list(range(row["span"][1] + 1, len(ids)))
    out = {"baseline": score(baseline, targets)}
    for name in SPANS:
        keys = row["spans"][name]
        if not keys:
            out[name] = None  # e.g. the asked final line is only the last token
            continue
        with knockout(model, layer, queries, keys):
            out[name] = score(logits_at(model, ids, positions, device), targets)
    return out


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
        last_update = next(s for s in reversed(statements) if s.variable == variable)
        row = {"id": item["id"], "group_id": item["group_id"], "split": split, "variable": variable,
               "prompt": item["prompt"], "input_ids": item["input_ids"], "answer": item["answer"], "span": list(span),
               "asked_last_update_kind": KIND[last_update.operation],
               "last_line_assigns_asked": statements[-1].variable == variable,
               "asked_copies_other": any(s.variable == variable and s.operation == "copy" for s in statements),
               "spans": spans_for(statements, variable, span, lines)}
        row["sequence"] = teacher_forced(row, tokenizer)
        rows.append(row)
    return rows


def summarize(rows: list[dict], results: list[dict]) -> dict:
    def table(keep):
        chosen = [r for row, r in zip(rows, results) if keep(row)]
        out = {"n": len(chosen), "baseline": float(np.mean([r["baseline"]["correct"] for r in chosen])) if chosen else None}
        for name in SPANS:
            scored = [r for r in chosen if r[name] is not None]
            out[name] = {"n": len(scored),
                         "accuracy": float(np.mean([r[name]["correct"] for r in scored])) if scored else None,
                         "loss": float(np.mean([r["baseline"]["correct"] - r[name]["correct"] for r in scored])) if scored else None}
        return out

    strata = {}
    for name, levels in (("asked_last_update_kind", ("literal", "arithmetic", "copy")),
                         ("last_line_assigns_asked", (True, False)), ("asked_copies_other", (True, False))):
        strata[name] = {str(level): table(lambda r, n=name, v=level: r[n] == v) for level in levels}
    return {"by_split": {split: table(lambda r, s=split: r["split"] == s) for split in ("calibration", "pilot")},
            "strata_all_splits": strata,
            "span_sizes": {name: float(np.mean([len(r["spans"][name]) for r in rows])) for name in SPANS}}


# ---------------------------------------------------------------- execution


def _family(lock: Path) -> str:
    return lock.stem.replace("model-lock-", "") if lock.stem != "model-lock" else "qwen2.5-7b"


def run(args, run_dir: RunDirectory, timings: dict) -> dict:
    family = _family(args.lock)
    route = ROUTES[family]
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
        "format": FORMAT, "family": family, "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(), "protocol_sha256": sha256_file(PROTOCOL), "source_file_sha256": source_identity(),
        "model_lock_sha256": sha256_file(args.lock), "route": route, "spans": SPANS,
        "inputs": {"calibration_run": args.calibration_run.name, "calibration_manifest_sha256": sha256_file(args.calibration_run / "manifest.json"),
                   "pilot_run": args.pilot_run.name, "pilot_manifest_sha256": sha256_file(args.pilot_run / "manifest.json")},
        "prompts": len(rows),
        "rows": [{k: r[k] for k in ("id", "split", "span", "spans", "asked_last_update_kind", "last_line_assigns_asked", "asked_copies_other")} for r in rows],
        "verified_artifact_bytes": verified, "software": compatibility(args.device),
    })
    start = time.perf_counter()
    model = load_model(paths["target"], "target", args.device)
    timings["loading"] = time.perf_counter() - start
    if route[1] >= len(decoder_layers(model)):
        raise ValueError("the frozen route layer exceeds the model's depth")
    start = time.perf_counter()
    results = []
    with (run_dir.path / "results.jsonl").open("x") as stream:
        for index, row in enumerate(rows):
            result = evaluate_row(model, row, route, args.device)
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
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p9-span-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
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
