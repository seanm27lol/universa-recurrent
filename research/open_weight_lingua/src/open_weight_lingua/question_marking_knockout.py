"""Phase Seventeen: which program tokens carry the asked variable when the question comes first? Exploratory.

protocols/phase_seventeen_question_marking_brief.md is the frozen design. Example:

    What is x? Reply with only the integer. / x = 7 / y = 16 / y = y + 1 / y = y - 3      -> 7  (y ends at 14)

Phase Sixteen found that when every program token is blocked from the
question, the late read falls back to the most recent assignment of either
variable. Here only some program tokens are blocked from the question, from
layer 0:

    program_blind      every program token (Phase Sixteen's condition)
    asked_blind        the asked variable's lines
    other_blind        the other variable's lines
    last_line_blind    the program's last line

If the program *tags the asked variable's statements*, `asked_blind` should
reproduce the fallback. If it *suppresses the other variable's*,
`other_blind` should. If it *carries the variable to the end*,
`last_line_blind` should. Question-first prompts only; teacher-forced scoring
of the answer and of the other variable's final value.
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

from .answer_knockout import knockout, logits_at, score
from .artifacts import RunDirectory, sha256_file, write_json
from .description_census import program, program_text
from .edit_diagnostics import PROJECT, _git_head
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .question_route_knockout import route_row
from .runner import release_models, source_identity
from .span_knockout import token_lines

PROTOCOL = PROJECT / "protocols" / "phase_seventeen_question_marking_brief.md"
FORMAT = "open_weight_lingua.question_marking_knockout.v1"
CONDITIONS = ("baseline", "program_blind", "asked_blind", "other_blind", "last_line_blind")


def marking_row(item: dict, split: str, tokenizer) -> dict:
    row = route_row(item, split, tokenizer)
    statements = program(item["prompt"])
    variable = item["variable"]
    span = tuple(row["span"])
    lines = token_lines(tokenizer, row["input_ids"], span, program_text(item["prompt"]))
    tokens = list(range(span[0], span[1] + 1))
    last = len(statements) - 1
    row["queries"] = {
        "program_blind": tokens,
        "asked_blind": [t for t, line in zip(tokens, lines) if statements[line].variable == variable],
        "other_blind": [t for t, line in zip(tokens, lines) if statements[line].variable != variable],
        "last_line_blind": [t for t, line in zip(tokens, lines) if line == last],
    }
    return row


def build_rows(calibration: dict, pilot: dict, tokenizer) -> list[dict]:
    return [marking_row(item, split, tokenizer)
            for split, manifest in (("calibration", calibration), ("pilot", pilot)) for item in manifest["inputs"]]


def evaluate_row(model, row: dict, device: str) -> dict:
    answer = row["sequences"]["answer"]
    baseline = logits_at(model, answer["ids"], answer["positions"], device)
    with knockout(model, 0, answer["positions"], []):
        if not torch.equal(logits_at(model, answer["ids"], answer["positions"], device), baseline):
            raise RuntimeError(f"{row['id']}: the no-op hook changed the logits")
    question = list(range(row["question"][0], row["question"][1] + 1))
    out = {}
    for condition in CONDITIONS:
        scored = {}
        for name, q in row["sequences"].items():
            with ExitStack() as stack:
                if condition != "baseline":
                    stack.enter_context(knockout(model, 0, row["queries"][condition], question))
                scored[name] = score(logits_at(model, q["ids"], q["positions"], device), q["targets"])
        out[condition] = scored
    return out


def summarize(rows: list[dict], results: list[dict]) -> dict:
    def table(keep):
        chosen = [(row, r) for row, r in zip(rows, results) if keep(row)]
        if not chosen:
            return {}
        out = {"n": len(chosen)}
        for condition in CONDITIONS:
            said = [r[condition] for row, r in chosen if "other" in row["values"]]
            out[condition] = {"accuracy": float(np.mean([r[condition]["answer"]["correct"] for _, r in chosen])),
                              "says_other": float(np.mean([s["other"]["correct"] for s in said])) if said else None}
        return out

    return {"all": table(lambda r: True),
            "by_split": {s: table(lambda r, s=s: r["split"] == s) for s in ("calibration", "pilot")},
            "last_line_other_no_dependency": table(lambda r: not r["last_line_assigns_asked"] and not r["depends_on_other"]),
            "last_line_asked": table(lambda r: r["last_line_assigns_asked"])}


# ---------------------------------------------------------------- execution


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
    rows = build_rows(calibration, pilot, tokenizer)
    stimuli = run_dir.path / "stimuli.json"
    write_json(stimuli, rows)
    write_json(run_dir.path / "manifest.json", {
        "format": FORMAT, "family": _family(args.lock), "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(), "protocol_path": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL),
        "source_file_sha256": source_identity(), "model_lock_sha256": sha256_file(args.lock),
        "inputs": {"calibration_run": args.calibration_run.name, "calibration_manifest_sha256": sha256_file(args.calibration_run / "manifest.json"),
                   "pilot_run": args.pilot_run.name, "pilot_manifest_sha256": sha256_file(args.pilot_run / "manifest.json")},
        "prompts": len(rows), "stimuli_sha256": sha256_file(stimuli),
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
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p17-marking-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
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
