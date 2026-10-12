"""Phase Fourteen: does knowing the question first change the late read? Exploratory.

protocols/phase_fourteen_question_order_brief.md is the frozen design. Example:

    question after:  x = 2 / y = 3 / x = x + 9 / y = 5 / What is x? ...      -> 11
    question first:  What is x? ... / x = 2 / y = 3 / x = x + 9 / y = 5      -> 11

With the question after the program, the model looks back at the asked
variable's statements late (Phases Eight to Thirteen). With the question
first, it knows which variable to follow while reading. If it then holds the
answer at the end of the program, the post-program positions should manage
with the program's last line alone, even when that line assigns the other
variable. Each prompt runs in both orders: baseline, the Phase Eight layer
sweep (every post-program position blocked from the program from layer k),
and `only_last_line` (from layer 0, every post-program position blocked from
every program token before the last line). Teacher-forced digit accuracy.
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

from .answer_knockout import KIND, RELEASE_TOLERANCE, knockout, layer_grid, logits_at, program_span, score, teacher_forced
from .architectures import decoder_layers
from .artifacts import RunDirectory, sha256_file, write_json
from .description_census import program, program_text
from .edit_diagnostics import PROJECT, _git_head
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .prompt_factorial import render_prompt
from .runner import release_models, source_identity
from .span_knockout import token_lines
from .target import TARGET_BUCKET
from .tasks import interpret

PROTOCOL = PROJECT / "protocols" / "phase_fourteen_question_order_brief.md"
FORMAT = "open_weight_lingua.question_order_knockout.v1"
ORDERS = {"after": "original_after", "first": "original_before"}


# ---------------------------------------------------------------- rows


def _ids(tokenizer, prompt: str) -> list[int]:
    return list(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True))


def order_row(item: dict, split: str, order: str, tokenizer) -> dict:
    """One prompt in one question order: its program span, the last line's tokens and the teacher-forced answer."""
    statements = program(item["prompt"])
    variable = item["variable"]
    if str(interpret(statements)[variable]) != item["answer"]:
        raise ValueError(f"{item['id']}: answer disagrees with the interpreter")
    body = program_text(item["prompt"])
    if item["prompt"] != render_prompt(body, variable, "original_after"):
        raise ValueError(f"{item['id']}: source prompt differs from the original wording")
    if _ids(tokenizer, item["prompt"]) != list(item["input_ids"]):
        raise ValueError(f"{item['id']}: recorded input ids differ from the chat template's")
    prompt = render_prompt(body, variable, ORDERS[order])
    ids = list(item["input_ids"]) if order == "after" else _ids(tokenizer, prompt)
    span = program_span(tokenizer, ids, body)
    lines = token_lines(tokenizer, ids, span, body)
    last_update = next(s for s in reversed(statements) if s.variable == variable)
    row = {"id": item["id"], "group_id": item["group_id"], "split": split, "order": order, "variable": variable,
           "prompt": prompt, "input_ids": ids, "answer": item["answer"], "span": list(span),
           "before_last_line": [t for t, line in zip(range(span[0], span[1] + 1), lines) if line < max(lines)],
           "asked_last_update_kind": KIND[last_update.operation],
           "last_line_assigns_asked": statements[-1].variable == variable}
    row["sequence"] = teacher_forced(row, tokenizer)
    if len(row["sequence"]["ids"]) > TARGET_BUCKET:
        raise ValueError(f"{item['id']}: prompt plus answer does not fit the pinned bucket")
    return row


def build_rows(calibration: dict, pilot: dict, tokenizer) -> list[dict]:
    return [order_row(item, split, order, tokenizer)
            for split, manifest in (("calibration", calibration), ("pilot", pilot))
            for item in manifest["inputs"] for order in ORDERS]


# ---------------------------------------------------------------- the knockouts


def conditions(layers: int) -> list[str]:
    return ["baseline"] + [f"after_program@{k}" for k in layer_grid(layers)] + ["only_last_line"]


def evaluate_row(model, row: dict, layers: int, device: str) -> dict:
    ids, positions, targets = row["sequence"]["ids"], row["sequence"]["positions"], row["sequence"]["targets"]
    baseline = logits_at(model, ids, positions, device)
    with knockout(model, 0, positions, []):
        if not torch.equal(logits_at(model, ids, positions, device), baseline):
            raise RuntimeError(f"{row['id']}: the no-op hook changed the logits")
    after = list(range(row["span"][1] + 1, len(ids)))
    out = {"baseline": score(baseline, targets)}
    for k in layer_grid(layers):
        with knockout(model, k, after, list(range(row["span"][0], row["span"][1] + 1))):
            out[f"after_program@{k}"] = score(logits_at(model, ids, positions, device), targets)
    with knockout(model, 0, after, row["before_last_line"]):
        out["only_last_line"] = score(logits_at(model, ids, positions, device), targets)
    return out


def summarize(rows: list[dict], results: list[dict], layers: int) -> dict:
    names = conditions(layers)

    def table(keep):
        chosen = [r for row, r in zip(rows, results) if keep(row)]
        return {"n": len(chosen), **{c: float(np.mean([r[c]["correct"] for r in chosen])) for c in names}} if chosen else {}

    out = {"layers": layers, "grid": layer_grid(layers), "orders": {}}
    for order in ORDERS:
        by_split = {split: table(lambda r, o=order, s=split: r["order"] == o and r["split"] == s) for split in ("calibration", "pilot")}
        release = {}
        for split, cells in by_split.items():
            ok = [cells[f"after_program@{k}"] >= cells["baseline"] - RELEASE_TOLERANCE for k in layer_grid(layers)]
            qualifying = [k for i, k in enumerate(layer_grid(layers)) if all(ok[i:])]
            release[split] = qualifying[0] if qualifying else None
        strata = {"last_line_assigns_other": table(lambda r, o=order: r["order"] == o and not r["last_line_assigns_asked"])}
        for kind in ("literal", "arithmetic", "copy"):
            strata[f"last_line_assigns_asked_{kind}"] = table(
                lambda r, o=order, k=kind: r["order"] == o and r["last_line_assigns_asked"] and r["asked_last_update_kind"] == k)
        out["orders"][order] = {"all": table(lambda r, o=order: r["order"] == o), "by_split": by_split,
                                "release_layer": release, "strata": strata}
    return out


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
        "prompts": len(rows), "orders": ORDERS, "stimuli_sha256": sha256_file(stimuli),
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
            stream.write(json.dumps({"id": row["id"], "order": row["order"], **result}) + "\n")
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
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p14-order-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
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
