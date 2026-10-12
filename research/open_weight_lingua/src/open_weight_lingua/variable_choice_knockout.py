"""Phase Fifteen: are question-first errors the late read choosing the wrong statements? Exploratory.

protocols/phase_fifteen_variable_choice_brief.md is the frozen design. Example:

    What is x? ... / x = 7 / y = 16 / y = y + 1 / y = y - 3      -> 7  (the other variable ends at 14)

With the question first, both models lose accuracy mainly on programs that
end on the other variable, and their wrong answers are often that variable's
value (Phases Six and Seven). Phase Fourteen found that a literal last line
of the asked variable, seen alone, restores accuracy. If the errors come from
the late read taking the wrong statements, hiding those statements from the
post-program positions should restore accuracy:

    block_other        every post-program position blocked from the other variable's lines, from layer 0
    block_superseded   ... from the asked variable's lines before its final assignment
                       (literal finals with an earlier assignment only)
    block_both         both (same prompts)

Each prompt runs in both question orders. Teacher-forced scoring of the
answer, the other variable's final value and the asked variable's previous
value (each when distinct).
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
from .artifacts import RunDirectory, sha256_file, write_json
from .description_census import program, program_text
from .edit_diagnostics import PROJECT, _git_head
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .prompt_factorial import render_prompt
from .question_order_knockout import ORDERS
from .runner import release_models, source_identity
from .span_knockout import token_lines
from .supersession_knockout import value_before
from .target import TARGET_BUCKET
from .tasks import interpret

PROTOCOL = PROJECT / "protocols" / "phase_fifteen_variable_choice_brief.md"
FORMAT = "open_weight_lingua.variable_choice_knockout.v1"
CONDITIONS = ("baseline", "block_other", "block_superseded", "block_both")
LITERAL_ONLY = ("block_superseded", "block_both")
VALUES = ("answer", "other", "previous")


def _ids(tokenizer, prompt: str) -> list[int]:
    return list(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True))


def depends_on_other(statements, variable: str) -> bool:
    """Whether the asked variable's final value passes through a copy of the other variable."""
    for s in reversed(statements):
        if s.variable != variable:
            continue
        if s.operation == "assign":
            return False
        if s.operation == "copy" and s.operand != variable:
            return True
    return False


def choice_row(item: dict, split: str, order: str, tokenizer) -> dict:
    statements = program(item["prompt"])
    variable = item["variable"]
    other = "y" if variable == "x" else "x"
    final_state = interpret(statements)
    if str(final_state[variable]) != item["answer"]:
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
    tokens = list(range(span[0], span[1] + 1))
    asked = [i for i, s in enumerate(statements) if s.variable == variable]
    final = asked[-1]
    literal_final = statements[final].operation == "assign"
    keys = {"block_other": [t for t, line in zip(tokens, lines) if statements[line].variable == other]}
    superseded = [t for t, line in zip(tokens, lines) if line in asked and line != final]
    if literal_final and superseded:  # an earlier assignment exists and the final one does not depend on it
        keys["block_superseded"] = superseded
        keys["block_both"] = sorted(keys["block_other"] + superseded)
    answer = int(item["answer"])
    values = {"answer": answer}
    for name, value in (("other", final_state[other]), ("previous", value_before(statements, final, variable))):
        if value is not None and value not in values.values():
            values[name] = value
    row = {"id": item["id"], "group_id": item["group_id"], "split": split, "order": order, "variable": variable,
           "prompt": prompt, "input_ids": ids, "answer": item["answer"], "span": list(span), "values": values, "keys": keys,
           "last_line_assigns_asked": statements[-1].variable == variable,
           "asked_last_update_kind": KIND[statements[final].operation],
           "depends_on_other": depends_on_other(statements, variable)}
    row["sequences"] = {name: teacher_forced({**row, "answer": str(value)}, tokenizer) for name, value in values.items()}
    if max(len(q["ids"]) for q in row["sequences"].values()) > TARGET_BUCKET:
        raise ValueError(f"{item['id']}: prompt plus answer does not fit the pinned bucket")
    return row


def build_rows(calibration: dict, pilot: dict, tokenizer) -> list[dict]:
    return [choice_row(item, split, order, tokenizer)
            for split, manifest in (("calibration", calibration), ("pilot", pilot))
            for item in manifest["inputs"] for order in ORDERS]


def evaluate_row(model, row: dict, device: str) -> dict:
    answer = row["sequences"]["answer"]
    baseline = logits_at(model, answer["ids"], answer["positions"], device)
    with knockout(model, 0, answer["positions"], []):
        if not torch.equal(logits_at(model, answer["ids"], answer["positions"], device), baseline):
            raise RuntimeError(f"{row['id']}: the no-op hook changed the logits")
    out = {}
    for condition in CONDITIONS:
        if condition != "baseline" and condition not in row["keys"]:
            out[condition] = None
            continue
        scored = {}
        for name, q in row["sequences"].items():
            after = list(range(row["span"][1] + 1, len(q["ids"])))
            keys = row["keys"][condition] if condition != "baseline" else []
            with knockout(model, 0, after, keys):
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
            scored = [(row, r[condition]) for row, r in chosen if r[condition] is not None]
            if not scored:
                continue
            out[condition] = {"n": len(scored), "accuracy": float(np.mean([s["answer"]["correct"] for _, s in scored]))}
            for name in ("other", "previous"):
                has = [s for row, s in scored if name in row["values"]]
                out[condition][f"says_{name}"] = float(np.mean([s[name]["correct"] for s in has])) if has else None
        return out

    out = {}
    for order in ORDERS:
        def mine(keep, o=order):
            return table(lambda r: r["order"] == o and keep(r))

        out[order] = {
            "all": mine(lambda r: True),
            "by_split": {s: mine(lambda r, s=s: r["split"] == s) for s in ("calibration", "pilot")},
            "last_line_other_no_dependency": mine(lambda r: not r["last_line_assigns_asked"] and not r["depends_on_other"]),
            "last_line_asked": mine(lambda r: r["last_line_assigns_asked"]),
            "depends_on_other": mine(lambda r: r["depends_on_other"]),
            "superseded_literal": mine(lambda r: "block_superseded" in r["keys"]),
            "superseded_literal_last_line_asked": mine(lambda r: "block_superseded" in r["keys"] and r["last_line_assigns_asked"]),
            "superseded_literal_last_line_other": mine(lambda r: "block_superseded" in r["keys"] and not r["last_line_assigns_asked"]),
        }
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
    timings["loading"] = time.perf_counter() - start
    start = time.perf_counter()
    results = []
    with (run_dir.path / "results.jsonl").open("x") as stream:
        for index, row in enumerate(rows):
            result = evaluate_row(model, row, args.device)
            results.append(result)
            stream.write(json.dumps({"id": row["id"], "order": row["order"], **result}) + "\n")
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
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p15-choice-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
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
