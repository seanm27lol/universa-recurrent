"""Phase Sixteen: with the question first, how does the asked variable reach the late read? Exploratory.

protocols/phase_sixteen_question_route_brief.md is the frozen design. Example:

    What is x? Reply with only the integer. / x = 7 / y = 16 / y = y + 1 / y = y - 3      -> 7  (y ends at 14)

With the question first, the late read happens at the chat-template positions
after the program (Phases Fourteen and Fifteen), and it must know which
variable was asked. That can arrive two ways:
- **directly**: the post-program positions attend to the question tokens;
- **through the program**: the program tokens, which come after the
  question, attend to it and mark the asked variable's statements.

Each route is cut by blocking attention from layer 0, to the whole question
or to its variable-name token alone:

    reader_blind[_name]    post-program positions -> question (name)
    program_blind[_name]   program tokens -> question (name)
    both_blind[_name]      both

Question-first prompts only. Teacher-forced scoring of the answer and of the
other variable's final value.
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

from .answer_knockout import KIND, knockout, logits_at, program_span, score, teacher_forced
from .artifacts import RunDirectory, sha256_file, write_json
from .description_census import program, program_text
from .edit_diagnostics import PROJECT, _git_head
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .prompt_factorial import ORIGINAL, render_prompt
from .runner import release_models, source_identity
from .target import TARGET_BUCKET
from .tasks import interpret
from .variable_choice_knockout import depends_on_other

PROTOCOL = PROJECT / "protocols" / "phase_sixteen_question_route_brief.md"
FORMAT = "open_weight_lingua.question_route_knockout.v1"
CONDITIONS = ("baseline", "reader_blind", "program_blind", "both_blind",
              "reader_blind_name", "program_blind_name", "both_blind_name")
VALUES = ("answer", "other")


def _ids(tokenizer, prompt: str) -> list[int]:
    return list(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True))


def name_token(tokenizer, ids: list[int], question: tuple[int, int], variable: str) -> int:
    """The one question token whose text is the asked variable's name (with or without a leading space)."""
    hits = [t for t in range(question[0], question[1] + 1) if tokenizer.decode([ids[t]], skip_special_tokens=False).strip() == variable]
    if len(hits) != 1:
        raise ValueError("the variable name is not exactly one question token")
    return hits[0]


def route_row(item: dict, split: str, tokenizer) -> dict:
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
    prompt = render_prompt(body, variable, "original_before")
    ids = _ids(tokenizer, prompt)
    span = program_span(tokenizer, ids, body)
    question = program_span(tokenizer, ids, ORIGINAL.format(variable=variable))
    if question[1] >= span[0]:
        raise ValueError(f"{item['id']}: question and program tokens overlap")
    name = name_token(tokenizer, ids, question, variable)
    values = {"answer": int(item["answer"])}
    if final_state[other] != values["answer"]:
        values["other"] = final_state[other]
    row = {"id": item["id"], "group_id": item["group_id"], "split": split, "variable": variable, "prompt": prompt,
           "input_ids": ids, "answer": item["answer"], "span": list(span), "question": list(question), "name": name,
           "values": values, "last_line_assigns_asked": statements[-1].variable == variable,
           "asked_last_update_kind": KIND[next(s for s in reversed(statements) if s.variable == variable).operation],
           "depends_on_other": depends_on_other(statements, variable)}
    row["sequences"] = {k: teacher_forced({**row, "answer": str(v)}, tokenizer) for k, v in values.items()}
    if max(len(q["ids"]) for q in row["sequences"].values()) > TARGET_BUCKET:
        raise ValueError(f"{item['id']}: prompt plus answer does not fit the pinned bucket")
    return row


def build_rows(calibration: dict, pilot: dict, tokenizer) -> list[dict]:
    return [route_row(item, split, tokenizer)
            for split, manifest in (("calibration", calibration), ("pilot", pilot)) for item in manifest["inputs"]]


def blocks(row: dict, length: int, condition: str) -> list[tuple[list[int], list[int]]]:
    """(queries, keys) pairs for one condition on a sequence of `length` tokens."""
    if condition == "baseline":
        return []
    question = [row["name"]] if condition.endswith("_name") else list(range(row["question"][0], row["question"][1] + 1))
    reader = list(range(row["span"][1] + 1, length))
    program_tokens = list(range(row["span"][0], row["span"][1] + 1))
    kind = condition.removesuffix("_name")
    return {"reader_blind": [(reader, question)], "program_blind": [(program_tokens, question)],
            "both_blind": [(reader, question), (program_tokens, question)]}[kind]


def evaluate_row(model, row: dict, device: str) -> dict:
    answer = row["sequences"]["answer"]
    baseline = logits_at(model, answer["ids"], answer["positions"], device)
    with knockout(model, 0, answer["positions"], []):
        if not torch.equal(logits_at(model, answer["ids"], answer["positions"], device), baseline):
            raise RuntimeError(f"{row['id']}: the no-op hook changed the logits")
    out = {}
    for condition in CONDITIONS:
        scored = {}
        for name, q in row["sequences"].items():
            with ExitStack() as stack:
                for queries, keys in blocks(row, len(q["ids"]), condition):
                    stack.enter_context(knockout(model, 0, queries, keys))
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
            has_other = [r[condition] for row, r in chosen if "other" in row["values"]]
            out[condition] = {"accuracy": float(np.mean([r[condition]["answer"]["correct"] for _, r in chosen])),
                              "says_other": float(np.mean([s["other"]["correct"] for s in has_other])) if has_other else None}
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
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p16-route-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
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
