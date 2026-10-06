"""Phase Seven confirmatory 2×2 on validation_a: question position × wording, answers only.

protocols/phase_seven_confirmatory_validation_a.md is the frozen design. Example:
asked "What is x at the end of this program?" before `x = 7 / y = 16 /
y = y + 1 / y = y - 3`, Gemma-3-12B answered 14 (y's value) in the
exploratory run. This module asks the 1,024 never-opened validation_a base
prompts in the four formats (original/expanded wording × question
before/after the program), then tests six pre-registered contrasts with a
Bonferroni-controlled whole-group bootstrap.

    run       one family on the GPU: build validation_a, record hashes, generate
    evaluate  CPU: cells, contrasts, verdicts, breakdowns over both families
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
import traceback
import uuid

import numpy as np
import torch

from .artifacts import RunDirectory, sha256_file, write_json
from .description_census import program, program_text
from .edit_diagnostics import PROJECT, _git_head
from .metrics import answer_text_matches
from .preflight import compatibility, load_model, model_paths, read_lock, verify_models
from .prompt_factorial import render_prompt, tokenize
from .runner import release_models, source_identity
from .splits import build_plan
from .target import TARGET_BUCKET, Site, Target, generation_eos_token_ids
from .tasks import interpret

PROTOCOL = PROJECT / "protocols" / "phase_seven_confirmatory_validation_a.md"
FORMAT = "open_weight_lingua.prompt_factorial_confirm.v1"
PLAN_HASH = "ff060012d35d92c480a51400a02597bea7421a525c44d97043a56dd3f3de667f"
SPLIT = "validation_a"
CONDITIONS = ("original_after", "original_before", "expanded_after", "expanded_before")
CONVENTIONS = {"gemma3-12b": "rstrip", "qwen2.5-7b": "raw"}
MAX_NEW_TOKENS = 8
ORDER_SEED = 2026100508
RESAMPLES = 10_000
BOOTSTRAP_SEEDS = {"gemma3-12b": 801100, "qwen2.5-7b": 801101}
FAMILY_TESTS = 6
CONFIRMATORY_COVERAGE = 1 - 0.05 / FAMILY_TESTS
SECONDARY_COVERAGE = 0.95
MARGIN = 0.05
HYPOTHESES = (
    ("G1", "gemma3-12b", "position_original", "negative"),
    ("G2", "gemma3-12b", "wording_before", "negative"),
    ("G3", "gemma3-12b", "interaction", "negative"),
    ("Q1", "qwen2.5-7b", "position_original", "negative"),
    ("Q2", "qwen2.5-7b", "wording_before", "equivalent"),
    ("Q3", "qwen2.5-7b", "interaction", "equivalent"),
)
KIND = {"assign": "literal", "add": "arithmetic", "subtract": "arithmetic", "copy": "copy"}


# ---------------------------------------------------------------- stimuli


def validation_rows(plan) -> list[dict]:
    """Four formats of every validation_a base prompt, with the labels the breakdowns need."""
    if plan.plan_hash != PLAN_HASH:
        raise ValueError("split plan differs from the frozen Phase Two plan")
    rows = []
    rng = random.Random(ORDER_SEED)
    for group in plan.groups[SPLIT]:
        for base in group.variants():
            body = program_text(base["prompt"])
            if base["prompt"] != render_prompt(body, base["variable"], "original_after"):
                raise ValueError("validation prompt differs from the frozen original wording")
            statements = program(base["prompt"])
            state = interpret(statements)
            variable = base["variable"]
            if str(state[variable]) != base["answer"]:
                raise ValueError("reference answer disagrees with the interpreter")
            last_update = next(s for s in reversed(statements) if s.variable == variable)
            order = list(CONDITIONS)
            rng.shuffle(order)
            for condition in order:
                rows.append({
                    "id": f"{base['id']}__{condition}", "base_id": base["id"], "group_id": base["group_id"],
                    "side": base["side"], "variable": variable, "condition": condition,
                    "program": body, "prompt": render_prompt(body, variable, condition),
                    "answer": base["answer"], "other_answer": str(state["y" if variable == "x" else "x"]),
                    "last_line_assigns_asked": statements[-1].variable == variable,
                    "last_program_operation": statements[-1].operation,
                    "asked_last_update_operation": last_update.operation,
                    "asked_last_update_kind": KIND[last_update.operation],
                })
    return rows


# ---------------------------------------------------------------- run (GPU)


def _family(lock: Path) -> str:
    return lock.stem.replace("model-lock-", "") if lock.stem != "model-lock" else "qwen2.5-7b"


def generate(target: Target, rows: list[dict], convention: str, destination: Path, eos: list[int]) -> int:
    """Greedy answers, appended and flushed one line at a time; no retries."""
    done = 0
    with destination.open("x") as stream:
        for row in rows:
            ids, mask = target.tensors([row["input_ids"]], [row["attention_mask"]])
            generated = target.greedy(ids, mask, Site(layer=0, position=row["answer_position"]), max_tokens=MAX_NEW_TOKENS)
            text = generated["text"]
            stream.write(json.dumps({
                **{k: v for k, v in row.items() if k not in ("input_ids", "attention_mask")},
                "generated_token_ids": generated["token_ids"], "text": text, "terminated": generated["terminated"],
                "resolved_eos_token_ids": eos, "correct": answer_text_matches(text, row["answer"], convention),
                "answered_other_value": text.strip() == row["other_answer"] and row["other_answer"] != row["answer"],
            }) + "\n")
            stream.flush()
            done += 1
            if done % 512 == 0:
                print(f"  generated {done}/{len(rows)}", flush=True)
    return done


def run(args, run_dir: RunDirectory, timings: dict) -> dict:
    family = _family(args.lock)
    convention = CONVENTIONS[family]
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    start = time.perf_counter()
    verified = verify_models({**lock, "models": {"target": lock["models"]["target"]}}, {"target": paths["target"]})
    timings["hash_verification"] = time.perf_counter() - start
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(paths["target"], local_files_only=True)
    rows = validation_rows(build_plan())
    tokenize(tokenizer, rows)
    stimuli = run_dir.path / "stimuli.json"
    write_json(stimuli, rows)
    write_json(run_dir.path / "manifest.json", {
        "format": FORMAT, "family": family, "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(), "protocol_sha256": sha256_file(PROTOCOL), "source_file_sha256": source_identity(),
        "model_lock_sha256": sha256_file(args.lock), "plan_hash": PLAN_HASH, "split": SPLIT,
        "stimuli_sha256": sha256_file(stimuli), "rows": len(rows), "base_prompts": len(rows) // len(CONDITIONS),
        "groups": len({r["group_id"] for r in rows}), "conditions": CONDITIONS, "answer_convention": convention,
        "max_new_tokens": MAX_NEW_TOKENS, "target_bucket": TARGET_BUCKET,
        "max_prompt_tokens": max(len(r["input_ids"]) for r in rows),
        "verified_artifact_bytes": verified, "software": compatibility(args.device),
    })
    start = time.perf_counter()
    model = load_model(paths["target"], "target", args.device)
    target = Target(model, tokenizer)
    eos = sorted(generation_eos_token_ids(model, tokenizer))
    timings["loading"] = time.perf_counter() - start
    start = time.perf_counter()
    done = generate(target, rows, convention, run_dir.path / "generations.jsonl", eos)
    timings["generation"] = time.perf_counter() - start
    del target, model
    release_models()
    return {"completed_rows": done, "expected_rows": len(rows), "resolved_eos_token_ids": eos}


# ---------------------------------------------------------------- evaluate (CPU)


CONTRASTS = {
    "position_original": lambda a: a["original_before"] - a["original_after"],
    "position_expanded": lambda a: a["expanded_before"] - a["expanded_after"],
    "wording_before": lambda a: a["expanded_before"] - a["original_before"],
    "wording_after": lambda a: a["expanded_after"] - a["original_after"],
    "interaction": lambda a: (a["expanded_before"] - a["expanded_after"]) - (a["original_before"] - a["original_after"]),
}


def load_family(run: Path) -> tuple[str, list[dict]]:
    manifest = json.loads((run / "manifest.json").read_text())
    completion = json.loads((run / "completion.json").read_text())
    rows = [json.loads(line) for line in (run / "generations.jsonl").open()]
    if completion["status"] != "COMPLETE" or len(rows) != manifest["rows"]:
        raise ValueError(f"{run}: incomplete runs never enter the confirmatory analysis")
    if manifest["plan_hash"] != PLAN_HASH or manifest["protocol_sha256"] != sha256_file(PROTOCOL):
        raise ValueError(f"{run}: plan or protocol differs from the frozen ones")
    for row in rows:
        if answer_text_matches(row["text"], row["answer"], manifest["answer_convention"]) != row["correct"]:
            raise ValueError(f"{row['id']}: stored correctness disagrees with rescoring")
    return manifest["family"], rows


def _group_matrix(rows, value=lambda r: r["correct"], keep=lambda r: True):
    """groups × conditions sums and counts, for prompts passing `keep`."""
    groups = sorted({r["group_id"] for r in rows})
    index = {g: i for i, g in enumerate(groups)}
    sums = {c: np.zeros(len(groups)) for c in CONDITIONS}
    counts = {c: np.zeros(len(groups)) for c in CONDITIONS}
    for r in rows:
        if keep(r):
            sums[r["condition"]][index[r["group_id"]]] += float(value(r))
            counts[r["condition"]][index[r["group_id"]]] += 1
    return groups, sums, counts


def bootstrap(rows, seed: int, keep=lambda r: True) -> dict:
    """Point estimates and resampled draws of every cell and contrast; whole groups resampled."""
    groups, sums, counts = _group_matrix(rows, keep=keep)
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(RESAMPLES, len(groups)))
    point = {c: sums[c].sum() / counts[c].sum() for c in CONDITIONS}
    resampled = {c: sums[c][draws].sum(axis=1) / counts[c][draws].sum(axis=1) for c in CONDITIONS}
    out = {"cells": point, "contrasts": {}, "draws": {}}
    for name, function in CONTRASTS.items():
        out["contrasts"][name] = float(function(point))
        out["draws"][name] = function(resampled)
    return out


def interval(draws: np.ndarray, coverage: float) -> list[float]:
    tail = (1 - coverage) / 2
    return [float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))]


def verdict(kind: str, bounds: list[float]) -> bool:
    if kind == "negative":
        return bounds[1] < 0
    if kind == "equivalent":
        return -MARGIN < bounds[0] and bounds[1] < MARGIN
    raise ValueError(kind)


def _stratum(rows, keep) -> dict:
    chosen = [r for r in rows if keep(r)]
    out = {}
    for c in CONDITIONS:
        cell = [r for r in chosen if r["condition"] == c]
        out[c] = {"n": len(cell), "correct": sum(r["correct"] for r in cell),
                  "accuracy": float(np.mean([r["correct"] for r in cell])) if cell else None,
                  "terminated": sum(r["terminated"] for r in cell),
                  "answered_other_value": sum(r["answered_other_value"] for r in cell)}
    return out


def breakdowns(rows) -> dict:
    out = {}
    for name, levels in (("variable", ("x", "y")), ("last_line_assigns_asked", (True, False)),
                         ("asked_last_update_kind", ("literal", "arithmetic", "copy")),
                         ("asked_last_update_operation", ("assign", "add", "subtract", "copy"))):
        out[name] = {str(level): _stratum(rows, lambda r, n=name, v=level: r[n] == v) for level in levels}
    out["variable_x_last_writer"] = {
        f"{v}__last_line_assigns_asked_{w}": _stratum(rows, lambda r, v=v, w=w: r["variable"] == v and r["last_line_assigns_asked"] == w)
        for v in ("x", "y") for w in (True, False)}
    return out


def family_summary(family: str, rows: list[dict]) -> dict:
    seed = BOOTSTRAP_SEEDS[family]
    full = bootstrap(rows, seed)
    contrasts = {name: {"point": full["contrasts"][name],
                        "confirmatory_interval": interval(full["draws"][name], CONFIRMATORY_COVERAGE),
                        "secondary_95_interval": interval(full["draws"][name], SECONDARY_COVERAGE)}
                 for name in CONTRASTS}
    other_last = bootstrap(rows, seed, keep=lambda r: not r["last_line_assigns_asked"])
    asked_last = bootstrap(rows, seed, keep=lambda r: r["last_line_assigns_asked"])
    concentration = {}
    for wording in ("original", "expanded"):
        name = f"position_{wording}"
        draws = other_last["draws"][name] - asked_last["draws"][name]
        concentration[wording] = {"point": other_last["contrasts"][name] - asked_last["contrasts"][name],
                                  "secondary_95_interval": interval(draws, SECONDARY_COVERAGE)}
    groups = sorted({r["group_id"] for r in rows})
    return {
        "family": family, "groups": len(groups), "base_prompts": len(rows) // len(CONDITIONS),
        "bootstrap": {"resamples": RESAMPLES, "seed": seed, "unit": "whole group, four base prompts × four formats"},
        "cells": _stratum(rows, lambda r: True), "contrasts": contrasts,
        "last_line_concentration": concentration, "breakdowns": breakdowns(rows),
    }


def evaluate(runs: list[Path]) -> dict:
    families = dict(load_family(run) for run in runs)
    summaries = {family: family_summary(family, rows) for family, rows in families.items()}
    verdicts = []
    for test_id, family, contrast, kind in HYPOTHESES:
        if family not in summaries:
            verdicts.append({"id": test_id, "family": family, "contrast": contrast, "kind": kind, "status": "NOT_RUN"})
            continue
        entry = summaries[family]["contrasts"][contrast]
        bounds = entry["confirmatory_interval"]
        verdicts.append({
            "id": test_id, "family": family, "contrast": contrast, "kind": kind, "point": entry["point"],
            "interval": bounds, "coverage": CONFIRMATORY_COVERAGE,
            "status": "SUPPORTED" if verdict(kind, bounds) else "NOT_SUPPORTED",
            "beyond_practical_reference": bounds[1] < -MARGIN if kind == "negative" else None,
        })
    return {"format": FORMAT + ".evaluation", "protocol_sha256": sha256_file(PROTOCOL), "plan_hash": PLAN_HASH,
            "runs": {family: str(run) for (family, _), run in zip(families.items(), runs)},
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
            print(json.dumps({k: h.get(k) for k in ("id", "contrast", "point", "interval", "status", "beyond_practical_reference")}))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RunDirectory(args.run_dir or PROJECT / "runs" / f"p7-confirm-{_family(args.lock)}-{stamp}-{uuid.uuid4().hex[:8]}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timings, started = {}, time.perf_counter()
    status, error, trace, details = "FAILED", None, None, {}
    try:
        details = run(args, run_dir, timings)
        status = "COMPLETE" if details["completed_rows"] == details["expected_rows"] else "PARTIAL"
    except BaseException as failure:  # recorded in completion.json, then re-raised
        error, trace = f"{type(failure).__name__}: {failure}", traceback.format_exc()
        raise
    finally:
        generations = run_dir.path / "generations.jsonl"
        write_json(run_dir.path / "completion.json", {
            "status": status, "error": error, "traceback": trace, **details,
            "rows_written": sum(1 for _ in generations.open()) if generations.exists() else 0,
            "generations_sha256": sha256_file(generations) if generations.exists() else None,
            "timings_seconds": timings, "wall_seconds": time.perf_counter() - started,
            "setup_seconds": float(os.environ["OWL_SETUP_SECONDS"]) if "OWL_SETUP_SECONDS" in os.environ else None,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        })
        print(f"{status}: {run_dir.path}", flush=True)


if __name__ == "__main__":
    main()
