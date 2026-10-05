"""Frozen Phase Seven behavior runner and CPU grouped evaluator.

`run` requires a both-family launch freeze and a shared three-hour campaign.
`evaluate` reads completed generations only; it never loads model weights.
"""
import argparse
import json
import os
import signal
import time
from pathlib import Path

import numpy as np

from .artifacts import sha256_file, write_json
from .metrics import answer_text_matches
from .preflight import compatibility, load_model, read_lock, verify_models
from .prompt_factorial import (
    CONDITIONS,
    CONVENTIONS,
    FORMAT,
    MAX_NEW_TOKENS,
    ORDER_SEED,
    PROJECT,
    PROTOCOL,
    SOURCE_INPUTS,
    build_rows,
)
from .runner import release_models, source_identity
from .target import TARGET_BUCKET, Site, Target, generation_eos_token_ids

FREEZE_FORMAT = "open_weight_lingua.prompt_factorial_launch.v1"
WALL_LIMIT_SECONDS = 10800
BOOTSTRAP_REPEATS = 10000
EXPECTED_ROWS = 7680
SEEDS = {"gemma3-12b": 701100, "qwen2.5-7b": 701101}
CONTRASTS = {
    "position_original": [1, -1, 0, 0, 0],
    "position_expanded": [0, 0, 1, -1, 0],
    "wording_before": [-1, 0, 1, 0, 0],
    "wording_after": [0, -1, 0, 1, 0],
    "interaction": [-1, 1, 1, -1, 0],
    "repeat_minus_before": [-1, 0, 0, 0, 1],
    "repeat_minus_after": [0, -1, 0, 0, 1],
}
STRATA = {
    "variable": ("x", "y"), "last_line_assigns_asked": (False, True),
    "asked_last_update_operation": ("assign", "add", "subtract", "copy"),
    "asked_last_update_kind": ("literal", "arithmetic", "copy"),
    "last_program_operation": ("assign", "add", "subtract", "copy"),
    "source_split": ("calibration", "pilot"), "terminated": (False, True),
}


def read(path):
    return json.loads(Path(path).read_text())


def check_deadline(deadline, clock=time.time):
    if clock() >= deadline:
        raise TimeoutError("shared three-hour campaign deadline reached")


def verify_freeze(path):
    frozen = read(path)
    if frozen.get("format") != FREEZE_FORMAT or frozen.get("wall_limit_seconds") != WALL_LIMIT_SECONDS:
        raise ValueError("wrong launch freeze format or wall limit")
    if set(frozen.get("family_stimuli", {})) != set(SOURCE_INPUTS):
        raise ValueError("launch freeze must bind both model families")
    files = frozen.get("files_sha256", {})
    required = {str((PROJECT / name).resolve()) for name in source_identity()}
    required.update(str(path.resolve()) for path in (PROJECT / "tests").rglob("*.py"))
    required.update(str(path.resolve()) for path in (PROJECT / "tests/fixtures").rglob("*") if path.is_file())
    required.update(str(Path(value).resolve()) for value in frozen["family_stimuli"].values())
    if not required <= set(files):
        raise ValueError("launch freeze omits current source, tests or stimuli")
    for filename, digest in files.items():
        if not Path(filename).is_absolute() or sha256_file(Path(filename)) != digest:
            raise ValueError(f"frozen artifact changed: {filename}")
    return frozen


def validate_stimuli(stimuli, family, files=None):
    if stimuli.get("format") != FORMAT or stimuli.get("family") != family:
        raise ValueError("wrong prepared stimulus format or family")
    if (stimuli.get("conditions") != list(CONDITIONS) or stimuli.get("max_new_tokens") != MAX_NEW_TOKENS
            or stimuli.get("target_bucket") != TARGET_BUCKET or stimuli.get("order_seed") != ORDER_SEED
            or stimuli.get("answer_convention") != CONVENTIONS[family]):
        raise ValueError("stimulus settings differ from the fixed protocol")
    if stimuli.get("protocol_sha256") != sha256_file(PROTOCOL):
        raise ValueError("prepared protocol differs from current frozen protocol")
    if stimuli.get("source_file_sha256") != source_identity():
        raise ValueError("prepared source inventory differs; regenerate before launch freeze")
    original = {}
    for split, (name, digest) in SOURCE_INPUTS[family].items():
        entry = stimuli["source_inputs"][split]
        path = Path(entry["path"])
        if path.parent.name != name or entry["sha256"] != digest:
            raise ValueError("only pinned already-opened input manifests may be read")
        if files is not None and files.get(str(path.resolve())) != digest:
            raise ValueError("source manifest missing from launch freeze")
        if sha256_file(path) != digest:
            raise ValueError("source manifest changed")
        original[split] = read(path)
    expected = build_rows(original["calibration"], original["pilot"])
    if len(stimuli["rows"]) != EXPECTED_ROWS or len(expected) != EXPECTED_ROWS:
        raise ValueError("all 7,680 prepared conditions are required")
    for row, reference in zip(stimuli["rows"], expected, strict=True):
        if any(row.get(key) != value for key, value in reference.items()):
            raise ValueError("prepared order, grouping, program, prompt or reference metadata differs")
        ids = row["input_ids"]
        if (not ids or any(type(token) is not int or token < 0 for token in ids)
                or len(ids) + MAX_NEW_TOKENS > TARGET_BUCKET or row["attention_mask"] != [1] * len(ids)
                or row["answer_position"] != len(ids) - 1):
            raise ValueError("invalid prepared tokens or bucket")


def verified_pair(freeze_path):
    frozen = verify_freeze(freeze_path)
    stimuli = {family: read(path) for family, path in frozen["family_stimuli"].items()}
    for family, item in stimuli.items():
        validate_stimuli(item, family, frozen["files_sha256"])
        lock_path = Path(item["model_lock_path"])
        if frozen["files_sha256"].get(str(lock_path.resolve())) != item["model_lock_sha256"]:
            raise ValueError("model lock absent from launch freeze")
        if sha256_file(lock_path) != item["model_lock_sha256"]:
            raise ValueError("model lock changed")
        lock = read_lock(lock_path)
        target_path = Path(item["tokenizer_path"])
        for filename, metadata in lock["models"]["target"]["files"].items():
            if not filename.endswith(".safetensors") and frozen["files_sha256"].get(str((target_path / filename).resolve())) != metadata["sha256"]:
                raise ValueError("target tokenizer/config metadata absent from freeze")
    first, second = stimuli.values()
    fields = ("id", "base_id", "group_id", "side", "variable", "source_split", "condition", "program", "prompt", "answer")
    if any(any(a[field] != b[field] for field in fields) for a, b in zip(first["rows"], second["rows"], strict=True)):
        raise ValueError("model families do not share identical paired stimuli")
    return frozen, stimuli


def campaign_record(directory, freeze_path, now=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "campaign.json"
    now = time.time() if now is None else now
    digest = sha256_file(freeze_path)
    try:
        with path.open("x") as stream:
            record = {"freeze_path": str(freeze_path.resolve()), "freeze_sha256": digest,
                      "started_unix": now, "deadline_unix": now + WALL_LIMIT_SECONDS,
                      "wall_limit_seconds": WALL_LIMIT_SECONDS}
            json.dump(record, stream, indent=2)
    except FileExistsError:
        record = read(path)
        if (record["freeze_sha256"] != digest or record["wall_limit_seconds"] != WALL_LIMIT_SECONDS
                or record["deadline_unix"] != record["started_unix"] + WALL_LIMIT_SECONDS):
            raise ValueError("campaign identity/deadline changed")
    check_deadline(record["deadline_unix"], lambda: now)
    return record


def generate_rows(target, stimuli, destination, deadline, clock=time.time, timings=None, progress=None):
    """Append and flush each completed observation; propagate failures without retries."""
    eos = sorted(generation_eos_token_ids(target.model, target.tokenizer))
    if eos != stimuli["expected_generation_eos_token_ids"]:
        raise ValueError("loaded model stop set differs from preparation")
    timings = {} if timings is None else timings
    progress = {} if progress is None else progress
    timings.update(generation=0., incremental_serialization=0.)
    with destination.open("x") as stream:
        for row in stimuli["rows"]:
            progress["active_stimulus_id"] = row["id"]
            check_deadline(deadline, clock)
            ids, mask = target.tensors([row["input_ids"]], [row["attention_mask"]])
            start = time.perf_counter()
            try:
                generated = target.greedy(ids, mask, Site(layer=0, position=row["answer_position"]), max_tokens=MAX_NEW_TOKENS)
            finally:
                timings["generation"] += time.perf_counter() - start
            start = time.perf_counter()
            record = {**row, "family": stimuli["family"], "generated_token_ids": generated["token_ids"],
                      "text": generated["text"], "terminated": generated["terminated"], "resolved_eos_token_ids": eos,
                      "stop_token_id": generated["token_ids"][-1] if generated["terminated"] else None,
                      "generated_token_count": len(generated["token_ids"]),
                      "correct": answer_text_matches(generated["text"], row["answer"], stimuli["answer_convention"])}
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            timings["incremental_serialization"] += time.perf_counter() - start
            progress["last_completed_stimulus_id"] = row["id"]
            progress["completed_rows"] = progress.get("completed_rows", 0) + 1
            if progress["completed_rows"] % 256 == 0:
                print(json.dumps({"family": stimuli["family"], "completed_rows": progress["completed_rows"],
                                  "total_rows": len(stimuli["rows"])}), flush=True)
            check_deadline(deadline, clock)


def run(args):
    if args.output.exists():
        raise FileExistsError("run destination exists; no implicit resume")
    # The shared clock starts before verification/loading, including second-stage gaps.
    campaign = campaign_record(args.campaign, args.freeze)
    active = args.campaign / ".active"
    with active.open("x") as stream:
        stream.write(args.family)
    try:
        with (args.campaign / (args.family + ".claimed")).open("x") as stream:
            stream.write(str(args.output.resolve()))
    except BaseException:
        active.unlink()
        raise
    args.output.mkdir(parents=True)
    started, timings, progress = time.perf_counter(), {}, {}
    status, error, model, tokenizer = "FAILED", None, None, None
    target, original_forward, bounded_forward = None, None, None
    generated_path = args.output / "generations.jsonl"
    deadline = campaign["deadline_unix"]
    old_handler = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError("shared three-hour campaign deadline reached")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.time()))
    try:
        start = time.perf_counter()
        frozen, both_stimuli = verified_pair(args.freeze)
        stimuli = both_stimuli[args.family]
        lock_path, target_path = Path(stimuli["model_lock_path"]), Path(stimuli["tokenizer_path"])
        lock = read_lock(lock_path)
        verified = verify_models({**lock, "models": {"target": lock["models"]["target"]}}, {"target": target_path})
        timings["hash_verification"] = time.perf_counter() - start
        check_deadline(deadline)
        import torch
        from transformers import AutoTokenizer
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True, trust_remote_code=False)
        # Re-tokenize before model loading to bind saved IDs to the pinned template.
        for row in stimuli["rows"]:
            actual = list(tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True))
            if actual != row["input_ids"]:
                raise ValueError("prepared token IDs disagree with pinned tokenizer")
        model = load_model(target_path, "target", args.device)
        model.eval()
        target = Target(model, tokenizer)
        original_forward = target.forward
        def bounded_forward(*positional, **keywords):
            check_deadline(deadline)
            value = original_forward(*positional, **keywords)
            check_deadline(deadline)
            return value
        target.forward = bounded_forward
        timings["tokenization_and_loading"] = time.perf_counter() - start
        write_json(args.output / "manifest.json", {"family": args.family, "freeze_sha256": sha256_file(args.freeze),
                    "freeze_path": str(args.freeze.resolve()), "stimuli_path": frozen["family_stimuli"][args.family],
                    "stimuli_sha256": sha256_file(Path(frozen["family_stimuli"][args.family])), "campaign": campaign,
                    "verified_target_bytes": verified, "software": compatibility(args.device),
                    "generation_settings": {"max_new_tokens": MAX_NEW_TOKENS, "bucket": TARGET_BUCKET, "greedy": True,
                                            "use_cache": False, "bf16": True, "eager_attention": True}})
        start = time.perf_counter()
        with torch.inference_mode():
            generate_rows(target, stimuli, generated_path, deadline, timings=timings, progress=progress)
        timings["generation_and_incremental_serialization"] = time.perf_counter() - start
        target, original_forward, bounded_forward, model = None, None, None, None
        release_models()
        start = time.perf_counter()
        verify_freeze(args.freeze)
        verify_models({**lock, "models": {"target": lock["models"]["target"]}}, {"target": target_path})
        timings["completion_hash_verification"] = time.perf_counter() - start
        check_deadline(deadline)
        status = "COMPLETE"
    except TimeoutError as failure:
        status, error = "PARTIAL_TIMEOUT", str(failure)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        target, original_forward, bounded_forward, model, tokenizer = None, None, None, None, None
        release_models()
        completed = sum(1 for line in generated_path.open() if line.strip()) if generated_path.exists() else 0
        write_json(args.output / "completion.json", {"status": status, "error": error, "completed_rows": completed,
                    "expected_rows": EXPECTED_ROWS, "timings_seconds": timings, "progress": progress, "wall_seconds": time.perf_counter() - started,
                    "campaign": campaign, "setup_seconds": float(os.environ.get("OWL_SETUP_SECONDS", "0")),
                    "generations_sha256": sha256_file(generated_path) if generated_path.exists() else None})
        active.unlink(missing_ok=True)
    return status


def validate_records(stimuli, records):
    if len(records) != len(stimuli["rows"]):
        raise ValueError("partial generations cannot enter the primary analysis")
    ids = set()
    for row, record in zip(stimuli["rows"], records, strict=True):
        if record["id"] in ids or any(record.get(key) != value for key, value in row.items()):
            raise ValueError("generation order, pairing or metadata differs from stimuli")
        ids.add(record["id"])
        tokens, eos = record["generated_token_ids"], stimuli["expected_generation_eos_token_ids"]
        terminated = record["terminated"]
        if (record["family"] != stimuli["family"] or record["resolved_eos_token_ids"] != eos
                or not tokens or len(tokens) > MAX_NEW_TOKENS or any(type(token) is not int or token < 0 for token in tokens)
                or type(terminated) is not bool or record["generated_token_count"] != len(tokens)
                or any(token in eos for token in tokens[:-1]) or terminated != (tokens[-1] in eos)
                or record["stop_token_id"] != (tokens[-1] if terminated else None)
                or (not terminated and len(tokens) != MAX_NEW_TOKENS)):
            raise ValueError("inconsistent generation or termination record")
        correct = answer_text_matches(record["text"], row["answer"], stimuli["answer_convention"])
        if type(record["correct"]) is not bool or record["correct"] != correct:
            raise ValueError("stored correctness disagrees with frozen text rule")


def cells(records):
    result = {}
    for condition in CONDITIONS:
        rows = [row for row in records if row["condition"] == condition]
        count = len(rows)
        hits = sum(row["correct"] for row in rows)
        terminated_hits = sum(row["correct"] and row["terminated"] for row in rows)
        result[condition] = {"correct": hits, "total": count, "accuracy": hits / count if count else None,
                             "terminated": sum(row["terminated"] for row in rows),
                             "correct_and_terminated": terminated_hits,
                             "correct_and_terminated_fraction": terminated_hits / count if count else None}
    return result


def breakdown(records):
    table = cells(records)
    differences = {}
    for wording in ("original", "expanded"):
        before, after = table[wording + "_before"]["accuracy"], table[wording + "_after"]["accuracy"]
        differences[wording] = None if before is None or after is None else before - after
    return {"cells": table, "before_minus_after": differences}


def grouped_analysis(records, family, repeats=BOOTSTRAP_REPEATS):
    """Pure CPU evaluator: form matched group contrasts, then resample groups."""
    groups = {}
    for row in records:
        bases = groups.setdefault(row["group_id"], {})
        values = bases.setdefault(row["base_id"], {})
        if row["condition"] in values:
            raise ValueError("duplicate paired condition")
        values[row["condition"]] = row["correct"]
    names = sorted(groups)
    grouped = []
    for name in names:
        bases = groups[name]
        if len(bases) != 4 or any(set(values) != set(CONDITIONS) for values in bases.values()):
            raise ValueError("group must keep four bases and all five conditions")
        grouped.append(np.array([[values[condition] for condition in CONDITIONS] for values in bases.values()], dtype=float).mean(0))
    grouped = np.asarray(grouped)
    contrasts = grouped @ np.asarray(list(CONTRASTS.values()), dtype=float).T
    rng = np.random.default_rng(SEEDS[family])
    draws = np.empty((repeats, len(CONTRASTS)))
    for begin in range(0, repeats, 128):
        count = min(128, repeats - begin)
        indices = rng.integers(0, len(names), (count, len(names)))
        draws[begin:begin + count] = contrasts[indices].mean(1)
    estimates = {}
    for index, name in enumerate(CONTRASTS):
        primary = index < 5
        tail = .0025 if primary else .025
        lower, upper = np.quantile(draws[:, index], [tail, 1 - tail])
        point = float(contrasts[:, index].mean())
        estimates[name] = {"difference": point, "interval": [float(lower), float(upper)],
                           "coverage": 1 - 2 * tail, "primary": primary,
                           "resolved_sign": "positive" if lower > 0 else "negative" if upper < 0 else "unresolved",
                           "practical_reference": .05, "point_absolute_at_least_005": abs(point) >= .05,
                           "interval_supports_at_least_005": bool(lower >= .05 or upper <= -.05)}
    strata = {key: {str(value): breakdown([row for row in records if row[key] == value]) for value in values}
              for key, values in STRATA.items()}
    joint = {f"{variable}__last_line_assigns_asked_{writer}": breakdown([row for row in records
                  if row["variable"] == variable and row["last_line_assigns_asked"] == writer])
             for variable in ("x", "y") for writer in (False, True)}
    summary = {"family": family, "groups": len(names), "base_prompts": 4 * len(names), "cells": cells(records),
               "contrasts": estimates, "strata": strata, "joint_asked_variable_last_writer": joint,
               "bootstrap": {"resamples": repeats, "seed": SEEDS[family], "unit": "whole group, all four bases and five conditions together"},
               "scope": "Reused calibration/pilot inputs; ten primary model-by-contrast comparisons form the Bonferroni family. Repeat diagnostics and subgroup summaries are separate.",
               "termination_strata_scope": "Cells condition on each format's observed termination; their differences are descriptive conditional-mean differences, not matched causal contrasts."}
    return summary, {"group_ids": np.array(names), "group_cell_means": grouped, "group_contrasts": contrasts,
                     "bootstrap_contrasts": draws, "contrast_names": np.array(list(CONTRASTS))}


def evaluate(args):
    if args.output.exists():
        raise FileExistsError("evaluation destination exists")
    started = time.perf_counter()
    frozen, both_stimuli = verified_pair(args.freeze)
    manifest, completion = read(args.run / "manifest.json"), read(args.run / "completion.json")
    if completion["status"] != "COMPLETE" or manifest["freeze_sha256"] != sha256_file(args.freeze):
        raise ValueError("only complete runs under this launch freeze may be evaluated")
    family = manifest["family"]
    stimuli = both_stimuli[family]
    stimulus_path = Path(frozen["family_stimuli"][family])
    if manifest["stimuli_sha256"] != sha256_file(stimulus_path):
        raise ValueError("run stimulus identity differs")
    generation_path = args.run / "generations.jsonl"
    if completion["generations_sha256"] != sha256_file(generation_path):
        raise ValueError("generations changed after completion")
    records = [json.loads(line) for line in generation_path.read_text().splitlines() if line.strip()]
    validate_records(stimuli, records)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(stimuli["tokenizer_path"], local_files_only=True, trust_remote_code=False)
    for record in records:
        tokens = record["generated_token_ids"][:-1] if record["terminated"] else record["generated_token_ids"]
        if tokenizer.decode(tokens, skip_special_tokens=False) != record["text"]:
            raise ValueError("recorded text disagrees with generated token IDs")
    summary, arrays = grouped_analysis(records, family)
    summary.update(run_directory=str(args.run.resolve()), freeze_sha256=sha256_file(args.freeze),
                   generations_sha256=sha256_file(generation_path), analysis_seconds=time.perf_counter() - started)
    verify_freeze(args.freeze)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "bootstrap.npz", **arrays)
    write_json(args.output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    running = commands.add_parser("run")
    running.add_argument("--family", choices=tuple(SOURCE_INPUTS), required=True)
    running.add_argument("--freeze", type=Path, required=True)
    running.add_argument("--campaign", type=Path, required=True)
    running.add_argument("--output", type=Path, required=True)
    running.add_argument("--device", default="cuda")
    evaluating = commands.add_parser("evaluate")
    evaluating.add_argument("--freeze", type=Path, required=True)
    evaluating.add_argument("--run", type=Path, required=True)
    evaluating.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        status = run(args)
        print(json.dumps({"status": status, "output": str(args.output)}))
        if status != "COMPLETE":
            raise SystemExit(2)
    else:
        summary = evaluate(args)
        print(json.dumps({"family": summary["family"], "groups": summary["groups"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
