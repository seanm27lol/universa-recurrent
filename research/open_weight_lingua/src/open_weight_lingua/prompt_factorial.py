"""Prepare Phase Seven's five prompt formats on CPU; never load model weights.

The four primary formats cross question wording with question placement.
The fifth repeats the original question before and after the same program.
Only the already-opened calibration and pilot manifests are accepted.
Preparation writes tokenized stimuli and identities, not behavioral results.
A separate source/analysis freeze precedes a GPU run.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from types import SimpleNamespace

from .artifacts import sha256_file, write_json
from .description_census import program, program_text
from .edit_diagnostics import PROJECT, _git_head
from .preflight import model_paths, read_lock, verify_models
from .runner import source_identity
from .target import TARGET_BUCKET, generation_eos_token_ids
from .tasks import interpret

FORMAT = "open_weight_lingua.prompt_factorial_stimuli.v1"
PROTOCOL = PROJECT / "protocols" / "phase_seven_prompt_factorial_v1.md"
ORIGINAL = "What is {variable}? Reply with only the integer."
EXPANDED = "What is {variable} at the end of this program? Reply with only the integer."
CONDITIONS = ("original_before", "original_after", "expanded_before", "expanded_after", "original_repeat")
ORDER_SEED = 2026100507
MAX_NEW_TOKENS = 8
CONVENTIONS = {"gemma3-12b": "rstrip", "qwen2.5-7b": "raw"}
SOURCE_INPUTS = {
    "gemma3-12b": {
        "calibration": ("calibration-20260923T042912Z-d0e9499f", "047046d1d9153d2aff5c14c806d2ea3aec555a8d9999dd968629280a01c90634"),
        "pilot": ("pilot-20260923T043613Z-f7e71d7b", "f14dd70859683108af829e79451874c69f7e1652fd04dd3761be6ae30b61f49c"),
    },
    "qwen2.5-7b": {
        "calibration": ("calibration-20260921T235017Z-fd111b21", "848251d278a8a107fe215268f20a84b577d6e7a06e256cc8874dd85891ea9287"),
        "pilot": ("pilot-20260921T235825Z-6164d210", "cb3ec24ff3187813eb0f2a25ee8397ec7bd94e5b6a8b615588c5615df38b6176"),
    },
}


def render_prompt(program_string: str, variable: str, condition: str) -> str:
    if variable not in ("x", "y") or condition not in CONDITIONS:
        raise ValueError("unknown variable or factorial condition")
    question = (EXPANDED if condition.startswith("expanded_") else ORIGINAL).format(variable=variable)
    if condition.endswith("_before"):
        return question + "\n" + program_string
    if condition.endswith("_after"):
        return program_string + "\n" + question
    return question + "\n" + program_string + "\n" + question


def build_rows(calibration: dict, pilot: dict) -> list[dict]:
    rows, seen, groups = [], set(), {}
    rng = random.Random(ORDER_SEED)
    for split, manifest in (("calibration", calibration), ("pilot", pilot)):
        for item in sorted(manifest["inputs"], key=lambda row: row["id"]):
            if item["id"] in seen or not item["group_id"].startswith(split + "-"):
                raise ValueError("duplicate input ID or unexpected source split")
            seen.add(item["id"])
            members = groups.setdefault(item["group_id"], set())
            member = (item["side"], item["variable"])
            if member in members:
                raise ValueError("duplicate group-side-variable input")
            members.add(member)
            body = program_text(item["prompt"])
            if item["prompt"] != render_prompt(body, item["variable"], "original_after"):
                raise ValueError("source prompt differs from the frozen original wording")
            instructions = program(item["prompt"])
            state = interpret(instructions)
            variable = item["variable"]
            if str(state[variable]) != item["answer"]:
                raise ValueError("source answer disagrees with reference execution")
            last_update = next(s for s in reversed(instructions) if s.variable == variable)
            operation = last_update.operation
            order = list(CONDITIONS)
            rng.shuffle(order)
            for condition in order:
                rows.append({
                    "id": item["id"] + "__" + condition,
                    "base_id": item["id"], "group_id": item["group_id"], "side": item["side"],
                    "variable": variable, "source_split": split, "condition": condition,
                    "primary_factorial": condition != "original_repeat",
                    "program": body, "prompt": render_prompt(body, variable, condition),
                    "answer": item["answer"], "other_answer": str(state["y" if variable == "x" else "x"]),
                    "last_program_variable": instructions[-1].variable,
                    "last_line_assigns_asked": instructions[-1].variable == variable,
                    "last_program_operation": instructions[-1].operation,
                    "asked_last_update_operation": operation,
                    "asked_last_update_kind": {"assign": "literal", "add": "arithmetic", "subtract": "arithmetic", "copy": "copy"}[operation],
                    "statement_count": len(instructions),
                })
    expected = {(side, variable) for side in ("A", "B") for variable in ("x", "y")}
    if any(members != expected for members in groups.values()):
        raise ValueError("every group must retain both sides and both questions")
    return rows


def tokenize(tokenizer, rows: list[dict]) -> None:
    for row in rows:
        ids = list(tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True))
        if not ids or len(ids) + MAX_NEW_TOKENS > TARGET_BUCKET:
            raise ValueError(f"{row['id']}: prompt plus generation does not fit the pinned bucket")
        row.update(input_ids=ids, attention_mask=[1] * len(ids), answer_position=len(ids) - 1)


def prepare(args) -> dict:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    family = args.family
    lock_digest = sha256_file(args.lock)
    manifests, inputs = {}, {}
    for split, directory in (("calibration", args.calibration_run), ("pilot", args.pilot_run)):
        expected_name, expected_hash = SOURCE_INPUTS[family][split]
        if directory.name != expected_name:
            raise ValueError(f"only the declared already-opened {split} run may be read")
        path = directory / "manifest.json"
        if sha256_file(path) != expected_hash:
            raise ValueError(f"{split} source manifest hash changed")
        manifests[split] = json.loads(path.read_text())
        if manifests[split]["model_lock_sha256"] != lock_digest:
            raise ValueError("source manifest and supplied model lock disagree")
        inputs[split] = {"path": str(path.resolve()), "sha256": expected_hash}
    lock = read_lock(args.lock)
    paths = model_paths(lock, args.cache)
    # Hash/tokenizer verification only: no weight shards are read or loaded.
    verified = verify_models({**lock, "models": {"target": lock["models"]["target"]}},
                             {"target": paths["target"]}, metadata_only=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(paths["target"], local_files_only=True, trust_remote_code=False)
    rows = build_rows(manifests["calibration"], manifests["pilot"])
    if len(rows) != 7680 or len({row["group_id"] for row in rows}) != 384:
        raise ValueError("expected the fixed 1,536 base prompts in 384 groups, each in five formats")
    tokenize(tokenizer, rows)
    config_path = paths["target"] / "generation_config.json"
    if not config_path.is_file():
        config_path = paths["target"] / "config.json"
    generation_config = json.loads(config_path.read_text())
    eos_ids = generation_eos_token_ids(SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=generation_config.get("eos_token_id"))), tokenizer)
    result = {
        "format": FORMAT, "family": family, "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(), "protocol_sha256": sha256_file(PROTOCOL),
        "source_file_sha256": source_identity(), "source_inputs": inputs,
        "model_lock_path": str(args.lock.resolve()), "model_lock_sha256": lock_digest,
        "tokenizer_path": str(paths["target"].resolve()), "metadata_verified_bytes": verified,
        "target_files_sha256": {name: entry["sha256"] for name, entry in lock["models"]["target"]["files"].items()},
        "conditions": list(CONDITIONS), "order_seed": ORDER_SEED, "answer_convention": CONVENTIONS[family],
        "max_new_tokens": MAX_NEW_TOKENS, "target_bucket": TARGET_BUCKET,
        "expected_generation_eos_token_ids": sorted(eos_ids),
        "prompts": len(rows), "base_prompts": len(rows) // len(CONDITIONS),
        "groups": 384, "max_prompt_tokens": max(len(row["input_ids"]) for row in rows), "rows": rows,
        "status": "STIMULI_ONLY_NO_MODEL_FORWARD_NO_LAUNCH_FREEZE",
        "scope": "Existing calibration/pilot programs only. Reserved validation splits remain unopened. No activation capture or model weights loaded.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=tuple(SOURCE_INPUTS), required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--calibration-run", type=Path, required=True)
    parser.add_argument("--pilot-run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args)
    print(json.dumps({key: result[key] for key in ("format", "family", "prompts", "max_prompt_tokens", "status")}))


if __name__ == "__main__":
    main()
