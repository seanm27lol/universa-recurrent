"""Independently rescore native generations without importing producer code."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REGISTER = re.compile(r"^r0=(0|[1-9][0-9]*) r1=(0|[1-9][0-9]*) r2=(0|[1-9][0-9]*) r3=(0|[1-9][0-9]*)$")
MODES = ("dense", "shared", "untyped", "typed")
SPLITS = ("train", "dev", "unseen_composition", "longer_length")
PROGRAM_STUDY_SPLITS = ("train", "dev", "initial_values", "unseen_composition", "longer_length")
PROGRAM_STUDY_FORMAT = "native-program-study-evaluation-v1"
OPERATIONS = ("set", "add", "multiply", "copy")
INSTRUCTION = re.compile(r"^(set|add|multiply|copy) r([0-3]) (?:r([0-3])|(0|[1-9][0-9]*))$")


class ScoringError(ValueError):
    pass


def execute_saved_program(record):
    if record.get("modulus") != 32:
        raise ScoringError(f"{record['example_id']}: modulus must be 32")
    initial = record.get("initial")
    if not isinstance(initial, list) or len(initial) != 4 or any(
            not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 32 for value in initial):
        raise ScoringError(f"{record['example_id']}: initial register values must be four integers in [0, 32)")
    state = tuple(initial)
    program = record.get("program")
    if not isinstance(program, list) or not program:
        raise ScoringError(f"{record['example_id']}: program must be nonempty")
    for text in program:
        match = INSTRUCTION.fullmatch(text)
        if match is None:
            raise ScoringError(f"{record['example_id']}: noncanonical instruction {text!r}")
        operation, destination, source, immediate = match.groups()
        destination = int(destination)
        if operation == "set":
            if source is not None or immediate is None:
                raise ScoringError(f"{record['example_id']}: malformed set instruction {text!r}")
            value = int(immediate)
            if not 0 <= value < 32:
                raise ScoringError(f"{record['example_id']}: immediate outside [0, 32): {text!r}")
            state = tuple(value if index == destination else current for index, current in enumerate(state))
        else:
            if source is None or immediate is not None:
                raise ScoringError(f"{record['example_id']}: malformed {operation} instruction {text!r}")
            source = int(source)
            if operation == "add":
                values = (state[destination] + state[source]) % 32
            elif operation == "multiply":
                values = (state[destination] * state[source]) % 32
            else:
                values = state[source]
            state = tuple(values if index == destination else current for index, current in enumerate(state))
    expected = registers(record["target_text"])
    if expected is None or expected != state:
        raise ScoringError(f"{record['example_id']}: stored target {record['target_text']!r} does not execute to {state!r}")
    import hashlib
    identity = hashlib.sha256((record["prompt"] + record["target_text"]).encode()).hexdigest()[:20]
    if record["example_id"] != f"structured-{record['split']}-{identity}":
        raise ScoringError(f"{record['example_id']}: example identity does not match prompt and target")
    return state


def load(path):
    return json.loads(Path(path).read_text())


def registers(text):
    if not isinstance(text, str):
        return None
    match = REGISTER.fullmatch(text)
    if match is None:
        return None
    values = tuple(int(value) for value in match.groups())
    return values if all(0 <= value < 32 for value in values) else None


def source_records(data, splits=SPLITS):
    data = Path(data)
    result = {}
    for split in splits:
        path = data / "structured" / f"{split}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            record = json.loads(line)
            result[record["example_id"]] = record
    return result


def prerequisite(record):
    prompt = record["prompt"]
    lines = prompt.splitlines()
    instructions = [line for line in lines if line.startswith(("set ", "add ", "multiply ", "copy "))]
    complete = (
        any(line.startswith("Initial: r0=") for line in lines)
        and "Program:" in lines
        and instructions == list(record["program"])
        and "Answer with the final four registers." in lines
    )
    return {
        "complete": complete,
        "initial_values_present": any(line.startswith("Initial: r0=") for line in lines),
        "program_source_present": "Program:" in lines and instructions == list(record["program"]),
        "instruction_order_preserved": instructions == list(record["program"]),
        "program_source_is_in_prompt": True,
    }


def error_details(completion, target, expected, actual):
    if actual is None:
        return {"kind": "format", "message": "completion is not exactly four canonical register assignments"}
    for index, (left, right) in enumerate(zip(actual, expected)):
        if left != right:
            return {"kind": "register_value", "register": index, "actual": left, "expected": right}
    return None


def summarize(records, source):
    exact = valid = eos = truncated = complete = incomplete = 0
    register_correct = Counter()
    register_absolute = Counter()
    value_loss = []
    first_errors = []
    details = []
    score_changes = []
    for record in records:
        source_record = source[record["example_id"]]
        if record.get("target_text") != source_record["target_text"]:
            raise ScoringError(f"{record['example_id']}: saved generation target_text does not match validated source target")
        expected = registers(source_record["target_text"])
        if expected is None:
            raise ScoringError(f"{record['example_id']}: validated source target is not canonical")
        actual = registers(record["completion_text"])
        is_exact = (
            expected is not None
            and actual is not None
            and actual == expected
            and record["completion_text"] == record["target_text"]
        )
        required = prerequisite(source_record)
        is_complete = required["complete"] and not record.get("prompt_truncated", False)
        complete += is_complete
        incomplete += not is_complete
        exact += is_exact
        valid += actual is not None
        eos += bool(record.get("stopped_on_eos"))
        truncated += bool(record.get("truncated"))
        first_error = error_details(record["completion_text"], record["target_text"], expected, actual)
        if record.get("correct") is not None and record["correct"] != is_exact:
            score_changes.append({"example_id": record["example_id"], "metric": "exact_match",
                                  "original": record["correct"], "independent": is_exact})
        if record.get("valid") is not None and record["valid"] != (actual is not None):
            score_changes.append({"example_id": record["example_id"], "metric": "format_valid",
                                  "original": record["valid"], "independent": actual is not None})
        if actual is not None:
            for index, (left, right) in enumerate(zip(actual, expected)):
                register_correct[index] += left == right
                register_absolute[index] += abs(left - right)
            value_loss.append(sum(abs(left - right) for left, right in zip(actual, expected)) / 4)
        details.append({
            "example_id": record["example_id"],
            "exact_match": is_exact,
            "format_valid": actual is not None,
            "answer_value_loss": None if actual is None else sum(
                abs(left - right) for left, right in zip(actual, expected)) / 4,
            "first_error": first_error,
            "stopped_on_eos": bool(record.get("stopped_on_eos")),
            "completion_truncated": bool(record.get("truncated")),
            "prerequisites_complete": is_complete,
            **required,
        })
        if first_error is not None and len(first_errors) < 8:
            first_errors.append({"example_id": record["example_id"], **first_error})
    count = len(records)
    return {
        "examples": count,
        "exact_match": exact / count if count else None,
        "exact_count": exact,
        "format_validity": valid / count if count else None,
        "format_valid_count": valid,
        "per_register_accuracy": {str(index): register_correct[index] / valid if valid else None for index in range(4)},
        "per_register_mean_absolute_error": {str(index): register_absolute[index] / valid if valid else None for index in range(4)},
        "answer_value_loss": sum(value_loss) / len(value_loss) if value_loss else None,
        "answer_value_loss_definition": "mean absolute register-value error over format-valid completions only",
        "eos_rate": eos / count if count else None,
        "completion_truncation_rate": truncated / count if count else None,
        "prerequisites": {
            "complete": complete,
            "truncated_or_incomplete": incomplete,
            "program_source_is_in_prompt": True,
        },
        "first_errors": first_errors,
        "score_changes": score_changes,
        "records": details,
    }


def summarize_program_study(records, source):
    result = summarize(records, source)
    result["by_final_operation"] = {}
    for operation in OPERATIONS:
        selected = [
            record for record in records
            if source[record["example_id"]]["program"][-1].split(" ", 1)[0] == operation
        ]
        summary = summarize(selected, source)
        result["by_final_operation"][operation] = {
            key: summary[key]
            for key in ("examples", "exact_count", "exact_match", "format_valid_count", "format_validity")
        }
    return result


def summarize_program_study_interventions(records, ordinary_records, source):
    baselines = {}
    for record in ordinary_records:
        example_id = record["example_id"]
        if example_id in baselines:
            raise ScoringError(f"{example_id}: duplicate ordinary generation for intervention join")
        baselines[example_id] = record
    intervened_records = []
    details = []
    score_changes = []
    identities = set()
    for record in records:
        example_id = record["example_id"]
        if example_id in identities:
            raise ScoringError(f"{example_id}: duplicate intervention record")
        identities.add(example_id)
        if example_id not in baselines:
            raise ScoringError(f"{example_id}: no primary or supplemental ordinary generation")
        intervened = record["intervened"]
        if intervened.get("example_id") != example_id:
            raise ScoringError(f"{example_id}: intervened generation has a different example identity")
        baseline = baselines[example_id]
        changed_completion = baseline["completion_ids"] != intervened["completion_ids"]
        changed_suffix = baseline["generated_ids"] != intervened["generated_ids"]
        if record.get("changed_completion") != changed_completion:
            score_changes.append({
                "example_id": example_id,
                "metric": "changed_completion",
                "original": record.get("changed_completion"),
                "independent": changed_completion,
            })
        if record.get("changed_generated_suffix") != changed_suffix:
            score_changes.append({
                "example_id": example_id,
                "metric": "changed_generated_suffix",
                "original": record.get("changed_generated_suffix"),
                "independent": changed_suffix,
            })
        final_operation = source[example_id]["program"][-1].split(" ", 1)[0]
        details.append({
            "example_id": example_id,
            "final_operation": final_operation,
            "changed_completion": changed_completion,
            "changed_generated_suffix": changed_suffix,
        })
        intervened_records.append(intervened)
    by_operation = {}
    for operation in OPERATIONS:
        selected = [record for record in details if record["final_operation"] == operation]
        by_operation[operation] = {
            "examples": len(selected),
            "changed_completions": sum(record["changed_completion"] for record in selected),
            "changed_generated_suffixes": sum(record["changed_generated_suffix"] for record in selected),
        }
    return {
        "examples": len(records),
        "intervened": summarize_program_study(intervened_records, source),
        "changed_completions": sum(record["changed_completion"] for record in details),
        "changed_generated_suffixes": sum(record["changed_generated_suffix"] for record in details),
        "by_final_operation": by_operation,
        "score_changes": score_changes,
        "records": details,
    }


def rescore(study, data, output, modes=None, *, evaluation_paths=None):
    study = Path(study)
    explicit_paths = evaluation_paths is not None
    if explicit_paths:
        if modes is not None:
            raise ValueError("modes and evaluation_paths are mutually exclusive")
        if not hasattr(evaluation_paths, "items"):
            raise ValueError("evaluation_paths must be a mapping from mode names to evaluation JSON paths")
        selected_modes = tuple(evaluation_paths)
        if not selected_modes:
            raise ValueError("evaluation_paths must not be empty")
        unknown = set(selected_modes) - set(MODES)
        if unknown:
            raise ValueError(f"unknown generation-rescore modes: {sorted(unknown)}")
        evaluation_paths = {mode: Path(evaluation_paths[mode]) for mode in selected_modes}
        for mode, path in evaluation_paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{mode}: explicit evaluation path does not exist: {path}")
    else:
        if modes is None:
            selected_modes = MODES
        else:
            if isinstance(modes, str):
                raise ValueError("modes must be an iterable of mode names, not a string")
            selected_modes = tuple(modes)
            if len(set(selected_modes)) != len(selected_modes):
                raise ValueError("modes must not contain duplicates")
            unknown = set(selected_modes) - set(MODES)
            if unknown:
                raise ValueError(f"unknown generation-rescore modes: {sorted(unknown)}")
        evaluation_paths = {
            mode: study / "arms" / mode / "evaluation.json"
            for mode in selected_modes
            if (study / "arms" / mode / "evaluation.json").is_file()
        }
    program_study = any(load(path).get("format") == PROGRAM_STUDY_FORMAT for path in evaluation_paths.values())
    sources = source_records(data, PROGRAM_STUDY_SPLITS if program_study else SPLITS)
    result = {"format": "native-independent-generation-rescore-v1", "study": str(study), "arms": {}}
    if explicit_paths:
        result["evaluation_paths"] = {mode: str(path.resolve()) for mode, path in evaluation_paths.items()}
    if program_study:
        result["evaluation_format"] = PROGRAM_STUDY_FORMAT
    validated = 0
    for record in sources.values():
        execute_saved_program(record)
        validated += 1
    result["target_validation"] = {"executed_programs": validated, "rejected_inconsistent_targets": []}
    for mode, evaluation_path in evaluation_paths.items():
        evaluation = load(evaluation_path)
        if evaluation.get("format") == PROGRAM_STUDY_FORMAT:
            arm = {
                "structured": {},
                "supplemental_baseline_generation": {},
                "interventions": {},
                "code": {"status": "not_run", "reason": "code generation is not part of native program-study evaluation"},
                "answer_token_cross_entropy": {},
            }
            for split in PROGRAM_STUDY_SPLITS:
                split_evaluation = evaluation["splits"][split]
                primary_records = split_evaluation["generation"]["records"]
                interventions = split_evaluation["interventions"]
                supplemental_records = interventions["supplemental_baseline_generation"]["records"]
                arm["structured"][split] = summarize_program_study(primary_records, sources)
                arm["supplemental_baseline_generation"][split] = summarize_program_study(
                    supplemental_records, sources
                )
                arm["interventions"][split] = summarize_program_study_interventions(
                    interventions["records"], [*primary_records, *supplemental_records], sources
                )
                answer_losses = split_evaluation["state_path"]["answer_token_cross_entropy"]
                arm["answer_token_cross_entropy"][split] = answer_losses[split]
            result["arms"][mode] = arm
            continue
        if evaluation.get("format") == "native-acquisition-evaluation-v1":
            arm = {
                "structured": {},
                "code": {"status": "not_run", "reason": "code generation is not part of acquisition evaluation"},
                "answer_token_cross_entropy": {},
            }
            for split in ("train", "dev"):
                if split not in evaluation["splits"]:
                    continue
                split_evaluation = evaluation["splits"][split]
                arm["structured"][split] = summarize(split_evaluation["generation"]["records"], sources)
                answer_losses = split_evaluation["state_path"]["answer_token_cross_entropy"]
                arm["answer_token_cross_entropy"][split] = answer_losses[split]
            result["arms"][mode] = arm
            continue
        arm = {"structured": {}, "code": None}
        for split in SPLITS:
            records = evaluation["structured_generation"]["splits"][split]["records"]
            arm["structured"][split] = summarize(records, sources)
        code_records = evaluation["code_generation"].get("records", [])
        exact = sum(record["completion_text"] == record["target_text"] for record in code_records)
        arm["code"] = {
            "status": "complete" if code_records else evaluation["code_generation"].get("status", "not_run"),
            "examples": len(code_records),
            "exact_match": exact / len(code_records) if code_records else None,
            "exact_count": exact,
            "eos_rate": sum(bool(record.get("stopped_on_eos")) for record in code_records) / len(code_records) if code_records else None,
            "completion_truncation_rate": sum(bool(record.get("truncated")) for record in code_records) / len(code_records) if code_records else None,
            "prerequisites": {
                "complete": sum(not record.get("prompt_truncated", False) for record in code_records),
                "truncated_or_incomplete": sum(bool(record.get("prompt_truncated", False)) for record in code_records),
            },
            "first_errors": [
                {"example_id": record["example_id"], "kind": "text", **record["first_divergence"]}
                for record in code_records[:8]
            ],
        }
        result["arms"][mode] = arm
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = rescore(Path(args.study), Path(args.data), args.output)
    print(json.dumps({"status": "complete", "arms": list(result["arms"]), "output": args.output}, sort_keys=True))


if __name__ == "__main__":
    main()
