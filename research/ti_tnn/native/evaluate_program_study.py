"""Evaluate the five-split native synthetic-program study on CPU.

The neural model receives only each complete tokenized prompt and its positions.
Reference programs are used separately for target validation and the supplied-program
executor baseline. Implementation and selection checks are not learning results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from . import evaluate as E
from .data import EOS_TOKEN_ID, sha256_file
from .diagnose_learned_state import final_instruction_anchor, suppress_after_state_update
from .evaluate_state_path import evaluate_state_path
from .structured import OP_ROUTES, Instruction, execute


FORMAT = "native-program-study-evaluation-v1"
SPLITS = ("train", "dev", "initial_values", "unseen_composition", "longer_length")
EVALUATION_SPLITS = ("dev", "initial_values", "unseen_composition", "longer_length")
OPERATIONS = tuple(OP_ROUTES)
EVALUATION_COUNT = 128
PER_OPERATION_COUNT = 32
INTERVENTION_COUNT = 32
INTERVENTIONS_PER_OPERATION = 8
INTERVENTION_LIMITS = (
    "An output effect is sensitivity to this local counterfactual, not evidence of semantic faithfulness or answer encoding.",
    "No output effect can reflect redundancy or unaffected paths and does not establish that learned state is absent.",
    "One final-instruction after-state update does not exhaust other tokens, passes, examples, or state uses.",
)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _final_operation(record: Mapping) -> str:
    program = record.get("program")
    if not isinstance(program, (list, tuple)) or not program:
        raise ValueError(f"{record.get('example_id')}: a nonempty saved program is required")
    instructions = []
    for text in program:
        if not isinstance(text, str):
            raise ValueError(f"{record.get('example_id')}: program instructions must be strings")
        instruction = Instruction.parse(text)
        if str(instruction) != text:
            raise ValueError(f"{record.get('example_id')}: noncanonical instruction {text!r}")
        instructions.append(instruction)
    return instructions[-1].op


def _load_sources(root: Path) -> tuple[dict[str, list[dict]], dict]:
    records = {}
    files = {}
    identities = set()
    for split in SPLITS:
        path = root / "structured" / f"{split}.jsonl"
        split_records = E._load_jsonl(path)
        for record in split_records:
            example_id = record["example_id"]
            if record.get("split") != split:
                raise ValueError(f"{example_id}: record split does not match {split}")
            if example_id in identities:
                raise ValueError(f"duplicate example_id across program-study sources: {example_id}")
            identities.add(example_id)
            operation = _final_operation(record)
            expected = execute(record["program"], record["initial"]).target_text
            if E.parse_final_registers(record["target_text"]) is None or record["target_text"] != expected:
                raise ValueError(f"{example_id}: target text is not the canonical executed final state")
            if operation not in OPERATIONS:
                raise ValueError(f"{example_id}: unsupported final operation {operation!r}")
        records[split] = split_records
        files[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "source_examples": len(split_records),
        }
    return records, files


def _ids_from_container(container, split: str, kind: str):
    if not isinstance(container, Mapping):
        return None
    for key in (f"{kind}_ids", f"{kind}_example_ids"):
        value = container.get(key)
        if isinstance(value, Mapping):
            value = value.get(split)
        if value is not None:
            return value
    return None


def _manifest_ids(manifest: Mapping, split: str, kind: str):
    """Read evaluation/intervention IDs while leaving unrelated split inventories alone."""
    split_values = manifest.get("splits", {}).get(split, {})
    evaluation = manifest.get("evaluation", {})
    intervention = manifest.get("interventions", manifest.get("intervention", {}))
    selections = manifest.get("selections", {})
    containers = [manifest, selections, split_values]
    if kind == "evaluation":
        containers.extend((evaluation, evaluation.get(split, {}) if isinstance(evaluation, Mapping) else {}))
        if isinstance(evaluation, Mapping) and isinstance(evaluation.get("splits"), Mapping):
            containers.append(evaluation["splits"].get(split, {}))
    else:
        containers.extend((intervention, intervention.get(split, {}) if isinstance(intervention, Mapping) else {}))
        if isinstance(intervention, Mapping) and isinstance(intervention.get("splits"), Mapping):
            containers.append(intervention["splits"].get(split, {}))
        if isinstance(evaluation, Mapping):
            containers.append(evaluation.get(split, {}))
            if isinstance(evaluation.get("splits"), Mapping):
                containers.append(evaluation["splits"].get(split, {}))
    found = []
    for container in containers:
        value = _ids_from_container(container, split, kind)
        if value is not None:
            found.append(value)
    if not found:
        return None
    first = found[0]
    if any(value != first for value in found[1:]):
        raise ValueError(f"{split}: conflicting manifest {kind} ID selections")
    if not isinstance(first, (list, tuple)) or any(not isinstance(value, str) for value in first):
        raise ValueError(f"{split}: manifest {kind} IDs must be a list of strings")
    if len(first) != len(set(first)):
        raise ValueError(f"{split}: manifest {kind} IDs contain duplicates")
    return list(first)


def _records_for_ids(records: Sequence[Mapping], ids: Sequence[str], split: str) -> list[dict]:
    by_id = {record["example_id"]: record for record in records}
    missing = [example_id for example_id in ids if example_id not in by_id]
    if missing:
        raise ValueError(f"{split}: manifest selected unknown example IDs, starting with {missing[:3]}")
    return [by_id[example_id] for example_id in ids]


def _check_operation_counts(records: Sequence[Mapping], split: str, expected: int) -> dict:
    counts = Counter(_final_operation(record) for record in records)
    wanted = {operation: expected for operation in OPERATIONS}
    if dict(counts) != wanted:
        raise ValueError(f"{split}: final-operation counts {dict(counts)} do not match {wanted}")
    return wanted


def select_evaluation_records(records: Sequence[Mapping], manifest: Mapping, split: str) -> tuple[list[dict], dict]:
    ids = _manifest_ids(manifest, split, "evaluation")
    if ids is None:
        selected = list(records[:EVALUATION_COUNT]) if split == "train" else list(records)
        source = "first_128_train" if split == "train" else "complete_split"
        selected_ids = [record["example_id"] for record in selected]
    else:
        selected = _records_for_ids(records, ids, split)
        source = "manifest"
        selected_ids = list(ids)
    if len(selected) != EVALUATION_COUNT:
        raise ValueError(f"{split}: evaluation requires exactly {EVALUATION_COUNT} examples, found {len(selected)}")
    operation_counts = _check_operation_counts(selected, split, PER_OPERATION_COUNT)
    return selected, {
        "source": source,
        "examples": len(selected),
        "evaluation_ids": selected_ids,
        "per_final_operation": operation_counts,
    }


def select_intervention_records(
    records: Sequence[Mapping], manifest: Mapping, split: str
) -> tuple[list[dict], dict]:
    ids = _manifest_ids(manifest, split, "intervention")
    if ids is None:
        selected = []
        for operation in OPERATIONS:
            selected.extend(
                [record for record in records if _final_operation(record) == operation][:INTERVENTIONS_PER_OPERATION]
            )
        source = "first_8_per_final_operation"
        selected_ids = [record["example_id"] for record in selected]
    else:
        selected = _records_for_ids(records, ids, split)
        source = "manifest"
        selected_ids = list(ids)
    if len(selected) != INTERVENTION_COUNT:
        raise ValueError(f"{split}: interventions require exactly {INTERVENTION_COUNT} examples, found {len(selected)}")
    operation_counts = _check_operation_counts(selected, split, INTERVENTIONS_PER_OPERATION)
    return selected, {
        "source": source,
        "examples": len(selected),
        "intervention_ids": selected_ids,
        "per_final_operation": operation_counts,
    }


def _first_error(completion: str, expected: tuple[int, ...], actual: tuple[int, ...] | None) -> dict | None:
    if actual is None:
        return {"kind": "format", "message": "completion is not exactly four canonical register assignments"}
    for index, (predicted, wanted) in enumerate(zip(actual, expected)):
        if predicted != wanted:
            return {
                "kind": "register_value",
                "register": index,
                "actual": predicted,
                "expected": wanted,
            }
    return None


def _measure_generation(record: Mapping, generated: Mapping) -> dict:
    result = dict(generated)
    prediction = result["completion_text"]
    target = record["target_text"]
    actual = E.parse_final_registers(prediction)
    expected = E.parse_final_registers(target)
    if expected is None:
        raise ValueError(f"{record['example_id']}: target is not canonical final-register text")
    correct = actual == expected and prediction == target
    first_error = _first_error(prediction, expected, actual)
    result.update({
        "example_id": record["example_id"],
        "final_operation": _final_operation(record),
        "target_text": target,
        "valid": actual is not None,
        "correct": correct,
        "completion_tokens": len(result["completion_ids"]),
        "first_divergence": E._first_text_divergence(prediction, target),
        "first_error": first_error,
        "per_register_correct": None if actual is None else [left == right for left, right in zip(actual, expected)],
        "per_register_absolute_error": None if actual is None else [abs(left - right) for left, right in zip(actual, expected)],
        "answer_value_loss": None if actual is None else sum(
            abs(left - right) for left, right in zip(actual, expected)
        ) / 4,
    })
    if result["prompt_truncated"]:
        raise RuntimeError(f"{record['example_id']}: prompt truncation is forbidden in program-study evaluation")
    return result


def _condition_summary(records: Sequence[Mapping]) -> dict:
    count = len(records)
    correct = sum(bool(record["correct"]) for record in records)
    valid = sum(bool(record["valid"]) for record in records)
    stopped = sum(bool(record["stopped_on_eos"]) for record in records)
    truncated = sum(bool(record["truncated"]) for record in records)
    return {
        "examples": count,
        "correct": correct,
        "exact_count": correct,
        "exact_match": _rate(correct, count),
        "valid": valid,
        "format_valid_count": valid,
        "valid_rate": _rate(valid, count),
        "format_validity": _rate(valid, count),
        "stopped_on_eos": stopped,
        "eos_rate": _rate(stopped, count),
        "truncated": truncated,
        "truncation_rate": _rate(truncated, count),
    }


def _generation_summary(records: Sequence[Mapping]) -> dict:
    result = _condition_summary(records)
    valid_records = [record for record in records if record["valid"]]
    valid = len(valid_records)
    result.update({
        "status": "complete",
        "complete": True,
        "evaluated": len(records),
        "prompt_truncated": sum(bool(record["prompt_truncated"]) for record in records),
        "prompt_truncation_rate": _rate(
            sum(bool(record["prompt_truncated"]) for record in records), len(records)
        ),
        "per_register_accuracy": {
            str(index): _rate(sum(record["per_register_correct"][index] for record in valid_records), valid)
            for index in range(4)
        },
        "per_register_mean_absolute_error": {
            str(index): _rate(sum(record["per_register_absolute_error"][index] for record in valid_records), valid)
            for index in range(4)
        },
        "per_register_denominator": valid,
        "answer_value_loss": _rate(
            sum(record["answer_value_loss"] for record in valid_records), valid
        ),
        "answer_value_loss_definition": "mean absolute register-value error over format-valid completions only; not answer-token cross-entropy",
        "first_errors": [
            {"example_id": record["example_id"], **record["first_error"]}
            for record in records if record["first_error"] is not None
        ],
        "by_final_operation": {
            operation: _condition_summary([record for record in records if record["final_operation"] == operation])
            for operation in OPERATIONS
        },
    })
    return result


@torch.no_grad()
def generation_metrics(
    model,
    tokenizer,
    records: Sequence[Mapping],
    *,
    max_new_tokens: int,
    eos_token_id: int = EOS_TOKEN_ID,
) -> dict:
    generated_records = []
    for record in records:
        generated = E.generate_prompt(
            model,
            tokenizer,
            record["prompt"],
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            use_cache=True,
            prompt_policy="error",
        )
        generated_records.append(_measure_generation(record, generated))
    return {
        **_generation_summary(generated_records),
        "metric": "complete greedy canonical final-register answer exact match",
        "model_inputs": ["tokenized prompt", "positions"],
        "forbidden_generation_channels": ["program", "execution trace", "route targets", "state targets", "target text"],
        "target_length_used_for_decoding": False,
        "prompt_policy": "error",
        "max_new_tokens": max_new_tokens,
        "greedy": True,
        "use_cache": True,
        "records": generated_records,
    }


def _baseline_flags(record: Mapping) -> dict:
    return {
        key: record[key]
        for key in ("correct", "valid", "stopped_on_eos", "truncated")
    }


@torch.no_grad()
def targeted_interventions(
    model,
    tokenizer,
    records: Sequence[Mapping],
    ordinary_records: Sequence[Mapping],
    *,
    max_new_tokens: int,
    eos_token_id: int = EOS_TOKEN_ID,
) -> dict:
    baseline_by_id = {record["example_id"]: record for record in ordinary_records}
    if len(baseline_by_id) != len(ordinary_records):
        raise ValueError("ordinary generation contains duplicate example IDs")
    details = []
    for record in records:
        example_id = record["example_id"]
        if example_id not in baseline_by_id:
            raise ValueError(f"{example_id}: no saved ordinary generation for intervention baseline")
        baseline = baseline_by_id[example_id]
        anchor = final_instruction_anchor(record, tokenizer)
        with suppress_after_state_update(model, anchor.token_index) as intervention:
            intervened = _measure_generation(record, E.generate_prompt(
                model,
                tokenizer,
                record["prompt"],
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                use_cache=True,
                prompt_policy="error",
            ))
        if baseline["prompt_ids"] != intervened["prompt_ids"]:
            raise RuntimeError(f"{example_id}: intervention changed the tokenized prompt")
        if intervention["completed_forwards"] != intervention["forwards"]:
            raise RuntimeError(f"{example_id}: incomplete instrumented forward")
        if intervention["targeted_forwards"] != 1 or intervention["suppressed_writer_updates"] != 4:
            raise RuntimeError(f"{example_id}: final-instruction after-state update was not targeted exactly once")
        changed_completion = baseline["completion_ids"] != intervened["completion_ids"]
        changed_suffix = baseline["generated_ids"] != intervened["generated_ids"]
        details.append({
            "example_id": example_id,
            "final_operation": _final_operation(record),
            "target_text": record["target_text"],
            "final_instruction": anchor.metadata(),
            "baseline": _baseline_flags(baseline),
            "intervention": dict(intervention),
            "intervened": intervened,
            "changed_completion": changed_completion,
            "changed_generated_suffix": changed_suffix,
            "first_generated_token_divergence": E._first_token_divergence(
                intervened["generated_ids"], baseline["generated_ids"]
            ),
            "first_completion_text_divergence": E._first_text_divergence(
                intervened["completion_text"], baseline["completion_text"]
            ),
        })
    baseline_records = [baseline_by_id[record["example_id"]] for record in records]
    intervened_records = [record["intervened"] for record in details]
    by_operation = {}
    for operation in OPERATIONS:
        operation_details = [record for record in details if record["final_operation"] == operation]
        operation_ids = {record["example_id"] for record in operation_details}
        by_operation[operation] = {
            "baseline": _condition_summary([record for record in baseline_records if record["example_id"] in operation_ids]),
            "intervened": _condition_summary([record for record in intervened_records if record["example_id"] in operation_ids]),
            "changed_completions": sum(record["changed_completion"] for record in operation_details),
            "changed_generated_suffixes": sum(record["changed_generated_suffix"] for record in operation_details),
        }
    changed_completions = sum(record["changed_completion"] for record in details)
    changed_suffixes = sum(record["changed_generated_suffix"] for record in details)
    return {
        "operation": "suppress every slot-writer output at the final-instruction anchor in zero-based pass two",
        "baseline_source": "saved ordinary generation records; no duplicate baseline decoding",
        "selection_examples": len(records),
        "baseline": _condition_summary(baseline_records),
        "intervened": _condition_summary(intervened_records),
        "changed_completions": changed_completions,
        "changed_completion_rate": _rate(changed_completions, len(details)),
        "changed_generated_suffixes": changed_suffixes,
        "changed_generated_suffix_rate": _rate(changed_suffixes, len(details)),
        "targeted_exactly_once": sum(record["intervention"]["targeted_forwards"] == 1 for record in details),
        "four_writer_updates_suppressed_per_example": sum(
            record["intervention"]["suppressed_writer_updates"] == 4 for record in details
        ),
        "by_final_operation": by_operation,
        "records": details,
        "limits": list(INTERVENTION_LIMITS),
        "evidence_boundary": "intervention sensitivity is not semantic faithfulness; no effect is not absence of learned state",
    }


def _baseline_report(all_records: Mapping[str, list[dict]], selected: Mapping[str, list[dict]]) -> dict:
    selected_report = E.structured_baseline_report(selected)
    full_train_lookup = E.structured_baseline_report({**selected, "train": all_records["train"]})
    exact_lookup = {
        split: selected_report["exact_lookup"][split] if split == "train" else full_train_lookup["exact_lookup"][split]
        for split in SPLITS
    }
    return {
        "metric": "complete target-text exact match on the same selected evaluation records",
        "lookup_fit_examples": len(all_records["train"]),
        "exact_lookup": exact_lookup,
        "supplied_program_executor": selected_report["supplied_program_executor"],
        "boundary": "lookup is memorization; the executor receives reference programs and is not the neural model",
    }


def _validate_manifest(manifest: Mapping) -> None:
    if manifest.get("position_policy") != "document":
        raise ValueError("program-study evaluation requires manifest position_policy='document'")
    for split, values in manifest.get("splits", {}).items():
        if split in SPLITS and values.get("position_policy", "document") != "document":
            raise ValueError(f"{split}: packed split does not declare document positions")


def evaluate(checkpoint, expected_sha256, data, tokenizer, output, max_new_tokens=48):
    if expected_sha256 is None:
        raise ValueError("expected_sha256 is required for program-study checkpoint validation")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    root = Path(data)
    output = Path(output)
    checkpoint_path = Path(checkpoint).resolve()
    output_paths = [output, *(output.parent / f"{split}_state_path.json" for split in SPLITS)]
    if any(path.resolve() == checkpoint_path for path in output_paths):
        raise ValueError("evaluation outputs must not overwrite the checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    _validate_manifest(manifest)
    sources, source_files = _load_sources(root)
    tokenizer_path = Path(tokenizer)
    model, descriptor = E.load_native_model(
        checkpoint, expected_sha256=expected_sha256, device="cpu"
    )
    tokenizer_object = E.load_gpt2_tokenizer(tokenizer_path)
    selected_by_split = {}
    result = {
        "format": FORMAT,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": descriptor["sha256"],
            "step": descriptor.get("step"),
            "config": descriptor.get("config"),
            "device": "cpu",
        },
        "data": {
            "path": str(root),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "position_policy": "document",
            "sources": source_files,
        },
        "generation": {
            "device": "cpu",
            "position_policy": "zero-based prompt and continuing generated positions",
            "prompt_or_target_truncation": False,
            "gold_targets_used_for_decoding": False,
            "oracle_inputs": False,
            "max_new_tokens": max_new_tokens,
        },
        "splits": {},
        "evidence_boundary": "selection and executor checks validate construction; only a future trained-checkpoint run can provide learning results",
    }
    for split in SPLITS:
        selected, selection = select_evaluation_records(sources[split], manifest, split)
        intervention_pool = sources[split] if _manifest_ids(manifest, split, "intervention") is not None else selected
        intervention_records, intervention_selection = select_intervention_records(intervention_pool, manifest, split)
        selected_by_split[split] = selected
        generation = generation_metrics(
            model, tokenizer_object, selected, max_new_tokens=max_new_tokens
        )
        ordinary_records = list(generation["records"])
        ordinary_ids = {record["example_id"] for record in ordinary_records}
        additional_records = [record for record in intervention_records if record["example_id"] not in ordinary_ids]
        if additional_records:
            supplemental = generation_metrics(
                model, tokenizer_object, additional_records, max_new_tokens=max_new_tokens
            )
            ordinary_records.extend(supplemental["records"])
        else:
            supplemental = {"status": "not_required", "examples": 0, "records": []}
        intervention_selection.update({
            "reused_evaluation_baselines": len(intervention_records) - len(additional_records),
            "additional_ordinary_baselines": len(additional_records),
        })
        interventions = targeted_interventions(
            model,
            tokenizer_object,
            intervention_records,
            ordinary_records,
            max_new_tokens=max_new_tokens,
        )
        interventions["supplemental_baseline_generation"] = supplemental
        state_path = evaluate_state_path(
            checkpoint,
            root / split,
            output.parent / f"{split}_state_path.json",
            expected_sha256=expected_sha256,
            batch_size=8,
            max_batches=None,
            structured_directory=root / "structured",
            tokenizer_path=tokenizer_path,
            split=split,
            evaluation_ids=selection["evaluation_ids"],
            position_policy="document",
        )
        if not state_path["complete"] or state_path["evaluated_rows"] != state_path["packed_rows"]:
            raise RuntimeError(f"{split}: state readout did not evaluate every packed row")
        answer_ce = state_path["answer_token_cross_entropy"][split]
        if not answer_ce["complete"] or answer_ce["evaluated"] != len(selected):
            raise RuntimeError(f"{split}: answer-token cross-entropy did not cover every selected example")
        result["splits"][split] = {
            "scope": "memorization fit" if split == "train" else f"held-out {split} diagnostic",
            "selection": {**selection, "interventions": intervention_selection},
            "generation": generation,
            "interventions": interventions,
            "state_path": state_path,
        }
    result["baselines"] = _baseline_report(sources, selected_by_split)
    if sha256_file(checkpoint) != descriptor["sha256"]:
        raise RuntimeError("checkpoint bytes changed during program-study evaluation")
    E._write_json(output, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m native.evaluate_program_study", description=__doc__.split("\n\n")[0])
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--expected-sha256", required=True)
    result.add_argument("--data", required=True)
    result.add_argument("--tokenizer", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--max-new-tokens", type=int, default=48)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    result = evaluate(
        args.checkpoint,
        args.expected_sha256,
        args.data,
        args.tokenizer,
        args.output,
        args.max_new_tokens,
    )
    print(json.dumps({
        "output": args.output,
        "checkpoint_sha256": result["checkpoint"]["sha256"],
        "exact": {
            split: values["generation"]["correct"]
            for split, values in result["splits"].items()
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
