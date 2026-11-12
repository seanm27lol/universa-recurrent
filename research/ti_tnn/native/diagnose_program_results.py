"""Run bounded answer diagnostics on completed native program-study checkpoints.

Teacher forcing uses the existing packed inputs and external answer masks. Saved
 generations are rescored, not regenerated. Changed-register accuracy conditions on
``target[register] != initial[register]`` and counts malformed completions as wrong;
it is a diagnostic complement, never a replacement for official exact match.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import evaluate as E
from . import evaluate_program_study as P
from .answer_analysis import (
    ANSWER_MASK,
    ANSWER_VALUE_MASK,
    FORMAT as MASK_FORMAT,
    FORMATTING_EOS_MASK,
    MASK_ARRAY_NAMES,
    PROGRAM_SPLITS,
    AnswerMaskedDataset,
    compare_answer_position_coverage,
)
from .data import IGNORE_INDEX, MODEL_INPUT_ARRAYS, sha256_file
from .structured import format_target

FORMAT = "native-program-answer-diagnostics-v1"
MANIFEST_FORMAT = "native-program-diagnostic-manifest-v1"
ARMS = ("dense", "untyped", "typed")
CE_GROUPS = (ANSWER_MASK, "numeric_answer_value", FORMATTING_EOS_MASK)
GROUP_MASKS = {
    ANSWER_MASK: ANSWER_MASK,
    "numeric_answer_value": ANSWER_VALUE_MASK,
    FORMATTING_EOS_MASK: FORMATTING_EOS_MASK,
}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _ce_metric(loss_sum: float, count: int) -> dict:
    return {
        "cross_entropy_sum": float(loss_sum),
        "token_count": int(count),
        "cross_entropy_mean": loss_sum / count if count else None,
    }


def _batch_length(batch: Mapping[str, torch.Tensor]) -> int:
    live = ~batch["key_padding_mask"].bool()
    if not bool(live.any()):
        raise ValueError("teacher-forced diagnostic encountered an all-padding batch")
    return int(torch.nonzero(live, as_tuple=False)[:, 1].max().item()) + 1


@torch.no_grad()
def teacher_forced_answer_cross_entropy(
    model,
    dataset: AnswerMaskedDataset,
    *,
    batch_size: int = 8,
    max_batches: int | None = None,
    device="cpu",
) -> dict:
    """Micro-average CE for all answer+EOS, numeric-value, and formatting/EOS labels."""
    device = torch.device(device)
    if device.type != "cpu":
        raise ValueError("bounded result diagnostics are CPU-only")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if max_batches is not None and (
        isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches <= 0
    ):
        raise ValueError("max_batches must be a positive integer or None")
    total_batches = math.ceil(len(dataset) / batch_size)
    batches_to_run = total_batches if max_batches is None else min(max_batches, total_batches)
    sums = {name: 0.0 for name in CE_GROUPS}
    counts = {name: 0 for name in CE_GROUPS}
    full_label_tokens = 0
    rows = 0
    was_training = model.training
    model.eval()
    try:
        for batch_index in range(batches_to_run):
            start = batch_index * batch_size
            indices = list(range(start, min(start + batch_size, len(dataset))))
            batch = dataset.batch(indices, as_torch=True, device=device)
            length = _batch_length(batch)
            inputs = {name: batch[name][:, :length] for name in MODEL_INPUT_ARRAYS}
            output = model(**inputs)
            shifted_logits = output.logits[:, :-1]
            shifted_labels = batch["labels"][:, 1:length]
            if shifted_logits.shape[:-1] != shifted_labels.shape:
                raise ValueError("packed labels and model logits have inconsistent shifted shapes")
            supervised = shifted_labels.ne(IGNORE_INDEX)
            full_label_tokens += int(supervised.sum())
            for group in CE_GROUPS:
                selected = supervised & batch[GROUP_MASKS[group]][:, 1:length].bool()
                count = int(selected.sum())
                if not count:
                    continue
                loss = F.cross_entropy(
                    shifted_logits[selected].float(), shifted_labels[selected].long(), reduction="sum"
                )
                sums[group] += float(loss)
                counts[group] += count
            rows += len(indices)
    finally:
        model.train(was_training)
    expected = {
        group: int(dataset.masks[GROUP_MASKS[group]].sum())
        for group in CE_GROUPS
    }
    complete = batches_to_run == total_batches
    if complete and counts != expected:
        raise RuntimeError(f"teacher-forced token counts {counts} differ from companion counts {expected}")
    if counts[ANSWER_MASK] != counts["numeric_answer_value"] + counts[FORMATTING_EOS_MASK]:
        raise RuntimeError("answer CE partitions do not sum to total answer token count")
    return {
        "teacher_forced": True,
        "shift": "logits[:, :-1] -> labels[:, 1:]; answer masks are shifted identically",
        "first_answer_token_included": True,
        "device": str(device),
        "batch_size": batch_size,
        "max_batches": max_batches,
        "batches": batches_to_run,
        "total_batches": total_batches,
        "rows_evaluated": rows,
        "rows_total": len(dataset),
        "complete": complete,
        "full_lm_label_tokens_evaluated": full_label_tokens,
        "expected_complete_counts": expected,
        "metrics": {group: _ce_metric(sums[group], counts[group]) for group in CE_GROUPS},
    }


def _final_operation(record: Mapping) -> str:
    saved = record.get("final_operation")
    program = record.get("program")
    if saved is None and isinstance(program, (list, tuple)) and program:
        saved = str(program[-1]).split()[0]
    if saved == "mul":
        saved = "multiply"
    if saved not in P.OPERATIONS:
        raise ValueError(f"{record.get('example_id')}: unsupported final operation {saved!r}")
    return saved


def _target_and_initial(record: Mapping) -> tuple[tuple[int, ...], tuple[int, ...]]:
    target = E.parse_final_registers(record.get("target_text"))
    initial = tuple(record.get("initial", ()))
    if target is None or len(initial) != 4 or any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 32 for value in initial
    ):
        raise ValueError(f"{record.get('example_id')}: canonical target and four modulo-32 initial registers are required")
    return target, initial


def _prediction_details(record: Mapping, prediction_text: str | None, saved_record: Mapping | None = None) -> dict:
    target, initial = _target_and_initial(record)
    predicted = E.parse_final_registers(prediction_text) if isinstance(prediction_text, str) else None
    exact = prediction_text == record["target_text"]
    changed = [target[index] != initial[index] for index in range(4)]
    correct = [predicted is not None and predicted[index] == target[index] for index in range(4)]
    detail = {
        "example_id": record["example_id"],
        "final_operation": _final_operation(record),
        "initial": list(initial),
        "target": list(target),
        "target_text": record["target_text"],
        "prediction_text": prediction_text,
        "predicted": None if predicted is None else list(predicted),
        "format_valid": predicted is not None,
        "full_exact": exact,
        "changed_registers": changed,
        "per_register_correct": correct,
        "changed_register_correct": [correct[index] for index in range(4) if changed[index]],
    }
    if not exact and saved_record is not None:
        detail["saved_failure_record"] = dict(saved_record)
    return detail


def _register_summary(details: Sequence[Mapping], *, include_failures: bool = True) -> dict:
    count = len(details)
    valid = sum(bool(item["format_valid"]) for item in details)
    per_register = {}
    changed_per_register = {}
    changed_valid_per_register = {}
    for register in range(4):
        correct = sum(bool(item["per_register_correct"][register]) for item in details)
        changed_items = [item for item in details if item["changed_registers"][register]]
        changed_correct = sum(bool(item["per_register_correct"][register]) for item in changed_items)
        changed_valid_items = [item for item in changed_items if item["format_valid"]]
        changed_valid_correct = sum(bool(item["per_register_correct"][register]) for item in changed_valid_items)
        per_register[str(register)] = {
            "correct": correct,
            "count": count,
            "accuracy": _rate(correct, count),
        }
        changed_per_register[str(register)] = {
            "correct": changed_correct,
            "count": len(changed_items),
            "accuracy": _rate(changed_correct, len(changed_items)),
        }
        changed_valid_per_register[str(register)] = {
            "correct": changed_valid_correct,
            "count": len(changed_valid_items),
            "accuracy": _rate(changed_valid_correct, len(changed_valid_items)),
        }
    overall_correct = sum(sum(map(bool, item["per_register_correct"])) for item in details)
    changed_opportunities = sum(sum(map(bool, item["changed_registers"])) for item in details)
    changed_correct = sum(
        bool(item["per_register_correct"][register])
        for item in details for register in range(4) if item["changed_registers"][register]
    )
    changed_valid_opportunities = sum(
        bool(item["changed_registers"][register])
        for item in details if item["format_valid"] for register in range(4)
    )
    changed_valid_correct = sum(
        bool(item["per_register_correct"][register])
        for item in details if item["format_valid"] for register in range(4)
        if item["changed_registers"][register]
    )
    result = {
        "examples": count,
        "full_exact": {
            "correct": sum(bool(item["full_exact"]) for item in details),
            "count": count,
            "accuracy": _rate(sum(bool(item["full_exact"]) for item in details), count),
        },
        "format_valid": {"count": valid, "rate": _rate(valid, count)},
        "per_register_overall": {
            "per_register": per_register,
            "micro": {"correct": overall_correct, "count": 4 * count, "accuracy": _rate(overall_correct, 4 * count)},
            "invalid_predictions_count_as_incorrect": True,
        },
        "changed_registers": {
            "per_register": changed_per_register,
            "micro": {
                "correct": changed_correct,
                "count": changed_opportunities,
                "accuracy": _rate(changed_correct, changed_opportunities),
            },
            "condition": "target[register] != initial[register]",
            "invalid_predictions_count_as_incorrect": True,
        },
        "changed_registers_format_valid_only": {
            "per_register": changed_valid_per_register,
            "micro": {
                "correct": changed_valid_correct,
                "count": changed_valid_opportunities,
                "accuracy": _rate(changed_valid_correct, changed_valid_opportunities),
            },
        },
        "examples_with_any_changed_register": sum(any(item["changed_registers"]) for item in details),
        "examples_with_no_changed_register": sum(not any(item["changed_registers"]) for item in details),
    }
    if include_failures:
        result["failure_records"] = [dict(item) for item in details if not item["full_exact"]]
        result["failure_count"] = len(result["failure_records"])
    return result


def register_prediction_metrics(
    records: Sequence[Mapping], predictions: Sequence[str | None], *, source: str
) -> dict:
    """Score exact text and overall/changed registers on an explicit existing population."""
    if len(records) != len(predictions):
        raise ValueError("records and predictions have different lengths")
    details = [
        _prediction_details(record, prediction)
        for record, prediction in zip(records, predictions)
    ]
    result = {
        "source": source,
        **_register_summary(details),
        "by_final_operation": {
            operation: _register_summary(
                [item for item in details if item["final_operation"] == operation], include_failures=False
            )
            for operation in P.OPERATIONS
        },
    }
    return result


def no_op_metrics(records: Sequence[Mapping]) -> dict:
    """Return the canonical initial registers without executing the supplied program."""
    return register_prediction_metrics(
        records,
        [format_target(_target_and_initial(record)[1]) for record in records],
        source="no-op canonical serialization of the four initial registers",
    )


def saved_generation_metrics(records: Sequence[Mapping], saved_records: Sequence[Mapping]) -> dict:
    """Rescore saved completions with no model calls and retain every failing original record."""
    by_id = {record.get("example_id"): record for record in saved_records}
    if len(by_id) != len(saved_records):
        raise ValueError("saved generation records contain duplicate example IDs")
    expected_ids = [record["example_id"] for record in records]
    if set(by_id) != set(expected_ids):
        raise ValueError("saved generation IDs differ from the selected source records")
    details = []
    for record in records:
        saved = by_id[record["example_id"]]
        if saved.get("target_text") != record["target_text"]:
            raise ValueError(f"{record['example_id']}: saved target differs from the source record")
        if saved.get("final_operation") not in (None, _final_operation(record)):
            raise ValueError(f"{record['example_id']}: saved final operation differs from the source record")
        details.append(_prediction_details(record, saved.get("completion_text"), saved))
    return {
        "source": "existing evaluation.json greedy generation records; no new decoding",
        **_register_summary(details),
        "by_final_operation": {
            operation: _register_summary(
                [item for item in details if item["final_operation"] == operation], include_failures=False
            )
            for operation in P.OPERATIONS
        },
        "saved_failure_records_preserved": True,
    }


def _selected_sources(sources: Mapping[str, list[dict]], evaluation: Mapping) -> dict[str, list[dict]]:
    selected = {}
    for split in PROGRAM_SPLITS:
        values = evaluation["splits"][split]
        generation = values["generation"]
        ids = values["selection"]["evaluation_ids"]
        by_id = {record["example_id"]: record for record in sources[split]}
        records = [by_id[example_id] for example_id in ids]
        if not generation.get("complete") or generation.get("evaluated") != len(records):
            raise ValueError(f"{split}: official saved generation is incomplete")
        selected[split] = records
    return selected


def _baseline_reports(sources: Mapping[str, list[dict]], selected: Mapping[str, list[dict]]) -> dict:
    selected_report = E.structured_baseline_report(selected)
    full_train_report = E.structured_baseline_report({**selected, "train": sources["train"]})
    return {
        "exact_lookup": {
            split: selected_report["exact_lookup"][split] if split == "train" else full_train_report["exact_lookup"][split]
            for split in PROGRAM_SPLITS
        },
        "supplied_program_executor": selected_report["supplied_program_executor"],
        "lookup_fit_examples": len(sources["train"]),
        "population": "the same official 128 selected source records per split",
        "boundary": "lookup is memorization; the supplied-program executor is separate and is not the neural model",
    }


def _validate_companion_manifest(mask_directory: Path, data_manifest_sha256: str) -> dict:
    path = mask_directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("format") != MASK_FORMAT:
        raise ValueError("unexpected companion mask manifest format")
    if manifest["source_manifest"]["sha256"] != data_manifest_sha256:
        raise ValueError("companion masks belong to a different source data manifest")
    for split in PROGRAM_SPLITS:
        for name in MASK_ARRAY_NAMES:
            entry = manifest["splits"][split]["arrays"][name]
            path = Path(entry["path"])
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"{split}/{name}: companion array hash differs from its manifest")
    return manifest


def _write_new_json(path: Path, payload: Mapping) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def diagnose(
    run_directory,
    data_root,
    mask_directory=None,
    *,
    batch_size: int = 8,
    max_batches_per_split: int | None = None,
    child_mappings: Mapping[str, Mapping] | None = None,
    manifest_output=None,
) -> dict:
    """Diagnose final checkpoints without selecting or rewriting their source artifacts.

    The default discovers the original three-arm run. Explicit child mappings instead
    accept run-ID keyed mode/train-result/evaluation/output paths for continuations.
    """
    run = Path(run_directory).resolve()
    root = Path(data_root).resolve()
    mask_root = Path(mask_directory).resolve() if mask_directory is not None else root / "answer_masks"
    explicit_children = child_mappings is not None
    if explicit_children:
        if not isinstance(child_mappings, Mapping) or not child_mappings:
            raise ValueError("child_mappings must be a nonempty run-ID keyed mapping")
        arms = tuple(child_mappings)
        output_paths = {}
        for run_id, mapping in child_mappings.items():
            if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
                raise ValueError("explicit child run IDs must be nonempty path components")
            if not isinstance(mapping, Mapping) or mapping.get("mode") not in ARMS:
                raise ValueError(f"{run_id}: explicit child mapping requires a valid mode")
            if mapping.get("output") is None:
                raise ValueError(f"{run_id}: explicit child mapping requires an output path")
            output_paths[run_id] = Path(mapping["output"]).resolve()
    else:
        arms = ARMS
        output_paths = {arm: run / "arms" / arm / "answer_diagnostics.json" for arm in ARMS}
    manifest_output = Path(manifest_output).resolve() if manifest_output is not None else run / "diagnostic_manifest.json"
    if len(set(output_paths.values())) != len(output_paths) or manifest_output in output_paths.values():
        raise ValueError("explicit answer-diagnostic outputs must be distinct")
    for path in [*output_paths.values(), manifest_output]:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing diagnostic output: {path}")
    launch = json.loads((run / "launch.json").read_text())
    if explicit_children:
        if not set(arms) <= set(launch.get("run_ids", [])):
            raise ValueError("explicit child mappings are not declared by the continuation launch")
    else:
        status = json.loads((run / "run_status.json").read_text())
        if not status.get("complete") or status.get("failure") is not None:
            raise ValueError("result diagnostics require a completed parent run with no run-level failure")
    launch_data = launch["data"] if isinstance(launch["data"], str) else launch["data"]["root"]
    if Path(launch_data).resolve() != root:
        raise ValueError("data root differs from the parent launch")
    data_manifest_path = root / "manifest.json"
    data_manifest_sha256 = sha256_file(data_manifest_path)
    launch_manifest_sha256 = launch["data_manifest_sha256"] if "data_manifest_sha256" in launch else launch["data"]["manifest_sha256"]
    if data_manifest_sha256 != launch_manifest_sha256:
        raise ValueError("data manifest differs from the parent launch")
    companion = _validate_companion_manifest(mask_root, data_manifest_sha256)
    sources, source_files = P._load_sources(root)
    datasets = {split: AnswerMaskedDataset(root, split, mask_root) for split in PROGRAM_SPLITS}
    coverage = compare_answer_position_coverage(datasets["train"], datasets["longer_length"])
    if not coverage["training"]["route_state_targets_exclude_every_answer_position"] or not coverage[
        "longer_program"
    ]["route_state_targets_exclude_every_answer_position"]:
        raise RuntimeError("route/state targets unexpectedly cover answer positions")
    arm_inputs = {}
    selected_by_arm = {}
    for arm in arms:
        if explicit_children:
            mapping = child_mappings[arm]
            train_result_path = Path(mapping["train_result"]).resolve()
            evaluation_path = Path(mapping["evaluation"]).resolve()
            expected_mode = mapping["mode"]
        else:
            train_result_path = run / "arms" / arm / "train-result.json"
            evaluation_path = run / "arms" / arm / "evaluation.json"
            expected_mode = arm
        train_result = json.loads(train_result_path.read_text())
        descriptor = train_result["final_checkpoint"]
        checkpoint_path = Path(descriptor["path"]).resolve()
        expected_sha256 = descriptor["sha256"]
        if train_result.get("mode") != expected_mode or descriptor.get("step") != train_result.get("step"):
            raise ValueError(f"{arm}: train-result mode or final step is inconsistent")
        if sha256_file(checkpoint_path) != expected_sha256:
            raise ValueError(f"{arm}: final checkpoint differs from train-result.json")
        evaluation = json.loads(evaluation_path.read_text())
        if evaluation["checkpoint"]["sha256"] != expected_sha256:
            raise ValueError(f"{arm}: official evaluation used a different checkpoint")
        selected_by_arm[arm] = _selected_sources(sources, evaluation)
        arm_inputs[arm] = {
            "mode": expected_mode,
            "train_result_path": train_result_path,
            "train_result_sha256": sha256_file(train_result_path),
            "evaluation_path": evaluation_path,
            "evaluation_sha256": sha256_file(evaluation_path),
            "checkpoint_path": checkpoint_path,
            "expected_sha256": expected_sha256,
            "descriptor": descriptor,
            "evaluation": evaluation,
        }
    shared_baselines = _baseline_reports(sources, selected_by_arm[arms[0]])
    shared_no_op = {
        split: no_op_metrics(selected_by_arm[arms[0]][split])
        for split in PROGRAM_SPLITS
    }
    arm_summaries = {}
    for arm in arms:
        inputs = arm_inputs[arm]
        model, checkpoint = E.load_native_model(
            inputs["checkpoint_path"], expected_sha256=inputs["expected_sha256"], device="cpu"
        )
        if model.config.mode != inputs["mode"] or model.training or next(model.parameters()).device.type != "cpu":
            raise RuntimeError(f"{arm}: expected an eval-mode CPU checkpoint for the matching mode")
        teacher = {}
        for split in PROGRAM_SPLITS:
            teacher[split] = teacher_forced_answer_cross_entropy(
                model,
                datasets[split],
                batch_size=batch_size,
                max_batches=max_batches_per_split,
                device="cpu",
            )
            print(json.dumps({
                "arm": arm,
                "split": split,
                "teacher_forced_complete": teacher[split]["complete"],
                "answer_eos": teacher[split]["metrics"][ANSWER_MASK],
                "numeric_answer_value": teacher[split]["metrics"]["numeric_answer_value"],
                "formatting_eos": teacher[split]["metrics"][FORMATTING_EOS_MASK],
            }, sort_keys=True), flush=True)
        saved = {
            split: saved_generation_metrics(
                selected_by_arm[arm][split], inputs["evaluation"]["splits"][split]["generation"]["records"]
            )
            for split in PROGRAM_SPLITS
        }
        checkpoint_after = sha256_file(inputs["checkpoint_path"])
        if checkpoint_after != inputs["expected_sha256"]:
            raise RuntimeError(f"{arm}: checkpoint bytes changed during diagnosis")
        if sha256_file(inputs["evaluation_path"]) != inputs["evaluation_sha256"]:
            raise RuntimeError(f"{arm}: official evaluation bytes changed during diagnosis")
        result = {
            "format": FORMAT,
            "status": "bounded_diagnostic_not_evidence_final",
            "evidence_final": False,
            "official_metric_replaced": False,
            "arm": arm,
            "device": "cpu",
            "training_performed": False,
            "new_generation_performed": False,
            "checkpoint": {
                "path": str(inputs["checkpoint_path"]),
                "expected_sha256": inputs["expected_sha256"],
                "sha256": checkpoint["sha256"],
                "sha256_after": checkpoint_after,
                "step": checkpoint["step"],
                "source": "final_checkpoint in this arm's train-result.json",
            },
            "train_result": {
                "path": str(inputs["train_result_path"]),
                "sha256": inputs["train_result_sha256"],
            },
            "official_evaluation_source": {
                "path": str(inputs["evaluation_path"]),
                "sha256": inputs["evaluation_sha256"],
                "rewritten": False,
            },
            "data": {
                "root": str(root),
                "manifest_sha256": data_manifest_sha256,
                "structured_sources": source_files,
                "companion_manifest": str(mask_root / "manifest.json"),
                "companion_manifest_sha256": sha256_file(mask_root / "manifest.json"),
            },
            "teacher_forced": {
                "splits": teacher,
                "complete": all(values["complete"] for values in teacher.values()),
                "loss_labels_only": True,
                "prompt_visibility": "complete packed model inputs retained; masks are never forward-pass arguments",
            },
            "no_op_baseline": {
                "splits": shared_no_op,
                "population": "the same official 128 selected source records per split",
                "operation": "return initial registers in canonical target format; do not execute the program",
            },
            "saved_generation_changed_registers": {
                "splits": saved,
                "invalid_predictions_count_as_incorrect": True,
                "all_failure_records_preserved": True,
            },
            "answer_position_coverage": coverage,
            "baselines": shared_baselines,
            "limits": [
                "Teacher-forced CE is not free-generation accuracy.",
                "Changed-register accuracy excludes unchanged target/initial pairs from its denominator and is not official exact match.",
                "Malformed generations have no parsed values and count as incorrect in overall and changed-register denominators.",
                "These bounded diagnostics do not select checkpoints, alter sealed evidence, or establish a continuation training result.",
            ],
        }
        _write_new_json(output_paths[arm], result)
        arm_summaries[arm] = {
            "answer_diagnostics": {
                "path": str(output_paths[arm]),
                "sha256": sha256_file(output_paths[arm]),
            },
            "train_result": result["train_result"],
            "official_evaluation_source": result["official_evaluation_source"],
            "checkpoint": result["checkpoint"],
            "teacher_forced_complete": result["teacher_forced"]["complete"],
            "answer_eos_ce": {
                split: values["metrics"][ANSWER_MASK]["cross_entropy_mean"]
                for split, values in teacher.items()
            },
            "numeric_answer_value_ce": {
                split: values["metrics"]["numeric_answer_value"]["cross_entropy_mean"]
                for split, values in teacher.items()
            },
            "formatting_eos_ce": {
                split: values["metrics"][FORMATTING_EOS_MASK]["cross_entropy_mean"]
                for split, values in teacher.items()
            },
            "no_op_changed_register_micro_accuracy": {
                split: values["changed_registers"]["micro"]["accuracy"]
                for split, values in shared_no_op.items()
            },
            "saved_changed_register_micro_accuracy": {
                split: values["changed_registers"]["micro"]["accuracy"]
                for split, values in saved.items()
            },
        }
        del model, checkpoint
        gc.collect()
    complete = all(values["teacher_forced_complete"] for values in arm_summaries.values())
    manifest = {
        "format": MANIFEST_FORMAT,
        "status": "complete_bounded_diagnostic_not_evidence_final" if complete else "partial_bounded_diagnostic_not_evidence_final",
        "evidence_final": False,
        "run_directory": str(run),
        "data_root": str(root),
        "data_manifest_sha256": data_manifest_sha256,
        "companion_manifest": {
            "path": str(mask_root / "manifest.json"),
            "sha256": sha256_file(mask_root / "manifest.json"),
            "source_tokenizer_sha256": companion["tokenizer"]["sha256"],
        },
        "code": {
            "answer_analysis.py": sha256_file(Path(__file__).with_name("answer_analysis.py")),
            "diagnose_program_results.py": sha256_file(Path(__file__)),
        },
        "device": "cpu",
        "batch_size": batch_size,
        "max_batches_per_split": max_batches_per_split,
        "new_training_or_generation": False,
        "checkpoint_selection": "none; exactly each arm's existing train-result.json final_checkpoint",
        "arms": arm_summaries,
        "answer_position_coverage": coverage,
        "baselines_kept_separate": ["no_op", "exact_lookup", "supplied_program_executor"],
    }
    _write_new_json(manifest_output, manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m native.diagnose_program_results", description=__doc__.split("\n\n")[0])
    result.add_argument("--run", required=True, help="completed three-arm program-study run directory")
    result.add_argument("--data", required=True, help="original five-split packed data root")
    result.add_argument("--mask-directory", help="companion root; defaults to DATA/answer_masks")
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--max-batches-per-split", type=int, help="explicit partial diagnostic cap; default is all fixed packed rows")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    result = diagnose(
        args.run,
        args.data,
        args.mask_directory,
        batch_size=args.batch_size,
        max_batches_per_split=args.max_batches_per_split,
    )
    print(json.dumps({
        "output": str(Path(args.run).resolve() / "diagnostic_manifest.json"),
        "status": result["status"],
        "arms": result["arms"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
