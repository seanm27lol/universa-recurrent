"""Self-contained continuation export, reproduction, and failure-preservation checks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from native.independent import rescore_generations
from native.independent.export_continuation import FORBIDDEN_SUFFIXES, SPLITS, export
from native.structured import Instruction, execute, format_prompt


A_ID = "78-dense-full_sequence"
B_ID = "78-dense-answer_eos"
FAILED_ID = "78-untyped-full_sequence"


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_record(split, supplemental=False):
    programs = {
        "train": [Instruction("add", 0, src=1)] if supplemental else [Instruction("set", 0, value=1)],
        "dev": [Instruction("copy", 0, src=1)],
        "initial_values": [Instruction("set", 0, value=31)],
        "unseen_composition": [Instruction("multiply", 0, src=1), Instruction("add", 2, src=3)],
        "longer_length": [Instruction("set", 1, value=2), Instruction("add", 0, src=1)],
    }
    initial = (24, 25, 26, 27) if split == "initial_values" else (2, 3, 4, 5)
    if supplemental:
        initial = (6, 7, 8, 9)
    instructions = programs[split]
    program = [str(instruction) for instruction in instructions]
    target = execute(instructions, initial).target_text
    prompt = format_prompt(instructions, initial)
    identity = hashlib.sha256((prompt + target).encode()).hexdigest()[:20]
    return {
        "example_id": f"structured-{split}-{identity}",
        "split": split,
        "prompt": prompt,
        "target_text": target,
        "program": program,
        "initial": list(initial),
        "modulus": 32,
    }


def registers(text):
    return [int(part.split("=")[1]) for part in text.split()]


def saved(record, token, *, correct=True, intervened=False):
    target = registers(record["target_text"])
    prediction = target if correct else [(target[0] + 1) % 32, *target[1:]]
    completion = " ".join(f"r{index}={value}" for index, value in enumerate(prediction))
    return {
        "example_id": record["example_id"],
        "completion_text": completion,
        "target_text": record["target_text"],
        "completion_ids": [token],
        "generated_ids": [token],
        "correct": correct,
        "valid": True,
        "stopped_on_eos": not intervened,
        "truncated": intervened,
        "prompt_truncated": False,
    }


def changed_metrics(record, generation, *, no_op=False):
    initial = record["initial"]
    target = registers(record["target_text"])
    prediction = initial if no_op else registers(generation["completion_text"])
    changed = [left != right for left, right in zip(target, initial)]
    correct = [left == right for left, right in zip(prediction, target)]
    changed_correct = sum(value for value, selected in zip(correct, changed) if selected)
    changed_count = sum(changed)
    per_register = {
        str(index): {
            "correct": int(correct[index]) if changed[index] else 0,
            "count": int(changed[index]),
            "accuracy": (int(correct[index]) if changed[index] else 0) / changed[index] if changed[index] else None,
        }
        for index in range(4)
    }
    exact = prediction == target
    failure = None if exact else {
        "example_id": record["example_id"],
        "predicted": prediction,
        "target": target,
        "changed_registers": changed,
        "changed_register_correct": [value for value, selected in zip(correct, changed) if selected],
    }
    operation = record["program"][-1].split(" ", 1)[0]
    values = {
        "examples": 1,
        "examples_with_any_changed_register": int(any(changed)),
        "examples_with_no_changed_register": int(not any(changed)),
        "failure_count": int(not exact),
        "failure_records": [failure] if failure else [],
        "format_valid": {"count": 1, "rate": 1.0},
        "full_exact": {"correct": int(exact), "count": 1, "accuracy": float(exact)},
        "changed_registers": {
            "condition": "target[register] != initial[register]",
            "invalid_predictions_count_as_incorrect": True,
            "micro": {
                "correct": changed_correct,
                "count": changed_count,
                "accuracy": changed_correct / changed_count if changed_count else None,
            },
            "per_register": per_register,
        },
        "changed_registers_format_valid_only": {
            "micro": {
                "correct": changed_correct,
                "count": changed_count,
                "accuracy": changed_correct / changed_count if changed_count else None,
            },
            "per_register": per_register,
        },
        "per_register_overall": {
            "invalid_predictions_count_as_incorrect": True,
            "micro": {"correct": sum(correct), "count": 4, "accuracy": sum(correct) / 4},
            "per_register": {
                str(index): {"correct": int(correct[index]), "count": 1, "accuracy": float(correct[index])}
                for index in range(4)
            },
        },
        "source": "synthetic saved generation" if not no_op else "synthetic no-op baseline",
    }
    values["by_final_operation"] = {
        operation: {key: value for key, value in values.items() if key not in {"by_final_operation", "failure_records"}}
    }
    if not no_op:
        values["saved_failure_records_preserved"] = True
    return values


def teacher_values(split, primary, *, answer_b):
    factor = 2.0 if answer_b else 1.0
    answer_sum = 1.7 * factor
    numeric_sum = 0.8 * factor
    formatting_sum = 0.9 * factor
    answer_ce = {
        "complete": True,
        "examples": 1,
        "evaluated": 1,
        "source_examples": 2 if split == "train" else 1,
        "count": 17,
        "sum": answer_sum,
        "mean": answer_sum / 17,
        "records": [{"example_id": primary["example_id"], "count": 17, "sum": answer_sum}],
    }
    state = {
        "format": "native-state-path-v1",
        "complete": True,
        "packed_rows": 1,
        "evaluated_rows": 1,
        "batches": 1,
        "counts": {"state_targets": 4},
        "metrics": {"pass_0": {"correct": 2, "count": 4, "accuracy": 0.5}},
        "state_schedule": "before,after,after,after",
        "pass_2_contract": "fixture contract",
        "document_position_validation": {"valid": True},
        "answer_loss_distinction": "CE is not register error",
        "evidence_boundary": "readout is not causal semantic use",
        "answer_token_cross_entropy": {split: answer_ce},
    }
    teacher = {
        "complete": True,
        "rows_evaluated": 1,
        "rows_total": 1,
        "batches": 1,
        "full_lm_label_tokens_evaluated": 20,
        "metrics": {
            "answer_mask": {
                "token_count": 17, "cross_entropy_sum": answer_sum, "cross_entropy_mean": answer_sum / 17,
            },
            "numeric_answer_value": {
                "token_count": 4, "cross_entropy_sum": numeric_sum, "cross_entropy_mean": numeric_sum / 4,
            },
            "formatting_eos_mask": {
                "token_count": 13, "cross_entropy_sum": formatting_sum, "cross_entropy_mean": formatting_sum / 13,
            },
        },
    }
    return state, teacher


def make_completed_child(run, data, mask_manifest, records, supplemental, run_id, mode, loss_mode, correctness):
    child = run / run_id
    answer_b = loss_mode == "answer_eos"
    parent_sha = ("a" if mode == "dense" else "b") * 64
    fork_sha = ("c" if not answer_b else "d") * 64
    final_sha = ("e" if not answer_b else "f") * 64
    final_name = "final-update-00000001.pt"
    final_manifest = child / "checkpoints" / f"{final_name}.{final_sha}.manifest.json"
    fork_manifest_path = child / "checkpoints" / f"update-00000000.pt.{fork_sha}.manifest.json"
    write(fork_manifest_path, {
        "format": "native-checkpoint-v1", "checkpoint_name": "update-00000000.pt",
        "step": 0, "sha256": fork_sha, "bytes": 100, "consumed_loss_tokens": 4,
        "state_hashes": {"model": "parent-model"},
    })
    write(final_manifest, {
        "format": "native-checkpoint-v1", "checkpoint_name": final_name,
        "step": 1, "sha256": final_sha, "bytes": 200, "consumed_loss_tokens": 10,
        "state_hashes": {"model": "child-model"},
    })
    parent = {
        "checkpoint_path": str(run / "parent.pt"), "checkpoint_sha256": parent_sha,
        "code_identity": "parent-code", "config_identity": "parent-config",
        "model_identity": "parent-model", "optimizer_identity": "parent-optimizer",
        "step": 0,
    }
    write(child / "run_config.json", {
        "format": "native-train-v1", "run_id": run_id, "reservation_id": "reservation-fixture",
        "model": {"mode": mode, "n_passes": 4},
        "code_identity": "code-fixture",
        "fork": {
            "parent": parent, "parent_exposure_tokens": 4, "target_epochs": 1,
            "target_updates": 1, "target_exposure_tokens": 10,
            "loss_mode": loss_mode,
        },
        "training": {"loss_mode": loss_mode, "planned_updates": 1},
    })
    final_checkpoint = {
        "path": str(child / "checkpoints" / final_name),
        "manifest_path": str(final_manifest), "step": 1, "sha256": final_sha,
        "bytes": 200, "consumed_loss_tokens": 10, "state_hashes": {"model": "child-model"},
    }
    additional_main = 2 if answer_b else 6
    write(child / "train-result.json", {
        "format": "native-train-v1", "step": 1, "mode": mode, "loss_mode": loss_mode,
        "new_updates": 1, "processed_updates_this_invocation": 1,
        "consumed_exposure_tokens": 10, "consumed_loss_tokens": 10,
        "processed_exposure_tokens_this_invocation": 6,
        "processed_main_loss_tokens_this_invocation": additional_main,
        "final_checkpoint": final_checkpoint,
        "training_seconds": 1.0, "data_loading_seconds": 0.1, "forward_seconds": 0.2,
        "backward_seconds": 0.3, "optimizer_seconds": 0.2, "checkpoint_seconds": 0.1,
        "actual_exposure_tokens_per_second": 6.0,
        "actual_main_loss_tokens_per_second": float(additional_main),
        "cuda_peak_allocated_bytes": 1000, "cuda_peak_reserved_bytes": 2000,
        "dev": {
            "loss": 0.5, "objective_loss": 0.7, "route_usage": [0, 1],
            "route_targets": 2, "route_loss": 0.1, "balance_decisions": 2, "balance_loss": 0.1,
            "cuda_peak_allocated_bytes": 1000, "checkpoint": final_checkpoint,
        },
    })
    (child / "learning_curve.jsonl").write_text("".join(json.dumps(event) + "\n" for event in [
        {"event": "train", "step": 0, "loss": 2.0},
        {"event": "train", "step": 1, "loss": 1.0},
        {"event": "dev", "step": 1, "loss": 0.5},
        {"event": "checkpoint", "step": 1, "loss": 0.5},
    ]))
    (child / "checkpoint_journal.jsonl").write_text("".join(json.dumps(event) + "\n" for event in [
        {"event": "checkpoint", "step": 0, "sha256": fork_sha},
        {"event": "batch", "step": 1, "loss_tokens": additional_main},
        {"event": "checkpoint", "step": 1, "sha256": final_sha},
    ]))
    (child / "recovery.jsonl").write_text(json.dumps({"event": "recovery", "invalid_checkpoints": []}) + "\n")

    splits = {}
    teachers = {}
    saved_changed = {}
    no_ops = {}
    for index, split in enumerate(SPLITS, 1):
        primary = saved(records[split], index, correct=correctness[split])
        supplemental_records = [saved(supplemental, 10, correct=not answer_b)] if split == "train" else []
        target = supplemental if split == "train" else records[split]
        intervened = saved(
            target,
            20 if split == "train" else index,
            correct=False if split == "train" else correctness[split],
            intervened=split == "train",
        )
        changed = split == "train"
        intervention = {
            "example_id": target["example_id"],
            "baseline": {"correct": True, "valid": True},
            "intervened": intervened,
            "changed_completion": changed,
            "changed_generated_suffix": changed,
        }
        state, teacher = teacher_values(split, primary, answer_b=answer_b)
        teachers[split] = teacher
        saved_changed[split] = changed_metrics(records[split], primary)
        no_ops[split] = changed_metrics(records[split], primary, no_op=True)
        write(child / f"{split}_state_path.json", state)
        splits[split] = {
            "scope": "fixture split",
            "selection": {
                "examples": 1,
                "interventions": {
                    "examples": 1,
                    "reused_evaluation_baselines": 0 if split == "train" else 1,
                    "additional_ordinary_baselines": 1 if split == "train" else 0,
                    "per_final_operation": {target["program"][-1].split(" ", 1)[0]: 1},
                },
            },
            "generation": {"complete": True, "examples": 1, "evaluated": 1, "records": [primary]},
            "interventions": {
                "selection_examples": 1,
                "supplemental_baseline_generation": {
                    "complete": bool(supplemental_records), "examples": len(supplemental_records),
                    "evaluated": len(supplemental_records), "records": supplemental_records,
                },
                "records": [intervention],
                "changed_completions": int(changed), "changed_completion_rate": float(changed),
                "changed_generated_suffixes": int(changed), "changed_generated_suffix_rate": float(changed),
                "targeted_exactly_once": 1, "four_writer_updates_suppressed_per_example": 1,
                "evidence_boundary": "sensitivity is not semantic faithfulness",
            },
            "state_path": state,
        }
    write(child / "evaluation.json", {
        "format": "native-program-study-evaluation-v1",
        "checkpoint": {"sha256": final_sha, "step": 1, "config": {"model": {"mode": mode}}},
        "splits": splits,
        "baselines": {
            "exact_lookup": {split: {"examples": 1, "correct": int(split == "train")} for split in SPLITS},
            "supplied_program_executor": {split: {"examples": 1, "correct": 1} for split in SPLITS},
        },
    })
    write(child / "reconstruction.json", {
        "format": "native-reconstruction-report-v1", "agreement": True,
        "execution": {"device": "cpu", "parameter_dtype": "torch.float32", "autocast_dtype": None},
        "checkpoint": {"sha256": final_sha, "step": 1, "path": "original-weight-path"},
        "teacher_forced": {
            "agrees": True,
            "agreement_policy": "relative RMS scale policy",
            "strict_allclose_agrees": False,
            "comparisons": {
                "logits": {
                    "agrees": False, "exact": False, "scale_agrees": True,
                    "maximum_absolute_difference": 0.25, "different_values": 3,
                    "first_divergence": {"absolute_difference": 0.125, "index": [0, 0, 0]},
                    "first_tolerance_violation": {"absolute_difference": 0.25, "index": [0, 1, 2]},
                    "atol": 0.01, "rtol": 0.01, "reference_rms": 2.0,
                },
                "h": {
                    "agrees": True, "exact": False, "scale_agrees": True,
                    "maximum_absolute_difference": 0.001, "different_values": 1,
                    "first_divergence": {"absolute_difference": 0.001, "index": [0, 0, 0, 0]},
                    "first_tolerance_violation": None, "atol": 0.01, "rtol": 0.01,
                },
                "route_logits": None,
            },
        },
        "generation": {"agrees": True},
    })
    write(child / "answer_diagnostics.json", {
        "format": "native-program-answer-diagnostics-v1", "arm": run_id,
        "status": "bounded_diagnostic_not_evidence_final", "evidence_final": False,
        "teacher_forced": {"complete": True, "splits": teachers},
        "saved_generation_changed_registers": {
            "all_failure_records_preserved": True, "splits": saved_changed,
        },
        "no_op_baseline": {
            "operation": "return initial registers", "population": "same selected source records",
            "splits": no_ops,
        },
        "answer_position_coverage": {"ordered_exposure_not_masked_token_matching": True},
        "official_evaluation_source": {"rewritten": False, "sha256": sha256(child / "evaluation.json")},
        "data": {
            "companion_manifest": str(mask_manifest),
            "companion_manifest_sha256": sha256(mask_manifest),
        },
    })
    write(child / "answer_diagnostic_manifest.json", {
        "format": "native-program-diagnostic-manifest-v1",
        "status": "complete_bounded_diagnostic_not_evidence_final",
        "evidence_final": False,
        "companion_manifest": {"path": str(mask_manifest), "sha256": sha256(mask_manifest)},
        "arms": {run_id: {"answer_diagnostics": {"sha256": sha256(child / "answer_diagnostics.json")}}},
    })
    rescore_generations.rescore(
        run, data, child / "generation_diagnostics.json", evaluation_paths={mode: child / "evaluation.json"}
    )
    exact = {split: int(correctness[split]) for split in SPLITS}
    write(child / "status.json", {
        "status": "completed", "updates": 1, "new_updates": 1,
        "consumed_exposure_tokens": 10, "additional_exposure_tokens": 6,
        "required_updates": 1, "required_exposure_tokens": 10,
        "exact": exact, "wall_seconds": 2.0, "train_seconds": 1.0,
        "measured_stage_seconds": {"official_evaluation": 0.5, "answer_diagnostics": 0.25},
        "train_result_sha256": sha256(child / "train-result.json"),
        "evaluation_sha256": sha256(child / "evaluation.json"),
        "generation_diagnostics_sha256": sha256(child / "generation_diagnostics.json"),
        "reconstruction_sha256": sha256(child / "reconstruction.json"),
        "answer_diagnostics_sha256": sha256(child / "answer_diagnostics.json"),
        "answer_diagnostic_manifest_sha256": sha256(child / "answer_diagnostic_manifest.json"),
        "fork_to_final_journal_sha256": sha256(child / "checkpoint_journal.jsonl"),
    })
    return {
        "run_id": run_id,
        "fork_sha256": fork_sha,
        "final_sha256": final_sha,
        "parent": parent,
        "fork_manifest": {
            "initial_checkpoint": {
                "path": str(child / "checkpoints" / "update-00000000.pt"),
                "manifest_path": str(fork_manifest_path), "step": 0,
                "sha256": fork_sha, "bytes": 100,
            },
        },
    }


def make_study(tmp_path):
    run = tmp_path / "continuation-run"
    data = tmp_path / "continuation-data"
    records = {split: source_record(split) for split in SPLITS}
    supplemental = source_record("train", supplemental=True)
    for split in SPLITS:
        split_records = [records[split]] + ([supplemental] if split == "train" else [])
        (data / "structured").mkdir(parents=True, exist_ok=True)
        (data / "structured" / f"{split}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in split_records)
        )
    write(data / "structured" / "baselines.json", {"boundary": "fixture construction baseline"})
    write(data / "manifest.json", {
        "format": "native-varied-program-study-data-v1",
        "counts": {split: 2 if split == "train" else 1 for split in SPLITS},
        "source_identity": {"external_corpora_used": False},
    })
    write(data / "audit.json", {"format": "fixture-audit", "targets_validated": True})
    write(data / "selections.json", {
        "evaluation_example_ids": {split: [records[split]["example_id"]] for split in SPLITS},
        "intervention_example_ids": {
            split: [(supplemental if split == "train" else records[split])["example_id"]]
            for split in SPLITS
        },
    })
    mask_manifest = tmp_path / "masks" / "manifest.json"
    write(mask_manifest, {
        "format": "native-program-answer-mask-companions-v1",
        "mask_arrays": ["answer_mask", "answer_value_mask", "formatting_eos_mask"],
        "splits": {split: {"arrays": {"answer_mask": {"sha256": "mask-fixture", "shape": [1, 8]}}} for split in SPLITS},
    })

    correctness_a = {"train": 1, "dev": 0, "initial_values": 0, "unseen_composition": 0, "longer_length": 0}
    correctness_b = {"train": 1, "dev": 1, "initial_values": 0, "unseen_composition": 0, "longer_length": 0}
    a = make_completed_child(run, data, mask_manifest, records, supplemental, A_ID, "dense", "full_sequence", correctness_a)
    b = make_completed_child(run, data, mask_manifest, records, supplemental, B_ID, "dense", "answer_eos", correctness_b)
    failed_status = {
        "status": "failed", "stage": "official_evaluation", "mode": "untyped",
        "loss_mode": "full_sequence", "error": "synthetic failure", "traceback": "fixture traceback",
    }
    write(run / FAILED_ID / "status.json", failed_status)
    write(run / FAILED_ID / "run_config.json", {
        "model": {"mode": "untyped"}, "fork": {"parent": {"checkpoint_sha256": "b" * 64}},
    })
    (run / FAILED_ID / "recovery.jsonl").write_text(json.dumps({"event": "failure", "error": "synthetic failure"}) + "\n")

    children = [
        {
            "run_id": A_ID, "child_id": "dense-A", "pair_id": "dense", "arm": "A",
            "mode": "dense", "loss_mode": "full_sequence", "additional_main_loss_tokens": 6,
            "target": {"epochs": 1, "updates": 1, "exposure_tokens": 10},
        },
        {
            "run_id": B_ID, "child_id": "dense-B", "pair_id": "dense", "arm": "B",
            "mode": "dense", "loss_mode": "answer_eos", "additional_main_loss_tokens": 2,
            "target": {"epochs": 1, "updates": 1, "exposure_tokens": 10},
        },
        {
            "run_id": FAILED_ID, "child_id": "untyped-A", "pair_id": "untyped", "arm": "A",
            "mode": "untyped", "loss_mode": "full_sequence", "additional_main_loss_tokens": 6,
            "target": {"epochs": 1, "updates": 1, "exposure_tokens": 10},
        },
    ]
    protocol = {
        "format": "native-six-child-continuation-protocol-v1",
        "continuation_study_id": "78-native-continuation",
        "children": children,
        "run_ids": [child["run_id"] for child in children],
        "endpoint": {
            "target": {"epochs": 1, "updates": 1, "exposure_tokens": 10}, "updates_per_epoch": 1,
            "paired_matching": "ordered input exposure and complete epochs, not masked-token counts",
        },
        "data": {"manifest_sha256": sha256(data / "manifest.json")},
        "masks": {
            "identity": "mask-identity", "manifest_path": str(mask_manifest),
            "manifest_sha256": sha256(mask_manifest), "mask_use": "main-loss label selection only",
        },
        "exposure_accounting": {
            "additional_main_loss_tokens": {"full_sequence": 6, "answer_eos": 2},
            "full_sequence_target_main_loss_tokens": 10,
            "answer_eos_whole_history_main_loss_tokens": 6,
        },
        "source_files_sha256": {"independent/rescore_generations.py": "fixture-source-identity"},
    }
    write(run / "continuation_protocol.json", protocol)
    protocol_sha = sha256(run / "continuation_protocol.json")
    (run / "continuation_protocol.sha256").write_text(protocol_sha + "\n")
    write(data / "study_protocol.json", protocol)
    (data / "study_protocol.sha256").write_text(sha256(data / "study_protocol.json") + "\n")
    write(run / "fork_manifest.json", {
        "children": {
            A_ID: a["fork_manifest"], B_ID: b["fork_manifest"], FAILED_ID: {},
        },
    })

    reservation = "reservation-fixture"
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("".join(json.dumps(event) + "\n" for event in [
        {
            "event": "start", "reservation": reservation,
            "native_authorization": {"additional_hours": 12.0},
            "extra": {"source_commit": "1" * 40},
        },
        {"event": "heartbeat", "reservation": reservation, "elapsed": 1.0},
        {"event": "end", "reservation": reservation, "hours": 0.25, "status": "ok"},
        {"event": "start", "reservation": "unrelated-reservation"},
    ]))
    write(run / "launch.json", {
        "format": "native-six-child-continuation-launch-v1",
        "continuation_study_id": "78-native-continuation",
        "source_commit": "1" * 40, "code_identity": "code-fixture",
        "children": children, "protocol_sha256": protocol_sha,
        "fork_manifest_sha256": sha256(run / "fork_manifest.json"),
        "data": {
            "manifest_sha256": sha256(data / "manifest.json"),
            "tokenizer_sha256": "tokenizer-fixture", "replay_sha256": "replay-fixture",
        },
        "ledger_before": {"ledger": str(ledger), "native_used_hours": 0.5},
    })
    write(run / "run_status.json", {
        "format": "native-six-child-continuation-run-status-v1",
        "continuation_study_id": "78-native-continuation",
        "source_commit": "1" * 40, "complete": False,
        "failure": {"error": "synthetic failure", "child": FAILED_ID},
        "failure_history": [{"error": "synthetic failure", "child": FAILED_ID}],
        "previous_status_history": [{"complete": False, "reason": "fixture prior partial status"}],
        "reservation_id": reservation, "charged_hours": 0.25,
        "cumulative_continuation_charged_hours": 0.25,
        "authorized_hours": 1.0, "allocation_limit_respected": True, "wall_seconds": 4.0,
        "children": {
            A_ID: json.loads((run / A_ID / "status.json").read_text()),
            B_ID: json.loads((run / B_ID / "status.json").read_text()),
            FAILED_ID: failed_status,
        },
        "ledger_before": {"native_used_hours": 0.5},
        "ledger_after": {
            "ledger": str(ledger), "native_used_hours": 0.75, "native_remaining_hours": 11.25,
        },
        "exposure_accounting": protocol["exposure_accounting"],
    })
    return run, data, mask_manifest, ledger, run / "evidence-final"


def source_hashes(run, data, mask_manifest, ledger):
    paths = [mask_manifest, ledger]
    paths.extend(path for root in (run, data) for path in root.rglob("*") if path.is_file())
    return {path: sha256(path) for path in paths}


def without_study(path):
    value = json.loads(Path(path).read_text())
    value.pop("study")
    return value


def test_continuation_export_reproduces_and_preserves_failures(tmp_path):
    run, data, mask_manifest, ledger, output = make_study(tmp_path)
    original = source_hashes(run, data, mask_manifest, ledger)
    summary = export(run, data, output)
    assert not summary["complete"]
    assert summary["failure"]["error"] == "synthetic failure"
    assert summary["failure_history"][0]["child"] == FAILED_ID
    assert summary["previous_status_history"][0]["reason"] == "fixture prior partial status"
    assert summary["child_failures"][FAILED_ID]["traceback"] == "fixture traceback"
    assert not summary["matched_endpoints_verified"]
    assert not summary["all_children_completed"]
    assert all(sha256(path) == expected for path, expected in original.items())

    diagnostics = json.loads((output / "generation_diagnostics.json").read_text())
    assert set(diagnostics["arms"]) == {A_ID, B_ID}
    assert diagnostics["evaluation_paths"][A_ID] == f"children/{A_ID}/evaluation.json"
    assert diagnostics["skipped_children"] == [FAILED_ID]
    assert diagnostics["arms"][A_ID]["supplemental_baseline_generation"]["train"]["examples"] == 1
    assert diagnostics["arms"][A_ID]["interventions"]["train"]["changed_completions"] == 1
    assert diagnostics["arms"][A_ID]["interventions"]["train"]["intervened"]["exact_count"] == 0
    assert without_study(output / "generation_diagnostics.reproduced.json") == without_study(
        output / "generation_diagnostics.json"
    )

    arm = summary["children"][A_ID]
    assert arm["endpoint"]["derived_completed_epochs"] == 1
    assert arm["checkpoint_hashes"] == {
        "parent_sha256": "a" * 64, "fork_sha256": "c" * 64, "final_sha256": "e" * 64,
    }
    assert len(arm["checkpoint_manifests"]) == 2
    assert arm["losses"]["train"]["last"]["loss"] == 1.0
    assert arm["routing"]["route_usage"] == [0, 1]
    assert arm["memory"]["training"]["cuda_peak_allocated_bytes"] == 1000
    assert arm["reconstruction"]["teacher_forced_scale_policy"]["agrees"]
    assert not arm["reconstruction"]["teacher_forced_strict_allclose_agrees"]
    assert not arm["reconstruction"]["bitwise_replay_claimed"]
    failure = arm["reconstruction"]["strict_allclose_failures"][0]
    assert failure["tensor"] == "logits"
    assert failure["maximum_absolute_difference"] == 0.25
    assert failure["first_tolerance_violation"]["index"] == [0, 1, 2]

    for split in SPLITS:
        values = arm["splits"][split]
        assert values["primary_generation"]["examples"] == 1
        assert values["teacher_forced_answer_ce"]["answer_eos"]["cross_entropy_sum"] == 1.7
        assert values["teacher_forced_answer_ce"]["numeric_answer_value"]["token_count"] == 4
        assert values["teacher_forced_answer_ce"]["formatting_eos"]["token_count"] == 13
        assert values["selected_evaluation_answer_ce"]["count"] == 17
        assert "records" not in values["selected_evaluation_answer_ce"]
        assert values["primary_generation"]["per_register"]["format_valid_denominator"] == 1
        assert set(values["primary_generation"]["by_final_operation"]) == {"set", "add", "multiply", "copy"}
        assert values["saved_generation_changed_registers"]["failure_records"]["count"] == int(split != "train")
        assert values["no_op_baseline"]["changed_registers"]["micro"]["correct"] == 0
        assert values["state_readout"]["metrics"]["pass_0"]["count"] == 4
        interventions = values["interventions"]
        assert interventions["selected_baseline"]["examples"] == 1
        assert interventions["intervened"]["examples"] == 1
        assert interventions["changed_counts"]["intervened_examples"] == 1
        assert (output / "children" / A_ID / f"{split}_state_path.json").is_file()
    train_interventions = arm["splits"]["train"]["interventions"]
    assert train_interventions["supplemental_baseline_generation"]["examples"] == 1
    assert train_interventions["selection"]["reused_primary_baselines"] == 0
    assert train_interventions["selection"]["additional_supplemental_baselines"] == 1
    assert train_interventions["changed_counts"]["changed_completions"] == 1
    assert train_interventions["intervened"]["truncation"]["completion_count"] == 1
    assert arm["splits"]["dev"]["interventions"]["changed_counts"]["changed_completions"] == 0

    dense = summary["ab_comparisons"]["dense"]
    assert dense["splits"]["dev"]["exact_answers"] == {
        "A": 0, "B": 1, "B_minus_A": 1, "strictly_better": True, "no_regression": True,
    }
    assert not dense["answer_eos_strictly_better_on_every_split"]
    assert dense["no_exact_answer_regression"]
    assert not summary["ab_comparisons"]["untyped"]["available"]
    assert summary["predeclared_rule"]["outcome"] == "retain_full_sequence"
    assert not summary["predeclared_rule"]["evaluation_complete"]
    assert not summary["paired_compute_match"]
    assert summary["cumulative_resource_use"]["native_usage"] == {
        "used_hours": 0.75, "authorized_additional_hours": 12.0,
        "remaining_hours": 11.25, "used_before_hours": 0.5,
    }

    assert (output / "children" / FAILED_ID / "status.json").is_file()
    assert (output / "children" / FAILED_ID / "recovery.jsonl").is_file()
    assert not [path for path in output.rglob("*") if path.suffix in FORBIDDEN_SUFFIXES]
    assert (output / "answer_masks" / "manifest.json").read_bytes() == mask_manifest.read_bytes()
    ledger_events = [json.loads(line) for line in (output / "reservation_ledger.jsonl").read_text().splitlines()]
    assert len(ledger_events) == 3
    assert {event["reservation"] for event in ledger_events} == {"reservation-fixture"}
    reproduction = (output / "REPRODUCE.md").read_text()
    assert "mktemp -d" in reproduction
    assert 'a.pop("study", None)' in reproduction
    assert "bitwise replay is not claimed" in reproduction

    repeated = tmp_path / "fresh-reproduction" / "generation_diagnostics.json"
    subprocess.run([
        sys.executable, str(output / "code" / "export_continuation.py"),
        "--rescore-bundle", str(output), "--output", str(repeated),
    ], check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert without_study(repeated) == without_study(output / "generation_diagnostics.json")

    hashed = {}
    for line in (output / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        hashed[relative] = expected
        assert sha256(output / relative) == expected
    actual = {
        str(path.relative_to(output)) for path in output.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "bundle_manifest.json"}
    }
    assert set(hashed) == actual
    manifest = json.loads((output / "bundle_manifest.json").read_text())
    assert manifest["files_hashed"] == len(actual)
    assert manifest["files_total"] == len(actual) + 2
    assert manifest["payload_bytes"] == sum((output / name).stat().st_size for name in actual)
    assert manifest["summary_sha256"] == sha256(output / "summary.json")
    assert manifest["generation_diagnostics_sha256"] == sha256(output / "generation_diagnostics.json")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        export(run, data, output)
