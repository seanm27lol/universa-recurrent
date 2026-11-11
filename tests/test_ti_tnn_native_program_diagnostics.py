"""Self-contained checks for bounded answer CE and changed-register diagnostics."""
from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
from native import answer_analysis as A  # noqa: E402
from native import diagnose_program_results as D  # noqa: E402
from native.data import IGNORE_INDEX  # noqa: E402


def record(example_id, initial, target, operation="set"):
    return {
        "example_id": example_id,
        "split": "train",
        "initial": list(initial),
        "target_text": target,
        "program": [f"{operation} r0 1" if operation == "set" else f"{operation} r0 r1"],
        "prompt": "self-contained prompt",
    }


def test_no_op_and_saved_changed_register_behavior_preserve_failures():
    first = record("changed", (1, 2, 3, 4), "r0=1 r1=9 r2=3 r3=7")
    second = record("unchanged", (1, 2, 3, 4), "r0=1 r1=2 r2=3 r3=4")
    records = [first, second]
    no_op = D.no_op_metrics(records)
    assert no_op["full_exact"] == {"correct": 1, "count": 2, "accuracy": 0.5}
    assert no_op["per_register_overall"]["micro"] == {"correct": 6, "count": 8, "accuracy": 0.75}
    assert no_op["changed_registers"]["micro"] == {"correct": 0, "count": 2, "accuracy": 0.0}
    assert no_op["changed_registers"]["per_register"]["0"]["count"] == 0
    assert no_op["changed_registers"]["per_register"]["0"]["accuracy"] is None
    assert no_op["changed_registers"]["per_register"]["1"]["count"] == 1
    assert no_op["changed_registers"]["per_register"]["3"]["count"] == 1
    assert no_op["examples_with_no_changed_register"] == 1
    assert no_op["failure_count"] == 1
    assert no_op["failure_records"][0]["prediction_text"] == "r0=1 r1=2 r2=3 r3=4"
    assert set(no_op["by_final_operation"]) == {"set", "add", "multiply", "copy"}

    saved = [
        {
            "example_id": "changed",
            "target_text": first["target_text"],
            "final_operation": "set",
            "completion_text": "r0=1 r1=8 r2=3 r3=7",
            "correct": False,
            "valid": True,
            "first_error": {"kind": "register_value", "register": 1},
        },
        {
            "example_id": "unchanged",
            "target_text": second["target_text"],
            "final_operation": "set",
            "completion_text": "not canonical registers",
            "correct": False,
            "valid": False,
            "first_error": {"kind": "format"},
        },
    ]
    generated = D.saved_generation_metrics(records, saved)
    assert generated["full_exact"]["correct"] == 0
    assert generated["format_valid"] == {"count": 1, "rate": 0.5}
    assert generated["per_register_overall"]["micro"] == {"correct": 3, "count": 8, "accuracy": 0.375}
    assert generated["changed_registers"]["micro"] == {"correct": 1, "count": 2, "accuracy": 0.5}
    assert generated["changed_registers_format_valid_only"]["micro"] == {"correct": 1, "count": 2, "accuracy": 0.5}
    assert generated["failure_count"] == 2
    assert generated["failure_records"][0]["saved_failure_record"]["first_error"]["kind"] == "register_value"
    assert generated["failure_records"][1]["saved_failure_record"]["first_error"]["kind"] == "format"
    assert generated["saved_failure_records_preserved"]


def coverage_arrays(first_answer, anchor):
    length = first_answer + 17
    width = 40
    arrays = {
        "input_ids": np.ones((1, width), dtype=np.int32),
        "labels": np.full((1, width), IGNORE_INDEX, dtype=np.int32),
        "positions": np.zeros((1, width), dtype=np.int32),
        "doc_ids": np.full((1, width), -1, dtype=np.int32),
        "key_padding_mask": np.ones((1, width), dtype=np.bool_),
        "source_ids": np.full((1, width), -1, dtype=np.int8),
        "route_targets": np.zeros((1, width, 4), dtype=np.int8),
        "route_mask": np.zeros((1, width, 4), dtype=np.bool_),
        "state_targets": np.zeros((1, width, 4, 4), dtype=np.int16),
        "state_mask": np.zeros((1, width, 4, 4), dtype=np.bool_),
        A.ANSWER_MASK: np.zeros((1, width), dtype=np.bool_),
        A.ANSWER_VALUE_MASK: np.zeros((1, width), dtype=np.bool_),
        A.FORMATTING_EOS_MASK: np.zeros((1, width), dtype=np.bool_),
    }
    arrays["positions"][0, :length] = np.arange(length)
    arrays["labels"][0, 1:length] = 1
    arrays["doc_ids"][0, :length] = 0
    arrays["key_padding_mask"][0, :length] = False
    arrays["source_ids"][0, :length] = 2
    arrays[A.ANSWER_MASK][0, first_answer:length] = True
    value_columns = [first_answer + 3, first_answer + 7, first_answer + 11, first_answer + 15]
    arrays[A.ANSWER_VALUE_MASK][0, value_columns] = True
    arrays[A.FORMATTING_EOS_MASK] = arrays[A.ANSWER_MASK] & ~arrays[A.ANSWER_VALUE_MASK]
    arrays["route_mask"][0, anchor] = True
    arrays["state_mask"][0, anchor] = True
    return arrays


def test_training_vs_longer_answer_position_coverage():
    training = coverage_arrays(first_answer=5, anchor=3)
    longer = coverage_arrays(first_answer=12, anchor=7)
    result = A.compare_answer_position_coverage(training, longer)
    for name in ("training", "longer_program"):
        values = result[name]
        assert values["documents"] == 1
        assert values["answer_eos_positions"] == values["answer_eos_positions_with_lm_labels"] == 17
        assert values["answer_value_positions"] == 4
        assert values["formatting_eos_positions"] == 13
        assert values["route_mask_true_at_answer_positions"] == 0
        assert values["state_mask_true_at_answer_positions"] == 0
        assert values["route_positions_with_any_target_at_answer"] == 0
        assert values["state_positions_with_any_target_at_answer"] == 0
        assert values["full_lm_supervision_includes_every_answer_position"]
        assert values["route_state_targets_exclude_every_answer_position"]
    assert result["training"]["first_answer_position"]["mean"] == 5
    assert result["training"]["last_instruction_anchor"]["mean"] == 3
    assert result["training"]["downstream_distance"]["mean"] == 2
    assert result["longer_program"]["first_answer_position"]["mean"] == 12
    assert result["longer_program"]["last_instruction_anchor"]["mean"] == 7
    assert result["longer_program"]["downstream_distance"]["mean"] == 5
    assert result["longer_minus_training_mean"] == {
        "first_answer_position": 7.0,
        "last_instruction_anchor": 4.0,
        "downstream_distance": 3.0,
        "intervening_tokens_between_anchor_and_answer": 3.0,
    }
    assert "LM labels include all 17" in result["supervision_boundary"]


class TinyDataset:
    def __init__(self, arrays):
        self.arrays = arrays
        self.masks = {name: arrays[name] for name in A.MASK_ARRAY_NAMES}

    def __len__(self):
        return len(self.arrays["input_ids"])

    def batch(self, indices, *, as_torch=True, device=None, model_inputs_only=False):
        names = list(self.arrays)
        if model_inputs_only:
            names = ["input_ids", "positions", "doc_ids", "key_padding_mask"]
        result = {}
        for name in names:
            tensor = torch.from_numpy(np.ascontiguousarray(self.arrays[name][indices]))
            if name in {"input_ids", "labels", "positions", "doc_ids", "route_targets", "state_targets"}:
                tensor = tensor.long()
            result[name] = tensor.to(device) if device is not None else tensor
        return result


class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, input_ids, positions=None, doc_ids=None, key_padding_mask=None):
        self.calls.append({
            "input_ids": input_ids.detach().clone(),
            "positions": positions.detach().clone(),
            "doc_ids": doc_ids.detach().clone(),
            "key_padding_mask": key_padding_mask.detach().clone(),
        })
        return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 64), dtype=torch.float32))


def test_teacher_forced_ce_includes_first_answer_and_separates_three_counts():
    arrays = coverage_arrays(first_answer=5, anchor=3)
    dataset = TinyDataset(arrays)
    model = ConstantModel()
    model.train()
    result = D.teacher_forced_answer_cross_entropy(model, dataset, batch_size=1)
    assert result["complete"]
    assert result["first_answer_token_included"]
    assert result["full_lm_label_tokens_evaluated"] == 21
    assert result["metrics"][A.ANSWER_MASK]["token_count"] == 17
    assert result["metrics"]["numeric_answer_value"]["token_count"] == 4
    assert result["metrics"][A.FORMATTING_EOS_MASK]["token_count"] == 13
    for metrics in result["metrics"].values():
        assert metrics["cross_entropy_mean"] == pytest.approx(math.log(64))
        assert metrics["cross_entropy_sum"] == pytest.approx(metrics["token_count"] * math.log(64))
    assert model.training
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["input_ids"].shape[1] == 22
    assert torch.equal(call["input_ids"], torch.from_numpy(arrays["input_ids"][:, :22]).long())
    assert not any(name in call for name in A.MASK_ARRAY_NAMES)
