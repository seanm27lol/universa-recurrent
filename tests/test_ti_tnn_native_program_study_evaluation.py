"""Construction checks for program-study evaluation, not future learning results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
from native import evaluate_program_study as P  # noqa: E402
from native import evaluate_state_path as S  # noqa: E402
from native.config import NativeConfig  # noqa: E402
from native.independent import rescore_generations as R  # noqa: E402
from native.model import NativeModel  # noqa: E402
from native.structured import Instruction, execute, format_prompt  # noqa: E402


OPERATIONS = ("set", "add", "multiply", "copy")


def make_record(split, operation, index=0, program=None):
    initial = (index % 32, (index + 1) % 32, (index + 2) % 32, (index + 3) % 32)
    if program is None:
        text = {
            "set": f"set r0 {index % 32}",
            "add": "add r0 r1",
            "multiply": "multiply r0 r1",
            "copy": "copy r0 r1",
        }[operation]
        program = [
            f"set r3 {P.SPLITS.index(split)}",
            f"set r2 {index % 32}",
            f"set r1 {index // 32}",
            text,
        ]
    instructions = [Instruction.parse(text) for text in program]
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


def make_split(split, per_operation=32):
    return [
        make_record(split, operation, index)
        for operation in OPERATIONS
        for index in range(per_operation)
    ]


def write_sources(root, records):
    structured = root / "structured"
    structured.mkdir(parents=True)
    for split, split_records in records.items():
        if isinstance(split_records, dict):
            split_records = [split_records]
        (structured / f"{split}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in split_records)
        )


def saved_generation(record, completion_text=None):
    completion_text = record["target_text"] if completion_text is None else completion_text
    return {
        "example_id": record["example_id"],
        "completion_text": completion_text,
        "target_text": record["target_text"],
        "correct": completion_text == record["target_text"],
        "valid": R.registers(completion_text) is not None,
        "stopped_on_eos": True,
        "truncated": False,
        "prompt_truncated": False,
    }


def test_manifest_selection_and_intervention_stratification():
    records = make_split("train", 33)
    evaluation_ids = [
        record["example_id"]
        for operation in OPERATIONS
        for record in [item for item in records if item["program"][-1].startswith(operation + " ")][:32]
    ]
    intervention_ids = [
        record["example_id"]
        for operation in OPERATIONS
        for record in [
            item for item in records
            if item["example_id"] in evaluation_ids and P._final_operation(item) == operation
        ][:8]
    ]
    manifest = {
        "selections": {
            "evaluation_example_ids": {"train": evaluation_ids},
            "intervention_example_ids": {"train": intervention_ids},
        }
    }
    selected, selection = P.select_evaluation_records(records, manifest, "train")
    intervened, intervention_selection = P.select_intervention_records(selected, manifest, "train")
    assert len(selected) == 128
    assert selection["source"] == "manifest"
    assert selection["evaluation_ids"] == evaluation_ids
    assert all(record["example_id"] in evaluation_ids for record in selected)
    assert selection["per_final_operation"] == {operation: 32 for operation in OPERATIONS}
    assert len(intervened) == 32
    assert intervention_selection["per_final_operation"] == {operation: 8 for operation in OPERATIONS}
    assert intervention_selection["intervention_ids"] == intervention_ids

    fallback, fallback_selection = P.select_intervention_records(selected, {}, "train")
    assert len(fallback) == 32
    assert fallback_selection["source"] == "first_8_per_final_operation"
    assert all(sum(P._final_operation(record) == operation for record in fallback) == 8
               for operation in OPERATIONS)


def test_generation_metrics_group_by_final_operation_and_keep_register_error_separate(monkeypatch):
    records = [make_record("dev", operation) for operation in OPERATIONS]
    wrong = list(R.registers(records[1]["target_text"]))
    wrong[0] = (wrong[0] + 1) % 32
    wrong_text = " ".join(f"r{index}={value}" for index, value in enumerate(wrong))
    completions = {
        "set": (records[0]["target_text"], True, False),
        "add": (wrong_text, True, False),
        "multiply": ("not canonical registers", False, True),
        "copy": (records[3]["target_text"], True, False),
    }
    calls = []

    def generate(model, tokenizer, prompt, **kwargs):
        operation = next(operation for operation, record in zip(OPERATIONS, records) if record["prompt"] == prompt)
        text, stopped, truncated = completions[operation]
        calls.append((prompt, kwargs))
        return {
            "prompt_ids": [1, 2],
            "positions": [0, 1],
            "original_prompt_tokens": 2,
            "prompt_truncated": False,
            "prompt_policy": "error",
            "sequence_ids": [1, 2, 9],
            "generated_ids": [9],
            "completion_ids": [9],
            "completion_text": text,
            "stopped_on_eos": stopped,
            "truncated": truncated,
            "max_new_tokens": kwargs["max_new_tokens"],
            "use_cache": True,
        }

    monkeypatch.setattr(P.E, "generate_prompt", generate)
    result = P.generation_metrics(object(), object(), records, max_new_tokens=7)
    assert result["examples"] == result["evaluated"] == 4
    assert result["correct"] == 2
    assert result["format_valid_count"] == 3
    assert result["eos_rate"] == 0.75
    assert result["truncation_rate"] == 0.25
    assert result["per_register_denominator"] == 3
    assert result["per_register_mean_absolute_error"]["0"] == pytest.approx(1 / 3)
    assert result["answer_value_loss"] == pytest.approx(1 / 12)
    assert "answer_token_cross_entropy" not in result
    assert len(result["records"]) == 4
    assert result["records"][1]["first_error"] == {
        "kind": "register_value", "register": 0, "actual": wrong[0], "expected": R.registers(records[1]["target_text"])[0],
    }
    assert result["records"][2]["first_error"]["kind"] == "format"
    assert set(result["by_final_operation"]) == set(OPERATIONS)
    for operation in OPERATIONS:
        metrics = result["by_final_operation"][operation]
        assert metrics["examples"] == 1
        assert metrics["correct"] == int(operation in ("set", "copy"))
        assert metrics["format_valid_count"] == int(operation != "multiply")
    assert all(call[1]["prompt_policy"] == "error" and call[1]["max_new_tokens"] == 7 for call in calls)


def test_evaluate_five_splits_forwards_selection_and_uncapped_state_policy(tmp_path, monkeypatch):
    root = tmp_path / "data"
    records = {split: make_split(split, 33 if split == "train" else 32) for split in P.SPLITS}
    write_sources(root, records)
    train_ids = [
        record["example_id"]
        for operation in OPERATIONS
        for record in [item for item in records["train"] if P._final_operation(item) == operation][:32]
    ]
    train_intervention_ids = []
    for operation in OPERATIONS:
        candidates = [record for record in records["train"] if P._final_operation(record) == operation]
        train_intervention_ids.extend(
            [record["example_id"] for record in candidates if record["example_id"] in train_ids][:7]
        )
        train_intervention_ids.extend(
            [record["example_id"] for record in candidates if record["example_id"] not in train_ids][:1]
        )
    manifest = {
        "position_policy": "document",
        "splits": {split: {"position_policy": "document"} for split in P.SPLITS},
        "selections": {
            "evaluation_example_ids": {"train": train_ids},
            "intervention_example_ids": {"train": train_intervention_ids},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("synthetic tokenizer")
    model = object()
    load_calls = []
    generation_calls = []
    intervention_calls = []
    state_calls = []
    monkeypatch.setattr(P.E, "load_native_model", lambda *args, **kwargs: (
        load_calls.append((args, kwargs)) or (model, {"sha256": checkpoint_sha, "step": 3, "config": {"mode": "dense"}})
    ))
    monkeypatch.setattr(P.E, "load_gpt2_tokenizer", lambda path: object())

    def generation(model_arg, tokenizer_arg, selected, **kwargs):
        generation_calls.append((selected, kwargs))
        return {
            "examples": len(selected),
            "correct": 0,
            "records": [
                {
                    **saved_generation(record, "invalid"),
                    "prompt_ids": [1],
                    "completion_ids": [2],
                    "generated_ids": [2],
                }
                for record in selected
            ],
        }

    def interventions(model_arg, tokenizer_arg, selected, ordinary, **kwargs):
        intervention_calls.append((selected, ordinary, kwargs))
        split = selected[0]["split"]
        assert len(selected) == 32
        assert len(ordinary) == (132 if split == "train" else 128)
        assert len({record["example_id"] for record in ordinary}) == len(ordinary)
        assert {record["example_id"] for record in selected} <= {record["example_id"] for record in ordinary}
        return {"selection_examples": 32, "records": []}

    def state(checkpoint_arg, packed, output, **kwargs):
        split = kwargs["split"]
        state_calls.append((packed, output, kwargs))
        return {
            "packed_rows": 3,
            "evaluated_rows": 3,
            "complete": True,
            "answer_token_cross_entropy": {split: {"complete": True, "evaluated": 128}},
        }

    monkeypatch.setattr(P, "generation_metrics", generation)
    monkeypatch.setattr(P, "targeted_interventions", interventions)
    monkeypatch.setattr(P, "evaluate_state_path", state)
    output = tmp_path / "arm" / "evaluation.json"
    result = P.evaluate(checkpoint, checkpoint_sha, root, tokenizer_path, output, max_new_tokens=7)
    assert json.loads(output.read_text()) == result
    assert result["format"] == "native-program-study-evaluation-v1"
    assert tuple(result["splits"]) == P.SPLITS
    assert load_calls[0][1] == {"expected_sha256": checkpoint_sha, "device": "cpu"}
    assert len(generation_calls) == 6
    assert len(intervention_calls) == len(state_calls) == 5
    assert all(kwargs["max_new_tokens"] == 7 and len(kwargs) == 1 for _, kwargs in generation_calls)
    assert [record["example_id"] for record in generation_calls[0][0]] == train_ids
    assert [record["example_id"] for record in generation_calls[1][0]] == [
        example_id for example_id in train_intervention_ids if example_id not in train_ids
    ]
    for split in P.SPLITS:
        assert result["splits"][split]["selection"]["examples"] == 128
        assert result["splits"][split]["selection"]["per_final_operation"] == {
            operation: 32 for operation in OPERATIONS
        }
        intervention_selection = result["splits"][split]["selection"]["interventions"]
        assert intervention_selection["additional_ordinary_baselines"] == (4 if split == "train" else 0)
        assert intervention_selection["reused_evaluation_baselines"] == (28 if split == "train" else 32)
    assert result["splits"]["train"]["interventions"]["supplemental_baseline_generation"]["examples"] == 4
    for split, (packed, state_output, kwargs) in zip(P.SPLITS, state_calls):
        assert packed == root / split
        assert state_output == output.parent / f"{split}_state_path.json"
        assert kwargs["max_batches"] is None
        assert kwargs["position_policy"] == "document"
        assert kwargs["evaluation_ids"] == result["splits"][split]["selection"]["evaluation_ids"]
    assert result["baselines"]["lookup_fit_examples"] == 132
    assert result["baselines"]["exact_lookup"]["train"]["exact_match"] == 1.0
    assert result["baselines"]["exact_lookup"]["initial_values"]["exact_match"] == 0.0
    for split in P.SPLITS:
        assert result["baselines"]["supplied_program_executor"][split]["exact_match"] == 1.0


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        ids = [ord(character) % 30 + 1 for character in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        return SimpleNamespace(ids=ids, offsets=offsets)

    def decode(self, ids, skip_special_tokens=False):
        return ""


def test_real_tiny_model_intervention_uses_saved_baseline_and_targets_final_instruction(monkeypatch):
    torch.manual_seed(11)
    model = NativeModel(NativeConfig(
        vocab_size=32,
        context_length=256,
        d_model=16,
        d_slot=8,
        d_ff=32,
        n_heads=4,
        mode="typed",
    ))
    model.eval()
    record = make_record("dev", "copy", program=["set r0 1", "copy r2 r1"])
    original_generate = P.E.generate_prompt
    calls = []

    def counted_generate(*args, **kwargs):
        calls.append((args, kwargs))
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(P.E, "generate_prompt", counted_generate)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    ordinary = P.generation_metrics(model, CharacterTokenizer(), [record], max_new_tokens=1, eos_token_id=31)
    result = P.targeted_interventions(
        model, CharacterTokenizer(), [record], ordinary["records"], max_new_tokens=1, eos_token_id=31
    )
    assert len(calls) == 2
    assert result["baseline_source"] == "saved ordinary generation records; no duplicate baseline decoding"
    assert result["targeted_exactly_once"] == 1
    assert result["four_writer_updates_suppressed_per_example"] == 1
    detail = result["records"][0]
    assert detail["final_instruction"]["instruction"] == "copy r2 r1"
    assert detail["final_instruction"]["instruction_index"] == 1
    assert detail["intervention"]["targeted_forwards"] == 1
    assert detail["intervention"]["suppressed_writer_updates"] == 4
    assert set(detail["baseline"]) == {"correct", "valid", "stopped_on_eos", "truncated"}
    assert "semantic faithfulness" in result["evidence_boundary"]
    assert "not absence" in result["evidence_boundary"]
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert all(parameter.grad is None for parameter in model.parameters())


class CETokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[1] if text.startswith("prompt") else [2])


class CEModel:
    config = SimpleNamespace(context_length=8)

    def __call__(self, input_ids):
        return SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], 50257))


def test_initial_values_answer_ce_uses_selected_ids_and_includes_eos(tmp_path):
    records = [
        {"example_id": "initial-a", "prompt": "prompt a", "target_text": "target a"},
        {"example_id": "initial-b", "prompt": "prompt b", "target_text": "target b"},
    ]
    (tmp_path / "initial_values.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    result = S._answer_cross_entropy(
        CEModel(), CETokenizer(), tmp_path, split="initial_values", evaluation_ids=["initial-b"]
    )
    assert set(result) == {"initial_values"}
    metrics = result["initial_values"]
    assert metrics["examples"] == metrics["evaluated"] == 1
    assert metrics["source_examples"] == 2
    assert metrics["count"] == 2
    assert metrics["complete"]
    assert metrics["records"] == [{
        "example_id": "initial-b",
        "sum": pytest.approx(float(2 * torch.log(torch.tensor(50257.0)))),
        "count": 2,
        "mean": pytest.approx(float(torch.log(torch.tensor(50257.0)))),
        "prompt_tokens": 1,
        "answer_tokens_including_eos": 2,
    }]


def test_document_position_validation_checks_every_row_and_reset():
    dataset = SimpleNamespace(arrays={
        "positions": np.asarray([[0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.int32),
        "doc_ids": np.asarray([[5, 5, 6, -1], [7, -1, -1, -1]], dtype=np.int32),
        "key_padding_mask": np.asarray([[False, False, False, True], [False, True, True, True]]),
    })
    result = S._validate_document_positions(dataset)
    assert result["valid"]
    assert result["packed_rows_checked"] == 2
    assert result["live_tokens_checked"] == 4
    assert result["document_starts_checked"] == 3
    dataset.arrays["positions"][0, 2] = 1
    with pytest.raises(ValueError, match="does not reset"):
        S._validate_document_positions(dataset)


def test_independent_program_study_rescore_filters_modes_and_scores_all_generations(tmp_path):
    root = tmp_path / "data"
    records = {split: make_record(split, OPERATIONS[index % 4], index) for index, split in enumerate(P.SPLITS)}
    supplemental = make_record("train", "add", 1)
    sources = {**records, "train": [records["train"], supplemental]}
    write_sources(root, sources)
    assert len(R.source_records(root)) == 5
    assert len(R.source_records(root, R.PROGRAM_STUDY_SPLITS)) == 6

    def generated(record):
        token_id = len(record["example_id"]) % 251 + 1
        return {
            **saved_generation(record),
            "completion_ids": [token_id],
            "generated_ids": [token_id],
        }

    def intervention(record, changed):
        baseline = generated(record)
        intervened = generated(record)
        if changed:
            intervened["completion_ids"] = [baseline["completion_ids"][0] + 1]
            intervened["generated_ids"] = [baseline["generated_ids"][0] + 1]
        return {
            "example_id": record["example_id"],
            "intervened": intervened,
            "changed_completion": baseline["completion_ids"] != intervened["completion_ids"],
            "changed_generated_suffix": baseline["generated_ids"] != intervened["generated_ids"],
        }

    losses = {split: {"sum": float(index), "count": 2, "mean": index / 2} for index, split in enumerate(P.SPLITS)}
    evaluation = {
        "format": P.FORMAT,
        "splits": {
            split: {
                "generation": {"records": [generated(record)]},
                "interventions": {
                    "supplemental_baseline_generation": {
                        "records": [generated(supplemental)] if split == "train" else [],
                    },
                    "records": [intervention(record, split == "train")],
                },
                "state_path": {"answer_token_cross_entropy": {split: losses[split]}},
            }
            for split, record in records.items()
        },
    }
    study = tmp_path / "study"
    evaluation_path = study / "arms" / "dense" / "evaluation.json"
    evaluation_path.parent.mkdir(parents=True)
    evaluation_path.write_text(json.dumps(evaluation) + "\n")
    original = evaluation_path.read_text()
    other_evaluation = json.loads(json.dumps(evaluation))
    other_evaluation["splits"]["dev"]["interventions"]["records"][0]["changed_completion"] = True
    other_path = study / "arms" / "untyped" / "evaluation.json"
    other_path.parent.mkdir(parents=True)
    other_path.write_text(json.dumps(other_evaluation) + "\n")

    output = tmp_path / "rescore.json"
    result = R.rescore(study, root, output, modes=("dense",))
    assert set(result["arms"]) == {"dense"}
    assert result["evaluation_format"] == P.FORMAT
    assert result["target_validation"]["executed_programs"] == 6
    arm = result["arms"]["dense"]
    assert set(arm["structured"]) == set(P.SPLITS)
    assert arm["structured"]["initial_values"]["exact_count"] == 1
    assert arm["answer_token_cross_entropy"] == losses
    assert arm["structured"]["initial_values"]["by_final_operation"]["multiply"]["examples"] == 1
    assert arm["supplemental_baseline_generation"]["train"]["examples"] == 1
    assert arm["supplemental_baseline_generation"]["dev"]["examples"] == 0
    assert arm["supplemental_baseline_generation"]["train"]["prerequisites"]["complete"] == 1
    assert arm["interventions"]["train"]["changed_completions"] == 1
    assert arm["interventions"]["train"]["changed_generated_suffixes"] == 1
    assert arm["interventions"]["dev"]["changed_completions"] == 0
    assert arm["interventions"]["train"]["intervened"]["prerequisites"]["complete"] == 1
    assert arm["interventions"]["train"]["score_changes"] == []
    assert evaluation_path.read_text() == original
    assert json.loads(output.read_text()) == result

    all_arms = R.rescore(study, root, tmp_path / "rescore-all.json")
    assert set(all_arms["arms"]) == {"dense", "untyped"}
    changes = all_arms["arms"]["untyped"]["interventions"]["dev"]["score_changes"]
    assert changes == [{
        "example_id": records["dev"]["example_id"],
        "metric": "changed_completion",
        "original": True,
        "independent": False,
    }]


def test_program_study_cli_delegates_with_default_generation_bound(monkeypatch, capsys):
    calls = []
    result = {
        "checkpoint": {"sha256": "checkpoint-sha"},
        "splits": {split: {"generation": {"correct": index}} for index, split in enumerate(P.SPLITS)},
    }
    monkeypatch.setattr(P, "evaluate", lambda *args: calls.append(args) or result)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_program_study", "--checkpoint", "checkpoint.pt", "--expected-sha256", "checkpoint-sha",
        "--data", "data", "--tokenizer", "tokenizer.json", "--output", "evaluation.json",
    ])
    assert P.main() == 0
    assert calls == [("checkpoint.pt", "checkpoint-sha", "data", "tokenizer.json", "evaluation.json", 48)]
    assert json.loads(capsys.readouterr().out)["exact"]["initial_values"] == 2
