import json

import pytest

from native.independent.rescore_generations import (
    SPLITS,
    ScoringError,
    execute_saved_program,
    registers,
    rescore,
    summarize,
)
from native.structured import Instruction, make_example


def source_example(split="train"):
    return make_example(
        split, [Instruction("set", 0, value=1)], (0, 0, 0, 0)
    ).to_record()


def saved_generation(record, completion_text=None, correct=True):
    completion_text = record["target_text"] if completion_text is None else completion_text
    return {
        "example_id": record["example_id"],
        "completion_text": completion_text,
        "target_text": record["target_text"],
        "correct": correct,
        "valid": registers(completion_text) is not None,
        "stopped_on_eos": True,
        "truncated": False,
        "prompt_truncated": False,
    }


def write_sources(root, records):
    structured = root / "structured"
    structured.mkdir(parents=True)
    for split, record in records.items():
        (structured / f"{split}.jsonl").write_text(json.dumps(record) + "\n")


def answer_losses():
    return {
        "train": {"sum": 0.25, "count": 2, "mean": 0.125},
        "dev": {"sum": 0.5, "count": 2, "mean": 0.25},
        "unseen_composition": {"sum": 0.0, "count": 0, "mean": None, "status": "not_run"},
        "longer_length": {"sum": 0.0, "count": 0, "mean": None, "status": "not_run"},
    }


def test_independent_parser_rejects_noncanonical_registers_and_immediates():
    assert registers("r0=0 r1=2 r2=31 r3=7") == (0, 2, 31, 7)
    invalid = [
        "r0=0 r1=2 r2=3",
        "r0=0 r1=2 r2=3 r3=4 extra",
        "r0=0 r1=2 r2=3 r3=32",
        "r0=-1 r1=2 r2=3 r3=4",
        " r0=0 r1=2 r2=3 r3=4",
    ]
    for index in range(4):
        values = ["0", "2", "3", "4"]
        values[index] = "01"
        invalid.append(" ".join(f"r{register}={value}" for register, value in enumerate(values)))
    assert all(registers(text) is None for text in invalid)

    record = source_example()
    for immediate in ("00", "01", "+1", "-0"):
        invalid_program = {**record, "program": [f"set r0 {immediate}"]}
        with pytest.raises(ScoringError, match="noncanonical instruction"):
            execute_saved_program(invalid_program)


def test_saved_target_must_match_source_and_invalid_completion_is_not_exact():
    record = source_example()
    invalid_completion = saved_generation(record, completion_text="not registers")
    metrics = summarize([invalid_completion], {record["example_id"]: record})
    assert metrics["exact_count"] == 0
    assert metrics["format_valid_count"] == 0
    assert metrics["records"][0]["exact_match"] is False
    assert metrics["score_changes"] == [{
        "example_id": record["example_id"],
        "metric": "exact_match",
        "original": True,
        "independent": False,
    }]

    tampered = saved_generation(record)
    tampered["target_text"] = "r0=2 r1=0 r2=0 r3=0"
    tampered["completion_text"] = tampered["target_text"]
    with pytest.raises(ScoringError, match="does not match validated source target"):
        summarize([tampered], {record["example_id"]: record})


def test_acquisition_rescore_uses_available_splits_and_separate_answer_loss(tmp_path):
    records = {split: source_example(split) for split in ("train", "dev")}
    data = tmp_path / "data"
    study = tmp_path / "study"
    write_sources(data, records)
    losses = answer_losses()
    evaluation = {
        "format": "native-acquisition-evaluation-v1",
        "splits": {
            split: {
                "generation": {"records": [saved_generation(record)]},
                "state_path": {"answer_token_cross_entropy": {split: losses[split]}},
            }
            for split, record in records.items()
        },
    }
    evaluation_path = study / "arms" / "dense" / "evaluation.json"
    evaluation_path.parent.mkdir(parents=True)
    evaluation_path.write_text(json.dumps(evaluation, indent=2) + "\n")
    original_evaluation = evaluation_path.read_text()

    output = tmp_path / "diagnostics" / "rescore.json"
    result = rescore(study, data, output)
    assert json.loads(output.read_text()) == result
    assert evaluation_path.read_text() == original_evaluation
    assert result["target_validation"]["executed_programs"] == 2
    arm = result["arms"]["dense"]
    assert set(arm["structured"]) == {"train", "dev"}
    assert arm["code"]["status"] == "not_run"
    assert arm["answer_token_cross_entropy"] == {
        split: losses[split] for split in ("train", "dev")
    }
    for split in ("train", "dev"):
        assert arm["structured"][split]["exact_count"] == 1
        assert arm["structured"][split]["answer_value_loss"] == 0.0
    output.write_text("preserved existing evidence\n")
    with pytest.raises(FileExistsError):
        rescore(study, data, output)
    assert output.read_text() == "preserved existing evidence\n"
    assert evaluation_path.read_text() == original_evaluation


def test_normal_four_split_rescore_schema_still_runs(tmp_path):
    records = {split: source_example(split) for split in SPLITS}
    data = tmp_path / "data"
    study = tmp_path / "study"
    write_sources(data, records)
    evaluation = {
        "format": "native-evaluation-v1",
        "structured_generation": {
            "splits": {
                split: {"records": [saved_generation(record)]}
                for split, record in records.items()
            }
        },
        "code_generation": {"status": "not_run"},
    }
    evaluation_path = study / "arms" / "typed" / "evaluation.json"
    evaluation_path.parent.mkdir(parents=True)
    evaluation_path.write_text(json.dumps(evaluation) + "\n")

    result = rescore(study, data, tmp_path / "rescore.json")
    arm = result["arms"]["typed"]
    assert set(arm["structured"]) == set(SPLITS)
    assert all(arm["structured"][split]["exact_count"] == 1 for split in SPLITS)
    assert arm["code"]["status"] == "not_run"
