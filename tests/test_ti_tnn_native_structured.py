"""Offline construction checks for the native executable register-machine data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
from native import structured  # noqa: E402


def tiny_tokenizer(examples):
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(
        [example.full_text for split in structured.SPLIT_NAMES for example in examples.get(split, ())],
        trainers.WordLevelTrainer(special_tokens=["[UNK]"]),
    )
    return tokenizer


def small_dataset():
    return structured.generate_dataset(train=10, dev=6, unseen_composition=6, longer_length=6, seed=17)


def test_execution_requires_composition_and_updates_all_operation_families():
    program = structured.execute(
        ["set r0 2", "add r0 r1", "multiply r2 r0", "copy r3 r2"], (3, 4, 5, 6))
    assert program.final == (6, 4, 30, 30)
    assert program.target_text == "r0=6 r1=4 r2=30 r3=30"
    assert [event.after for event in program.events] == [
        (2, 4, 5, 6), (6, 4, 5, 6), (6, 4, 30, 6), (6, 4, 30, 30)]
    assert program.events[0].routes == (0, 0, 0, 3)
    assert program.events[1].routes == (1, 0, 1, 3)
    assert program.events[2].routes == (2, 0, 2, 3)
    assert program.events[3].routes == (3, 0, 3, 3)
    for event in program.events:
        assert event.states[0] == event.before
        assert event.states[1:] == (event.after,) * 3


def test_modular_execution_and_validation():
    result = structured.execute(["add r0 r1", "multiply r2 r2"], (31, 1, 31, 0))
    assert result.final == (0, 1, 1, 0)
    with pytest.raises(ValueError):
        structured.execute([], (0, 0, 0, 0))
    with pytest.raises(ValueError):
        structured.execute(["set r0 32"], (0, 0, 0, 0))
    with pytest.raises(ValueError):
        structured.Instruction("add", 0, value=2)


def test_holdouts_are_separated_and_reproducible():
    first = small_dataset()
    second = small_dataset()
    structured.assert_holdout_separation(first)
    assert {split: [e.full_text for e in examples] for split, examples in first.items()} == {
        split: [e.full_text for e in examples] for split, examples in second.items()}
    ordinary = first["train"] + first["dev"]
    assert all(structured.RESERVED_COMPOSITION not in structured.composition_pairs(e.program) for e in ordinary)
    assert all(2 <= len(e.program) <= 4 for e in ordinary)
    assert all(structured.RESERVED_COMPOSITION in structured.composition_pairs(e.program)
               for e in first["unseen_composition"])
    assert all(6 <= len(e.program) <= 8 for e in first["longer_length"])
    prompts = [e.prompt for examples in first.values() for e in examples]
    assert len(prompts) == len(set(prompts))


def test_offsets_align_per_pass_routes_and_four_slot_states_without_dropping_text():
    dataset = small_dataset()
    tokenizer = tiny_tokenizer(dataset)
    example = dataset["train"][0]
    aligned = structured.align_example(example, tokenizer, eos_token_id=50256)
    plain_ids, offsets = structured.encode_with_offsets(tokenizer, example.full_text)
    assert aligned["input_ids"].tolist() == [*plain_ids, 50256]
    assert offsets[-1][1] == len(example.full_text)
    assert len(aligned["anchors"]) == len(example.program)
    assert np.all(np.diff(aligned["anchors"]) > 0)
    for anchor, event in zip(aligned["anchors"], example.execution.events):
        start, end = offsets[anchor]
        assert example.full_text[start:end] == str(event.instruction).split()[-1]
        assert aligned["route_targets"][anchor].tolist() == list(event.routes)
        assert aligned["route_mask"][anchor].tolist() == [True] * 4
        assert aligned["state_targets"][anchor].tolist() == [list(state) for state in event.states]
        assert aligned["state_mask"][anchor].tolist() == [[True] * 4] * 4
    outside = np.ones(len(aligned["input_ids"]), dtype=bool)
    outside[aligned["anchors"]] = False
    assert not aligned["route_mask"][outside].any()
    assert not aligned["state_mask"][outside].any()
    assert int(aligned["route_mask"].sum()) == 4 * len(example.program)
    assert int(aligned["state_mask"].sum()) == 16 * len(example.program)
    assert aligned["state_targets"].max() < structured.N_STATE_CLASSES


def test_free_generation_records_exclude_gold_trace_channels(tmp_path):
    example = small_dataset()["train"][0]
    inference = example.inference_record()
    assert inference == {"example_id": example.example_id, "prompt": example.prompt}
    assert example.full_text == example.prompt + example.target_text
    record = example.to_record()
    assert "trace" not in record and "route_targets" not in record and "state_targets" not in record
    program, initial = structured.parse_prompt(example.prompt)
    assert program == example.program and initial == example.initial
    files = structured.save_structured_jsonl({"train": [example]}, tmp_path)
    text = Path(files["train"]["path"]).read_text()
    loaded = json.loads(text)
    assert loaded["target_text"] == example.target_text
    assert "trace" not in loaded and "route_mask" not in loaded and "state_mask" not in loaded


def test_exact_lookup_and_supplied_program_executor_baselines():
    dataset = small_dataset()
    report = structured.baseline_report(dataset)
    assert report["exact_lookup"]["train"]["exact_match"] == 1.0
    for split in ("dev", "unseen_composition", "longer_length"):
        assert report["exact_lookup"][split]["exact_match"] == 0.0
        assert report["exact_lookup"][split]["missing"] == len(dataset[split])
    for split in structured.SPLIT_NAMES:
        assert report["supplied_program_executor"][split]["exact_match"] == 1.0
    example = dataset["dev"][0]
    assert structured.SuppliedProgramExecutorBaseline().predict(example.prompt) == example.target_text


def test_structured_only_preparation_runs_without_network(tmp_path):
    from native import data, prepare_data

    defaults = prepare_data.parser().parse_args(["--structured-only"])
    assert defaults.code_generation_count == data.CODE_GENERATION_DEFAULT_COUNT == 64
    assert defaults.code_generation_target_chars == data.CODE_GENERATION_DEFAULT_TARGET_CHARS == 256
    status = prepare_data.main([
        "--structured-only", "--output", str(tmp_path), "--structured-train", "3",
        "--structured-dev", "2", "--structured-unseen", "2", "--structured-longer", "2",
    ])
    assert status == 0
    report = json.loads((tmp_path / "structured" / "baselines.json").read_text())
    assert all(report["supplied_program_executor"][split]["exact_match"] == 1.0
               for split in structured.SPLIT_NAMES)
    assert not (tmp_path / "train").exists()
    assert not (tmp_path / "code_generation.jsonl").exists()
