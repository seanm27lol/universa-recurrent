"""Construction-only checks for the varied native executable-program study."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
from native import data, prepare_program_study, program_study, structured  # noqa: E402
from native.independent.rescore_generations import execute_saved_program  # noqa: E402


def small_counts():
    return {"train": 16, "dev": 8, "initial_values": 8, "unseen_composition": 8, "longer_length": 8}


def small_dataset(seed=17):
    return program_study.generate_dataset(**small_counts(), seed=seed)


def program_key(example):
    return tuple(str(instruction) for instruction in example.program)


def pair_key(example):
    return program_key(example), example.initial


def tiny_tokenizer(dataset):
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(
        [example.full_text for split in program_study.SPLIT_NAMES for example in dataset[split]],
        trainers.WordLevelTrainer(special_tokens=["[UNK]"]),
    )
    return tokenizer


def test_defaults_describe_the_real_study_without_changing_legacy_position_default():
    args = prepare_program_study.parser().parse_args(["--output", "out", "--tokenizer", "tokenizer.json"])
    assert (args.train, args.dev, args.initial_values) == (4096, 128, 128)
    assert (args.unseen_composition, args.longer_length) == (128, 128)
    assert args.seed == 20261112
    assert args.position_policy == "document"
    assert args.train_evaluation_count == 128 and args.intervention_count == 32
    from native import prepare_data
    assert prepare_data.parser().parse_args([]).position_policy == "row"


def test_small_population_is_reproducible_disjoint_balanced_and_independently_executable():
    first = small_dataset()
    second = small_dataset()
    assert {split: [example.to_record() for example in examples] for split, examples in first.items()} == {
        split: [example.to_record() for example in examples] for split, examples in second.items()}
    report = program_study.assert_program_study_invariants(first, expected_counts=small_counts())
    assert report["globally_unique_program_initial_pairs"] == 48
    assert report["globally_unique_prompt_target_contents"] == 48
    assert report["unique_instruction_sequences_outside_initial_values"] == 40
    assert program_study.validate_targets(first)["executed_programs"] == 48
    assert structured.SPLIT_NAMES == ("train", "dev", "unseen_composition", "longer_length")

    all_pairs = [pair_key(example) for examples in first.values() for example in examples]
    all_contents = [example.full_text for examples in first.values() for example in examples]
    assert len(all_pairs) == len(set(all_pairs))
    assert len(all_contents) == len(set(all_contents))
    train_programs = {program_key(example) for example in first["train"]}
    ordinary_programs = [program_key(example) for split in ("train", "dev", "unseen_composition", "longer_length")
                         for example in first[split]]
    assert len(ordinary_programs) == len(set(ordinary_programs))

    for split, examples in first.items():
        assert len(examples) == small_counts()[split]
        assert Counter(example.program[-1].op for example in examples) == {
            operation: len(examples) // 4 for operation in structured.OP_ROUTES}
        for example in examples:
            execute_saved_program(example.to_record())
            has_reserved = structured.RESERVED_COMPOSITION in structured.composition_pairs(example.program)
            assert has_reserved == (split == "unseen_composition")
            if split == "longer_length":
                assert 8 <= len(example.program) <= 12
            else:
                assert 2 <= len(example.program) <= 6
            if split == "initial_values":
                assert all(24 <= value < 32 for value in example.initial)
                assert program_key(example) in train_programs
            else:
                assert all(0 <= value < 24 for value in example.initial)


def test_balanced_selection_records_only_deterministic_ids_and_construction_baselines():
    dataset = small_dataset()
    first = program_study.select_subsets(
        dataset, seed=17, train_evaluation_count=8, intervention_count=4)
    second = program_study.select_subsets(
        dataset, seed=17, train_evaluation_count=8, intervention_count=4)
    assert first == second
    assert first["outcomes_recorded"] is False
    for split in program_study.SPLIT_NAMES:
        evaluation = first["evaluation_example_ids"][split]
        intervention = first["intervention_example_ids"][split]
        assert len(evaluation) == len(set(evaluation)) == 8
        assert len(intervention) == len(set(intervention)) == 4
        assert set(first["evaluation_final_operation_counts"][split].values()) == {2}
        assert set(first["intervention_final_operation_counts"][split].values()) == {1}
        split_ids = {example.example_id for example in dataset[split]}
        assert set(evaluation) <= split_ids and set(intervention) <= split_ids
    baselines = program_study.baseline_report(dataset)
    assert baselines["exact_lookup"]["train"]["exact_match"] == 1.0
    for split in program_study.SPLIT_NAMES[1:]:
        assert baselines["exact_lookup"][split]["exact_match"] == 0.0
    for split in program_study.SPLIT_NAMES:
        assert baselines["supplied_program_executor"][split]["exact_match"] == 1.0
    assert "no model" in baselines["evidence_boundary"]


def test_unbalanced_counts_and_oversized_subsets_are_refused():
    with pytest.raises(ValueError, match="multiple of four"):
        program_study.generate_dataset(**{**small_counts(), "dev": 6})
    dataset = small_dataset()
    with pytest.raises(ValueError, match="requested 12"):
        program_study.select_subsets(dataset, train_evaluation_count=12, intervention_count=12)


def test_actual_small_builder_packs_every_answer_eos_and_document_position(tmp_path, monkeypatch, capsys):
    dataset = small_dataset()
    tokenizer = tiny_tokenizer(dataset)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("local tokenizer test double\n", encoding="utf-8")
    monkeypatch.setattr(prepare_program_study.D, "load_gpt2_tokenizer", lambda path: tokenizer)
    output = tmp_path / "built"
    argv = [
        "--output", str(output), "--tokenizer", str(tokenizer_path),
        "--train", "16", "--dev", "8", "--initial-values", "8",
        "--unseen-composition", "8", "--longer-length", "8", "--seed", "17",
        "--train-evaluation-count", "8", "--intervention-count", "4",
    ]
    assert prepare_program_study.main(argv) == 0
    cli = json.loads(capsys.readouterr().out)
    manifest = json.loads((output / "manifest.json").read_text())
    audit = json.loads((output / "audit.json").read_text())
    assert manifest["counts"] == small_counts()
    assert manifest["packing"] == "intact"
    assert manifest["position_policy"] == "document"
    assert manifest["structured_overflow"] == "error"
    assert audit["manifest_sha256"] == data.sha256_file(output / "manifest.json")
    assert audit["target_validation"]["executed_programs"] == 48
    assert audit["construction_invariants"]["globally_unique_prompt_target_contents"] == 48
    assert audit["source_identity"]["qwen_or_old_evaluation_populations_used"] is False
    assert "no training" in audit["evidence_boundary"]

    expected_train_loss = 0
    for split in program_study.SPLIT_NAMES:
        values = manifest["splits"][split]
        checks = values["construction_checks"]
        assert values["examples"] == small_counts()[split]
        assert values["structured_examples_intact"] == small_counts()[split]
        assert values["long_examples_excluded"] == 0
        assert checks["documents_dropped"] == checks["documents_split"] == 0
        assert checks["eos_tokens_supervised"] == small_counts()[split]
        assert checks["answer_tokens_supervised"] > 0
        assert checks["padding_label_leakage_tokens"] == 0
        assert checks["noncausal_or_mismatched_labels"] == 0
        assert checks["maximum_document_tokens"] <= 256
        assert cli[split]["rows"] == values["rows"]
        assert cli[split]["loss_tokens"] == values["loss_tokens"]
        arrays = {name: np.load(output / split / f"{name}.npy", allow_pickle=False) for name in data.ARRAY_NAMES}
        assert np.all(arrays["labels"][arrays["key_padding_mask"]] == data.IGNORE_INDEX)
        consumed = [json.loads(line) for line in (output / f"{split}_consumed_documents.jsonl").read_text().splitlines()]
        assert len(consumed) == small_counts()[split]
        expected_loss = 0
        for item in consumed:
            row, start, end = item["row"], item["start_column"], item["end_column_exclusive"]
            length = end - start
            assert item["tokens_total"] == item["tokens_consumed"] == length
            assert item["tokens_unconsumed"] == 0 and item["partial"] is False
            assert np.count_nonzero(arrays["doc_ids"] == item["doc_id"]) == length
            assert arrays["positions"][row, start:end].tolist() == list(range(length))
            assert arrays["labels"][row, start] == data.IGNORE_INDEX
            assert arrays["labels"][row, start + 1:end].tolist() == arrays["input_ids"][row, start + 1:end].tolist()
            for column in [*item["answer_token_columns"], item["eos_token_column"]]:
                assert arrays["labels"][row, column] == arrays["input_ids"][row, column]
                assert arrays["labels"][row, column] != data.IGNORE_INDEX
            assert arrays["labels"][row, item["eos_token_column"]] == data.EOS_TOKEN_ID
            expected_loss += length - 1
        assert values["loss_tokens"] == expected_loss
        assert values["tokens"] + values["padding_tokens"] == values["rows"] * 256
        assert values["consumed_document_sha256"] == data.sha256_file(output / f"{split}_consumed_documents.jsonl")
        for name, entry in values["arrays"].items():
            assert entry["sha256"] == data.sha256_file(output / split / f"{name}.npy")
        if split == "train":
            expected_train_loss = expected_loss

    assert manifest["train_epoch"]["loss_tokens_per_epoch"] == expected_train_loss
    assert manifest["train_epoch"]["updates_per_epoch"] == math.ceil(manifest["splits"]["train"]["rows"] / 8)
    assert cli["train_epoch"] == manifest["train_epoch"]
    for relative, expected_hash in audit["artifact_sha256"].items():
        assert data.sha256_file(output / relative) == expected_hash
    with pytest.raises(RuntimeError, match="nonempty"):
        prepare_program_study.main(argv)
