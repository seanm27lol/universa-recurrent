"""Construction-only checks for the varied native executable-program study."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
from native import prepare_program_study, program_study, structured  # noqa: E402
from native.independent.rescore_generations import execute_saved_program  # noqa: E402


def small_counts():
    return {"train": 16, "dev": 8, "initial_values": 8, "unseen_composition": 8, "longer_length": 8}


def small_dataset(seed=17):
    return program_study.generate_dataset(**small_counts(), seed=seed)


def program_key(example):
    return tuple(str(instruction) for instruction in example.program)


def pair_key(example):
    return program_key(example), example.initial


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
