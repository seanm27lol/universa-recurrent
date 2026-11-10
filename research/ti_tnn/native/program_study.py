"""Generate varied synthetic register programs with explicit generalization holdouts.

The checks in this module establish data construction properties only. They do not
train a model or provide evidence that a model learns any of the held-out programs.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from . import structured
from .independent.rescore_generations import execute_saved_program

DEFAULT_SEED = 20261112
SPLIT_NAMES = ("train", "dev", "initial_values", "unseen_composition", "longer_length")
DEFAULT_COUNTS = {
    "train": 4096,
    "dev": 128,
    "initial_values": 128,
    "unseen_composition": 128,
    "longer_length": 128,
}
FINAL_OPERATIONS = tuple(structured.OP_ROUTES)
LOW_INITIAL_RANGE = (0, 24)
HIGH_INITIAL_RANGE = (24, 32)
PROGRAM_LENGTHS = {
    "train": (2, 6),
    "dev": (2, 6),
    "initial_values": (2, 6),
    "unseen_composition": (2, 6),
    "longer_length": (8, 12),
}


def _program_key(program: Sequence[structured.Instruction]) -> tuple[str, ...]:
    return tuple(str(instruction) for instruction in program)


def _pair_key(example: structured.StructuredExample) -> tuple[tuple[str, ...], tuple[int, ...]]:
    return _program_key(example.program), example.initial


def _make_example(split: str, program: Sequence[structured.Instruction],
                  initial: Sequence[int]) -> structured.StructuredExample:
    if split != "initial_values":
        return structured.make_example(split, program, initial)
    original_splits = structured.SPLIT_NAMES
    try:
        structured.SPLIT_NAMES = (*original_splits, "initial_values")
        return structured.make_example(split, program, initial)
    finally:
        structured.SPLIT_NAMES = original_splits


def _validate_counts(counts: Mapping[str, int]) -> None:
    if set(counts) != set(SPLIT_NAMES):
        raise ValueError(f"counts must contain exactly {SPLIT_NAMES}")
    for split, count in counts.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 4 or count % 4:
            raise ValueError(f"{split}: count must be a positive multiple of four, got {count!r}")


def _sample_program(rng: random.Random, length: int, final_operation: str,
                    *, reserved: bool) -> tuple[structured.Instruction, ...]:
    operations = [rng.choice(FINAL_OPERATIONS) for _ in range(length - 1)]
    operations.append(final_operation)
    has_reserved = structured.RESERVED_COMPOSITION in {
        (left, right) for left, right in zip(operations, operations[1:])}
    if has_reserved != reserved:
        raise ValueError("sampled program does not have the requested composition status")
    instructions = []
    for operation in operations:
        destination = rng.randrange(structured.N_SLOTS)
        if operation == "set":
            instructions.append(structured.Instruction(
                operation, destination, value=rng.randrange(structured.N_STATE_CLASSES)))
        else:
            instructions.append(structured.Instruction(
                operation, destination, src=rng.randrange(structured.N_SLOTS)))
    return tuple(instructions)


def _generate_balanced_split(split: str, count: int, rng: random.Random,
                             seen_programs: set, seen_pairs: set) -> list:
    examples = []
    quota = count // len(FINAL_OPERATIONS)
    minimum_length, maximum_length = PROGRAM_LENGTHS[split]
    for final_operation in FINAL_OPERATIONS:
        accepted = 0
        attempts = 0
        while accepted < quota:
            attempts += 1
            if attempts > max(10_000, quota * 2_000):
                raise RuntimeError(f"could not generate {quota} unique {split}/{final_operation} examples")
            length = rng.randint(minimum_length, maximum_length)
            try:
                program = _sample_program(
                    rng, length, final_operation, reserved=split == "unseen_composition")
            except ValueError:
                continue
            initial = tuple(rng.randrange(*LOW_INITIAL_RANGE) for _ in range(structured.N_SLOTS))
            program_key = _program_key(program)
            pair_key = (program_key, initial)
            if program_key in seen_programs or pair_key in seen_pairs:
                continue
            seen_programs.add(program_key)
            seen_pairs.add(pair_key)
            examples.append(_make_example(split, program, initial))
            accepted += 1
    rng.shuffle(examples)
    return examples


def _generate_initial_values(train: Sequence[structured.StructuredExample], count: int,
                             rng: random.Random, seen_pairs: set) -> list:
    programs = {operation: [] for operation in FINAL_OPERATIONS}
    for example in train:
        programs[example.program[-1].op].append(example.program)
    examples = []
    quota = count // len(FINAL_OPERATIONS)
    for final_operation in FINAL_OPERATIONS:
        candidates = programs[final_operation]
        if len(candidates) < quota:
            raise RuntimeError(f"initial_values needs {quota} distinct train programs ending in {final_operation}")
        rng.shuffle(candidates)
        for program in candidates[:quota]:
            initial = tuple(rng.randrange(*HIGH_INITIAL_RANGE) for _ in range(structured.N_SLOTS))
            pair_key = (_program_key(program), initial)
            if pair_key in seen_pairs:
                raise AssertionError("high initial values unexpectedly overlap an existing program/initial pair")
            seen_pairs.add(pair_key)
            examples.append(_make_example("initial_values", program, initial))
    rng.shuffle(examples)
    return examples


def generate_dataset(*, train: int = DEFAULT_COUNTS["train"], dev: int = DEFAULT_COUNTS["dev"],
                     initial_values: int = DEFAULT_COUNTS["initial_values"],
                     unseen_composition: int = DEFAULT_COUNTS["unseen_composition"],
                     longer_length: int = DEFAULT_COUNTS["longer_length"],
                     seed: int = DEFAULT_SEED) -> dict[str, list[structured.StructuredExample]]:
    """Generate fresh deterministic programs, balanced by each program's final operation."""
    counts = {
        "train": train,
        "dev": dev,
        "initial_values": initial_values,
        "unseen_composition": unseen_composition,
        "longer_length": longer_length,
    }
    _validate_counts(counts)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    rng = random.Random(seed)
    seen_programs = set()
    seen_pairs = set()
    dataset = {split: [] for split in SPLIT_NAMES}
    for split in ("train", "dev", "unseen_composition", "longer_length"):
        dataset[split] = _generate_balanced_split(
            split, counts[split], rng, seen_programs, seen_pairs)
    dataset["initial_values"] = _generate_initial_values(
        dataset["train"], counts["initial_values"], rng, seen_pairs)
    assert_program_study_invariants(dataset, expected_counts=counts)
    validate_targets(dataset)
    return dataset


def _operation_counts(examples: Sequence[structured.StructuredExample]) -> dict[str, int]:
    counts = Counter(example.program[-1].op for example in examples)
    return {operation: counts[operation] for operation in FINAL_OPERATIONS}


def assert_program_study_invariants(dataset: Mapping[str, Sequence[structured.StructuredExample]],
                                    *, expected_counts: Mapping[str, int] | None = None) -> dict:
    """Check split ranges, disjointness, final-operation balance, and content uniqueness."""
    if set(dataset) != set(SPLIT_NAMES):
        raise AssertionError(f"dataset must contain exactly {SPLIT_NAMES}")
    if expected_counts is not None:
        _validate_counts(expected_counts)
    all_pairs = []
    all_full_text = []
    all_ids = []
    ordinary_programs = []
    train_programs = {_program_key(example.program) for example in dataset["train"]}
    report = {"splits": {}, "globally_unique_program_initial_pairs": 0,
              "globally_unique_prompt_target_contents": 0}
    for split in SPLIT_NAMES:
        examples = dataset[split]
        count = len(examples)
        if count < 4 or count % 4:
            raise AssertionError(f"{split}: count must be a positive multiple of four")
        if expected_counts is not None and count != expected_counts[split]:
            raise AssertionError(f"{split}: got {count} examples, expected {expected_counts[split]}")
        operation_counts = _operation_counts(examples)
        if set(operation_counts.values()) != {count // 4}:
            raise AssertionError(f"{split}: final operations are not balanced: {operation_counts}")
        minimum_length, maximum_length = PROGRAM_LENGTHS[split]
        for example in examples:
            if example.split != split:
                raise AssertionError(f"{example.example_id}: record split does not match {split}")
            if not minimum_length <= len(example.program) <= maximum_length:
                raise AssertionError(f"{example.example_id}: program length is outside the split contract")
            expected_initial = HIGH_INITIAL_RANGE if split == "initial_values" else LOW_INITIAL_RANGE
            if len(example.initial) != 4 or any(
                    not expected_initial[0] <= value < expected_initial[1] for value in example.initial):
                raise AssertionError(f"{example.example_id}: initial values violate {expected_initial}")
            has_reserved = structured.RESERVED_COMPOSITION in structured.composition_pairs(example.program)
            if has_reserved != (split == "unseen_composition"):
                raise AssertionError(f"{example.example_id}: reserved-composition membership is incorrect")
            if split == "initial_values" and _program_key(example.program) not in train_programs:
                raise AssertionError(f"{example.example_id}: instruction sequence is not present in train")
            all_pairs.append(_pair_key(example))
            all_full_text.append(example.full_text)
            all_ids.append(example.example_id)
            if split != "initial_values":
                ordinary_programs.append(_program_key(example.program))
        lengths = Counter(len(example.program) for example in examples)
        report["splits"][split] = {
            "examples": count,
            "final_operation_counts": operation_counts,
            "length_counts": {str(length): lengths[length] for length in sorted(lengths)},
            "initial_range": list(HIGH_INITIAL_RANGE if split == "initial_values" else LOW_INITIAL_RANGE),
            "reserved_composition_examples": sum(
                structured.RESERVED_COMPOSITION in structured.composition_pairs(example.program)
                for example in examples),
        }
    if len(all_pairs) != len(set(all_pairs)):
        raise AssertionError("a complete program/initial pair occurs more than once")
    if len(all_full_text) != len(set(all_full_text)):
        raise AssertionError("a complete prompt+target content occurs more than once")
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("an example id occurs more than once")
    if len(ordinary_programs) != len(set(ordinary_programs)):
        raise AssertionError("instruction-sequence reuse is allowed only for initial_values")
    report["globally_unique_program_initial_pairs"] = len(set(all_pairs))
    report["globally_unique_prompt_target_contents"] = len(set(all_full_text))
    report["unique_instruction_sequences_outside_initial_values"] = len(set(ordinary_programs))
    report["initial_values_instruction_sequences_seen_in_train"] = len(dataset["initial_values"])
    report["reserved_composition"] = list(structured.RESERVED_COMPOSITION)
    report["evidence_boundary"] = "construction invariants only; no learning result"
    return report


def validate_targets(dataset: Mapping[str, Sequence[structured.StructuredExample]]) -> dict:
    """Independently execute every saved-form record and compare its complete target."""
    count = 0
    for split in SPLIT_NAMES:
        for example in dataset[split]:
            execute_saved_program(example.to_record())
            count += 1
    return {
        "executor": "native.independent.rescore_generations.execute_saved_program",
        "executed_programs": count,
        "all_targets_match_independent_execution": True,
        "evidence_boundary": "target construction validation only; not model evaluation",
    }


def save_structured_jsonl(dataset: Mapping[str, Sequence[structured.StructuredExample]],
                          directory: str | os.PathLike) -> dict:
    """Write complete scoring records for all five roles without execution traces."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in SPLIT_NAMES:
        path = directory / f"{split}.jsonl"
        text = "".join(
            json.dumps(example.to_record(), sort_keys=True, ensure_ascii=False) + "\n"
            for example in dataset[split])
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
        files[split] = {
            "path": str(path),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "examples": len(dataset[split]),
        }
    return files


def baseline_report(dataset: Mapping[str, Sequence[structured.StructuredExample]]) -> dict:
    """Evaluate construction baselines, explicitly excluding neural-learning claims."""
    lookup = structured.ExactLookupBaseline().fit(dataset["train"])
    executor = structured.SuppliedProgramExecutorBaseline()
    return {
        "metric": "exact free-generation target-text match",
        "exact_lookup": {
            split: structured.evaluate_baseline(lookup, dataset[split]) for split in SPLIT_NAMES},
        "supplied_program_executor": {
            split: structured.evaluate_baseline(executor, dataset[split]) for split in SPLIT_NAMES},
        "boundary": "the executor receives reference programs; the neural model does not",
        "evidence_boundary": "construction baselines only; no model was trained or evaluated",
    }


def _selection_key(example: structured.StructuredExample, seed: int, role: str, split: str) -> str:
    material = f"native-program-study:{seed}:{role}:{split}:{example.content_sha256}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _balanced_subset(examples: Sequence[structured.StructuredExample], count: int,
                     *, seed: int, role: str, split: str) -> list:
    if not isinstance(count, int) or isinstance(count, bool) or count < 4 or count % 4:
        raise ValueError(f"{role}: subset size must be a positive multiple of four")
    if count > len(examples):
        raise ValueError(f"{role}: requested {count} examples from only {len(examples)} in {split}")
    selected = []
    quota = count // 4
    for operation in FINAL_OPERATIONS:
        candidates = [example for example in examples if example.program[-1].op == operation]
        candidates.sort(key=lambda example: (
            _selection_key(example, seed, role, split), example.content_sha256, example.example_id))
        if len(candidates) < quota:
            raise ValueError(f"{role}: {split}/{operation} has {len(candidates)} examples, needs {quota}")
        selected.extend(candidates[:quota])
    selected.sort(key=lambda example: (
        _selection_key(example, seed, role, split), example.content_sha256, example.example_id))
    return selected


def select_subsets(dataset: Mapping[str, Sequence[structured.StructuredExample]], *,
                   seed: int = DEFAULT_SEED, train_evaluation_count: int = 128,
                   intervention_count: int = 32) -> dict:
    """Select IDs deterministically; this records no outcome or model-dependent quantity."""
    evaluation = {}
    intervention = {}
    for split in SPLIT_NAMES:
        examples = dataset[split]
        if split == "train":
            evaluation_examples = _balanced_subset(
                examples, train_evaluation_count, seed=seed, role="evaluation", split=split)
        else:
            evaluation_examples = _balanced_subset(
                examples, len(examples), seed=seed, role="evaluation", split=split)
        intervention_examples = _balanced_subset(
            examples, intervention_count, seed=seed, role="intervention", split=split)
        evaluation[split] = [example.example_id for example in evaluation_examples]
        intervention[split] = [example.example_id for example in intervention_examples]
    return {
        "selection_rule": (
            "sort by sha256('native-program-study:<seed>:<role>:<split>:<content_sha256>') "
            "within each final-operation family, take equal quotas, then sort selected IDs by the same key"),
        "evaluation_example_ids": evaluation,
        "intervention_example_ids": intervention,
        "evaluation_final_operation_counts": {
            split: _operation_counts([
                example for example in dataset[split] if example.example_id in set(evaluation[split])])
            for split in SPLIT_NAMES},
        "intervention_final_operation_counts": {
            split: _operation_counts([
                example for example in dataset[split] if example.example_id in set(intervention[split])])
            for split in SPLIT_NAMES},
        "outcomes_recorded": False,
        "evidence_boundary": "deterministic example IDs only; no ad hoc outcomes",
    }
