"""Executable register-machine examples with trace-aligned auxiliary targets.

A prompt supplies four registers and a short program.  The free-generation target is
only the final register line.  During data construction, the independent executor
below supplies four pass labels and four register-state labels at each instruction's
last token.  Those labels are external training auxiliaries: inference records contain
only an example id and prompt, never a program trace or auxiliary tensor.

Arithmetic is modulo ``N_STATE_CLASSES``.  The held-out composition contains an
immediate ``multiply`` -> ``add`` transition, which train and ordinary dev never
contain.  The length holdout uses six to eight instructions; train and dev use two to
four.  These are synthetic construction checks, not evidence of learned reasoning.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

N_PASSES = 4
N_EXPERTS = 4
N_SLOTS = 4
N_STATE_CLASSES = 32
SPLIT_NAMES = ("train", "dev", "unseen_composition", "longer_length")
RESERVED_COMPOSITION = ("multiply", "add")
OP_ROUTES = {"set": 0, "add": 1, "multiply": 2, "copy": 3}
PASS_STAGES = ("decode", "read", "compute", "commit")


def _register(value: int) -> str:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < N_SLOTS:
        raise ValueError(f"register must be an integer in [0, 4), got {value!r}")
    return f"r{value}"


@dataclass(frozen=True)
class Instruction:
    """One canonical instruction: ``set dst value`` or ``op dst src``."""

    op: str
    dst: int
    src: int | None = None
    value: int | None = None

    def __post_init__(self) -> None:
        op = {"mul": "multiply"}.get(self.op, self.op)
        object.__setattr__(self, "op", op)
        if op not in OP_ROUTES:
            raise ValueError(f"unknown operation {self.op!r}")
        _register(self.dst)
        if op == "set":
            if self.src is not None or not isinstance(self.value, int) or isinstance(self.value, bool):
                raise ValueError("set requires an integer immediate and no source register")
            if not 0 <= self.value < N_STATE_CLASSES:
                raise ValueError(f"immediate must be in [0, {N_STATE_CLASSES})")
        elif self.value is not None or self.src is None:
            raise ValueError(f"{op} requires a source register and no immediate")
        else:
            _register(self.src)

    @classmethod
    def parse(cls, text: str) -> "Instruction":
        parts = text.strip().split()
        if len(parts) == 3 and parts[0] == "set":
            return cls("set", int(parts[1][1:]), value=int(parts[2]))
        if len(parts) == 3 and parts[0] in {"add", "multiply", "mul", "copy"}:
            return cls(parts[0], int(parts[1][1:]), src=int(parts[2][1:]))
        raise ValueError(f"invalid instruction {text!r}")

    def __str__(self) -> str:
        if self.op == "set":
            return f"set {_register(self.dst)} {self.value}"
        return f"{self.op} {_register(self.dst)} {_register(self.src)}"


@dataclass(frozen=True)
class ExecutionEvent:
    """The state before/after one instruction and its four semantic pass routes."""

    instruction_index: int
    instruction: Instruction
    before: tuple[int, ...]
    after: tuple[int, ...]

    @property
    def routes(self) -> tuple[int, int, int, int]:
        operation = OP_ROUTES[self.instruction.op]
        return operation, 0, operation, OP_ROUTES["copy"]

    @property
    def states(self) -> tuple[tuple[int, ...], ...]:
        return self.before, self.before, self.before, self.after


@dataclass(frozen=True)
class Execution:
    initial: tuple[int, ...]
    final: tuple[int, ...]
    events: tuple[ExecutionEvent, ...]
    target_text: str


class RegisterMachine:
    """Four-register machine with modular add/multiply and copy/set state updates."""

    def __init__(self, modulus: int = N_STATE_CLASSES):
        if modulus != N_STATE_CLASSES:
            raise ValueError(f"this trace schema has exactly {N_STATE_CLASSES} state classes")
        self.modulus = modulus

    def execute(self, program: Sequence[Instruction | str], initial: Sequence[int]) -> Execution:
        registers = tuple(initial)
        if len(registers) != N_SLOTS or any(
                not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < self.modulus for v in registers):
            raise ValueError(f"initial state must contain four integers in [0, {self.modulus})")
        instructions = tuple(Instruction.parse(p) if isinstance(p, str) else p for p in program)
        if not instructions:
            raise ValueError("a program must contain at least one instruction")
        events = []
        state = registers
        for index, instruction in enumerate(instructions):
            before = state
            values = list(state)
            if instruction.op == "set":
                values[instruction.dst] = instruction.value
            elif instruction.op == "add":
                values[instruction.dst] = (state[instruction.dst] + state[instruction.src]) % self.modulus
            elif instruction.op == "multiply":
                values[instruction.dst] = (state[instruction.dst] * state[instruction.src]) % self.modulus
            else:
                values[instruction.dst] = state[instruction.src]
            state = tuple(values)
            events.append(ExecutionEvent(index, instruction, before, state))
        return Execution(registers, state, tuple(events), format_target(state))


def execute(program: Sequence[Instruction | str], initial: Sequence[int]) -> Execution:
    return RegisterMachine().execute(program, initial)


def format_target(registers: Sequence[int]) -> str:
    if len(registers) != N_SLOTS:
        raise ValueError("exactly four registers are required")
    return " ".join(f"r{i}={int(value)}" for i, value in enumerate(registers))


def format_prompt(program: Sequence[Instruction], initial: Sequence[int]) -> str:
    lines = [
        f"Register machine: four registers, arithmetic modulo {N_STATE_CLASSES}.",
        f"Initial: {format_target(initial)}",
        "Program:",
        *(str(instruction) for instruction in program),
        "Answer with the final four registers.",
        "Answer:\n",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class StructuredExample:
    example_id: str
    split: str
    prompt: str
    target_text: str
    program: tuple[Instruction, ...]
    initial: tuple[int, ...]
    execution: Execution = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"unknown structured split {self.split!r}")
        execution = execute(self.program, self.initial)
        if execution.target_text != self.target_text:
            raise ValueError(f"{self.example_id}: target text does not match execution")
        object.__setattr__(self, "execution", execution)

    @property
    def full_text(self) -> str:
        return self.prompt + self.target_text

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.full_text.encode("utf-8")).hexdigest()

    def inference_record(self) -> dict:
        """The complete model-facing free-generation input; gold fields are absent."""
        return {"example_id": self.example_id, "prompt": self.prompt}

    def to_record(self) -> dict:
        """Serializable scoring record; traces are deliberately not serialized."""
        return {
            "example_id": self.example_id,
            "split": self.split,
            "prompt": self.prompt,
            "target_text": self.target_text,
            "program": [str(instruction) for instruction in self.program],
            "initial": list(self.initial),
            "modulus": N_STATE_CLASSES,
        }


def make_example(split: str, program: Sequence[Instruction], initial: Sequence[int]) -> StructuredExample:
    program = tuple(program)
    initial = tuple(initial)
    prompt = format_prompt(program, initial)
    target = execute(program, initial).target_text
    identity = hashlib.sha256((prompt + target).encode("utf-8")).hexdigest()[:20]
    return StructuredExample(f"structured-{split}-{identity}", split, prompt, target, program, initial)


def composition_pairs(program: Sequence[Instruction]) -> set[tuple[str, str]]:
    return {(a.op, b.op) for a, b in zip(program, program[1:])}


def _program(rng: random.Random, length: int, *, unseen: bool) -> tuple[Instruction, ...]:
    operations = tuple(OP_ROUTES)
    ops = [rng.choice(operations) for _ in range(length)]
    if unseen:
        start = rng.randrange(length - 1)
        ops[start:start + 2] = RESERVED_COMPOSITION
    elif any((a, b) == RESERVED_COMPOSITION for a, b in zip(ops, ops[1:])):
        raise _Retry
    instructions = []
    for op in ops:
        dst = rng.randrange(N_SLOTS)
        if op == "set":
            instructions.append(Instruction(op, dst, value=rng.randrange(N_STATE_CLASSES)))
        else:
            instructions.append(Instruction(op, dst, src=rng.randrange(N_SLOTS)))
    return tuple(instructions)


class _Retry(Exception):
    pass


def generate_dataset(*, train: int = 1024, dev: int = 128, unseen_composition: int = 128,
                     longer_length: int = 128, seed: int = 20261108) -> dict[str, list[StructuredExample]]:
    """Generate deterministic, globally disjoint prompts and the two explicit holdouts."""
    counts = dict(zip(SPLIT_NAMES, (train, dev, unseen_composition, longer_length)))
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts.values()):
        raise ValueError("split sizes must be nonnegative integers")
    rng = random.Random(seed)
    result = {name: [] for name in SPLIT_NAMES}
    seen = set()
    for split in SPLIT_NAMES:
        attempts = 0
        while len(result[split]) < counts[split]:
            attempts += 1
            if attempts > max(1000, counts[split] * 1000):
                raise RuntimeError(f"could not generate {counts[split]} unique {split} examples")
            if split == "longer_length":
                length = rng.randint(6, 8)
            elif split == "unseen_composition":
                length = rng.randint(3, 5)
            else:
                length = rng.randint(2, 4)
            try:
                program = _program(rng, length, unseen=split == "unseen_composition")
            except _Retry:
                continue
            initial = tuple(rng.randrange(N_STATE_CLASSES) for _ in range(N_SLOTS))
            key = (tuple(str(instruction) for instruction in program), initial)
            if key in seen:
                continue
            seen.add(key)
            result[split].append(make_example(split, program, initial))
    assert_holdout_separation(result)
    return result


def assert_holdout_separation(dataset: Mapping[str, Sequence[StructuredExample]]) -> None:
    """Fail if prompt identity or either declared holdout leaks into ordinary data."""
    prompts = [example.prompt for split in SPLIT_NAMES for example in dataset.get(split, ())]
    if len(prompts) != len(set(prompts)):
        raise AssertionError("structured splits share a prompt")
    ordinary = [*dataset.get("train", ()), *dataset.get("dev", ())]
    if any(RESERVED_COMPOSITION in composition_pairs(example.program) for example in ordinary):
        raise AssertionError("unseen composition leaked into train/dev")
    if any(not 2 <= len(example.program) <= 4 for example in ordinary):
        raise AssertionError("ordinary train/dev program length is outside [2, 4]")
    for example in dataset.get("unseen_composition", ()):
        if RESERVED_COMPOSITION not in composition_pairs(example.program):
            raise AssertionError("unseen-composition example lacks the reserved transition")
    if any(not 6 <= len(example.program) <= 8 for example in dataset.get("longer_length", ())):
        raise AssertionError("length-holdout program is outside [6, 8]")


def encode_with_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Encode through a tokenizers or Transformers fast tokenizer without special tokens."""
    if callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise ValueError("expected one sequence")
            ids, offsets = ids[0], offsets[0]
        return list(ids), [tuple(map(int, offset)) for offset in offsets]
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if not hasattr(encoded, "ids") or not hasattr(encoded, "offsets"):
        raise TypeError("tokenizer must return token ids and character offsets")
    return list(encoded.ids), [tuple(map(int, offset)) for offset in encoded.offsets]


def align_example(example: StructuredExample, tokenizer, *, eos_token_id: int | None = None) -> dict:
    """Attach each execution event to the last token overlapping its source line.

    Route arrays use ``[token, pass]`` and state arrays ``[token, pass, slot]``.
    Unlabelled entries are zero with a false mask; no trace is an inference input.
    """
    ids, offsets = encode_with_offsets(tokenizer, example.full_text)
    if eos_token_id is not None:
        ids.append(int(eos_token_id))
        offsets.append((len(example.full_text), len(example.full_text)))
    length = len(ids)
    route_targets = np.zeros((length, N_PASSES), dtype=np.int8)
    route_mask = np.zeros((length, N_PASSES), dtype=np.bool_)
    state_targets = np.zeros((length, N_PASSES, N_SLOTS), dtype=np.int16)
    state_mask = np.zeros((length, N_PASSES, N_SLOTS), dtype=np.bool_)
    anchors = []
    cursor = 0
    for event in example.execution.events:
        line = str(event.instruction)
        start = example.prompt.find(line, cursor)
        if start < 0:
            raise ValueError(f"{example.example_id}: cannot locate {line!r}")
        end = start + len(line)
        cursor = end
        overlaps = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
        if not overlaps:
            raise ValueError(f"{example.example_id}: no token overlaps {line!r}")
        anchor = overlaps[-1]
        if anchor in anchors:
            raise ValueError(f"{example.example_id}: two instructions share token {anchor}")
        anchors.append(anchor)
        route_targets[anchor] = event.routes
        route_mask[anchor] = True
        for pass_index, state in enumerate(event.states):
            state_targets[anchor, pass_index] = state
            state_mask[anchor, pass_index] = True
    return {
        "input_ids": np.asarray(ids, dtype=np.int32),
        "route_targets": route_targets,
        "route_mask": route_mask,
        "state_targets": state_targets,
        "state_mask": state_mask,
        "anchors": np.asarray(anchors, dtype=np.int32),
    }


class ExactLookupBaseline:
    """Memorize training prompt/target pairs; unseen prompts receive no prediction."""

    def __init__(self):
        self.table = {}

    def fit(self, examples: Iterable[StructuredExample]) -> "ExactLookupBaseline":
        for example in examples:
            previous = self.table.setdefault(example.prompt, example.target_text)
            if previous != example.target_text:
                raise ValueError("conflicting targets for one lookup prompt")
        return self

    def predict(self, prompt: str) -> str | None:
        return self.table.get(prompt)


class SuppliedProgramExecutorBaseline:
    """Execute the supplied reference program; this baseline is not the neural model."""

    def predict(self, example_or_prompt: StructuredExample | str) -> str:
        if isinstance(example_or_prompt, StructuredExample):
            return execute(example_or_prompt.program, example_or_prompt.initial).target_text
        program, initial = parse_prompt(example_or_prompt)
        return execute(program, initial).target_text


def evaluate_baseline(baseline, examples: Sequence[StructuredExample]) -> dict:
    correct = 0
    missing = 0
    for example in examples:
        query = example.prompt if isinstance(baseline, ExactLookupBaseline) else example
        prediction = baseline.predict(query)
        missing += prediction is None
        correct += prediction == example.target_text
    total = len(examples)
    return {"examples": total, "correct": correct, "missing": missing,
            "exact_match": correct / total if total else None}


def baseline_report(dataset: Mapping[str, Sequence[StructuredExample]]) -> dict:
    lookup = ExactLookupBaseline().fit(dataset.get("train", ()))
    executor = SuppliedProgramExecutorBaseline()
    return {
        "metric": "exact free-generation target-text match",
        "exact_lookup": {split: evaluate_baseline(lookup, dataset.get(split, ())) for split in SPLIT_NAMES},
        "supplied_program_executor": {
            split: evaluate_baseline(executor, dataset.get(split, ())) for split in SPLIT_NAMES},
        "boundary": "the executor receives reference programs; the neural model does not",
    }


def parse_prompt(prompt: str) -> tuple[tuple[Instruction, ...], tuple[int, ...]]:
    """Parse the canonical supplied-program format for the executor baseline."""
    initial_match = re.search(r"(?m)^Initial: r0=(\d+) r1=(\d+) r2=(\d+) r3=(\d+)$", prompt)
    if initial_match is None:
        raise ValueError("prompt has no canonical initial register line")
    try:
        body = prompt.split("Program:\n", 1)[1].split("Answer with", 1)[0]
    except IndexError as error:
        raise ValueError("prompt has no canonical program block") from error
    program = tuple(Instruction.parse(line) for line in body.splitlines() if line)
    if not program:
        raise ValueError("prompt has an empty program")
    return program, tuple(map(int, initial_match.groups()))


def save_structured_jsonl(dataset: Mapping[str, Sequence[StructuredExample]], directory: str | os.PathLike) -> dict:
    """Write deterministic scoring records, without execution traces, and return their hashes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in SPLIT_NAMES:
        path = directory / f"{split}.jsonl"
        text = "".join(json.dumps(example.to_record(), sort_keys=True, ensure_ascii=False) + "\n"
                       for example in dataset.get(split, ()))
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
        files[split] = {"path": str(path), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "examples": len(dataset.get(split, ()))}
    return files
