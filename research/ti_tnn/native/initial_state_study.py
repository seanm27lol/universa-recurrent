"""Paired initial-state diversity data; construction only, never model execution.

repeat_one repeats one state eight times per exact instruction pattern; multi_state
shows eight distinct states. Patterns, document slots and their ordering match.
Historical JSONL inputs are used only as exclusion data, never as training examples.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from . import structured as S
from .independent.dependency_diagnostic import analyze_program
from .independent.rescore_generations import execute_saved_program

CONDITIONS = ("repeat_one", "multi_state")
POPULATIONS = ("train_anchor", "train_sample", "transition_values_id", "transition_values_ood",
               "program_values_id", "program_values_ood", "dev", "composition", "longer")


def pattern_key(program):
    return tuple(str(instruction) for instruction in program)


def pair_key(program, initial):
    return pattern_key(program), tuple(initial)


def read_history(paths):
    """Require explicit historical structured files; fail closed on missing/invalid data."""
    if not paths:
        raise ValueError("historical exclusion files are required")
    pairs, patterns, prompts, hashes = set(), set(), set(), {}
    for path in paths:
        path = Path(path).resolve()
        raw = path.read_bytes()
        if not raw.strip():
            raise ValueError(f"empty historical file: {path}")
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        for line in raw.decode().splitlines():
            record = json.loads(line)
            execute_saved_program(record)
            pairs.add((tuple(record["program"]), tuple(record["initial"])))
            patterns.add(tuple(record["program"]))
            prompts.add(record["prompt"])
    return {"pairs": pairs, "patterns": patterns, "prompts": prompts, "sha256": hashes}


def generate(history, *, seed=2026111201, composed_patterns=192, states_per_pattern=8,
             evaluation_count=128):
    """Generate a bounded pair and disjoint evaluation sets with identical treatments.

    All 64 atomic instructions are training patterns: 16 set instructions (four
    immediates per destination), and all 16 dst/src pairs for each other operation.
    Composed programs use only these exact primitives. Only the initial-state
    diversity differs between conditions; there is no staged curriculum.
    """
    if not 64 <= composed_patterns <= 512 or not 2 <= states_per_pattern <= 16:
        raise ValueError("bounded preparation requires 64..512 composed patterns and 2..16 states")
    if evaluation_count < 4 or evaluation_count > 128 or evaluation_count % 2:
        raise ValueError("evaluation_count must be even and in [4,128]")
    rng = random.Random(seed)
    primitives = [S.Instruction("set", dst, value=value) for dst in range(4)
                  for value in rng.sample(range(32), 4)]
    primitives += [S.Instruction(op, dst, src=src) for op in ("copy", "add", "multiply")
                   for dst in range(4) for src in range(4)]
    used_patterns = set(history["patterns"])
    used_pairs = set(history["pairs"])
    atomic = [(instruction,) for instruction in primitives]
    # Exact atomic patterns may have occurred historically, but no old state/prompt
    # can recur. Multi-instruction patterns are excluded wholesale below.
    used_patterns.update(pattern_key(p) for p in atomic)

    def has_reserved(program):
        return any(a.op == "multiply" and b.op == "add" for a, b in zip(program, program[1:]))

    def novel_program(length_range, composition=False):
        for _ in range(20000):
            length = rng.randint(*length_range)
            if composition:
                prefix = tuple(rng.choice(primitives) for _ in range(length - 2))
                multiply = rng.choice([i for i in primitives if i.op == "multiply"])
                dst = rng.choice([i for i in range(4) if i != multiply.dst])
                program = (*prefix, multiply, S.Instruction("add", dst, src=multiply.dst))
                if has_reserved(prefix + (multiply,)):
                    continue
                dependency = analyze_program(pattern_key(program))[dst]
                if length - 2 not in dependency["relevant_instructions"] or not dependency["depends_on_initial"]:
                    continue
            else:
                program = tuple(rng.choice(primitives) for _ in range(length))
                if has_reserved(program):
                    continue
            key = pattern_key(program)
            if key in used_patterns:
                continue
            if max(d["dependency_depth"] for d in analyze_program(key)) < 2:
                continue
            used_patterns.add(key)
            return program
        raise RuntimeError("bounded unique-program sampling exhausted")

    def fresh_example(program, high=False, split="dev"):
        for _ in range(20000):
            initial = tuple(rng.randrange(24, 32) if high else rng.randrange(24) for _ in range(4))
            key = pair_key(program, initial)
            if key in used_pairs:
                continue
            example = S.make_example(split, program, initial)
            if example.prompt in history["prompts"]:
                continue
            execute_saved_program(example.to_record())
            used_pairs.add(key)
            return example
        raise RuntimeError("bounded initial-state sampling exhausted")

    composed = [novel_program((2, 4)) for _ in range(composed_patterns)]
    patterns = atomic + composed
    states = []
    for program in patterns:
        examples = [fresh_example(program, split="train") for _ in range(states_per_pattern)]
        # Diversity in initial inputs is required; output diversity is measured,
        # not forced by selecting on neural outcomes or pruning constant programs.
        states.append(examples)
    slots = [(p, state) for p in range(len(patterns)) for state in range(states_per_pattern)]
    rng.shuffle(slots)
    training = {condition: [states[p][0 if condition == "repeat_one" else state] for p, state in slots]
                for condition in CONDITIONS}
    half = evaluation_count // 2
    selected_atomic = rng.sample(list(range(64)), half)
    selected_composed = rng.sample(list(range(64, len(patterns))), half)
    selected = selected_atomic + selected_composed
    common = {"train_anchor": [states[p][0] for p in selected]}
    for name, indexes, high in (
            ("transition_values_id", selected_atomic, False),
            ("transition_values_ood", selected_atomic, True),
            ("program_values_id", selected_composed, False),
            ("program_values_ood", selected_composed, True)):
        common[name] = [fresh_example(patterns[p], high=high) for p in indexes for _ in range(2)]
    common["dev"] = [fresh_example(novel_program((2, 4))) for _ in range(evaluation_count)]
    common["composition"] = [fresh_example(novel_program((2, 4), True), split="unseen_composition")
                             for _ in range(evaluation_count)]
    common["longer"] = [fresh_example(novel_program((6, 8)), split="longer_length")
                        for _ in range(evaluation_count)]
    evaluations = {}
    for condition in CONDITIONS:
        evaluations[condition] = {**common, "train_sample": [
            states[p][0 if condition == "repeat_one" else 1] for p in selected]}
    return {"training": training, "evaluations": evaluations,
            "patterns": [list(pattern_key(p)) for p in patterns], "slots": slots,
            "seed": seed, "states_per_pattern": states_per_pattern,
            "historical_sha256": history["sha256"],
            "unique_history_pairs": len(history["pairs"]),
            "distinct_target_counts": [len({e.target_text for e in group}) for group in states]}


def audit(dataset, history):
    """Check the treatment contrast, holdout isolation and historical exclusions."""
    training, evaluations = dataset["training"], dataset["evaluations"]
    pairs = {condition: {pair_key(e.program, e.initial) for e in examples}
             for condition, examples in training.items()}
    patterns = {pattern_key(e.program) for e in training["multi_state"]}
    primitives = {p[0] for p in patterns if len(p) == 1}
    if len(training["repeat_one"]) != len(training["multi_state"]):
        raise ValueError("treatment document count mismatch")
    for left, right in zip(training["repeat_one"], training["multi_state"]):
        if left.program != right.program:
            raise ValueError("treatment program order mismatch")
    per_pattern = {condition: {} for condition in CONDITIONS}
    for condition in CONDITIONS:
        for example in training[condition]:
            if any(v >= 24 for v in example.initial):
                raise ValueError("training initial-value range mismatch")
            if S.RESERVED_COMPOSITION in S.composition_pairs(example.program):
                raise ValueError("reserved composition leaked into training")
            per_pattern[condition].setdefault(pattern_key(example.program), set()).add(example.initial)
        expected = 1 if condition == "repeat_one" else dataset["states_per_pattern"]
        if any(len(states) != expected for states in per_pattern[condition].values()):
            raise ValueError("initial-state diversity mismatch")
        for name, examples in evaluations[condition].items():
            current = {pair_key(e.program, e.initial) for e in examples}
            if len(current) != len(examples):
                raise ValueError("duplicate evaluation pair")
            if name.startswith("train_"):
                if not current <= pairs[condition]:
                    raise ValueError("training fit selection is not in its treatment")
            elif current & (pairs["repeat_one"] | pairs["multi_state"]):
                raise ValueError("heldout pair overlaps training")
            if name.endswith("ood") and any(v < 24 for e in examples for v in e.initial):
                raise ValueError("high-value holdout mismatch")
            if not name.endswith("ood") and any(v >= 24 for e in examples for v in e.initial):
                raise ValueError("low-value population mismatch")
            if name.startswith(("transition_values", "program_values")):
                if any(pattern_key(e.program) not in patterns for e in examples):
                    raise ValueError("value holdout must reuse a training pattern")
                atomic = name.startswith("transition_")
                if any((len(e.program) == 1) != atomic for e in examples):
                    raise ValueError("transition/program population mismatch")
            for example in examples:
                if any(str(i) not in primitives for i in example.program):
                    raise ValueError("evaluation contains an untrained primitive")
                reserved = S.RESERVED_COMPOSITION in S.composition_pairs(example.program)
                if reserved != (name == "composition"):
                    raise ValueError("composition population mismatch")
                if name == "composition":
                    last = example.program[-1]
                    dependency = analyze_program(pattern_key(example.program))[last.dst]
                    if (last.op != "add" or example.program[-2].op != "multiply"
                            or len(example.program) - 2 not in dependency["relevant_instructions"]
                            or not dependency["depends_on_initial"]):
                        raise ValueError("reserved dependency does not survive to a target")
            if name in ("dev", "composition", "longer"):
                if any(pattern_key(e.program) in patterns | history["patterns"] for e in examples):
                    raise ValueError("novel-program holdout contains an old pattern")
        heldout = [e for n, es in evaluations[condition].items() if not n.startswith("train_") for e in es]
        if len({e.prompt for e in heldout}) != len(heldout):
            raise ValueError("overlap between evaluation populations")
        for example in [*training[condition], *heldout]:
            if pair_key(example.program, example.initial) in history["pairs"] or example.prompt in history["prompts"]:
                raise ValueError("historical example reused")
            execute_saved_program(example.to_record())
    for name in POPULATIONS:
        if name != "train_sample" and evaluations["repeat_one"][name] != evaluations["multi_state"][name]:
            raise ValueError("paired evaluation population mismatch")
    return {"historical_pair_overlap": 0, "historical_unique_pairs": len(history["pairs"]),
            "historical_files": len(history["sha256"]), "heldout_train_overlap": 0,
            "training_documents_per_condition": len(training["multi_state"]),
            "training_patterns": len(patterns), "unique_train_pairs": {k: len(v) for k, v in pairs.items()},
            "evaluation_counts": {k: len(v) for k, v in evaluations["multi_state"].items()},
            "patterns_with_multiple_targets": sum(n > 1 for n in dataset["distinct_target_counts"])}
