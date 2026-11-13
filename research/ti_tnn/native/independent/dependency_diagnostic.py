"""Reference-program dependency categories for saved answers; no model imports.

The read graph is built from versioned registers, independently of the producer.
Its depth includes copy/set writes, even if arithmetic later cancels their effect.
Functional initial dependence is checked exhaustively over the finite domain,
separately from that structural graph. See DEPENDENCY_DIAGNOSTIC.md.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from .rescore_generations import (
    INSTRUCTION, PROGRAM_STUDY_SPLITS, ScoringError, execute_saved_program,
    prerequisite, registers,
)

WRITES = ("untouched", "set", "copy", "add", "multiply")


def _graph(program):
    """Return immutable nodes (operation, operands, immediate) and final roots."""
    nodes = [("initial", (), i) for i in range(4)]
    roots = list(range(4))
    writes = [[] for _ in range(4)]
    if not program or len(program) > 32:
        raise ScoringError("dependency analysis requires 1..32 instructions")
    for index, text in enumerate(program):
        match = INSTRUCTION.fullmatch(text)
        if match is None:
            raise ScoringError(f"invalid instruction: {text!r}")
        op, dst, src, value = match.groups()
        dst = int(dst)
        if op == "set":
            if src is not None or value is None or not 0 <= int(value) < 32:
                raise ScoringError("set needs an immediate in [0,32)")
            operands = ()
        else:
            if src is None or value is not None:
                raise ScoringError(f"{op} needs a source register")
            operands = (roots[int(src)],) if op == "copy" else (roots[dst], roots[int(src)])
        nodes.append((op, operands, None if value is None else int(value)))
        roots[dst] = len(nodes) - 1
        writes[dst].append(index)
    return nodes, roots, writes


def _ancestors(nodes, root):
    found = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if node not in found:
            found.add(node)
            pending.extend(nodes[node][1])
    return found


def _functional_support(nodes, root, initial_support):
    """Exact global support: vary each initial coordinate over all other values.

    At most 32**4 assignments. uint16 is safe: intermediates are reduced modulo
    32 after each instruction, so the largest product before reduction is 961.
    A nonzero polynomial coefficient alone would not suffice over Z/32Z.
    """
    if not initial_support:
        return []
    dimensions = len(initial_support)
    values = {}
    for axis, register in enumerate(initial_support):
        shape = [1] * dimensions
        shape[axis] = 32
        values[register] = np.arange(32, dtype=np.uint16).reshape(shape)

    def evaluate(node):
        if node in values:
            return values[node]
        op, operands, immediate = nodes[node]
        if op == "set":
            result = np.asarray(immediate, dtype=np.uint16)
        elif op == "copy":
            result = evaluate(operands[0])
        else:
            left, right = (evaluate(part) for part in operands)
            result = (left + right if op == "add" else left * right) % 32
        values[node] = result
        return result

    table = np.broadcast_to(evaluate(root), (32,) * dimensions)
    return [register for axis, register in enumerate(initial_support)
            if np.any(table != np.take(table, [0], axis=axis))]


@lru_cache(maxsize=8192)
def analyze_program(program: tuple[str, ...]) -> tuple[dict, ...]:
    """Classify each final register without reading saved targets or completions."""
    nodes, roots, writes = _graph(program)
    depths = [0] * 4
    for _, operands, _ in nodes[4:]:
        depths.append(1 + max((depths[node] for node in operands), default=0))
    result = []
    for register, root in enumerate(roots):
        ancestors = _ancestors(nodes, root)
        relevant = sorted(node - 4 for node in ancestors if node >= 4)
        initial = sorted(node for node in ancestors if node < 4)
        functional = _functional_support(nodes, root, initial)
        prior_writes = writes[register][:-1]
        discarded = sorted(set(prior_writes) - set(relevant))
        result.append({
            "register": register,
            "final_write": "untouched" if root < 4 else nodes[root][0],
            "final_instruction": None if root < 4 else root - 4,
            "relevant_instructions": relevant,
            "read_initial_registers": initial,
            "functional_initial_registers": functional,
            "depends_on_initial": bool(functional),
            "dependency_depth": depths[root],
            "prior_target_writes": prior_writes,
            "discarded_prior_target_writes": discarded,
            "has_discarded_prior_write": bool(discarded),
        })
    return tuple(result)


def _counts(rows):
    total = len(rows)
    valid = sum(row["valid"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    return {"correct": correct, "total": total, "valid": valid,
            "malformed": total - valid,
            "accuracy_all": correct / total if total else None,
            "accuracy_valid_only": correct / valid if valid else None,
            "complete_answer_correct": sum(row["answer_correct"] for row in rows)}


def _group(rows, key):
    return {str(value).lower(): _counts([row for row in rows if row[key] == value])
            for value in sorted({row[key] for row in rows})}


def score_population(saved, source, *, expected_count=None):
    """Score complete answers and all register positions; malformed text fails all four."""
    ids = [record["example_id"] for record in saved]
    if len(ids) != len(set(ids)) or (expected_count is not None and len(ids) != expected_count):
        raise ScoringError("duplicate records or changed population denominator")
    answers, targets = [], []
    for record in saved:
        example = source[record["example_id"]]
        expected = execute_saved_program(example)
        if record["target_text"] != example["target_text"]:
            raise ScoringError("saved target does not match source")
        if not prerequisite(example)["complete"] or record.get("prompt_truncated", False):
            raise ScoringError("incomplete input prerequisite")
        actual = registers(record["completion_text"])
        valid = actual is not None
        exact = valid and actual == expected
        dependencies = analyze_program(tuple(example["program"]))
        answers.append({"valid": valid, "correct": exact, "answer_correct": exact,
                        "any_initial_dependence": any(d["depends_on_initial"] for d in dependencies),
                        "maximum_dependency_depth": max(d["dependency_depth"] for d in dependencies),
                        "any_discarded_prior_write": any(d["has_discarded_prior_write"] for d in dependencies)})
        for index, dependency in enumerate(dependencies):
            targets.append({**dependency, "example_id": record["example_id"],
                            "valid": valid, "correct": valid and actual[index] == expected[index],
                            "answer_correct": exact})
    report = {
        "complete_answers": _counts(answers),
        "eos": sum(bool(r.get("stopped_on_eos")) for r in saved),
        "truncated": sum(bool(r.get("truncated")) for r in saved),
        "registers": _counts(targets),
        "per_register": {str(i): _counts([r for r in targets if r["register"] == i]) for i in range(4)},
        "by_final_write": {op: _counts([r for r in targets if r["final_write"] == op]) for op in WRITES},
        "by_initial_dependence": _group(targets, "depends_on_initial"),
        "by_dependency_depth": _group(targets, "dependency_depth"),
        "by_discarded_prior_write": _group(targets, "has_discarded_prior_write"),
        "answer_by_any_initial_dependence": _group(answers, "any_initial_dependence"),
        "answer_by_maximum_depth": _group(answers, "maximum_dependency_depth"),
        "answer_by_any_discarded_prior_write": _group(answers, "any_discarded_prior_write"),
        "joint_register_categories": {},
    }
    for row in targets:
        key = f"{row['final_write']}|initial={row['depends_on_initial']}|depth={row['dependency_depth']}|discard={row['has_discarded_prior_write']}"
        report["joint_register_categories"].setdefault(key, []).append(row)
    report["joint_register_categories"] = {key: _counts(rows) for key, rows in
                                            sorted(report["joint_register_categories"].items())}
    return report, targets


def analyze_bundle(bundle, output):
    """Analyze only the six fixed primary populations; never regenerate outputs."""
    bundle, output = Path(bundle), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    source, input_hashes = {}, {}

    def read(path):
        raw = path.read_bytes()
        input_hashes[str(path.relative_to(bundle))] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    for split in PROGRAM_STUDY_SPLITS:
        for line in read(bundle / "data" / "structured" / f"{split}.jsonl").splitlines():
            record = json.loads(line)
            if record["example_id"] in source or record["split"] != split:
                raise ScoringError("duplicate source or split mismatch")
            execute_saved_program(record)
            source[record["example_id"]] = record
    report = {"format": "native-saved-dependency-v1", "populations": {}, "input_sha256": input_hashes,
              "scope": "primary saved #79 outputs only; 128 examples/512 register positions per child/split",
              "excluded": "180 supplemental and 960 intervened outputs; reused selected baselines are not additional observations",
              "interpretation": "reference task categories, not neural mechanisms or fresh confirmation"}
    details = []
    selections = {}
    for mode in ("dense", "untyped", "typed"):
        for objective in ("full_sequence", "answer_eos"):
            child = f"78-{mode}-{objective}"
            evaluation = json.loads(read(bundle / "children" / child / "evaluation.json"))
            report["populations"][child] = {}
            for split in PROGRAM_STUDY_SPLITS:
                population = evaluation["splits"][split]
                generation = population["generation"]
                ids = [r["example_id"] for r in generation["records"]]
                if ids != population["selection"]["evaluation_ids"]:
                    raise ScoringError("generation order does not match saved selection")
                if ids != selections.setdefault(split, ids):
                    raise ScoringError("children do not share the same selected examples")
                if any(source[i]["split"] != split for i in ids):
                    raise ScoringError("generation split mismatch")
                summary, records = score_population(generation["records"], source, expected_count=128)
                counts = summary["complete_answers"]
                if (counts["total"], counts["correct"], counts["valid"]) != (
                        generation["evaluated"], generation["exact_count"], generation["format_valid_count"]):
                    raise ScoringError("historical scores disagree; stop rather than rewrite")
                report["populations"][child][split] = summary
                details.extend({"child": child, "split": split, **r} for r in records)
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "register_details.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in details))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True, help="new directory, never overwritten")
    args = parser.parse_args()
    result = analyze_bundle(args.bundle, args.output)
    print(json.dumps({"children": len(result["populations"]), "primary_outputs": 3840,
                      "model_calls": 0, "output": str(args.output)}))


if __name__ == "__main__":
    main()
