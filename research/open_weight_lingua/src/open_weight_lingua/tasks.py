"""Paired straight-line programs with an explicit, bounded interpreter.

Reference execution establishes program answers, never activation semantics.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import random

VARIABLES = ("x", "y")
SPLIT_SEEDS = {
    "smoke": 201001,
    "calibration": 202001,
    "pilot": 203001,
    "validation_a": 204001,
    "validation_b": 205001,
}
PROMPT_TEMPLATE = "{program}\nWhat is {variable}? Reply with only the integer."


@dataclass(frozen=True)
class Statement:
    variable: str
    operation: str
    operand: int | str

    def render(self) -> str:
        if self.operation in ("assign", "copy"):
            rhs = str(self.operand)
        else:
            symbol = "+" if self.operation == "add" else "-"
            rhs = f"{self.variable} {symbol} {self.operand}"
        return f"{self.variable} = {rhs}"


def interpret(program: tuple[Statement, ...]) -> dict[str, int]:
    """Execute only four operations; reject undefined/out-of-range values."""
    state: dict[str, int] = {}
    for instruction in program:
        variable, operation, operand = (
            instruction.variable,
            instruction.operation,
            instruction.operand,
        )
        if variable not in VARIABLES:
            raise ValueError("unknown variable")
        if operation == "copy":
            if operand not in state:
                raise ValueError("copy reads an undefined variable")
            value = state[operand]
        elif operation in ("assign", "add", "subtract") and type(operand) is int:
            if operation == "assign":
                value = operand
            elif variable not in state:
                raise ValueError("arithmetic reads an undefined variable")
            else:
                value = state[variable] + (operand if operation == "add" else -operand)
        else:
            raise ValueError("unsupported operation or operand")
        if not 0 <= value <= 19:
            raise ValueError("intermediate value outside 0..19")
        state[variable] = value
    if set(state) != set(VARIABLES):
        raise ValueError("both variables must be initialized")
    return state


def canonical(program: tuple[Statement, ...]) -> str:
    return "\n".join(instruction.render() for instruction in program)


@dataclass(frozen=True)
class ProblemGroup:
    group_id: str
    split: str
    source: tuple[Statement, ...]
    counterfactual: tuple[Statement, ...]
    affected: str
    edit_index: int

    def validate(self) -> None:
        differences = [
            i
            for i, pair in enumerate(zip(self.source, self.counterfactual))
            if pair[0] != pair[1]
        ]
        if len(self.source) != len(self.counterfactual) or differences != [
            self.edit_index
        ]:
            raise ValueError("pair must differ in exactly the declared assignment")
        if (
            self.source[self.edit_index].operation != "assign"
            or self.counterfactual[self.edit_index].operation != "assign"
            or self.source[self.edit_index].variable != self.affected
            or self.counterfactual[self.edit_index].variable != self.affected
        ):
            raise ValueError("pair must change an assignment to the affected variable")
        a, b = interpret(self.source), interpret(self.counterfactual)
        other = next(v for v in VARIABLES if v != self.affected)
        if a[self.affected] == b[self.affected] or a[other] != b[other]:
            raise ValueError("intended answer must change and control answer must not")

    def variants(self) -> list[dict]:
        rows = []
        # Alternation balances which query appears first; both are always evaluated.
        queries = (self.affected, next(v for v in VARIABLES if v != self.affected))
        for source_name, program in (("A", self.source), ("B", self.counterfactual)):
            for variable in queries:
                rows.append(
                    {
                        "id": f"{self.group_id}-{source_name}-{variable}",
                        "group_id": self.group_id,
                        "side": source_name,
                        "variable": variable,
                        "affected": variable == self.affected,
                        "prompt": PROMPT_TEMPLATE.format(
                            program=canonical(program), variable=variable
                        ),
                        "answer": str(interpret(program)[variable]),
                    }
                )
        return rows


def generate_groups(
    split: str, count: int, *, excluded_programs=()
) -> tuple[list[ProblemGroup], dict]:
    """Deterministic groups; callers carry the exclusion inventory across splits.

    Separate seeds alone do not guarantee disjointness. Both members are checked
    against the supplied prior canonical-program inventory before acceptance.
    No model outputs enter generation or rejection.
    """
    if split not in SPLIT_SEEDS or count < 1:
        raise ValueError("unknown split or nonpositive group count")
    rng = random.Random(SPLIT_SEEDS[split])
    seen = set(excluded_programs)
    groups = []
    stats = {"attempts": 0, "duplicate_programs": 0, "invalid_pairs": 0}
    while len(groups) < count:
        stats["attempts"] += 1
        if stats["attempts"] > count * 10000:
            raise RuntimeError("generation budget exhausted")
        affected = VARIABLES[len(groups) % 2]
        statements = [Statement(v, "assign", rng.randrange(1, 18)) for v in VARIABLES]
        for _ in range(rng.randrange(1, 5)):
            variable = rng.choice(VARIABLES)
            operation = rng.choice(("add", "subtract", "copy", "assign"))
            operand = (
                rng.choice(VARIABLES) if operation == "copy" else rng.randrange(1, 5)
            )
            statements.append(Statement(variable, operation, operand))
        possible = [
            i
            for i, s in enumerate(statements)
            if s.variable == affected and s.operation == "assign"
        ]
        index = rng.choice(possible)
        donor = list(statements)
        donor[index] = replace(
            donor[index], operand=donor[index].operand + rng.choice((-1, 1))
        )
        source, counterfactual = tuple(statements), tuple(donor)
        group = ProblemGroup(
            f"{split}-{len(groups):04d}", split, source, counterfactual, affected, index
        )
        try:
            group.validate()
        except ValueError:
            stats["invalid_pairs"] += 1
            continue
        programs = {canonical(source), canonical(counterfactual)}
        if programs & seen:
            stats["duplicate_programs"] += 1
            continue
        seen.update(programs)
        groups.append(group)
    stats["groups"] = len(groups)
    stats["prompt_variants"] = 4 * len(groups)
    stats["inventory_sha256"] = hashlib.sha256(
        json.dumps([asdict(g) for g in groups], sort_keys=True).encode()
    ).hexdigest()
    return groups, stats


def tokenize_groups(groups, tokenizer, *, excluded_tokenized_prompts=()) -> list[dict]:
    """Use the last assistant-prefix token in each context; allow unequal lengths.

    Explicit relative-boundary alignment: A and B each use their own last
    non-padding prompt token. Absolute offsets are retained, never transferred.
    """
    rows, seen = [], {tuple(ids) for ids in excluded_tokenized_prompts}
    for group in groups:
        pair_tokens = {}
        for row in group.variants():
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if not ids or tuple(ids) in seen:
                raise ValueError(
                    "empty or duplicated tokenized prompt; no silent replacement"
                )
            seen.add(tuple(ids))
            row.update(
                input_ids=ids,
                attention_mask=[1] * len(ids),
                position=len(ids) - 1,
                decoded_token=tokenizer.decode([ids[-1]]),
                alignment="last-assistant-prefix-token-in-each-context",
            )
            pair_tokens.setdefault(row["variable"], []).append(ids[-1])
            rows.append(row)
        if any(a != b for a, b in pair_tokens.values()):
            raise ValueError("source/donor boundary tokens disagree")
    return rows
