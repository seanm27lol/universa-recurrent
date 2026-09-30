"""Text-only census helpers shared by the edit-structure analysis and diagnostics.

Example: for the prompt ``x = 9\\ny = 8\\nx = x + 1\\nWhat is x? ...``,
``program`` recovers three statements, ``tasks.interpret`` gives x = 10, and
``bound_values("so x is 10", "x")`` returns ``[10]``. Nothing here reads a model
or decides whether a statement is true of an activation.
"""

import re

from .tasks import Statement

NUMBER = re.compile(r"(?<![0-9A-Za-z_.])(\d{1,2})(?![0-9])")
GUARD = r"(?<![A-Za-z0-9_])"
MARK = r"[`'\"*]*"
FROZEN_FORMS = ("{v} is currently", "{v} is now", "the current value of {v} is")
BOUND_VERBS = r"(?:=|==|:|is now|is currently|is|equals|becomes|holds)"


def program_text(prompt: str) -> str:
    """The program part of a task prompt, before the question."""
    return prompt.split("\nWhat is")[0]


def program(prompt: str) -> tuple[Statement, ...]:
    """Parse a rendered task prompt back into statements (tasks.Statement.render inverse)."""
    statements = []
    for line in program_text(prompt).split("\n"):
        variable, rhs = (part.strip() for part in line.split("="))
        arithmetic = re.fullmatch(r"([xy]) ([+-]) (\d+)", rhs)
        if arithmetic:
            operation = "add" if arithmetic.group(2) == "+" else "subtract"
            statements.append(Statement(variable, operation, int(arithmetic.group(3))))
        elif rhs in ("x", "y"):
            statements.append(Statement(variable, "copy", rhs))
        else:
            statements.append(Statement(variable, "assign", int(rhs)))
    return tuple(statements)


def literals(prompt: str) -> set[int]:
    """Every integer written in the prompt; a value outside this set is computed."""
    return {int(n) for n in re.findall(r"\d+", prompt)}


def numbers(text: str) -> set[int]:
    return {int(n) for n in NUMBER.findall(text)}


def names(text: str, variable: str) -> bool:
    return bool(re.search(GUARD + MARK + variable + MARK + r"(?![A-Za-z0-9_])", text))


def bound_values(text: str, variable: str) -> list[int]:
    """Values stated for a variable in any bound form (`x = N`, `x is N`, ...)."""
    pattern = re.compile(
        GUARD + MARK + variable + MARK + r"\s*" + BOUND_VERBS + r"\s*" + MARK
        + r"(\d{1,2})(?![0-9])"
    )
    return [int(m.group(1)) for m in pattern.finditer(text)]


def relaxed_frozen_hit(text: str) -> bool:
    """Frozen-rule forms after removing markdown/quotes and case."""
    flat = re.sub(r"[`*\"']", "", text).lower()
    return any(
        re.search(GUARD + form.format(v=v) + r"\s*(\d{1,2})(?![0-9])", flat)
        for v in ("x", "y")
        for form in FROZEN_FORMS
    )


def integer_answer(generation: dict | None, convention: str) -> int | None:
    """A terminated generation's canonical integer under the run's convention."""
    if not generation or not generation.get("terminated"):
        return None
    text = generation["text"].rstrip() if convention == "rstrip" else generation["text"]
    return int(text) if re.fullmatch(r"0|[1-9][0-9]*", text) else None
