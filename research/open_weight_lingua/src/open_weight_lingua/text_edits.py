"""Frozen deterministic parser/editor for explicit current-value statements.

Establishes Milestone 2 edit-coverage accounting (brief Section 7): which
descriptions contain exactly one unambiguous statement of a variable's current
value, and the minimal value-span replacement when they do. It does not
establish that any statement is true of an activation; agreement with the
reference program is recorded separately and never gates an edit. Matching is
case-sensitive and limited to the frozen forms in the rule definition; every
other phrasing, including quoted initial assignments, is absent by
construction and surfaced in coverage, never guessed.
"""

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
import re

from .tasks import VARIABLES, ProblemGroup, Statement, interpret

RULE_VERSION = "1.0.0"
STATUSES = ("eligible", "absent", "ambiguous")

_VALUE_FORM = r"(?:0|1[0-9]|[1-9])"
_RULE_TEMPLATES = (
    ("is-currently", "{var} is currently {value}"),
    ("is-now", "{var} is now {value}"),
    ("current-value-of", "the current value of {var} is {value}"),
)
_LEFT_GUARD = r"(?<![A-Za-z0-9_])"
_RIGHT_GUARD = r"(?![A-Za-z0-9_])"


def _compile(variable: str) -> tuple[tuple[str, re.Pattern], ...]:
    patterns = []
    for name, template in _RULE_TEMPLATES:
        body = template.replace("{var}", variable).replace(
            "{value}", f"(?P<value>{_VALUE_FORM})"
        )
        patterns.append((name, re.compile(_LEFT_GUARD + body + _RIGHT_GUARD)))
    return tuple(patterns)


_PATTERNS = {variable: _compile(variable) for variable in VARIABLES}


@dataclass(frozen=True)
class ParseHit:
    rule: str
    value: int
    statement_span: tuple[int, int]
    value_span: tuple[int, int]


@dataclass(frozen=True)
class VariableStatus:
    status: str
    value: int | None
    value_span: tuple[int, int] | None
    detail: str | None
    hits: tuple[ParseHit, ...]


@dataclass(frozen=True)
class EditResult:
    variable: str
    requested_value: int
    status: str
    description: str | None
    original_value: int | None
    value_span: tuple[int, int] | None


def parse(description: str) -> dict[str, VariableStatus]:
    """Classify each variable: one current-value statement, none, or several.

    Two or more statements are ambiguous even when they agree, because editing
    one span would leave the description contradicting itself.
    """
    if not isinstance(description, str):
        raise ValueError("description must be a string")
    results = {}
    for variable in VARIABLES:
        hits = []
        for rule, pattern in _PATTERNS[variable]:
            for match in pattern.finditer(description):
                hits.append(
                    ParseHit(
                        rule,
                        int(match.group("value")),
                        match.span(),
                        match.span("value"),
                    )
                )
        hits.sort(key=lambda hit: (hit.statement_span, hit.rule))
        if not hits:
            status, value, span, detail = "absent", None, None, None
        elif len(hits) == 1:
            status, value, span, detail = (
                "eligible",
                hits[0].value,
                hits[0].value_span,
                None,
            )
        else:
            status, value, span = "ambiguous", None, None
            detail = (
                "conflicting values"
                if len({hit.value for hit in hits}) > 1
                else "repeated value"
            )
        results[variable] = VariableStatus(status, value, span, detail, tuple(hits))
    return results


def edit(description: str, variable: str, new_value: int) -> EditResult:
    """Replace only the value span of an eligible statement; never fabricate.

    The requested value is the intervention. Truth of the original statement
    and the reference program are not consulted here.
    """
    if variable not in VARIABLES:
        raise ValueError("unknown variable")
    if type(new_value) is not int or not 0 <= new_value <= 19:
        raise ValueError("new value must be a canonical integer in 0..19")
    parsed = parse(description)[variable]
    if parsed.status != "eligible":
        return EditResult(variable, new_value, parsed.status, None, None, None)
    start, end = parsed.value_span
    replaced = description[:start] + str(new_value) + description[end:]
    return EditResult(
        variable, new_value, "edited", replaced, parsed.value, (start, end)
    )


def agreement(description: str, program: tuple[Statement, ...]) -> dict:
    """Record whether each parsed current value matches interpret(program).

    Independent of edit(): its outcome is evidence about the description, and
    must not gate any requested edit.
    """
    reference = interpret(program)
    parsed = parse(description)
    return {
        variable: {
            "status": parsed[variable].status,
            "parsed_value": parsed[variable].value,
            "reference_value": reference[variable],
            "agrees": (
                parsed[variable].value == reference[variable]
                if parsed[variable].status == "eligible"
                else None
            ),
        }
        for variable in VARIABLES
    }


def coverage(group_description_pairs: Iterable[tuple[ProblemGroup, str]]) -> dict:
    """Edit eligibility over all groups; one pair per group, never reweighted.

    Each pair carries the single description the frozen rule would edit for
    that group (the source side). Exclusion reasons keep zero counts so the
    full distribution is always reported.
    """
    pairs = list(group_description_pairs)
    identities = [group.group_id for group, _ in pairs]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate group id in coverage input")
    per_variable = {v: {status: 0 for status in STATUSES} for v in VARIABLES}
    exclusions = {status: 0 for status in STATUSES if status != "eligible"}
    eligible = 0
    for group, description in pairs:
        if group.affected not in VARIABLES:
            raise ValueError("unknown affected variable")
        parsed = parse(description)
        for variable in VARIABLES:
            per_variable[variable][parsed[variable].status] += 1
        status = parsed[group.affected].status
        if status == "eligible":
            eligible += 1
        else:
            exclusions[status] += 1
    return {
        "groups": len(pairs),
        "eligible_groups": eligible,
        "eligible_fraction": eligible / len(pairs) if pairs else 0.0,
        "exclusion_reasons": exclusions,
        "per_variable_status": per_variable,
        "rule_version": RULE_VERSION,
        "rule_sha256": RULE_SHA256,
    }


def rule_definition() -> dict:
    """Canonical serialization of the frozen rule, for manifest locking."""
    return {
        "version": RULE_VERSION,
        "variables": list(VARIABLES),
        "value_range": [0, 19],
        "value_form": _VALUE_FORM,
        "templates": [
            {"name": name, "template": template} for name, template in _RULE_TEMPLATES
        ],
    }


def rule_sha256() -> str:
    canonical = json.dumps(rule_definition(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


RULE_SHA256 = rule_sha256()
