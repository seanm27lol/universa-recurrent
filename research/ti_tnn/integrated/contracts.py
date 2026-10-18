"""Typed contracts for the integrated runtime (integration plan, section 5.1).

A concrete example: "Which company operates the airport serving the largest city of Ardel?" is the program
largest_city_of -> airport_of -> operator_of. Each step consumes one typed value (a Country, then a City,
then an Airport) and produces the next. The records below say what a type, a relation, a module version, a
value, a program, a decision and a budget are, and reject malformed ones. Nothing in this file learns or
decides; it only makes the interfaces explicit and checkable.

Semantic types are not tensor shapes: Country and City are different types even if a learned module
represents both as vectors of the same width. Two relations with the same endpoints (capital_of and
largest_city_of, both Country -> City) keep distinct identities.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

DECISIONS = ("execute", "commit", "abstain")
STATUSES = ("answered", "abstained", "rejected")


def content_identity(payload: Any) -> str:
    """A stable SHA-256 over a canonical JSON rendering. It identifies an artifact; it authenticates nothing."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _identifier(name: str, what: str) -> str:
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"{what} must be a Python-style identifier, got {name!r}")
    return name


@dataclass(frozen=True)
class TypeSpec:
    """A semantic type with a finite domain of canonical value identifiers."""

    type_id: str
    values: tuple[str, ...]
    version: int = 1
    semantic_domain: str = ""
    _members: frozenset = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        _identifier(self.type_id, "type_id")
        if not self.values or len(set(self.values)) != len(self.values):
            raise ValueError(f"type {self.type_id} needs a non-empty domain of distinct values")
        if not all(isinstance(v, str) and v for v in self.values):
            raise ValueError(f"type {self.type_id}: values must be non-empty strings")
        object.__setattr__(self, "_members", frozenset(self.values))

    def contains(self, value: str) -> bool:
        return value in self._members

    def value(self, value: str) -> "TypedValue":
        """A typed value of this type; rejects anything outside the domain."""
        if not self.contains(value):
            raise ValueError(f"{value!r} is not a {self.type_id}")
        return TypedValue(self.type_id, value, self.version)


@dataclass(frozen=True)
class RelationSpec:
    """A relation with a unique identity, a source type and a target type (arity 1 here)."""

    relation_id: str
    source: str
    target: str
    description: str = ""
    properties: tuple[str, ...] = ()  # declared, e.g. "functional", "many_to_one"; checked against facts

    def __post_init__(self):
        _identifier(self.relation_id, "relation_id")
        _identifier(self.source, "source type")
        _identifier(self.target, "target type")


@dataclass(frozen=True)
class TypedValue:
    """A canonical value of a declared type. The value is all a module receives; provenance lives in the trace."""

    type_id: str
    value: str
    type_version: int = 1


def canonical(value: TypedValue) -> TypedValue:
    """A fresh plain TypedValue with the same three fields. Whatever else a module attached to its output (a
    subclass, an extra attribute) is dropped, so nothing outside the declared value can reach the next module."""
    return TypedValue(value.type_id, value.value, value.type_version)


@dataclass(frozen=True)
class ModuleVersion:
    """An immutable module identity: which relation it implements, and a content hash of what implements it."""

    module_id: str
    relation_id: str
    kind: str  # "table" here; learned kinds arrive in plan PR 4
    identity: str  # content hash of code identity, configuration and parameters
    precision: str = "exact"
    calibration: str | None = None

    def __post_init__(self):
        _identifier(self.module_id, "module_id")
        _identifier(self.relation_id, "relation_id")
        if len(self.identity) != 64 or any(c not in "0123456789abcdef" for c in self.identity):
            raise ValueError("identity must be a SHA-256 hex digest")


@dataclass(frozen=True)
class Program:
    """A supplied program: relations applied in order to a typed input."""

    steps: tuple[str, ...]
    input: TypedValue
    expected_output_type: str | None = None

    def __post_init__(self):
        if not self.steps:
            raise ValueError("a program needs at least one step")
        for step in self.steps:
            _identifier(step, "program step")


@dataclass(frozen=True)
class Budget:
    """Explicit resource limits. Here: the number of module applications."""

    max_steps: int
    policy: str = "module-applications-v1"

    def __post_init__(self):
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")


@dataclass(frozen=True)
class Decision:
    """What the runtime decided, and why."""

    kind: str
    relation_id: str | None = None
    module_id: str | None = None
    reason: str = ""

    def __post_init__(self):
        if self.kind not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")


@dataclass(frozen=True)
class StepRecord:
    """One executed step: the exact value consumed and produced, and the module version that ran."""

    index: int
    relation_id: str
    module_id: str
    module_identity: str
    consumed: TypedValue
    produced: TypedValue | None
    cost: int = 1
    intervened: bool = False  # the consumed value was supplied by an intervention, not by the previous step


@dataclass(frozen=True)
class RunState:
    """Everything needed to continue: the current value, the program position, steps used, and the registry
    snapshot the run is bound to (bounded recurrence builds on this in plan PR 7)."""

    current: TypedValue
    position: int
    steps_used: int
    registry_identity: str


@dataclass(frozen=True)
class Execution:
    """The outcome of running a program: a status, the final decision, the answer if any, and the trace."""

    status: str
    decision: Decision
    answer: TypedValue | None
    trace: tuple[StepRecord, ...]
    final_state: RunState | None
    problems: tuple[str, ...] = ()

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if (self.status == "answered") != (self.answer is not None):
            raise ValueError("an execution has an answer exactly when it is answered")


def facts_mapping(table: Mapping[str, str]) -> dict[str, str]:
    """A plain copy of a relation's facts, for hashing and checks."""
    return dict(sorted(table.items()))
