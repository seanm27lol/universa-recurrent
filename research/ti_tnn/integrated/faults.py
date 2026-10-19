"""Deliberately faulty module sets, used only as positive controls for the checks in `interventions.py` and
`verification.py` (plan PR 5). Each answers ordinary questions correctly or lawfully, and each breaks a
different guarantee, so each check can be shown to catch what it claims to catch.

| Registry | What is wrong | Ordinary answers | What should catch it |
|---|---|---|---|
| `question_reading_registry` | `country_of` answers with the country in the question when there is one | all correct | interventions (history independence) |
| `hidden_dependency_registry` | `capital_of` quietly calls the `largest_city_of` module | all correct | the probe (undeclared calls); disabling unexecuted computation |
| `swapped_registry` | `capital_of` returns the largest city | wrong where they differ, and every declared law still holds | the checker's semantics, against the facts |
| `payload_registry` | `capital_of` attaches hidden data to its output, and `country_of` reads it | correct as long as the data travels | the runtime: it rebuilds every value, so the data is dropped |

None of these is ever used to answer anything.
"""
from __future__ import annotations

from dataclasses import dataclass

from .contracts import ModuleVersion, TypedValue, TypeSpec, content_identity, facts_mapping
from .registry import Registry, TableModule, table_module
from .relations import World


def _version(relation_id: str, kind: str, payload) -> ModuleVersion:
    return ModuleVersion(f"{relation_id}_{kind}", relation_id, kind, content_identity({"kind": kind, **payload}))


def _registry(world: World, replacements: dict) -> Registry:
    registry = Registry(world.relations)
    for rid in sorted(world.relations):
        registry = registry.register(replacements.get(rid) or table_module(world, rid))
    return registry


class Question:
    """What a host shares when a question arrives. Modules should never see it; one fault reads it."""

    def __init__(self):
        self.input: TypedValue | None = None

    def __call__(self, program):  # the `before` hook in interventions.py
        self.input = program.input


@dataclass(frozen=True)
class QuestionReadingModule:
    """`country_of` that answers with the question's country whenever the question starts from a Country."""

    version: ModuleVersion
    table: dict
    target: TypeSpec
    question: Question

    def apply(self, value: TypedValue) -> TypedValue | None:
        asked = self.question.input
        if asked is not None and asked.type_id == self.target.type_id:
            return self.target.value(asked.value)  # the shortcut: never looks at the city it was given
        out = self.table.get(value.value)
        return None if out is None else self.target.value(out)


def question_reading_registry(world: World) -> tuple[Registry, Question]:
    """Every relation by table, except `country_of`, which reads the question. Returns the host hook too."""
    question = Question()
    table = facts_mapping(world.facts["country_of"])
    module = QuestionReadingModule(_version("country_of", "reads_question", {"table": table}), table,
                                   world.types["Country"], question)
    return _registry(world, {"country_of": module}), question


@dataclass(frozen=True)
class HiddenDependencyModule:
    """`capital_of` that asks the `largest_city_of` module first and uses its answer when the two agree."""

    version: ModuleVersion
    table: dict
    target: TypeSpec
    other: object  # a hidden reference to another registered module

    def apply(self, value: TypedValue) -> TypedValue | None:
        guess = self.other.apply(value)
        if guess is not None and guess.value == self.table.get(value.value):
            return guess
        out = self.table.get(value.value)
        return None if out is None else self.target.value(out)


def hidden_dependency_registry(world: World) -> Registry:
    largest = table_module(world, "largest_city_of")
    table = facts_mapping(world.facts["capital_of"])
    module = HiddenDependencyModule(_version("capital_of", "hidden_dependency", {"table": table}), table,
                                    world.types["City"], largest)
    return _registry(world, {"capital_of": module, "largest_city_of": largest})


def swapped_registry(world: World) -> Registry:
    """`capital_of` implemented with the largest-city table: a lawful but wrong change (country_of of either
    city is the country, so both declared round-trip laws still hold)."""
    table = facts_mapping(world.facts["largest_city_of"])
    module = TableModule(_version("capital_of", "swapped", {"table": table}), table, world.types["City"])
    return _registry(world, {"capital_of": module})


@dataclass(frozen=True)
class PayloadModule:
    """`capital_of` whose output carries the country it came from as an extra attribute."""

    version: ModuleVersion
    table: dict
    target: TypeSpec

    def apply(self, value: TypedValue) -> TypedValue | None:
        out = self.target.value(self.table[value.value])
        object.__setattr__(out, "smuggled_country", value.value)
        return out


@dataclass(frozen=True)
class PayloadReadingModule:
    """`country_of` that returns the smuggled country whenever its input carries one."""

    version: ModuleVersion
    table: dict
    target: TypeSpec

    def apply(self, value: TypedValue) -> TypedValue | None:
        smuggled = getattr(value, "smuggled_country", None)
        return self.target.value(smuggled if smuggled is not None else self.table[value.value])


def payload_registry(world: World) -> Registry:
    cap = facts_mapping(world.facts["capital_of"])
    country = facts_mapping(world.facts["country_of"])
    return _registry(world, {
        "capital_of": PayloadModule(_version("capital_of", "payload", {"table": cap}), cap, world.types["City"]),
        "country_of": PayloadReadingModule(_version("country_of", "payload_reader", {"table": country}), country,
                                           world.types["Country"])})
