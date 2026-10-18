"""An independent reference evaluator for the finite world.

It reads the world's facts directly and type-checks paths itself. It does not import the runtime, the
registry or any module (a test checks this), so agreement between the two is evidence, not a tautology.
It is the semantic oracle that later experiments compare learned modules and their traces against.

Example: `reference_answer(ardel_world(), ("largest_city_of", "airport_of", "operator_of"), "Ardel")` is
"Tessaline Air"; with `capital_of` in place of `largest_city_of` it is "Northline".
"""
from __future__ import annotations

from itertools import product

from .relations import LAWS, World


def path_types(world: World, path: tuple[str, ...]) -> tuple[str, str] | None:
    """(source type, target type) of a well-typed path, or None."""
    if not path or any(r not in world.relations for r in path):
        return None
    for a, b in zip(path, path[1:]):
        if world.relations[a].target != world.relations[b].source:
            return None
    return world.relations[path[0]].source, world.relations[path[-1]].target


def reference_answer(world: World, path: tuple[str, ...], start: str) -> str | None:
    """Follow the facts along a path; None if the path is ill-typed or the start is outside its type."""
    types = path_types(world, path)
    if types is None or not world.types[types[0]].contains(start):
        return None
    value = start
    for rid in path:
        value = world.facts[rid][value]
    return value


def well_typed_paths(world: World, max_length: int) -> list[tuple[str, ...]]:
    """Every well-typed relation path of length 1..max_length."""
    rels = sorted(world.relations)
    return [p for n in range(1, max_length + 1) for p in product(rels, repeat=n) if path_types(world, p)]


def law_violations(world: World) -> list[str]:
    """The declared restricted-domain laws that fail on this world (empty if every law holds)."""
    out = []
    for law in LAWS:
        for x in world.types[law["on"]].values:
            if reference_answer(world, law["path"], x) != x:
                out.append(f"{' then '.join(law['path'])} does not return {x}")
    return out
