"""An immutable registry of module versions, one per relation identity.

Example: the registry for the Ardel world binds `capital_of` and `largest_city_of` to two different modules.
They share their endpoint types, but each is registered under its own relation identity, and a module can
only be registered for the relation its `ModuleVersion` names.

Registering returns a new registry; the old one is unchanged. The registry's identity is a hash over its
module versions, so a run can record exactly which snapshot it used. Activating a new snapshot is a
separate, later concern (integration plan PR 8): nothing here swaps modules under a running program.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol

from .contracts import ModuleVersion, RelationSpec, TypedValue, TypeSpec, content_identity, facts_mapping


class Module(Protocol):
    """What the runtime may call: one typed value in, one typed value (or None, for no answer) out."""

    version: ModuleVersion

    def apply(self, value: TypedValue) -> TypedValue | None: ...


@dataclass(frozen=True)
class TableModule:
    """A module that looks its answer up in a fixed table (the symbolic implementation of a relation)."""

    version: ModuleVersion
    table: Mapping[str, str]
    target: TypeSpec

    def __post_init__(self):
        object.__setattr__(self, "table", MappingProxyType(dict(self.table)))

    def apply(self, value: TypedValue) -> TypedValue | None:
        out = self.table.get(value.value)
        return None if out is None else self.target.value(out)


def table_module(world, relation_id: str, module_id: str | None = None) -> TableModule:
    """A table module implementing a world relation, with a content identity over its table."""
    spec = world.relations[relation_id]
    table = facts_mapping(world.facts[relation_id])
    version = ModuleVersion(module_id or f"{relation_id}_table", relation_id, "table",
                            content_identity({"kind": "table", "relation": relation_id, "table": table}))
    return TableModule(version, table, world.types[spec.target])


@dataclass(frozen=True)
class Registry:
    """Relation specs plus an immutable binding relation_id -> module."""

    relations: Mapping[str, RelationSpec]
    modules: Mapping[str, Module] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "relations", MappingProxyType(dict(self.relations)))
        object.__setattr__(self, "modules", MappingProxyType(dict(self.modules)))
        for rid, module in self.modules.items():
            if rid not in self.relations:
                raise ValueError(f"module bound to unknown relation {rid}")
            if module.version.relation_id != rid:
                raise ValueError(f"module {module.version.module_id} implements {module.version.relation_id}, not {rid}")

    def register(self, module: Module) -> "Registry":
        """A new registry with `module` bound to its relation; refuses to rebind an existing relation."""
        rid = module.version.relation_id
        if rid not in self.relations:
            raise ValueError(f"no relation {rid} to implement")
        if rid in self.modules:
            raise ValueError(f"relation {rid} already has a module; replacing one is a separate, checked operation")
        return Registry(self.relations, {**self.modules, rid: module})

    def module(self, relation_id: str) -> Module:
        return self.modules[relation_id]

    @property
    def identity(self) -> str:
        return content_identity({rid: [m.version.module_id, m.version.identity] for rid, m in sorted(self.modules.items())})


def table_registry(world) -> Registry:
    """A registry with every world relation implemented by its table module."""
    registry = Registry(world.relations)
    for rid in sorted(world.relations):
        registry = registry.register(table_module(world, rid))
    return registry
