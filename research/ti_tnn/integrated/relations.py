"""The finite world: semantic types, relations with unique identities, and their facts.

Example (the fixture `ardel_world`): Ardel's capital is Ardel Crossing, but its largest city is Port Vey.
So `capital_of` and `largest_city_of` share their endpoints (Country -> City) yet give different answers,
and any system that treats all Country -> City relations alike fails on Ardel. Port Vey's airport, Vey
International, is operated by Tessaline Air, which also operates another country's airport (`operator_of`
is many-to-one). Brenmark's capital is also its largest city, so the two relations agree there.

Facts are total and functional on each relation's source type: every source value has exactly one
target. Declared properties ("many_to_one", "one_to_one") are checked against the facts. The structure
here is supplied by the benchmark; a model that uses it has not discovered it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .contracts import RelationSpec, TypeSpec, content_identity

COUNTRY, CITY, AIRPORT, OPERATOR = "Country", "City", "Airport", "Operator"

RELATIONS: Mapping[str, RelationSpec] = MappingProxyType({
    "capital_of": RelationSpec("capital_of", COUNTRY, CITY, "the country's capital city"),
    "largest_city_of": RelationSpec("largest_city_of", COUNTRY, CITY, "the country's most populous city"),
    "country_of": RelationSpec("country_of", CITY, COUNTRY, "the country a city is in", ("many_to_one",)),
    "airport_of": RelationSpec("airport_of", CITY, AIRPORT, "the airport serving a city", ("one_to_one",)),
    "operator_of": RelationSpec("operator_of", AIRPORT, OPERATOR, "the company operating an airport", ("many_to_one",)),
})

# Equations that hold on a restricted domain: going to the capital (or the largest city) and back to its
# country returns the country. The reverse round trip, country_of then capital_of, is NOT the identity on
# cities: it maps every city to its country's capital. Many-to-one relations have no inverse.
LAWS = (
    {"path": ("capital_of", "country_of"), "equals": "identity", "on": COUNTRY},
    {"path": ("largest_city_of", "country_of"), "equals": "identity", "on": COUNTRY},
)


@dataclass(frozen=True)
class World:
    """Types, relations and facts (relation -> {source value: target value}), validated on construction."""

    name: str
    types: Mapping[str, TypeSpec]
    relations: Mapping[str, RelationSpec]
    facts: Mapping[str, Mapping[str, str]]
    seed: int | None = None
    identity: str = field(init=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "types", MappingProxyType(dict(self.types)))
        object.__setattr__(self, "relations", MappingProxyType(dict(self.relations)))
        object.__setattr__(self, "facts", MappingProxyType({r: MappingProxyType(dict(t)) for r, t in self.facts.items()}))
        problems = world_problems(self)
        if problems:
            raise ValueError("invalid world: " + "; ".join(problems))
        object.__setattr__(self, "identity", content_identity({
            "types": {t: list(s.values) for t, s in sorted(self.types.items())},
            "relations": {r: [s.source, s.target] for r, s in sorted(self.relations.items())},
            "facts": {r: dict(sorted(t.items())) for r, t in sorted(self.facts.items())}}))


def world_problems(world: World) -> list[str]:
    """Why a world is malformed (empty if it is not)."""
    out = []
    for rid, spec in world.relations.items():
        if spec.source not in world.types or spec.target not in world.types:
            out.append(f"{rid}: unknown endpoint type")
            continue
        table = world.facts.get(rid)
        if table is None:
            out.append(f"{rid}: no facts")
            continue
        source, target = world.types[spec.source], world.types[spec.target]
        if set(table) != set(source.values):
            out.append(f"{rid}: facts must cover every {spec.source} exactly once")
        bad = [v for v in table.values() if not target.contains(v)]
        if bad:
            out.append(f"{rid}: targets outside {spec.target}: {bad[:3]}")
        images = list(table.values())
        if "one_to_one" in spec.properties and len(set(images)) != len(images):
            out.append(f"{rid}: declared one_to_one, but two sources share a target")
        if "many_to_one" in spec.properties and len(set(images)) == len(images):
            out.append(f"{rid}: declared many_to_one, but no two sources share a target")
    for rid in world.facts:
        if rid not in world.relations:
            out.append(f"{rid}: facts for an undeclared relation")
    return out


def _world(name, countries, cities, airports, operators, facts, seed=None) -> World:
    types = {COUNTRY: TypeSpec(COUNTRY, tuple(countries)), CITY: TypeSpec(CITY, tuple(cities)),
             AIRPORT: TypeSpec(AIRPORT, tuple(airports)), OPERATOR: TypeSpec(OPERATOR, tuple(operators))}
    return World(name, types, RELATIONS, facts, seed)


def ardel_world() -> World:
    """The small hand-written world from the README example."""
    country_cities = {"Ardel": ["Ardel Crossing", "Port Vey", "Lunholm"],
                      "Brenmark": ["Brenhaven", "Esk"],
                      "Corvala": ["Corva", "Talmont"]}
    capital = {"Ardel": "Ardel Crossing", "Brenmark": "Brenhaven", "Corvala": "Corva"}
    largest = {"Ardel": "Port Vey", "Brenmark": "Brenhaven", "Corvala": "Talmont"}
    airport = {"Ardel Crossing": "Crossing Airfield", "Port Vey": "Vey International", "Lunholm": "Lunholm Field",
               "Brenhaven": "Brenhaven Airport", "Esk": "Esk Strip", "Corva": "Corva Regional",
               "Talmont": "Talmont International"}
    operator = {"Crossing Airfield": "Northline", "Vey International": "Tessaline Air", "Lunholm Field": "Fieldway",
                "Brenhaven Airport": "Northline", "Esk Strip": "Fieldway", "Corva Regional": "Fieldway",
                "Talmont International": "Tessaline Air"}
    cities = [c for cs in country_cities.values() for c in cs]
    facts = {"capital_of": capital, "largest_city_of": largest,
             "country_of": {city: country for country, cs in country_cities.items() for city in cs},
             "airport_of": airport, "operator_of": operator}
    return _world("ardel", list(country_cities), cities, list(airport.values()),
                  sorted(set(operator.values())), facts)


def generate_world(seed: int, n_countries: int = 24, cities_per_country: tuple[int, int] = (2, 5),
                   n_operators: int = 6, same_capital_share: float = 0.2) -> World:
    """A random world. Capital and largest city coincide in about `same_capital_share` of countries, so the two
    relations usually differ; airports are one per city; operators run several airports each."""
    rng = random.Random(seed)
    countries = [f"country_{i}" for i in range(n_countries)]
    country_cities, capital, largest = {}, {}, {}
    for c in countries:
        k = rng.randint(*cities_per_country)
        cs = [f"{c}_city_{j}" for j in range(k)]
        country_cities[c] = cs
        capital[c] = rng.choice(cs)
        others = [x for x in cs if x != capital[c]]
        largest[c] = capital[c] if rng.random() < same_capital_share else rng.choice(others)
    cities = [x for cs in country_cities.values() for x in cs]
    airport = {city: f"{city}_airport" for city in cities}
    operators = [f"operator_{k}" for k in range(n_operators)]
    operator = {a: rng.choice(operators) for a in airport.values()}
    used = sorted(set(operator.values()))
    facts = {"capital_of": capital, "largest_city_of": largest,
             "country_of": {city: c for c, cs in country_cities.items() for city in cs},
             "airport_of": airport, "operator_of": operator}
    return _world(f"generated_{seed}", countries, cities, list(airport.values()), used, facts, seed)


# Plan PR 7: genuine longer paths. A route network on airports: every airport has exactly one onward connection,
# to another airport, or to itself at a terminal (no onward route). Following it from any airport eventually
# reaches a terminal or closes a cycle, so "the first airport reached from X operated by O" may be found after
# several steps, or never (a cycle or a terminal first). Earlier worlds are unchanged: this is a separate world.
ROUTE_RELATIONS: Mapping[str, RelationSpec] = MappingProxyType({
    **RELATIONS,
    "connects_to": RelationSpec("connects_to", AIRPORT, AIRPORT,
                                "the airport's one onward connection (the airport itself at a terminal)"),
})


def route_world(seed: int, n_countries: int = 48, terminal_share: float = 0.08, **kwargs) -> World:
    """`generate_world(seed, n_countries, ...)` plus a random route network on its airports."""
    base = generate_world(seed, n_countries=n_countries, **kwargs)
    rng = random.Random(f"routes|{seed}")
    airports = list(base.types[AIRPORT].values)
    connects = {a: a if rng.random() < terminal_share else rng.choice([b for b in airports if b != a]) for a in airports}
    return World(f"route_{seed}", base.types, ROUTE_RELATIONS, {**base.facts, "connects_to": connects}, seed)
