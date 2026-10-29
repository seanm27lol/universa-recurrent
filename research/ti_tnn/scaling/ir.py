"""tirel-1: the typed relational program that C4 and T4 predict, its validator and its deterministic SQLite compiler.

**Example.** The training question "Who is the director of the movie Sex, Drink and Bloodshed?" (`movie_platform`)
has gold SQL `SELECT director_name FROM movies WHERE movie_title = 'Sex, Drink and Bloodshed'`. Its tirel-1 plan is

    Select(from_=(FromItem(TableSource("movies"), "FIRST", None),),
           where=Compare("=", Column(0, 0, "movie_title"), Literal("text", "Sex, Drink and Bloodshed")),
           items=(Column(0, 0, "director_name"),))

and `compile(plan)` emits

    SELECT "T1"."director_name" FROM "movies" AS "T1" WHERE "T1"."movie_title" = 'Sex, Drink and Bloodshed'

**Intuition.** A program is a tree of typed nodes: row sets (`Select`, `Compound`), the sources a SELECT reads
(`TableSource`, `DerivedSource`) and scalar expressions. A column is never a free-text name. It is a reference
`Column(level, source, name)`: `level` counts SELECT boundaries outward from the clause that uses it (0 = the SELECT
that owns the clause, 1 = its enclosing SELECT, for correlated subqueries), and `source` indexes that SELECT's `from_`
list. Aliases are therefore not part of the program; the compiler invents unique ones (`T1`, `T2`, ... in a fixed
traversal order), so a correlated reference can never be shadowed.

**Sorts and typing.** tirel-1 has two sorts: row sets (with an arity, the number of output columns) and scalars.
SQLite is dynamically typed: `'abc' + 1`, `SUM(x > 0)` and `WHERE 5` all run, so no scalar operation is forbidden
for type reasons. The validator checks only what SQLite itself rejects when it prepares a statement, plus
well-formedness:

| Rule | SQLite's own error |
|---|---|
| every column reference resolves to a visible source that has the column | `no such column` |
| a derived table (subquery in FROM) cannot see the other sources of its own SELECT | `no such column` |
| the ON clause of a LEFT JOIN at position k sees only sources 0..k (an INNER JOIN's ON sees them all) | `ON clause references tables to its right` |
| no aggregate directly in WHERE, ON or GROUP BY, or directly inside another aggregate | `misuse of aggregate` |
| HAVING, and an aggregate in ORDER BY, need an aggregate query (GROUP BY, or an aggregate in the SELECT list) | `HAVING clause on a non-aggregate query`, `misuse of aggregate` |
| scalar and IN subqueries return one column; compound members have equal arity | `sub-select returns N columns`, `... same number of result columns` |
| `*` needs a FROM clause; compound members carry no ORDER BY or LIMIT | `no tables specified`, `ORDER BY clause should come after UNION` |
| no literal as a whole GROUP BY or ORDER BY term (an integer is a result-column position; other constants do nothing) | `GROUP BY term out of range`, or a silent change of meaning |
| a GROUP BY or ORDER BY term sees its own SELECT's sources, not an enclosing query's (even through a nested subquery) | `no such column` |

Each row was checked on this host's SQLite (3.45.1); see `tests/test_ti_tnn_scaling_ir.py`. Anything else that
SQLite accepts, the validator accepts. Declared column types are information only, never a restriction. CAST is the
one place where type names matter, and there tirel-1 stores the *affinity* SQLite derives from the type name
(INTEGER, REAL, NUMERIC, TEXT, BLOB; https://www.sqlite.org/datatype3.html section 3.1), which fully determines the
conversion. `CAST(x AS STRING)` is NUMERIC in SQLite, for instance, and stays NUMERIC here.

**Limits.** tirel-1 covers what the training release's gold SQL uses (counts in `pins/ir_coverage_v1.json`), plus
the constructs the work order names (EXISTS, INTERSECT, EXCEPT, COALESCE, IFNULL, NULLIF, simple CASE). Window
functions, GROUP_CONCAT, RIGHT/FULL/NATURAL joins, ESCAPE, COLLATE, GLOB, row values and CTEs are outside it;
`sql_to_ir.py` reports them as `Unsupported(reason)`. LIMIT and OFFSET are non-negative integer constants. The
validator does not reproduce every SQLite preparation error (for example an aggregate of an outer query hidden inside
a subquery's WHERE, or a GROUP BY term that SQLite folds to the integer 0, such as `x AND 0`); a plan it accepts can
still fail when executed, and that failure is then recorded, never hidden.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, fields, is_dataclass
from typing import Union

IR_VERSION = "tirel-1"

JOIN_KINDS = ("FIRST", "INNER", "LEFT", "CROSS")
SET_OPS = ("UNION", "UNION ALL", "INTERSECT", "EXCEPT")
ARITH_OPS = ("+", "-", "*", "/", "%", "||")
COMPARE_OPS = ("=", "<>", "<", "<=", ">", ">=")
AGG_FUNCS = ("COUNT", "SUM", "AVG", "MIN", "MAX")
AFFINITIES = ("INTEGER", "REAL", "NUMERIC", "TEXT", "BLOB")
LITERAL_KINDS = ("text", "integer", "real", "null")
INT64 = (-(2 ** 63), 2 ** 63 - 1)

# Scalar functions: name -> (minimum, maximum) argument count (None: no maximum), as SQLite accepts them.
# Measured in the training release's gold SQL: STRFTIME, SUBSTR, REPLACE, LENGTH, INSTR, TRIM, LOWER, ROUND,
# JULIANDAY, DATE, DATETIME, TIME, CURRENT_TIMESTAMP. Named by the work order although unused there: COALESCE, IFNULL,
# NULLIF, ABS, UPPER, LTRIM, RTRIM and the multi-argument (scalar) MAX/MIN.
FUNCTIONS = {
    "STRFTIME": (1, None), "JULIANDAY": (0, None), "DATE": (0, None), "DATETIME": (0, None), "TIME": (0, None),
    "CURRENT_TIMESTAMP": (0, 0),  # the keyword; compiled without parentheses
    "SUBSTR": (2, 3), "INSTR": (2, 2), "LENGTH": (1, 1), "REPLACE": (3, 3),
    "TRIM": (1, 2), "LTRIM": (1, 2), "RTRIM": (1, 2), "LOWER": (1, 1), "UPPER": (1, 1),
    "ROUND": (1, 2), "ABS": (1, 1),
    "COALESCE": (2, None), "IFNULL": (2, 2), "NULLIF": (2, 2),
    "MAX": (2, None), "MIN": (2, None),  # scalar forms; one-argument MAX/MIN are aggregates (Agg)
}


# --- row sets and sources ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TableSource:
    """A base table of the request's database (the schema's canonical spelling)."""
    table: str


@dataclass(frozen=True)
class DerivedSource:
    """A subquery in FROM. Its output columns are referenced by position (`DerivedColumn`)."""
    query: "Query"


@dataclass(frozen=True)
class FromItem:
    """One FROM entry. The first has join "FIRST"; later ones join with "INNER" (ON required), "LEFT" (ON optional)
    or "CROSS" (no ON: a Cartesian product, compiled as an unconstrained JOIN so SQLite's planner may reorder it)."""
    source: Union[TableSource, DerivedSource]
    join: str = "FIRST"
    on: "Scalar | None" = None


@dataclass(frozen=True)
class Star:
    """`*` (source None: every source, in FROM order) or `T.*` (one source)."""
    source: "int | None" = None


@dataclass(frozen=True)
class OrderItem:
    expr: "Scalar"
    descending: bool = False


@dataclass(frozen=True)
class Select:
    """One SELECT core. Clause fields follow SQL; `items` may mix scalars and `Star`."""
    from_: tuple = ()             # tuple[FromItem, ...]; empty for SELECT without FROM
    where: "Scalar | None" = None
    group_by: tuple = ()          # tuple[Scalar, ...]
    having: "Scalar | None" = None
    distinct: bool = False
    items: tuple = ()             # tuple[Scalar | Star, ...], at least one
    order_by: tuple = ()          # tuple[OrderItem, ...]
    limit: "int | None" = None
    offset: "int | None" = None   # only with a limit (SQLite has no OFFSET without LIMIT)


@dataclass(frozen=True)
class Compound:
    """`left op right`. Chains are left-nested, as SQLite evaluates them; members carry no ORDER BY or LIMIT."""
    op: str
    left: "Query"
    right: Select


# --- scalars -------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Column:
    """A column of a base-table source: `level` SELECTs outward, `source` index in that SELECT's FROM list."""
    level: int
    source: int
    name: str


@dataclass(frozen=True)
class DerivedColumn:
    """Output column `position` (0-based) of a derived-table source."""
    level: int
    source: int
    position: int


@dataclass(frozen=True)
class Literal:
    """kind "text" (str), "integer" (int), "real" (canonical decimal text, `repr(float(...))`) or "null" (None)."""
    kind: str
    value: "str | int | None" = None


@dataclass(frozen=True)
class Unary:
    """Arithmetic negation of a non-literal (negative numbers are literals)."""
    op: str
    arg: "Scalar"


@dataclass(frozen=True)
class Binary:
    """Arithmetic (+ - * / %) or string concatenation (||)."""
    op: str
    left: "Scalar"
    right: "Scalar"


@dataclass(frozen=True)
class Compare:
    op: str
    left: "Scalar"
    right: "Scalar"


@dataclass(frozen=True)
class And:
    args: tuple                   # two or more


@dataclass(frozen=True)
class Or:
    args: tuple                   # two or more


@dataclass(frozen=True)
class Not:
    arg: "Scalar"


@dataclass(frozen=True)
class Like:
    arg: "Scalar"
    pattern: "Scalar"
    negated: bool = False


@dataclass(frozen=True)
class Between:
    arg: "Scalar"
    low: "Scalar"
    high: "Scalar"
    negated: bool = False


@dataclass(frozen=True)
class InList:
    arg: "Scalar"
    items: tuple                  # one or more scalars
    negated: bool = False


@dataclass(frozen=True)
class InQuery:
    arg: "Scalar"
    query: "Query"
    negated: bool = False


@dataclass(frozen=True)
class IsNull:
    arg: "Scalar"
    negated: bool = False         # IS NOT NULL


@dataclass(frozen=True)
class Exists:
    query: "Query"
    negated: bool = False


@dataclass(frozen=True)
class When:
    condition: "Scalar"           # for a simple CASE, the value compared with the operand
    result: "Scalar"


@dataclass(frozen=True)
class Case:
    """Searched CASE (operand None) or simple CASE. IIF(c, a, b) is CASE WHEN c THEN a ELSE b END in SQLite."""
    operand: "Scalar | None"
    whens: tuple                  # tuple[When, ...], one or more
    default: "Scalar | None" = None


@dataclass(frozen=True)
class Cast:
    arg: "Scalar"
    affinity: str


@dataclass(frozen=True)
class Func:
    name: str
    args: tuple = ()


@dataclass(frozen=True)
class Agg:
    func: str
    arg: "Scalar"
    distinct: bool = False


@dataclass(frozen=True)
class CountStar:
    pass


@dataclass(frozen=True)
class ScalarQuery:
    """A scalar subquery: one column; SQLite takes the first row, or NULL for none."""
    query: "Query"


Query = Union[Select, Compound]
Scalar = Union[Column, DerivedColumn, Literal, Unary, Binary, Compare, And, Or, Not, Like, Between, InList, InQuery,
               IsNull, Exists, Case, Cast, Func, Agg, CountStar, ScalarQuery]
NODE_TYPES = {cls.__name__: cls for cls in (
    TableSource, DerivedSource, FromItem, Star, OrderItem, Select, Compound, Column, DerivedColumn, Literal, Unary,
    Binary, Compare, And, Or, Not, Like, Between, InList, InQuery, IsNull, Exists, When, Case, Cast, Func, Agg,
    CountStar, ScalarQuery)}
QUERY_TYPES = (Select, Compound)
SCALAR_TYPES = (Column, DerivedColumn, Literal, Unary, Binary, Compare, And, Or, Not, Like, Between, InList, InQuery,
                IsNull, Exists, Case, Cast, Func, Agg, CountStar, ScalarQuery)


# --- literals ----------------------------------------------------------------------------------------------------------
def canonical_real(text) -> str:
    """The canonical text of a REAL literal: Python's shortest round-trip repr (always has '.' or 'e')."""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"not a finite real: {text!r}")
    return repr(value)


def text(value: str) -> Literal:
    return Literal("text", value)


def integer(value: int) -> Literal:
    return Literal("integer", int(value))


def real(value) -> Literal:
    return Literal("real", canonical_real(value))


NULL = Literal("null", None)


def affinity_of(type_name: str) -> str:
    """SQLite's affinity for a declared or CAST type name (datatype3.html, section 3.1, rules in order)."""
    t = (type_name or "").upper()
    if "INT" in t:
        return "INTEGER"
    if "CHAR" in t or "CLOB" in t or "TEXT" in t:
        return "TEXT"
    if "BLOB" in t or not t.strip():
        return "BLOB"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "NUMERIC"


# --- serialization -------------------------------------------------------------------------------------------------
def to_json(node):
    """A JSON-ready dict tree: {"_": node type, field: value...}; tuples become lists."""
    if is_dataclass(node):
        out = {"_": type(node).__name__}
        for f in fields(node):
            out[f.name] = to_json(getattr(node, f.name))
        return out
    if isinstance(node, (tuple, list)):
        return [to_json(x) for x in node]
    return node


def from_json(obj):
    """Inverse of `to_json` (exact: `from_json(to_json(p)) == p`)."""
    if isinstance(obj, dict):
        cls = NODE_TYPES[obj["_"]]
        return cls(**{k: from_json(v) for k, v in obj.items() if k != "_"})
    if isinstance(obj, list):
        return tuple(from_json(x) for x in obj)
    return obj


def dumps(plan) -> str:
    return json.dumps({"version": IR_VERSION, "plan": to_json(plan)}, sort_keys=True, separators=(",", ":"))


def loads(s: str):
    obj = json.loads(s)
    if obj.get("version") != IR_VERSION:
        raise ValueError(f"not a {IR_VERSION} plan: {obj.get('version')!r}")
    return from_json(obj["plan"])


def children(node):
    """The direct child nodes of a node (dataclass fields, tuples flattened), in field order."""
    out = []
    for f in fields(node):
        value = getattr(node, f.name)
        if is_dataclass(value):
            out.append(value)
        elif isinstance(value, tuple):
            out.extend(v for v in value if is_dataclass(v))
    return out


def walk(node):
    """Pre-order traversal of every node (crossing subquery boundaries)."""
    yield node
    for child in children(node):
        yield from walk(child)


# --- the schema the validator checks against -----------------------------------------------------------------------
class SchemaIndex:
    """Table and column names of one database: canonical spellings, matched case-insensitively (as SQLite does)."""

    def __init__(self, tables: dict):
        self.tables = {str(t): tuple(str(c) for c in cols) for t, cols in tables.items()}
        self._table_by_lower = {t.lower(): t for t in self.tables}
        self._column_by_lower = {t: {c.lower(): c for c in cols} for t, cols in self.tables.items()}

    @classmethod
    def from_schema(cls, schema) -> "SchemaIndex":
        """From `contracts.Schema`."""
        return cls({t.name: [c.name for c in t.columns] for t in schema.tables})

    @classmethod
    def from_serialized(cls, serialized) -> "SchemaIndex":
        """From `contracts.SerializedSchema` (its items list every table and column, in order)."""
        tables: dict = {}
        for item in serialized.items:
            if item.kind == "table":
                tables.setdefault(item.table, [])
            else:
                tables.setdefault(item.table, []).append(item.column)
        return cls(tables)

    @classmethod
    def from_sqlite(cls, conn) -> "SchemaIndex":
        """From an open SQLite connection (sqlite_master order, PRAGMA table_info)."""
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE "
                                            "'sqlite\\_%' ESCAPE '\\' ORDER BY rowid")]
        out = {}
        for name in names:
            out[name] = [r[1] for r in conn.execute(f"PRAGMA table_info({quote(name)})")]
        return cls(out)

    @classmethod
    def coerce(cls, schema) -> "SchemaIndex | None":
        if schema is None or isinstance(schema, SchemaIndex):
            return schema
        if isinstance(schema, dict):
            return cls(schema)
        if hasattr(schema, "items") and hasattr(schema, "text"):
            return cls.from_serialized(schema)
        if hasattr(schema, "tables"):
            return cls.from_schema(schema)
        raise TypeError(f"cannot read a schema from {type(schema).__name__}")

    def table(self, name: str) -> "str | None":
        return self._table_by_lower.get(str(name).lower())

    def column(self, table: str, name: str) -> "str | None":
        return self._column_by_lower.get(table, {}).get(str(name).lower())

    def columns(self, table: str) -> tuple:
        return self.tables[table]


# --- validation ----------------------------------------------------------------------------------------------------
class PlanError(ValueError):
    """A plan that is not well formed, or that SQLite would refuse to prepare."""


@dataclass(frozen=True)
class _Scope:
    select: Select
    visible: bool          # False for the SELECT that contains a derived table, seen from inside that table
    sources: tuple         # indices of this SELECT's FROM items visible here
    hidden_by: str = "derived table"  # why it is not visible: "derived table", or "GROUP BY/ORDER BY term"


@dataclass(frozen=True)
class _Ctx:
    clause: str            # ON | WHERE | GROUP | HAVING | SELECT | ORDER | OTHER
    in_agg: bool = False


AGG_FORBIDDEN_CLAUSES = ("ON", "WHERE", "GROUP")


def query_arity(query, schema: "SchemaIndex | None") -> "int | None":
    """Output columns of a row set (None when a `*` over an unknown schema makes it unknowable)."""
    if isinstance(query, Compound):
        return query_arity(query.left, schema)
    total = 0
    for item in query.items:
        if isinstance(item, Star):
            chosen = range(len(query.from_)) if item.source is None else [item.source]
            for k in chosen:
                if not 0 <= k < len(query.from_):
                    return None
                n = source_arity(query.from_[k].source, schema)
                if n is None:
                    return None
                total += n
        else:
            total += 1
    return total


def source_arity(source, schema: "SchemaIndex | None") -> "int | None":
    if isinstance(source, DerivedSource):
        return query_arity(source.query, schema)
    if schema is None or schema.table(source.table) is None:
        return None
    return len(schema.columns(schema.table(source.table)))


def _direct_aggregates(expr):
    """Aggregates in an expression that are not inside a nested query or another aggregate's argument."""
    if isinstance(expr, (Agg, CountStar)):
        yield expr
        return
    if isinstance(expr, (ScalarQuery, InQuery, Exists)):
        if isinstance(expr, InQuery):
            yield from _direct_aggregates(expr.arg)
        return
    for child in children(expr):
        yield from _direct_aggregates(child)


def _outer_owned_aggregates(expr):
    """Aggregates inside nested queries of `expr` whose argument columns all point outside their own SELECT.

    SQLite assigns such an aggregate to an outer SELECT (`SELECT (SELECT COUNT(T1.x) FROM t2) FROM t1 AS T1` counts
    rows of t1). Counting every such aggregate for the enclosing SELECT can only make `is_aggregate_query` answer
    True more often, so the validator stays at most as strict as SQLite."""
    if isinstance(expr, (Agg, CountStar)):
        return
    for node in children(expr):
        if isinstance(node, QUERY_TYPES):
            for sub in walk(node):
                if isinstance(sub, Agg):
                    levels = [c.level for c in walk(sub.arg) if isinstance(c, (Column, DerivedColumn))]
                    if levels and min(levels) >= 1:
                        yield sub
        else:
            yield from _outer_owned_aggregates(node)


def is_aggregate_query(select: Select) -> bool:
    """GROUP BY, or an aggregate in the SELECT list (SQLite's test for HAVING and for aggregates in ORDER BY)."""
    if select.group_by:
        return True
    for item in select.items:
        if isinstance(item, Star):
            continue
        if any(True for _ in _direct_aggregates(item)) or any(True for _ in _outer_owned_aggregates(item)):
            return True
    return False


class _Validator:
    def __init__(self, schema: "SchemaIndex | None"):
        self.schema = schema
        self.errors: list[str] = []

    def err(self, path: str, message: str) -> None:
        self.errors.append(f"{path}: {message}")

    # row sets
    def query(self, q, stack: tuple, path: str) -> "int | None":
        if isinstance(q, Select):
            return self.select(q, stack, path)
        if isinstance(q, Compound):
            if q.op not in SET_OPS:
                self.err(path, f"unknown set operation {q.op!r}")
            if not isinstance(q.right, Select):
                self.err(path, "the right member of a compound must be a simple SELECT")
            for side, member in (("left", q.left), ("right", q.right)):
                if isinstance(member, Select) and (member.order_by or member.limit is not None
                                                   or member.offset is not None):
                    self.err(f"{path}.{side}", "a compound member cannot carry ORDER BY or LIMIT")
            a = self.query(q.left, stack, path + ".left")
            b = self.query(q.right, stack, path + ".right")
            if a is not None and b is not None and a != b:
                self.err(path, f"compound members have {a} and {b} columns")
            return a
        self.err(path, f"expected a row set, found {type(q).__name__}")
        return None

    def select(self, s: Select, stack: tuple, path: str) -> "int | None":
        if not isinstance(s.from_, tuple) or not isinstance(s.items, tuple) or not isinstance(s.group_by, tuple) \
                or not isinstance(s.order_by, tuple):
            self.err(path, "clause lists must be tuples")
            return None
        n = len(s.from_)
        everything = tuple(range(n))
        here = stack + (_Scope(s, True, everything),)
        for k, item in enumerate(s.from_):
            p = f"{path}.from_[{k}]"
            if not isinstance(item, FromItem):
                self.err(p, f"expected FromItem, found {type(item).__name__}")
                continue
            if (k == 0) != (item.join == "FIRST") or item.join not in JOIN_KINDS:
                self.err(p, f"join kind {item.join!r} is not valid at position {k}")
            if isinstance(item.source, TableSource):
                if self.schema is not None and self.schema.table(item.source.table) != item.source.table:
                    self.err(p, f"no such table {item.source.table!r} (canonical spelling required)")
            elif isinstance(item.source, DerivedSource):
                # a derived table sees neither its own SELECT's sources nor that SELECT (no LATERAL in SQLite)
                self.query(item.source.query, stack + (_Scope(s, False, ()),), p + ".query")
            else:
                self.err(p, f"expected a source, found {type(item.source).__name__}")
            if item.join == "CROSS" and item.on is not None:
                self.err(p, "a CROSS item has no ON (an ON makes it INNER)")
            if item.join == "INNER" and item.on is None:
                self.err(p, "an INNER item needs ON (without it, it is CROSS)")
            if item.join == "FIRST" and item.on is not None:
                self.err(p, "the first FROM item has no ON")
        for k, item in enumerate(s.from_):
            if isinstance(item, FromItem) and item.on is not None:
                visible = tuple(range(k + 1)) if item.join == "LEFT" else everything
                self.expr(item.on, stack + (_Scope(s, True, visible),), _Ctx("ON"), f"{path}.from_[{k}].on")
        if s.where is not None:
            self.expr(s.where, here, _Ctx("WHERE"), path + ".where")
        # GROUP BY and ORDER BY terms see this SELECT's sources (and their own subqueries') but no enclosing query's
        terms = tuple(_Scope(sc.select, False, (), "GROUP BY/ORDER BY term") for sc in stack) + (here[-1],)
        for k, g in enumerate(s.group_by):
            if isinstance(g, Literal):
                self.err(f"{path}.group_by[{k}]", "a literal GROUP BY term (SQLite reads an integer as a column position)")
            self.expr(g, terms, _Ctx("GROUP"), f"{path}.group_by[{k}]")
        if not s.items:
            self.err(path, "a SELECT needs at least one result column")
        for k, item in enumerate(s.items):
            p = f"{path}.items[{k}]"
            if isinstance(item, Star):
                if n == 0:
                    self.err(p, "* needs a FROM clause (no tables specified)")
                elif item.source is not None and not (isinstance(item.source, int) and 0 <= item.source < n):
                    self.err(p, f"* of source {item.source} out of range")
            else:
                self.expr(item, here, _Ctx("SELECT"), p)
        aggregate_query = is_aggregate_query(s)
        if s.having is not None:
            if not aggregate_query:
                self.err(path + ".having", "HAVING clause on a non-aggregate query")
            self.expr(s.having, here, _Ctx("HAVING"), path + ".having")
        for k, o in enumerate(s.order_by):
            p = f"{path}.order_by[{k}]"
            if not isinstance(o, OrderItem) or not isinstance(o.descending, bool):
                self.err(p, "expected OrderItem")
                continue
            if isinstance(o.expr, Literal):
                self.err(p, "a literal ORDER BY term (SQLite reads an integer as a column position)")
            if not aggregate_query and any(True for _ in _direct_aggregates(o.expr)):
                self.err(p, "an aggregate in ORDER BY needs an aggregate query")
            self.expr(o.expr, terms, _Ctx("ORDER"), p)
        for name in ("limit", "offset"):
            value = getattr(s, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                self.err(f"{path}.{name}", f"{name} must be a non-negative integer")
        if s.offset is not None and s.limit is None:
            self.err(path, "OFFSET needs LIMIT")
        if not isinstance(s.distinct, bool):
            self.err(path, "distinct must be a bool")
        return query_arity(s, self.schema)

    # scalars
    def column(self, e, stack: tuple, path: str):
        if not isinstance(e.level, int) or e.level < 0 or e.level >= len(stack):
            self.err(path, f"level {e.level} is outside the {len(stack)} enclosing SELECTs")
            return None
        scope = stack[-1 - e.level]
        if not scope.visible:
            self.err(path, "a derived table cannot reference its own SELECT's sources" if scope.hidden_by == "derived table"
                     else "a GROUP BY or ORDER BY term cannot reference an enclosing query's columns")
            return None
        if e.source not in scope.sources:
            if 0 <= e.source < len(scope.select.from_):
                self.err(path, "ON clause references tables to its right (LEFT JOIN)")
            else:
                self.err(path, f"source {e.source} out of range")
            return None
        return scope.select.from_[e.source].source

    def expr(self, e, stack: tuple, ctx: _Ctx, path: str) -> None:
        if isinstance(e, Column):
            src = self.column(e, stack, path)
            if src is None:
                return
            if not isinstance(src, TableSource):
                self.err(path, "Column refers to a derived table; use DerivedColumn")
            elif self.schema is not None:
                table = self.schema.table(src.table)
                if table is not None and self.schema.column(table, e.name) != e.name:
                    self.err(path, f"no such column {src.table}.{e.name} (canonical spelling required)")
        elif isinstance(e, DerivedColumn):
            src = self.column(e, stack, path)
            if src is None:
                return
            if not isinstance(src, DerivedSource):
                self.err(path, "DerivedColumn refers to a base table; use Column")
                return
            inner = src.query
            first = inner
            while isinstance(first, Compound):
                first = first.left
            if any(isinstance(i, Star) for i in first.items):
                self.err(path, "a reference into a derived table whose SELECT list has * is outside tirel-1")
            arity = query_arity(inner, self.schema)
            if not isinstance(e.position, int) or e.position < 0 or (arity is not None and e.position >= arity):
                self.err(path, f"derived column {e.position} out of range ({arity} columns)")
        elif isinstance(e, Literal):
            self.literal(e, path)
        elif isinstance(e, Unary):
            if e.op != "-":
                self.err(path, f"unknown unary operator {e.op!r}")
            self.expr(e.arg, stack, ctx, path + ".arg")
        elif isinstance(e, (Binary, Compare)):
            ops = ARITH_OPS if isinstance(e, Binary) else COMPARE_OPS
            if e.op not in ops:
                self.err(path, f"unknown operator {e.op!r}")
            self.expr(e.left, stack, ctx, path + ".left")
            self.expr(e.right, stack, ctx, path + ".right")
        elif isinstance(e, (And, Or)):
            if not isinstance(e.args, tuple) or len(e.args) < 2:
                self.err(path, f"{type(e).__name__} needs two or more arguments")
            for k, a in enumerate(e.args or ()):
                self.expr(a, stack, ctx, f"{path}.args[{k}]")
        elif isinstance(e, Not):
            self.expr(e.arg, stack, ctx, path + ".arg")
        elif isinstance(e, Like):
            self.expr(e.arg, stack, ctx, path + ".arg")
            self.expr(e.pattern, stack, ctx, path + ".pattern")
        elif isinstance(e, Between):
            for name in ("arg", "low", "high"):
                self.expr(getattr(e, name), stack, ctx, f"{path}.{name}")
        elif isinstance(e, InList):
            self.expr(e.arg, stack, ctx, path + ".arg")
            if not isinstance(e.items, tuple) or not e.items:
                self.err(path, "IN needs one or more items")
            for k, a in enumerate(e.items or ()):
                self.expr(a, stack, ctx, f"{path}.items[{k}]")
        elif isinstance(e, InQuery):
            self.expr(e.arg, stack, ctx, path + ".arg")
            self.subquery(e.query, stack, path + ".query", want_one=True)
        elif isinstance(e, IsNull):
            self.expr(e.arg, stack, ctx, path + ".arg")
        elif isinstance(e, Exists):
            self.subquery(e.query, stack, path + ".query", want_one=False)
        elif isinstance(e, ScalarQuery):
            self.subquery(e.query, stack, path + ".query", want_one=True)
        elif isinstance(e, Case):
            if e.operand is not None:
                self.expr(e.operand, stack, ctx, path + ".operand")
            if not isinstance(e.whens, tuple) or not e.whens:
                self.err(path, "CASE needs one or more WHEN")
            for k, w in enumerate(e.whens or ()):
                if not isinstance(w, When):
                    self.err(f"{path}.whens[{k}]", "expected When")
                    continue
                self.expr(w.condition, stack, ctx, f"{path}.whens[{k}].condition")
                self.expr(w.result, stack, ctx, f"{path}.whens[{k}].result")
            if e.default is not None:
                self.expr(e.default, stack, ctx, path + ".default")
        elif isinstance(e, Cast):
            if e.affinity not in AFFINITIES:
                self.err(path, f"unknown affinity {e.affinity!r}")
            self.expr(e.arg, stack, ctx, path + ".arg")
        elif isinstance(e, Func):
            spec = FUNCTIONS.get(e.name)
            if spec is None:
                self.err(path, f"function {e.name!r} is outside tirel-1")
            else:
                lo, hi = spec
                if not isinstance(e.args, tuple) or len(e.args) < lo or (hi is not None and len(e.args) > hi):
                    self.err(path, f"{e.name} takes {lo}..{hi if hi is not None else 'n'} arguments")
            for k, a in enumerate(e.args or ()):
                self.expr(a, stack, ctx, f"{path}.args[{k}]")
        elif isinstance(e, (Agg, CountStar)):
            if ctx.clause in AGG_FORBIDDEN_CLAUSES:
                self.err(path, f"misuse of aggregate: not allowed in {ctx.clause}")
            if ctx.in_agg:
                self.err(path, "misuse of aggregate: nested inside another aggregate")
            if isinstance(e, Agg):
                if e.func not in AGG_FUNCS:
                    self.err(path, f"unknown aggregate {e.func!r}")
                if not isinstance(e.distinct, bool):
                    self.err(path, "distinct must be a bool")
                self.expr(e.arg, stack, _Ctx(ctx.clause, True), path + ".arg")
        else:
            self.err(path, f"expected a scalar, found {type(e).__name__}")

    def subquery(self, q, stack: tuple, path: str, want_one: bool) -> None:
        arity = self.query(q, stack, path)
        if want_one and arity is not None and arity != 1:
            self.err(path, f"sub-select returns {arity} columns - expected 1")

    def literal(self, e: Literal, path: str) -> None:
        kind, value = e.kind, e.value
        if kind == "text":
            ok = isinstance(value, str)
        elif kind == "integer":
            ok = isinstance(value, int) and not isinstance(value, bool) and INT64[0] <= value <= INT64[1]
        elif kind == "real":
            try:
                ok = isinstance(value, str) and canonical_real(value) == value
            except ValueError:
                ok = False
        elif kind == "null":
            ok = value is None
        else:
            ok = False
        if not ok:
            self.err(path, f"invalid {kind} literal {value!r}")


def validate(plan, schema=None) -> list[str]:
    """Every problem found (empty: valid). `schema` (contracts.Schema, SerializedSchema, a {table: [columns]} dict
    or a SchemaIndex) enables table, column and `*`-arity checks; without it only structure and scope are checked."""
    v = _Validator(SchemaIndex.coerce(schema))
    v.query(plan, (), "plan")
    return v.errors


def check(plan, schema=None) -> None:
    errors = validate(plan, schema)
    if errors:
        raise PlanError("; ".join(errors[:5]) + (f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""))


# --- the compiler ----------------------------------------------------------------------------------------------------
def quote(name: str) -> str:
    """A double-quoted SQLite identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


# precedence, higher binds tighter (https://www.sqlite.org/lang_expr.html, "Operators")
ATOM, UNARY, CONCAT, MUL, ADD, LT, EQ, NOT_P, AND_P, OR_P = 100, 10, 9, 8, 7, 5, 4, 3, 2, 1
_BINARY_PREC = {"||": CONCAT, "*": MUL, "/": MUL, "%": MUL, "+": ADD, "-": ADD}
_COMPARE_PREC = {"<": LT, "<=": LT, ">": LT, ">=": LT, "=": EQ, "<>": EQ}


class _Compiler:
    def __init__(self):
        self.counter = 0
        self.stack: list[list[str]] = []

    def alias(self) -> str:
        self.counter += 1
        return f"T{self.counter}"

    def query(self, q, derived_names: bool = False) -> str:
        if isinstance(q, Compound):
            left = self.query(q.left, derived_names)
            right = self.query(q.right, derived_names)
            return f"{left} {q.op} {right}"
        return self.select(q, derived_names)

    def select(self, s: Select, derived_names: bool) -> str:
        aliases = [self.alias() for _ in s.from_]
        self.stack.append(aliases)
        try:
            # compiled in the action grammar's order (so alias numbers are stable), emitted in SQL order
            from_parts = []
            for k, item in enumerate(s.from_):
                if isinstance(item.source, TableSource):
                    src = f"{quote(item.source.table)} AS {quote(aliases[k])}"
                else:
                    src = f"({self.query(item.source.query, derived_names=True)}) AS {quote(aliases[k])}"
                from_parts.append(src)
            joins = []
            for k, item in enumerate(s.from_):
                if k == 0:
                    joins.append(from_parts[0])
                    continue
                keyword = {"INNER": "INNER JOIN", "LEFT": "LEFT JOIN", "CROSS": "JOIN"}[item.join]
                on = f" ON {self.expr(item.on)[0]}" if item.on is not None else ""
                joins.append(f"{keyword} {from_parts[k]}{on}")
            where = self.expr(s.where)[0] if s.where is not None else None
            group = [self.expr(g)[0] for g in s.group_by]
            items = []
            for k, item in enumerate(s.items):
                if isinstance(item, Star):
                    items.append("*" if item.source is None else f"{quote(aliases[item.source])}.*")
                else:
                    sql = self.expr(item)[0]
                    items.append(f"{sql} AS {quote(f'c{k}')}" if derived_names else sql)
            having = self.expr(s.having)[0] if s.having is not None else None
            order = [self.expr(o.expr)[0] + (" DESC" if o.descending else "") for o in s.order_by]
        finally:
            self.stack.pop()
        out = "SELECT " + ("DISTINCT " if s.distinct else "") + ", ".join(items)
        if joins:
            out += " FROM " + " ".join(joins)
        if where is not None:
            out += " WHERE " + where
        if group:
            out += " GROUP BY " + ", ".join(group)
        if having is not None:
            out += " HAVING " + having
        if order:
            out += " ORDER BY " + ", ".join(order)
        if s.limit is not None:
            out += f" LIMIT {s.limit}"
            if s.offset is not None:
                out += f" OFFSET {s.offset}"
        return out

    def ref(self, level: int, source: int) -> str:
        return quote(self.stack[-1 - level][source])

    def wrap(self, e, at_most: int) -> str:
        """Compile `e`, parenthesized when its precedence is at most `at_most` (conservative on ties)."""
        sql, prec = self.expr(e)
        return f"({sql})" if prec <= at_most else sql

    def expr(self, e) -> tuple[str, int]:
        if isinstance(e, Column):
            return f"{self.ref(e.level, e.source)}.{quote(e.name)}", ATOM
        if isinstance(e, DerivedColumn):
            return f"{self.ref(e.level, e.source)}.{quote(f'c{e.position}')}", ATOM
        if isinstance(e, Literal):
            if e.kind == "text":
                return sql_string(e.value), ATOM
            if e.kind == "null":
                return "NULL", ATOM
            sql = str(e.value)
            return sql, (UNARY if sql.startswith("-") else ATOM)
        if isinstance(e, Unary):
            return f"-{self.wrap(e.arg, ATOM - 1)}", UNARY
        if isinstance(e, Binary):
            p = _BINARY_PREC[e.op]
            return f"{self.wrap(e.left, p)} {e.op} {self.wrap(e.right, p)}", p
        if isinstance(e, Compare):
            p = _COMPARE_PREC[e.op]
            return f"{self.wrap(e.left, p)} {e.op} {self.wrap(e.right, p)}", p
        if isinstance(e, And):
            return " AND ".join(self.wrap(a, AND_P) for a in e.args), AND_P
        if isinstance(e, Or):
            return " OR ".join(self.wrap(a, OR_P) for a in e.args), OR_P
        if isinstance(e, Not):
            return f"NOT {self.wrap(e.arg, NOT_P)}", NOT_P
        if isinstance(e, Like):
            op = "NOT LIKE" if e.negated else "LIKE"
            return f"{self.wrap(e.arg, EQ)} {op} {self.wrap(e.pattern, EQ)}", EQ
        if isinstance(e, Between):
            op = "NOT BETWEEN" if e.negated else "BETWEEN"
            return f"{self.wrap(e.arg, EQ)} {op} {self.wrap(e.low, EQ)} AND {self.wrap(e.high, EQ)}", EQ
        if isinstance(e, InList):
            op = "NOT IN" if e.negated else "IN"
            return f"{self.wrap(e.arg, EQ)} {op} ({', '.join(self.expr(i)[0] for i in e.items)})", EQ
        if isinstance(e, InQuery):
            op = "NOT IN" if e.negated else "IN"
            return f"{self.wrap(e.arg, EQ)} {op} ({self.query(e.query)})", EQ
        if isinstance(e, IsNull):
            return f"{self.wrap(e.arg, EQ)} IS {'NOT ' if e.negated else ''}NULL", EQ
        if isinstance(e, Exists):
            return (f"NOT EXISTS ({self.query(e.query)})", NOT_P) if e.negated else \
                (f"EXISTS ({self.query(e.query)})", ATOM)
        if isinstance(e, ScalarQuery):
            return f"({self.query(e.query)})", ATOM
        if isinstance(e, Case):
            parts = ["CASE"]
            if e.operand is not None:
                parts.append(self.expr(e.operand)[0])
            for w in e.whens:
                parts.append(f"WHEN {self.expr(w.condition)[0]} THEN {self.expr(w.result)[0]}")
            if e.default is not None:
                parts.append(f"ELSE {self.expr(e.default)[0]}")
            parts.append("END")
            return " ".join(parts), ATOM
        if isinstance(e, Cast):
            return f"CAST({self.expr(e.arg)[0]} AS {e.affinity})", ATOM
        if isinstance(e, Func):
            if e.name == "CURRENT_TIMESTAMP":
                return "CURRENT_TIMESTAMP", ATOM
            return f"{e.name}({', '.join(self.expr(a)[0] for a in e.args)})", ATOM
        if isinstance(e, Agg):
            return f"{e.func}({'DISTINCT ' if e.distinct else ''}{self.expr(e.arg)[0]})", ATOM
        if isinstance(e, CountStar):
            return "COUNT(*)", ATOM
        raise PlanError(f"cannot compile {type(e).__name__}")


def compile(plan) -> str:  # noqa: A001 - the work order's name for the compiler
    """Deterministic SQLite SQL for a plan: quoted identifiers, aliases T1, T2, ... in the action grammar's traversal
    order (FROM, ON, WHERE, GROUP BY, SELECT, HAVING, ORDER BY), conservative parentheses."""
    if not isinstance(plan, QUERY_TYPES):
        raise PlanError(f"a plan is a row set, not {type(plan).__name__}")
    return _Compiler().query(plan)


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def operator_inventory(plan) -> dict:
    """Operator -> count for one plan (the coverage pin's inventory vocabulary)."""
    counts: dict = {}

    def bump(key):
        counts[key] = counts.get(key, 0) + 1

    for node in walk(plan):
        name = type(node).__name__
        if isinstance(node, Select):
            bump("SELECT")
            if node.distinct:
                bump("SELECT DISTINCT")
            if node.where is not None:
                bump("WHERE")
            if node.group_by:
                bump("GROUP BY")
            if node.having is not None:
                bump("HAVING")
            if node.order_by:
                bump("ORDER BY")
            if node.limit is not None:
                bump("LIMIT")
            if node.offset is not None:
                bump("OFFSET")
            if not node.from_:
                bump("SELECT without FROM")
        elif isinstance(node, Compound):
            bump(node.op)
        elif isinstance(node, FromItem):
            if node.join != "FIRST":
                bump(f"JOIN {node.join}")
        elif isinstance(node, DerivedSource):
            bump("derived table")
        elif isinstance(node, OrderItem):
            bump("ORDER DESC" if node.descending else "ORDER ASC")
        elif isinstance(node, Star):
            bump("SELECT *" if node.source is None else "SELECT T.*")
        elif isinstance(node, (Binary, Compare)):
            bump(node.op)
        elif isinstance(node, Unary):
            bump("unary -")
        elif isinstance(node, (Like, Between, InList, InQuery, IsNull, Exists)):
            bump(("NOT " if node.negated else "") + {"Like": "LIKE", "Between": "BETWEEN", "InList": "IN list",
                                                       "InQuery": "IN subquery", "IsNull": "IS NULL",
                                                       "Exists": "EXISTS"}[name])
        elif isinstance(node, (And, Or, Not)):
            bump(name.upper())
        elif isinstance(node, Case):
            bump("CASE simple" if node.operand is not None else "CASE")
        elif isinstance(node, Cast):
            bump(f"CAST {node.affinity}")
        elif isinstance(node, Func):
            bump(f"{node.name}/{len(node.args)}")
        elif isinstance(node, Agg):
            bump(f"{node.func}{' DISTINCT' if node.distinct else ''}")
        elif isinstance(node, CountStar):
            bump("COUNT(*)")
        elif isinstance(node, ScalarQuery):
            bump("scalar subquery")
        elif isinstance(node, Literal):
            bump(f"literal {node.kind}")
        elif isinstance(node, DerivedColumn):
            bump("derived column")
        elif isinstance(node, Column) and node.level > 0:
            bump("correlated column")
    return counts
