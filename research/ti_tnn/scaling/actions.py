"""The action grammar: a deterministic, invertible traversal of tirel-1 plans, one typed decision per step.

**Example.** The plan for "Who is the director of the movie Sex, Drink and Bloodshed?" is built by the actions

    QUERY_SELECT               projection   the row set is one SELECT
    FROM_TABLE(movies)         schema       pointer to the schema item `movies`
    END                        schema       no joins
    EQ                         predicate    WHERE <left> = <right>
    COL0(movies.movie_title)   predicate    pointer to the schema item `movies.movie_title`
    TEXT_AS_NONE(span)         predicate    pointer to the evidence tokens 'Sex, Drink and Bloodshed'
    END                        aggregate    no GROUP BY
    SELECT_ALL                 projection   no DISTINCT
    COL0(movies.director_name) projection   the first result column
    END                        projection   no more result columns
    NONE                       predicate    no HAVING
    END                        projection   no ORDER BY
    NONE                       projection   no LIMIT

and then STOP. `decode(actions)` rebuilds exactly the same plan (`encode` and `decode` are inverse on every plan the
grammar covers; the tests check it on every convertible gold query).

**The traversal (ASDL / TRANX style; Yin and Neubig, "TRANX", EMNLP 2018).** A plan is filled in pre-order over typed
slots. The machine keeps the partial plan and an explicit stack of unfinished subproblems: a query slot, a SELECT in
one of its phases, an operator waiting for its operands, or a scalar slot. A SELECT is built in the logical order
FROM (all sources, then each join's ON) -> WHERE -> GROUP BY -> SELECT list -> HAVING -> ORDER BY -> LIMIT -> OFFSET,
so every column pointer is chosen after the tables it may refer to. The SELECT list precedes HAVING and ORDER BY so
that the masks know whether the query aggregates.

**Families.** Every action belongs to one of four semantic families, or is STOP:

| Family | Decides | Always this family |
|---|---|---|
| schema (0) | FROM tables, joins, join keys (the whole ON clause), derived tables | FROM_*, JOIN_* |
| predicate (1) | WHERE/HAVING predicates, comparison operators, literals | EQ...NOT, TEXT_*, NUM, CONST_*, NULL (comparisons inside ON belong to schema) |
| aggregate (2) | aggregate functions, GROUP BY | AGG_*, COUNT_STAR |
| projection (3) | SELECT items, DISTINCT, ORDER BY, LIMIT, set operations | QUERY_*, SELECT_*, STAR*, ORDER_*, LIMIT_*, OFFSET_* |

The remaining operators (column pointers, arithmetic, CAST, functions, CASE, scalar subqueries, END, NONE) take the
family of the clause they are in: ON -> schema, WHERE and HAVING -> predicate, GROUP BY -> aggregate, SELECT list and
ORDER BY -> projection. Inside a predicate's operands the clause family becomes predicate (schema inside ON), and inside
an aggregate's argument it becomes aggregate. So `SUM(CASE WHEN T1.gender = 'M' THEN 1 ELSE 0 END)` is SUM (aggregate),
CASE (aggregate), `=` (predicate), the column (predicate), 'M' and the constants (predicate).

**Arguments.** Pointers go to schema items (`SerializedSchema.items` indices: tables, columns) and to token spans of
the question and evidence (absolute token indices into the encoded context). Everything else is a closed vocabulary
folded into the operator: comparison and arithmetic operators, function names with their arity, CAST affinities,
small integers (LIMIT 1..20, OFFSET 0..20, the k-th occurrence of a self-joined table, derived-table positions) and a
constant vocabulary measured on the training pool.

**Literal grounding v1.** A literal is
- `NULL`;
- a constant (`CONST_i`) when its value is in `CONSTANTS`, always, so small integers never use spans;
- otherwise a token span: `NUM(span)` for numbers, or `TEXT_<case>_<wildcard>(span)` for text.

A span decodes as follows: take the characters the tokens cover; strip surrounding whitespace; drop one leading quote;
drop a trailing quote together with any `;,.:!?)]` that follows it. NUM then removes thousands separators
(`1,000` -> 1000) and parses an integer, or a real (canonical repr). TEXT applies a case form (AS is unchanged, CAP
upper-cases the first character, UP, LO, TTL is Python's `str.title`) and a wildcard (NONE, ANY `%x%`, PRE `x%`,
SUF `%x`). The gold span is the first match in this order:
- wildcard NONE before ANY, PRE, SUF;
- case AS, CAP, UP, LO, TTL;
- the evidence before the question;
- the shortest span, then the earliest.

A literal with no such span is not groundable. Its question is unsupported for supervised training, but it stays in
every evaluation denominator.

**Masks** come from the grammar state and scope only, and they are no stricter than SQLite. They enforce:
- columns of visible sources only (a derived table cannot see its siblings; a LEFT JOIN's ON sees sources 0..k);
- no aggregate in WHERE, ON or GROUP BY, or inside another aggregate;
- HAVING, and aggregates in ORDER BY, only in an aggregate query;
- single-column scalar and IN subqueries, and equal arity across compound members;
- `*` only with a FROM clause.

Three canonical-form rules mirror `sql_to_ir`: no AND directly inside AND or OR inside OR (flattening), NOT never
directly over a negatable predicate (use NOT_LIKE etc.), and unary minus never directly over a number (negative
numbers are literals). None of them removes a query SQLite could run; each removes a second spelling of one.

**Limits.** The closed vocabularies cap what can be encoded: 4 occurrences of one table in scope, 2 derived tables in
scope × 8 positions, LIMIT <= 20, OFFSET <= 20, the listed (function, arity) pairs, CASE with <= 4 WHENs (simple
CASE <= 2), <= 10 AND/OR operands, <= 12 IN items, 8 FROM items, 16 result columns, 8 GROUP BY and 4 ORDER BY items,
query nesting depth 6. Plans outside them raise `NotEncodable(reason)`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import ir

GRAMMAR_VERSION = "tirel-1/actions-v1"

FAMILIES = ("schema", "predicate", "aggregate", "projection")
SCHEMA, PREDICATE, AGGREGATE, PROJECTION = 0, 1, 2, 3
STOP = 4
N_CLASSES = 5  # four families + STOP

# --- closed vocabularies -------------------------------------------------------------------------------------------
LIMIT_VALUES = tuple(range(1, 21))
OFFSET_VALUES = tuple(range(0, 21))
OCCURRENCES = 4                # COL0..COL3: the k-th visible occurrence of the column's table
DERIVED_SOURCES = 2            # DCOL_<s>_<p>: s-th visible derived table ...
DERIVED_POSITIONS = 8          # ... output position p
MAX_CASE_WHENS = 4
MAX_SIMPLE_CASE_WHENS = 2
MAX_SPAN_TOKENS = 32
MAX_FROM_ITEMS, MAX_GROUP, MAX_ITEMS, MAX_ORDER = 8, 8, 16, 4
MAX_BOOL_ARGS, MAX_IN_ITEMS, MAX_QUERY_DEPTH = 10, 12, 6

# Literal constants: every value whose occurrences in training-pool gold SQL (55 databases, 5,071 questions) were not
# found in the question or evidence at least 3 times, restricted for text to symbols and STRFTIME format codes (label
# words such as 'false' or 'YES' stay ungroundable), plus the small integers 0..10.
CONSTANTS = (
    ("integer", -4), ("integer", 0), ("integer", 1), ("integer", 2), ("integer", 3), ("integer", 4), ("integer", 5),
    ("integer", 6), ("integer", 7), ("integer", 8), ("integer", 9), ("integer", 10), ("integer", 30), ("integer", 60),
    ("integer", 100), ("integer", 365), ("integer", 100000),
    ("real", "1.0"), ("real", "1.8"), ("real", "100.0"),
    ("text", ""), ("text", ","), ("text", "$"), ("text", "%Y"), ("text", "%m"), ("text", "%d"), ("text", "%H"),
    ("text", "%M"), ("text", "%S"), ("text", "%J"), ("text", "%Y-%m"), ("text", "%Y-%m-%d"),
)
CONSTANT_INDEX = {c: i for i, c in enumerate(CONSTANTS)}

# (function, arity) pairs the grammar can emit: those in the training release's gold SQL, and the work order's.
# IFNULL is absent: sql_to_ir writes IFNULL(a, b) as COALESCE(a, b).
FUNCTION_ARITIES = (
    ("STRFTIME", 2), ("STRFTIME", 3), ("SUBSTR", 2), ("SUBSTR", 3), ("INSTR", 2), ("LENGTH", 1), ("REPLACE", 3),
    ("TRIM", 1), ("TRIM", 2), ("LTRIM", 1), ("LTRIM", 2), ("RTRIM", 1), ("RTRIM", 2), ("LOWER", 1), ("UPPER", 1),
    ("ROUND", 1), ("ROUND", 2), ("ABS", 1), ("JULIANDAY", 1), ("DATE", 1), ("DATE", 2), ("DATETIME", 1),
    ("DATETIME", 2), ("TIME", 1), ("CURRENT_TIMESTAMP", 0), ("COALESCE", 2), ("COALESCE", 3), ("NULLIF", 2),
    ("MAX", 2), ("MIN", 2),
)
CASE_FORMS = ("AS", "CAP", "UP", "LO", "TTL")
WILDCARDS = ("NONE", "ANY", "PRE", "SUF")
FORM_FN = {"AS": lambda s: s, "CAP": lambda s: s[:1].upper() + s[1:], "UP": str.upper, "LO": str.lower,
           "TTL": str.title}
WILD_FN = {"NONE": lambda s: s, "ANY": lambda s: "%" + s + "%", "PRE": lambda s: s + "%", "SUF": lambda s: "%" + s}

# --- operators -----------------------------------------------------------------------------------------------------
QUERY_OPS = ("QUERY_SELECT", "QUERY_UNION", "QUERY_UNION_ALL", "QUERY_INTERSECT", "QUERY_EXCEPT")
SET_OP = {"QUERY_UNION": "UNION", "QUERY_UNION_ALL": "UNION ALL", "QUERY_INTERSECT": "INTERSECT",
          "QUERY_EXCEPT": "EXCEPT"}
SET_OP_ACTION = {v: k for k, v in SET_OP.items()}
JOIN_KINDS = ("INNER", "LEFT", "CROSS")
FROM_OPS = ("FROM_TABLE", "FROM_DERIVED", "FROM_NONE") + tuple(
    f"JOIN_{k}_{s}" for k in JOIN_KINDS for s in ("TABLE", "DERIVED"))
SHAPE_OPS = ("SELECT_ALL", "SELECT_DISTINCT", "STAR") + tuple(f"STAR_{k}" for k in range(4)) + (
    "ORDER_ASC", "ORDER_DESC") + tuple(f"LIMIT_{v}" for v in LIMIT_VALUES) + tuple(f"OFFSET_{v}" for v in OFFSET_VALUES)
COMPARE_ACTIONS = {"EQ": "=", "NE": "<>", "LT": "<", "LE": "<=", "GT": ">", "GE": ">="}
COMPARE_OF = {v: k for k, v in COMPARE_ACTIONS.items()}
NEGATABLE = {"LIKE": "NOT_LIKE", "BETWEEN": "NOT_BETWEEN", "IN_LIST": "NOT_IN_LIST", "IN_QUERY": "NOT_IN_QUERY",
             "IS_NULL": "IS_NOT_NULL", "EXISTS": "NOT_EXISTS"}
PREDICATE_OPS = tuple(COMPARE_ACTIONS) + tuple(x for pair in NEGATABLE.items() for x in pair) + ("AND", "OR", "NOT")
TEXT_OPS = tuple(f"TEXT_{f}_{w}" for f in CASE_FORMS for w in WILDCARDS)
LITERAL_OPS = TEXT_OPS + ("NUM", "NULL") + tuple(f"CONST_{i}" for i in range(len(CONSTANTS)))
AGG_OPS = tuple(f"AGG_{f}{d}" for f in ir.AGG_FUNCS for d in ("", "_DISTINCT")) + ("COUNT_STAR",)
ARITH_ACTIONS = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/", "MOD": "%", "CONCAT": "||"}
ARITH_OF = {v: k for k, v in ARITH_ACTIONS.items()}
CAST_OPS = tuple(f"CAST_{a}" for a in ir.AFFINITIES)
FN_OPS = tuple(f"FN_{n}_{a}" for n, a in FUNCTION_ARITIES)
CASE_OPS = tuple(f"CASE_{k}{e}" for k in range(1, MAX_CASE_WHENS + 1) for e in ("", "_ELSE")) + tuple(
    f"CASE_SIMPLE_{k}{e}" for k in range(1, MAX_SIMPLE_CASE_WHENS + 1) for e in ("", "_ELSE"))
COL_OPS = tuple(f"COL{k}" for k in range(OCCURRENCES))
DCOL_OPS = tuple(f"DCOL_{s}_{p}" for s in range(DERIVED_SOURCES) for p in range(DERIVED_POSITIONS))
NEUTRAL_OPS = ("END", "NONE") + COL_OPS + DCOL_OPS + tuple(ARITH_ACTIONS) + ("NEG",) + CAST_OPS + FN_OPS + CASE_OPS \
    + ("SUBQ",)

FAMILY_OPS = {
    SCHEMA: FROM_OPS + PREDICATE_OPS + NEUTRAL_OPS,
    PREDICATE: PREDICATE_OPS + LITERAL_OPS + NEUTRAL_OPS,
    AGGREGATE: AGG_OPS + NEUTRAL_OPS,
    PROJECTION: QUERY_OPS + SHAPE_OPS + NEUTRAL_OPS,
}
# The C4 output space: the disjoint union of the four families' operator vocabularies, in family order. T4's head f
# outputs exactly the slice of family f.
UNION = tuple((f, op) for f in range(4) for op in FAMILY_OPS[f])
UNION_INDEX = {key: i for i, key in enumerate(UNION)}
FAMILY_SLICES = {}
_start = 0
for _f in range(4):
    FAMILY_SLICES[_f] = (_start, _start + len(FAMILY_OPS[_f]))
    _start += len(FAMILY_OPS[_f])
OP_NAMES = tuple(sorted({op for _, op in UNION}))
OP_ID = {op: i + 1 for i, op in enumerate(OP_NAMES)}  # 0: no parent operator

_SET = {name: frozenset(ops) for name, ops in (("literal", LITERAL_OPS), ("agg", AGG_OPS), ("pred", PREDICATE_OPS),
                                                ("from", FROM_OPS), ("query", QUERY_OPS), ("shape", SHAPE_OPS))}
POINTER_KIND = {op: None for _, op in UNION}
for _op in ("FROM_TABLE",) + tuple(f"JOIN_{k}_TABLE" for k in JOIN_KINDS):
    POINTER_KIND[_op] = "table"
for _op in COL_OPS:
    POINTER_KIND[_op] = "column"
for _op in TEXT_OPS + ("NUM",):
    POINTER_KIND[_op] = "span"
POINTER_KINDS = ("table", "column", "span")

SLOT_KINDS = ("QUERY", "FROM_FIRST", "FROM_MORE", "ON", "WHERE", "GROUP", "DISTINCT", "ITEM", "HAVING", "ORDER_DIR",
              "ORDER_EXPR", "LIMIT", "OFFSET", "ARG", "MORE", "DONE")
SLOT_ID = {k: i for i, k in enumerate(SLOT_KINDS)}
CLAUSES = ("QUERY", "FROM", "ON", "WHERE", "GROUP", "SELECT", "HAVING", "ORDER", "LIMIT")
N_FEATURES = 40


class NotEncodable(Exception):
    """A plan (or literal) outside what the grammar's closed vocabularies can express."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class IllegalAction(ValueError):
    """An action the masks forbid at this step."""


@dataclass(frozen=True)
class Action:
    """`op` plus its pointer: a schema item index (table or column) or an inclusive token span (start, end)."""
    op: str
    arg: object = None

    def as_json(self) -> list:
        return [self.op, list(self.arg) if isinstance(self.arg, tuple) else self.arg]

    @staticmethod
    def from_json(obj) -> "Action":
        op, arg = obj
        return Action(op, tuple(arg) if isinstance(arg, list) else arg)


def family_of(op: str, clause: str, context_family: int) -> int:
    if op in _SET["literal"]:
        return PREDICATE
    if op in _SET["agg"]:
        return AGGREGATE
    if op in _SET["pred"]:
        return SCHEMA if clause == "ON" else PREDICATE
    if op in _SET["from"]:
        return SCHEMA
    if op in _SET["query"] or op in _SET["shape"]:
        return PROJECTION
    return context_family


# --- literal spans -------------------------------------------------------------------------------------------------
QUOTES = "'\"`‘’“”"
TRAILING = ";,.:!?)]"
_INT = re.compile(r"-?\d+")
_REAL = re.compile(r"-?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?|-?\d+[eE][-+]?\d+")
_THOUSANDS = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")


def normalize_span_text(raw: str) -> str:
    """Strip whitespace, one leading quote, and a trailing quote with any punctuation after it."""
    t = raw.strip()
    if t[:1] and t[0] in QUOTES:
        t = t[1:]
    u = t.rstrip(TRAILING)
    if u != t and u[-1:] and u[-1] in QUOTES:
        t = u[:-1]
    elif t[-1:] and t[-1] in QUOTES:
        t = t[:-1]
    return t


def number_literal(raw: str) -> "ir.Literal | None":
    t = normalize_span_text(raw).strip()
    if _THOUSANDS.fullmatch(t):
        t = t.replace(",", "")
    if _INT.fullmatch(t):
        value = int(t)
        return ir.integer(value) if ir.INT64[0] <= value <= ir.INT64[1] else None
    if _REAL.fullmatch(t):
        try:
            return ir.real(t)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class Region:
    """One pointable text region (the question or the evidence) and its tokens: (absolute token index, start, end),
    character offsets into `text`, clipped to it."""
    name: str
    text: str
    tokens: tuple


class SpanIndex:
    """The question/evidence tokens a literal pointer may select, and the span decoder shared by encode and decode."""
    PRIORITY = ("evidence", "question")

    def __init__(self, regions):
        self.regions = tuple(regions)
        self._where = {}
        for r, region in enumerate(self.regions):
            for local, (absolute, _s, _e) in enumerate(region.tokens):
                self._where[absolute] = (r, local)
        self._valid = None
        self._ground = {}

    def positions(self) -> list:
        return sorted(self._where)

    def raw(self, start: int, end: int) -> "str | None":
        a, b = self._where.get(start), self._where.get(end)
        if a is None or b is None or a[0] != b[0] or b[1] < a[1] or b[1] - a[1] >= MAX_SPAN_TOKENS:
            return None
        region = self.regions[a[0]]
        return region.text[region.tokens[a[1]][1]:region.tokens[b[1]][2]]

    def text(self, start: int, end: int) -> "str | None":
        raw = self.raw(start, end)
        if raw is None:
            return None
        t = normalize_span_text(raw)
        return t if t else None

    def number(self, start: int, end: int) -> "ir.Literal | None":
        raw = self.raw(start, end)
        return None if raw is None else number_literal(raw)

    def _spans(self):
        """Every (start, end) span within MAX_SPAN_TOKENS, in grounding priority order."""
        order = sorted(range(len(self.regions)), key=lambda r: (self.PRIORITY.index(self.regions[r].name)
                                                                if self.regions[r].name in self.PRIORITY else 9, r))
        out = []
        for r in order:
            toks = self.regions[r].tokens
            spans = [(j - i, toks[i][0], toks[j][0]) for i in range(len(toks))
                     for j in range(i, min(len(toks), i + MAX_SPAN_TOKENS))]
            spans.sort()
            out.extend((s, e) for _, s, e in spans)
        return out

    def valid(self):
        """{"text": {start: [ends]}, "number": {start: [ends]}}: spans that decode to a literal."""
        if self._valid is None:
            text, number = {}, {}
            for s, e in self._spans():
                if self.text(s, e) is not None:
                    text.setdefault(s, []).append(e)
                if self.number(s, e) is not None:
                    number.setdefault(s, []).append(e)
            self._valid = {"text": {k: sorted(v) for k, v in text.items()},
                           "number": {k: sorted(v) for k, v in number.items()}}
        return self._valid

    def starts(self, kind: str) -> tuple:
        return tuple(sorted(self.valid()[kind]))

    def ends(self, kind: str, start: int) -> tuple:
        return tuple(self.valid()[kind].get(start, ()))

    def ground_text(self, core: str, form: str):
        table = self._ground.get(form)
        if table is None:
            table = {}
            fn = FORM_FN[form]
            for s, e in self._spans():
                t = self.text(s, e)
                if t is not None:
                    table.setdefault(fn(t), (s, e))
            self._ground[form] = table
        return table.get(core)

    def ground_number(self, literal: ir.Literal):
        table = self._ground.get("#")
        if table is None:
            table = {}
            for s, e in self._spans():
                lit = self.number(s, e)
                if lit is not None:
                    table.setdefault((lit.kind, lit.value), (s, e))
            self._ground["#"] = table
        return table.get((literal.kind, literal.value))


_WORD = re.compile(r"\s*(?:\w+|[^\w\s])", re.UNICODE)


def simple_regions(question: str, evidence: str, first_index: int = 0) -> SpanIndex:
    """A tokenizer-free SpanIndex (words and single punctuation marks, leading whitespace attached, as byte-level BPE
    pre-tokenizers do). For tests and CPU tools; the controller uses the backbone tokenizer's offsets (encode.py)."""
    regions, k = [], first_index
    for name, text in (("question", question), ("evidence", evidence)):
        toks = []
        for m in _WORD.finditer(text):
            toks.append((k, m.start(), m.end()))
            k += 1
        regions.append(Region(name, text, tuple(toks)))
    return SpanIndex(regions)


# --- the schema the pointers address -------------------------------------------------------------------------------
class SchemaItems:
    """Pointer targets: `SerializedSchema.items` (kind, table, column), in order; tables and columns by index."""

    def __init__(self, items):
        self.items = tuple(items)
        self.table_item, self.column_item, self.columns_of = {}, {}, {}
        tables: dict = {}
        for i, it in enumerate(self.items):
            if it.kind == "table":
                self.table_item[it.table] = i
                tables.setdefault(it.table, [])
                self.columns_of.setdefault(it.table, [])
            else:
                self.column_item[(it.table, it.column)] = i
                tables.setdefault(it.table, []).append(it.column)
                self.columns_of.setdefault(it.table, []).append(i)
        self.index = ir.SchemaIndex(tables)
        self.tables = tuple(sorted(self.table_item.values()))

    @classmethod
    def coerce(cls, schema) -> "SchemaItems":
        if isinstance(schema, SchemaItems):
            return schema
        if hasattr(schema, "items") and not isinstance(schema, dict):
            return cls(schema.items)
        if isinstance(schema, dict):  # {table: [columns]}: synthetic item list
            items = []
            for t, cols in schema.items():
                items.append(_Item("table", t, ""))
                items.extend(_Item("column", t, c) for c in cols)
            return cls(items)
        raise TypeError(f"cannot read schema items from {type(schema).__name__}")


@dataclass(frozen=True)
class _Item:
    kind: str
    table: str
    column: str


# --- the machine ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Ctx:
    clause: str
    family: int
    scope: tuple = ()            # (_Entry, ...) outermost first
    depth: int = 0               # query nesting depth
    in_agg: bool = False
    in_pred: bool = False


@dataclass(frozen=True)
class _Entry:
    select: "_SelectFrame"
    visible: bool = True
    limit: "int | None" = None   # a LEFT JOIN's ON at position k sees sources 0..k


@dataclass
class _Src:
    table: "str | None"
    plan: object = None          # derived: the inner plan
    arity: int = 0
    has_star: bool = False


END_MARK = object()


def _visible(ctx: _Ctx):
    """Visible base-table occurrences {table: [(level, index), ...]} and derived sources [(level, index, src)], both
    innermost SELECT first, then FROM order (the order COL<k> and DCOL_<s>_<p> count in)."""
    tables: dict = {}
    derived = []
    for level, entry in enumerate(reversed(ctx.scope)):
        if not entry.visible:
            continue
        srcs = entry.select.sources
        n = len(srcs) if entry.limit is None else min(len(srcs), entry.limit + 1)
        for idx in range(n):
            s = srcs[idx]
            if s.table is not None:
                tables.setdefault(s.table, []).append((level, idx))
            else:
                derived.append((level, idx, s))
    return tables, derived


class _Frame:
    kind = "DONE"
    done = False
    value = None
    ctx: _Ctx = _Ctx("QUERY", PROJECTION)
    parent_op = None
    child_index = 0

    def options(self, m) -> dict:
        raise NotImplementedError

    def apply(self, action: Action, m) -> None:
        raise NotImplementedError

    def child_done(self, value, m) -> None:
        raise NotImplementedError


class _QuerySlot(_Frame):
    kind = "QUERY"

    def __init__(self, scope: tuple, depth: int, *, want=None, select_only=False, member=False, parent_op=None,
                 child_index=0):
        self.ctx = _Ctx("QUERY", PROJECTION, scope, depth)
        self.want, self.select_only, self.member = want, select_only, member
        self.parent_op, self.child_index = parent_op, child_index

    def options(self, m) -> dict:
        ops = {"QUERY_SELECT": None}
        if not self.select_only and self.ctx.depth < MAX_QUERY_DEPTH:
            for op in SET_OP:
                ops[op] = None
        return ops

    def apply(self, action, m):
        if action.op == "QUERY_SELECT":
            m.push(_SelectFrame(self.ctx.scope, self.ctx.depth, self.want, self.member))
        else:
            m.push(_CompoundFrame(SET_OP[action.op], self.ctx.scope, self.ctx.depth, self.want))

    def child_done(self, value, m):
        self.value, self.done = value, True


class _CompoundFrame(_Frame):
    kind = "QUERY"

    def __init__(self, op: str, scope: tuple, depth: int, want):
        self.op, self.scope, self.depth, self.want = op, scope, depth, want
        self.ctx = _Ctx("QUERY", PROJECTION, scope, depth)
        self.left = None
        self.started = False

    def start(self, m):
        m.push(_QuerySlot(self.scope, self.depth, want=self.want, member=True, parent_op=SET_OP_ACTION[self.op]))

    def child_done(self, value, m):
        if self.left is None:
            self.left = value
            arity = ir.query_arity(value, m.schema.index)
            m.push(_QuerySlot(self.scope, self.depth, want=arity, select_only=True, member=True,
                              parent_op=SET_OP_ACTION[self.op], child_index=1))
        else:
            self.value, self.done = ir.Compound(self.op, self.left, value), True


class _SelectFrame(_Frame):
    """One SELECT, built phase by phase."""

    def __init__(self, scope: tuple, depth: int, want, member: bool):
        self.parent_scope, self.depth, self.want, self.member = scope, depth, want, member
        self.sources: list = []
        self.joins: list = []        # join kind per FROM item
        self.ons: list = []
        self.where = None
        self.group: list = []
        self.distinct = False
        self.items: list = []
        self.arity = 0
        self.having = None
        self.order: list = []
        self.limit = None
        self.offset = None
        self.phase = "FROM_FIRST"
        self.on_queue: list = []
        self.pending_desc = None
        self.current_on = None

    # scope
    @property
    def own(self) -> tuple:
        return self.parent_scope + (_Entry(self),)

    @property
    def ctx(self) -> _Ctx:
        family = SCHEMA if self.phase in ("FROM_FIRST", "FROM_MORE") else PROJECTION
        return _Ctx("FROM" if family == SCHEMA else ("LIMIT" if self.phase in ("LIMIT", "OFFSET") else "SELECT"),
                     family, self.own, self.depth)

    @property
    def kind(self) -> str:
        return self.phase

    def is_aggregate(self) -> bool:
        return ir.is_aggregate_query(ir.Select(group_by=tuple(self.group), items=tuple(self.items)))

    def source_arity(self, k: int) -> int:
        return self.sources[k].arity

    def star_arity(self, op: str) -> int:
        if op == "STAR":
            return sum(s.arity for s in self.sources)
        return self.sources[int(op.split("_")[1])].arity

    # options in the structural phases (expression phases push slots)
    def options(self, m) -> dict:
        if self.phase == "FROM_FIRST":
            ops = {"FROM_TABLE": ("table", m.schema.tables), "FROM_NONE": None}
            if self.depth < MAX_QUERY_DEPTH:
                ops["FROM_DERIVED"] = None
            return ops
        if self.phase == "FROM_MORE":
            ops = {"END": None}
            if len(self.sources) < MAX_FROM_ITEMS:
                for k in JOIN_KINDS:
                    ops[f"JOIN_{k}_TABLE"] = ("table", m.schema.tables)
                    if self.depth < MAX_QUERY_DEPTH:
                        ops[f"JOIN_{k}_DERIVED"] = None
            return ops
        if self.phase == "DISTINCT":
            return {"SELECT_ALL": None, "SELECT_DISTINCT": None}
        if self.phase == "ORDER_DIR":
            ops = {"END": None}
            if len(self.order) < MAX_ORDER:
                ops.update({"ORDER_ASC": None, "ORDER_DESC": None})
            return ops
        if self.phase == "LIMIT":
            return {"NONE": None, **{f"LIMIT_{v}": None for v in LIMIT_VALUES}}
        if self.phase == "OFFSET":
            return {"NONE": None, **{f"OFFSET_{v}": None for v in OFFSET_VALUES}}
        raise IllegalAction(f"no direct action in phase {self.phase}")

    def apply(self, action, m):
        op = action.op
        if self.phase == "FROM_FIRST":
            if op == "FROM_NONE":
                return self.goto_where(m)
            self.joins.append("FIRST")
            if op == "FROM_TABLE":
                self.add_table(action.arg, m)
                self.phase = "FROM_MORE"
            else:
                m.push(_QuerySlot(self.parent_scope + (_Entry(self, visible=False),), self.depth + 1,
                                  parent_op=op))
            return None
        if self.phase == "FROM_MORE":
            if op == "END":
                self.on_queue = [k for k in range(1, len(self.sources)) if self.joins[k] in ("INNER", "LEFT")]
                return self.next_on(m)
            _, kind, what = op.split("_")
            self.joins.append(kind)
            if what == "TABLE":
                self.add_table(action.arg, m)
            else:
                m.push(_QuerySlot(self.parent_scope + (_Entry(self, visible=False),), self.depth + 1,
                                  parent_op=op))
            return None
        if self.phase == "DISTINCT":
            self.distinct = op == "SELECT_DISTINCT"
            self.phase = "ITEM"
            return self.push_item(m)
        if self.phase == "ORDER_DIR":
            if op == "END":
                self.phase = "LIMIT"
                return None
            self.pending_desc = op == "ORDER_DESC"
            self.phase = "ORDER_EXPR"
            m.push(_ExprSlot(_Ctx("ORDER", PROJECTION, self.own, self.depth), "ORDER_EXPR", parent_op=op))
            return None
        if self.phase == "LIMIT":
            if op == "NONE":
                return self.finish(m)
            self.limit = int(op.split("_")[1])
            self.phase = "OFFSET"
            return None
        if self.phase == "OFFSET":
            if op != "NONE":
                self.offset = int(op.split("_")[1])
            return self.finish(m)
        raise IllegalAction(f"{op} in phase {self.phase}")

    def add_table(self, item_index, m):
        table = m.schema.items[item_index].table
        self.sources.append(_Src(table=table, arity=len(m.schema.index.columns(table))))
        self.ons.append(None)

    def child_done(self, value, m):
        phase = self.phase
        if phase in ("FROM_FIRST", "FROM_MORE"):  # a derived table
            arity = ir.query_arity(value, m.schema.index) or 0
            first = value
            while isinstance(first, ir.Compound):
                first = first.left
            self.sources.append(_Src(table=None, plan=value, arity=arity,
                                     has_star=any(isinstance(i, ir.Star) for i in first.items)))
            self.ons.append(None)
            self.phase = "FROM_MORE"
        elif phase == "ON":
            self.ons[self.current_on] = None if value is None else value
            self.next_on(m)
        elif phase == "WHERE":
            self.where = value
            self.phase = "GROUP"
            self.push_group(m)
        elif phase == "GROUP":
            if value is END_MARK:
                self.phase = "DISTINCT"
            else:
                self.group.append(value)
                self.push_group(m)
        elif phase == "ITEM":
            if value is END_MARK:
                self.phase = "HAVING"
                m.push(_ExprSlot(_Ctx("HAVING", PREDICATE, self.own, self.depth), "HAVING", allow_none=True,
                                 allow_scalar=self.is_aggregate()))
            else:
                self.items.append(value)
                self.arity += self.item_arity(value)
                self.push_item(m)
        elif phase == "HAVING":
            self.having = value
            if self.member:
                self.finish(m)
            else:
                self.phase = "ORDER_DIR"
        elif phase == "ORDER_EXPR":
            self.order.append(ir.OrderItem(value, self.pending_desc))
            self.phase = "ORDER_DIR"
        else:
            raise IllegalAction(f"unexpected child in phase {phase}")

    def item_arity(self, item) -> int:
        if isinstance(item, ir.Star):
            return sum(s.arity for s in self.sources) if item.source is None else self.sources[item.source].arity
        return 1

    def next_on(self, m):
        if self.on_queue:
            k = self.on_queue.pop(0)
            self.current_on = k
            self.phase = "ON"
            left = self.joins[k] == "LEFT"
            scope = self.parent_scope + (_Entry(self, True, k if left else None),)
            m.push(_ExprSlot(_Ctx("ON", SCHEMA, scope, self.depth), "ON", allow_none=left))
            return None
        return self.goto_where(m)

    def goto_where(self, m):
        self.phase = "WHERE"
        m.push(_ExprSlot(_Ctx("WHERE", PREDICATE, self.own, self.depth), "WHERE", allow_none=True))

    def push_group(self, m):
        m.push(_ExprSlot(_Ctx("GROUP", AGGREGATE, self.own, self.depth), "GROUP", allow_end=True,
                         allow_scalar=len(self.group) < MAX_GROUP, child_index=len(self.group)))

    def push_item(self, m):
        n, want, a = len(self.items), self.want, self.arity
        allow_end = n >= 1 and (want is None or a == want)
        room = n < MAX_ITEMS
        allow_scalar = room and (want is None or a + 1 <= want)
        stars = {}
        if room and self.sources:
            for op in ("STAR",) + tuple(f"STAR_{k}" for k in range(min(4, len(self.sources)))):
                if want is None or a + self.star_arity(op) <= want:
                    stars[op] = None
        m.push(_ExprSlot(_Ctx("SELECT", PROJECTION, self.own, self.depth), "ITEM", allow_end=allow_end,
                         allow_scalar=allow_scalar, stars=stars, child_index=n))

    def finish(self, m):
        from_ = tuple(ir.FromItem(self.source_ir(k), self.joins[k], self.ons[k]) for k in range(len(self.sources)))
        self.value = ir.Select(from_=from_, where=self.where, group_by=tuple(self.group), having=self.having,
                               distinct=self.distinct, items=tuple(self.items), order_by=tuple(self.order),
                               limit=self.limit, offset=self.offset)
        self.done = True

    def source_ir(self, k: int):
        s = self.sources[k]
        return ir.TableSource(s.table) if s.table is not None else ir.DerivedSource(s.plan)


_NUMERIC_CONSTANTS = tuple(f"CONST_{i}" for i, (kind, _v) in enumerate(CONSTANTS) if kind in ("integer", "real"))
_NOT_FORBIDS = frozenset(op for pair in NEGATABLE.items() for op in pair)


class _ExprSlot(_Frame):
    """A scalar position. It accepts a leaf (column, literal, COUNT(*), *) or an operator that opens a _NodeFrame."""

    def __init__(self, ctx: _Ctx, kind: str, *, allow_none=False, allow_end=False, allow_scalar=True, stars=None,
                 forbid=frozenset(), parent_op=None, child_index=0):
        self.ctx, self.kind = ctx, kind
        self.allow_none, self.allow_end, self.allow_scalar = allow_none, allow_end, allow_scalar
        self.stars = stars or {}
        self.forbid = forbid
        self.parent_op, self.child_index = parent_op, child_index

    def agg_ok(self) -> bool:
        ctx = self.ctx
        if ctx.in_agg or ctx.clause not in ("SELECT", "HAVING", "ORDER"):
            return False
        if ctx.clause == "ORDER":
            return ctx.scope[-1].select.is_aggregate()
        return True

    def options(self, m) -> dict:
        ops: dict = {}
        if self.allow_none:
            ops["NONE"] = None
        if self.allow_end:
            ops["END"] = None
        ops.update(self.stars)
        if not self.allow_scalar:
            return ops
        tables, derived = _visible(self.ctx)
        for k in range(OCCURRENCES):
            cols = [c for t, occ in tables.items() if len(occ) > k for c in m.schema.columns_of[t]]
            if cols:
                ops[f"COL{k}"] = ("column", tuple(sorted(cols)))
        for s, (_level, _idx, src) in enumerate(derived[:DERIVED_SOURCES]):
            if not src.has_star:
                for p in range(min(src.arity, DERIVED_POSITIONS)):
                    ops[f"DCOL_{s}_{p}"] = None
        if m.spans is not None:
            text_starts = m.spans.starts("text")
            if text_starts:
                for op in TEXT_OPS:
                    ops[op] = ("span", text_starts)
            number_starts = m.spans.starts("number")
            if number_starts:
                ops["NUM"] = ("span", number_starts)
        ops["NULL"] = None
        for i in range(len(CONSTANTS)):
            ops[f"CONST_{i}"] = None
        if self.agg_ok():
            for op in AGG_OPS:
                ops[op] = None
        nest = self.ctx.depth < MAX_QUERY_DEPTH
        for op in PREDICATE_OPS:
            if op in ("IN_QUERY", "NOT_IN_QUERY", "EXISTS", "NOT_EXISTS") and not nest:
                continue
            ops[op] = None
        for op in tuple(ARITH_ACTIONS) + ("NEG",) + CAST_OPS + FN_OPS + CASE_OPS:
            ops[op] = None
        if nest:
            ops["SUBQ"] = None
        for op in self.forbid:
            ops.pop(op, None)
        return ops

    def apply(self, action, m):
        op = action.op
        if op == "NONE":
            self.value, self.done = None, True
        elif op == "END":
            self.value, self.done = END_MARK, True
        elif op == "STAR":
            self.value, self.done = ir.Star(None), True
        elif op.startswith("STAR_"):
            self.value, self.done = ir.Star(int(op.split("_")[1])), True
        elif op.startswith("COL"):
            k = int(op[3:])
            item = m.schema.items[action.arg]
            tables, _ = _visible(self.ctx)
            level, idx = tables[item.table][k]
            self.value, self.done = ir.Column(level, idx, item.column), True
        elif op.startswith("DCOL_"):
            _, s, p = op.split("_")
            _, derived = _visible(self.ctx)
            level, idx, _src = derived[int(s)]
            self.value, self.done = ir.DerivedColumn(level, idx, int(p)), True
        elif op in TEXT_OPS:
            _, form, wild = op.split("_")
            core = m.spans.text(*action.arg)
            if core is None:
                raise IllegalAction("the span does not decode to text")
            self.value, self.done = ir.text(WILD_FN[wild](FORM_FN[form](core))), True
        elif op == "NUM":
            lit = m.spans.number(*action.arg)
            if lit is None:
                raise IllegalAction("the span does not decode to a number")
            self.value, self.done = lit, True
        elif op == "NULL":
            self.value, self.done = ir.NULL, True
        elif op.startswith("CONST_"):
            kind, value = CONSTANTS[int(op.split("_")[1])]
            self.value, self.done = ir.Literal(kind, value), True
        elif op == "COUNT_STAR":
            self.value, self.done = ir.CountStar(), True
        else:
            m.push(_NodeFrame(op, self.ctx))

    def child_done(self, value, m):
        self.value, self.done = value, True


def _node_children(op: str):
    """(number of required scalar children, list-continuation rule) for an operator; queries are handled apart."""
    if op in COMPARE_ACTIONS or op in ARITH_ACTIONS or op in ("LIKE", "NOT_LIKE"):
        return 2
    if op in ("BETWEEN", "NOT_BETWEEN"):
        return 3
    if op in ("NEG", "NOT", "IS_NULL", "IS_NOT_NULL") or op in CAST_OPS or op.startswith("AGG_"):
        return 1
    if op.startswith("FN_"):
        return int(op.rsplit("_", 1)[1])
    raise KeyError(op)


class _NodeFrame(_Frame):
    """An operator waiting for its operands."""
    kind = "ARG"

    def __init__(self, op: str, ctx: _Ctx):
        self.op = op
        self.parent_ctx = ctx
        if op in _SET["pred"]:
            self.ctx = _Ctx(ctx.clause, SCHEMA if ctx.clause == "ON" else PREDICATE, ctx.scope, ctx.depth,
                            ctx.in_agg, True)
        elif op in _SET["agg"]:
            self.ctx = _Ctx(ctx.clause, AGGREGATE, ctx.scope, ctx.depth, True, ctx.in_pred)
        else:
            self.ctx = ctx
        self.children: list = []
        self.started = False

    def query_ctx_scope(self):
        return self.ctx.scope

    def start(self, m):
        self.next_child(m)

    def push_scalar(self, m, *, kind="ARG", allow_end=False, allow_scalar=True, forbid=frozenset()):
        m.push(_ExprSlot(self.ctx, kind, allow_end=allow_end, allow_scalar=allow_scalar, forbid=forbid,
                         parent_op=self.op, child_index=len(self.children)))

    def push_query(self, m, want):
        m.push(_QuerySlot(self.ctx.scope, self.ctx.depth + 1, want=want, parent_op=self.op,
                          child_index=len(self.children)))

    def next_child(self, m):
        op, n = self.op, len(self.children)
        if op in ("AND", "OR"):
            if n < 2:
                return self.push_scalar(m, forbid=frozenset({op}))
            return self.push_scalar(m, kind="MORE", allow_end=True, allow_scalar=n < MAX_BOOL_ARGS,
                                    forbid=frozenset({op}))
        if op in ("IN_LIST", "NOT_IN_LIST"):
            if n < 2:
                return self.push_scalar(m)
            return self.push_scalar(m, kind="MORE", allow_end=True, allow_scalar=n < MAX_IN_ITEMS + 1)
        if op in ("IN_QUERY", "NOT_IN_QUERY"):
            return self.push_scalar(m) if n == 0 else self.push_query(m, 1) if n == 1 else self.finish()
        if op in ("EXISTS", "NOT_EXISTS"):
            return self.push_query(m, None) if n == 0 else self.finish()
        if op == "SUBQ":
            return self.push_query(m, 1) if n == 0 else self.finish()
        if op.startswith("CASE_"):
            parts = op.split("_")
            simple = parts[1] == "SIMPLE"
            k = int(parts[2] if simple else parts[1])
            total = int(simple) + 2 * k + int(op.endswith("_ELSE"))
            if n < total:
                return self.push_scalar(m)
            return self.finish()
        if op == "NEG":
            return self.push_scalar(m, forbid=frozenset(("NUM",) + _NUMERIC_CONSTANTS)) if n == 0 else self.finish()
        if op == "NOT":
            return self.push_scalar(m, forbid=_NOT_FORBIDS) if n == 0 else self.finish()
        need = _node_children(op)
        if n < need:
            return self.push_scalar(m)
        return self.finish()

    def child_done(self, value, m):
        if value is END_MARK:
            return self.finish()
        self.children.append(value)
        return self.next_child(m)

    def finish(self):
        op, c = self.op, self.children
        if op in COMPARE_ACTIONS:
            node = ir.Compare(COMPARE_ACTIONS[op], c[0], c[1])
        elif op in ARITH_ACTIONS:
            node = ir.Binary(ARITH_ACTIONS[op], c[0], c[1])
        elif op == "NEG":
            node = ir.Unary("-", c[0])
        elif op in ("AND", "OR"):
            node = (ir.And if op == "AND" else ir.Or)(tuple(c))
        elif op == "NOT":
            node = ir.Not(c[0])
        elif op in ("LIKE", "NOT_LIKE"):
            node = ir.Like(c[0], c[1], op == "NOT_LIKE")
        elif op in ("BETWEEN", "NOT_BETWEEN"):
            node = ir.Between(c[0], c[1], c[2], op == "NOT_BETWEEN")
        elif op in ("IN_LIST", "NOT_IN_LIST"):
            node = ir.InList(c[0], tuple(c[1:]), op == "NOT_IN_LIST")
        elif op in ("IN_QUERY", "NOT_IN_QUERY"):
            node = ir.InQuery(c[0], c[1], op == "NOT_IN_QUERY")
        elif op in ("IS_NULL", "IS_NOT_NULL"):
            node = ir.IsNull(c[0], op == "IS_NOT_NULL")
        elif op in ("EXISTS", "NOT_EXISTS"):
            node = ir.Exists(c[0], op == "NOT_EXISTS")
        elif op == "SUBQ":
            node = ir.ScalarQuery(c[0])
        elif op in CAST_OPS:
            node = ir.Cast(c[0], op[5:])
        elif op.startswith("FN_"):
            name = op[3:].rsplit("_", 1)[0]
            node = ir.Func(name, tuple(c))
        elif op.startswith("AGG_"):
            name = op[4:]
            node = ir.Agg(name.replace("_DISTINCT", ""), c[0], name.endswith("_DISTINCT"))
        elif op.startswith("CASE_"):
            simple = op.startswith("CASE_SIMPLE")
            has_else = op.endswith("_ELSE")
            operand = c[0] if simple else None
            body = c[1:] if simple else c
            default = body[-1] if has_else else None
            pairs = body[:-1] if has_else else body
            node = ir.Case(operand, tuple(ir.When(pairs[i], pairs[i + 1]) for i in range(0, len(pairs), 2)), default)
        else:
            raise IllegalAction(f"cannot build {op}")
        self.value, self.done = node, True


class Machine:
    """The grammar state: the partial plan and the stack of unfinished subproblems.

    `options()` maps every legal operator at this step to its pointer constraint (None, or (kind, legal targets));
    `legal_families()` and `legal_ops(family)` derive the family and operator masks from it; `apply(action)` checks
    legality and advances. When the stack is empty the only legal decision is STOP and `plan` holds the result."""

    def __init__(self, schema, spans: "SpanIndex | None" = None, *, max_steps: int = 512):
        self.schema = SchemaItems.coerce(schema)
        self.spans = spans
        self.max_steps = max_steps
        self.stack: list = []
        self.steps = 0
        self.plan = None
        self._options = None
        self.push(_QuerySlot((), 0))

    # stack discipline
    def push(self, frame: _Frame):
        self.stack.append(frame)
        if isinstance(frame, (_CompoundFrame, _NodeFrame)):
            frame.start(self)

    def _collapse(self):
        while self.stack and self.stack[-1].done:
            finished = self.stack.pop()
            if self.stack:
                self.stack[-1].child_done(finished.value, self)
            else:
                self.plan = finished.value

    @property
    def finished(self) -> bool:
        return not self.stack

    @property
    def top(self) -> "_Frame | None":
        return self.stack[-1] if self.stack else None

    # masks
    def options(self) -> dict:
        if self._options is None:
            self._options = {} if self.finished else self.top.options(self)
        return self._options

    def family(self, op: str) -> int:
        top = self.top
        ctx = top.ctx
        return family_of(op, ctx.clause, ctx.family)

    def legal_families(self) -> list:
        mask = [False] * N_CLASSES
        if self.finished:
            mask[STOP] = True
            return mask
        for op in self.options():
            mask[self.family(op)] = True
        return mask

    def legal_ops(self, family: int) -> list:
        return [op for op in self.options() if self.family(op) == family]

    def legal_union_ids(self) -> list:
        return sorted(UNION_INDEX[(self.family(op), op)] for op in self.options())

    def pointer_targets(self, op: str):
        """(kind, legal targets) for a pointer operator: item indices, or span start positions."""
        return self.options()[op]

    def span_ends(self, op: str, start: int) -> tuple:
        return self.spans.ends("number" if op == "NUM" else "text", start)

    def is_legal(self, action: Action) -> bool:
        opts = self.options()
        if action.op not in opts:
            return False
        spec = opts[action.op]
        if spec is None:
            return action.arg is None
        kind, targets = spec
        if kind == "span":
            if not isinstance(action.arg, tuple) or len(action.arg) != 2:
                return False
            start, end = action.arg
            return start in targets and end in self.span_ends(action.op, start)
        return action.arg in targets

    def apply(self, action: Action) -> None:
        if self.finished:
            raise IllegalAction("the plan is complete; only STOP is legal")
        if not self.is_legal(action):
            raise IllegalAction(f"{action.op}({action.arg}) is not legal in {self.top.kind}")
        self.steps += 1
        self._options = None
        self.top.apply(action, self)
        self._collapse()

    # state features for the controller
    def features(self) -> dict:
        """slot kind id, parent operator id and a fixed-length numeric vector describing the partial plan."""
        vec = [0.0] * N_FEATURES
        if self.finished:
            return {"slot": SLOT_ID["DONE"], "parent_op": 0, "vector": vec}
        top = self.top
        ctx = top.ctx
        vec[CLAUSES.index(ctx.clause)] = 1.0
        vec[9 + ctx.family] = 1.0
        vec[13] = float(ctx.in_agg)
        vec[14] = float(ctx.in_pred)
        vec[15] = getattr(top, "child_index", 0) / 8.0
        vec[16] = len(self.stack) / 16.0
        vec[17] = ctx.depth / 4.0
        select = ctx.scope[-1].select if ctx.scope else (top if isinstance(top, _SelectFrame) else None)
        if isinstance(top, _SelectFrame):
            select = top
        if select is not None:
            vec[18] = len(select.sources) / 8.0
            vec[19] = len(select.items) / 8.0
            vec[20] = len(select.group) / 4.0
            vec[21] = len(select.order) / 4.0
            vec[22] = float(select.where is not None)
            vec[23] = float(bool(select.items) and select.is_aggregate())
            if select.want is not None:
                vec[26] = 1.0
                vec[27] = select.want / 8.0
                vec[28] = (select.want - select.arity) / 8.0
        tables, derived = _visible(ctx)
        vec[24] = sum(len(v) for v in tables.values()) / 8.0
        vec[25] = len(derived) / 2.0
        for frame in self.stack:
            if isinstance(frame, (_QuerySlot, _CompoundFrame)):
                vec[29] += 1 / 8.0
            elif isinstance(frame, _SelectFrame):
                vec[30] += 1 / 8.0
            elif isinstance(frame, _NodeFrame):
                vec[31] += 1 / 8.0
                op = frame.op
                cat = 33 if op in _SET["pred"] else 34 if op in _SET["agg"] else 35 if (
                    op in ARITH_ACTIONS or op == "NEG" or op in CAST_OPS) else 36 if op.startswith("FN_") else \
                    37 if op.startswith("CASE_") else 38
                vec[cat] += 1 / 4.0
            else:
                vec[32] += 1 / 8.0
        vec[39] = self.steps / 128.0
        parent = getattr(top, "parent_op", None)
        return {"slot": SLOT_ID.get(top.kind, SLOT_ID["ARG"]), "parent_op": OP_ID.get(parent, 0) if parent else 0,
                "vector": vec}


# --- gold encoding (plan -> actions) ---------------------------------------------------------------------------------
class _Encoder:
    def __init__(self, machine: Machine, record=None):
        self.m = machine
        self.actions: list = []
        self.record = record

    def emit(self, op: str, arg=None):
        action = Action(op, arg)
        m = self.m
        if not m.is_legal(action):
            if op not in m.options():
                raise NotEncodable("grammar rejects gold action", f"{op} in {m.top.kind}")
            raise NotEncodable("grammar rejects gold pointer", f"{op}({arg})")
        if self.record is not None:
            self.record(m, action)
        m.apply(action)
        self.actions.append(action)

    def query(self, q, member=False):
        if isinstance(q, ir.Compound):
            self.emit(SET_OP_ACTION[q.op])
            self.query(q.left, member=True)
            self.emit("QUERY_SELECT")
            self.select(q.right, member=True)
            return
        self.emit("QUERY_SELECT")
        self.select(q, member=member)

    def select(self, s: ir.Select, member: bool):
        if len(s.from_) > MAX_FROM_ITEMS:
            raise NotEncodable("too many FROM items")
        if not s.from_:
            self.emit("FROM_NONE")
        else:
            for k, item in enumerate(s.from_):
                if isinstance(item.source, ir.TableSource):
                    op = "FROM_TABLE" if k == 0 else f"JOIN_{item.join}_TABLE"
                    self.emit(op, self.m.schema.table_item[item.source.table])
                else:
                    self.emit("FROM_DERIVED" if k == 0 else f"JOIN_{item.join}_DERIVED")
                    self.query(item.source.query)
            self.emit("END")
            for k, item in enumerate(s.from_):
                if item.join == "INNER":
                    self.expr(item.on)
                elif item.join == "LEFT":
                    if item.on is None:
                        self.emit("NONE")
                    else:
                        self.expr(item.on)
        if s.where is None:
            self.emit("NONE")
        else:
            self.expr(s.where)
        if len(s.group_by) > MAX_GROUP:
            raise NotEncodable("too many GROUP BY items")
        for g in s.group_by:
            self.expr(g)
        self.emit("END")
        self.emit("SELECT_DISTINCT" if s.distinct else "SELECT_ALL")
        if len(s.items) > MAX_ITEMS:
            raise NotEncodable("too many result columns")
        for item in s.items:
            if isinstance(item, ir.Star):
                self.emit("STAR" if item.source is None else f"STAR_{item.source}")
            else:
                self.expr(item)
        self.emit("END")
        if s.having is None:
            self.emit("NONE")
        else:
            self.expr(s.having)
        if member:
            return
        if len(s.order_by) > MAX_ORDER:
            raise NotEncodable("too many ORDER BY items")
        for o in s.order_by:
            self.emit("ORDER_DESC" if o.descending else "ORDER_ASC")
            self.expr(o.expr)
        self.emit("END")
        if s.limit is None:
            self.emit("NONE")
        else:
            if s.limit not in LIMIT_VALUES:
                raise NotEncodable("LIMIT outside the vocabulary", str(s.limit))
            self.emit(f"LIMIT_{s.limit}")
            if s.offset is None:
                self.emit("NONE")
            else:
                if s.offset not in OFFSET_VALUES:
                    raise NotEncodable("OFFSET outside the vocabulary", str(s.offset))
                self.emit(f"OFFSET_{s.offset}")

    def expr(self, e):
        m = self.m
        if isinstance(e, ir.Column):
            tables, _ = _visible(m.top.ctx)
            table = self.source_table(e)
            occ = tables.get(table, [])
            if (e.level, e.source) not in occ:
                raise NotEncodable("column not visible to the grammar", f"{table}.{e.name}")
            k = occ.index((e.level, e.source))
            if k >= OCCURRENCES:
                raise NotEncodable("table occurrence outside the vocabulary", str(k))
            self.emit(f"COL{k}", m.schema.column_item[(table, e.name)])
        elif isinstance(e, ir.DerivedColumn):
            _, derived = _visible(m.top.ctx)
            keys = [(lv, ix) for lv, ix, _ in derived]
            if (e.level, e.source) not in keys:
                raise NotEncodable("derived column not visible to the grammar")
            s = keys.index((e.level, e.source))
            if s >= DERIVED_SOURCES or e.position >= DERIVED_POSITIONS:
                raise NotEncodable("derived column outside the vocabulary", f"{s}/{e.position}")
            self.emit(f"DCOL_{s}_{e.position}")
        elif isinstance(e, ir.Literal):
            op, arg = self.literal(e)
            self.emit(op, arg)
        elif isinstance(e, ir.Unary):
            self.emit("NEG")
            self.expr(e.arg)
        elif isinstance(e, ir.Binary):
            self.emit(ARITH_OF[e.op])
            self.expr(e.left)
            self.expr(e.right)
        elif isinstance(e, ir.Compare):
            self.emit(COMPARE_OF[e.op])
            self.expr(e.left)
            self.expr(e.right)
        elif isinstance(e, (ir.And, ir.Or)):
            if len(e.args) > MAX_BOOL_ARGS:
                raise NotEncodable("too many AND/OR operands")
            self.emit("AND" if isinstance(e, ir.And) else "OR")
            for a in e.args:
                self.expr(a)
            self.emit("END")
        elif isinstance(e, ir.Not):
            self.emit("NOT")
            self.expr(e.arg)
        elif isinstance(e, ir.Like):
            self.emit("NOT_LIKE" if e.negated else "LIKE")
            self.expr(e.arg)
            self.expr(e.pattern)
        elif isinstance(e, ir.Between):
            self.emit("NOT_BETWEEN" if e.negated else "BETWEEN")
            self.expr(e.arg)
            self.expr(e.low)
            self.expr(e.high)
        elif isinstance(e, ir.InList):
            if len(e.items) > MAX_IN_ITEMS:
                raise NotEncodable("too many IN items")
            self.emit("NOT_IN_LIST" if e.negated else "IN_LIST")
            self.expr(e.arg)
            for item in e.items:
                self.expr(item)
            self.emit("END")
        elif isinstance(e, ir.InQuery):
            self.emit("NOT_IN_QUERY" if e.negated else "IN_QUERY")
            self.expr(e.arg)
            self.query(e.query)
        elif isinstance(e, ir.IsNull):
            self.emit("IS_NOT_NULL" if e.negated else "IS_NULL")
            self.expr(e.arg)
        elif isinstance(e, ir.Exists):
            self.emit("NOT_EXISTS" if e.negated else "EXISTS")
            self.query(e.query)
        elif isinstance(e, ir.ScalarQuery):
            self.emit("SUBQ")
            self.query(e.query)
        elif isinstance(e, ir.Case):
            k = len(e.whens)
            simple = e.operand is not None
            if k > (MAX_SIMPLE_CASE_WHENS if simple else MAX_CASE_WHENS):
                raise NotEncodable("too many CASE branches")
            self.emit(f"CASE_{'SIMPLE_' if simple else ''}{k}{'_ELSE' if e.default is not None else ''}")
            if simple:
                self.expr(e.operand)
            for w in e.whens:
                self.expr(w.condition)
                self.expr(w.result)
            if e.default is not None:
                self.expr(e.default)
        elif isinstance(e, ir.Cast):
            self.emit(f"CAST_{e.affinity}")
            self.expr(e.arg)
        elif isinstance(e, ir.Func):
            if (e.name, len(e.args)) not in FUNCTION_ARITIES:
                raise NotEncodable("function outside the vocabulary", f"{e.name}/{len(e.args)}")
            self.emit(f"FN_{e.name}_{len(e.args)}")
            for a in e.args:
                self.expr(a)
        elif isinstance(e, ir.Agg):
            self.emit(f"AGG_{e.func}{'_DISTINCT' if e.distinct else ''}")
            self.expr(e.arg)
        elif isinstance(e, ir.CountStar):
            self.emit("COUNT_STAR")
        else:
            raise NotEncodable("node outside the grammar", type(e).__name__)

    def source_table(self, e: ir.Column) -> str:
        scope = self.m.top.ctx.scope
        entry = scope[-1 - e.level]
        return entry.select.sources[e.source].table

    def literal(self, e: ir.Literal):
        if e.kind == "null":
            return "NULL", None
        key = (e.kind, e.value)
        if key in CONSTANT_INDEX:
            return f"CONST_{CONSTANT_INDEX[key]}", None
        spans = self.m.spans
        if spans is None:
            raise NotEncodable("literal not groundable", "no question/evidence tokens")
        if e.kind in ("integer", "real"):
            span = spans.ground_number(e)
            if span is None:
                raise NotEncodable("literal not groundable", f"number {e.value}")
            return "NUM", span
        value = e.value
        for wild in WILDCARDS:
            core = _unwild(value, wild)
            if core is None:
                continue
            for form in CASE_FORMS:
                span = spans.ground_text(core, form)
                if span is not None:
                    return f"TEXT_{form}_{wild}", span
        raise NotEncodable("literal not groundable", f"text {value[:60]!r}")


def _unwild(value: str, wild: str):
    if wild == "NONE":
        return value if value else None
    if wild == "ANY":
        return value[1:-1] if len(value) >= 3 and value[0] == "%" and value[-1] == "%" else None
    if wild == "PRE":
        return value[:-1] if len(value) >= 2 and value[-1] == "%" and value[0] != "%" else None
    if wild == "SUF":
        return value[1:] if len(value) >= 2 and value[0] == "%" and value[-1] != "%" else None
    raise KeyError(wild)


def encode(plan, schema, spans: "SpanIndex | None" = None) -> list:
    """The gold action sequence of a plan (without the final STOP), or NotEncodable."""
    m = Machine(schema, spans)
    enc = _Encoder(m)
    enc.query(plan)
    if not m.finished:
        raise NotEncodable("plan did not complete the grammar")
    return enc.actions


def decode(actions, schema, spans: "SpanIndex | None" = None):
    """Replay actions through the machine and return the plan (raises IllegalAction on an illegal step)."""
    m = Machine(schema, spans)
    for a in actions:
        m.apply(a)
    if not m.finished:
        raise IllegalAction("the actions leave unfinished subproblems")
    return m.plan


@dataclass
class Step:
    """One supervised step: the masks and features before the action, the gold action and its targets."""
    family: int                 # 0..3, or STOP
    op: "str | None"
    union_id: int               # index in UNION (-1 for STOP)
    legal_families: list
    legal_union_ids: list
    pointer: "str | None"       # table | column | span
    target: object              # item index, or (start, end)
    pointer_legal: tuple        # legal item indices, or legal span starts
    end_legal: tuple            # span only: legal ends given the gold start
    features: dict = field(default_factory=dict)
    action: "Action | None" = None


def trace(plan, schema, spans: "SpanIndex | None" = None) -> list:
    """Teacher-forcing targets: every step of the gold traversal with its masks, then STOP."""
    steps: list = []

    def record(m: Machine, action: Action):
        family = m.family(action.op)
        spec = m.options()[action.op]
        pointer = POINTER_KIND[action.op]
        legal, ends = (), ()
        if spec is not None:
            legal = tuple(spec[1])
            if pointer == "span":
                ends = m.span_ends(action.op, action.arg[0])
        steps.append(Step(family=family, op=action.op, union_id=UNION_INDEX[(family, action.op)],
                          legal_families=m.legal_families(), legal_union_ids=m.legal_union_ids(), pointer=pointer,
                          target=action.arg, pointer_legal=legal, end_legal=ends, features=m.features(),
                          action=action))

    m = Machine(schema, spans)
    _Encoder(m, record).query(plan)
    if not m.finished:
        raise NotEncodable("plan did not complete the grammar")
    steps.append(Step(family=STOP, op=None, union_id=-1, legal_families=m.legal_families(), legal_union_ids=[],
                      pointer=None, target=None, pointer_legal=(), end_legal=(), features=m.features()))
    return steps


def vocabulary_summary() -> dict:
    return {"grammar": GRAMMAR_VERSION, "families": list(FAMILIES),
            "family_vocabulary_sizes": {FAMILIES[f]: len(FAMILY_OPS[f]) for f in range(4)},
            "union_size": len(UNION), "constants": [list(c) for c in CONSTANTS],
            "functions": [f"{n}/{a}" for n, a in FUNCTION_ARITIES], "limit_values": [LIMIT_VALUES[0], LIMIT_VALUES[-1]],
            "offset_values": [OFFSET_VALUES[0], OFFSET_VALUES[-1]], "max_span_tokens": MAX_SPAN_TOKENS}
