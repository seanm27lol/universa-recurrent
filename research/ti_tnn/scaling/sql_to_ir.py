"""Gold SQL to tirel-1: parse with sqlglot (read="sqlite"), resolve every name against the database schema, and return
a plan or an explicit `Unsupported(reason)`.

**Example.** On `movie_platform`,

    SELECT T1.list_movie_number, T2.user_has_payment_method FROM lists AS T1
    INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id ORDER BY T1.list_movie_number DESC LIMIT 1

becomes a `Select` with two FROM items (`lists`, then `lists_users` joined INNER on
`Compare("=", Column(0, 0, "list_id"), Column(0, 1, "list_id"))`), two result columns, one descending `OrderItem`
and `limit=1`. The aliases T1 and T2 disappear: a column is (level, source index, canonical name).

**What conversion decides, and why it is safe.** Each rewrite below preserves what SQLite computes; the round trip
by outcome (`coverage`) checks it on every gold query.

| Source form | tirel-1 form | Why it is the same query in SQLite |
|---|---|---|
| `IIF(c, a, b)` | `Case(None, (When(c, a),), b)` | SQLite defines IIF as that CASE |
| `IFNULL(a, b)` | `Func("COALESCE", (a, b))` | identical for two arguments (sqlglot cannot tell them apart) |
| `a JOIN b` (no ON), `a, b`, `CROSS JOIN` | CROSS item | all are Cartesian products; CROSS JOIN only pins SQLite's join order |
| `CROSS JOIN b ON p` | INNER item with ON p | SQLite runs it as an inner join |
| `JOIN b USING (c)` | INNER with ON equality | same rows, except for `*` (then Unsupported) |
| `NOT (x IN ...)`, `NOT x LIKE p`, ... | the negated predicate | SQLite defines `x NOT IN` as `NOT (x IN)`, and so on |
| nested `AND` / `OR`, parentheses | flat `And(args)` / `Or(args)` | associativity |
| a result alias used in WHERE, GROUP BY, HAVING or ORDER BY | the aliased expression, inlined | SQLite resolves the alias to that expression |
| `ORDER BY 2`, `GROUP BY 1` | the second (first) result expression | SQLite's positional reference |
| `-5`, `TRUE`, `FALSE` | integer literals -5, 1, 0 | TRUE and FALSE are aliases for 1 and 0 |
| `CAST(x AS VARCHAR(10))`, `CAST(x AS STRING)` | `Cast(x, "TEXT")`, `Cast(x, "NUMERIC")` | CAST depends only on the type name's affinity, read from the original text |
| `STRFTIME(f)` | `STRFTIME(f, CURRENT_TIMESTAMP)` | both mean "now" (sqlglot cannot tell them apart) |
| a double-quoted name that is no column | a text literal | SQLite's double-quoted-string fallback |

Name resolution follows SQLite: a qualified name searches the innermost SELECT outward for that alias; an unqualified
name searches each SELECT's sources (innermost first), and at the SELECT itself also the result aliases (in ORDER BY,
aliases are tried first). Ambiguity is Unsupported, never guessed.

**Limits.** Everything outside tirel-1 raises `Unsupported` with a stable reason (window functions, GROUP_CONCAT,
RIGHT/FULL/NATURAL joins, IS with a non-NULL operand, ESCAPE, COLLATE, row values, CTEs, ORDER BY on a compound, ...).
A query can convert and still differ in outcome if a rewrite above were wrong; that is what the round trip measures.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.dialects.sqlite import SQLite
from sqlglot.tokens import TokenType

from . import ir

CONVERTER_VERSION = "sql_to_ir-v1"


class Unsupported(Exception):
    """A gold query outside tirel-1 (or one this converter cannot resolve). `reason` is a stable category."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# --- CAST types: read from the original text, because sqlglot maps STRING and TEXT to the same type ----------------
_AFFINITY_KEYWORD = {"INTEGER": "INTEGER", "REAL": "REAL", "NUMERIC": "NUMERIC", "TEXT": "TEXT", "BLOB": "BLOB"}
_SQLGLOT_TYPE_TO_AFFINITY = {"INT": "INTEGER", "FLOAT": "REAL", "DECIMAL": "NUMERIC", "TEXT": "TEXT",
                             "VARBINARY": "BLOB", "BLOB": "BLOB"}


def canonical_cast_types(sql: str) -> str:
    """Rewrite every `CAST(x AS <type name>)` to `CAST(x AS <affinity>)`, with the affinity SQLite derives from the
    original type name. CAST's result depends only on that affinity, so the query computes the same values."""
    try:
        tokens = SQLite().tokenize(sql)
    except Exception as error:  # sqlglot's TokenError
        raise Unsupported("parse error", str(error).splitlines()[0][:200]) from error
    edits = []
    for i, tok in enumerate(tokens):
        if tok.token_type != TokenType.VAR or tok.text.upper() != "CAST" or i + 1 >= len(tokens) \
                or tokens[i + 1].token_type != TokenType.L_PAREN:
            continue
        depth, j, as_at = 0, i + 1, None
        while j < len(tokens):
            t = tokens[j]
            if t.token_type == TokenType.L_PAREN:
                depth += 1
            elif t.token_type == TokenType.R_PAREN:
                depth -= 1
                if depth == 0:
                    break
            elif t.token_type == TokenType.ALIAS and depth == 1:
                as_at = j
            j += 1
        if as_at is None or j >= len(tokens) or as_at + 1 >= j:
            continue
        start, end = tokens[as_at + 1].start, tokens[j - 1].end + 1
        edits.append((start, end, ir.affinity_of(sql[start:end])))
    for start, end, affinity in sorted(edits, reverse=True):
        sql = sql[:start] + affinity + sql[end:]
    return sql


# --- resolution state ----------------------------------------------------------------------------------------------
@dataclass
class _Source:
    alias: "str | None"          # lower-case binding name (table name when unaliased); None for an unaliased subquery
    table: "str | None" = None   # canonical base table
    names: tuple = ()            # derived: lower-case output names by position (None: unnamed or unknown)
    stars: frozenset = frozenset()  # derived: output positions produced by '*'


@dataclass
class _Level:
    sources: list = field(default_factory=list)
    visible: bool = True
    left_limit: "int | None" = None   # ON of a LEFT JOIN at position k: only sources 0..k are visible
    aliases: dict = field(default_factory=dict)   # lower-case result alias -> sqlglot expression
    items: list = field(default_factory=list)     # sqlglot result expressions (aliases unwrapped)
    expanding: set = field(default_factory=set)   # aliases being inlined (cycle guard)


_SELECT_ARGS = {"expressions", "from", "from_", "joins", "where", "group", "having", "order", "limit", "offset",
                "distinct"}
_COMPARE = {exp.EQ: "=", exp.NEQ: "<>", exp.LT: "<", exp.LTE: "<=", exp.GT: ">", exp.GTE: ">="}
_ARITH = {exp.Add: "+", exp.Sub: "-", exp.Mul: "*", exp.Div: "/", exp.Mod: "%", exp.DPipe: "||"}
_AGG = {exp.Count: "COUNT", exp.Sum: "SUM", exp.Avg: "AVG", exp.Min: "MIN", exp.Max: "MAX"}
_ANONYMOUS_OK = {"JULIANDAY", "DATETIME", "TIME", "DATE", "STRFTIME", "SUBSTR", "INSTR", "LENGTH", "REPLACE", "TRIM",
                 "LTRIM", "RTRIM", "LOWER", "UPPER", "ROUND", "ABS", "COALESCE", "IFNULL", "NULLIF"}


def _empty(value) -> bool:
    return value is None or value is False or (isinstance(value, (list, tuple)) and not value)


def _unparen(node):
    while isinstance(node, exp.Paren):
        node = node.this
    return node


class _Converter:
    def __init__(self, schema: ir.SchemaIndex, sql: str):
        self.schema = schema
        self.sql = sql

    # row sets
    def query(self, node, stack: list) -> "ir.Query":
        node = _unparen(node)
        while isinstance(node, exp.Subquery) and _empty(node.args.get("alias")):
            node = _unparen(node.this)
        if isinstance(node, exp.Select):
            return self.select(node, stack)
        if isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
            for key in ("order", "limit", "offset"):
                if not _empty(node.args.get(key)):
                    raise Unsupported("ORDER BY or LIMIT on a compound")
            for key in ("by_name", "side", "kind", "on"):
                if not _empty(node.args.get(key)):
                    raise Unsupported("compound modifier", key)
            distinct = bool(node.args.get("distinct"))
            if isinstance(node, exp.Union):
                op = "UNION" if distinct else "UNION ALL"
            elif not distinct:
                raise Unsupported(f"{type(node).__name__.upper()} ALL")
            else:
                op = type(node).__name__.upper()
            left = self.query(node.this, stack)
            right_node = _unparen(node.expression)
            if not isinstance(right_node, exp.Select):
                raise Unsupported("compound member is not a simple SELECT")
            return ir.Compound(op, left, self.select(right_node, stack))
        if isinstance(node, exp.With) or (isinstance(node, exp.Expression) and not _empty(node.args.get("with_"))):
            raise Unsupported("WITH (CTE)")
        raise Unsupported("statement kind", type(node).__name__)

    def select(self, node: exp.Select, stack: list) -> ir.Select:
        if not _empty(node.args.get("with_")) or not _empty(node.args.get("with")):
            raise Unsupported("WITH (CTE)")
        for key, value in node.args.items():
            if key not in _SELECT_ARGS and not _empty(value):
                raise Unsupported("SELECT clause", key)
        distinct = node.args.get("distinct")
        if distinct is not None and not _empty(distinct.args.get("on")):
            raise Unsupported("DISTINCT ON")
        level = _Level()
        here = stack + [level]
        # FROM: the sources first (a derived table cannot see its siblings), then ON / USING
        entries = []
        from_node = node.args.get("from_") or node.args.get("from")
        if from_node is not None:
            if not _empty(from_node.args.get("expressions")):
                raise Unsupported("FROM with several expressions")
            entries.append(("FIRST", from_node.this, None))
        for j in node.args.get("joins") or []:
            entries.append((self.join_kind(j), j.this, j))
        sources = []
        for k, (kind, src, _join) in enumerate(entries):
            source, binding = self.source(src, stack)
            sources.append(source)
            level.sources.append(binding)
        level.items = []
        for item in node.expressions:
            if isinstance(item, exp.Alias):
                level.aliases.setdefault(item.alias.lower(), item.this)
                level.items.append(item.this)
            else:
                level.items.append(item)
        from_items = []
        star_select = any(isinstance(i, exp.Star) or (isinstance(i, exp.Column) and isinstance(i.this, exp.Star))
                          for i in node.expressions)
        for k, (kind, _src, join) in enumerate(entries):
            on = None
            if join is not None:
                on_node = join.args.get("on")
                using = join.args.get("using")
                if isinstance(on_node, exp.Boolean) and on_node.this is True and _empty(using):
                    on_node = None
                if not _empty(using):
                    if on_node is not None:
                        raise Unsupported("join with both ON and USING")
                    if star_select:
                        raise Unsupported("USING with *")
                    on = self.using(using, k, level)
                    kind = "INNER" if kind == "CROSS" else kind
                elif on_node is not None:
                    limit = k if kind == "LEFT" else None
                    on = self.expr(on_node, stack + [_Level(level.sources, True, limit, level.aliases, level.items)],
                                   "ON")
                    kind = "INNER" if kind == "CROSS" else kind
                elif kind == "INNER":
                    kind = "CROSS"
            from_items.append(ir.FromItem(sources[k], kind, on))
        where = self.expr(node.args["where"].this, here, "WHERE") if node.args.get("where") is not None else None
        group = ()
        if node.args.get("group") is not None:
            g = node.args["group"]
            for key, value in g.args.items():
                if key != "expressions" and not _empty(value):
                    raise Unsupported("GROUP BY modifier", key)
            group = tuple(self.positional_or_expr(e, here, "GROUP") for e in g.expressions)
        items = tuple(self.item(i, here) for i in node.expressions)
        having = self.expr(node.args["having"].this, here, "HAVING") if node.args.get("having") is not None else None
        order = ()
        if node.args.get("order") is not None:
            order = tuple(self.order_item(o, here) for o in node.args["order"].expressions)
        limit = self.count(node.args.get("limit"), "LIMIT")
        offset = self.count(node.args.get("offset"), "OFFSET")
        if offset is not None and limit is None:
            raise Unsupported("OFFSET without LIMIT")
        return ir.Select(from_=tuple(from_items), where=where, group_by=group, having=having,
                         distinct=distinct is not None, items=items, order_by=order, limit=limit, offset=offset)

    def join_kind(self, j: exp.Join) -> str:
        method = (j.args.get("method") or "").upper()
        side = (j.args.get("side") or "").upper()
        kind = (j.args.get("kind") or "").upper()
        if method:
            raise Unsupported(f"{method} JOIN")
        if side in ("RIGHT", "FULL"):
            raise Unsupported(f"{side} JOIN")
        if kind in ("SEMI", "ANTI", "ASOF", "STRAIGHT_JOIN", "ARRAY", "POSITIONAL"):
            raise Unsupported(f"{kind} JOIN")
        if side == "LEFT":
            return "LEFT"
        if kind == "CROSS":
            return "CROSS"
        return "INNER"

    def source(self, node, stack: list):
        node = _unparen(node)
        if isinstance(node, exp.Table):
            for key in ("db", "catalog", "joins", "pivots", "laterals", "hints", "sample", "version"):
                if not _empty(node.args.get(key)):
                    raise Unsupported("table modifier", key)
            name = node.name
            table = self.schema.table(name)
            if table is None:
                raise Unsupported("unknown table", name)
            alias = node.args.get("alias")
            if alias is not None and not _empty(alias.args.get("columns")):
                raise Unsupported("table alias with column list")
            binding = (alias.name if alias is not None and alias.name else name).lower()
            return ir.TableSource(table), _Source(alias=binding, table=table)
        if isinstance(node, exp.Subquery):
            alias = node.args.get("alias")
            if alias is not None and not _empty(alias.args.get("columns")):
                raise Unsupported("derived table alias with column list")
            # a derived table sees the enclosing SELECTs but not its own SELECT's sources (no LATERAL)
            inner = self.query(node.this, stack + [_Level(visible=False)])
            names, stars = self.output_names(node.this, inner)
            binding = alias.name.lower() if alias is not None and alias.name else None
            return ir.DerivedSource(inner), _Source(alias=binding, names=names, stars=stars)
        raise Unsupported("FROM source", type(node).__name__)

    def output_names(self, node, plan) -> tuple:
        """Lower-case output names of a derived query, by position, and the positions that come from `*`."""
        node = _unparen(node)
        while isinstance(node, (exp.Union, exp.Intersect, exp.Except, exp.Subquery)):
            node = _unparen(node.this)
        first = plan
        while isinstance(first, ir.Compound):
            first = first.left
        names, stars = [], set()
        for k, item in enumerate(node.expressions):
            if isinstance(first.items[k], ir.Star):
                star = first.items[k]
                chosen = range(len(first.from_)) if star.source is None else [star.source]
                for s in chosen:
                    src = first.from_[s].source
                    if isinstance(src, ir.TableSource):
                        cols = [c.lower() for c in self.schema.columns(src.table)]
                    else:
                        arity = ir.query_arity(src.query, self.schema)
                        cols = [None] * (arity or 0)
                    for c in cols:
                        stars.add(len(names))
                        names.append(c)
                continue
            if isinstance(item, exp.Alias):
                names.append(item.alias.lower())
            elif isinstance(item, exp.Column) and not isinstance(item.this, exp.Star):
                names.append(item.name.lower())
            else:
                names.append(None)
        return tuple(names), frozenset(stars)

    def using(self, using: list, k: int, level: _Level):
        terms = []
        for ident in using:
            name = ident.name if hasattr(ident, "name") else str(ident)
            left = None
            for idx in range(k):
                if self.has_column(level.sources[idx], name):
                    left = self.column_of(level.sources[idx], idx, 0, name)
                    break
            if left is None or not self.has_column(level.sources[k], name):
                raise Unsupported("USING column not found", name)
            terms.append(ir.Compare("=", left, self.column_of(level.sources[k], k, 0, name)))
        return terms[0] if len(terms) == 1 else ir.And(tuple(terms))

    def count(self, node, what: str):
        if node is None:
            return None
        value = _unparen(node.args.get("expression") if node.args.get("expression") is not None else node.this)
        if isinstance(value, exp.Literal) and not value.is_string and re.fullmatch(r"\d+", value.this):
            return int(value.this)
        raise Unsupported(f"{what} expression")

    # result columns and ordering
    def item(self, node, stack: list):
        if isinstance(node, exp.Alias):
            return self.expr(node.this, stack, "SELECT")
        if isinstance(node, exp.Star):
            if not _empty(node.args.get("except")) or not _empty(node.args.get("replace")):
                raise Unsupported("* modifier")
            return ir.Star(None)
        if isinstance(node, exp.Column) and isinstance(node.this, exp.Star):
            qualifier = node.table.lower()
            for idx, src in enumerate(stack[-1].sources):
                if src.alias == qualifier:
                    return ir.Star(idx)
            raise Unsupported("unresolved T.*", node.table)
        return self.expr(node, stack, "SELECT")

    def positional_or_expr(self, node, stack: list, mode: str):
        bare = _unparen(node)
        if isinstance(bare, exp.Literal) and not bare.is_string and re.fullmatch(r"\d+", bare.this):
            k = int(bare.this)
            items = stack[-1].items
            if not 1 <= k <= len(items):
                raise Unsupported(f"{mode} position out of range")
            target = items[k - 1]
            if isinstance(target, exp.Star) or (isinstance(target, exp.Column) and isinstance(target.this, exp.Star)):
                raise Unsupported(f"{mode} position refers to *")
            return self.expr(target, stack, mode)
        return self.expr(node, stack, mode)

    def order_item(self, node, stack: list) -> ir.OrderItem:
        if not isinstance(node, exp.Ordered):
            node = exp.Ordered(this=node)
        for key, value in node.args.items():
            if key not in ("this", "desc", "nulls_first") and not _empty(value):
                raise Unsupported("ORDER BY modifier", key)
        descending = bool(node.args.get("desc"))
        nulls_first = node.args.get("nulls_first")
        if nulls_first is not None and bool(nulls_first) == descending:
            raise Unsupported("explicit NULLS FIRST/LAST")
        term = _unparen(node.this)
        level = stack[-1]
        if isinstance(term, exp.Column) and not term.table and isinstance(term.this, exp.Identifier) \
                and term.name.lower() in level.aliases:
            return ir.OrderItem(self.inline_alias(term.name.lower(), stack, "ORDER"), descending)
        return ir.OrderItem(self.positional_or_expr(node.this, stack, "ORDER"), descending)

    def inline_alias(self, name: str, stack: list, mode: str):
        level = stack[-1]
        if name in level.expanding:
            raise Unsupported("recursive alias", name)
        level.expanding.add(name)
        try:
            return self.expr(level.aliases[name], stack, mode)
        finally:
            level.expanding.discard(name)

    # names
    def has_column(self, src: _Source, name: str) -> bool:
        if src.table is not None:
            return self.schema.column(src.table, name) is not None
        return name.lower() in src.names

    def column_of(self, src: _Source, idx: int, depth: int, name: str):
        if src.table is not None:
            canonical = self.schema.column(src.table, name)
            if canonical is None:
                raise Unsupported("unknown column", f"{src.table}.{name}")
            return ir.Column(depth, idx, canonical)
        positions = [p for p, n in enumerate(src.names) if n == name.lower()]
        if not positions:
            raise Unsupported("unresolved derived column", name)
        if len(positions) > 1:
            raise Unsupported("ambiguous derived column", name)
        if src.stars:
            raise Unsupported("reference into a derived table with *", name)
        return ir.DerivedColumn(depth, idx, positions[0])

    def visible(self, level: _Level):
        for idx, src in enumerate(level.sources):
            if level.left_limit is None or idx <= level.left_limit:
                yield idx, src

    def resolve(self, node: exp.Column, stack: list, mode: str):
        if not _empty(node.args.get("db")) or not _empty(node.args.get("catalog")):
            raise Unsupported("schema-qualified column")
        name = node.name
        qualifier = node.table.lower() if node.table else None
        for depth in range(len(stack)):
            level = stack[-1 - depth]
            if not level.visible:
                continue
            if qualifier is not None:
                for idx, src in self.visible(level):
                    if src.alias == qualifier:
                        return self.column_of(src, idx, depth, name)
                continue
            matches = [(idx, src) for idx, src in self.visible(level) if self.has_column(src, name)]
            if len(matches) > 1:
                raise Unsupported("ambiguous column", name)
            if matches:
                idx, src = matches[0]
                return self.column_of(src, idx, depth, name)
            if depth == 0 and mode in ("WHERE", "GROUP", "HAVING", "ORDER") and name.lower() in level.aliases:
                return self.inline_alias(name.lower(), stack, mode)
        if qualifier is None and node.this.args.get("quoted") and f'"{name}"' in self.sql:
            return ir.text(name)  # SQLite's double-quoted string literal fallback
        raise Unsupported("unresolved column", f"{qualifier + '.' if qualifier else ''}{name}")

    # scalars
    def expr(self, node, stack: list, mode: str):
        node = _unparen(node)
        if isinstance(node, exp.Column):
            if isinstance(node.this, exp.Star):
                raise Unsupported("* outside the SELECT list")
            return self.resolve(node, stack, mode)
        if isinstance(node, exp.Literal):
            return self.literal(node, negative=False)
        if isinstance(node, exp.Null):
            return ir.NULL
        if isinstance(node, exp.Boolean):
            return ir.integer(1 if node.this else 0)
        if isinstance(node, exp.Neg):
            inner = _unparen(node.this)
            if isinstance(inner, exp.Literal) and not inner.is_string:
                return self.literal(inner, negative=True)
            return ir.Unary("-", self.expr(inner, stack, mode))
        if type(node) in _ARITH:
            return ir.Binary(_ARITH[type(node)], self.expr(node.this, stack, mode),
                             self.expr(node.expression, stack, mode))
        if type(node) in _COMPARE:
            return ir.Compare(_COMPARE[type(node)], self.expr(node.this, stack, mode),
                              self.expr(node.expression, stack, mode))
        if isinstance(node, (exp.And, exp.Or)):
            cls = ir.And if isinstance(node, exp.And) else ir.Or
            return cls(tuple(self.expr(a, stack, mode) for a in self.flatten(node, type(node))))
        if isinstance(node, exp.Not):
            inner = self.expr(node.this, stack, mode)
            if isinstance(inner, (ir.Like, ir.Between, ir.InList, ir.InQuery, ir.IsNull, ir.Exists)):
                return type(inner)(**{**{f: getattr(inner, f) for f in inner.__dataclass_fields__},
                                      "negated": not inner.negated})
            return ir.Not(inner)
        if isinstance(node, exp.Is):
            if isinstance(_unparen(node.expression), exp.Null):
                return ir.IsNull(self.expr(node.this, stack, mode))
            raise Unsupported("IS with a non-NULL operand")
        if isinstance(node, exp.Like):
            if not _empty(node.args.get("escape")):
                raise Unsupported("LIKE ESCAPE")
            return ir.Like(self.expr(node.this, stack, mode), self.expr(node.expression, stack, mode),
                           bool(node.args.get("negate")))
        if isinstance(node, exp.Escape):
            raise Unsupported("LIKE ESCAPE")
        if isinstance(node, exp.Between):
            if not _empty(node.args.get("symmetric")):
                raise Unsupported("BETWEEN SYMMETRIC")
            return ir.Between(self.expr(node.this, stack, mode), self.expr(node.args["low"], stack, mode),
                              self.expr(node.args["high"], stack, mode))
        if isinstance(node, exp.In):
            for key in ("unnest", "field", "is_global"):
                if not _empty(node.args.get(key)):
                    raise Unsupported("IN form", key)
            if isinstance(_unparen(node.this), exp.Tuple):
                raise Unsupported("row value")
            arg = self.expr(node.this, stack, mode)
            if node.args.get("query") is not None:
                return ir.InQuery(arg, self.query(node.args["query"], stack))
            if not node.expressions:
                raise Unsupported("empty IN list")
            if len(node.expressions) == 1 and isinstance(_unparen(node.expressions[0]), (exp.Select, exp.Union)):
                return ir.InQuery(arg, self.query(node.expressions[0], stack))
            return ir.InList(arg, tuple(self.expr(e, stack, mode) for e in node.expressions))
        if isinstance(node, exp.Exists):
            return ir.Exists(self.query(node.this, stack))
        if isinstance(node, (exp.Subquery, exp.Select, exp.Union, exp.Intersect, exp.Except)):
            return ir.ScalarQuery(self.query(node, stack))
        if isinstance(node, exp.Case):
            operand = self.expr(node.this, stack, mode) if node.this is not None else None
            whens = tuple(ir.When(self.expr(i.this, stack, mode), self.expr(i.args["true"], stack, mode))
                          for i in node.args.get("ifs") or [])
            default = self.expr(node.args["default"], stack, mode) if node.args.get("default") is not None else None
            if not whens:
                raise Unsupported("CASE without WHEN")
            return ir.Case(operand, whens, default)
        if isinstance(node, exp.If):  # IIF(c, a, b)
            false = node.args.get("false")
            return ir.Case(None, (ir.When(self.expr(node.this, stack, mode), self.expr(node.args["true"], stack,
                                                                                      mode)),),
                           self.expr(false, stack, mode) if false is not None else None)
        if isinstance(node, exp.Cast):
            to = node.args["to"]
            key = to.this.value if hasattr(to.this, "value") else str(to.this)
            affinity = _SQLGLOT_TYPE_TO_AFFINITY.get(str(key).upper())
            if affinity is None:
                raise Unsupported("CAST type", str(key))
            return ir.Cast(self.expr(node.this, stack, mode), affinity)
        if type(node) in _AGG:
            return self.aggregate(node, stack, mode)
        if isinstance(node, (exp.Window, exp.WindowSpec)):
            raise Unsupported("window function")
        if isinstance(node, exp.GroupConcat):
            raise Unsupported("aggregate GROUP_CONCAT")
        return self.function(node, stack, mode)

    @staticmethod
    def flatten(node, cls) -> list:
        out = []
        for side in (node.this, node.expression):
            side = _unparen(side)
            if isinstance(side, cls):
                out.extend(_Converter.flatten(side, cls))
            else:
                out.append(side)
        return out

    def literal(self, node: exp.Literal, negative: bool):
        if node.is_string:
            if negative:
                raise Unsupported("negated string literal")
            return ir.text(node.this)
        t = node.this.strip()
        if re.fullmatch(r"\d+", t):
            value = -int(t) if negative else int(t)
            if ir.INT64[0] <= value <= ir.INT64[1]:
                return ir.integer(value)
            return ir.real(value)
        try:
            return ir.real(("-" if negative else "") + t)
        except ValueError as error:
            raise Unsupported("numeric literal", t) from error

    def aggregate(self, node, stack: list, mode: str):
        func = _AGG[type(node)]
        arg = node.this
        extra = node.args.get("expressions") or []
        if extra:
            if func in ("MIN", "MAX"):  # the scalar multi-argument form
                return ir.Func(func, tuple(self.expr(a, stack, mode) for a in [arg] + list(extra)))
            raise Unsupported(f"{func} with several arguments")
        for key, value in node.args.items():
            if key not in ("this", "expressions", "big_int") and not _empty(value):
                raise Unsupported(f"{func} modifier", key)
        if func == "COUNT" and isinstance(arg, exp.Star):
            return ir.CountStar()
        distinct = False
        if isinstance(arg, exp.Distinct):
            if len(arg.expressions) != 1:
                raise Unsupported("DISTINCT aggregate with several arguments")
            distinct, arg = True, arg.expressions[0]
        if arg is None:
            raise Unsupported(f"{func} without argument")
        return ir.Agg(func, self.expr(arg, stack, mode), distinct)

    def function(self, node, stack: list, mode: str):
        def args(*nodes):
            return tuple(self.expr(a, stack, mode) for a in nodes if a is not None)

        if isinstance(node, exp.TimeToStr):
            fmt = node.args.get("format")
            value = _unparen(node.this)
            if isinstance(value, exp.TsOrDsToTimestamp):
                value = _unparen(value.this)
            if isinstance(value, exp.CurrentTimestamp):
                return ir.Func("STRFTIME", (self.expr(fmt, stack, mode), ir.Func("CURRENT_TIMESTAMP", ())))
            return ir.Func("STRFTIME", args(fmt, value))
        if isinstance(node, exp.CurrentTimestamp):
            if not _empty(node.args.get("this")):
                raise Unsupported("CURRENT_TIMESTAMP with precision")
            return ir.Func("CURRENT_TIMESTAMP", ())
        if isinstance(node, exp.TsOrDsToTimestamp):
            raise Unsupported("internal timestamp conversion")
        if isinstance(node, exp.Date):
            extra = [node.args.get("zone")] + list(node.args.get("expressions") or [])
            return ir.Func("DATE", args(node.this, *extra))
        if isinstance(node, exp.Substring):
            return ir.Func("SUBSTR", args(node.this, node.args.get("start"), node.args.get("length")))
        if isinstance(node, exp.StrPosition):
            if not _empty(node.args.get("position")) or not _empty(node.args.get("occurrence")):
                raise Unsupported("INSTR with a start position")
            return ir.Func("INSTR", args(node.this, node.args.get("substr")))
        if isinstance(node, exp.Length):
            return ir.Func("LENGTH", args(node.this))
        if isinstance(node, exp.Replace):
            return ir.Func("REPLACE", args(node.this, node.expression, node.args.get("replacement")))
        if isinstance(node, exp.Trim):
            position = (node.args.get("position") or "").upper()
            name = {"": "TRIM", "BOTH": "TRIM", "LEADING": "LTRIM", "TRAILING": "RTRIM"}.get(position)
            if name is None or not _empty(node.args.get("collation")):
                raise Unsupported("TRIM form")
            return ir.Func(name, args(node.this, node.expression))
        if isinstance(node, exp.Lower):
            return ir.Func("LOWER", args(node.this))
        if isinstance(node, exp.Upper):
            return ir.Func("UPPER", args(node.this))
        if isinstance(node, exp.Abs):
            return ir.Func("ABS", args(node.this))
        if isinstance(node, exp.Round):
            if not _empty(node.args.get("truncate")):
                raise Unsupported("ROUND form")
            return ir.Func("ROUND", args(node.this, node.args.get("decimals")))
        if isinstance(node, exp.Coalesce):
            return ir.Func("COALESCE", args(node.this, *(node.expressions or [])))
        if isinstance(node, exp.Nullif):
            return ir.Func("NULLIF", args(node.this, node.expression))
        if isinstance(node, exp.Anonymous):
            name = str(node.this).upper()
            if name not in _ANONYMOUS_OK:
                raise Unsupported(f"function {name}")
            result = ir.Func(name, args(*node.expressions))
            lo, hi = ir.FUNCTIONS[name]
            if len(result.args) < lo or (hi is not None and len(result.args) > hi):
                raise Unsupported(f"function {name} arity")
            return result
        name = getattr(node, "sql_name", lambda: type(node).__name__)()
        raise Unsupported(f"function {name}" if isinstance(node, exp.Func) else "expression",
                          type(node).__name__)


def parse(sql: str):
    """One sqlglot statement (read="sqlite"), with CAST types rewritten to their affinity."""
    rewritten = canonical_cast_types(sql)
    try:
        statements = [s for s in sqlglot.parse(rewritten, read="sqlite") if s is not None]
    except Exception as error:  # sqlglot's ParseError / TokenError
        raise Unsupported("parse error", str(error).splitlines()[0][:200]) from error
    if len(statements) != 1:
        raise Unsupported("statement count", str(len(statements)))
    return statements[0]


def convert(sql: str, schema) -> "ir.Query":
    """Gold SQL to a validated tirel-1 plan, or `Unsupported`. `schema`: anything `ir.SchemaIndex.coerce` reads."""
    index = ir.SchemaIndex.coerce(schema)
    if index is None:
        raise ValueError("conversion needs the database schema")
    root = parse(sql)
    try:
        plan = _Converter(index, sql).query(root, [])
    except RecursionError as error:
        raise Unsupported("nesting too deep") from error
    errors = ir.validate(plan, index)
    if errors:
        raise Unsupported("invalid plan", "; ".join(errors[:3]))
    return plan


def try_convert(sql: str, schema) -> tuple:
    """(plan, None) or (None, Unsupported)."""
    try:
        return convert(sql, schema), None
    except Unsupported as error:
        return None, error


# --- coverage and the round trip by outcome --------------------------------------------------------------------------
COVERAGE_PIN = "ir_coverage_v1.json"


def _round_trip_job(batch):
    """Worker: score each compiled plan against its gold SQL with the benchmark's scorer (one shared 30 s deadline
    per pair, as the official harness does)."""
    from . import contracts as C
    from . import scorer

    out = []
    for task_id, snap, compiled, gold in batch:
        result = scorer.score(C.Snapshot(*snap), compiled, gold)
        out.append((task_id, {"status": result["status"], "correct": result["correct"],
                              "elapsed_seconds": result["elapsed_seconds"],
                              "compiled": result["predicted"]["status"], "gold": result["gold"]["status"],
                              "compiled_error": result["predicted"]["error"], "gold_error": result["gold"]["error"]}))
    return out


def _round_trip(jobs: list, workers: int, log) -> dict:
    """Parallel pass, then a serial re-run of every pair that timed out or used over half the timeout (the parallel
    pass shares the CPU); the serial result is final and both are kept."""
    import concurrent.futures

    from . import contracts as C

    timeout = C.Limits().sql_timeout_seconds
    by_db: dict = {}
    for job in jobs:
        by_db.setdefault(job[1][0], []).append(job)
    batches = [b[i:i + 16] for b in by_db.values() for i in range(0, len(b), 16)]
    results: dict = {}
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for k, done in enumerate(pool.map(_round_trip_job, batches)):
                results.update(done)
                if log and k % 50 == 0:
                    log(f"round trip: {len(results)}/{len(jobs)}")
    else:
        for batch in batches:
            results.update(_round_trip_job(batch))
    slow = [j for j in jobs if results[j[0]]["status"] in ("timeout", "predicted_timeout", "gold_timeout")
            or results[j[0]]["elapsed_seconds"] > timeout / 2]
    for job in slow:
        first = results[job[0]]
        again = _round_trip_job([job])[0][1]
        again["parallel_pass"] = first
        results[job[0]] = again
    if log:
        log(f"round trip: {len(results)} pairs, {len(slow)} re-run serially")
    return results


def coverage(task_ids=None, workers: int = 6, execute: bool = True, log=print) -> dict:
    """Convert, compile and round-trip every gold query of the training release (or `task_ids`).

    Returns `{"summary": per-split counts, "details": per-task outcome}`. A pair is round-trip-equivalent when the
    benchmark scorer gives EX(compiled) == EX(gold), i.e. set equality of the full result sets on the pinned snapshot.
    Exact SQL equality is reported only as a diagnostic."""
    from collections import Counter

    from . import data
    from . import schema as schema_mod

    dev = set(data.dev_databases())
    rows = data.rows() if task_ids is None else [data.row(t) for t in task_ids]
    indexes: dict = {}
    details: dict = {}
    jobs = []
    for r in rows:
        if r.db_id not in indexes:
            indexes[r.db_id] = ir.SchemaIndex.from_schema(schema_mod.load_schema(r.db_id))
        split = "dev" if r.db_id in dev else "train"
        entry = {"split": split, "db_id": r.db_id}
        plan, error = try_convert(r.sql, indexes[r.db_id])
        if error is not None:
            entry.update(converted=False, reason=error.reason, detail=error.detail[:300])
        else:
            compiled = compile_plan(plan)
            back, back_error = try_convert(compiled, indexes[r.db_id])
            entry.update(converted=True, compiled=compiled, plan_round_trip=back == plan and back_error is None,
                         sql_identical=" ".join(compiled.split()) == " ".join(r.sql.split()),
                         inventory=ir.operator_inventory(plan))
            if execute:
                snap = data.snapshot(r.db_id)
                jobs.append((r.task_id, (snap.db_id, snap.path, snap.sha256, snap.bytes), compiled, r.sql))
        details[r.task_id] = entry
    if execute and jobs:
        for task_id, outcome in _round_trip(jobs, workers, log).items():
            details[task_id]["round_trip"] = outcome
    summary: dict = {}
    for split in ("train", "dev"):
        items = [d for d in details.values() if d["split"] == split]
        converted = [d for d in items if d["converted"]]
        inventory: Counter = Counter()
        for d in converted:
            inventory.update(d["inventory"])
        s = {"questions": len(items), "converted": len(converted),
             "unsupported": dict(sorted(Counter(d["reason"] for d in items if not d["converted"]).items())),
             "plan_round_trip_exact": sum(d["plan_round_trip"] for d in converted),
             "sql_text_identical": sum(d["sql_identical"] for d in converted),
             "operator_inventory": dict(sorted(inventory.items()))}
        if execute:
            outcomes = Counter()
            for d in converted:
                rt = d.get("round_trip", {})
                if rt.get("gold") != "ok":
                    outcomes["gold_not_executable"] += 1
                elif rt.get("correct"):
                    outcomes["equivalent"] += 1
                elif rt.get("compiled") != "ok":
                    outcomes["compiled_" + str(rt.get("compiled"))] += 1
                else:
                    outcomes[str(rt.get("status"))] += 1
            s["round_trip"] = dict(sorted(outcomes.items()))
            s["round_trip_equivalent"] = outcomes["equivalent"]
        summary[split] = s
    return {"summary": summary, "details": details}


def compile_plan(plan) -> str:
    return ir.compile(plan)


def write_coverage_pin(result: dict, extra: dict | None = None) -> dict:
    """pins/ir_coverage_v1.json: compact counts only (per-question details go under runs/)."""
    import json
    import sqlite3

    from . import contracts as C
    from . import data

    payload = {"version": "ir_coverage_v1", "study": C.STUDY, "ir_version": ir.IR_VERSION,
               "converter": CONVERTER_VERSION, "sqlglot": sqlglot.__version__, "sqlite_version": sqlite3.sqlite_version,
               "split_pin_sha256": data.pin_digest(data.PIN_SPLIT),
               "snapshots_pin_sha256": data.pin_digest(data.PIN_SNAPSHOTS),
               "rule": "convert gold SQL with sql_to_ir.convert; compile the plan with ir.compile; score the compiled SQL "
                       "against the gold SQL with scorer.score (EX set equality, 30 s shared deadline); pairs that "
                       "time out or exceed 15 s in the parallel pass are re-run serially",
               "splits": result["summary"]}
    path = C.PINS / COVERAGE_PIN
    if path.exists():  # keep the sections other producers wrote (action_grammar, smoke_train, notes)
        kept = json.loads(path.read_text(encoding="utf-8"))
        payload = {**{k: v for k, v in kept.items() if k not in payload}, **payload}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def write_coverage_details(result: dict) -> str:
    import json

    from . import contracts as C

    path = C.RUNS / "ir" / "coverage_details_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result["details"], indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return str(path)


def main(argv=None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="python -m scaling.sql_to_ir",
                                     description="tirel-1 coverage and round trip by outcome")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-execute", action="store_true")
    parser.add_argument("--write-pin", action="store_true")
    args = parser.parse_args(argv)
    result = coverage(workers=args.workers, execute=not args.no_execute)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "operator_inventory"}
                      for k, v in result["summary"].items()}, indent=1))
    print("details:", write_coverage_details(result))
    if args.write_pin:
        write_coverage_pin(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
