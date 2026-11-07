"""Database evidence for one proposed SQL query: a deterministic, bounded lookup (CPU; no model; no reference).

**Inputs** are only what an inference path may see:
- the question and BIRD's evidence;
- the proposed SQL;
- the schema, with its descriptions and declared keys;
- the read-only snapshot.

No task id, reference SQL, expected answer or audit label is a parameter. Nothing here opens any file but the
snapshot.

**What it retrieves.** Each kind is a ranked list of candidates with their provenance (table, column, method):
- **value:** for each literal the proposal compares with a column, whether that column stores it verbatim, and the
  nearest stored values of that column. "Nearest" covers case, whitespace, punctuation, prefix, containment, shared
  words and character similarity.
- **value_elsewhere:** the text columns that store the literal, or contain it. The columns searched are those of the
  proposal's tables, their declared neighbours and the best-matching other tables.
- **column:** for each column the proposal filters on or returns, the columns of the same and neighbouring tables
  whose names are close to it or whose names and descriptions match the question and evidence.
- **table:** tables the proposal does not use whose names, columns and descriptions match the question and evidence,
  or whose names are close to the proposal's tables.
- **relationship:** the declared foreign keys among and around the proposal's tables, each proposal join checked
  against them, and composite keys joined on only part.
- **representation:** for each column the proposal uses numerically, orders by or compares with a literal, its
  declared type, its stored types and the text formats of its values (thousands separators, currency, percent, dates,
  numbers stored as text).

**What it does not do.** It never rewrites the proposal and never substitutes a literal: the output is evidence, not
a repair. An empty result does not show that a value is absent: scans stop at the limits below, and every limit hit is
recorded.

**Bounds.** All are deterministic:
- distinct values scanned per column, columns searched per literal, literals per proposal and candidates per item;
- SQLite VM instructions per query and per proposal, counted by a progress handler, so a stopped query stops at the
  same point on every run (a "timeout" here);
- the rendered payload's characters.

Wall-clock latency is reported, and a wall-clock safety stop is recorded separately if it ever fires.

**Example.** The proposal `SELECT COUNT(*) FROM shop WHERE city = 'springfield'` on a database that stores
'Springfield' gives:
- a value item: no verbatim row, and the candidate 'Springfield' (case) from shop.city;
- a representation item for shop.city.
The SQL is unchanged.
"""
from __future__ import annotations

import difflib
import re
import sqlite3
import time
import unicodedata
from dataclasses import asdict, dataclass

EVIDENCE_VERSION = "db-evidence-v1.4"   # columns bound in their SQL scope, tables kept in SQL order
# v1: one global alias map. v1.1 (8e729cf, superseded): scopes, but the table order changed by accident.
# v1.2 (352284e, superseded): bound a column a derived table or CTE of its scope might also supply.
# v1.3 (d04220b, superseded): read a CTE's or derived table's selects, not its column list.


@dataclass(frozen=True)
class Limits:
    distinct_scan: int = 20_000              # distinct values read from one column for the nearest-value search
    columns_per_literal: int = 40            # text columns searched for one literal elsewhere
    literals: int = 8                        # literals looked up per proposal
    candidates: int = 5                      # candidates kept per item
    columns_examined: int = 8                # proposal columns given a column item
    represented_columns: int = 12            # columns given a representation item
    alternative_tables: int = 3              # best-matching unused tables added to the searched tables
    sample_rows: int = 5_000                 # rows sampled for a column's stored types and formats
    query_instructions: int = 20_000_000     # SQLite VM instructions per query
    proposal_instructions: int = 400_000_000
    payload_chars: int = 6_000               # the rendered evidence a model would read
    wall_seconds: float = 120.0              # a safety stop only; the instruction budgets are the bound


STOPWORDS = frozenset("""a an the of in on at to for from by with and or not no is are was were be been being has have
had do does did what which who whom whose when where why how many much list name names give show find tell among
there their this that these those it its as than then also each per all any only most more least less please refers
refer refer_to equal equals value values number numbers""".split())


# --- text ------------------------------------------------------------------------------------------------------------
def _split_identifier(name: str) -> list[str]:
    """'nameGiven' -> ['name', 'given']; 'award_category' -> ['award', 'category']; 'TeamsSC' -> ['teams', 'sc']."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name))
    return [w.lower() for w in re.split(r"[^A-Za-z0-9]+", spaced) if w]


def _stem(word: str) -> str:
    for suffix in ("ies", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix) and not word.endswith("ss"):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def terms(text: str) -> set[str]:
    """The content words of a text or an identifier, lowercased and singular."""
    words = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*|\d+", text or ""):
        words += _split_identifier(token) if re.search(r"[a-z][A-Z]|_", token) else [token.lower()]
    return {_stem(w) for w in words if w not in STOPWORDS and len(w) > 1}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[\W_]+", " ", text).strip()


def value_match(literal: str, stored: str) -> "tuple[float, str] | None":
    """How close a stored value is to the proposal's literal: (score, method), or None."""
    if stored == literal:
        return 1.0, "verbatim"
    if stored.casefold().strip() == literal.casefold().strip():
        return 0.95, "case/whitespace"
    a, b = _norm(literal), _norm(stored)
    if not a or not b:
        return None
    if a == b:
        return 0.9, "punctuation"
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 2 and len(short) / len(long_) >= 0.5:
        if long_.startswith(short):
            return 0.8, "prefix"
        if short in long_:
            return 0.7, "contains"
    ta, tb = set(a.split()), set(b.split())
    jaccard = len(ta & tb) / len(ta | tb)
    if jaccard >= 0.5:
        return round(0.5 + 0.2 * jaccard, 4), "words"
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    if ratio >= 0.75:
        return round(0.6 * ratio, 4), "characters"
    return None


def _name_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _overlap(want: set, have: set) -> float:
    return len(want & have) / len(want) if want else 0.0


# --- the bounded connection ------------------------------------------------------------------------------------------
class _Budget:
    """Counts SQLite VM instructions through a progress handler and stops a query past its budget."""
    STEP = 1_000

    def __init__(self, conn: sqlite3.Connection, limits: Limits):
        self.conn, self.limits = conn, limits
        self.total = self.query = self.queries = 0
        self.timeouts, self.errors = [], []
        self.wall_stop = False
        self.start = time.perf_counter()
        conn.set_progress_handler(self._tick, self.STEP)

    def _tick(self) -> int:
        self.query += self.STEP
        self.total += self.STEP
        if self.query > self.limits.query_instructions or self.total > self.limits.proposal_instructions:
            return 1
        if time.perf_counter() - self.start > self.limits.wall_seconds:
            self.wall_stop = True
            return 1
        return 0

    def spent(self) -> bool:
        return self.total > self.limits.proposal_instructions or self.wall_stop

    def run(self, what: str, sql: str, params=()) -> "list | None":
        if self.spent():
            self.timeouts.append({"what": what, "reason": "proposal budget spent before the query"})
            return None
        self.query, self.queries = 0, self.queries + 1
        try:
            return self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as error:
            if "interrupt" in str(error):
                reason = ("wall-clock safety stop" if self.wall_stop else "proposal instruction budget"
                          if self.total > self.limits.proposal_instructions else "query instruction budget")
                self.timeouts.append({"what": what, "reason": reason})
            else:
                self.errors.append({"what": what, "error": str(error)[:200]})
            return None


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


# --- the proposal ----------------------------------------------------------------------------------------------------
def parse_proposal(sql: "str | None", schema) -> dict:
    """Tables, columns, literal predicates, numeric uses and joins of the proposal, resolved against the schema.

    **Scopes (db-evidence-v1.4).** Every column is bound in its own SELECT scope, as SQLite binds it:
    - A qualified column `q.c` binds to the source named `q` in its scope. In a subquery, if `q` is not a source
      there, the lookup moves outward through the enclosing scopes: a correlated column.
    - An unqualified column binds to the one source of its scope that has it. If two may have it, it is ambiguous,
      including when one of them is a derived table or CTE exposing that name (or a star). A derived table or CTE
      exposes its column list if it has one, else its select names. If none has it, a
      subquery or a set-operation branch looks outward.
    - A source that is a derived table or a CTE is not a stored table: its columns are not bound.

    A column that cannot be bound reliably is recorded under `unresolved` with the reason, and gets no
    column-specific evidence; nothing is guessed. A parse failure gives an empty structure.

    **Example.** In `SELECT x.name FROM customers AS x WHERE x.city = 'Paris' AND EXISTS (SELECT 1 FROM suppliers
    AS x WHERE x.city = 'Rome')`, 'Paris' binds to customers.city and 'Rome' to suppliers.city."""
    out = {"parsed": False, "error": None, "tables": [], "columns": [], "predicates": [], "numeric": [],
           "joins": [], "outputs": [], "unresolved": []}
    if not sql:
        out["error"] = "no SQL"
        return out
    import sqlglot
    from sqlglot import exp
    from sqlglot.optimizer.scope import ScopeType, traverse_scope
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        scopes = list(traverse_scope(tree)) if tree is not None else []
    except Exception as error:  # the parser or the scope analysis failed: retrieval continues from the question
        out["error"] = f"{type(error).__name__}: {str(error)[:200]}"
        return out
    if tree is None:
        out["error"] = "empty parse"
        return out
    by_name = {t.name.lower(): t for t in schema.tables}
    cols_of = {t.name: {c.name.lower(): c.name for c in t.columns} for t in schema.tables}
    used = []

    def sources(scope) -> dict:
        """alias or name (lowercase) -> ("stored", table, source id), or ("derived", names) for a derived table or CTE,
        where names is the set of columns it exposes or None when unknown (a star, or a table not in the schema)."""
        result = {}
        for key, src in scope.sources.items():
            if isinstance(src, exp.Table):
                real = by_name.get(src.name.lower())
                result[key.lower()] = ("stored", real.name, id(src)) if real is not None else ("derived", None)
            else:   # a CTE's column list, `WITH c(x) AS ...`, is what SQLite exposes; else the select names
                parent = src.expression.parent
                alias = parent.args.get("alias") if isinstance(parent, (exp.CTE, exp.Subquery)) else None
                listed = [c.name for c in alias.columns] if alias is not None and alias.columns else None
                named = [n.lower() for n in (listed or getattr(src.expression, "named_selects", None) or ["*"])]
                result[key.lower()] = ("derived", None if "*" in named else frozenset(named))
        return result

    def outward(scope):
        """The scope a correlated column is looked up in next: a subquery's enclosing scope; a set-operation branch's
        set operation (itself looked through if it is a subquery). Derived tables and CTEs see no outer scope."""
        if scope.scope_type in (ScopeType.SUBQUERY, ScopeType.SET_OPERATION):
            return scope.parent
        return None

    binding = {}   # id(column node) -> (table, column, source id) or a reason it is unresolved

    def bind(node, scope):
        name, qual = node.name.lower(), (node.table or "").lower()
        s = scope
        while s is not None:
            src = sources(s)
            if qual:
                if qual in src:
                    hit = src[qual]
                    if hit[0] == "derived":
                        return "the source is a derived table, a CTE or not in the schema"
                    return (hit[1], cols_of[hit[1]][name], hit[2]) if name in cols_of[hit[1]] else \
                        f"{hit[1]} has no column {node.name}"
            else:
                owners = [v for v in src.values() if v[0] == "stored" and name in cols_of[v[1]]]
                maybe = [v for v in src.values() if v[0] == "derived" and (v[1] is None or name in v[1])]
                if len(owners) + len(maybe) > 1:
                    return "ambiguous: more than one source of its scope has this column" if not maybe else \
                        "ambiguous: a derived table or CTE of its scope may also have this column"
                if owners:
                    return (owners[0][1], cols_of[owners[0][1]][name], owners[0][2])
                if maybe:
                    return "comes from a derived table or CTE"
            s = outward(s)
        return "no source in its scope or an enclosing one" + (f" is named {node.table}" if qual else " has it")

    stored = {id(src) for scope in scopes for src in scope.sources.values() if isinstance(src, exp.Table)}
    for t in tree.find_all(exp.Table):   # the stored tables, in the order the SQL names them (as in v1)
        real = by_name.get(t.name.lower())
        if id(t) in stored and real is not None and real.name not in used:
            used.append(real.name)
    for scope in scopes:                 # innermost scopes first: a node is bound where it appears
        for node in scope.columns:
            if id(node) not in binding:
                binding[id(node)] = bind(node, scope)
    for node in tree.find_all(exp.Column):
        b = binding.get(id(node), "outside any analysed scope (for example an alias of the select list)")
        if not isinstance(b, tuple):
            item = {"column": node.sql(dialect="sqlite"), "reason": b}
            if item not in out["unresolved"]:
                out["unresolved"].append(item)

    def unwrap(node):
        while isinstance(node, (exp.Paren, exp.Lower, exp.Upper, exp.Trim, exp.Cast)):
            node = node.this
        return node

    def resolve(node) -> "tuple | None":
        node = unwrap(node)
        b = binding.get(id(node)) if isinstance(node, exp.Column) else None
        return (b[0], b[1]) if isinstance(b, tuple) else None

    def source_of(node):
        b = binding.get(id(unwrap(node)))
        return b[2] if isinstance(b, tuple) else None

    def literal(node) -> "tuple | None":
        neg = False
        while isinstance(node, (exp.Paren, exp.Neg)):
            neg ^= isinstance(node, exp.Neg)
            node = node.this
        if isinstance(node, exp.Literal):
            return (("-" if neg else "") + node.this, node.is_string)
        return None

    def add(key, item):
        if item not in out[key]:
            out[key].append(item)

    for node in tree.find_all(exp.Column):
        r = resolve(node)
        if r:
            add("columns", list(r))
    comparisons = (exp.EQ, exp.NEQ, exp.Like, exp.ILike, exp.GT, exp.GTE, exp.LT, exp.LTE)
    for node in tree.find_all(*comparisons):
        op = type(node).__name__.lower()
        for side, other in ((node.this, node.expression), (node.expression, node.this)):
            col, lit = resolve(side), literal(other)
            if col and lit:
                add("predicates", {"table": col[0], "column": col[1], "op": op, "literal": lit[0], "string": lit[1]})
                if not lit[1] and op not in ("eq", "neq"):
                    add("numeric", list(col))
        if isinstance(node, exp.EQ):
            a, b = resolve(node.this), resolve(node.expression)
            qa = (node.this.table or "").lower() if isinstance(node.this, exp.Column) else ""
            qb = (node.expression.table or "").lower() if isinstance(node.expression, exp.Column) else ""
            if a and b and (a != b or source_of(node.this) != source_of(node.expression)):
                add("joins", {"left": list(a), "right": list(b), "left_alias": qa, "right_alias": qb})
    for node in tree.find_all(exp.In):
        col = resolve(node.this)
        for e in node.expressions:
            lit = literal(e)
            if col and lit:
                add("predicates", {"table": col[0], "column": col[1], "op": "in", "literal": lit[0], "string": lit[1]})
    for node in tree.find_all(exp.Between):
        col = resolve(node.this)
        if col:
            add("numeric", list(col))
    numeric_ops = (exp.Sum, exp.Avg, exp.Min, exp.Max, exp.Add, exp.Sub, exp.Mul, exp.Div)
    for node in tree.find_all(*numeric_ops):
        for c in node.find_all(exp.Column):
            r = resolve(c)
            if r:
                add("numeric", list(r))
    for node in tree.find_all(exp.Ordered):
        for c in node.find_all(exp.Column):
            r = resolve(c)
            if r:
                add("numeric", list(r))
    if isinstance(tree, exp.Select):
        for e in tree.expressions:
            for c in e.find_all(exp.Column):
                r = resolve(c)
                if r:
                    add("outputs", list(r))
    out.update(parsed=True, tables=used)
    return out


# --- the schema ------------------------------------------------------------------------------------------------------
def _neighbours(schema, tables: list) -> list:
    """Tables one declared foreign key away from any of `tables`, in schema order."""
    names = set(tables)
    near = set()
    for t in schema.tables:
        for _col, ref, _to in t.foreign_keys:
            if t.name in names and ref not in names:
                near.add(ref)
            if ref in names and t.name not in names:
                near.add(t.name)
    return [t.name for t in schema.tables if t.name in near]


def _table_terms(table) -> set:
    return terms(table.name) | {w for c in table.columns for w in terms(c.name)}


def _column_terms(column) -> set:
    return terms(column.name) | terms(column.description)


def _is_texty(column) -> bool:
    declared = (column.type or "").upper()
    return not declared or any(k in declared for k in ("CHAR", "TEXT", "CLOB", "VARCHAR", "STRING"))


def rank_tables(schema, question_terms: set, proposal_tables: list, limit: int) -> list:
    """Unused tables, by the share of the question's and evidence's terms their names and columns cover and by name
    closeness to a proposal table."""
    rows = []
    for t in schema.tables:
        if t.name in proposal_tables:
            continue
        overlap = _overlap(question_terms, _table_terms(t))
        close = max((_name_similarity(t.name, p) for p in proposal_tables), default=0.0)
        score = round(0.6 * overlap + 0.4 * close, 4)
        if score > 0:
            rows.append({"table": t.name, "score": score, "provenance": {
                "method": "question/evidence terms in the table's name and columns; name closeness to the proposal",
                "term_overlap": round(overlap, 4), "closest_proposal_table": round(close, 4)}})
    rows.sort(key=lambda r: (-r["score"], r["table"]))
    return rows[:limit]


# --- the retrieval ---------------------------------------------------------------------------------------------------
def _quoted(text: str) -> list:
    return [a or b for a, b in re.findall(r"'([^']+)'|\"([^\"]+)\"", text or "")]


def retrieve(question: str, evidence: str, sql: "str | None", schema, snapshot_path: str,
             limits: Limits = Limits()) -> dict:
    """The evidence package for one proposal. Deterministic for a given snapshot, schema and SQLite build, except
    the latency fields."""
    from .tools import connect_readonly
    t0 = time.perf_counter()
    proposal = parse_proposal(sql, schema)
    table_of = {t.name: t for t in schema.tables}
    column_of = {(t.name, c.name): c for t in schema.tables for c in t.columns}
    qterms = terms(question) | terms(evidence)
    conn = connect_readonly(snapshot_path)
    conn.text_factory = lambda b: b.decode("utf-8", "replace")
    budget = _Budget(conn, limits)
    items, scan_limits = [], []
    try:
        alt_tables = rank_tables(schema, qterms, proposal["tables"], limits.candidates)
        searched = list(proposal["tables"]) + [t for t in _neighbours(schema, proposal["tables"])]
        for row in alt_tables[: limits.alternative_tables]:
            if row["table"] not in searched:
                searched.append(row["table"])

        # value: the proposal's string literals against their own columns
        literals, seen = [], set()
        for p in proposal["predicates"]:
            lit = p["literal"].strip("%") if p["op"] in ("like", "ilike") else p["literal"]
            key = (p["table"], p["column"], lit)
            if not p["string"] or not lit or key in seen or len(literals) >= limits.literals:
                continue
            seen.add(key)
            literals.append({"literal": lit, "table": p["table"], "column": p["column"], "op": p["op"],
                             "pattern": p["literal"] if p["op"] in ("like", "ilike") else None,
                             "source": "proposal"})
        for lit in _quoted(evidence) + _quoted(question):   # literals the evidence or question quotes
            lit = lit.strip("%")
            if len(literals) >= limits.literals:
                break
            if lit and not any(x["literal"] == lit for x in literals):
                literals.append({"literal": lit, "table": None, "column": None, "op": None, "pattern": None,
                                 "source": "evidence/question"})
        for x in literals:
            if x["table"] is None:
                continue
            t, c = x["table"], x["column"]
            if x["pattern"] is not None:      # a LIKE predicate: count what its own pattern matches
                exact = budget.run(f"value pattern {t}.{c}", f"SELECT COUNT(*) FROM {_q(t)} WHERE {_q(c)} LIKE ?",
                                   (x["pattern"],))
            else:
                exact = budget.run(f"value exact {t}.{c}", f"SELECT COUNT(*) FROM {_q(t)} WHERE {_q(c)} = ?",
                                   (x["literal"],))
            x["verbatim_rows"] = exact[0][0] if exact else None
            stored = budget.run(f"value scan {t}.{c}",
                                f"SELECT DISTINCT {_q(c)} FROM {_q(t)} WHERE {_q(c)} IS NOT NULL LIMIT ?",
                                (limits.distinct_scan + 1,))
            hit = stored is not None and len(stored) > limits.distinct_scan
            if hit:
                scan_limits.append({"what": f"value scan {t}.{c}", "limit": limits.distinct_scan})
            cands = []
            for (v,) in (stored or [])[: limits.distinct_scan]:
                m = value_match(x["literal"], str(v))
                if m:
                    cands.append({"value": str(v), "score": m[0], "provenance": {
                        "table": t, "column": c, "method": m[1]}})
            cands.sort(key=lambda r: (-r["score"], abs(len(r["value"]) - len(x["literal"])), r["value"]))
            items.append({"kind": "value", "subject": {"literal": x["literal"], "table": t, "column": c,
                                                       "op": x["op"], "pattern": x["pattern"]},
                          "verbatim_rows": x["verbatim_rows"],
                          "scanned_distinct": min(len(stored or []), limits.distinct_scan),
                          "scan_limit_hit": hit, "candidates": cands[: limits.candidates]})

        # value_elsewhere: a literal its own column does not store (or that only the evidence or question quotes),
        # in the other text columns of the searched tables
        for x in literals:
            if x["table"] is not None and x.get("verbatim_rows"):
                continue
            own = (x["table"], x["column"])
            pool = [(t, c.name) for t in searched for c in table_of[t].columns
                    if _is_texty(c) and (t, c.name) != own]
            pool.sort(key=lambda tc: (searched.index(tc[0]) if tc[0] in proposal["tables"] else len(searched),
                                      -_name_similarity(tc[1], x["column"] or x["literal"]), tc[0], tc[1]))
            if len(pool) > limits.columns_per_literal:
                scan_limits.append({"what": f"columns searched for {x['literal']!r}",
                                    "limit": limits.columns_per_literal, "available": len(pool)})
            cands = []
            for t, c in pool[: limits.columns_per_literal]:
                rows = budget.run(f"elsewhere {t}.{c}", f"SELECT {_q(c)}, COUNT(*) FROM {_q(t)} WHERE {_q(c)} = ? "
                                  f"COLLATE NOCASE GROUP BY {_q(c)} ORDER BY 2 DESC LIMIT 3", (x["literal"],))
                if not rows and len(x["literal"]) >= 3:
                    pattern = "%" + re.sub(r"([%_\\])", r"\\\1", x["literal"]) + "%"
                    rows = budget.run(f"elsewhere contains {t}.{c}",
                                      f"SELECT DISTINCT {_q(c)}, NULL FROM {_q(t)} WHERE {_q(c)} LIKE ? ESCAPE '\\' "
                                      f"LIMIT 3", (pattern,))
                for v, n in rows or []:
                    m = value_match(x["literal"], str(v))
                    if m:
                        cands.append({"value": str(v), "score": m[0], "provenance": {
                            "table": t, "column": c, "method": m[1], "rows": n,
                            "table_role": "proposal" if t in proposal["tables"] else "neighbour or alternative"}})
            cands.sort(key=lambda r: (-r["score"], r["provenance"]["table"], r["provenance"]["column"], r["value"]))
            items.append({"kind": "value_elsewhere", "subject": {"literal": x["literal"], "table": x["table"],
                                                                 "column": x["column"], "source": x["source"]},
                          "columns_searched": min(len(pool), limits.columns_per_literal),
                          "candidates": cands[: limits.candidates]})

        # column: alternatives to each examined proposal column
        examined = []
        for col in ([[p["table"], p["column"]] for p in proposal["predicates"]] + proposal["outputs"]
                    + proposal["numeric"]):
            if col not in examined and len(examined) < limits.columns_examined:
                examined.append(col)
        used_cols = {tuple(c) for c in proposal["columns"]}
        for t, c in examined:
            near = [t] + _neighbours(schema, [t])
            cands = []
            for nt in near:
                for other in table_of[nt].columns:
                    if (nt, other.name) in used_cols:
                        continue
                    close = _name_similarity(other.name, c)
                    overlap = _overlap(qterms, _column_terms(other))
                    score = round(0.5 * close + 0.5 * overlap, 4)
                    if score >= 0.3:
                        cands.append({"column": [nt, other.name], "score": score, "provenance": {
                            "method": "name closeness to the proposal's column; question/evidence terms in the name "
                                      "and description", "name_closeness": round(close, 4),
                            "term_overlap": round(overlap, 4),
                            "description": (other.description or "")[:120]}})
            cands.sort(key=lambda r: (-r["score"], r["column"]))
            items.append({"kind": "column", "subject": {"table": t, "column": c,
                                                        "description": (column_of[(t, c)].description or "")[:120]},
                          "candidates": cands[: limits.candidates]})

        # table
        items.append({"kind": "table", "subject": {"proposal_tables": proposal["tables"]},
                      "candidates": alt_tables})

        # relationship
        fks = []
        for t in schema.tables:
            for col, ref, to in t.foreign_keys:
                if t.name in proposal["tables"] or ref in proposal["tables"]:
                    fks.append([t.name, col, ref, to])
        declared = {(a, b, c, d) for a, b, c, d in fks} | {(c, d, a, b) for a, b, c, d in fks}
        checks = []
        for j in proposal["joins"]:
            (lt, lc), (rt, rc) = j["left"], j["right"]
            status = "declared" if (lt, lc, rt, rc) in declared else "not declared"
            note = None
            if lt == rt and lc == rc:
                note = "joins a column to itself"
            for a, b in ((lt, rt), (rt, lt)):
                pk = [c.name for c in table_of[a].columns if c.primary_key]
                if len(pk) > 1 and (lc in pk or rc in pk):
                    rest = [k for k in pk if k not in (lc, rc) and k in {x.name for x in table_of[b].columns}]
                    joined = {(x["left"][1], x["right"][1]) for x in proposal["joins"]
                              if {x["left"][0], x["right"][0]} == {a, b}}
                    missing = [k for k in rest if (k, k) not in joined]
                    if missing:
                        note = f"{a} has the composite key ({', '.join(pk)}); {b} also has {', '.join(missing)}"
            checks.append({**j, "status": status, "note": note})
        items.append({"kind": "relationship", "subject": {"proposal_tables": proposal["tables"]},
                      "declared": fks[: 4 * limits.candidates], "declared_total": len(fks), "joins": checks,
                      "candidates": [{"join": c, "score": 1.0 if c["status"] == "not declared" or c["note"] else 0.0,
                                      "provenance": {"method": "declared foreign keys and primary keys"}}
                                     for c in checks if c["status"] == "not declared" or c["note"]]})

        # representation
        rep_cols = []
        for col in proposal["numeric"] + [[p["table"], p["column"]] for p in proposal["predicates"]]:
            if col not in rep_cols and len(rep_cols) < limits.represented_columns:
                rep_cols.append(col)
        numeric_cols = {tuple(c) for c in proposal["numeric"]}
        for t, c in rep_cols:
            rows = budget.run(f"representation {t}.{c}",
                              f"SELECT typeof({_q(c)}), {_q(c)} FROM {_q(t)} LIMIT ?", (limits.sample_rows,))
            if rows is None:
                items.append({"kind": "representation", "subject": {"table": t, "column": c}, "sampled": 0,
                              "candidates": []})
                continue
            kinds, formats, examples = {}, {}, {}
            for typ, v in rows:
                kinds[typ] = kinds.get(typ, 0) + 1
                if typ == "text":
                    f = text_format(v)
                    formats[f] = formats.get(f, 0) + 1
                    examples.setdefault(f, [])
                    if len(examples[f]) < 3 and v not in examples[f]:
                        examples[f].append(v)
            literal_kinds = sorted({"string" if p["string"] else "number" for p in proposal["predicates"]
                                    if [p["table"], p["column"]] == [t, c]})
            cands = []
            nonnumeric_text = sum(n for f, n in formats.items() if f not in ("number as text",))
            if (t, c) in numeric_cols and nonnumeric_text:
                cands.append({"note": "used numerically, but stores text that is not a plain number",
                              "score": 1.0, "provenance": {"method": "typeof and text formats of sampled rows"}})
            if formats.get("number as text"):
                cands.append({"note": "stores numbers as text (text comparison and sorting differ from numeric)",
                              "score": 0.8, "provenance": {"method": "typeof and text formats of sampled rows"}})
            if "number" in literal_kinds and kinds.get("text") and not kinds.get("integer") and not kinds.get("real"):
                cands.append({"note": "compared with a number, but every sampled value is text", "score": 0.7,
                              "provenance": {"method": "typeof of sampled rows"}})
            if "string" in literal_kinds and not kinds.get("text") and (kinds.get("integer") or kinds.get("real")):
                cands.append({"note": "compared with a string, but every sampled value is a number", "score": 0.7,
                              "provenance": {"method": "typeof of sampled rows"}})
            items.append({"kind": "representation", "subject": {"table": t, "column": c},
                          "declared_type": column_of[(t, c)].type, "sampled": len(rows),
                          "scan_limit_hit": len(rows) >= limits.sample_rows,
                          "stored_types": dict(sorted(kinds.items())), "text_formats": dict(sorted(formats.items())),
                          "examples": {k: examples[k] for k in sorted(examples)}, "candidates": cands})
    finally:
        conn.close()
    for item in items:
        for rank, cand in enumerate(item["candidates"], 1):
            cand["rank"] = rank
    text, shown = render(items, limits.payload_chars)
    full, _ = render(items, None)
    return {"version": EVIDENCE_VERSION, "limits": asdict(limits), "proposal": proposal, "items": items,
            "rendered": text, "rendered_items": shown, "payload_chars": len(text), "payload_chars_untruncated":
            len(full), "truncated": len(full) > len(text),
            "queries": budget.queries, "instructions": budget.total, "timeouts": budget.timeouts,
            "errors": budget.errors, "scan_limits": scan_limits, "wall_safety_stop": budget.wall_stop,
            "latency_seconds": time.perf_counter() - t0}


FORMATS = (("empty", re.compile(r"^\s*$")),
           ("number as text", re.compile(r"^[+-]?\d+(\.\d+)?$")),
           ("thousands separators", re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")),
           ("currency", re.compile(r"^[+-]?\s*[$€£¥]\s*[\d,]+(\.\d+)?$|^[\d,]+(\.\d+)?\s*[$€£¥]$")),
           ("percent", re.compile(r"^[+-]?[\d.]+\s*%$")),
           ("date", re.compile(r"^\d{4}-\d{1,2}-\d{1,2}([ T].*)?$|^\d{1,2}/\d{1,2}/\d{2,4}$")),
           ("null-like text", re.compile(r"^(null|none|n/?a|nan|-)$", re.I)))


def text_format(value: str) -> str:
    for name, pattern in FORMATS:
        if pattern.match(value):
            return name
    return "other text"


# --- what a model would read -----------------------------------------------------------------------------------------
ORDER = ("value", "value_elsewhere", "relationship", "representation", "column", "table")


def _line(item: dict) -> list:
    k, s, c = item["kind"], item["subject"], item["candidates"]
    if k == "value":
        stores = "match the pattern " + repr(s["pattern"]) if s.get("pattern") else "store it verbatim"
        head = (f"Value {s['literal']!r} in {s['table']}.{s['column']}: {item['verbatim_rows']} rows {stores}"
                if item["verbatim_rows"] is not None else
                f"Value {s['literal']!r} in {s['table']}.{s['column']}: the verbatim count was not obtained")
        head += f" (distinct values scanned: {item['scanned_distinct']}{', limit reached' if item['scan_limit_hit'] else ''})"
        return [head] + [f"  {x['rank']}. stored {x['value']!r} ({x['provenance']['method']})" for x in c
                         if x["provenance"]["method"] != "verbatim"]
    if k == "value_elsewhere":
        if not c:
            return [f"Value {s['literal']!r} elsewhere: not found in {item['columns_searched']} columns searched "
                    "(not proof of absence)"]
        return [f"Value {s['literal']!r} elsewhere:"] + [
            f"  {x['rank']}. {x['provenance']['table']}.{x['provenance']['column']} stores {x['value']!r} "
            f"({x['provenance']['method']})" for x in c]
    if k == "relationship":
        out = [f"Declared keys around {', '.join(s['proposal_tables'])}: " + "; ".join(
            f"{a}.{b} -> {cc}.{d}" for a, b, cc, d in item["declared"]) +
            (f" (+{item['declared_total'] - len(item['declared'])} more)"
             if item["declared_total"] > len(item["declared"]) else "")]
        for j in item["joins"]:
            if j["status"] == "not declared" or j["note"]:
                out.append(f"  Join {'.'.join(j['left'])} = {'.'.join(j['right'])}: {j['status']}"
                           + (f"; {j['note']}" if j["note"] else ""))
        return out
    if k == "representation":
        if not item.get("sampled"):
            return [f"Column {s['table']}.{s['column']}: not sampled"]
        fm = ", ".join(f"{f} {n}" + (f" e.g. {item['examples'][f][:2]}" if f in item["examples"] else "")
                       for f, n in item["text_formats"].items())
        return [f"Column {s['table']}.{s['column']} (declared {item['declared_type'] or 'none'}): stored "
                + ", ".join(f"{t} {n}" for t, n in item["stored_types"].items()) + (f"; text: {fm}" if fm else "")] + \
            [f"  note: {x['note']}" for x in c]
    if k == "column":
        if not c:
            return []
        return [f"Columns near {s['table']}.{s['column']}: " + "; ".join(
            f"{x['rank']}. {x['column'][0]}.{x['column'][1]}"
            + (f" ({x['provenance']['description'][:60]})" if x["provenance"]["description"] else "") for x in c)]
    if k == "table":
        if not c:
            return []
        return ["Other tables matching the question: " + "; ".join(f"{x['rank']}. {x['table']}" for x in c)]
    return []


def render(items: list, budget_chars: "int | None") -> tuple:
    """(text, indexes of the items fully shown). Items are added whole, in ORDER, until the budget is spent."""
    lines, shown, used = [], [], 0
    for kind in ORDER:
        for i, item in enumerate(items):
            if item["kind"] != kind:
                continue
            block = "\n".join(_line(item))
            if not block:
                continue
            if budget_chars is not None and used + len(block) + 1 > budget_chars:
                continue
            lines.append(block)
            shown.append(i)
            used += len(block) + 1
    return "\n".join(lines), sorted(shown)
