"""The no-fit control L: retrieved training SQL used as templates, re-aimed at the request's own database.

**Example** (`test_single_table_template_with_a_quoted_value` in `tests/test_ti_tnn_scaling_baseline_l.py`). A
request on a `shop` database asks "What is the city of the customer named 'Ann'?". Retrieval returns a training
example from another database whose gold SQL is `SELECT city FROM users WHERE user_name = 'Bob'`. L maps `users` to
`customers` and `user_name` to `customer_name` by lexical similarity, and replaces 'Bob' with 'Ann', the quoted string
in the question. It runs `SELECT city FROM customers WHERE customer_name = 'Ann'` with `execute_select` on the
request's snapshot, gets one row, and submits it. Nothing is learned or tuned, so L is a floor: it shows what
retrieval plus deterministic mechanics achieve.

**The algorithm.** For each retrieved example in rank order (the same k = 3 every arm receives):
1. Parse its SQL with sqlglot. Resolve each column to its source table using the source database's schema (training
   material): first among the tables of the column's own SELECT, then of the enclosing ones.
2. Map source tables to target tables, and choose up to 2 assignments. A target table scores by its name's overlap
   with the source table's name and with the request's question plus evidence, and by its best matches for the
   source columns used. A join between two source tables earns a bonus when the two target tables are linked by a
   foreign key. Small templates (3 tables or fewer) are searched exhaustively over each table's 6 best candidates;
   larger ones greedily.
3. Map columns within the chosen tables, greedily in descending score with each target column used once if possible.
   A column's score combines name and description overlap with the source column, relevance to the request, type
   agreement, and primary-key agreement. Join columns follow the target foreign key. A column the evidence names in
   `<column> = <value>` gets a large bonus when the template compares it with a literal.
4. Rename consistently. References to a derived table's outputs follow its renamed select list. `JOIN ... USING`
   names are renamed too, and become `ON a.x = b.y` when the two sides map to different names. An unqualified column
   gets its table's alias when the mapping would otherwise make it ambiguous.
5. Refill each literal compared with a column (`=`, `<>`, `<`, `>`, `<=`, `>=`, LIKE, IN, BETWEEN). In order, use the
   value the evidence binds to that target column, then quoted strings, then numbers from the question and evidence,
   then a capitalized phrase from the question confirmed with `lookup_values` (at most 2 lookups per template).
   A literal with no grounded value keeps its template value, and the attempt is marked `ungrounded`.
6. Run the candidate with `execute_select`.

**Selection.** Submit the first candidate that runs with at least one row; otherwise the first that runs at all;
otherwise abstain with the reason. Every attempt is recorded: the template id, the mappings, the fills, the SQL and
each tool call. Tool calls (lookups plus executions) are capped at `contracts.Limits().max_tool_calls` (8), the
same per-request cap as the other arms.

**What L may see:** the `Task`, the retrieved training examples (training-pool gold SQL is permitted material), the
schemas of those examples' training databases, the request's own schema, and the tools on the request's own
snapshot. It never receives a `Gold`.
"""
from __future__ import annotations

import itertools
import math
import re
import time
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from . import contracts as C
from . import schema as schema_mod
from . import tools

L_VERSION = "baseline-l-v1"
ASSIGNMENTS_PER_TEMPLATE = 2
TABLE_CANDIDATES = 6
EXACT_SEARCH_TABLES = 3
LOOKUPS_PER_TEMPLATE = 2
_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.ILike, exp.In, exp.Between)
_QUESTION_WORDS = {"who", "what", "which", "when", "where", "how", "why", "list", "name", "give", "provide", "please",
                   "among", "find", "state", "show", "calculate", "indicate", "describe", "identify", "is", "are",
                   "was", "were", "do", "does", "did", "in", "of", "for", "the", "a", "an", "tell", "count", "mention"}
_NUMERIC_TYPES = ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC", "BIT", "BOOL")


# --- words ------------------------------------------------------------------------------------------------------------
def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and (word.endswith("ses") or word.endswith("xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def words(text: str) -> list[str]:
    """Lowercase stemmed words; identifiers are split at underscores, spaces and camelCase."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text or ""))
    return [_stem(w) for w in re.findall(r"[a-z]+|[0-9]+", text.lower()) if len(w) > 1 or w.isdigit()]


def _alnum(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", str(text).lower())


def _is_numeric_type(declared: str) -> bool:
    return any(t in (declared or "").upper() for t in _NUMERIC_TYPES)


# --- the request's side -----------------------------------------------------------------------------------------------
@dataclass
class Target:
    """The request's schema, indexed for lexical scoring."""
    schema: C.Schema
    question_words: set
    tables: dict = field(default_factory=dict)       # lower name -> SchemaTable
    columns: dict = field(default_factory=dict)      # (table, column) -> SchemaColumn
    name_words: dict = field(default_factory=dict)   # (table, column) -> set
    desc_words: dict = field(default_factory=dict)   # (table, column) -> set
    idf: dict = field(default_factory=dict)
    relevance: dict = field(default_factory=dict)    # (table, column) -> float
    table_relevance: dict = field(default_factory=dict)
    bindings: list = field(default_factory=list)     # [(table, column, op, value)] from the evidence
    bound: set = field(default_factory=set)          # {(table, column)} named by an evidence binding

    @classmethod
    def build(cls, schema: C.Schema, question: str, evidence: str) -> "Target":
        target = cls(schema=schema, question_words=set(words(f"{question} {evidence}")))
        document_frequency: dict[str, int] = {}
        for table in schema.tables:
            target.tables[table.name.lower()] = table
            for column in table.columns:
                key = (table.name, column.name)
                target.columns[key] = column
                target.name_words[key] = set(words(column.name))
                target.desc_words[key] = set(words(" ".join(
                    (column.description, column.value_description, schema_mod.compact_meaning(column.meaning)))))
                for w in target.name_words[key] | target.desc_words[key]:
                    document_frequency[w] = document_frequency.get(w, 0) + 1
        n = max(1, len(target.columns))
        target.idf = {w: math.log(1.0 + n / df) for w, df in document_frequency.items()}
        target.bindings = evidence_bindings(evidence, schema)
        target.bound = {(t, c) for t, c, _op, _v in target.bindings}
        mentioned = _alnum(f"{question} {evidence}")
        q = target.question_words
        for table in schema.tables:
            own_words = set(words(table.name))
            for column in table.columns:
                key = (table.name, column.name)
                # a column's own words, minus words that only repeat its table's name (orders.order_id: "id")
                score = 1.0 * sum(target.idf.get(w, 1.0) for w in (target.name_words[key] - own_words) & q)
                score += 0.3 * sum(target.idf.get(w, 1.0) for w in target.desc_words[key] & q)
                if len(_alnum(column.name)) >= 4 and _alnum(column.name) in mentioned:
                    score += 3.0
                target.relevance[key] = score
            own = 2.0 * sum(target.idf.get(w, 1.0) for w in own_words & q)
            best = sorted((target.relevance[(table.name, c.name)] for c in table.columns), reverse=True)[:3]
            bound_here = 10.0 if any(t == table.name for t, _c in target.bound) else 0.0
            target.table_relevance[table.name] = own + sum(best) + bound_here
        return target


_BINDING = re.compile(
    r"(?P<lhs>[^=<>!;]{1,80}?)\s*(?P<op>=|!=|<>|>=|<=|>|<)\s*(?P<val>'(?:[^']|'')*'|\"[^\"]*\"|-?\d+(?:\.\d+)?)")


def evidence_bindings(evidence: str, schema: C.Schema) -> list[tuple]:
    """[(table, column, op, value)] for evidence phrases like "refers to movie title = 'Heat'" or "year = 2010".

    The column is the longest run of the last one to five words before the operator that names a target column,
    ignoring case, spaces, underscores and backticks."""
    by_alnum: dict[str, list] = {}
    tables_by_alnum = {_alnum(t.name): t.name for t in schema.tables}
    for table in schema.tables:
        for column in table.columns:
            by_alnum.setdefault(_alnum(column.name), []).append((table.name, column.name))
    out = []
    for match in _BINDING.finditer(evidence or ""):
        lhs = match.group("lhs")
        qualifier = None
        if "." in lhs:  # a qualified name: T.column or table.column
            head, lhs = lhs.rsplit(".", 1)
            last = re.findall(r"[\w`\"\[\]]+", head)
            qualifier = tables_by_alnum.get(_alnum(last[-1])) if last else None
        lhs = re.split(r"\brefers? to\b|\bmeans?\b|\bwhere\b|\bwhen\b|[,(:]", lhs, flags=re.IGNORECASE)[-1]
        parts = re.findall(r"[`\"\[]?[\w#%-]+[`\"\]]?", lhs)
        hits = []
        for size in range(min(5, len(parts)), 0, -1):
            hits = by_alnum.get(_alnum("".join(parts[-size:])), [])
            if hits:
                break
        if qualifier is not None:
            hits = [h for h in hits if h[0] == qualifier] or hits
        if not hits:
            continue
        raw = match.group("val")
        if raw[0] in "'\"":
            value = raw[1:-1].replace("''", "'") if raw[0] == "'" else raw[1:-1]
        else:
            value = float(raw) if "." in raw else int(raw)
        out.extend((table, column, match.group("op"), value) for table, column in hits)
    return out


def question_values(question: str, evidence: str) -> dict:
    """Candidate literals from the request: quoted strings, numbers, capitalized phrases (in order of appearance)."""
    quoted = [a or b for a, b in re.findall(r"'([^']+)'|\"([^\"]+)\"", f"{question} {evidence}")]
    numbers = []
    for raw in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])", question) + \
            re.findall(r"(?<![\w.'\"])-?\d+(?:\.\d+)?(?![\w'\"])", evidence or ""):
        numbers.append(float(raw) if "." in raw else int(raw))
    phrases = []
    tokens_ = re.findall(r"[A-Za-z0-9][\w'&.-]*|[,?;:!]", question)
    current: list[str] = []
    for position, token in enumerate(tokens_):
        capital = token[0].isupper() and not (position == 0 and token.lower() in _QUESTION_WORDS)
        if capital or (current and token.lower() in {"of", "and", "the", "de", "la", "du", "van", "von", "&"}):
            current.append(token)
            continue
        while current and not current[-1][0].isupper():
            current.pop()
        if current:
            phrases.append(" ".join(current))
        current = []
    while current and not current[-1][0].isupper():
        current.pop()
    if current:
        phrases.append(" ".join(current))
    phrases = [p.strip(".") for p in phrases if p.lower() not in _QUESTION_WORDS]
    return {"quoted": quoted, "numbers": numbers, "phrases": phrases}


# --- the template's side ----------------------------------------------------------------------------------------------
@dataclass
class Template:
    example: C.RetrievedExample
    tree: exp.Expression
    tables: list            # source table names in order of first appearance
    columns: dict           # id(Column node) -> (source table, source column)
    used: dict              # source table -> [source columns] in order of first use
    joins: list             # [((table, column), (table, column))] equalities between two source tables
    filters: set = field(default_factory=set)  # {(table, column)} compared with a literal


def _scopes(node: exp.Expression) -> list:
    """The SELECTs enclosing a node, innermost first (a subquery's columns resolve in its own FROM first)."""
    chain, select = [], node.find_ancestor(exp.Select)
    while select is not None:
        chain.append(select)
        select = select.find_ancestor(exp.Select)
    return chain


def parse_template(example: C.RetrievedExample, source: C.Schema) -> Template:
    tree = sqlglot.parse_one(example.sql, read="sqlite")
    real = {t.name.lower(): t for t in source.tables}
    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    table_nodes, tables = [], []
    for node in tree.find_all(exp.Table):
        name = node.name.lower()
        if name in cte_names or name not in real:
            continue
        table_nodes.append(node)
        if real[name].name not in tables:
            tables.append(real[name].name)

    def has(table: str, name: str) -> str | None:
        return next((c.name for c in real[table.lower()].columns if c.name.lower() == name.lower()), None)

    def in_scope(select) -> list:
        return [n for n in table_nodes if n.find_ancestor(exp.Select) is select]

    def resolve(node: exp.Expression, qualifier: str, name: str):
        """(table, column) for a name: in the innermost scope that has it, then outer scopes, then anywhere."""
        for select in _scopes(node) + [None]:
            nodes = in_scope(select) if select is not None else table_nodes
            if qualifier:
                nodes = [n for n in nodes if n.alias_or_name.lower() == qualifier.lower()
                         or (not n.alias and n.name.lower() == qualifier.lower())]
            for n in nodes:
                table = real[n.name.lower()].name
                column = has(table, name)
                if column is not None:
                    return table, column
        return None

    columns, used = {}, {t: [] for t in tables}

    def use(key):
        if key[1] not in used[key[0]]:
            used[key[0]].append(key[1])

    for node in tree.find_all(exp.Column):
        if isinstance(node.this, exp.Star):
            continue
        key = resolve(node, node.table, node.name)
        if key is not None:
            columns[id(node)] = key
            use(key)
    joins = []
    for eq in tree.find_all(exp.EQ):
        left, right = eq.this, eq.expression
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            a, b = columns.get(id(left)), columns.get(id(right))
            if a and b and a[0] != b[0]:
                joins.append((a, b))
    for join in tree.find_all(exp.Join):  # JOIN t USING (x): an equality on x between t and an earlier table
        right = join.this
        if not isinstance(right, exp.Table) or right.name.lower() not in real:
            continue
        right_table = real[right.name.lower()].name
        earlier = [n for n in in_scope(join.find_ancestor(exp.Select)) if n is not right]
        for identifier in join.args.get("using") or []:
            right_column = has(right_table, identifier.name)
            left = next(((real[n.name.lower()].name, has(real[n.name.lower()].name, identifier.name))
                         for n in earlier if has(real[n.name.lower()].name, identifier.name)), None)
            if right_column and left:
                use((right_table, right_column))
                use(left)
                joins.append((left, (right_table, right_column)))
    filters = {key for _literal, key, _comparison in _slots(tree, columns)}
    return Template(example=example, tree=tree, tables=tables, columns=columns, used=used, joins=joins,
                    filters=filters)


# --- mapping ------------------------------------------------------------------------------------------------------------
def _overlap(a: set, b: set, idf: dict) -> float:
    return sum(idf.get(w, 1.0) for w in a & b)


def _column_score(target: Target, source_column: C.SchemaColumn, key: tuple, is_filter: bool = False) -> float:
    column = target.columns[key]
    src_name, src_desc = set(words(source_column.name)), set(words(source_column.description))
    score = 3.0 * _overlap(src_name, target.name_words[key], target.idf)
    score += 0.5 * _overlap(src_name | src_desc, target.desc_words[key], target.idf)
    score += target.relevance[key]
    if is_filter and key in target.bound:
        score += 20.0
    if _is_numeric_type(source_column.type) == _is_numeric_type(column.type):
        score += 0.5
    if source_column.primary_key and column.primary_key:
        score += 1.0
    return score


def _fk_links(target: Target, a: str, b: str) -> list[tuple]:
    """[(column of a, column of b)] for foreign keys between target tables a and b, either direction."""
    links = []
    for from_col, ref, to_col in target.tables[a.lower()].foreign_keys:
        if ref.lower() == b.lower() and to_col:
            links.append((from_col, to_col))
    for from_col, ref, to_col in target.tables[b.lower()].foreign_keys:
        if ref.lower() == a.lower() and to_col:
            links.append((to_col, from_col))
    return links


def table_assignments(template: Template, target: Target, source: C.Schema, limit: int) -> list[dict]:
    """Up to `limit` injective maps {source table: target table}, best first (ties by target names)."""
    src = {t.name: t for t in source.tables}
    names = [t.name for t in target.schema.tables]
    if not template.tables or len(names) < len(template.tables):
        return []
    fit = {}
    for s in template.tables:
        for t in names:
            table = target.tables[t.lower()]
            best = 0.0
            for column_name in template.used[s]:
                source_column = next(c for c in src[s].columns if c.name == column_name)
                is_filter = (s, column_name) in template.filters
                best += max((_column_score(target, source_column, (t, c.name), is_filter)
                             for c in table.columns), default=0.0)
            fit[(s, t)] = target.table_relevance[t] + best + 1.5 * _overlap(set(words(s)), set(words(t)), target.idf)

    def total(assignment: dict) -> float:
        value = sum(fit[(s, t)] for s, t in assignment.items())
        for (sa, _ca), (sb, _cb) in template.joins:
            if sa in assignment and sb in assignment:
                value += 5.0 if _fk_links(target, assignment[sa], assignment[sb]) else -5.0
        return value

    options = {s: sorted(names, key=lambda t: (-fit[(s, t)], t))[:TABLE_CANDIDATES] for s in template.tables}
    scored = []
    if len(template.tables) <= EXACT_SEARCH_TABLES:
        for combo in itertools.product(*(options[s] for s in template.tables)):
            if len(set(combo)) == len(combo):
                assignment = dict(zip(template.tables, combo))
                scored.append((-total(assignment), combo, assignment))
    else:
        assignment: dict = {}
        for s in template.tables:
            free = [t for t in names if t not in assignment.values()]
            assignment[s] = min(free, key=lambda t: (-total({**assignment, s: t}), t))
        scored.append((-total(assignment), tuple(assignment.values()), assignment))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [assignment for _neg, _combo, assignment in scored[:limit]]


def column_map(template: Template, target: Target, source: C.Schema, tables: dict) -> dict:
    """{(source table, source column): (target table, target column)}."""
    src = {t.name: t for t in source.tables}
    mapping: dict = {}
    for (sa, ca), (sb, cb) in template.joins:  # join columns follow a target foreign key when there is one
        ta, tb = tables[sa], tables[sb]
        links = _fk_links(target, ta, tb)
        if links and (sa, ca) not in mapping and (sb, cb) not in mapping:
            x, y = max(links, key=lambda l: (_overlap(set(words(ca)) | set(words(cb)), set(words(l[0])) |
                                                      set(words(l[1])), target.idf), l))
            mapping[(sa, ca)], mapping[(sb, cb)] = (ta, x), (tb, y)
    for s, t in tables.items():
        pending = [c for c in template.used[s] if (s, c) not in mapping]
        taken = {mapping[k][1] for k in mapping if k[0] == s}
        pairs = []
        for column_name in pending:
            source_column = next(c for c in src[s].columns if c.name == column_name)
            for c in target.tables[t.lower()].columns:
                pairs.append((-_column_score(target, source_column, (t, c.name), (s, column_name) in template.filters),
                              column_name, c.name))
        pairs.sort()
        for _neg, column_name, target_column in pairs:
            if (s, column_name) not in mapping and target_column not in taken:
                mapping[(s, column_name)] = (t, target_column)
                taken.add(target_column)
        for column_name in pending:  # more source columns than target columns: reuse the best
            if (s, column_name) not in mapping:
                best = min((p for p in pairs if p[1] == column_name), default=None)
                if best is not None:
                    mapping[(s, column_name)] = (t, best[2])
    return mapping


# --- building a candidate -------------------------------------------------------------------------------------------------
def _identifier(name: str) -> exp.Identifier:
    plain = re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name) and name.upper() not in schema_mod.SQLITE_KEYWORDS
    return exp.to_identifier(name, quoted=not plain)


def _slots(tree: exp.Expression, columns: dict) -> list[tuple]:
    """[(literal node, (source table, source column), comparison node)] for literals compared with one column."""
    out = []
    for comparison in tree.find_all(*_COMPARISONS):
        if isinstance(comparison, exp.In):
            sides, literals = [comparison.this], [e for e in comparison.expressions if isinstance(e, exp.Literal)]
        elif isinstance(comparison, exp.Between):
            sides = [comparison.this]
            literals = [e for e in (comparison.args.get("low"), comparison.args.get("high"))
                        if isinstance(e, exp.Literal)]
        else:
            left, right = comparison.this, comparison.expression
            sides, literals = ([left], [right]) if isinstance(right, exp.Literal) else ([right], [left])
            literals = [l for l in literals if isinstance(l, exp.Literal)]
        referenced = [columns[id(c)] for side in sides for c in side.find_all(exp.Column) if id(c) in columns]
        if len(set(referenced)) == 1:
            for literal in literals:
                out.append((literal, referenced[0], comparison))
    return out


def _literal(value, like_template: str | None = None) -> exp.Literal:
    if isinstance(value, str):
        if like_template is not None:
            prefix = "%" if like_template.startswith("%") else ""
            suffix = "%" if like_template.endswith("%") and len(like_template) > 1 else ""
            value = f"{prefix}{value}{suffix}"
        return exp.Literal.string(value)
    return exp.Literal.number(value)


def fill_literals(tree: exp.Expression, column_keys: dict, target: Target, values: dict, lookup) -> tuple:
    """Choose a value for each literal compared with one column (see the module docstring).

    `column_keys` maps id(Column node) in `tree` to its (target table, target column). Returns (fills, ungrounded)."""
    used_quoted, used_numbers, used_bindings = set(), set(), set()
    fills, ungrounded = [], False
    for literal, target_key, comparison in _slots(tree, column_keys):
        is_string = literal.is_string
        like = literal.this if isinstance(comparison, (exp.Like, exp.ILike)) and is_string else None
        numeric_text = is_string and re.fullmatch(r"-?\d+(\.\d+)?", literal.this or "") is not None
        value, origin = None, None
        for index, (table, column, _op, bound) in enumerate(target.bindings):
            if index not in used_bindings and (table, column) == target_key:
                used_bindings.add(index)
                value, origin = (str(bound) if is_string else bound), "evidence"
                break
        if origin is None and is_string and not numeric_text:
            for index, text in enumerate(values["quoted"]):
                if index not in used_quoted:
                    used_quoted.add(index)
                    value, origin = text, "quoted"
                    break
        if origin is None and (not is_string or numeric_text):
            for index, number in enumerate(values["numbers"]):
                if index not in used_numbers:
                    used_numbers.add(index)
                    value, origin = (str(number) if is_string else number), "number"
                    break
        if origin is None and is_string and not numeric_text:
            for phrase in values["phrases"]:
                found = lookup(target_key[0], target_key[1], phrase)
                if found is not None:
                    value, origin = found, "lookup"
                    break
        before = literal.sql(dialect="sqlite")
        if origin is None:
            ungrounded = True
            fills.append({"literal": before, "column": list(target_key), "value": None, "source": "template"})
            continue
        new = _literal(value, like)
        fills.append({"literal": before, "column": list(target_key), "value": new.sql(dialect="sqlite"),
                      "source": origin})
        literal.replace(new)
    return fills, ungrounded


def render(template: Template, tables: dict, columns: dict) -> tuple[exp.Expression, dict]:
    """A copy of the template with source tables and columns renamed to their targets, and
    {id(Column node in the copy): (target table, target column)} for the columns that were mapped.

    Also renamed: a column qualified by an unaliased table's own name (`users.city`); a reference to a derived
    table's output (`T.city` for `(SELECT T1.city ...) AS T`), as that output was renamed; and the names in
    `JOIN ... USING (x)`, which become `ON a.x1 = b.x2` when the two sides map to different names."""
    tree = template.tree.copy()
    renamed_tables = {s.lower(): t for s, t in tables.items()}
    unaliased, keys = {}, {}
    pairs = list(zip(template.tree.find_all(exp.Table, exp.Column), tree.find_all(exp.Table, exp.Column)))
    for _old, new in pairs:
        if isinstance(new, exp.Table):
            target_table = renamed_tables.get(new.name.lower())
            if target_table is not None:
                if not new.alias:
                    unaliased[new.name.lower()] = target_table
                new.set("this", _identifier(target_table))
    for old, new in pairs:
        if isinstance(new, exp.Column):
            if new.table and new.table.lower() in unaliased:
                new.set("table", _identifier(unaliased[new.table.lower()]))
            if id(old) in template.columns:
                mapped = columns.get(template.columns[id(old)])
                if mapped is not None:
                    new.set("this", _identifier(mapped[1]))
                    keys[id(new)] = mapped
                    if not new.table:  # unambiguous in the template; keep it so when its scope joins tables
                        scope = new.find_ancestor(exp.Select)
                        local = [t for t in tree.find_all(exp.Table) if t.find_ancestor(exp.Select) is scope]
                        owners = [t for t in local if t.name.lower() == mapped[0].lower()]
                        if len(local) > 1 and len(owners) == 1:
                            new.set("table", _identifier(owners[0].alias_or_name))
    outputs = {}  # derived-table alias -> {old output name: new output name}
    for old_sub, new_sub in zip(template.tree.find_all(exp.Subquery), tree.find_all(exp.Subquery)):
        if new_sub.alias and isinstance(new_sub.this, exp.Select):
            renames = {}
            for old_p, new_p in zip(old_sub.this.expressions, new_sub.this.expressions):
                if isinstance(old_p, exp.Column) and isinstance(new_p, exp.Column) and old_p.name != new_p.name:
                    renames[old_p.name.lower()] = new_p.name
            if renames:
                outputs[new_sub.alias.lower()] = renames
    for column in tree.find_all(exp.Column):
        renames = outputs.get((column.table or "").lower())
        if renames and column.name.lower() in renames and id(column) not in keys:
            column.set("this", _identifier(renames[column.name.lower()]))
    for old_join, new_join in zip(template.tree.find_all(exp.Join), tree.find_all(exp.Join)):
        using = new_join.args.get("using") or []
        if not using or not isinstance(old_join.this, exp.Table):
            continue
        right_source = next((s for s in tables if s.lower() == old_join.this.name.lower()), None)
        conditions, rewritten = [], False
        for identifier in using:
            pair = next(((a, b) for a, b in template.joins if b[0] == right_source
                         and b[1].lower() == identifier.name.lower()), None)
            if pair is None or columns.get(pair[0]) is None or columns.get(pair[1]) is None:
                conditions.append(None)
                continue
            (left_table, left_column), (_rt, right_column) = columns[pair[0]], columns[pair[1]]
            left_alias = next((n.alias_or_name for n in tree.find_all(exp.Table) if n.name == left_table), left_table)
            conditions.append((left_alias, left_column, new_join.this.alias_or_name, right_column))
            rewritten = rewritten or left_column != right_column
        if rewritten and all(conditions):
            on = None
            for left_alias, left_column, right_alias, right_column in conditions:
                eq = exp.EQ(this=exp.column(_identifier(left_column), table=_identifier(left_alias)),
                            expression=exp.column(_identifier(right_column), table=_identifier(right_alias)))
                on = eq if on is None else exp.and_(on, eq)
            new_join.set("using", None)
            new_join.set("on", on)
        else:
            for identifier, condition in zip(using, conditions):
                if condition is not None:
                    identifier.replace(_identifier(condition[3]))
    return tree, keys


# --- the arm ------------------------------------------------------------------------------------------------------------
def run(task: C.Task, snapshot: C.Snapshot, *, examples=None, target_schema: C.Schema | None = None,
        source_schema=None, max_tool_calls: int | None = None) -> dict:
    """Answer one request with L. Returns `final` ({"kind", "sql", "reason"}) and every attempt."""
    from . import retrieval

    started = time.perf_counter()
    budget = C.Limits().max_tool_calls if max_tool_calls is None else max_tool_calls
    examples = list(retrieval.retrieve(task) if examples is None else examples)
    target_schema = target_schema or schema_mod.load_schema(task.db_id, snapshot=snapshot)
    source_schema = source_schema or (lambda db_id: schema_mod.load_schema(db_id))
    target = Target.build(target_schema, task.question, task.evidence)
    values = question_values(task.question, task.evidence)
    calls: list[dict] = []
    attempts: list[dict] = []
    chosen = fallback = None

    def call(fn, *args):
        result = fn(snapshot, *args)
        calls.append(result)
        return result

    for rank, example in enumerate(examples, start=1):
        if chosen is not None or len(calls) >= budget:
            break
        record = {"rank": rank, "template_task_id": example.task_id, "template_db_id": example.db_id,
                  "retrieval_score": example.score, "status": "", "candidates": []}
        attempts.append(record)
        try:
            source = source_schema(example.db_id)
            template = parse_template(example, source)
        except Exception as error:  # unparsable template SQL or unknown source database
            record["status"] = f"template_unusable: {type(error).__name__}: {str(error)[:200]}"
            continue
        assignments = table_assignments(template, target, source, ASSIGNMENTS_PER_TEMPLATE)
        if not assignments:
            record["status"] = "no_table_assignment"
            continue
        lookups_left = [LOOKUPS_PER_TEMPLATE]

        def lookup(table, column, phrase):
            if lookups_left[0] <= 0 or len(calls) >= budget - 1:  # always keep one call for the execution
                return None
            lookups_left[0] -= 1
            result = call(tools.lookup_values, table, column, phrase, C.Limits().lookup_max_values)
            if result["status"] != "ok" or not result["values"]:
                return None
            exact = [v for v in result["values"] if isinstance(v, str) and v.lower() == phrase.lower()]
            return exact[0] if exact else result["values"][0]

        record["status"] = "executed"
        for tables in assignments:
            if len(calls) >= budget:
                record["status"] = "budget_exhausted"
                break
            columns = column_map(template, target, source, tables)
            tree, keys = render(template, tables, columns)
            fills, ungrounded = fill_literals(tree, keys, target, values, lookup)
            sql = tree.sql(dialect="sqlite")
            entry = {"tables": dict(tables), "columns": {f"{s}.{c}": f"{t}.{d}" for (s, c), (t, d) in columns.items()},
                     "fills": fills, "ungrounded": ungrounded, "sql": sql}
            record["candidates"].append(entry)
            if len(calls) >= budget:
                entry["status"] = record["status"] = "budget_exhausted"
                break
            result = call(tools.execute_select, sql, C.Limits().preview_rows)
            entry.update(status=result["status"], row_count=result["row_count"], error=result["error"])
            if result["status"] == "ok" and result["row_count"]:
                chosen = sql
                break
            if result["status"] == "ok" and fallback is None:
                fallback = sql
    if chosen is not None:
        final = {"kind": "sql", "sql": chosen, "reason": "first candidate that ran with at least one row"}
    elif fallback is not None:
        final = {"kind": "sql", "sql": fallback, "reason": "no candidate returned rows; first that ran"}
    else:
        statuses = [a["status"] for a in attempts] or ["no retrieved examples"]
        final = {"kind": "abstain", "sql": None,
                 "reason": "no candidate ran: " + "; ".join(statuses)[:500] +
                           (" (tool budget exhausted)" if len(calls) >= budget else "")}
    return {"arm": "L", "version": L_VERSION, "task_id": task.task_id, "db_id": task.db_id,
            "snapshot_sha256": snapshot.sha256, "retrieved": [e.task_id for e in examples], "final": final,
            "attempts": attempts, "tool_calls": calls, "tool_call_count": len(calls), "tool_call_budget": budget,
            "elapsed_seconds": round(time.perf_counter() - started, 6)}
