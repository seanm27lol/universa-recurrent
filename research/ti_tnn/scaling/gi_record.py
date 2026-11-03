"""The grounding record and how a controller consumes it (grounding-interface study, development).

**Example.** For "Identify by their id all the apps that belong to the game-stress reliever category." a grounding
step may produce

    {"tables": ["label_categories", "app_labels"],
     "columns": [{"table": "label_categories", "column": "category"}, {"table": "app_labels", "column": "app_id"},
                 {"table": "label_categories", "column": "label_id"}],
     "literals": [{"span": "game-stress reliever", "value": "game-stress reliever",
                   "table": "label_categories", "column": "category"}],
     "status": "ok"}

**What a record may hold.** Only:
- ranked table identifiers;
- ranked column identifiers;
- candidate literal bindings (a question or evidence span, a proposed value, an associated column);
- a status, "ok" or "unresolved".

No SQL, plan or prose. Identifiers are the schema's own table and column names, matched without regard to case and
without one layer of SQL quoting (`clean_name`).

**Parsing.** The last fenced block of the reply, or the whole reply, is read as JSON; if that fails, again with invalid
backslash escapes removed, and then the outermost braces alone.

**Validation.** A record is valid when it parses as one JSON object of exactly these keys and types, every table
and column reference resolves to an item of the request's schema, and the output that produced it was not
truncated. Anything else is invalid, and the request falls back to the original policy (G0). Literal bindings are
validated one by one (the span and the value must occur in the question or evidence, the column must resolve). An
invalid binding is dropped and counted; it never invalidates the record, because literals are not consumed in this
version.

**Consumption: what the controller reads.** Only the ranked tables and columns, through a fixed prior added to the
table- and column-pointer logits at every decoding step: a candidate at rank r (1-based, its first occurrence)
gains ln 2 / r, and an unlisted item gains 0. At most `MAX_TABLES` tables and `MAX_COLUMNS` columns are read, in
order. Every legality mask stays as it is, and operator and family decisions are untouched. The literals and the
status are recorded and scored, not consumed.
"""
from __future__ import annotations

import contextlib
import json
import math
import re
from dataclasses import asdict, dataclass, field

RECORD_VERSION = "gi-record-v1"
KEYS = ("tables", "columns", "literals", "status")
STATUSES = ("ok", "unresolved")
MAX_TABLES = 8
MAX_COLUMNS = 16


def rank_bias(rank: int) -> float:
    """The fixed ranking prior: ln 2 / r for 1-based rank r."""
    return math.log(2.0) / rank


@dataclass
class Grounding:
    """A validated record, resolved to item indexes of one request's schema."""
    valid: bool
    reason: "str | None" = None
    tables: list = field(default_factory=list)       # item indexes, ranked, first occurrences
    columns: list = field(default_factory=list)
    literals: list = field(default_factory=list)     # valid bindings: {span, value, item}
    invalid_literals: list = field(default_factory=list)
    status: "str | None" = None
    listed: dict = field(default_factory=dict)        # {"tables": n, "columns": n} as the record listed them

    def as_dict(self) -> dict:
        return asdict(self)

    def biases(self, n_items: int) -> "list[float] | None":
        """The additive prior per item (None for an invalid record: the original policy)."""
        if not self.valid:
            return None
        out = [0.0] * n_items
        for ranked, cap in ((self.tables, MAX_TABLES), (self.columns, MAX_COLUMNS)):
            for r, i in enumerate(ranked[:cap], 1):
                out[i] = rank_bias(r)
        return out


QUOTES = (("`", "`"), ('"', '"'), ("[", "]"), ("'", "'"))


def clean_name(name: str) -> str:
    """An identifier as the model wrote it, without surrounding whitespace or one layer of SQL quoting (the schema
    text quotes some names; a model may copy the quotes)."""
    n = name.strip()
    for a, b in QUOTES:
        if len(n) >= 2 and n.startswith(a) and n.endswith(b):
            return n[1:-1].strip()
    return n


def _lookup(items):
    tables = {it.table.lower(): i for i, it in enumerate(items.items) if it.kind == "table"}
    columns = {(it.table.lower(), it.column.lower()): i for i, it in enumerate(items.items) if it.kind == "column"}
    return tables, columns


def _occurs(text: str, needle) -> bool:
    if needle is None:
        return False
    return str(needle).strip().lower() in text.lower() if str(needle).strip() else False


def resolve(obj, items, question: str, evidence: str, truncated: bool = False) -> Grounding:
    """Validate a parsed record against a request's schema items (`actions.SchemaItems`)."""
    if truncated:
        return Grounding(False, "truncated output")
    if not isinstance(obj, dict):
        return Grounding(False, "not a JSON object")
    if set(obj) != set(KEYS):
        return Grounding(False, f"keys {sorted(obj)} != {sorted(KEYS)}")
    if not isinstance(obj["tables"], list) or not all(isinstance(t, str) for t in obj["tables"]):
        return Grounding(False, "tables is not a list of names")
    if not isinstance(obj["columns"], list) or not all(
            isinstance(c, dict) and set(c) == {"table", "column"} and all(isinstance(v, str) for v in c.values())
            for c in obj["columns"]):
        return Grounding(False, "columns is not a list of {table, column}")
    if not isinstance(obj["literals"], list) or not all(isinstance(x, dict) for x in obj["literals"]):
        return Grounding(False, "literals is not a list of objects")
    if obj["status"] not in STATUSES:
        return Grounding(False, f"status {obj['status']!r}")
    tables, columns = _lookup(items)
    t_out, c_out = [], []
    for t in obj["tables"]:
        i = tables.get(clean_name(t).lower())
        if i is None:
            return Grounding(False, f"unknown table {t!r}")
        if i not in t_out:
            t_out.append(i)
    for c in obj["columns"]:
        i = columns.get((clean_name(c["table"]).lower(), clean_name(c["column"]).lower()))
        if i is None:
            return Grounding(False, f"unknown column {c['table']!r}.{c['column']!r}")
        if i not in c_out:
            c_out.append(i)
    text = f"{question}\n{evidence or ''}"
    lits, bad = [], []
    for x in obj["literals"]:
        col = columns.get((clean_name(str(x.get("table", ""))).lower(), clean_name(str(x.get("column", ""))).lower()))
        ok = set(x) == {"span", "value", "table", "column"} and _occurs(text, x.get("span")) and \
            _occurs(text, x.get("value")) and col is not None
        (lits if ok else bad).append({"span": x.get("span"), "value": x.get("value"), "item": col,
                                      **({} if ok else {"raw": x})})
    return Grounding(True, None, t_out, c_out, lits, bad, obj["status"],
                     {"tables": len(obj["tables"]), "columns": len(obj["columns"])})


FENCED = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.S | re.I)
INVALID_ESCAPE = re.compile(r'\\(?![\\"/bfnrtu])')


def parse(text: str):
    """The record's JSON object from a model reply: a fenced block if there is one, else the whole reply. Returns
    (object or None, reason)."""
    blocks = FENCED.findall(text or "")
    body = (blocks[-1] if blocks else (text or "")).strip()
    first_error = None
    for candidate in (body, INVALID_ESCAPE.sub("", body)):  # then with invalid backslash escapes (e.g. \\') removed
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError as error:
            first_error = first_error or error
            start, end = candidate.find("{"), candidate.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(candidate[start:end + 1]), None
                except json.JSONDecodeError:
                    pass
    return None, f"not JSON: {first_error.msg}"


@contextlib.contextmanager
def pointer_prior(model, biases: "list[float] | None"):
    """Decode with `biases` added to the table- and column-pointer scores of every step (no-op for None). The
    trunk's own `pointer_scores` is wrapped on the instance and restored afterwards."""
    if biases is None:
        yield
        return
    import torch
    trunk = model.trunk
    original = trunk.pointer_scores

    def biased(mem, queries, start_index=None, relations=None):
        items, start, end = original(mem, queries, start_index, relations)
        b = torch.tensor(biases, dtype=items.dtype, device=items.device)
        if b.shape[0] != items.shape[-1]:
            raise ValueError(f"{b.shape[0]} biases for {items.shape[-1]} items")
        return items + b, start, end

    trunk.pointer_scores = biased
    try:
        yield
    finally:
        del trunk.pointer_scores
        assert trunk.pointer_scores == original


def oracle_record(gold_plan, items) -> dict:
    """The evaluator-information record of a gold program: the tables and columns every SELECT of it uses (columns
    resolved in their own SELECT's scope, `diagnose.select_features`), every one at rank 1. The set is unordered; the
    first FROM table is not singled out. Never used outside the marked oracle diagnostic."""
    from . import diagnose as D
    tables, columns = set(), set()
    for sel, outer, _path in D.all_selects(gold_plan):
        f = D.select_features(sel, outer)
        tables |= {t.lower() for t in f["tables"]}
        columns |= {tuple(c.lower().split(".", 1)) for c in f["columns"]}
    t_idx, c_idx = _lookup(items)
    return {"tables": sorted(t_idx[t] for t in tables if t in t_idx),
            "columns": sorted(c_idx[c] for c in columns if c in c_idx)}


def oracle_biases(oracle: dict, n_items: int) -> list:
    """Every gold table and column at rank 1: ln 2 each, unordered."""
    out = [0.0] * n_items
    for i in oracle["tables"] + oracle["columns"]:
        out[i] = rank_bias(1)
    return out
