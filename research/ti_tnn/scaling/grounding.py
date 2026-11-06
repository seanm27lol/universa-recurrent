"""Relational grounding (grounding v2, performance study v2): the schema's relations, given to the controller.

**Why.** The pilot's diagnosis (`diagnose.py`, CPU only, on the 64 historical development questions and a fixed
training-pool sample) found that the dominant failure on unseen databases is grounding: under teacher forcing the
first FROM table is wrong at 49% of steps (17% on training questions), the joined table at 52% (16%), SELECT columns
at 42% (10%) and WHERE columns at 33% (10%). The predicted table set equals the gold one on only 16-21 of the 64, and
when it does the join keys are almost always right. Two structural reasons:
- the frozen backbone reads causally, and a table's name comes before its columns in the schema text, so a table's
  state cannot see its own columns; under teacher forcing the first-table pointer is right 76% of the time when the
  table and one of the gold query's columns of it are both named in the question or evidence, and 52% when only the
  column is;
- nothing in the v1 controller represents the declared keys: which column references which, which tables a foreign
  key links, which columns are primary keys. The 42 pilot outputs with a same-reference join (`T1.x = T1.x`) are the
  visible symptom: in 40 of them the table set differs from the gold program's (in 24 the joined table is one the
  gold program does not use), and all six scorer timeouts are among them.

**What grounding v2 adds** (identical in T4 and C4_params; it is part of the shared trunk):
1. **Item relations.** A relation-aware encoder (`controller.SchemaEncoder`) lets every schema item attend to every
   other item and to the question and evidence tokens, with a learned bias per head for each relation type below
   (Shaw et al., NAACL 2018; Wang et al., "RAT-SQL", ACL 2020). A table now sees its columns and the tables its keys
   link; a column sees its table, its siblings and its key partners.
2. **Item features:** primary key, foreign key (out, in), and whether the item's name is named in the question or
   the evidence (exactly, or by one of its words).
3. **Plan-relative pointer relations** (`actions.Machine.pointer_relations`): each table or column candidate carries
   bits saying how it relates to the partial plan (already a source; linked by a declared key to a source; shares a
   column name with a source; a key partner or namesake of the comparison's first operand; on the joined source;
   already referenced). The pointer adds `(W_r^T q) . bits` to its score, so the second side of a join key is
   scored against the first.
4. **The constraint:** a comparison in ON between one column reference and itself is not legal. It relates no two
   sources (`X = X` is `X IS NOT NULL`). A self-join (`T1.x = T2.x`) and any other ON predicate stay legal.

**Limits.** The relations are those the database declares. A join on an undeclared key (8% of the training pool's
gold join equalities) gets no key bit, only the shared-name bit when names agree; it stays fully representable.
"""
from __future__ import annotations

import re

from . import actions as A

RELATIONS = ("none", "self", "table->own column", "column->own table", "sibling column", "column references column",
             "column referenced by column", "table links to table", "column shares a name with column",
             "table->column of a linked table", "column->table linked to its table")
QUESTION_TOKEN, EVIDENCE_TOKEN = len(RELATIONS), len(RELATIONS) + 1   # relation ids of the item -> token keys
N_RELATIONS = len(RELATIONS) + 2
ITEM_FEATURES = ("table", "primary key", "foreign key (references)", "foreign key (referenced)",
                 "name in the question", "name in the evidence", "a name word in the question or evidence")
N_ITEM_FEATURES = len(ITEM_FEATURES)
_STOP = frozenset({"id", "the", "of", "and", "in", "to", "a", "is", "name", "no", "num", "date", "type", "code"})


def name_tokens(name: str) -> list:
    """Lower-case words of an identifier: camelCase, snake_case and spaces split."""
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return [t for t in re.split(r"[^A-Za-z0-9]+", name.lower()) if t]


def _mentions(name: str, text: str) -> bool:
    words = " ".join(name_tokens(text))
    phrase = " ".join(name_tokens(name))
    return bool(phrase) and (re.search(r"\b" + re.escape(phrase) + r"\b", words) is not None or
                             re.search(r"\b" + re.escape(name.lower()) + r"\b", text.lower()) is not None)


def item_relations(schema: A.SchemaItems) -> list:
    """[N][N] relation ids between schema items (RELATIONS order). Without keys, only membership relations."""
    items, keys = schema.items, schema.keys
    n = len(items)
    rel = [[0] * n for _ in range(n)]
    links = keys.table_links if keys is not None else {}
    for i in range(n):
        for j in range(n):
            a, b = items[i], items[j]
            if i == j:
                r = 1
            elif a.kind == "table" and b.kind == "column":
                r = 2 if b.table == a.table else 9 if b.table in links.get(a.table, ()) else 0
            elif a.kind == "column" and b.kind == "table":
                r = 3 if a.table == b.table else 10 if b.table in links.get(a.table, ()) else 0
            elif a.kind == "table" and b.kind == "table":
                r = 7 if b.table in links.get(a.table, ()) else 0
            else:
                if keys is not None and j in keys.fk_out.get(i, ()):
                    r = 5
                elif keys is not None and j in keys.fk_in.get(i, ()):
                    r = 6
                elif a.table == b.table:
                    r = 4
                elif a.column.lower() == b.column.lower():
                    r = 8
                else:
                    r = 0
            rel[i][j] = r
    return rel


def item_features(schema: A.SchemaItems, question: str, evidence: str) -> list:
    """[N][N_ITEM_FEATURES] 0/1 features of every item."""
    keys = schema.keys
    words = set(name_tokens(question + " " + evidence))
    out = []
    for i, it in enumerate(schema.items):
        name = it.table if it.kind == "table" else it.column
        partial = any(w in words for w in name_tokens(name) if w not in _STOP and len(w) >= 3)
        out.append([float(it.kind == "table"),
                    float(keys is not None and i in keys.pk),
                    float(keys is not None and bool(keys.fk_out.get(i))),
                    float(keys is not None and bool(keys.fk_in.get(i))),
                    float(_mentions(name, question)), float(_mentions(name, evidence)), float(partial)])
    return out


def question_evidence_tokens(token_types: list) -> tuple:
    """(indices, relation ids) of the question and evidence tokens, the keys items attend to besides items."""
    from .encode import REGIONS
    q, e = REGIONS.index("question"), REGIONS.index("evidence")
    idx = [k for k, t in enumerate(token_types) if t in (q, e)]
    return idx, [QUESTION_TOKEN if token_types[k] == q else EVIDENCE_TOKEN for k in idx]
