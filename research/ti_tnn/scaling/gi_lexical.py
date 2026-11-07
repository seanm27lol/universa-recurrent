"""Glex: cheap, deterministic lexical grounding (grounding-interface study, development). No model, and no evaluation
label: it reads only the question, the evidence and the schema's names and descriptions.

**Example.** For "Identify by their id all the apps that belong to the game-stress reliever category." with the
evidence "id refers to device_id":
- `label_categories.category` names "category" in the question (a whole-name mention, 3 points);
- `app_labels.app_id` shares the word "app" (1 point per shared word);
- the tables follow their best columns.

**Scores.** Words are the lower-case alphanumeric words of an identifier or a text, with camelCase and snake_case
split (`grounding.name_tokens`), minus a few stop words:
- **A column** scores 3 if its whole name is mentioned in the question or evidence (`grounding._mentions`), plus 1
  per distinct word of its name that occurs there, plus 0.5 per distinct description word (column description and
  meaning, at most 4) that occurs there.
- **A table** scores 3 if its whole name is mentioned, plus 1 per distinct word of its name that occurs, plus half its
  best column's score.

Candidates with a positive score are ranked by score, ties broken by schema order. At most `gi_record.MAX_TABLES`
tables and `gi_record.MAX_COLUMNS` columns are kept. Glex proposes no literal bindings.
"""
from __future__ import annotations

from . import gi_record as GR
from . import grounding as G

STOP = frozenset({"the", "of", "and", "in", "to", "a", "an", "is", "are", "was", "were", "be", "by", "for", "on",
                  "with",
                  "what", "which", "who", "whom", "how", "many", "much", "list", "name", "names", "give", "show",
                  "refers", "refer", "that", "this", "all", "from", "as", "at", "or", "it", "its", "their", "has",
                  "have", "do", "does", "please", "among", "id"})


def _words(text: str) -> set:
    return {w for w in G.name_tokens(text or "") if w not in STOP and len(w) > 1}


def _column_score(table: str, column, text: str, words: set) -> float:
    name = column.name
    score = 3.0 if G._mentions(name, text) else 0.0
    score += len(_words(name) & words)
    desc = _words(f"{column.description} {column.meaning}") - _words(name) - _words(table)
    score += 0.5 * min(4, len(desc & words))
    return score


def ground(question: str, evidence: str, schema, items) -> dict:
    """A grounding record (the `gi_record` format) for one request. `schema` is the `contracts.Schema` (names and
    descriptions); `items` the request's `actions.SchemaItems`, which decides what can be listed."""
    text = f"{question}\n{evidence or ''}"
    words = _words(text)
    listed = {(it.table.lower(), it.column.lower()) for it in items.items if it.kind == "column"}
    listed_tables = {it.table.lower() for it in items.items if it.kind == "table"}
    order = {(it.table.lower(), it.column.lower() if it.kind == "column" else None): i
             for i, it in enumerate(items.items)}
    col_scores, table_scores = [], []
    for t in schema.tables:
        if t.name.lower() not in listed_tables:
            continue
        best = 0.0
        for c in t.columns:
            if (t.name.lower(), c.name.lower()) not in listed:
                continue
            s = _column_score(t.name, c, text, words)
            best = max(best, s)
            if s > 0:
                col_scores.append((-s, order[(t.name.lower(), c.name.lower())], t.name, c.name))
        ts = (3.0 if G._mentions(t.name, text) else 0.0) + len(_words(t.name) & words) + 0.5 * best
        if ts > 0:
            table_scores.append((-ts, order[(t.name.lower(), None)], t.name))
    tables = [name for *_k, name in sorted(table_scores)][:GR.MAX_TABLES]
    columns = [{"table": t, "column": c} for *_k, t, c in sorted(col_scores)][:GR.MAX_COLUMNS]
    return {"tables": tables, "columns": columns, "literals": [], "status": "ok" if tables else "unresolved"}
