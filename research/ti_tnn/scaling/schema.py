"""The schema every arm receives: tables, columns, keys and BIRD's descriptions, as structure and as text.

**Example.** For `european_football_1`, `load_schema` reads the snapshot (`PRAGMA table_info`, `PRAGMA
foreign_key_list`, table order from `sqlite_master`), merges BIRD's `database_description/divisions.csv` and
`train_column_meaning.json`, and `serialize` turns it into lines like

    DATABASE european_football_1
    TABLE divisions
      division TEXT PK -- division id | meaning: stores text identifiers for different football divisions, such as 'G1', 'E2', 'F2'.
    TABLE matchs
      Div TEXT FK -> divisions.division -- Division Id | ...

**The text format (version `schema-text-v1`).** One line per item, each ending in a newline:
- `DATABASE <db_id>` first;
- then, in `sqlite_master` order, `TABLE <name>` followed by its columns in declaration order;
- a column line is two spaces, `<name>`, the declared type (omitted when empty), `PK` when the column is part of the
  primary key, `FK -> <table>.<column>` for each foreign key it belongs to, and, when there is any, ` -- ` then the
  descriptions joined by ` | `: the column description (dropped when it only repeats the name), `values: <value
  description>`, `meaning: <train_column_meaning summary>`. Phrases of the meaning that only restate the column's
  table and database ("In the 'y' table of the 'z' database, the 'x' column, an integer,") are removed
  (`compact_meaning`). All whitespace runs, including newlines, become one space.
- Names that are not plain identifiers, or that are SQLite keywords, are quoted with backticks (`` `Date received` ``),
  which SQLite accepts in queries.

Every table and column has a `SchemaItem` whose span `[start, end)` covers its line without the indentation, the
`TABLE ` keyword or the newline, so `text[start:end]` always begins with the (quoted) name.

**Budget.** `serialize(schema, budget_chars=n)` keeps every table and column. If the text is longer than `n`, it first
shortens the generated meaning summaries to the longest common length that fits (down to none), because they are the
most redundant part. If that is not enough, it cuts the remaining BIRD descriptions the same way. Cuts end in "...".
With no room for descriptions at all, it returns the bare structure even if that is still longer than `n`.

**Robustness.** BIRD's description CSVs are decoded as UTF-8 (with or without a byte-order mark), then cp1252, then
Latin-1. Tables are matched to CSV files by case-insensitive file name, and columns by `original_column_name`
(case-insensitive, then ignoring non-alphanumeric characters). `description_report(db_id)` lists whatever did not
match. Unmatched CSVs and rows are left out, never guessed.
"""
from __future__ import annotations

import csv
import difflib
import functools
import io
import json
import re
from pathlib import Path

from . import contracts as C

SCHEMA_TEXT_VERSION = "schema-text-v1"
# Suggested budget for the schema text in prompts. Measured with the Qwen3.5-4B tokenizer on the 69 training
# databases (about 4.2 characters per token): at 24,000 characters every schema is at most 5,940 tokens, and only 5
# change (works_cycles, hockey, legislator, professional_basketball, mondial_geo; only their meanings get shorter).
# Without a budget, the median is 1,570 tokens and the largest is works_cycles at 26,049.
PROMPT_BUDGET_CHARS = 24_000
COLUMN_MEANING = C.DATA_RAW / "hf" / "bird23_train_filtered.train_column_meaning.json"

# SQLite's keywords (https://www.sqlite.org/lang_keywords.html); a name equal to one of them is quoted.
SQLITE_KEYWORDS = frozenset("""
ABORT ACTION ADD AFTER ALL ALTER ALWAYS ANALYZE AND AS ASC ATTACH AUTOINCREMENT BEFORE BEGIN BETWEEN BY CASCADE CASE
CAST CHECK COLLATE COLUMN COMMIT CONFLICT CONSTRAINT CREATE CROSS CURRENT CURRENT_DATE CURRENT_TIME CURRENT_TIMESTAMP
DATABASE DEFAULT DEFERRABLE DEFERRED DELETE DESC DETACH DISTINCT DO DROP EACH ELSE END ESCAPE EXCEPT EXCLUDE EXCLUSIVE
EXISTS EXPLAIN FAIL FILTER FIRST FOLLOWING FOR FOREIGN FROM FULL GENERATED GLOB GROUP GROUPS HAVING IF IGNORE IMMEDIATE
IN INDEX INDEXED INITIALLY INNER INSERT INSTEAD INTERSECT INTO IS ISNULL JOIN KEY LAST LEFT LIKE LIMIT MATCH
MATERIALIZED NATURAL NO NOT NOTHING NOTNULL NULL NULLS OF OFFSET ON OR ORDER OTHERS OUTER OVER PARTITION PLAN PRAGMA
PRECEDING PRIMARY QUERY RAISE RANGE RECURSIVE REFERENCES REGEXP REINDEX RELEASE RENAME REPLACE RESTRICT RETURNING RIGHT
ROLLBACK ROW ROWS SAVEPOINT SELECT SET TABLE TEMP TEMPORARY THEN TIES TO TRANSACTION TRIGGER UNBOUNDED UNION UNIQUE
UPDATE USING VACUUM VALUES VIEW VIRTUAL WHEN WHERE WINDOW WITH WITHOUT
""".split())
_PLAIN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SPACE = re.compile(r"\s+")
_Q = r"(?:'[^']*'|\"[^\"]*\"|`[^`]*`|[\w.-]+)"  # a quoted or bare name
_TYPE = r"(?:\w+ )?"                                # an optional type word: "text", "integer", ...
_TYPED = r"(?:,? (?:an? \w+(?: type)?|of type \w+|\(\w+(?: type)?\)),?)?"  # ", an integer," / " (text type)"
# Openings that only restate where the column is (the schema text already says so), most specific first.
_MEANING_OPENINGS = [re.compile(pattern, re.IGNORECASE) for pattern in (
    rf"^In the {_Q} table of (?:the )?(?:{_Q} database|database {_Q}|{_Q} db), the {_Q} {_TYPE}column{_TYPED} ",
    rf"^In the {_Q} database, (?:within|in) the {_Q} table, the {_Q} {_TYPE}column{_TYPED} ",
    rf"^The {_Q} {_TYPE}column{_TYPED} (?:in|of|within) the {_Q} table"
    rf"(?: (?:of|in|within) the {_Q} database| \(db id: [^)]*\)|, identified by (?:db|database) id {_Q},)?"
    rf"{_TYPED} ",
    rf"^The {_Q} {_TYPE}column{_TYPED} (?!(?:in|of|within) )",
)]
# Location phrases anywhere in a meaning ("... in the 'Player' table of the 'soccer_2016' database").
_MEANING_LOCATIONS = re.compile(
    rf",? (?:in|of|within) the {_Q} table (?:of|in|within) (?:the )?(?:{_Q} database|database {_Q}|{_Q} db)"
    rf"|,? (?:in|of|within) the {_Q} database", re.IGNORECASE)
ELLIPSIS = "..."


def display_name(name: str) -> str:
    """How a table or column name is written in the schema text (and how it can be written in SQLite)."""
    if _PLAIN.match(name) and name.upper() not in SQLITE_KEYWORDS:
        return name
    return "`" + name.replace("`", "``") + "`"


def _clean(text) -> str:
    return _SPACE.sub(" ", str(text or "")).strip()


def _alnum(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.lower())


def compact_meaning(meaning: str) -> str:
    """A train_column_meaning summary without the phrases that only restate the column's table and database.

    Removes one opening such as "In the 'matchs' table of the 'european_football_1' database, the 'Div' column, an
    integer," and then any "in the 'x' table of the 'y' database" phrase; the rest is kept verbatim."""
    text = _clean(meaning)
    for pattern in _MEANING_OPENINGS:
        shorter = pattern.sub("", text, count=1)
        if shorter != text:
            text = shorter
            break
    text = _MEANING_LOCATIONS.sub("", text).strip()
    return text


# --- reading BIRD's descriptions --------------------------------------------------------------------------------------
def decode_bytes(data: bytes) -> tuple[str, str]:
    """(text, encoding) for a BIRD CSV: UTF-8 (BOM optional), then cp1252, then Latin-1 (which cannot fail)."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"


def read_description_csv(path: Path) -> tuple[list[dict], str]:
    """Rows of one database_description CSV as dicts with lower-cased, stripped header keys, and the encoding."""
    text, encoding = decode_bytes(path.read_bytes())
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    header_at = next((i for i, row in enumerate(rows) if any(cell.strip() for cell in row)), None)
    if header_at is None:
        return [], encoding
    header = [cell.strip().lower() for cell in rows[header_at]]
    out = []
    for row in rows[header_at + 1:]:
        if not any(cell.strip() for cell in row):
            continue
        out.append({header[i]: row[i] for i in range(min(len(header), len(row))) if header[i]})
    return out, encoding


@functools.lru_cache(maxsize=1)
def column_meanings(path: str = str(COLUMN_MEANING)) -> dict:
    """train_column_meaning.json keyed by lower-cased (db, table, column)."""
    p = Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for key, value in raw.items():
        parts = key.split("|")
        if len(parts) == 3 and isinstance(value, str):
            out[tuple(part.strip().lower() for part in parts)] = value
    return out


def _match_column(table_columns: list[str], name: str) -> str | None:
    by_lower = {c.lower(): c for c in table_columns}
    key = name.strip().lower()
    if key in by_lower:
        return by_lower[key]
    by_alnum = {}
    for c in table_columns:
        by_alnum.setdefault(_alnum(c), c)
    return by_alnum.get(_alnum(name)) if _alnum(name) else None


def _match_files(paths: list[Path], tables: list[str]) -> dict:
    """{csv path: (table, how)}. By file name (case-insensitive) first. A file left over then matches the one leftover
    table whose name ends its name ("googleplaystore.csv" -> playstore), or else the one leftover table whose name is
    at least 90% similar (difflib; "filed_for_bankruptcy.csv" -> filed_for_bankrupcy). Ambiguity means no match."""
    by_lower = {t.lower(): t for t in tables}
    out = {}
    for path in paths:
        table = by_lower.get(path.stem.strip().lower())
        if table is not None and table not in {t for t, _how in out.values()}:
            out[path] = (table, "name")
    for how in ("suffix", "similar"):
        free = [t for t in tables if t not in {t for t, _how in out.values()}]
        for path in paths:
            if path in out:
                continue
            stem = _alnum(path.stem)
            if how == "suffix":
                hits = [t for t in free if _alnum(t) and stem.endswith(_alnum(t))]
                longest = max((len(_alnum(t)) for t in hits), default=0)
                hits = [t for t in hits if len(_alnum(t)) == longest]
            else:
                hits = [t for t in free if difflib.SequenceMatcher(None, stem, _alnum(t)).ratio() >= 0.9]
            if len(hits) == 1:
                out[path] = (hits[0], how)
                free.remove(hits[0])
    return out


def _descriptions(db_dir: Path, tables: dict[str, list[str]]) -> tuple[dict, dict]:
    """{(table, column): (description, value_description)} and a report of what did not match."""
    found, report = {}, {"files": {}, "unmatched_files": [], "unmatched_rows": []}
    desc_dir = db_dir / "database_description"
    if not desc_dir.is_dir():
        report["missing_directory"] = True
        return found, report
    paths = [p for p in sorted(desc_dir.iterdir())
             if not p.name.startswith(".") and p.suffix.lower() == ".csv" and p.is_file()]
    table_of = _match_files(paths, list(tables))
    report["fallback_files"] = {p.name: t for p, (t, how) in table_of.items() if how != "name"}
    for path in paths:
        if path not in table_of:
            report["unmatched_files"].append(path.name)
            continue
        table = table_of[path][0]
        rows, encoding = read_description_csv(path)
        report["files"][path.name] = encoding
        for row in rows:
            original = (row.get("original_column_name") or "").strip()
            column = _match_column(tables[table], original) if original else None
            if column is None and row.get("column_name"):
                column = _match_column(tables[table], row["column_name"])
            if column is None:
                report["unmatched_rows"].append(f"{path.name}:{original or row.get('column_name', '')}")
                continue
            description = _clean(row.get("column_description"))
            expanded = _clean(row.get("column_name"))
            if not description and expanded and _alnum(expanded) != _alnum(column):
                description = expanded
            if (table, column) not in found:  # the first row for a column wins
                found[(table, column)] = (description, _clean(row.get("value_description")))
    return found, report


# --- the structured schema ----------------------------------------------------------------------------------------------
def _database_dir(db_id: str) -> Path:
    return C.DATABASES / db_id


def _snapshot_of(db_id: str) -> C.Snapshot:
    from . import data

    return data.snapshot(db_id)


@functools.lru_cache(maxsize=256)
def _load(db_id: str, path: str, sha256: str, desc_root: str, meaning_path: str) -> tuple[C.Schema, dict]:
    from .tools import connect_readonly, quote_identifier

    conn = connect_readonly(path)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' "
            "ORDER BY rowid")]
        info, fks = {}, {}
        for name in names:
            info[name] = conn.execute(f"PRAGMA table_info({quote_identifier(name)})").fetchall()
            fks[name] = conn.execute(f"PRAGMA foreign_key_list({quote_identifier(name)})").fetchall()
    finally:
        conn.close()
    columns_of = {name: [row[1] for row in info[name]] for name in names}
    pk_of = {name: [row[1] for row in sorted(info[name], key=lambda r: r[5]) if row[5]] for name in names}
    descriptions, report = _descriptions(Path(desc_root) / db_id, columns_of)
    meanings = column_meanings(meaning_path)
    tables = []
    for name in names:
        columns = []
        for _cid, column, ctype, _notnull, _default, pk in info[name]:
            description, value_description = descriptions.get((name, column), ("", ""))
            meaning = meanings.get((db_id.lower(), name.lower(), column.lower()), "")
            columns.append(C.SchemaColumn(table=name, name=column, type=ctype or "", primary_key=bool(pk),
                                          description=description, value_description=value_description,
                                          meaning=_clean(meaning)))
        foreign_keys = []
        by_lower = {n.lower(): n for n in names}
        for _id, seq, ref_table, from_col, to_col, *_ in sorted(fks[name], key=lambda r: (r[0], r[1])):
            ref_real = by_lower.get(str(ref_table).lower(), ref_table)
            if to_col is None:  # REFERENCES parent with no column list: the parent's primary key, by position
                parent_pk = pk_of.get(ref_real, [])
                to_col = parent_pk[seq] if seq < len(parent_pk) else ""
            foreign_keys.append((from_col, ref_real, to_col))
        tables.append(C.SchemaTable(name=name, columns=tuple(columns), foreign_keys=tuple(foreign_keys)))
    report["meanings_matched"] = sum(1 for t in tables for c in t.columns if c.meaning)
    report["columns"] = sum(len(t.columns) for t in tables)
    return C.Schema(db_id=db_id, snapshot_sha256=sha256, tables=tuple(tables)), report


def load_schema(db_id: str, snapshot: C.Snapshot | None = None, *, descriptions_root: Path | None = None,
                meaning_path: Path | None = None) -> C.Schema:
    """The schema of one database snapshot, merged with BIRD's descriptions and column meanings."""
    snap = snapshot or _snapshot_of(db_id)
    return _load(db_id, snap.path, snap.sha256, str(descriptions_root or C.DATABASES),
                 str(meaning_path or COLUMN_MEANING))[0]


def description_report(db_id: str, snapshot: C.Snapshot | None = None, **kw) -> dict:
    """What `load_schema` could not match for this database (files, rows) and the encodings it used."""
    snap = snapshot or _snapshot_of(db_id)
    return _load(db_id, snap.path, snap.sha256, str(kw.get("descriptions_root") or C.DATABASES),
                 str(kw.get("meaning_path") or COLUMN_MEANING))[1]


# --- the text ----------------------------------------------------------------------------------------------------------
def _column_head(table: C.SchemaTable, column: C.SchemaColumn) -> str:
    head = display_name(column.name)
    if column.type:
        head += " " + column.type
    if column.primary_key:
        head += " PK"
    for from_col, ref_table, to_col in table.foreign_keys:
        if from_col.lower() == column.name.lower():
            head += f" FK -> {display_name(ref_table)}.{display_name(to_col) if to_col else '?'}"
    return head


def _column_notes(column: C.SchemaColumn, meaning_cap: int | None = None) -> str:
    parts = []
    if column.description and _alnum(column.description) != _alnum(column.name):
        parts.append(column.description)
    if column.value_description:
        parts.append("values: " + column.value_description)
    meaning = _cut(compact_meaning(column.meaning), meaning_cap)
    if meaning:
        parts.append("meaning: " + meaning)
    return " | ".join(parts)


def _cut(text: str, cap: int | None) -> str:
    if cap is None or len(text) <= cap:
        return text
    if cap <= len(ELLIPSIS):
        return ""
    return text[: cap - len(ELLIPSIS)].rstrip() + ELLIPSIS


def _render(schema: C.Schema, meaning_cap: int | None = None, notes_cap: int | None = None) -> C.SerializedSchema:
    pieces, items, offset = [], [], 0

    def emit(text: str):
        nonlocal offset
        pieces.append(text)
        offset += len(text)

    emit(f"DATABASE {schema.db_id}\n")
    for table in schema.tables:
        emit("TABLE ")
        start = offset
        emit(display_name(table.name))
        items.append(C.SchemaItem(kind="table", table=table.name, column="", start=start, end=offset))
        emit("\n")
        for column in table.columns:
            emit("  ")
            start = offset
            emit(_column_head(table, column))
            notes = _cut(_column_notes(column, meaning_cap), notes_cap)
            if notes:
                emit(" -- " + notes)
            items.append(C.SchemaItem(kind="column", table=table.name, column=column.name, start=start, end=offset))
            emit("\n")
    return C.SerializedSchema(db_id=schema.db_id, snapshot_sha256=schema.snapshot_sha256, text="".join(pieces),
                              items=tuple(items))


def _largest_fitting(render, high: int, budget: int) -> int:
    """The largest cap in [0, high] whose rendering fits the budget (length is non-decreasing in the cap)."""
    low = 0
    while low < high:
        mid = (low + high + 1) // 2
        if len(render(mid).text) <= budget:
            low = mid
        else:
            high = mid - 1
    return low


def serialize(schema: C.Schema, budget_chars: int | None = None) -> C.SerializedSchema:
    """Deterministic schema text with the character span of every table and column (see the module docstring).

    Over budget, the meaning summaries are shortened first (one common cap, down to none); only if the text still
    does not fit are the remaining descriptions cut (again one common cap). No table or column is ever dropped."""
    full = _render(schema)
    if budget_chars is None or len(full.text) <= budget_chars:
        return full
    columns = [c for t in schema.tables for c in t.columns]
    if len(_render(schema, meaning_cap=0).text) <= budget_chars:
        longest = max((len(compact_meaning(c.meaning)) for c in columns), default=0)
        return _render(schema, meaning_cap=_largest_fitting(lambda cap: _render(schema, meaning_cap=cap), longest,
                                                              budget_chars))
    longest = max((len(_column_notes(c, 0)) for c in columns), default=0)
    cap = _largest_fitting(lambda cap: _render(schema, meaning_cap=0, notes_cap=cap), longest, budget_chars)
    return _render(schema, meaning_cap=0, notes_cap=cap)
