"""The shared read-only tools: what every arm may do to the request's own database, and nothing more.

**Example.** An arm answering `bird23tf-00000` ("Who is the director of the movie Sex, Drink and Bloodshed?") on
`movie_platform` could call

    lookup_values(snap, "movies", "movie_title", pattern="Sex, Drink")
    execute_select(snap, "SELECT director_name FROM movies WHERE movie_title = 'Sex, Drink and Bloodshed'")

where `snap` is the pinned `contracts.Snapshot` of `movie_platform`. The second call returns at most 20 rows for the
model to read, plus the total row count, the column names and a status. The scorer never uses these previews: it
re-executes the submitted SQL in full (`scorer.py`).

**The contract.**

| Tool | Returns (besides `tool`, `args`, `db_id`, `snapshot_sha256`, `status`, `error`, `elapsed_seconds`) |
|---|---|
| `describe_schema(snap, tables=None)` | `text` (`schema.serialize`, budget `schema.PROMPT_BUDGET_CHARS`), `tables`, `unknown_tables` |
| `lookup_values(snap, table, column, pattern=None, limit=20)` | `values` (distinct, most frequent first), `truncated` |
| `execute_select(snap, sql, preview_rows=20)` | `columns`, `rows` (the preview), `row_count` (all rows), `truncated` |

`status` is one of `ok`, `error` (SQLite raised), `timeout` (the wall-clock limit, 30 s by default, was reached) or
`rejected` (not a single read-only SELECT, or arguments outside the contract). Every return value is a plain
JSON-serializable dict.

**Read-only, three ways.** Connections open the file through the URI `file:<path>?mode=ro&immutable=1` and set
`PRAGMA query_only=ON`, with ATTACH disabled. `execute_select` also parses the SQL with sqlglot (`read="sqlite"`) and
accepts exactly one statement whose root is a SELECT, a WITH ... SELECT or a set operation. And an SQLite authorizer
refuses any action other than reading and calling functions. The parse check is a policy, so a model gets a clear
`rejected`. The other two are the actual barrier: a write fails even if the parse check were wrong.

**Binding.** Every call first checks that the snapshot file is the pinned one (its sha256 and byte count). The full
hash runs once per process and file. Later calls re-check the file's size, mtime, inode and device, and re-hash if
any of them changed (`verify_snapshot(snap, rehash=True)` forces a full hash). The public benchmark's reserved
databases are refused outright.

**Limits.** The timeout is a progress handler that checks a wall-clock deadline, counted from the connection (after
the binding check), every 10,000 SQLite virtual-machine instructions. Long single operations, such as one large in-memory sort, can overrun it by their own duration. A call
that finishes after its deadline is still reported as `timeout`. Preview cells longer than 300 characters are cut and
marked. Blobs are shown as `<blob N bytes>`.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import sqlglot
from sqlglot import exp

from . import contracts as C

TOOL_VERSION = "tools-v1"
_LIMITS = C.Limits()
PROGRESS_INSTRUCTIONS = 10_000   # SQLite VM instructions between deadline checks
PREVIEW_CELL_CHARS = 300         # longer text cells are cut in previews and lookups
_READ_ROOTS = (exp.Select, exp.SetOperation)


class SnapshotMismatch(RuntimeError):
    """The file at a snapshot's path is not the pinned file."""


# --- binding: the file must be the pinned one ----------------------------------------------------------------------
_VERIFIED: dict[tuple, str] = {}


def sha256_file(path: str | os.PathLike) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _refuse_reserved(path: str) -> None:
    reserved = os.path.realpath(C.RESERVED_DATABASES)
    real = os.path.realpath(path)
    if real == reserved or real.startswith(reserved + os.sep):
        raise SnapshotMismatch(f"{path} is a reserved public-benchmark database; no arm may query it")


def verify_snapshot(snapshot: C.Snapshot, *, rehash: bool = False) -> None:
    """Raise `SnapshotMismatch` unless the file at `snapshot.path` has the pinned byte count and sha256."""
    _refuse_reserved(snapshot.path)
    try:
        stat = os.stat(snapshot.path)
    except FileNotFoundError as missing:
        raise SnapshotMismatch(f"{snapshot.db_id}: no file at {snapshot.path}") from missing
    if stat.st_size != snapshot.bytes:
        raise SnapshotMismatch(f"{snapshot.db_id}: {stat.st_size} bytes on disk, {snapshot.bytes} pinned")
    signature = (os.path.realpath(snapshot.path), stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev)
    if not rehash and _VERIFIED.get(signature) == snapshot.sha256:
        return
    digest = sha256_file(snapshot.path)
    if digest != snapshot.sha256:
        raise SnapshotMismatch(f"{snapshot.db_id}: sha256 {digest} on disk, {snapshot.sha256} pinned")
    _VERIFIED[signature] = digest


# --- connections and the deadline ---------------------------------------------------------------------------------
def readonly_uri(path: str | os.PathLike) -> str:
    return "file:" + urllib.parse.quote(os.path.abspath(path)) + "?mode=ro&immutable=1"


def connect_readonly(path: str | os.PathLike) -> sqlite3.Connection:
    """A read-only, immutable connection with query_only on and ATTACH disabled. Python's default text factory is
    kept, as in the official harness, so text that is not valid UTF-8 raises the same error there and here."""
    _refuse_reserved(str(path))
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    conn = sqlite3.connect(readonly_uri(path), uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
    return conn


class Deadline:
    """A wall-clock deadline enforced by an SQLite progress handler: past it, the running statement is interrupted."""

    def __init__(self, conn: sqlite3.Connection, seconds: float, start: float | None = None):
        self.start = time.perf_counter() if start is None else start
        self.seconds = float(seconds)
        self.at = self.start + self.seconds
        self.fired = False
        conn.set_progress_handler(self._check, PROGRESS_INSTRUCTIONS)

    def _check(self) -> int:
        if time.perf_counter() > self.at:
            self.fired = True
            return 1
        return 0

    def expired(self) -> bool:
        return self.fired or time.perf_counter() > self.at


_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


def _authorize_reads(action, *_):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def check_read_only(sql: str) -> str | None:
    """None if `sql` is exactly one SELECT / WITH ... SELECT / set operation (sqlglot, read="sqlite"); else why not."""
    if not isinstance(sql, str) or not sql.strip():
        return "empty statement"
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except Exception as error:  # sqlglot raises ParseError / TokenError
        return f"could not parse as SQLite: {str(error).splitlines()[0][:300]}"
    if len(statements) != 1:
        return f"exactly one statement is allowed, found {len(statements)}"
    root = statements[0]
    while isinstance(root, exp.Subquery):  # a parenthesized query
        root = root.this
    if not isinstance(root, _READ_ROOTS):
        return f"only SELECT, WITH ... SELECT and set operations are allowed, not {type(root).__name__}"
    for node in root.walk():
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Pragma, exp.Attach,
                             exp.Detach, exp.Command, exp.Transaction, exp.Commit, exp.Rollback, exp.Alter)):
            return f"statement contains {type(node).__name__}"
    return None


# --- JSON-safe values for previews ----------------------------------------------------------------------------------
def display_value(value):
    """A JSON-safe rendering of one SQLite value for a model to read (never for scoring)."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else repr(value)
    if isinstance(value, bytes):
        return f"<blob {len(value)} bytes>"
    text = str(value)
    if len(text) > PREVIEW_CELL_CHARS:
        return text[:PREVIEW_CELL_CHARS] + f"...[+{len(text) - PREVIEW_CELL_CHARS} chars]"
    return text


def _base(tool: str, snapshot: C.Snapshot, args: dict) -> dict:
    return {"tool": tool, "args": args, "db_id": snapshot.db_id, "snapshot_sha256": snapshot.sha256,
            "status": "ok", "error": None, "elapsed_seconds": 0.0}


def _finish(result: dict, start: float) -> dict:
    result["elapsed_seconds"] = round(time.perf_counter() - start, 6)
    return result


def _sqlite_failure(result: dict, error: Exception, deadline: Deadline | None) -> dict:
    if deadline is not None and deadline.expired():
        result["status"], result["error"] = "timeout", f"exceeded {deadline.seconds:g} s"
    else:
        result["status"], result["error"] = "error", f"{type(error).__name__}: {error}"
    return result


# --- the three tools ------------------------------------------------------------------------------------------------
def describe_schema(snapshot: C.Snapshot, tables=None, budget_chars: int | None = -1) -> dict:
    """The serialized schema (all tables, or the listed ones in database order) of the bound snapshot.

    By default the text uses `schema.PROMPT_BUDGET_CHARS`, so a call without `tables` returns the prompt's text;
    a few listed tables usually fit whole. Pass `budget_chars=None` for the untruncated text."""
    from . import schema as schema_mod

    if budget_chars == -1:
        budget_chars = schema_mod.PROMPT_BUDGET_CHARS
    if isinstance(tables, str):  # one table name
        tables = [tables]
    start = time.perf_counter()
    result = _base("describe_schema", snapshot, {"tables": list(tables) if tables is not None else None,
                                                 "budget_chars": budget_chars})
    try:
        verify_snapshot(snapshot)
        full = schema_mod.load_schema(snapshot.db_id, snapshot=snapshot)
    except (SnapshotMismatch, sqlite3.Error, OSError) as error:
        result["status"], result["error"] = "error", f"{type(error).__name__}: {error}"
        return _finish(result, start)
    chosen, unknown = full.tables, []
    if tables is not None:
        wanted = [str(t) for t in tables]
        by_lower = {t.name.lower(): t for t in full.tables}
        unknown = [t for t in wanted if t.lower() not in by_lower]
        keep = {t.lower() for t in wanted}
        chosen = tuple(t for t in full.tables if t.name.lower() in keep)
        if not chosen:
            result.update(status="error", error="no such table: " + ", ".join(unknown), text="", tables=[],
                          unknown_tables=unknown)
            return _finish(result, start)
    subset = C.Schema(db_id=full.db_id, snapshot_sha256=full.snapshot_sha256, tables=tuple(chosen))
    serialized = schema_mod.serialize(subset, budget_chars=budget_chars)
    result.update(text=serialized.text, tables=[t.name for t in chosen], unknown_tables=unknown)
    return _finish(result, start)


def _resolve(schema: C.Schema, table: str, column: str):
    for t in schema.tables:
        if t.name.lower() == str(table).lower():
            for c in t.columns:
                if c.name.lower() == str(column).lower():
                    return t.name, c.name
            return t.name, None
    return None, None


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def lookup_values(snapshot: C.Snapshot, table: str, column: str, pattern: str | None = None,
                  limit: int = _LIMITS.lookup_max_values, timeout_seconds: float | None = None) -> dict:
    """Distinct non-NULL values of one column, most frequent first (ties by value), at most `limit` (<= 20).

    `pattern` is a plain substring (no wildcards; `%`, `_` and `\\` are escaped) matched with SQLite's LIKE, so
    ASCII letters match case-insensitively. It is passed as a bound parameter, never spliced into the SQL."""
    from . import schema as schema_mod

    start = time.perf_counter()
    result = _base("lookup_values", snapshot, {"table": table, "column": column, "pattern": pattern, "limit": limit})
    result.update(values=[], truncated=False)
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= _LIMITS.lookup_max_values:
        result.update(status="rejected", error=f"limit must be an integer in 1..{_LIMITS.lookup_max_values}")
        return _finish(result, start)
    if pattern is not None and not isinstance(pattern, str):
        result.update(status="rejected", error="pattern must be a string or null")
        return _finish(result, start)
    try:
        verify_snapshot(snapshot)
        schema = schema_mod.load_schema(snapshot.db_id, snapshot=snapshot)
    except (SnapshotMismatch, sqlite3.Error, OSError) as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        return _finish(result, start)
    real_table, real_column = _resolve(schema, table, column)
    if real_table is None or real_column is None:
        result.update(status="error", error=f"no such column: {table}.{column}")
        return _finish(result, start)
    qcol, qtab = quote_identifier(real_column), quote_identifier(real_table)
    sql = f"SELECT {qcol}, COUNT(*) AS n FROM {qtab} WHERE {qcol} IS NOT NULL"
    params: list = []
    if pattern is not None:
        escaped = pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        sql += f" AND {qcol} LIKE ? ESCAPE '\\'"
        params.append(f"%{escaped}%")
    sql += f" GROUP BY {qcol} ORDER BY n DESC, {qcol} LIMIT ?"
    params.append(limit + 1)
    deadline = None
    conn = connect_readonly(snapshot.path)
    try:
        deadline = Deadline(conn, timeout_seconds or _LIMITS.sql_timeout_seconds)  # starts at the connection
        rows = conn.execute(sql, params).fetchall()
        if deadline.expired():
            raise sqlite3.OperationalError("interrupted")
        result["values"] = [display_value(r[0]) for r in rows[:limit]]
        result["truncated"] = len(rows) > limit
    except sqlite3.Error as error:
        _sqlite_failure(result, error, deadline)
    finally:
        conn.close()
    return _finish(result, start)


def execute_select(snapshot: C.Snapshot, sql: str, preview_rows: int = _LIMITS.preview_rows,
                   timeout_seconds: float | None = None) -> dict:
    """Run one read-only SELECT in full; return its first `preview_rows` rows, the total row count and a status."""
    start = time.perf_counter()
    result = _base("execute_select", snapshot, {"sql": sql, "preview_rows": preview_rows})
    result.update(columns=[], rows=[], row_count=None, truncated=False,
                  timeout_seconds=float(timeout_seconds or _LIMITS.sql_timeout_seconds))
    if not isinstance(preview_rows, int) or isinstance(preview_rows, bool) or preview_rows < 0:
        result.update(status="rejected", error="preview_rows must be a non-negative integer")
        return _finish(result, start)
    reason = check_read_only(sql)
    if reason is not None:
        result.update(status="rejected", error=reason)
        return _finish(result, start)
    try:
        verify_snapshot(snapshot)
    except SnapshotMismatch as error:
        result.update(status="error", error=f"SnapshotMismatch: {error}")
        return _finish(result, start)
    deadline = None
    conn = connect_readonly(snapshot.path)
    try:
        conn.set_authorizer(_authorize_reads)
        deadline = Deadline(conn, result["timeout_seconds"])  # starts at the connection, after verification
        cursor = conn.execute(sql)
        result["columns"] = [d[0] for d in (cursor.description or ())]
        preview, count = [], 0
        while True:
            batch = cursor.fetchmany(1000)
            if not batch:
                break
            if len(preview) < preview_rows:
                preview.extend([display_value(v) for v in row] for row in batch[: preview_rows - len(preview)])
            count += len(batch)
        if deadline.expired():
            raise sqlite3.OperationalError("interrupted")
        result.update(rows=preview, row_count=count, truncated=count > len(preview))
    except sqlite3.Error as error:
        result.update(rows=[], row_count=None)
        _sqlite_failure(result, error, deadline)
    finally:
        conn.close()
    return _finish(result, start)


@dataclass(frozen=True)
class Toolbox:
    """The three tools bound to one snapshot. Creating it verifies the file (sha256 and byte count)."""
    snapshot: C.Snapshot

    def __post_init__(self):
        verify_snapshot(self.snapshot)

    def describe_schema(self, tables=None, budget_chars=-1) -> dict:
        return describe_schema(self.snapshot, tables, budget_chars)

    def lookup_values(self, table, column, pattern=None, limit=_LIMITS.lookup_max_values) -> dict:
        return lookup_values(self.snapshot, table, column, pattern, limit)

    def execute_select(self, sql, preview_rows=_LIMITS.preview_rows) -> dict:
        return execute_select(self.snapshot, sql, preview_rows)


def bind(snapshot: C.Snapshot) -> Toolbox:
    return Toolbox(snapshot)


def snapshot_for_file(db_id: str, path: str | os.PathLike) -> C.Snapshot:
    """A Snapshot for a file that has no pin yet (tests, synthetic databases): hashes it as it is now."""
    path = Path(path)
    return C.Snapshot(db_id=db_id, path=str(path.resolve()), sha256=sha256_file(path), bytes=path.stat().st_size)
