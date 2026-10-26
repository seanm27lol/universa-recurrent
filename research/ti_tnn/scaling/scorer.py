"""The development scorer: BIRD execution accuracy (EX), replicating the pinned official rule.

**Example.** Gold `SELECT name FROM t WHERE age > 30` returns the rows `[('Ann',), ('Bo',), ('Ann',)]`. A prediction
`SELECT DISTINCT name FROM t WHERE age > 30 ORDER BY name DESC` returns `[('Bo',), ('Ann',)]`. The official rule
compares `set(predicted_rows) == set(gold_rows)`, so this prediction scores 1: row order and duplicates do not count.

**The rule** (`contracts.OFFICIAL_SCORER`, bird-bench/mini_dev at `abd11b6`, `evaluation/evaluation_ex.py`):
`calculate_ex` returns 1 when `set(fetchall(predicted)) == set(fetchall(gold))` on the same SQLite file, and 0
otherwise. Any exception scores 0. So does exceeding `meta_time_out` (30 s). The official harness wraps one call to
`execute_sql` in that single timeout. That call opens the connection and runs the predicted query, then the gold
query, then the comparison. This scorer does the same in the same order under one shared deadline, with Python's
`cursor.execute` and `fetchall` and the default text factory.

**Where the official semantics surprise people** (each is tested on synthetic databases in
`tests/test_ti_tnn_scaling_scorer.py`):

| Behaviour | Consequence |
|---|---|
| Row order is ignored | `ORDER BY ... DESC` versus `ASC` both score 1, even for "top 3" questions whose order is the point |
| Duplicates collapse (`set`) | `SELECT name` and `SELECT DISTINCT name` are equal; so are 1 row and 5 copies of it |
| `1 == 1.0` in Python | an INTEGER result matches the same REAL value (`hash(1) == hash(1.0)`) |
| NULL is `None` | `(None,)` equals `(None,)`, unlike SQL's `NULL = NULL` |
| Column order counts | `(a, b)` rows never equal `(b, a)` rows; column names never count |
| Floats compare exactly | `100.0 * x / y` and `x * 100.0 / y` can differ in the last bit and then score 0 |
| Empty gold | when the gold returns no rows, any prediction that returns no rows scores 1, including a blank or comment-only string (Python's `execute("")` returns no rows) |
| One deadline for both | a gold query taking 25 s leaves the prediction 5 s |

**Deviations, all deliberate.**
1. The connection is the read-only, immutable URI from `tools.py`, not a plain `sqlite3.connect`. That cannot
   change what a SELECT returns. It makes every write fail and score 0. In the official harness, a predicted
   `DELETE` runs inside an uncommitted transaction that the gold query then reads, and a predicted `CREATE TABLE`
   autocommits.
2. An abstention (`predicted_sql=None`) scores 0 with status `abstain`. The official harness would execute a blank
   string, which returns no rows. The records layer, not the scorer, turns arm outputs into SQL strings.
3. The timeout interrupts the running SQLite statement through a progress handler. `func_timeout` instead abandons
   the thread. Both give 0.
4. The SQLite library is this host's (`sqlite3.sqlite_version`, recorded in every result). Functions and rounding
   can differ between SQLite versions.

The scorer never applies the tools' preview cap or the parse check: it runs exactly the submitted string.
"""
from __future__ import annotations

import ast
import hashlib
import sqlite3
import time
import urllib.request
from pathlib import Path

from . import contracts as C
from .tools import Deadline, connect_readonly, verify_snapshot

SCORER_VERSION = "scorer-v1"
OFFICIAL_CACHE = C.DATA_RAW / "official_scorer"


def _run_side(conn: sqlite3.Connection, sql: str, deadline: Deadline, shared: bool = True) -> tuple[dict, list | None]:
    """Execute one side as the official harness does (`cursor.execute`, then `fetchall`). `shared`: the deadline
    also covers the other query of the pair (scoring), rather than this query alone (the gold check)."""
    start = time.perf_counter()
    side = {"status": "ok", "row_count": None, "distinct_rows": None, "error": None, "elapsed_seconds": 0.0}
    rows = None
    if deadline.expired():
        side["status"] = "not_run"
        side["error"] = "no time left under the shared deadline"
        return side, None
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        if deadline.expired():
            raise sqlite3.OperationalError("interrupted")
        side["row_count"] = len(rows)
        side["distinct_rows"] = len(set(rows))
    except Exception as error:  # the official harness catches every exception and scores 0
        rows = None
        if deadline.expired():
            side["status"] = "timeout"
            side["error"] = f"exceeded {deadline.seconds:g} s" + (" (one deadline for both queries)" if shared else "")
        else:
            side["status"], side["error"] = "error", f"{type(error).__name__}: {error}"[:500]
    side["elapsed_seconds"] = round(time.perf_counter() - start, 6)
    return side, rows


def _result(snapshot: C.Snapshot, timeout: float) -> dict:
    return {"correct": 0, "status": "", "rule": C.OFFICIAL_SCORER["rule"], "scorer": SCORER_VERSION,
            "official_commit": C.OFFICIAL_SCORER["commit"], "sqlite_version": sqlite3.sqlite_version,
            "db_id": snapshot.db_id, "snapshot_sha256": snapshot.sha256, "timeout_seconds": timeout,
            "elapsed_seconds": 0.0, "gold_empty": None,
            "predicted": None, "gold": None}


def score(snapshot: C.Snapshot, predicted_sql: str | None, gold_sql: str, *,
          timeout_seconds: float | None = None) -> dict:
    """Score one prediction: `correct` is 1 exactly when the official rule gives 1 (see the module docstring)."""
    timeout = float(timeout_seconds or C.Limits().sql_timeout_seconds)
    result = _result(snapshot, timeout)
    start = time.perf_counter()
    verify_snapshot(snapshot)
    if predicted_sql is None:
        result["status"] = "abstain"
        result["predicted"] = {"status": "abstain", "row_count": None, "distinct_rows": None, "error": None,
                               "elapsed_seconds": 0.0}
        result["gold"] = {"status": "not_run", "row_count": None, "distinct_rows": None, "error": None,
                          "elapsed_seconds": 0.0}
        return result
    conn = connect_readonly(snapshot.path)
    try:
        start = time.perf_counter()  # the official timeout starts with the connection
        deadline = Deadline(conn, timeout, start)
        predicted, predicted_rows = _run_side(conn, str(predicted_sql), deadline)
        gold, gold_rows = _run_side(conn, gold_sql, deadline)
        same = predicted_rows is not None and gold_rows is not None and set(predicted_rows) == set(gold_rows)
        over = time.perf_counter() - start > timeout
    finally:
        conn.close()
    result["elapsed_seconds"] = round(time.perf_counter() - start, 6)
    result["predicted"], result["gold"] = predicted, gold
    result["gold_empty"] = None if gold_rows is None else len(gold_rows) == 0
    if over and predicted["status"] == "ok" and gold["status"] == "ok":
        result["status"] = "timeout"  # finished, but after the shared deadline (func_timeout would have fired)
    elif predicted["status"] != "ok":
        result["status"] = "predicted_" + predicted["status"]
    elif gold["status"] != "ok":
        result["status"] = "gold_" + gold["status"]
    else:
        result["status"] = "correct" if same else "incorrect"
    result["correct"] = int(result["status"] == "correct")
    return result


def execute_gold(snapshot: C.Snapshot, gold_sql: str, *, timeout_seconds: float | None = None) -> dict:
    """Run one gold query alone with the scorer's executor and timeout (the inventory's executability check)."""
    timeout = float(timeout_seconds or C.Limits().sql_timeout_seconds)
    verify_snapshot(snapshot)
    conn = connect_readonly(snapshot.path)
    try:
        side, _rows = _run_side(conn, gold_sql, Deadline(conn, timeout), shared=False)
    finally:
        conn.close()
    side["timeout_seconds"] = timeout
    return side


# --- the official source, for agreement checks ------------------------------------------------------------------------
def official_source(relative: str = "evaluation/evaluation_ex.py", *, cache: Path = OFFICIAL_CACHE,
                    timeout: float = 15.0) -> str:
    """The pinned official file's text, from the local cache or from GitHub at the pinned commit (sha256 checked)."""
    expected = C.OFFICIAL_SCORER["files"][relative]
    local = Path(cache) / C.OFFICIAL_SCORER["commit"] / relative
    if local.exists():
        data = local.read_bytes()
    else:
        url = ("https://raw.githubusercontent.com/bird-bench/mini_dev/"
               f"{C.OFFICIAL_SCORER['commit']}/{relative}")
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise ValueError(f"{relative}: sha256 {digest}, pinned {expected}")
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
    return data.decode("utf-8")


def official_functions(source: str, names: tuple[str, ...], namespace: dict | None = None) -> dict:
    """Compile only the named top-level functions of an official file (its imports need func_timeout, psycopg2)."""
    tree = ast.parse(source)
    chosen = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    missing = set(names) - {node.name for node in chosen}
    if missing:
        raise KeyError(f"not in the official source: {sorted(missing)}")
    module = ast.Module(body=chosen, type_ignores=[])
    scope = dict(namespace or {})
    exec(compile(module, "<official bird-bench/mini_dev>", "exec"), scope)  # noqa: S102 - pinned, hash-checked text
    return {name: scope[name] for name in names}
