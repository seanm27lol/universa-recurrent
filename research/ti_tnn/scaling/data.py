"""Sources, identities, the database split, the development inventory and the pins.

**Example.** Line 0 of the pinned training release (`birdsql/bird23-train-filtered` at `4068469`) is a question about
`movie_platform`: "Who is the director of the movie Sex, Drink and Bloodshed?". It gets the stable id
`bird23tf-00000` (0-based line index) and a `content_sha256` over its db_id, question, evidence and SQL, joined by the
unit separator `\\x1f`. `movie_platform` is one of the 69 databases. The split puts each database, never each
question, on one side: development or training pool.

**What this module produces** (run `PYTHONPATH=research/ti_tnn python -m scaling.data all` from the repo root):

| Pin (`research/ti_tnn/scaling/pins/`) | Holds |
|---|---|
| `snapshots_v1.json` | sha256 and size of each of the 69 training database files and of their description CSVs |
| `split_v1.json` | the 14 development and 55 training-pool databases, the rule, the salt, questions per database |
| `inventory_v1.json` | the ordered development inventory (<= 512; its first 64 and 16 are the subsets) and the exclusions |
| `smoke_train_candidates_v1.json` | the first 1,024 training-pool ids in the smoke-train order (the IR filters them) |
| `reserved_public_pin_v1.json` | the reserved Mini-Dev file's hash, its 500 question ids and db_ids, its 11 database hashes |

Recomputing a pin that already exists must reproduce it exactly, or the build stops.

**The rules** are `contracts.SPLIT`, `contracts.INVENTORY` and `contracts.SMOKE_TRAIN`. Each orders items by
`sha256(salt + ":" + item)` as lowercase hex. The inventory takes one question per development database per round,
databases in split order. Questions whose gold SQL errors, or exceeds the scorer's 30 s timeout, on the pinned
snapshot are dropped first and listed with their reasons.

**The reserved benchmark** (Mini-Dev, 500 questions) is only pinned. `reserved_pin` keeps each record's
`question_id` and `db_id` and discards everything else as soon as the file is parsed. Only the 11 SQLite files are
extracted from `minidev.zip`, so they can be hashed. They are made read-only, and `tools.py` refuses to open them.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import functools
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import contracts as C

TRAIN_JSONL = C.DATA_RAW / "hf" / "bird23_train_filtered.train-00000-of-00001.jsonl"
TRAIN_JSONL_KEY = "data/train-00000-of-00001.jsonl"
COLUMN_MEANING = C.DATA_RAW / "hf" / "bird23_train_filtered.train_column_meaning.json"
COLUMN_MEANING_KEY = "train_column_meaning.json"
RESERVED_JSON = C.DATA_RAW / "hf" / "mini_dev_sqlite.reserved.json"
TRAIN_ARCHIVE = C.DATA_RAW / "bird_train" / "train.zip"
NESTED_ARCHIVE = C.DATA_RAW / "bird_train" / "train_databases.zip"
RESERVED_ARCHIVE = C.DATA_RAW / "mini_dev" / "minidev.zip"
GOLD_CHECK = C.DATA / "gold_check_dev_v1.json"  # per-question executability details (data product, not a pin)

PIN_SNAPSHOTS = C.PINS / "snapshots_v1.json"
PIN_SPLIT = C.PINS / "split_v1.json"
PIN_INVENTORY = C.PINS / "inventory_v1.json"
PIN_SMOKE = C.PINS / "smoke_train_candidates_v1.json"
PIN_RESERVED = C.PINS / "reserved_public_pin_v1.json"
SMOKE_IDS_WRITTEN = 1024
SEPARATOR = "\x1f"


class PinMismatch(RuntimeError):
    """A recomputed identity differs from the committed pin."""


# --- hashing and pins ---------------------------------------------------------------------------------------------
def sha256_file(path: str | os.PathLike) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def order_key(salt: str, item: str) -> str:
    """The ordering key of every rule in contracts.py: sha256(salt + ':' + item), lowercase hex."""
    return sha256_text(f"{salt}:{item}")


def verify_file(path: Path, expected_sha256: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise PinMismatch(f"{path}: sha256 {digest}, pinned {expected_sha256}")


def dump_pin(payload: dict) -> str:
    return json.dumps(payload, indent=1, ensure_ascii=False) + "\n"


def write_pin(path: Path, payload: dict, *, rewrite: bool = False) -> dict:
    """Write a pin, or check that an existing pin has exactly this content."""
    text = dump_pin(payload)
    if path.exists() and not rewrite:
        old = path.read_text(encoding="utf-8")
        if old != text:
            raise PinMismatch(f"{path.name}: the recomputed pin differs from the committed one (use --rewrite only "
                              "if the change is intended and reviewed)")
        return payload
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return payload


def read_pin(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; build it with `python -m scaling.data all`")
    return json.loads(path.read_text(encoding="utf-8"))


def pin_digest(path: Path) -> str:
    return sha256_file(path)


# --- the training release -------------------------------------------------------------------------------------------
def task_id_for(index: int) -> str:
    return f"bird23tf-{index:05d}"


def content_sha256(db_id: str, question: str, evidence: str, sql: str) -> str:
    return sha256_text(SEPARATOR.join((db_id, question, evidence, sql)))


@dataclass(frozen=True)
class Row:
    """One line of the training release. Inference paths get `task()`; only the scorer and trainer get `gold()`."""
    task_id: str
    index: int
    db_id: str
    question: str
    evidence: str
    sql: str
    content_sha256: str

    def task(self, split: str) -> C.Task:
        return C.Task(task_id=self.task_id, db_id=self.db_id, question=self.question, evidence=self.evidence,
                      split=split)

    def gold(self) -> C.Gold:
        return C.Gold(task_id=self.task_id, sql=self.sql)


def parse_rows(lines: list[str]) -> list[Row]:
    """Rows from JSONL lines; the id is the 0-based line index. Only a final empty line is tolerated."""
    if lines and lines[-1] == "":
        lines = lines[:-1]
    out = []
    for index, line in enumerate(lines):
        if not line.strip():
            raise ValueError(f"empty line {index} would shift every later id")
        record = json.loads(line)
        db_id, question, evidence, sql = (str(record[k]) for k in ("db_id", "question", "evidence", "SQL"))
        out.append(Row(task_id=task_id_for(index), index=index, db_id=db_id, question=question, evidence=evidence,
                       sql=sql, content_sha256=content_sha256(db_id, question, evidence, sql)))
    return out


@functools.lru_cache(maxsize=1)
def rows() -> tuple[Row, ...]:
    """The pinned training release (sha256 verified), one Row per line."""
    verify_file(TRAIN_JSONL, C.TRAIN_RELEASE["files"][TRAIN_JSONL_KEY])
    parsed = parse_rows(TRAIN_JSONL.read_text(encoding="utf-8").split("\n"))
    if len(parsed) != C.TRAIN_RELEASE["rows"]:
        raise PinMismatch(f"{len(parsed)} rows, pinned {C.TRAIN_RELEASE['rows']}")
    if len({r.db_id for r in parsed}) != C.TRAIN_RELEASE["databases"]:
        raise PinMismatch("database count differs from the pin")
    return tuple(parsed)


@functools.lru_cache(maxsize=1)
def _rows_by_id() -> dict[str, Row]:
    return {r.task_id: r for r in rows()}


def row(task_id: str) -> Row:
    return _rows_by_id()[task_id]


def database_ids() -> list[str]:
    return sorted({r.db_id for r in rows()})


def verify_column_meaning() -> None:
    verify_file(COLUMN_MEANING, C.TRAIN_RELEASE["files"][COLUMN_MEANING_KEY])


# --- the split ------------------------------------------------------------------------------------------------------
def split_databases(db_ids, salt: str = C.SPLIT["salt"], fraction: float = C.SPLIT["dev_fraction"]):
    """(development, training pool): databases ordered by order_key; the first round(fraction * n) are development."""
    ordered = sorted(set(db_ids), key=lambda d: (order_key(salt, d), d))
    n_dev = round(fraction * len(ordered))
    return ordered[:n_dev], ordered[n_dev:]


@functools.lru_cache(maxsize=1)
def _split() -> tuple[tuple[str, ...], tuple[str, ...]]:
    dev, pool = split_databases(database_ids())
    if set(dev) & set(pool):
        raise AssertionError("split is not disjoint")
    return tuple(dev), tuple(pool)


def dev_databases() -> list[str]:
    return list(_split()[0])


def train_pool_databases() -> list[str]:
    return list(_split()[1])


def split_of(db_id: str) -> str:
    dev, pool = _split()
    if db_id in dev:
        return "dev"
    if db_id in pool:
        return "train"
    raise KeyError(f"{db_id} is not a training-release database")


def split_payload() -> dict:
    dev, pool = _split()
    counts = Counter(r.db_id for r in rows())
    return {
        "version": "split_v1", "study": C.STUDY, "rule": C.SPLIT["rule"], "salt": C.SPLIT["salt"],
        "dev_fraction": C.SPLIT["dev_fraction"],
        "source": {"repo": C.TRAIN_RELEASE["repo"], "revision": C.TRAIN_RELEASE["revision"],
                   "file": TRAIN_JSONL_KEY, "sha256": C.TRAIN_RELEASE["files"][TRAIN_JSONL_KEY]},
        "databases": len(dev) + len(pool),
        "dev": list(dev), "train_pool": list(pool),
        "dev_questions": sum(counts[d] for d in dev), "train_pool_questions": sum(counts[d] for d in pool),
        "questions_per_database": {d: counts[d] for d in list(dev) + list(pool)},
    }


def split(write: bool = True, rewrite: bool = False) -> dict:
    """The database split (contracts.SPLIT); writes or checks pins/split_v1.json."""
    payload = split_payload()
    if len(payload["dev"]) != 14 or len(payload["train_pool"]) != 55:
        raise AssertionError(f"expected 14 development / 55 pool databases, got {len(payload['dev'])} / "
                             f"{len(payload['train_pool'])}")
    return write_pin(PIN_SPLIT, payload, rewrite=rewrite) if write else payload


# --- ordering rules (pure) ------------------------------------------------------------------------------------------
def round_robin(per_database: dict[str, list[str]], database_order: list[str], size: int) -> list[str]:
    """One item per database per round, databases in the given order, skipping exhausted ones; the first `size`."""
    queues = {d: list(per_database.get(d, [])) for d in database_order}
    out, position = [], 0
    while len(out) < size and any(queues[d][position:] for d in database_order):
        for d in database_order:
            if position < len(queues[d]):
                out.append(queues[d][position])
                if len(out) == size:
                    break
        position += 1
    return out


def inventory_order(questions: dict[str, list[str]], database_order: list[str], excluded=frozenset(),
                    salt: str = C.INVENTORY["salt"], size: int = C.INVENTORY["size"]) -> list[str]:
    """contracts.INVENTORY: order each database's questions by order_key, drop exclusions, then round-robin."""
    per_database = {d: sorted((t for t in questions.get(d, []) if t not in excluded),
                              key=lambda t: (order_key(salt, t), t)) for d in database_order}
    return round_robin(per_database, database_order, size)


def smoke_order(task_ids, salt: str = C.SMOKE_TRAIN["salt"]) -> list[str]:
    return sorted(task_ids, key=lambda t: (order_key(salt, t), t))


# --- archives and snapshots -----------------------------------------------------------------------------------------
def _check_archive_bytes(path: Path, expected: int) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing (is the download finished?)")
    size = path.stat().st_size
    if size != expected:
        raise PinMismatch(f"{path.name}: {size} bytes, pinned {expected} (incomplete download?)")


def _safe_parts(name: str) -> list[str] | None:
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts) or parts[0].startswith("__MACOSX"):
        return None
    if any(p.startswith(".") or p.startswith("._") for p in parts):
        return None
    return parts


def _extract_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path) -> None:
    """Copy one member to destination (CRC checked by zipfile at end of stream); skip if already complete."""
    if destination.exists() and destination.stat().st_size == info.file_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".part")
    if tmp.exists():
        tmp.unlink()
    with zf.open(info) as source, open(tmp, "wb") as sink:
        shutil.copyfileobj(source, sink, length=16 * 1024 * 1024)
    if destination.exists():
        destination.chmod(0o644)
    os.replace(tmp, destination)


def _database_members(zf: zipfile.ZipFile, wanted: set[str]) -> dict[str, dict]:
    """{db_id: {"sqlite": info, "descriptions": [(file name, info)]}} for `<db>/<db>.sqlite` style members."""
    found: dict[str, dict] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = _safe_parts(info.filename)
        if parts is None:
            continue
        if len(parts) >= 2 and parts[-1] == f"{parts[-2]}.sqlite" and parts[-2] in wanted:
            entry = found.setdefault(parts[-2], {"sqlite": None, "descriptions": []})
            if entry["sqlite"] is not None:
                raise ValueError(f"two SQLite files for {parts[-2]} in {zf.filename}")
            entry["sqlite"] = info
        elif (len(parts) >= 3 and parts[-2] == "database_description" and parts[-3] in wanted
              and parts[-1].lower().endswith(".csv")):
            found.setdefault(parts[-3], {"sqlite": None, "descriptions": []})["descriptions"].append(
                (parts[-1], info))
    return found


def extract_training_databases(log=print) -> dict:
    """Extract the 69 training databases (+ database_description CSVs) from the official BIRD training archive."""
    _check_archive_bytes(TRAIN_ARCHIVE, C.TRAIN_RELEASE["database_archive"]["bytes"])
    wanted = set(database_ids())
    with zipfile.ZipFile(TRAIN_ARCHIVE) as outer:
        members = _database_members(outer, wanted)
        source, nested_name = outer, None
        if not members:
            nested = [i for i in outer.infolist()
                      if _safe_parts(i.filename) and i.filename.endswith("train_databases.zip")]
            if len(nested) != 1:
                raise ValueError(f"no databases and {len(nested)} nested train_databases.zip in {TRAIN_ARCHIVE}")
            nested_name = nested[0].filename
            log(f"extracting nested {nested_name} ({nested[0].file_size} bytes)")
            _extract_member(outer, nested[0], NESTED_ARCHIVE)
        if nested_name is None:
            return _extract_from(source, members, wanted, log)
    with zipfile.ZipFile(NESTED_ARCHIVE) as inner:
        result = _extract_from(inner, _database_members(inner, wanted), wanted, log)
    result["nested_archive"] = nested_name
    return result


def _extract_from(zf: zipfile.ZipFile, members: dict, wanted: set[str], log) -> dict:
    missing = sorted(d for d in wanted if d not in members or members[d]["sqlite"] is None)
    if missing:
        raise ValueError(f"databases missing from the archive: {missing}")
    for db_id in sorted(wanted):
        entry = members[db_id]
        target = C.DATABASES / db_id / f"{db_id}.sqlite"
        _extract_member(zf, entry["sqlite"], target)
        target.chmod(0o444)
        for name, info in sorted(entry["descriptions"], key=lambda x: x[0]):
            _extract_member(zf, info, C.DATABASES / db_id / "database_description" / name)
        log(f"  {db_id}: {entry['sqlite'].file_size} bytes, {len(entry['descriptions'])} description files")
    return {"databases": len(wanted)}


def _description_digest(db_dir: Path) -> tuple[str, int]:
    desc = db_dir / "database_description"
    files = sorted(p for p in desc.iterdir() if p.is_file() and not p.name.startswith(".")) if desc.is_dir() else []
    listing = "".join(f"{p.name}\t{sha256_file(p)}\n" for p in files)
    return sha256_text(listing), len(files)


def snapshots(write: bool = True, rewrite: bool = False, archive_sha256: bool = True) -> dict:
    """sha256 and size of every training database file (and a digest of its description CSVs); pins them."""
    databases = {}
    for db_id in database_ids():
        path = C.DATABASES / db_id / f"{db_id}.sqlite"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run `python -m scaling.data extract` first")
        description_sha256, description_files = _description_digest(path.parent)
        databases[db_id] = {"file": f"{db_id}/{db_id}.sqlite", "sha256": sha256_file(path),
                            "bytes": path.stat().st_size, "description_sha256": description_sha256,
                            "description_files": description_files}
    archive = dict(C.TRAIN_RELEASE["database_archive"])
    if archive_sha256:
        archive["sha256"] = sha256_file(TRAIN_ARCHIVE)
        if NESTED_ARCHIVE.exists():
            with zipfile.ZipFile(TRAIN_ARCHIVE) as outer:
                names = [i.filename for i in outer.infolist()
                         if _safe_parts(i.filename) and i.filename.endswith("train_databases.zip")]
            archive["nested"] = {"name": names[0] if len(names) == 1 else names,
                                 "bytes": NESTED_ARCHIVE.stat().st_size, "sha256": sha256_file(NESTED_ARCHIVE)}
    payload = {"version": "snapshots_v1", "study": C.STUDY, "root": "runs/ti_tnn_scaling/data/databases",
               "description_digest": "sha256 over lines '<file name>\\t<sha256>\\n' of database_description/*, "
                                     "sorted by name",
               "archive": archive, "databases": databases}
    return write_pin(PIN_SNAPSHOTS, payload, rewrite=rewrite) if write else payload


@functools.lru_cache(maxsize=1)
def _snapshot_pin() -> dict:
    return read_pin(PIN_SNAPSHOTS)


def snapshot(db_id: str) -> C.Snapshot:
    """The pinned snapshot of one training-release database (tools.py verifies the file against it)."""
    entry = _snapshot_pin()["databases"].get(db_id)
    if entry is None:
        raise KeyError(f"{db_id} has no pinned snapshot (reserved and unknown databases have none)")
    return C.Snapshot(db_id=db_id, path=str(C.DATABASES / entry["file"]), sha256=entry["sha256"],
                      bytes=entry["bytes"])


# --- gold executability (development databases) ---------------------------------------------------------------------
def _gold_job(batch):
    from . import scorer

    out = []
    for task_id, snap_fields, sql, timeout in batch:
        out.append((task_id, scorer.execute_gold(C.Snapshot(*snap_fields), sql, timeout_seconds=timeout)))
    return out


def check_gold(task_ids: list[str], workers: int = 8, timeout_seconds: float | None = None, log=print) -> dict:
    """Run each question's gold SQL once with the scorer's executor; re-run timeouts and slow items alone.

    The parallel pass shares the CPU, so anything that timed out or used more than half the timeout is run again
    serially, and the serial result is final (both are kept)."""
    from . import scorer

    timeout = float(timeout_seconds or C.Limits().sql_timeout_seconds)
    by_db: dict[str, list] = {}
    for task_id in task_ids:
        r = row(task_id)
        snap = snapshot(r.db_id)
        by_db.setdefault(r.db_id, []).append((task_id, (snap.db_id, snap.path, snap.sha256, snap.bytes), r.sql,
                                              timeout))
    batches = []
    for db_id, items in sorted(by_db.items()):
        for i in range(0, len(items), 25):
            batches.append(items[i:i + 25])
    results: dict[str, dict] = {}
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for done in pool.map(_gold_job, batches):
            for task_id, result in done:
                results[task_id] = result
            log(f"  gold check: {len(results)}/{len(task_ids)} ({time.perf_counter() - started:.0f} s)")
    for task_id in sorted(results):
        first = results[task_id]
        if first["status"] == "timeout" or first["elapsed_seconds"] > timeout / 2:
            r = row(task_id)
            again = scorer.execute_gold(snapshot(r.db_id), r.sql, timeout_seconds=timeout)
            again["parallel_pass"] = first
            results[task_id] = again
            log(f"  serial re-run {task_id}: {first['status']} {first['elapsed_seconds']:.1f} s -> "
                f"{again['status']} {again['elapsed_seconds']:.1f} s")
    return results


def gold_check(workers: int = 8, log=print) -> dict:
    """The development gold check, written to runs/ti_tnn_scaling/data/gold_check_dev_v1.json."""
    dev = set(dev_databases())
    ids = [r.task_id for r in rows() if r.db_id in dev]
    results = check_gold(ids, workers=workers, log=log)
    payload = {"version": "gold_check_dev_v1", "sqlite_version": sqlite3.sqlite_version,
               "python": sys.version.split()[0], "timeout_seconds": C.Limits().sql_timeout_seconds,
               "executor": "scorer.execute_gold (read-only immutable URI, cursor.execute + fetchall)",
               "snapshots_pin_sha256": pin_digest(PIN_SNAPSHOTS), "checked": len(results),
               "results": {k: results[k] for k in sorted(results)}}
    GOLD_CHECK.parent.mkdir(parents=True, exist_ok=True)
    GOLD_CHECK.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return payload


def _gold_results() -> dict:
    if not GOLD_CHECK.exists():
        raise FileNotFoundError(f"{GOLD_CHECK} is missing; run `python -m scaling.data gold-check` first")
    return json.loads(GOLD_CHECK.read_text(encoding="utf-8"))


# --- the inventory and the smoke-train order ------------------------------------------------------------------------
def inventory(write: bool = True, rewrite: bool = False) -> dict:
    """contracts.INVENTORY over the development databases, excluding non-executable gold; pins it."""
    dev = dev_databases()
    checked = _gold_results()
    results = checked["results"]
    questions = {d: [] for d in dev}
    for r in rows():
        if r.db_id in questions:
            questions[r.db_id].append(r.task_id)
    unchecked = [t for ids in questions.values() for t in ids if t not in results]
    if unchecked:
        raise ValueError(f"{len(unchecked)} development questions have no gold check")
    excluded = {t: results[t] for ids in questions.values() for t in ids if results[t]["status"] != "ok"}
    ordered = inventory_order(questions, dev, frozenset(excluded))
    composition = {}
    for n in (*sorted(C.INVENTORY["subsets"]), C.INVENTORY["size"]):
        counts = Counter(row(t).db_id for t in ordered[:n])
        composition[str(n)] = {d: counts.get(d, 0) for d in dev}
    payload = {
        "version": "inventory_v1", "study": C.STUDY, "rule": C.INVENTORY["rule"], "salt": C.INVENTORY["salt"],
        "size": len(ordered), "subsets": list(C.INVENTORY["subsets"]),
        "split_pin_sha256": pin_digest(PIN_SPLIT), "snapshots_pin_sha256": pin_digest(PIN_SNAPSHOTS),
        "gold_check": {"timeout_seconds": checked["timeout_seconds"], "executor": checked["executor"],
                       "sqlite_version": checked["sqlite_version"], "checked": checked["checked"],
                       "empty_result_in_inventory": sum(1 for t in ordered if results[t]["row_count"] == 0)},
        "excluded": [{"task_id": t, "db_id": row(t).db_id,
                      "reason": f"{v['status']}: {v['error']}"} for t, v in sorted(excluded.items())],
        "composition": composition,
        "task_ids": ordered,
        "db_ids": [row(t).db_id for t in ordered],
    }
    return write_pin(PIN_INVENTORY, payload, rewrite=rewrite) if write else payload


def smoke_train_candidates(write: bool = True, rewrite: bool = False) -> dict:
    """contracts.SMOKE_TRAIN's order over the training pool (order only); pins the first 1,024 ids."""
    pool = set(train_pool_databases())
    ordered = smoke_order([r.task_id for r in rows() if r.db_id in pool])
    payload = {"version": "smoke_train_candidates_v1", "study": C.STUDY, "rule": C.SMOKE_TRAIN["rule"],
               "salt": C.SMOKE_TRAIN["salt"], "note": "order only; the relational IR decides which compile",
               "split_pin_sha256": pin_digest(PIN_SPLIT), "pool_questions": len(ordered),
               "written": min(SMOKE_IDS_WRITTEN, len(ordered)), "task_ids": ordered[:SMOKE_IDS_WRITTEN]}
    return write_pin(PIN_SMOKE, payload, rewrite=rewrite) if write else payload


# --- the reserved public benchmark: pinned only ---------------------------------------------------------------------
def reserved_identities(path: Path = RESERVED_JSON, expected_sha256: str = C.RESERVED_PUBLIC["file_sha256"]):
    """[(question_id, db_id)] in file order. Every other field is discarded as soon as the file is parsed."""
    verify_file(path, expected_sha256)
    with open(path, "rb") as handle:
        records = json.load(handle)
    identities = [(int(record["question_id"]), str(record["db_id"])) for record in records]
    del records
    return identities


def extract_reserved_databases(db_ids: set[str], log=print) -> dict[str, dict]:
    """Extract only `<db>/<db>.sqlite` for the reserved db_ids, read-only, and hash them. Nothing is ever queried."""
    _check_archive_bytes(RESERVED_ARCHIVE, C.RESERVED_PUBLIC["database_archive"]["bytes"])
    out = {}
    with zipfile.ZipFile(RESERVED_ARCHIVE) as zf:
        members = _database_members(zf, db_ids)
        missing = sorted(d for d in db_ids if d not in members or members[d]["sqlite"] is None)
        if missing:
            raise ValueError(f"reserved databases missing from minidev.zip: {missing}")
        for db_id in sorted(db_ids):
            target = C.RESERVED_DATABASES / db_id / f"{db_id}.sqlite"
            _extract_member(zf, members[db_id]["sqlite"], target)
            target.chmod(0o444)
            out[db_id] = {"sha256": sha256_file(target), "bytes": target.stat().st_size}
            log(f"  reserved {db_id}: {out[db_id]['bytes']} bytes")
    return out


def reserved_pin(write: bool = True, rewrite: bool = False, log=print) -> dict:
    identities = reserved_identities()
    if len(identities) != C.RESERVED_PUBLIC["questions"] or len({q for q, _ in identities}) != len(identities):
        raise PinMismatch("reserved file: expected 500 distinct question ids")
    reserved_dbs = sorted({d for _, d in identities})
    training_dbs = set(database_ids())
    if training_dbs & set(reserved_dbs):
        raise AssertionError(f"reserved databases overlap the training release: {training_dbs & set(reserved_dbs)}")
    hashes = extract_reserved_databases(set(reserved_dbs), log=log)
    training_hashes = {v["sha256"] for v in _snapshot_pin()["databases"].values()}
    if training_hashes & {v["sha256"] for v in hashes.values()}:
        raise AssertionError("a reserved database file is byte-identical to a training database file")
    archive = dict(C.RESERVED_PUBLIC["database_archive"])
    archive["sha256"] = sha256_file(RESERVED_ARCHIVE)
    payload = {"version": "reserved_public_pin_v1", "study": C.STUDY,
               "note": "pinned only: no arm is run on these questions or databases, and their SQL is never read",
               "repo": C.RESERVED_PUBLIC["repo"], "revision": C.RESERVED_PUBLIC["revision"],
               "file": C.RESERVED_PUBLIC["file"], "file_sha256": C.RESERVED_PUBLIC["file_sha256"],
               "questions": len(identities), "archive": archive,
               "disjoint_from_training": {"by_db_id": True, "by_file_sha256": True},
               "databases": hashes,
               "question_ids": [q for q, _ in identities], "db_ids": [d for _, d in identities]}
    return write_pin(PIN_RESERVED, payload, rewrite=rewrite) if write else payload


# --- accessors ------------------------------------------------------------------------------------------------------
def task(task_id: str) -> C.Task:
    """What every arm receives for one request: no gold."""
    r = row(task_id)
    return r.task(split_of(r.db_id))


def tasks(split_name: str) -> list[C.Task]:
    """All tasks of one side of the split ("dev" or "train"), in release order."""
    if split_name not in ("dev", "train"):
        raise ValueError(split_name)
    return [r.task(split_name) for r in rows() if split_of(r.db_id) == split_name]


def gold(task_id: str) -> C.Gold:
    """Scorer and trainer only."""
    return row(task_id).gold()


@functools.lru_cache(maxsize=1)
def _inventory_pin() -> dict:
    return read_pin(PIN_INVENTORY)


def inventory_ids(n: int = C.INVENTORY["size"]) -> list[str]:
    """The first n ids of the pinned development inventory (n = 16, 64 or 512)."""
    ids = _inventory_pin()["task_ids"]
    if n > len(ids):
        raise ValueError(f"the inventory has {len(ids)} items")
    return list(ids[:n])


def inventory_tasks(n: int = C.INVENTORY["size"]) -> list[C.Task]:
    return [task(t) for t in inventory_ids(n)]


# --- command line ---------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scaling.data", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["extract", "snapshots", "split", "gold-check", "inventory", "smoke",
                                            "reserved", "all"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rewrite", action="store_true", help="overwrite pins that differ (review the diff!)")
    args = parser.parse_args(argv)
    steps = ["extract", "snapshots", "split", "gold-check", "inventory", "smoke", "reserved"]
    for step in steps if args.command == "all" else [args.command]:
        started = time.perf_counter()
        print(f"[{step}]")
        if step == "extract":
            extract_training_databases()
        elif step == "snapshots":
            snapshots(rewrite=args.rewrite)
        elif step == "split":
            payload = split(rewrite=args.rewrite)
            print(f"  dev {payload['dev']}")
        elif step == "gold-check":
            payload = gold_check(workers=args.workers)
            print(f"  {Counter(v['status'] for v in payload['results'].values())}")
        elif step == "inventory":
            payload = inventory(rewrite=args.rewrite)
            print(f"  {payload['size']} items, {len(payload['excluded'])} excluded")
        elif step == "smoke":
            smoke_train_candidates(rewrite=args.rewrite)
        elif step == "reserved":
            payload = reserved_pin(rewrite=args.rewrite)
            print(f"  {payload['questions']} ids, {len(payload['databases'])} databases hashed")
        print(f"  ({time.perf_counter() - started:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
