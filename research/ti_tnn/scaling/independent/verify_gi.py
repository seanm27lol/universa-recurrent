#!/usr/bin/env python3
"""Independent verifier for the TI-TNN grounding-interface study (development), the reviewer's, written apart from
the producers.

    taskset -c 0-4,10-14 nice -n 10 python verify_gi.py --runs-root RUNS [--out DIR] [--frozen-tree TREE]
        [--threads 4] [--ex-workers 3] [--no-worker] [--no-report]
    (or TI_TNN_SCALING_RUNS=RUNS instead of --runs-root; there is no machine default)

The checkout defaults to the one holding this file (--repo); outputs go to ./verify_gi_out unless --out says
otherwise; nothing is ever written under the runs root.

**What it checks** (the work order's ten items, one section each, plus the bundle):
1. inventories: the inventory rebuilt from the raw release (own split, order and gold-executability rule); dev =
   positions 0-127 = perf_v2_eval; new = positions 128-191 = gi_new_eval.json; own prior-use scan of the runs root
   (outside grounding_interface/), the evidence bundles and the pins, by task id and by question text; every run
   covers exactly its population, once.
2. permissions: the system prompts; own BM25 retrieval against every recorded `retrieved` list; the G4 and direct
   prompts rebuilt and token-counted against the recorded prompt_tokens; gold SQL absent from every prompt; Glex's
   and the decode path's inputs read from the frozen source; no oracle outside the dev diagnostic.
3. consumption: the prior's code at the frozen commit; every eval-new decode re-decoded on the CPU from the cached
   anchored states with this file's own bias vector; pointer targets of the right kind; literals and status removed
   leave the bias and the decode unchanged.
4. fallback: own parse and validation of every G4 reply; invalid records decode as G0; the analysis counts them.
5. scores: every final SQL (eval, direct, constraint, the v2 dev decode, swap) and the gold SQL re-executed on the
   pinned snapshots with this file's own EX rule.
6. timing: the timed span's order in the frozen source; per-record spans; G4 calls in their own reservation; the
   analysis's cost arithmetic; own ledger parse, windows, the work order's limit; co-tenancy logs.
7. integrity: cache files hashed by this file's own safetensors reader against the anchors; fresh states = anchors.
8. constraint: own tallies from own scores; own SQL-level same-reference and self-join counts; v1 = saved decode;
   the restricted rule re-derived from v2's push_scalar and re-decoded.
9. analysis: counts, paired comparisons, conditions, interaction, direct control, decodes, costs, swap, reading,
   against analysis.json; then `python -m scaling.gi_report` (producer code, a separate process, read-only).
10. dev: the gate, the dev EX table, the G0 reproduction, the record validity, the git order of the freeze.

**Discipline.** No producer module is imported by this process. Producer constants and sources are read from git
(`git show <commit>:path`) and parsed with ast. Exceptions, flagged in the output (section "producer_code"):
- the worker (`--worker items|decode`), a separate process importing the frozen worktree's producer modules, for
  what cannot be done otherwise: the schema serialization and context reconstruction (schema items and the
  prompt's schema text), reading cached states into the producer's encoding, loading checkpoints and decoding.
  The bias vectors it adds are built here, from the recorded records and the worker's item list; the restricted
  machine is this file's own (v2's push_scalar run on a v1 machine);
- `python -m scaling.gi_report`, run once on the bundle as the brief asks.
Third-party: sqlite3, sqlglot (SQL parsing for the diagnostics), transformers' tokenizer (prompt token counts).

Output: <out>/verify_gi.json, {"sections": {name: {"checks": [{name, passed, detail}], "failures": n,
"info": {...}}}, "checks": total, "failures": total}, and <out>/verify_gi_summary.txt.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import subprocess
import sys
import time
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[4]
H = "research/ti_tnn/scaling/"
RAW_RELEASE = "data/raw/hf/bird23_train_filtered.train-00000-of-00001.jsonl"
RELEASE_KEY = "data/train-00000-of-00001.jsonl"
FREEZE = "fb4e342"         # the protocol's frozen_at (the brief)
FROZEN_RUN = "9c139ac"     # the commit the new population ran at (the brief)
GATE_COMMIT = "e759e58"    # the gate rule must be committed here or earlier (the brief)
BUNDLE_COMMIT = "a16168e"  # the bundle's commit (the brief)
WORK_ORDER_LIMIT = 29.416  # the brief's limit (gi_run: min(32, 21.415907 + 8) = 29.415907)
DEV_DECODE = "20261103T193217Z"
V2_FITS = [f"{a}-v2/{s}" for s in (100, 101, 102) for a in ("T4", "C4")]
V1_FITS = [f"{a}-v1/{s}" for a in ("C4", "T4") for s in (100, 101, 102)]
POLICIES = ("G0", "Glex", "G4")
LN2 = math.log(2.0)
NEW_TIMED = ("gi encode-new", "gi ground new", "gi eval-new", "gi direct-new")


# --- the check list ------------------------------------------------------------------------------------------------
class Section:
    def __init__(self, name: str):
        self.name, self.checks, self.info = name, [], {}

    def check(self, ok, name: str, detail="") -> bool:
        ok = bool(ok)
        self.checks.append({"name": name, "passed": ok, "detail": "" if ok and not detail else detail})
        return ok

    def as_dict(self) -> dict:
        return {"checks": self.checks, "passed": sum(c["passed"] for c in self.checks),
                "failures": sum(not c["passed"] for c in self.checks), "info": self.info}


def git(repo: Path, *args, check=True) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=check).stdout


def git_show(repo: Path, commit: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=repo, capture_output=True, check=True).stdout


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_path(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def jl(path: Path) -> list:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def order_key(salt: str, item: str):
    return (hashlib.sha256(f"{salt}:{item}".encode()).hexdigest(), item)


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def run_time(name: str) -> float:
    """A run directory's UTC start time from its name (20261103T195032Z[_new])."""
    return datetime.strptime(name[:16], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()


def literals_of(source: bytes, names) -> dict:
    """Module-level literal assignments, by name (ast.literal_eval)."""
    out = {}
    for node in ast.parse(source).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) \
            else []
        for t in targets:
            if isinstance(t, ast.Name) and t.id in names:
                with contextlib.suppress(ValueError):
                    out[t.id] = ast.literal_eval(node.value)
            if isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple):
                for a, b in zip(t.elts, node.value.elts):
                    if isinstance(a, ast.Name) and a.id in names:
                        with contextlib.suppress(ValueError):
                            out[a.id] = ast.literal_eval(b)
    return out


def func_source(source: bytes, name: str, cls: "str | None" = None) -> str:
    text = source.decode()
    tree = ast.parse(text)
    nodes = tree.body
    if cls is not None:
        nodes = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    return ast.get_source_segment(text, node)


def imports_of(source: bytes) -> set:
    out = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            out |= {f"{'.' * node.level}{node.module or ''}:{a.name}" for a in node.names}
        elif isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
    return out


def names_in(text: str) -> set:
    return {n.id for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Name)} | \
        {n.attr for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Attribute)}


def close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


# --- own EX rule (the official mini_dev rule: set(rows) equality, one deadline, predicted first) -------------------------
def connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect("file:" + urllib.parse.quote(str(Path(path).resolve())) + "?mode=ro", uri=True)


def ex(path: str, predicted, gold: str, timeout: float) -> dict:
    """Predicted, then gold, on one read-only connection under one deadline. {correct, status, pred, gold_status,
    gold_rows, seconds}. status: correct | incorrect | abstain | timeout | error | gold_error."""
    if predicted is None:
        return {"correct": 0, "status": "abstain", "pred": "abstain", "gold_status": "not_run", "seconds": 0.0}
    conn = connect(Path(path))
    t0 = time.perf_counter()
    deadline = t0 + timeout
    conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 1000)
    sides = []
    try:
        for sql in (predicted, gold):
            if time.perf_counter() > deadline:
                sides.append(("timeout", None))
                continue
            try:
                cur = conn.cursor()
                cur.execute(sql)
                rows = cur.fetchall()
                sides.append(("timeout", None) if time.perf_counter() > deadline else ("ok", rows))
            except Exception as e:  # the official harness scores any exception 0
                sides.append(("timeout" if time.perf_counter() > deadline else "error", None))
    finally:
        conn.close()
    dt = time.perf_counter() - t0
    (ps, pr), (gs, gr) = sides
    if ps != "ok":
        status = ps
    elif gs != "ok":
        status = "gold_" + gs
    elif dt > timeout:
        status = "timeout"
    else:
        status = "correct" if set(pr) == set(gr) else "incorrect"
    return {"correct": int(status == "correct"), "status": status, "pred": ps, "gold_status": gs,
            "gold_rows": len(gr) if gr is not None else None, "seconds": dt}


def _ex_job(job):
    return ex(*job)


def ex_batch(jobs: dict, workers: int) -> dict:
    if not jobs:
        return {}
    keys = list(jobs)
    if workers <= 1:
        return {k: _ex_job(jobs[k]) for k in keys}
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        res = list(pool.map(_ex_job, [jobs[k] for k in keys], chunksize=1))
    return dict(zip(keys, res))


def status_class(s) -> str:
    s = str(s)
    if s in ("correct", "incorrect", "abstain", "ok"):
        return s
    if "timeout" in s:
        return "timeout"
    if "error" in s:
        return "error"
    return s


# --- the raw release, the split and the inventory (own reconstruction) ------------------------------------------------
class Release:
    def __init__(self, runs: Path, repo: Path, commit: str):
        self.runs = runs
        self.contracts = literals_of(git_show(repo, commit, H + "contracts.py"),
                                     ("SPLIT", "INVENTORY", "TRAIN_RELEASE", "OFFICIAL_SCORER"))
        raw = (runs / RAW_RELEASE).read_bytes()
        self.raw_sha256 = sha256_bytes(raw)
        self.raw_ok = self.raw_sha256 == self.contracts["TRAIN_RELEASE"]["files"][RELEASE_KEY]
        lines = raw.decode("utf-8").split("\n")
        lines = lines[:-1] if lines and lines[-1] == "" else lines
        rows = [json.loads(x) for x in lines]
        self.ids = [f"bird23tf-{i:05d}" for i in range(len(rows))]
        self.by_id = {t: {k: str(r[k]) for k in ("db_id", "question", "evidence", "SQL")} for t, r in zip(self.ids, rows)}
        sp = self.contracts["SPLIT"]
        dbs = sorted({r["db_id"] for r in rows}, key=lambda d: order_key(sp["salt"], d))
        n_dev = round(sp["dev_fraction"] * len(dbs))
        self.dev, self.pool = dbs[:n_dev], set(dbs[n_dev:])
        self.snaps = json.loads(git_show(repo, commit, H + "pins/snapshots_v1.json"))["databases"]
        self.timeout = float(self.contracts["OFFICIAL_SCORER"]["meta_time_out_seconds"])

    def db_path(self, db: str) -> Path:
        return self.runs / "data/databases" / self.snaps[db]["file"]

    def inventory(self, n: int, cache: dict, cache_path: Path) -> tuple:
        """The first n of the development inventory: within each development database by sha256(salt:task_id),
        one question per database per round in split order, skipping a question whose gold does not execute
        within the timeout (own execution)."""
        salt = self.contracts["INVENTORY"]["salt"]
        queues = {d: sorted((t for t in self.ids if self.by_id[t]["db_id"] == d), key=lambda t: order_key(salt, t))
                  for d in self.dev}
        ptr, order, excluded = {d: 0 for d in self.dev}, [], {}
        while len(order) < n and any(ptr[d] < len(queues[d]) for d in self.dev):
            for d in self.dev:
                while ptr[d] < len(queues[d]):
                    t = queues[d][ptr[d]]
                    ptr[d] += 1
                    key = f"{t}:{self.snaps[d]['sha256']}:{self.timeout}"
                    if key not in cache:
                        t0 = time.perf_counter()
                        r = ex(str(self.db_path(d)), self.by_id[t]["SQL"], "SELECT 1", self.timeout)
                        cache[key] = {"status": r["pred"], "seconds": round(time.perf_counter() - t0, 3)}
                        cache_path.write_text(json.dumps(cache, indent=0, sort_keys=True))
                    if cache[key]["status"] == "ok":
                        order.append(t)
                        break
                    excluded[t] = cache[key]
                if len(order) == n:
                    break
        return order, excluded


# --- own BM25 retrieval (the retrieval docstring's specification; the stopword list read with ast) -------------------
class BM25:
    def __init__(self, rel: Release, stopwords: set, k1: float, b: float):
        self.stop, self.k1, self.b = stopwords, k1, b
        self.docs = sorted((t for t in rel.ids if rel.by_id[t]["db_id"] in rel.pool))
        self.db = {t: rel.by_id[t]["db_id"] for t in self.docs}
        self.lengths, self.post = [], {}
        for pos, t in enumerate(self.docs):
            r = rel.by_id[t]
            c = Counter(self.tokens(f"{r['question']} {r['evidence']}"))
            self.lengths.append(sum(c.values()))
            for term, tf in sorted(c.items()):
                self.post.setdefault(term, []).append((pos, tf))
        n = len(self.docs)
        self.avgdl = sum(self.lengths) / n
        self.idf = {t: math.log(1.0 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def tokens(self, text: str) -> list:
        return [t for t in re.findall(r"[^\W_]+", (text or "").lower(), flags=re.UNICODE) if t not in self.stop]

    def top(self, question: str, evidence: str, k: int, task_id: str, db_id: str) -> list:
        scores = {}
        for term in sorted(set(self.tokens(f"{question} {evidence}"))):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for pos, tf in self.post[term]:
                norm = self.k1 * (1.0 - self.b + self.b * self.lengths[pos] / self.avgdl)
                scores[pos] = scores.get(pos, 0.0) + idf * tf * (self.k1 + 1.0) / (tf + norm)
        cand = sorted((-scores.get(p, 0.0), t) for p, t in enumerate(self.docs) if t != task_id and self.db[t] != db_id)
        return [t for _s, t in cand[:k]]


# --- own record parser and validator (the protocol's parser and validation rules) -------------------------------------
QUOTE_PAIRS = (("`", "`"), ('"', '"'), ("[", "]"), ("'", "'"))
KEYS = {"tables", "columns", "literals", "status"}


def own_clean(name: str) -> str:
    n = name.strip()
    for a, b in QUOTE_PAIRS:
        if len(n) >= 2 and n.startswith(a) and n.endswith(b):
            return n[1:-1].strip()
    return n


def own_parse(text: str):
    """The last ```-fenced block (optionally ```json) or the whole reply, as JSON; then with backslashes that start
    no JSON escape removed; each candidate also as its outermost braces."""
    text = text or ""
    blocks = re.findall(r"```(?:json)?\s*\n?(.*?)```", text, flags=re.S | re.I)
    body = (blocks[-1] if blocks else text).strip()
    for cand in (body, re.sub(r'\\(?![\\"/bfnrtu])', "", body)):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            i, j = cand.find("{"), cand.rfind("}")
            if 0 <= i < j:
                try:
                    return json.loads(cand[i:j + 1])
                except json.JSONDecodeError:
                    pass
    return None


def own_validate(obj, items: list, question: str, evidence: str, truncated: bool) -> dict:
    """{valid, reason, tables, columns (item indexes, first occurrences), literals_ok, literals_bad}."""
    def bad(reason):
        return {"valid": False, "reason": reason}
    if truncated:
        return bad("truncated")
    if obj is None:
        return bad("not JSON")
    if not isinstance(obj, dict):
        return bad("not an object")
    if set(obj) != KEYS:
        return bad("keys")
    if not isinstance(obj["tables"], list) or any(not isinstance(t, str) for t in obj["tables"]):
        return bad("tables type")
    if not isinstance(obj["columns"], list) or any(
            not isinstance(c, dict) or set(c) != {"table", "column"} or any(not isinstance(v, str) for v in c.values())
            for c in obj["columns"]):
        return bad("columns type")
    if not isinstance(obj["literals"], list) or any(not isinstance(x, dict) for x in obj["literals"]):
        return bad("literals type")
    if obj["status"] not in ("ok", "unresolved"):
        return bad("status")
    tab = {it[1].lower(): i for i, it in enumerate(items) if it[0] == "table"}
    col = {(it[1].lower(), it[2].lower()): i for i, it in enumerate(items) if it[0] == "column"}
    ts, cs = [], []
    for t in obj["tables"]:
        i = tab.get(own_clean(t).lower())
        if i is None:
            return bad("unknown table")
        if i not in ts:
            ts.append(i)
    for c in obj["columns"]:
        i = col.get((own_clean(c["table"]).lower(), own_clean(c["column"]).lower()))
        if i is None:
            return bad("unknown column")
        if i not in cs:
            cs.append(i)
    text = f"{question}\n{evidence or ''}".lower()

    def occurs(v):
        return v is not None and bool(str(v).strip()) and str(v).strip().lower() in text
    ok = badl = 0
    for x in obj["literals"]:
        c = col.get((own_clean(str(x.get("table", ""))).lower(), own_clean(str(x.get("column", ""))).lower()))
        good = set(x) == {"span", "value", "table", "column"} and occurs(x.get("span")) and occurs(x.get("value")) \
            and c is not None
        ok, badl = ok + good, badl + (not good)
    return {"valid": True, "reason": None, "tables": ts, "columns": cs, "literals_ok": ok, "literals_bad": badl,
            "status": obj["status"]}


def own_biases(tables: list, columns: list, n: int) -> list:
    """The protocol's prior: rank r (1-based, first occurrences) gains ln 2 / r; at most 8 tables, 16 columns."""
    out = [0.0] * n
    for ranked, cap in ((tables, 8), (columns, 16)):
        for r, i in enumerate(ranked[:cap], 1):
            out[i] = LN2 / r
    return out


def reason_class(reason) -> str:
    r = str(reason or "")
    for key, cls in (("truncated", "truncated"), ("not JSON", "not JSON"), ("not a JSON object", "not an object"),
                     ("keys", "keys"), ("tables is not", "tables type"), ("columns is not", "columns type"),
                     ("literals is not", "literals type"), ("status", "status"), ("unknown table", "unknown table"),
                     ("unknown column", "unknown column")):
        if key in r:
            return cls
    return r


# --- own SQL diagnostics (sqlglot) --------------------------------------------------------------------------------
def sql_tables(sql: str) -> "set | None":
    import sqlglot
    from sqlglot import exp
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    return {t.name.lower() for t in tree.find_all(exp.Table) if t.name and t.name.lower() not in ctes}


def self_join_selects(sql: str) -> "int | None":
    """SELECTs whose own FROM/JOIN sources name one base table twice or more."""
    import sqlglot
    from sqlglot import exp
    if sql is None:
        return 0
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    n = 0
    for sel in tree.find_all(exp.Select):
        names = []
        frm = sel.args.get("from") or sel.args.get("from_")
        srcs = ([frm.this] if frm is not None else []) + [j.this for j in sel.args.get("joins") or []]
        for s in srcs:
            if isinstance(s, exp.Table):
                names.append(s.name.lower())
        n += len(names) != len(set(names))
    return n


def same_reference_on(sql: str) -> "int | None":
    """ON-clause comparisons (=, <>, <, <=, >, >=) of one column reference with itself (case and quoting ignored)."""
    import sqlglot
    from sqlglot import exp
    if sql is None:
        return 0
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    n = 0
    for j in tree.find_all(exp.Join):
        on = j.args.get("on")
        if on is None:
            continue
        for c in on.find_all(exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE):
            a, b = c.left, c.right
            if isinstance(a, exp.Column) and isinstance(b, exp.Column) and \
                    (a.table.lower(), a.name.lower()) == (b.table.lower(), b.name.lower()):
                n += 1
    return n


# --- own safetensors reader ---------------------------------------------------------------------------------------
def safetensors_digest(path: Path, name="states") -> tuple:
    """(sha256 of the tensor's raw bytes, dtype, shape, metadata) read with this file's own header parser."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header.get("__metadata__") or {}
        t = header[name]
        a, b = t["data_offsets"]
        f.seek(8 + n + a)
        h, left = hashlib.sha256(), b - a
        while left:
            chunk = f.read(min(left, 1 << 24))
            if not chunk:
                break
            h.update(chunk)
            left -= len(chunk)
    return h.hexdigest(), t["dtype"], t["shape"], meta


# --- own ledger parser (ledger.py's documented charge rule) ------------------------------------------------------------
def parse_ledger(path: Path, heartbeat=30.0) -> tuple:
    res, order, problems = {}, [], []
    for n, line in enumerate(path.read_text().split("\n"), 1):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError as err:
            problems.append(f"line {n}: not JSON ({err})")
            continue
        rid, ev = e.get("reservation"), e.get("event")
        if ev == "start":
            if rid in res:
                problems.append(f"line {n}: second start {rid}")
                continue
            res[rid] = {"start": e, "beats": [], "end": None}
            order.append(rid)
        elif ev in ("heartbeat", "end"):
            r = res.get(rid)
            if r is None:
                problems.append(f"line {n}: {ev} before start {rid}")
                continue
            if r["end"] is not None:
                problems.append(f"line {n}: {ev} after end {rid}")
            if ev == "heartbeat":
                r["beats"].append(e)
            else:
                r["end"] = e
        else:
            problems.append(f"line {n}: unknown event {ev!r}")
    out = []
    for rid in order:
        r = res[rid]
        st, beats, end = r["start"], r["beats"], r["end"]
        el = [float(b["elapsed"]) for b in beats]
        if el != sorted(el):
            problems.append(f"{rid}: heartbeats not monotone")
        if end is not None:
            seconds, t_end = float(end["elapsed"]), end["t"]
        else:
            seconds = (el[-1] if el else 0.0) + heartbeat
            t_end = (beats[-1]["t"] if beats else st["t"]) + heartbeat
        out.append({"id": rid, "command": st.get("command"), "start": st, "end": end, "seconds": seconds,
                    "hours": seconds / 3600.0, "t_start": st["t"], "t_end": t_end, "open": end is None,
                    "beats": len(beats)})
    return out, problems


def cotenancy(path: Path) -> list:
    """[(t, load1, running, total)] from a co-tenancy log ('ISO load1 load5 load15 running/total lastpid')."""
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 5 and "/" in p[4]:
            with contextlib.suppress(ValueError):
                t = datetime.strptime(p[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
                run, tot = p[4].split("/")
                out.append((t, float(p[1]), int(run), int(tot)))
    return out


# --- own paired difference (the protocol's three conditions) -----------------------------------------------------------
def own_paired(correct: dict, a: str, b: str, seeds: list, ids: list, db_of: dict, threshold=10) -> dict:
    per_seed = {s: sum(correct[(a, s)][t] - correct[(b, s)][t] for t in ids) for s in seeds}
    pooled = sum(per_seed.values())
    dbs = sorted({db_of[t] for t in ids})
    per_db = {d: sum(correct[(a, s)][t] - correct[(b, s)][t] for s in seeds for t in ids if db_of[t] == d) for d in dbs}
    loo = {d: pooled - per_db[d] for d in dbs}
    sign = (pooled > 0) - (pooled < 0)
    cond = {"pooled_at_least_threshold": abs(pooled) >= threshold,
            "same_sign_every_seed": sign != 0 and all(v * sign > 0 for v in per_seed.values()),
            "same_sign_without_any_one_database": sign != 0 and all(v * sign > 0 for v in loo.values())}
    hist = Counter(sum(correct[(a, s)][t] - correct[(b, s)][t] for s in seeds) for t in ids)
    return {"per_seed": {str(s): v for s, v in per_seed.items()}, "pooled": pooled, "sign": sign, "conditions": cond,
            "all_conditions": all(cond.values()), "per_database": per_db, "leave_one_database_out": loo,
            "per_question_histogram": {str(k): hist[k] for k in sorted(hist)}, "pooled_out_of": len(ids) * len(seeds)}


# === the worker (a separate process; producer code from the frozen worktree; flagged) ==================================
def worker_main(args) -> int:
    """Runs in its own process with the frozen worktree's research/ti_tnn on sys.path. Writes only --worker-out."""
    sys.path.insert(0, args.producer_tree)
    import torch
    torch.set_num_threads(args.threads)
    from scaling import actions as A
    from scaling import contracts as C
    from scaling import controller as K
    from scaling import data, encode, infer_controller, pilot_run, retrieval, train
    from scaling import gi_record as GR
    from scaling import schema as schema_mod
    job = json.loads(Path(args.worker_in).read_text())
    tok = pilot_run._tokenizer()
    reader = encode.Backbone.cache_reader(job["backbone_identity"], tok)
    if args.worker == "items":
        out = {"tasks": {}, "schema_text": {}}
        for tid in job["task_ids"]:
            task = data.task(tid)
            sch = schema_mod.load_schema(task.db_id)
            exs = retrieval.retrieve(task)
            ctx, ser, ids, _off = encode.fit_context(task, sch, exs, tok, C.Limits().max_context_tokens)
            plain = schema_mod.serialize(sch)
            items = [[it.kind, it.table, it.column] for it in ser.items]
            out["tasks"][tid] = {"db_id": task.db_id, "items": items,
                                 "items_equal_unbudgeted": items == [[i.kind, i.table, i.column] for i in plain.items],
                                 "retrieved": [e.task_id for e in exs], "cache_key": reader.cache_key(ids),
                                 "tokens": len(ids), "examples_used": ctx.examples_used}
            if task.db_id not in out["schema_text"]:
                out["schema_text"][task.db_id] = schema_mod.serialize(
                    sch, budget_chars=schema_mod.PROMPT_BUDGET_CHARS).text
        Path(args.worker_out).write_text(json.dumps(out))
        return 0

    # decode
    models, results = {}, []
    orig_push_scalar = A._NodeFrame.push_scalar

    def own_push_scalar(self, m, **kw):
        # the restriction re-derived from v2 itself: v2's push_scalar condition, evaluated on a v1 machine
        if getattr(m, "_own_restricted", False) and m.grounding == "v1":
            m.grounding = "v2"
            try:
                return orig_push_scalar(self, m, **kw)
            finally:
                m.grounding = "v1"
        return orig_push_scalar(self, m, **kw)

    orig_init = A.Machine.__init__

    @contextlib.contextmanager
    def own_restricted():
        def init(self, *a, **kw):
            orig_init(self, *a, **kw)
            self._own_restricted = True
        A.Machine.__init__ = init
        A._NodeFrame.push_scalar = own_push_scalar
        try:
            yield
        finally:
            A.Machine.__init__ = orig_init
            A._NodeFrame.push_scalar = orig_push_scalar

    @contextlib.contextmanager
    def own_prior(model, biases):
        if biases is None:
            yield
            return
        trunk = model.trunk
        orig = trunk.pointer_scores
        bt = torch.tensor(biases, dtype=torch.float32)

        def wrapped(mem, queries, start_index=None, relations=None):
            items, start, end = orig(mem, queries, start_index, relations)
            if items.shape[-1] != bt.shape[0]:
                raise ValueError("bias length")
            return items + bt.to(items.dtype), start, end
        trunk.pointer_scores = wrapped
        try:
            yield
        finally:
            del trunk.pointer_scores

    by_task = defaultdict(list)
    for j in job["jobs"]:
        by_task[j["task_id"]].append(j)
    t0 = time.time()
    for n, (tid, js) in enumerate(by_task.items()):
        task = data.task(tid)
        sch = schema_mod.load_schema(task.db_id)
        enc = encode.encode_request(reader, task, sch, retrieval.retrieve(task), cache_root=encode.PILOT_CACHE,
                                    require_hit=True, require_verified=True)
        preps = {}
        for j in js:
            if j["ckpt"] not in models:
                models[j["ckpt"]] = train.model_from_checkpoint(train.load_checkpoint(j["ckpt"]), "cpu").eval()
            model = models[j["ckpt"]]
            if j["ckpt"] not in preps:
                preps[j["ckpt"]] = K.prepare(enc, grounding=model.cfg.grounding)
            prep, extra = preps[j["ckpt"]], {}
            prec = j.get("precision", "highest")
            if j["kind"] == "bias":
                with own_prior(model, j["biases"]):
                    dec = infer_controller._decode_at(prec, model, prep, infer_controller.DEFAULT_MAX_STEPS)
            elif j["kind"] == "producer":   # the producer's own resolve, biases and pointer_prior (flagged)
                g = GR.resolve(j["record"], enc.schema, task.question, task.evidence,
                               truncated=j.get("finish_reason", "eos") != "eos")
                b = g.biases(len(enc.schema.items)) if g.valid else None
                extra["producer_bias_nonzero"] = None if b is None else {str(i): v for i, v in enumerate(b) if v}
                with GR.pointer_prior(model, b):
                    dec = infer_controller._decode_at(prec, model, prep, infer_controller.DEFAULT_MAX_STEPS)
            elif j["kind"] == "margin":   # the item scores of every operator step, for a CPU/GPU difference
                calls = []
                trunk = model.trunk
                orig_ps = trunk.pointer_scores

                def rec_ps(mem, queries, start_index=None, relations=None, _o=orig_ps, _c=calls, _b=j["biases"]):
                    items_, start_, end_ = _o(mem, queries, start_index, relations)
                    if _b is not None:
                        items_ = items_ + torch.tensor(_b, dtype=items_.dtype)
                    if start_index is None:
                        _c.append(items_[0, 0].detach().clone())
                    return items_, start_, end_
                trunk.pointer_scores = rec_ps
                try:
                    dec = infer_controller._decode_at(prec, model, prep, infer_controller.DEFAULT_MAX_STEPS)
                finally:
                    del trunk.pointer_scores
                k, row = j["step"], 0 if j["pointer"] == "table" else 1
                sc = calls[k][row]
                extra["margin"] = {"cpu_choice": j["cpu_choice"], "gpu_choice": j["gpu_choice"],
                                   "score_cpu_choice": float(sc[j["cpu_choice"]]),
                                   "score_gpu_choice": float(sc[j["gpu_choice"]]),
                                   "score_scale": float(sc.abs().max())}
            elif j["kind"] == "restricted_own":
                with own_restricted():
                    dec = infer_controller._decode_at(prec, model, prep, infer_controller.DEFAULT_MAX_STEPS)
            else:
                dec = infer_controller._decode_at(prec, model, prep, infer_controller.DEFAULT_MAX_STEPS)
            rec = dec.as_record()
            results.append({"id": j["id"], "actions": rec["actions"], "sql": rec["final"]["sql"],
                            "kind": rec["final"]["kind"], "reason": rec["final"].get("reason"),
                            "states_sha256": enc.cost.get("states_sha256"), "cache_key": enc.cost.get("cache_key"),
                            **extra})
        print(json.dumps({"task": n + 1, "of": len(by_task), "seconds": round(time.time() - t0)}), flush=True)
    Path(args.worker_out).write_text(json.dumps({"results": results}))
    return 0


def run_worker(ctx: dict, mode: str, payload: dict, tag: str) -> dict:
    """The worker in a separate process, cached in the output directory by its input's hash."""
    out = ctx["out"] / "worker"
    out.mkdir(parents=True, exist_ok=True)
    import inspect
    blob = json.dumps(payload, sort_keys=True)
    code = sha256_bytes(inspect.getsource(worker_main).encode())   # a changed worker never reuses a cached result
    key = sha256_bytes(f"{mode}:{ctx['frozen_tree']}:{code}:{blob}".encode())[:16]
    inp, res = out / f"{tag}_{key}.in.json", out / f"{tag}_{key}.out.json"
    if res.exists():
        return json.loads(res.read_text())
    inp.write_text(blob)
    env = {**os.environ, "TI_TNN_SCALING_RUNS": str(ctx["runs"]), "CUDA_VISIBLE_DEVICES": "",
           "OMP_NUM_THREADS": str(ctx["threads"]), "MKL_NUM_THREADS": str(ctx["threads"]),
           "TOKENIZERS_PARALLELISM": "false"}
    cmd = [ctx["python"], str(Path(__file__).resolve()), "--worker", mode, "--worker-in", str(inp), "--worker-out",
           str(res), "--producer-tree", str(Path(ctx["frozen_tree"]) / "research/ti_tnn"), "--threads",
           str(ctx["threads"])]
    t0 = time.time()
    with open(out / f"{tag}_{key}.log", "w") as log:
        p = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(out))
    ctx["producer_runs"].append({"tag": tag, "mode": mode, "cmd": " ".join(cmd[1:4]), "exit": p.returncode,
                                 "seconds": round(time.time() - t0, 1)})
    if p.returncode != 0:
        raise RuntimeError(f"worker {tag} failed; see {out / f'{tag}_{key}.log'}")
    return json.loads(res.read_text())


# === sections ===========================================================================================================
SECTIONS = {}


def section(fn):
    SECTIONS[fn.__name__.removeprefix("sec_")] = fn
    return fn


def gi(ctx) -> Path:
    return ctx["runs"] / "grounding_interface"


def only_run(sec: Section, ctx: dict, kind: str, suffix: str) -> Path:
    found = sorted(d for d in (gi(ctx) / kind).glob(f"*{suffix}") if d.is_dir())
    sec.check(len(found) == 1, f"exactly one {kind} run *{suffix}", f"{[d.name for d in found]}")
    return found[-1]


@section
def sec_bundle(sec: Section, ctx: dict) -> None:
    repo, bundle = ctx["repo"], ctx["bundle"]
    sums = {}
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            h, p = line.split(None, 1)
            sums[p.strip()] = h
    files = {str(f.relative_to(bundle)) for f in bundle.rglob("*") if f.is_file()}
    sec.check(files == set(sums) | {"SHA256SUMS"}, "bundle: files are exactly SHA256SUMS's list",
              f"{sorted(files ^ (set(sums) | {'SHA256SUMS'}))[:5]}")
    badsum = [p for p, h in sums.items() if not (bundle / p).exists() or sha256_path(bundle / p) != h]
    sec.check(not badsum, f"bundle: all {len(sums)} files hash to SHA256SUMS", f"{badsum[:5]}")
    run_files = {str(f.relative_to(gi(ctx))) for f in gi(ctx).rglob("*") if f.is_file()}
    sec.check(run_files == set(sums) - {"gpu_ledger.jsonl"}, "bundle: the runs root's grounding_interface/ files, "
              "exactly", f"{sorted(run_files ^ (set(sums) - {'gpu_ledger.jsonl'}))[:5]}")
    diff = [p for p in run_files & set(sums) if (gi(ctx) / p).read_bytes() != (bundle / p).read_bytes()]
    sec.check(not diff, "bundle: every bundled run file equals its runs-root original byte for byte", f"{diff[:5]}")
    led_run, led_b = (ctx["runs"] / "gpu_ledger.jsonl").read_bytes(), (bundle / "gpu_ledger.jsonl").read_bytes()
    sec.check(led_run.startswith(led_b), "bundle: the bundled ledger is a prefix of the live ledger (append-only)",
              f"{len(led_b)} vs {len(led_run)} bytes")
    sec.info["live_ledger_grew_since_bundle_bytes"] = len(led_run) - len(led_b)
    changed = git(repo, "diff", "--name-only", BUNDLE_COMMIT, "HEAD", "--", H + "evidence/grounding_interface_v1")
    sec.check(not changed.strip(), f"bundle unchanged from {BUNDLE_COMMIT} to HEAD", changed[:300])
    gi_changed = git(repo, "diff", "--name-only", FROZEN_RUN, "HEAD", "--", *[H + f for f in (
        "gi_constraint.py", "gi_record.py", "gi_lexical.py", "gi_llm.py", "gi_run.py", "pins/gi_protocol.json",
        "pins/gi_new_eval.json", "docs/PROTOCOL_grounding_interface.md")])
    sec.check(not gi_changed.strip(), f"study code, pins and protocol unchanged from {FROZEN_RUN} to HEAD",
              gi_changed[:300])
    sec.info["head"] = git(repo, "rev-parse", "--short", "HEAD").strip()
    ft = Path(ctx["frozen_tree"]) if ctx["frozen_tree"] else None
    if ft and ft.exists():
        head = git(ft, "rev-parse", "HEAD").strip()
        sec.check(head.startswith(FROZEN_RUN), f"frozen worktree at {FROZEN_RUN}", head)
        dirty = git(ft, "status", "--porcelain", "--", "research").strip()
        sec.check(not dirty, "frozen worktree: research/ clean", dirty[:300])


@section
def sec_pins(sec: Section, ctx: dict) -> None:
    """Item 1 (populations): the inventory rebuilt; dev and new."""
    repo, rel = ctx["repo"], ctx["rel"]
    pin = json.loads((repo / H / "pins/gi_protocol.json").read_text())
    proto_b = (repo / H / "docs/PROTOCOL_grounding_interface.md").read_bytes()
    newpin_b = (repo / H / "pins/gi_new_eval.json").read_bytes()
    sec.check(sha256_bytes(proto_b) == pin["sha256"], "protocol text hashes to the pin's sha256", pin["sha256"])
    sec.check(sha256_bytes(newpin_b) == pin["new_eval_pin_sha256"], "gi_new_eval.json hashes to the pin's sha256")
    sec.check(pin["frozen_at"].startswith(FREEZE), f"the pin's frozen_at is {FREEZE}", pin["frozen_at"])
    sec.check(sha256_bytes(git_show(repo, FREEZE, H + "docs/PROTOCOL_grounding_interface.md")) == pin["sha256"] and
              git_show(repo, FREEZE, H + "pins/gi_new_eval.json") == newpin_b,
              f"protocol text and new pin at {FREEZE} equal the pinned ones")
    sec.check(git(repo, "rev-parse", f"{FROZEN_RUN}^").strip().startswith(FREEZE) and
              git_show(repo, FROZEN_RUN, H + "pins/gi_protocol.json") == (repo / H / "pins/gi_protocol.json").read_bytes(),
              f"{FROZEN_RUN} (the run commit) is {FREEZE}'s child and holds this protocol pin")
    sec.check(rel.raw_ok, "raw release sha256 equals contracts.TRAIN_RELEASE", rel.raw_sha256)
    sec.check(len(rel.dev) == 14 and len(rel.pool) == 55, "own split: 14 development, 55 pool databases")
    split = json.loads((repo / H / "pins/split_v1.json").read_text())
    sec.check(split["dev"] == rel.dev, "own split order equals split_v1.json's dev list")
    inv = json.loads((repo / H / "pins/inventory_v1.json").read_text())
    cache_path = ctx["out"] / "cache" / "gold_exec.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    mine, excl = rel.inventory(192, cache, cache_path)
    sec.info["own_inventory_excluded_within_192"] = excl
    sec.check(mine == inv["task_ids"][:192], "own inventory rebuild: positions 0-191 equal inventory_v1.json",
              f"first difference at {next((i for i, (a, b) in enumerate(zip(mine, inv['task_ids'])) if a != b), None)}")
    sec.check([rel.by_id[t]["db_id"] for t in mine] == inv["db_ids"][:192], "inventory db_ids 0-191 equal")
    pinned_excl = {e["task_id"] for e in inv["excluded"]}
    sec.check(set(excl) <= pinned_excl, "own exclusions within 0-191 are the pin's", f"{sorted(excl)} vs {pinned_excl}")
    ev = json.loads((repo / H / "pins/perf_v2_eval.json").read_text())
    dev = ev["populations"]["historical"]["task_ids"] + ev["populations"]["new"]["task_ids"]
    sec.check(dev == mine[:128], "dev = perf_v2_eval historical + new = own inventory positions 0-127")
    new = json.loads(newpin_b)
    sec.check(new["task_ids"] == mine[128:192], "new = own inventory positions 128-191, in order")
    sec.check(new["positions"] == [128, 192] and new["inventory"] == rel.contracts["INVENTORY"]["salt"],
              "gi_new_eval: positions [128, 192) of the pinned inventory")
    sec.check(new["db_ids"] == [rel.by_id[t]["db_id"] for t in new["task_ids"]], "gi_new_eval db_ids from the release")
    sec.check(not set(dev) & set(new["task_ids"]) and len(set(new["task_ids"])) == 64 and len(set(dev)) == 128,
              "dev (128) and new (64) are disjoint and duplicate-free")
    sec.check(all(rel.by_id[t]["db_id"] in rel.dev for t in dev + new["task_ids"]), "all 192 from development databases")
    # the gold SQL of every evaluated question executes (own executor)
    gold_ok = [t for t in dev + new["task_ids"] if cache.get(f"{t}:{rel.snaps[rel.by_id[t]['db_id']]['sha256']}:"
                                                              f"{rel.timeout}", {}).get("status") == "ok"]
    sec.check(len(gold_ok) == 192, "every evaluated question's gold SQL executes within 30 s (own run)",
              f"{len(gold_ok)}")
    # the pinned snapshots
    snap_bad = [d for d in rel.dev if sha256_path(rel.db_path(d)) != rel.snaps[d]["sha256"]]
    sec.check(not snap_bad, "the 14 development snapshots hash to snapshots_v1.json", f"{snap_bad}")
    ctx["dev_ids"], ctx["new_ids"] = dev, new["task_ids"]
    ctx["pop_of"] = {**{t: "historical" for t in ev["populations"]["historical"]["task_ids"]},
                     **{t: "new" for t in ev["populations"]["new"]["task_ids"]}}
    sec.info["new_databases"] = dict(Counter(new["db_ids"]))


SKIP_SUFFIX = {".pt", ".safetensors", ".sqlite", ".zip", ".npy", ".npz", ".lock", ".bin", ".gz", ".png"}


@section
def sec_prior_use(sec: Section, ctx: dict) -> None:
    """Item 1 (prior use): own scan by task id and by question text."""
    repo, runs, rel = ctx["repo"], ctx["runs"], ctx["rel"]
    new = ctx["new_ids"]
    want = set(new)
    questions = {t: rel.by_id[t]["question"] for t in new}
    files = []
    for f in sorted(runs.rglob("*")):
        if not f.is_file() or f.suffix in SKIP_SUFFIX:
            continue
        parts = f.relative_to(runs).parts
        if parts[0] in ("grounding_interface", "cache"):
            continue
        files.append(("runs", f.relative_to(runs), f))
    for f in sorted((repo / H / "evidence").rglob("*")):
        if f.is_file() and f.suffix not in SKIP_SUFFIX and "grounding_interface_v1" not in f.parts:
            files.append(("evidence", f.relative_to(repo / H / "evidence"), f))
    for f in sorted((repo / H / "pins").glob("*")):
        if f.is_file() and f.name != "gi_new_eval.json":
            files.append(("pins", f.relative_to(repo / H / "pins"), f))
    id_hits, q_hits = [], defaultdict(list)
    for root, relp, f in files:
        text = f.read_bytes().decode("utf-8", errors="ignore")
        n = sum(1 for t in want if t in text)
        if n:
            id_hits.append({"root": root, "file": str(relp), "ids": n})
        if str(relp) == RAW_RELEASE:
            continue
        for t, q in questions.items():
            if len(q) >= 20 and (q in text or json.dumps(q)[1:-1] in text):
                q_hits[f"{root}/{relp}"].append(t)
    sec.info["scanned_files"] = len(files)
    sec.info["id_hits"] = id_hits
    expected = {("pins", "inventory_v1.json"), ("runs", "data/gold_check_dev_v1.json"),
                ("runs", "ir/coverage_details_v1.json")}
    unexpected = [h for h in id_hits if (h["root"], h["file"]) not in expected]
    sec.check(not unexpected, f"no file outside grounding_interface/ names a new id except the three expected "
              f"({len(files)} files scanned)", f"{unexpected}")
    pinned = {(("runs" if h["root"] == "ti_tnn_scaling" else h["root"]), h["file"]) for h in
              json.loads((repo / H / "pins/gi_new_eval.json").read_text())["prior_use"]}
    sec.check({(h["root"], h["file"]) for h in id_hits} == pinned, "own hits equal gi_new_eval.json's prior_use list",
              f"{sorted({(h['root'], h['file']) for h in id_hits})}")
    sec.info["question_text_hits"] = {k: len(v) for k, v in q_hits.items()}
    allowed_q = {f"{r}/{f}" for r, f in expected}
    q_other = {k: v for k, v in q_hits.items() if k not in allowed_q}
    # a question text can recur in another task of the same database; such hits are listed with the task ids
    dup = {}
    for t, q in questions.items():
        others = [u for u in rel.ids if u != t and rel.by_id[u]["question"] == q]
        if others:
            dup[t] = others
    sec.info["new_questions_with_identical_text_elsewhere"] = dup
    real = {k: [t for t in v if t not in dup] for k, v in q_other.items()}
    real = {k: v for k, v in real.items() if v}
    sec.check(not real, "no file outside the three expected (and the raw release) contains a new question's text",
              f"{real}")
    # no earlier encoding of the new contexts: anchored cache keys absent from the generic cache, pilot_v1 entries new
    enc_dir = only_run(sec, ctx, "encode", "_new")
    entries = jl(enc_dir / "entries.jsonl")
    pre = [e["task_id"] for e in entries if (runs / "cache/encodings" / e["cache_key"][:2] /
                                              f"{e['cache_key']}.safetensors").exists()]
    sec.check(not pre, "no new context's cache key exists in cache/encodings (never encoded by another run)", f"{pre}")
    t_start = run_time(enc_dir.name)
    old = [e["task_id"] for e in entries if (runs / "cache/pilot_v1" / e["cache_key"][:2] /
                                              f"{e['cache_key']}.safetensors").stat().st_mtime < t_start]
    sec.check(not old, "every new context's pilot_v1 cache entry was written after encode-new started", f"{old}")


@section
def sec_coverage(sec: Section, ctx: dict) -> None:
    """Item 1 (coverage): every run covers exactly its population, once."""
    dev, new = ctx["dev_ids"], ctx["new_ids"]
    R = {}
    for kind, suf in (("ground", "_dev"), ("lexical", "_dev"), ("decode_dev", ""), ("constraint", "")):
        R[kind + suf] = sorted(d for d in (gi(ctx) / kind).glob(f"*{suf}") if d.is_dir())
    sec.info["runs"] = {k: [d.name for d in v] for k, v in R.items()}
    for kind in ("encode", "ground", "lexical", "eval", "direct", "swap"):
        R[f"{kind}_new"] = [only_run(sec, ctx, kind, "_new")]
    R["analysis"] = sorted(d for d in (gi(ctx) / "analysis").glob("*") if d.is_dir())
    sec.check(len(R["analysis"]) == 1, "one analysis run")
    ctx["R"] = R
    for d in R["ground_dev"]:
        ids = [r["task_id"] for r in jl(d / "records.jsonl")]
        sec.check(ids == dev, f"ground {d.name}: the 128 dev questions, once, in order")
        sec.check(json.loads((d / "manifest.json").read_text())["config"]["requested"] == dev,
                  f"ground {d.name}: manifest requested = dev")
    for d in R["lexical_dev"]:
        sec.check([r["task_id"] for r in jl(d / "records.jsonl")] == dev, f"lexical {d.name}: dev once, in order")
    for d in R["decode_dev"]:
        rows = jl(d / "records.jsonl")
        keys = [(r["fit"], r["task_id"]) for r in rows]
        sec.check(sorted(keys) == sorted((f, t) for f in V2_FITS for t in dev) and
                  len(set(keys)) == len(keys) == 768, f"decode_dev {d.name}: 6 v2 fits x 128, once")
        sec.check(all(set(r["policies"]) <= {"G0", "Glex", "G4", "oracle", "G4-swap"} for r in rows),
                  f"decode_dev {d.name}: policies G0, Glex, G4, oracle, G4-swap only")
    for d in R["constraint"]:
        rows = jl(d / "records.jsonl")
        keys = [(r["fit"], r["task_id"]) for r in rows]
        sec.check(sorted(keys) == sorted((f, t) for f in V1_FITS for t in dev) and len(set(keys)) == 768,
                  f"constraint {d.name}: 6 v1 fits x 128, once")
        sec.check(all(r["population"] == ctx["pop_of"][r["task_id"]] for r in rows),
                  f"constraint {d.name}: population labels = perf_v2_eval")
    e = R["encode_new"][0]
    ent = jl(e / "entries.jsonl")
    sec.check([x["task_id"] for x in ent] == new, "encode-new: the 64 new, once, in order")
    m = json.loads((e / "manifest.json").read_text())["config"]
    sec.check(m["ids"] == new and m["recheck"] == dev[:8], "encode-new manifest: ids = new, recheck = dev[:8]")
    for kind in ("ground_new", "lexical_new", "direct_new"):
        d = R[kind][0]
        sec.check([r["task_id"] for r in jl(d / "records.jsonl")] == new, f"{kind}: the 64 new, once, in order")
        cfg = json.loads((d / "manifest.json").read_text())["config"]
        sec.check(cfg.get("requested") == new, f"{kind}: manifest requested = new")
    ev = R["eval_new"][0]
    recs = jl(ev / "records.jsonl")
    keys = [(r["fit"], r["policy"], r["task_id"]) for r in recs]
    want = {(f, p, t) for f in V2_FITS for p in POLICIES for t in new}
    sec.check(set(keys) == want and len(keys) == len(want) == 1152, "eval-new: 6 v2 fits x 3 policies x 64, once")
    decs = json.loads((ev / "decodes.json").read_text())
    sec.check([(d["fit"], d["policy"], d["task_id"]) for d in decs] == keys,
              "eval-new: decodes.json holds the same requests in the same order as records.jsonl")
    same = all({k: v for k, v in d.items() if k != "plan"} == {k: v for k, v in r.items() if k != "score"}
               for d, r in zip(decs, recs))
    sec.check(same, "eval-new: each record equals its decodes.json entry (minus plan) plus its score")
    cfg = json.loads((ev / "manifest.json").read_text())["config"]
    sec.check(cfg["requested"] == new and cfg["policies"] == list(POLICIES) and sorted(cfg["checkpoints"]) ==
              sorted(V2_FITS), "eval-new manifest: requested = new, policies G0/Glex/G4, the six v2 fits")
    order = []
    for f, _p, _t in keys:
        if not order or order[-1] != f:
            order.append(f)
    sec.check(order == V2_FITS, "eval-new ran by seed, both architectures together (T4, C4 per seed)", f"{order}")
    sw = jl(R["swap_new"][0] / "records.jsonl")
    sk = [(r["fit"], r["task_id"]) for r in sw]
    sec.check(set(sk) == {(f, t) for f in V2_FITS for t in new} and len(sk) == 384,
              "swap-new: 6 fits x 64 (every new question has a partner), once")
    ctx["eval_recs"], ctx["eval_decs"] = recs, decs


@section
def sec_worker_items(sec: Section, ctx: dict) -> None:
    """The producer's schema items and prompt schema text (flagged worker)."""
    if ctx["no_worker"]:
        sec.check(False, "worker disabled (--no-worker): items unavailable")
        return
    ident = json.loads((ctx["R"]["encode_new"][0] / "backbone.json").read_text())
    ctx["backbone_identity"] = ident
    out = run_worker(ctx, "items", {"task_ids": ctx["dev_ids"] + ctx["new_ids"], "backbone_identity": ident}, "items")
    ctx["items"] = {t: v["items"] for t, v in out["tasks"].items()}
    ctx["schema_text"] = out["schema_text"]
    ctx["worker_items"] = out["tasks"]
    sec.check(all(v["items_equal_unbudgeted"] for v in out["tasks"].values()),
              "the encoding's schema items equal the unbudgeted serialization's (the ground run's items)")
    ent = {e["task_id"]: e for e in jl(ctx["R"]["encode_new"][0] / "entries.jsonl")}
    bad = [t for t in ctx["new_ids"] if out["tasks"][t]["cache_key"] != ent[t]["cache_key"] or
           out["tasks"][t]["tokens"] != ent[t]["tokens"]]
    sec.check(not bad, "each new context, rebuilt, has its anchored cache key and token count", f"{bad[:5]}")


@section
def sec_permissions(sec: Section, ctx: dict) -> None:
    """Item 2."""
    repo, rel, R = ctx["repo"], ctx["rel"], ctx["R"]
    dev, new = ctx["dev_ids"], ctx["new_ids"]
    llm_src = git_show(repo, FROZEN_RUN, H + "gi_llm.py")
    L = literals_of(llm_src, ("SYSTEM_GROUND", "SYSTEM_DIRECT", "PROMPT_VERSION", "DIRECT_VERSION", "MAX_NEW_TOKENS"))
    g_cfg = json.loads((R["ground_new"][0] / "manifest.json").read_text())["config"]
    d_cfg = json.loads((R["direct_new"][0] / "manifest.json").read_text())["config"]
    sec.check(g_cfg["system"] == L["SYSTEM_GROUND"] and g_cfg["prompt"] == L["PROMPT_VERSION"] == "gi-g4-prompt-v2",
              "ground-new: system prompt = gi_llm.SYSTEM_GROUND (v2) at the run commit")
    sec.check(d_cfg["system"] == L["SYSTEM_DIRECT"], "direct-new: system prompt = gi_llm.SYSTEM_DIRECT at the run commit")
    for d in R["ground_dev"]:
        c = json.loads((d / "manifest.json").read_text())
        L2 = literals_of(git_show(repo, c["git"]["commit"], H + "gi_llm.py"), ("SYSTEM_GROUND",))
        sec.check(c["config"]["system"] == L2["SYSTEM_GROUND"], f"ground {d.name}: system prompt = its commit's")
    sec.check(json.loads((R["ground_dev"][-1] / "manifest.json").read_text())["config"]["system"] == L["SYSTEM_GROUND"],
              "the v2 dev ground run used the frozen system prompt")
    for cfg, tag in ((g_cfg, "ground"), (d_cfg, "direct")):
        gen = cfg["generation"]
        sec.check(gen.get("max_new_tokens") == L["MAX_NEW_TOKENS"] == 512 and gen.get("thinking") is False,
                  f"{tag}-new generation: non-thinking, 512 new tokens", f"{ {k: gen.get(k) for k in ('thinking', 'max_new_tokens')} }")
    # own retrieval
    rsrc = git_show(repo, FROZEN_RUN, H + "retrieval.py")
    stop = None
    for node in ast.parse(rsrc).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "STOPWORDS":
            stop = set(next(n.value for n in ast.walk(node.value) if isinstance(n, ast.Constant)
                            and isinstance(n.value, str)).split())
    kb = literals_of(rsrc, ("K1", "B"))
    bm = BM25(rel, stop, kb["K1"], kb["B"])
    ctx["bm25"] = bm
    sec.info["bm25"] = {"documents": len(bm.docs), "stopwords": len(stop), "k1": kb["K1"], "b": kb["B"]}
    mine = {t: bm.top(rel.by_id[t]["question"], rel.by_id[t]["evidence"], 3, t, rel.by_id[t]["db_id"])
            for t in dev + new}
    ctx["retrieved"] = mine
    dev_inventory = set(json.loads((repo / H / "pins/inventory_v1.json").read_text())["task_ids"])
    for d in R["ground_dev"] + R["ground_new"]:
        recs = jl(d / "records.jsonl")
        bad = [r["task_id"] for r in recs if r["retrieved"] != mine[r["task_id"]]]
        sec.check(not bad, f"ground {d.name}: recorded retrieved ids = own BM25 top 3 ({len(recs)})", f"{bad[:5]}")
        leak = [r["task_id"] for r in recs for e in r["retrieved"] if rel.by_id[e]["db_id"] not in rel.pool or
                rel.by_id[e]["db_id"] == r_db(rel, r) or e in dev_inventory]
        sec.check(not leak, f"ground {d.name}: every retrieved example from the training pool, another database, "
                  f"outside the dev inventory")
    if ctx.get("worker_items"):
        bad = [t for t in dev + new if ctx["worker_items"][t]["retrieved"] != mine[t]]
        sec.check(not bad, "the encoder contexts' retrieved examples (producer, flagged) = own BM25", f"{bad[:5]}")
    # prompts rebuilt and token-counted
    if not ctx.get("schema_text"):
        sec.check(False, "prompt reconstruction needs the worker's schema text")
    else:
        tok, files_ok = load_tokenizer(ctx, g_cfg)
        sec.check(files_ok, "tokenizer files hash to the run manifest's model identity")
        golds_norm = {t: norm_sql(rel.by_id[t]["SQL"]) for t in dev + new}
        n_match, mism, leaks, prompts = 0, [], [], {}
        runs = [(R["ground_new"][0], L["SYSTEM_GROUND"], "G4-new"), (R["direct_new"][0], L["SYSTEM_DIRECT"], "direct-new"),
                (R["ground_dev"][-1], L["SYSTEM_GROUND"], "G4-dev-v2")]
        for d, system, tag in runs:
            ok = 0
            for r in jl(d / "records.jsonl"):
                t = r["task_id"]
                msg = own_user_message(rel, t, ctx["schema_text"][rel.by_id[t]["db_id"]], mine[t])
                n = prompt_tokens(tok, system, msg)
                if n == r["call"]["prompt_tokens"]:
                    ok += 1
                else:
                    mism.append((tag, t, n, r["call"]["prompt_tokens"]))
                low = norm_sql(system + "\n" + msg)
                for g, gs in golds_norm.items():
                    if len(gs) > 20 and gs in low:
                        leaks.append((tag, t, g))
                if any(x in msg for x in dev + new):
                    leaks.append((tag, t, "task id"))
                prompts[f"{tag}/{t}"] = sha256_bytes((system + "\x1f" + msg).encode())
            sec.check(ok == len(jl(d / "records.jsonl")), f"{tag}: every rebuilt prompt has the recorded prompt_tokens "
                      f"({ok}/{len(jl(d / 'records.jsonl'))})", f"{mism[:5]}")
            n_match += ok
        sec.check(not leaks, "no evaluated question's gold SQL (192) or task id occurs in any rebuilt G4 or direct prompt",
                  f"{leaks[:5]}")
        ctx["prompt_hashes"] = prompts
        sec.info["prompts_rebuilt_and_matched"] = n_match
    # code: what the calls and Glex read (frozen source)
    run_src = git_show(repo, FROZEN_RUN, H + "gi_run.py")
    for fn in ("ground_cmd", "direct_new_cmd"):
        s = func_source(run_src, fn)
        before = s.split("_score(")[0] if "_score(" in s else s
        sec.check("gold" not in before, f"gi_run.{fn}: nothing named gold before scoring")
    for fn in ("ground_call", "direct_call", "messages"):
        s = func_source(llm_src, fn)
        sec.check("gold" not in s and not names_in(s) & {"gold", "data", "scorer"},
                  f"gi_llm.{fn}: reads no gold, data or scorer")
    sec.check("def user_message(task: C.Task, schema_text: str, examples: list)" in
              git_show(repo, FROZEN_RUN, H + "llm_agent.py").decode(),
              "llm_agent.user_message takes the task, the schema text and the examples only")
    lex_src = git_show(repo, FROZEN_RUN, H + "gi_lexical.py")
    sec.check(imports_of(lex_src) == {"__future__:annotations", ".:gi_record", ".:grounding"},
              "gi_lexical imports only gi_record and grounding", f"{sorted(imports_of(lex_src))}")
    gsrc = func_source(lex_src, "ground")
    sec.check(gsrc.startswith("def ground(question: str, evidence: str, schema, items)") and
              not names_in(gsrc) & {"gold", "data", "scorer", "sql", "plan", "oracle"},
              "gi_lexical.ground(question, evidence, schema, items) reads no gold, data, scorer or plan")
    gr_src = git_show(repo, FROZEN_RUN, H + "grounding.py")
    for fn in ("name_tokens", "_mentions"):
        sec.check(not names_in(func_source(gr_src, fn)) & {"gold", "data", "scorer", "open", "Path"},
                  f"grounding.{fn} is a text function")
    ev_src = func_source(run_src, "eval_new_cmd")
    sec.check("gi_lexical.ground(task.question, task.evidence, schema, enc.schema)" in ev_src,
              "eval-new computes Glex from the question, evidence, schema and the encoding's items")
    sec.check("oracle" not in ev_src and "gold" not in ev_src.split("_score(")[0],
              "eval-new's decode path names no oracle and no gold before scoring")
    # records: Glex record identical in every fit and equal to the lexical run; no oracle in any new run
    lex = {r["task_id"]: r["record"] for r in jl(R["lexical_new"][0] / "records.jsonl")}
    glex = [r for r in ctx["eval_recs"] if r["policy"] == "Glex"]
    sec.check(all(r["lexical_record"] == lex[r["task_id"]] for r in glex),
              "every eval Glex record (6 fits) equals the lexical-new record of its question (deterministic)")
    sec.check(all(r["lexical_record"] is None for r in ctx["eval_recs"] if r["policy"] != "Glex"),
              "no lexical record outside Glex requests")
    sec.check(all(x["literals"] == [] for x in lex.values()), "Glex proposes no literal bindings")
    hits = []
    for kind in ("encode_new", "ground_new", "lexical_new", "eval_new", "direct_new", "swap_new"):
        for f in R[kind][0].rglob("*"):
            if f.is_file() and '"oracle"' in f.read_text(errors="ignore"):
                hits.append(str(f.relative_to(gi(ctx))))
    sec.check(not hits, "no new-population run file holds an oracle key", f"{hits}")
    sec.check(sum('"oracle"' in json.dumps(r) for r in jl(R["decode_dev"][-1] / "records.jsonl")) == 768,
              "the oracle appears in the dev diagnostic decode")
    sec.info["note"] = ("analysis.json uses gold annotations (gold_reference/oracle_record) only for the diagnostics "
                        "(record recall, alternative table sets), after scoring")


def r_db(rel, r):
    return rel.by_id[r["task_id"]]["db_id"]


def norm_sql(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def own_user_message(rel: Release, tid: str, schema_text: str, retrieved: list) -> str:
    """The B4 harness's user message (question, evidence, schema text, the retrieved examples), as its docstring and
    the protocol describe it."""
    r = rel.by_id[tid]
    parts = [f"Question: {r['question']}", f"Evidence: {r['evidence'] or '(none)'}", "", "Database schema:", schema_text]
    if retrieved:
        parts += ["", "Examples from other databases (question, evidence and a correct query):"]
        for i, e in enumerate(retrieved, 1):
            x = rel.by_id[e]
            parts += [f"Example {i}. Question: {x['question']}", f"Evidence: {x['evidence'] or '(none)'}",
                      f"SQL: {x['SQL']}"]
    return "\n".join(parts)


def load_tokenizer(ctx: dict, manifest_cfg: dict):
    from transformers import AutoTokenizer
    m = json.loads((ctx["R"]["ground_new"][0] / "manifest.json").read_text())["model_identity"]
    d = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    ok = all(sha256_path(d / f) == m["files"][f]["sha256"] for f in ("tokenizer.json", "tokenizer_config.json",
                                                                     "chat_template.jinja"))
    return AutoTokenizer.from_pretrained(str(d), local_files_only=True), ok


def prompt_tokens(tok, system: str, user: str) -> int:
    text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   tools=None, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return len(tok(text, add_special_tokens=False)["input_ids"])


@section
def sec_scores(sec: Section, ctx: dict) -> None:
    """Item 5: own EX for every final SQL."""
    rel, R = ctx["rel"], ctx["R"]
    jobs, uses = {}, []

    def want(tid, sql, tag, rec_correct, rec_status, rec_pred_status=None):
        key = (tid, sql)
        if key not in jobs:
            db = rel.by_id[tid]["db_id"]
            jobs[key] = (str(rel.db_path(db)), sql, rel.by_id[tid]["SQL"], rel.timeout)
        uses.append((key, tag, rec_correct, rec_status, rec_pred_status))
    for r in ctx["eval_recs"]:
        sql = r["final"]["sql"] if r["final"]["kind"] == "sql" else None
        want(r["task_id"], sql, "eval", r["score"]["correct"], r["score"]["status"], r["score"]["predicted"]["status"])
    for r in jl(R["direct_new"][0] / "records.jsonl"):
        sql = r["final"]["sql"] if r["final"]["kind"] == "sql" else None
        want(r["task_id"], sql, "direct", r["score"]["correct"], r["score"]["status"], r["score"]["predicted"]["status"])
    for d in R["constraint"]:
        for r in jl(d / "records.jsonl"):
            for p in ("v1", "restricted"):
                want(r["task_id"], r[p]["sql"] if r[p]["kind"] == "sql" else None, "constraint", r[p]["correct"], None,
                     r[p]["status"])
    for r in jl(R["decode_dev"][-1] / "records.jsonl"):
        for p, v in r["policies"].items():
            want(r["task_id"], v["sql"] if v["kind"] == "sql" else None, "dev_v2", v["correct"], None, v["status"])
    for r in jl(R["swap_new"][0] / "records.jsonl"):
        for p, v in r["policies"].items():
            want(r["task_id"], v["sql"] if v["kind"] == "sql" else None, "swap", v["correct"], None, v["status"])
    cache_path = ctx["out"] / "cache" / "ex.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def ck(key):
        db = rel.by_id[key[0]]["db_id"]
        return f"{key[0]}:{rel.snaps[db]['sha256'][:16]}:{sha256_bytes((key[1] or '').encode()) if key[1] is not None else 'None'}"
    todo = {k: v for k, v in jobs.items() if ck(k) not in cache}
    t0 = time.time()
    res = ex_batch(todo, ctx["ex_workers"])
    for k, v in res.items():
        cache[ck(k)] = v
    cache_path.write_text(json.dumps(cache, sort_keys=True))
    sec.info["distinct_executions"] = len(jobs)
    sec.info["executed_now"] = len(todo)
    sec.info["seconds"] = round(time.time() - t0, 1)
    mine = {k: cache[ck(k)] for k in jobs}
    # This verifier runs on cores 0-4,10-14, the host's Cortex-A725 efficiency cores (about half the single-thread
    # speed of the Cortex-X925 cores). A query that finished within 30 s for the producer can time out here: such a
    # pair is re-executed once with a 4x deadline, and then compared on its result.
    slow = {key for key, tag, rc, rs, rps in uses if status_class(mine[key]["status"]) == "timeout" and
            status_class(rps or rs) != "timeout"}
    ext_todo = {k: jobs[k][:3] + (4 * rel.timeout,) for k in slow if ck(k) + ":x4" not in cache}
    for k, v in ex_batch(ext_todo, ctx["ex_workers"]).items():
        cache[ck(k) + ":x4"] = v
    cache_path.write_text(json.dumps(cache, sort_keys=True))
    ext = {k: cache[ck(k) + ":x4"] for k in slow}
    sec.info["host_speed_timeouts"] = [{"task": k[0], "sql_sha": sha256_bytes((k[1] or "").encode())[:12],
                                        "at_30s": mine[k]["status"], "at_120s": ext[k]["status"],
                                        "seconds_at_120s": round(ext[k]["seconds"], 1)} for k in sorted(slow)]
    ctx["ex"] = {k: ext.get(k) or v for k, v in mine.items()}   # downstream tallies: host-speed timeouts at 4x
    ctx["ex_ext"] = ext
    by_tag = defaultdict(lambda: [0, 0, 0, []])
    for key, tag, rc, rs, rps in uses:
        m = ext.get(key) or mine[key]
        b = by_tag[tag]
        b[0] += 1
        ok_c = m["correct"] == rc
        ok_s = (rs is None or status_class(rs) == status_class(m["status"])) and \
            (rps is None or status_class(rps) == status_class(m["pred"]))
        b[1] += ok_c
        b[2] += ok_s
        if not (ok_c and ok_s):
            b[3].append({"task": key[0], "sql": (key[1] or "")[:120], "recorded": [rc, rs, rps],
                         "mine": [m["correct"], m["status"], m["pred"], round(m["seconds"], 2)]})
    for tag, (n, c, s, bad) in by_tag.items():
        sec.check(c == n, f"{tag}: own EX equals every recorded score ({c}/{n})", f"{bad[:5]}")
        sec.check(s == n, f"{tag}: own status class equals every recorded status ({s}/{n}; host-speed timeouts "
                  f"compared at the 4x deadline)", f"{bad[:5]}")
    gold_bad = [k for k, v in mine.items() if v["status"] != "abstain" and v["pred"] != "timeout" and
                v["gold_status"] != "ok"]
    sec.check(not gold_bad, "the gold SQL executed cleanly in every pair whose prediction left it time",
              f"{gold_bad[:3]}")
    snaps = {r["score"]["snapshot_sha256"] == rel.snaps[r["db_id"]]["sha256"] for r in ctx["eval_recs"]}
    sec.check(snaps == {True}, "eval scores name the pinned snapshot sha256")
    sec.info["status_counts"] = {tag: dict(Counter(ctx["ex"][k]["status"] for k, t, *_ in uses if t == tag))
                                 for tag in by_tag}


@section
def sec_fallback(sec: Section, ctx: dict) -> None:
    """Item 4."""
    if not ctx.get("items"):
        sec.check(False, "needs the worker's items")
        return
    rel, R, items = ctx["rel"], ctx["R"], ctx["items"]
    mine_valid = {}
    for d in R["ground_dev"] + R["ground_new"]:
        recs = jl(d / "records.jsonl")
        agree, idx_agree, lit_agree, bad = 0, 0, 0, []
        for r in recs:
            t = r["task_id"]
            q, e = rel.by_id[t]["question"], rel.by_id[t]["evidence"]
            obj = own_parse(r["call"]["content"])
            v = own_validate(obj, items[t], q, e, r["call"]["finish_reason"] != "eos")
            g = r["grounding"]
            same = v["valid"] == g["valid"] and (v["valid"] or v["reason"] == reason_class(g["reason"]))
            agree += same
            if not same:
                bad.append((t, v.get("reason"), g["reason"]))
            if v["valid"] and g["valid"]:
                idx_agree += v["tables"] == g["tables"] and v["columns"] == g["columns"]
                lit_agree += v["literals_ok"] == len(g["literals"]) and v["literals_bad"] == len(g["invalid_literals"])
            if obj != r["record"] and not (obj is None and r["record"] is None):
                bad.append((t, "parsed record differs"))
            mine_valid[(d.name, t)] = v
        nv = sum(1 for r in recs if r["grounding"]["valid"])
        prompt = json.loads((d / "manifest.json").read_text())["config"]["prompt"]
        if prompt != "gi-g4-prompt-v2":   # the first dev run predates the parser repairs: descriptive only
            sec.info[f"{d.name} ({prompt}, parser before the repairs)"] = {
                "recorded_valid": nv, "own_valid_with_repaired_parser": sum(mine_valid[(d.name, r["task_id"])]["valid"]
                                                                           for r in recs), "disagreements": bad[:10]}
            continue
        sec.check(agree == len(recs), f"ground {d.name}: own parse/validation agrees on validity and reason "
                  f"({agree}/{len(recs)}; {len(recs) - nv} invalid)", f"{bad[:5]}")
        sec.check(idx_agree == nv, f"ground {d.name}: own resolution = recorded item indexes for every valid record")
        sec.check(lit_agree == nv, f"ground {d.name}: own literal check = recorded valid/invalid bindings")
        sec.info[d.name] = {"valid": nv, "invalid": len(recs) - nv, "truncated": sum(r["call"]["finish_reason"] != "eos"
                                                                                 for r in recs),
                            "reasons": dict(Counter(reason_class(r["grounding"]["reason"]) for r in recs
                                                    if not r["grounding"]["valid"]))}
    ctx["own_valid"] = mine_valid
    gnew = R["ground_new"][0].name
    invalid = [t for t in ctx["new_ids"] if not mine_valid[(gnew, t)]["valid"]]
    ctx["g4_invalid_new"] = invalid
    sec.info["new_invalid"] = invalid
    ev = {(r["fit"], r["policy"], r["task_id"]): r for r in ctx["eval_recs"]}
    eq = [(f, t) for f in V2_FITS for t in invalid
          if ev[(f, "G4", t)]["actions"] == ev[(f, "G0", t)]["actions"] and
          ev[(f, "G4", t)]["final"] == ev[(f, "G0", t)]["final"] and not (ev[(f, "G4", t)]["grounding"] or {}).get("valid")]
    sec.check(len(eq) == 6 * len(invalid), f"eval-new: every invalid G4 request ({6 * len(invalid)}) decodes exactly as "
              "G0 of the same fit and records an invalid grounding")
    valid_flag = [(f, t) for f in V2_FITS for t in ctx["new_ids"]
                  if bool((ev[(f, "G4", t)]["grounding"] or {}).get("valid")) != mine_valid[(gnew, t)]["valid"]]
    sec.check(not valid_flag, "eval-new: every G4 request's recorded validity = own validation", f"{valid_flag[:5]}")
    dv = {(r["fit"], r["task_id"]): r for r in jl(R["decode_dev"][-1] / "records.jsonl")}
    gdev = R["ground_dev"][-1].name
    inv_dev = [t for t in ctx["dev_ids"] if not mine_valid[(gdev, t)]["valid"]]
    eqd = [(f, t) for f in V2_FITS for t in inv_dev
           if dv[(f, t)]["policies"]["G4"]["actions"] == dv[(f, t)]["policies"]["G0"]["actions"]]
    sec.check(len(eqd) == 6 * len(inv_dev), f"dev v2 decode: every invalid G4 request ({6 * len(inv_dev)}) = G0")
    a = json.loads((ctx["R"]["analysis"][0] / "analysis.json").read_text())
    sec.check(all(a["decodes"][f"{x}-G4"]["fallback_to_G0"] == 3 * len(invalid) for x in ("T4", "C4")),
              f"analysis: fallback_to_G0 = 3 x {len(invalid)} per architecture")
    sec.check(a["records"]["G4"]["valid"] == 64 - len(invalid) and
              sum(a["records"]["G4"]["fallback_reasons"].values()) == len(invalid) and
              a["records"]["G4"]["truncated"] == sec.info[gnew]["truncated"],
              "analysis: G4 valid, fallback reasons and truncation counts = own")
    sec.check(all(v["valid"] or v["reason"] in ("unknown column", "unknown table", "keys", "status", "not JSON",
                                                "truncated", "columns type", "tables type", "literals type",
                                                "not an object") for v in mine_valid.values()),
              "every fallback is a parse, key, type, name or truncation failure (never a literal)")


@section
def sec_consumption(sec: Section, ctx: dict) -> None:
    """Item 3: the code, then every eval-new decode re-decoded on the CPU with own bias vectors (flagged worker)."""
    repo, R = ctx["repo"], ctx["R"]
    rec_src = git_show(repo, FROZEN_RUN, H + "gi_record.py")
    lit = literals_of(rec_src, ("MAX_TABLES", "MAX_COLUMNS", "KEYS", "STATUSES"))
    sec.check(lit.get("MAX_TABLES") == 8 and lit.get("MAX_COLUMNS") == 16, "gi_record: MAX_TABLES 8, MAX_COLUMNS 16")
    rb = func_source(rec_src, "rank_bias")
    sec.check("return math.log(2.0) / rank" in rb, "gi_record.rank_bias = ln 2 / rank")
    bs = func_source(rec_src, "biases", cls="Grounding")
    sec.check("if not self.valid:" in bs and "return None" in bs and "ranked[:cap]" in bs and
              "out[i] = rank_bias(r)" in bs and "[0.0] * n_items" in bs,
              "Grounding.biases: None if invalid; zeros; the first 8 tables and 16 columns at ln 2 / r")
    pp = func_source(rec_src, "pointer_prior")
    sec.check("return items + b, start, end" in pp and "trunk.pointer_scores = biased" in pp and
              "if biases is None:" in pp, "pointer_prior adds the vector to the item scores only (start/end untouched)")
    rs = func_source(rec_src, "resolve")
    sec.check("lits" in rs and "Grounding(True, None, t_out, c_out" in rs,
              "resolve: literals are recorded beside the ranked tables and columns, which alone feed biases()")
    run_src = git_show(repo, FROZEN_RUN, H + "gi_run.py")
    ev = func_source(run_src, "eval_new_cmd")
    sec.check(ev.count("with GR.pointer_prior(model, biases):") == 2 and ev.count("infer_controller._decode_at(") == 2,
              "eval-new: the timed decode and its cache reconstruction both run under the same prior")
    sec.check("biases = g.biases(len(enc.schema.items)) if g is not None else None" in ev,
              "eval-new: the vector is the record's biases over the encoding's items (None for G0)")
    ic = git_show(repo, FROZEN_RUN, H + "infer_controller.py").decode()
    sec.check('scores = items[b, 0, 0 if kind == "table" else 1]' in ic and "_, targets = machines[b].pointer_targets(op)"
              in ic and "scores.masked_fill(~mask, K.NEG).argmax()" in ic,
              "decode: table/column pointers read the item scores; legal targets come from the machine and mask after")
    if ctx["no_worker"] or not ctx.get("items"):
        sec.check(False, "the CPU re-decode needs the worker")
        return
    rel, items = ctx["rel"], ctx["items"]
    ckpt = ctx["ckpt_paths"]
    g4 = {r["task_id"]: r for r in jl(R["ground_new"][0] / "records.jsonl")}
    jobs, expect = [], {}
    bias_agree, n_bias = 0, 0
    for r in ctx["eval_recs"]:
        t, f, p = r["task_id"], r["fit"], r["policy"]
        n = len(items[t])
        b = None
        if p == "G4":
            v = own_validate(own_parse(g4[t]["call"]["content"]), items[t], rel.by_id[t]["question"],
                             rel.by_id[t]["evidence"], g4[t]["call"]["finish_reason"] != "eos")
            b = own_biases(v["tables"], v["columns"], n) if v["valid"] else None
        elif p == "Glex":
            v = own_validate(r["lexical_record"], items[t], rel.by_id[t]["question"], rel.by_id[t]["evidence"], False)
            b = own_biases(v["tables"], v["columns"], n) if v["valid"] else None
        g = r["grounding"]
        if g is not None and g.get("valid"):
            n_bias += 1
            bias_agree += b == own_biases(g["tables"], g["columns"], n)
        jid = f"eval|{f}|{p}|{t}"
        jobs.append({"id": jid, "task_id": t, "ckpt": ckpt[f], "kind": "bias", "biases": b, "precision": "highest"})
        expect[jid] = r
    sec.check(bias_agree == n_bias, f"own bias vector = the vector of the recorded resolved record ({bias_agree}/{n_bias})")
    strip = []
    for f in V2_FITS:
        for t in ctx["new_ids"]:
            r = g4[t]
            if r["grounding"]["valid"]:
                rec = dict(r["record"])
                rec2 = {**rec, "literals": [], "status": "unresolved" if rec["status"] == "ok" else "ok"}
                for tag, rr in (("orig", rec), ("stripped", rec2)):
                    jid = f"producer|{tag}|{f}|{t}"
                    jobs.append({"id": jid, "task_id": t, "ckpt": ckpt[f], "kind": "producer", "record": rr,
                                 "finish_reason": r["call"]["finish_reason"], "precision": "highest"})
                strip.append((f, t))
    out = run_worker(ctx, "decode", {"jobs": jobs, "backbone_identity": ctx["backbone_identity"]}, "decode_eval")
    got = {x["id"]: x for x in out["results"]}
    ctx["redecode_eval"] = got
    same_a = same_s = 0
    diffs = []
    for jid, r in expect.items():
        x = got[jid]
        a_ok = x["actions"] == r["actions"]
        s_ok = x["sql"] == r["final"]["sql"]
        same_a += a_ok
        same_s += s_ok
        if not a_ok:
            first = next((i for i, (u, v) in enumerate(zip(x["actions"], r["actions"])) if u != v),
                         min(len(x["actions"]), len(r["actions"])))
            diffs.append({"id": jid, "first_step": first, "cpu": x["actions"][first:first + 2],
                          "gpu": r["actions"][first:first + 2], "sql_equal": s_ok})
    sec.info["redecode_eval"] = {"decodes": len(expect), "actions_equal": same_a, "sql_equal": same_s,
                                 "differences": diffs}
    sec.check(same_a >= 0.99 * len(expect), f"CPU re-decode with own biases reproduces the GPU actions "
              f"({same_a}/{len(expect)}; differences listed in info)", f"{diffs[:3]}")
    by_pol = Counter(d["id"].split("|")[2] for d in diffs)
    sec.info["redecode_differences_by_policy"] = dict(by_pol)
    if diffs:   # the likely cause: how close the two choices were on the CPU at the first differing step
        mj = []
        for dd in diffs:
            _e, f, p, t = dd["id"].split("|")
            op, cpu_arg = dd["cpu"][0]
            mj.append({"id": dd["id"], "task_id": t, "ckpt": ckpt[f], "kind": "margin",
                       "biases": next(j["biases"] for j in jobs if j["id"] == dd["id"]), "step": dd["first_step"],
                       "pointer": "table" if op.endswith("_TABLE") else "column", "cpu_choice": cpu_arg,
                       "gpu_choice": dd["gpu"][0][1], "precision": "highest"})
            mj.append({**mj[-1], "id": dd["id"] + "|high", "kind": "bias", "precision": "high"})
        mo = {x["id"]: x for x in run_worker(ctx, "decode", {"jobs": mj, "backbone_identity": ctx["backbone_identity"]},
                                             "decode_margin")["results"]}
        for dd in diffs:
            m = mo[dd["id"]]["margin"]
            dd["cpu_margin_over_gpu_choice"] = m["score_cpu_choice"] - m["score_gpu_choice"]
            dd["relative_margin"] = dd["cpu_margin_over_gpu_choice"] / max(m["score_scale"], 1e-9)
            dd["cpu_high_precision_equals_gpu"] = mo[dd["id"] + "|high"]["actions"] == expect[dd["id"]]["actions"]
        sec.check(all(0 <= dd["cpu_margin_over_gpu_choice"] < 0.05 for dd in diffs),
                  "every CPU/GPU difference is a near-tie at its first differing pointer step (CPU margin < 0.05)",
                  f"{[(dd['id'], round(dd['cpu_margin_over_gpu_choice'], 5)) for dd in diffs]}")
    # pointer kinds
    wrong = []
    for jid, r in expect.items():
        it = items[r["task_id"]]
        for op, arg in r["actions"]:
            if isinstance(arg, int) and (op.endswith("_TABLE") or op.startswith("COL")):
                kind = "table" if op.endswith("_TABLE") else "column"
                if it[arg][0] != kind:
                    wrong.append((jid, op, arg, it[arg][0]))
    sec.check(not wrong, "every table-pointer action chose a table item and every column action a column item",
              f"{wrong[:5]}")
    # producer path: own bias = producer bias; literals and status stripped leave bias and decode unchanged
    pb_ok = st_ok = dec_ok = 0
    for f, t in strip:
        o, s = got[f"producer|orig|{f}|{t}"], got[f"producer|stripped|{f}|{t}"]
        n = len(items[t])
        v = own_validate(own_parse(g4[t]["call"]["content"]), items[t], rel.by_id[t]["question"],
                         rel.by_id[t]["evidence"], False)
        mineb = {str(i): x for i, x in enumerate(own_biases(v["tables"], v["columns"], n)) if x}
        pb_ok += {k: round(x, 12) for k, x in o["producer_bias_nonzero"].items()} == \
            {k: round(x, 12) for k, x in mineb.items()}
        st_ok += o["producer_bias_nonzero"] == s["producer_bias_nonzero"]
        dec_ok += o["actions"] == s["actions"] and o["sql"] == s["sql"]
    sec.check(pb_ok == len(strip), f"the producer's resolve+biases (flagged) = own vector for every valid G4 record "
              f"({pb_ok}/{len(strip)})")
    sec.check(st_ok == len(strip), f"literals removed and status flipped: identical bias vector ({st_ok}/{len(strip)})")
    sec.check(dec_ok == len(strip), f"literals removed and status flipped: identical decode ({dec_ok}/{len(strip)})")
    own_vs_prod = sum(got[f"producer|orig|{f}|{t}"]["actions"] == got[f"eval|{f}|G4|{t}"]["actions"] for f, t in strip)
    sec.check(own_vs_prod == len(strip), "own-bias decode = producer-path decode for every valid G4 request (CPU)")


@section
def sec_integrity(sec: Section, ctx: dict) -> None:
    """Item 7."""
    runs, R, repo = ctx["runs"], ctx["R"], ctx["repo"]
    proto_sha = json.loads((repo / H / "pins/gi_protocol.json").read_text())["sha256"]
    e = R["encode_new"][0]
    m = json.loads((e / "manifest.json").read_text())
    sec.check(not m["git"]["research_tree_dirty"] and m["config"]["protocol"]["sha256"] == proto_sha,
              "encode-new: clean tree, the frozen protocol (its anchors are accepted)")
    anchors = {x["task_id"]: x for x in jl(e / "entries.jsonl")}
    bad, meta_bad, shape_bad = [], [], []
    for t, a in anchors.items():
        p = runs / "cache/pilot_v1" / a["cache_key"][:2] / f"{a['cache_key']}.safetensors"
        h, dt, shape, meta = safetensors_digest(p)
        if h != a["states_sha256"]:
            bad.append(t)
        if meta.get("states_sha256") != a["states_sha256"]:
            meta_bad.append(t)
        if dt != "BF16" or shape != [a["tokens"], 2560]:
            shape_bad.append((t, dt, shape))
    sec.check(not bad, "own hash of every new cache entry's tensor bytes = its anchor (64)", f"{bad[:5]}")
    sec.check(not meta_bad, "every new cache entry's metadata records its anchor sha256")
    sec.check(not shape_bad, "every new cache entry: bf16 [tokens, 2560]", f"{shape_bad[:3]}")
    summ = json.loads((e / "summary.json").read_text())
    sec.check(summ["encoded"] == 64 and summ["already_present"] == 0 and len(summ["rechecked"]) == 8 and
              all(x["equal"] for x in summ["rechecked"]) and [x["task_id"] for x in summ["rechecked"]] ==
              ctx["dev_ids"][:8], "encode-new: 8 dev contexts re-encoded equal to their anchors, 64 written fresh")
    # the 8 rechecks' anchors (pilot / perf_v2 encode runs)
    prior = {}
    for root in (runs / "pilot/encode", runs / "perf_v2/encode"):
        for en in sorted(root.glob("*/entries.jsonl")):
            for x in jl(en):
                prior.setdefault(x["task_id"], x)
    sec.check(all(t in prior for t in ctx["dev_ids"][:8]), "the 8 rechecked dev contexts have earlier anchors")
    recs = ctx["eval_recs"]
    sec.check(all(r["states_sha256"] == anchors[r["task_id"]]["states_sha256"] for r in recs),
              "eval-new: every fresh GPU forward's states sha256 = its anchor (1152)")
    sec.check(all(r["encode_cost"]["cache_key"] == anchors[r["task_id"]]["cache_key"] for r in recs),
              "eval-new: every request's context has its anchored cache key")
    sec.check(all(r["anchor_equal"] and r["reconstruction_equal"] for r in recs),
              "eval-new: every record's anchor_equal and reconstruction_equal flags are true")
    s = json.loads((R["eval_new"][0] / "summary.json").read_text())
    sec.check(s["integrity"] == {"requests": 1152, "anchor_or_reconstruction_failures": 0}, "eval-new summary: 0 failures")
    cfg = json.loads((R["eval_new"][0] / "manifest.json").read_text())
    sec.check(cfg["config"]["matmul_precision"] == "high" and not cfg["git"]["research_tree_dirty"] and
              cfg["config"]["protocol"]["sha256"] == proto_sha, "eval-new: precision high, clean tree, frozen protocol")
    got = ctx.get("redecode_eval")
    if got:
        sk = {x["states_sha256"] for x in got.values()}
        sec.check(all(x["states_sha256"] == anchors[x["id"].split("|")[-1]]["states_sha256"] for x in got.values()),
                  "the CPU re-decodes read cached states equal to the anchors")
        sec.info["cpu_redecode_from_cache"] = ctx["redecode_eval_summary"] = {
            "equal": sum(got[f"eval|{r['fit']}|{r['policy']}|{r['task_id']}"]["actions"] == r["actions"] for r in recs),
            "of": len(recs), "distinct_states": len(sk)}


@section
def sec_timing(sec: Section, ctx: dict) -> None:
    """Item 6."""
    repo, runs, R = ctx["repo"], ctx["runs"], ctx["R"]
    run_src = git_show(repo, FROZEN_RUN, H + "gi_run.py")
    ev = func_source(run_src, "eval_new_cmd")
    marks = ["t0 = time.perf_counter()", "enc = encode.encode_request(bb, task, schema, examples, use_cache=False)",
             "lex = gi_lexical.ground(", "dec = infer_controller._decode_at(", "torch.cuda.synchronize()",
             "elapsed = time.perf_counter() - t0", "digest = hashlib.sha256(", "cached = encode.encode_request(reader"]
    pos = [ev.find(x) for x in marks]
    sec.check(all(p >= 0 for p in pos) and pos == sorted(pos),
              "eval-new timed span: t0, fresh forward, Glex record, decode, synchronize, elapsed; digest and cache "
              "reconstruction after", f"{dict(zip(marks, pos))}")
    recs = ctx["eval_recs"]
    sec.check(all(not r["encode_cost"]["cache_hit"] and not r["encode_cost"]["cache_used"] and
                  r["encode_cost"]["forward_seconds"] > 0 and r["encode_cost"]["device_type"] == "cuda" for r in recs),
              "every eval request ran its own fresh GPU forward (no cache read)")
    resid = defaultdict(list)
    short = []
    for r in recs:
        inner = r["encode_cost"]["seconds"] + r["controller_work"]["batch_seconds"]
        resid[r["policy"]].append(r["controller_seconds"] - inner)
        if r["controller_seconds"] < inner:
            short.append(r["task_id"])
    sec.check(not short, "controller_seconds >= the encode span + the decode span for every request", f"{short[:5]}")
    sec.info["controller_seconds_minus_encode_and_decode"] = {
        p: {"mean": sum(v) / len(v), "max": max(v)} for p, v in resid.items()}
    # G4 calls: per call, in their own reservation
    g = jl(R["ground_new"][0] / "records.jsonl")
    gs = json.loads((R["ground_new"][0] / "summary.json").read_text())
    sec.check(all(r["seconds"] == r["call"]["seconds"] and r["seconds"] > 0 for r in g),
              "every G4 call records its own seconds")
    ledger, problems = parse_ledger(runs / "gpu_ledger.jsonl")
    sec.check(not problems, "own ledger parse: well-formed, one start per reservation, nothing after an end",
              f"{problems[:5]}")
    byid = {x["id"]: x for x in ledger}
    gires = [x for x in ledger if (x["command"] or "").startswith("gi ")]
    sec.info["gi_reservations"] = [{"command": x["command"], "id": x["id"], "start": iso(x["t_start"]),
                                    "end": iso(x["t_end"]), "hours": x["hours"],
                                    "projected": x["start"].get("projected_hours"),
                                    "run_dir": (x["start"].get("extra") or {}).get("run_dir")} for x in gires]
    sec.check(len(gires) == 6 and all(not x["open"] and x["end"]["status"] == "ok" for x in gires),
              "six gi reservations (2 ground dev, encode/ground/eval/direct new), all ended ok")
    sec.check(all(x["start"].get("foreign_gpu_processes") == [] and x["end"].get("foreign_gpu_processes") == []
                  for x in gires), "no foreign GPU process at the start or end of any gi reservation")
    sec.check(all(x["start"].get("authorization") == 2 and x["start"].get("cap_hours") == 32.0 and
                  close((x["start"].get("extra") or {}).get("work_order_limit"), 29.415907, 1e-9) for x in gires),
              "every gi start entry: authorization v2, cap 32 h, the work order's limit 29.415907 h")
    load = gs["load_seconds"]
    gres = byid[gs["reservation"]["id"]]
    sec.check(sum(r["seconds"] for r in g) + load <= gres["seconds"], "ground-new: the calls and the load fit inside "
              "its own reservation", f"{sum(r['seconds'] for r in g) + load:.1f} s vs {gres['seconds']:.1f} s")
    # each run's summary binds its reservation
    pairs = [("encode_new", "gi encode-new"), ("ground_new", "gi ground new"), ("eval_new", "gi eval-new"),
             ("direct_new", "gi direct-new")] + [(f"ground_dev:{d.name}", "gi ground dev") for d in R["ground_dev"]]
    t_used = 0.0
    used_before = {}
    for x in ledger:
        used_before[x["id"]] = t_used
        t_used += x["hours"]
    for kind, cmd in pairs:
        d = R[kind][0] if ":" not in kind else next(d for d in R["ground_dev"] if d.name == kind.split(":")[1])
        s = json.loads((d / "summary.json").read_text())["reservation"]
        x = byid.get(s["id"])
        ok = x is not None and x["command"] == cmd and abs(x["hours"] - s["hours"]) < 1e-4 and \
            (x["start"].get("extra") or {}).get("run_dir") == f"grounding_interface/{d.relative_to(gi(ctx)).as_posix()}"
        sec.check(ok, f"{d.relative_to(gi(ctx))}: summary reservation = ledger ({cmd}, run_dir, hours)",
                  f"{s} vs {None if x is None else (x['command'], x['hours'])}")
        if x is None:
            continue
        sec.check(run_time(d.name) <= x["t_start"] + 1, f"{d.name}: run directory created before its reservation")
        files_in = [f for f in d.iterdir() if f.name in ("entries.jsonl", "decodes.json") or
                    (f.name == "records.jsonl" and cmd in ("gi ground new", "gi ground dev"))]
        late = [f.name for f in files_in if f.stat().st_mtime > x["t_end"] + 2]
        sec.check(not late, f"{d.name}: its GPU outputs were written by the reservation's end (within 2 s)", f"{late}")
        ub = x["start"].get("used_before_hours")
        sec.check(close(ub, used_before[x["id"]], 1e-6), f"{cmd} {x['id']}: used_before_hours = own cumulative "
                  f"({used_before[x['id']]:.6f})", f"{ub}")
        worst = used_before[x["id"]] + 1.5 * x["start"]["projected_hours"]
        sec.check(worst <= WORK_ORDER_LIMIT, f"{cmd} {x['id']}: cumulative {used_before[x['id']]:.3f} + 1.5 x "
                  f"{x['start']['projected_hours']} = {worst:.3f} <= {WORK_ORDER_LIMIT}")
        sec.check(x["hours"] <= 1.5 * x["start"]["projected_hours"], f"{cmd}: charged {x['hours']:.4f} h within 1.5 x "
                  f"its projection")
    # overlaps
    srt = sorted(ledger, key=lambda x: x["t_start"])
    ov = [(a["id"], b["id"]) for a, b in zip(srt, srt[1:]) if b["t_start"] < a["t_end"]]
    sec.check(not ov, "no two reservations in the ledger overlap", f"{ov[:3]}")
    first_gi = gires[0]
    lit = literals_of(run_src, ("WORK_ORDER_USED_AT_START",))
    sec.check(close(lit["WORK_ORDER_USED_AT_START"], used_before[first_gi["id"]], 1e-6),
              "gi_run.WORK_ORDER_USED_AT_START = own cumulative before the first gi reservation",
              f"{lit['WORK_ORDER_USED_AT_START']} vs {used_before[first_gi['id']]}")
    t_freeze = int(git(repo, "show", "-s", "--format=%ct", FREEZE).strip())
    at_freeze = sum(x["hours"] for x in ledger if x["t_end"] <= t_freeze)
    sec.check(abs(at_freeze - 22.042) < 0.001, f"cumulative at the freeze commit = 22.042 h (own {at_freeze:.4f})")
    total = sum(x["hours"] for x in ledger)
    study = sum(x["hours"] for x in gires)
    sec.info["ledger"] = {"total_hours": total, "study_hours": study, "at_freeze": at_freeze,
                          "new_population_hours": sum(x["hours"] for x in gires if x["command"] in NEW_TIMED)}
    sec.check(total + 0 <= 32.0 and study <= 8.0, f"cumulative {total:.4f} h within 32; the study {study:.4f} h within 8")
    ctx["ledger"], ctx["ledger_gi"] = ledger, gires
    # manifests' budget blocks
    for kind, cmd in pairs[:4]:
        b = json.loads((R[kind][0] / "manifest.json").read_text())["config"]["budget"]
        x = byid[json.loads((R[kind][0] / "summary.json").read_text())["reservation"]["id"]]
        sec.check(close(b["used_hours"], used_before[x["id"]], 1e-6) and b["fits"] and
                  close(b["worst_case_hours"], used_before[x["id"]] + 1.5 * b["projected_hours"], 1e-9),
                  f"{kind} manifest budget block = own ledger arithmetic")
    # co-tenancy
    logs = gi(ctx) / "logs"
    co = {f.name: cotenancy(f) for f in logs.glob("cotenancy_*.log")}
    sec.info["cotenancy"] = {}
    for x in gires:
        lines = [(t, l1, rn) for f, v in co.items() for (t, l1, rn, _tot) in v if x["t_start"] <= t <= x["t_end"]]
        covered = any(v and v[0][0] <= x["t_start"] + 31 and v[-1][0] >= x["t_end"] - 31 for v in co.values())
        info = {"samples": len(lines), "covered": covered,
                "max_load1": max((l for _t, l, _r in lines), default=None),
                "mean_load1": (sum(l for _t, l, _r in lines) / len(lines)) if lines else None,
                "max_running": max((r for *_x, r in lines), default=None)}
        sec.info["cotenancy"][f"{x['command']} {x['id']}"] = info
        if x["command"] in NEW_TIMED:
            sec.check(covered and info["max_load1"] is not None and info["max_load1"] < 2.5,
                      f"{x['command']}: co-tenancy log covers the window; 1-min load stays below 2.5 "
                      f"(max {info['max_load1']}, mean {info['mean_load1']:.2f})")
    # CPU runs of the study against the timed windows
    cpu_runs = []
    for kind in ("constraint", "decode_dev", "lexical_dev", "lexical_new", "swap_new", "analysis"):
        for d in R[kind]:
            ts = [f.stat().st_mtime for f in d.iterdir()]
            cpu_runs.append((d.relative_to(gi(ctx)).as_posix(), run_time(d.name), max(ts)))
    over = defaultdict(list)
    for x in gires:
        for name, a, b in cpu_runs:
            if a < x["t_end"] and b > x["t_start"]:
                over[x["command"] + " " + x["id"]].append(name)
    sec.info["study_cpu_runs_overlapping_timed_windows"] = dict(over)
    sec.check(not any(k.split(" 2026")[0] in NEW_TIMED for k in over),
              "no CPU run of the study overlapped a new-population timed reservation", f"{dict(over)}")
    # analysis cost arithmetic
    a = json.loads((R["analysis"][0] / "analysis.json").read_text())
    g4sec = {r["task_id"]: r["seconds"] for r in g}
    bad = []
    for arch in ("T4", "C4"):
        for p in POLICIES:
            rows = [r for r in recs if r["fit"].startswith(arch) and r["policy"] == p]
            ctrl = sum(r["controller_seconds"] for r in rows)
            grd = sum(g4sec[r["task_id"]] for r in rows) if p == "G4" else 0.0
            exe = sum(r["score"]["predicted"]["elapsed_seconds"] for r in rows)
            ok = sum(ctx["ex"][(r["task_id"], r["final"]["sql"] if r["final"]["kind"] == "sql" else None)]["correct"]
                     for r in rows)
            c = a["cost"][f"{arch}-{p}"]
            mine = {"requests": len(rows), "ex_correct": ok, "controller_seconds": ctrl, "grounding_call_seconds": grd,
                    "submitted_execution_seconds": exe, "per_request_seconds": (ctrl + grd + exe) / len(rows),
                    "per_ex_correct_seconds": (ctrl + grd + exe) / ok if ok else None}
            for k, v in mine.items():
                if not close(v, c[k]):
                    bad.append((f"{arch}-{p}", k, v, c[k]))
    d = jl(R["direct_new"][0] / "records.jsonl")
    dsec = sum(r["seconds"] for r in d) + sum(r["score"]["predicted"]["elapsed_seconds"] for r in d)
    dok = sum(ctx["ex"][(r["task_id"], r["final"]["sql"] if r["final"]["kind"] == "sql" else None)]["correct"] for r in d)
    for k, v in {"per_request_seconds": dsec / 64, "per_ex_correct_seconds": dsec / dok, "ex_correct": dok}.items():
        if not close(v, a["cost"]["direct-B4-short"][k]):
            bad.append(("direct", k, v, a["cost"]["direct-B4-short"][k]))
    sec.check(not bad, "analysis costs = own arithmetic: controller seconds + the G4 call for every G4 request "
              "(fallbacks included) + submitted-query execution", f"{bad[:5]}")
    mean = {p: sum(r["controller_seconds"] for r in recs if r["policy"] == p) / 384 for p in POLICIES}
    sec.info["controller_seconds_mean"] = mean
    sec.check(abs(mean["G4"] - mean["G0"]) < 0.5, "G4's controller_seconds excludes the G4 call (mean within 0.5 s "
              f"of G0's; a call is ~9 s): {mean}")
    sec.check(close(a["cost"]["T4-G4"]["grounding_call_seconds"], 3 * sum(g4sec.values())),
              "the G4 call is charged once per seed for each of the 64 questions")


@section
def sec_constraint(sec: Section, ctx: dict) -> None:
    """Item 8."""
    repo, runs, R = ctx["repo"], ctx["runs"], ctx["R"]
    d = R["constraint"][0]
    rows = jl(d / "records.jsonl")
    m = json.loads((d / "manifest.json").read_text())
    sec.check(not m["git"]["research_tree_dirty"] and m["config"]["device"] == "cpu", "constraint: clean tree, CPU")
    commit = m["git"]["commit"]
    sec.check(git_show(repo, commit, H + "gi_constraint.py") == git_show(repo, FROZEN_RUN, H + "gi_constraint.py"),
              f"gi_constraint.py at the run commit {commit[:7]} = at {FROZEN_RUN}")
    ex_ = ctx["ex"]

    def corr(t, v):
        return ex_[(t, v["sql"] if v["kind"] == "sql" else None)]["correct"]
    saved = ctx["saved"]
    own_rows = []
    sr_bad, sj_bad = [], []
    for r in rows:
        t, f = r["task_id"], r["fit"]
        o = {"fit": f, "pop": r["population"], "v1c": corr(t, r["v1"]), "rc": corr(t, r["restricted"]),
             "changed": r["v1"]["actions"] != r["restricted"]["actions"],
             "changed_sql": r["v1"]["sql"] != r["restricted"]["sql"],
             "v1_sr": same_reference_on(r["v1"]["sql"]), "r_sr": same_reference_on(r["restricted"]["sql"]),
             "v1_sj": self_join_selects(r["v1"]["sql"]), "r_sj": self_join_selects(r["restricted"]["sql"]),
             "v1_to": ex_[(t, r["v1"]["sql"])]["pred"] == "timeout", "r_to": ex_[(t, r["restricted"]["sql"])]["pred"]
             == "timeout", "v1_err": ex_[(t, r["v1"]["sql"])]["pred"] == "error",
             "r_err": ex_[(t, r["restricted"]["sql"])]["pred"] == "error",
             "v1_abs": r["v1"]["kind"] == "abstain", "r_abs": r["restricted"]["kind"] == "abstain",
             "repro": r["v1"]["actions"] == saved[(f, t)]}
        for p, k in (("v1", "v1_sr"), ("restricted", "r_sr")):
            if (o[k] > 0) != (r[p]["same_reference_on"] > 0):
                sr_bad.append((f, t, p, o[k], r[p]["same_reference_on"]))
        for p, k in (("v1", "v1_sj"), ("restricted", "r_sj")):
            if (o[k] > 0) != (r[p]["self_joins"] > 0):
                sj_bad.append((f, t, p, o[k], r[p]["self_joins"]))
        own_rows.append(o)
        if o["repro"] != r["v1_equals_saved_decode"]:
            sec.check(False, f"{f} {t}: v1_equals_saved_decode = own comparison")
    sec.check(not sr_bad, "own SQL-level same-reference ON count agrees (present/absent) with every record",
              f"{sr_bad[:5]}")
    sec.check(not sj_bad, "own SQL-level self-join count agrees (present/absent) with every record", f"{sj_bad[:5]}")
    sec.check(all(o["r_sr"] == 0 for o in own_rows), "no restricted decode has a same-reference ON comparison (own)")

    def tally(rs):
        n = lambda f: sum(1 for o in rs if f(o))  # noqa: E731
        return {"requests": len(rs), "v1_correct": n(lambda o: o["v1c"]), "restricted_correct": n(lambda o: o["rc"]),
                "changed_decisions": n(lambda o: o["changed"]), "changed_sql": n(lambda o: o["changed_sql"]),
                "wrong_to_correct": n(lambda o: not o["v1c"] and o["rc"]),
                "correct_to_wrong": n(lambda o: o["v1c"] and not o["rc"]),
                "v1_timeouts": n(lambda o: o["v1_to"]), "restricted_timeouts": n(lambda o: o["r_to"]),
                "v1_errors": n(lambda o: o["v1_err"]), "restricted_errors": n(lambda o: o["r_err"]),
                "v1_same_reference_outputs": n(lambda o: o["v1_sr"] > 0),
                "restricted_same_reference_outputs": n(lambda o: o["r_sr"] > 0),
                "v1_self_join_outputs": n(lambda o: o["v1_sj"] > 0),
                "restricted_self_join_outputs": n(lambda o: o["r_sj"] > 0),
                "v1_abstentions": n(lambda o: o["v1_abs"]), "restricted_abstentions": n(lambda o: o["r_abs"]),
                "v1_reproduces_saved_decode": n(lambda o: o["repro"])}
    summ = json.loads((d / "summary.json").read_text())
    mine = {"by_fit_population": {}, "by_architecture_population": {}}
    for o in own_rows:
        mine["by_fit_population"].setdefault(f"{o['fit']}/{o['pop']}", []).append(o)
        mine["by_architecture_population"].setdefault(f"{o['fit'].split('/')[0]}/{o['pop']}", []).append(o)
    for k in mine:
        mine[k] = {kk: tally(v) for kk, v in sorted(mine[k].items())}
    diffs = [(k, kk, f, mine[k][kk][f], summ[k][kk][f]) for k in mine for kk in mine[k] for f in mine[k][kk]
             if mine[k][kk][f] != summ[k][kk].get(f)]
    sec.check(not diffs and set(mine["by_fit_population"]) == set(summ["by_fit_population"]),
              "constraint summary: every count (by fit and population, by architecture) = own tally", f"{diffs[:6]}")
    pooled = {a: {f: sum(mine["by_architecture_population"][f"{a}/{p}"][f] for p in ("historical", "new"))
                  for f in ("requests", "v1_correct", "restricted_correct", "changed_decisions", "wrong_to_correct",
                            "correct_to_wrong", "v1_same_reference_outputs", "restricted_same_reference_outputs",
                            "v1_reproduces_saved_decode")} for a in ("T4-v1", "C4-v1")}
    sec.info["pooled_by_architecture"] = pooled
    # code: the same condition as v2's push_scalar, applied to v1 only
    act = git_show(repo, FROZEN_RUN, H + "actions.py").decode()
    ps = func_source(act.encode(), "push_scalar", cls="_NodeFrame")
    sec.check('m.grounding == "v2" and self.op in COMPARE_ACTIONS and len(self.children) == 1' in ps and
              "isinstance(self.children[0], ir.Column)" in ps and 'if self.ctx.clause == "ON":' in ps and
              "exclude = first" in ps and "child_index=len(self.children)" in ps,
              "v2's push_scalar: second operand of a comparison in ON whose first operand is a column -> exclude")
    gc = git_show(repo, FROZEN_RUN, H + "gi_constraint.py").decode()
    cs = func_source(gc.encode(), "push", cls="RestrictedV1Machine")
    sec.check("parent.op in A.COMPARE_ACTIONS" in cs and "frame.child_index == 1" in cs and
              "len(parent.children) == 1" in cs and "isinstance(parent.children[0], ir.Column)" in cs and
              'parent.ctx.clause == "ON"' in cs and "frame.exclude = parent.children[0]" in cs and
              "frame.exclude is None" in cs, "RestrictedV1Machine.push: the same four conditions, the same exclusion")
    sec.check("first" not in cs.replace("children[0]", ""),
              "RestrictedV1Machine sets no pointer-relation operand (`first`): nothing else of v2")
    sec.check("def _excluded(self, m, item: int, occurrence: tuple) -> bool:" in act and
              "grounding" not in func_source(act.encode(), "_excluded", cls="_ExprSlot"),
              "the slot's exclusion is honoured by any machine (no grounding check in _ExprSlot._excluded)")
    run_src = func_source(git_show(repo, commit, H + "gi_run.py"), "constraint_cmd")
    sec.check('K.prepare(encs[tid], grounding="v1")' in run_src and "V1_ARMS" in run_src and
              "GC.machine_class(cls)" in run_src, "constraint_cmd: v1 checkpoints, v1 preparation, the class swapped in")
    all_src = git_show(repo, FROZEN_RUN, H + "gi_run.py").decode()
    sec.check(all_src.count("RestrictedV1Machine") == 1 and all_src.count("machine_class") == 1,
              "the restricted machine is used only by the constraint command")
    # own re-decode: v1 and v1 with v2's push_scalar condition (flagged worker)
    if ctx["no_worker"]:
        return
    ck = ctx["ckpt_paths"]
    jobs = []
    for r in rows:
        for kind in ("v1", "restricted_own"):
            jobs.append({"id": f"{kind}|{r['fit']}|{r['task_id']}", "task_id": r["task_id"], "ckpt": ck[r["fit"]],
                         "kind": kind, "precision": "highest"})
    out = run_worker(ctx, "decode", {"jobs": jobs, "backbone_identity": ctx["backbone_identity"]}, "decode_constraint")
    got = {x["id"]: x for x in out["results"]}
    v1_eq = sum(got[f"v1|{r['fit']}|{r['task_id']}"]["actions"] == r["v1"]["actions"] for r in rows)
    re_eq = sum(got[f"restricted_own|{r['fit']}|{r['task_id']}"]["actions"] == r["restricted"]["actions"] for r in rows)
    sec.info["own_redecode"] = {"v1_equal": v1_eq, "restricted_equal": re_eq, "rows": len(rows)}
    sec.check(v1_eq >= len(rows) - 3, f"own CPU re-decode of v1 = the recorded v1 decode ({v1_eq}/{len(rows)})")
    sec.check(re_eq >= len(rows) - 3, f"own restricted machine (v2's push_scalar on v1) = the recorded restricted "
              f"decode ({re_eq}/{len(rows)})")


@section
def sec_dev(sec: Section, ctx: dict) -> None:
    """Item 10."""
    repo, R = ctx["repo"], ctx["R"]
    d = gi(ctx) / "decode_dev" / DEV_DECODE
    rows = jl(d / "records.jsonl")
    m = json.loads((d / "manifest.json").read_text())
    pin = json.loads((repo / H / "pins/gi_protocol.json").read_text())
    sec.check(m["config"]["ground_run"] == pin["selected_on_dev"]["dev_ground_run"] and
              pin["selected_on_dev"]["dev_decode_run"].endswith(DEV_DECODE),
              "the gate's decode run used the pinned dev ground run (v2 prompt)")
    inf = {}
    for p in ("Glex", "G4", "oracle"):
        n = sum(r["policies"][p]["sql"] != r["policies"]["G0"]["sql"] for r in rows)
        inf[p] = {"changed_sql": n, "share": n / len(rows)}
    sec.info["influence"] = inf
    ds = json.loads((d / "dev_summary.json").read_text())
    sec.check(len(rows) == 768 and all(inf[p]["changed_sql"] == ds["influence"][p]["changed_sql"] for p in inf),
              "gate: own changed-SQL counts = dev_summary", f"{inf}")
    sec.check((inf["Glex"]["changed_sql"], inf["G4"]["changed_sql"], inf["oracle"]["changed_sql"]) == (80, 78, 68),
              "gate: Glex 80, G4 78, oracle 68 of 768, as the protocol states")
    sec.check(all(round(inf[p]["share"], 4) == pin["gate"]["shares"][p] for p in inf), "gate shares = the pin's")
    passes = any(v["share"] >= 0.10 for v in inf.values())
    sec.check(passes == pin["gate"]["passes"] is True and ds["gate"]["passes"], "the gate passes (some share >= 10%)")
    sec.info["gate_margin"] = {p: inf[p]["changed_sql"] - 76.8 for p in inf}
    act = {p: sum(r["policies"][p]["actions"] != r["policies"]["G0"]["actions"] for r in rows) for p in inf}
    sec.info["changed_actions"] = act
    # EX table (own scores)
    ex_ = ctx["ex"]
    table = {}
    for arch in ("T4-v2", "C4-v2"):
        rs = [r for r in rows if r["fit"].startswith(arch)]
        table[arch] = {p: sum(ex_[(r["task_id"], r["policies"][p]["sql"] if r["policies"][p]["kind"] == "sql" else None)]
                              ["correct"] for r in rs if p in r["policies"]) for p in ("G0", "Glex", "G4", "oracle", "G4-swap")}
    sec.info["dev_ex"] = table
    sec.check(all(table[a][p] == ds["by_arch"][a][p] for a in table for p in table[a]), "dev EX per policy = dev_summary")
    sec.check([table["C4-v2"][p] for p in ("G0", "Glex", "G4", "oracle")] == [68, 69, 71, 74] and
              [table["T4-v2"][p] for p in ("G0", "Glex", "G4", "oracle")] == [70, 71, 69, 69],
              "dev EX table = the protocol's (C4 68/69/71/74, T4 70/71/69/69)")
    swap = sum(r["policies"]["G4-swap"]["sql"] != r["policies"]["G0"]["sql"] for r in rows if "G4-swap" in r["policies"])
    sec.check(swap == 36, f"swap check: another question's record changes {swap} of 768 (protocol: 36)")
    saved = ctx["saved"]
    rep = sum(r["policies"]["G0"]["actions"] == saved[(r["fit"], r["task_id"])] for r in rows)
    sec.check(rep == 766 == ds["g0_reproduces_saved_gpu_decode"], f"G0 on the CPU = the saved GPU decode: own {rep}/768")
    if ctx.get("own_valid"):
        gname = pin["selected_on_dev"]["dev_ground_run"].split("/")[-1]
        nv = sum(ctx["own_valid"][(gname, t)]["valid"] for t in ctx["dev_ids"])
        sec.check(nv == 124 == ds["records"]["G4"]["valid"], f"dev G4 records valid: own {nv}/128 (protocol 124)")
        g1 = R["ground_dev"][0].name
        sec.info["dev_v1_prompt_valid"] = sum(ctx["own_valid"][(g1, t)]["valid"] for t in ctx["dev_ids"])
    # the partner rule
    db = {t: ctx["rel"].by_id[t]["db_id"] for t in ctx["dev_ids"]}
    by = defaultdict(list)
    for t in ctx["dev_ids"]:
        by[db[t]].append(t)
    partner = {t: q[(k + 1) % len(q)] for q in by.values() if len(q) > 1 for k, t in enumerate(q)}
    sec.check(all(r["swap_partner"] == partner.get(r["task_id"]) for r in rows), "dev swap partners = own rule")
    # git order
    t_freeze = int(git(repo, "show", "-s", "--format=%ct", FREEZE).strip())
    t_gate = int(git(repo, "show", "-s", "--format=%ct", GATE_COMMIT).strip())
    dev_runs = R["decode_dev"] + R["ground_dev"] + R["lexical_dev"]
    ends = {x.name: max(f.stat().st_mtime for f in x.iterdir()) for x in dev_runs}
    sec.info["dev_run_ends"] = {k: iso(v) for k, v in ends.items()}
    sec.info["freeze_commit_time"] = iso(t_freeze)
    sec.info["gate_commit_time"] = iso(t_gate)
    sec.check(all(v < t_freeze for v in ends.values()), f"{FREEZE} postdates every dev run's last write "
              f"(latest {iso(max(ends.values()))}, freeze {iso(t_freeze)})")
    starts = {x.name: run_time(x.name) for x in R["decode_dev"]}
    sec.check(all(v > t_gate for v in starts.values()), f"both dev decode runs started after {GATE_COMMIT} "
              f"({iso(t_gate)}): {sorted(iso(v) for v in starts.values())}")
    gsrc = git_show(repo, GATE_COMMIT, H + "gi_run.py")
    sec.check(literals_of(gsrc, ("GATE_SHARE",)).get("GATE_SHARE") == 0.10, f"{GATE_COMMIT}: GATE_SHARE = 0.10")
    ptxt = git_show(repo, GATE_COMMIT, H + "docs/PROTOCOL_grounding_interface.md").decode()
    sec.check("least 10% of the 768 development decodes" in re.sub(r"\s+", " ", ptxt) and
              "If none does, the study stops before the new population" in re.sub(r"\s+", " ", ptxt),
              f"{GATE_COMMIT}: the protocol draft states the 10% gate and the stop")
    for c in (m["git"]["commit"], FREEZE, FROZEN_RUN):
        sec.check(literals_of(git_show(repo, c, H + "gi_run.py"), ("GATE_SHARE",)).get("GATE_SHARE") == 0.10,
                  f"GATE_SHARE = 0.10 at {c[:7]}")
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", GATE_COMMIT, m["git"]["commit"]], cwd=repo).returncode
    sec.check(anc == 0, f"{GATE_COMMIT} is an ancestor of the decode runs' commit {m['git']['commit'][:7]}")
    first_ground = R["ground_dev"][0]
    sec.info["note_first_ground_dev_before_gate_commit"] = run_time(first_ground.name) < t_gate


@section
def sec_analysis(sec: Section, ctx: dict) -> None:
    """Item 9."""
    R, rel = ctx["R"], ctx["rel"]
    a = json.loads((R["analysis"][0] / "analysis.json").read_text())
    am = json.loads((R["analysis"][0] / "manifest.json").read_text())
    sec.check(a["eval_run"] == f"grounding_interface/eval/{R['eval_new'][0].name}" and not am["git"]["research_tree_dirty"]
              and am["git"]["commit"].startswith(FROZEN_RUN), "analysis: the eval-new run, clean tree at the run commit")
    ex_ = ctx["ex"]
    new = ctx["new_ids"]
    db_of = {t: rel.by_id[t]["db_id"] for t in new}
    seeds = [100, 101, 102]
    correct = defaultdict(dict)
    by = {}
    for r in ctx["eval_recs"]:
        arch, s = r["fit"].split("-")[0], int(r["fit"].split("/")[1])
        c = ex_[(r["task_id"], r["final"]["sql"] if r["final"]["kind"] == "sql" else None)]["correct"]
        correct[(f"{arch}-{r['policy']}", s)][r["task_id"]] = c
        by[(arch, r["policy"], s, r["task_id"])] = r
    counts = {f"{k}/{s}": sum(v.values()) for (k, s), v in sorted(correct.items())}
    sec.info["counts"] = counts
    sec.check(counts == a["counts"], "EX counts per policy, architecture and seed = analysis.json", f"{counts}")
    comp = {}
    for arch in ("T4", "C4"):
        comp[arch] = {}
        for x, y in (("G4", "G0"), ("Glex", "G0"), ("G4", "Glex")):
            comp[arch][f"{x}-{y}"] = own_paired(correct, f"{arch}-{x}", f"{arch}-{y}", seeds, new, db_of)
    inter = {("I", s): {t: correct[("T4-G4", s)][t] - correct[("T4-G0", s)][t] for t in new} for s in seeds}
    inter.update({("Z", s): {t: correct[("C4-G4", s)][t] - correct[("C4-G0", s)][t] for t in new} for s in seeds})
    ia = own_paired(inter, "I", "Z", seeds, new, db_of)
    keys = ("per_seed", "pooled", "sign", "conditions", "all_conditions", "per_database", "leave_one_database_out",
            "per_question_histogram", "pooled_out_of")
    bad = [(arch, k, f) for arch in comp for k in comp[arch] for f in keys if comp[arch][k][f] != a["comparisons"][arch][k][f]]
    bad += [("interaction", f) for f in keys if ia[f] != a["interaction_T4_minus_C4_of_G4_minus_G0"][f]]
    sec.check(not bad, "paired G4-G0, Glex-G0, G4-Glex per architecture and the interaction = analysis.json "
              "(per seed, pooled, conditions, per database, leave-one-out, histogram)", f"{bad[:6]}")
    sec.info["comparisons"] = {arch: {k: {"per_seed": v["per_seed"], "pooled": v["pooled"],
                                          "conditions": v["conditions"], "consistent": v["all_conditions"]}
                                      for k, v in c.items()} for arch, c in comp.items()}
    sec.info["interaction"] = {"per_seed": ia["per_seed"], "pooled": ia["pooled"], "conditions": ia["conditions"]}
    sec.check(not any(v["all_conditions"] for c in comp.values() for v in c.values()) and not ia["all_conditions"],
              "no comparison and no interaction is a consistent difference")
    # direct control
    d = jl(R["direct_new"][0] / "records.jsonl")
    d_ok = sum(ex_[(r["task_id"], r["final"]["sql"] if r["final"]["kind"] == "sql" else None)]["correct"] for r in d)
    best = {str(s): max(counts[f"{x}-{p}/{s}"] for x in ("T4", "C4") for p in POLICIES) for s in seeds}
    mine_d = {"correct": d_ok, "requests": len(d), "abstentions": sum(r["final"]["kind"] == "abstain" for r in d),
              "truncated": sum(r["call"]["finish_reason"] != "eos" for r in d),
              "timeouts": sum(ex_[(r["task_id"], r["final"]["sql"])]["pred"] == "timeout" for r in d),
              "best_controller_policy_per_seed": best, "beats_best_controller_every_seed": all(d_ok > v for v in best.values())}
    dd = [(k, v, a["direct"][k]) for k, v in mine_d.items() if v != a["direct"][k]]
    sec.check(not dd, "direct control = analysis.json (correct, abstentions, truncation, timeouts, best per seed, "
              "beats every seed)", f"{dd}")
    sec.check(close(sum(r["seconds"] for r in d) / len(d), a["direct"]["call_seconds_mean"]), "direct call mean seconds")
    sec.info["direct"] = mine_d
    # direct vs the best controller policy, paired per seed on the 64 questions
    dcorr = {r["task_id"]: ex_[(r["task_id"], r["final"]["sql"])]["correct"] for r in d}
    sec.info["direct_minus_policy_per_seed"] = {f"{x}-{p}": {str(s): d_ok - counts[f"{x}-{p}/{s}"] for s in seeds}
                                                for x in ("T4", "C4") for p in POLICIES}
    sec.info["direct_only_vs_any_controller"] = {
        "direct_correct_no_controller_correct": sum(1 for t in new if dcorr[t] and not any(
            correct[(f"{x}-{p}", s)][t] for x in ("T4", "C4") for p in POLICIES for s in seeds)),
        "controller_some_correct_direct_wrong": sum(1 for t in new if not dcorr[t] and any(
            correct[(f"{x}-{p}", s)][t] for x in ("T4", "C4") for p in POLICIES for s in seeds))}
    # decodes
    bad = []
    mine_dec = {}
    for arch in ("T4", "C4"):
        for p in ("Glex", "G4"):
            rows = [(by[(arch, p, s, t)], by[(arch, "G0", s, t)]) for s in seeds for t in new]
            cc = lambda r: ex_[(r["task_id"], r["final"]["sql"] if r["final"]["kind"] == "sql" else None)]["correct"]  # noqa
            mine = {"changed_decisions_vs_G0": sum(x["actions"] != g["actions"] for x, g in rows),
                    "changed_sql_vs_G0": sum(x["final"]["sql"] != g["final"]["sql"] for x, g in rows),
                    "wrong_to_correct": sum(cc(x) > cc(g) for x, g in rows),
                    "correct_to_wrong": sum(cc(x) < cc(g) for x, g in rows),
                    "fallback_to_G0": sum(not (x["grounding"] or {}).get("valid", False) for x, _g in rows)}
            mine_dec[f"{arch}-{p}"] = mine
            bad += [(f"{arch}-{p}", k, v, a["decodes"][f"{arch}-{p}"][k]) for k, v in mine.items()
                    if v != a["decodes"][f"{arch}-{p}"][k]]
    sec.check(not bad, "decode diagnostics (changed decisions and SQL, recoveries, regressions, fallbacks) = analysis",
              f"{bad[:5]}")
    sec.info["decodes"] = mine_dec
    # swap
    sw = jl(R["swap_new"][0] / "records.jsonl")
    cs = lambda t, v: ex_[(t, v["sql"] if v["kind"] == "sql" else None)]["correct"]  # noqa
    ms = {"decodes": len(sw), "changed_sql": sum(r["policies"]["G4"]["sql"] != r["policies"]["G4-swap"]["sql"] for r in sw),
          "own_correct": sum(cs(r["task_id"], r["policies"]["G4"]) for r in sw),
          "swapped_correct": sum(cs(r["task_id"], r["policies"]["G4-swap"]) for r in sw)}
    sec.check(all(ms[k] == a["swap"][k] for k in ms), "swap check = analysis.json", f"{ms}")
    sec.info["swap"] = ms
    byft = {(r["fit"], r["task_id"]): r for r in ctx["eval_recs"] if r["policy"] == "G4"}
    sec.info["swap_own_G4_equals_eval_G4_actions"] = sum(r["policies"]["G4"]["actions"] == byft[(r["fit"], r["task_id"])]
                                                         ["actions"] for r in sw)
    partner = {}
    byd = defaultdict(list)
    for t in new:
        byd[db_of[t]].append(t)
    for q in byd.values():
        if len(q) > 1:
            for k, t in enumerate(q):
                partner[t] = q[(k + 1) % len(q)]
    sec.check(all(r["partner"] == partner[r["task_id"]] for r in sw) and
              json.loads((R["swap_new"][0] / "manifest.json").read_text())["config"]["partners"] == partner,
              "swap partners = own rule (next same-database question, cyclic)")
    # gpu hours and ledger
    gires = ctx.get("ledger_gi") or []
    by_cmd = defaultdict(list)
    for x in gires:
        by_cmd[x["command"]].append(x["hours"])
    sec.info["gpu_hours_by_command"] = dict(by_cmd)
    sec.check(all(close(a["gpu_hours"][c], v[-1], 1e-9) for c, v in by_cmd.items()),
              "analysis gpu_hours: each command's (last) reservation = own ledger")
    sec.check(set(a["gpu_hours"]) == set(by_cmd) and all(len(v) == 1 for c, v in by_cmd.items() if c != "gi ground dev"),
              "analysis gpu_hours names every gi command (the two dev ground runs share one key)")
    if ctx.get("ledger"):
        sec.check(close(a["ledger"]["used_hours"], sum(x["hours"] for x in ctx["ledger"]), 1e-9),
                  "analysis ledger used_hours = own total")
    # reading
    up = lambda dd: dd["all_conditions"] and dd["sign"] > 0  # noqa
    reading = []
    g4g0 = {x: up(comp[x]["G4-G0"]) for x in comp}
    g4gl = {x: up(comp[x]["G4-Glex"]) for x in comp}
    glg0 = {x: up(comp[x]["Glex-G0"]) for x in comp}
    if mine_d["beats_best_controller_every_seed"]:
        reading.append("direct beats best controller policy in every seed")
    if not any(g4g0.values()) and not any(glg0.values()):
        reading.append("nothing improves consistently")
    sec.info["own_reading"] = reading
    sec.check(len(reading) == 2 and len(a["reading"]) == 2 and "direct short-budget B4" in a["reading"][0] and
              "nothing improves" in a["reading"][1] and not any(g4gl.values()),
              "the readings: the direct control beats the best policy every seed; nothing improves consistently")
    # record diagnostics (own, sqlglot) beside the analysis's (gold annotations by the producer's sql_to_ir)
    if ctx.get("items"):
        g4 = {r["task_id"]: r for r in jl(R["ground_new"][0] / "records.jsonl")}
        rec, prec, first = [], [], []
        for t in new:
            v = ctx["own_valid"][(R["ground_new"][0].name, t)]
            gt = sql_tables(rel.by_id[t]["SQL"])
            if not v["valid"] or not gt:
                continue
            ts = {ctx["items"][t][i][1].lower() for i in v["tables"]}
            rec.append(len(ts & gt) / len(gt))
            prec.append(len(ts & gt) / len(ts) if ts else 0)
            first.append(ctx["items"][t][v["tables"][0]][1].lower() in gt if v["tables"] else False)
        mine_r = {"scored": len(rec), "table_recall": sum(rec) / len(rec), "table_precision": sum(prec) / len(prec),
                  "first_in_gold_set": sum(first) / len(first)}
        sec.info["own_G4_record_tables_vs_gold (sqlglot)"] = mine_r
        sec.info["analysis_G4_record_tables"] = {k: a["records"]["G4"][k] for k in mine_r}
        sec.check(abs(mine_r["table_recall"] - a["records"]["G4"]["table_recall"]) < 0.03 and
                  abs(mine_r["table_precision"] - a["records"]["G4"]["table_precision"]) < 0.03,
                  "G4 record table recall and precision: own (sqlglot gold tables) within 0.03 of analysis.json")
        sec.check(close(sum(r["seconds"] for r in g4.values()) / 64, a["records"]["G4"]["call_seconds_mean"]),
                  "G4 call mean seconds = analysis")
    # the producer's report (flagged; read-only; on the bundle)
    if not ctx["no_report"]:
        env = {**os.environ, "TI_TNN_SCALING_RUNS": str(ctx["runs"]), "CUDA_VISIBLE_DEVICES": ""}
        p = subprocess.run([ctx["python"], "-m", "scaling.gi_report"], cwd=ctx["repo"] / "research/ti_tnn", env=env,
                           capture_output=True, text=True)
        (ctx["out"] / "gi_report_output.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
        ctx["producer_runs"].append({"tag": "gi_report", "cmd": "python -m scaling.gi_report", "exit": p.returncode})
        sec.check(p.returncode == 0, "python -m scaling.gi_report exits 0 on the bundle (producer code, flagged)",
                  (p.stdout + p.stderr)[-400:])
        sec.info["gi_report_tail"] = p.stdout[-1500:]


# --- context and main ---------------------------------------------------------------------------------------------
def saved_decodes(ctx: dict) -> dict:
    """{(fit, task): actions} of the performance study's one passing evaluation per seed (own selection)."""
    out = {}
    for s in (100, 101, 102):
        runs = [d for d in sorted((ctx["runs"] / "perf_v2/eval").glob(f"s{s}_*")) if
                json.loads((d / "summary.json").read_text()).get("passed") is True]
        assert len(runs) == 1, f"seed {s}: {len(runs)} passing perf_v2 evaluations"
        for lab in ("T4-v2", "C4-v2", "T4-v1", "C4-v1"):
            for r in jl(runs[0] / lab / "records.jsonl"):
                out[(f"{lab}/{s}", r["task_id"])] = r["attempts"][0]["actions"]
    return out


def checkpoints(sec: Section, ctx: dict) -> dict:
    """{fit: path} of the twelve final checkpoints, hashed here and checked against every manifest naming them."""
    runs, R = ctx["runs"], ctx["R"]
    out = {}
    for fit in V2_FITS + V1_FITS:
        lab, s = fit.split("/")
        found = sorted((runs / "perf_v2/train" / f"{lab}_s{s}" / "checkpoints").glob("ckpt_u*_*.pt"))
        final = [p for p in found if p.name.startswith("ckpt_u002999_")]
        sec.check(len(final) == 1 and found[-1] == final[0], f"{fit}: one final checkpoint u002999, the last")
        p = final[0]
        h = sha256_path(p)
        sec.check(p.stem.split("_")[-1] == h[:16], f"{fit}: checkpoint bytes hash to its name")
        out[fit] = (str(p), h)
    for kind in ("eval_new", "decode_dev", "constraint"):
        for d in R[kind]:
            cks = json.loads((d / "manifest.json").read_text())["config"]["checkpoints"]
            bad = [k for k, v in cks.items() if v["sha256"] != out[k][1] or not out[k][0].endswith(v["path"])]
            sec.check(not bad and len(cks) == 6, f"{d.relative_to(gi(ctx))}: its six checkpoints = own hashes", f"{bad}")
    return {k: v[0] for k, v in out.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=str(REPO_DEFAULT))
    ap.add_argument("--runs-root", default=os.environ.get("TI_TNN_SCALING_RUNS"))
    ap.add_argument("--out", default="verify_gi_out")
    ap.add_argument("--frozen-tree", default=None, help="the frozen worktree at 9c139ac (the worker's producer code)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ex-workers", type=int, default=3)
    ap.add_argument("--no-worker", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--sections", nargs="*", default=None)
    ap.add_argument("--worker", choices=("items", "decode"), default=None)
    ap.add_argument("--worker-in")
    ap.add_argument("--worker-out")
    ap.add_argument("--producer-tree")
    args = ap.parse_args(argv)
    if args.worker:
        return worker_main(args)
    if not args.runs_root:
        ap.error("--runs-root (or TI_TNN_SCALING_RUNS) is required")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    runs = Path(args.runs_root).resolve()
    out = Path(args.out).resolve()
    if runs in out.parents or out == runs:
        ap.error("--out may not be under the runs root")
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo).resolve()
    ctx = {"repo": repo, "runs": runs, "out": out, "frozen_tree": args.frozen_tree, "python": args.python,
           "threads": args.threads, "ex_workers": args.ex_workers, "no_worker": args.no_worker or not args.frozen_tree,
           "no_report": args.no_report, "producer_runs": [],
           "bundle": repo / H / "evidence/grounding_interface_v1"}
    t0 = time.time()
    ctx["rel"] = Release(runs, repo, FROZEN_RUN)
    order = ["bundle", "pins", "prior_use", "coverage", "worker_items", "permissions", "scores", "fallback",
             "consumption", "integrity", "timing", "constraint", "dev", "analysis"]
    results = {}
    for name in order:
        if args.sections and name not in args.sections and name not in ("pins", "coverage", "worker_items", "scores"):
            continue
        sec = Section(name)
        t1 = time.time()
        try:
            if name == "coverage":
                SECTIONS[name](sec, ctx)
                ctx["ckpt_paths"] = checkpoints(sec, ctx)
                ctx["saved"] = saved_decodes(ctx)
            else:
                SECTIONS[name](sec, ctx)
        except Exception as e:  # a crash is a failed check, never a silent pass
            import traceback
            sec.check(False, f"section {name} raised", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
        sec.info["seconds"] = round(time.time() - t1, 1)
        results[name] = sec.as_dict()
        print(f"[{name}] {results[name]['passed']} passed, {results[name]['failures']} failed "
              f"({sec.info['seconds']} s)", flush=True)
    own = Section("producer_code")
    own.check("scaling" not in {m.split(".")[0] for m in sys.modules}, "this process imported no producer module")
    own.info["separate_processes"] = ctx["producer_runs"]
    results["producer_code"] = own.as_dict()
    total = sum(s["passed"] + s["failures"] for s in results.values())
    fails = sum(s["failures"] for s in results.values())
    me = Path(__file__).resolve()
    report = {"verifier": str(me.relative_to(repo)) if repo in me.parents else str(me),
              "verifier_sha256": sha256_path(me), "repo_head": git(repo, "rev-parse", "HEAD").strip(),
              "runs_root": str(runs), "seconds": round(time.time() - t0, 1), "checks": total, "failures": fails,
              "sections": results}
    (out / "verify_gi.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    lines = [f"verify_gi: {total} checks, {fails} failures ({report['seconds']} s); sha256 {report['verifier_sha256']}"]
    for n, s in results.items():
        lines.append(f"[{n}] {s['passed']} passed, {s['failures']} failed")
        for c in s["checks"]:
            if not c["passed"]:
                lines.append(f"   FAIL {c['name']}: {str(c['detail'])[:600]}")
    (out / "verify_gi_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
