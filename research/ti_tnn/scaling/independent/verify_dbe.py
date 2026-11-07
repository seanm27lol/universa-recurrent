#!/usr/bin/env python3
"""Independent verifier for the TI-TNN database-evidence revision run (D0, D1, V1), the reviewer's, written apart
from the producers.

    CUDA_VISIBLE_DEVICES= taskset -c 0-4,10-14 nice -n 10 python verify_dbe.py --runs-root RUNS --out DIR
        [--repo REPO] [--frozen-tree TREE] [--workers 4] [--no-worker]
    (no machine default for --runs-root or --out; --repo defaults to the checkout holding this file)

Nothing is written under the runs root; a stat listing of the runs root before and after is compared.

**What it checks** (the brief's six items, plus the bundle):
1. scores: every D0, D1 and V1 SQL and its gold re-executed on the pinned snapshots (hashed here) with this file's own
   EX rule; every record mapped to its snapshot sha256 and SQL sha256; correct, status and row counts re-derived.
2. lineage: the shared initial call; own extraction rule; D1 = own raw observation + the frozen instruction; V1 = D1
   + exactly the frozen header and the recorded payload (or the frozen NOTHING / FAILED text); the recorded payload =
   a fresh db-evidence-v1.4 lookup (worker, latency aside); the order rule; the prompt token counts of all three calls
   re-rendered from the recorded contents (same prefix); the remaining time; own selection from own execution; the
   lookup failure and limit inventory; the code of `serve` (seeds, prefix, remaining time, lookup before it).
3. paired results: counts, V1-D1 / D1-D0 / V1-D0 per seed, pooled, per database, leaving each database out,
   recoveries, regressions, the +8 rule and the decision; the bootstrap re-implemented from the protocol text; the
   exact sign-flip; transitions; signal groups (own reading of the rendered payload); the per-question table.
4. costs from the records; the ledger (own parse): the reservation, the 1-hour allowance, 1.5 x projection; the
   co-tenancy log.
5. no reference access: no gold SQL, gold result or task id in any prompt, feedback message or payload; the run's
   manifests name the frozen commit with a clean tree; the frozen worktree is still at it and clean.
6. integrity: one request per question and seed, one score per policy, seed and question, one run directory, one
   reservation; and the bundle (SHA256SUMS, identity with the runs-root originals).
7. the results report (docs/REPORT_database_evidence_revision.md at the checkout's HEAD): every table row and number
   rebuilt from this file's own results, the signal-group statements, and the six discordant requests against the
   saved records.

**Discipline.** No producer module is imported by this process; producer constants and sources are read from git
(`git show <commit>:path`, ast). One flagged exception, the worker (`--worker`), a separate process importing the
frozen worktree's producer modules: the fresh db-evidence-v1.4 lookup, and the prompt's schema text and retrieved
examples. Its outputs are only compared with the records.
Third-party: sqlite3, sqlglot (read-only check, bare-statement extraction), transformers' tokenizer.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import subprocess
import sys
import time
import urllib.parse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[4]
H = "research/ti_tnn/scaling/"
FROZEN = "8e1ee1b"           # the commit the run executed at (the brief)
FROZEN_AT = "644c090"        # the protocol pin's frozen_at
RUN, ANALYSIS = "20261107T182054Z", "20261107T184900Z"
USED_AT_START, LIMIT = 23.489974, 24.489974
RELEASE = "data/raw/hf/bird23_train_filtered.train-00000-of-00001.jsonl"
POLICIES = ("D0", "D1", "V1")
COMPARISONS = (("V1", "D1"), ("D1", "D0"), ("V1", "D0"))
SEEDS = (200, 201)
LEAD = 8
TIMEOUT = 30.0
BOOT_SEED, BOOT_R = 20261107, 10_000


# --- the check list and own SQL machinery (as in verify_spr.py, the reviewer's) ------------------------------------
class Section:
    def __init__(self, name):
        self.name, self.checks, self.info = name, [], {}

    def check(self, ok, name, detail=""):
        ok = bool(ok)
        self.checks.append({"name": name, "passed": ok, "detail": detail if (not ok or detail) else ""})
        return ok

    def as_dict(self):
        return {"checks": self.checks, "passed": sum(c["passed"] for c in self.checks),
                "failures": sum(not c["passed"] for c in self.checks), "info": self.info}


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout


def git_show(repo, commit, path) -> bytes:
    b = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=repo, capture_output=True, check=True).stdout
    record_input(f"git:{commit}:{path}", sha_b(b), len(b))
    return b


def sha_b(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_t(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# --- the inputs this run read (path or git object -> sha256, bytes), recorded in the output ---------------------------
INPUTS: dict = {}


def record_input(label: str, sha: str, size: int) -> None:
    INPUTS[label] = {"sha256": sha, "bytes": size}


def rd(p) -> bytes:
    """Read an input file's bytes and record its sha256."""
    b = Path(p).read_bytes()
    record_input(str(Path(p).resolve()), sha_b(b), len(b))
    return b


def rt(p) -> str:
    return rd(p).decode("utf-8")


def sha_f(p) -> str:
    with open(p, "rb") as f:
        h = hashlib.file_digest(f, "sha256").hexdigest()
    record_input(str(Path(p).resolve()), h, os.path.getsize(p))
    return h


def jl(p) -> list:
    return [json.loads(x) for x in rt(p).splitlines() if x.strip()]


def utc(name: str) -> float:
    return datetime.strptime(name[:16], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def literals(source: bytes, names) -> dict:
    """Module-level literal assignments by name (ast.literal_eval); frozenset({...}) calls are unwrapped."""
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id in names and node.value is not None:
                    v = node.value
                    if isinstance(v, ast.Call) and getattr(v.func, "id", None) == "frozenset" and v.args:
                        v = v.args[0]
                    with contextlib.suppress(ValueError):
                        out[t.id] = ast.literal_eval(v)
    return out


def class_defaults(source: bytes, cls: str) -> dict:
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return {s.target.id: ast.literal_eval(s.value) for s in node.body
            if isinstance(s, ast.AnnAssign) and s.value is not None}


def functions(source: bytes) -> dict:
    """{name: ast node} of module-level functions and of class methods as Class.method."""
    tree = ast.parse(source)
    out = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            out[n.name] = n
        elif isinstance(n, ast.ClassDef):
            out[n.name] = n
            for m in n.body:
                if isinstance(m, ast.FunctionDef):
                    out[f"{n.name}.{m.name}"] = m
    return out


def idents(node) -> set:
    s = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            s.add(n.id)
        elif isinstance(n, ast.Attribute):
            s.add(n.attr)
        elif isinstance(n, ast.alias):
            s.add(n.name)
    return s


def closure(fns: dict, roots) -> set:
    """Module-internal functions reachable from roots: bare-name calls; a reached class reaches all its methods; a
    method reaches same-class methods it names (self.x)."""
    seen, todo = set(), list(roots)
    while todo:
        f = todo.pop()
        if f in seen or f not in fns:
            continue
        seen.add(f)
        names = idents(fns[f])
        cls = f.split(".")[0] if "." in f else None
        for g in fns:
            if g in seen:
                continue
            if "." not in g:
                if g in names:
                    todo.append(g)
            else:
                base, meth = g.split(".", 1)
                if base == f or base in names or (cls == base and meth in names):
                    todo.append(g)
    return seen


# --- own SQL machinery ---------------------------------------------------------------------------------------------
def ro_connect(path):
    conn = sqlite3.connect("file:" + urllib.parse.quote(str(Path(path).resolve())) + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def own_readonly(sql) -> "str | None":
    """None if exactly one statement whose root is a SELECT or a set operation with no write node (sqlglot, sqlite)."""
    import sqlglot
    from sqlglot import exp
    if not isinstance(sql, str) or not sql.strip():
        return "empty"
    try:
        stmts = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except Exception as e:  # noqa: BLE001
        return f"parse: {type(e).__name__}"
    if len(stmts) != 1:
        return f"{len(stmts)} statements"
    root = stmts[0]
    while isinstance(root, exp.Subquery):
        root = root.this
    if not isinstance(root, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        return f"root {type(root).__name__}"
    bad = tuple(getattr(exp, n) for n in ("Insert", "Update", "Delete", "Create", "Drop", "Pragma", "Attach",
                                           "Detach", "Command", "Transaction", "Commit", "Rollback", "Alter")
                if hasattr(exp, n))
    if any(isinstance(n, bad) for n in root.walk()):
        return "write node"
    return None


def cell_preview(v):
    """A preview cell as the model's tool showed it (tools contract: blobs named, text over 300 characters cut)."""
    if v is None or isinstance(v, (bool, int)):
        return v
    if isinstance(v, float):
        return v if v == v and v not in (float("inf"), float("-inf")) else repr(v)
    if isinstance(v, bytes):
        return f"<blob {len(v)} bytes>"
    t = str(v)
    return t if len(t) <= 300 else t[:300] + f"...[+{len(t) - 300} chars]"


def observe(job):
    """Own execution of one candidate: (key, {status, error, columns, row_count, rows (20-row preview), seconds,
    readonly, result_hash}). Status: no_sql | rejected | ok | error | timeout."""
    key, path, sql = job
    if not sql:
        return key, {"status": "no_sql", "row_count": None, "rows": [], "columns": [], "seconds": 0.0}
    why = own_readonly(sql)
    if why is not None:
        return key, {"status": "rejected", "error": why, "row_count": None, "rows": [], "columns": [], "seconds": 0.0}
    conn = ro_connect(path)
    t0 = time.perf_counter()
    dl = t0 + TIMEOUT
    conn.set_progress_handler(lambda: 1 if time.perf_counter() > dl else 0, 1000)
    out = {"columns": [], "rows": [], "row_count": None}
    try:
        cur = conn.execute(sql)
        out["columns"] = [d[0] for d in (cur.description or ())]
        rows = cur.fetchall()
        if time.perf_counter() > dl:
            raise sqlite3.OperationalError("interrupted")
        out.update(status="ok", error=None, row_count=len(rows), rows=[[cell_preview(v) for v in r] for r in rows[:20]],
                   truncated=len(rows) > 20,
                   result_hash=sha_t(repr(sorted({repr(r) for r in rows}))))
    except Exception as e:  # noqa: BLE001
        if time.perf_counter() > dl:
            out.update(status="timeout", error=f"exceeded {TIMEOUT:g} s")
        else:
            out.update(status="error", error=f"{type(e).__name__}: {e}")
        out["rows"], out["row_count"] = [], None
    finally:
        conn.close()
    out["seconds"] = time.perf_counter() - t0
    return key, out


def ex_pair(job):
    """Own EX: (key, {correct, status, pred, gold, seconds, ...}). Predicted, then gold, one read-only connection, one
    30 s deadline from the connection; correct iff both finish in time and set(predicted rows) == set(gold rows)."""
    key, path, pred, gold = job
    if pred is None:
        return key, {"correct": 0, "status": "abstain", "pred": "abstain", "gold": "not_run", "seconds": 0.0}
    conn = ro_connect(path)
    t0 = time.perf_counter()
    dl = t0 + TIMEOUT
    conn.set_progress_handler(lambda: 1 if time.perf_counter() > dl else 0, 1000)
    res = {}
    try:
        for side, sql in (("pred", pred), ("gold", gold)):
            if time.perf_counter() > dl:
                res[side], res[side + "_rows"] = "not_run", None
                continue
            s0 = time.perf_counter()
            try:
                rows = conn.execute(sql).fetchall()
                if time.perf_counter() > dl:
                    raise sqlite3.OperationalError("interrupted")
                res[side], res[side + "_rows"] = "ok", rows
            except Exception as e:  # noqa: BLE001
                res[side] = "timeout" if time.perf_counter() > dl else "error"
                res[side + "_error"] = f"{type(e).__name__}: {e}"[:300]
                res[side + "_rows"] = None
            res[side + "_seconds"] = time.perf_counter() - s0
    finally:
        conn.close()
    total = time.perf_counter() - t0
    pr, gr = res.pop("pred_rows"), res.pop("gold_rows")
    if total > TIMEOUT and res["pred"] == "ok" and res["gold"] == "ok":
        status = "timeout"
    elif res["pred"] != "ok":
        status = "predicted_" + res["pred"]
    elif res["gold"] != "ok":
        status = "gold_" + res["gold"]
    else:
        status = "correct" if set(pr) == set(gr) else "incorrect"
    res.update(correct=int(status == "correct"), status=status, seconds=total,
               pred_count=None if pr is None else len(pr), gold_count=None if gr is None else len(gr),
               pred_set=None if pr is None else sha_t(repr(sorted({repr(r) for r in pr}))),
               pred_list=None if pr is None else sha_t(repr([repr(r) for r in pr])),
               gold_set=None if gr is None else sha_t(repr(sorted({repr(r) for r in gr}))),
               gold_preview=None if gr is None else [[cell_preview(v) for v in r] for r in gr[:20]])
    return key, res


def run_jobs(fn, jobs, workers):
    """Parallel first pass; anything slow (> 5 s) or timed out runs again alone, and the serial result is final."""
    out = {}
    if not jobs:
        return out, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, v in pool.map(fn, jobs, chunksize=4):
            out[k] = v
    reruns = []
    for j in jobs:
        v = out[j[0]]
        if v.get("seconds", 0) > 5 or "timeout" in str(v.get("status")):
            k, again = fn(j)
            again["parallel_pass"] = {"status": v["status"], "seconds": v["seconds"]}
            out[k] = again
            reruns.append({"key": list(k) if isinstance(k, tuple) else k, "first": v["status"],
                           "first_seconds": round(v["seconds"], 2), "serial": again["status"],
                           "serial_seconds": round(again["seconds"], 2)})
    return out, reruns


# --- own extraction (the B4 harness's declared rule) ---------------------------------------------------------------
def own_first_statement(text: str) -> str:
    import sqlglot
    q, i, cut = None, 0, len(text)
    while i < len(text):
        ch = text[i]
        if q:
            if ch == q:
                if i + 1 < len(text) and text[i + 1] == q:
                    i += 1
                else:
                    q = None
        elif ch in "'\"`":
            q = ch
        elif ch == "[":
            q = "]"
        elif ch == ";":
            cut = i
            break
        i += 1
    text = text[:cut]
    ends = [len(text)] + sorted((m.start() for m in re.finditer(r"\n[ \t\r\f\v]*\n", text)), reverse=True)
    for e in ends:
        cand = text[:e].strip()
        try:
            if cand and len([s for s in sqlglot.parse(cand, read="sqlite") if s is not None]) == 1:
                return cand
        except Exception:  # noqa: BLE001
            continue
    return re.split(r"\n\s*\n", text, maxsplit=1)[0].strip()


def own_extract(content: str):
    """(sql, how): the last ``` block labelled sql, sqlite or nothing; else a bare reply starting with SELECT/WITH,
    its first statement. Fences are paired in order of appearance."""
    pos = [m.start() for m in re.finditer("```", content)]
    blocks = []
    for a, b in zip(pos[0::2], pos[1::2]):
        inner = content[a + 3:b]
        m = re.match(r"([A-Za-z0-9_+-]*)[ \t]*\n?", inner)
        label = m.group(1)
        if label.lower() in ("", "sql", "sqlite"):
            blocks.append(inner[m.end():])
    if blocks:
        sql = blocks[-1].strip().rstrip(";").strip()
        return (sql or None), "fence"
    s = content.strip()
    if re.match(r"(?is)^(select|with)\b", s):
        return (own_first_statement(s) or None), "bare"
    return None, "none"


# --- own feedback renderings (the declared message formats) --------------------------------------------------------
def own_raw(ex: dict, cell_chars=60, limit=4000) -> str:
    st = ex.get("status")
    if st != "ok":
        return f"Execution status: {st}. Error: {ex.get('error')}"
    rows = ex.get("rows") or []

    def c(v):
        s = "NULL" if v is None else str(v)
        return s if len(s) <= cell_chars else s[:cell_chars] + "..."
    lines = [f"Execution status: ok. Rows returned: {ex.get('row_count')}"
             + (f" (showing the first {len(rows)})" if ex.get("truncated") else "") + ".",
             "Columns: " + ", ".join(map(str, ex.get("columns") or []))]
    lines += ["  " + " | ".join(c(v) for v in r) for r in rows]
    t = "\n".join(lines)
    return t if len(t) <= limit else t[:limit] + "\n  ... (preview truncated)"




# --- the worker (producer code, separate process, flagged) ---------------------------------------------------------
def worker_main(job_path: str, out_path: str, frozen_tree: str) -> None:
    sys.path.insert(0, str(Path(frozen_tree) / "research" / "ti_tnn"))
    from scaling import contracts as C  # noqa: PLC0415  (producer code: the flagged worker only)
    from scaling import data, db_evidence, gi_llm, gi_run, retrieval  # noqa: PLC0415
    from scaling import schema as schema_mod  # noqa: PLC0415
    job = json.loads(Path(job_path).read_text())
    out = {"prompts": {}, "lookup": {}, "version": db_evidence.EVIDENCE_VERSION,
           "db_evidence_file": db_evidence.__file__}
    limits, index, schemas = C.Limits(), retrieval.training_pool_index(), {}
    for tid in job["prompts"]:
        task = data.task(tid)
        text, examples = gi_run._prompt_inputs(task, limits, index)
        msgs = gi_llm.messages(gi_llm.SYSTEM_DIRECT, task, text, examples)
        out["prompts"][tid] = {"system": msgs[0]["content"], "user": msgs[1]["content"], "schema_text": text,
                               "retrieved": [e.task_id for e in examples]}
    for key, item in job["lookup"].items():
        task = data.task(item["task_id"])
        snap = data.snapshot(task.db_id)
        if task.db_id not in schemas:
            schemas[task.db_id] = schema_mod.load_schema(task.db_id, snap)
        try:
            pkg = db_evidence.retrieve(task.question, task.evidence, item["sql"], schemas[task.db_id], snap.path)
            pkg.pop("latency_seconds", None)
            out["lookup"][key] = {"ok": True, "package": pkg}
        except Exception as e:  # noqa: BLE001
            out["lookup"][key] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    Path(out_path).write_text(json.dumps(out, default=str))


def own_signals(text: str) -> dict:
    """Signal classes read from the rendered payload a model saw: value not stored with candidates; other (a literal
    found elsewhere, a flagged join, a representation note)."""
    blocks, cur = [], None
    for line in (text or "").split("\n"):
        if line.startswith("  ") and cur is not None:
            cur.append(line)
        else:
            cur = [line]
            blocks.append(cur)
    vns = any(re.match(r"Value .* in \S+: 0 rows ", b[0]) and any(" stored " in l for l in b[1:]) for b in blocks)
    other = any((re.match(r"Value .* elsewhere:$", b[0]) and len(b) > 1) or
                any(l.startswith("  Join ") for l in b[1:]) or
                (b[0].startswith("Column ") and any(l.startswith("  note: ") for l in b[1:])) for b in blocks)
    return {"value_not_stored": vns, "other": other}


class V:
    def __init__(self, a):
        self.repo, self.runs, self.out = Path(a.repo).resolve(), Path(a.runs_root).resolve(), Path(a.out).resolve()
        self.frozen_tree, self.workers, self.no_worker = a.frozen_tree, a.workers, a.no_worker
        self.out.mkdir(parents=True, exist_ok=True)
        if self.runs in self.out.parents or self.out == self.runs:
            raise SystemExit("--out must not be under the runs root")
        self.sections, self.W = {}, None
        self.dbe = self.runs / "dbe"

    def sec(self, name):
        return self.sections.setdefault(name, Section(name))

    # -- inputs --
    def load(self):
        R, f = self.repo, FROZEN
        self.D_src = git_show(R, f, H + "dbe_run.py")
        self.D = literals(self.D_src, {"EVIDENCE_HEADER", "NOTHING", "FAILED", "EVIDENCE_VERSION", "BOOTSTRAP_SEED",
                                       "BOOTSTRAP_RESAMPLES", "WORK_ORDER_USED_AT_START", "PROJECTED_HOURS",
                                       "PROMPT_VERSION"})
        self.S = literals(git_show(R, f, H + "spr_run.py"), {"REVISE", "REQUEST_DEADLINE_SECONDS", "MAX_NEW_TOKENS"})
        self.G = literals(git_show(R, f, H + "gi_llm.py"), {"SYSTEM_DIRECT"})
        self.C = literals(git_show(R, f, H + "contracts.py"), {"MODELS", "TRAIN_RELEASE"})
        self.proto = json.loads(git_show(R, f, H + "pins/dbe_protocol.json"))
        self.pin = json.loads(git_show(R, f, H + "pins/dbe_new_eval.json"))
        self.inv = json.loads(git_show(R, f, H + "pins/inventory_v1.json"))["task_ids"]
        self.snaps = json.loads(git_show(R, f, H + "pins/snapshots_v1.json"))["databases"]
        rel = self.runs / RELEASE
        self.release_sha = sha_f(rel)
        lines = rt(rel).split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        self.rows = {f"bird23tf-{i:05d}": {k: str(json.loads(x)[k]) for k in ("db_id", "question", "evidence", "SQL")}
                     for i, x in enumerate(lines)}
        self.snap = {d: {"path": str(self.runs / "data" / "databases" / e["file"]), "sha256": e["sha256"],
                         "bytes": e["bytes"]} for d, e in self.snaps.items()}
        self.run_dir = self.dbe / "run" / RUN
        self.reqs = jl(self.run_dir / "requests.jsonl")
        self.scores = jl(self.run_dir / "scores.jsonl")
        self.manifest = json.loads(rt(self.run_dir / "manifest.json"))
        self.summary = json.loads(rt(self.run_dir / "summary.json"))
        self.analysis = json.loads(rt(self.dbe / "analysis" / ANALYSIS / "analysis.json"))
        self.amanifest = json.loads(rt(self.dbe / "analysis" / ANALYSIS / "manifest.json"))
        self.ids = self.pin["task_ids"]
        self.reqs_by_key = {(r["seed"], r["task_id"]): r for r in self.reqs}
        self.db_of = dict(zip(self.ids, self.pin["db_ids"]))

    def stat_listing(self):
        out = {}
        for p in self.runs.rglob("*"):
            with contextlib.suppress(OSError):
                if p.is_file():
                    st = p.stat()
                    out[str(p.relative_to(self.runs))] = (st.st_size, st.st_mtime_ns)
        return out

    # -- the worker --
    def worker(self):
        if self.no_worker or not self.frozen_tree:
            return
        job = self.out / "worker_job.json"
        res = self.out / "worker_out.json"
        job.write_text(json.dumps({"prompts": self.ids, "lookup": {f"{r['seed']}|{r['task_id']}": {
            "task_id": r["task_id"], "sql": r["original"]["sql"]} for r in self.reqs}}))
        env = {**os.environ, "TI_TNN_SCALING_RUNS": str(self.runs), "CUDA_VISIBLE_DEVICES": ""}
        t0 = time.time()
        p = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--worker", str(job), str(res),
                            str(self.frozen_tree)], cwd=self.out, env=env, capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit(f"worker failed: {p.stderr[-3000:]}")
        self.W = json.loads(rt(res))
        self.worker_info = {"frozen_tree": str(self.frozen_tree),
                            "head": git(Path(self.frozen_tree), "rev-parse", "HEAD").strip(),
                            "research_dirty": git(Path(self.frozen_tree), "status", "--porcelain", "--",
                                                  "research/ti_tnn").strip(),
                            "db_evidence_file": self.W["db_evidence_file"], "version": self.W["version"],
                            "seconds": round(time.time() - t0, 1)}

    # -- executions (own) --
    def executions(self):
        obs, exj = {}, {}
        for r in self.reqs:
            p = self.snap[r["db_id"]]["path"]
            for sql in (r["original"]["sql"], r["D1"]["revised"]["sql"], r["V1"]["revised"]["sql"]):
                obs[(r["db_id"], sql)] = p
        for x in self.scores:
            exj[(x["task_id"], x["sql"])] = self.snap[x["db_id"]]["path"]
        t0 = time.time()
        self.obs, self.obs_reruns = run_jobs(observe, [(k, v, k[1]) for k, v in obs.items()], self.workers)
        self.ex, self.ex_reruns = run_jobs(ex_pair, [(k, v, k[1], self.rows[k[0]]["SQL"]) for k, v in exj.items()],
                                           self.workers)
        self.exec_seconds = time.time() - t0

    # -- 1. scores --
    def scores_section(self):
        s = self.sec("1_scores")
        used = sorted({r["db_id"] for r in self.reqs})
        bad = {}
        self.own_sha = {}
        for d in used:
            h = sha_f(self.snap[d]["path"])
            self.own_sha[d] = h
            if h != self.snap[d]["sha256"] or os.path.getsize(self.snap[d]["path"]) != self.snap[d]["bytes"]:
                bad[d] = h
        s.check(not bad, f"own sha256 and size of the {len(used)} snapshot files = the snapshots pin", str(bad))
        byreq = {(r["seed"], r["task_id"]): r for r in self.reqs}
        fails, mapping = defaultdict(list), []
        self.correct = {}
        for x in self.scores:
            r = byreq[(x["seed"], x["task_id"])]
            k = f"{x['seed']}|{x['task_id']}|{x['policy']}"
            if (x["source"], x["sql"]) != (r[x["policy"]]["source"], r[x["policy"]]["sql"]):
                fails["scored SQL = the request's selected SQL"].append(k)
            mine, sc = self.ex[(x["task_id"], x["sql"])], x["score"]
            if sc["snapshot_sha256"] != self.own_sha[x["db_id"]]:
                fails["score's snapshot sha256 = own hash"].append(k)
            if sc["correct"] != mine["correct"]:
                fails["recorded EX = own EX"].append(f"{k}: {sc['correct']} vs {mine['correct']}")
            if sc["status"] != mine["status"]:
                fails["recorded status = own status"].append(f"{k}: {sc['status']} vs {mine['status']}")
            if sc["predicted"]["status"] != mine["pred"]:
                fails["recorded predicted status = own"].append(k)
            if mine["status"] != "abstain" and (sc["gold"]["status"] != mine["gold"] or
                                                sc["gold"]["row_count"] != mine["gold_count"] or
                                                sc["predicted"]["row_count"] != mine["pred_count"]):
                fails["recorded gold status and row counts = own"].append(k)
            self.correct[(x["policy"], x["seed"], x["task_id"])] = mine["correct"]
            mapping.append({"seed": x["seed"], "task_id": x["task_id"], "policy": x["policy"], "db_id": x["db_id"],
                            "snapshot_sha256": self.own_sha[x["db_id"]],
                            "sql_sha256": None if x["sql"] is None else sha_t(x["sql"]),
                            "source": x["source"], "recorded_correct": sc["correct"], "own_correct": mine["correct"],
                            "recorded_status": sc["status"], "own_status": mine["status"],
                            "own_pred_rows": mine.get("pred_count"), "own_gold_rows": mine.get("gold_count")})
        for n in ("scored SQL = the request's selected SQL", "score's snapshot sha256 = own hash", "recorded EX = own EX",
                  "recorded status = own status", "recorded predicted status = own",
                  "recorded gold status and row counts = own"):
            s.check(not fails.get(n), n + f" ({len(mapping)} records)", "; ".join(fails.get(n, [])[:8]))
        (self.out / "score_mapping.jsonl").write_text("".join(json.dumps(m, sort_keys=True) + "\n" for m in mapping))
        s.info.update(records=len(mapping), own_status=dict(Counter(m["own_status"] for m in mapping)),
                      reruns={"observe": self.obs_reruns, "ex": self.ex_reruns},
                      execution_seconds=round(self.exec_seconds, 1), sqlite_version=sqlite3.sqlite_version)

    # -- 2. lineage --
    def lineage(self):
        s = self.sec("2_lineage")
        REVISE, HEADER, NOTHING, FAILED = self.S["REVISE"], self.D["EVIDENCE_HEADER"], self.D["NOTHING"], self.D["FAILED"]
        s.check(sha_t(REVISE) == self.proto["revise_sha256"] and HEADER == self.proto["evidence_header"] and
                NOTHING == self.proto["nothing"] and FAILED == self.proto["failed"],
                "the frozen REVISE, header, NOTHING and FAILED texts = the protocol pin")
        cfg = self.manifest["config"]
        s.check(cfg["revise"] == REVISE and cfg["evidence_header"] == HEADER and cfg["system"] == self.G["SYSTEM_DIRECT"]
                and cfg["generation"]["max_new_tokens"] == 512 and not cfg["generation"]["thinking"] and
                cfg["lookup"] == "db-evidence-v1.4", "run manifest: system, instruction, header, 512 tokens, "
                "non-thinking, db-evidence-v1.4")
        tok = None
        if self.W:
            from transformers import AutoTokenizer
            m = self.C["MODELS"]["B4"]
            root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
            d = root / f"models--{m['repo'].replace('/', '--')}" / "snapshots" / m["revision"]
            tok = AutoTokenizer.from_pretrained(str(d), local_files_only=True)
            s.check(sha_f(d / "chat_template.jinja") ==
                    self.manifest["model_identity"]["files"]["chat_template.jinja"]["sha256"],
                    "the local chat template = the run manifest's model identity")

        def count(msgs):
            text = tok.apply_chat_template(msgs, tools=None, add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False)
            return len(tok(text, add_special_tokens=False)["input_ids"])
        fails, st = defaultdict(list), Counter()
        self.sel = {}
        for r in self.reqs:
            k = f"{r['seed']}|{r['task_id']}"
            o = r["original"]
            if r["D0"] != {"source": "original", "sql": o["sql"]}:
                fails["D0 = the shared initial proposal"].append(k)
            for name, c in (("original", o), ("D1", r["D1"]["revised"]), ("V1", r["V1"]["revised"])):
                sql, how = own_extract(c["call"]["content"])
                st[f"extract_{how}"] += 1
                if sql != c["sql"]:
                    fails["own B4 extraction = the recorded SQL"].append(f"{k}|{name}")
                if c["call"]["generated_tokens"] > 512:
                    fails["generated tokens <= 512"].append(f"{k}|{name}")
                st[f"{name}_finish_{c['call']['finish_reason']}"] += 1
            h = int(hashlib.sha256(f"spr-order:{r['seed']}:{r['task_id']}".encode()).hexdigest()[:8], 16)
            want = ["D1", "V1"] if h % 2 == 0 else ["V1", "D1"]
            st["D1_first"] += want[0] == "D1"
            if r["revision_order"] != want:
                fails["revision order = the seed rule"].append(k)
            raw = own_raw(o["execution"])
            d1, v1 = r["D1"]["revised"]["feedback"], r["V1"]["revised"]["feedback"]
            if d1 != f"{raw}\n\n{REVISE}":
                fails["D1 message = own raw observation + the frozen instruction"].append(k)
            ev = r["evidence"]
            body = FAILED if not ev["ok"] else (ev["package"]["rendered"] or NOTHING)
            if v1 != f"{raw}\n\n{HEADER}\n{body}\n\n{REVISE}":
                fails["V1 message = D1's + exactly the header and the recorded payload"].append(k)
            st["lookup_failed"] += not ev["ok"]
            st["payload_empty"] += bool(ev["ok"] and not ev["package"]["rendered"])
            mine = self.obs[(r["db_id"], o["sql"])]
            if not same_obs(o["execution"], mine):
                fails["own execution of the initial SQL = the recorded execution"].append(
                    f"{k}: {o['execution'].get('status')} vs {mine['status']}")
            if o["execution"].get("status") not in ("timeout", "rejected") and \
                    own_raw(o["execution"]) != own_raw(obs_as_record(mine)):
                fails["the raw observation rebuilt from own execution = the recorded one"].append(k)
            for p in ("D1", "V1"):
                rv = r[p]["revised"]
                if not same_obs(rv["execution"], self.obs[(r["db_id"], rv["sql"])]):
                    fails["own execution of each revision = the recorded execution"].append(f"{k}|{p}")
            allowed = r["revision_seconds_allowed"]
            spent = o["call"]["seconds"] + o["execution"]["seconds"] + ev["seconds"]
            gap = self.S["REQUEST_DEADLINE_SECONDS"] - allowed - spent
            st["max_deadline_gap_x1000"] = max(st["max_deadline_gap_x1000"], int(gap * 1000))
            if not (0 <= gap < 2.0):
                fails["remaining time = 180 s - (initial call + execution + lookup), within 2 s"].append(
                    f"{k}: {gap:.3f}")
            for p in ("D1", "V1"):
                if r[p]["revised"]["call"]["seconds"] > allowed:
                    fails["each revision within the shared remaining time"].append(f"{k}|{p}")
            # selection (own execution, own read-only check)
            for p in ("D1", "V1"):
                rv = r[p]["revised"]

                def valid(sql, ob):
                    return bool(sql) and own_readonly(sql) is None and ob["status"] == "ok"
                if valid(rv["sql"], self.obs[(r["db_id"], rv["sql"])]):
                    want_sel = ("revised", rv["sql"])
                elif valid(o["sql"], mine):
                    want_sel = ("original", o["sql"])
                else:
                    want_sel = ("abstain", None)
                self.sel[(r["seed"], r["task_id"], p)] = want_sel
                st[f"{p}_{want_sel[0]}"] += 1
                if (r[p]["source"], r[p]["sql"]) != want_sel:
                    fails["every selection, fallback and abstention = own rule on own execution"].append(
                        f"{k}|{p}: {r[p]['source']} vs {want_sel[0]}")
            # the fresh lookup (worker)
            if self.W:
                w = self.W["lookup"][k]
                rec = dict(ev["package"] or {})
                rec.pop("latency_seconds", None)
                if not (w["ok"] == ev["ok"] and json.loads(json.dumps(w.get("package"), default=str)) ==
                        json.loads(json.dumps(rec, default=str))):
                    fails["recorded payload = a fresh db-evidence-v1.4 lookup (worker; latency aside)"].append(k)
                pr = self.W["prompts"][r["task_id"]]
                if pr["retrieved"] != r["retrieved"]:
                    fails["retrieved examples = the producer's retrieval (worker)"].append(k)
                user = self.own_user(r["task_id"], pr["schema_text"], r["retrieved"])
                if user != pr["user"]:
                    fails["own user message = the producer's (worker)"].append(k)
                base = [{"role": "system", "content": self.G["SYSTEM_DIRECT"]}, {"role": "user", "content": user}]
                n0 = count(base)
                if n0 != o["call"]["prompt_tokens"]:
                    fails["initial prompt tokens = own rendering"].append(f"{k}: {n0} vs {o['call']['prompt_tokens']}")
                for p in ("D1", "V1"):
                    msgs = base + [{"role": "assistant", "content": o["call"]["content"]},
                                   {"role": "user", "content": r[p]["revised"]["feedback"]}]
                    n = count(msgs)
                    if n != r[p]["revised"]["call"]["prompt_tokens"]:
                        fails[f"{p} prompt tokens = own rendering of the shared prefix + its message"].append(
                            f"{k}: {n} vs {r[p]['revised']['call']['prompt_tokens']}")
        names = ["D0 = the shared initial proposal", "own B4 extraction = the recorded SQL", "generated tokens <= 512",
                 "revision order = the seed rule", "D1 message = own raw observation + the frozen instruction",
                 "V1 message = D1's + exactly the header and the recorded payload",
                 "own execution of the initial SQL = the recorded execution",
                 "the raw observation rebuilt from own execution = the recorded one",
                 "own execution of each revision = the recorded execution",
                 "remaining time = 180 s - (initial call + execution + lookup), within 2 s",
                 "each revision within the shared remaining time",
                 "every selection, fallback and abstention = own rule on own execution"]
        if self.W:
            names += ["recorded payload = a fresh db-evidence-v1.4 lookup (worker; latency aside)",
                      "retrieved examples = the producer's retrieval (worker)",
                      "own user message = the producer's (worker)", "initial prompt tokens = own rendering",
                      "D1 prompt tokens = own rendering of the shared prefix + its message",
                      "V1 prompt tokens = own rendering of the shared prefix + its message"]
        for n in names:
            s.check(not fails.get(n), n, "; ".join(fails.get(n, [])[:8]))
        s.info["stats"] = dict(st)
        self.code_serve(s)
        # the lookup inventory against analysis.json
        ev = [r["evidence"] for r in self.reqs]
        pk = [e["package"] for e in ev if e["ok"]]
        inv = {"failed": sum(not e["ok"] for e in ev), "timeouts": sum(len(p["timeouts"]) for p in pk),
               "with_scan_limits": sum(1 for p in pk if p["scan_limits"]), "truncated": sum(p["truncated"] for p in pk),
               "with_unresolved": sum(1 for p in pk if p["proposal"].get("unresolved")),
               "payload_chars_max": max((len(p["rendered"]) for p in pk), default=0),
               "wall_safety_stops": sum(p["wall_safety_stop"] for p in pk)}
        rec = {k: self.analysis["lookup"][k] for k in inv}
        s.check(inv == rec, "lookup failure and limit inventory (own) = analysis.json", json.dumps(inv))
        s.check(all(p["version"] == "db-evidence-v1.4" and p["payload_chars"] == len(p["rendered"]) and
                    p["limits"] == self.default_limits() for p in pk),
                "every package: db-evidence-v1.4, payload_chars = len(rendered), the protocol's default limits")
        s.check(abs(sum(e["seconds"] for e in ev) - self.analysis["lookup"]["seconds_total"]) < 1e-9 and
                abs(max(e["seconds"] for e in ev) - self.analysis["lookup"]["seconds_max"]) < 1e-12,
                "lookup seconds total and max = analysis.json")
        s.info["lookup_inventory"] = inv | {"scan_limit_kinds": dict(Counter(
            " ".join(x["what"].split()[:2]) for p in pk for x in p["scan_limits"]))}

    def default_limits(self):
        lim = class_defaults(git_show(self.repo, FROZEN, H + "db_evidence.py"), "Limits")
        return lim

    def own_user(self, tid, schema_text, retrieved):
        q = self.rows[tid]
        parts = [f"Question: {q['question']}", f"Evidence: {q['evidence'] or '(none)'}", "", "Database schema:",
                 schema_text]
        if retrieved:
            parts += ["", "Examples from other databases (question, evidence and a correct query):"]
            for i, t in enumerate(retrieved, 1):
                e = self.rows[t]
                parts += [f"Example {i}. Question: {e['question']}", f"Evidence: {e['evidence'] or '(none)'}",
                          f"SQL: {e['SQL']}"]
        return "\n".join(parts)

    def code_serve(self, s):
        src = self.D_src.decode()
        fns = functions(self.D_src)
        serve = ast.get_source_segment(src, fns["serve"])
        flat = re.sub(r"\s+", " ", serve)
        s.check('S._call(rt, base, cfg, S.seed_of("initial", seed, task.task_id), t_start + S.REQUEST_DEADLINE_SECONDS)'
                in flat and flat.count("S._call(") == 2, "code: one initial call per request with seed_of('initial')")
        s.check(re.search(r"for policy in order:.*S\._call\(rt, msgs, cfg, S\.seed_of\(\"revise\", seed, task\.task_id\), "
                          r"time\.monotonic\(\) \+ remaining\)", flat),
                "code: both revisions use seed_of('revise') and time.monotonic() + remaining")
        s.check('msgs = base + [{"role": "assistant", "content": turn0.content}, {"role": "user", "content": '
                'messages[policy]}]' in flat, "code: both revisions = the shared base + the initial reply + their message")
        i_lookup, i_rem = flat.find("evidence = lookup(task, schema, snap, sql0)"), flat.find("remaining = max(0.0")
        s.check(0 < i_lookup < i_rem and flat.count("remaining = ") == 1,
                "code: the lookup runs before the remaining time is taken, once")
        s.check('order = ("D1", "V1") if S.seed_of("order", seed, task.task_id) % 2 == 0 else ("V1", "D1")' in flat,
                "code: the order by seed_of('order') parity")
        s.check('out[policy] = {"revised": revised, **S.select(original, revised)}' in flat,
                "code: selection is spr_run.select")
        lk = ast.get_source_segment(src, fns["lookup"])
        s.check("retry" not in re.sub(r"#.*|\"\"\".*?\"\"\"", "", lk, flags=re.S).lower() and
                lk.count("db_evidence.retrieve(") == 1, "code: the lookup is called once, failures recorded, no retry")
        spr_now = git_show(self.repo, FROZEN, H + "spr_run.py")
        s.check(spr_now == git_show(self.repo, "2f76bb7", H + "spr_run.py"),
                "spr_run.py at the run commit = #71's frozen spr_run.py (_call, _execute, select, seed_of, REVISE)")

    # -- 3. paired results --
    def paired(self):
        s = self.sec("3_paired")
        A = self.analysis
        ids, seeds, db_of = self.ids, SEEDS, self.db_of
        c = lambda p, sd, t: self.correct[(p, sd, t)]  # noqa: E731
        counts = {f"{p}/{sd}": sum(c(p, sd, t) for t in ids) for p in POLICIES for sd in seeds}
        pooled = {p: sum(counts[f"{p}/{sd}"] for sd in seeds) for p in POLICIES}
        s.check(counts == A["counts"] and pooled == A["pooled"], "counts per policy and seed, and pooled = analysis.json",
                json.dumps(counts))
        own = {}
        for a, b in COMPARISONS:
            per_seed = {str(sd): sum(c(a, sd, t) - c(b, sd, t) for t in ids) for sd in seeds}
            tot = sum(per_seed.values())
            per_db = defaultdict(int)
            for t in ids:
                per_db[db_of[t]] += sum(c(a, sd, t) - c(b, sd, t) for sd in seeds)
            loo = {d: tot - v for d, v in sorted(per_db.items())}
            sign = 1 if tot > 0 else -1 if tot < 0 else 0
            cond = {"pooled_at_least_lead": abs(tot) >= LEAD,
                    "same_sign_every_seed": sign != 0 and all(v * sign > 0 for v in per_seed.values()),
                    "same_sign_without_any_one_database": sign != 0 and all(v * sign > 0 for v in loo.values())}
            nz = [v for v in per_db.values() if v != 0]
            tot_a = ext = 0
            for signs in product((1, -1), repeat=len(nz)):
                tot_a += 1
                ext += abs(sum(x * y for x, y in zip(signs, nz))) >= abs(tot)
            flip = {"observed": tot, "blocks": len(per_db), "nonzero_blocks": len(nz), "assignments": tot_a,
                    "p_two_sided": ext / tot_a if tot_a else 1.0,
                    "smallest_attainable_p": (2 / tot_a) if nz else 1.0}
            hist = Counter(sum(c(a, sd, t) - c(b, sd, t) for sd in seeds) for t in ids)
            d = {"per_seed": per_seed, "pooled": tot, "per_database": dict(sorted(per_db.items())),
                 "leave_one_database_out": loo, "sign": sign, "conditions": cond, "all_conditions": all(cond.values()),
                 "threshold": LEAD,
                 "recoveries": sum(c(a, sd, t) > c(b, sd, t) for sd in seeds for t in ids),
                 "regressions": sum(c(a, sd, t) < c(b, sd, t) for sd in seeds for t in ids),
                 "per_question_histogram": {str(k): v for k, v in sorted(hist.items())},
                 "exploratory_sign_flip": flip}
            own[f"{a}-{b}"] = d
            rec = A["comparisons"][f"{a}-{b}"]
            diff = [f for f in d if rec.get(f) != d[f]]
            s.check(not diff, f"{a} - {b}: per seed, pooled, per database, leave-one-out, sign, +8 conditions, "
                    "recoveries, regressions, histogram and exact sign-flip = analysis.json",
                    f"differs: {diff}" if diff else json.dumps({k: d[k] for k in ("per_seed", "pooled",
                                                                                 "recoveries", "regressions")}))
            s.check(d["recoveries"] - d["regressions"] == tot, f"{a} - {b}: recoveries - regressions = pooled")
        lead = lambda d: d["all_conditions"] and d["sign"] > 0  # noqa: E731
        harm = lambda d: d["all_conditions"] and d["sign"] < 0  # noqa: E731
        if lead(own["V1-D1"]):
            dec = "V1 meets the continuation rule against D1: database evidence has earned further investigation"
        elif harm(own["V1-D1"]) or harm(own["V1-D0"]):
            dec = "V1 harms correctness by the rule: inspect its regressions"
        else:
            dec = "V1 did not meet the prespecified continuation threshold on this population: this version stops"
        s.check(dec == A["decision"], "own decision (protocol's branches, +8) = analysis.json", dec)
        # bootstrap, re-implemented from the protocol text
        diff = {f"{a}-{b}": [sum(c(a, sd, t) - c(b, sd, t) for sd in seeds) for t in ids] for a, b in COMPARISONS}
        rng = random.Random(BOOT_SEED)
        draws = {k: [] for k in diff}
        for _ in range(BOOT_R):
            pick = [rng.randrange(len(ids)) for _ in ids]
            for k, dv in diff.items():
                draws[k].append(sum(dv[i] for i in pick))
        lo, hi = math.floor(0.025 * BOOT_R), math.ceil(0.975 * BOOT_R) - 1
        boot = {k: {"observed": sum(dv), "low": sorted(draws[k])[lo], "high": sorted(draws[k])[hi]} for k, dv in diff.items()}
        s.check(all({f: A["uncertainty"][k][f] for f in ("observed", "low", "high")} == boot[k] and
                    A["uncertainty"][k]["seed"] == BOOT_SEED and A["uncertainty"][k]["resamples"] == BOOT_R for k in boot),
                "bootstrap re-implemented from the protocol (Random(20261107).randrange, 10,000, floor/ceil indices) = "
                "analysis.json", json.dumps(boot))
        s.check(self.D["BOOTSTRAP_SEED"] == BOOT_SEED and self.D["BOOTSTRAP_RESAMPLES"] == BOOT_R,
                "the frozen harness's bootstrap constants = the protocol's")
        # transitions
        key = [(sd, t) for sd in seeds for t in ids]
        cc = {p: {(sd, t): c(p, sd, t) for sd, t in key} for p in POLICIES}
        tr = {"V1_fixes_D1_misses": sum(1 for k in key if cc["V1"][k] and not cc["D1"][k]),
              "D1_right_V1_breaks": sum(1 for k in key if cc["D1"][k] and not cc["V1"][k]),
              "initially_correct": sum(cc["D0"].values()),
              "initially_correct_broken_by": {p: sum(1 for k in key if cc["D0"][k] and not cc[p][k]) for p in ("D1", "V1")},
              "initially_wrong_fixed_by": {p: sum(1 for k in key if not cc["D0"][k] and cc[p][k]) for p in ("D1", "V1")}}
        s.check(tr == A["transitions"], "transitions = analysis.json", json.dumps(tr))
        # signal groups, own reading of the rendered payload
        groups = {}
        for r in self.reqs:
            ev = r["evidence"]
            if not ev["ok"]:
                g = "lookup failed"
            else:
                sg = own_signals(ev["package"]["rendered"])
                g = "value not stored" if sg["value_not_stored"] else "other signal" if sg["other"] else "no signal"
            d = groups.setdefault(g, {"requests": 0, **{p: 0 for p in POLICIES}, "V1_fixes_D1_misses": 0,
                                      "D1_right_V1_breaks": 0, "D0_right_V1_breaks": 0})
            k = (r["seed"], r["task_id"])
            d["requests"] += 1
            for p in POLICIES:
                d[p] += cc[p][k]
            d["V1_fixes_D1_misses"] += cc["V1"][k] and not cc["D1"][k]
            d["D1_right_V1_breaks"] += cc["D1"][k] and not cc["V1"][k]
            d["D0_right_V1_breaks"] += cc["D0"][k] and not cc["V1"][k]
        rec_groups = {k: v for k, v in A["signal_groups"].items() if k != "note"}
        s.check(groups == rec_groups, "signal groups (own reading of the rendered payload) = analysis.json",
                json.dumps(groups))
        perq = {t: {p: [c(p, sd, t) for sd in seeds] for p in POLICIES} | {"db": db_of[t]} for t in ids}
        s.check(perq == A["per_question"], "per-question table = analysis.json")
        bydb = {d: {p: sum(c(p, sd, t) for sd in seeds for t in ids if db_of[t] == d) for p in POLICIES}
                for d in sorted(set(db_of.values()))}
        s.info.update(own={k: {f: v[f] for f in ("per_seed", "pooled", "leave_one_database_out", "recoveries",
                                                 "regressions", "conditions", "exploratory_sign_flip")}
                           for k, v in own.items()}, bootstrap=boot, decision=dec, by_database=bydb, transitions=tr)
        L = A["lineage"]
        for p in ("D1", "V1"):
            mine = {"final_source": dict(Counter(r[p]["source"] for r in self.reqs)),
                    "revision_unchanged_sql": sum(r[p]["revised"]["sql"] == r["original"]["sql"] for r in self.reqs),
                    "revision_truncated": sum(r[p]["revised"]["call"]["finish_reason"] != "eos" for r in self.reqs)}
            s.check(mine == L[p], f"lineage tallies {p} = analysis.json", json.dumps(mine))
        ini = {"no_sql": sum(not r["original"]["sql"] for r in self.reqs),
               "execution": dict(Counter(self.obs[(r["db_id"], r["original"]["sql"])]["status"] for r in self.reqs))}
        s.check(ini == A["initial"], "initial tallies (own execution) = analysis.json", json.dumps(ini))
        fs = {p: dict(Counter("abstain" if self.sel_or_d0(p, r)[1] is None else
                              self.ex[(r["task_id"], self.sel_or_d0(p, r)[1])]["pred"] for r in self.reqs)) for p in POLICIES}
        s.check(fs == A["final_status"], "final statuses (own) = analysis.json", json.dumps(fs))

    def sel_or_d0(self, p, r):
        return ("original", r["original"]["sql"]) if p == "D0" else self.sel[(r["seed"], r["task_id"], p)]

    # -- 4. costs, ledger, co-tenancy --
    def costs(self):
        s = self.sec("4_costs")
        A, R = self.analysis["cost"], self.reqs
        sc = {(x["policy"], x["seed"], x["task_id"]): x for x in self.scores}
        for p in POLICIES:
            calls = [r["original"]["call"] for r in R] + ([r[p]["revised"]["call"] for r in R] if p != "D0" else [])
            gen = sum(x["seconds"] for x in calls)
            tool = sum(r["original"]["execution"]["seconds"] + r[p]["revised"]["execution"]["seconds"] for r in R) \
                if p != "D0" else 0.0
            look = sum(r["evidence"]["seconds"] for r in R) if p == "V1" else 0.0
            exe = sum(sc[(p, r["seed"], r["task_id"])]["score"]["predicted"]["elapsed_seconds"] for r in R)
            tot = gen + tool + look + exe
            ok = self.analysis["pooled"][p]
            mine = {"generation_seconds": gen, "lookup_seconds": look, "tool_seconds": tool,
                    "submitted_execution_seconds": exe, "total_seconds": tot, "per_request_seconds": tot / len(R),
                    "per_ex_correct_seconds": tot / ok if ok else None,
                    "prompt_tokens": sum(x["prompt_tokens"] for x in calls),
                    "generated_tokens": sum(x["generated_tokens"] for x in calls),
                    "peak_memory_bytes": max(x["peak_memory_bytes"] for x in calls),
                    "calls_per_request": 1 if p == "D0" else 2}
            close = all((mine[k] is None and A[p][k] is None) or abs(mine[k] - A[p][k]) <= 1e-6 * max(1, abs(A[p][k]))
                        for k in mine)
            s.check(close and set(mine) == set(A[p]), f"{p}: generation, lookup, tool, submitted-execution, tokens, "
                    "peak memory = analysis.json (lookup charged to V1 only)", json.dumps(mine))
            s.info[p] = mine
        work = sum(r["original"]["call"]["seconds"] + r["original"]["execution"]["seconds"] + r["evidence"]["seconds"] +
                   sum(r[p]["revised"]["call"]["seconds"] + r[p]["revised"]["execution"]["seconds"] for p in ("D1", "V1"))
                   for r in R)
        L = jl(self.runs / "gpu_ledger.jsonl")
        res = {}
        for e in L:
            r = res.setdefault(e["reservation"], {"id": e["reservation"], "start": None, "elapsed": 0.0, "end": None, "hb": []})
            if e["event"] == "start":
                r.update(start=e["t"], command=e.get("command"), projected=e.get("projected_hours"),
                         used_before=e.get("used_before_hours"), limit=e.get("limit_hours"), extra=e.get("extra"),
                         foreign=e.get("foreign_gpu_processes"))
            elif e["event"] == "heartbeat":
                r["elapsed"] = max(r["elapsed"], float(e["elapsed"]))
                r["hb"].append(e["t"])
            elif e["event"] == "end":
                r.update(end=e["t"], status=e.get("status"), elapsed=max(r["elapsed"], float(e["elapsed"])),
                         end_foreign=e.get("foreign_gpu_processes"))
        order = sorted((r for r in res.values() if r["start"] is not None), key=lambda r: r["start"])
        cum = 0.0
        for r in order:
            r["hours"] = (r["elapsed"] if r["end"] is not None else r["elapsed"] + 30.0) / 3600.0
            r["cum_before"] = cum
            cum += r["hours"]
        dbe = [r for r in order if (r.get("command") or "").startswith("dbe ")]
        s.check(len(dbe) == 1 and dbe[0]["command"] == "dbe run", "exactly one dbe reservation: 'dbe run'",
                json.dumps([r.get("command") for r in dbe]))
        r = dbe[0]
        s.check(abs(r["cum_before"] - USED_AT_START) < 1e-5 and abs(r["used_before"] - r["cum_before"]) < 1e-6,
                "used at start = 23.489974 h (own cumulative = recorded)", f"{r['cum_before']:.6f}")
        worst = r["cum_before"] + 1.5 * r["projected"]
        s.check(r["projected"] == 0.65 and worst <= LIMIT and abs(r["limit"] - worst) < 1e-6,
                f"projection 0.65 h; used + 1.5 x projection = {worst:.6f} <= {LIMIT}; watchdog limit = that")
        s.check(r["end"] is not None and r["status"] == "ok" and r["cum_before"] + r["hours"] <= LIMIT and
                r["hours"] <= 1.0, f"the run ended ok: {r['hours']:.4f} h, cumulative {r['cum_before'] + r['hours']:.6f} "
                f"<= {LIMIT}")
        gaps = [b - a for a, b in zip([r["start"]] + r["hb"], r["hb"] + [r["end"]])]
        s.check(max(gaps) < 45 and abs((r["end"] - r["start"]) - r["elapsed"]) < 5, "heartbeats every ~30 s",
                f"max gap {max(gaps):.1f} s")
        overlap = [o["id"] for o in order if o is not r and o["start"] < r["end"] and
                   (o["end"] or o["start"] + o["elapsed"]) > r["start"]]
        s.check(not overlap, "no overlapping reservation", str(overlap))
        s.info["foreign_gpu_processes"] = {"start": r.get("foreign"), "end": r.get("end_foreign")}
        s.check(abs(r["hours"] - self.summary["reservation"]["hours"]) < 1e-5 and
                self.summary["reservation"]["id"] == r["id"] and r["extra"]["run_dir"] == f"dbe/run/{RUN}",
                "the run summary's reservation = the ledger's")
        s.check(self.summary["load_seconds"] + work <= r["elapsed"],
                "load + every recorded call, execution and lookup fits inside the reservation",
                f"{self.summary['load_seconds'] + work:.1f} s of {r['elapsed']:.1f} s")
        g = self.analysis["gpu_reservations"]
        s.check(len(g) == 1 and g[0]["id"] == r["id"] and abs(g[0]["hours"] - r["hours"]) < 1e-5,
                "analysis.json's reservation = own")
        b = self.manifest["config"]["budget"]
        s.check(abs(b["used_hours"] - r["cum_before"]) < 1e-6 and b["work_order_limit_hours"] == LIMIT and b["fits"],
                "the run manifest's budget check = own", json.dumps(b))
        s.info["ledger_total_hours_now"] = cum
        s.info["reservation"] = {"id": r["id"], "hours": r["hours"], "start": iso(r["start"]), "end": iso(r["end"])}
        log = rt(self.dbe / "logs" / "cotenancy_20261107T182053Z.log").splitlines()
        samples = []
        for ln in log:
            p = ln.split()
            if len(p) >= 4 and p[0][:4].isdigit():
                t = datetime.strptime(p[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
                samples.append((t, float(p[1]), p[4] if len(p) > 4 else ""))
        inw = [x for x in samples if r["start"] - 31 <= x[0] <= r["end"] + 31]
        ts = [x[0] for x in inw]
        s.check(ts and ts[0] <= r["start"] + 31 and ts[-1] >= r["end"] - 31 and max(b2 - a2 for a2, b2 in zip(ts, ts[1:])) <= 45,
                "the co-tenancy log covers the run every ~30 s", f"{len(inw)} samples")
        s.info["cotenancy"] = {"samples": len(inw), "load1_max": max(x[1] for x in inw),
                               "load1_mean": round(sum(x[1] for x in inw) / len(inw), 2)}
        s.check(max(x[1] for x in inw) < 8, "co-tenancy: 1-minute load stayed low during the run",
                f"max {max(x[1] for x in inw)}")

    # -- 5. no reference access --
    def reference(self):
        s = self.sec("5_reference")
        norm = lambda t: re.sub(r"\s+", " ", t or "").strip().lower().rstrip(";").strip()  # noqa: E731
        idpat = re.compile(r"bird23tf-\d{5}")
        fails, explained, checked = defaultdict(list), 0, 0
        for r in self.reqs:
            k = f"{r['seed']}|{r['task_id']}"
            gold = norm(self.rows[r["task_id"]]["SQL"])
            texts = {"D1 message": r["D1"]["revised"]["feedback"], "V1 message": r["V1"]["revised"]["feedback"],
                     "payload": (r["evidence"]["package"] or {}).get("rendered", "")}
            if self.W:
                pr = self.W["prompts"][r["task_id"]]
                texts["prompt"] = pr["system"] + "\n" + pr["user"]
            for name, t in texts.items():
                checked += 1
                if gold and gold in norm(t):
                    fails["no gold SQL in any prompt, feedback message or payload"].append(f"{k}|{name}")
                if set(idpat.findall(t)):
                    fails["no task id in any prompt, feedback message or payload"].append(f"{k}|{name}")
            g = self.ex.get((r["task_id"], r["original"]["sql"]))
            gp = None
            for p in POLICIES:
                e = self.ex.get((r["task_id"], r[p]["sql"]))
                if e and e.get("gold_preview"):
                    gp = e["gold_preview"]
                    break
            if gp:
                block = "\n".join("  " + " | ".join(("NULL" if v is None else str(v))[:60] +
                                                    ("..." if len("NULL" if v is None else str(v)) > 60 else "")
                                                    for v in row) for row in gp)
                own_obs = own_raw(obs_as_record(self.obs[(r["db_id"], r["original"]["sql"])]))
                for name in ("D1 message", "V1 message"):
                    # whole lines only: a one-value gold result such as "  2" must not match a payload rank "  2. ..."
                    if block and ("\n" + block + "\n") in ("\n" + texts[name] + "\n"):
                        if (g and g.get("pred_set") == g.get("gold_set")) or block in own_obs:
                            explained += 1
                        else:
                            fails["a gold result block appears in a message only where the model's own result shows it"].append(
                                f"{k}|{name}")
        for n in ("no gold SQL in any prompt, feedback message or payload",
                  "no task id in any prompt, feedback message or payload",
                  "a gold result block appears in a message only where the model's own result shows it"):
            s.check(not fails.get(n), n, "; ".join(fails.get(n, [])[:8]))
        s.info.update(texts_checked=checked, gold_block_explained_by_own_result=explained, prompts_checked=bool(self.W))
        # code: the lookup's and the prompt's inputs; gold only after the reservation
        fns = functions(self.D_src)
        src = self.D_src.decode()
        lk = ast.get_source_segment(src, fns["lookup"])
        s.check("db_evidence.retrieve(task.question, task.evidence, sql, schema, snap.path)" in lk,
                "code: the lookup sees only question, evidence, SQL, schema and snapshot")
        run = fns["run_cmd"]
        w = next(n for n in ast.walk(run) if isinstance(n, ast.With))
        gl = [n.lineno for n in ast.walk(run) if isinstance(n, ast.Attribute) and n.attr == "gold"]
        s.check(gl and min(gl) > w.end_lineno, "code: gold is read only after the reservation (scoring)", str(gl))
        sv = ast.get_source_segment(src, fns["serve"]) + ast.get_source_segment(src, fns["lookup"]) + \
            ast.get_source_segment(src, fns["v1_message"])
        s.check("gold" not in sv.lower() and "scorer" not in sv, "code: serve, lookup and v1_message never name gold")
        Ctask = ast.get_source_segment(git_show(self.repo, FROZEN, H + "contracts.py").decode(),
                                       functions(git_show(self.repo, FROZEN, H + "contracts.py"))["Task"])
        s.check("sql" not in re.findall(r"^\s+(\w+):", Ctask, re.M), "contracts.Task has no SQL field",
                str(re.findall(r"^\s+(\w+):", Ctask, re.M)))
        # the frozen commit, clean, throughout
        for name, m in (("run", self.manifest), ("analysis", self.amanifest)):
            s.check(m["git"]["commit"].startswith(FROZEN) and not m["git"]["dirty_paths"] and
                    not m["git"]["research_tree_dirty"], f"{name} manifest: commit {FROZEN}, clean tree",
                    json.dumps(m["git"]))
        if self.frozen_tree:
            ft = Path(self.frozen_tree)
            s.check(git(ft, "rev-parse", "HEAD").strip().startswith(FROZEN) and
                    not git(ft, "status", "--porcelain").strip(), "the frozen worktree is at 8e1ee1b and clean (now)")
        pin = self.proto
        s.check(self.manifest["config"]["protocol"]["sha256"] == pin["sha256"] and
                self.manifest["config"]["protocol"]["frozen_at"].startswith(FROZEN_AT),
                "the run manifest names the frozen protocol (sha256, frozen_at 644c090)")
        files = {"docs/PROTOCOL_database_evidence_revision.md": pin["sha256"], "pins/dbe_new_eval.json":
                 pin["new_eval_pin_sha256"], "pins/dbe_prior_use_scan.json": pin["prior_use_scan_sha256"],
                 "db_evidence.py": pin["lookup_sha256"], "dbe_run.py": pin["harness_sha256"]}
        s.check(all(sha_b(git_show(self.repo, FROZEN, H + f)) == h for f, h in files.items()),
                "the five pinned files at 8e1ee1b match dbe_protocol.json")
        s.check(self.manifest["sources"].get("dbe_run.py") == pin["harness_sha256"] and
                self.manifest["sources"].get("db_evidence.py") == pin["lookup_sha256"],
                "the run manifest's source hashes = the pinned harness and lookup")
        changed = git(self.repo, "diff", "--name-only", FROZEN, "HEAD", "--", H).split()
        s.info["changed_since_frozen"] = changed[:20]

    # -- 6. integrity and the bundle --
    def integrity(self):
        s = self.sec("6_integrity")
        s.check(self.pin["task_ids"] == self.inv[256:320] and self.pin["positions"] == [256, 320],
                "population = inventory positions 256-319")
        rk = Counter((r["seed"], r["task_id"]) for r in self.reqs)
        s.check(sorted(rk) == sorted((sd, t) for sd in SEEDS for t in self.ids) and max(rk.values()) == 1,
                "one request per question and seed: 64 x 2, none missing or extra", f"{len(self.reqs)}")
        sk = Counter((x["seed"], x["task_id"], x["policy"]) for x in self.scores)
        s.check(sorted(sk) == sorted((sd, t, p) for sd in SEEDS for t in self.ids for p in POLICIES) and
                max(sk.values()) == 1, "one score per policy, seed and question: 384", f"{len(self.scores)}")
        s.check(all(self.rows[r["task_id"]]["db_id"] == r["db_id"] == self.db_of[r["task_id"]] for r in self.reqs),
                "every request on its pinned database")
        s.check(self.manifest["config"]["requested"] == self.ids and self.manifest["config"]["seeds"] == list(SEEDS),
                "run manifest: requested = the pin, seeds [200, 201]")
        s.check(self.analysis["requests_per_policy"] == 128 and self.analysis["questions"] == 64,
                "every policy's denominator is 128 (64 questions x 2 seeds)")
        runs = sorted(p.name for p in (self.dbe / "run").iterdir())
        s.check(runs == [RUN] and sorted(p.name for p in (self.dbe / "analysis").iterdir()) == [ANALYSIS],
                "one run directory and one analysis directory (no rerun)", str(runs))
        log = rt(self.dbe / "logs" / "run_20261107T182053Z.log").splitlines()
        lines = [json.loads(x) for x in log if x.startswith("{")]
        s.check(len(lines) == 128 and [(x["seed"], x["task"]) for x in lines] == [(r["seed"], r["task_id"]) for r in self.reqs],
                "the run log shows 128 requests once each, in the records' order")
        s.check(self.analysis["run"] == f"dbe/run/{RUN}", "analysis.json names the run")
        # bundle
        b = self.repo / H / "evidence" / "dbe_v1"
        sums = {ln.split(None, 1)[1].strip(): ln.split(None, 1)[0] for ln in rt(b / "SHA256SUMS").splitlines()}
        s.check(all(sha_f(b / p) == h for p, h in sums.items()), f"SHA256SUMS verifies {len(sums)} files")
        files = sorted(str(p.relative_to(b)) for p in b.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
        orig = sorted(str(p.relative_to(self.dbe)) for p in self.dbe.rglob("*") if p.is_file())
        s.check(files == sorted(sums) and sorted(f for f in files if f != "gpu_ledger.jsonl") == orig and
                all(sha_f(b / f) == sha_f(self.dbe / f) for f in orig),
                "the bundle = every file of <runs>/dbe/ (byte-identical) plus the ledger, all under SHA256SUMS")
        s.check(sha_f(b / "gpu_ledger.jsonl") == sha_f(self.runs / "gpu_ledger.jsonl"),
                "the bundled ledger = the live ledger (now)")

    # -- 7. the results report --
    def report(self):
        s = self.sec("7_report")
        path = H + "docs/REPORT_database_evidence_revision.md"
        text = git_show(self.repo, "HEAD", path).decode()
        s.info["report"] = {"commit": self.head, "sha256": sha_t(text)}
        P = self.sections["3_paired"].info
        own, boot = P["own"], P["bootstrap"]
        cc = {(p, sd, t): self.correct[(p, sd, t)] for p in POLICIES for sd in SEEDS for t in self.ids}
        cnt = {p: [sum(cc[(p, sd, t)] for t in self.ids) for sd in SEEDS] for p in POLICIES}

        flat = re.sub(r"\s+", " ", text)

        def contains(fragment):   # line wrapping in the Markdown is ignored
            return re.sub(r"\s+", " ", fragment) in flat

        def has(fragment, what):
            return s.check(contains(fragment), f"report: {what}", fragment)
        for p in POLICIES:
            has(f"| {p} | {cnt[p][0]} | {cnt[p][1]} | {sum(cnt[p])} |", f"{p} counts row")
        fmt = lambda v: f"+{v}" if v > 0 else (f"+0" if v == 0 else str(v))  # noqa: E731
        for k in ("V1-D1", "D1-D0", "V1-D0"):
            d, b = own[k], boot[k]
            ps = ", ".join(fmt(d["per_seed"][str(sd)]) for sd in SEEDS)
            iv = f"[{fmt(b['low']) if b['low'] > 0 else b['low']}, {fmt(b['high'])}]"
            loo = fmt(min(d["leave_one_database_out"].values())) if min(d["leave_one_database_out"].values()) > 0 \
                else str(min(d["leave_one_database_out"].values()))
            row = (f"| {k} | {fmt(d['pooled'])} | {ps} | {d['recoveries']} | {d['regressions']} | {iv} | {loo} | "
                   f"{all(d['conditions'].values())} | {d['exploratory_sign_flip']['p_two_sided']:.3f} |")
            has(row, f"{k} comparison row (pooled, seeds, recoveries, regressions, interval, LOO min, rule, sign-flip)")
        b = boot["V1-D1"]
        has(f"[{b['low']}, +{b['high']}] correct answers out of 128, about {b['low'] / 128 * 100:.1f} to "
            f"+{b['high'] / 128 * 100:.1f}", "V1-D1 interval and its percentage points")
        for k, w in (("D1-D0", "about +{:.1f} to +{:.1f} percentage points"), ("V1-D0", "about +{:.1f} to +{:.1f} points")):
            has(w.format(boot[k]["low"] / 128 * 100, boot[k]["high"] / 128 * 100), f"{k} interval in percentage points")
        tr = P["transitions"]
        for frag, what in ((f"| V1 right where D1 is wrong | {tr['V1_fixes_D1_misses']} |", "transition V1 fixes"),
                           (f"| D1 right where V1 is wrong | {tr['D1_right_V1_breaks']} |", "transition D1 right V1 wrong"),
                           (f"| initially correct (D0) | {tr['initially_correct']} |", "initially correct"),
                           (f"| initially correct, broken by D1 / V1 | {tr['initially_correct_broken_by']['D1']} / "
                            f"{tr['initially_correct_broken_by']['V1']} |", "broken by"),
                           (f"| initially wrong, fixed by D1 / V1 | {tr['initially_wrong_fixed_by']['D1']} / "
                            f"{tr['initially_wrong_fixed_by']['V1']} |", "fixed by")):
            has(frag, what)
        for db, v in P["by_database"].items():
            nq = sum(1 for t in self.ids if self.db_of[t] == db)
            has(f"| {db} | {nq} | {v['D0']} | {v['D1']} | {v['V1']} |", f"database row {db}")
        cost = {p: self.sections["4_costs"].info[p] for p in POLICIES}
        for p in POLICIES:
            c = cost[p]
            row = (f"| {p} | {c['generation_seconds']:.1f} | {c['lookup_seconds']:.1f} | {c['tool_seconds']:.1f} | "
                   f"{c['submitted_execution_seconds']:.1f} | {c['total_seconds']:.1f} | {c['per_request_seconds']:.2f} | "
                   f"{c['per_ex_correct_seconds']:.1f} | {c['prompt_tokens']:,} | {c['generated_tokens']:,} | "
                   f"{c['peak_memory_bytes'] / 1e9:.2f} |")
            has(row, f"{p} cost row")
        pct = (cost["V1"]["total_seconds"] / cost["D1"]["total_seconds"] - 1) * 100
        has(f"{pct:.1f}% higher than D1's", "V1's end-to-end cost increase")
        has(f"V1's end-to-end cost was {cost['V1']['total_seconds']:,.0f} s, against D1's "
            f"{cost['D1']['total_seconds']:,.0f} s (+{pct:.1f}%)", "cost sentence")
        ev = [r["evidence"] for r in self.reqs]
        pk = [e["package"] for e in ev if e["ok"]]
        import statistics
        secs = sorted(e["seconds"] for e in ev)
        has(f"6.2 s in total, median {statistics.median(secs):.3f} s, maximum {max(secs):.2f} s",
            "lookup seconds total, median, max")
        s.check(f"{sum(secs):.1f}" == "6.2", "lookup total = 6.2 s (own)", f"{sum(secs):.3f}")
        has(f"the payload median was {statistics.median(len(p['rendered']) for p in pk):.0f} characters, with a maximum "
            f"of {max(len(p['rendered']) for p in pk):,}", "payload median and maximum")
        inv = self.sections["2_lineage"].info["lookup_inventory"]
        has(f"{inv['with_scan_limits']} hit a scan limit", "scan-limit count")
        has(f"{inv['with_unresolved']} recorded unresolved columns", "unresolved count")
        same_final = sum(r["D1"]["sql"] == r["V1"]["sql"] for r in self.reqs)
        ch = {p: sum(r[p]["revised"]["sql"] != r["original"]["sql"] for r in self.reqs) for p in ("D1", "V1")}
        has(f"V1 and D1 end on the same final SQL in {same_final} of 128", "same final SQL count")
        has(f"({ch['V1']} revisions changed it, against {ch['D1']})", "revisions that changed the initial SQL")
        loo = set(own["V1-D1"]["leave_one_database_out"].values())
        s.check(loo <= {-1, 0, 1} and contains("V1 - D1 is -1, 0 or +1 for every database left out"),
                "report: V1 - D1 leaving out one database is in {-1, 0, +1}", str(sorted(loo)))
        broken = [(sd, t) for sd in SEEDS for t in self.ids if cc[("D0", sd, t)] and not cc[("V1", sd, t)]]
        brokenD1 = [(sd, t) for sd in SEEDS for t in self.ids if cc[("D0", sd, t)] and not cc[("D1", sd, t)]]
        s.check(broken == [(200, "bird23tf-00441")] and not brokenD1 and
                contains("V1 broke 1 of the 70 initially correct answers (bird23tf-00441, seed 200)"),
                "report: V1 broke only bird23tf-00441/200, D1 none", str(broken))
        # signal groups
        byk = {(r["seed"], r["task_id"]): r for r in self.reqs}
        grp = {}
        for r in self.reqs:
            sg = own_signals(r["evidence"]["package"]["rendered"]) if r["evidence"]["ok"] else None
            grp[(r["seed"], r["task_id"])] = "lookup failed" if sg is None else "value not stored" if \
                sg["value_not_stored"] else "other signal" if sg["other"] else "no signal"
        G = self.analysis["signal_groups"]
        for g, label in (("no signal", "no flagged signal"), ("other signal", "other signal"),
                         ("value not stored", "value not stored")):
            d = G[g]
            has(f"| {label} | {d['requests']} | {d['D0']} | {d['D1']} | {d['V1']} | {d['V1_fixes_D1_misses']} | "
                f"{d['D1_right_V1_breaks']} | {d['D0_right_V1_breaks']} |", f"signal-group row {label}")
        v1only = [k for k in byk if cc[("V1",) + k] and not cc[("D1",) + k]]
        s.check(len(v1only) == 3 and all(grp[k] == "no signal" for k in v1only) and
                contains("All three V1-only recoveries occurred in the no-flagged-signal group."),
                "report: all three V1-only recoveries are in the no-flagged-signal group (own grouping)",
                str([(k, grp[k]) for k in v1only]))
        s.check(len(broken) == 1 and grp[broken[0]] == "other signal" and
                contains("The one V1 regression of an initially correct\nanswer occurred in the other-signal group."),
                "report: the one V1 regression of an initially correct answer is in the other-signal group")
        nosig = [byk[k]["evidence"]["package"]["rendered"] for k in byk if grp[k] == "no signal"]
        kinds = ("Declared keys around", "Column ", "Columns near", "Other tables matching the question")
        s.check(all(t.strip() for t in nosig) and all(any(l.startswith(kinds) for l in t.split("\n")) for t in nosig),
                f"report: every no-flagged-signal payload ({len(nosig)}) is non-empty and shows declared keys, column "
                "notes, near-miss columns or tables")
        s.info["no_signal_payload_kinds"] = {k: sum(1 for t in nosig if any(l.startswith(k) for l in t.split("\n")))
                                             for k in kinds}
        vns = [k for k in byk if grp[k] == "value not stored"]
        s.check(len(vns) == 1 and all(not cc[(p,) + vns[0]] for p in POLICIES) and
                contains('"Value not stored, candidates shown" occurred in 1 of 128 requests, and that\n  request stayed wrong '
                'under all three policies'), "report: the one 'value not stored' request stayed wrong under all three")
        # the six discordant requests
        disc = sorted((k for k in byk if cc[("D1",) + k] != cc[("V1",) + k]), key=lambda k: (k[0], k[1]))
        want = {(200, "bird23tf-01144"): (0, 1, 0), (200, "bird23tf-00441"): (1, 1, 0), (200, "bird23tf-03981"): (0, 0, 1),
                (201, "bird23tf-01645"): (0, 1, 0), (201, "bird23tf-02677"): (0, 0, 1), (201, "bird23tf-04913"): (0, 0, 1)}
        s.check(set(disc) == set(want) and all(tuple(cc[(p,) + k] for p in POLICIES) == v for k, v in want.items()),
                "report: the six discordant requests and their D0 / D1 / V1 correctness", str(disc))
        for k, v in want.items():
            has(f"| {k[1]}, seed {k[0]} ({byk[k]['db_id']}) | {' / '.join(map(str, v))} |", f"six-request row {k[1]}/{k[0]}")
        R = lambda k: byk[k]  # noqa: E731
        sq = lambda k, p: R(k)[p]["revised"]["sql"] or ""  # noqa: E731
        err = lambda k: R(k)["original"]["execution"]["error"] or ""  # noqa: E731
        pay = lambda k: R(k)["evidence"]["package"]["rendered"]  # noqa: E731
        unres = lambda k: [u["column"] for u in R(k)["evidence"]["package"]["proposal"].get("unresolved", [])]  # noqa: E731
        k = (200, "bird23tf-01144")
        s.check("T3.award" in err(k) and "T3.award" not in sq(k, "D1") and R(k)["D1"]["revised"]["execution"]["status"] == "ok"
                and "T3.award" in sq(k, "V1") and "LIKE 'California%'" in sq(k, "V1") and "stored 'California" in pay(k)
                and R(k)["V1"]["revised"]["execution"]["status"] == "error" and R(k)["V1"]["source"] == "abstain"
                and "player_allstar.award" not in pay(k) and "T3.award" in unres(k),
                "01144/200: as described (D1 drops T3.award; V1 keeps it, LIKE 'California%', fails, abstains; payload "
                "silent on the column, recorded unresolved)")
        k = (200, "bird23tf-00441")
        tables_line = next(l for l in pay(k).split("\n") if l.startswith("Other tables matching the question"))
        s.check(R(k)["original"]["sql"] == self.ex_sql_correct(k, "D0") and sq(k, "D1").startswith("SELECT DISTINCT") and
                "events_relevant" in sq(k, "V1") and tables_line.startswith("Other tables matching the question: 1. events_relevant")
                and R(k)["V1"]["revised"]["execution"]["row_count"] == 0,
                "00441/200: as described (D1 adds DISTINCT; V1 swaps in events_relevant, first listed table, 0 rows)")
        k = (200, "bird23tf-03981")
        s.check("T1.device_id" in err(k) and sq(k, "D1") == R(k)["original"]["sql"] and R(k)["D1"]["source"] == "abstain"
                and all(x in sq(k, "V1") for x in ("events_relevant", "app_events", "gender_age")) and
                R(k)["V1"]["revised"]["execution"]["status"] == "ok" and "events_relevant" in pay(k),
                "03981/200: as described (D1 unchanged, abstains; V1 joins through events_relevant, which the payload "
                "lists in its declared keys)")
        s.info["03981_events_relevant_where"] = [l[:40] for l in pay(k).split("\n") if "events_relevant" in l]
        k = (201, "bird23tf-01645")
        s.check("T1.name" in err(k) and "T2.person" in sq(k, "D1") and R(k)["D1"]["revised"]["execution"]["status"] == "ok"
                and "T1.name" in sq(k, "V1") and R(k)["V1"]["source"] == "abstain" and "Episode.name" not in pay(k)
                and "T1.name" in unres(k),
                "01645/201: as described (D1 uses Award.person; V1 keeps T1.name, abstains; payload silent, unresolved)")
        k = (201, "bird23tf-02677")
        s.check("height_in_cm" in err(k) and "AVG(T2.height)" in sq(k, "D1") and "height_info" in sq(k, "V1") and
                "PlayerInfo.height -> height_info.height_id" in pay(k),
                "02677/201: as described (D1 averages PlayerInfo.height; V1 joins height_info on the listed key)")
        k = (201, "bird23tf-04913")
        s.check("OBJ_CLASS" in err(k) and sq(k, "D1") == R(k)["original"]["sql"] and R(k)["D1"]["source"] == "abstain" and
                "OBJ_CLASSES" in sq(k, "V1") and "IMG_OBJ.OBJ_CLASS_ID -> OBJ_CLASSES.OBJ_CLASS_ID" in pay(k),
                "04913/201: as described (D1 unchanged, abstains; V1 joins OBJ_CLASSES on the listed key)")
        # other statements
        co = self.sections["4_costs"].info["cotenancy"]
        s.check(co["load1_max"] <= 3.0 and contains("The host load stayed at\nor below 3.0"),
                "report: host load at or below 3.0", str(co["load1_max"]))
        fg = self.sections["4_costs"].info["foreign_gpu_processes"]
        s.check(fg["start"] and fg["start"] == fg["end"] and "533 MiB" in str(fg["start"]) and contains("533 MiB"),
                "report: a 533 MiB foreign process on the GPU, as the ledger recorded at start and end", str(fg["start"]))
        rs = self.sections["4_costs"].info["reservation"]
        has(f"The run took {rs['hours']:.3f} GPU-reservation hours", "reservation hours")
        has(f"ledger stands at {self.sections['4_costs'].info['ledger_total_hours_now']:.3f}", "ledger total")
        spr = json.loads(git_show(self.repo, "HEAD", H + "evidence/spr_v1/analysis/20261107T030307Z_corrected_v2/analysis_v2.json"))
        s.check(spr["comparisons"]["D1-D0"]["pooled"] == 2 and contains("In PR #71's population, D1 - D0 was +2."),
                "report: PR #71's D1 - D0 was +2")

    def ex_sql_correct(self, k, p):
        return self.reqs_by_key[k][p]["sql"]

    def run(self):
        self.started = time.time()
        self.self_sha = sha_b(Path(__file__).resolve().read_bytes())
        before = self.stat_listing()
        self.head = git(self.repo, "rev-parse", "HEAD").strip()
        self.load()
        self.worker()
        self.executions()
        self.scores_section()
        self.lineage()
        self.paired()
        self.costs()
        self.reference()
        self.integrity()
        self.report()
        after = self.stat_listing()
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        s = self.sec("0_discipline")
        s.check(not [c for c in changed if c.startswith(("dbe/", "data/")) or c == "gpu_ledger.jsonl"],
                "nothing under dbe/, data/ or the ledger changed during this verification", str(changed[:10]))
        s.check(sha_b(Path(__file__).resolve().read_bytes()) == self.self_sha, "the verifier did not change while it ran")
        s.info.update(runs_root_changes_elsewhere=changed[:50], cpu_affinity=sorted(os.sched_getaffinity(0)),
                      cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      producer_code={"worker": getattr(self, "worker_info", None),
                                     "use": "the fresh db-evidence-v1.4 lookup and the prompt's schema text and "
                                            "retrieved examples; compared with the records only"})
        secs = {k: v.as_dict() for k, v in sorted(self.sections.items())}
        ended = time.time()
        out = {"run": {"verifier": str(Path(__file__).resolve()), "verifier_sha256": self.self_sha,
                       "started_utc": iso(self.started), "ended_utc": iso(ended),
                       "runtime_seconds": round(ended - self.started, 1), "repo_head": self.head,
                       "python": sys.version.split()[0], "sqlite_version": sqlite3.sqlite_version},
               "sections": secs, "checks": sum(len(v["checks"]) for v in secs.values()),
               "failures": sum(v["failures"] for v in secs.values()), "inputs": dict(sorted(INPUTS.items()))}
        (self.out / "verify_dbe.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
        lines = [f"verify_dbe: {out['checks']} checks, {out['failures']} failures",
                 f"verifier sha256 {self.self_sha} (computed at run time); head {self.head}",
                 f"started {out['run']['started_utc']}, ended {out['run']['ended_utc']}, runtime "
                 f"{out['run']['runtime_seconds']} s; {len(INPUTS)} inputs hashed"]
        for k, v in secs.items():
            lines.append(f"\n[{k}] {v['passed']} passed, {v['failures']} failed")
            for c in v["checks"]:
                lines.append(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']}" +
                             (f"  -- {c['detail'][:400]}" if c["detail"] else ""))
        (self.out / "verify_dbe_summary.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return out


def same_obs(rec, mine) -> bool:
    if rec.get("status") != mine["status"]:
        return False
    if mine["status"] == "ok":
        return rec.get("row_count") == mine["row_count"] and rec.get("columns") == mine["columns"] and \
            json.loads(json.dumps(rec.get("rows"))) == json.loads(json.dumps(mine["rows"])) and \
            bool(rec.get("truncated")) == bool(mine.get("truncated"))
    if mine["status"] == "error":
        return rec.get("error") == mine.get("error")
    return True


def obs_as_record(m):
    return {"status": m["status"], "error": m.get("error"), "rows": m.get("rows", []), "columns": m.get("columns", []),
            "row_count": m.get("row_count"), "truncated": m.get("truncated", False)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--worker":
        worker_main(*argv[1:4])
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-root", default=os.environ.get("TI_TNN_SCALING_RUNS"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=str(REPO_DEFAULT))
    ap.add_argument("--frozen-tree", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-worker", action="store_true")
    a = ap.parse_args(argv)
    if not a.runs_root:
        ap.error("--runs-root (or TI_TNN_SCALING_RUNS) is required")
    V(a).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
