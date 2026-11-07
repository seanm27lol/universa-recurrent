#!/usr/bin/env python3
"""Independent verifier for the TI-TNN semantic-proposal-repair study (development), the reviewer's, written apart
from the producers.

    CUDA_VISIBLE_DEVICES= taskset -c 5-9,15-19 nice -n 10 python verify_spr.py --runs-root RUNS --out DIR
        [--repo REPO] [--frozen-tree TREE] [--workers 4] [--no-worker] [--producer-report]
    (no machine default for --runs-root or --out; --repo defaults to the checkout holding this file)

Nothing is written under the runs root; a stat listing of the runs root before and after is compared.

**What it checks** (the brief's nine items, one section each, plus the bundle):
1. inventories: the split and the inventory rebuilt from the raw release (own order rule); positions 192-255 = the
   pin; own prior-use scan of the runs root, the evidence bundles and the pins (task ids, question texts, and the
   JSON headers of safetensors files); the frozen run's coverage (64 x 2 seeds x 3 policies, once each).
2. lineage: the shared initial proposal; own extraction rule on every recorded reply; own revision order; own raw
   observation and own rendering of the recorded typed analysis against both feedback messages; the deadline
   arithmetic; the prompt token counts of all three calls re-rendered with the model's own tokenizer from the recorded
   contents; the code of `serve` (same seed, prefix and remaining time).
3. selection: own read-only check, own execution and own rule, against every recorded source and SQL; no gold read
   in the selection path (code).
4. scores: every scored SQL (run, dev) and the preservation test's queries re-executed with this file's own EX rule
   on the pinned snapshots (hashed here), each record mapped to its snapshot sha256 and SQL sha256.
5. gold isolation: the diagnostic path's code (transitive within the modules), and every prompt, feedback message and
   typed analysis searched for gold SQL, gold results and evaluated task ids.
6. preservation: own re-execution of the original and compiled queries; outcome re-derived.
7. paired differences, the rule and the decision, from own scores; agreement with analysis.json.
8. costs (from records), the ledger (own parse), the co-tenancy log.
9. freeze order (git), the dev runs' populations and claims.
10. the bundle: SHA256SUMS and identity with the runs-root originals; independent/ = the first review's outputs.
11. the correction pass (2026-11-07): the known historical defect (the original analysis.json's rule fields at 7 of
    128) checked as still present exactly as known and recorded under "known_defects", not as a failure; the
    corrected analysis (spr-rule-v2) checked for +8 everywhere, its conditions against own computation, and its counts,
    comparison values and decision against the original's and own.
12. (--producer-report) `python -m scaling.spr_report` with TI_TNN_SCALING_RUNS unset, and the spr tests: producer
    code in separate processes, flagged; their outcome is recorded, never used for an own verdict.

**The run record** (verify_spr.json "run", "inputs"): the verifier's sha256 computed by the running process from its
own bytes at start (re-read at the end), the start and end times and runtime, the head, and the sha256 of every input
read (release, pins, manifests, records, analyses, ledger, logs, snapshots, tokenizer files, git objects; the
prior-use scan as one digest). Nothing relies on file modification times.

History: first run at 1289d7e (sha256 81935f4b..., 167 checks, 1 failure: the rule fields); its outputs are in
evidence/spr_v1/independent/. This version adds items 11-12 and the run record; the failure is now item 11's known
defect.

**Discipline.** No producer module is imported by this process; producer constants are read from git
(`git show <commit>:path`, ast). One flagged exception, the worker (`--worker`), a separate process importing the
frozen worktree's producer modules, for what cannot be done otherwise without re-implementing the schema serializer,
the BM25 retrieval and the typed IR: the prompt's schema text and retrieved examples, the typed analysis recomputed
from the recorded SQL and execution, and the preservation test's compilation. Its outputs are only compared with
the records; every verdict of items 3, 4, 6 and 7 comes from this file's own code.
Third-party: sqlite3, sqlglot (the read-only check and bare-statement extraction), transformers' tokenizer.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.parse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[4]
H = "research/ti_tnn/scaling/"
FROZEN = "2f76bb7"          # the commit the frozen run ran at (the brief)
FREEZE_AT = "6e1003f"       # the protocol's frozen_at (the brief)
CORRECTED = "20261107T030307Z_corrected_v2"   # the correction pass's versioned analysis (spr-rule-v2)
CORRECTION_COMMIT = "1d042a0"                 # the commit the corrected analysis ran at
FIRST_REVIEW = {                              # this verifier's first run at 1289d7e (167 checks, 1 failure)
    "verify_spr.py": "81935f4be0f3ed163e02fa98b6ff911cd31b27e03e9a98c2f6c8f55581799fe0",
    "verify_spr.json": "97f30334204e046e9eb88480e9fc084e55921ef754c501937d35416428d5e411",
    "verify_spr_summary.txt": "3931a84236232bfe6958476682e3111e908848a2d841df44da4e464283792518",
    "score_mapping.jsonl": "809850b3e5489d8fc8603f9beefb4c7faa5ee6a0a5848acc76a4b344cfaa1815",
    "preservation_own.jsonl": "523fc506da5a8791690980dc27e692c6c39ff91b9618e141f3023bc973d143a0"}
# The known historical defect, found by the first run: the original analysis.json's per-comparison rule fields came
# from perf_analysis.paired (threshold 7 of 128), not the study's +8. Recorded as known, not as a new failure.
KNOWN_DEFECT_ANALYSIS_SHA256 = "ceb208f628c5224b17f42be9c34ee61b6e3dc747395339794106434e604f349f"
KNOWN_DEFECT_THRESHOLD = 7
# Modules the frozen run, the scoring and the analysis executed (they must be unchanged from 2f76bb7 to the head).
RUN_MODULES = ("spr_run", "spr_typed", "spr_feedback", "gi_llm", "gi_run", "llm_agent", "tools", "scorer", "contracts",
               "data", "schema", "retrieval", "sql_to_ir", "ir", "diagnose", "runtime", "ledger", "identity",
               "perf_analysis", "pilot_run")
WORK_ORDER_LIMIT = 28.823   # the brief's limit
RUN, ANALYSIS, PRESERVE = "20261106T204616Z", "20261106T211558Z", "20261106T150505Z"
DEVS = ("20261106T150736Z", "20261106T151520Z")
GI_DIRECT = "grounding_interface/direct/20261103T202939Z_new"
RELEASE = "data/raw/hf/bird23_train_filtered.train-00000-of-00001.jsonl"
POLICIES = ("D0", "D1", "TI1")
TIMEOUT = 30.0


# --- the check list ------------------------------------------------------------------------------------------------
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


def own_render(a: dict, limit=4000) -> str:
    out = ["Typed analysis of your query (deterministic checks against this database's schema; the warnings are "
           "observations, not verdicts):"]
    out += (["Typed plan:"] + a["plan_lines"]) if a["typed"] else ["Typed plan: not available."]
    out += ["Errors:"] + ([f"  - {e}" for e in a["errors"]] or ["  (none)"])
    out += ["Warnings:"] + ([f"  - {w}" for w in a["warnings"]] or ["  (none)"])
    t = "\n".join(out)
    return t if len(t) <= limit else t[:limit] + "\n  ... (analysis truncated)"


ERROR_PATTERNS = (r"^no SQL query was found in the reply$", r"^(unknown table|unknown column|unresolved column|"
                  r"ambiguous column|unresolved T\.\*|unresolved derived column|USING column not found) against this "
                  r"database's schema: ", r"^validate: ", r"^execution timed out: ", r"^execution \w+: ")
WARNING_PATTERNS = (r"^the query uses a construct outside the typed representation \(", r"^in .*, the ON predicate of the "
                    r"join with .* references only ", r"^in .*, the join with .* has no ON predicate",
                    r"^in .*, the literal .* does not occur verbatim in the question or evidence",
                    r"^the query returned no rows \(this can be the right answer\)$")


def occurs(value: str, text: str) -> bool:
    v = value.strip().strip("'\"").strip()
    return bool(v) and v.lower() in text.lower()


# --- the worker (producer code, separate process, flagged) ---------------------------------------------------------
def worker_main(job_path: str, out_path: str, frozen_tree: str) -> None:
    sys.path.insert(0, str(Path(frozen_tree) / "research" / "ti_tnn"))
    from scaling import contracts as C  # noqa: PLC0415  (producer code: the flagged worker only)
    from scaling import data, gi_llm, gi_run, retrieval, spr_feedback, spr_typed  # noqa: PLC0415
    job = json.loads(Path(job_path).read_text())
    out = {"prompts": {}, "typed": {}, "compile": {}, "modules": {}}
    limits = C.Limits()
    index = retrieval.training_pool_index()
    for tid in job["prompts"]:
        task = data.task(tid)
        text, examples = gi_run._prompt_inputs(task, limits, index)
        msgs = gi_llm.messages(gi_llm.SYSTEM_DIRECT, task, text, examples)
        out["prompts"][tid] = {"system": msgs[0]["content"], "user": msgs[1]["content"], "schema_text": text,
                               "retrieved": [e.task_id for e in examples], "retrieved_db": [e.db_id for e in examples]}
    for key, item in job["typed"].items():
        task = data.task(item["task_id"])
        a = spr_feedback.typed_analysis(task, data.snapshot(task.db_id), item["sql"], item["execution"])
        a.pop("seconds", None)
        out["typed"][key] = a
    for key, item in job["compile"].items():
        task = data.task(item["task_id"])
        snap = data.snapshot(task.db_id)
        p = spr_typed.parse(item["sql"], spr_typed.schema_index(task, snap))
        out["compile"][key] = {"supported": p["supported"], "reason": p["reason"], "compiled": p["compiled"]}
    out["modules"] = {m: getattr(sys.modules.get(f"scaling.{m}"), "__file__", None)
                      for m in ("gi_llm", "gi_run", "spr_feedback", "spr_typed", "retrieval", "data")}
    Path(out_path).write_text(json.dumps(out, default=str))


# --- the verifier --------------------------------------------------------------------------------------------------
class V:
    def __init__(self, a):
        self.repo, self.runs, self.out = Path(a.repo).resolve(), Path(a.runs_root).resolve(), Path(a.out).resolve()
        self.frozen_tree, self.workers, self.no_worker = a.frozen_tree, a.workers, a.no_worker
        self.producer_report = a.producer_report
        self.known_defects = []
        self.out.mkdir(parents=True, exist_ok=True)
        if self.runs in self.out.parents or self.out == self.runs:
            raise SystemExit("--out must not be under the runs root")
        self.sections = {}
        self.spr = self.runs / "spr"

    def sec(self, name):
        return self.sections.setdefault(name, Section(name))

    # -- shared inputs --
    def load(self):
        R, f = self.repo, FROZEN
        self.contracts_src = git_show(R, f, H + "contracts.py")
        self.C = literals(self.contracts_src, {"TRAIN_RELEASE", "SPLIT", "INVENTORY", "MODELS", "OFFICIAL_SCORER"})
        self.limits = class_defaults(self.contracts_src, "Limits")
        self.spr_src = git_show(R, f, H + "spr_run.py")
        self.S = literals(self.spr_src, {"SEEDS", "DEV_SEED", "DEV_SAMPLE", "DEV_SALT", "MAX_NEW_TOKENS",
                                         "REQUEST_DEADLINE_SECONDS", "POLICIES", "LEAD", "REVISE", "PROMPT_VERSION",
                                         "WORK_ORDER_USED_AT_START"})
        self.fb_src = git_show(R, f, H + "spr_feedback.py")
        self.F = literals(self.fb_src, {"FEEDBACK_VERSION", "IDENTIFIER_REASONS", "CELL_CHARS", "FEEDBACK_CHARS"})
        self.gi_src = git_show(R, f, H + "gi_llm.py")
        self.G = literals(self.gi_src, {"SYSTEM_DIRECT", "MAX_NEW_TOKENS"})
        self.pins = {n: json.loads(git_show(R, f, H + f"pins/{n}.json")) for n in
                     ("spr_new_eval", "spr_protocol", "inventory_v1", "split_v1", "snapshots_v1", "perf_v2_eval",
                      "gi_new_eval")}
        # the raw release, own parse
        rel = self.runs / RELEASE
        self.release_sha = sha_f(rel)
        lines = rt(rel).split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        self.rows = {}
        for i, line in enumerate(lines):
            r = json.loads(line)
            self.rows[f"bird23tf-{i:05d}"] = {k: str(r[k]) for k in ("db_id", "question", "evidence", "SQL")}
        snaps = self.pins["snapshots_v1"]["databases"]
        self.snap = {d: {"path": str(self.runs / "data" / "databases" / e["file"]), "sha256": e["sha256"],
                         "bytes": e["bytes"]} for d, e in snaps.items()}
        self.run_dir = self.spr / "run" / RUN
        self.reqs = jl(self.run_dir / "requests.jsonl")
        self.scores = jl(self.run_dir / "scores.jsonl")
        self.dev = {d: {"reqs": jl(self.spr / "dev" / d / "requests.jsonl"),
                        "scores": jl(self.spr / "dev" / d / "scores.jsonl"),
                        "manifest": json.loads(rt(self.spr / "dev" / d / "manifest.json"))} for d in DEVS}
        self.manifest = json.loads(rt(self.run_dir / "manifest.json"))
        self.analysis = json.loads(rt(self.spr / "analysis" / ANALYSIS / "analysis.json"))
        self.pres = jl(self.spr / "preserve" / PRESERVE / "records.jsonl")
        self.pres_summary = json.loads(rt(self.spr / "preserve" / PRESERVE / "summary.json"))
        self.pres_manifest = json.loads(rt(self.spr / "preserve" / PRESERVE / "manifest.json"))
        self.new_ids = self.pins["spr_new_eval"]["task_ids"]

    def stat_listing(self):
        out = {}
        for p in self.runs.rglob("*"):
            with contextlib.suppress(OSError):
                if p.is_file():
                    st = p.stat()
                    out[str(p.relative_to(self.runs))] = (st.st_size, st.st_mtime_ns)
        return out

    # -- 1. inventories --
    def inventories(self):
        s = self.sec("1_inventories")
        TR, SP, INV = self.C["TRAIN_RELEASE"], self.C["SPLIT"], self.C["INVENTORY"]
        s.check(self.release_sha == TR["files"]["data/train-00000-of-00001.jsonl"], "raw release sha256 = contracts pin",
                self.release_sha)
        s.check(len(self.rows) == TR["rows"] and len({r["db_id"] for r in self.rows.values()}) == TR["databases"],
                "release rows and databases", f"{len(self.rows)} rows")
        dbs = sorted({r["db_id"] for r in self.rows.values()},
                     key=lambda d: (hashlib.sha256(f"{SP['salt']}:{d}".encode()).hexdigest(), d))
        n_dev = round(SP["dev_fraction"] * len(dbs))
        dev = dbs[:n_dev]
        s.check(dev == self.pins["split_v1"]["dev"], "own split = split pin (14 dev databases, same order)", str(dev))
        excluded = {e["task_id"] for e in self.pins["inventory_v1"]["excluded"]}
        gc = json.loads(rt(self.runs / "data" / "gold_check_dev_v1.json"))["results"]
        s.check({t for t, v in gc.items() if v["status"] != "ok"} == excluded,
                "the inventory pin's exclusions = the gold check's non-ok entries", str(sorted(excluded)))

        def order(excl):
            per = {d: sorted((t for t, r in self.rows.items() if r["db_id"] == d and t not in excl),
                             key=lambda t: (hashlib.sha256(f"{INV['salt']}:{t}".encode()).hexdigest(), t)) for d in dev}
            out, pos = [], 0
            while len(out) < INV["size"] and any(per[d][pos:] for d in dev):
                for d in dev:
                    if pos < len(per[d]):
                        out.append(per[d][pos])
                        if len(out) == INV["size"]:
                            break
                pos += 1
            return out
        inv = order(excluded)
        inv_noexcl = order(set())
        s.check(inv == self.pins["inventory_v1"]["task_ids"], "own inventory (512) = inventory pin")
        s.check(inv[192:256] == self.new_ids, "positions 192-255 rebuilt from the release = spr_new_eval.json")
        s.check([self.rows[t]["db_id"] for t in self.new_ids] == self.pins["spr_new_eval"]["db_ids"],
                "spr_new_eval db_ids = the release's")
        s.check(self.pins["spr_new_eval"]["positions"] == [192, 256] and len(set(self.new_ids)) == 64,
                "64 distinct ids at positions [192, 256)")
        s.info["exclusion_changes_positions_0_255"] = inv_noexcl[:256] != inv[:256]
        s.info["new_population_by_db"] = dict(Counter(self.rows[t]["db_id"] for t in self.new_ids))
        # the excluded gold really exceeds the timeout (own executor)
        if excluded:
            t = sorted(excluded)[0]
            _k, r = observe((t, self.snap[self.rows[t]["db_id"]]["path"], self.rows[t]["SQL"]))
            s.check(r["status"] == "timeout", f"the excluded {t}'s gold times out under the own executor", r["status"])
        # earlier populations
        perf = self.pins["perf_v2_eval"]["populations"]
        earlier = perf["historical"]["task_ids"] + perf["new"]["task_ids"]
        s.check(earlier == inv[:128], "the earlier dev questions (perf_v2 historical + new) = positions 0-127")
        s.check(self.pins["gi_new_eval"]["task_ids"] == inv[128:192], "gi_new_eval = positions 128-191")
        self.inv = inv
        self.earlier = earlier
        # prior-use scan
        s.info["prior_use"] = self.prior_use_scan(s)
        # coverage of the frozen run
        keys = Counter((r["seed"], r["task_id"]) for r in self.reqs)
        s.check(sorted(keys) == sorted((sd, t) for sd in (200, 201) for t in self.new_ids) and max(keys.values()) == 1,
                "frozen run: 64 questions x seeds 200, 201, each request exactly once", f"{len(self.reqs)} requests")
        sk = Counter((x["seed"], x["task_id"], x["policy"]) for x in self.scores)
        s.check(sorted(sk) == sorted((sd, t, p) for sd in (200, 201) for t in self.new_ids for p in POLICIES)
                and max(sk.values()) == 1, "frozen run scores: 64 x 2 x 3 policies, each exactly once",
                f"{len(self.scores)} scores")
        s.check(all(all(p in r and r[p].get("source") in ("original", "revised", "abstain") for p in POLICIES)
                    for r in self.reqs), "every request carries all three policies")
        s.check(self.manifest["config"]["requested"] == self.new_ids and self.manifest["config"]["seeds"] == [200, 201],
                "run manifest: requested = the pin, seeds [200, 201]")
        s.check(all(self.rows[r["task_id"]]["db_id"] == r["db_id"] for r in self.reqs),
                "every request's db_id = the release's")

    def prior_use_scan(self, s):
        ids = set(self.new_ids)
        qtexts = {t: self.rows[t]["question"] for t in self.new_ids}
        qforms = {t: {q, json.dumps(q)[1:-1], json.dumps(q, ensure_ascii=False)[1:-1]} for t, q in qtexts.items()}
        # questions duplicated in the release (their text would match other rows too)
        dup = {t for t, q in qtexts.items() if sum(1 for r in self.rows.values() if r["question"] == q) > 1}
        idpat = re.compile(r"bird23tf-\d{5}")
        hits_id, hits_q, scanned, headers = {}, {}, 0, 0
        scan_digest = hashlib.sha256()
        roots = [("runs", self.runs), ("evidence", self.repo / H / "evidence"), ("pins", self.repo / H / "pins")]
        binary = {".sqlite", ".pt", ".zip", ".npy", ".bin", ".gguf"}
        for rname, root in roots:
            for p in sorted(root.rglob("*")):
                if not p.is_file():
                    continue
                rel = f"{rname}:{p.relative_to(root)}"
                if p.suffix in binary:
                    continue
                if p.suffix == ".safetensors":
                    with open(p, "rb") as f:
                        n = int.from_bytes(f.read(8), "little")
                        raw = f.read(min(n, 50_000_000))
                    text = raw.decode("utf-8", "ignore")
                    headers += 1
                else:
                    if p.stat().st_size > 400_000_000:
                        continue
                    raw = p.read_bytes()
                    text = raw.decode("utf-8", "ignore")
                scanned += 1
                scan_digest.update(f"{rel}\0{sha_b(raw)}\n".encode())
                n_id = len(ids & set(idpat.findall(text)))
                if n_id:
                    hits_id[rel] = n_id
                n_q = sum(1 for t, forms in qforms.items() if any(fm in text for fm in forms))
                if n_q:
                    hits_q[rel] = n_q
        expected = {"pins:inventory_v1.json", "pins:spr_new_eval.json", "runs:data/gold_check_dev_v1.json",
                    "runs:ir/coverage_details_v1.json"}

        def own_study(rel):
            return rel.startswith("runs:spr/run/") or rel.startswith("runs:spr/analysis/") or \
                rel.startswith("runs:spr/logs/run_") or rel.startswith("runs:spr/logs/analyze_") or \
                rel.startswith("evidence:spr_v1/run/") or rel.startswith("evidence:spr_v1/analysis/") or \
                rel.startswith("evidence:spr_v1/logs/run_") or rel.startswith("evidence:spr_v1/logs/analyze_") or \
                rel.startswith("evidence:spr_v1/independent/")   # this verifier's own first-run output
        unexpected_id = {k: v for k, v in hits_id.items() if k not in expected and not own_study(k)}
        q_expected = {"runs:" + RELEASE}
        unexpected_q = {k: v for k, v in hits_q.items() if k not in q_expected and k not in expected and not own_study(k)}
        s.check(not unexpected_id, "no file outside the pins, gold check, coverage analysis and this study's own run "
                "names a new-population task id", json.dumps(unexpected_id)[:2000])
        s.check(not unexpected_q, "no file outside the release (and those) holds a new-population question text",
                json.dumps(unexpected_q)[:2000])
        s.check(expected - {"pins:spr_new_eval.json"} <= set(hits_id),
                "the three declared prior uses are found (inventory pin, gold check, coverage analysis)", json.dumps({k: hits_id.get(k) for k in sorted(expected)}))
        record_input("prior-use scan (runs root, evidence bundles, pins; text files and safetensors headers)",
                     scan_digest.hexdigest(), scanned)
        return {"files_scanned": scanned, "safetensors_headers": headers, "id_hits": hits_id, "question_hits": hits_q,
                "scan_digest": scan_digest.hexdigest(),
                "duplicated_question_texts_in_release": sorted(dup)}

    # -- executions (own) --
    def executions(self):
        """Own observation of every candidate (run, dev) and own EX of every scored SQL, preservation included."""
        obs_jobs, ex_jobs = {}, {}
        for rq in self.all_requests():
            path = self.snap[rq["db_id"]]["path"]
            cands = [rq["original"]["sql"]] + [rq[p]["revised"]["sql"] for p in ("D1", "TI1")]
            for sql in cands:
                obs_jobs[(rq["db_id"], sql)] = path
        for sc in self.all_scores():
            ex_jobs[(sc["task_id"], sc["sql"])] = self.snap[sc["db_id"]]["path"]
        for r in self.pres:
            p = self.snap[r["db_id"]]["path"]
            ex_jobs[(r["task_id"], r["sql"])] = p
            obs_jobs[(r["db_id"], r["sql"])] = p
            if r.get("compiled"):
                ex_jobs[(r["task_id"], r["compiled"])] = p
                obs_jobs[(r["db_id"], r["compiled"])] = p
        t0 = time.time()
        self.obs, self.obs_reruns = run_jobs(observe, [(k, v, k[1]) for k, v in obs_jobs.items()], self.workers)
        self.ex, self.ex_reruns = run_jobs(ex_pair, [(k, v, k[1], self.rows[k[0]]["SQL"]) for k, v in ex_jobs.items()],
                                           self.workers)
        self.exec_seconds = time.time() - t0

    def all_requests(self):
        yield from self.reqs
        for d in DEVS:
            yield from self.dev[d]["reqs"]

    def all_scores(self):
        yield from self.scores
        for d in DEVS:
            yield from self.dev[d]["scores"]

    # -- the worker --
    def worker(self):
        self.W = None
        if self.no_worker or not self.frozen_tree:
            return
        typed, comp = {}, {}
        for tag, reqs in [("run", self.reqs)] + [(d, self.dev[d]["reqs"]) for d in DEVS]:
            for r in reqs:
                base = f"{tag}|{r['seed']}|{r['task_id']}"
                typed[base + "|original"] = {"task_id": r["task_id"], "sql": r["original"]["sql"],
                                             "execution": r["original"]["execution"]}
                for p in ("D1", "TI1"):
                    typed[f"{base}|{p}"] = {"task_id": r["task_id"], "sql": r[p]["revised"]["sql"],
                                            "execution": r[p]["revised"]["execution"]}
        for r in self.pres:
            comp[r["task_id"]] = {"task_id": r["task_id"], "sql": r["sql"]}
        ids = sorted({r["task_id"] for r in self.all_requests()})
        job = self.out / "worker_job.json"
        res = self.out / "worker_out.json"
        job.write_text(json.dumps({"prompts": ids, "typed": typed, "compile": comp}, default=str))
        env = {**os.environ, "TI_TNN_SCALING_RUNS": str(self.runs), "CUDA_VISIBLE_DEVICES": ""}
        t0 = time.time()
        p = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--worker", str(job), str(res),
                            str(self.frozen_tree)], cwd=self.out, env=env, capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit(f"worker failed: {p.stderr[-3000:]}")
        self.W = json.loads(rt(res))
        self.worker_seconds = time.time() - t0
        frozen_head = git(Path(self.frozen_tree), "rev-parse", "HEAD").strip()
        dirty = git(Path(self.frozen_tree), "status", "--porcelain", "--", "research/ti_tnn").strip()
        self.worker_info = {"frozen_tree": str(self.frozen_tree), "head": frozen_head, "research_dirty": dirty,
                            "modules": self.W["modules"], "seconds": round(self.worker_seconds, 1)}

    # -- tokenizer --
    def tokenizer(self):
        from transformers import AutoTokenizer
        m = self.C["MODELS"]["B4"]
        root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
        d = root / f"models--{m['repo'].replace('/', '--')}" / "snapshots" / m["revision"]
        tok = AutoTokenizer.from_pretrained(str(d), local_files_only=True)
        tpl = sha_f(d / "chat_template.jinja")
        for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            if (d / f).exists():
                sha_f(d / f)
        rec = self.manifest["model_identity"]["files"]["chat_template.jinja"]["sha256"]
        return tok, tpl == rec, tpl

    # -- 2 + 3 lineage and selection --
    def lineage(self):
        s2, s3 = self.sec("2_lineage"), self.sec("3_selection")
        tok, tpl_ok, _ = self.tokenizer()
        s2.check(tpl_ok, "the local chat template = the run manifest's model identity")
        REVISE, SYSTEM = self.S["REVISE"], self.G["SYSTEM_DIRECT"]
        s2.check(self.manifest["config"]["revise"] == REVISE and self.manifest["config"]["system"] == SYSTEM,
                 "run manifest's system prompt and revision instruction = the frozen source's")
        s2.check(self.manifest["config"]["generation"]["max_new_tokens"] == 512 and
                 not self.manifest["config"]["generation"]["thinking"], "512 new tokens, non-thinking")
        revise_v1 = literals(git_show(self.repo, "d73c30e", H + "spr_run.py"), {"REVISE"})["REVISE"]
        populations = [("run", self.reqs, REVISE, (200, 201))] + \
            [(d, self.dev[d]["reqs"], self.dev[d]["manifest"]["config"]["revise"], (900,)) for d in DEVS]
        s2.check(self.dev[DEVS[0]]["manifest"]["config"]["revise"] == revise_v1 and
                 self.dev[DEVS[1]]["manifest"]["config"]["revise"] == REVISE, "dev run 1 used the v1 instruction "
                 "(d73c30e), dev run 2 the frozen v2")
        fails = defaultdict(list)
        stats = defaultdict(Counter)
        self.sel = {}
        for tag, reqs, revise, seeds in populations:
            st = stats[tag]
            for r in reqs:
                k = f"{tag}|{r['seed']}|{r['task_id']}"
                o = r["original"]
                st["requests"] += 1
                # shared initial proposal
                if not (r["D0"] == {"source": "original", "sql": o["sql"]}):
                    fails["D0 = the initial proposal"].append(k)
                # extraction
                for name, c in [("original", o)] + [(p, r[p]["revised"]) for p in ("D1", "TI1")]:
                    sql, how = own_extract(c["call"]["content"])
                    st[f"extract_{how}"] += 1
                    if sql != c["sql"]:
                        fails["own extraction = the recorded SQL"].append(f"{k}|{name}")
                    if c["call"]["finish_reason"] != "eos":
                        st[f"{name}_not_eos"] += 1
                    if c["call"]["generated_tokens"] > 512:
                        fails["generated tokens <= 512"].append(f"{k}|{name}")
                # revision order
                h = int(hashlib.sha256(f"spr-order:{r['seed']}:{r['task_id']}".encode()).hexdigest()[:8], 16)
                want = ["D1", "TI1"] if h % 2 == 0 else ["TI1", "D1"]
                st["D1_first"] += want[0] == "D1"
                if r["revision_order"] != want:
                    fails["revision order = the declared hash rule"].append(k)
                # feedback messages
                raw = own_raw(o["execution"])
                if r["D1"]["revised"]["feedback"] != f"{raw}\n\n{revise}":
                    fails["D1 message = own raw observation + the instruction"].append(k)
                ti = r["TI1"]["revised"]["feedback"]
                if ti != f"{raw}\n\n{own_render(o['typed'])}\n\n{revise}":
                    fails["TI1 message = D1's raw observation + own rendering of the recorded analysis + the "
                          "instruction"].append(k)
                d1 = r["D1"]["revised"]["feedback"]
                if not (ti.startswith(raw + "\n\n") and ti.endswith("\n\n" + revise) and d1.startswith(raw)):
                    fails["TI1 message = D1's plus the analysis only (prefix/suffix)"].append(k)
                # own execution of the initial SQL = the recorded observation
                mine = self.obs[(r["db_id"], o["sql"])]
                rec = o["execution"]
                if not self.same_obs(rec, mine):
                    fails["own execution of the initial SQL = the recorded execution"].append(
                        f"{k}: {rec.get('status')}/{rec.get('row_count')} vs {mine['status']}/{mine.get('row_count')}")
                if rec.get("status") in ("timeout", "rejected"):
                    st[f"raw_rebuild_skipped_{rec.get('status')}"] += 1  # wording is the producer's (status checked)
                elif own_raw(rec) != own_raw(self.obs_as_record(mine)):
                    fails["the raw observation rebuilt from own execution = the recorded one (ok and error statuses)"].append(k)
                for p in ("D1", "TI1"):
                    rv = r[p]["revised"]
                    m2 = self.obs[(r["db_id"], rv["sql"])]
                    if not self.same_obs(rv["execution"], m2):
                        fails["own execution of each revision = the recorded execution"].append(
                            f"{k}|{p}: {rv['execution'].get('status')} vs {m2['status']}")
                # deadline
                allowed = r["revision_seconds_allowed"]
                spent = o["call"]["seconds"] + o["execution"]["seconds"] + o["typed"]["seconds"]
                gap = self.S["REQUEST_DEADLINE_SECONDS"] - allowed - spent
                st["max_deadline_gap"] = max(st["max_deadline_gap"], gap)
                if not (0 <= gap < 2.0):
                    fails["remaining time = 180 s - (initial call + execution + analysis), within 2 s"].append(
                        f"{k}: gap {gap:.3f}")
                for p in ("D1", "TI1"):
                    if r[p]["revised"]["call"]["seconds"] > allowed:
                        fails["each revision finished within the shared remaining time"].append(f"{k}|{p}")
                # prompt token counts
                if self.W:
                    pr = self.W["prompts"][r["task_id"]]
                    if pr["retrieved"] != r["retrieved"]:
                        fails["retrieved examples = the producer's retrieval (worker)"].append(k)
                    if not all(self.rows[t]["db_id"] in self.pins["split_v1"]["train_pool"] for t in r["retrieved"]):
                        fails["retrieved examples from training-pool databases only"].append(k)
                    user = self.own_user(r["task_id"], pr["schema_text"], r["retrieved"])
                    if user != pr["user"]:
                        fails["own user message = the producer's (worker)"].append(k)
                    if pr["system"] != SYSTEM:
                        fails["system prompt = SYSTEM_DIRECT"].append(k)
                    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
                    n0 = self.count(tok, base)
                    if n0 != o["call"]["prompt_tokens"]:
                        fails["initial prompt tokens = own rendering"].append(f"{k}: {n0} vs {o['call']['prompt_tokens']}")
                    for p in ("D1", "TI1"):
                        msgs = base + [{"role": "assistant", "content": o["call"]["content"]},
                                       {"role": "user", "content": r[p]["revised"]["feedback"]}]
                        n = self.count(tok, msgs)
                        if n != r[p]["revised"]["call"]["prompt_tokens"]:
                            fails[f"{p} revision prompt tokens = own rendering of the shared prefix + its message"].append(
                                f"{k}: {n} vs {r[p]['revised']['call']['prompt_tokens']}")
                    # typed analysis recomputed by the producer (worker) = recorded
                    for name, c in [("original", o)] + [(p, r[p]["revised"]) for p in ("D1", "TI1")]:
                        w = self.W["typed"][f"{k}|{name}"]
                        if {x: c["typed"][x] for x in w} != w:
                            fails["recorded typed analysis = recomputed (worker)"].append(f"{k}|{name}")
                # own consistency checks of the recorded analysis
                self.typed_checks(r, o, fails, st)
                for p in ("D1", "TI1"):
                    self.typed_checks(r, r[p]["revised"], fails, st, label=p)
                # selection
                for p in ("D1", "TI1"):
                    rv = r[p]["revised"]
                    mine_rev, mine_orig = self.obs[(r["db_id"], rv["sql"])], mine

                    def valid(sql, ob):
                        return bool(sql) and own_readonly(sql) is None and ob["status"] == "ok"
                    if valid(rv["sql"], mine_rev):
                        want_sel = ("revised", rv["sql"])
                    elif valid(o["sql"], mine_orig):
                        want_sel = ("original", o["sql"])
                    else:
                        want_sel = ("abstain", None)
                    # the same rule on the recorded executions
                    if bool(rv["sql"]) and own_readonly(rv["sql"]) is None and rv["execution"]["status"] == "ok":
                        rec_sel = ("revised", rv["sql"])
                    elif bool(o["sql"]) and own_readonly(o["sql"]) is None and o["execution"]["status"] == "ok":
                        rec_sel = ("original", o["sql"])
                    else:
                        rec_sel = ("abstain", None)
                    self.sel[(tag, r["seed"], r["task_id"], p)] = want_sel
                    st[f"{p}_{want_sel[0]}"] += 1
                    if (r[p]["source"], r[p]["sql"]) != want_sel:
                        fails[f"selection (own execution) = recorded ({p})"].append(
                            f"{k}: {r[p]['source']} vs {want_sel[0]}")
                    if (r[p]["source"], r[p]["sql"]) != rec_sel:
                        fails[f"selection (recorded executions) = recorded ({p})"].append(k)
                self.sel[(tag, r["seed"], r["task_id"], "D0")] = ("original", o["sql"])
        names2 = ["D0 = the initial proposal", "own extraction = the recorded SQL", "generated tokens <= 512",
                  "revision order = the declared hash rule", "D1 message = own raw observation + the instruction",
                  "TI1 message = D1's raw observation + own rendering of the recorded analysis + the instruction",
                  "TI1 message = D1's plus the analysis only (prefix/suffix)",
                  "own execution of the initial SQL = the recorded execution",
                  "the raw observation rebuilt from own execution = the recorded one (ok and error statuses)",
                  "own execution of each revision = the recorded execution",
                  "remaining time = 180 s - (initial call + execution + analysis), within 2 s",
                  "each revision finished within the shared remaining time",
                  "typed analysis: errors only of the declared definite kinds",
                  "typed analysis: warnings only of the declared observational kinds",
                  "typed analysis: execution errors and timeouts reported iff the execution failed",
                  "typed analysis: the empty-result warning iff ok with 0 rows",
                  "typed analysis: own literal-occurrence test = each literal line",
                  "typed analysis: untyped iff a reason; identifier reasons as errors, others as warnings"]
        if self.W:
            names2 += ["retrieved examples = the producer's retrieval (worker)",
                       "retrieved examples from training-pool databases only", "own user message = the producer's (worker)",
                       "system prompt = SYSTEM_DIRECT", "initial prompt tokens = own rendering",
                       "D1 revision prompt tokens = own rendering of the shared prefix + its message",
                       "TI1 revision prompt tokens = own rendering of the shared prefix + its message",
                       "recorded typed analysis = recomputed (worker)"]
        for n in names2:
            s2.check(not fails.get(n), n, "; ".join(map(str, fails.get(n, [])[:8])) + (
                f" (+{len(fails[n]) - 8})" if len(fails.get(n, [])) > 8 else ""))
        for n in ["selection (own execution) = recorded (D1)", "selection (own execution) = recorded (TI1)",
                  "selection (recorded executions) = recorded (D1)", "selection (recorded executions) = recorded (TI1)"]:
            s3.check(not fails.get(n), n, "; ".join(fails.get(n, [])[:8]))
        s2.info["stats"] = {k: dict(v) for k, v in stats.items()}
        s2.info["revision_seeds_sample"] = {f"{r['seed']}|{r['task_id']}": {
            "initial": int(hashlib.sha256(f"spr-initial:{r['seed']}:{r['task_id']}".encode()).hexdigest()[:8], 16),
            "revise": int(hashlib.sha256(f"spr-revise:{r['seed']}:{r['task_id']}".encode()).hexdigest()[:8], 16)}
            for r in self.reqs[:3]}
        s2.info["identifier_errors_vs_sqlite"] = self.ident_vs_sqlite
        self.code_serve(s2, s3)

    ident_vs_sqlite = Counter()

    def typed_checks(self, r, c, fails, st, label="original"):
        a, ex = c["typed"], c["execution"]
        q = self.rows[r["task_id"]]
        text = f"{q['question']}\n{q['evidence'] or ''}"
        for e in a["errors"]:
            if not any(re.match(p, e) for p in ERROR_PATTERNS):
                fails["typed analysis: errors only of the declared definite kinds"].append(e[:80])
        for w in a["warnings"]:
            if not any(re.match(p, w) for p in WARNING_PATTERNS):
                fails["typed analysis: warnings only of the declared observational kinds"].append(w[:80])
        stx = ex.get("status")
        has_exec_err = any(e.startswith("execution ") for e in a["errors"])
        if has_exec_err != (stx not in ("ok", None)):
            fails["typed analysis: execution errors and timeouts reported iff the execution failed"].append(
                f"{r['task_id']}|{label}")
        empty = any(w.startswith("the query returned no rows") for w in a["warnings"])
        if empty != (stx == "ok" and ex.get("row_count") == 0):
            fails["typed analysis: the empty-result warning iff ok with 0 rows"].append(f"{r['task_id']}|{label}")
        for line in a["plan_lines"]:
            m = re.match(r"^  literal (.*): (occurs|does not occur) verbatim in the question or evidence$", line)
            if m:
                st["literal_lines"] += 1
                if occurs(m.group(1), text) != (m.group(2) == "occurs"):
                    fails["typed analysis: own literal-occurrence test = each literal line"].append(line[:80])
        idr = set(self.F["IDENTIFIER_REASONS"])
        ok = (a["typed"] == (a["reason"] is None)) if c["sql"] else (not a["typed"])
        if c["sql"] and not a["typed"]:
            if a["reason"] in idr:
                ok = ok and any(e.startswith(a["reason"] + " against") for e in a["errors"])
                if label == "original":
                    pop = "dev" if r["seed"] == 900 else "run"
                    self.ident_vs_sqlite[f"{pop}: {a['reason']} -> recorded execution {stx}"] += 1
            else:
                ok = ok and any(w.startswith("the query uses a construct outside") for w in a["warnings"])
        if not ok:
            fails["typed analysis: untyped iff a reason; identifier reasons as errors, others as warnings"].append(
                f"{r['task_id']}|{label}")

    @staticmethod
    def obs_as_record(m):
        return {"status": m["status"], "error": m.get("error"), "rows": m.get("rows", []), "columns": m.get("columns", []),
                "row_count": m.get("row_count"), "truncated": m.get("truncated", False)}

    @staticmethod
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

    @staticmethod
    def count(tok, msgs):
        text = tok.apply_chat_template(msgs, tools=None, add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    def code_serve(self, s2, s3):
        fns = functions(self.spr_src)
        src = self.spr_src.decode()
        serve = ast.get_source_segment(src, fns["serve"])
        flat = re.sub(r"\s+", " ", serve)
        s2.check(flat.count('seed_of("revise", seed, task.task_id)') == 1 and "for policy in order:" in flat and
                 re.search(r"for policy in order:.*_call\(rt, msgs, cfg, seed_of\(\"revise\", seed, task\.task_id\), "
                           r"time\.monotonic\(\) \+ remaining\)", flat),
                 "code: both revisions call with seed_of('revise', seed, task_id) and time.monotonic() + remaining")
        assigns = [n for n in ast.walk(fns["serve"]) if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == "remaining" for t in n.targets)]
        loop = next(n for n in ast.walk(fns["serve"]) if isinstance(n, ast.For) and getattr(n.target, "id", "") == "policy")
        s2.check(len(assigns) == 1 and assigns[0].lineno < loop.lineno,
                 "code: the remaining time is computed once, before the revision loop")
        s2.check('msgs = base + [{"role": "assistant", "content": turn0.content}, {"role": "user", "content": message}]'
                 in flat, "code: both revisions = the shared base messages + the initial reply + their own message")
        s2.check('messages = {"D1": f"{raw}\\n\\n{REVISE}", "TI1": f"{raw}\\n\\n{F.render_typed(analysis0)}\\n\\n{REVISE}"}'
                 in flat, "code: the D1 and TI1 messages differ only by the rendered analysis of the initial proposal")
        s2.check('order = ("D1", "TI1") if seed_of("order", seed, task.task_id) % 2 == 0 else ("TI1", "D1")' in flat,
                 "code: the revision order by seed_of('order', ...) parity")
        s2.check('turn0, call0 = _call(rt, base, cfg, seed_of("initial", seed, task.task_id), deadline)' in flat,
                 "code: one initial call per question and seed, seed_of('initial', ...)")
        sel = ast.get_source_segment(src, fns["select"]) + ast.get_source_segment(src, fns["_valid"]) + \
            ast.get_source_segment(src, fns["_execute"])
        s3.check("gold" not in sel.lower() and "scorer" not in sel, "code: select, _valid and _execute never read "
                 "gold or the scorer")
        s3.check(re.search(r"if _valid\(revised\[\"sql\"\], revised\[\"execution\"\]\):.*return \{\"source\": "
                           r"\"revised\".*if _valid\(original\[\"sql\"\], original\[\"execution\"\]\):.*"
                           r"\"original\".*\"abstain\"", re.sub(r"\s+", " ", sel)) and
                 "check_read_only(sql) is None and execution[\"status\"] == \"ok\"" in sel,
                 "code: the rule is revised if valid, else original if valid, else abstain; valid = SQL, read-only, ok")

    # -- 4. scores --
    def scores_section(self):
        s = self.sec("4_scores")
        # snapshots: own hash of every file used
        used = {r["db_id"] for r in self.all_requests()} | {r["db_id"] for r in self.pres}
        bad = {}
        self.own_snap_sha = {}
        for d in sorted(used):
            h = sha_f(self.snap[d]["path"])
            self.own_snap_sha[d] = h
            if h != self.snap[d]["sha256"] or os.path.getsize(self.snap[d]["path"]) != self.snap[d]["bytes"]:
                bad[d] = h
        s.check(not bad, f"own sha256 and size of the {len(used)} snapshot files = the snapshots pin", str(bad))
        mapping = []
        fails = defaultdict(list)
        sets = [("run", self.reqs, self.scores)] + [(d, self.dev[d]["reqs"], self.dev[d]["scores"]) for d in DEVS]
        self.correct = {}
        for tag, reqs, scores in sets:
            byreq = {(r["seed"], r["task_id"]): r for r in reqs}
            for x in scores:
                r = byreq[(x["seed"], x["task_id"])]
                k = f"{tag}|{x['seed']}|{x['task_id']}|{x['policy']}"
                if (x["source"], x["sql"]) != (r[x["policy"]]["source"], r[x["policy"]]["sql"]):
                    fails["scored SQL = the request's final SQL"].append(k)
                if (x["source"], x["sql"]) != self.sel[(tag, x["seed"], x["task_id"], x["policy"])]:
                    fails["scored SQL = own selection"].append(k)
                mine = self.ex[(x["task_id"], x["sql"])]
                sc = x["score"]
                if sc["snapshot_sha256"] != self.own_snap_sha[x["db_id"]]:
                    fails["score's snapshot sha256 = own hash"].append(k)
                if sc["correct"] != mine["correct"]:
                    fails["recorded EX = own EX"].append(f"{k}: {sc['correct']} vs {mine['correct']}")
                if sc["status"] != mine["status"]:
                    fails["recorded status = own status"].append(f"{k}: {sc['status']} vs {mine['status']}")
                pst = sc["predicted"]["status"]
                if pst != mine["pred"]:
                    fails["recorded predicted status = own"].append(k)
                if mine["status"] not in ("abstain",) and (sc["gold"]["status"] != mine["gold"] or
                                                           sc["gold"]["row_count"] != mine["gold_count"] or
                                                           sc["predicted"]["row_count"] != mine["pred_count"]):
                    fails["recorded gold status and row counts = own"].append(k)
                self.correct[(tag, x["policy"], x["seed"], x["task_id"])] = mine["correct"]
                mapping.append({"run": tag, "seed": x["seed"], "task_id": x["task_id"], "policy": x["policy"],
                                "db_id": x["db_id"], "snapshot_sha256": self.own_snap_sha[x["db_id"]],
                                "sql_sha256": None if x["sql"] is None else sha_t(x["sql"]),
                                "gold_sha256": sha_t(self.rows[x["task_id"]]["SQL"]), "source": x["source"],
                                "recorded_correct": sc["correct"], "own_correct": mine["correct"],
                                "recorded_status": sc["status"], "own_status": mine["status"]})
        for n in ["scored SQL = the request's final SQL", "scored SQL = own selection", "score's snapshot sha256 = own hash",
                  "recorded EX = own EX", "recorded status = own status", "recorded predicted status = own",
                  "recorded gold status and row counts = own"]:
            s.check(not fails.get(n), n + f" ({len(mapping)} records: run 384, dev 96)", "; ".join(fails.get(n, [])[:8]))
        (self.out / "score_mapping.jsonl").write_text("".join(json.dumps(m, sort_keys=True) + "\n" for m in mapping))
        s.info["records_mapped"] = len(mapping)
        s.info["own_status_counts_run"] = dict(Counter(m["own_status"] for m in mapping if m["run"] == "run"))
        s.info["reruns"] = {"observe": self.obs_reruns, "ex": self.ex_reruns}
        s.info["execution_seconds"] = round(self.exec_seconds, 1)
        s.info["sqlite_version"] = sqlite3.sqlite_version
        s.info["dev_counts"] = {d: {p: sum(self.correct[(d, p, 900, t)] for t in {r["task_id"] for r in self.dev[d]["reqs"]})
                                    for p in POLICIES} for d in DEVS}

    # -- 5. gold isolation --
    def gold_isolation(self):
        s = self.sec("5_gold_isolation")
        # code: the diagnostic path, transitive within each module
        mods = {"spr_feedback": ["typed_analysis", "raw_observation", "render_typed"],
                "spr_typed": ["schema_index", "parse"],
                "diagnose": ["all_selects", "select_features", "Scope"],
                "sql_to_ir": ["try_convert"], "ir": ["validate", "compile", "to_json", "from_json", "SchemaIndex"],
                "schema": ["load_schema", "serialize"], "tools": ["execute_select", "check_read_only"],
                "gi_llm": ["messages", "config"], "llm_agent": ["user_message", "extract_final"],
                "spr_run": ["serve", "select", "_valid", "_execute", "_call", "seed_of"]}
        found = {}
        for m, roots in mods.items():
            src = git_show(self.repo, FROZEN, H + f"{m}.py")
            fns = functions(src)
            reach = closure(fns, roots)
            text = src.decode()
            bad = []
            for f in sorted(reach):
                node = fns[f]
                seg = ast.get_source_segment(text, node) or ""
                code_only = "\n".join(ln for ln in seg.splitlines())
                ids = idents(node)
                if {"gold", "Gold", "scorer", "execute_gold", "gold_check"} & ids or re.search(r"\.gold\(", code_only):
                    bad.append(f)
            found[m] = {"reachable": sorted(reach), "touching_gold": bad}
            # imports at module level that run code
            body_calls = [n for n in ast.parse(src).body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
            if body_calls:
                bad.append("<module-level calls>")
        s.check(all(not v["touching_gold"] for v in found.values()),
                "code: no function reachable from the diagnostic and serving path names gold, Gold or the scorer",
                json.dumps({m: v["touching_gold"] for m, v in found.items() if v["touching_gold"]}))
        s.info["reachable"] = {m: len(v["reachable"]) for m, v in found.items()}
        # _serve_population: gold read only after the reservation's with block (all requests served)
        fns = functions(self.spr_src)
        sp = fns["_serve_population"]
        with_node = next(n for n in ast.walk(sp) if isinstance(n, ast.With))
        gold_lines = [n.lineno for n in ast.walk(sp) if isinstance(n, ast.Attribute) and n.attr == "gold"]
        within = [ln for ln in gold_lines if with_node.lineno <= ln <= with_node.end_lineno]
        s.check(gold_lines and not within and min(gold_lines) > with_node.end_lineno,
                "code: _serve_population reads gold only after every request is served and written",
                f"gold at lines {gold_lines}, serving block {with_node.lineno}-{with_node.end_lineno}")
        rq_m, sc_m = os.stat(self.run_dir / "requests.jsonl").st_mtime, os.stat(self.run_dir / "scores.jsonl").st_mtime
        s.info["requests_mtime"], s.info["scores_mtime"] = iso(rq_m), iso(sc_m)
        # empirical
        evaluated = set(self.new_ids) | {r["task_id"] for d in DEVS for r in self.dev[d]["reqs"]}
        idpat = re.compile(r"bird23tf-\d{5}")
        norm = lambda t: re.sub(r"\s+", " ", t).strip().lower().rstrip(";").strip()  # noqa: E731
        fails = defaultdict(list)
        explained = []
        texts_checked = 0
        for tag, reqs in [("run", self.reqs)] + [(d, self.dev[d]["reqs"]) for d in DEVS]:
            for r in reqs:
                k = f"{tag}|{r['seed']}|{r['task_id']}"
                gold = norm(self.rows[r["task_id"]]["SQL"])
                texts = {"D1 message": r["D1"]["revised"]["feedback"], "TI1 message": r["TI1"]["revised"]["feedback"]}
                for name, c in [("original", r["original"])] + [(p, r[p]["revised"]) for p in ("D1", "TI1")]:
                    texts[f"{name} typed analysis"] = json.dumps(c["typed"], ensure_ascii=False)
                if self.W:
                    texts["prompt"] = self.W["prompts"][r["task_id"]]["user"] + "\n" + \
                        self.W["prompts"][r["task_id"]]["system"]
                for name, t in texts.items():
                    texts_checked += 1
                    if gold and gold in norm(t):
                        fails["no gold SQL in any prompt, message or analysis"].append(f"{k}|{name}")
                    named = set(idpat.findall(t))
                    if named & evaluated:
                        fails["no evaluated task id in any prompt, message or analysis"].append(f"{k}|{name}")
                    if named:
                        fails["no task id of any kind in any prompt, message or analysis"].append(f"{k}|{name}")
                # gold result: the gold preview block in a message only when the model's own result is the gold's
                ge = self.ex[(r["task_id"], r["original"]["sql"])] if (r["task_id"], r["original"]["sql"]) in self.ex \
                    else None
                gp = None
                for key in [(r["task_id"], r[p]["sql"]) for p in POLICIES]:
                    if key in self.ex and self.ex[key].get("gold_preview"):
                        gp = self.ex[key]["gold_preview"]
                        break
                if gp:
                    block = "\n".join("  " + " | ".join(("NULL" if v is None else str(v))[:60] +
                                                        ("..." if len("NULL" if v is None else str(v)) > 60 else "")
                                                        for v in row) for row in gp)
                    for name in ("D1 message", "TI1 message"):
                        if block and block in texts[name]:
                            own_same = ge is not None and ge.get("pred_set") == ge.get("gold_set")
                            own_lines = own_raw(self.obs_as_record(self.obs[(r["db_id"], r["original"]["sql"])]))
                            if own_same or block in own_lines:
                                explained.append(f"{k}|{name}")
                            else:
                                fails["a gold result block appears only where the model's own result shows it"].append(
                                    f"{k}|{name}")
        for n in ["no gold SQL in any prompt, message or analysis", "no evaluated task id in any prompt, message or "
                  "analysis", "no task id of any kind in any prompt, message or analysis",
                  "a gold result block appears only where the model's own result shows it"]:
            s.check(not fails.get(n), n, "; ".join(fails.get(n, [])[:8]))
        s.info["texts_checked"] = texts_checked
        s.info["gold_result_block_in_message_explained_by_own_result"] = len(explained)
        s.info["prompts_checked"] = bool(self.W)

    # -- 6. preservation --
    def preservation(self):
        s = self.sec("6_preservation")
        src = self.runs / GI_DIRECT / "records.jsonl"
        s.check(sha_f(src) == self.pres_manifest["config"]["source_sha256"], "source records sha256 = the manifest's")
        srcr = {r["task_id"]: r for r in jl(src)}
        s.check(sorted(srcr) == sorted(self.pins["gi_new_eval"]["task_ids"]) and len(self.pres) == 64 and
                sorted(r["task_id"] for r in self.pres) == sorted(srcr),
                "64 records = the grounding-interface new population's 64 saved direct outputs")
        s.check(all(r["sql"] == srcr[r["task_id"]]["sql"] for r in self.pres), "each record's SQL = the saved output")
        s.check(all(r["snapshot_sha256"] == self.own_snap_sha[r["db_id"]] for r in self.pres),
                "each record's snapshot sha256 = own hash")
        fails = defaultdict(list)
        out = Counter()
        unsup = Counter()
        rows = []
        for r in self.pres:
            k = r["task_id"]
            row = {"task_id": k, "db_id": r["db_id"], "snapshot_sha256": self.own_snap_sha[r["db_id"]],
                   "sql_sha256": sha_t(r["sql"]) if r["sql"] else None,
                   "compiled_sha256": sha_t(r["compiled"]) if r.get("compiled") else None}
            if self.W:
                w = self.W["compile"][k]
                if (w["supported"], w["reason"], w["compiled"]) != (r["supported"], r["reason"], r["compiled"]):
                    fails["compilation recomputed (worker) = recorded"].append(k)
            if not (r["supported"] and r.get("compiled")):
                o = "unsupported"
                unsup[r["reason"]] += 1
                ob = self.obs[(r["db_id"], r["sql"])]
                row["own_original_execution"] = ob["status"]
                unsup[f"sqlite {ob['status']}"] += 1
                eo = self.ex[(k, r["sql"])]
                if eo["correct"] != 0:
                    fails["unsupported outputs: none was EX-correct (own)"].append(k)
            else:
                eo, ec = self.ex[(k, r["sql"])], self.ex[(k, r["compiled"])]
                same_set = eo["pred_set"] is not None and eo["pred_set"] == ec["pred_set"]
                same_list = eo["pred_list"] is not None and eo["pred_list"] == ec["pred_list"]
                both_failed = eo["pred"] == ec["pred"] and eo["pred_count"] is None and ec["pred_count"] is None
                o = "preserved" if eo["correct"] == ec["correct"] and (same_set or both_failed) else "CHANGED"
                out["ex_original"] += eo["correct"]
                out["ex_compiled"] += ec["correct"]
                out["same_set"] += same_set
                out["same_list"] += same_list
                out["both_failed"] += both_failed
                if (eo["correct"], ec["correct"]) != (r["original_correct"], r["compiled_correct"]):
                    fails["own EX of original and compiled = recorded"].append(k)
                if (same_set, same_list) != (r["same_result_set"], r["same_row_list"]):
                    fails["own result-set and row-list comparisons = recorded"].append(k)
                if eo["pred"] != r["original_status"] or ec["pred"] != r["compiled_status"]:
                    fails["own execution statuses = recorded"].append(f"{k}: {eo['pred']}/{ec['pred']} vs "
                                                                     f"{r['original_status']}/{r['compiled_status']}")
                row.update(own_original=eo["status"], own_compiled=ec["status"], same_set=same_set,
                           same_list=same_list)
                gi = srcr[k]["score"]["correct"]
                if eo["correct"] != gi:
                    fails["own EX of the original = the grounding-interface study's recorded score"].append(k)
            out[o] += 1
            row["outcome"] = o
            if o != r["outcome"]:
                fails["own outcome = recorded"].append(f"{k}: {o} vs {r['outcome']}")
            rows.append(row)
        for n in ["own outcome = recorded", "own EX of original and compiled = recorded",
                  "own result-set and row-list comparisons = recorded", "own execution statuses = recorded",
                  "own EX of the original = the grounding-interface study's recorded score",
                  "unsupported outputs: none was EX-correct (own)"] + \
                (["compilation recomputed (worker) = recorded"] if self.W else []):
            s.check(not fails.get(n), n, "; ".join(fails.get(n, [])[:8]))
        s.check(out["preserved"] == 47 and out["CHANGED"] == 0 and out["unsupported"] == 17,
                "47 supported and preserved, 0 unexplained changes, 17 unsupported", json.dumps(out))
        s.check(out["same_set"] == 46 and out["same_list"] == 46 and out["both_failed"] == 1 and
                out["ex_original"] == 32 and out["ex_compiled"] == 32,
                "46 identical result sets and row lists, 1 failing alike both ways, 32 EX-correct either way")
        ident = sum(v for kk, v in unsup.items() if kk in set(self.F["IDENTIFIER_REASONS"]))
        s.check(ident == 16 and unsup.get("function YEAR") == 1, "unsupported: 16 identifier reasons, 1 YEAR",
                json.dumps(dict(unsup)))
        s.check(self.pres_summary["preserved"] == 47 and self.pres_summary["changed"] == [] and
                self.pres_summary["supported"] == 47, "summary.json agrees")
        s.info["own"] = dict(out)
        s.info["unsupported"] = dict(unsup)
        (self.out / "preservation_own.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))

    # -- 7. paired differences --
    def paired(self):
        s = self.sec("7_paired")
        A = self.analysis
        ids, seeds = self.new_ids, (200, 201)
        db_of = {t: self.rows[t]["db_id"] for t in ids}
        c = lambda p, sd, t: self.correct[("run", p, sd, t)]  # noqa: E731
        counts = {f"{p}/{sd}": sum(c(p, sd, t) for t in ids) for p in POLICIES for sd in seeds}
        pooled = {p: sum(counts[f"{p}/{sd}"] for sd in seeds) for p in POLICIES}
        s.check(counts == A["counts"], "own counts per policy and seed = analysis.json", json.dumps(counts))
        s.check(pooled == A["pooled"], "own pooled counts = analysis.json", json.dumps(pooled))
        own = {}
        lead = self.S["LEAD"]
        for a, b in (("TI1", "D1"), ("D1", "D0"), ("TI1", "D0")):
            per_seed = {str(sd): sum(c(a, sd, t) - c(b, sd, t) for t in ids) for sd in seeds}
            tot = sum(per_seed.values())
            per_db = defaultdict(int)
            for t in ids:
                per_db[db_of[t]] += sum(c(a, sd, t) - c(b, sd, t) for sd in seeds)
            loo = {d: tot - v for d, v in sorted(per_db.items())}
            rec = sum(c(a, sd, t) > c(b, sd, t) for sd in seeds for t in ids)
            reg = sum(c(a, sd, t) < c(b, sd, t) for sd in seeds for t in ids)
            hist = Counter(sum(c(a, sd, t) - c(b, sd, t) for sd in seeds) for t in ids)
            lead_ok = tot >= lead and all(v > 0 for v in per_seed.values()) and all(v > 0 for v in loo.values())
            harm = tot <= -lead and all(v < 0 for v in per_seed.values()) and all(v < 0 for v in loo.values())
            own[f"{a}-{b}"] = {"per_seed": per_seed, "pooled": tot, "per_database": dict(sorted(per_db.items())),
                               "leave_one_database_out": loo, "recoveries": rec, "regressions": reg,
                               "histogram": {str(k): v for k, v in sorted(hist.items())}, "lead": lead_ok, "harm": harm}
            d = A["comparisons"][f"{a}-{b}"]
            s.check(d["per_seed"] == per_seed and d["pooled"] == tot and d["leave_one_database_out"] == loo and
                    d["per_database"] == dict(sorted(per_db.items())) and d["recoveries"] == rec and
                    d["regressions"] == reg and d["per_question_histogram"] == {str(k): v for k, v in sorted(hist.items())},
                    f"{a} - {b}: per seed, pooled, per database, leave-one-out, recoveries, regressions, histogram = "
                    "analysis.json", json.dumps(own[f"{a}-{b}"]))
            s.check(rec - reg == tot, f"{a} - {b}: recoveries - regressions = pooled difference")
        if own["TI1-D1"]["lead"]:
            decision = "TI1 beats D1: structured feedback has earned further investigation"
        elif own["TI1-D1"]["harm"] or own["TI1-D0"]["harm"]:
            decision = "TI1 harms correctness: inspect its rejected or revised correct proposals"
        elif own["D1-D0"]["lead"] and own["TI1-D0"]["lead"]:
            decision = "TI1 ties D1 but both beat D0: ordinary revision explains the improvement"
        else:
            decision = "neither revision helps by the rule: retain direct Qwen and stop this repair direction"
        s.check(lead == 8, "the rule's lead is 8 of 128 (frozen spr_run.LEAD)")
        s.check(decision == A["decision"], "own decision (protocol's four branches, +8 rule) = analysis.json", decision)
        # the original's rule fields: the known historical defect (section 11 checks it and the correction)
        thr = {k: v.get("threshold") for k, v in A["comparisons"].items()}
        s.info["original_rule_fields"] = {"threshold": thr, "status": "known historical defect (see 11_correction)"
                                          if all(t == KNOWN_DEFECT_THRESHOLD for t in thr.values()) else "UNKNOWN"}
        s.info["own"] = own
        s.info["decision"] = decision
        # lineage tallies against analysis.json
        L = A["lineage"]
        for p in ("D1", "TI1"):
            src = Counter(r[p]["source"] for r in self.reqs)
            mine = {"final_source": dict(src), "revision_rejected": sum(r[p]["source"] != "revised" for r in self.reqs),
                    "revision_unchanged_sql": sum(r[p]["revised"]["sql"] == r["original"]["sql"] for r in self.reqs),
                    "revision_truncated": sum(r[p]["revised"]["call"]["finish_reason"] != "eos" for r in self.reqs),
                    "revised_typed": sum(bool(r[p]["revised"]["typed"]["typed"]) for r in self.reqs)}
            s.check(mine == L[p], f"lineage tallies {p} = analysis.json", json.dumps(mine))
        ini = {"no_sql": sum(not r["original"]["sql"] for r in self.reqs),
               "truncated": sum(r["original"]["call"]["finish_reason"] != "eos" for r in self.reqs),
               "typed": sum(bool(r["original"]["typed"]["typed"]) for r in self.reqs),
               "untyped_reasons": dict(Counter(r["original"]["typed"]["reason"] for r in self.reqs
                                               if not r["original"]["typed"]["typed"])),
               "execution": dict(Counter(self.obs[(r["db_id"], r["original"]["sql"])]["status"] for r in self.reqs)),
               "with_errors": sum(bool(r["original"]["typed"]["errors"]) for r in self.reqs),
               "with_warnings": sum(bool(r["original"]["typed"]["warnings"]) for r in self.reqs)}
        s.check(ini == A["initial"], "initial-proposal tallies (execution from own runs) = analysis.json", json.dumps(ini))
        fs = {p: dict(Counter(("abstain" if self.sel[("run", r["seed"], r["task_id"], p)][1] is None else
                               self.ex[(r["task_id"], self.sel[("run", r["seed"], r["task_id"], p)][1])]["pred"])
                              for r in self.reqs)) for p in POLICIES}
        s.check(fs == A["final_status"], "final execution statuses (own) = analysis.json", json.dumps(fs))
        bydb = {d: {p: sum(c(p, sd, t) for sd in seeds for t in ids if db_of[t] == d) for p in POLICIES} |
                {"questions": sum(1 for t in ids if db_of[t] == d)} for d in sorted(set(db_of.values()))}
        s.check(bydb == A["by_database"], "by-database counts = analysis.json")
        # D1 vs TI1 invalid-query reduction (engineering benefit, reported)
        s.info["submitted_invalid"] = {p: sum(1 for r in self.reqs if self.sel[("run", r["seed"], r["task_id"], p)][0]
                                              == "abstain" or (self.sel[("run", r["seed"], r["task_id"], p)][1] and
                                              self.ex[(r["task_id"], self.sel[("run", r["seed"], r["task_id"], p)][1])]
                                              ["pred"] != "ok")) for p in POLICIES}

    # -- 8. costs, ledger, co-tenancy --
    def costs(self):
        s = self.sec("8_costs")
        A, R = self.analysis["cost"], self.reqs
        sc = {(x["policy"], x["seed"], x["task_id"]): x for x in self.scores}
        pooled = self.analysis["pooled"]
        for p in POLICIES:
            gen = sum(r["original"]["call"]["seconds"] for r in R) + \
                (sum(r[p]["revised"]["call"]["seconds"] for r in R) if p != "D0" else 0.0)
            tool = sum(r["original"]["execution"]["seconds"] + r[p]["revised"]["execution"]["seconds"] for r in R) \
                if p != "D0" else 0.0
            val = sum(r["original"]["typed"]["seconds"] for r in R) if p == "TI1" else 0.0
            exe = sum(sc[(p, r["seed"], r["task_id"])]["score"]["predicted"]["elapsed_seconds"] for r in R)
            tp = sum(r["original"]["call"]["prompt_tokens"] for r in R) + \
                (sum(r[p]["revised"]["call"]["prompt_tokens"] for r in R) if p != "D0" else 0)
            tg = sum(r["original"]["call"]["generated_tokens"] for r in R) + \
                (sum(r[p]["revised"]["call"]["generated_tokens"] for r in R) if p != "D0" else 0)
            peak = max([r["original"]["call"]["peak_memory_bytes"] for r in R] +
                       ([r[p]["revised"]["call"]["peak_memory_bytes"] for r in R] if p != "D0" else []))
            tot = gen + tool + val + exe
            mine = {"generation_seconds": gen, "validation_seconds": val, "tool_seconds": tool,
                    "submitted_execution_seconds": exe, "total_seconds": tot, "per_request_seconds": tot / len(R),
                    "per_ex_correct_seconds": tot / pooled[p], "prompt_tokens": tp, "generated_tokens": tg,
                    "peak_memory_bytes": peak, "calls_per_request": 1 if p == "D0" else 2}
            close = all(abs(mine[k] - A[p][k]) <= 1e-6 * max(1, abs(A[p][k])) for k in mine)
            s.check(close, f"{p} costs, tokens and peak memory re-derived = analysis.json", json.dumps(mine))
            s.info[p] = mine
        s.info["note_validation"] = ("validation_seconds counts only the initial proposal's typed analysis for TI1; "
                                     "the revised candidates' analyses (diagnostic only, computed for D1 too) are "
                                     "not charged to any policy")
        s.info["revised_typed_seconds_uncharged"] = sum(r[p]["revised"]["typed"]["seconds"] for r in R for p in ("D1", "TI1"))
        # the reservation covers the work it timed
        work = sum(r["original"]["call"]["seconds"] + r["original"]["execution"]["seconds"] + r["original"]["typed"]["seconds"]
                   + sum(r[p]["revised"]["call"]["seconds"] + r[p]["revised"]["execution"]["seconds"] +
                         r[p]["revised"]["typed"]["seconds"] for p in ("D1", "TI1")) for r in R)
        summ = json.loads(rt(self.run_dir / "summary.json"))
        s.info["recorded_work_seconds"] = work
        s.info["load_seconds"] = summ["load_seconds"]
        # ledger
        L = jl(self.runs / "gpu_ledger.jsonl")
        res = {}
        for e in L:
            r = res.setdefault(e["reservation"], {"id": e["reservation"], "start": None, "elapsed": 0.0, "end": None,
                                                  "hb": []})
            if e["event"] == "start":
                r.update(start=e["t"], command=e.get("command"), projected=e.get("projected_hours"),
                         used_before=e.get("used_before_hours"), limit=e.get("limit_hours"), extra=e.get("extra"),
                         foreign=e.get("foreign_gpu_processes"))
            elif e["event"] == "heartbeat":
                r["elapsed"] = max(r["elapsed"], float(e["elapsed"]))
                r["hb"].append(e["t"])
            elif e["event"] == "end":
                r.update(end=e["t"], status=e.get("status"), elapsed=max(r["elapsed"], float(e["elapsed"])),
                         end_foreign=e.get("foreign_gpu_processes"), end_hours=e.get("hours"))
        order = sorted((r for r in res.values() if r["start"] is not None), key=lambda r: r["start"])
        cum = 0.0
        spr = []
        for r in order:
            r["hours"] = (r["elapsed"] if r["end"] is not None else r["elapsed"] + 30.0) / 3600.0
            r["cum_before"] = cum
            cum += r["hours"]
            if (r.get("command") or "").startswith("spr "):
                spr.append(r)
        s.info["ledger_total_hours"] = cum
        s.info["open_reservations"] = [{"id": r["id"], "command": r.get("command"), "charged_hours": r["hours"]}
                                       for r in order if r["end"] is None]
        s.check(len(spr) == 3 and [r["command"] for r in spr] == ["spr dev", "spr dev", "spr run"],
                "three spr reservations: dev, dev, run", json.dumps([r["command"] for r in spr]))
        s.check(abs(spr[0]["cum_before"] - self.S["WORK_ORDER_USED_AT_START"]) < 1e-5,
                "the work order's starting usage (22.822751 h) = own ledger total before the first spr reservation",
                f"{spr[0]['cum_before']:.6f}")
        prev_end = None
        for r in spr:
            nm = f"{r['command']} {r['id']}"
            s.check(abs(r["used_before"] - r["cum_before"]) < 1e-6, f"{nm}: recorded used_before = own cumulative",
                    f"{r['used_before']:.6f} vs {r['cum_before']:.6f}")
            worst = r["cum_before"] + 1.5 * r["projected"]
            s.check(worst <= WORK_ORDER_LIMIT and abs(r["limit"] - worst) < 1e-6,
                    f"{nm}: used + 1.5 x projection ({r['projected']} h) = {worst:.3f} h <= {WORK_ORDER_LIMIT}")
            s.check(r["end"] is not None and r["status"] == "ok" and r["cum_before"] + r["hours"] <= r["limit"],
                    f"{nm}: ended ok within its own limit", f"{r['hours']:.4f} h")
            gaps = [b - a for a, b in zip([r["start"]] + r["hb"], r["hb"] + [r["end"]])]
            s.check(max(gaps) < 45 and abs((r["end"] - r["start"]) - r["elapsed"]) < 5,
                    f"{nm}: heartbeats every ~30 s, wall time = monotonic elapsed", f"max gap {max(gaps):.1f} s")
            overlap = [o["id"] for o in order if o is not r and o["start"] < r["end"] and
                       (o["end"] or o["start"] + o["elapsed"]) > r["start"]]
            s.check(not overlap and not r.get("foreign") and not r.get("end_foreign"),
                    f"{nm}: no overlapping reservation; no foreign GPU process at start or end", str(overlap))
            r_ext = (r.get("extra") or {}).get("run_dir")
            s.info[nm] = {"hours": r["hours"], "projected": r["projected"], "cum_before": r["cum_before"],
                          "run_dir": r_ext, "start": iso(r["start"]), "end": iso(r["end"])}
            prev_end = r
        run_res = spr[2]
        after = run_res["cum_before"] + run_res["hours"]
        s.check(after <= WORK_ORDER_LIMIT, f"cumulative after the frozen run {after:.4f} h <= {WORK_ORDER_LIMIT}")
        s.check(abs(run_res["hours"] - summ["reservation"]["hours"]) < 1e-5 and
                summ["reservation"]["id"] == run_res["id"] and run_res["extra"]["run_dir"] == f"spr/run/{RUN}",
                "the run's summary reservation = the ledger's")
        s.check(summ["load_seconds"] + work <= run_res["elapsed"],
                "load + every recorded call, execution and analysis fits inside the reservation",
                f"{summ['load_seconds'] + work:.1f} s of {run_res['elapsed']:.1f} s")
        gp = {g["id"]: g for g in self.analysis["gpu_reservations"]}
        s.check(all(abs(gp[r["id"]]["hours"] - r["hours"]) < 1e-5 for r in spr) and len(gp) == 3,
                "analysis.json's spr reservations = own")
        s.check(abs(self.analysis["ledger"]["used_hours"] - (spr[2]["cum_before"] + spr[2]["hours"])) < 0.05,
                "analysis.json's ledger total (at analysis time) ~ own total after the run",
                f"{self.analysis['ledger']['used_hours']:.6f} vs {spr[2]['cum_before'] + spr[2]['hours']:.6f}")
        s.check(sum(r["hours"] for r in spr) <= 6.0, "spr reservations total <= the work order's 6 h",
                f"{sum(r['hours'] for r in spr):.4f} h")
        m = self.manifest["config"]["budget"]
        s.check(abs(m["used_hours"] - run_res["cum_before"]) < 1e-6 and m["projected_hours"] == 0.6 and
                m["fits"], "the run manifest's budget check = own", json.dumps(m))
        # co-tenancy
        log = rt(self.spr / "logs" / f"cotenancy_new_{RUN}.log").splitlines()
        samples = []
        for ln in log:
            p = ln.split()
            if len(p) >= 4:
                t = datetime.strptime(p[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
                samples.append((t, float(p[1]), float(p[2]), float(p[3]), p[4] if len(p) > 4 else ""))
        inwin = [x for x in samples if run_res["start"] - 31 <= x[0] <= run_res["end"] + 31]
        ts = [x[0] for x in inwin]
        cover = ts and ts[0] <= run_res["start"] + 31 and ts[-1] >= run_res["end"] - 31 and \
            max(b - a for a, b in zip(ts, ts[1:])) <= 45
        s.check(cover, "the co-tenancy log covers the frozen run every ~30 s", f"{len(inwin)} samples")
        s.info["cotenancy"] = {"samples": len(inwin), "load1_max": max(x[1] for x in inwin),
                               "load1_mean": sum(x[1] for x in inwin) / len(inwin),
                               "running_max": max(int(x[4].split("/")[0]) for x in inwin if "/" in x[4]),
                               "fields": "loadavg 1/5/15 min, running/total tasks (no GPU process list)"}
        s.check(max(x[1] for x in inwin) < 8, "co-tenancy: 1-minute load stayed low (< 8 on 20 cores) during the run",
                f"max {max(x[1] for x in inwin)}")

    # -- 9. freeze order --
    def freeze(self):
        s = self.sec("9_freeze")
        R = self.repo
        pin = self.pins["spr_protocol"]
        proto = git_show(R, FREEZE_AT, H + "docs/PROTOCOL_semantic_proposal_repair.md")
        newpin = git_show(R, FREEZE_AT, H + "pins/spr_new_eval.json")
        s.check(sha_b(proto) == pin["sha256"], "protocol at frozen_at 6e1003f: sha256 = the pin")
        s.check(sha_b(newpin) == pin["new_eval_pin_sha256"], "spr_new_eval.json at 6e1003f: sha256 = the pin")
        HEAD = self.head
        s.check(git_show(R, FROZEN, H + "pins/spr_protocol.json") == git_show(R, HEAD, H + "pins/spr_protocol.json"),
                "the protocol pin is unchanged from 2f76bb7 to the head")
        s.check(sha_b(git_show(R, HEAD, H + "docs/PROTOCOL_semantic_proposal_repair.md")) == pin["sha256"] and
                sha_b(git_show(R, HEAD, H + "pins/spr_new_eval.json")) == pin["new_eval_pin_sha256"],
                "the protocol and the population pin are unchanged at the head")
        s.check(pin["frozen_at"].startswith(FREEZE_AT), "pin frozen_at = 6e1003f")
        changed = git(R, "diff", "--name-only", FROZEN, HEAD, "--", H).split()
        touched = [m for m in RUN_MODULES if H + f"{m}.py" in changed]
        s.check(not touched, f"the {len(RUN_MODULES)} modules the run, scoring and analysis executed are unchanged "
                f"from {FROZEN} to the head {HEAD[:7]}", str(touched))
        s.check(not [c for c in changed if c.startswith(H + "pins/")], "no pin changed from the frozen commit to the head")
        s.info["changed_since_frozen"] = {"spr": sorted(c for c in changed if "spr" in c and "/evidence/" not in c),
                                          "evidence_spr_v1": sum(c.startswith(H + "evidence/spr_v1/") for c in changed),
                                          "other": sorted(c for c in changed if "spr" not in c)[:40]}
        for c in ("03759ac", "d73c30e", FREEZE_AT):
            ok = subprocess.run(["git", "merge-base", "--is-ancestor", c, FROZEN], cwd=R).returncode == 0
            s.check(ok, f"{c} is an ancestor of the frozen commit {FROZEN}")
        ct = {c: int(git(R, "show", "-s", "--format=%ct", c).strip()) for c in ("d73c30e", "03759ac", FREEZE_AT, FROZEN)}
        run_start = utc(RUN)
        s.check(ct[FROZEN] < run_start and ct[FREEZE_AT] < run_start,
                f"the freeze commits ({iso(ct[FREEZE_AT])}, {iso(ct[FROZEN])}) precede the frozen run ({iso(run_start)})")
        s.check(self.manifest["git"]["commit"].startswith(FROZEN) and not self.manifest["git"]["dirty_paths"] and
                not self.manifest["git"]["research_tree_dirty"], "the run's manifest: commit 2f76bb7, clean tree")
        s.check(self.manifest["config"]["protocol"]["sha256"] == pin["sha256"] and
                self.manifest["config"]["prompts"] == "spr-prompts-v2" == self.S["PROMPT_VERSION"] == pin["prompts"],
                "the run's manifest: the pinned protocol and spr-prompts-v2")
        v2 = literals(git_show(R, "03759ac", H + "spr_run.py"), {"REVISE", "PROMPT_VERSION"})
        s.check(v2["REVISE"] == self.S["REVISE"] and v2["PROMPT_VERSION"] == "spr-prompts-v2",
                "the v2 instruction committed at 03759ac = the frozen one")
        s.check(ct["03759ac"] <= utc(DEVS[1]) + 1 and ct["d73c30e"] <= utc(DEVS[0]),
                "each dev run started at or after its commit", f"{iso(ct['d73c30e'])} / {iso(ct['03759ac'])}")
        s.check(utc(DEVS[1]) < ct[FREEZE_AT] and utc(PRESERVE) < ct[FREEZE_AT], "the preservation test and both dev "
                "runs precede the freeze")
        if self.frozen_tree:
            ft = Path(self.frozen_tree)
            s.check(git(ft, "rev-parse", "HEAD").strip().startswith(FROZEN) and
                    not git(ft, "status", "--porcelain", "--", "research/ti_tnn").strip(),
                    "the frozen worktree is at 2f76bb7 with a clean research tree (now)")
        # dev runs: populations
        dev_ids = sorted(self.earlier, key=lambda t: hashlib.sha256(f"{self.S['DEV_SALT']}:{t}".encode()).hexdigest())[:16]
        for d in DEVS:
            m = self.dev[d]["manifest"]
            got = m["config"]["requested"]
            s.check(got == dev_ids and {r["task_id"] for r in self.dev[d]["reqs"]} == set(dev_ids) and
                    len(self.dev[d]["reqs"]) == 16 and all(r["seed"] == 900 for r in self.dev[d]["reqs"]),
                    f"dev run {d}: the 16 own-sampled earlier dev questions at seed 900")
            s.check(not set(got) & set(self.new_ids) and set(got) <= set(self.inv[:128]),
                    f"dev run {d}: within positions 0-127, none of 192-255")
        s.check(self.dev[DEVS[0]]["manifest"]["git"]["commit"].startswith("d73c30e") and
                self.dev[DEVS[1]]["manifest"]["git"]["commit"].startswith("03759ac") and
                self.pres_manifest["git"]["commit"].startswith("d73c30e"),
                "dev run 1 and the preservation test at d73c30e, dev run 2 at 03759ac")
        # the protocol's dev-run claims
        claims = {}
        for d in DEVS:
            rq = self.dev[d]["reqs"]
            claims[d] = {
                "revisions_not_eos": {p: sum(r[p]["revised"]["call"]["finish_reason"] != "eos" for r in rq)
                                      for p in ("D1", "TI1")},
                "revision_generated_mean": sum(r[p]["revised"]["call"]["generated_tokens"] for r in rq
                                               for p in ("D1", "TI1")) / 32,
                "revision_seconds_mean": {p: sum(r[p]["revised"]["call"]["seconds"] for r in rq) / 16 for p in ("D1", "TI1")},
                "initial_seconds_mean": sum(r["original"]["call"]["seconds"] for r in rq) / 16,
                "peak_gb": max(max(r["original"]["call"]["peak_memory_bytes"], r["D1"]["revised"]["call"]["peak_memory_bytes"],
                                   r["TI1"]["revised"]["call"]["peak_memory_bytes"]) for r in rq) / 1e9,
                "ex": self.sections["4_scores"].info["dev_counts"][d]}
        s.info["dev_claims"] = claims
        c1, c2 = claims[DEVS[0]], claims[DEVS[1]]
        s.check(sum(c1["revisions_not_eos"].values()) == 4 and c1["revisions_not_eos"] == {"D1": 3, "TI1": 1},
                "dev run 1: 4 of 32 revisions hit the ceiling (3 D1, 1 TI1)", json.dumps(c1["revisions_not_eos"]))
        s.check(sum(c2["revisions_not_eos"].values()) == 0, "dev run 2: no truncation")
        s.check(c2["ex"] == {"D0": 8, "D1": 8, "TI1": 9}, "dev run 2 EX (own) = 8, 8, 9", json.dumps(c2["ex"]))
        s.check(abs(c2["peak_gb"] - 11.1) < 0.1 and 3.9 <= c2["initial_seconds_mean"] <= 4.3 and
                all(4.0 <= v <= 4.4 for v in c2["revision_seconds_mean"].values()) and 40 <= c2["revision_generated_mean"] <= 60,
                "dev run 2: ~50 generated tokens, 4.1-4.3 s revisions, 4.1 s initial, 11.1 GB", json.dumps(c2))

    def manifest_res_id(self):
        return json.loads(rt(self.run_dir / "summary.json"))["reservation"]["id"]

    # -- 10. the bundle --
    def bundle(self):
        s = self.sec("10_bundle")
        b = self.repo / H / "evidence" / "spr_v1"
        sums = {}
        for ln in rt(b / "SHA256SUMS").splitlines():
            h, p = ln.split(None, 1)
            sums[p.strip()] = h
        bad = [p for p, h in sums.items() if sha_f(b / p) != h]
        s.check(not bad, f"SHA256SUMS: {len(sums)} files verified", str(bad))
        files = sorted(str(p.relative_to(b)) for p in b.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
        s.check(files == sorted(sums), "every bundled file is under SHA256SUMS")
        orig = sorted(str(p.relative_to(self.spr)) for p in self.spr.rglob("*") if p.is_file())
        s.check(sorted(f for f in files if f != "gpu_ledger.jsonl" and not f.startswith("independent/")) == orig,
                "the bundle holds every file of <runs>/spr/ and nothing else (plus the ledger and independent/)")
        ind = {f.split("/", 1)[1]: sums[f] for f in sums if f.startswith("independent/")}
        first = {k: v for k, v in FIRST_REVIEW.items() if k != "verify_spr.py"}
        s.check(all(ind.get(k) == v for k, v in first.items()) and set(ind) == set(first) | {"README.txt"},
                "independent/: the first review's four outputs, byte-identical to that run, plus the producer's README",
                json.dumps(ind))
        s.check(sha_b(git_show(self.repo, "760eef5", H + "independent/verify_spr.py")) == FIRST_REVIEW["verify_spr.py"],
                "the committed verifier at 760eef5 = the first review's script (81935f4b...)")
        diff = [f for f in orig if sha_f(b / f) != sha_f(self.spr / f)]
        s.check(not diff, "every bundled file = its runs-root original", str(diff))
        s.check(sha_f(b / "gpu_ledger.jsonl") == sha_f(self.runs / "gpu_ledger.jsonl"),
                "the bundled ledger = the live ledger (now)")
        tracked = git(self.repo, "ls-files", H + "evidence/spr_v1").split()
        s.check(len(tracked) == len(sums) + 1, "every bundle file is committed", f"{len(tracked)} tracked")

    # -- 11. the correction pass: the known historical defect apart from the corrected analysis --
    def correction(self):
        s = self.sec("11_correction")
        A = self.analysis
        rule_fields = ("threshold", "conditions", "all_conditions", "sign", "rule_version")
        own = self.sections["7_paired"].info["own"]
        orig_path = self.spr / "analysis" / ANALYSIS / "analysis.json"
        bund = self.repo / H / "evidence" / "spr_v1" / "analysis"
        orig_sha = sha_f(orig_path)
        s.check(orig_sha == KNOWN_DEFECT_ANALYSIS_SHA256 and sha_f(bund / ANALYSIS / "analysis.json") == orig_sha,
                "the original analysis.json is kept unchanged (sha256 ceb208f6...) in the runs root and the bundle",
                orig_sha)

        def conds(d, lead):
            sign = 1 if d["pooled"] > 0 else -1 if d["pooled"] < 0 else 0
            return sign, {"at_least": abs(d["pooled"]) >= lead,
                          "same_sign_every_seed": sign != 0 and all(v * sign > 0 for v in d["per_seed"].values()),
                          "same_sign_without_any_one_database": sign != 0 and all(
                              v * sign > 0 for v in d["leave_one_database_out"].values())}
        # the known defect: present, exactly as known (threshold 7 and conditions computed with 7)
        thr = {k: v.get("threshold") for k, v in A["comparisons"].items()}
        as_known = True
        flips = {}
        for k, d in A["comparisons"].items():
            sign7, c7 = conds(own[k], KNOWN_DEFECT_THRESHOLD)
            _s8, c8 = conds(own[k], self.S["LEAD"])
            rec = d.get("conditions", {})
            as_known &= d.get("threshold") == KNOWN_DEFECT_THRESHOLD and rec == {
                "pooled_at_least_threshold": c7["at_least"], "same_sign_every_seed": c7["same_sign_every_seed"],
                "same_sign_without_any_one_database": c7["same_sign_without_any_one_database"]} and \
                d.get("all_conditions") == all(c7.values()) and d.get("sign") == sign7
            flips[k] = c7 != c8
        s.check(as_known, "KNOWN DEFECT (historical, recorded as known, not a new failure): the original's rule fields "
                "still carry threshold 7 with conditions and all_conditions computed at 7, exactly as found by the "
                "first review", json.dumps(thr))
        s.check(not any(flips.values()), "the known defect changes no condition: with pooled +2, +2, +4, every "
                "condition is the same at 7 and at 8", json.dumps(flips))
        self.known_defects.append({
            "id": "spr-analysis-v1-rule-fields", "found_by": "the first review (167 checks, 1 failure, 81935f4b...)",
            "file": f"spr/analysis/{ANALYSIS}/analysis.json", "sha256": orig_sha,
            "fields": "comparisons.*.threshold, conditions, all_conditions (from perf_analysis.paired)",
            "recorded_threshold": thr, "registered_threshold": self.S["LEAD"],
            "status": "present and unchanged, as known" if as_known else "CHANGED FROM THE KNOWN FORM",
            "changes_a_condition_or_the_decision": any(flips.values()),
            "corrected_by": f"spr/analysis/{CORRECTED}/analysis_v2.json (spr-rule-v2)"})
        # the corrected analysis
        cdir = self.spr / "analysis" / CORRECTED
        C2 = json.loads(rt(cdir / "analysis_v2.json"))
        M2 = json.loads(rt(cdir / "manifest.json"))
        s.check(sha_f(cdir / "analysis_v2.json") == sha_f(bund / CORRECTED / "analysis_v2.json") and
                sha_f(cdir / "manifest.json") == sha_f(bund / CORRECTED / "manifest.json"),
                "the corrected analysis and its manifest: runs root = bundle")
        s.check(C2.get("rule_version") == "spr-rule-v2" and all(d.get("rule_version") == "spr-rule-v2"
                                                              for d in C2["comparisons"].values()),
                "the corrected analysis is versioned spr-rule-v2 (top level and every comparison)")
        s.check(all(d.get("threshold") == self.S["LEAD"] == 8 for d in C2["comparisons"].values()),
                "the corrected analysis uses +8 in every comparison's threshold",
                json.dumps({k: d.get("threshold") for k, d in C2["comparisons"].items()}))
        cond_bad, val_bad, rest_bad = [], [], []
        for k, d in C2["comparisons"].items():
            sign8, c8 = conds(own[k], self.S["LEAD"])
            want = {"pooled_at_least_lead": c8["at_least"], "same_sign_every_seed": c8["same_sign_every_seed"],
                    "same_sign_without_any_one_database": c8["same_sign_without_any_one_database"]}
            if d["conditions"] != want or d["all_conditions"] != all(want.values()) or d["sign"] != sign8:
                cond_bad.append(k)
            mine = own[k]
            if (d["per_seed"], d["pooled"], d["per_database"], d["leave_one_database_out"], d["recoveries"],
                    d["regressions"], d["per_question_histogram"]) != (
                    mine["per_seed"], mine["pooled"], mine["per_database"], mine["leave_one_database_out"],
                    mine["recoveries"], mine["regressions"], mine["histogram"]):
                val_bad.append(k)
            o = {f: v for f, v in A["comparisons"][k].items() if f not in rule_fields}
            n = {f: v for f, v in d.items() if f not in rule_fields}
            if o != n:
                rest_bad.append(f"{k}: {sorted(set(o) ^ set(n)) or [f for f in o if o[f] != n.get(f)]}")
        s.check(not cond_bad, "the corrected conditions, all_conditions and sign = own computation at +8 (each "
                "comparison)", str(cond_bad))
        s.check(not val_bad, "the corrected per-seed, pooled, per-database, leave-one-out, recoveries, regressions and "
                "histogram = own", str(val_bad))
        s.check(not rest_bad, "every non-rule field of each corrected comparison = the original's", str(rest_bad))
        own_dec = self.sections["7_paired"].info["decision"]

        def lead(d):
            return d["all_conditions"] and d["sign"] > 0

        def harm(d):
            return d["all_conditions"] and d["sign"] < 0
        cc = C2["comparisons"]
        dec_from_fields = ("TI1 beats D1: structured feedback has earned further investigation" if lead(cc["TI1-D1"])
                           else "TI1 harms correctness: inspect its rejected or revised correct proposals"
                           if harm(cc["TI1-D1"]) or harm(cc["TI1-D0"])
                           else "TI1 ties D1 but both beat D0: ordinary revision explains the improvement"
                           if lead(cc["D1-D0"]) and lead(cc["TI1-D0"])
                           else "neither revision helps by the rule: retain direct Qwen and stop this repair direction")
        s.check(C2["decision"] == A["decision"] == own_dec == dec_from_fields,
                "the corrected decision = the original's = own = the decision implied by the corrected fields", own_dec)
        counts = {f"{p}/{sd}": sum(self.correct[("run", p, sd, t)] for t in self.new_ids) for p in POLICIES
                  for sd in (200, 201)}
        s.check(C2["counts"] == A["counts"] == counts and C2["pooled"] == A["pooled"],
                "the corrected counts and pooled counts = the original's = own", json.dumps(C2["counts"]))
        other = [k for k in set(A) | set(C2) if k not in ("comparisons", "decision", "rule_version", "corrects", "inputs")
                 and A.get(k) != C2.get(k)]
        s.check(not other and set(C2) - set(A) == {"rule_version", "corrects", "inputs"},
                "every other field of the corrected analysis = the original's; it adds only rule_version, corrects, "
                "inputs", str(other))
        sc_sha = sha_f(self.run_dir / "scores.jsonl")
        s.check(C2["corrects"]["analysis_sha256"] == orig_sha and C2["corrects"]["analysis"] == f"spr/analysis/{ANALYSIS}"
                and C2["inputs"]["scores_sha256"] == sc_sha and C2["inputs"]["scores"] == f"spr/run/{RUN}/scores.jsonl",
                "the corrected analysis names the original (sha256) and the frozen run's scores (sha256)")
        s.check(M2["git"]["commit"].startswith(CORRECTION_COMMIT) and not M2["git"]["dirty_paths"] and
                not M2["git"]["research_tree_dirty"] and M2["config"]["rule_version"] == "spr-rule-v2",
                f"the corrected run's manifest: commit {CORRECTION_COMMIT}, clean")
        src_at = git_show(self.repo, CORRECTION_COMMIT, H + "spr_rule.py")
        src_head = git_show(self.repo, self.head, H + "spr_rule.py")
        s.check(M2["sources"].get("spr_rule.py") == sha_b(src_at),
                "the manifest's spr_rule.py hash = the file at 1d042a0", M2["sources"].get("spr_rule.py"))
        s.check(ast.dump(ast.parse(src_at)) == ast.dump(ast.parse(src_head)),
                "spr_rule.py at the head differs from 1d042a0 only in layout (identical syntax trees: the print "
                "re-wrap)", f"{sha_b(src_at)[:12]} -> {sha_b(src_head)[:12]}")
        lit = literals(src_head, {"LEAD", "RULE_VERSION"})
        s.check(lit == {"LEAD": 8, "RULE_VERSION": "spr-rule-v2"}, "spr_rule: LEAD 8, RULE_VERSION spr-rule-v2")
        for path in ("spr_run.py", "perf_analysis.py"):
            s.check(git_show(self.repo, FROZEN, H + path) == git_show(self.repo, self.head, H + path),
                    f"{path} unchanged from the frozen commit to the head")
        s.info["corrected"] = {k: {"threshold": d["threshold"], "conditions": d["conditions"],
                                   "all_conditions": d["all_conditions"]} for k, d in C2["comparisons"].items()}

    # -- 12. the producer's report and tests (producer code, separate processes, flagged) --
    def producer_checks(self):
        s = self.sec("12_producer_report")
        env = {k: v for k, v in os.environ.items() if k not in ("TI_TNN_SCALING_RUNS", "PYTHONPATH")}
        env["CUDA_VISIBLE_DEVICES"] = ""
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "scaling.spr_report"], cwd=self.repo / "research" / "ti_tnn",
                           env=env, capture_output=True, text=True)
        (self.out / "spr_report_output.txt").write_text(p.stdout + p.stderr)
        s.check(p.returncode == 0 and "neither revision helps by the rule" in p.stdout and
                "| TI1-D1 | +0 | +2 | +2 |" in p.stdout,
                "python -m scaling.spr_report (TI_TNN_SCALING_RUNS unset): exit 0, reconstructs, the decision and "
                "TI1 - D1 = +0/+2/+2", f"exit {p.returncode}; {p.stderr[-300:]}")
        s.info["spr_report"] = {"exit": p.returncode, "stdout_sha256": sha_t(p.stdout), "seconds": round(time.time() - t0, 1)}
        t0 = time.time()
        q = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            "tests/test_ti_tnn_scaling_spr.py"], cwd=self.repo, env=env, capture_output=True, text=True)
        (self.out / "spr_tests_output.txt").write_text(q.stdout + q.stderr)
        m = re.search(r"(\d+) passed", q.stdout)
        s.check(q.returncode == 0 and m and not re.search(r"\d+ (failed|skipped|error)", q.stdout),
                "the spr tests pass (none skipped)", q.stdout.strip().splitlines()[-1] if q.stdout.strip() else "")
        s.info["tests"] = {"exit": q.returncode, "summary": q.stdout.strip().splitlines()[-1] if q.stdout.strip() else "",
                           "seconds": round(time.time() - t0, 1)}
        s.info["flag"] = "producer code, run in separate processes; not used for any of this verifier's own verdicts"

    def run(self):
        self.started = time.time()
        self.self_sha_start = sha_b(Path(__file__).resolve().read_bytes())
        before = self.stat_listing()
        t0 = time.time()
        self.head = git(self.repo, "rev-parse", "HEAD").strip()
        self.load()
        self.worker()
        self.executions()
        self.inventories()
        self.lineage()
        self.scores_section()
        self.gold_isolation()
        self.preservation()
        self.paired()
        self.costs()
        self.freeze()
        self.bundle()
        self.correction()
        if self.producer_report:
            self.producer_checks()
        after = self.stat_listing()
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        s = self.sec("0_discipline")
        s.check(not [c for c in changed if c.startswith("spr/") or c == "gpu_ledger.jsonl" or c.startswith("data/")],
                "nothing under spr/, data/ or the ledger changed during this verification", str(changed[:10]))
        s.info["runs_root_changes_elsewhere"] = changed[:50]
        s.info["producer_code"] = {"worker": getattr(self, "worker_info", None),
                                   "use": "schema text and retrieval for the prompts; the typed analysis recomputed; "
                                          "the preservation compile. Compared with the records only."}
        s.info["seconds"] = round(time.time() - t0, 1)
        self_end = sha_b(Path(__file__).resolve().read_bytes())
        s.check(self_end == self.self_sha_start, "the verifier file did not change while it ran")
        s.info["verifier_sha256"] = self.self_sha_start
        s.info["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        s.info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        secs = {k: v.as_dict() for k, v in sorted(self.sections.items())}
        ended = time.time()
        out = {"run": {"verifier": str(Path(__file__).resolve()), "verifier_sha256": self.self_sha_start,
                       "verifier_sha256_how": "sha256 of the script's bytes read by the running process at start "
                                              "(and re-read at the end: equal)",
                       "started_utc": iso(self.started), "ended_utc": iso(ended),
                       "runtime_seconds": round(ended - self.started, 1), "repo_head": self.head,
                       "runs_root": str(self.runs), "python": sys.version.split()[0],
                       "sqlite_version": sqlite3.sqlite_version, "argv": sys.argv[1:]},
               "known_defects": self.known_defects,
               "sections": secs, "checks": sum(len(v["checks"]) for v in secs.values()),
               "failures": sum(v["failures"] for v in secs.values()),
               "inputs": dict(sorted(INPUTS.items()))}
        (self.out / "verify_spr.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
        lines = [f"verify_spr: {out['checks']} checks, {out['failures']} failures",
                 f"verifier sha256 {self.self_sha_start} (computed at run time); head {self.head}",
                 f"started {out['run']['started_utc']}, ended {out['run']['ended_utc']}, runtime "
                 f"{out['run']['runtime_seconds']} s; {len(INPUTS)} inputs hashed (verify_spr.json 'inputs')",
                 "known historical defects (recorded, not failures): " +
                 "; ".join(f"{d['id']}: {d['status']}" for d in self.known_defects)]
        for k, v in secs.items():
            lines.append(f"\n[{k}] {v['passed']} passed, {v['failures']} failed")
            for c in v["checks"]:
                lines.append(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']}" +
                             (f"  -- {c['detail'][:400]}" if c["detail"] else ""))
        (self.out / "verify_spr_summary.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return out


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
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
    ap.add_argument("--producer-report", action="store_true",
                    help="also run `python -m scaling.spr_report` and the spr tests (producer code, separate processes)")
    a = ap.parse_args(argv)
    if not a.runs_root:
        ap.error("--runs-root (or TI_TNN_SCALING_RUNS) is required")
    V(a).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
