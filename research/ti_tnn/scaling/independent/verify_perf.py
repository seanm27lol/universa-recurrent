#!/usr/bin/env python3
"""Independent verifier for the TI-TNN scaling performance study v2 (the reviewer's, written apart from the producers).

    public (a checkout alone; no runs root, databases or checkpoints):
        python3 verify_perf.py --public [--out DIR] [--unchanged-since COMMIT]
    local (the runs root as well):
        taskset -c 0-4,10-14 nice -n 10 python3 verify_perf.py [SECTION ...] --runs-root RUNS [--out DIR] [--no-rehash]
        (or TI_TNN_SCALING_RUNS=RUNS instead of --runs-root; there is no machine default)

The checkout defaults to the one holding this file (--repo); outputs go to ./verify_perf_out unless --out says
otherwise; nothing is ever written under the runs root.

**Public sections** read the pins (pins/*.json), the protocol, git history and the committed bundles
research/ti_tnn/scaling/evidence/{perf_v2, dev_v1, pilot_v1}:
- bundles: each bundle against its SHA256SUMS; nothing listed at --unchanged-since (default 0f2f9b1) changed since,
  the evidence only gained files; the bundled ledgers append-only across bundles.
- pins_public: the protocol pin's sha256s (here and at frozen_at); historical = inventory 0-63 = the pilot's 64,
  new = 64-127, disjoint; perf_v2_train = pilot_train_v1 minus bird23tf-00003; no evaluation id in training.
- ledger: the bundled ledger, parsed and charged by ledger.py's documented rule.
- train: training data and the own sampling stream from the bundled logs and configs, recipes, config differences,
  checkpoint records (not weights), parameter reports, one reservation per fit.
- eval: eligibility from the inventories recomputed out of decodes.json, anchors (pilot_v1 and perf_v2 encode
  entries), records = timed decodes, fresh states, bindings; the recorded scores' internal consistency (statuses,
  executions, correct bit, scored SQL); summaries and serving totals recomputed.
- compare: counts, Q1/Q2 and their rules, leave-one-database-out, sign flips, per-database values, serving, MACs,
  baselines' counts, the pilot's counts, the training and ledger blocks: every leaf of analysis.json except the
  strata (whose classification needs the databases; their marginal sums are checked).
- macs: the executed controller MACs per request, re-derived with this file's own reading of decode_batch.

**Local sections** add what needs the runs root: pins (the inventory rebuilt from the raw release, gold execution),
history (pilot run files and checkpoints), baselines (every final and gold SQL re-executed), and in train, eval and
compare the checkpoint re-hashes and digests, SQL re-execution with this file's own EX rule, and the strata tallies.

Discipline: no producer module is imported. Pins and contracts are read as files or with git show, producer
constants with ast. The EX rule, inventory rule, ledger parser, sampling stream, digests and rules are this file's
own. Exception, flagged in the output: the gold-program strata classification (scaling.perf_analysis.strata), local
mode only, run in a separate process from the frozen worktree. torch (third party) loads checkpoints with
weights_only=True (numpy globals allowlisted, nothing else) and draws the sampling stream.

Output: <out>/verify_perf_<mode>_<sections>.json, {"sections": {name: {"checks": [{name, passed, detail}],
"failures": n, "info": {...}}}, "checks": total, "failures": total}.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = "research/ti_tnn/scaling/"
PIN_DIR = HARNESS + "pins"
CONTRACTS = HARNESS + "contracts.py"
RAW_RELEASE = "data/raw/hf/bird23_train_filtered.train-00000-of-00001.jsonl"  # relative to the runs root
RELEASE_KEY = "data/train-00000-of-00001.jsonl"
DEFAULT_OUT = Path("verify_perf_out")   # relative to the working directory; --out overrides
UNCHANGED_SINCE = "0f2f9b1"              # the bundle state every later commit may only add to (--unchanged-since)
PILOT_HEAD = "9a534b5"            # the pilot PR's head
DROPPED = "bird23tf-00003"         # the one question perf_v2_train drops
HEARTBEAT_SECONDS = 30.0           # ledger.py's documented heartbeat interval (open reservations: last heartbeat + one)
PROJECTION_MARGIN = 0.5            # ledger.py's documented limit: min(cap, used_before + projected x 1.5)
PILOT_ERA_HOURS = 6.878            # stated in the protocol and in ledger.py's docstring
PILOT_ARMS, PILOT_SEEDS, PILOT_UPDATES = ("T4", "c4_params"), (100, 101), 1500
B4_P16 = "20261102T222845Z_B4_thinking_p16-64"
B4_P0 = "20261029T203119Z_B4_thinking_p0-16"
Q27_P0 = "20261029T205933Z_Q27_thinking_low_p0-16"
Q27_P16_GLOB = "*_Q27_thinking_low_p16-64"
HARNESS_REJECTIONS = ("the tool-call budget is exhausted", "the request deadline has passed",
                      "tool-call budget exhausted without a final answer")


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


# --- own helpers (copied from verify_baselines.py, the reviewer's; no producer code) ----------------------------------
def git(repo: Path, *args, check=True) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=check).stdout


def git_show(repo: Path, commit: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=repo, capture_output=True, check=True).stdout


def contract_literals(source: bytes, names=("SPLIT", "INVENTORY", "TRAIN_RELEASE", "OFFICIAL_SCORER")) -> dict:
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_path(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def order_key(salt: str, item: str):
    return (hashlib.sha256(f"{salt}:{item}".encode()).hexdigest(), item)


def jl(path: Path) -> list:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def nonfinite(obj, path="record") -> list:
    if isinstance(obj, float) and not math.isfinite(obj):
        return [path]
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in nonfinite(v, f"{path}.{k}")]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in nonfinite(v, f"{path}[{i}]")]
    return []


def is_count(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


# --- own EX rule (the official mini_dev rule, as in verify_baselines.py) -------------------------------------------------
def connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect("file:" + urllib.parse.quote(str(path.resolve())) + "?mode=ro", uri=True)


def run_sql(path: Path, sqls: list, timeout: float):
    """Execute the statements in order on one read-only connection under one deadline. (rows list | None, status)."""
    conn = connect(path)
    deadline = time.perf_counter() + timeout
    conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 1000)
    try:
        cur, out = conn.cursor(), []
        for sql in sqls:
            cur.execute(sql)
            out.append(cur.fetchall())
        if time.perf_counter() > deadline:
            return None, "timeout"
        return out, "ok"
    except Exception as e:  # the official harness scores any exception 0
        return None, "timeout" if time.perf_counter() > deadline else f"error: {type(e).__name__}: {str(e)[:120]}"
    finally:
        conn.close()


def ex(path: Path, predicted, gold: str, timeout: float):
    """Predicted first, then gold, on one connection under one deadline; set(rows) equality. (correct, status, s)."""
    if predicted is None:
        return 0, "abstain", 0.0
    t0 = time.perf_counter()
    rows, status = run_sql(path, [predicted, gold], timeout)
    dt = time.perf_counter() - t0
    if rows is None:
        return 0, status, dt
    same = set(rows[0]) == set(rows[1])
    return int(same), "correct" if same else "incorrect", dt


class GoldCache:
    """Is a gold query executable within the timeout (the inventory's exclusion rule)? Cached in the output dir."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def executes(self, db: Path, sha: str, task_id: str, gold: str, timeout: float) -> tuple:
        key = f"{task_id}:{sha}:{timeout}"
        if key not in self.data:
            t0 = time.perf_counter()
            _rows, status = run_sql(db, [gold], timeout)
            self.data[key] = {"status": status, "seconds": round(time.perf_counter() - t0, 3)}
            self.path.write_text(json.dumps(self.data, indent=0, sort_keys=True))
        return self.data[key]["status"] == "ok", self.data[key]


# --- the raw release and the inventory rule (own reconstruction) -----------------------------------------------------
class Release:
    def __init__(self, runs: Path, repo: Path, commit: str, cache: GoldCache):
        self.runs, self.repo, self.commit, self.cache = runs, repo, commit, cache
        self.contracts = contract_literals(git_show(repo, commit, CONTRACTS))
        raw = (runs / RAW_RELEASE).read_bytes()
        self.raw_sha256 = sha256_bytes(raw)
        self.raw_ok = self.raw_sha256 == self.contracts["TRAIN_RELEASE"]["files"][RELEASE_KEY]
        lines = raw.decode("utf-8").split("\n")
        lines = lines[:-1] if lines and lines[-1] == "" else lines
        rows = [json.loads(x) for x in lines]
        self.ids = [f"bird23tf-{i:05d}" for i in range(len(rows))]
        self.by_id = dict(zip(self.ids, rows))
        sp = self.contracts["SPLIT"]
        dbs = sorted({r["db_id"] for r in rows}, key=lambda d: order_key(sp["salt"], d))
        n_dev = round(sp["dev_fraction"] * len(dbs))
        self.dev, self.pool = dbs[:n_dev], set(dbs[n_dev:])
        self.snaps = json.loads(git_show(repo, commit, PIN_DIR + "/snapshots_v1.json"))["databases"]
        self.timeout = float(self.contracts["OFFICIAL_SCORER"]["meta_time_out_seconds"])

    def db_path(self, db: str) -> Path:
        return self.runs / "data/databases" / self.snaps[db]["file"]

    def inventory(self, n: int) -> tuple:
        """The first n of the development inventory: within each development database order by
        sha256(salt:task_id), then one question per database per round in split order, skipping a question whose gold
        does not execute within the timeout. Returns (ids, excluded {task_id: info})."""
        salt = self.contracts["INVENTORY"]["salt"]
        queues = {d: sorted((t for t in self.ids if self.by_id[t]["db_id"] == d), key=lambda t: order_key(salt, t))
                  for d in self.dev}
        ptr, order, excluded = {d: 0 for d in self.dev}, [], {}
        while len(order) < n and any(ptr[d] < len(queues[d]) for d in self.dev):
            for d in self.dev:
                while ptr[d] < len(queues[d]):
                    t = queues[d][ptr[d]]
                    ptr[d] += 1
                    ok, info = self.cache.executes(self.db_path(d), self.snaps[d]["sha256"], t, self.by_id[t]["SQL"],
                                                   self.timeout)
                    if ok:
                        order.append(t)
                        break
                    excluded[t] = info
                if len(order) == n:
                    break
        return order, excluded


# === sections ===========================================================================================================
SECTIONS = {}


def section(fn):
    SECTIONS[fn.__name__.removeprefix("sec_")] = fn
    return fn


def pilot_64(repo: Path) -> list:
    """The pilot's 64, from the committed pilot bundle's evaluation manifests (all must agree)."""
    ids = [json.loads(m.read_text())["config"]["ids"]
           for m in sorted((repo / HARNESS / "evidence/pilot_v1/eval").glob("*/manifest.json"))]
    assert ids and all(x == ids[0] for x in ids), "the pilot's evaluation manifests disagree on their ids"
    return ids[0]


_ON_EQ = re.compile(r"""((?:[A-Za-z_][\w]*|`[^`]+`|"[^"]+"|\[[^\]]+\])\s*\.\s*(?:[A-Za-z_][\w]*|`[^`]+`|"[^"]+"|\[[^\]]+\]))"""
                    r"""\s*=\s*"""
                    r"""((?:[A-Za-z_][\w]*|`[^`]+`|"[^"]+"|\[[^\]]+\])\s*\.\s*(?:[A-Za-z_][\w]*|`[^`]+`|"[^"]+"|\[[^\]]+\]))""")


def on_clauses(sql: str) -> list:
    """The text of every ON clause: from each ON keyword to the next JOIN/WHERE/GROUP/ORDER/LIMIT/HAVING/UNION or
    closing parenthesis at depth zero (string literals skipped)."""
    out, low = [], sql
    for m in re.finditer(r"\bON\b", low, flags=re.I):
        i, depth, quote = m.end(), 0, None
        j = i
        while j < len(sql):
            ch = sql[j]
            if quote:
                if ch == quote:
                    quote = None
            elif ch in "'\"`":
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0 and re.match(r"\b(INNER|LEFT|RIGHT|CROSS|FULL|NATURAL|JOIN|WHERE|GROUP|ORDER|LIMIT|HAVING|UNION|"
                                         r"EXCEPT|INTERSECT)\b", sql[j:], flags=re.I) and re.match(r"\W", sql[j - 1]):
                break
            j += 1
        out.append(sql[i:j].strip())
    return out


def mask_string_literals(sql: str) -> str:
    """Replace the inside of every single-quoted string literal ('' escapes included) with underscores, same length."""
    out, i, n = [], 0, len(sql)
    while i < n:
        if sql[i] == "'":
            j = i + 1
            while j < n:
                if sql[j] == "'" and j + 1 < n and sql[j + 1] == "'":
                    j += 2
                    continue
                if sql[j] == "'":
                    break
                j += 1
            out.append("'" + "_" * (min(j, n) - i - 1) + ("'" if j < n else ""))
            i = j + 1
        else:
            out.append(sql[i])
            i += 1
    return "".join(out)


def same_reference_equalities(sql: str) -> list:
    """Every ON-clause equality whose two sides are the same column reference (case, spacing and quoting ignored)."""
    norm = lambda s: re.sub(r"[\s`\"\[\]]", "", s).lower()  # noqa: E731
    found = []
    for clause in on_clauses(mask_string_literals(sql)):
        for a, b in _ON_EQ.findall(clause):
            if norm(a) == norm(b):
                found.append(f"{a.strip()} = {b.strip()}")
    return found


@section
def sec_pins(sec: Section, ctx: dict) -> None:
    repo, runs = ctx["repo"], ctx["runs"]
    pins = repo / PIN_DIR
    proto_pin = json.loads((pins / "perf_v2_protocol.json").read_text())
    ev_bytes, tr_bytes = (pins / "perf_v2_eval.json").read_bytes(), (pins / "perf_v2_train.json").read_bytes()
    ev, tr = json.loads(ev_bytes), json.loads(tr_bytes)
    inv_bytes, ptr_bytes = (pins / "inventory_v1.json").read_bytes(), (pins / "pilot_train_v1.json").read_bytes()
    inv, ptr = json.loads(inv_bytes), json.loads(ptr_bytes)
    frozen = proto_pin["frozen_at"]

    # --- the protocol pin: sha256s equal the files, here and at the freeze commit ----------------------------------
    proto_file = repo / proto_pin["file"]
    sec.check(sha256_path(proto_file) == proto_pin["sha256"], "protocol pin: PROTOCOL_perf_v2.md sha256 = pin",
              f"{sha256_path(proto_file)} vs {proto_pin['sha256']}")
    sec.check(sha256_bytes(tr_bytes) == proto_pin["train_pin_sha256"], "protocol pin: perf_v2_train.json sha256 = pin",
              f"{sha256_bytes(tr_bytes)} vs {proto_pin['train_pin_sha256']}")
    sec.check(sha256_bytes(ev_bytes) == proto_pin["eval_pin_sha256"], "protocol pin: perf_v2_eval.json sha256 = pin",
              f"{sha256_bytes(ev_bytes)} vs {proto_pin['eval_pin_sha256']}")
    at_freeze = {name: sha256_bytes(git_show(repo, frozen, path)) for name, path in
                 (("protocol", proto_pin["file"]), ("train", PIN_DIR + "/perf_v2_train.json"),
                  ("eval", PIN_DIR + "/perf_v2_eval.json"))}
    sec.check(at_freeze == {"protocol": proto_pin["sha256"], "train": proto_pin["train_pin_sha256"],
                            "eval": proto_pin["eval_pin_sha256"]},
              "protocol pin: the three files at frozen_at hash to the pin", f"{at_freeze}")
    sec.check(subprocess.run(["git", "merge-base", "--is-ancestor", frozen, "HEAD"], cwd=repo).returncode == 0,
              "protocol pin: frozen_at is an ancestor of HEAD", frozen)
    since = git(repo, "log", "--format=%h %s", f"{frozen}..HEAD", "--", proto_pin["file"], PIN_DIR + "/perf_v2_train.json",
                PIN_DIR + "/perf_v2_eval.json").strip()
    sec.check(since == "", "protocol, train and eval pins unchanged by any commit after frozen_at", since)
    dirty = git(repo, "status", "--porcelain", "--", proto_pin["file"], PIN_DIR).strip()
    sec.check(dirty == "", "no uncommitted change to the protocol or the pins", dirty)
    sec.check(proto_pin.get("updates") == 3000, "protocol pin: U = 3000", f"{proto_pin.get('updates')}")
    sec.info["protocol_pin"] = proto_pin

    # --- the evaluation pin -------------------------------------------------------------------------------------
    sec.check(sha256_bytes(inv_bytes) == ev["inventory_pin_sha256"], "eval pin names inventory_v1.json by its sha256",
              f"{sha256_bytes(inv_bytes)} vs {ev['inventory_pin_sha256']}")
    hist, new = ev["populations"]["historical"], ev["populations"]["new"]
    sec.check(hist["positions"] == [0, 64] and new["positions"] == [64, 128], "eval pin positions: [0,64) and [64,128)",
              f"{hist['positions']} {new['positions']}")
    sec.check(sorted(ev["populations"]) == ["historical", "new"], "eval pin has exactly two populations")
    sec.check(inv["task_ids"][0:64] == hist["task_ids"], "historical = inventory_v1 positions 0-63, in order")
    sec.check(inv["task_ids"][64:128] == new["task_ids"], "new = inventory_v1 positions 64-127, in order")
    p64 = pilot_64(repo)
    sec.check(hist["task_ids"] == p64, "historical = the pilot's 64 (evidence/pilot_v1 evaluation manifests), in order")
    sec.check(len(hist["task_ids"]) == len(set(hist["task_ids"])) == 64 and len(new["task_ids"]) == len(set(new["task_ids"])) == 64,
              "each population: 64 distinct ids")
    overlap = sorted(set(hist["task_ids"]) & set(new["task_ids"]))
    sec.check(not overlap, "historical and new do not overlap", f"{overlap}")

    # rebuilt from the raw release with the own inventory rule (contracts and snapshots at the freeze commit)
    rel = Release(runs, repo, frozen, ctx["cache"])
    sec.check(rel.raw_ok, "raw release sha256 = contracts.TRAIN_RELEASE at frozen_at", rel.raw_sha256)
    sec.check(inv["salt"] == rel.contracts["INVENTORY"]["salt"], "inventory pin salt = contracts.INVENTORY salt at frozen_at")
    split_sha = sha256_bytes(git_show(repo, frozen, PIN_DIR + "/split_v1.json"))
    sec.check(inv["split_pin_sha256"] == split_sha == ptr["split_pin_sha256"], "inventory and pilot_train pins name split_v1 at frozen_at")
    split = json.loads(git_show(repo, frozen, PIN_DIR + "/split_v1.json"))
    sk = {k: (len(v) if isinstance(v, list) else v) for k, v in split.items()}
    sec.check(list(split.get("dev", [])) == rel.dev and set(split.get("train_pool", [])) == rel.pool,
              "split pin's dev / train_pool databases = own split rule", f"{split.get('dev')} vs {rel.dev}; keys {sk}")
    order, excluded = rel.inventory(128)
    sec.check(order[:64] == hist["task_ids"], "historical rebuilt from the raw release with the own rule = pin",
              f"first difference at {next((i for i, (a, b) in enumerate(zip(order, hist['task_ids'])) if a != b), None)}")
    sec.check(order[64:128] == new["task_ids"], "new rebuilt from the raw release with the own rule = pin",
              f"first difference at {next((i + 64 for i, (a, b) in enumerate(zip(order[64:], new['task_ids'])) if a != b), None)}")
    pin_excl = {e["task_id"]: e for e in inv["excluded"]}
    sec.check(set(excluded) <= set(pin_excl), "every exclusion met while rebuilding is a pinned exclusion", f"{excluded}")
    sec.info["exclusions_met_rebuilding_0_128"] = excluded
    for name, pop in (("historical", hist), ("new", new)):
        dbs = [rel.by_id[t]["db_id"] for t in pop["task_ids"]]
        sec.check(dbs == pop["db_ids"], f"{name}: db_ids = the raw release's", "")
        sec.check(all(d in rel.dev for d in dbs), f"{name}: every question from a development database")

    # --- the training pin --------------------------------------------------------------------------------------------
    sec.check(sha256_bytes(ptr_bytes) == tr["pilot_train_pin_sha256"], "perf_v2_train names pilot_train_v1.json by its sha256")
    expect = [t for t in ptr["task_ids"] if t != DROPPED]
    sec.check(DROPPED in ptr["task_ids"] and len(expect) == len(ptr["task_ids"]) - 1, f"{DROPPED} is in pilot_train_v1 exactly once")
    sec.check(tr["task_ids"] == expect, f"perf_v2_train = pilot_train_v1 minus exactly {DROPPED}, in order",
              f"len {len(tr['task_ids'])} vs {len(expect)}; missing {sorted(set(expect) - set(tr['task_ids']))[:5]}, "
              f"extra {sorted(set(tr['task_ids']) - set(expect))[:5]}")
    sec.check(sorted(tr["dropped"]) == [DROPPED], "perf_v2_train's dropped list is exactly bird23tf-00003", f"{tr['dropped']}")
    sec.check((tr["pilot_questions"], tr["kept"], len(tr["task_ids"]), len(set(tr["task_ids"]))) == (4741, 4740, 4740, 4740),
              "perf_v2_train counts 4741 -> 4740, ids distinct", f"{tr['pilot_questions']} {tr['kept']} {len(tr['task_ids'])}")
    sec.check(all(rel.by_id[t]["db_id"] in rel.pool for t in tr["task_ids"]), "every training id from a training-pool database")
    ev_ids = set(hist["task_ids"]) | set(new["task_ids"])
    leak = sorted(ev_ids & (set(tr["task_ids"]) | set(ptr["task_ids"])))
    sec.check(not leak, "no evaluation id in perf_v2_train or pilot_train_v1", f"{leak}")
    ev_dbs = {rel.by_id[t]["db_id"] for t in ev_ids}
    tr_dbs = {rel.by_id[t]["db_id"] for t in tr["task_ids"]}
    sec.check(not ev_dbs & tr_dbs, "no evaluation database among the training databases", f"{sorted(ev_dbs & tr_dbs)}")

    # bird23tf-00003's gold: a same-reference ON equality, read here
    gold = rel.by_id[DROPPED]["SQL"]
    same = same_reference_equalities(gold)
    sec.check(any(re.sub(r"\s+", "", s).lower() == "t2.list_id=t2.list_id" for s in same),
              f"{DROPPED}: gold SQL has the same-reference ON equality T2.list_id = T2.list_id", f"found {same}; SQL: {gold}")
    sec.info[DROPPED] = {"db_id": rel.by_id[DROPPED]["db_id"], "gold_sql": gold, "on_clauses": on_clauses(gold),
                         "same_reference_equalities": same}
    # every other kept training question: no same-reference ON equality (the reason it alone was dropped)
    others = {t: s for t in tr["task_ids"] if (s := same_reference_equalities(rel.by_id[t]["SQL"]))}
    sec.check(not others, "no kept training question's gold has a same-reference ON equality", f"{list(others.items())[:5]}")
    sec.info["population_dbs"] = {"historical": sorted(set(hist["db_ids"])), "new": sorted(set(new["db_ids"]))}


def read_sums(bundle: Path) -> list:
    out = []
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            h, p = line.split("  ", 1)
            out.append((h, p))
    return out


def tensor_digest(state: dict) -> str:
    """As in verify_pilot.py (the reviewer's): name, dtype and shape as JSON, then the raw bytes, names sorted."""
    import torch
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().to("cpu").contiguous().reshape(-1)
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def safe_load(path: Path):
    """torch.load with weights_only=True, allowlisting numpy globals only (a checkpoint's RNG state is a numpy array).
    Any other global (e.g. a producer class) is refused, never unpickled. Returns (state, allowlisted names)."""
    import importlib
    import torch
    allowed = []
    for _ in range(12):
        try:
            with torch.serialization.safe_globals(allowed):
                return torch.load(path, map_location="cpu", weights_only=True), [f"{o.__module__}.{o.__name__}" for o in allowed]
        except Exception as e:  # noqa: BLE001
            m = re.search(r"Unsupported global: GLOBAL ([\w.]+)", str(e)) or \
                re.search(r"allowlisted via `add_safe_globals`, but got <class '([\w.]+)'>", str(e))
            if not m or not m.group(1).startswith("numpy."):
                raise
            mod, _, name = m.group(1).rpartition(".")
            allowed.append(getattr(importlib.import_module(mod), name))
    raise RuntimeError("too many globals")


@section
def sec_history(sec: Section, ctx: dict) -> None:
    repo, runs = ctx["repo"], ctx["runs"]
    ev = repo / HARNESS / "evidence"
    pilot = runs / "pilot"

    # --- the committed bundles: unchanged since the pilot PR's head ---------------------------------------------------
    sec.check(subprocess.run(["git", "merge-base", "--is-ancestor", PILOT_HEAD, "HEAD"], cwd=repo).returncode == 0,
              f"{PILOT_HEAD} (the pilot PR's head) is an ancestor of HEAD")
    paths = [HARNESS + "evidence/pilot_v1", HARNESS + "evidence/dev_v1"]
    stat = git(repo, "diff", "--stat", PILOT_HEAD, "HEAD", "--", *paths).strip()
    sec.check(stat == "", f"git diff --stat {PILOT_HEAD} HEAD -- evidence/pilot_v1 evidence/dev_v1 is empty", stat)
    porcelain = git(repo, "status", "--porcelain", "--untracked-files=all", "--", *paths).strip()
    sec.check(porcelain == "", "no uncommitted or untracked change in evidence/pilot_v1 or evidence/dev_v1", porcelain)
    for name in ("pilot_v1", "dev_v1"):
        b = ev / name
        sums = read_sums(b)
        bad = [p for h, p in sums if not (b / p).is_file() or sha256_path(b / p) != h]
        listed = {p for _h, p in sums}
        unlisted = sorted(str(x.relative_to(b)) for x in b.rglob("*") if x.is_file() and x.name != "SHA256SUMS"
                          and str(x.relative_to(b)) not in listed)
        sec.check(not bad and not unlisted, f"evidence/{name}: every file matches SHA256SUMS and every file is listed",
                  f"mismatched {bad[:5]}, unlisted {unlisted[:5]}")
        sec.info[f"{name}_files"] = len(sums)

    # --- the pilot's run files = the committed pilot bundle -------------------------------------------------------
    b = ev / "pilot_v1"
    sums = read_sums(b)
    differ, missing, compared, no_original = [], [], 0, []
    for h, p in sums:
        q = p.split("/")
        if p == "gpu_ledger.jsonl":
            continue  # the live ledger is append-only and has grown; checked as a byte prefix below
        if q[0] == "independent":
            no_original.append(p)
            continue
        orig = runs / "baselines" / "/".join(q[1:]) if q[0] == "L" else pilot / p
        if not orig.is_file():
            missing.append(p)
            continue
        compared += 1
        if sha256_path(orig) != h:
            differ.append(p)
    sec.check(not differ, "every bundled pilot file equals its run original (SHA256SUMS)", f"{differ}")
    sec.check(not missing, "every bundled pilot file still has its run original", f"{missing}")
    sec.info["pilot_bundle_compared"] = {"compared_with_originals": compared, "differ": differ, "missing": missing,
                                         "verification_outputs_without_original": no_original}
    led_b, led_live = (b / "gpu_ledger.jsonl").read_bytes(), (runs / "gpu_ledger.jsonl").read_bytes()
    sec.check(led_live.startswith(led_b), "the bundled ledger is a byte prefix of the live ledger (append-only, nothing rewritten)",
              f"bundled {len(led_b)} bytes, live {len(led_live)} bytes")
    sec.info["ledger_bytes"] = {"bundled": len(led_b), "live": len(led_live)}
    bundled = {p for _h, p in sums}
    not_bundled = sorted(str(x.relative_to(pilot)) for x in pilot.rglob("*") if x.is_file() and x.suffix in (".json", ".jsonl")
                         and str(x.relative_to(pilot)) not in bundled)
    sec.info["pilot_json_not_bundled"] = not_bundled
    head_t = int(git(repo, "log", "-1", "--format=%ct", PILOT_HEAD).strip())
    later = sorted(str(x.relative_to(pilot)) for x in pilot.rglob("*") if x.is_file() and x.stat().st_mtime > head_t)
    sec.info["pilot_files_modified_after_pilot_head_commit_time"] = later

    # --- the pilot's final checkpoints -----------------------------------------------------------------------------
    evaluated = {}
    for m in sorted((b / "eval").glob("*/manifest.json")):
        man = json.loads(m.read_text())
        for arm, c in man["config"]["checkpoints"].items():
            evaluated.setdefault((arm, man["config"]["seed"]), set()).add((c["sha256"], c["model_digest"], c["update"]))
    ck_info = {}
    torch = None
    if ctx["rehash"]:
        try:
            import torch  # third party, CPU only, weights_only=True
            torch.set_num_threads(4)
        except Exception as e:  # noqa: BLE001
            sec.info["torch_unavailable"] = repr(e)
    for s in PILOT_SEEDS:
        for arm in PILOT_ARMS:
            rd = pilot / "train" / f"{arm}_s{s}"
            finals = [r for r in jl(rd / "checkpoints.jsonl") if r["update"] == PILOT_UPDATES - 1]
            sec.check(len({r["sha256"] for r in finals}) == 1, f"{arm} s{s}: one final (update 1499) checkpoint recorded",
                      f"{[r['sha256'][:16] for r in finals]}")
            f = finals[-1]
            path = rd / "checkpoints" / Path(f["path"]).name
            sec.check(Path(f["path"]).parent == rd / "checkpoints", f"{arm} s{s}: recorded path is in the run's checkpoints dir", f["path"])
            sec.check(evaluated.get((arm, s)) == {(f["sha256"], f["model_digest"], PILOT_UPDATES - 1)},
                      f"{arm} s{s}: the bundled evaluations name this final checkpoint", f"{evaluated.get((arm, s))}")
            if not ctx["rehash"]:
                continue
            h = sha256_path(path)
            stem = path.name.removesuffix(".pt")
            sec.check(stem == f"ckpt_u{PILOT_UPDATES - 1:06d}_{h[:16]}" and h == f["sha256"],
                      f"{arm} s{s}: final checkpoint re-hashes to its name and record", f"{path.name}: {h}")
            sec.check(path.stat().st_size == f.get("bytes"), f"{arm} s{s}: final checkpoint size = record", f"{path.stat().st_size} vs {f.get('bytes')}")
            info = {"file": path.name, "sha256": h}
            if torch is not None:
                try:
                    ck, allowed = safe_load(path)
                    info["allowlisted_globals"] = allowed
                    md = tensor_digest(ck["model"])
                    sec.check(md == f["model_digest"], f"{arm} s{s}: own model digest = checkpoints.jsonl", f"{md} vs {f['model_digest']}")
                    sec.check(ck.get("update") == PILOT_UPDATES - 1, f"{arm} s{s}: checkpoint's own update = 1499", f"{ck.get('update')}")
                    info["model_digest"] = md
                except Exception as e:  # noqa: BLE001 (weights_only refuses a pickled object: report, do not unpickle)
                    info["model_digest"] = f"not recomputed: {type(e).__name__}: {str(e)[:200]}"
                    sec.check(False, f"{arm} s{s}: own model digest recomputed", info["model_digest"])
            ck_info[f"{arm}/s{s}"] = info
    sec.info["final_checkpoints"] = ck_info


def parse_ledger(path: Path) -> tuple:
    """Own parse of gpu_ledger.jsonl. Returns (reservations in start order, problems). Each reservation:
    start entry, heartbeats, end entry or None; the charge by ledger.py's documented rule."""
    res, problems, order = {}, [], []
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
                problems.append(f"line {n}: second start for {rid}")
                continue
            res[rid] = {"id": rid, "start": e, "beats": [], "end": None, "line": n}
            order.append(rid)
        elif ev in ("heartbeat", "end"):
            r = res.get(rid)
            if r is None:
                problems.append(f"line {n}: {ev} before any start for {rid}")
                continue
            if r["end"] is not None:
                problems.append(f"line {n}: {ev} after the end of {rid}")
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
            problems.append(f"{rid}: heartbeat elapsed not monotone")
        ts = [st["t"]] + [b["t"] for b in beats] + ([end["t"]] if end else [])
        if ts != sorted(ts):
            problems.append(f"{rid}: wall-clock t not monotone across its entries")
        last_beat = el[-1] if el else 0.0
        if end is not None:
            seconds = float(end["elapsed"])
            if seconds < last_beat:
                problems.append(f"{rid}: end elapsed {seconds} below its last heartbeat {last_beat}")
            t_end = end["t"]
        else:
            seconds = last_beat + HEARTBEAT_SECONDS
            t_end = (beats[-1]["t"] if beats else st["t"]) + HEARTBEAT_SECONDS
        out.append({"id": rid, "start": st, "end": end, "beats": len(beats), "last_beat_elapsed": last_beat,
                    "last_beat_t": beats[-1]["t"] if beats else None, "seconds": seconds, "hours": seconds / 3600.0,
                    "t_start": st["t"], "t_end": t_end, "open": end is None})
    return out, problems


@section
def sec_ledger(sec: Section, ctx: dict) -> None:
    path = ledger_file(ctx)
    named = {r["start"].get("ledger") for r in parse_ledger(path)[0]}
    sec.check(len(named) == 1 and str(next(iter(named))).endswith("/gpu_ledger.jsonl"), "every start entry names one and the same ledger",
              f"{named}")
    sec.info["ledger_file"] = str(path)
    sec.info["ledger_named_by_its_entries"] = sorted(map(str, named))
    now = time.time()
    rs, problems = parse_ledger(path)
    sec.check(not problems, "ledger: every line JSON, one start per reservation, nothing after an end, monotone entries",
              f"{problems[:10]}")
    sec.info["reservations"] = len(rs)

    # --- charges and the producer's own accounting -----------------------------------------------------------------
    cum, bad_used, bad_limit, bad_hours, over = 0.0, [], [], [], []
    for r in rs:
        st = r["start"]
        if abs(st.get("used_before_hours", -1) - cum) > 1e-6:
            bad_used.append(f"{r['id']} ({st['command']}): used_before {st.get('used_before_hours')} vs own {cum:.9f}")
        limit = min(st["cap_hours"], st["used_before_hours"] + st["projected_hours"] * (1 + PROJECTION_MARGIN))
        if abs(st.get("limit_hours", -1) - limit) > 1e-9:
            bad_limit.append(f"{r['id']}: limit {st.get('limit_hours')} vs {limit}")
        if r["end"] is not None and abs(r["end"].get("hours", -1) - r["end"]["elapsed"] / 3600.0) > 1e-9:
            bad_hours.append(f"{r['id']}: end hours {r['end'].get('hours')} vs elapsed/3600 {r['end']['elapsed'] / 3600.0}")
        # the watchdog stops at the limit; allow one heartbeat plus the documented 90 s grace
        if st["used_before_hours"] + r["hours"] > st["limit_hours"] + (HEARTBEAT_SECONDS + 90.0) / 3600.0:
            over.append(f"{r['id']} ({st['command']}): used_before + charge {st['used_before_hours'] + r['hours']:.4f} h > limit {st['limit_hours']:.4f} h")
        if not ctx["public"] and st.get("ledger") != str(path):
            bad_used.append(f"{r['id']}: start names ledger {st.get('ledger')}")
        r["cum_before"] = cum
        cum += r["hours"]
    sec.check(not bad_used, "every start's used_before_hours = own sum of earlier charges (and names this ledger)", f"{bad_used[:8]}")
    sec.check(not bad_limit, "every start's limit_hours = min(cap, used_before + projected x 1.5)", f"{bad_limit[:8]}")
    sec.check(not bad_hours, "every end's hours = its elapsed / 3600", f"{bad_hours[:8]}")
    sec.check(not over, "no reservation ran past its limit (one heartbeat + 90 s grace allowed)", f"{over[:8]}")
    total = cum

    # --- authorization versions ------------------------------------------------------------------------------------
    unversioned = [i for i, r in enumerate(rs) if "authorization" not in r["start"]]
    versioned = [i for i, r in enumerate(rs) if "authorization" in r["start"]]
    sec.check(not unversioned or not versioned or max(unversioned) < min(versioned),
              "every unversioned start precedes every versioned one", f"{unversioned[-3:]} {versioned[:3]}")
    bad8 = [(rs[i]["id"], rs[i]["start"]["cap_hours"]) for i in unversioned if rs[i]["start"]["cap_hours"] != 8.0]
    sec.check(not bad8, "every entry before the versioning carries cap_hours 8", f"{bad8}")
    bad32 = [(rs[i]["id"], rs[i]["start"].get("authorization"), rs[i]["start"]["cap_hours"]) for i in versioned
             if (rs[i]["start"].get("authorization"), rs[i]["start"]["cap_hours"]) != (2, 32.0)]
    sec.check(not bad32, "every versioned entry carries authorization 2 and cap_hours 32", f"{bad32}")

    # --- the pilot era and since ------------------------------------------------------------------------------------
    first_v2 = min(versioned) if versioned else len(rs)
    era = rs[:first_v2]
    pilot_cmds = [r for r in era if r["start"]["command"].startswith("pilot-")]
    last_pilot = max((r["t_end"] for r in pilot_cmds), default=None)
    after_pilot_v1 = [r["id"] for r in era if last_pilot is not None and r["t_start"] > last_pilot]
    sec.check(not after_pilot_v1, "no unversioned reservation after the pilot's last reservation", f"{after_pilot_v1}")
    era_hours = sum(r["hours"] for r in era)
    sec.check(round(era_hours, 3) == PILOT_ERA_HOURS, f"pilot-era total (through the end of the pilot) = {PILOT_ERA_HOURS} h",
              f"own {era_hours:.6f} h")
    if versioned:
        ub = rs[first_v2]["start"]["used_before_hours"]
        sec.check(abs(ub - era_hours) < 1e-6, "the first v2 start's used_before_hours = own pilot-era total", f"{ub} vs {era_hours}")
    sec.check(era_hours <= 8.0, "pilot-era total within authorization v1's 8 h", f"{era_hours}")
    sec.check(total <= 32.0, "total within authorization v2's 32 h", f"{total}")
    era_open = [{"id": r["id"], "command": r["start"]["command"], "started": r["start"]["iso"], "beats": r["beats"],
                 "charged_hours": round(r["hours"], 6)} for r in era if r["open"]]
    sec.info["pilot_era"] = {"reservations": len(era), "hours": era_hours, "pilot_commands_hours": sum(r["hours"] for r in pilot_cmds),
                             "never_ended (charged to last heartbeat + 30 s)": era_open,
                             "not_ok": [{"id": r["id"], "command": r["start"]["command"], "status": r["end"]["status"],
                                         "hours": round(r["hours"], 6), "error": (r["end"].get("error") or "")[:200]}
                                        for r in era if r["end"] and r["end"]["status"] != "ok"],
                             "statuses": {s: sum((r["end"] or {}).get("status") == s for r in era)
                                          for s in sorted({(r["end"] or {}).get("status") for r in era if r["end"]})}}
    since = []
    for r in rs[first_v2:]:
        st, end = r["start"], r["end"]
        item = {"reservation": r["id"], "command": st["command"], "started": st["iso"],
                "ended": end["iso"] if end else None, "status": end["status"] if end else "OPEN",
                "hours": round(r["hours"], 6), "projected_hours": st["projected_hours"], "limit_hours": st["limit_hours"],
                "used_before_hours": st["used_before_hours"], "authorization": st.get("authorization"),
                "cap_hours": st["cap_hours"], "run_dir": (st.get("extra") or {}).get("run_dir"), "pid": st.get("pid")}
        if end is None:
            age = now - (r["last_beat_t"] or st["t"])
            alive = Path(f"/proc/{st.get('pid')}").exists()
            item.update(last_heartbeat_age_seconds=round(age, 1), pid_alive=alive,
                        note="open: charged to last heartbeat + 30 s; " + ("live" if age < 120 and alive else "STALE heartbeat or dead pid"))
        if end is not None and end.get("error"):
            item["error"] = end["error"]
        since.append(item)
    sec.info["since_pilot"] = since
    sec.info["totals_hours"] = {"total": total, "pilot_era": era_hours, "since_pilot": total - era_hours,
                                "remaining_under_32": 32.0 - total, "open_reservations": [x["reservation"] for x in since if x["status"] == "OPEN"]}
    stale = [x for x in since if x["status"] == "OPEN" and "STALE" in x["note"]]
    if not ctx["public"]:
        sec.check(not stale, "every open reservation since the pilot is live (recent heartbeat, pid alive)", f"{stale}")
    else:
        sec.info["open_reservations_note"] = "public mode: a bundled ledger is a snapshot; liveness is not checked"

    # --- one GPU owner: no two reservations overlap in wall-clock time --------------------------------------------
    srt = sorted(rs, key=lambda r: r["t_start"])
    overlaps = [f"{a['id']} ({a['start']['command']}, to {iso(a['t_end'])}) overlaps {b['id']} ({b['start']['command']}, from {iso(b['t_start'])})"
                for a, b in zip(srt, srt[1:]) if b["t_start"] < a["t_end"]]
    # also any non-adjacent overlap (a long reservation spanning several)
    for i, a in enumerate(srt):
        for b in srt[i + 2:]:
            if b["t_start"] < a["t_end"]:
                overlaps.append(f"{a['id']} overlaps {b['id']} (non-adjacent)")
    sec.check(not overlaps, "no two reservations overlap in wall-clock time (exactly one GPU owner)", f"{overlaps[:8]}")
    sec.check([r["id"] for r in srt] == [r["id"] for r in rs], "file order = wall-clock start order")
    drift = [f"{r['id']}: wall {r['t_end'] - r['t_start']:.1f} s vs monotonic {r['seconds']:.1f} s" for r in rs
             if not r["open"] and abs((r["t_end"] - r["t_start"]) - r["seconds"]) > 5.0]
    sec.info["wall_vs_monotonic_over_5s"] = drift
    gaps = sorted(((b["t_start"] - a["t_end"]), a["id"], b["id"]) for a, b in zip(srt, srt[1:]))[:3]
    sec.info["smallest_gaps_seconds"] = [(round(g, 1), x, y) for g, x, y in gaps]


# --- baselines -------------------------------------------------------------------------------------------------------
AGENT_SOURCES = ("llm_agent.py", "tools.py", "schema.py", "retrieval.py", "runtime.py", "scorer.py", "contracts.py",
                 "records.py", "data.py")   # what decides an LLM baseline's behaviour and its record


QWEN_PATH_SOURCES = ("llm_agent.py", "runtime.py", "tools.py", "retrieval.py", "schema.py", "scorer.py", "data.py",
                     "contracts.py")   # must be byte-identical between a new-population run and its 16-63 reference


def verify_llm_run(sec: Section, ctx: dict, run_dir: Path, ref_dir: Path, tag: str, expect=(16, 64), ref_expect=(0, 16),
                   strict_sources=()) -> dict:
    """One LLM baseline run (B4 or Q27) against its earlier configuration. Returns the per-question table."""
    repo, runs = ctx["repo"], ctx["runs"]
    man = json.loads((run_dir / "manifest.json").read_text())
    ref = json.loads((ref_dir / "manifest.json").read_text())
    recs = jl(run_dir / "records.jsonl")
    summary = json.loads((run_dir / "summary.json").read_text())
    cfg, commit = man["config"], man["git"]["commit"]
    arm, limits, mode = cfg["arm"], cfg["limits"], cfg.get("mode")
    c = lambda ok, name, detail="": sec.check(ok, f"{tag}: {name}", detail)  # noqa: E731
    warnings = []

    # 1. identity
    c(man["git"].get("research_tree_dirty") is False and not man["git"].get("dirty_paths"), "research tree clean at the run's commit",
      f"{man['git'].get('research_tree_dirty')} {man['git'].get('dirty_paths')}")
    c(subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=repo).returncode == 0,
      "the run's commit is an ancestor of HEAD", commit)
    cid = sha256_bytes(canonical(cfg).encode())
    c(cid == man["config_identity"], "config hashes to config_identity (own canonical JSON)", f"{cid} vs {man['config_identity']}")
    bad = [r["task_id"] for r in recs if r.get("config_identity") != man["config_identity"] or r.get("arm") != arm]
    c(not bad, "every record carries the manifest's config_identity and arm", f"{bad[:5]}")
    for name in ("inventory_v1", "snapshots_v1"):
        c(sha256_bytes(git_show(repo, commit, f"{PIN_DIR}/{name}.json")) == cfg.get(f"{name.split('_')[0]}_pin_sha256"),
          f"{name} pin hash (config vs the pin at the run's commit)")
    mi = man.get("model_identity") or {}
    c(mi.get("pin_sha256") == sha256_bytes(git_show(repo, commit, f"{PIN_DIR}/models_v1.json")),
      "model identity checked against models_v1.json at the run's commit", f"{mi.get('pin_sha256')}")
    models_pin = json.loads(git_show(repo, commit, f"{PIN_DIR}/models_v1.json"))
    c((mi.get("repo"), mi.get("revision")) == (cfg["model"]["repo"], cfg["model"]["revision"]), "model identity = configured model")
    pin_entry = json.dumps(models_pin)
    c(cfg["model"]["revision"] in pin_entry and cfg["model"]["repo"] in pin_entry, "configured model repo and revision are in models_v1.json")

    # 2. the configuration equals the earlier run's except positions and requested ids
    diff = sorted(k for k in set(cfg) | set(ref["config"]) if cfg.get(k) != ref["config"].get(k))
    c(diff == ["positions", "requested"], f"configuration = {ref_dir.name}'s except positions and requested", f"differs in {diff}")
    for k in ("mode", "limits", "generation", "model", "reasoning_effort", "dtype", "linear_attention", "retrieval", "scorer",
              "schema_text", "schema_budget_chars", "seed_rule", "warm_up_task", "runtime"):
        c(cfg.get(k) == ref["config"].get(k), f"{k} equal to the earlier run's", f"{cfg.get(k)} vs {ref['config'].get(k)}")
    c(mi == ref.get("model_identity"), "model_identity (every file hash) equal to the earlier run's")
    env_diff = {k: (ref["environment"].get(k), man["environment"].get(k)) for k in set(man["environment"]) | set(ref["environment"])
                if man["environment"].get(k) != ref["environment"].get(k)}
    c(not env_diff, "software environment equal to the earlier run's", f"{env_diff}")
    src_diff = sorted(k for k in set(man["sources"]) | set(ref["sources"]) if man["sources"].get(k) != ref["sources"].get(k))
    agent_diff = [k for k in src_diff if k in AGENT_SOURCES]
    # a changed agent source is accepted only if the change is the placement of `import torch` (behaviour-neutral)
    neutral = []
    for k in agent_diff:
        a = git_show(repo, ref["git"]["commit"], HARNESS + k).decode().splitlines()
        b = git_show(repo, commit, HARNESS + k).decode().splitlines()
        strip = lambda ls: [x for x in ls if x.strip() != "import torch"]  # noqa: E731
        if strip(a) == strip(b) and sha256_bytes(git_show(repo, commit, HARNESS + k)) == man["sources"][k] \
                and sha256_bytes(git_show(repo, ref["git"]["commit"], HARNESS + k)) == ref["sources"][k]:
            neutral.append(k)
    if strict_sources:
        byte = {k: (sha256_bytes(git_show(repo, ref["git"]["commit"], HARNESS + k)), sha256_bytes(git_show(repo, commit, HARNESS + k)),
                    ref["sources"].get(k), man["sources"].get(k)) for k in strict_sources}
        bad = {k: v for k, v in byte.items() if len(set(v)) != 1}
        c(not bad, f"Qwen-path sources byte-identical to the earlier run's ({', '.join(strict_sources)}), at both commits and in both manifests",
          f"{bad}")
        sec.info[f"{tag}_qwen_path_sha256"] = {k: v[0] for k, v in byte.items()}
    else:
        c(set(agent_diff) == set(neutral), "agent/scorer sources equal to the earlier run's, or differing only in where `import torch` sits",
          f"changed: {agent_diff}; import-placement-only: {neutral}")
    sec.info[f"{tag}_sources_changed_since_{ref_dir.name}"] = {"all": src_diff, "agent_relevant": agent_diff,
                                                               "agent_changes_only_import_torch_placement": neutral,
                                                               "commits": [ref["git"]["commit"], commit]}

    # 3. requests: rebuilt from the raw release with the own rule (contracts at the run's commit)
    rel = Release(runs, repo, commit, ctx["cache"])
    c(rel.raw_ok, "raw release sha256 = contracts at the run's commit")
    lo, hi = cfg["positions"]
    rlo, rhi = ref["config"]["positions"]
    order, _excl = rel.inventory(max(hi, rhi))
    c((lo, hi) == tuple(expect) and (rlo, rhi) == tuple(ref_expect), f"positions: this run {list(expect)}, the earlier run {list(ref_expect)}",
      f"{(lo, hi)} {(rlo, rhi)}")
    c(cfg["requested"] == order[lo:hi], f"requested = own inventory rebuild at [{lo},{hi})")
    c(ref["config"]["requested"] == order[rlo:rhi], f"the earlier run's requested = own rebuild at [{rlo},{rhi})")
    ev = json.loads((repo / PIN_DIR / "perf_v2_eval.json").read_text())
    if (lo, hi) == (16, 64):
        c(ref["config"]["requested"] + cfg["requested"] == ev["populations"]["historical"]["task_ids"],
          "the two runs together request exactly the historical 64 of perf_v2_eval, in order")
    if (lo, hi) == (64, 128):
        c(cfg["requested"] == ev["populations"]["new"]["task_ids"], "requested = the new 64 of perf_v2_eval, in order")
    ids = [r["task_id"] for r in recs]
    c(sorted(ids) == sorted(cfg["requested"]) and len(set(ids)) == len(ids), "every requested task exactly once, nothing else",
      f"missing {sorted(set(cfg['requested']) - set(ids))}, extra {sorted(set(ids) - set(cfg['requested']))}")
    wt = cfg.get("warm_up_task")
    c(wt in rel.by_id and rel.by_id[wt]["db_id"] in rel.pool, "warm-up task is a training-pool question (excluded from records)", f"{wt}")

    # 4. snapshots
    unbound = [r["task_id"] for r in recs
               if not (rel.by_id[r["task_id"]]["db_id"] == r["db_id"] and r.get("snapshot_sha256") == rel.snaps[r["db_id"]]["sha256"]
                       and (r.get("score") or {}).get("snapshot_sha256") == rel.snaps[r["db_id"]]["sha256"]
                       and all(((x.get("result") or {}).get("snapshot_sha256") in (None, rel.snaps[r["db_id"]]["sha256"]))
                               for a in r["attempts"] for x in a.get("tool_results", [])))]
    c(not unbound, "every record's db_id is the release's, and its snapshot, score and tool calls are on the pinned snapshot", f"{unbound}")
    if ctx["rehash"]:
        for d in sorted({r["db_id"] for r in recs}):
            h = sha256_path(rel.db_path(d))
            c(h == rel.snaps[d]["sha256"], f"{d}: database file re-hashes to the pin", h)

    # 5. scores re-derived with the own EX rule
    table, mism = [], []
    for r in recs:
        t, f = r["task_id"], r["final"]
        pred = f.get("sql") if f.get("kind") == "sql" else None
        mine, status, secs = ex(rel.db_path(r["db_id"]), pred, rel.by_id[t]["SQL"], rel.timeout)
        sc = r.get("score") or {}
        timing = status == "timeout" or secs > 0.8 * rel.timeout or sc.get("status") == "timeout"
        if sc.get("correct") != mine or sc.get("status") != status:
            (warnings if timing else mism).append(f"{t}: recorded {sc.get('correct')}/{sc.get('status')}, own {mine}/{status} ({secs:.2f} s)")
        if sc.get("scored_sql_sha256") != sha256_bytes((pred or "").encode()):
            mism.append(f"{t}: scored_sql_sha256 is not the submitted SQL's")
        table.append({"position": cfg["requested"].index(t) + lo, "task_id": t, "db_id": r["db_id"], "final": f.get("kind"),
                      "reason": f.get("reason"), "recorded_correct": sc.get("correct"), "own_correct": mine, "own_status": status,
                      "own_exec_seconds": round(secs, 3), "elapsed_seconds": round(r["elapsed_seconds"], 1),
                      "tool_calls": r["tool_calls"], "repairs": r["repairs"], "turns": len(r["attempts"]),
                      "generated_tokens": (r.get("tokens") or {}).get("generated"), "timed_out": r.get("timed_out")})
    c(not mism, "every recorded score (and status) re-derived by executing final and gold SQL with the own EX rule", f"{mism}")
    table.sort(key=lambda x: x["position"])

    # 6. record rules
    max_calls, max_rep = limits["max_tool_calls"], limits["max_repairs_llm"]
    per_turn = limits["max_new_tokens_thinking" if mode == "thinking" else "max_new_tokens_nonthinking"]
    per_req = limits["max_generated_tokens_thinking" if mode == "thinking" else "max_generated_tokens_nonthinking"]
    tmo = limits["request_timeout_seconds_27b" if arm == "Q27" else "request_timeout_seconds_4b"]
    viol = []
    for r in recs:
        t, f, atts = r["task_id"], r["final"], r["attempts"]
        v = lambda ok, what: ok or viol.append(f"{t}: {what}")  # noqa: E731
        v(not nonfinite(r), f"non-finite numbers {nonfinite(r)[:3]}")
        v(isinstance(r["elapsed_seconds"], (int, float)) and r["elapsed_seconds"] > 0, "elapsed not positive")
        v(f.get("kind") in ("sql", "abstain"), f"final kind {f.get('kind')}")
        v(f.get("kind") != "sql" or (isinstance(f.get("sql"), str) and f["sql"].strip()), "an answer without SQL")
        v(f.get("kind") != "abstain" or bool(f.get("reason")), "an abstention without a reason")
        v(r.get("error") is None or f.get("kind") == "abstain", "an error that did not abstain")
        v(isinstance(r.get("timed_out"), bool) and r["timed_out"] == (f.get("reason") == "request timeout"), "timed_out vs reason")
        for e in r.get("retrieved") or []:
            v(rel.by_id[e]["db_id"] in rel.pool and rel.by_id[e]["db_id"] != r["db_id"] and e != t, f"retrieved {e} not a training-pool example of another db")
        executed, used, budget_hit_at = 0, 0, None
        for i, a in enumerate(atts):
            v(a.get("turn") == i, f"turn {i} numbered {a.get('turn')}")
            res = a.get("tool_results", [])
            ran = [x for x in res if (x.get("result") or {}).get("status") != "not_run"
                   and not str((x.get("result") or {}).get("error", "")).startswith(HARNESS_REJECTIONS)]
            v(not (used >= max_calls and ran), f"turn {i} executed a tool after the budget was spent")
            v(not (used >= max_calls and res and i != len(atts) - 1), f"turn {i} called tools after the budget and the request went on")
            if a.get("kind") == "finalize":
                v(budget_hit_at == i - 1, f"turn {i} labelled finalize does not follow the turn that spent the budget")
            elif budget_hit_at == i - 1 and i > 0:
                v(False, f"turn {i} after the budget was spent is not labelled finalize")
            before = used
            used += len(ran)
            executed += len(ran)
            if before < max_calls <= used:
                budget_hit_at = i
            if a.get("final_status") not in (None, "ok") and i + 1 < len(atts):
                v(atts[i + 1].get("kind") == "repair", f"failed final check at turn {i} not followed by a labelled repair")
            calls = (a.get("generation") or {}).get("tool_calls") or []
            v(len(calls) == len(res), f"turn {i}: {len(calls)} tool calls generated, {len(res)} results")
        repairs = sum(a.get("kind") == "repair" for a in atts)
        v(r["tool_calls"] == executed and executed <= max_calls, f"tool_calls {r['tool_calls']} vs executed {executed} (<= {max_calls})")
        v(r["repairs"] == repairs and repairs <= max_rep, f"repairs {r['repairs']} vs {repairs} (<= {max_rep})")
        v(len(atts) <= max_calls + repairs + 1, f"{len(atts)} turns > bound {max_calls + repairs + 1}")
        if f.get("kind") == "sql":
            v(bool(atts) and atts[-1].get("final_sql") == f["sql"] and atts[-1].get("final_status") == "ok",
              "final SQL is not the last attempt's checked final")
        # abstention reasons, derived from the attempts
        if f.get("kind") == "abstain":
            reason = f.get("reason")
            if reason == "tool-call budget exhausted without a final answer":
                v(executed == max_calls and not any(a.get("final_status") == "ok" for a in atts),
                  f"budget-exhausted abstention with {executed} executed calls / an ok final")
            elif reason == "request timeout":
                v(r["elapsed_seconds"] >= tmo - 5, f"timeout abstention at {r['elapsed_seconds']:.0f} s < {tmo} s")
            elif reason == "turn truncated at the token limit without a final answer":
                last = (atts[-1].get("generation") or {}) if atts else {}
                v(bool(atts) and last.get("finish_reason") == "max_new_tokens" and last.get("generated_tokens") == per_turn
                  and not last.get("tool_calls") and not any(a.get("final_status") == "ok" for a in atts),
                  f"truncation abstention whose last turn did not stop at {per_turn} new tokens without tools or a final")
            else:
                warnings.append(f"{t}: abstention reason {reason!r} not derived here")
        g = [a.get("generation") or {} for a in atts]
        tok = r.get("tokens") or {}
        for k, field in (("generated", "generated_tokens"), ("prefill", "prompt_tokens"), ("reasoning", "reasoning_tokens"),
                         ("content", "content_tokens")):
            v(all(is_count(x.get(field)) for x in g) and tok.get(k) == sum(x[field] for x in g),
              f"tokens.{k} {tok.get(k)} != sum over turns {sum(x.get(field) or 0 for x in g)}")
        v(all(x.get("reasoning_tokens", 0) + x.get("content_tokens", 0) <= x.get("generated_tokens", 0) for x in g),
          "a turn's reasoning + content tokens exceed its generated tokens")
        if atts:
            v(tok.get("initial_prompt") == g[0].get("prompt_tokens"), "tokens.initial_prompt != first turn's prompt")
        v(all((x.get("generated_tokens") or 0) <= per_turn for x in g), f"a turn over {per_turn} new tokens")
        v((tok.get("generated") or 0) <= per_req, f"request over {per_req} generated tokens")
        v(all((x.get("prompt_tokens") or 0) + (x.get("generated_tokens") or 0) <= limits["max_context_tokens"] for x in g),
          "a turn over the context limit")
        if r["elapsed_seconds"] > tmo + 60:
            warnings.append(f"{t}: elapsed {r['elapsed_seconds']:.0f} s > timeout {tmo} + 60 s")
    c(not viol, "record rules: turn and tool budgets, finalize/repair labelling, token sums and limits, abstention reasons",
      f"{viol[:12]}")

    # 7. summary recomputed
    s = summary.get("summary") or {}
    n, corr = len(cfg["requested"]), sum(x["own_correct"] for x in table)
    el = sum(r["elapsed_seconds"] for r in recs)
    close = lambda a, b: (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9))  # noqa: E731
    mine = {"requested": n, "correct": corr, "answered": sum(r["final"]["kind"] == "sql" for r in recs),
            "abstained": sum(r["final"]["kind"] == "abstain" for r in recs), "errors": sum(r.get("error") is not None for r in recs),
            "timeouts": sum(bool(r.get("timed_out")) for r in recs), "repairs": sum(r["repairs"] for r in recs),
            "tool_calls": sum(r["tool_calls"] for r in recs)}
    d = {k: (s.get(k), v) for k, v in mine.items() if s.get(k) != v}
    c(not d, "summary counts recomputed (requested, correct, answered, abstained, errors, timeouts, repairs, tool calls)", f"{d}")
    c(close(s.get("quality"), corr / n), "quality = correct / requested", f"{s.get('quality')} vs {corr / n}")
    c(close(s.get("elapsed_seconds_total"), el), "elapsed_seconds_total = sum of record elapsed", f"{s.get('elapsed_seconds_total')} vs {el}")
    c(close(s.get("cost_seconds_per_correct"), el / corr if corr else None), "cost_seconds_per_correct = sum elapsed / correct")
    reasons = {}
    for r in recs:
        if r["final"]["kind"] == "abstain":
            reasons[r["final"].get("reason")] = reasons.get(r["final"].get("reason"), 0) + 1
    c(s.get("abstention_reasons") == reasons, "abstention_reasons recomputed", f"{s.get('abstention_reasons')} vs {reasons}")
    tok_sum = {k: sum((r.get("tokens") or {}).get(k, 0) for r in recs) for k in (s.get("tokens") or {})}
    c(s.get("tokens") == tok_sum, "summary token totals = sums over records", f"{s.get('tokens')} vs {tok_sum}")
    c(s.get("peak_memory_bytes_max") == max(r.get("peak_memory_bytes") or 0 for r in recs), "peak_memory_bytes_max = max over records")
    c(s.get("config_identity") == man["config_identity"] and s.get("arm") == arm, "summary names the run's arm and configuration")

    # 8. the GPU reservation behind the run
    rs, _p = parse_ledger(runs / "gpu_ledger.jsonl")
    mine_r = [r for r in rs if (r["start"].get("extra") or {}).get("run_dir") == str(run_dir)]
    if c(len(mine_r) == 1, "exactly one ledger reservation names this run directory", f"{[r['id'] for r in mine_r]}"):
        lr = mine_r[0]
        sr = summary.get("reservation") or {}
        c(sr.get("id") == lr["id"], "summary's reservation id = the ledger's", f"{sr.get('id')} vs {lr['id']}")
        c((lr["start"].get("authorization"), lr["start"]["cap_hours"]) == (2, 32.0), "the run's reservation is under authorization 2 (cap 32)")
        c(lr["end"] is not None and lr["end"]["status"] == "ok", "the run's reservation ended ok", f"{(lr['end'] or {}).get('status')}")
        c(abs((sr.get("hours") or 0) - lr["hours"]) < 60 / 3600, "summary's reservation hours within 60 s of the ledger charge",
          f"{sr.get('hours')} vs {lr['hours']}")
        c(el / 3600 <= lr["hours"], "the records' elapsed time fits inside the reservation", f"{el / 3600:.4f} vs {lr['hours']:.4f} h")
        sec.info[f"{tag}_reservation"] = {"id": lr["id"], "hours": lr["hours"], "projected": lr["start"]["projected_hours"]}

    out = {"run": run_dir.name, "commit": commit, "reference_run": ref_dir.name, "requested": n, "correct": corr,
           "answered": mine["answered"], "abstained": mine["abstained"], "timeouts": mine["timeouts"], "errors": mine["errors"],
           "abstention_reasons": reasons, "quality": corr / n, "elapsed_seconds_total": el,
           "cost_seconds_per_correct": el / corr if corr else None, "warnings": warnings, "per_question": table}
    # the earlier 0-16 run's recorded scores, for the historical 64 total (re-derived as well)
    ref_recs = jl(ref_dir / "records.jsonl")
    ref_mism, ref_corr = [], 0
    for r in ref_recs:
        pred = r["final"].get("sql") if r["final"].get("kind") == "sql" else None
        m, st, secs = ex(rel.db_path(r["db_id"]), pred, rel.by_id[r["task_id"]]["SQL"], rel.timeout)
        ref_corr += m
        if m != (r.get("score") or {}).get("correct"):
            ref_mism.append(f"{r['task_id']}: recorded {(r.get('score') or {}).get('correct')}, own {m} ({st}, {secs:.2f} s)")
    c(not ref_mism, f"the earlier run {ref_dir.name}: recorded scores re-derived", f"{ref_mism}")
    out["reference_run_correct"] = {"run": ref_dir.name, "correct": ref_corr, "of": len(ref_recs)}
    if (lo, hi) == (16, 64):
        out["historical_64"] = {"p0_16_correct": ref_corr, "p16_64_correct": corr, "total_correct": ref_corr + corr, "of": len(ref_recs) + n}
    return out


@section
def sec_baselines(sec: Section, ctx: dict) -> None:
    base = ctx["runs"] / "baselines"

    def one(glob_pat: str):
        found = sorted(base.glob(glob_pat))
        if len(found) != 1:
            return None, f"{len(found)} run directories match {glob_pat}"
        if not (found[0] / "summary.json").exists():
            n = len(jl(found[0] / "records.jsonl")) if (found[0] / "records.jsonl").exists() else 0
            return None, f"SKIPPED: {found[0].name} has no summary.json yet (still running; {n} records so far)"
        return found[0], None

    sec.info["B4_p16_64"] = verify_llm_run(sec, ctx, base / B4_P16, base / B4_P0, "B4 16-63")
    qd, why = one(Q27_P16_GLOB)
    sec.info["Q27_p16_64"] = verify_llm_run(sec, ctx, qd, base / Q27_P0, "Q27 16-63") if qd else why
    # the new population: each run against its 16-63 run, Qwen-path sources byte-identical
    train_mans = sorted((perf_root(ctx) / "train").glob("*/attempts/*/manifest.json"))
    frozen = json.loads(train_mans[0].read_text())["git"]["commit"] if train_mans else None
    for arm, pat, ref in (("B4", "*_B4_thinking_p64-128", base / B4_P16), ("Q27", "*_Q27_thinking_low_p64-128", qd)):
        d, why = one(pat)
        if d is None or ref is None:
            sec.info[f"{arm}_p64_128"] = why or "no 16-63 reference run"
            continue
        sec.info[f"{arm}_p64_128"] = verify_llm_run(sec, ctx, d, ref, f"{arm} 64-127", expect=(64, 128), ref_expect=(16, 64),
                                                    strict_sources=QWEN_PATH_SOURCES)
        commit = json.loads((d / "manifest.json").read_text())["git"]["commit"]
        sec.check(frozen is not None and commit == frozen, f"{arm} 64-127: run from the frozen commit of the fits and evaluations",
                  f"{commit} vs {frozen}")


# === phase 2: training, evaluation, comparison ==========================================================================
PERF_LABELS = ("T4-v1", "C4-v1", "T4-v2", "C4-v2")      # the protocol's arm order
PERF_SEEDS = (100, 101, 102)
PERF_UPDATES, PERF_CKPT_EVERY = 3000, 500
LABEL_ARM = {"T4-v1": ("T4", "v1"), "C4-v1": ("c4_params", "v1"), "T4-v2": ("T4", "v2"), "C4-v2": ("c4_params", "v2")}
V2_ADDED_PARTS = ("trunk.schema_encoder", "trunk.item_features", "trunk.pointer_relation")  # modules or tensors
PROTOCOL_V2_ADDED = 4_218_064                             # the protocol's stated trunk difference
RECIPE = {"lr": 3e-4, "weight_decay": 0.01, "grad_clip": 1.0, "effective_batch": 32, "micro_batch": 8,
          "matmul_precision": "high", "checkpoint_every": PERF_CKPT_EVERY, "updates": PERF_UPDATES,
          "loss_weights": {"family": 1.0, "op": 1.0, "pointer": 1.0}, "betas": [0.9, 0.999], "eps": 1e-8}
# config paths allowed to differ between the four arms of one seed (arm, grounding, the C4 hidden width, the
# parameter report, the grammar summary, the label)
ARM_PATHS = ("label", "controller.arm", "controller.grounding", "controller.c4_params_hidden", "parameters", "grammar",
             "train.arm")


def optimizer_digest(state: dict) -> str:
    """As in verify_pilot.py (the reviewer's), from the documented definition: param groups as canonical JSON, then
    each parameter's state entries in (index, key) order, tensors by their digest."""
    import torch
    h = hashlib.sha256(json.dumps(state["param_groups"], sort_keys=True, default=str).encode())
    for index in sorted(state["state"]):
        for key in sorted(state["state"][index]):
            v = state["state"][index][key]
            h.update(json.dumps([index, key]).encode())
            h.update(tensor_digest({"v": v}).encode() if torch.is_tensor(v) else json.dumps(v).encode())
    return h.hexdigest()


def example_stream(n: int, seed: int, updates: int, batch: int) -> list:
    """Own implementation of train.SAMPLING_RULE: the concatenation of torch.randperm(n) permutations, epoch e drawn
    with a torch.Generator seeded by int(sha256(f'{seed}:epoch:{e}')[:15], 16); update u takes stream positions
    [u * batch, (u + 1) * batch). Returns one index list per update."""
    import torch
    perms = {}
    out = []
    for u in range(updates):
        row = []
        for pos in range(u * batch, (u + 1) * batch):
            e, k = divmod(pos, n)
            if e not in perms:
                g = torch.Generator().manual_seed(int(hashlib.sha256(f"{seed}:epoch:{e}".encode()).hexdigest()[:15], 16))
                perms[e] = torch.randperm(n, generator=g).tolist()
            row.append(perms[e][k])
        out.append(row)
    return out


def flat(obj, prefix="") -> dict:
    """Nested dicts flattened to dotted paths; lists and scalars are leaves."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(flat(v, f"{prefix}{k}."))
        return out if out else {prefix.rstrip("."): {}}
    return {prefix.rstrip("."): obj}


def diff_paths(a: dict, b: dict) -> list:
    fa, fb = flat(a), flat(b)
    return sorted(k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>"))


def allowed_path(p: str, allowed) -> bool:
    return any(p == x or p.startswith(x + ".") for x in allowed)


def identity_of(obj) -> str:
    """The run-configuration identity as documented (train.identity_of): sha256 of sort_keys JSON with compact
    separators, ASCII-escaped, non-JSON values as str."""
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode())


def perf_root(ctx) -> Path:
    """This study's run tree: <runs>/perf_v2 (local) or the committed bundle evidence/perf_v2 (public)."""
    return ctx["perf_root"]


def ledger_file(ctx) -> Path:
    return ctx["ledger_path"]


def pilot_root(ctx) -> Path:
    return ctx["pilot_root"]


def baseline_dirs(ctx, pattern: str) -> list:
    return sorted(p for root in ctx["baseline_roots"] if root.exists() for p in root.glob(pattern) if p.is_dir())


def score_problems(r: dict) -> list:
    """The recorded score's internal consistency (what can be checked without the database): its kind, status,
    the predicted and gold executions, the correct bit and the scored-SQL binding."""
    sc, f = r.get("score") or {}, r["final"]
    p, g = sc.get("predicted") or {}, sc.get("gold") or {}
    pred = f.get("sql") if f.get("kind") == "sql" else None
    out = []
    if sc.get("scored_sql_sha256") != sha256_bytes((pred or "").encode()):
        out.append("scored SQL binding")
    st, c = sc.get("status"), sc.get("correct")
    expect = {"abstain": (0, "abstain", "not_run"), "correct": (1, "ok", "ok"), "incorrect": (0, "ok", "ok"),
              "predicted_error": (0, "error", "ok"), "predicted_timeout": (0, "timeout", "not_run")}.get(st)
    if expect is None:
        out.append(f"unknown status {st}")
    elif (c, p.get("status"), g.get("status")) != expect:
        out.append(f"status {st} with correct {c}, predicted {p.get('status')}, gold {g.get('status')}")
    if (f.get("kind") == "abstain") != (st == "abstain"):
        out.append(f"final {f.get('kind')} scored {st}")
    if st == "correct" and (p.get("distinct_rows") != g.get("distinct_rows") or p.get("distinct_rows") is None):
        out.append("correct with different distinct row counts")
    if st == "predicted_error" and not p.get("error"):
        out.append("predicted_error without an error")
    return out


def perf_commit_checks(sec: Section, ctx: dict, commits: set, what: str) -> "str | None":
    """Every run of a kind at one commit, which carries the frozen protocol and the code of frozen_at."""
    repo = ctx["repo"]
    pin = json.loads((repo / PIN_DIR / "perf_v2_protocol.json").read_text())
    if not sec.check(len(commits) == 1, f"{what}: every run at one commit", f"{sorted(commits)}"):
        return None
    commit = next(iter(commits))
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", pin["frozen_at"], commit], cwd=repo).returncode == 0
    sec.check(anc, f"{what}: the run commit contains frozen_at", commit)
    proto = sha256_bytes(git_show(repo, commit, pin["file"]))
    sec.check(proto == pin["sha256"], f"{what}: PROTOCOL_perf_v2.md at the run commit = pin", proto)
    code = git(repo, "diff", "--stat", pin["frozen_at"], commit, "--", HARNESS + "*.py").strip()
    sec.check(code == "", f"{what}: no producer code changed between frozen_at and the run commit", code)
    return commit


def ckpt_numbers(state: dict) -> dict:
    total = sum(v.numel() for v in state.values())
    trunk = sum(v.numel() for k, v in state.items() if k.startswith("trunk."))
    sem = sum(v.numel() for k, v in state.items() if k.startswith("semantic."))
    return {"total": total, "trunk": trunk, "semantic": sem, "other": total - trunk - sem}


@section
def sec_train(sec: Section, ctx: dict) -> None:
    repo = ctx["repo"]
    root = perf_root(ctx) / "train"
    pin = json.loads((repo / PIN_DIR / "perf_v2_train.json").read_text())["task_ids"]
    proto = json.loads((repo / PIN_DIR / "perf_v2_protocol.json").read_text())
    dirs = sorted(p.name for p in root.iterdir() if p.is_dir())
    want = sorted(f"{lab}_s{s}" for lab in PERF_LABELS for s in PERF_SEEDS)
    sec.check(dirs == want, "train/: exactly the 12 fits (4 arms x seeds 100-102)", f"{dirs}")
    rs, _p = parse_ledger(ledger_file(ctx))
    by_cmd = {}
    for r in rs:
        by_cmd.setdefault(r["start"]["command"], []).append(r)
    try:
        import torch
        torch.set_num_threads(4)
    except Exception as e:  # noqa: BLE001
        sec.check(False, "torch importable for the checkpoint digests", repr(e))
        torch = None
    cfgs, logs, finals, numbers, keys, commits, fit_info = {}, {}, {}, {}, {}, set(), {}
    for s in PERF_SEEDS:
        stream = example_stream(len(pin), s, PERF_UPDATES, RECIPE["effective_batch"])
        for lab in PERF_LABELS:
            rd = root / f"{lab}_s{s}"
            tag = f"{lab} s{s}"
            arm, g = LABEL_ARM[lab]
            cfg = json.loads((rd / "config.json").read_text())
            cfgs[(lab, s)] = cfg
            # --- training data and recipe ------------------------------------------------------------------------
            sec.check(cfg["examples"] == pin, f"{tag}: config examples = perf_v2_train task_ids, in order")
            sec.check(cfg.get("requested_task_ids") == pin and cfg.get("skipped") == {},
                      f"{tag}: requested_task_ids = the pin and nothing skipped", f"skipped {cfg.get('skipped')}")
            t = cfg["train"]
            bad = {k: (t.get(k), v) for k, v in RECIPE.items() if t.get(k) != v}
            sec.check(not bad and t.get("seed") == s and t.get("arm") == arm,
                      f"{tag}: recipe = protocol (lr, wd, clip, batch 32 / micro 8, TF32 'high', every 500, U 3000, weights 1/1/1), seed, arm",
                      f"{bad} seed {t.get('seed')} arm {t.get('arm')}")
            sec.check(cfg["controller"].get("arm") == arm and cfg["controller"].get("grounding") == g and cfg.get("label") == lab,
                      f"{tag}: controller arm / grounding / label", f"{cfg['controller'].get('arm')} {cfg['controller'].get('grounding')} {cfg.get('label')}")
            pp = cfg.get("perf_protocol") or {}
            sec.check((pp.get("sha256"), pp.get("frozen_at"), pp.get("updates")) == (proto["sha256"], proto["frozen_at"], PERF_UPDATES),
                      f"{tag}: config names the frozen protocol", f"{pp}")
            # --- the log ----------------------------------------------------------------------------------------------
            log = jl(rd / "train_log.jsonl")
            logs[(lab, s)] = log
            sec.check([x["update"] for x in log] == list(range(PERF_UPDATES)), f"{tag}: train_log holds updates 0..2999 once each, in order",
                      f"{len(log)} lines")
            sup = rd / "train_log.superseded.jsonl"
            fit_info[f"{lab}/s{s}"] = {"superseded_lines": len(jl(sup)) if sup.exists() else 0}
            badu = [x["update"] for x in log if x["examples"] != [pin[i] for i in stream[x["update"]]]]
            sec.check(not badu, f"{tag}: every update's examples = the own sampling stream (SAMPLING_RULE)", f"first bad updates {badu[:5]}")
            sec.check(t.get("sampling", "").startswith("a stream of example indices: the concatenation of torch.randperm(n)"),
                      f"{tag}: config records the sampling rule implemented here")
            nf = [x["update"] for x in log if nonfinite(x)]
            sec.check(not nf, f"{tag}: train log numbers finite", f"{nf[:5]}")
            # --- attempts, ledger ------------------------------------------------------------------------------------
            atts = sorted(p for p in (rd / "attempts").iterdir() if p.is_dir())
            mans = [json.loads((a / "manifest.json").read_text()) for a in atts]
            sums = [json.loads((a / "summary.json").read_text()) if (a / "summary.json").exists() else None for a in atts]
            commits |= {m["git"]["commit"] for m in mans}
            sec.check(all(m["git"]["research_tree_dirty"] is False and not m["git"]["dirty_paths"] for m in mans),
                      f"{tag}: every attempt from a clean research tree")
            sec.check(len(atts) == 1 and mans[0]["config"].get("resume_from") is None and sums[0] is not None
                      and sums[0].get("resumed_from") is None, f"{tag}: one attempt, not resumed, with a summary",
                      f"{len(atts)} attempts; resume_from {[m['config'].get('resume_from') for m in mans]}")
            sec.check(all(m["config"].get("train_pin_sha256") == proto["train_pin_sha256"] and m["config"].get("label") == lab
                          and m["config"].get("seed") == s and m["config"].get("updates") == PERF_UPDATES for m in mans),
                      f"{tag}: attempt manifest names the train pin, label, seed and U")
            res = [r for r in by_cmd.get(f"perf-train {lab} s{s}", [])]
            ok_res = sec.check(len(res) == len(atts) == 1, f"{tag}: one ledger reservation 'perf-train {lab} s{s}' per attempt",
                               f"{len(res)} reservations for {len(atts)} attempts")
            if res:
                r = res[-1]
                sec.check(r["end"] is not None and r["end"]["status"] == "ok", f"{tag}: its reservation ended ok",
                          f"{(r['end'] or {}).get('status')}")
                sec.check((r["start"].get("authorization"), r["start"]["cap_hours"]) == (2, 32.0), f"{tag}: reservation under authorization 2")
                if ok_res and sums[0]:
                    sec.check((sums[0].get("reservation") or {}).get("id") == r["id"], f"{tag}: attempt summary names this reservation")
                fit_info[f"{lab}/s{s}"].update(reservation=r["id"], hours=round(r["hours"], 4), status=(r["end"] or {}).get("status", "OPEN"),
                                               first_loss=log[0]["loss"] if log else None, last_loss=log[-1]["loss"] if log else None)
            # --- checkpoints -----------------------------------------------------------------------------------------
            cks = jl(rd / "checkpoints.jsonl")
            sec.check([c["update"] for c in cks] == list(range(PERF_CKPT_EVERY - 1, PERF_UPDATES, PERF_CKPT_EVERY)),
                      f"{tag}: checkpoints at updates 499, 999, ..., 2999, once each", f"{[c['update'] for c in cks]}")
            if sums and sums[0]:
                sec.check([(c["sha256"], c["model_digest"], c["optimizer_digest"]) for c in sums[0]["checkpoints"]]
                          == [(c["sha256"], c["model_digest"], c["optimizer_digest"]) for c in cks],
                          f"{tag}: attempt summary's checkpoints = checkpoints.jsonl")
            cid = identity_of(cfg)
            sec.check(all(c.get("config_identity") == cid for c in cks) and (sums[0] or {}).get("config_identity") == cid,
                      f"{tag}: config.json hashes to the checkpoints' and summary's config_identity", cid)
            f = [c for c in cks if c["update"] == PERF_UPDATES - 1][-1]
            finals[(lab, s)] = f
            path = rd / "checkpoints" / Path(f["path"]).name
            sec.check(Path(f["path"]).parts[-3:-1] == (rd.name, "checkpoints") and Path(f["path"]).name == f"ckpt_u{PERF_UPDATES - 1:06d}_{f['sha256'][:16]}.pt",
                      f"{tag}: final checkpoint record: in the fit's checkpoints dir, named by its sha256", f["path"])
            if not ctx["rehash"]:
                continue
            h = sha256_path(path)
            sec.check(path.name == f"ckpt_u{PERF_UPDATES - 1:06d}_{h[:16]}.pt" and h == f["sha256"] and path.stat().st_size == f["bytes"],
                      f"{tag}: final checkpoint re-hashes to its name and record", f"{path.name} {h}")
            if torch is None:
                continue
            try:
                ck, _allowed = safe_load(path)
            except Exception as e:  # noqa: BLE001
                sec.check(False, f"{tag}: final checkpoint loads with weights_only", repr(e)[:300])
                continue
            md, od = tensor_digest(ck["model"]), optimizer_digest(ck["optimizer"])
            sec.check(md == f["model_digest"], f"{tag}: own model digest = checkpoints.jsonl", md)
            sec.check(od == f["optimizer_digest"], f"{tag}: own optimizer digest = checkpoints.jsonl", od)
            sec.check(ck.get("update") == PERF_UPDATES - 1 and canonical(ck.get("config")) == canonical(cfg),
                      f"{tag}: checkpoint's own update = 2999 and its config = config.json")
            nums = ckpt_numbers(ck["model"])
            numbers[(lab, s)] = nums
            keys[(lab, s)] = {k: tuple(v.shape) for k, v in ck["model"].items()}
            p = cfg["parameters"]
            sec.check((nums["total"], nums["trunk"], nums["semantic"], nums["other"]) == (p["total"], p["trunk"], p["semantic"], 0),
                      f"{tag}: parameters counted from the state dict = config's report (total, trunk, semantic)",
                      f"{nums} vs {p}")
            fit_info[f"{lab}/s{s}"].update(parameters=nums, model_digest=md[:16])
            del ck

    # --- configs within a seed and across seeds -----------------------------------------------------------------
    for s in PERF_SEEDS:
        base = cfgs[("T4-v1", s)]
        for lab in PERF_LABELS[1:]:
            d = diff_paths(base, cfgs[(lab, s)])
            other = [p for p in d if not allowed_path(p, ARM_PATHS)]
            sec.check(not other, f"s{s}: T4-v1 vs {lab} configs differ only in arm / grounding / c4 hidden / parameters / grammar / label",
                      f"other differences: {other}")
            sec.info.setdefault("config_differences_within_seed", {})[f"s{s} T4-v1 vs {lab}"] = d
    for lab in PERF_LABELS:
        for s in PERF_SEEDS[1:]:
            d = diff_paths(cfgs[(lab, PERF_SEEDS[0])], cfgs[(lab, s)])
            sec.check(d == ["train.seed"], f"{lab}: s{PERF_SEEDS[0]} vs s{s} configs differ only in train.seed", f"{d}")
    for s in PERF_SEEDS:
        ex0 = [x["examples"] for x in logs[("T4-v1", s)]]
        sec.check(all([x["examples"] for x in logs[(lab, s)]] == ex0 for lab in PERF_LABELS[1:]),
                  f"s{s}: identical examples at every update across the four arms")
    commit = perf_commit_checks(sec, ctx, commits, "training")
    sec.info["run_commit"] = ctx["train_commit"] = commit

    # --- parameter matching ------------------------------------------------------------------------------------------
    if not numbers:   # public mode (no weights): the configs' parameter reports
        for s in PERF_SEEDS:
            p = {lab: cfgs[(lab, s)]["parameters"] for lab in PERF_LABELS}
            for g in ("v1", "v2"):
                sec.check((p[f"T4-{g}"]["total"], p[f"T4-{g}"]["trunk"]) == (p[f"C4-{g}"]["total"], p[f"C4-{g}"]["trunk"]),
                          f"s{s} {g}: T4 and C4 parameter totals and trunks equal (config reports)", f"{p[f'T4-{g}']} {p[f'C4-{g}']}")
            for arch in ("T4", "C4"):
                dt = p[f"{arch}-v2"]["trunk"] - p[f"{arch}-v1"]["trunk"]
                sec.check(dt == PROTOCOL_V2_ADDED and p[f"{arch}-v2"]["total"] - p[f"{arch}-v1"]["total"] == PROTOCOL_V2_ADDED
                          and p[f"{arch}-v2"]["semantic"] == p[f"{arch}-v1"]["semantic"],
                          f"s{s} {arch}: v2 adds {PROTOCOL_V2_ADDED:,} trunk parameters and nothing else (config reports)", f"{dt}")
        sec.info["parameters_from"] = "config reports (no weights in the bundle)"
    if numbers:
        for s in PERF_SEEDS:
            for g in ("v1", "v2"):
                a, b = numbers.get((f"T4-{g}", s)), numbers.get((f"C4-{g}", s))
                if a and b:
                    sec.check(a["total"] == b["total"] and a["trunk"] == b["trunk"], f"s{s} {g}: T4 and C4 parameter totals (and trunks) equal",
                              f"T4 {a} C4 {b}")
            for arch in ("T4", "C4"):
                k1, k2 = keys.get((f"{arch}-v1", s)), keys.get((f"{arch}-v2", s))
                if not (k1 and k2):
                    continue
                added = {k: v for k, v in k2.items() if k not in k1}
                sec.check(all(k in k2 and k2[k] == v for k, v in k1.items()), f"s{s} {arch}: every v1 tensor exists in v2 with the same shape")
                sec.check(all(any(allowed_path(k, (part,)) for part in V2_ADDED_PARTS) for k in added),
                          f"s{s} {arch}: v2's added tensors are all schema encoder / item features / pointer relations",
                          f"{sorted(k for k in added if not any(allowed_path(k, (part,)) for part in V2_ADDED_PARTS))[:5]}")
                size = sum(math.prod(v) for v in added.values())
                dtrunk = numbers[(f"{arch}-v2", s)]["trunk"] - numbers[(f"{arch}-v1", s)]["trunk"]
                parts = {part: sum(math.prod(v) for k, v in added.items() if allowed_path(k, (part,))) for part in V2_ADDED_PARTS}
                sec.check(dtrunk == size == PROTOCOL_V2_ADDED, f"s{s} {arch}: v2 trunk - v1 trunk = added parts = {PROTOCOL_V2_ADDED:,}",
                          f"dtrunk {dtrunk}, added {size}, parts {parts}")
                sec.info["v2_added_parameters"] = parts
    sec.info["fits"] = fit_info
    sec.info["training_hours"] = round(sum(v.get("hours", 0) for v in fit_info.values()), 4)
    sec.info["not_checked"] = ("the shared trunk initialization at update 0 (no update-0 checkpoint is kept); the losses "
                               "themselves (no re-training); intermediate checkpoints beyond their records")


# --- evaluation -----------------------------------------------------------------------------------------------------
def _ex_job(job):
    db, pred, gold, timeout = job
    return ex(Path(db), pred, gold, timeout)


def ex_batch(jobs: dict, workers: int) -> dict:
    """{key: (db path, predicted | None, gold, timeout)} -> {key: (correct, status, seconds)}, in worker processes
    (each inherits this process's CPU affinity)."""
    if not jobs:
        return {}
    keys = list(jobs)
    if workers <= 1:
        return {k: _ex_job(jobs[k]) for k in keys}
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(_ex_job, [tuple(map(str, jobs[k][:1])) + jobs[k][1:] for k in keys], chunksize=2))
    return dict(zip(keys, res))


def status_class(s) -> str:
    """The recorded scorer statuses and this file's, in common classes."""
    s = str(s)
    if s in ("correct", "incorrect", "abstain"):
        return s
    if "timeout" in s:
        return "timeout"
    if "error" in s:
        return "error"
    return s


def rescore(ctx: dict, rel: Release, records: list) -> dict:
    """Own EX for every record (deduplicated by question and SQL; cached across sections). {(task, sql sha): result}."""
    cache = ctx.setdefault("ex_cache", {})
    jobs = {}
    for r in records:
        pred = r["final"].get("sql") if r["final"].get("kind") == "sql" else None
        key = (r["task_id"], sha256_bytes((pred or "").encode()) if pred is not None else None)
        if key not in cache and key not in jobs:
            jobs[key] = (rel.db_path(r["db_id"]), pred, rel.by_id[r["task_id"]]["SQL"], rel.timeout)
    t0 = time.time()
    cache.update(ex_batch(jobs, ctx.get("workers", 3)))
    ctx.setdefault("ex_seconds", 0.0)
    ctx["ex_seconds"] += time.time() - t0
    return cache


def ex_key(r: dict) -> tuple:
    pred = r["final"].get("sql") if r["final"].get("kind") == "sql" else None
    return (r["task_id"], sha256_bytes((pred or "").encode()) if pred is not None else None)


def load_anchors(sec: Section, ctx: dict) -> dict:
    """{task_id: anchor entry}: the pilot's anchors (clean encode runs under the pilot protocol) and this study's
    (clean perf-encode runs under the perf protocol); a conflicting anchor is a failed check."""
    repo = ctx["repo"]
    pilot_sha = json.loads((repo / PIN_DIR / "pilot_protocol_v1.json").read_text())["sha256"]
    perf_sha = json.loads((repo / PIN_DIR / "perf_v2_protocol.json").read_text())["sha256"]
    out, conflicts, source = {}, [], {}
    for root, sha, tag in ((pilot_root(ctx) / "encode", pilot_sha, "pilot"), (perf_root(ctx) / "encode", perf_sha, "perf")):
        for entries in sorted(root.glob("*/entries.jsonl")):
            m = json.loads((entries.parent / "manifest.json").read_text())
            if m["git"]["research_tree_dirty"] or (m["config"].get("protocol") or {}).get("sha256") != sha:
                continue
            for e in jl(entries):
                old = out.get(e["task_id"])
                if old is not None and (old["states_sha256"], old["cache_key"]) != (e["states_sha256"], e["cache_key"]):
                    conflicts.append(f"{e['task_id']}: {source[e['task_id']]} vs {tag}/{entries.parent.name}")
                if old is None:
                    out[e["task_id"]], source[e["task_id"]] = e, f"{tag}/{entries.parent.name}"
    sec.check(not conflicts, "anchors: no task anchored to two different states", f"{conflicts[:5]}")
    ctx["anchor_source"] = source
    return out


def eval_population_ids(repo: Path) -> tuple:
    ev = json.loads((repo / PIN_DIR / "perf_v2_eval.json").read_text())
    return ev["populations"]["historical"]["task_ids"], ev["populations"]["new"]["task_ids"]


@section
def sec_eval(sec: Section, ctx: dict) -> None:
    repo = ctx["repo"]
    proto = json.loads((repo / PIN_DIR / "perf_v2_protocol.json").read_text())
    hist, new = eval_population_ids(repo)
    ids = hist + new
    rs, _p = parse_ledger(ledger_file(ctx))
    by_cmd = {}
    for r in rs:
        by_cmd.setdefault(r["start"]["command"], []).append(r)
    evpin = json.loads((repo / PIN_DIR / "perf_v2_eval.json").read_text())["populations"]
    db_pin = dict(zip(evpin["historical"]["task_ids"] + evpin["new"]["task_ids"],
                      evpin["historical"]["db_ids"] + evpin["new"]["db_ids"]))
    snaps = json.loads(git_show(repo, proto["frozen_at"], PIN_DIR + "/snapshots_v1.json"))["databases"]

    # --- this study's encode of the new 64 and the anchors -----------------------------------------------------------
    enc = sorted((perf_root(ctx) / "encode").glob("*"))
    sec.check(len(enc) == 1, "exactly one perf-encode run", f"{[e.name for e in enc]}")
    em = json.loads((enc[0] / "manifest.json").read_text())
    es = json.loads((enc[0] / "summary.json").read_text())
    ee = jl(enc[0] / "entries.jsonl")
    sec.check(em["git"]["research_tree_dirty"] is False and em["config"]["protocol"]["sha256"] == proto["sha256"],
              "perf-encode: clean tree, frozen protocol")
    sec.check(em["config"]["ids"] == new and [e["task_id"] for e in ee] == new, "perf-encode: requested and anchored exactly the new 64, in order")
    rec = es.get("rechecked") or []
    sec.check(len(rec) == 8 and all(x.get("equal") is True for x in rec) and [x["task_id"] for x in rec] == em["config"]["recheck"]
              and set(em["config"]["recheck"]) <= set(hist),
              "perf-encode: 8 historical contexts recomputed in a new process, all equal to their pilot anchors", f"{rec}")
    sec.check((es.get("encoded"), es.get("requested"), es.get("already_present")) == (64, 64, 0), "perf-encode: 64 encoded, none already present")
    er = by_cmd.get("perf-encode new", [])
    sec.check(len(er) == 1 and er[0]["end"] and er[0]["end"]["status"] == "ok" and (es.get("reservation") or {}).get("id") == er[0]["id"],
              "perf-encode: one ledger reservation, ended ok, named by its summary")
    anchors = load_anchors(sec, ctx)
    missing = [t for t in ids if t not in anchors]
    sec.check(not missing, "anchors exist for all 128 evaluation questions", f"{missing[:5]}")
    src = ctx["anchor_source"]
    sec.check(all(src.get(t, "").startswith("pilot/") for t in hist) and all(src.get(t, "").startswith("perf/") for t in new),
              "historical anchors are the pilot's; new anchors are perf-encode's",
              f"{sorted({src.get(t) for t in hist})} / {sorted({src.get(t) for t in new})}")
    sec.info["anchor_sources"] = {"historical": sorted({src.get(t) for t in hist}), "new": sorted({src.get(t) for t in new})}

    # --- the fits' finals ----------------------------------------------------------------------------------------------
    finals = {}
    for lab in PERF_LABELS:
        for s in PERF_SEEDS:
            rd = perf_root(ctx) / "train" / f"{lab}_s{s}"
            finals[(lab, s)] = [c for c in jl(rd / "checkpoints.jsonl") if c["update"] == PERF_UPDATES - 1][-1]

    # --- every evaluation run: eligibility derived here ---------------------------------------------------------------
    rel = None if ctx["public"] else Release(ctx["runs"], repo, proto["frozen_at"], ctx["cache"])
    all_runs, chosen, recs_of, commits = [], {}, {}, set()
    for d in sorted((perf_root(ctx) / "eval").glob("s*_*")):
        man = json.loads((d / "manifest.json").read_text())
        summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        dec = json.loads((d / "decodes.json").read_text()) if (d / "decodes.json").exists() else {}
        cfg, s = man["config"], man["config"]["seed"]
        commits.add(man["git"]["commit"])
        problems = []
        if man["git"]["research_tree_dirty"] or man["git"]["dirty_paths"]:
            problems.append("dirty research tree")
        if (cfg.get("protocol") or {}).get("sha256") != proto["sha256"]:
            problems.append("another protocol")
        if sha256_bytes(canonical(cfg).encode()) != man["config_identity"]:
            problems.append("manifest config does not hash to its config_identity")
        if cfg["ids"] != ids:
            problems.append("ids are not historical + new in pin order")
        for lab in PERF_LABELS:
            c, f = (cfg.get("checkpoints") or {}).get(lab) or {}, finals[(lab, s)]
            if (c.get("sha256"), c.get("model_digest"), c.get("update")) != (f["sha256"], f["model_digest"], PERF_UPDATES - 1) \
                    or c.get("path") != f"perf_v2/train/{lab}_s{s}/checkpoints/{Path(f['path']).name}":
                problems.append(f"{lab}: evaluated checkpoint is not the fit's final")
        if sorted(summ.get("arms") or {}) != sorted(PERF_LABELS):
            problems.append(f"summary arms {sorted(summ.get('arms') or {})}")
        states = {}
        for lab in PERF_LABELS:
            o = (summ.get("arms") or {}).get(lab) or {}
            inv = o.get("reconstruction") or []
            if [x.get("task_id") for x in inv] != ids:
                problems.append(f"{lab}: inventory does not list the 128 once each, in order")
            false = [(x.get("task_id"), k) for x in inv for k in ("same_actions", "same_final", "fresh_states_equal_anchor") if x.get(k) is not True]
            if false:
                problems.append(f"{lab}: {len(false)} false inventory entries, first {false[:3]}")
            # the inventory recomputed here from the decodes
            dl = dec.get(lab) or {}
            timed, recon = dl.get("timed") or [], dl.get("reconstruction") or []
            if [x.get("task_id") for x in timed] != ids or [x.get("task_id") for x in recon] != ids:
                problems.append(f"{lab}: decodes do not list the 128 in order")
            else:
                mine = [{"task_id": a["task_id"], "same_actions": a["actions"] == b["actions"], "same_final": a["final"] == b["final"],
                         "fresh_states_equal_anchor": a["states_sha256"] == anchors.get(a["task_id"], {}).get("states_sha256")
                         and b["states_sha256"] == a["states_sha256"]} for a, b in zip(timed, recon)]
                if mine != [{k: x.get(k) for k in ("task_id", "same_actions", "same_final", "fresh_states_equal_anchor")} for x in inv]:
                    problems.append(f"{lab}: the inventory recomputed from decodes.json differs from the summary's")
            if o.get("record_errors"):
                problems.append(f"{lab}: record errors")
            flags = {"reconstruction_all_equal": bool(inv) and all(x.get("same_actions") is True and x.get("same_final") is True for x in inv),
                     "fresh_states_all_equal_anchor": bool(inv) and all(x.get("fresh_states_equal_anchor") is True for x in inv)}
            for k, v in flags.items():
                if o.get(k) is not v:
                    problems.append(f"{lab}: summary {k} = {o.get(k)}, inventory gives {v}")
            # the records
            recs = jl(d / lab / "records.jsonl") if (d / lab / "records.jsonl").exists() else []
            recs_of[(d.name, lab)] = recs
            arm, g = LABEL_ARM[lab]
            if [r["task_id"] for r in recs] != ids:
                problems.append(f"{lab}: records are not the 128 once each, in pin order")
                continue
            cid = sha256_bytes(canonical({"run": man["config_identity"], "arm": lab,
                                          "model_digest": (cfg.get("checkpoints") or {}).get(lab, {}).get("model_digest")}).encode())
            for r, a in zip(recs, timed if len(timed) == len(recs) else [None] * len(recs)):
                t = r["task_id"]
                pred = r["final"].get("sql") if r["final"].get("kind") == "sql" else None
                if r.get("config_identity") != cid or o.get("config_identity") != cid:
                    problems.append(f"{lab} {t}: config_identity is not sha256({{run, arm: label, model_digest}})")
                if (r.get("label"), {"C4_params": "c4_params"}.get(r.get("arm"), r.get("arm")), r.get("grounding")) != (lab, arm, g):
                    problems.append(f"{lab} {t}: label/arm/grounding {(r.get('label'), r.get('arm'), r.get('grounding'))}")
                if r["db_id"] != db_pin[t] or r["snapshot_sha256"] != snaps[r["db_id"]]["sha256"] \
                        or r["score"].get("snapshot_sha256") != snaps[r["db_id"]]["sha256"]:
                    problems.append(f"{lab} {t}: database / snapshot binding")
                if r["score"].get("scored_sql_sha256") != sha256_bytes((pred or "").encode()):
                    problems.append(f"{lab} {t}: scored SQL binding")
                if r.get("states_sha256") != anchors.get(t, {}).get("states_sha256"):
                    problems.append(f"{lab} {t}: fresh states differ from the anchor")
                if a is not None and (r["attempts"][0].get("actions") != a["actions"] or r["final"] != a["final"]
                                      or r["states_sha256"] != a["states_sha256"]):
                    problems.append(f"{lab} {t}: record differs from the timed decode")
                if nonfinite(r):
                    problems.append(f"{lab} {t}: non-finite number")
                states.setdefault(t, set()).add(r.get("states_sha256"))
        arm_names = sorted({str(r.get("arm")) for (dn, _l), rr in recs_of.items() if dn == d.name for r in rr})
        sec.info.setdefault("record_arm_spellings", {})[d.name] = arm_names
        cross = sorted(t for t, v in states.items() if len(v) != 1)
        if cross:
            problems.append(f"states differ across the arms: {cross[:3]}")
        if summ.get("states_differ_across_arms"):
            problems.append("producer: states differ across the arms")
        eligible = not problems and summ.get("passed") is True
        sec.check((summ.get("passed") is True) == (not problems), f"{d.name}: producer's passed flag agrees with the checks here",
                  f"passed={summ.get('passed')}; problems {problems[:6]}")
        all_runs.append({"run": f"perf_v2/eval/{d.name}", "seed": s, "eligible": eligible, "producer_passed": summ.get("passed"),
                         "problems": problems[:10]})
        if eligible:
            if s in chosen:
                sec.check(False, f"seed {s}: more than one passing evaluation", f"{chosen[s].name}, {d.name}")
            chosen[s] = d
        # ledger
        lr = by_cmd.get(f"perf-eval s{s}", [])
        mine_r = [r for r in lr if (summ.get("reservation") or {}).get("id") == r["id"]]
        sec.check(len(mine_r) == 1 and mine_r[0]["end"] and mine_r[0]["end"]["status"] == "ok"
                  and mine_r[0]["start"].get("authorization") == 2, f"{d.name}: its ledger reservation exists, ended ok, authorization 2")
    for s in PERF_SEEDS:
        sec.check(s in chosen, f"seed {s}: exactly one passing evaluation", f"{[r['run'] for r in all_runs if r['seed'] == s]}")
    sec.info["evaluations"] = all_runs
    sec.info["eval_commit"] = perf_commit_checks(sec, ctx, commits, "evaluation")
    sec.check(sec.info["eval_commit"] is not None and sec.info["eval_commit"] == ctx.get("train_commit", sec.info["eval_commit"]),
              "evaluations at the training commit (when the train section ran)")

    # --- scores re-derived with the own EX rule -----------------------------------------------------------------------
    counted = [rr for (dn, lab), rr in recs_of.items() if any(dn == chosen[s].name for s in chosen)]
    if rel is None:   # public: the recorded scores, checked for internal consistency
        cache = {ex_key(r): (r["score"]["correct"], r["score"]["status"], 0.0) for rr in counted for r in rr}
        incons = [f"{r['task_id']} {r.get('label')}: {p_}" for rr in counted for r in rr for p_ in score_problems(r)]
        sec.check(not incons, "public: every counted record's recorded score is internally consistent (status, executions, "
                  "correct bit, scored SQL)", f"{incons[:10]}")
    else:
        cache = rescore(ctx, rel, [r for rr in counted for r in rr])
    correct, per_arm, mism, warn = {}, {}, [], []
    for s, d in sorted(chosen.items()):
        summ = json.loads((d / "summary.json").read_text())
        for lab in PERF_LABELS:
            recs = recs_of[(d.name, lab)]
            mine = {}
            for r in recs:
                own, st, secs = cache[ex_key(r)]
                mine[r["task_id"]] = own
                rc, rst = r["score"].get("correct"), r["score"].get("status")
                if rc != own or status_class(rst) != status_class(st):
                    timing = "timeout" in (status_class(st), status_class(rst)) or secs > 0.8 * 30.0
                    (warn if timing else mism).append(f"s{s} {lab} {r['task_id']}: recorded {rc}/{rst}, own {own}/{st} ({secs:.1f} s)")
            correct[(lab, s)] = mine
            n_ok = sum(mine.values())
            gen = sum(r["elapsed_seconds"] for r in recs)
            exe = sum(((r["score"].get("predicted") or {}).get("elapsed_seconds") or 0.0) for r in recs)
            own_t = {"requested": len(recs), "correct": n_ok, "answered": sum(r["final"]["kind"] == "sql" for r in recs),
                     "abstained": sum(r["final"]["kind"] == "abstain" for r in recs),
                     "errors": sum(r.get("error") is not None for r in recs), "timeouts": sum(bool(r.get("timed_out")) for r in recs),
                     "scorer_timeouts_own": sum(status_class(cache[ex_key(r)][1]) == "timeout" for r in recs),
                     "scorer_errors_own": sum(status_class(cache[ex_key(r)][1]) == "error" for r in recs),
                     "time_to_submission_seconds": gen, "submitted_execution_seconds": exe, "together_seconds": gen + exe,
                     "time_to_submission_per_correct": gen / n_ok if n_ok else None,
                     "together_per_correct": (gen + exe) / n_ok if n_ok else None}
            per_arm[f"{lab}/s{s}"] = own_t
            ps = ((summ.get("arms") or {}).get(lab) or {}).get("summary") or {}
            d_ = {k: (ps.get(k), own_t[k]) for k in ("requested", "correct", "answered", "abstained", "errors", "timeouts") if ps.get(k) != own_t[k]}
            sec.check(not d_, f"s{s} {lab}: summary counts = own counts (correct from own EX)", f"{d_}")
            sec.check(math.isclose(ps.get("elapsed_seconds_total", -1), gen, rel_tol=1e-9)
                      and (ps.get("cost_seconds_per_correct") is None if not n_ok else math.isclose(ps.get("cost_seconds_per_correct") or -1, gen / n_ok, rel_tol=1e-9)),
                      f"s{s} {lab}: time to submission total and per EX-correct = summary")
    if rel is not None:
        sec.check(not mism, "every counted record's score and status re-derived with the own EX rule", f"{mism[:10]}")
    sec.info["timing_sensitive_score_differences"] = warn
    sec.info["per_arm"] = per_arm
    sec.info["ex_executions"] = {"unique_question_sql_pairs": len(cache), "seconds": round(ctx.get("ex_seconds", 0), 1)}
    sec.info["record_arm_spelling_note"] = ("records spell the shared arm 'C4_params'; configs and summaries 'c4_params' "
                                            "(accepted as the same arm)")
    ctx["perf_eval"] = {"correct": correct, "chosen": {s: d.name for s, d in chosen.items()}, "all_runs": all_runs,
                        "db_of": {t: db_pin[t] for t in ids}, "hist": hist, "new": new, "rel": rel,
                        "records": {(lab, s): recs_of[(chosen[s].name, lab)] for s in chosen for lab in PERF_LABELS}}


# --- comparison -------------------------------------------------------------------------------------------------------
THRESHOLD_THREE_SEEDS = 10
BASELINE_TAGS = {("L", "historical"): ("L_nonthinking_p0-64",), ("L", "new"): ("L_nonthinking_p64-128",),
                 ("B4", "historical"): ("B4_thinking_p0-16", "B4_thinking_p16-64"), ("B4", "new"): ("B4_thinking_p64-128",),
                 ("Q27", "historical"): ("Q27_thinking_low_p0-16", "Q27_thinking_low_p16-64"),
                 ("Q27", "new"): ("Q27_thinking_low_p64-128",)}


def own_sign_flip(values: list) -> dict:
    """Exact two-sided sign flip over database blocks: every assignment of signs to the nonzero blocks;
    p = share with |sum| >= |observed|; the smallest attainable p is 2 / assignments."""
    import itertools
    observed = sum(values)
    nz = [v for v in values if v]
    total = extreme = 0
    for signs in itertools.product((1, -1), repeat=len(nz)):
        total += 1
        extreme += abs(sum(a * b for a, b in zip(signs, nz))) >= abs(observed)
    return {"observed": observed, "blocks": len(values), "nonzero_blocks": len(nz), "assignments": total,
            "p_two_sided": extreme / total if total else 1.0, "smallest_attainable_p": 2 / total if nz else 1.0}


def own_paired(correct: dict, a: str, b: str, seeds: list, ids: list, db_of: dict) -> dict:
    """The protocol's difference a - b: per seed, pooled, per database, leave-one-database-out, its three conditions
    (|pooled| >= 10 x seeds / 3 rounded up; the same sign in every seed; the same sign without any one database)."""
    per_seed = {s: sum(correct[(a, s)][t] for t in ids) - sum(correct[(b, s)][t] for t in ids) for s in seeds}
    pooled = sum(per_seed.values())
    dbs = sorted({db_of[t] for t in ids})
    per_db = {d: sum(correct[(a, s)][t] - correct[(b, s)][t] for s in seeds for t in ids if db_of[t] == d) for d in dbs}
    loo = {d: pooled - per_db[d] for d in dbs}
    thr = -(-THRESHOLD_THREE_SEEDS * len(seeds) // 3)
    sign = (pooled > 0) - (pooled < 0)
    cond = {"pooled_at_least_threshold": abs(pooled) >= thr,
            "same_sign_every_seed": sign != 0 and all(v * sign > 0 for v in per_seed.values()),
            "same_sign_without_any_one_database": sign != 0 and all(v * sign > 0 for v in loo.values())}
    hist = {}
    for t in ids:
        k = sum(correct[(a, s)][t] - correct[(b, s)][t] for s in seeds)
        hist[str(k)] = hist.get(str(k), 0) + 1
    return {"a": a, "b": b, "per_seed": {str(s): v for s, v in per_seed.items()}, "pooled": pooled, "threshold": thr,
            "pooled_out_of": len(ids) * len(seeds), "sign": sign, "conditions": cond, "all_conditions": all(cond.values()),
            "per_database": per_db, "leave_one_database_out": loo,
            "per_question_histogram": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
            "exploratory_sign_flip": own_sign_flip([per_db[d] for d in dbs])}


def q1_rule(d: dict) -> str:
    return "I1" if d["all_conditions"] and d["sign"] > 0 else "I2" if d["all_conditions"] and d["sign"] < 0 else "I3"


def q2_rule(d: dict) -> str:
    return "A1" if d["all_conditions"] and d["sign"] > 0 else "A2" if d["all_conditions"] and d["sign"] < 0 else "A3"


def own_reading(t4: str, c4: str, q2: str) -> str:
    """The protocol's 'Reading' bullets; anything else is descriptive."""
    if q2 == "A1":
        return "A1: motivates a separately frozen confirmation on fresh data; does not establish one"
    if t4 == c4 == "I1" and q2 == "A3":
        return "I1 for both with A3: an engineering gain, not attributed to specialization"
    if t4 == c4 == "I3":
        return "I3 for both: diagnose why and recommend a distinct next experiment"
    return f"no protocol reading for T4 {t4}, C4 {c4}, {q2}: descriptive"


def baseline_population(sec: Section, ctx: dict, rel: Release, arm: str, pop: str, ids: list) -> "dict | None":
    """A baseline's records covering a population (re-scored with the own EX rule), or None if any run is missing or
    still running."""
    recs, refs = {}, []
    for tag in BASELINE_TAGS[(arm, pop)]:
        found = baseline_dirs(ctx, f"*_{tag}")
        if len(found) != 1:
            return {"status": "not yet run" if not found else f"{len(found)} runs tagged {tag}"}
        d = found[0]
        if not (d / "summary.json").exists():
            n = len(jl(d / "records.jsonl")) if (d / "records.jsonl").exists() else 0
            return {"status": f"not yet complete ({d.name}: no summary.json, {n} records so far)"}
        man = json.loads((d / "manifest.json").read_text())
        sec.check(man["git"].get("research_tree_dirty") is False and sha256_bytes(canonical(man["config"]).encode()) == man["config_identity"],
                  f"{arm} {pop} {d.name}: clean tree and config identity")
        refs.append(d.name)
        for r in jl(d / "records.jsonl"):
            recs[r["task_id"]] = r
    sec.check(sorted(recs) == sorted(ids), f"{arm} {pop}: the runs cover exactly the population", f"{len(recs)} of {len(ids)}")
    if sorted(recs) != sorted(ids):
        return {"status": "incomplete coverage", "runs": refs}
    if rel is None:
        own = {t: recs[t]["score"]["correct"] for t in ids}
        incons = [f"{t}: {p_}" for t in ids for p_ in score_problems(recs[t])]
        sec.check(not incons, f"{arm} {pop}: recorded scores internally consistent (public)", f"{incons[:5]}")
    else:
        cache = rescore(ctx, rel, list(recs.values()))
        own = {t: cache[ex_key(recs[t])][0] for t in ids}
        mism = [t for t in ids if own[t] != recs[t]["score"]["correct"]]
        sec.check(not mism, f"{arm} {pop}: recorded scores re-derived with the own EX rule", f"{mism}")
    return {"status": "complete", "runs": refs, "correct": sum(own.values()), "per_question": own,
            "abstained": sum(recs[t]["final"]["kind"] == "abstain" for t in ids),
            "time_to_submission_seconds": sum(recs[t]["elapsed_seconds"] for t in ids), "_records": [recs[t] for t in ids]}


# --- the analysis.json mirror: each field re-derived from the records, the own scores and the ledger ------------------
def own_serving(records: list, per_q: "dict | None", flat: "list | None" = None) -> dict:
    """Time to SQL submission (record elapsed) and the scorer's execution of the submitted query (recorded
    score.predicted), separately and together; EX-correct from the own scores; timeouts as recorded by the scorer."""
    ok = sum(flat) if flat is not None else sum(per_q[r["task_id"]] for r in records)
    gen = sum(r["elapsed_seconds"] for r in records)
    exe = sum(((r["score"].get("predicted") or {}).get("elapsed_seconds") or 0.0) for r in records)
    return {"requests": len(records), "ex_correct": ok, "time_to_submission_seconds": gen,
            "submitted_execution_seconds": exe, "together_seconds": gen + exe,
            "mean_time_to_submission": gen / len(records) if records else None,
            "timeouts": sum((r["score"].get("predicted") or {}).get("status") == "timeout" for r in records),
            "abstentions": sum(r["final"]["kind"] == "abstain" for r in records),
            "generation_per_ex_correct": gen / ok if ok else None,
            "together_per_ex_correct": (gen + exe) / ok if ok else None}


def own_macs(records: list) -> dict:
    """Mean active MACs per request (backbone request MACs + the controller's trunk, semantic and pointer MACs, as
    recorded) and mean decode steps. Re-derives the aggregation only; the per-request MAC counts are the producer's."""
    import statistics
    per = [r["controller_work"]["request_macs"] + sum(r["controller_work"]["macs"].values()) for r in records
           if r.get("controller_work") and "macs" in r["controller_work"]]
    steps = [r["controller_work"]["steps"] for r in records if r.get("controller_work")]
    return {"active_macs_per_request_mean": statistics.mean(per) if per else None,
            "decode_steps_mean": statistics.mean(steps) if steps else None}


PRODUCER_STRATA = """
import json, sys
from scaling import data, ir, sql_to_ir
from scaling import schema as schema_mod
from scaling import perf_analysis as A
out = {}
for t in json.loads(sys.stdin.read()):
    index = ir.SchemaIndex.from_schema(schema_mod.load_schema(data.task(t).db_id))
    plan, err = sql_to_ir.try_convert(data.gold(t).sql, index)
    out[t] = A.strata(plan if err is None else None)
print("STRATA " + json.dumps(out))
"""


def producer_strata(ctx: dict, worktree: Path, commit: str, ids: list) -> tuple:
    """FLAGGED PRODUCER CODE: the per-question semantic strata of the gold programs, from scaling.perf_analysis.strata
    over scaling.sql_to_ir (the gold-program conversion is the producer's grammar, not re-implemented here). Run in a
    separate process from the frozen worktree at the analysis commit, read-only (-B, no bytecode written)."""
    head = git(worktree, "rev-parse", "HEAD").strip()
    dirty = git(worktree, "status", "--porcelain", "--", "research/ti_tnn").strip()
    if head != commit or dirty:
        return None, f"worktree {worktree} at {head} (want {commit}), dirty: {dirty[:200]}"
    env = dict(os.environ, PYTHONPATH=str(worktree / "research" / "ti_tnn"), PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES="", TI_TNN_SCALING_RUNS=str(ctx["runs"]))
    res = subprocess.run([sys.executable, "-B", "-c", PRODUCER_STRATA], input=json.dumps(ids), capture_output=True,
                         text=True, env=env, cwd=ctx["out"])
    line = [x for x in res.stdout.splitlines() if x.startswith("STRATA ")]
    if res.returncode or not line:
        return None, f"exit {res.returncode}: {res.stderr[-600:]}"
    return json.loads(line[-1][len("STRATA "):]), None


def strata_tally(strata_of: dict, correct: dict, seeds: list, ids: list) -> dict:
    out = {}
    for t in ids:
        for dim, value in strata_of[t].items():
            row = out.setdefault(dim, {}).setdefault(value, {"questions": 0, **{lab: 0 for lab in PERF_LABELS}})
            row["questions"] += 1
            for lab in PERF_LABELS:
                row[lab] += sum(correct[(lab, s)][t] for s in seeds)
    return out


def training_mirror(ctx: dict, seeds: list) -> dict:
    import statistics
    out = {}
    for lab in PERF_LABELS:
        for s in seeds:
            rd = perf_root(ctx) / "train" / f"{lab}_s{s}"
            logs = jl(rd / "train_log.jsonl")
            atts = [json.loads((a / "summary.json").read_text()) for a in sorted((rd / "attempts").glob("*"))
                    if (a / "summary.json").exists()]
            cfg = json.loads((rd / "config.json").read_text())
            out[f"{lab}/{s}"] = {"updates": len(logs), "loss_first_last": [logs[0]["loss"], logs[-1]["loss"]],
                                 "seconds_per_update_mean": statistics.mean(r["wall_seconds"] for r in logs),
                                 "reservation_hours": sum(a["reservation"]["hours"] for a in atts), "attempts": len(atts),
                                 "peak_memory_bytes": max((a.get("peak_memory_bytes") or 0) for a in atts) if atts else None,
                                 "parameters": cfg["parameters"]}
    return out


def ledger_at(path: Path, t_cut: float) -> dict:
    """The ledger's usage as it stood at wall-clock time t_cut (ledger.py's rule: ended = its elapsed; open = last
    heartbeat + 30 s)."""
    res = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e["t"] > t_cut:
            continue
        r = res.setdefault(e["reservation"], {"elapsed": 0.0, "ended": False})
        if e["event"] in ("heartbeat", "end"):
            r["elapsed"] = max(r["elapsed"], float(e.get("elapsed", 0.0)))
        if e["event"] == "end":
            r["ended"] = True
    used = sum((r["elapsed"] if r["ended"] else r["elapsed"] + HEARTBEAT_SECONDS) for r in res.values()) / 3600.0
    return {"used_hours": used, "open": [k for k, r in res.items() if not r["ended"]]}


def leaf_equal(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def deep_diff(theirs, mine, path="", out=None) -> dict:
    out = out if out is not None else {"compared": 0, "differ": [], "only_in_analysis": [], "only_in_mine": []}
    if isinstance(theirs, dict) and isinstance(mine, dict):
        for k in theirs:
            if k not in mine:
                out["only_in_analysis"].append(f"{path}.{k}")
            else:
                deep_diff(theirs[k], mine[k], f"{path}.{k}", out)
        out["only_in_mine"] += [f"{path}.{k}" for k in mine if k not in theirs]
    elif isinstance(theirs, list) and isinstance(mine, list):
        if len(theirs) != len(mine):
            out["differ"].append([path, f"length {len(theirs)}", f"length {len(mine)}"])
        for i, (x, y) in enumerate(zip(theirs, mine)):
            deep_diff(x, y, f"{path}[{i}]", out)
    else:
        out["compared"] += 1
        if not leaf_equal(theirs, mine):
            out["differ"].append([path, theirs, mine])
    return out


READING_KEYWORDS = {"A1": ("confirmation",), "I1 for both with A3": ("engineering gain", "specialization"),
                    "I3 for both": ("diagnose",)}


@section
def sec_compare(sec: Section, ctx: dict) -> None:
    if "perf_eval" not in ctx:
        sub = Section("eval (for compare)")
        sec_eval(sub, ctx)
        sec.check(not any(not c["passed"] for c in sub.checks), "the eval section (re-run for compare) passed",
                  f"{[c['name'] for c in sub.checks if not c['passed']][:5]}")
    pe = ctx["perf_eval"]
    correct, db_of, rel = pe["correct"], pe["db_of"], pe["rel"]
    seeds = sorted(pe["chosen"])
    sec.check(seeds == list(PERF_SEEDS), "all three seeds have a counted evaluation (no symmetric reduction)", f"{seeds}")
    an_dirs = sorted((perf_root(ctx) / "analysis").glob("*/analysis.json")) if (perf_root(ctx) / "analysis").exists() else []
    an = json.loads(an_dirs[-1].read_text()) if an_dirs else None
    sec.info["analysis_json"] = str(an_dirs[-1].relative_to(perf_root(ctx).parent)) if an_dirs else "not yet produced (perf-analyze has not run)"
    if an_dirs:
        sec.check(len(an_dirs) == 1, "exactly one analysis.json", f"{[str(p) for p in an_dirs]}")
    out, mirror_pops, strata_ids = {}, {}, {}
    for pop, ids in (("historical", pe["hist"]), ("new", pe["new"])):
        counts = {f"{lab}/{s}": sum(correct[(lab, s)][t] for t in ids) for lab in PERF_LABELS for s in seeds}
        q1 = {arch: own_paired(correct, f"{arch}-v2", f"{arch}-v1", seeds, ids, db_of) for arch in ("T4", "C4")}
        q2 = own_paired(correct, "T4-v2", "C4-v2", seeds, ids, db_of)
        q2v1 = own_paired(correct, "T4-v1", "C4-v1", seeds, ids, db_of)
        for arch in q1:
            q1[arch]["rule"] = q1_rule(q1[arch])
        q2["rule"], q2v1["rule"] = q2_rule(q2), q2_rule(q2v1)
        base = {arm: baseline_population(sec, ctx, rel, arm, pop, ids) for arm in ("L", "B4", "Q27")}
        dbs = sorted({db_of[t] for t in ids})
        table = {}
        for d in dbs:
            qs = [t for t in ids if db_of[t] == d]
            row = {"questions": len(qs), **{f"{lab}/{s}": sum(correct[(lab, s)][t] for t in qs) for lab in PERF_LABELS for s in seeds}}
            for arm, b in base.items():
                if b and b.get("status") == "complete":
                    row[arm] = sum(b["per_question"][t] for t in qs)
            table[d] = row
        o = {"questions": len(ids), "counts": counts,
             "pooled_counts": {lab: sum(counts[f"{lab}/{s}"] for s in seeds) for lab in PERF_LABELS},
             "q1": q1, "q2": q2, "q2_v1_descriptive": q2v1,
             "reading": own_reading(q1["T4"]["rule"], q1["C4"]["rule"], q2["rule"]),
             "per_database": table,
             "baselines": {arm: ({k: v for k, v in b.items() if k not in ("per_question", "_records")} if b else None)
                           for arm, b in base.items()}}
        out[pop] = o
        # --- this population's half of the analysis.json mirror (from the own scores and the records) ----------------
        recs_pop = {key: {r["task_id"]: r for r in rr} for key, rr in pe["records"].items()}
        cor = lambda lab, s: {t: correct[(lab, s)][t] for t in ids}  # noqa: E731
        mirror_pops[pop] = {
            "questions": len(ids), "counts": counts,
            "q1": {arch: {**q1[arch], "rule": q1[arch]["rule"]} for arch in q1}, "q2": q2, "q2_v1_descriptive": q2v1,
            "per_question": {t: {f"{lab}/{s}": correct[(lab, s)][t] for lab in PERF_LABELS for s in seeds} for t in ids},
            "serving": {f"{lab}/{s}": own_serving([recs_pop[(lab, s)][t] for t in ids], cor(lab, s)) for lab in PERF_LABELS for s in seeds},
            "serving_pooled": {lab: own_serving([recs_pop[(lab, s)][t] for s in seeds for t in ids],
                                                None, [correct[(lab, s)][t] for s in seeds for t in ids]) for lab in PERF_LABELS},
            "macs": {lab: own_macs([recs_pop[(lab, s)][t] for s in seeds for t in ids]) for lab in PERF_LABELS},
            "strata": "pending", "baselines": {}, "reading": o["reading"]}
        strata_ids[pop] = ids
        for arm, b in base.items():
            if b and b.get("status") == "complete":
                rr = b["_records"]
                mirror_pops[pop]["baselines"][arm] = {
                    "runs": [f"baselines/{x}" for x in b["runs"]], "correct": b["correct"], "per_question": b["per_question"],
                    "serving": own_serving(rr, b["per_question"]), "tool_calls": sum(r["tool_calls"] for r in rr),
                    "repairs": sum(r["repairs"] for r in rr),
                    "abstention_reasons": {k: sum(1 for r in rr if r["final"]["kind"] == "abstain" and r["final"].get("reason") == k)
                                           for k in sorted({r["final"].get("reason") for r in rr if r["final"]["kind"] == "abstain"})}}
            else:
                mirror_pops[pop]["baselines"][arm] = None

    # --- the pilot's 1,500-update fits on the historical 64, separately (re-scored here) -------------------------------
    pilot_eval = pilot_root(ctx) / "eval"
    pil = {}
    for d in sorted(pilot_eval.glob("s*_*")):
        s = json.loads((d / "manifest.json").read_text())["config"]["seed"]
        for arm in PILOT_ARMS:
            recs = jl(d / arm / "records.jsonl")
            if rel is None:
                bad = [f"{r['task_id']}: {p_}" for r in recs for p_ in score_problems(r)]
                sec.check(not bad, f"pilot {d.name} {arm}: recorded scores internally consistent (public)", f"{bad[:5]}")
                pil[f"{arm}/{s}"] = sum(r["score"]["correct"] for r in recs)
            else:
                cache = rescore(ctx, rel, recs)
                pil[f"{arm}/{s}"] = sum(cache[ex_key(r)][0] for r in recs)
    out["pilot_1500_updates_historical"] = pil
    pan = sorted((ctx["repo"] / HARNESS / "evidence/pilot_v1/analysis").glob("*/analysis.json"))
    if pan:
        pc = json.loads(pan[-1].read_text())["counts"]
        sec.check(all(pc.get(k) == v for k, v in pil.items()), "pilot 1,500-update counts (re-scored here) = the pilot bundle's analysis.json",
                  f"{pil} vs {pc}")
    # the new population's L run: the configuration of the historical L run except positions and requested ids
    lh, ln = baseline_dirs(ctx, "*_L_nonthinking_p0-64"), baseline_dirs(ctx, "*_L_nonthinking_p64-128")
    if len(lh) == 1 and len(ln) == 1:
        a_, b_ = (json.loads((x[0] / "manifest.json").read_text()) for x in (lh, ln))
        dk = sorted(k for k in set(a_["config"]) | set(b_["config"]) if a_["config"].get(k) != b_["config"].get(k))
        sec.check(dk == ["positions", "requested"] and b_["config"]["positions"] == [64, 128] and b_["config"]["requested"] == pe["new"],
                  "L on the new 64: configuration = the historical L run's except positions [64,128) and the new ids", f"{dk}")

    # --- every field of analysis.json against the mirror ---------------------------------------------------------------
    if an is None:
        out["analysis_comparison"] = "analysis.json not yet produced"
        sec.info.update(out)
        return
    an_dir = an_dirs[-1].parent
    an_man = json.loads((an_dir / "manifest.json").read_text())
    sec.check(an_man["git"]["research_tree_dirty"] is False and not an_man["git"]["dirty_paths"], "analysis: run from a clean tree",
              f"{an_man['git']}")
    sec.check(an_man["git"]["commit"] == ctx.get("train_commit", an_man["git"]["commit"]), "analysis: at the commit of the fits",
              an_man["git"]["commit"])
    # strata: locally, the producer's classification (flagged), tallied here with the own scores; in public mode the
    # classification needs the databases, so only the tallies' marginal sums are checked
    if not ctx["public"]:
        strata_of, why = producer_strata(ctx, Path(an_man["git"]["worktree"]), an_man["git"]["commit"], pe["hist"] + pe["new"])
        flagged = ctx.setdefault("producer_modules_used", [])
        flagged.append("scaling.perf_analysis.strata + scaling.sql_to_ir/ir/schema/data (gold-program strata classification only; "
                       "subprocess, frozen worktree)")
        if sec.check(strata_of is not None, "producer strata classification ran (flagged)", why or ""):
            for pop, ids in strata_ids.items():
                mirror_pops[pop]["strata"] = strata_tally(strata_of, correct, seeds, ids)
    else:
        for pop, ids in strata_ids.items():
            st_ = ((an.get("populations") or {}).get(pop) or {}).get("strata") or {}
            pooled = {lab: sum(correct[(lab, s)][t] for s in seeds for t in ids) for lab in PERF_LABELS}
            bad = []
            for dim, vals in st_.items():
                q = sum(v["questions"] for v in vals.values())
                if dim == "tables" and q != len(ids) or dim != "tables" and q > len(ids):
                    bad.append(f"{dim}: {q} questions")
                full = dim == "tables" or q == len(ids)
                for lab in PERF_LABELS:
                    tot = sum(v[lab] for v in vals.values())
                    if (full and tot != pooled[lab]) or tot > pooled[lab] or any(v[lab] > 3 * v["questions"] for v in vals.values()):
                        bad.append(f"{dim} {lab}: {tot} vs pooled {pooled[lab]}")
            sec.check(not bad and st_, f"{pop} strata: marginal sums consistent with the own counts (public; classification not re-derived)",
                      f"{bad[:6]}")
            mirror_pops[pop]["strata"] = st_   # not re-derivable without the databases; excluded from the leaf comparison
        sec.info["strata_note"] = "public mode: strata classification needs the databases; marginal sums checked only"
    pin = json.loads((ctx["repo"] / PIN_DIR / "perf_v2_protocol.json").read_text())
    t_cut = datetime.strptime(an_dir.name, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
    led = ledger_at(ledger_file(ctx), t_cut + 600)   # usage() runs at the end of the analysis
    led_early = ledger_at(ledger_file(ctx), t_cut)
    rs_all, _p = parse_ledger(ledger_file(ctx))
    named = sorted({str(r["start"].get("ledger")) for r in rs_all})
    never_ended = sorted(r["id"] for r in rs_all if r["open"] and r["t_start"] < t_cut)
    sec.check(abs(led["used_hours"] - led_early["used_hours"]) < 1e-12 and led["open"] == led_early["open"]
              and sorted(led["open"]) == never_ended,
              "ledger: usage identical at the analysis start and 10 min later; the only unended reservations are ones that never ended",
              f"{led_early} vs {led}; never ended {never_ended}")
    mirror = {"protocol": {k: pin[k] for k in ("file", "sha256", "frozen_at", "updates")}, "seeds": seeds,
              "threshold": THRESHOLD_THREE_SEEDS, "populations": mirror_pops,
              "inputs": {"evaluations_counted": {str(s): f"perf_v2/eval/{n}" for s, n in pe["chosen"].items()},
                         "evaluations_all": [{"passed": r["eligible"], "run": r["run"], "seed": r["seed"]} for r in pe["all_runs"]]},
              "pilot_1500_updates_historical": {f"{'C4_params' if k.startswith('c4_params') else 'T4'}/{k.split('/')[1]}": v for k, v in pil.items()},
              "training": training_mirror(ctx, seeds),
              "ledger": {"authorization": 2, "cap_hours": 32.0, "ledger": named[0] if len(named) == 1 else named,
                         "used_hours": led["used_hours"], "remaining_hours": 32.0 - led["used_hours"]}}
    # the rules: compare the code; the readings: by the protocol's bullet
    theirs = json.loads(json.dumps(an))
    rule_texts = {}
    for pop, ap in (theirs.get("populations") or {}).items():
        for key in ("q2", "q2_v1_descriptive"):
            if isinstance(ap.get(key), dict) and "rule" in ap[key]:
                rule_texts[f"{pop}.{key}"] = ap[key]["rule"]
                ap[key]["rule"] = str(ap[key]["rule"]).split(":")[0]
        for arch, d in (ap.get("q1") or {}).items():
            rule_texts[f"{pop}.q1.{arch}"] = d.get("rule")
            d["rule"] = str(d.get("rule")).split(":")[0]
        reading = ap.pop("reading", None)
        mine_reading = mirror_pops.get(pop, {}).pop("reading", "")
        cat = next((k for k in READING_KEYWORDS if mine_reading.startswith(k)), None)
        ok = cat is not None and all(w in str(reading) for w in READING_KEYWORDS[cat])
        sec.check(ok, f"{pop}: analysis.json's reading matches the protocol's reading for the own rules ({mine_reading.split(':')[0]})",
                  f"analysis: {reading!r}")
    sec.info["analysis_rule_texts"] = rule_texts
    if ctx["public"]:   # the strata need the databases: marginal sums checked above, leaves not compared
        for pop in mirror_pops:
            (theirs.get("populations") or {}).get(pop, {}).pop("strata", None)
            mirror_pops[pop].pop("strata", None)
    diff = deep_diff(theirs, mirror)
    groups = sorted({p.split(".")[1] if not p.startswith(".populations") else ".".join(p.split(".")[1:4]).split("[")[0]
                     for p in [x[0] for x in diff["differ"]] + diff["only_in_analysis"]})
    top = ["protocol", "seeds", "threshold", "inputs", "pilot_1500_updates_historical", "training", "ledger"] + \
          [f"populations.{pop}.{k}" for pop in ("historical", "new") for k in
           ("questions", "counts", "q1", "q2", "q2_v1_descriptive", "per_question", "serving", "serving_pooled", "macs", "baselines")
           + ((("strata",)) if not ctx["public"] else ())]
    for g in top:
        bad = [x for x in diff["differ"] if x[0] == f".{g}" or x[0].startswith(f".{g}.") or x[0].startswith(f".{g}[")]
        missing = [x for x in diff["only_in_analysis"] if x == f".{g}" or x.startswith(f".{g}.")]
        sec.check(not bad and not missing, f"analysis.json {g}: every field equals the own re-derivation",
                  f"{len(bad)} differ, first {bad[:4]}; not re-derived {missing[:4]}")
    other = [x for x in diff["only_in_analysis"] if not any(x == f".{g}" or x.startswith(f".{g}.") for g in top)]
    sec.check(not other, "analysis.json has no field outside the groups re-derived here", f"{other[:8]}")
    out["analysis_comparison"] = {"file": str(an_dirs[-1].relative_to(perf_root(ctx).parent)), "leaves_compared": diff["compared"],
                                  "differ": diff["differ"][:50], "differ_count": len(diff["differ"]),
                                  "not_re_derived": diff["only_in_analysis"], "only_in_mirror": diff["only_in_mine"][:20],
                                  "groups_with_differences": groups}
    # the attempt summaries' reservation hours against the ledger charges
    rs, _p = parse_ledger(ledger_file(ctx))
    by_cmd = {r["start"]["command"]: r for r in rs}
    far = {k: (v["reservation_hours"], by_cmd.get(f"perf-train {k.split('/')[0]} s{k.split('/')[1]}", {}).get("hours"))
           for k, v in mirror["training"].items()}
    sec.check(all(b is not None and abs(a - b) < 60 / 3600 for a, b in far.values()),
              "training reservation_hours (attempt summaries) within 60 s of the ledger charges", f"{far}")
    sec.info.update(out)


# === public mode: the committed bundles only (no runs root, no databases, no checkpoints) ===========================
BUNDLES = ("perf_v2", "dev_v1", "pilot_v1")


def evidence_dir(ctx) -> Path:
    return ctx["repo"] / HARNESS / "evidence"


@section
def sec_bundles(sec: Section, ctx: dict) -> None:
    """Each committed bundle against its SHA256SUMS; no file listed at --unchanged-since changed since; the bundled
    ledgers grow by appending only."""
    repo, ev = ctx["repo"], evidence_dir(ctx)
    since = ctx["unchanged_since"]
    for name in BUNDLES:
        b = ev / name
        sums = read_sums(b)
        bad = [p for h, p in sums if not (b / p).is_file() or sha256_path(b / p) != h]
        listed = {p for _h, p in sums}
        unlisted = sorted(str(x.relative_to(b)) for x in b.rglob("*") if x.is_file() and x.name != "SHA256SUMS"
                          and str(x.relative_to(b)) not in listed)
        sec.check(not bad and not unlisted, f"evidence/{name}: every file matches SHA256SUMS, and every file is listed",
                  f"mismatched {bad[:5]}, unlisted {unlisted[:5]}")
        sec.info[f"{name}_files"] = len(sums)
        try:
            old = [x.split("  ", 1) for x in git_show(repo, since, f"{HARNESS}evidence/{name}/SHA256SUMS").decode().splitlines() if x.strip()]
        except subprocess.CalledProcessError:
            sec.check(False, f"evidence/{name}: SHA256SUMS readable at {since}")
            continue
        now = {p: h for h, p in sums}
        changed = [p for h, p in old if now.get(p) != h]
        sec.check(not changed, f"evidence/{name}: every file listed in SHA256SUMS at {since} is listed and hashes the same now",
                  f"{changed[:8]}")
        sec.info[f"{name}_added_since_{since}"] = sorted(set(now) - {p for _h, p in old})
    status = git(repo, "diff", "--name-status", since, "HEAD", "--", HARNESS + "evidence").splitlines()
    other = [x for x in status if not (x.startswith("A\t") or (x.startswith("M\t") and x.endswith("/SHA256SUMS")))]
    sec.check(not other, f"git: since {since} the evidence only gained files (A) and rewrote SHA256SUMS (M)", f"{other[:8]}")
    sec.info[f"evidence_changes_since_{since}"] = status
    dirty = git(repo, "status", "--porcelain", "--untracked-files=all", "--", HARNESS + "evidence").strip()
    sec.check(dirty == "", "no uncommitted or untracked change in the evidence", dirty[:300])
    chain = [ev / n / "gpu_ledger.jsonl" for n in ("dev_v1", "pilot_v1", "perf_v2")]
    blobs = [c.read_bytes() for c in chain]
    sec.check(blobs[1].startswith(blobs[0]) and blobs[2].startswith(blobs[1]),
              "bundled ledgers: dev_v1 is a byte prefix of pilot_v1, which is a byte prefix of perf_v2 (append-only)",
              f"{[len(x) for x in blobs]} bytes")


@section
def sec_pins_public(sec: Section, ctx: dict) -> None:
    """The pins, from the checkout alone (the raw-release rebuild and the gold SQL need the runs root: section pins)."""
    repo = ctx["repo"]
    pins = repo / PIN_DIR
    proto = json.loads((pins / "perf_v2_protocol.json").read_text())
    evb, trb = (pins / "perf_v2_eval.json").read_bytes(), (pins / "perf_v2_train.json").read_bytes()
    ev, tr = json.loads(evb), json.loads(trb)
    invb, ptrb = (pins / "inventory_v1.json").read_bytes(), (pins / "pilot_train_v1.json").read_bytes()
    inv, ptr = json.loads(invb), json.loads(ptrb)
    sec.check((sha256_path(repo / proto["file"]), sha256_bytes(trb), sha256_bytes(evb))
              == (proto["sha256"], proto["train_pin_sha256"], proto["eval_pin_sha256"]), "protocol pin: protocol, train and eval pins hash to it")
    at = tuple(sha256_bytes(git_show(repo, proto["frozen_at"], x)) for x in (proto["file"], PIN_DIR + "/perf_v2_train.json", PIN_DIR + "/perf_v2_eval.json"))
    sec.check(at == (proto["sha256"], proto["train_pin_sha256"], proto["eval_pin_sha256"]), "protocol pin: the three files at frozen_at hash to it")
    since = git(repo, "log", "--format=%h %s", f"{proto['frozen_at']}..HEAD", "--", proto["file"], PIN_DIR + "/perf_v2_train.json",
                PIN_DIR + "/perf_v2_eval.json").strip()
    sec.check(since == "", "protocol, train and eval pins unchanged by any commit after frozen_at", since)
    sec.check(sha256_bytes(invb) == ev["inventory_pin_sha256"] and sha256_bytes(ptrb) == tr["pilot_train_pin_sha256"],
              "eval and train pins name inventory_v1 and pilot_train_v1 by sha256")
    h, n = ev["populations"]["historical"], ev["populations"]["new"]
    sec.check(h["positions"] == [0, 64] and n["positions"] == [64, 128] and inv["task_ids"][:64] == h["task_ids"]
              and inv["task_ids"][64:128] == n["task_ids"], "historical = inventory positions 0-63, new = 64-127, in order")
    sec.check(h["task_ids"] == pilot_64(repo), "historical = the pilot's 64 (evidence/pilot_v1 evaluation manifests)")
    sec.check(not set(h["task_ids"]) & set(n["task_ids"]) and len(set(h["task_ids"] + n["task_ids"])) == 128,
              "the populations: 128 distinct ids, no overlap")
    sec.check(len(h["db_ids"]) == len(n["db_ids"]) == 64, "each population lists a database per question")
    expect = [t for t in ptr["task_ids"] if t != DROPPED]
    sec.check(tr["task_ids"] == expect and sorted(tr["dropped"]) == [DROPPED] and (tr["pilot_questions"], tr["kept"]) == (4741, 4740),
              f"perf_v2_train = pilot_train_v1 minus exactly {DROPPED}, in order")
    leak = sorted((set(h["task_ids"]) | set(n["task_ids"])) & (set(tr["task_ids"]) | set(ptr["task_ids"])))
    sec.check(not leak, "no evaluation id in either training pin", f"{leak}")
    sec.info["not_checked_public"] = ("the inventory rebuilt from the raw release, its gold-execution exclusions and "
                                      f"{DROPPED}'s gold SQL need the runs root (local section pins)")


# --- executed controller MACs, re-derived here ------------------------------------------------------------------------
def producer_literals(repo: Path, commit: str) -> dict:
    """Constants of the producer's grammar and controller, read as literals with ast (never imported)."""
    out = {}
    for f, names in (("actions.py", ("POINTER_RELATIONS", "N_CLASSES", "N_FEATURES")), ("grounding.py", ("ITEM_FEATURES",)),
                     ("controller.py", ("N_POINTERS",))):
        for node in ast.parse(git_show(repo, commit, HARNESS + f)).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id in names:
                out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def own_executed_macs(rec: dict, ctl: dict, grammar: dict, K: dict) -> dict:
    """One saved request (batch size one): N recovered from the recorded pointer MACs (or the trunk), every recorded
    term re-derived, and the executed MACs of decode_batch:
    - each operator step: trunk + semantic module + pointer_scores call 1 (item scores for the table and column
      queries 2 N d, span start T d, under v2 the relation terms 2 R d + 2 N R);
    - each span operator: pointer_scores call 2 (item scores 2 N d, start T d, end condition d^2, end T d; no
      relations are passed);
    - each STOP step: the trunk only (the loop stops before the semantic module);
    - once: request_macs (memory, keys, initial state, v2 item features and schema encoder)."""
    w = rec["controller_work"]
    d, gl = ctl["d_model"], ctl["gru_layers"]
    R, v2 = len(K["POINTER_RELATIONS"]), ctl["grounding"] == "v2"
    acts = rec["attempts"][0]["actions"] or []
    stops = w["families"].count("STOP")
    ops = w["steps"] - stops
    T = int(rec["tokens"]["context"])
    kind = ["none" if a[1] is None else "span" if isinstance(a[1], (list, tuple)) else "item" for a in acts]
    n_item, n_span = kind.count("item"), kind.count("span")
    prob = []
    if w.get("batch_size", 1) != 1:
        prob.append("batch size")
    if ops != len(acts):
        prob.append(f"{ops} operator steps for {len(acts)} actions (a dead-end step would be uncharged)")
    if stops > 1 or (stops == 1 and w["families"][-1] != "STOP"):
        prob.append("STOP not last")
    width = sum(ctl[k] for k in ("op_embed", "family_embed", "slot_embed", "parent_embed", "feature_embed", "arg_embed"))
    C = K["N_FEATURES"] * ctl["feature_embed"] + d * ctl["arg_embed"] + width * d + gl * 6 * d * d + 4 * d * d + d * K["N_CLASSES"]
    trunk_step, rem = divmod(w["macs"]["trunk"], ops) if ops else (None, 0)
    N = None
    if n_item:
        num = w["macs"]["pointer"] - n_span * (2 * T * d + d * d) - (n_item * R * d if v2 else 0)
        N, rem2 = divmod(num, n_item * (d + (R if v2 else 0)))
        if rem2:
            prob.append("pointer MACs give no integral N")
    if trunk_step is not None:
        if rem:
            prob.append("trunk MACs not a multiple of the operator steps")
        L, rem3 = divmod(trunk_step - C, 2 * d)
        if rem3:
            prob.append("trunk per step - constant not a multiple of 2d")
        if N is None:
            N = L - T
        elif L - T != N:
            prob.append(f"N from the pointer ({N}) != N from the trunk ({L - T})")
    if N is None:
        return {"problems": prob + ["N not recoverable"]}
    # the recorded terms, re-derived
    ptr_rec = n_item * (N * d + ((R * d + N * R) if v2 else 0)) + n_span * (2 * T * d + d * d)
    if ptr_rec != w["macs"]["pointer"]:
        prob.append("recorded pointer MACs not re-derived")
    P = K["N_POINTERS"] * d
    if ctl["arm"] == "T4":
        sem = sum(d * ctl["head_hidden"] + ctl["head_hidden"] * (grammar["family_vocabulary_sizes"][f] + P)
                  for f in w["families"] if f != "STOP")
    else:
        h = ctl["c4_params_hidden"]
        sem = ops * (d * h + h * (grammar["union_size"] + P))
    if sem != w["macs"]["semantic"]:
        prob.append(f"recorded semantic MACs {w['macs']['semantic']} != own {sem}")
    D = ctl["backbone_width"]
    base_req = (T + N) * D * d + (T + N) * d * d + 2 * (T + N) * d * d + D * gl * d
    if not v2 and base_req != w["request_macs"]:
        prob.append("v1 request MACs not re-derived")
    n_qe = None
    if v2:   # solve the question/evidence key count from the schema encoder's terms; it must be a whole number >= 0
        extra = w["request_macs"] - base_req - N * len(K["ITEM_FEATURES"]) * d
        per_layer0 = N * d * d + 2 * N * d * d + 2 * N * N * d + N * d * d + 2 * N * d * ctl["encoder_ff"]
        n_qe, rem4 = divmod(extra - ctl["encoder_layers"] * per_layer0, ctl["encoder_layers"] * (2 * d * d + 2 * N * d))
        if rem4 or n_qe < 0:
            prob.append(f"v2 request MACs give no whole question/evidence count ({n_qe}, remainder {rem4})")
    recorded = w["request_macs"] + sum(w["macs"].values())
    call1 = 2 * N * d + T * d + ((2 * R * d + 2 * N * R) if v2 else 0)
    call2 = 2 * N * d + T * d + d * d + T * d
    executed = (w["request_macs"] + (ops + stops) * (C + 2 * (T + N) * d) + w["macs"]["semantic"]
                + ops * call1 + n_span * call2)
    return {"problems": prob, "T": T, "N": N, "n_qe": n_qe, "ops": ops, "stops": stops, "spans": n_span,
            "recorded": recorded, "executed": executed}


def report_mac_table(repo: Path) -> dict:
    """The corrected report's table 'Controller MACs per request: as recorded and as executed', parsed as data."""
    text = (repo / HARNESS / "docs" / "REPORT_perf_v2.md").read_text()
    out = {}
    for m in re.finditer(r"^\| (historical|new) \| ([TC]4-v[12]) \| (\d+) \| ([\d.]+) G \| ([\d.]+) G \| \+([\d.]+)% \|$", text, re.M):
        out[(m.group(1), m.group(2))] = {"requests": int(m.group(3)), "recorded_G": float(m.group(4)),
                                         "executed_G": float(m.group(5)), "difference_pct": float(m.group(6))}
    return out


@section
def sec_macs(sec: Section, ctx: dict) -> None:
    """Executed controller MACs per request, re-derived with this file's own reading of decode_batch and
    pointer_scores, from the records and configs alone, and compared with the corrected report's table."""
    import statistics
    repo = ctx["repo"]
    proto = json.loads((repo / PIN_DIR / "perf_v2_protocol.json").read_text())
    K = producer_literals(repo, proto["frozen_at"])
    sec.info["literals"] = {"R": len(K["POINTER_RELATIONS"]), "N_ITEM_FEATURES": len(K["ITEM_FEATURES"]),
                            **{k: K[k] for k in ("N_CLASSES", "N_FEATURES", "N_POINTERS")}}
    hist, new = eval_population_ids(repo)
    pops = {"historical": set(hist), "new": set(new)}
    evals = {}
    for d in sorted((perf_root(ctx) / "eval").glob("s*_*")):
        summ = json.loads((d / "summary.json").read_text())
        if summ.get("passed") is True:
            evals.setdefault(json.loads((d / "manifest.json").read_text())["config"]["seed"], []).append(d)
    sec.check(sorted(evals) == list(PERF_SEEDS) and all(len(v) == 1 for v in evals.values()), "one passing evaluation per seed")
    rows, problems, n_of_db, consts = {}, [], {}, {}
    an = sorted((perf_root(ctx) / "analysis").glob("*/analysis.json"))
    an = json.loads(an[-1].read_text()) if an else None
    for s, (d,) in sorted(evals.items()):
        for lab in PERF_LABELS:
            cfg = json.loads((perf_root(ctx) / "train" / f"{lab}_s{s}" / "config.json").read_text())
            for r in jl(d / lab / "records.jsonl"):
                x = own_executed_macs(r, cfg["controller"], cfg["grammar"], K)
                if x["problems"]:
                    problems.append(f"s{s} {lab} {r['task_id']}: {x['problems']}")
                    continue
                n_of_db.setdefault((r["db_id"], cfg["controller"]["grounding"]), set()).add(x["N"])
                consts.setdefault((r["task_id"], cfg["controller"]["grounding"]), set()).add((x["N"], x["n_qe"]))
                pop = "historical" if r["task_id"] in pops["historical"] else "new"
                rows.setdefault((pop, lab), []).append(x)
    sec.check(not problems, "every request's recorded trunk, semantic, pointer and request MACs re-derived from (T, N) "
              "and the config; no dead-end or uncharged step", f"{len(problems)}: {problems[:6]}")
    sec.check(all(len(v) == 1 for v in n_of_db.values()), "one schema-item count N per database (all arms, seeds, both groundings)",
              f"{[(k, v) for k, v in n_of_db.items() if len(v) != 1][:5]}")
    ng = {}
    for (db, g), v in n_of_db.items():
        ng.setdefault(db, set()).update(v)
    sec.check(all(len(v) == 1 for v in ng.values()), "the same N per database under v1 and v2")
    sec.check(all(len(v) == 1 for v in consts.values()), "one (N, question/evidence count) per question across arms and seeds")
    table, rep = {}, report_mac_table(repo)
    for (pop, lab), xs in sorted(rows.items()):
        rec, exe = statistics.mean(x["recorded"] for x in xs), statistics.mean(x["executed"] for x in xs)
        table[f"{pop}/{lab}"] = {"requests": len(xs), "recorded_mean": rec, "executed_mean": exe,
                                 "difference_pct": 100 * (exe / rec - 1), "stop_steps": sum(x["stops"] for x in xs),
                                 "span_ops": sum(x["spans"] for x in xs)}
        if an is not None:
            am = (((an.get("populations") or {}).get(pop) or {}).get("macs") or {}).get(lab) or {}
            sec.check(leaf_equal(am.get("active_macs_per_request_mean"), rec), f"{pop} {lab}: recorded mean = analysis.json macs",
                      f"{am.get('active_macs_per_request_mean')} vs {rec}")
        rr = rep.get((pop, lab))
        sec.check(rr is not None and rr["requests"] == len(xs) and f"{rec / 1e9:.3f}" == f"{rr['recorded_G']:.3f}"
                  and f"{exe / 1e9:.3f}" == f"{rr['executed_G']:.3f}" and f"{100 * (exe / rec - 1):.2f}" == f"{rr['difference_pct']:.2f}",
                  f"{pop} {lab}: own executed MACs = the corrected report's table (3 decimals in G, 2 in %)",
                  f"own {rec / 1e9:.4f} G -> {exe / 1e9:.4f} G (+{100 * (exe / rec - 1):.3f}%), report {rr}")
    sec.info["executed_macs"] = table
    sec.info["reading"] = ("re-derived from controller.step_macs / request_macs (trunk, semantic, pointer terms), "
                           "controller.pointer_scores and infer_controller.decode_batch at batch size one; the backbone's "
                           "forward is in neither count")


LOCAL_SECTIONS = ("pins", "history", "ledger", "baselines", "train", "eval", "compare", "macs")
PUBLIC_SECTIONS = ("bundles", "pins_public", "ledger", "train", "eval", "compare", "macs")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sections", nargs="*", default=None)
    ap.add_argument("--public", action="store_true",
                    help="bundle-only mode: the checkout's pins and committed evidence; no runs root, databases or checkpoints")
    ap.add_argument("--repo", type=Path, default=HERE.parents[3], help="the checkout (default: the one holding this file)")
    ap.add_argument("--runs-root", type=Path, default=None,
                    help="the shared runs directory (local mode; default: $TI_TNN_SCALING_RUNS, no machine default)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory (default: ./verify_perf_out)")
    ap.add_argument("--no-rehash", action="store_true", help="skip re-hashing database files and checkpoints")
    ap.add_argument("--workers", type=int, default=3, help="processes executing SQL (each inherits the CPU affinity)")
    ap.add_argument("--unchanged-since", default=UNCHANGED_SINCE, help="commit whose SHA256SUMS every bundle file must still match")
    a = ap.parse_args(argv)
    names = a.sections or list(PUBLIC_SECTIONS if a.public else LOCAL_SECTIONS)
    repo = a.repo.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    ev = repo / HARNESS / "evidence"
    if a.public:
        bad = [n for n in names if n not in PUBLIC_SECTIONS]
        if bad:
            ap.error(f"sections {bad} need the runs root; public sections are {', '.join(PUBLIC_SECTIONS)}")
        ctx = {"repo": repo, "runs": None, "public": True, "rehash": False, "workers": a.workers, "out": a.out,
               "perf_root": ev / "perf_v2", "ledger_path": ev / "perf_v2" / "gpu_ledger.jsonl", "pilot_root": ev / "pilot_v1",
               "baseline_roots": [ev / "perf_v2" / "baselines", ev / "dev_v1" / "baselines", ev / "pilot_v1" / "L"],
               "unchanged_since": a.unchanged_since, "cache": None}
    else:
        runs_arg = a.runs_root or (Path(os.environ["TI_TNN_SCALING_RUNS"]) if os.environ.get("TI_TNN_SCALING_RUNS") else None)
        if runs_arg is None:
            ap.error("local mode needs --runs-root or TI_TNN_SCALING_RUNS (or use --public)")
        runs = runs_arg.resolve()
        if a.out.resolve().is_relative_to(runs):
            ap.error("the output directory must not be under the runs root")
        ctx = {"repo": repo, "runs": runs, "public": False, "rehash": not a.no_rehash, "workers": a.workers, "out": a.out,
               "perf_root": runs / "perf_v2", "ledger_path": runs / "gpu_ledger.jsonl", "pilot_root": runs / "pilot",
               "baseline_roots": [runs / "baselines"], "unchanged_since": a.unchanged_since,
               "cache": GoldCache(a.out / "gold_exec_cache.json")}
    result = {"verifier": {"file": HARNESS + "independent/verify_perf.py", "sha256": sha256_path(Path(__file__)),
                           "checkout": str(repo), "checkout_commit": git(repo, "rev-parse", "HEAD").strip(),
                           "mode": "public (bundles only)" if a.public else "local (runs root)",
                           "runs_root": str(ctx["runs"]) if ctx["runs"] else None,
                           "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "producer_modules_imported": []},
              "sections": {}}
    for n in names:
        sec = Section(n)
        t0 = time.time()
        try:
            SECTIONS[n](sec, ctx)
        except Exception:  # a crash is a failed check, never a silent pass
            import traceback
            sec.check(False, f"{n}: section completed without an exception", traceback.format_exc()[-2000:])
        sec.info["seconds"] = round(time.time() - t0, 1)
        result["sections"][n] = sec.as_dict()
        d = result["sections"][n]
        print(f"== {n}: {d['passed']} passed, {d['failures']} failed ({sec.info['seconds']} s)", flush=True)
        for c in sec.checks:
            if not c["passed"]:
                print(f"   FAIL {c['name']}: {str(c['detail'])[:400]}", flush=True)
    result["verifier"]["producer_modules_imported"] = ctx.get("producer_modules_used", [])
    result["checks"] = sum(s_["passed"] + s_["failures"] for s_ in result["sections"].values())
    result["failures"] = sum(s_["failures"] for s_ in result["sections"].values())
    out = a.out / f"verify_perf_{'public' if a.public else 'local'}_{'_'.join(names)}.json"
    out.write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(f"wrote {out}; {result['checks']} checks, failures {result['failures']}")
    return int(bool(result["failures"]))


if __name__ == "__main__":
    sys.exit(main())
