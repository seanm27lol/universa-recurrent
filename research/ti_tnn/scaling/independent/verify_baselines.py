#!/usr/bin/env python3
"""Independent verifier for TI-TNN scaling baseline runs (arms L, B4, Q27).

    usage: nice -n 19 python3 verify_baselines.py RUN_DIR [RUN_DIR ...] [--repo REPO] [--runs-root RUNS] [--no-rehash] [--json OUT]

Written apart from the producers. It imports nothing from the scaling package. Its inputs are:
- the run's own files (manifest.json, records.jsonl, summary.json);
- the raw training release, sha256-checked;
- the pins and contracts *at the run's commit*, read with `git show` and parsed as literals (never imported);
- the pinned SQLite files, re-hashed;
- its own execution-accuracy (EX) implementation: the official mini_dev rule, predicted then gold on one read-only
  connection, `set(rows)` equality, one shared 30 s deadline, any exception scores 0, an abstention scores 0.

What it establishes, per run:
1. Identity: the manifest's config hashes to its config_identity; every record and the summary carry it; the research
   tree was clean; the pin hashes the config names equal the pins at the run's commit.
2. Requests: the requested ids for the run's positions, reconstructed from the raw release with the split and inventory
   rules. Every gold query the reconstruction passes is re-executed here, which settles the exclusions it depends on.
   Every requested task appears exactly once, and no other task does.
3. Snapshots: each record's snapshot_sha256 equals the pin, and the file on disk re-hashes to it.
4. Scores: every record's final SQL and gold SQL are re-executed and `correct` is re-derived; abstentions count 0.
5. Records: finite numbers; error and timeout flags consistent with the final; retrieved examples from the training
   pool and another database.
   - LLM arms: tool counts derived from the attempts, and repairs counted; no hidden retry; the turn bound
     max_tool_calls + repairs + 1; finalize labelling; token sums; per-turn and per-request token limits.
   - L: the tool count equals the tool log, and the final SQL is a candidate that ran.
6. Summary: quality, cost_seconds_per_correct (the sum of elapsed / correct) and the counts, recomputed and compared.

What it does not establish: that the logged elapsed times are true wall-clock times; that the model produced the
recorded text (no re-generation); token counts against a tokenizer; input_identity (the schema text is not stored);
GPU-hour accounting (the ledger is out of scope).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

RAW_RELEASE = "data/raw/hf/bird23_train_filtered.train-00000-of-00001.jsonl"   # relative to the runs root
RELEASE_KEY = "data/train-00000-of-00001.jsonl"
PIN_DIR = "research/ti_tnn/scaling/pins"
CONTRACTS = "research/ti_tnn/scaling/contracts.py"
# the harness's own refusals (a call that never ran): status "not_run" (539067d on), or these texts (older records);
# every other tool result was executed, whatever its status (a tool's own "rejected" is a call)
HARNESS_REJECTIONS = ("the tool-call budget is exhausted", "the request deadline has passed",
                      "tool-call budget exhausted without a final answer")
LLM_ARMS = ("B4", "Q27")
CACHE = Path(os.environ.get("TI_TNN_VERIFY_CACHE", Path.home() / ".cache" / "ti_tnn_scaling_verify"))


class Report:
    def __init__(self, run: str):
        self.run, self.fails, self.warns, self.passes = run, [], [], []

    def check(self, ok: bool, what: str, detail: str = "", warn: bool = False):
        if ok:
            self.passes.append(what)
        else:
            (self.warns if warn else self.fails).append(f"{what}: {detail}" if detail else what)
        return ok


# --- inputs read without the producers -----------------------------------------------------------------------------
def git_show(repo: Path, commit: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=repo, capture_output=True, check=True).stdout


def contract_literals(source: bytes, names=("SPLIT", "INVENTORY", "TRAIN_RELEASE", "OFFICIAL_SCORER", "SMOKE_TRAIN")):
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


# --- execution ------------------------------------------------------------------------------------------------------
def connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect("file:" + urllib.parse.quote(str(path.resolve())) + "?mode=ro", uri=True)


def run_sql(path: Path, sqls: list, timeout: float):
    """Execute the statements in order on one read-only connection under one deadline. Returns (rows list, status)."""
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
    """The official EX on one connection: predicted first, then gold, one deadline. (correct, status, seconds)."""
    if predicted is None:
        return 0, "abstain", 0.0
    t0 = time.perf_counter()
    rows, status = run_sql(path, [predicted, gold], timeout)
    dt = time.perf_counter() - t0
    if rows is None:
        return 0, status, dt
    return int(set(rows[0]) == set(rows[1])), "correct" if set(rows[0]) == set(rows[1]) else "incorrect", dt


def gold_executes(path: Path, sha: str, task_id: str, gold: str, timeout: float) -> tuple:
    """Is this gold query executable within the timeout (the inventory's exclusion rule)? Cached per (task, file)."""
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / "gold_exec.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    key = f"{task_id}:{sha}:{timeout}"
    if key not in cache:
        t0 = time.perf_counter()
        rows, status = run_sql(path, [gold], timeout)
        cache[key] = {"status": status, "seconds": round(time.perf_counter() - t0, 3)}
        cache_file.write_text(json.dumps(cache, indent=0, sort_keys=True))
    return cache[key]["status"] == "ok", cache[key]


def nonfinite(obj, path="record"):
    """Paths of NaN/inf numbers anywhere in a JSON object."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return [path]
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in nonfinite(v, f"{path}.{k}")]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in nonfinite(v, f"{path}[{i}]")]
    return []


def is_count(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


# --- the verification -----------------------------------------------------------------------------------------------
def verify(run_dir: Path, repo: Path, rehash: bool = True, runs_root: Path | None = None) -> Report:
    rep = Report(run_dir.name)
    runs_root = runs_root or run_dir.resolve().parents[1]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines() if line.strip()]
    summary_file = run_dir / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else None
    config, commit = manifest["config"], manifest["git"]["commit"]
    arm = config["arm"]
    limits = config["limits"]

    # 1. identity
    rep.check(manifest["git"].get("research_tree_dirty") is False, "research tree clean at the run's commit",
              f"research_tree_dirty={manifest['git'].get('research_tree_dirty')}, {manifest['git'].get('dirty_paths')}")
    cid = sha256_bytes(canonical(config).encode())
    rep.check(cid == manifest["config_identity"], "config hashes to config_identity", f"{cid} != {manifest['config_identity']}")
    bad = [r["task_id"] for r in records if r.get("config_identity") != manifest["config_identity"] or r.get("arm") != arm]
    rep.check(not bad, "every record carries the manifest's config_identity and arm", f"{bad[:5]}")
    contracts = contract_literals(git_show(repo, commit, CONTRACTS))
    pins = {name: git_show(repo, commit, f"{PIN_DIR}/{name}.json") for name in ("split_v1", "inventory_v1", "snapshots_v1")}
    rep.check(sha256_bytes(pins["inventory_v1"]) == config.get("inventory_pin_sha256"), "inventory pin hash (config vs commit)")
    rep.check(sha256_bytes(pins["snapshots_v1"]) == config.get("snapshots_pin_sha256"), "snapshots pin hash (config vs commit)")
    if arm in LLM_ARMS:
        mi = manifest.get("model_identity") or {}
        try:
            models_pin = git_show(repo, commit, f"{PIN_DIR}/models_v1.json")
            rep.check(mi.get("pin_sha256") == sha256_bytes(models_pin), "model identity checked against models_v1 at the commit",
                      f"{mi.get('pin_sha256')}")
        except subprocess.CalledProcessError:
            rep.check(False, "models_v1.json exists at the run's commit")
        rep.check((mi.get("repo"), mi.get("revision")) == (config["model"]["repo"], config["model"]["revision"]),
                  "model identity matches the configured model")

    # 2. requests, reconstructed from the raw release and the rules
    raw = (runs_root / RAW_RELEASE).read_bytes()
    rep.check(sha256_bytes(raw) == contracts["TRAIN_RELEASE"]["files"][RELEASE_KEY], "raw release sha256 (contracts at commit)")
    lines = raw.decode("utf-8").split("\n")
    lines = lines[:-1] if lines and lines[-1] == "" else lines
    rows = [json.loads(line) for line in lines]
    tid = [f"bird23tf-{i:05d}" for i in range(len(rows))]
    by_id = dict(zip(tid, rows))
    split, inv_rule = contracts["SPLIT"], contracts["INVENTORY"]
    dbs = sorted({r["db_id"] for r in rows}, key=lambda d: order_key(split["salt"], d))
    n_dev = round(split["dev_fraction"] * len(dbs))
    dev, pool = dbs[:n_dev], set(dbs[n_dev:])
    snaps = json.loads(pins["snapshots_v1"])["databases"]
    db_path = lambda d: runs_root / "data/databases" / snaps[d]["file"]   # noqa: E731
    timeout = float(contracts["OFFICIAL_SCORER"]["meta_time_out_seconds"])
    lo, hi = config["positions"]
    queues = {d: sorted((t for t in tid if by_id[t]["db_id"] == d), key=lambda t: order_key(inv_rule["salt"], t)) for d in dev}
    pointer = {d: 0 for d in dev}
    order, excluded = [], {}
    while len(order) < hi and any(pointer[d] < len(queues[d]) for d in dev):
        for d in dev:
            while pointer[d] < len(queues[d]):
                t = queues[d][pointer[d]]
                pointer[d] += 1
                ok, info = gold_executes(db_path(d), snaps[d]["sha256"], t, by_id[t]["SQL"], timeout)
                if ok:
                    order.append(t)
                    break
                excluded[t] = info
            if len(order) == hi:
                break
    requested = order[lo:hi]
    inv_pin = json.loads(pins["inventory_v1"])
    rep.check(requested == config["requested"], "requested ids = reconstruction from the raw release and the rules",
              f"reconstructed {requested} vs manifest {config['requested']}")
    rep.check(inv_pin["task_ids"][lo:hi] == requested, "pinned inventory agrees with the reconstruction at these positions")
    pin_excluded = {e["task_id"] for e in inv_pin["excluded"]}
    rep.check(set(excluded) <= pin_excluded, "every exclusion met on the way is a pinned exclusion",
              f"re-executed and excluded here: {excluded}", warn=True)
    ids = [r["task_id"] for r in records]
    rep.check(sorted(ids) == sorted(requested) and len(set(ids)) == len(ids), "every requested task exactly once, nothing else",
              f"missing {sorted(set(requested) - set(ids))}, extra {sorted(set(ids) - set(requested))}, "
              f"duplicates {sorted({i for i in ids if ids.count(i) > 1})}")

    # 3. snapshots
    hashed = {}
    for r in records:
        d = r["db_id"]
        rep.check(by_id.get(r["task_id"], {}).get("db_id") == d, f"{r['task_id']}: db_id is the release's")
        rep.check(r.get("snapshot_sha256") == snaps[d]["sha256"], f"{r['task_id']}: snapshot_sha256 = pin")
        rep.check((r.get("score") or {}).get("snapshot_sha256") == snaps[d]["sha256"], f"{r['task_id']}: score bound to the pinned snapshot")
        if rehash and d not in hashed:
            hashed[d] = sha256_path(db_path(d))
            rep.check(hashed[d] == snaps[d]["sha256"], f"{d}: file re-hashes to the pin", hashed[d])
        for e in r.get("tool_log") or []:
            rep.check(e.get("snapshot_sha256") in (None, snaps[d]["sha256"]), f"{r['task_id']}: tool call on the pinned snapshot")

    # 4. scores, re-derived
    rederived = {}
    for r in records:
        t, f = r["task_id"], r["final"]
        pred = f.get("sql") if f.get("kind") == "sql" else None
        correct, status, secs = ex(db_path(r["db_id"]), pred, by_id[t]["SQL"], timeout)
        rederived[t] = correct
        recorded = (r.get("score") or {}).get("correct")
        timing = status == "timeout" or secs > 0.8 * timeout or (r.get("score") or {}).get("status") == "timeout"
        rep.check(recorded == correct, f"{t}: correct re-derived ({status}, {secs:.2f} s)", f"recorded {recorded}, re-derived {correct}",
                  warn=timing)
        rep.check((r.get("score") or {}).get("scored_sql_sha256") == sha256_bytes((pred or "").encode()),
                  f"{t}: score bound to the submitted SQL")

    # 5. records
    max_calls, max_rep = limits["max_tool_calls"], limits["max_repairs_llm"]
    mode = config.get("mode")
    for r in records:
        t, f, atts = r["task_id"], r["final"], r["attempts"]
        rep.check(not nonfinite(r), f"{t}: all numbers finite", f"{nonfinite(r)[:5]}")
        rep.check(isinstance(r["elapsed_seconds"], (int, float)) and r["elapsed_seconds"] > 0, f"{t}: elapsed positive")
        rep.check(f.get("kind") in ("sql", "abstain"), f"{t}: final kind")
        rep.check(f.get("kind") != "sql" or (isinstance(f.get("sql"), str) and f["sql"].strip()), f"{t}: an answer has SQL")
        rep.check(f.get("kind") != "abstain" or bool(f.get("reason")), f"{t}: an abstention has a reason")
        rep.check(r.get("error") is None or f.get("kind") == "abstain", f"{t}: an error abstains")
        rep.check(isinstance(r.get("timed_out"), bool), f"{t}: timed_out is a bool")
        for e in r.get("retrieved") or []:
            rep.check(by_id[e]["db_id"] in pool and by_id[e]["db_id"] != r["db_id"] and e != t,
                      f"{t}: retrieved {e} is a training-pool example from another database")
        if arm in LLM_ARMS:
            rep.check(r["timed_out"] == (f.get("reason") == "request timeout"), f"{t}: timed_out agrees with the reason")
            executed, used, budget_hit_at = 0, 0, None
            for i, a in enumerate(atts):
                res = a.get("tool_results", [])
                ran = [x for x in res if (x.get("result") or {}).get("status") != "not_run"
                       and not str((x.get("result") or {}).get("error", "")).startswith(HARNESS_REJECTIONS)]
                if used >= max_calls and ran:
                    rep.check(False, f"{t}: turn {i} executed a tool after the budget was spent")
                if used >= max_calls and res and i != len(atts) - 1:
                    rep.check(False, f"{t}: turn {i} called tools after the budget was spent and the request went on")
                if a.get("kind") == "finalize":
                    rep.check(budget_hit_at == i - 1, f"{t}: turn {i} labelled finalize follows the turn that spent the budget")
                elif budget_hit_at == i - 1 and i > 0:
                    rep.check(False, f"{t}: turn {i}, after the budget was spent, is not labelled finalize")
                before = used
                used += len(ran)
                executed += len(ran)
                if before < max_calls <= used:
                    budget_hit_at = i
                if a.get("final_status") not in (None, "ok") and i + 1 < len(atts):
                    rep.check(atts[i + 1].get("kind") == "repair", f"{t}: failed final check at turn {i} followed by a labelled repair")
            repairs = sum(a.get("kind") == "repair" for a in atts)
            rep.check(r["tool_calls"] == executed and executed <= max_calls, f"{t}: tool_calls = executed calls in the attempts (<= {max_calls})",
                      f"recorded {r['tool_calls']}, executed {executed}")
            rep.check(r["repairs"] == repairs and repairs <= max_rep, f"{t}: repairs counted (<= {max_rep})", f"{r['repairs']} vs {repairs}")
            rep.check(len(atts) <= max_calls + repairs + 1, f"{t}: turn bound {max_calls} + repairs + 1",
                      f"{len(atts)} turns (bound {max_calls + repairs + 1})")
            if f.get("kind") == "sql":
                rep.check(bool(atts) and atts[-1].get("final_sql") == f["sql"], f"{t}: final SQL is the last attempt's")
            g = [a.get("generation") or {} for a in atts]
            tok = r.get("tokens") or {}
            sums = {"generated": "generated_tokens", "prefill": "prompt_tokens", "reasoning": "reasoning_tokens", "content": "content_tokens"}
            for k, field in sums.items():
                rep.check(all(is_count(x.get(field)) for x in g) and tok.get(k) == sum(x[field] for x in g),
                          f"{t}: tokens.{k} = sum over turns", f"{tok.get(k)} vs {sum(x.get(field) or 0 for x in g)}")
            if atts:
                rep.check(tok.get("initial_prompt") == g[0].get("prompt_tokens"), f"{t}: tokens.initial_prompt = first turn's prompt")
            per_turn = limits["max_new_tokens_thinking" if mode == "thinking" else "max_new_tokens_nonthinking"]
            per_req = limits["max_generated_tokens_thinking" if mode == "thinking" else "max_generated_tokens_nonthinking"]
            rep.check(all((x.get("generated_tokens") or 0) <= per_turn for x in g), f"{t}: every turn within {per_turn} new tokens")
            rep.check((tok.get("generated") or 0) <= per_req, f"{t}: request within {per_req} generated tokens")
            rep.check(all((x.get("prompt_tokens") or 0) + (x.get("generated_tokens") or 0) <= limits["max_context_tokens"] for x in g),
                      f"{t}: every turn within the context limit")
            tmo = limits["request_timeout_seconds_27b" if arm == "Q27" else "request_timeout_seconds_4b"]
            rep.check(r["elapsed_seconds"] <= tmo + 60, f"{t}: elapsed within the request timeout + 60 s", f"{r['elapsed_seconds']:.0f} s", warn=True)
        elif arm == "L":
            rep.check(r["tool_calls"] == len(r.get("tool_log") or []), f"{t}: L tool_calls = tool log entries")
            rep.check(r.get("repairs") == 0 and r.get("tokens") == {} and r.get("timed_out") is False, f"{t}: L has no repairs, tokens or timeouts")
            if f.get("kind") == "sql":
                ran = [c["sql"] for a in atts for c in a.get("candidates", []) if c.get("status") == "ok"]
                rep.check(f["sql"] in ran, f"{t}: L's final SQL is a candidate that ran")

    # 6. summary, recomputed from the re-derived scores
    if rep.check(summary is not None and "summary" in summary, "summary.json has a summary", (summary or {}).get("summary_error", "missing")):
        s = summary["summary"]
        n = len(requested)
        correct = sum(rederived.get(t, 0) for t in requested)
        elapsed = sum(r["elapsed_seconds"] for r in records)
        close = lambda a, b: (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9))  # noqa: E731
        rep.check(s.get("requested") == n and s.get("correct") == correct, "summary requested/correct recomputed",
                  f"summary {s.get('requested')}/{s.get('correct')} vs {n}/{correct}")
        rep.check(close(s.get("quality"), correct / n if n else None), "quality recomputed", f"{s.get('quality')} vs {correct / n if n else None}")
        rep.check(close(s.get("elapsed_seconds_total"), elapsed), "elapsed total recomputed")
        rep.check(close(s.get("cost_seconds_per_correct"), elapsed / correct if correct else None), "cost_seconds_per_correct = sum elapsed / correct",
                  f"{s.get('cost_seconds_per_correct')} vs {elapsed / correct if correct else None}")
        counts = {"answered": sum(r["final"]["kind"] == "sql" for r in records), "abstained": sum(r["final"]["kind"] == "abstain" for r in records),
                  "errors": sum(r.get("error") is not None for r in records), "timeouts": sum(bool(r.get("timed_out")) for r in records),
                  "repairs": sum(r.get("repairs", 0) for r in records), "tool_calls": sum(r.get("tool_calls", 0) for r in records)}
        diff = {k: (s.get(k), v) for k, v in counts.items() if s.get(k) != v}
        rep.check(not diff, "summary counts recomputed", f"{diff}")
        rep.check(s.get("config_identity", manifest["config_identity"]) == manifest["config_identity"] and s.get("arm", arm) == arm,
                  "summary names the run's arm and configuration")
    return rep


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--repo", type=Path, default=None, help="a worktree of the repository (default: from the run dir)")
    p.add_argument("--runs-root", type=Path, default=None, help="the shared runs directory (default: two levels above the run dir)")
    p.add_argument("--no-rehash", action="store_true", help="skip re-hashing the database files")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)
    status, out = 0, {}
    for run in args.runs:
        repo = args.repo or run.resolve().parents[3]
        rep = verify(run, repo, rehash=not args.no_rehash, runs_root=args.runs_root)
        out[rep.run] = {"fails": rep.fails, "warns": rep.warns, "passes": len(rep.passes)}
        print(f"== {rep.run}: {'FAIL' if rep.fails else 'PASS'} ({len(rep.passes)} checks passed, {len(rep.fails)} failed, {len(rep.warns)} warnings)")
        for x in rep.fails:
            print("   FAIL", x)
        for x in rep.warns:
            print("   WARN", x)
        status |= bool(rep.fails)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
    return int(status)


if __name__ == "__main__":
    sys.exit(main())
