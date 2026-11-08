"""The database-evidence revision study's evidence bundle and report tables (written after the frozen run).

    python -m scaling.dbe_report bundle    # copy the study's run, analysis and logs, and the ledger, into the bundle
    python -m scaling.dbe_report           # recompute from the bundle, compare with analysis.json, print tables

**The bundle** is `evidence/dbe_v1/`: every file of `<runs root>/dbe/` and the GPU ledger, under `SHA256SUMS`.

**The report** checks the sums and then, before building anything:
- **The frozen pins:** the protocol, population, prior-use scan, lookup and harness sha256 (`dbe_run.protocol_identity`).
- **The population binding:** every request and score names a pinned question and its pinned database.
- **The run identity:** the one run and its analysis name each other. Both manifests record the same clean commit,
  the frozen protocol, the pinned questions, seeds and lookup. The summary's reservation is the ledger's, and the
  ledger's start entry for that reservation names this run directory.
- **The raw values,** with the protections of #73:
  - integer seeds;
  - integer 0/1 correctness;
  - nonnegative integer token, row and memory counts;
  - genuine Booleans;
  - finite nonnegative generation, lookup, tool, submitted-execution and allowed times.

It then recomputes every analysed quantity from the bundled records with the frozen `dbe_run.analyze`. It recomputes
the GPU reservation and the ledger totals from the bundled ledger with `ledger.usage`. It compares all of them, and
the analysis's exact key set, with `analysis.json` (types checked before any tolerance), and refuses any
disagreement. A matching file hash alone verifies nothing displayed.

This module changes nothing the run used: the policies, the lookup, the rule and the analysis are the frozen
harness's.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from . import contracts as C
from . import evidence
from .spr_report import Inconsistent, _check, _count, _differ, _flag, _num, _time

BUNDLE = C.SCALING / "evidence" / "dbe_v1"
LEDGER_FIELDS = ("cap_hours", "authorization", "used_hours", "remaining_hours")
STORED_ONLY = ("protocol", "run", "gpu_reservations", "ledger")   # analysis.json fields not produced by analyze


def bundle(dest: Path = BUNDLE) -> int:
    from . import dbe_run
    n = 0
    for f in sorted(dbe_run.ROOT.rglob("*")):
        if f.is_file():
            target = dest / f.relative_to(dbe_run.ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            n += 1
    shutil.copy2(C.RUNS / "gpu_ledger.jsonl", dest / "gpu_ledger.jsonl")
    evidence.write_sums(dest)
    return n + 1


def _load(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _validate(reqs: list, scores: list, ids: list, seeds, db_of: "dict | None" = None) -> None:
    """The exact inventory, the population binding and every raw value, before anything is built."""
    from collections import Counter

    from . import dbe_run as D
    for x in reqs + scores:
        _check(type(x["seed"]) is int, f"{x['task_id']}: seed {x['seed']!r} is not an integer")
    want_r = {(s, t) for s in seeds for t in ids}
    got_r = Counter((r["seed"], r["task_id"]) for r in reqs)
    if set(got_r) != want_r or any(n != 1 for n in got_r.values()):
        raise Inconsistent("requests: not exactly one per question and seed")
    want_s = {(p, s, t) for p in D.POLICIES for s in seeds for t in ids}
    got_s = Counter((x["policy"], x["seed"], x["task_id"]) for x in scores)
    if set(got_s) != want_s or any(n != 1 for n in got_s.values()):
        raise Inconsistent("scores: not exactly one per policy, seed and question")
    if db_of is not None:
        for x in reqs + scores:
            if x["db_id"] != db_of[x["task_id"]]:
                raise Inconsistent(f"{x['task_id']}: database {x['db_id']!r} is not the pinned one")
    for x in scores:
        w = f"score {x['policy']}/{x['seed']}/{x['task_id']}"
        c = x["score"]["correct"]
        if not (_num(c) and isinstance(c, int) and c in (0, 1)):
            raise Inconsistent(f"{w}: correctness {c!r} is not integer 0 or 1")
        pr = x["score"]["predicted"]
        _time(pr["elapsed_seconds"], f"{w}: predicted.elapsed_seconds")
        for k in ("row_count", "distinct_rows"):
            if pr.get(k) is not None:
                _count(pr[k], f"{w}: predicted.{k}")
    for r in reqs:
        w = f"request {r['seed']}/{r['task_id']}"
        if r["D0"] != {"source": "original", "sql": r["original"]["sql"]}:
            raise Inconsistent(f"{r['task_id']}/{r['seed']}: D0 is not the shared initial proposal")
        for name, cand in [("original", r["original"])] + [(f"{p}.revised", r[p]["revised"]) for p in ("D1", "V1")]:
            call, ex = cand["call"], cand["execution"]
            _time(call["seconds"], f"{w}: {name}.call.seconds (generation)")
            for k in ("prompt_tokens", "generated_tokens", "peak_memory_bytes"):
                _count(call[k], f"{w}: {name}.call.{k}")
            _time(ex["seconds"], f"{w}: {name}.execution.seconds (tool)")
            if ex.get("row_count") is not None:
                _count(ex["row_count"], f"{w}: {name}.execution.row_count")
            _flag(ex.get("truncated", False), f"{w}: {name}.execution.truncated")
        ev = r["evidence"]
        _flag(ev["ok"], f"{w}: evidence.ok")
        _time(ev["seconds"], f"{w}: evidence.seconds (lookup)")
        if ev["ok"]:
            pkg = ev["package"]
            for k in ("truncated", "wall_safety_stop"):
                _flag(pkg[k], f"{w}: evidence.package.{k}")
            for k in ("payload_chars", "payload_chars_untruncated", "queries", "instructions"):
                _count(pkg[k], f"{w}: evidence.package.{k}")
            _time(pkg["latency_seconds"], f"{w}: evidence.package.latency_seconds")
        _time(r["revision_seconds_allowed"], f"{w}: revision_seconds_allowed")


def _run_identity(run: Path, adir: Path, analysis: dict, identity: dict, ids: list) -> None:
    """The one run and its analysis name each other; both manifests record the same clean commit and the frozen
    protocol; the run's manifest records the pinned questions, seeds and lookup."""
    from . import dbe_run as D
    problems = [f"analysis.protocol{p}" for p in _differ(analysis["protocol"], identity)]
    if analysis["run"] != "dbe/run/" + run.name:
        problems.append(f"analysis.run {analysis['run']!r} is not the bundled run dbe/run/{run.name}")
    rm, am = (json.loads((d / "manifest.json").read_text()) for d in (run, adir))
    for name, m in (("run", rm), ("analysis", am)):
        g = m["git"]
        if not (isinstance(g["commit"], str) and len(g["commit"]) == 40) or g["commit"] != rm["git"]["commit"]:
            problems.append(f"{name} manifest commit")
        if g["research_tree_dirty"] is not False or g["dirty_paths"] != []:
            problems.append(f"{name} manifest: the tree was not clean")
        problems += [f"{name} manifest protocol{p}" for p in _differ(m["config"]["protocol"], identity)]
    problems += [f"run manifest requested{p}" for p in _differ(rm["config"]["requested"], ids)]
    problems += [f"run manifest seeds{p}" for p in _differ(rm["config"]["seeds"], list(D.SEEDS))]
    if rm["config"]["lookup"] != D.EVIDENCE_VERSION or rm["config"]["command"] != "dbe run":
        problems.append("run manifest lookup or command")
    if am["config"]["run"] != analysis["run"] or am["config"]["command"] != "dbe analyze":
        problems.append("analysis manifest run or command")
    if problems:
        raise Inconsistent("the run identity disagrees: " + "; ".join(problems[:12]))


def _ledger(run: Path) -> tuple:
    """(reservations, totals) recomputed from the bundled ledger; exactly one 'dbe' reservation, the summary's."""
    from . import ledger
    u = ledger.usage(BUNDLE / "gpu_ledger.jsonl")
    res = [{"command": r["command"], "id": r["reservation"], "hours": r["hours"], "status": r["status"]}
           for r in u["reservations"] if r["command"].startswith("dbe ")]
    summary = json.loads((run / "summary.json").read_text())
    if len(res) != 1 or res[0]["id"] != summary["reservation"]["id"] or res[0]["status"] != "ok":
        raise Inconsistent(f"the ledger does not hold exactly the run's one completed reservation: {res}")
    starts = [e for e in ledger.read(BUNDLE / "gpu_ledger.jsonl")
              if e.get("event") == "start" and e.get("reservation") == res[0]["id"]]
    if len(starts) != 1 or (starts[0].get("extra") or {}).get("run_dir") != "dbe/run/" + run.name:
        raise Inconsistent(f"the run identity disagrees: the ledger's reservation names "
                           f"{[(e.get('extra') or {}).get('run_dir') for e in starts]}, not dbe/run/{run.name}")
    return res, {k: u[k] for k in LEDGER_FIELDS}


def reconstruct() -> tuple:
    """(recomputed, analysis): the pins, the population, the run identity and every raw value checked first; then
    every analysed quantity, the GPU reservation and the ledger totals recomputed and compared with analysis.json."""
    from . import dbe_run as D
    evidence._checked(BUNDLE)
    runs = sorted(p for p in BUNDLE.glob("run/*") if p.is_dir())
    adirs = sorted(p for p in BUNDLE.glob("analysis/*") if p.is_dir())
    if len(runs) != 1 or len(adirs) != 1 or not (runs[0] / "scores.jsonl").exists():
        raise Inconsistent(f"{len(runs)} runs and {len(adirs)} analyses in the bundle; exactly one each is required")
    run, adir = runs[0], adirs[0]
    analysis = json.loads((adir / "analysis.json").read_text())
    try:
        identity = D.protocol_identity()
    except SystemExit as stop:
        raise Inconsistent(f"the frozen pins do not hold: {stop}") from None
    pin = json.loads(D.NEW_PIN.read_text())
    ids, db_of = pin["task_ids"], dict(zip(pin["task_ids"], pin["db_ids"]))
    reqs, scores = _load(run / "requests.jsonl"), _load(run / "scores.jsonl")
    _validate(reqs, scores, ids, D.SEEDS, db_of)
    _run_identity(run, adir, analysis, identity, ids)
    got = D.analyze(reqs, scores, ids, db_of, D.SEEDS)
    expected = set(got) | set(STORED_ONLY)
    if set(analysis) != expected:
        raise Inconsistent(f"analysis.json fields differ: {sorted(set(analysis) ^ expected)}")
    reservations, totals = _ledger(run)
    if not isinstance(analysis["ledger"].get("ledger"), str):
        raise Inconsistent("analysis.ledger.ledger is not a path")
    problems = [p for k in got for p in _differ(got[k], analysis[k], f".{k}")]
    problems += _differ(reservations, analysis["gpu_reservations"], ".gpu_reservations")
    problems += _differ(totals, {k: analysis["ledger"].get(k) for k in LEDGER_FIELDS}, ".ledger")
    if problems:
        raise Inconsistent("the records disagree with analysis.json: " + "; ".join(problems[:12]))
    got = {**got, "gpu_reservations": reservations, "ledger": totals}
    return got, analysis


def tables() -> str:
    got, analysis = reconstruct()
    seeds = got["seeds"]
    out = ["| policy | " + " | ".join(f"seed {s}" for s in seeds) + " | pooled (of 128) |",
           "|---|" + "---|" * len(seeds) + "---|"]
    for p in ("D0", "D1", "V1"):
        out.append(f"| {p} | " + " | ".join(str(got["counts"][f"{p}/{s}"]) for s in seeds) +
                   f" | {got['pooled'][p]} |")
    out += ["", "| comparison | pooled | per seed | recoveries | regressions | 95% interval (questions) | "
            "leave-one-database-out min | all conditions | sign-flip p (exploratory) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k, d in got["comparisons"].items():
        u = got["uncertainty"][k]
        out.append(f"| {k} | {d['pooled']:+d} | " + ", ".join(f"{v:+d}" for v in d["per_seed"].values()) +
                   f" | {d['recoveries']} | {d['regressions']} | [{u['low']:+d}, {u['high']:+d}] | "
                   f"{min(d['leave_one_database_out'].values()):+d} | {d['all_conditions']} | "
                   f"{d['exploratory_sign_flip']['p_two_sided']:.3f} |")
    out += ["", f"**Decision:** {got['decision']}.", "", "| transition | count |", "|---|---|"]
    t = got["transitions"]
    out += [f"| V1 right where D1 is wrong | {t['V1_fixes_D1_misses']} |",
            f"| D1 right where V1 is wrong | {t['D1_right_V1_breaks']} |",
            f"| initially correct (D0) | {t['initially_correct']} |",
            f"| initially correct, broken by D1 / V1 | {t['initially_correct_broken_by']['D1']} / "
            f"{t['initially_correct_broken_by']['V1']} |",
            f"| initially wrong, fixed by D1 / V1 | {t['initially_wrong_fixed_by']['D1']} / "
            f"{t['initially_wrong_fixed_by']['V1']} |"]
    out += ["", "| signal group (descriptive) | requests | D0 | D1 | V1 | V1 right, D1 wrong | D1 right, V1 wrong | "
            "D0 right, V1 wrong |", "|---|---|---|---|---|---|---|---|"]
    for g, d in got["signal_groups"].items():
        if g == "note":
            continue
        out.append(f"| {g} | {d['requests']} | {d['D0']} | {d['D1']} | {d['V1']} | {d['V1_fixes_D1_misses']} | "
                   f"{d['D1_right_V1_breaks']} | {d['D0_right_V1_breaks']} |")
    out += ["", "| policy | generation s | lookup s | tool s | submitted execution s | total s | per request s | "
            "per EX-correct s | prompt tokens | generated tokens | peak memory GB |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, c in got["cost"].items():
        per_ok = f"{c['per_ex_correct_seconds']:.1f}" if c["per_ex_correct_seconds"] else "-"
        out.append(f"| {p} | {c['generation_seconds']:.1f} | {c['lookup_seconds']:.1f} | {c['tool_seconds']:.1f} | "
                   f"{c['submitted_execution_seconds']:.1f} | {c['total_seconds']:.1f} | "
                   f"{c['per_request_seconds']:.2f} | {per_ok} | {c['prompt_tokens']:,} | "
                   f"{c['generated_tokens']:,} | {c['peak_memory_bytes'] / 1e9:.2f} |")
    lk = got["lookup"]
    out += ["", f"Lookups: {lk['failed']} failed; {lk['timeouts']} instruction-budget stops; "
            f"{lk['with_scan_limits']} with scan limits; {lk['truncated']} truncated; {lk['with_unresolved']} with "
            f"unresolved columns; {lk['wall_safety_stops']} wall-clock safety stops; total {lk['seconds_total']:.1f} s, "
            f"max {lk['seconds_max']:.2f} s; largest payload {lk['payload_chars_max']} characters.",
            "", "GPU reservations: " + "; ".join(f"{r['command']} {r['hours']:.3f} h ({r['status']})"
                                                 for r in got["gpu_reservations"]) +
            f". Ledger: {got['ledger']['used_hours']:.3f} of {got['ledger']['cap_hours']} h."]
    out += ["", "| database | questions | D0 | D1 | V1 |", "|---|---|---|---|---|"]
    by_db = {}
    for t, q in got["per_question"].items():
        d = by_db.setdefault(q["db"], {"questions": 0, "D0": 0, "D1": 0, "V1": 0})
        d["questions"] += 1
        for p in ("D0", "D1", "V1"):
            d[p] += sum(q[p])
    for db, d in sorted(by_db.items()):
        out.append(f"| {db} | {d['questions']} | {d['D0']} | {d['D1']} | {d['V1']} |")
    return "\n".join(out)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["bundle"]:
        print(json.dumps({"files": bundle()}))
        return 0
    print(tables())
    return 0


if __name__ == "__main__":
    sys.exit(main())
