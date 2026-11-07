"""The semantic-proposal-repair study's evidence bundle and report tables (development).

    python -m scaling.spr_report bundle    # copy the study's runs (weight-free) and the ledger into the bundle
    python -m scaling.spr_report           # recompute from the bundle, compare with analysis.json, print tables

**The bundle** is `evidence/spr_v1/`: every file of `<runs root>/spr/` and the GPU ledger, under `SHA256SUMS`.

**The report** checks the sums, then recomputes the counts, the paired comparisons (with recoveries, regressions and
leave-one-database-out), the decision, the lineage and the costs from the bundled records. It refuses any
disagreement with `analysis.json`.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

from . import contracts as C
from . import evidence
from . import perf_analysis as PA

BUNDLE = C.SCALING / "evidence" / "spr_v1"
POLICIES = ("D0", "D1", "TI1")


class Inconsistent(SystemExit):
    pass


def bundle(dest: Path = BUNDLE) -> int:
    from . import spr_run
    n = 0
    for f in sorted(spr_run.ROOT.rglob("*")):
        if f.is_file():
            target = dest / f.relative_to(spr_run.ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            n += 1
    shutil.copy2(C.RUNS / "gpu_ledger.jsonl", dest / "gpu_ledger.jsonl")
    evidence.write_sums(dest)
    return n + 1


def _load(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _validate_inventory(reqs: list, scores: list, ids: list, db_of: dict, seeds: list) -> None:
    """The exact frozen inventory before anything is built: one request per (seed, question) and one score per
    (policy, seed, question), each on the pinned database; no duplicate, missing or extra record."""
    from collections import Counter
    want_r = {(s, t) for s in seeds for t in ids}
    got_r = Counter((r["seed"], r["task_id"]) for r in reqs)
    if set(got_r) != want_r or any(n != 1 for n in got_r.values()):
        raise Inconsistent(f"requests: {sum(n > 1 for n in got_r.values())} duplicated, "
                           f"{len(want_r - set(got_r))} missing, {len(set(got_r) - want_r)} extra")
    want_s = {(p, s, t) for p in POLICIES for s in seeds for t in ids}
    got_s = Counter((x["policy"], x["seed"], x["task_id"]) for x in scores)
    if set(got_s) != want_s or any(n != 1 for n in got_s.values()):
        raise Inconsistent(f"scores: {sum(n > 1 for n in got_s.values())} duplicated, "
                           f"{len(want_s - set(got_s))} missing, {len(set(got_s) - want_s)} extra")
    for x in list(reqs) + list(scores):
        if x["db_id"] != db_of[x["task_id"]]:
            raise Inconsistent(f"{x['task_id']}: database {x['db_id']} is not the pinned {db_of[x['task_id']]}")


def recompute(reqs: list, scores: list, ids: list, db_of: dict, seeds: list, ledger_path: Path) -> dict:
    """Every displayed quantity from the raw records and the bundled ledger."""
    from collections import Counter

    from . import ledger
    from . import spr_rule as RULE
    from . import spr_run
    req = {(r["seed"], r["task_id"]): r for r in reqs}
    sc = {(x["policy"], x["seed"], x["task_id"]): x for x in scores}
    for (s, t), r in req.items():   # the selected candidates, re-derived from the recorded executions
        if r["D0"] != {"source": "original", "sql": r["original"]["sql"]}:
            raise Inconsistent(f"{t}/{s}: D0 is not the initial proposal")
        for p in ("D1", "TI1"):
            sel = spr_run.select(r["original"], r[p]["revised"])
            if {k: r[p][k] for k in ("source", "sql")} != sel:
                raise Inconsistent(f"{t}/{s}: the recorded {p} selection differs from the rule")
        for p in POLICIES:
            x = sc[(p, s, t)]
            if (x["source"], x["sql"]) != (r[p]["source"], r[p]["sql"]):
                raise Inconsistent(f"{t}/{s}: the scored {p} candidate is not the selected one")
    correct = {(p, s): {t: sc[(p, s, t)]["score"]["correct"] for t in ids} for p in POLICIES for s in seeds}
    out = {"counts": {f"{p}/{s}": sum(correct[(p, s)].values()) for p in POLICIES for s in seeds},
           "pooled": {p: sum(sum(correct[(p, s)].values()) for s in seeds) for p in POLICIES},
           "comparisons": RULE.comparisons(correct, seeds, ids, db_of)}
    out["decision"] = RULE.decision(out["comparisons"])
    out["historical_rule"] = {f"{a}-{b}": {f: d[f] for f in HISTORICAL_FIELDS}       # the original's known defect
                              for a, b in RULE.COMPARISONS
                              for d in [PA.paired(correct, a, b, seeds, ids, db_of)]}
    out["requests_per_policy"] = len(ids) * len(seeds)
    out["by_database"] = {db: {p: sum(correct[(p, s)][t] for s in seeds for t in ids if db_of[t] == db)
                               for p in POLICIES} | {"questions": sum(1 for t in ids if db_of[t] == db)}
                          for db in sorted(set(db_of.values()))}
    rs = list(req.values())
    out["lineage"] = {p: {"final_source": dict(Counter(r[p]["source"] for r in rs)),
                          "revision_rejected": sum(1 for r in rs if r[p]["source"] != "revised"),
                          "revision_unchanged_sql": sum(1 for r in rs
                                                        if r[p]["revised"]["sql"] == r["original"]["sql"]),
                          "revision_truncated": sum(1 for r in rs if r[p]["revised"]["call"]["finish_reason"] != "eos"),
                          "revised_typed": sum(1 for r in rs if r[p]["revised"]["typed"]["typed"])}
                      for p in ("D1", "TI1")}
    out["initial"] = {"no_sql": sum(1 for r in rs if not r["original"]["sql"]),
                      "truncated": sum(1 for r in rs if r["original"]["call"]["finish_reason"] != "eos"),
                      "typed": sum(1 for r in rs if r["original"]["typed"]["typed"]),
                      "untyped_reasons": dict(Counter(r["original"]["typed"]["reason"] for r in rs
                                                      if not r["original"]["typed"]["typed"])),
                      "execution": dict(Counter(r["original"]["execution"]["status"] for r in rs)),
                      "with_errors": sum(1 for r in rs if r["original"]["typed"]["errors"]),
                      "with_warnings": sum(1 for r in rs if r["original"]["typed"]["warnings"])}
    out["final_status"] = {p: dict(Counter(sc[(p, r["seed"], r["task_id"])]["score"]["predicted"]["status"]
                                           for r in rs)) for p in POLICIES}
    cost = {}
    for p in POLICIES:
        calls = [r["original"]["call"] for r in rs] + ([r[p]["revised"]["call"] for r in rs] if p != "D0" else [])
        gen = sum(c["seconds"] for c in calls)
        tool = sum(r["original"]["execution"]["seconds"] + r[p]["revised"]["execution"]["seconds"] for r in rs) \
            if p != "D0" else 0.0
        valid = sum(r["original"]["typed"]["seconds"] for r in rs) if p == "TI1" else 0.0
        exe = sum(sc[(p, r["seed"], r["task_id"])]["score"]["predicted"]["elapsed_seconds"] for r in rs)
        total, ok = gen + tool + valid + exe, out["pooled"][p]
        cost[p] = {"generation_seconds": gen, "validation_seconds": valid, "tool_seconds": tool,
                   "submitted_execution_seconds": exe, "total_seconds": total, "per_request_seconds": total / len(rs),
                   "per_ex_correct_seconds": total / ok if ok else None,
                   "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
                   "generated_tokens": sum(c["generated_tokens"] for c in calls),
                   "peak_memory_bytes": max(c["peak_memory_bytes"] for c in calls),
                   "calls_per_request": 1 if p == "D0" else 2}
    out["cost"] = cost
    u = ledger.usage(ledger_path)
    out["gpu_reservations"] = [{"command": r["command"], "id": r["reservation"], "hours": r["hours"],
                                "status": r["status"]} for r in u["reservations"] if r["command"].startswith("spr ")]
    out["ledger"] = {k: u[k] for k in LEDGER_FIELDS}
    out["ledger_used_hours"] = u["used_hours"]
    return out


def _num(v) -> bool:
    """A real number that is not a Boolean (bool is a subclass of int in Python)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _differ(a, b, path="") -> list:
    """Paths where two JSON-like values differ. Types are checked before any equality or tolerance:
    - Booleans equal only Booleans (True is not 1);
    - numbers must be finite (no NaN, no +/-Infinity) and of the same kind (an int against a float differs);
    - integers compare exactly, floats to 1e-9 relative (legitimate roundoff passes);
    - dicts and lists compare element by element."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b)) for p in (_differ(a.get(k), b.get(k), f"{path}.{k}")
                                                              if k in a and k in b else [f"{path}.{k}"])]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}[len]"]
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in _differ(x, y, f"{path}[{i}]")]
    if isinstance(a, bool) or isinstance(b, bool):
        return [] if isinstance(a, bool) and isinstance(b, bool) and a == b else [path]
    if _num(a) or _num(b):
        if not (_num(a) and _num(b)) or not (math.isfinite(a) and math.isfinite(b)):
            return [path]
        if isinstance(a, int) != isinstance(b, int):
            return [path]
        if isinstance(a, int):
            return [] if a == b else [path]
        return [] if abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)) else [path]
    return [] if type(a) is type(b) and a == b else [path]


def _check(ok: bool, what: str) -> None:
    if not ok:
        raise Inconsistent(f"invalid record value: {what}")


def _count(v, what: str) -> None:
    _check(_num(v) and isinstance(v, int) and v >= 0, f"{what} = {v!r} is not a nonnegative integer")


def _time(v, what: str) -> None:
    _check(_num(v) and math.isfinite(v) and v >= 0, f"{what} = {v!r} is not a finite nonnegative time")


def _flag(v, what: str) -> None:
    _check(isinstance(v, bool), f"{what} = {v!r} is not a Boolean")


def _validate_values(reqs: list, scores: list) -> None:
    """Every value a displayed quantity is computed from, before it is used: correctness is integer 0 or 1, counts
    are nonnegative integers, times are finite and nonnegative, Boolean fields are Booleans."""
    for x in reqs + scores:
        _check(type(x["seed"]) is int, f"{x['task_id']}: seed {x['seed']!r} is not an integer")
    for x in scores:
        w = f"score {x['policy']}/{x['seed']}/{x['task_id']}"
        c = x["score"]["correct"]
        _check(_num(c) and isinstance(c, int) and c in (0, 1), f"{w}: correct = {c!r} is not integer 0 or 1")
        pr = x["score"]["predicted"]
        _time(pr["elapsed_seconds"], f"{w}: predicted.elapsed_seconds")
        for k in ("row_count", "distinct_rows"):
            if pr.get(k) is not None:
                _count(pr[k], f"{w}: predicted.{k}")
    for r in reqs:
        w = f"request {r['seed']}/{r['task_id']}"
        cands = [("original", r["original"])] + [(f"{p}.revised", r[p]["revised"]) for p in ("D1", "TI1")]
        for name, cand in cands:
            call, ex, ty = cand["call"], cand["execution"], cand["typed"]
            _time(call["seconds"], f"{w}: {name}.call.seconds")
            for k in ("prompt_tokens", "generated_tokens", "peak_memory_bytes"):
                _count(call[k], f"{w}: {name}.call.{k}")
            _time(ex["seconds"], f"{w}: {name}.execution.seconds")
            if ex.get("row_count") is not None:
                _count(ex["row_count"], f"{w}: {name}.execution.row_count")
            _flag(ex.get("truncated", False), f"{w}: {name}.execution.truncated")
            _time(ty["seconds"], f"{w}: {name}.typed.seconds")
            _flag(ty["typed"], f"{w}: {name}.typed.typed")
        _time(r["revision_seconds_allowed"], f"{w}: revision_seconds_allowed")


RULE_FIELDS = ("threshold", "conditions", "all_conditions", "sign", "rule_version")
HISTORICAL_FIELDS = ("threshold", "conditions", "all_conditions", "sign")   # as the frozen analysis wrote them
LEDGER_FIELDS = ("cap_hours", "authorization", "used_hours", "remaining_hours")
RECOMPUTED = ("counts", "pooled", "comparisons", "decision", "by_database", "lineage", "initial", "final_status",
              "cost", "gpu_reservations", "requests_per_policy")
ORIGINAL_KEYS = frozenset(RECOMPUTED + ("ledger", "protocol", "run", "seeds"))
CORRECTED_KEYS = ORIGINAL_KEYS | {"corrects", "inputs", "rule_version"}


def _identities(original: dict, corrected: dict, original_dir: Path, run: Path) -> list:
    """The fields not recomputed from the records: the key sets, the protocol, the run, the ledger path, and the
    corrected analysis's references, checked against the bundled files' own sha256."""
    from . import identity
    from . import spr_rule as RULE
    problems = []
    for name, a, keys in (("original", original, ORIGINAL_KEYS), ("corrected", corrected, CORRECTED_KEYS)):
        if set(a) != keys:
            problems.append(f"{name} fields {sorted(set(a) ^ keys)}")
            continue
        proto = json.loads((C.PINS / "spr_protocol.json").read_text())
        problems += [f"{name}{p}" for p in _differ(a["protocol"], {"sha256": proto["sha256"],
                                                                    "frozen_at": proto["frozen_at"]}, ".protocol")]
        if not isinstance(a["ledger"], dict) or not isinstance(a["ledger"].get("ledger"), str):
            problems.append(f"{name}.ledger.ledger")
    if problems:
        return problems
    rel = lambda path: "spr/" + path.relative_to(BUNDLE).as_posix()   # noqa: E731
    what = corrected["corrects"].get("what") if isinstance(corrected["corrects"], dict) else None
    if not isinstance(what, str) or not what:
        problems.append("corrected.corrects.what")
    expected = {"run": rel(run), "rule_version": RULE.RULE_VERSION,
                "corrects": {"analysis": rel(original_dir),
                             "analysis_sha256": identity.sha256_file(original_dir / "analysis.json"), "what": what},
                "inputs": {"scores": rel(run / "scores.jsonl"),
                           "scores_sha256": identity.sha256_file(run / "scores.jsonl")}}
    problems += [f"corrected{p}" for k, v in expected.items() for p in _differ(corrected[k], v, f".{k}")]
    problems += [f"original{p}" for p in _differ(original["run"], expected["run"], ".run")]
    return problems


def reconstruct() -> tuple:
    """(recomputed, original analysis, corrected analysis): the inventory validated first, every quantity
    recomputed from the raw records and checked against every field of both analyses. The original's rule fields are
    its known defect; they are checked against the shared helper's +7 values that produced them."""
    import copy
    evidence._checked(BUNDLE)
    adirs = sorted(BUNDLE.glob("analysis/*"))
    original_dir = next(d for d in adirs if (d / "analysis.json").exists())
    original = json.loads((original_dir / "analysis.json").read_text())
    v2_paths = [d / "analysis_v2.json" for d in adirs if (d / "analysis_v2.json").exists()]
    if len(v2_paths) != 1:
        raise Inconsistent(f"{len(v2_paths)} corrected analyses in the bundle; exactly one is required")
    corrected = json.loads(v2_paths[0].read_text())
    runs = sorted(BUNDLE.glob("run/*"))
    if len(runs) != 1 or not isinstance(original.get("run"), str) or \
            original["run"] != "spr/" + runs[0].relative_to(BUNDLE).as_posix():
        raise Inconsistent(f"the original analysis's run {original.get('run')!r} is not the bundle's one run")
    run = runs[0]
    reqs, scores = _load(run / "requests.jsonl"), _load(run / "scores.jsonl")
    pin = json.loads((C.PINS / "spr_new_eval.json").read_text())
    proto = json.loads((C.PINS / "spr_protocol.json").read_text())
    ids, db_of, seeds = pin["task_ids"], dict(zip(pin["task_ids"], pin["db_ids"])), proto["seeds"]
    if _differ(original.get("seeds"), seeds) or _differ(corrected.get("seeds"), seeds):
        raise Inconsistent("the analyses' seeds are not the frozen ones")
    _validate_inventory(reqs, scores, ids, db_of, seeds)
    _validate_values(reqs, scores)
    got = recompute(reqs, scores, ids, db_of, seeds, BUNDLE / "gpu_ledger.jsonl")
    problems = _identities(original, corrected, original_dir, run)
    if problems:
        raise Inconsistent("the analyses' fields disagree with the bundle: " + "; ".join(problems[:12]))
    for k in RECOMPUTED:
        problems += [f"corrected{p}" for p in _differ(got[k], corrected[k], f".{k}")]
        orig, mine = copy.deepcopy(original[k]), copy.deepcopy(got[k])
        if k == "comparisons":            # the original's rule fields are its known defect, pinned to the helper's
            for c, v in orig.items():
                problems += [f"original{p}" for p in _differ({f: v.get(f) for f in HISTORICAL_FIELDS},
                                                             got["historical_rule"][c], f".comparisons.{c}")]
            for v in orig.values():       # the original never had a rule_version: one there is a difference
                for f in HISTORICAL_FIELDS:
                    v.pop(f, None)
            for v in mine.values():
                for f in RULE_FIELDS:
                    v.pop(f, None)
        problems += [f"original{p}" for p in _differ(mine, orig, f".{k}")]
    for name, a in (("original", original), ("corrected", corrected)):
        problems += [f"{name}{p}" for p in _differ({k: a["ledger"].get(k) for k in LEDGER_FIELDS}, got["ledger"],
                                                   ".ledger")]
    if problems:
        raise Inconsistent("the records disagree with the analyses: " + "; ".join(problems[:12]))
    return got, original, corrected


def tables() -> str:
    got, original, corrected = reconstruct()
    seeds = corrected["seeds"]
    out = [f"Recomputed from the bundled raw records (inventory validated: seeds {seeds}, the 64 pinned questions, "
           f"{len(POLICIES)} policies), equal to the corrected analysis ({corrected['rule_version']}) and to the "
           "original analysis outside its rule fields.", "",
           "| policy | " + " | ".join(f"seed {s} (of 64)" for s in seeds) + " | pooled (of 128) |",
           "|---|" + "---|" * (len(seeds) + 1)]
    for p in POLICIES:
        v = [got["counts"][f"{p}/{s}"] for s in seeds]
        out.append(f"| {p} | " + " | ".join(map(str, v)) + f" | {sum(v)} |")
    out += ["", "| comparison | " + " | ".join(f"seed {s}" for s in seeds) + " | pooled | leaving one database out "
            "(min .. max) | recoveries | regressions | rule (+8): pooled / seeds / databases | meets the rule |",
            "|---|" + "---|" * (len(seeds) + 6)]
    for name, d in got["comparisons"].items():
        loo, c = d["leave_one_database_out"], d["conditions"]
        yn = lambda x: "yes" if x else "no"  # noqa: E731
        out.append(f"| {name} | " + " | ".join(f"{d['per_seed'][str(s)]:+d}" for s in seeds) +
                   f" | {d['pooled']:+d} | {min(loo.values()):+d} .. {max(loo.values()):+d} | {d['recoveries']} | "
                   f"{d['regressions']} | {yn(c['pooled_at_least_lead'])} / {yn(c['same_sign_every_seed'])} / "
                   f"{yn(c['same_sign_without_any_one_database'])} | {yn(d['all_conditions'])} |")
    out += ["", f"Decision (the registered rule): {got['decision']}.", "",
            "| database | questions | " + " | ".join(f"{p}, both seeds" for p in POLICIES) + " |",
            "|---|---|" + "---|" * len(POLICIES)]
    for db, v in got["by_database"].items():
        out.append(f"| {db} | {v['questions']} | " + " | ".join(str(v[p]) for p in POLICIES) + " |")
    ini = got["initial"]
    out += ["", f"Initial proposals (shared): {sum(1 for _ in seeds) * 64} requests; no SQL {ini['no_sql']}; "
                f"truncated {ini['truncated']}; typed {ini['typed']}; untyped reasons {ini['untyped_reasons']}; "
                f"execution {ini['execution']}; with typed errors {ini['with_errors']}; with warnings "
                f"{ini['with_warnings']}.", "",
            "| repair policy | final: revised / original / abstain | revision rejected | revision unchanged SQL | "
            "revision truncated | revised typed |", "|---|---|---|---|---|---|"]
    for p in ("D1", "TI1"):
        v = got["lineage"][p]
        src = v["final_source"]
        out.append(f"| {p} | {src.get('revised', 0)} / {src.get('original', 0)} / {src.get('abstain', 0)} | "
                   f"{v['revision_rejected']} | {v['revision_unchanged_sql']} | {v['revision_truncated']} | "
                   f"{v['revised_typed']} |")
    out += ["", "| policy | final execution status | generation s | validation s | tool s | submitted execution s | "
            "total s | per request s | per EX-correct s | prompt tokens | generated tokens | peak GB |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in POLICIES:
        c = got["cost"][p]
        pe = "—" if c["per_ex_correct_seconds"] is None else f"{c['per_ex_correct_seconds']:.1f}"
        out.append(f"| {p} | {got['final_status'][p]} | {c['generation_seconds']:.0f} | "
                   f"{c['validation_seconds']:.1f} | {c['tool_seconds']:.1f} | "
                   f"{c['submitted_execution_seconds']:.1f} | "
                   f"{c['total_seconds']:.0f} | {c['per_request_seconds']:.2f} | {pe} | {c['prompt_tokens']:,} | "
                   f"{c['generated_tokens']:,} | {c['peak_memory_bytes'] / 1e9:.1f} |")
    out += ["", "| GPU reservation (this work order) | hours | status |", "|---|---|---|"]
    out += [f"| {r['command']} ({r['id']}) | {r['hours']:.3f} | {r['status']} |" for r in got["gpu_reservations"]]
    out += [f"| total | {sum(r['hours'] for r in got['gpu_reservations']):.3f} | |", "",
            f"Ledger total in the bundled ledger: {got['ledger_used_hours']:.3f} of 32.0 h."]
    return "\n".join(out)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["bundle"]:
        print(f"{bundle()} files bundled")
    else:
        print(tables())
    return 0


if __name__ == "__main__":
    sys.exit(main())
