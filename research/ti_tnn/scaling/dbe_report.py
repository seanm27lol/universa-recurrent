"""The database-evidence revision study's evidence bundle and report tables (written after the frozen run).

    python -m scaling.dbe_report bundle    # copy the study's run, analysis and logs, and the ledger, into the bundle
    python -m scaling.dbe_report           # recompute from the bundle, compare with analysis.json, print tables

**The bundle** is `evidence/dbe_v1/`: every file of `<runs root>/dbe/` and the GPU ledger, under `SHA256SUMS`.

**The report.** It checks the sums and the frozen pins, then validates the records:
- exactly one request per question and seed;
- one score per policy, seed and question;
- integer 0/1 correctness;
- finite, nonnegative times.

It recomputes every analysed quantity from the bundled records with the frozen `dbe_run.analyze`, and refuses any
disagreement with `analysis.json`. This module changes nothing the run used: the policies, the lookup, the rule and
the analysis are the frozen harness's.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

from . import contracts as C
from . import evidence
from .spr_report import Inconsistent, _differ

BUNDLE = C.SCALING / "evidence" / "dbe_v1"


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


def _validate(reqs: list, scores: list, ids: list, seeds) -> None:
    from collections import Counter

    from . import dbe_run as D
    want_r = {(s, t) for s in seeds for t in ids}
    got_r = Counter((r["seed"], r["task_id"]) for r in reqs)
    if set(got_r) != want_r or any(n != 1 for n in got_r.values()):
        raise Inconsistent("requests: not exactly one per question and seed")
    want_s = {(p, s, t) for p in D.POLICIES for s in seeds for t in ids}
    got_s = Counter((x["policy"], x["seed"], x["task_id"]) for x in scores)
    if set(got_s) != want_s or any(n != 1 for n in got_s.values()):
        raise Inconsistent("scores: not exactly one per policy, seed and question")
    for x in scores:
        c = x["score"]["correct"]
        if type(c) is not int or c not in (0, 1):
            raise Inconsistent(f"{x['task_id']}/{x['seed']}/{x['policy']}: correctness {c!r}")
    for r in reqs:
        times = [r["original"]["call"]["seconds"], r["evidence"]["seconds"]] + \
            [r[p]["revised"]["call"]["seconds"] for p in ("D1", "V1")]
        if any(not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0)
               for v in times):
            raise Inconsistent(f"{r['task_id']}/{r['seed']}: a time is not finite and nonnegative")
        if r["D0"] != {"source": "original", "sql": r["original"]["sql"]}:
            raise Inconsistent(f"{r['task_id']}/{r['seed']}: D0 is not the shared initial proposal")


def reconstruct() -> tuple:
    """(recomputed, analysis): every analysed quantity from the bundled records, checked against analysis.json."""
    from . import dbe_run as D
    evidence._checked(BUNDLE)
    (run,) = sorted(p for p in BUNDLE.glob("run/*") if (p / "scores.jsonl").exists())
    (adir,) = sorted(BUNDLE.glob("analysis/*"))
    analysis = json.loads((adir / "analysis.json").read_text())
    pin = json.loads(D.NEW_PIN.read_text())
    ids, db_of = pin["task_ids"], dict(zip(pin["task_ids"], pin["db_ids"]))
    reqs, scores = _load(run / "requests.jsonl"), _load(run / "scores.jsonl")
    _validate(reqs, scores, ids, D.SEEDS)
    got = D.analyze(reqs, scores, ids, db_of, D.SEEDS)
    problems = [p for k in got for p in _differ(got[k], analysis.get(k), f".{k}")]
    if problems:
        raise Inconsistent("the records disagree with analysis.json: " + "; ".join(problems[:12]))
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
                                                 for r in analysis["gpu_reservations"]) +
            f". Ledger: {analysis['ledger']['used_hours']:.3f} of {analysis['ledger']['cap_hours']} h."]
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
