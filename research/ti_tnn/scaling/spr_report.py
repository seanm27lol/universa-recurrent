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


def reconstruct() -> tuple:
    from . import spr_run
    evidence._checked(BUNDLE)
    (adir,) = sorted(BUNDLE.glob("analysis/*"))[-1:]
    analysis = json.loads((adir / "analysis.json").read_text())
    run = BUNDLE.joinpath(*Path(analysis["run"]).parts[1:])
    scores, reqs = _load(run / "scores.jsonl"), _load(run / "requests.jsonl")
    pin = json.loads((C.PINS / "spr_new_eval.json").read_text())
    ids, db_of = pin["task_ids"], dict(zip(pin["task_ids"], pin["db_ids"]))
    seeds = analysis["seeds"]
    correct = {}
    for x in scores:
        correct.setdefault((x["policy"], x["seed"]), {})[x["task_id"]] = x["score"]["correct"]
    problems = []
    counts = {f"{p}/{s}": sum(correct[(p, s)].values()) for p in POLICIES for s in seeds}
    if counts != analysis["counts"]:
        problems.append("counts")
    comps = {}
    for a, b in (("TI1", "D1"), ("D1", "D0"), ("TI1", "D0")):
        d = PA.paired(correct, a, b, seeds, ids, db_of)
        d["recoveries"] = sum(correct[(a, s)][t] > correct[(b, s)][t] for s in seeds for t in ids)
        d["regressions"] = sum(correct[(a, s)][t] < correct[(b, s)][t] for s in seeds for t in ids)
        comps[f"{a}-{b}"] = d
        full = analysis["comparisons"][f"{a}-{b}"]
        for k in ("pooled", "per_seed", "leave_one_database_out", "recoveries", "regressions"):
            if d[k] != full[k]:
                problems.append(f"{a}-{b} {k}")
    if spr_run._decision(comps) != analysis["decision"]:
        problems.append("decision")
    for p in ("D1", "TI1"):
        if sum(1 for r in reqs if r[p]["source"] != "revised") != analysis["lineage"][p]["revision_rejected"]:
            problems.append(f"lineage {p}")
    if problems:
        raise Inconsistent("analysis.json disagrees with the records: " + "; ".join(problems))
    return analysis, counts, comps


def tables() -> str:
    analysis, counts, comps = reconstruct()
    seeds = analysis["seeds"]
    out = [f"Recomputed from the bundled records and equal to analysis.json (seeds {seeds}).", "",
           "| policy | " + " | ".join(f"seed {s} (of 64)" for s in seeds) + " | pooled (of 128) |",
           "|---|" + "---|" * (len(seeds) + 1)]
    for p in POLICIES:
        v = [counts[f"{p}/{s}"] for s in seeds]
        out.append(f"| {p} | " + " | ".join(map(str, v)) + f" | {sum(v)} |")
    out += ["", "| comparison | " + " | ".join(f"seed {s}" for s in seeds) + " | pooled | leaving one database out "
            "(min .. max) | recoveries | regressions |", "|---|" + "---|" * (len(seeds) + 4)]
    for name, d in comps.items():
        loo = d["leave_one_database_out"]
        out.append(f"| {name} | " + " | ".join(f"{d['per_seed'][str(s)]:+d}" for s in seeds) +
                   f" | {d['pooled']:+d} | {min(loo.values()):+d} .. {max(loo.values()):+d} | {d['recoveries']} | "
                   f"{d['regressions']} |")
    out += ["", f"Decision (the protocol's rule): {analysis['decision']}.", "",
            "| database | questions | " + " | ".join(f"{p}, both seeds" for p in POLICIES) + " |",
            "|---|---|" + "---|" * len(POLICIES)]
    for db, v in analysis["by_database"].items():
        out.append(f"| {db} | {v['questions']} | " + " | ".join(str(v[p]) for p in POLICIES) + " |")
    ini = analysis["initial"]
    out += ["", f"Initial proposals (shared): {analysis['requests_per_policy']} requests; no SQL {ini['no_sql']}; "
                f"truncated {ini['truncated']}; typed {ini['typed']}; untyped reasons {ini['untyped_reasons']}; "
                f"execution {ini['execution']}; with typed errors {ini['with_errors']}; with warnings "
                f"{ini['with_warnings']}.", "",
            "| repair policy | final: revised / original / abstain | revision rejected | revision unchanged SQL | "
            "revision truncated | revised typed |", "|---|---|---|---|---|---|"]
    for p in ("D1", "TI1"):
        v = analysis["lineage"][p]
        src = v["final_source"]
        out.append(f"| {p} | {src.get('revised', 0)} / {src.get('original', 0)} / {src.get('abstain', 0)} | "
                   f"{v['revision_rejected']} | {v['revision_unchanged_sql']} | {v['revision_truncated']} | "
                   f"{v['revised_typed']} |")
    out += ["", "| policy | final execution status | generation s | validation s | tool s | submitted execution s | "
            "total s | per request s | per EX-correct s | prompt tokens | generated tokens | peak GB |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in POLICIES:
        c = analysis["cost"][p]
        pe = "—" if c["per_ex_correct_seconds"] is None else f"{c['per_ex_correct_seconds']:.1f}"
        out.append(f"| {p} | {analysis['final_status'][p]} | {c['generation_seconds']:.0f} | "
                   f"{c['validation_seconds']:.1f} | {c['tool_seconds']:.1f} | {c['submitted_execution_seconds']:.1f} | "
                   f"{c['total_seconds']:.0f} | {c['per_request_seconds']:.2f} | {pe} | {c['prompt_tokens']:,} | "
                   f"{c['generated_tokens']:,} | {c['peak_memory_bytes'] / 1e9:.1f} |")
    out += ["", "| GPU reservation (this work order) | hours | status |", "|---|---|---|"]
    out += [f"| {r['command']} ({r['id']}) | {r['hours']:.3f} | {r['status']} |" for r in analysis["gpu_reservations"]]
    out += [f"| total | {sum(r['hours'] for r in analysis['gpu_reservations']):.3f} | |", "",
            f"Ledger at the analysis: {analysis['ledger']['used_hours']:.3f} of {analysis['ledger']['cap_hours']} h."]
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
