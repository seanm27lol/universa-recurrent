"""The grounding-interface study's evidence bundle and report tables (development).

    python -m scaling.gi_report bundle    # copy the study's runs (weight-free) and the ledger into the bundle; sums
    python -m scaling.gi_report           # recompute from the bundle, compare with analysis.json, print the tables

**The bundle** is `evidence/grounding_interface_v1/`. It holds every file of `<runs root>/grounding_interface/`
(no weights are written there) and the GPU ledger, under `SHA256SUMS`.

**The report** checks the sums, then recomputes from the bundled records alone:
- the EX counts per policy, architecture and seed;
- the paired comparisons and their three conditions;
- the architecture interaction;
- the direct control;
- the swap check;
- the costs.

It stops on any disagreement with `analysis.json`. The record diagnostics against the gold annotations need the
BIRD release, so they are printed from `analysis.json` and checked only for internal consistency.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from . import contracts as C
from . import evidence
from . import perf_analysis as PA

BUNDLE = C.SCALING / "evidence" / "grounding_interface_v1"
POLICIES = ("G0", "Glex", "G4")


class Inconsistent(SystemExit):
    pass


def bundle(dest: Path = BUNDLE) -> int:
    from . import gi_run
    n = 0
    for f in sorted(gi_run.ROOT.rglob("*")):
        if f.is_file():
            target = dest / f.relative_to(gi_run.ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            n += 1
    shutil.copy2(C.RUNS / "gpu_ledger.jsonl", dest / "gpu_ledger.jsonl")
    evidence.write_sums(dest)
    return n + 1


def _load(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _latest(kind: str, suffix: str = "") -> Path:
    runs = sorted(d for d in (BUNDLE / kind).glob(f"*{suffix}") if d.is_dir())
    if not runs:
        raise Inconsistent(f"no {kind} run in the bundle")
    return runs[-1]


def reconstruct() -> tuple:
    evidence._checked(BUNDLE)
    analysis = json.loads((_latest("analysis") / "analysis.json").read_text())
    ev = BUNDLE.joinpath(*Path(analysis["eval_run"]).parts[1:])
    pin = json.loads((C.PINS / "gi_new_eval.json").read_text())
    ids, db_of = pin["task_ids"], dict(zip(pin["task_ids"], pin["db_ids"]))
    recs = _load(ev / "records.jsonl")
    seeds = analysis["seeds"]
    correct, by = {}, {}
    for r in recs:
        arch, seed = r["fit"].split("-")[0], int(r["fit"].split("/")[1])
        correct.setdefault((f"{arch}-{r['policy']}", seed), {})[r["task_id"]] = r["score"]["correct"]
        by[(arch, r["policy"], seed, r["task_id"])] = r
    counts = {f"{lab}/{s}": sum(v.values()) for (lab, s), v in sorted(correct.items())}
    problems = []
    if counts != analysis["counts"]:
        problems.append("counts")
    comps = {}
    for arch in ("T4", "C4"):
        comps[arch] = {}
        for a, b in (("G4", "G0"), ("Glex", "G0"), ("G4", "Glex")):
            d = PA.paired(correct, f"{arch}-{a}", f"{arch}-{b}", seeds, ids, db_of)
            comps[arch][f"{a}-{b}"] = d
            full = analysis["comparisons"][arch][f"{a}-{b}"]
            for k in ("pooled", "per_seed", "all_conditions", "leave_one_database_out"):
                if d[k] != full[k]:
                    problems.append(f"{arch} {a}-{b} {k}")
    inter = {("I", s): {t: correct[("T4-G4", s)][t] - correct[("T4-G0", s)][t] for t in ids} for s in seeds}
    inter.update({("Z", s): {t: correct[("C4-G4", s)][t] - correct[("C4-G0", s)][t] for t in ids} for s in seeds})
    di = PA.paired(inter, "I", "Z", seeds, ids, db_of)
    if (di["pooled"], di["all_conditions"]) != (analysis["interaction_T4_minus_C4_of_G4_minus_G0"]["pooled"],
                                                analysis["interaction_T4_minus_C4_of_G4_minus_G0"]["all_conditions"]):
        problems.append("interaction")
    direct = _load(BUNDLE.joinpath(*Path(analysis["direct"]["run"]).parts[1:]) / "records.jsonl")
    if sum(r["score"]["correct"] for r in direct) != analysis["direct"]["correct"]:
        problems.append("direct")
    g4 = {r["task_id"]: r for r in _load(BUNDLE.joinpath(*Path(analysis["records"]["G4"]["ground_run"]).parts[1:])
                                         / "records.jsonl")}
    for key, v in analysis["cost"].items():
        if key == "direct-B4-short":
            continue
        arch, p = key.split("-", 1)
        rows = [by[(arch, p, s, t)] for s in seeds for t in ids]
        tot = sum(r["controller_seconds"] for r in rows) + sum(r["score"]["predicted"]["elapsed_seconds"] for r in rows)
        tot += sum(g4[r["task_id"]]["seconds"] for r in rows) if p == "G4" else 0.0
        if abs(tot / len(rows) - v["per_request_seconds"]) > 1e-6:
            problems.append(f"cost {key}")
    if problems:
        raise Inconsistent("analysis.json disagrees with the records: " + "; ".join(problems))
    return analysis, counts, comps, di


def tables() -> str:
    analysis, counts, comps, di = reconstruct()
    seeds = analysis["seeds"]
    out = [f"Recomputed from the bundled records and equal to analysis.json (seeds {seeds}).", "",
           "| arm | policy | " + " | ".join(f"seed {s}" for s in seeds) + " | pooled of 192 |",
           "|---|---|" + "---|" * (len(seeds) + 1)]
    for arch in ("T4", "C4"):
        for p in POLICIES:
            v = [counts[f"{arch}-{p}/{s}"] for s in seeds]
            out.append(f"| {arch}-v2 | {p} | " + " | ".join(map(str, v)) + f" | {sum(v)} |")
    d = analysis["direct"]
    out.append(f"| direct Qwen3.5-4B (short budget) | one run | — | — | — | {d['correct']} of 64 |")
    out += ["", "| comparison | " + " | ".join(f"seed {s}" for s in seeds) + " | pooled | leaving one database out "
            "(min .. max) | consistent | sign-flip p (exploratory) |", "|---|" + "---|" * (len(seeds) + 4)]
    rows = [(f"{a}: {k}", comps[a][k]) for a in ("T4", "C4") for k in ("G4-G0", "Glex-G0", "G4-Glex")]
    rows.append(("(T4: G4-G0) - (C4: G4-G0)", di))
    for name, c in rows:
        loo = c["leave_one_database_out"]
        out.append(f"| {name} | " + " | ".join(f"{c['per_seed'][str(s)]:+d}" for s in seeds) +
                   f" | {c['pooled']:+d} | {min(loo.values()):+d} .. {max(loo.values()):+d} | "
                   f"{'yes' if c['all_conditions'] else 'no'} | {c['exploratory_sign_flip']['p_two_sided']:.3f} |")
    out += ["", "Readings:"] + [f"- {r}" for r in analysis["reading"]]
    out += ["", "| arm-policy | requests | EX-correct | per request, s | per EX-correct, s | controller s | G4 call s | "
            "execution s | timeouts | abstentions |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in analysis["cost"].items():
        out.append(f"| {k} | {v['requests']} | {v['ex_correct']} | {v['per_request_seconds']:.2f} | "
                   f"{'—' if v['per_ex_correct_seconds'] is None else format(v['per_ex_correct_seconds'], '.1f')} | "
                   f"{v.get('controller_seconds', 0):.0f} | {v.get('grounding_call_seconds', 0):.0f} | "
                   f"{v.get('submitted_execution_seconds', 0):.0f} | {v.get('timeouts', '—')} | "
                   f"{v.get('abstentions', '—')} |")
    out += ["", "| decodes against G0 | changed decisions | changed SQL | wrong to correct | correct to wrong | fallback "
            "to G0 | EX-correct with another table set |", "|---|---|---|---|---|---|---|"]
    for k, v in analysis["decodes"].items():
        out.append(f"| {k} | {v['changed_decisions_vs_G0']} | {v['changed_sql_vs_G0']} | {v['wrong_to_correct']} | "
                   f"{v['correct_to_wrong']} | {v['fallback_to_G0']} | {v['ex_correct_with_another_table_set']} |")
    out += ["", "| record (new population) | valid | table recall | table precision | table set exact | first listed in "
            "gold set | first = gold first FROM | column coverage | literal value accuracy | literal pair accuracy |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for p in ("G4", "Glex"):
        v = analysis["records"][p]

        def f(x):
            return "—" if x is None else f"{x:.2f}"
        out.append(f"| {p} | {v.get('valid', v['scored'])} | {f(v['table_recall'])} | {f(v['table_precision'])} | "
                   f"{f(v['table_set_exact'])} | {f(v['first_in_gold_set'])} | {f(v['first_equals_gold_first'])} | "
                   f"{f(v['column_coverage'])} | {f(v['literal_value_accuracy'])} | {f(v['literal_pair_accuracy'])} |")
    if "swap" in analysis:
        s = analysis["swap"]
        out += ["", f"Swap check: another question's G4 record (same database) changes the final SQL of "
                    f"{s['changed_sql']} of {s['decodes']} decodes; EX-correct {s['own_correct']} with the own "
                    f"record, {s['swapped_correct']} with the swapped one."]
    out += ["", "GPU-reservation hours: " + ", ".join(f"{k} {v:.3f}" for k, v in analysis["gpu_hours"].items()) +
            f". Ledger: {analysis['ledger']['used_hours']:.3f} of {analysis['ledger']['cap_hours']} h."]
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
