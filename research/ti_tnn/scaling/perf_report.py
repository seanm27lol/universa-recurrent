"""The performance study's report tables, rebuilt from the committed evidence bundles alone.

    python -m scaling.perf_report              # recompute from the records, compare with analysis.json, print tables
    python -m scaling.perf_report --questions  # the same checks, then every question of both populations
    python -m scaling.perf_report --corrections  # the 2026-11-03 corrections beside the recorded values

**What it reads:**
- `evidence/perf_v2/`: the counted evaluations' records, the training logs, the analysis, the baselines run in this
  study;
- `evidence/dev_v1/`: B4 and Q27 on the first 16;
- `evidence/pilot_v1/`: L on the 64, and the pilot's 1,500-update evaluations.

It checks every bundle against its `SHA256SUMS`. It recomputes the counts, the Q1 and Q2 rules (with the per-seed
and leave-one-database-out conditions), the per-database differences and the serving totals from the records, and
stops naming any disagreement with `analysis.json`. Then it prints the tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import contracts as C
from . import evidence
from . import perf
from . import perf_analysis as A

BUNDLE = C.SCALING / "evidence" / "perf_v2"
DEV = C.SCALING / "evidence" / "dev_v1"
PILOT = C.SCALING / "evidence" / "pilot_v1"


class Inconsistent(SystemExit):
    pass


def _load(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _in_bundle(ref: str) -> Path:
    """A run reference relative to the runs root ("perf_v2/eval/s100_...") inside the bundle."""
    parts = Path(ref).parts
    return BUNDLE.joinpath(*parts[1:]) if parts[0] == "perf_v2" else BUNDLE.joinpath(*parts)


def _baseline(arm: str, pop: str, ids: list):
    """Records covering a population from the bundles, or None."""
    recs = {}
    for tag in perf.BASELINE_RUNS[(arm, pop)]:
        found = sorted(BUNDLE.glob(f"baselines/*_{tag}")) or sorted(DEV.glob(f"baselines/*_{tag}")) or \
            sorted(PILOT.glob(f"L/*_{tag}"))
        if not found:
            return None
        for r in _load(found[0] / "records.jsonl"):
            recs[r["task_id"]] = r
    if sorted(recs) != sorted(ids):
        raise Inconsistent(f"{arm} on {pop}: the bundled runs cover {len(recs)} of {len(ids)}")
    return [recs[t] for t in ids]


def reconstruct() -> tuple:
    """(analysis, recomputed): every rule input recomputed from the bundled records."""
    for b in (BUNDLE, DEV, PILOT):
        evidence._checked(b)
    (path,) = sorted(BUNDLE.glob("analysis/*/analysis.json"))[-1:]
    analysis = json.loads(path.read_text())
    seeds = analysis["seeds"]
    eval_pin = json.loads(perf.EVAL_PIN.read_text())["populations"]
    db_of = {t: db for p in eval_pin.values() for t, db in zip(p["task_ids"], p["db_ids"])}
    correct, records = {}, {}
    for s in seeds:
        d = _in_bundle(analysis["inputs"]["evaluations_counted"][str(s)])
        for a in perf.ARMS:
            lab = perf.label(*a)
            recs = _load(d / lab / "records.jsonl")
            records[(lab, s)] = {r["task_id"]: r for r in recs}
            correct[(lab, s)] = {r["task_id"]: r["score"]["correct"] for r in recs}
    recomputed = {}
    for pop, p in eval_pin.items():
        ids = p["task_ids"]
        q1 = {arch: A.paired(correct, f"{arch}-v2", f"{arch}-v1", seeds, ids, db_of) for arch in ("T4", "C4")}
        q2 = A.paired(correct, "T4-v2", "C4-v2", seeds, ids, db_of)
        recomputed[pop] = {
            "counts": A.counts(correct, ids),
            "q1": {k: {"pooled": v["pooled"], "per_seed": v["per_seed"], "rule": A.rule_q1(v),
                       "leave_one_database_out": v["leave_one_database_out"]} for k, v in q1.items()},
            "q2": {"pooled": q2["pooled"], "per_seed": q2["per_seed"], "rule": A.rule_q2(q2),
                   "leave_one_database_out": q2["leave_one_database_out"]},
            "serving_pooled": {perf.label(*a): A.serving([records[(perf.label(*a), s)][t] for s in seeds for t in ids])
                               for a in perf.ARMS},
            "baselines": {arm: (lambda r: None if r is None else {"correct": sum(x["score"]["correct"] for x in r),
                                                                 "serving": A.serving(r)})(_baseline(arm, pop, ids))
                          for arm in ("L", "B4", "Q27")}}
    problems = []
    for pop, r in recomputed.items():
        a = analysis["populations"][pop]
        if r["counts"] != a["counts"]:
            problems.append(f"{pop}: counts")
        for k in ("T4", "C4"):
            for f in ("pooled", "per_seed", "rule", "leave_one_database_out"):
                if r["q1"][k][f] != a["q1"][k][f]:
                    problems.append(f"{pop}: q1 {k} {f}")
        for f in ("pooled", "per_seed", "rule", "leave_one_database_out"):
            if r["q2"][f] != a["q2"][f]:
                problems.append(f"{pop}: q2 {f}")
        for lab, v in r["serving_pooled"].items():
            if any(abs(v[k] - a["serving_pooled"][lab][k]) > 1e-6 for k in ("time_to_submission_seconds",
                                                                            "submitted_execution_seconds")):
                problems.append(f"{pop}: serving {lab}")
        for arm, v in r["baselines"].items():
            if (v is None) != (a["baselines"][arm] is None) or (v and v["correct"] != a["baselines"][arm]["correct"]):
                problems.append(f"{pop}: baseline {arm}")
    if problems:
        raise Inconsistent("analysis.json disagrees with the records: " + "; ".join(problems))
    return analysis, recomputed


def _f(x, digits=1):
    return "—" if x is None else f"{x:.{digits}f}"


def tables() -> str:
    analysis, rec = reconstruct()
    seeds = analysis["seeds"]
    labels = [perf.label(*a) for a in perf.ARMS]
    out = [f"Recomputed from the bundled records and equal to analysis.json (seeds {seeds}).", ""]
    for pop, r in rec.items():
        a = analysis["populations"][pop]
        out += [f"### {pop} population ({a['questions']} questions)", "",
                "| arm | " + " | ".join(f"seed {s}" for s in seeds) + " | pooled |",
                "|---|" + "---|" * (len(seeds) + 1)]
        for lab in labels:
            vals = [r["counts"][f"{lab}/{s}"] for s in seeds]
            out.append(f"| {lab} | " + " | ".join(map(str, vals)) + f" | {sum(vals)} of {len(seeds) * a['questions']} |")
        for arm in ("L", "B4", "Q27"):  # one run each, no training seed
            b = r["baselines"][arm]
            out.append(f"| {arm} | " + " | ".join(["—"] * len(seeds)) +
                       (f" | {b['correct']} of {a['questions']} (one run) |" if b else " | not run |"))
        out += ["", "| comparison | " + " | ".join(f"seed {s}" for s in seeds) + " | pooled | leaving one database "
                "out (min .. max) | rule | sign-flip p (exploratory) |", "|---|" + "---|" * (len(seeds) + 4)]
        for name, d, full in (("T4: v2 - v1", r["q1"]["T4"], a["q1"]["T4"]), ("C4: v2 - v1", r["q1"]["C4"],
                                                                            a["q1"]["C4"]),
                              ("v2: T4 - C4", r["q2"], a["q2"]), ("v1: T4 - C4 (descriptive)",
                                                                  a["q2_v1_descriptive"], a["q2_v1_descriptive"])):
            loo = d["leave_one_database_out"]
            out.append(f"| {name} | " + " | ".join(f"{d['per_seed'][str(s)]:+d}" for s in seeds) +
                       f" | {d['pooled']:+d} | {min(loo.values()):+d} .. {max(loo.values()):+d} | {d['rule']} | "
                       f"{full['exploratory_sign_flip']['p_two_sided']:.3f} |")
        out += ["", f"Reading: {a['reading']}", "", "| arm (pooled over seeds) | requests | EX-correct | time to "
                "submission, mean s | submitted execution, s (timeouts) | generation per EX-correct, s | generation + "
                "execution per EX-correct, s | controller MACs per request, as recorded |", "|---|---|---|---|---|---|---|---|"]
        for lab in labels:
            v = r["serving_pooled"][lab]
            m = a["macs"][lab]["active_macs_per_request_mean"]
            out.append(f"| {lab} | {v['requests']} | {v['ex_correct']} | {_f(v['mean_time_to_submission'], 2)} | "
                       f"{_f(v['submitted_execution_seconds'])} ({v['timeouts']}) | "
                       f"{_f(v['generation_per_ex_correct'])} | {_f(v['together_per_ex_correct'])} | "
                       f"{m / 1e9:.2f} G |")
        for arm in ("L", "B4", "Q27"):
            b = r["baselines"][arm]
            if b:
                v = b["serving"]
                out.append(f"| {arm} | {v['requests']} | {v['ex_correct']} | {_f(v['mean_time_to_submission'], 2)} | "
                           f"{_f(v['submitted_execution_seconds'])} ({v['timeouts']}) | "
                           f"{_f(v['generation_per_ex_correct'])} | {_f(v['together_per_ex_correct'])} | — |")
        out += ["", "| database | questions | " + " | ".join(labels) + " | T4 v2-v1 | C4 v2-v1 | v2 T4-C4 |",
                "|---|---|" + "---|" * (len(labels) + 3)]
        dbs = sorted(a["q2"]["per_database"])
        per_q = a["per_question"]
        for db in dbs:
            ids = [t for t in per_q if analysis_db(t) == db]
            cells = [sum(per_q[t][f"{lab}/{s}"] for t in ids for s in seeds) for lab in labels]
            out.append(f"| {db} | {len(ids)} | " + " | ".join(map(str, cells)) +
                       f" | {a['q1']['T4']['per_database'][db]:+d} | {a['q1']['C4']['per_database'][db]:+d} | "
                       f"{a['q2']['per_database'][db]:+d} |")
        out += ["", "| stratum | value | questions | " + " | ".join(labels) + " |", "|---|---|---|" + "---|" * len(labels)]
        for dim, vals in a["strata"].items():
            for value, row in sorted(vals.items()):
                out.append(f"| {dim} | {value} | {row['questions']} | " + " | ".join(str(row[lab]) for lab in labels)
                           + " |")
        out.append("")
    out += ["### Training", "", "| fit | updates | loss first -> last | s per update | reservation h | attempts | "
            "peak memory GB | trainable parameters |", "|---|---|---|---|---|---|---|---|"]
    for k, v in analysis["training"].items():
        out.append(f"| {k} | {v['updates']} | {v['loss_first_last'][0]:.3f} -> {v['loss_first_last'][1]:.3f} | "
                   f"{v['seconds_per_update_mean']:.3f} | {v['reservation_hours']:.3f} | {v['attempts']} | "
                   f"{(v['peak_memory_bytes'] or 0) / 1e9:.1f} | {v['parameters']['total']:,} |")
    out += ["", "The pilot's fits (1,500 updates, seeds 100-101) on the historical 64: " + ", ".join(
        f"{k} {v}" for k, v in analysis["pilot_1500_updates_historical"].items()) + ".", "",
        f"Ledger: {analysis['ledger']['used_hours']:.3f} of {analysis['ledger']['cap_hours']} h used."]
    return "\n".join(out)


def questions() -> str:
    """Every question of both populations: each arm's EX-correct seeds (with submitted-query timeouts and errors),
    and each baseline's outcome with its tool calls. The SQL itself is in the bundled records."""
    analysis, rec = reconstruct()
    seeds = analysis["seeds"]
    labels = [perf.label(*a) for a in perf.ARMS]
    records = {}
    for s in seeds:
        d = _in_bundle(analysis["inputs"]["evaluations_counted"][str(s)])
        for lab in labels:
            records[(lab, s)] = {r["task_id"]: r for r in _load(d / lab / "records.jsonl")}
    pin = json.loads(perf.EVAL_PIN.read_text())["populations"]
    out = ["Per question: an arm's cell is its EX-correct seeds of " + str(len(seeds)) + ", with t = a submitted-query "
           "timeout and e = an execution error (one letter per seed). A baseline's cell is 1 or 0 (a = abstained), "
           "with its tool calls. Recomputed from the bundled records.", ""]
    for pop, p in pin.items():
        base = {arm: (None if _baseline(arm, pop, p["task_ids"]) is None else
                      {r["task_id"]: r for r in _baseline(arm, pop, p["task_ids"])}) for arm in ("L", "B4", "Q27")}
        out += [f"### {pop} population", "", "| question | database | " + " | ".join(labels) + " | L | B4 | Q27 |",
                "|---|---|" + "---|" * (len(labels) + 3)]
        for t, db in zip(p["task_ids"], p["db_ids"]):
            cells = []
            for lab in labels:
                rs = [records[(lab, s)][t] for s in seeds]
                flags = "".join({"timeout": "t", "error": "e"}.get(r["score"]["predicted"]["status"], "") for r in rs)
                cells.append(f"{sum(r['score']['correct'] for r in rs)}{' ' + flags if flags else ''}")
            for arm in ("L", "B4", "Q27"):
                r = base[arm][t] if base[arm] else None
                cells.append("—" if r is None else
                             f"{'a' if r['final']['kind'] == 'abstain' else r['score']['correct']} ({r['tool_calls']})")
            out.append(f"| {t} | {db} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def corrections() -> str:
    """The 2026-11-03 corrections, from the bundle alone and beside the recorded values (which stay as they are):
    controller MACs as executed, the machine's load while Qwen was timed, the first-table split and the costs kept
    apart."""
    import glob
    import statistics

    from . import controller as K
    from . import ledger
    from . import perf_corrections as PC
    analysis, _rec = reconstruct()
    seeds = analysis["seeds"]
    labels = [perf.label(*a) for a in perf.ARMS]
    pin = json.loads(perf.EVAL_PIN.read_text())["populations"]
    out = ["### Controller MACs per request: as recorded and as executed", "",
           "Controller only; the frozen 4B backbone's forward over the context is not included in either column. "
           "As recorded: the chosen operator's pointer kind only, and no STOP step. As executed: every STOP step's "
           "trunk, and every pointer head `decode_batch` runs (`perf_corrections.executed_macs`).", "",
           "| population | arm | requests | as recorded, mean | as executed, mean | difference |", "|---|---|---|---|---|---|"]
    for pop, p in pin.items():
        ids = set(p["task_ids"])
        for lab in labels:
            rows = []
            for s in seeds:
                cfg = K.ControllerConfig(**json.loads((BUNDLE / "train" / f"{lab}_s{s}" / "config.json").read_text())
                                         ["controller"])
                d = _in_bundle(analysis["inputs"]["evaluations_counted"][str(s)])
                rows += [PC.executed_macs(r, cfg) for r in _load(d / lab / "records.jsonl") if r["task_id"] in ids]
            rec = statistics.mean(r["recorded"] for r in rows)
            exe = statistics.mean(r["executed"] for r in rows)
            out.append(f"| {pop} | {lab} | {len(rows)} | {rec / 1e9:.3f} G | {exe / 1e9:.3f} G | "
                       f"+{100 * (exe / rec - 1):.2f}% |")
    lines = [json.loads(x) for x in (BUNDLE / "gpu_ledger.jsonl").read_text().splitlines() if x.strip()]
    windows = PC.qwen_windows(lines)
    logs = [x for f in sorted(glob.glob(str(BUNDLE / "logs" / "cotenancy_*.log"))) for x in Path(f).read_text()
            .splitlines()]
    overlap = json.loads((BUNDLE / "corrections" / "cpu_overlap.json").read_text())
    out += ["", "### Machine load while Qwen was timed (disclosure; no correction applied)", "",
            "| reservation | minutes | load samples | 1-min load mean | median | max | share above 6 | overlapping "
            "producer or reviewer CPU work |", "|---|---|---|---|---|---|---|---|"]
    for name, w in overlap["windows"].items():
        cmd = w["reservation_command"]
        a, b = windows[cmd]
        v = PC.load_during(logs, {cmd: (a, b)})[cmd]
        out.append(f"| {name} | {(b - a) / 60:.0f} | {v['samples']} | {v['mean']:.2f} | {v['median']:.2f} | "
                   f"{v['max']:.2f} | {v['share_above_6']:.2f} | {overlap['scope'][name]} |")
    (ft_path,) = sorted(BUNDLE.glob("diagnosis/*_first_table_*/first_table.json"))[-1:]
    ft = json.loads(ft_path.read_text())["fits"]
    out += ["", "### Teacher-forced FROM-table decisions: wrong order or irrelevant table", "",
            "The pointer only (a wrong family or operator at the step is not counted here). In gold set: a table the "
            "gold program uses, chosen in another order; irrelevant: a table the gold program does not use.", "",
            "| population | arm | FROM-table steps | right | wrong, in gold set | wrong, irrelevant | first FROM of the "
            "program: right / in gold set / irrelevant |", "|---|---|---|---|---|---|---|"]
    for pop in pin:
        for lab in labels:
            c = {k: sum(ft[f"{lab}/{s}/{pop}"][k] for s in seeds) for k in ft[f"{lab}/{seeds[0]}/{pop}"]}
            n, t = c["steps"], c["top_steps"]
            out.append(f"| {pop} | {lab} | {n} | {c['right'] / n:.2f} | {c['wrong_in_gold_set'] / n:.2f} | "
                       f"{c['wrong_irrelevant'] / n:.2f} | {c['top_right'] / t:.2f} / "
                       f"{c['top_wrong_in_gold_set'] / t:.2f} / {c['top_wrong_irrelevant'] / t:.2f} |")
    cost = PC.cost_separation(ledger.usage(BUNDLE / "gpu_ledger.jsonl")["reservations"])
    st, rs = cost["setup"], cost["research"]
    out += ["", "### Costs, kept apart (GPU-reservation hours; never added together)", "",
            "| kind | what | hours |", "|---|---|---|",
            f"| standalone setup | the training pool's backbone states, completed encode reservations "
            f"({st['training_states_reservations']}) | {st['training_states_hours']:.3f} |"]
    out += [f"| standalone setup | one {lab} fit, mean of {len(seeds)} | {h:.3f} |"
            for lab, h in st["one_fit_hours_mean"].items()]
    out += [f"| research | encode attempts that did not complete (the pilot's: one cut off by the host crash, one failed) | "
            f"{st['training_states_interrupted_hours']:.3f} |",
            f"| research | everything under authorization v2 ({rs['authorization_v2_reservations']} reservations) | "
            f"{rs['authorization_v2_hours']:.3f} |",
            f"| research | the whole ledger, the pilot and earlier included ({rs['all_reservations']} reservations) | "
            f"{rs['all_hours']:.3f} |", "",
            "Marginal serving is per request and per EX-correct answer, measured in the records (the serving tables "
            "above); it is not a ledger quantity."]
    return "\n".join(out)


_DB = None


def analysis_db(task_id: str) -> str:
    global _DB
    if _DB is None:
        pin = json.loads(perf.EVAL_PIN.read_text())["populations"]
        _DB = {t: db for p in pin.values() for t, db in zip(p["task_ids"], p["db_ids"])}
    return _DB[task_id]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(questions() if argv == ["--questions"] else corrections() if argv == ["--corrections"] else tables())
    return 0


if __name__ == "__main__":
    sys.exit(main())
