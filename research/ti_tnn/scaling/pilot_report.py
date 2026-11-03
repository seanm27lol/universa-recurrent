"""The pilot report's tables, rebuilt from the pilot evidence bundle alone (`evidence/pilot_v1`).

    python -m scaling.pilot_report [BUNDLE]

**What it does first.** It refuses to run if any bundled file differs from `SHA256SUMS`. Then it recomputes the
decision from the decision records and compares it with the bundled `analysis.json`; any disagreement stops it. The
recomputation:
- **eligibility:** each evaluation run's eligibility, from its complete reconstruction inventory
  (`pilot_run.eval_failures`), giving the one counted run per seed;
- **records:** each counted arm's records and the L run's, validated, cover the 64 exactly once;
- **the analysis:** the counts, L statuses, paired differences, database totals, per-question histogram, rule and
  sign-flip block statistic, recomputed with the frozen analysis (`pilot.analyze`) from each record's recorded EX
  score;
- **the totals:** each run's totals, recomputed from its records against its summary.

Runs are matched by their reference relative to the runs root (`identity.run_ref`), never by absolute path, so the
bundle can be moved or read from another checkout.

This is the public arithmetic reconstruction: it reads only the bundle. Re-executing the SQL, re-hashing checkpoints
and reading cached states need the local runs (`independent/verify_pilot.py --local`).

**Then it prints the tables:**
- the counts on the full 64-question denominator;
- per-seed and pooled differences;
- each arm's "cleared L in both seeds" status and the registered rule outcome;
- per-database results;
- the exploratory block analysis;
- gold-program representability, with performance within its strata;
- serving: time to SQL submission and generation cost per EX-correct submission (the frozen metric), with the
  scorer's recorded execution of each submitted query reported separately as descriptive accounting;
- projected against actual GPU hours per stage, with every reservation in the ledger, failed, interrupted and
  superseded ones included.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path, PurePath

from . import evidence, identity
from . import pilot as P
from . import pilot_run
from . import records as R
from . import verify

COMPARED = ("counts", "cleared_L_in_both_seeds", "differences", "rule", "lead_threshold",
            "per_question_difference_histogram", "databases", "exploratory_sign_flip")


class Inconsistent(SystemExit):
    """The bundle's analysis.json or summaries disagree with its decision records."""


def _load(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def bundle_dir(bundle: Path, recorded) -> Path:
    """A recorded run (absolute path or reference) -> its directory in the bundle: `pilot/<x>` is at `<x>`, and an L
    run `baselines/<x>` is at `L/<x>`."""
    parts = PurePath(identity.run_ref(recorded)).parts
    if parts[0] == "pilot":
        return bundle.joinpath(*parts[1:])
    if parts[0] == "baselines":
        return bundle.joinpath("L", *parts[1:])
    raise Inconsistent(f"{recorded}: not a pilot or L run")


def serving(records: list) -> dict:
    """Serving accounting from one run's records (descriptive except where marked):
    - `submission_seconds`: time to SQL submission, summed (the records' `elapsed_seconds`: from the request's start,
      including the fresh backbone forward, or L's retrieval and candidate queries, until the final SQL is submitted);
    - `generation_cost_per_ex_correct`: that sum / EX-correct submissions, the frozen protocol's cost unit;
    - `submitted_execution_seconds`: the scorer's recorded execution of each submitted query (`score.predicted`),
      not part of the frozen metric; queries stopped by the scorer's 30 s timeout are counted as recorded;
    - `with_execution_per_ex_correct`: (generation + that execution) / EX-correct submissions, descriptive."""
    n = len(records)
    correct = sum(r["score"]["correct"] for r in records)
    gen = sum(r["elapsed_seconds"] for r in records)
    execution = sum(((r["score"].get("predicted") or {}).get("elapsed_seconds") or 0.0) for r in records)
    return {"requests": n, "ex_correct": correct, "submission_seconds": gen, "submission_seconds_mean": gen / n,
            "generation_cost_per_ex_correct": gen / correct if correct else None,
            "submitted_execution_seconds": execution,
            "submitted_timeouts": sum((r["score"].get("predicted") or {}).get("status") == "timeout" for r in records),
            "with_execution_per_ex_correct": (gen + execution) / correct if correct else None}


def reconstruct(bundle: Path = evidence.PILOT_BUNDLE) -> dict:
    """The decision recomputed from the bundle's records, after comparing it with analysis.json and every summary.
    Raises `Inconsistent` listing each disagreement."""
    bundle = Path(bundle)
    evidence._checked(bundle)
    problems = []
    analyses = sorted((bundle / "analysis").glob("*/analysis.json"))
    if len(analyses) != 1:
        raise Inconsistent(f"{len(analyses)} analyses in the bundle; exactly one is expected")
    a = json.loads(analyses[0].read_text())

    # --- which evaluation each seed counts, from the reconstruction inventories ---------------------------------------
    runs, ids = [], None
    for d in sorted((bundle / "eval").glob("s*_*")):
        config = json.loads((d / "manifest.json").read_text())["config"]
        summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        failures = pilot_run.eval_failures(summary, config["ids"])
        if summary.get("passed") is True and failures:
            problems.append(f"{d.name}: its summary records passed, but {failures[0]}")
        runs.append({"run": identity.run_ref(f"pilot/eval/{d.name}"), "dir": d, "seed": config["seed"],
                     "ids": config["ids"], "passed": summary.get("passed") is True and not failures,
                     "summary": summary})
    counted = {}
    for seed in P.SEEDS:
        passing = [r for r in runs if r["seed"] == seed and r["passed"]]
        if len(passing) != 1:
            problems.append(f"seed {seed}: {len(passing)} eligible evaluations; exactly one is required")
        else:
            counted[seed] = passing[0]
    recorded_counted = {s: identity.run_ref(x) for s, x in a["inputs"]["evaluations_counted"].items()}
    if recorded_counted != {str(s): r["run"] for s, r in counted.items()}:
        problems.append(f"evaluations counted: analysis.json {recorded_counted}, the inventories give "
                        f"{ {str(s): r['run'] for s, r in counted.items()} }")
    recorded_all = sorted((identity.run_ref(r["run"]), r["seed"], r["passed"]) for r in a["inputs"]["evaluations_all"])
    if recorded_all != sorted((r["run"], r["seed"], r["passed"]) for r in runs):
        problems.append("evaluations_all in analysis.json differs from the bundled evaluation runs")
    if problems and len(counted) != len(P.SEEDS):
        raise Inconsistent("the bundle is inconsistent:\n- " + "\n- ".join(problems))
    ids = counted[P.SEEDS[0]]["ids"]
    if any(r["ids"] != ids for r in counted.values()):
        problems.append("the counted evaluations requested different questions")

    # --- the decision records ----------------------------------------------------------------------------------------
    correct, db_of, totals = {}, {}, {}

    def records_of(run_records: list, where: str, recorded_summary: "dict | None") -> dict:
        try:
            recomputed = R.summarize(run_records, ids)  # validates; exactly the requested questions, once each
        except R.RecordError as e:
            problems.append(f"{where}: {e}")
            return {}
        if recorded_summary is not None:
            verify._check_summary(recomputed, recorded_summary, problems, where)
        for r in run_records:
            if db_of.setdefault(r["task_id"], r["db_id"]) != r["db_id"]:
                problems.append(f"{where} {r['task_id']}: database {r['db_id']}, elsewhere {db_of[r['task_id']]}")
        totals[where] = {**recomputed, **serving(run_records)}
        return {r["task_id"]: r["score"]["correct"] for r in run_records}

    for seed, run in counted.items():
        for arm in P.ARMS:
            arm_summary = (run["summary"].get("arms", {}).get(arm) or {}).get("summary")
            correct[(arm, seed)] = records_of(_load(run["dir"] / arm / "records.jsonl"), f"{arm}/{seed}", arm_summary)
    l_dirs = sorted(p.parent for p in (bundle / "L").glob("*/records.jsonl"))
    if len(l_dirs) != 1:
        raise Inconsistent(f"{len(l_dirs)} L runs in the bundle; exactly one is expected")
    if identity.run_ref(a["inputs"]["L"]) != f"baselines/{l_dirs[0].name}":
        problems.append(f"L: analysis.json names {identity.run_ref(a['inputs']['L'])}, the bundle holds {l_dirs[0].name}")
    l_summary = json.loads((l_dirs[0] / "summary.json").read_text()).get("summary")
    l_correct = records_of(_load(l_dirs[0] / "records.jsonl"), "L", l_summary)
    if problems:
        raise Inconsistent("the bundle is inconsistent:\n- " + "\n- ".join(problems))

    # --- the frozen analysis, recomputed, against analysis.json --------------------------------------------------------
    result = P.analyze(correct, l_correct, db_of, ids)
    for key in COMPARED:
        if result[key] != a.get(key):
            problems.append(f"{key}: analysis.json {json.dumps(a.get(key))[:160]}, recomputed "
                            f"{json.dumps(result[key])[:160]}")
    rep = a["gold_representability"]  # the per-question trace needs the tokenizer: taken from analysis.json
    per_q = rep["per_question"]
    if sorted(per_q) != sorted(ids):
        problems.append("gold_representability.per_question does not cover the 64")
    else:
        unrep = [t for t in ids if per_q[t] != "representable"]
        if rep["representable"] != len(ids) - len(unrep) or rep["of"] != len(ids):
            problems.append("gold_representability: its count differs from its per-question list")
        on_unrep = {f"{x}/{s}": sum(correct[(x, s)][t] for t in unrep) for x in P.ARMS for s in P.SEEDS}
        if on_unrep != rep["correct_on_unrepresentable"]:
            problems.append(f"correct_on_unrepresentable: analysis.json {rep['correct_on_unrepresentable']}, "
                            f"recomputed {on_unrep}")
    if problems:
        raise Inconsistent("analysis.json disagrees with the decision records:\n- " + "\n- ".join(problems))
    return {"analysis": a, "recomputed": result, "ids": ids, "correct": correct, "l_correct": l_correct,
            "counted": counted, "runs": runs, "l_dir": l_dirs[0], "totals": totals}


def _f(x, digits: int = 1) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def tables(bundle: Path = evidence.PILOT_BUNDLE) -> str:
    bundle = Path(bundle)
    rc = reconstruct(bundle)
    a, res, ids = rc["analysis"], rc["recomputed"], rc["ids"]
    c, d = res["counts"], res["differences"]
    lines = ["Recomputed from the decision records and equal to analysis.json: counts, L statuses, differences, rule, "
             "database totals, histogram, sign-flip block statistic; every run's totals equal its summary.", "",
             "## Counts (64 questions, every request in the denominator)", "",
             "| | seed 100 | seed 101 | pooled (of 128) |", "|---|---|---|---|"]
    for arm in P.ARMS:
        lines.append(f"| {P.LABELS[arm]} | {c[f'{arm}/100']} | {c[f'{arm}/101']} | "
                     f"{c[f'{arm}/100'] + c[f'{arm}/101']} |")
    lines.append(f"| L (deterministic, no fit) | {c['L']} | {c['L']} | {2 * c['L']} |")
    lines += ["", f"- T4 − C4_params: seed 100 {d['per_seed']['100']:+d}, seed 101 {d['per_seed']['101']:+d}, "
                  f"pooled {d['pooled']:+d} of {d['pooled_out_of']}.",
              f"- Cleared L in both seeds: T4 {res['cleared_L_in_both_seeds']['T4']}, "
              f"C4_params {res['cleared_L_in_both_seeds']['c4_params']}.",
              f"- Registered rule outcome: **{res['rule']}** (lead threshold {res['lead_threshold']}).",
              f"- Per-question (T4 − C4_params, summed over seeds): {res['per_question_difference_histogram']}.", ""]
    lines += ["## Databases", "", "| database | questions | L | T4 s100 | T4 s101 | C4_params s100 | C4_params s101 | "
              "pooled difference |", "|---|---|---|---|---|---|---|---|"]
    for db, row in res["databases"].items():
        lines.append(f"| {db} | {row['questions']} | {row['L']} | {row['T4/100']} | {row['T4/101']} | "
                     f"{row['c4_params/100']} | {row['c4_params/101']} | {row['difference']:+d} |")
    sf = res["exploratory_sign_flip"]
    lines += ["", f"**Exploratory database-block sign-flip analysis:** observed pooled difference {sf['observed']:+d}; "
                  f"{sf['nonzero_blocks']} of {sf['blocks']} blocks nonzero; {sf['assignments']} sign assignments "
                  f"enumerated; two-sided p = {sf['p_two_sided']:.4f} (smallest attainable "
                  f"{sf['smallest_attainable_p']:.4f}). Exploratory: it conditions on the four fitted models.", ""]
    rep = a["gold_representability"]
    strata = defaultdict(list)
    for t in ids:
        strata["representable" if rep["per_question"][t] == "representable" else "not representable"].append(t)
    lines += [f"## Gold-program representability ({rep['representable']} of {rep['of']} representable)", "",
              f"Reasons, where not representable: {rep['reasons']}. Training exclusion rate: "
              f"{a['training_exclusion_rate']['excluded']} of {a['training_exclusion_rate']['pool']} "
              f"({a['training_exclusion_rate']['rate']:.1%}), a rate for training, not an evaluation ceiling. "
              f"(The per-question trace is analysis.json's; the strata are recomputed from the records.)", "",
              "| stratum | questions | L | T4 s100 | T4 s101 | C4_params s100 | C4_params s101 |", "|---|---|---|---|---|---|---|"]
    for name in ("representable", "not representable"):
        members = strata[name]
        cells = [str(sum(rc["l_correct"][t] for t in members))]
        for arm in P.ARMS:
            for seed in P.SEEDS:
                cells.append(str(sum(rc["correct"][(arm, seed)][t] for t in members)))
        lines.append(f"| {name} | {len(members)} | " + " | ".join(cells) + " |")
    lines += ["", "## Serving: time to SQL submission and generation cost per EX-correct submission", "",
              "Time to SQL submission runs from a request's start until its final SQL is submitted (for the "
              "controllers, the fresh backbone forward and the decode; for L, retrieval and its candidate queries). "
              "Generation cost per EX-correct submission is that total over the 64 divided by the EX-correct "
              "submissions: the frozen protocol's cost unit, unchanged. The scorer's recorded execution of each "
              "submitted query is not part of it; it is shown separately, with the sum of the two, as descriptive "
              "accounting.", "",
              "| seed | arm | EX-correct | answered | abstained | time to SQL submission, s (64 total / mean) | "
              "generation cost per EX-correct submission, s | recorded submitted-query execution, s (scorer "
              "timeouts) | generation + execution per EX-correct, s (descriptive) | reconstruction | fresh = anchor |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for seed, run in rc["counted"].items():
        for arm in P.ARMS:
            t, o = rc["totals"][f"{arm}/{seed}"], run["summary"]["arms"][arm]
            lines.append(f"| {seed} | {P.LABELS[arm]} | {t['ex_correct']} | {t['answered']} | {t['abstained']} | "
                         f"{t['submission_seconds']:.1f} / {t['submission_seconds_mean']:.2f} | "
                         f"{_f(t['generation_cost_per_ex_correct'])} | {t['submitted_execution_seconds']:.1f} "
                         f"({t['submitted_timeouts']}) | {_f(t['with_execution_per_ex_correct'])} | "
                         f"{o['reconstruction_all_equal']} | {o['fresh_states_all_equal_anchor']} |")
    t = rc["totals"]["L"]
    lines.append(f"| — | L | {t['ex_correct']} | {t['answered']} | {t['abstained']} | {t['submission_seconds']:.1f} / "
                 f"{t['submission_seconds_mean']:.2f} | {_f(t['generation_cost_per_ex_correct'])} | "
                 f"{t['submitted_execution_seconds']:.1f} ({t['submitted_timeouts']}) | "
                 f"{_f(t['with_execution_per_ex_correct'])} | — | — |")
    lines += ["", "## Every evaluation run (eligibility from its reconstruction inventory)", ""]
    for r in rc["runs"]:
        lines.append(f"- {r['run']}: seed {r['seed']}, passed {r['passed']}")
    lines += ["", "## GPU reservations (the whole study's ledger)", "",
              "| reservation | command | status | projected h | charged h |", "|---|---|---|---|---|"]
    from . import ledger as ledger_mod
    u = ledger_mod.usage(bundle / "gpu_ledger.jsonl")  # the ledger's own charging rule
    total = pilot_total = 0.0
    for r in u["reservations"]:
        hours = r["hours"]
        total += hours
        if str(r.get("command", "")).startswith("pilot-"):
            pilot_total += hours
        status = r["status"] if r["ended"] else "INTERRUPTED by the host crash (no end entry; charged to the last " \
                                                 "heartbeat plus one interval)"
        lines.append(f"| {r['reservation']} | {r['command']} | {status} | {r.get('projected_hours')} | {hours:.3f} |")
    lines.append(f"| total | | | | {total:.3f} of 8 (pilot: {pilot_total:.3f}) |")
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(tables(Path(argv[0]) if argv else evidence.PILOT_BUNDLE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
