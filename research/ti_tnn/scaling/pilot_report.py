"""The pilot report's tables, rebuilt from the pilot evidence bundle alone (`evidence/pilot_v1`).

    python -m scaling.pilot_report [BUNDLE]

**The tables:**
- the counts on the full 64-question denominator;
- per-seed and pooled differences;
- each arm's "cleared L in both seeds" status and the registered rule outcome;
- per-database results;
- the exploratory block analysis;
- gold-program representability, with performance within its strata;
- the feasibility measurements;
- projected against actual GPU hours per stage, with every reservation in the ledger, failed, interrupted and
  superseded ones included.

It refuses to run if any bundled file differs from `SHA256SUMS`.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from . import evidence
from . import pilot as P


def _load(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _counted_evals(bundle: Path, analysis: dict) -> dict:
    out = {}
    for seed, run in analysis["inputs"]["evaluations_counted"].items():
        out[int(seed)] = bundle / "eval" / Path(run).name
    return out


def tables(bundle: Path = evidence.PILOT_BUNDLE) -> str:
    evidence._checked(bundle)
    lines = []
    analyses = sorted((bundle / "analysis").glob("*/analysis.json"))
    if len(analyses) != 1:
        raise SystemExit(f"{len(analyses)} analyses in the bundle; exactly one is expected")
    a = json.loads(analyses[0].read_text())
    c, d = a["counts"], a["differences"]
    lines += ["## Counts (64 questions, every request in the denominator)", "",
              "| | seed 100 | seed 101 | pooled (of 128) |", "|---|---|---|---|"]
    for arm in P.ARMS:
        lines.append(f"| {P.LABELS[arm]} | {c[f'{arm}/100']} | {c[f'{arm}/101']} | "
                     f"{c[f'{arm}/100'] + c[f'{arm}/101']} |")
    lines.append(f"| L (deterministic, no fit) | {c['L']} | {c['L']} | {2 * c['L']} |")
    lines += ["", f"- T4 − C4_params: seed 100 {d['per_seed']['100']:+d}, seed 101 {d['per_seed']['101']:+d}, "
                  f"pooled {d['pooled']:+d} of {d['pooled_out_of']}.",
              f"- Cleared L in both seeds: T4 {a['cleared_L_in_both_seeds']['T4']}, "
              f"C4_params {a['cleared_L_in_both_seeds']['c4_params']}.",
              f"- Registered rule outcome: **{a['rule']}** (lead threshold {a['lead_threshold']}).",
              f"- Per-question (T4 − C4_params, summed over seeds): {a['per_question_difference_histogram']}.", ""]
    lines += ["## Databases", "", "| database | questions | L | T4 s100 | T4 s101 | C4_params s100 | C4_params s101 | "
              "pooled difference |", "|---|---|---|---|---|---|---|---|"]
    for db, row in a["databases"].items():
        lines.append(f"| {db} | {row['questions']} | {row['L']} | {row['T4/100']} | {row['T4/101']} | "
                     f"{row['c4_params/100']} | {row['c4_params/101']} | {row['difference']:+d} |")
    sf = a["exploratory_sign_flip"]
    lines += ["", f"**Exploratory database-block sign-flip analysis:** observed pooled difference {sf['observed']:+d}; "
                  f"{sf['nonzero_blocks']} of {sf['blocks']} blocks nonzero; {sf['assignments']} sign assignments "
                  f"enumerated; two-sided p = {sf['p_two_sided']:.4f} (smallest attainable "
                  f"{sf['smallest_attainable_p']:.4f}). Exploratory: it conditions on the four fitted models.", ""]
    rep = a["gold_representability"]
    per_q = rep["per_question"]
    evals = _counted_evals(bundle, a)
    correct = {}
    for seed, run in evals.items():
        for arm in P.ARMS:
            for r in _load(run / arm / "records.jsonl"):
                correct[(arm, seed, r["task_id"])] = r["score"]["correct"]
    l_runs = sorted((bundle / "L").glob("*/records.jsonl"))
    l_correct = {r["task_id"]: r["score"]["correct"] for r in _load(l_runs[-1])} if l_runs else {}
    strata = defaultdict(list)
    for t, v in per_q.items():
        strata["representable" if v == "representable" else "not representable"].append(t)
    lines += [f"## Gold-program representability ({rep['representable']} of {rep['of']} representable)", "",
              f"Reasons, where not representable: {rep['reasons']}. Training exclusion rate: "
              f"{a['training_exclusion_rate']['excluded']} of {a['training_exclusion_rate']['pool']} "
              f"({a['training_exclusion_rate']['rate']:.1%}), a rate for training, not an evaluation ceiling.", "",
              "| stratum | questions | L | T4 s100 | T4 s101 | C4_params s100 | C4_params s101 |", "|---|---|---|---|---|---|---|"]
    for name in ("representable", "not representable"):
        ids = strata[name]
        cells = [str(sum(l_correct.get(t, 0) for t in ids))]
        for arm in P.ARMS:
            for seed in P.SEEDS:
                cells.append(str(sum(correct.get((arm, seed, t), 0) for t in ids)))
        lines.append(f"| {name} | {len(ids)} | " + " | ".join(cells) + " |")
    lines += ["", "## Cost and integrity per evaluation", "",
              "| seed | arm | correct | answered | abstained | elapsed s (64) | s per correct | reconstruction | "
              "fresh = anchor | passed |", "|---|---|---|---|---|---|---|---|---|---|"]
    for seed, run in evals.items():
        s = json.loads((run / "summary.json").read_text())
        for arm in P.ARMS:
            o = s["arms"][arm]
            sm = o["summary"]
            cpc = sm["cost_seconds_per_correct"]
            lines.append(f"| {seed} | {P.LABELS[arm]} | {sm['correct']} | {sm['answered']} | {sm['abstained']} | "
                         f"{sm['elapsed_seconds_total']:.1f} | {'—' if cpc is None else f'{cpc:.1f}'} | "
                         f"{o['reconstruction_all_equal']} | {o['fresh_states_all_equal_anchor']} | {s['passed']} |")
    lines += ["", "## Every evaluation run", ""]
    for r in a["inputs"]["evaluations_all"]:
        lines.append(f"- {Path(r['run']).name}: seed {r['seed']}, passed {r['passed']}")
    lines += ["", "## GPU reservations (the whole study's ledger)", "",
              "| reservation | command | status | projected h | charged h |", "|---|---|---|---|---|"]
    ledger = _load(bundle / "gpu_ledger.jsonl")
    starts = {e["reservation"]: e for e in ledger if e.get("event") == "start"}
    ends = {e["reservation"]: e for e in ledger if e.get("event") == "end"}
    beats = defaultdict(float)
    for e in ledger:
        if e.get("event") == "heartbeat":
            beats[e["reservation"]] = max(beats[e["reservation"]], e.get("elapsed", 0.0))
    total = pilot_total = 0.0
    for rid, st in starts.items():
        en = ends.get(rid)
        hours = (en["elapsed"] if en else beats[rid]) / 3600
        total += hours
        if str(st.get("command", "")).startswith("pilot-"):
            pilot_total += hours
        status = en.get("status") if en else "INTERRUPTED (no end entry; charged to the last heartbeat)"
        lines.append(f"| {rid} | {st.get('command')} | {status} | {st.get('projected_hours')} | {hours:.3f} |")
    lines.append(f"| total | | | | {total:.3f} of 8 (pilot: {pilot_total:.3f}) |")
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(tables(Path(argv[0]) if argv else evidence.PILOT_BUNDLE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
