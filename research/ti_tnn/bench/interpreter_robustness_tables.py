"""Every table the interpreter robustness pilot reports, rebuilt from one run file (protocols/interpreter_robustness_v1.md).

    .venv-phase2/bin/python research/ti_tnn/bench/interpreter_robustness_tables.py runs/titnn_robustness_pilot/<run>/interpreter_robustness.json

Reported per unit (world, init) and per arm:
- the certification sequence (threshold, accepted, errors, p-value, outcome) and its stopping reason;
- every category under every policy (coverage, accepted errors with the 95% bound, abstention);
- S4a, S4b and S4_L3 by stratum;
- the categories at the fixed diagnostic cutoff, recomputed from full-precision scores. This is never a
  certificate.
- certification power, for the accepted counts observed, at assumed true error rates.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrated import calibration as K  # noqa: E402

PRIMARY = ("ID", "R", "T1", "S2", "Q_base", "S3", "S4a", "S4a:answer_ambiguous", "S4a:program_ambiguous_one_answer",
           "S4a:type_determined", "S4b", "S4b:answer_ambiguous", "S4b:program_ambiguous_one_answer",
           "S4b:type_determined", "S1:length3", "S1:round_trip")
ABSTAIN = ("R", "S4a", "S4b", "S4_L3")


def cell(cat, m):
    if m is None:
        return "-"
    if cat.split(":")[0] in ABSTAIN:
        return f"abst {m['abstention']:.3f} ({m['n'] - m['answered']}/{m['n']})"
    return f"cov {m['coverage']:.3f} err {m['errors']}/{m['answered']} ub {m['error_upper_95']:.4f}"


def at_cutoff(records):
    agg = defaultdict(lambda: {"n": 0, "answered": 0, "errors": 0})
    for r in records:
        keys = [r["category"]] + ([f"{r['category']}:{r['stratum']}"] if r.get("stratum") else []) + \
               ([f"S1:{r['family']}"] if r["category"] == "S1" else [])
        for k in keys:
            a = agg[k]
            a["n"] += 1
            if r["accepted_at_diagnostic_cutoff"]:
                a["answered"] += 1
                a["errors"] += not r["correct"]
    out = {}
    for k, a in agg.items():
        out[k] = {**a, "coverage": a["answered"] / a["n"], "abstention": 1 - a["answered"] / a["n"],
                  "error_upper_95": K.clopper_pearson_upper(a["errors"], a["answered"])}
    return out


def main(path):
    r = json.loads(Path(path).read_text())
    print(f"grid {r['grid']}; diagnostic cutoff {r.get('diagnostic_cutoff')}")
    for w, info in r["worlds"].items():
        print(f"world {w}: collisions {info['collisions']}")
    for u in r["units"]:
        print(f"\n=== world {u['world']} init {u['init']}")
        for arm, a in u["arms"].items():
            c = a["policies"]["certified"]["certificate"]
            steps = "; ".join(f"{s['threshold']}: {s['accepted']} acc, {s['errors']} err, p={s['p_value']:.3g}"
                              f"{'' if s['rejected'] else ' (stop)'}" for s in c["steps"])
            print(f"  {arm:8s} certificate {c['threshold']} [{c['stop_reason']}] {steps}")
            for s in c["steps"]:
                if s["accepted"]:
                    print(f"           power if true error 0.1/0.25/0.5% at n={s['accepted']}: "
                          + "/".join(f"{K.certification_power(e, s['accepted']):.2f}" for e in (0.001, 0.0025, 0.005)))
                    break
            for policy in ("none", "certified", "novelty"):
                cats = a["policies"][policy]["categories"]
                print(f"    {policy:9s} " + " | ".join(f"{k} {cell(k, cats.get(k))}" for k in PRIMARY))
            d = at_cutoff(a["records"])
            print(f"    @{r.get('diagnostic_cutoff')}   " + " | ".join(f"{k} {cell(k, d.get(k))}" for k in PRIMARY))
        g = u["grammar"]["categories"]
        print(f"  grammar   " + " | ".join(f"{k} {cell(k, g.get(k))}" for k in PRIMARY))


if __name__ == "__main__":
    main(sys.argv[1])
