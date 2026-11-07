"""The semantic-proposal-repair study's registered decision rule, applied consistently (correction of 2026-11-07).

**What it corrects.** The frozen analysis (`spr_run analyze`) took its per-comparison `threshold`, `conditions` and
`all_conditions` from the shared paired-comparison helper (`perf_analysis.paired`). That helper scales the
performance study's +10-of-192 rule to two seeds, giving 7 of 128. Its final `decision` string correctly used the
study's own rule, +8 (`spr_run.LEAD`), so the exported fields disagreed with the decision.

This module applies the registered rule, +8 of 128, the same sign in both seeds, and the same sign leaving any one
database out, to every one of those fields. It does not change `perf_analysis.paired`, the performance study's
defaults, the frozen `spr_run.py` or the original `analysis.json`.

**Example.** A pooled TI1 - D1 of +7, positive in both seeds and in every leave-one-database-out sum, meets two of the
three conditions, not the first, so it is not a lead. At +8 it is.

    python -m scaling.spr_rule        # the versioned corrected analysis of the frozen run (CPU; reads only records)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

from . import perf_analysis as PA

LEAD = 8                      # the registered research-priority rule (spr_run.LEAD, the frozen protocol)
RULE_VERSION = "spr-rule-v2"  # v1: the frozen analysis's mixed fields


def paired_rule(correct: dict, a: str, b: str, seeds, ids: list, db_of: dict, lead: int = LEAD) -> dict:
    """The difference a - b with the registered rule in every field. The per-seed, per-database,
    leave-one-database-out, histogram and sign-flip values are the shared helper's; the rule fields are this
    study's."""
    d = PA.paired(correct, a, b, seeds, ids, db_of)
    pooled = d["pooled"]
    sign = 1 if pooled > 0 else -1 if pooled < 0 else 0
    conditions = {"pooled_at_least_lead": abs(pooled) >= lead,
                  "same_sign_every_seed": sign != 0 and all(v * sign > 0 for v in d["per_seed"].values()),
                  "same_sign_without_any_one_database": sign != 0 and all(
                      v * sign > 0 for v in d["leave_one_database_out"].values())}
    return {**d, "threshold": lead, "sign": sign, "conditions": conditions,
            "all_conditions": all(conditions.values()), "rule_version": RULE_VERSION,
            "recoveries": sum(correct[(a, s)][t] > correct[(b, s)][t] for s in seeds for t in ids),
            "regressions": sum(correct[(a, s)][t] < correct[(b, s)][t] for s in seeds for t in ids)}


def lead(d: dict) -> bool:
    return d["all_conditions"] and d["sign"] > 0


def harm(d: dict) -> bool:
    return d["all_conditions"] and d["sign"] < 0


def decision(cmp: dict) -> str:
    """The protocol's four branches, in its order (TI1 - D1 primary)."""
    if lead(cmp["TI1-D1"]):
        return "TI1 beats D1: structured feedback has earned further investigation"
    if harm(cmp["TI1-D1"]) or harm(cmp["TI1-D0"]):
        return "TI1 harms correctness: inspect its rejected or revised correct proposals"
    if lead(cmp["D1-D0"]) and lead(cmp["TI1-D0"]):
        return "TI1 ties D1 but both beat D0: ordinary revision explains the improvement"
    return "neither revision helps by the rule: retain direct Qwen and stop this repair direction"


COMPARISONS = (("TI1", "D1"), ("D1", "D0"), ("TI1", "D0"))


def comparisons(correct: dict, seeds, ids: list, db_of: dict) -> dict:
    return {f"{a}-{b}": paired_rule(correct, a, b, seeds, ids, db_of) for a, b in COMPARISONS}


def corrected_cmd(args) -> None:
    """Write analysis/<stamp>_corrected_v2/analysis_v2.json: the original analysis's content with the rule fields
    recomputed by this module from the same scores, and a check that every count and the decision are unchanged."""
    from . import identity, pilot_run, spr_run
    analyses = sorted(d for d in (spr_run.ROOT / "analysis").glob("*") if (d / "analysis.json").exists())
    original_dir = analyses[0]
    original = json.loads((original_dir / "analysis.json").read_text())
    run = spr_run.C.RUNS.joinpath(original["run"])
    scores_path = run / "scores.jsonl"
    scores = [json.loads(x) for x in scores_path.read_text().splitlines() if x.strip()]
    pin = json.loads(spr_run.NEW_PIN.read_text())
    ids, db_of = pin["task_ids"], dict(zip(pin["task_ids"], pin["db_ids"]))
    seeds = original["seeds"]
    correct = {}
    for x in scores:
        correct.setdefault((x["policy"], x["seed"]), {})[x["task_id"]] = x["score"]["correct"]
    cmp = comparisons(correct, seeds, ids, db_of)
    out = {**original, "comparisons": cmp, "decision": decision(cmp), "rule_version": RULE_VERSION,
           "corrects": {"analysis": identity.run_ref(original_dir),
                        "analysis_sha256": identity.sha256_file(original_dir / "analysis.json"),
                        "what": "threshold, conditions and all_conditions from the registered +8 rule"},
           "inputs": {"scores": identity.run_ref(run) + "/scores.jsonl",
                      "scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest()}}
    for k in ("TI1-D1", "D1-D0", "TI1-D0"):
        for f in ("pooled", "per_seed", "per_database", "leave_one_database_out", "recoveries", "regressions"):
            if out["comparisons"][k][f] != original["comparisons"][k][f]:
                raise SystemExit(f"{k} {f} differs from the original analysis")
    if out["decision"] != original["decision"]:
        raise SystemExit("the corrected decision differs from the original")
    run_dir = pilot_run._new_dir(spr_run.ROOT / "analysis", "corrected_v2")
    identity.write_manifest(run_dir, config={"command": "spr_rule corrected", "rule_version": RULE_VERSION,
                                             **out["corrects"], **out["inputs"]}, require_clean=not args.allow_dirty)
    (run_dir / "analysis_v2.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: {"pooled": v["pooled"], "threshold": v["threshold"], "conditions": v["conditions"],
                          "all_conditions": v["all_conditions"]} for k, v in cmp.items()} | {"decision": out["decision"]},
                     indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--allow-dirty", action="store_true")
    corrected_cmd(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
