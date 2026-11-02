"""The performance study's analysis, fixed by `docs/PROTOCOL_perf_v2.md` (pure functions over EX-correct records).

**Example.** With EX-correct counts on the new 64 of T4-v1 = 6, 7, 5 and T4-v2 = 10, 11, 9 for seeds 100-102, the
intervention's pooled difference for T4 is D = +12 of 192 question-seed pairs. It is positive in each seed, so I1
(material improvement) holds if D stays positive when any one database's questions are removed; otherwise the
gain is concentrated in one database and the outcome is I3.

The rules (the protocol's Q1 and Q2) use the same three conditions:
- a pooled difference of at least `THRESHOLD` (10 of 192 pairs, 5.2 points);
- the same sign in every seed;
- the same sign after removing any one database.
"""
from __future__ import annotations

from collections import Counter

from . import pilot as P

THRESHOLD = 10          # with three seeds (192 question-seed pairs)


def threshold(n_seeds: int) -> int:
    """10 of 192 pairs with three seeds; the same share, rounded up, if a symmetric reduction leaves fewer."""
    return -(-THRESHOLD * n_seeds // 3)


def counts(correct: dict, ids: list) -> dict:
    """{label/seed: EX-correct over ids} from correct {(label, seed): {task_id: 0/1}}."""
    return {f"{lab}/{s}": sum(v[t] for t in ids) for (lab, s), v in sorted(correct.items())}


def paired(correct: dict, a: str, b: str, seeds, ids: list, db_of: dict) -> dict:
    """The difference a - b in EX-correct counts: per seed, pooled, per database, leave-one-database-out, the per
    question histogram, the rule's three conditions and the exploratory sign-flip."""
    per_seed = {s: sum(correct[(a, s)][t] - correct[(b, s)][t] for t in ids) for s in seeds}
    pooled = sum(per_seed.values())
    by_db: dict = {}
    for t in ids:
        by_db[db_of[t]] = by_db.get(db_of[t], 0) + sum(correct[(a, s)][t] - correct[(b, s)][t] for s in seeds)
    loo = {db: pooled - v for db, v in sorted(by_db.items())}
    per_question = Counter(sum(correct[(a, s)][t] - correct[(b, s)][t] for s in seeds) for t in ids)
    sign = 1 if pooled > 0 else -1 if pooled < 0 else 0
    conditions = {"pooled_at_least_threshold": abs(pooled) >= threshold(len(list(seeds))),
                  "same_sign_every_seed": sign != 0 and all(d * sign > 0 for d in per_seed.values()),
                  "same_sign_without_any_one_database": sign != 0 and all(v * sign > 0 for v in loo.values())}
    return {"a": a, "b": b, "per_seed": {str(s): d for s, d in per_seed.items()}, "pooled": pooled,
            "threshold": threshold(len(per_seed)),
            "pooled_out_of": len(ids) * len(seeds), "sign": sign, "conditions": conditions,
            "all_conditions": all(conditions.values()), "per_database": dict(sorted(by_db.items())),
            "leave_one_database_out": loo,
            "per_question_histogram": {str(k): per_question[k] for k in sorted(per_question)},
            "exploratory_sign_flip": P.sign_flip(dict(sorted(by_db.items())))}


def rule_q1(d: dict) -> str:
    if d["all_conditions"] and d["sign"] > 0:
        return "I1: material improvement"
    if d["all_conditions"] and d["sign"] < 0:
        return "I2: material harm"
    return "I3: no material change"


def rule_q2(d: dict) -> str:
    if d["all_conditions"] and d["sign"] > 0:
        return "A1: T4 ahead"
    if d["all_conditions"] and d["sign"] < 0:
        return "A2: C4_params ahead"
    return "A3: no consistent difference"


def reading(q1: dict, q2: str) -> str:
    """The protocol's reading of the combined outcomes (Q1 for both architectures, Q2 for v2)."""
    t4, c4 = q1["T4"].split(":")[0], q1["C4"].split(":")[0]
    if q2.startswith("A1"):
        return ("T4-v2 has a consistent advantage over C4-v2 on the new population: propose a separately frozen "
                "confirmation on fresh data; nothing is confirmed here")
    if t4 == c4 == "I1" and q2.startswith("A3"):
        return ("both architectures improved materially and they do not differ consistently: an engineering gain of "
                "the intervention, not attributable to specialization")
    if t4 == c4 == "I1" and q2.startswith("A2"):
        return ("both architectures improved materially, and C4-v2 is consistently ahead of T4-v2: the gain is the "
                "intervention's, and the shared organization did better")
    if t4 == c4 == "I3":
        return "neither architecture improved materially: diagnose why and recommend a distinct next experiment"
    return f"mixed: T4 {t4}, C4 {c4}, architectures {q2.split(':')[0]}; read descriptively"


def serving(records: list) -> dict:
    """Time to SQL submission and the scorer's execution of the submitted query, separately and together."""
    gen = sum(r["elapsed_seconds"] for r in records)
    exe = sum(r["score"]["predicted"]["elapsed_seconds"] for r in records)
    ok = sum(r["score"]["correct"] for r in records)
    return {"requests": len(records), "ex_correct": ok, "time_to_submission_seconds": gen,
            "submitted_execution_seconds": exe, "together_seconds": gen + exe,
            "mean_time_to_submission": gen / len(records) if records else None,
            "timeouts": sum(r["score"]["predicted"]["status"] == "timeout" for r in records),
            "abstentions": sum(r["final"]["kind"] == "abstain" for r in records),
            "generation_per_ex_correct": gen / ok if ok else None,
            "together_per_ex_correct": (gen + exe) / ok if ok else None}


def strata(plan) -> dict:
    """Semantic strata of a gold program: tables, aggregation, grouping, nesting, ordering, literals."""
    from . import diagnose as D
    from . import ir
    if plan is None:
        return {"tables": "unconverted"}
    top = D.top_select(plan)
    f = D.select_features(top, D.Scope())
    n_tables = len(f["tables"])
    literals = sum(1 for n in ir.walk(plan) if isinstance(n, ir.Literal))
    return {"tables": "1" if n_tables <= 1 else "2" if n_tables == 2 else "3+",
            "aggregation": "yes" if f["aggregates"] else "no",
            "grouping": "yes" if f["group"] else "no",
            "nesting": "yes" if (f["nested"] or isinstance(plan, ir.Compound)) else "no",
            "ordering": "yes" if (f["order"] or f["limit"] is not None) else "no",
            "literals": "0" if literals == 0 else "1" if literals == 1 else "2+"}


def by_stratum(correct: dict, labels, seeds, ids: list, stratum_of: dict) -> dict:
    """EX-correct per stratum value, pooled over seeds, for each label: {dimension: {value: {questions, label: n}}}."""
    out: dict = {}
    for t in ids:
        for dim, value in stratum_of[t].items():
            row = out.setdefault(dim, {}).setdefault(value, {"questions": 0, **{lab: 0 for lab in labels}})
            row["questions"] += 1
            for lab in labels:
                row[lab] += sum(correct[(lab, s)][t] for s in seeds if (lab, s) in correct)
    return out
