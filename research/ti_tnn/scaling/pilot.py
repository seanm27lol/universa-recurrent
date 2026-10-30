"""The C4/T4 learning pilot: its fixed settings, the encoding anchors, and the analysis the frozen protocol defines.

**Example.** With both seeds evaluated on the 64 development questions:
- T4 answers 9 (seed 100) and 7 (seed 101) correctly;
- C4_params answers 5 and 4;
- L answers 3.

Both arms *cleared L in both seeds*. The per-seed differences are +4 and +3, and the pooled lead is +7, so rule A1
("T4 ahead in this pilot") holds. The exploratory database-block sign-flip analysis is reported beside it. It
treats each database's pooled difference (both seeds, all its questions) as one block and enumerates every sign
assignment. That analysis is not part of the rule.

**What the sign-flip analysis assumes:**
- the 14 databases are independent units;
- with no architecture effect, each block's pooled difference is symmetric about zero; swapping the arm labels for
  all of a block's questions and both seeds together leaves the joint distribution unchanged.

**What it leaves out.** It conditions on the four trained models. Training randomness is shared across every database
(two seeds only), so the analysis captures between-database variation, not training-seed variation, and it can
understate the uncertainty. It is exploratory; no false-positive rate is claimed.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

from . import contracts as C

ARMS = ("T4", "c4_params")
LABELS = {"T4": "T4", "c4_params": "C4_params"}
SEEDS = (100, 101)
UPDATES = 1500
CHECKPOINT_EVERY = 250
MATMUL_PRECISION = "high"
EVAL_SIZE = 64
LEAD = 7                     # pooled successes: 7 / 128 >= 5 percentage points
PROTOCOL = C.SCALING / "docs" / "PROTOCOL_pilot.md"
ROOT = C.RUNS / "pilot"


# --- the training set and the encoding anchors -------------------------------------------------------------------------
def train_ids() -> list:
    pin = json.loads((C.PINS / "pilot_train_v1.json").read_text())
    return list(pin["task_ids"])


def eval_ids() -> list:
    from . import data
    return data.inventory_ids(EVAL_SIZE)


def load_anchors(root: Path = None) -> tuple:
    """({task_id: anchor}, backbone identity) from every encoding run's entries.jsonl. Each anchor was written by the
    run that created (or verified) the cache entry: {task_id, cache_key, states_sha256, tokens, ...}. Two anchors for
    one task must agree, and every anchor must come from one backbone identity."""
    root = Path(root) if root is not None else ROOT / "encode"
    anchors, identities = {}, set()
    for entries in sorted(root.glob("*/entries.jsonl")):
        identities.add(json.dumps(json.loads((entries.parent / "backbone.json").read_text()), sort_keys=True))
        for line in entries.read_text().splitlines():
            if not line.strip():
                continue
            a = json.loads(line)
            old = anchors.get(a["task_id"])
            if old is not None and (old["cache_key"], old["states_sha256"]) != (a["cache_key"], a["states_sha256"]):
                raise ValueError(f"{a['task_id']}: two encoding runs anchored different entries")
            anchors[a["task_id"]] = a
    if len(identities) > 1:
        raise ValueError(f"the anchors come from {len(identities)} backbone identities")
    return anchors, (json.loads(identities.pop()) if identities else None)


def feasibility_sample(ids: list, lengths: dict, n: int = 160) -> list:
    """n training ids at evenly spaced quantiles of context length (ties by id), so that the sample's lengths
    represent the pool's."""
    order = sorted(ids, key=lambda t: (lengths[t], t))
    if n >= len(order):
        return order
    picks = [round(i * (len(order) - 1) / (n - 1)) for i in range(n)]
    return [order[i] for i in picks]


# --- the analysis (fixed by the frozen protocol) -----------------------------------------------------------------------
def sign_flip(blocks: dict) -> dict:
    """Exact two-sided sign-flip test of the pooled difference: blocks {database: pooled difference}. Every assignment
    of signs to the blocks is enumerated; p = share with |sum| >= |observed|."""
    values = list(blocks.values())
    observed = sum(values)
    nonzero = [v for v in values if v != 0]
    total = extreme = 0
    for signs in itertools.product((1, -1), repeat=len(nonzero)):
        s = sum(a * b for a, b in zip(signs, nonzero))
        total += 1
        extreme += abs(s) >= abs(observed)
    return {"observed": observed, "blocks": len(values), "nonzero_blocks": len(nonzero), "assignments": total,
            "p_two_sided": extreme / total if total else 1.0,
            "smallest_attainable_p": (2 / total) if len(nonzero) else 1.0}


def analyze(correct: dict, l_correct: dict, db_of: dict, ids: list) -> dict:
    """correct: {(arm, seed): {task_id: 0 or 1}} for every arm and seed; l_correct: {task_id: 0 or 1}. Returns the
    counts, the L gate, the paired differences, the rule outcome, the database table and the sign-flip analysis."""
    for key, rec in list(correct.items()) + [(("L", None), l_correct)]:
        if sorted(rec) != sorted(ids):
            raise ValueError(f"{key}: the records do not cover exactly the {len(ids)} evaluation questions")
    c = {f"{arm}/{seed}": sum(correct[(arm, seed)].values()) for arm in ARMS for seed in SEEDS}
    c_l = sum(l_correct.values())
    cleared = {arm: all(c[f"{arm}/{s}"] > c_l for s in SEEDS) for arm in ARMS}
    per_seed = {s: c[f"T4/{s}"] - c[f"c4_params/{s}"] for s in SEEDS}
    pooled = sum(per_seed.values())
    if pooled >= LEAD and all(d > 0 for d in per_seed.values()):
        rule = "A1: T4 ahead in this pilot"
    elif pooled <= -LEAD and all(d < 0 for d in per_seed.values()):
        rule = "A2: C4_params ahead in this pilot"
    else:
        rule = "A3: no pilot difference by the pre-set rule"
    by_db: dict = {}
    for t in ids:
        row = by_db.setdefault(db_of[t], {"questions": 0, "L": 0, **{f"{a}/{s}": 0 for a in ARMS for s in SEEDS}})
        row["questions"] += 1
        row["L"] += l_correct[t]
        for a in ARMS:
            for s in SEEDS:
                row[f"{a}/{s}"] += correct[(a, s)][t]
    for row in by_db.values():
        row["difference"] = sum(row[f"T4/{s}"] - row[f"c4_params/{s}"] for s in SEEDS)
    per_question = {t: sum(correct[("T4", s)][t] - correct[("c4_params", s)][t] for s in SEEDS) for t in ids}
    q_hist = Counter(per_question.values())
    return {"counts": {**c, "L": c_l, "questions": len(ids)},
            "cleared_L_in_both_seeds": cleared,
            "differences": {"per_seed": {str(s): d for s, d in per_seed.items()}, "pooled": pooled,
                            "pooled_out_of": len(ids) * len(SEEDS)},
            "rule": rule, "lead_threshold": LEAD,
            "per_question_difference_histogram": {str(k): q_hist[k] for k in sorted(q_hist)},
            "databases": dict(sorted(by_db.items())),
            "exploratory_sign_flip": sign_flip({db: row["difference"] for db, row in sorted(by_db.items())})}


def representability(ids: list, tokenizer) -> dict:
    """{task_id: "representable" | reason}: whether each question's gold program converts to tirel-1 and traces through
    the action grammar with its literals in the question/evidence tokens. Reported beside the results (evaluator-side;
    no arm reads it)."""
    from . import actions as A
    from . import data, encode, retrieval
    from . import schema as schema_mod
    from .train import gold_steps
    out = {}
    for tid in ids:
        task = data.task(tid)
        ctx, serialized, _ids, offsets = encode.fit_context(task, schema_mod.load_schema(task.db_id),
                                                            retrieval.retrieve(task), tokenizer,
                                                            C.Limits().max_context_tokens)

        class _Enc:
            schema = A.SchemaItems(serialized.items)
            spans = encode.span_index(ctx, offsets)

        steps, reason = gold_steps(data.gold(tid).sql, _Enc)
        out[tid] = "representable" if steps is not None else reason
    return out
