"""The performance study's decision rules (`perf_analysis`), on synthetic EX-correct tables.

- I1/A1 need all three conditions: a pooled lead of at least 10 of 192 pairs, the same sign in every seed, and the
  same sign after removing any one database. Missing any one gives I3/A3, and the mirror image gives I2/A2.
- A lead concentrated in one database fails the leave-one-database-out condition.
- With fewer seeds (a symmetric reduction) the threshold keeps its share, rounded up.
- The readings: A1 proposes a confirmation; I1 for both with A3 is an engineering gain; I3 for both asks for a new
  experiment; I1 for both with A2 names the shared organization as ahead.
- Serving totals keep generation and the submitted query's execution apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from scaling import perf_analysis as A  # noqa: E402

SEEDS = (100, 101, 102)
DBS = [f"db{k}" for k in range(8)]
IDS = [f"q{i:02d}" for i in range(64)]
DB_OF = {t: DBS[i % 8] for i, t in enumerate(IDS)}


def _table(wins: dict) -> dict:
    """correct[(label, seed)][task]: 'a' wins on the listed (seed, task) pairs, 'b' never."""
    out = {}
    for s in SEEDS:
        out[("a", s)] = {t: int((s, t) in wins) for t in IDS}
        out[("b", s)] = {t: 0 for t in IDS}
    return out


def _spread(n_per_seed: int) -> set:
    """n wins per seed, spread across the databases."""
    return {(s, IDS[i]) for s in SEEDS for i in range(n_per_seed)}


def test_a_spread_lead_in_every_seed_is_i1_and_a1():
    d = A.paired(_table(_spread(4)), "a", "b", SEEDS, IDS, DB_OF)
    assert d["pooled"] == 12 and d["all_conditions"]
    assert A.rule_q1(d).startswith("I1") and A.rule_q2(d).startswith("A1")
    mirror = A.paired(_table(_spread(4)), "b", "a", SEEDS, IDS, DB_OF)
    assert A.rule_q1(mirror).startswith("I2") and A.rule_q2(mirror).startswith("A2")


def test_each_condition_is_needed():
    small = A.paired(_table(_spread(3)), "a", "b", SEEDS, IDS, DB_OF)        # 9 < 10
    assert not small["conditions"]["pooled_at_least_threshold"] and A.rule_q1(small).startswith("I3")
    one_seed = {(100, IDS[i]) for i in range(12)}                            # 12, all in seed 100
    d = A.paired(_table(one_seed), "a", "b", SEEDS, IDS, DB_OF)
    assert d["pooled"] == 12 and not d["conditions"]["same_sign_every_seed"] and A.rule_q2(d).startswith("A3")
    one_db = {(s, t) for s in SEEDS for t in IDS if DB_OF[t] == "db0"}       # 8 per seed, one database
    d = A.paired(_table(one_db), "a", "b", SEEDS, IDS, DB_OF)
    assert d["pooled"] == 24 and d["conditions"]["same_sign_every_seed"]
    assert not d["conditions"]["same_sign_without_any_one_database"] and A.rule_q1(d).startswith("I3")
    assert d["leave_one_database_out"]["db0"] == 0


def test_the_threshold_keeps_its_share_with_fewer_seeds():
    assert [A.threshold(n) for n in (1, 2, 3)] == [4, 7, 10]
    two = (100, 101)
    wins = {(s, IDS[i]) for s in two for i in range(4)}
    d = A.paired(_table(wins), "a", "b", two, IDS, DB_OF)
    assert d["threshold"] == 7 and d["pooled"] == 8 and d["all_conditions"]


def test_readings():
    assert "confirmation" in A.reading({"T4": "I3: x", "C4": "I3: x"}, "A1: T4 ahead")
    assert "engineering gain" in A.reading({"T4": "I1: x", "C4": "I1: x"}, "A3: none")
    assert "shared organization" in A.reading({"T4": "I1: x", "C4": "I1: x"}, "A2: C4_params ahead")
    assert "distinct next experiment" in A.reading({"T4": "I3: x", "C4": "I3: x"}, "A3: none")
    assert A.reading({"T4": "I1: x", "C4": "I3: x"}, "A3: none").startswith("mixed")


def test_serving_keeps_generation_and_execution_apart():
    recs = [{"elapsed_seconds": 1.0, "final": {"kind": "sql"},
             "score": {"correct": c, "predicted": {"status": st, "elapsed_seconds": e}}}
            for c, st, e in ((1, "ok", 0.5), (0, "timeout", 30.0), (1, "ok", 0.5))]
    v = A.serving(recs)
    assert v["time_to_submission_seconds"] == 3.0 and v["submitted_execution_seconds"] == 31.0
    assert v["generation_per_ex_correct"] == 1.5 and v["together_per_ex_correct"] == 17.0 and v["timeouts"] == 1
