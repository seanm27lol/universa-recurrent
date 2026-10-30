"""The pilot's fixed analysis (scaling/pilot.py): the L gate, the paired differences, the pre-set rules A1-A3, the
database table, and the exploratory database-block sign-flip test, on hand-made records (no data, no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "ti_tnn"))
from scaling import pilot as P  # noqa: E402

IDS = [f"q{i}" for i in range(8)]
DB = {t: ("db_a" if i < 4 else "db_b") for i, t in enumerate(IDS)}


def _rec(ones):
    return {t: int(t in ones) for t in IDS}


def _correct(t4_100, t4_101, c4_100, c4_101):
    return {("T4", 100): _rec(t4_100), ("T4", 101): _rec(t4_101),
            ("c4_params", 100): _rec(c4_100), ("c4_params", 101): _rec(c4_101)}


def test_the_sign_flip_enumerates_every_assignment():
    out = P.sign_flip({"a": 2, "b": 0, "c": 1})
    assert (out["observed"], out["nonzero_blocks"], out["assignments"]) == (3, 2, 4)
    assert out["p_two_sided"] == 0.5 and out["smallest_attainable_p"] == 0.5
    assert P.sign_flip({"a": 0, "b": 0})["p_two_sided"] == 1.0


def test_a1_needs_the_lead_and_both_seeds(monkeypatch):
    monkeypatch.setattr(P, "LEAD", 3)
    out = P.analyze(_correct({"q0", "q1", "q4"}, {"q0", "q5", "q6"}, {"q0"}, {"q0"}), _rec({"q7"}), DB, IDS)
    assert out["differences"] == {"per_seed": {"100": 2, "101": 2}, "pooled": 4, "pooled_out_of": 16}
    assert out["rule"].startswith("A1") and out["cleared_L_in_both_seeds"] == {"T4": True, "c4_params": False}
    assert out["databases"]["db_a"]["difference"] == 1 and out["databases"]["db_b"]["difference"] == 3
    one_seed = P.analyze(_correct({"q0", "q1", "q4", "q5"}, {"q0"}, {"q0"}, {"q0"}), _rec(set()), DB, IDS)
    assert one_seed["differences"]["pooled"] == 3 and one_seed["rule"].startswith("A3")  # seed 101 is 0, not > 0


def test_a2_mirrors_a1_and_l_must_be_beaten_in_both_seeds(monkeypatch):
    monkeypatch.setattr(P, "LEAD", 2)
    out = P.analyze(_correct(set(), set(), {"q0"}, {"q1"}), _rec({"q2"}), DB, IDS)
    assert out["rule"].startswith("A2") and out["cleared_L_in_both_seeds"]["c4_params"] is False  # ties with L


def test_records_must_cover_exactly_the_questions():
    bad = _correct(set(), set(), set(), set())
    bad[("T4", 100)].pop("q0")
    with pytest.raises(ValueError, match="exactly"):
        P.analyze(bad, _rec(set()), DB, IDS)


def test_the_lead_keeps_five_points_of_128():
    assert P.LEAD == 7 and P.LEAD / (P.EVAL_SIZE * len(P.SEEDS)) >= 0.05


def test_the_feasibility_sample_spans_the_lengths():
    ids = [f"t{i}" for i in range(100)]
    lengths = {t: 1000 + 37 * ((i * 13) % 100) for i, t in enumerate(ids)}
    s = P.feasibility_sample(ids, lengths, 5)
    got = sorted(lengths[t] for t in s)
    assert got[0] == min(lengths.values()) and got[-1] == max(lengths.values()) and len(set(s)) == 5
