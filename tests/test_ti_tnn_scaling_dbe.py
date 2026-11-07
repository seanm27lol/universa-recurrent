"""The database-evidence revision study's harness, with a fake model (no GPU, no BIRD data).

- **Lineage:** D0, D1 and V1 share the initial proposal. D1 and V1 share the revision seed, the conversation prefix
  and the remaining time. V1's message is D1's with the evidence inserted, and nothing else.
- **A failed lookup** is recorded, says so in V1's message, is not retried, and V1 is still revised.
- **The rule:** +7 of 128 is not a lead and +8 is (spr_rule, not the old +7 helper). The decision's wording does
  not call a miss proof of no effect.
- **Uncertainty:** the bootstrap is deterministic for its pinned seed and brackets the observed difference.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
pytest.importorskip("sqlglot")
from scaling import dbe_run as D  # noqa: E402
from scaling import spr_run as S  # noqa: E402


class _Rt:
    """Answers the initial call with one query and each revision with another; records every call."""

    def __init__(self):
        self.calls = []

    def generate_turn(self, messages, tools, cfg, seed, deadline=None):
        self.calls.append({"messages": [dict(m) for m in messages], "seed": seed, "deadline": deadline})
        sql = "SELECT 1" if len(messages) <= 3 else "SELECT 2"
        return SimpleNamespace(content=f"```sql\n{sql}\n```", prompt_tokens=10, generated_tokens=5,
                               finish_reason="eos")


@pytest.fixture
def served(monkeypatch):
    def run(lookup_result):
        rt = _Rt()
        monkeypatch.setattr(S, "_call", lambda rt_, msgs, cfg, seed, deadline: (
            rt_.generate_turn(msgs, None, cfg, seed, deadline),
            {"seconds": 0.1, "prompt_tokens": 10, "generated_tokens": 5, "finish_reason": "eos",
             "peak_memory_bytes": 1, "content": ""}))
        monkeypatch.setattr(S, "_execute", lambda snap, sql: {"status": "ok", "error": None, "rows": [[1]],
                                                              "columns": ["x"], "row_count": 1, "truncated": False,
                                                              "seconds": 0.01})
        monkeypatch.setattr(D, "lookup", lambda task, schema, snap, sql: lookup_result)
        import scaling.gi_llm as G
        monkeypatch.setattr(G, "messages", lambda system, task, text, examples: [
            {"role": "system", "content": "sys"}, {"role": "user", "content": "question"}])
        task = SimpleNamespace(task_id="t1", db_id="db", question="q", evidence="e")
        return D.serve(rt, None, task, None, None, "schema", [], 200), rt
    return run


OK = {"ok": True, "package": {"rendered": "Value 'a' in t.c: 0 rows store it verbatim"}, "seconds": 0.02,
      "error": None}


def test_the_policies_share_the_initial_proposal_and_differ_only_by_the_evidence(served):
    row, rt = served(OK)
    assert row["D0"] == {"source": "original", "sql": "SELECT 1"}
    d1, v1 = row["D1"]["revised"]["feedback"], row["V1"]["revised"]["feedback"]
    assert v1 == d1.replace(f"\n\n{S.REVISE}", f"\n\n{D.EVIDENCE_HEADER}\n{OK['package']['rendered']}\n\n{S.REVISE}")
    revisions = rt.calls[1:]
    assert len(revisions) == 2 and revisions[0]["seed"] == revisions[1]["seed"]
    assert revisions[0]["messages"][:-1] == revisions[1]["messages"][:-1]       # same prefix, initial reply included
    assert row["revision_order"] in (["D1", "V1"], ["V1", "D1"])
    assert row["D1"]["source"] == row["V1"]["source"] == "revised"


def test_a_failed_lookup_is_recorded_announced_and_not_retried(served):
    failed = {"ok": False, "package": None, "seconds": 0.5, "error": "RuntimeError: boom"}
    row, rt = served(failed)
    assert row["evidence"] == failed and D.FAILED in row["V1"]["revised"]["feedback"]
    assert len(rt.calls) == 3 and row["V1"]["source"] == "revised"


def test_an_empty_package_says_nothing_was_retrieved():
    assert D.NOTHING in D.v1_message("obs", "", False)
    assert D.v1_message("obs", "x", False).endswith(S.REVISE)


def test_the_lookup_records_an_exception(monkeypatch):
    import scaling.db_evidence as E
    monkeypatch.setattr(E, "retrieve", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad")))
    out = D.lookup(SimpleNamespace(question="q", evidence="e"), None, SimpleNamespace(path="x"), "SELECT 1")
    assert not out["ok"] and out["error"].startswith("ValueError") and out["seconds"] >= 0


SEEDS = (200, 201)
QIDS = [f"q{i:02d}" for i in range(64)]
DB_OF = {t: f"db{i % 8}" for i, t in enumerate(QIDS)}


def _correct(v1_wins: dict) -> dict:
    """V1 right on the first v1_wins[s] questions of seed s; D0 and D1 never right."""
    out = {}
    for s in SEEDS:
        out[("D0", s)] = {t: 0 for t in QIDS}
        out[("D1", s)] = {t: 0 for t in QIDS}
        out[("V1", s)] = {t: int(i < v1_wins[s]) for i, t in enumerate(QIDS)}
    return out


@pytest.mark.parametrize("wins, lead", [({200: 4, 201: 3}, False), ({200: 4, 201: 4}, True)])
def test_the_decision_uses_the_registered_plus_8_rule(wins, lead):
    from scaling import spr_rule as RULE
    c = _correct(wins)
    cmp = {f"{a}-{b}": RULE.paired_rule(c, a, b, SEEDS, QIDS, DB_OF) for a, b in D.COMPARISONS}
    assert cmp["V1-D1"]["threshold"] == 8 and cmp["V1-D1"]["pooled"] == sum(wins.values())
    text = D.decision(cmp)
    assert text.startswith("V1 meets") if lead else text.startswith("V1 did not meet the prespecified threshold")
    assert "no effect" not in text and "small" not in text


def test_the_bootstrap_is_pinned_shared_across_comparisons_and_brackets_the_observation():
    c = _correct({200: 5, 201: 5})
    a = D.bootstrap_all(c, D.COMPARISONS, SEEDS, QIDS, resamples=2000)
    assert a == D.bootstrap_all(c, D.COMPARISONS, SEEDS, QIDS, resamples=2000)
    v = a["V1-D1"]
    assert v["observed"] == 10 and v["low"] <= 10 <= v["high"] and v["seed"] == D.BOOTSTRAP_SEED
    assert a["V1-D0"] == {**v, "observed": 10}            # D0 and D1 are equal here: the same draws give the same interval
    assert a["D1-D0"]["low"] == a["D1-D0"]["high"] == 0


@pytest.mark.parametrize("root, rel, allowed", [
    ("pins", "inventory_v1.json", True), ("ti_tnn_scaling", "data/gold_check_dev_v1.json", True),
    ("ti_tnn_scaling", "ir/coverage_details_v1.json", True), ("ti_tnn_scaling", "data/raw/bird_train/x.json", True),
    ("ti_tnn_scaling", "spr/run/x/requests.jsonl", False), ("evidence", "spr_v1/run/x/scores.jsonl", False),
    ("docs", "AUDIT_x.md", False), ("coverage", "requirements_v1.json", False), ("scaling", "verify.py", False)])
def test_prior_uses_are_classified(root, rel, allowed):
    assert D.classify_use(root, rel).startswith("allowed") is allowed


def _records(v1_right: set):
    """One request per question and seed, D0 and D1 wrong everywhere, V1 right on v1_right; one failed lookup."""
    reqs, scores = [], []
    call = {"seconds": 1.0, "prompt_tokens": 10, "generated_tokens": 5, "finish_reason": "eos",
            "peak_memory_bytes": 7}
    ex = {"status": "ok", "seconds": 0.01}
    pkg = {"items": [{"kind": "value", "verbatim_rows": 0, "candidates": [{"value": "x"}]}], "rendered_items": [0],
           "timeouts": [], "scan_limits": [], "truncated": False, "proposal": {"unresolved": []},
           "payload_chars": 50, "wall_safety_stop": False}
    for s in SEEDS:
        for i, t in enumerate(QIDS):
            ok = i != 0
            reqs.append({"seed": s, "task_id": t, "original": {"sql": "SELECT 1", "call": call, "execution": ex},
                         "evidence": {"ok": ok, "package": pkg if ok else None, "seconds": 0.1,
                                      "error": None if ok else "E"},
                         **{p: {"source": "revised", "sql": "SELECT 2",
                                "revised": {"sql": "SELECT 2", "call": call, "execution": ex}} for p in ("D1", "V1")}})
            for p in D.POLICIES:
                right = int(p == "V1" and (s, t) in v1_right)
                scores.append({"seed": s, "task_id": t, "policy": p, "score": {
                    "correct": right, "predicted": {"status": "ok", "elapsed_seconds": 0.01}}})
    return reqs, scores


def test_the_analysis_keeps_every_request_in_the_denominator_and_charges_the_lookup_to_v1():
    right = {(s, QIDS[i]) for s in SEEDS for i in range(1, 5)}
    reqs, scores = _records(right)
    out = D.analyze(reqs, scores, QIDS, DB_OF, SEEDS)
    assert out["requests_per_policy"] == 128 and out["pooled"] == {"D0": 0, "D1": 0, "V1": 8}
    assert out["comparisons"]["V1-D1"]["all_conditions"] is True       # +8 over four databases, 4 in each seed
    assert out["decision"].startswith("V1 meets the continuation rule")
    assert out["lookup"]["failed"] == 2 and out["signal_groups"]["lookup failed"]["requests"] == 2
    assert out["cost"]["V1"]["lookup_seconds"] == pytest.approx(12.8) and out["cost"]["D1"]["lookup_seconds"] == 0
    assert out["transitions"]["V1_fixes_D1_misses"] == 8 and out["uncertainty"]["V1-D1"]["observed"] == 8
    with pytest.raises(D.DbeStop):
        D.analyze(reqs[:-1], scores, QIDS, DB_OF, SEEDS)            # a missing request is refused, not dropped


def test_a_gain_in_one_database_does_not_meet_the_rule():
    one_db = {(s, t) for s in SEEDS for t in QIDS if DB_OF[t] == "db1"}       # 8 questions x 2 seeds, all in db1
    out = D.analyze(*_records(one_db), QIDS, DB_OF, SEEDS)
    assert out["pooled"]["V1"] == 16 and not out["comparisons"]["V1-D1"]["all_conditions"]
    assert out["decision"].startswith("V1 did not meet the prespecified threshold")
