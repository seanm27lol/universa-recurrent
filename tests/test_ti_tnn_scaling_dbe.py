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
    assert text.startswith("V1 meets") if lead else text.startswith("V1 did not meet the prespecified continuation threshold")
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
    ex = {"status": "ok", "seconds": 0.01, "row_count": 1, "truncated": False}
    pkg = {"items": [{"kind": "value", "verbatim_rows": 0, "candidates": [{"value": "x"}]}], "rendered_items": [0],
           "timeouts": [], "scan_limits": [], "truncated": False, "proposal": {"unresolved": []},
           "payload_chars": 50, "payload_chars_untruncated": 50, "queries": 3, "instructions": 1000,
           "latency_seconds": 0.1, "wall_safety_stop": False}
    for s in SEEDS:
        for i, t in enumerate(QIDS):
            ok = i != 0
            reqs.append({"seed": s, "task_id": t, "db_id": DB_OF[t], "revision_seconds_allowed": 170.0,
                         "original": {"sql": "SELECT 1", "call": call, "execution": ex},
                         "evidence": {"ok": ok, "package": pkg if ok else None, "seconds": 0.1,
                                      "error": None if ok else "E"},
                         **{p: {"source": "revised", "sql": "SELECT 2",
                                "revised": {"sql": "SELECT 2", "call": call, "execution": ex}} for p in ("D1", "V1")}})
            for p in D.POLICIES:
                right = int(p == "V1" and (s, t) in v1_right)
                scores.append({"seed": s, "task_id": t, "db_id": DB_OF[t], "policy": p, "score": {
                    "correct": right, "predicted": {"status": "ok", "elapsed_seconds": 0.01, "row_count": 1,
                                                    "distinct_rows": 1}}})
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
    assert out["decision"].startswith("V1 did not meet the prespecified continuation threshold")


# --- the post-run report (dbe_report) --------------------------------------------------------------------------------
from scaling import dbe_report as R  # noqa: E402


def test_the_report_validation_refuses_broken_records():
    reqs, scores = _records(set())
    for r in reqs:
        r["D0"] = {"source": "original", "sql": r["original"]["sql"]}
    R._validate(reqs, scores, QIDS, SEEDS)                                  # the intact records pass
    with pytest.raises(R.Inconsistent, match="requests"):
        R._validate(reqs + reqs[:1], scores, QIDS, SEEDS)
    with pytest.raises(R.Inconsistent, match="scores"):
        R._validate(reqs, scores[:-1], QIDS, SEEDS)
    bad = [dict(x, score={**x["score"], "correct": True}) if i == 0 else x for i, x in enumerate(scores)]
    with pytest.raises(R.Inconsistent, match="correctness"):
        R._validate(reqs, bad, QIDS, SEEDS)
    reqs[0]["evidence"] = {**reqs[0]["evidence"], "seconds": float("nan")}
    with pytest.raises(R.Inconsistent, match="finite"):
        R._validate(reqs, scores, QIDS, SEEDS)


# --- the public reconstruction against edited bundles (checksums regenerated, so only reconstruction can refuse) ---
from scaling import evidence  # noqa: E402


@pytest.fixture
def bundle_copy(tmp_path, monkeypatch):
    if not (R.BUNDLE / "SHA256SUMS").exists():
        pytest.skip("no dbe bundle")
    import shutil
    copy = tmp_path / "dbe_v1"
    shutil.copytree(R.BUNDLE, copy)
    monkeypatch.setattr(R, "BUNDLE", copy)
    (run,) = sorted(p for p in copy.glob("run/*") if p.is_dir())
    (adir,) = sorted(p for p in copy.glob("analysis/*") if p.is_dir())
    return copy, run, adir


def _edit_json(path, edit):
    import json as J
    obj = J.loads(path.read_text())
    edit(obj)
    path.write_text(J.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")


def _edit_lines(path, edit):
    import json as J
    rows = [J.loads(x) for x in path.read_text().splitlines() if x.strip()]
    edit(rows)
    path.write_text("".join(J.dumps(r, sort_keys=True) + "\n" for r in rows))


def test_the_bundle_reconstructs_with_the_recorded_results():
    if not (R.BUNDLE / "SHA256SUMS").exists():
        pytest.skip("no dbe bundle")
    got, _analysis = R.reconstruct()
    assert got["pooled"] == {"D0": 70, "D1": 80, "V1": 80} and got["comparisons"]["V1-D1"]["pooled"] == 0
    assert got["decision"].startswith("V1 did not meet the prespecified continuation threshold")
    assert [r["command"] for r in got["gpu_reservations"]] == ["dbe run"]


def _set(*path_value):
    *path, value = path_value

    def edit(obj):
        node = obj
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
    return edit


@pytest.mark.parametrize("edit, match", [
    (_set("protocol", "sha256", "0" * 64), "run identity"),                       # a changed protocol identity
    (_set("protocol", "frozen_at", "f" * 40), "run identity"),
    (_set("run", "dbe/run/20990101T000000Z"), "run identity"),                    # a changed run identity
    (_set("ledger", "used_hours", 23.0), "ledger.used_hours"),                    # ledger usage
    (_set("ledger", "remaining_hours", float("nan")), "ledger.remaining_hours"),
    (_set("gpu_reservations", 0, "hours", 0.2), "gpu_reservations"),              # reservation duration
    (_set("gpu_reservations", 0, "status", "killed"), "gpu_reservations"),
    (_set("note", "added"), "fields differ"),                                      # an extra analysis field
    (_set("seeds", [200, 201.0]), "seeds"),
    (_set("pooled", "V1", 81), "pooled")])
def test_a_changed_analysis_is_refused(bundle_copy, edit, match):
    copy, _run, adir = bundle_copy
    _edit_json(adir / "analysis.json", edit)
    evidence.write_sums(copy)
    with pytest.raises(R.Inconsistent, match=match):
        R.reconstruct()


def _first(field_path, value):
    def edit(rows):
        node = rows[0]
        for k in field_path[:-1]:
            node = node[k]
        node[field_path[-1]] = value
    return edit


@pytest.mark.parametrize("target, edit, match", [
    ("requests.jsonl", _first(("seed",), 200.0), "not an integer"),                        # float seeds
    ("scores.jsonl", _first(("seed",), 201.0), "not an integer"),
    ("requests.jsonl", _first(("original", "execution", "seconds"), -0.5), "finite nonnegative"),   # execution time
    ("requests.jsonl", _first(("V1", "revised", "execution", "seconds"), float("inf")), "finite nonnegative"),
    ("requests.jsonl", _first(("evidence", "seconds"), float("nan")), "finite nonnegative"),        # lookup time
    ("requests.jsonl", _first(("D1", "revised", "call", "seconds"), -1.0), "finite nonnegative"),   # generation time
    ("scores.jsonl", _first(("score", "predicted", "elapsed_seconds"), float("inf")), "finite nonnegative"),
    ("requests.jsonl", _first(("original", "call", "generated_tokens"), True), "nonnegative integer"),  # Boolean counts
    ("requests.jsonl", _first(("V1", "revised", "call", "prompt_tokens"), 12.0), "nonnegative integer"),
    ("requests.jsonl", _first(("original", "call", "peak_memory_bytes"), -1), "nonnegative integer"),
    ("requests.jsonl", _first(("evidence", "ok"), 1), "not a Boolean"),                    # genuine Booleans
    ("scores.jsonl", _first(("score", "correct"), True), "integer 0 or 1"),
    ("scores.jsonl", _first(("score", "correct"), 1.0), "integer 0 or 1"),
    ("requests.jsonl", _first(("db_id",), "world"), "pinned"),                             # population binding
    ("scores.jsonl", lambda rows: rows.append(dict(rows[0])), "scores")])
def test_invalid_records_are_refused(bundle_copy, target, edit, match):
    copy, run, _adir = bundle_copy
    if target == "requests.jsonl" and match == "pinned":
        rows = (run / target).read_text()
        if '"db_id": "world"' in rows.splitlines()[0]:
            edit = _first(("db_id",), "disney")
    _edit_lines(run / target, edit)
    evidence.write_sums(copy)
    with pytest.raises(R.Inconsistent, match=match):
        R.reconstruct()


def test_a_changed_ledger_or_manifest_is_refused(bundle_copy):
    import json as J
    copy, run, adir = bundle_copy
    lines = (copy / "gpu_ledger.jsonl").read_text().splitlines()
    (end,) = [i for i, x in enumerate(lines) if '"event": "end"' in x and "dbe run" in "".join(
        y for y in lines if J.loads(y)["reservation"] == J.loads(x)["reservation"])]
    e = J.loads(lines[end])
    e["elapsed"] = e["elapsed"] + 360.0                        # a longer reservation in the ledger itself
    lines[end] = J.dumps(e)
    (copy / "gpu_ledger.jsonl").write_text("\n".join(lines) + "\n")
    evidence.write_sums(copy)
    with pytest.raises(R.Inconsistent, match="gpu_reservations|ledger"):
        R.reconstruct()


@pytest.mark.parametrize("which, edit, match", [
    ("run", _set("git", "research_tree_dirty", True), "not clean"),
    ("analysis", _set("git", "commit", "a" * 40), "commit"),
    ("run", _set("config", "seeds", [200]), "seeds"),
    ("run", _set("config", "lookup", "db-evidence-v1.3"), "lookup")])
def test_a_changed_manifest_is_refused(bundle_copy, which, edit, match):
    copy, run, adir = bundle_copy
    _edit_json((run if which == "run" else adir) / "manifest.json", edit)
    evidence.write_sums(copy)
    with pytest.raises(R.Inconsistent, match=match):
        R.reconstruct()


def test_legitimate_roundoff_passes(bundle_copy):
    copy, _run, adir = bundle_copy
    _edit_json(adir / "analysis.json", lambda a: (a["cost"]["V1"].__setitem__(
        "total_seconds", a["cost"]["V1"]["total_seconds"] * (1 + 1e-13)), a["ledger"].__setitem__(
        "used_hours", a["ledger"]["used_hours"] * (1 + 1e-13))))
    evidence.write_sums(copy)
    got, _a = R.reconstruct()
    assert got["pooled"]["V1"] == 80


def test_a_consistently_renamed_run_directory_is_refused(bundle_copy):
    """The reporting-correction review's finding: renaming the run and every reference to it in the analysis was
    accepted, because only the ledger's start entry, written by the run itself, still names the original."""
    copy, run, adir = bundle_copy
    new = run.with_name("20991231T000000Z")
    run.rename(new)
    _edit_json(adir / "analysis.json", _set("run", "dbe/run/" + new.name))
    _edit_json(adir / "manifest.json", _set("config", "run", "dbe/run/" + new.name))
    evidence.write_sums(copy)
    with pytest.raises(R.Inconsistent, match="ledger's reservation names"):
        R.reconstruct()
