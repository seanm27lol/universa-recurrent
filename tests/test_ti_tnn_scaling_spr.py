"""The semantic-proposal-repair study: the typed analysis, the raw observation and the selection rule.

- **Selection** is the same for D1 and TI1 and never reads gold. The revised proposal is kept if it is valid (one
  read-only statement that executes), else the original if valid, else the policy abstains.
- **The revision seed** is shared by D1 and TI1.
- **The typed analysis** separates errors (an identifier the schema does not have, a SQLite error) from observational
  warnings (an ON predicate that references only one source, a literal not in the question, an empty result). A
  legitimate self-join on a real key gets no warning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
sys.path.insert(0, str(REPO / "tests"))
pytest.importorskip("sqlglot")
from scaling import ir  # noqa: E402
from scaling import spr_feedback as F  # noqa: E402
from scaling import spr_run as S  # noqa: E402
from scaling import spr_typed as T  # noqa: E402

COLUMNS = {"movies": ["movie_id", "title", "director", "year"], "ratings": ["rating_id", "movie_id", "score", "critic"],
           "people": ["person_id", "name", "mentor_id"]}


class _Task:
    task_id, db_id = "t1", "cinema"
    question = "Which movies directed by Ada have a score above 3?"
    evidence = "Ada refers to director = 'Ada'"


@pytest.fixture
def analysis(monkeypatch):
    monkeypatch.setattr(T, "schema_index", lambda task, snap: ir.SchemaIndex(COLUMNS))
    ok = {"status": "ok", "row_count": 2, "rows": [[1], [2]], "columns": ["title"], "truncated": False}
    return lambda sql, ex=ok: F.typed_analysis(_Task(), None, sql, ex)


def test_selection_keeps_a_valid_revision_else_a_valid_original_else_abstains():
    ok, bad = {"status": "ok"}, {"status": "error"}
    o = {"sql": "SELECT 1", "execution": ok}
    assert S.select(o, {"sql": "SELECT 2", "execution": ok}) == {"source": "revised", "sql": "SELECT 2"}
    assert S.select(o, {"sql": "SELECT 2", "execution": bad})["source"] == "original"
    assert S.select(o, {"sql": None, "execution": {"status": "no_sql"}})["source"] == "original"
    assert S.select(o, {"sql": "DELETE FROM movies", "execution": ok})["source"] == "original"   # not read-only
    assert S.select({"sql": "SELECT 1", "execution": bad}, {"sql": "SELECT 2", "execution": bad}) == \
        {"source": "abstain", "sql": None}


def test_d1_and_ti1_share_the_revision_seed_and_seeds_differ_by_kind():
    assert S.seed_of("revise", 200, "t1") == S.seed_of("revise", 200, "t1")
    assert S.seed_of("revise", 200, "t1") != S.seed_of("initial", 200, "t1") != S.seed_of("revise", 201, "t1")


def test_the_raw_observation_is_bounded():
    rows = [["x" * 500, i] for i in range(20)]
    text = F.raw_observation({"status": "ok", "row_count": 1000, "rows": rows, "columns": ["a", "b"],
                              "truncated": True})
    assert "Rows returned: 1000 (showing the first 20)" in text and len(text) <= F.FEEDBACK_CHARS + 40
    assert F.raw_observation({"status": "error", "error": "no such column: q"}).startswith("Execution status: error")


def test_a_one_source_on_predicate_is_a_warning_and_a_real_self_join_is_not(analysis):
    a = analysis("SELECT T2.title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T1.movie_id")
    assert a["typed"] and not a["errors"]
    assert any("references only ratings (source 0)" in w for w in a["warnings"])
    b = analysis("SELECT T2.name FROM people AS T1 INNER JOIN people AS T2 ON T1.mentor_id = T2.person_id")
    assert b["typed"] and not any("references only" in w for w in b["warnings"])


def test_unknown_identifiers_are_errors_and_observations_are_warnings(analysis):
    a = analysis("SELECT budget FROM movies")
    assert not a["typed"] and any("unknown column" in e or "unresolved column" in e for e in a["errors"])
    b = analysis("SELECT title FROM movies WHERE director = 'Bob'",
                 {"status": "ok", "row_count": 0, "rows": [], "columns": ["title"], "truncated": False})
    assert not b["errors"]
    assert any("'Bob' does not occur verbatim" in w for w in b["warnings"])
    assert any("returned no rows" in w for w in b["warnings"])
    c = analysis("SELECT title FROM movies WHERE director = 'Ada'")
    assert not any("'Ada'" in w for w in c["warnings"])                     # the evidence names it
    d = analysis("SELECT title FROM movies", {"status": "error", "error": "misuse of aggregate: MAX()"})
    assert any("misuse of aggregate" in e for e in d["errors"])
    rendered = F.render_typed(a)
    assert "Errors:" in rendered and "Warnings:" in rendered


def test_a_construct_outside_the_ir_is_an_untyped_warning_not_a_rejection(analysis):
    a = analysis("WITH x AS (SELECT title FROM movies) SELECT title FROM x")
    assert not a["typed"] and not a["errors"] and any("outside the typed representation" in w for w in a["warnings"])


# --- the registered rule (spr_rule, v2) ------------------------------------------------------------------------------
from scaling import spr_rule as RULE  # noqa: E402

SEEDS = (200, 201)
DBS = [f"db{k}" for k in range(8)]
QIDS = [f"q{i:02d}" for i in range(64)]
DB_OF = {t: DBS[i % 8] for i, t in enumerate(QIDS)}


def _table(wins: dict, losses: dict = None) -> dict:
    """correct[(policy, seed)][task]: 'a' right where listed in wins; 'b' right where listed in losses."""
    losses = losses or set()
    return {**{("a", s): {t: int((s, t) in wins) for t in QIDS} for s in SEEDS},
            **{("b", s): {t: int((s, t) in losses) for t in QIDS} for s in SEEDS}}


def _spread(n_per_seed: dict) -> set:
    """n_per_seed[s] wins in seed s, spread over the databases."""
    return {(s, QIDS[i]) for s, n in n_per_seed.items() for i in range(n)}


@pytest.mark.parametrize("wins, meets", [({200: 4, 201: 3}, False), ({200: 4, 201: 4}, True)])
def test_plus_7_is_not_a_lead_and_plus_8_is(wins, meets):
    d = RULE.paired_rule(_table(_spread(wins)), "a", "b", SEEDS, QIDS, DB_OF)
    assert d["pooled"] == sum(wins.values()) and d["threshold"] == 8
    assert d["conditions"]["same_sign_every_seed"] and d["conditions"]["same_sign_without_any_one_database"]
    assert d["conditions"]["pooled_at_least_lead"] is meets and d["all_conditions"] is meets
    assert RULE.lead(d) is meets and not RULE.harm(d)


@pytest.mark.parametrize("losses, meets", [({200: 4, 201: 3}, False), ({200: 4, 201: 4}, True)])
def test_minus_7_is_not_harm_and_minus_8_is(losses, meets):
    d = RULE.paired_rule(_table(set(), _spread(losses)), "a", "b", SEEDS, QIDS, DB_OF)
    assert d["pooled"] == -sum(losses.values()) and d["all_conditions"] is meets
    assert RULE.harm(d) is meets and not RULE.lead(d)


def test_a_zero_difference_seed_fails_the_seed_condition():
    d = RULE.paired_rule(_table(_spread({200: 10, 201: 0})), "a", "b", SEEDS, QIDS, DB_OF)
    assert d["pooled"] == 10 and d["per_seed"] == {"200": 10, "201": 0}
    assert d["conditions"]["pooled_at_least_lead"] and not d["conditions"]["same_sign_every_seed"]
    assert not d["all_conditions"] and not RULE.lead(d)


def test_a_gain_concentrated_in_one_database_fails_the_database_condition():
    one_db = {(s, t) for s in SEEDS for t in QIDS if DB_OF[t] == "db0"}   # 8 per seed, all in db0
    d = RULE.paired_rule(_table(one_db), "a", "b", SEEDS, QIDS, DB_OF)
    assert d["pooled"] == 16 and d["conditions"]["same_sign_every_seed"]
    assert d["leave_one_database_out"]["db0"] == 0 and not d["conditions"]["same_sign_without_any_one_database"]
    assert not RULE.lead(d)


def test_the_decision_branches():
    base = RULE.paired_rule(_table(set()), "a", "b", SEEDS, QIDS, DB_OF)
    win = RULE.paired_rule(_table(_spread({200: 4, 201: 4})), "a", "b", SEEDS, QIDS, DB_OF)
    loss = RULE.paired_rule(_table(set(), _spread({200: 4, 201: 4})), "a", "b", SEEDS, QIDS, DB_OF)
    assert RULE.decision({"TI1-D1": win, "D1-D0": base, "TI1-D0": base}).startswith("TI1 beats D1")
    assert RULE.decision({"TI1-D1": base, "D1-D0": base, "TI1-D0": loss}).startswith("TI1 harms")
    assert RULE.decision({"TI1-D1": base, "D1-D0": win, "TI1-D0": win}).startswith("TI1 ties D1")
    assert RULE.decision({"TI1-D1": base, "D1-D0": base, "TI1-D0": base}).startswith("neither")


# --- public reconstruction from the bundle -------------------------------------------------------------------------
from scaling import evidence  # noqa: E402
from scaling import spr_report as SR  # noqa: E402


def _bundle_copy(tmp_path, monkeypatch):
    if not (SR.BUNDLE / "SHA256SUMS").exists():
        pytest.skip("no spr bundle")
    import shutil
    copy = tmp_path / "spr_v1"
    shutil.copytree(SR.BUNDLE, copy)
    monkeypatch.setattr(SR, "BUNDLE", copy)
    (run,) = sorted(copy.glob("run/*"))
    return copy, run


def _rewrite(path, edit):
    import json as J
    rows = [J.loads(x) for x in path.read_text().splitlines() if x.strip()]
    rows = edit(rows)
    path.write_text("".join(J.dumps(r, sort_keys=True) + "\n" for r in rows))


def test_the_bundle_reconstructs_and_keeps_the_observed_verdict():
    if not (SR.BUNDLE / "SHA256SUMS").exists():
        pytest.skip("no spr bundle")
    got, original, corrected = SR.reconstruct()
    assert got["pooled"] == {"D0": 74, "D1": 76, "TI1": 78}
    assert got["decision"] == original["decision"] == corrected["decision"]
    assert got["comparisons"]["TI1-D1"]["threshold"] == 8 and not got["comparisons"]["TI1-D1"]["all_conditions"]
    assert original["comparisons"]["TI1-D1"]["threshold"] == 7   # the preserved historical defect


def _edit_cost(rows):
    rows[0]["original"]["call"]["seconds"] += 1.0
    return rows


def _edit_db(rows):
    rows[0]["db_id"] = "world" if rows[0]["db_id"] != "world" else "disney"
    return rows


def _edit_selection(rows):
    r = next(r for r in rows if r["D1"]["source"] == "revised")
    r["D1"]["source"], r["D1"]["sql"] = "original", r["original"]["sql"]
    return rows


def _edit_seed(rows):
    rows[0]["seed"] = 202
    return rows


@pytest.mark.parametrize("target, edit, match", [
    ("requests.jsonl", _edit_cost, "corrected.cost"),
    ("requests.jsonl", _edit_db, "not the pinned"),
    ("requests.jsonl", _edit_selection, "selection differs"),
    ("requests.jsonl", _edit_seed, "requests:"),
    ("scores.jsonl", lambda rows: rows + rows[:1], "scores: 1 duplicated")])
def test_corruption_is_refused_even_with_regenerated_checksums(tmp_path, monkeypatch, target, edit, match):
    copy, run = _bundle_copy(tmp_path, monkeypatch)
    _rewrite(run / target, edit)
    evidence.write_sums(copy)          # the sums now match the corrupted bundle: only reconstruction can catch it
    with pytest.raises(SR.Inconsistent, match=match):
        SR.reconstruct()


# --- value validation before comparison (2026-11-07) -----------------------------------------------------------------
def test_differ_checks_types_before_equality_and_tolerance():
    inf = float("inf")
    assert SR._differ(1.0, inf, "x") == ["x"] and SR._differ(inf, inf, "x") == ["x"]
    assert SR._differ(-inf, 5.0, "x") == ["x"] and SR._differ(float("nan"), 1.0, "x") == ["x"]
    assert SR._differ(True, 1, "x") == ["x"] and SR._differ(0, False, "x") == ["x"]
    assert SR._differ(1.0, True, "x") == ["x"]
    assert SR._differ(74, 74.0, "x") == ["x"]                         # an integer count is not a float
    assert SR._differ(0.1 + 0.2, 0.3, "x") == [] and SR._differ(1525.123, 1525.123 * (1 + 1e-13), "x") == []
    assert SR._differ(True, True, "x") == [] and SR._differ(3, 3, "x") == [] and SR._differ(None, None, "x") == []
    assert SR._differ([1, 2.0], [1, 2.0], "x") == [] and SR._differ([True], [1], "x") == ["x[0]"]
    assert SR._differ([1], [1, 2], "x") == ["x[len]"]


def _edit_analysis(copy, name, edit):
    import json as J
    (path,) = [p for p in copy.glob(f"analysis/*/{name}")]
    obj = J.loads(path.read_text())
    edit(obj)
    path.write_text(J.dumps(obj, indent=1, sort_keys=True) + "\n")   # json writes Infinity for inf


@pytest.mark.parametrize("name, edit, match", [
    ("analysis_v2.json", lambda a: a["cost"]["TI1"].__setitem__("total_seconds", float("inf")), "corrected.cost"),
    ("analysis_v2.json", lambda a: a["cost"]["D1"].__setitem__("tool_seconds", float("-inf")), "corrected.cost"),
    ("analysis.json", lambda a: a["cost"]["D0"].__setitem__(
        "per_request_seconds", a["cost"]["D0"]["per_request_seconds"] + 1.0), "original.cost"),
    ("analysis.json", lambda a: a["counts"].__setitem__("D0/200", 37.0), "original.counts"),
    ("analysis_v2.json", lambda a: a["comparisons"]["TI1-D1"].__setitem__("all_conditions", 0),
     "corrected.comparisons")])
def test_changed_analysis_values_are_refused_with_regenerated_checksums(tmp_path, monkeypatch, name, edit, match):
    copy, _run = _bundle_copy(tmp_path, monkeypatch)
    _edit_analysis(copy, name, edit)
    evidence.write_sums(copy)
    with pytest.raises(SR.Inconsistent, match=match):
        SR.reconstruct()


def test_legitimate_roundoff_in_an_analysis_cost_passes(tmp_path, monkeypatch):
    copy, _run = _bundle_copy(tmp_path, monkeypatch)
    _edit_analysis(copy, "analysis_v2.json", lambda a: a["cost"]["TI1"].__setitem__(
        "total_seconds", a["cost"]["TI1"]["total_seconds"] * (1 + 1e-13)))
    evidence.write_sums(copy)
    got, _o, _c = SR.reconstruct()
    assert got["pooled"]["TI1"] == 78


def _score_edit(value):
    def edit(rows):
        rows[0]["score"]["correct"] = value
        return rows
    return edit


def _request_edit(path, value):
    def edit(rows):
        node = rows[0]
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
        return rows
    return edit


@pytest.mark.parametrize("target, edit, match", [
    ("scores.jsonl", _score_edit(2), "not integer 0 or 1"),
    ("scores.jsonl", _score_edit(True), "not integer 0 or 1"),
    ("scores.jsonl", _score_edit(1.0), "not integer 0 or 1"),
    ("requests.jsonl", _request_edit(("original", "call", "seconds"), -1.0), "finite nonnegative time"),
    ("requests.jsonl", _request_edit(("original", "call", "seconds"), float("inf")), "finite nonnegative time"),
    ("requests.jsonl", _request_edit(("D1", "revised", "call", "generated_tokens"), 3.5), "nonnegative integer"),
    ("requests.jsonl", _request_edit(("TI1", "revised", "typed", "typed"), 1), "not a Boolean")])
def test_invalid_record_values_are_refused_with_regenerated_checksums(tmp_path, monkeypatch, target, edit, match):
    copy, run = _bundle_copy(tmp_path, monkeypatch)
    _rewrite(run / target, edit)
    evidence.write_sums(copy)
    with pytest.raises(SR.Inconsistent, match=match):
        SR.reconstruct()
