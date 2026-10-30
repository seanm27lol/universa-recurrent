"""The pilot's commands end to end on the CPU, with a tiny random Qwen3.5 backbone and a synthetic database (no
downloaded data, no GPU, no ledger):

    feasibility (measure, project) -> encode dev (with the determinism recheck) -> encode train chunks
    -> train both arms -> final-checkpoint gate -> evaluate (timed + reconstruction) -> analyze

**The refusals and stops it checks:**
- an unfrozen or changed protocol;
- a missing anchor;
- anchors from a dirty tree or another protocol;
- a torn last anchor line;
- a run directory without `--resume`, and `--resume` on a completed run;
- a training log that is not complete;
- a final checkpoint whose attempt summary was lost (recovered from `checkpoints.jsonl`);
- a recomputed development context that differs;
- a fresh-vs-anchor mismatch, and a cross-arm states mismatch, at evaluation;
- an analysis with two passing evaluations for one seed."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "ti_tnn"))
torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_5")
pytest.importorskip("sqlglot")
from scaling import contracts as C  # noqa: E402
from scaling import encode as E  # noqa: E402
from scaling import pilot as P  # noqa: E402
from scaling import pilot_run as R  # noqa: E402
from scaling import schema as schema_mod  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ti_tnn_scaling_controller as TC  # noqa: E402

REAL_PROTOCOL_IDENTITY = R.protocol_identity
QUESTIONS = [("Who directed the movie Alpha?", "Alpha refers to title = 'Alpha';",
              "SELECT director FROM movies WHERE title = 'Alpha'"),
             ("How many movies did Jo direct after 1999?", "Jo refers to director = 'Jo';",
              "SELECT COUNT(*) FROM movies WHERE director = 'Jo' AND year > 1999"),
             ("Which movie has the highest average score?", "",
              "SELECT T1.title FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
              "GROUP BY T1.title ORDER BY AVG(T2.score) DESC LIMIT 1"),
             ("Who directed Beta?", "Beta refers to title = 'Beta';", "SELECT director FROM movies WHERE title = 'Beta'")]
TRAIN = [(f"tr{i}", f"{q} (training {i})", e, sql) for i, (q, e, sql) in enumerate(QUESTIONS)]
DEV = [(f"dv{i}", f"{q} (development {i})", e, sql) for i, (q, e, sql) in enumerate(QUESTIONS)]
PROTO = {"file": "PROTOCOL_pilot.md", "sha256": "f" * 64}


class _FakeReservation:
    def __init__(self, command, projected_hours, extra=None, **kw):
        self.id = f"fake-{command}"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def check(self):
        pass

    def elapsed_hours(self):
        return 0.0


@pytest.fixture()
def world(tmp_path, monkeypatch):
    from scaling import cli, data, identity, ledger, retrieval, scorer, verify
    tasks = {tid: C.Task(task_id=tid, db_id="cinema", question=q, evidence=e,
                         split="train" if tid.startswith("tr") else "dev") for tid, q, e, _ in TRAIN + DEV}
    golds = {tid: C.Gold(task_id=tid, sql=sql) for tid, _, _, sql in TRAIN + DEV}
    bb = E.Backbone.tiny_random(E.MockTokenizer(), hidden_size=16, layers=2, seed=0)
    monkeypatch.setattr(data, "task", lambda tid: tasks[tid])
    monkeypatch.setattr(data, "gold", lambda tid: golds[tid])
    monkeypatch.setattr(data, "snapshot", lambda db: C.Snapshot(db_id=db, path="/x", sha256="0" * 64, bytes=1))
    monkeypatch.setattr(schema_mod, "load_schema", lambda db: TC.SCHEMA)
    monkeypatch.setattr(retrieval, "retrieve", lambda task: [TC.EXAMPLE])
    monkeypatch.setattr(scorer, "score", lambda snap, sql, gold: {"correct": int(bool(sql) and "director" in sql),
                                                                 "status": "scored"})
    monkeypatch.setattr(cli, "_warm_databases", lambda dbs: 0.0)
    monkeypatch.setattr(identity, "model_identity", lambda key, verify=False: {"key": key})
    monkeypatch.setattr(identity, "git_state", lambda: {"commit": "c" * 40, "branch": "t", "worktree": "w",
                                                        "research_tree_dirty": False, "dirty_paths": []})
    monkeypatch.setattr(ledger, "Reservation", _FakeReservation)
    monkeypatch.setattr(ledger, "usage", lambda: {"used_hours": 2.979})
    monkeypatch.setattr(verify, "verify_run", lambda d: {"problems": []})
    monkeypatch.setattr(C, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(P, "ROOT", tmp_path / "pilot")
    monkeypatch.setattr(E, "PILOT_CACHE", tmp_path / "cache")
    monkeypatch.setattr(P, "train_ids", lambda: [t for t, *_ in TRAIN])
    monkeypatch.setattr(P, "eval_ids", lambda: [t for t, *_ in DEV])
    monkeypatch.setattr(P, "UPDATES", 4)
    monkeypatch.setattr(P, "CHECKPOINT_EVERY", 2)
    monkeypatch.setattr(P, "MATMUL_PRECISION", "highest")
    monkeypatch.setattr(R, "TRAIN_CHUNKS", ("0:2", "2:4"))
    monkeypatch.setattr(R, "DEVICE", "cpu")
    monkeypatch.setattr(R, "_load_backbone", lambda: (None, bb))
    monkeypatch.setattr(R, "_tokenizer", lambda: bb.tokenizer)
    monkeypatch.setattr(R, "protocol_identity", lambda: PROTO)
    monkeypatch.setattr(P, "representability", lambda ids, tok: {t: "representable" for t in ids})
    return tmp_path, bb


def _args(**kw):
    return argparse.Namespace(allow_dirty=True, projected_hours=0.1, **kw)


def _l_run(tmp_path):
    from scaling import records
    d = C.RUNS / "baselines" / "20261030T000000Z_L_nonthinking_p0-64"
    d.mkdir(parents=True)
    for tid, *_ in DEV:
        rec = {"task_id": tid, "db_id": "cinema", "arm": "L", "config_identity": "c", "input_identity": "i",
               "snapshot_sha256": "0" * 64, "attempts": [], "final": {"kind": "abstain", "sql": None, "reason": "x"},
               "score": {"correct": 0, "scored_sql_sha256": records.sql_sha256(None), "snapshot_sha256": "0" * 64},
               "tokens": {}, "controller_work": None, "elapsed_seconds": 0.01, "peak_memory_bytes": None,
               "timed_out": False, "error": None, "repairs": 0, "tool_calls": 0, "tool_log": []}
        with open(d / "records.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    return d


def _run_to_trained(monkeypatch):
    R.feasibility_cmd(_args(sample=2, dev=2, updates=3, checkpoint_every=1, seed=999))
    R.encode_cmd(_args(what="dev", range=None))
    for chunk in R.TRAIN_CHUNKS:
        R.encode_cmd(_args(what="train", range=chunk))
    for seed in P.SEEDS:
        R.train_cmd(_args(seed=seed, arms=["T4", "c4_params"], resume=False))


def test_the_pilot_runs_end_to_end_and_reconstructs(world, monkeypatch):
    tmp_path, bb = world
    _run_to_trained(monkeypatch)
    feas = json.loads(sorted((P.ROOT / "feasibility").glob("*/summary.json"))[-1].read_text())
    assert feas["encode"]["encoded"] == 4 and set(feas["arms"]) == set(P.ARMS)
    assert feas["projection"]["fits"] in (True, False) and feas["projection"]["needed_hours"] > 0
    dev_run = sorted((P.ROOT / "encode").glob("*_dev"))[-1]
    dev_sum = json.loads((dev_run / "summary.json").read_text())
    assert dev_sum["rechecked"] == 2 and dev_sum["rechecked_equal"] == 2  # the feasibility run's 2 dev contexts
    anchors, ident, ignored = P.load_anchors(protocol_sha256=PROTO["sha256"])
    assert set(anchors) == {t for t, *_ in TRAIN + DEV} and ident == bb.identity() and not ignored
    for seed in P.SEEDS:
        R.eval_cmd(_args(seed=seed))
    _l_run(tmp_path)
    R.analyze_cmd(_args())
    out = json.loads(sorted((P.ROOT / "analysis").glob("*/analysis.json"))[-1].read_text())
    assert out["counts"]["questions"] == len(DEV) and out["rule"][:2] in ("A1", "A2", "A3")
    assert set(out["inputs"]["evaluations_counted"]) == {"100", "101"} and len(out["inputs"]["evaluations_all"]) == 2


def test_a_lost_summary_does_not_strand_a_finished_run_and_resume_is_refused(world, monkeypatch):
    _run_to_trained(monkeypatch)
    run_dir = P.ROOT / "train" / "T4_s100"
    for s in run_dir.glob("attempts/*/summary.json"):
        s.unlink()  # a stop after the final save, before the summary
    path, rec = R.final_checkpoint(run_dir)
    assert rec["update"] == 3
    with pytest.raises(R.PilotStop, match="already complete"):
        R.train_cmd(_args(seed=100, arms=["T4"], resume=True))
    log = run_dir / "train_log.jsonl"
    log.write_text("".join(line + "\n" for line in log.read_text().splitlines()[:-1]))
    with pytest.raises(R.PilotStop, match="does not hold updates"):
        R.final_checkpoint(run_dir)


def test_evaluation_stops_on_a_fresh_states_mismatch(world, monkeypatch):
    _run_to_trained(monkeypatch)
    entries = sorted((P.ROOT / "encode").glob("*_dev"))[-1] / "entries.jsonl"
    lines = [json.loads(x) for x in entries.read_text().splitlines()]
    # a fresh forward that disagrees with the anchors: the evaluation reports it and stops
    real = E.encode_request

    def drifting(bb, task, *a, use_cache=True, **k):
        enc = real(bb, task, *a, use_cache=use_cache, **k)
        if not use_cache and task.task_id == lines[0]["task_id"]:
            enc.states = enc.states + 1e-3
        return enc
    monkeypatch.setattr(E, "encode_request", drifting)
    with pytest.raises(R.PilotStop, match="integrity"):
        R.eval_cmd(_args(seed=100))
    summary = json.loads(sorted((P.ROOT / "eval").glob("s100_*"))[-1].joinpath("summary.json").read_text())
    assert summary["passed"] is False and summary["arms"]["T4"]["fresh_states_all_equal_anchor"] is False


def test_analysis_counts_exactly_one_passing_evaluation_per_seed(world, monkeypatch):
    tmp_path, _ = world
    _run_to_trained(monkeypatch)
    for seed in P.SEEDS:
        R.eval_cmd(_args(seed=seed))
    _l_run(tmp_path)
    first = sorted((P.ROOT / "eval").glob("s100_*"))[0]
    shutil.copytree(first, first.with_name(first.name + "_rerun"))  # a second passing seed-100 evaluation
    with pytest.raises(R.PilotStop, match="exactly one"):
        R.analyze_cmd(_args())
    rerun = first.with_name(first.name + "_rerun") / "summary.json"
    s = json.loads(rerun.read_text())
    s["states_differ_across_arms"] = ["dv0"]
    rerun.write_text(json.dumps(s))  # a failed rerun is reported but not counted
    R.analyze_cmd(_args())
    out = json.loads(sorted((P.ROOT / "analysis").glob("*/analysis.json"))[-1].read_text())
    assert out["inputs"]["evaluations_counted"]["100"] == str(first)
    assert sum(not r["passed"] for r in out["inputs"]["evaluations_all"]) == 1


def test_a_recomputed_development_context_that_differs_stops_the_pilot(world, monkeypatch):
    R.feasibility_cmd(_args(sample=2, dev=2, updates=3, checkpoint_every=1, seed=999))
    real = E.encode_request

    def drifting(bb, task, *a, use_cache=True, **k):
        enc = real(bb, task, *a, use_cache=use_cache, **k)
        if not use_cache:
            enc.states = enc.states + 1e-3
        return enc
    monkeypatch.setattr(E, "encode_request", drifting)
    with pytest.raises(R.PilotStop, match="cross-process"):
        R.encode_cmd(_args(what="dev", range=None))


def test_the_pilot_refuses_what_the_protocol_forbids(world, monkeypatch):
    tmp_path, bb = world
    monkeypatch.setattr(R, "protocol_identity", REAL_PROTOCOL_IDENTITY)
    monkeypatch.setattr(R, "PROTOCOL_PIN", tmp_path / "no_pin.json")
    with pytest.raises(R.PilotStop, match="not frozen"):
        R.encode_cmd(_args(what="dev", range=None))
    protocol = tmp_path / "PROTOCOL_pilot.md"
    protocol.write_text("frozen text\n")
    monkeypatch.setattr(P, "PROTOCOL", protocol)
    (tmp_path / "pin.json").write_text(json.dumps({"sha256": "0" * 64}))
    monkeypatch.setattr(R, "PROTOCOL_PIN", tmp_path / "pin.json")
    with pytest.raises(R.PilotStop, match="frozen hash"):
        R.encode_cmd(_args(what="dev", range=None))
    monkeypatch.setattr(R, "protocol_identity", lambda: PROTO)
    with pytest.raises(R.PilotStop, match="chunks"):
        R.encode_cmd(_args(what="train", range="0:3"))
    R.encode_cmd(_args(what="train", range="0:2"))
    with pytest.raises(R.PilotStop, match="no encoding anchor"):
        R.train_cmd(_args(seed=100, arms=["T4"], resume=False))
    with pytest.raises(R.PilotStop, match="not one of"):
        R.train_cmd(_args(seed=7, arms=["T4"], resume=False))
    entries = sorted((P.ROOT / "encode").glob("*_train_0-2"))[-1] / "entries.jsonl"
    entries.write_text(entries.read_text() + '{"task_id": "tr9", "cache')  # a torn last line
    found, _, ignored = P.load_anchors(protocol_sha256=PROTO["sha256"])
    assert "tr9" not in found and ignored
    manifest = entries.parent / "manifest.json"
    m = json.loads(manifest.read_text())
    m["config"]["protocol"]["sha256"] = "1" * 64
    manifest.write_text(json.dumps(m))
    with pytest.raises(R.PilotStop, match="another protocol"):
        R.train_cmd(_args(seed=100, arms=["T4"], resume=False))


def test_the_projection_counts_every_remaining_step():
    m = {"encode": {"seconds_per_new_context": 2.0}, "load_seconds": 60.0,
         "arms": {a: {"update_seconds_after_first": 0.5, "first_update_seconds": 3.0, "checkpoint_save_seconds_mean": 2.0,
                      "fixed_seconds": 10.0, "eval_seconds_per_request": 1.2,
                      "reconstruction_seconds_per_request": 0.1} for a in P.ARMS}}
    import scaling.ledger as L
    orig = L.usage
    L.usage = lambda: {"used_hours": 3.0}
    try:
        out = R.project(m, n_pool=4741, n_eval=64, n_sample=160, n_dev=8)
    finally:
        L.usage = orig
    enc = (((4741 - 160) + (64 - 8)) * 2.0 + 8 * 2.0 + 5 * 60) / 3600
    fit = 1499 * 0.5 + 3.0 + 6 * 2.0 + 10.0
    ev = (2 * 60 + 2 * 2 * 64 * 1.3) / 3600
    items = list(out["items_hours"].values())
    assert items[0] == pytest.approx(enc) and items[1] == pytest.approx(2 * 2 * fit / 3600)
    assert items[2] == pytest.approx(ev) and out["remaining_hours"] == pytest.approx(5.0)
    assert out["fits"] == (enc + 4 * fit / 3600 + ev <= 5.0)
    with pytest.raises(R.PilotStop, match="not measured"):
        R.project({**m, "encode": {"seconds_per_new_context": None}}, 4741, 64, 160, 8)
