"""The pilot's commands end to end on the CPU, with a tiny random Qwen3.5 backbone and a synthetic database (no
downloaded data, no GPU, no ledger):

    encode (with anchors) -> prepare (anchored, verified, lazy) -> train both arms (checkpoints, resume binding)
    -> final-checkpoint gate -> evaluate (timed + reconstruction, cross-arm states) -> analyze

It also checks the refusals: an unfrozen protocol, a missing anchor, a run directory without --resume, and a
training log that is not complete."""
from __future__ import annotations

import argparse
import contextlib
import json
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
TRAIN = [(f"tr{i}", q, e, sql) for i, (_t, q, e, sql) in enumerate(TC.REQUESTS)]
DEV = [(f"dv{i}", q, e, sql) for i, (_t, q, e, sql) in enumerate(TC.REQUESTS)]


@pytest.fixture()
def world(tmp_path, monkeypatch):
    from scaling import cli, data, identity, ledger, retrieval, scorer
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
    monkeypatch.setattr(ledger, "Reservation", _FakeReservation)
    monkeypatch.setattr(ledger, "usage", lambda: {"used_hours": 2.979})
    monkeypatch.setattr(P, "ROOT", tmp_path / "pilot")
    monkeypatch.setattr(E, "PILOT_CACHE", tmp_path / "cache")
    monkeypatch.setattr(P, "train_ids", lambda: [t for t, *_ in TRAIN])
    monkeypatch.setattr(P, "eval_ids", lambda: [t for t, *_ in DEV])
    monkeypatch.setattr(P, "UPDATES", 4)
    monkeypatch.setattr(P, "CHECKPOINT_EVERY", 2)
    monkeypatch.setattr(P, "MATMUL_PRECISION", "highest")
    monkeypatch.setattr(R, "DEVICE", "cpu")
    monkeypatch.setattr(R, "_load_backbone", lambda: (None, bb))
    monkeypatch.setattr(R, "_tokenizer", lambda: bb.tokenizer)
    monkeypatch.setattr(R, "protocol_identity", lambda: {"file": "PROTOCOL_pilot.md", "sha256": "f" * 64})
    monkeypatch.setattr(P, "representability", lambda ids, tok: {t: "representable" for t in ids})
    return tmp_path, bb


class _FakeReservation:
    def __init__(self, command, projected_hours, extra=None, **kw):
        self.id, self.projected = f"fake-{command}", projected_hours

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def check(self):
        pass

    def elapsed_hours(self):
        return 0.0


def _args(**kw):
    return argparse.Namespace(allow_dirty=True, projected_hours=0.1, **kw)


def test_the_pilot_pipeline_runs_end_to_end_and_reconstructs(world):
    tmp_path, bb = world
    R.encode_cmd(_args(what="dev", range=None))
    R.encode_cmd(_args(what="train", range="0:3"))
    anchors, ident = P.load_anchors()
    assert set(anchors) == {t for t, *_ in TRAIN + DEV} and ident == bb.identity()
    for seed in P.SEEDS:
        R.train_cmd(_args(seed=seed, arms=["T4", "c4_params"], resume=False))
        for arm in P.ARMS:
            path, rec = R.final_checkpoint(P.ROOT / "train" / f"{arm}_s{seed}")
            assert rec["update"] == 3 and path.name.startswith("ckpt_u000003_")
    evals = []
    for seed in P.SEEDS:
        R.eval_cmd(_args(seed=seed))
        evals.append(sorted((P.ROOT / "eval").glob(f"s{seed}_*"))[-1])
        summary = json.loads((evals[-1] / "summary.json").read_text())
        assert summary["states_differ_across_arms"] == []
        for arm in P.ARMS:
            assert summary["arms"][arm]["reconstruction_all_equal"]
            assert summary["arms"][arm]["fresh_states_all_equal_anchor"]
            assert summary["arms"][arm]["summary"]["requested"] == len(DEV)
    l_dir = tmp_path / "L"
    l_dir.mkdir()
    for tid, *_ in DEV:  # a stand-in L run that answers nothing
        rec = {"task_id": tid, "db_id": "cinema", "arm": "L", "config_identity": "c", "input_identity": "i",
               "snapshot_sha256": "0" * 64, "attempts": [], "final": {"kind": "abstain", "sql": None, "reason": "x"},
               "score": {"correct": 0, "scored_sql_sha256": __import__("scaling.records", fromlist=["x"]).sql_sha256(None),
                         "snapshot_sha256": "0" * 64}, "tokens": {}, "controller_work": None,
               "elapsed_seconds": 0.01, "peak_memory_bytes": None, "timed_out": False, "error": None,
               "repairs": 0, "tool_calls": 0, "tool_log": []}
        with open(l_dir / "records.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    R.analyze_cmd(_args(evals=[str(e) for e in evals], l=str(l_dir)))
    out = json.loads(sorted((P.ROOT / "analysis").glob("*/analysis.json"))[-1].read_text())
    assert out["counts"]["questions"] == len(DEV) and out["rule"].startswith("A")
    assert set(out["databases"]) == {"cinema"} and "exploratory_sign_flip" in out


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
    from scaling import identity
    (tmp_path / "pin.json").write_text(json.dumps({"sha256": identity.sha256_file(protocol)}))
    assert R.protocol_identity()["sha256"] == identity.sha256_file(protocol)
    monkeypatch.setattr(R, "protocol_identity", lambda: {"sha256": "f" * 64})
    R.encode_cmd(_args(what="train", range="0:2"))  # tr2 is not encoded
    with pytest.raises(R.PilotStop, match="no encoding anchor"):
        R.train_cmd(_args(seed=100, arms=["T4"], resume=False))
    R.encode_cmd(_args(what="train", range="2:3"))
    R.train_cmd(_args(seed=100, arms=["T4"], resume=False))
    with pytest.raises(R.PilotStop, match="--resume"):
        R.train_cmd(_args(seed=100, arms=["T4"], resume=False))
    with pytest.raises(R.PilotStop, match="not one of"):
        R.train_cmd(_args(seed=7, arms=["T4"], resume=False))
    log = P.ROOT / "train" / "T4_s100" / "train_log.jsonl"
    log.write_text("".join(line + "\n" for line in log.read_text().splitlines()[:-1]))
    with pytest.raises(R.PilotStop, match="does not hold updates"):
        R.final_checkpoint(P.ROOT / "train" / "T4_s100")


def test_the_projection_counts_every_remaining_step():
    m = {"encode": {"seconds_per_new_context": 2.0}, "load_seconds": 60.0,
         "arms": {a: {"update_seconds_after_first": 0.5, "checkpoint_save_seconds_mean": 2.0,
                      "eval_seconds_per_request": 1.2, "reconstruction_seconds_per_request": 0.1} for a in P.ARMS}}
    import scaling.ledger as L
    orig = L.usage
    L.usage = lambda: {"used_hours": 3.0}
    try:
        out = R.project(m, n_pool=4741, n_eval=64, n_sample=160, n_dev=8)
    finally:
        L.usage = orig
    enc = ((4741 - 160) + (64 - 8)) * 2.0 / 3600 + 5 * 60 / 3600
    train = 2 * 2 * (1500 * 0.5 + 6 * 2.0) / 3600
    ev = (2 * 60 + 2 * 2 * 64 * 1.3) / 3600
    assert out["items_hours"]["encode remaining contexts"] == pytest.approx(enc)
    assert out["items_hours"]["train 2 seeds x 2 arms"] == pytest.approx(train)
    assert out["items_hours"]["evaluate and reconstruct 4 final checkpoints"] == pytest.approx(ev)
    assert out["remaining_hours"] == pytest.approx(5.0) and out["fits"] == (enc + train + ev <= 5.0)
