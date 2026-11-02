"""The performance study's commands end to end on the CPU, with the pilot test's tiny backbone and synthetic database
(no downloaded data, no GPU, no ledger):

    the pilot's encodings (anchors) -> perf-encode the new population (with the cross-process recheck)
    -> perf-feasibility (the paired smoke, all four arms) -> perf-train (four arms, one seed)
    -> perf-eval (timed + reconstruction, both populations) -> selection

**What it checks:**
- identical supervision and data order: every arm's configuration lists the same training ids, and every update
  the same examples, across groundings and architectures;
- the four arms' configurations differ only in arm, grounding, the c4_params width and the parameter report;
- v2 runs record the relational grammar; v1 runs record the pilot's;
- the evaluation inventory is both populations in pin order, for every arm, and every reconstruction holds;
- refusals: an unfrozen protocol, a missing anchor, a recomputed context that differs, a run directory without
  `--resume`, an incomplete log at evaluation, a false reconstruction entry, two passing evaluations for a seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "ti_tnn"))
torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_5")
pytest.importorskip("sqlglot")
from scaling import contracts as C  # noqa: E402
from scaling import perf as PF  # noqa: E402
from scaling import pilot as P  # noqa: E402
from scaling import pilot_run as R  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ti_tnn_scaling_pilot_run as TP  # noqa: E402
from test_ti_tnn_scaling_pilot_run import world  # noqa: E402,F401  (the pilot test's fixture)

NEW = [(f"nw{i}", f"{q} (new {i})", e, sql) for i, (q, e, sql) in enumerate(TP.QUESTIONS)]
PERF_PROTO = {"file": "PROTOCOL_perf_v2.md", "sha256": "e" * 64, "updates": 4}


@pytest.fixture()
def perf_world(world, monkeypatch):
    from scaling import data
    tmp_path, bb = world
    tasks = {tid: C.Task(task_id=tid, db_id="cinema", question=q, evidence=e,
                         split="train" if tid.startswith("tr") else "dev")
             for tid, q, e, _ in TP.TRAIN + TP.DEV + NEW}
    golds = {tid: C.Gold(task_id=tid, sql=sql) for tid, _, _, sql in TP.TRAIN + TP.DEV + NEW}
    monkeypatch.setattr(data, "task", lambda tid: tasks[tid])
    monkeypatch.setattr(data, "gold", lambda tid: golds[tid])
    monkeypatch.setattr(PF, "ROOT", C.RUNS / "perf_v2")
    monkeypatch.setattr(PF, "DEVICE", "cpu")
    monkeypatch.setattr(PF, "SEEDS", (100,))
    monkeypatch.setattr(PF, "CHECKPOINT_EVERY", 2)
    monkeypatch.setattr(PF, "MATMUL_PRECISION", "highest")
    monkeypatch.setattr(PF, "RECHECK", 2)
    monkeypatch.setattr(PF, "protocol_identity", lambda: PERF_PROTO)
    monkeypatch.setattr(PF, "updates", lambda: 4)
    monkeypatch.setattr(PF, "train_ids", lambda: [t for t, *_ in TP.TRAIN])
    pops = {"historical": [t for t, *_ in TP.DEV], "new": [t for t, *_ in NEW]}
    monkeypatch.setattr(PF, "eval_ids", lambda population=None: pops[population] if population else
                        pops["historical"] + pops["new"])
    # the pilot's encodings: the anchors this study reuses
    R.encode_cmd(TP._args(what="dev", range=None))
    for chunk in R.TRAIN_CHUNKS:
        R.encode_cmd(TP._args(what="train", range=chunk))
    return tmp_path, bb


def _args(**kw):
    return argparse.Namespace(allow_dirty=True, projected_hours=0.1, **kw)


def _trained(monkeypatch):
    PF.encode_cmd(_args())
    PF.feasibility_cmd(_args(sample=2, decode=2, updates=3, seed=999))
    PF.train_cmd(_args(seed=100, arms=None, resume=False))


def test_the_study_runs_end_to_end_with_identical_supervision(perf_world, monkeypatch):
    _trained(monkeypatch)
    configs = {PF.label(*a): json.loads((PF.run_dir_of(*a, 100) / "config.json").read_text()) for a in PF.ARMS}
    ids = {n: c["examples"] for n, c in configs.items()}
    assert len({json.dumps(v) for v in ids.values()}) == 1                  # identical training ids, same order
    logs = {n: [json.loads(x)["examples"] for x in (PF.run_dir_of(*a, 100) / "train_log.jsonl").read_text()
                .splitlines()] for n, a in zip(ids, PF.ARMS)}
    assert len({json.dumps(v) for v in logs.values()}) == 1                 # every update, the same examples
    differ = {}
    for n, c in configs.items():
        for k in c:
            if json.dumps(c[k], sort_keys=True) != json.dumps(configs["T4-v1"][k], sort_keys=True):
                differ.setdefault(n, set()).add(k)
    assert differ.get("T4-v1") is None
    for n, keys in differ.items():
        assert keys <= {"controller", "parameters", "grammar", "label", "train"}, (n, keys)
        train = {k: v for k, v in configs[n]["train"].items() if k != "arm"}
        assert train == {k: v for k, v in configs["T4-v1"]["train"].items() if k != "arm"}  # the recipe
    for n, c in configs.items():
        ctrl = {k: v for k, v in c["controller"].items() if k not in ("arm", "grounding", "c4_params_hidden")}
        base = {k: v for k, v in configs["T4-v1"]["controller"].items()
                if k not in ("arm", "grounding", "c4_params_hidden")}
        assert ctrl == base
    assert "grounding" not in configs["T4-v1"]["grammar"] and "grounding" not in configs["C4-v1"]["grammar"]
    assert configs["T4-v2"]["grammar"]["grounding"].endswith("+relational-grounding-v1")
    PF.eval_cmd(_args(seed=100))
    chosen, runs = PF.select_evaluations()
    assert list(chosen) == [100] and len(runs) == 1 and runs[0]["passed"]
    summary = json.loads((chosen[100] / "summary.json").read_text())
    want = PF.eval_ids()
    for n, o in summary["arms"].items():
        assert [r["task_id"] for r in o["reconstruction"]] == want            # the inventory, in pin order
        recs = [json.loads(x) for x in (chosen[100] / n / "records.jsonl").read_text().splitlines()]
        assert [r["task_id"] for r in recs] == want and {r["label"] for r in recs} == {n}
        assert {r["grounding"] for r in recs} == {o["grounding"]}
        assert {r["controller_work"]["grounding"] for r in recs} == {o["grounding"]}
    feas = json.loads(next((PF.ROOT / "feasibility").glob("*/summary.json")).read_text())
    assert set(feas["arms"]) == {PF.label(*a) for a in PF.ARMS}
    for a in feas["arms"].values():
        assert a["grad_norm_finite"] and a["checkpoint_reload_digest_equal"] and a["reconstruction_equal"]
    p = {n: a["parameters"] for n, a in feas["arms"].items()}
    assert p["T4-v1"]["trunk"] == p["C4-v1"]["trunk"] < p["T4-v2"]["trunk"] == p["C4-v2"]["trunk"]


def test_refusals(perf_world, monkeypatch):
    from scaling import encode
    with pytest.raises(PF.PerfStop, match="exists"):
        PF.encode_cmd(_args())
        PF.train_cmd(_args(seed=100, arms=["T4-v1"], resume=False))
        PF.train_cmd(_args(seed=100, arms=["T4-v1"], resume=False))
    with pytest.raises(PF.PerfStop, match="not one of"):
        PF.train_cmd(_args(seed=7, arms=None, resume=False))
    with pytest.raises(PF.PerfStop, match="training log"):          # the other arms never trained
        PF.eval_cmd(_args(seed=100))
    monkeypatch.setattr(PF, "train_ids", lambda: [t for t, *_ in TP.TRAIN] + ["tr_missing"])
    with pytest.raises(PF.PerfStop, match="no encoding anchor"):
        PF.prepare(PF.train_ids(), perf_world[1].tokenizer)
    real = encode.states_sha256
    monkeypatch.setattr(encode, "states_sha256", lambda states: "0" * 64)
    with pytest.raises(PF.PerfStop, match="differs from its pilot anchor"):
        PF.encode_cmd(_args())
    monkeypatch.setattr(encode, "states_sha256", real)


def test_an_unfrozen_protocol_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(PF, "PROTOCOL_PIN", tmp_path / "absent.json")
    with pytest.raises(PF.PerfStop, match="not frozen"):
        PF.protocol_identity()


def test_a_false_reconstruction_entry_fails_the_evaluation_and_two_passing_runs_stop(perf_world, monkeypatch):
    _trained(monkeypatch)
    PF.eval_cmd(_args(seed=100))
    (run,) = sorted((PF.ROOT / "eval").glob("s100_*"))
    summary = json.loads((run / "summary.json").read_text())
    ids = PF.eval_ids()
    assert PF.eval_failures(summary, ids) == []
    tampered = json.loads(json.dumps(summary))
    tampered["arms"]["C4-v2"]["reconstruction"][3]["same_actions"] = False
    assert any("not true" in f for f in PF.eval_failures(tampered, ids))
    PF.eval_cmd(_args(seed=100))
    with pytest.raises(PF.PerfStop, match="2 passing"):
        PF.select_evaluations()
