import json

import pytest

from open_weight_lingua import answer_slot, edit_replication as er

TEXT = 'Result: {v}.\n\nFinal token "\n" ends a label, expecting "{v}" or "{v}."'


def _inputs(a, c):
    rows = {}
    for side, answer in (("A", a), ("B", c)):
        rows[f"g-0000-{side}-x"] = {"side": side, "affected": True, "answer": str(answer), "variable": "x"}
        rows[f"g-0000-{side}-y"] = {"side": side, "affected": False, "answer": "3", "variable": "y"}
    return rows


def test_d3_plan_rewrites_every_mention_for_the_everywhere_conditions():
    (entry,) = er.d3_plan(_inputs(10, 11), {"g-0000-A-x": TEXT.format(v=10)})
    assert entry["population"] == "primary"
    assert set(entry["conditions"]) == {"E0", "E1", "E1g", "E2g", "E3g"}
    assert "Result: 10" in entry["conditions"]["E1"]["text"]  # slot-only leaves the repeat
    for name, value in (("E1g", 11), ("E2g", 9), ("E3g", 0)):
        text = entry["conditions"][name]["text"]
        assert f"Result: {value}" in text
        assert answer_slot.parse(text).candidates == (value, value)


def _record(e1g_answer, population="primary"):
    def condition(answer, value):
        return {
            "status": "ok",
            "value": value,
            "generation": {"text": f"{answer}\n", "terminated": True, "token_ids": []},
            "scores": {"10": -1.0, "11": -1.0 if answer == 11 else -6.0, "9": -6.0, "0": -6.0},
        }

    return {
        "population": population,
        "values": {"a": 10, "c": 11, "o": 9, "d": 0},
        "description": {"status": "ok"},
        "conditions": {
            "E0": condition(10, None),
            "E1": condition(10, 11),
            "E1g": condition(e1g_answer, 11),
            "E2g": condition(10, 9),
            "E3g": condition(10, 0),
        },
    }


def test_summarize_d3_readings_follow_the_frozen_definitions():
    records = [_record(11)] * 5 + [_record(10)] * 5 + [_record(10, "excluded")]
    summary = er.summarize_d3(records, {"harness_reproduced": True}, resamples=200, seed=1)
    assert summary["populations"] == {"primary": 10, "secondary": 0, "excluded": 1}
    assert summary["edits"]["E1g"]["hit_rate"] == 0.5
    assert summary["readings"]["R3a_consistent_edit_moves_behavior"] is True
    assert summary["readings"]["R3b_number_general"] is False
    assert summary["readings"]["comparator_E1g_hit_rate_at_least_0.30"] is True
    assert summary["readings"]["interpretable"] is True
    blocked = er.summarize_d3(records, {"harness_reproduced": False}, resamples=50, seed=1)
    assert blocked["readings"]["interpretable"] is False


def test_audit_recounts_and_detects_tampering(tmp_path):
    records = [_record(11)] * 3 + [_record(10)]
    summary = er.summarize_d3(records, {"harness_reproduced": True}, resamples=50, seed=1)
    (tmp_path / "results.json").write_text(json.dumps(records))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert er.audit(tmp_path)["status"] == "PASS"
    summary["edits"]["E1g"]["hit_rate"] = 0.0
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert er.audit(tmp_path)["status"] == "FAIL"


def test_d3_is_frozen_for_the_12b_lock_only(tmp_path):
    common = ["--calibration-run", str(tmp_path), "--pilot-run", str(tmp_path), "--d1-run", str(tmp_path)]
    with pytest.raises(SystemExit):
        er.parse_args(["--lock", "configs/model-lock-gemma3-27b.json", *common])
    assert er.parse_args(["--lock", "configs/model-lock-gemma3-12b.json", *common]).d1_run == tmp_path


def test_d3_end_to_end_with_d1_replay_on_fixtures(
    tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
):
    """Fixture pilot doubles as the calibration run; D1 on it provides the replay records."""
    from test_edit_diagnostics import _fixture_pilot
    from test_pilot_stages import _stub_verbalizer

    from open_weight_lingua import edit_diagnostics as ed
    from open_weight_lingua.artifacts import RunDirectory, write_json

    args, _ = _fixture_pilot(tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory)
    d1_run = RunDirectory(tmp_path / "d1")
    _, d1_records, d1_summary = ed.run_d1(args, d1_run, {})
    write_json(d1_run.path / "results.json", d1_records)
    write_json(d1_run.path / "summary.json", d1_summary)

    inputs = {r["id"]: r for r in json.loads((args.pilot_run / "manifest.json").read_text())["inputs"]}
    texts = [TEXT.format(v=inputs[rid]["answer"]) for rid in er.receivers(inputs)]
    monkeypatch.setattr(er, "Verbalizer", _stub_verbalizer(texts))
    for name in ("load_model", "read_lock", "model_paths", "verify_models", "compatibility", "inspect_metadata"):
        monkeypatch.setattr(er, name, getattr(ed, name))
    d3_args = er.parse_args(
        ["--lock", str(args.lock), "--calibration-run", str(args.pilot_run), "--pilot-run", str(args.pilot_run),
         "--d1-run", str(d1_run.path), "--device", "cpu"]
    )
    run = RunDirectory(tmp_path / "d3")
    records, summary = er.run_d3(d3_args, run, {})
    assert summary["gates"]["capture_equals_calibration_original"] == len(records)
    assert summary["gates"]["d1_replay_bitwise"] == summary["gates"]["d1_replayed"] > 0
    assert summary["gates"]["harness_reproduced"] is True
    assert summary["populations"]["primary"] == len(records)
    assert all(set(r["conditions"]) >= {"E0", "E1", "E1g", "E3g"} for r in records)
    assert json.loads((run.path / "manifest.json").read_text())["format"] == er.FORMAT
