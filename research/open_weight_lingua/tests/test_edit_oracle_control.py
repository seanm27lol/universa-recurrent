import json

import pytest

from open_weight_lingua import edit_oracle_control as eo
from open_weight_lingua.text_edits import parse as frozen_parse


def test_oracle_texts_render_the_frozen_templates():
    texts = eo.oracle_texts("x", 11)
    assert texts["O_terse"] == "The current value of x is 11."
    # A state statement in substance, but the case-sensitive v1.0.0 rule wants a
    # lowercase "the current value of"; the capitalized template is not eligible.
    assert frozen_parse(texts["O_terse"])["x"].status == "absent"
    assert frozen_parse(texts["O_terse"].replace("The", "the"))["x"].status == "eligible"
    assert '"\n"' in texts["O_slot"] and texts["O_slot"].endswith('like "11".')
    assert texts["O_number"] == "A short note that mentions the number 11."
    assert 'strongly expecting "11"' in texts["O_structured"]


def test_d4_plan_keeps_d3_primary_receivers_and_their_e1g_text():
    d3_plan = [
        {"id": "r1", "population": "primary", "values": {"a": 10, "c": 11, "o": 9, "d": 0},
         "conditions": {"E1g": {"text": "edited 11"}}},
        {"id": "r2", "population": "excluded", "values": {"a": 3, "c": 4, "o": 2, "d": 13}, "conditions": {}},
    ]
    (entry,) = eo.d4_plan(d3_plan, {"r1": {"variable": "y"}, "r2": {"variable": "x"}})
    assert entry["id"] == "r1"
    assert set(entry["conditions"]) == {"E1g", *eo.ORACLES}
    assert entry["conditions"]["E1g"]["text"] == "edited 11"
    assert entry["conditions"]["O_terse"]["text"] == "The current value of y is 11."


def _condition(answer):
    return {
        "status": "ok",
        "generation": {"text": f"{answer}\n", "terminated": True, "token_ids": []},
        "scores": {"10": -1.0, "11": -1.0 if answer == 11 else -6.0, "9": -6.0, "0": -6.0},
        "cosine": 0.5,
    }


def _record(e1g, oracle):
    return {
        "values": {"a": 10, "c": 11, "o": 9, "d": 0},
        "d3_E0": _condition(10),
        "conditions": {"E1g": _condition(e1g), **{name: _condition(oracle) for name in eo.ORACLES}},
    }


def test_summarize_d4_reads_r4a_when_the_av_text_beats_every_oracle():
    records = [_record(11, 10)] * 8 + [_record(10, 10)] * 2
    summary = eo.summarize_d4(records, {"harness_reproduced": True}, resamples=200, seed=1)
    assert summary["conditions"]["E1g"]["hit_rate"] == 0.8
    assert summary["readings"]["R4a_av_description_adds_beyond_number"] is True
    assert summary["readings"]["R4b_hand_written_text_moves_answer_as_often"] is False
    assert summary["readings"]["R4c_oracle_moves_answer"] == {name: False for name in eo.ORACLES}


def test_summarize_d4_reads_r4b_when_an_oracle_does_as_well():
    records = [_record(11, 11)] * 5 + [_record(10, 10)] * 5
    summary = eo.summarize_d4(records, {"harness_reproduced": False}, resamples=200, seed=1)
    assert summary["readings"]["R4b_hand_written_text_moves_answer_as_often"] is True
    assert summary["readings"]["R4a_av_description_adds_beyond_number"] is False
    assert summary["readings"]["interpretable"] is False
    assert summary["conditions"]["O_slot"]["off_target_rate"] == pytest.approx(0.0)


def test_audit_recounts_and_detects_tampering(tmp_path):
    records = [_record(11, 10)] * 3 + [_record(10, 11)]
    summary = eo.summarize_d4(records, {"harness_reproduced": True}, resamples=50, seed=1)
    (tmp_path / "results.json").write_text(json.dumps(records))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert eo.audit(tmp_path)["status"] == "PASS"
    summary["conditions"]["O_number"]["hit_rate"] = 0.9
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert eo.audit(tmp_path)["status"] == "FAIL"


def test_d4_is_frozen_for_the_12b_lock_only(tmp_path):
    common = ["--d3-run", str(tmp_path), "--calibration-run", str(tmp_path)]
    with pytest.raises(SystemExit):
        eo.parse_args(["--lock", "configs/model-lock-gemma3-27b.json", *common])
    assert eo.parse_args(["--lock", "configs/model-lock-gemma3-12b.json", *common]).d3_run == tmp_path


def test_d4_end_to_end_replays_d3_e1g_bitwise_on_fixtures(
    tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
):
    """Fixture chain D1 -> D3 -> D4; the fixture pilot doubles as the calibration run."""
    from test_edit_diagnostics import _fixture_pilot
    from test_edit_replication import TEXT
    from test_pilot_stages import _stub_verbalizer

    from open_weight_lingua import edit_diagnostics as ed, edit_replication as er
    from open_weight_lingua.artifacts import RunDirectory, write_json

    args, _ = _fixture_pilot(tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory)
    d1 = RunDirectory(tmp_path / "d1")
    _, d1_records, d1_summary = ed.run_d1(args, d1, {})
    write_json(d1.path / "results.json", d1_records)
    write_json(d1.path / "summary.json", d1_summary)
    inputs = {r["id"]: r for r in json.loads((args.pilot_run / "manifest.json").read_text())["inputs"]}
    monkeypatch.setattr(er, "Verbalizer", _stub_verbalizer([TEXT.format(v=inputs[r]["answer"]) for r in er.receivers(inputs)]))
    for name in ("load_model", "read_lock", "model_paths", "verify_models", "compatibility", "inspect_metadata"):
        monkeypatch.setattr(er, name, getattr(ed, name))
        monkeypatch.setattr(eo, name, getattr(ed, name))
    d3 = RunDirectory(tmp_path / "d3")
    d3_records, d3_summary = er.run_d3(
        er.parse_args(["--lock", str(args.lock), "--calibration-run", str(args.pilot_run), "--pilot-run",
                       str(args.pilot_run), "--d1-run", str(d1.path), "--device", "cpu"]),
        d3, {},
    )
    write_json(d3.path / "results.json", d3_records)
    write_json(d3.path / "summary.json", d3_summary)
    run = RunDirectory(tmp_path / "d4")
    records, summary = eo.run_d4(
        eo.parse_args(["--lock", str(args.lock), "--d3-run", str(d3.path), "--calibration-run",
                       str(args.pilot_run), "--device", "cpu"]),
        run, {},
    )
    assert summary["receivers"] == len(records) == d3_summary["populations"]["primary"] > 0
    assert summary["gates"]["harness_reproduced"] is True
    assert all(set(r["conditions"]) == {"E1g", *eo.ORACLES} for r in records)
    assert json.loads((run.path / "manifest.json").read_text())["oracle_templates"] == eo.ORACLES
