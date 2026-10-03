import json

import pytest

from open_weight_lingua import answer_slot, edit_consistency_dose as ec

TEXT = (
    "Result: 10. The value 10 follows x = 10.\n\n"
    'Final token "\n" ends a label, expecting "10" or "10" or "10."'
)  # 6 mentions of 10: three in the narrative, three quoted slot candidates


def test_mentions_are_standalone_and_in_text_order():
    spans = ec.mention_spans(TEXT + " Also 100 and x10.", 10)
    assert len(spans) == 6
    assert spans == sorted(spans)


def test_doses_rewrite_the_first_fraction_and_ns_spares_the_slot():
    out = ec.dose_texts(TEXT, 10, 11)
    assert (out["mentions"], out["outside_slot"]) == (6, 3)
    texts = out["texts"]
    assert texts["K25"].count("11") == 2  # ceil(0.25 * 6)
    assert texts["K50"].count("11") == 3
    assert texts["K75"].count("11") == 5
    assert texts["K25"].startswith("Result: 11. The value 11")
    assert answer_slot.parse(texts["NS"]).candidates == (10, 10, 10)
    assert "Result: 11. The value 11 follows x = 11." in texts["NS"]


def test_d6_plan_uses_d3_e1g_as_the_full_dose():
    plan = [
        {"id": "r", "population": "primary", "values": {"a": 10, "c": 11, "o": 9, "d": 0},
         "conditions": {"E0": {"text": TEXT}, "E1g": {"text": answer_slot.edit_everywhere(TEXT, 11)}}},
        {"id": "x", "population": "excluded", "values": {}, "conditions": {}},
    ]
    (entry,) = ec.d6_plan(plan)
    assert set(entry["conditions"]) == set(ec.CONDITIONS)
    full = ec.rewrite_spans(TEXT, ec.mention_spans(TEXT, 10), 11)
    assert entry["conditions"]["K100"]["text"] == full


def _condition(answer):
    return {
        "status": "ok",
        "generation": {"text": f"{answer}\n", "terminated": True, "token_ids": []},
        "scores": {"10": -1.0, "11": -1.0 if answer == 11 else -6.0, "9": -6.0, "0": -6.0},
        "cosine": 0.9,
    }


def _record(answers):
    return {
        "values": {"a": 10, "c": 11, "o": 9, "d": 0},
        "d3_E0": _condition(10),
        "d3_E1": _condition(10),
        "conditions": {name: _condition(answer) for name, answer in zip(ec.CONDITIONS, answers)},
    }


def test_summarize_d6_reads_a_dose_response_and_slot_independence():
    #                K100 K25 K50 K75 NS
    records = [_record((11, 10, 11, 11, 11))] * 6 + [_record((10, 10, 10, 10, 10))] * 4
    summary = ec.summarize_d6(records, {"harness_reproduced": True}, resamples=200, seed=1)
    assert summary["conditions"]["K75"]["hit_rate"] == 0.6
    assert summary["readings"]["R6a_more_agreement_more_flips"] is True
    assert summary["readings"]["R6b_slot_not_needed"] is True
    flat = [_record((11, 11, 11, 11, 10))] * 5 + [_record((10, 10, 10, 10, 10))] * 5
    summary = ec.summarize_d6(flat, {"harness_reproduced": False}, resamples=200, seed=1)
    assert summary["readings"] == {
        "R6a_more_agreement_more_flips": False,
        "R6b_slot_not_needed": False,
        "interpretable": False,
    }


def test_audit_recounts_and_detects_tampering(tmp_path):
    records = [_record((11, 10, 11, 11, 10))] * 3 + [_record((10,) * 5)]
    summary = ec.summarize_d6(records, {"harness_reproduced": True}, resamples=50, seed=1)
    (tmp_path / "results.json").write_text(json.dumps(records))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert ec.audit(tmp_path)["status"] == "PASS"
    summary["conditions"]["K50"]["hit_rate"] = 0.0
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert ec.audit(tmp_path)["status"] == "FAIL"


def test_d6_is_frozen_for_the_12b_lock_only(tmp_path):
    common = ["--d3-run", str(tmp_path), "--calibration-run", str(tmp_path)]
    with pytest.raises(SystemExit):
        ec.parse_args(["--lock", "configs/model-lock-gemma3-27b.json", *common])
    assert ec.parse_args(["--lock", "configs/model-lock-gemma3-12b.json", *common]).d3_run == tmp_path


def test_d6_end_to_end_replays_d3_e1g_bitwise_on_fixtures(
    tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
):
    """Fixture chain D1 -> D3 -> D6; the fixture pilot doubles as the calibration run."""
    from test_edit_diagnostics import _fixture_pilot
    from test_edit_replication import TEXT as D3_TEXT
    from test_pilot_stages import _stub_verbalizer

    from open_weight_lingua import edit_diagnostics as ed, edit_replication as er
    from open_weight_lingua.artifacts import RunDirectory, write_json

    args, _ = _fixture_pilot(tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory)
    d1 = RunDirectory(tmp_path / "d1")
    _, d1_records, d1_summary = ed.run_d1(args, d1, {})
    write_json(d1.path / "results.json", d1_records)
    write_json(d1.path / "summary.json", d1_summary)
    inputs = {r["id"]: r for r in json.loads((args.pilot_run / "manifest.json").read_text())["inputs"]}
    monkeypatch.setattr(er, "Verbalizer", _stub_verbalizer([D3_TEXT.format(v=inputs[r]["answer"]) for r in er.receivers(inputs)]))
    for name in ("load_model", "read_lock", "model_paths", "verify_models", "compatibility", "inspect_metadata"):
        monkeypatch.setattr(er, name, getattr(ed, name))
        monkeypatch.setattr(ec, name, getattr(ed, name))
    d3 = RunDirectory(tmp_path / "d3")
    d3_records, d3_summary = er.run_d3(
        er.parse_args(["--lock", str(args.lock), "--calibration-run", str(args.pilot_run), "--pilot-run",
                       str(args.pilot_run), "--d1-run", str(d1.path), "--device", "cpu"]),
        d3, {},
    )
    write_json(d3.path / "results.json", d3_records)
    write_json(d3.path / "summary.json", d3_summary)
    run = RunDirectory(tmp_path / "d6")
    records, summary = ec.run_d6(
        ec.parse_args(["--lock", str(args.lock), "--d3-run", str(d3.path), "--calibration-run",
                       str(args.pilot_run), "--device", "cpu"]),
        run, {},
    )
    assert summary["receivers"] == len(records) == d3_summary["populations"]["primary"] > 0
    assert summary["gates"]["harness_reproduced"] is True
    assert all(set(r["conditions"]) == set(ec.CONDITIONS) for r in records)
