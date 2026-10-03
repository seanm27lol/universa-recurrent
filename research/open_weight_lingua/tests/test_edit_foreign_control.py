import json

import pytest

from open_weight_lingua import answer_slot, edit_foreign_control as ef

TEXT = 'Result: {v}.\n\nFinal token "\n" ends a label, expecting "{v}" or "{v}."'


def _d3_plan():
    rows = []
    for index, (variable, a, c) in enumerate((("x", 7, 8), ("y", 3, 2), ("x", 4, 5), ("y", 12, 11))):
        rows.append({
            "id": f"r{index}", "population": "primary",
            "values": {"a": a, "c": c, "o": 2 * a - c, "d": (a + 10) % 20},
            "conditions": {"E0": {"text": TEXT.format(v=a)}, "E1g": {"text": TEXT.format(v=c)}},
        })
    rows.append({"id": "rx", "population": "excluded", "values": {"a": 1, "c": 2, "o": 0, "d": 11}, "conditions": {}})
    inputs = {r["id"]: {"variable": v} for r, v in zip(rows, ("x", "y", "x", "y", "x"))}
    return rows, inputs


def test_donor_is_the_next_primary_receiver_with_the_same_variable():
    rows, inputs = _d3_plan()
    assert ef.donors(["r0", "r1", "r2", "r3"], inputs) == {"r0": "r2", "r1": "r3", "r2": "r0", "r3": "r1"}
    with pytest.raises(ValueError):
        ef.donors(["r0", "r1"], inputs)  # no other receiver asks the same variable


def test_d5_plan_rewrites_the_donor_to_this_receivers_counterfactual():
    rows, inputs = _d3_plan()
    plan = {e["id"]: e for e in ef.d5_plan(rows, inputs)}
    assert set(plan) == {"r0", "r1", "r2", "r3"}
    entry = plan["r0"]  # a=7, c=8; donor r2 states 4
    assert entry["donor_id"] == "r2" and entry["values"]["donor_a"] == 4
    assert entry["conditions"]["F0"]["text"] == TEXT.format(v=8)
    assert entry["conditions"]["F1"]["text"] == TEXT.format(v=8)  # donor's 4 rewritten to 8
    assert entry["conditions"]["F2"]["text"] == TEXT.format(v=4)
    assert answer_slot.parse(entry["conditions"]["F1"]["text"]).lead == 8
    assert entry["donor_states_c_already"] is False


def _condition(answer, cosine=0.9):
    return {
        "status": "ok",
        "generation": {"text": f"{answer}\n", "terminated": True, "token_ids": []},
        "scores": {"7": -1.0, "8": -1.0 if answer == 8 else -6.0, "6": -6.0, "17": -6.0, "4": -6.0},
        "cosine": cosine,
    }


def _record(f0, f1, f2):
    return {
        "values": {"a": 7, "c": 8, "o": 6, "d": 17, "donor_a": 4},
        "donor_states_c_already": False,
        "d3_E0": _condition(7),
        "conditions": {"F0": _condition(f0), "F1": _condition(f1), "F2": _condition(f2)},
    }


def test_summarize_d5_reads_r5b_when_the_foreign_edit_does_as_well():
    records = [_record(8, 8, 4)] * 5 + [_record(7, 7, 7)] * 5
    summary = ef.summarize_d5(records, {"harness_reproduced": True}, resamples=200, seed=1)
    assert summary["readings"]["R5b_foreign_description_works_as_well"] is True
    assert summary["readings"]["R5a_own_description_matters"] is False
    assert summary["donor_adoption_F2"] == {"receivers": 10, "rate": 0.5}


def test_summarize_d5_reads_r5a_when_only_the_own_description_works():
    records = [_record(8, 7, 7)] * 8 + [_record(7, 7, 7)] * 2
    summary = ef.summarize_d5(records, {"harness_reproduced": False}, resamples=200, seed=1)
    assert summary["readings"]["R5a_own_description_matters"] is True
    assert summary["readings"]["R5b_foreign_description_works_as_well"] is False
    assert summary["readings"]["interpretable"] is False


def test_audit_recounts_and_detects_tampering(tmp_path):
    records = [_record(8, 8, 4)] * 3 + [_record(7, 7, 7)]
    summary = ef.summarize_d5(records, {"harness_reproduced": True}, resamples=50, seed=1)
    (tmp_path / "results.json").write_text(json.dumps(records))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert ef.audit(tmp_path)["status"] == "PASS"
    summary["conditions"]["F1"]["hit_rate_c"] = 0.0
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert ef.audit(tmp_path)["status"] == "FAIL"


def test_d5_is_frozen_for_the_12b_lock_only(tmp_path):
    common = ["--d3-run", str(tmp_path), "--calibration-run", str(tmp_path)]
    with pytest.raises(SystemExit):
        ef.parse_args(["--lock", "configs/model-lock-gemma3-27b.json", *common])
    assert ef.parse_args(["--lock", "configs/model-lock-gemma3-12b.json", *common]).d3_run == tmp_path


def test_d5_end_to_end_replays_d3_e1g_bitwise_on_fixtures(
    tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
):
    """Fixture chain D1 -> D3 -> D5; the fixture pilot doubles as the calibration run."""
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
        monkeypatch.setattr(ef, name, getattr(ed, name))
    d3 = RunDirectory(tmp_path / "d3")
    d3_records, d3_summary = er.run_d3(
        er.parse_args(["--lock", str(args.lock), "--calibration-run", str(args.pilot_run), "--pilot-run",
                       str(args.pilot_run), "--d1-run", str(d1.path), "--device", "cpu"]),
        d3, {},
    )
    write_json(d3.path / "results.json", d3_records)
    write_json(d3.path / "summary.json", d3_summary)
    run = RunDirectory(tmp_path / "d5")
    records, summary = ef.run_d5(
        ef.parse_args(["--lock", str(args.lock), "--d3-run", str(d3.path), "--calibration-run",
                       str(args.pilot_run), "--device", "cpu"]),
        run, {},
    )
    assert summary["receivers"] == len(records) == d3_summary["populations"]["primary"] > 1
    assert summary["gates"]["harness_reproduced"] is True
    assert all(r["donor_id"] != r["id"] for r in records)
    assert json.loads((run.path / "manifest.json").read_text())["format"] == ef.FORMAT
