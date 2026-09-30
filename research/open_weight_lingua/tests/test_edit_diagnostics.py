import json

import pytest

from open_weight_lingua import answer_slot, edit_diagnostics as ed

SLOT = 'Structured answer format.\n\nFinal token "\n" ends a label, expecting "{v}" or "{v}."'


def _inputs(a_answer, b_answer, affected="x"):
    other = "y" if affected == "x" else "x"
    rows = {}
    for side, answer in (("A", a_answer), ("B", b_answer)):
        rows[f"g-0000-{side}-{affected}"] = {
            "side": side, "affected": True, "answer": str(answer), "group_id": "g-0000",
            "variable": affected, "prompt": f"{affected} = {answer}\n{other} = 3\nWhat is {affected}?",
        }
        rows[f"g-0000-{side}-{other}"] = {
            "side": side, "affected": False, "answer": "3", "group_id": "g-0000",
            "variable": other, "prompt": f"{affected} = {answer}\n{other} = 3\nWhat is {other}?",
        }
    return rows


def test_requested_values_and_primary_plan_edit_every_condition():
    inputs = _inputs(10, 11)
    plan = ed.d1_plan(inputs, {"g-0000-A-x": SLOT.format(v=10)})
    (entry,) = plan
    assert entry["population"] == "primary"
    assert entry["values"] == {"a": 10, "c": 11, "o": 9, "d": 0}
    assert set(entry["conditions"]) == {"E0", "E1", "E1g", "E2", "E3"}
    for name, value in (("E1", 11), ("E2", 9), ("E3", 0)):
        text = entry["conditions"][name]["text"]
        assert answer_slot.parse(text).candidates == (value, value)
    assert all(len(c["text_sha256"]) == 64 for c in entry["conditions"].values())


def test_out_of_range_neighbour_drops_e2_and_wrong_lead_is_secondary():
    inputs = _inputs(19, 18)
    (entry,) = ed.d1_plan(inputs, {"g-0000-A-x": SLOT.format(v=18)})
    assert entry["values"]["o"] is None and entry["values"]["d"] == 9
    assert entry["population"] == "secondary"
    assert "E2" not in entry["conditions"]


def test_missing_or_ambiguous_slots_are_excluded_without_conditions():
    inputs = _inputs(10, 11)
    ambiguous = 'Final token "\n" likely "10" or "11".'
    for descriptions in ({}, {"g-0000-A-x": ambiguous}):
        (entry,) = ed.d1_plan(inputs, descriptions)
        assert entry["population"] == "excluded" and entry["conditions"] == {}


class CharTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def test_program_end_position_is_the_token_completing_the_program():
    prompt = "x = 11\ny = 3\nWhat is x? Reply with only the integer."
    ids = [ord(ch) for ch in "<user>" + prompt + "<model>\n"]
    position = ed.program_end_position(CharTokenizer(), ids, prompt)
    assert "".join(map(chr, ids[: position + 1])).endswith("y = 3")
    with pytest.raises(ValueError):
        ed.program_end_position(CharTokenizer(), [ord("a")], prompt)


def _record(population, hits, base, gates_ok=True):
    """hits/base: answers under E1 and E0 for a receiver whose a=10, c=11."""
    gates = {g: gates_ok for g in ("G0_logits", "G0_greedy", "G1_direction", "G1_replacement", "G1_greedy")}

    def condition(answer, value):
        return {
            "status": "ok",
            "value": value,
            "generation": {"text": f"{answer}\n", "terminated": True, "token_ids": []},
            "scores": {"10": -1.0, "11": -1.0 if answer == 11 else -5.0, "9": -6.0, "0": -7.0},
        }

    return {
        "population": population,
        "values": {"a": 10, "c": 11, "o": 9, "d": 0},
        "gates": gates,
        "conditions": {
            "E0": condition(base, None),
            "E1": condition(hits, 11),
            "E1g": condition(hits, 11),
            "E2": condition(10, 9),
            "E3": condition(10, 0),
        },
    }


def test_summarize_d1_counts_hits_against_the_unedited_base_rate():
    records = [_record("primary", 11, 10)] * 6 + [_record("primary", 10, 10)] * 4
    summary = ed.summarize_d1(records, resamples=200, seed=1)
    e1 = summary["edits"]["E1"]
    assert (e1["receivers"], e1["hit_rate"], e1["base_rate"]) == (10, 0.6, 0.0)
    assert e1["delta"]["lower"] > 0 and e1["logprob_shift"]["lower"] > 0
    assert summary["readings"]["slot_edit_moves_behavior"] is True
    assert summary["readings"]["number_general"] is False  # E2/E3 never hit
    assert summary["readings"]["comparator_hit_rate_at_least_0.30"] is True
    assert summary["gates"]["harness_reproduced"] is True


def test_failed_condition_counts_as_a_miss_and_failed_gate_blocks_interpretation():
    record = _record("primary", 11, 10, gates_ok=False)
    record["conditions"]["E1"] = {"status": "failed", "error": "boom", "value": 11}
    summary = ed.summarize_d1([record, _record("primary", 11, 10)], resamples=50, seed=1)
    assert summary["edits"]["E1"]["hit_rate"] == 0.5
    assert summary["edits"]["E1"]["logprob_shift_excluded"] == 1
    assert summary["readings"]["interpretable"] is False


def test_census_distinguishes_computed_state_from_echoed_literals():
    prompt = "x = 9\ny = 8\nx = x + 1\nWhat is x? Reply with only the integer."
    stated = ed.census("After the update x is now 10 and y = 8.", prompt, "x")
    assert stated["x"]["computed"] and stated["x"]["bound_true"]
    assert stated["x"]["frozen_status"] == "eligible"
    assert not stated["y"]["computed"] and stated["y"]["bound_true"]
    echoed = ed.census("Code with x = 9 then an increment.", prompt, "x")
    assert not echoed["x"]["bound_true"] and echoed["x"]["bound_any"]
    assert ed.census(None, prompt, "x")["available"] is False


def test_summarize_d2_readings_use_both_floors():
    prompt = "x = 9\ny = 8\nx = x + 1\nWhat is x? Reply with only the integer."
    stated = ed.census("x is now 10", prompt, "x")
    silent = ed.census("Structured code.", prompt, "x")
    summary = ed.summarize_d2([stated] * 32 + [silent] * 96, [silent] * 128)
    assert summary["end_of_program"]["affected_bound_true"] == 32
    assert summary["readings"]["states_variable_state"] is True
    summary = ed.summarize_d2([stated] * 31 + [silent] * 97, [silent] * 128)
    assert summary["readings"]["states_variable_state"] is False


def test_audit_recounts_d1_hits_and_detects_tampering(tmp_path):
    records = [_record("primary", 11, 10)] * 3 + [_record("primary", 10, 10)]
    summary = ed.summarize_d1(records, resamples=50, seed=1)
    (tmp_path / "manifest.json").write_text(json.dumps({"part": "d1"}))
    (tmp_path / "results.json").write_text(json.dumps(records))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert ed.audit(tmp_path)["status"] == "PASS"
    summary["edits"]["E1"]["hit_rate"] = 1.0
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert ed.audit(tmp_path)["status"] == "FAIL"


def test_d2_is_frozen_for_the_12b_lock_only(tmp_path):
    with pytest.raises(SystemExit):
        ed.parse_args(["--part", "d2", "--lock", "configs/model-lock-gemma3-27b.json", "--pilot-run", str(tmp_path)])
    args = ed.parse_args(["--part", "d2", "--lock", "configs/model-lock-gemma3-12b.json", "--pilot-run", str(tmp_path)])
    assert args.part == "d2"


def _fixture_pilot(tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory):
    """A tiny random-model pilot whose receivers carry answer slots (fixtures only)."""
    import torch
    from safetensors.torch import save_file
    from test_pilot_stages import TINY_COUNTS, _inputs, _stub_load_model, _stub_verbalizer

    from open_weight_lingua import runner
    from open_weight_lingua.artifacts import RunDirectory, write_json
    from open_weight_lingua.metrics import answer_tokens
    from open_weight_lingua.splits import build_plan

    plan = build_plan(TINY_COUNTS)
    monkeypatch.setattr(runner, "load_model", _stub_load_model(model_factory))
    calibration = RunDirectory(tmp_path / "calibration")
    runner.execute_calibration(
        calibration, {r: tmp_path for r in ("target", "av", "ar")}, {"target": tokenizer},
        metadata_factory("av"), plan, _inputs(plan.groups["calibration"]), "cpu", {}, {},
    )
    groups = {group.group_id: group for group in plan.groups["pilot"]}
    inputs = _inputs(plan.groups["pilot"])
    for row in inputs:
        row["answer_token_ids_including_eos"] = answer_tokens(tokenizer, row["answer"])
    texts = [
        SLOT.format(v=row["answer"]) if row["side"] == "A" and row["affected"] else "No slot here."
        for row in inputs
    ]
    monkeypatch.setattr(runner, "Verbalizer", _stub_verbalizer(texts))
    save_file({"weight": torch.eye(16)}, tmp_path / "value_head.safetensors")
    run = RunDirectory(tmp_path / "pilot")
    fit, sidecar = runner.load_calibration_inputs(calibration.path)
    rows = runner.execute_pilot(
        run, {r: tmp_path for r in ("target", "av", "ar")}, {r: tokenizer for r in ("target", "av", "ar")},
        metadata_factory("av"), metadata_factory("ar"), inputs, groups, fit, sidecar["median_norm"], "cpu", {}, {},
    )
    lock = tmp_path / "model-lock-gemma3-12b.json"
    lock.write_text("{}")
    assert (run.path / "results.json").is_file()  # execute_pilot writes its own results
    assert len(rows) == len(inputs)
    write_json(
        run.path / "manifest.json",
        {"inputs": inputs, "plan_hash": plan.plan_hash, "answer_convention": "rstrip",
         "model_lock_sha256": ed.sha256_file(lock)},
    )
    monkeypatch.setattr(ed, "load_model", _stub_load_model(model_factory))
    monkeypatch.setattr(ed, "read_lock", lambda path: {"models": {}})
    monkeypatch.setattr(ed, "model_paths", lambda lock, cache: {r: tmp_path for r in ("target", "av", "ar")})
    monkeypatch.setattr(ed, "verify_models", lambda lock, paths: {})
    monkeypatch.setattr(ed, "compatibility", lambda device: {"device": device})
    monkeypatch.setattr(
        ed, "inspect_metadata",
        lambda paths, lock: ({r: tokenizer for r in ("target", "av", "ar")}, metadata_factory("av"), metadata_factory("ar"), {}),
    )
    args = ed.parse_args(["--part", "d1", "--lock", str(lock), "--pilot-run", str(run.path), "--device", "cpu"])
    return args, texts


def test_d1_end_to_end_reproduces_the_pilot_p2_bitwise_on_fixtures(
    tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
):
    from open_weight_lingua.artifacts import RunDirectory, write_json

    args, _ = _fixture_pilot(tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory)
    run = RunDirectory(tmp_path / "d1")
    manifest, records, summary = ed.run_d1(args, run, {})
    executed = [r for r in records if r["population"] != "excluded"]
    assert executed and all(r["population"] == "primary" for r in executed)
    assert all(all(r["gates"].values()) and len(r["gates"]) == 5 for r in executed)
    assert summary["gates"]["harness_reproduced"] is True
    assert all(set(r["conditions"]) >= {"E0", "E1", "E1g", "E3"} for r in executed)
    assert json.loads((run.path / "manifest.json").read_text())["part"] == "d1"
    assert manifest["answer_slot_rule"]["sha256"] == answer_slot.RULE_SHA256
    write_json(run.path / "results.json", records)
    write_json(run.path / "summary.json", summary)
    assert ed.audit(run.path)["status"] == "PASS"


def test_d2_end_to_end_gates_and_census_on_fixtures(
    tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
):
    from test_pilot_stages import _stub_verbalizer

    from open_weight_lingua.artifacts import RunDirectory

    args, texts = _fixture_pilot(tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory)
    args.part = "d2"
    receivers = [text for text in texts if text != "No slot here."]
    monkeypatch.setattr(ed, "Verbalizer", _stub_verbalizer(["x is now 5"] * (len(receivers) * 2)))
    monkeypatch.setattr(ed, "program_end_position", lambda tokenizer, ids, prompt: 1)
    run = RunDirectory(tmp_path / "d2")
    _, records, summary = ed.run_d2(args, run, {})
    assert summary["gates"]["G2_pilot_site_capture"] == len(records) == len(receivers)
    # Fixture rows use unrelated token ids per query, so the shared-prefix
    # equality that real prompts should show cannot hold here; it is recorded.
    assert summary["gates"]["end_capture_equals_other_query"] == 0
    assert summary["gates"]["G3_av_replay"] == summary["gates"]["G3_replayed"] == len(records)
    assert summary["gates"]["harness_reproduced"] is True
    assert summary["end_of_program"]["descriptions"] == len(records)
    assert summary["pilot_site"]["slot_status"]["eligible"] == len(records)
