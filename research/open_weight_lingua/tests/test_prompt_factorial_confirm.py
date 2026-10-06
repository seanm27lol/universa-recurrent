import json

import pytest

from open_weight_lingua import prompt_factorial_confirm as pfc
from open_weight_lingua.splits import build_plan


def _tiny_plan():
    from test_pilot_stages import TINY_COUNTS

    return build_plan(TINY_COUNTS)


def test_frozen_plan_hash_is_the_one_the_closed_runs_recorded():
    assert build_plan().plan_hash == pfc.PLAN_HASH


def test_validation_rows_render_four_formats_with_breakdown_labels(monkeypatch):
    plan = _tiny_plan()
    monkeypatch.setattr(pfc, "PLAN_HASH", plan.plan_hash)
    rows = pfc.validation_rows(plan)
    bases = {r["base_id"] for r in rows}
    assert len(rows) == 4 * len(bases) == 4 * 4 * len(plan.groups[pfc.SPLIT])
    assert all(r["group_id"].startswith("validation_a-") for r in rows)
    by_base = {}
    for r in rows:
        by_base.setdefault(r["base_id"], {})[r["condition"]] = r
    example = next(iter(by_base.values()))
    body, v = example["original_after"]["program"], example["original_after"]["variable"]
    assert example["original_after"]["prompt"] == f"{body}\nWhat is {v}? Reply with only the integer."
    assert example["expanded_before"]["prompt"] == f"What is {v} at the end of this program? Reply with only the integer.\n{body}"
    assert set(example) == set(pfc.CONDITIONS)
    last_variable = body.split("\n")[-1].split(" = ")[0]
    assert example["original_after"]["last_line_assigns_asked"] == (last_variable == v)


def test_a_different_plan_is_refused():
    with pytest.raises(ValueError, match="frozen Phase Two plan"):
        pfc.validation_rows(_tiny_plan())


def _fake_run(tmp_path, family, accuracy, groups=40):
    """Synthetic generations: in each group, base prompt k of a format is correct iff k < 4 * accuracy[format]."""
    run = tmp_path / family
    run.mkdir()
    rows = []
    for g in range(groups):
        for k, (side, variable) in enumerate((("A", "x"), ("A", "y"), ("B", "x"), ("B", "y"))):
            for condition in pfc.CONDITIONS:
                correct = (k + g) % 4 < round(4 * accuracy[condition])
                rows.append({"id": f"validation_a-{g:04d}-{side}-{variable}__{condition}", "base_id": f"validation_a-{g:04d}-{side}-{variable}",
                             "group_id": f"validation_a-{g:04d}", "side": side, "variable": variable, "condition": condition,
                             "answer": "5", "other_answer": "6", "text": "5" if correct else "6", "correct": correct, "terminated": True,
                             "answered_other_value": not correct, "last_line_assigns_asked": variable == "y",
                             "asked_last_update_kind": "literal", "asked_last_update_operation": "assign"})
    (run / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "manifest.json").write_text(json.dumps({"family": family, "rows": len(rows), "plan_hash": pfc.PLAN_HASH,
                                                   "protocol_sha256": pfc.sha256_file(pfc.PROTOCOL), "answer_convention": "raw"}))
    (run / "completion.json").write_text(json.dumps({"status": "COMPLETE"}))
    return run


def test_evaluation_verdicts_follow_the_frozen_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(pfc, "RESAMPLES", 2000)
    gemma = _fake_run(tmp_path, "gemma3-12b", {"original_after": 1.0, "original_before": 0.75, "expanded_after": 1.0, "expanded_before": 0.25})
    qwen = _fake_run(tmp_path, "qwen2.5-7b", {"original_after": 1.0, "original_before": 0.5, "expanded_after": 1.0, "expanded_before": 0.5})
    result = pfc.evaluate([gemma, qwen])
    status = {h["id"]: h["status"] for h in result["hypotheses"]}
    assert status == {"G1": "SUPPORTED", "G2": "SUPPORTED", "G3": "SUPPORTED", "Q1": "SUPPORTED", "Q2": "SUPPORTED", "Q3": "SUPPORTED"}
    gemma_summary = result["families"]["gemma3-12b"]
    assert gemma_summary["contrasts"]["interaction"]["point"] == pytest.approx(-0.5)
    assert gemma_summary["cells"]["expanded_before"]["accuracy"] == pytest.approx(0.25)
    assert set(gemma_summary["breakdowns"]) >= {"variable", "last_line_assigns_asked", "asked_last_update_kind", "variable_x_last_writer"}


def test_equivalence_fails_when_the_interval_reaches_the_margin(tmp_path, monkeypatch):
    monkeypatch.setattr(pfc, "RESAMPLES", 2000)
    qwen = _fake_run(tmp_path, "qwen2.5-7b", {"original_after": 1.0, "original_before": 0.5, "expanded_after": 1.0, "expanded_before": 0.25})
    result = pfc.evaluate([qwen])
    by_id = {h["id"]: h for h in result["hypotheses"]}
    assert by_id["Q2"]["status"] == "NOT_SUPPORTED" and by_id["Q1"]["status"] == "SUPPORTED"
    assert by_id["G1"]["status"] == "NOT_RUN"


def test_incomplete_or_rescored_runs_are_refused(tmp_path):
    run = _fake_run(tmp_path, "gemma3-12b", {c: 1.0 for c in pfc.CONDITIONS})
    (run / "completion.json").write_text(json.dumps({"status": "PARTIAL"}))
    with pytest.raises(ValueError, match="incomplete"):
        pfc.evaluate([run])
    (run / "completion.json").write_text(json.dumps({"status": "COMPLETE"}))
    lines = (run / "generations.jsonl").read_text().splitlines()
    tampered = json.loads(lines[0]) | {"correct": False}
    (run / "generations.jsonl").write_text("\n".join([json.dumps(tampered)] + lines[1:]) + "\n")
    with pytest.raises(ValueError, match="rescoring"):
        pfc.evaluate([run])


def test_run_end_to_end_on_a_fixture_model(tmp_path, monkeypatch, model_factory, tokenizer):
    """Tiny random model and fixture tokens; plumbing only, never a real result."""
    from test_pilot_stages import _stub_load_model
    import transformers

    plan = _tiny_plan()
    lock = tmp_path / "model-lock-gemma3-12b.json"
    lock.write_text(json.dumps({"models": {"target": {"revision": "r", "files": {}}}}))
    monkeypatch.setattr(pfc, "PLAN_HASH", plan.plan_hash)
    monkeypatch.setattr(pfc, "build_plan", lambda: plan)
    monkeypatch.setattr(pfc, "load_model", _stub_load_model(model_factory))
    monkeypatch.setattr(pfc, "read_lock", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(pfc, "model_paths", lambda lock, cache: {"target": tmp_path})
    monkeypatch.setattr(pfc, "verify_models", lambda lock, paths: {})
    monkeypatch.setattr(pfc, "compatibility", lambda device: {})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path, **kwargs: tokenizer)
    run_dir = tmp_path / "confirm"
    pfc.main(["run", "--lock", str(lock), "--device", "cpu", "--run-dir", str(run_dir)])
    completion = json.loads((run_dir / "completion.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert completion["status"] == "COMPLETE" and completion["rows_written"] == manifest["rows"] == 4 * 4 * len(plan.groups["validation_a"])
    assert manifest["stimuli_sha256"] == pfc.sha256_file(run_dir / "stimuli.json") and completion["traceback"] is None
    result = pfc.evaluate([run_dir])
    assert {h["id"]: h["status"] for h in result["hypotheses"]}["Q1"] == "NOT_RUN"
