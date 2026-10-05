import json
from types import SimpleNamespace

import numpy as np
import pytest
from open_weight_lingua import prompt_factorial_run as run


def records_fixture():
    records = []
    # Group0: original placement +1, expanded placement -1, interaction -2.
    # Group1: original placement 0, expanded placement +1, interaction +1.
    cells = ((1, 0, 0, 1, 1), (0, 0, 1, 0, 1))
    for group, outcomes in enumerate(cells):
        for side in ("A", "B"):
            for variable in ("x", "y"):
                for condition, correct in zip(run.CONDITIONS, outcomes):
                    records.append({"id": f"{group}-{side}-{variable}-{condition}", "base_id": f"{group}-{side}-{variable}",
                        "group_id": str(group), "condition": condition, "correct": bool(correct), "terminated": True,
                        "variable": variable, "last_line_assigns_asked": variable == "x", "asked_last_update_operation": "add",
                        "asked_last_update_kind": "arithmetic", "last_program_operation": "add", "source_split": "pilot"})
    return records


def test_grouped_contrast_signs_and_bootstrap_preserve_complete_groups():
    summary, arrays = run.grouped_analysis(records_fixture(), "gemma3-12b", repeats=300)
    contrasts = summary["contrasts"]
    assert contrasts["position_original"]["difference"] == .5
    assert contrasts["position_expanded"]["difference"] == 0
    assert contrasts["interaction"]["difference"] == -.5
    assert contrasts["wording_before"]["difference"] == 0
    np.testing.assert_array_equal(arrays["group_contrasts"][:, 4], [-2, 1])
    # Every draw resamples two whole groups: no impossible partial-group value.
    assert set(arrays["bootstrap_contrasts"][:, 4]) <= {-2., -.5, 1.}
    assert contrasts["interaction"]["coverage"] == .995
    assert contrasts["repeat_minus_before"]["coverage"] == .95
    assert summary["strata"]["source_split"]["calibration"]["cells"]["original_before"]["total"] == 0
    assert summary["strata"]["source_split"]["calibration"]["before_minus_after"]["original"] is None


def test_analysis_rejects_missing_and_duplicate_pair_cells():
    records = records_fixture()
    with pytest.raises(ValueError, match="five conditions"):
        run.grouped_analysis(records[:-1], "gemma3-12b", 10)
    with pytest.raises(ValueError, match="duplicate"):
        run.grouped_analysis(records + [records[0]], "gemma3-12b", 10)


class ScriptedTarget:
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[1, 106]))
    tokenizer = SimpleNamespace(eos_token_id=1)

    def __init__(self, fail_at=None):
        self.calls, self.fail_at = 0, fail_at

    def tensors(self, ids, masks):
        return ids, masks

    def greedy(self, ids, mask, site, max_tokens):
        assert max_tokens == 8
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("fixture failure")
        return {"text": "7\n", "token_ids": [7, 106], "terminated": True}


def stimuli_fixture(n=3):
    return {"family": "gemma3-12b", "answer_convention": "rstrip", "expected_generation_eos_token_ids": [1, 106],
            "rows": [{"id": str(index), "input_ids": [2, 3], "attention_mask": [1, 1], "answer_position": 1,
                       "answer": "7"} for index in range(n)]}


def test_incremental_rows_preserve_secondary_eos_and_text_scoring(tmp_path):
    destination = tmp_path / "rows.jsonl"
    stimuli = stimuli_fixture()
    run.generate_rows(ScriptedTarget(), stimuli, destination, 10, clock=lambda: 0)
    records = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(records) == 3 and all(row["correct"] and row["terminated"] for row in records)
    assert all(row["stop_token_id"] == 106 and row["generated_token_count"] == 2 for row in records)
    run.validate_records(stimuli, records)
    records[0]["correct"] = False
    with pytest.raises(ValueError, match="correctness"):
        run.validate_records(stimuli, records)


def test_failure_and_timeout_keep_finished_rows_without_retry(tmp_path):
    path = tmp_path / "failure.jsonl"
    target = ScriptedTarget(fail_at=2)
    with pytest.raises(RuntimeError):
        run.generate_rows(target, stimuli_fixture(), path, 10, clock=lambda: 0)
    assert target.calls == 2 and len(path.read_text().splitlines()) == 1
    values = iter([0, 11])
    path = tmp_path / "timeout.jsonl"
    target = ScriptedTarget()
    with pytest.raises(TimeoutError):
        run.generate_rows(target, stimuli_fixture(), path, 10, clock=lambda: next(values))
    assert target.calls == 1 and len(path.read_text().splitlines()) == 1


def test_shared_campaign_deadline_cannot_reset_between_families(tmp_path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}")
    first = run.campaign_record(tmp_path / "campaign", freeze, now=100)
    second = run.campaign_record(tmp_path / "campaign", freeze, now=1000)
    assert first == second and second["deadline_unix"] == 10900
    with pytest.raises(TimeoutError):
        run.campaign_record(tmp_path / "campaign", freeze, now=10900)


def test_freeze_requires_both_families_before_any_model_load(tmp_path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"format": run.FREEZE_FORMAT, "wall_limit_seconds": 10800,
                                 "family_stimuli": {"gemma3-12b": "missing"}, "files_sha256": {}}))
    with pytest.raises(ValueError, match="both model"):
        run.verify_freeze(freeze)


def test_pairing_order_is_required_even_when_all_ids_are_present(tmp_path):
    destination = tmp_path / "rows.jsonl"
    stimuli = stimuli_fixture()
    run.generate_rows(ScriptedTarget(), stimuli, destination, 10, clock=lambda: 0)
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    with pytest.raises(ValueError, match="pairing"):
        run.validate_records(stimuli, list(reversed(rows)))


def test_run_and_evaluate_end_to_end_with_both_family_freeze(tmp_path, monkeypatch):
    """Exercise real freeze/manifest/completion/JSONL/evaluator logic, with no weights.

    Only input count, external model/tokenizer loading and immutable model-lock
    schema are reduced/stubbed; pairing, source pins and hash checks stay live.
    """
    import transformers
    from open_weight_lingua import prompt_factorial as pf
    from test_prompt_factorial import manifest as source_manifest

    protocol = tmp_path / "protocol.md"
    protocol.write_text("fixture protocol")
    source = tmp_path / "fixture.py"
    source.write_text("# fixture source\n")
    inventory = {path.name: run.sha256_file(path) for path in (protocol, source)}
    monkeypatch.setattr(run, "PROJECT", tmp_path)
    monkeypatch.setattr(run, "PROTOCOL", protocol)
    monkeypatch.setattr(run, "source_identity", lambda: {name: run.sha256_file(tmp_path / name) for name in inventory})
    monkeypatch.setattr(run, "EXPECTED_ROWS", 40)
    monkeypatch.setattr(run, "read_lock", run.read)
    monkeypatch.setattr(run, "verify_models", lambda *args, **kwargs: {"target": 0})
    monkeypatch.setattr(run, "compatibility", lambda device: {"fixture": True})
    monkeypatch.setattr(run, "release_models", lambda: None)

    class Tokenizer:
        eos_token_id = 1

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize and add_generation_prompt
            return [2, 3]

        def decode(self, tokens, skip_special_tokens):
            assert tokens == [7] and skip_special_tokens is False
            return "7\n"

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[1, 106])

        def eval(self):
            return self

    class Target(ScriptedTarget):
        def __init__(self, model, tokenizer):
            super().__init__()
            self.model, self.tokenizer = model, tokenizer

        def forward(self):
            return None

        def greedy(self, *args, **kwargs):
            self.forward()
            return super().greedy(*args, **kwargs)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(run, "load_model", lambda *args: Model())
    monkeypatch.setattr(run, "Target", Target)
    files = {str(path.resolve()): run.sha256_file(path) for path in (protocol, source)}
    pinned, stimulus_paths = {}, {}
    for family in run.SOURCE_INPUTS:
        family_dir = tmp_path / family
        family_dir.mkdir()
        lock_path = family_dir / "lock.json"
        lock_path.write_text(json.dumps({"models": {"target": {"files": {}}}}))
        files[str(lock_path.resolve())] = run.sha256_file(lock_path)
        inputs, originals, pinned[family] = {}, {}, {}
        for split in ("calibration", "pilot"):
            directory = family_dir / split
            directory.mkdir()
            path = directory / "manifest.json"
            originals[split] = source_manifest(split)
            path.write_text(json.dumps(originals[split]))
            digest = run.sha256_file(path)
            files[str(path.resolve())] = digest
            pinned[family][split] = (split, digest)
            inputs[split] = {"path": str(path.resolve()), "sha256": digest}
        rows = pf.build_rows(originals["calibration"], originals["pilot"])
        for row in rows:
            row.update(input_ids=[2, 3], attention_mask=[1, 1], answer_position=1)
        stimuli = {"format": pf.FORMAT, "family": family, "conditions": list(run.CONDITIONS),
                   "max_new_tokens": 8, "target_bucket": 128, "order_seed": run.ORDER_SEED,
                   "answer_convention": run.CONVENTIONS[family], "protocol_sha256": run.sha256_file(protocol),
                   "source_file_sha256": inventory, "source_inputs": inputs,
                   "model_lock_path": str(lock_path.resolve()), "model_lock_sha256": run.sha256_file(lock_path),
                   "tokenizer_path": str(family_dir.resolve()), "expected_generation_eos_token_ids": [1, 106], "rows": rows}
        path = family_dir / "stimuli.json"
        path.write_text(json.dumps(stimuli))
        stimulus_paths[family] = str(path.resolve())
        files[str(path.resolve())] = run.sha256_file(path)
    monkeypatch.setattr(run, "SOURCE_INPUTS", pinned)
    freeze = tmp_path / "launch_freeze.json"
    freeze.write_text(json.dumps({"format": run.FREEZE_FORMAT, "wall_limit_seconds": 10800,
                                 "family_stimuli": stimulus_paths, "files_sha256": files}))
    args = SimpleNamespace(output=tmp_path / "run", campaign=tmp_path / "campaign", freeze=freeze,
                           family="gemma3-12b", device="cpu")
    assert run.run(args) == "COMPLETE"
    completion = run.read(args.output / "completion.json")
    assert completion["completed_rows"] == 40 and completion["progress"]["completed_rows"] == 40
    assert completion["timings_seconds"]["generation"] >= 0
    analysis_args = SimpleNamespace(run=args.output, freeze=freeze, output=tmp_path / "analysis")
    analysis = run.evaluate(analysis_args)
    assert analysis["groups"] == 2 and analysis["base_prompts"] == 8
    assert (analysis_args.output / "bootstrap.npz").exists()
    source.write_text("# changed after freeze\n")
    analysis_args.output = tmp_path / "rejected_analysis"
    with pytest.raises(ValueError, match="frozen artifact changed"):
        run.evaluate(analysis_args)
