"""Whole-prerequisite allocation, fixed capability populations and saved scoring."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ti_tnn"))
from native import data as D, mixed_learning as M, structured as S
from native.evaluate_mixed_learning import score_continuations, generate_continuations


@pytest.fixture
def inputs():
    docs = [D.Document(source, f"{source}-{i}", " ".join(f"word{n}" for n in range(180 + i)))
            for source in ("wikipedia", "code") for i in range(4)]
    examples = [S.make_example("train", (S.Instruction("set", 0, value=i),), (1, 2, 3, 4)) for i in range(3)]
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator([d.text for d in docs] + [e.full_text for e in examples],
                            trainers.WordLevelTrainer(special_tokens=["[UNK]"]))
    return docs, examples, tok


def test_quota_never_cuts_the_last_structured_answer(inputs, tmp_path):
    docs, examples, tok = inputs
    length = len(D.tokenize_structured(examples[0], tok).input_ids)
    runs, audit = M.bounded_runs(docs, examples, tok, {"wikipedia": 5, "code": 5, "structured": length + 1})
    structured = [r for r in runs if r.source == "structured"]
    assert len(structured) == 1 and structured[0].length == length
    assert audit["structured"]["underfill"] == 1
    assert all(not x["partial"] for x in audit["structured"]["documents"])
    stats = M.write_packed(runs, tmp_path)
    arrays = D.PackedDataset(tmp_path).arrays
    locations = np.argwhere(arrays["doc_ids"] == structured[0].doc_id)
    assert len(set(locations[:, 0])) == 1
    row, start = locations[0]
    assert arrays["positions"][row, start:start + length].tolist() == list(range(length))
    assert arrays["input_ids"][row, start:start + length].tolist() == structured[0].tokenized.input_ids.tolist()
    assert arrays["labels"][row, start] == -100
    assert np.all(arrays["labels"][row, start + 1:start + length] != -100)
    assert stats["loss_tokens"] == sum(stats["source_loss_tokens"].values())
    assert not arrays["state_mask"][arrays["source_ids"] != 2].any()
    mask = structured[0].tokenized.state_mask[:, 1, :].all(axis=-1)
    assert np.array_equal(structured[0].tokenized.state_targets[mask, 1],
                          structured[0].tokenized.state_targets[mask, 2])


def test_long_exclusion_and_finite_shortfall_fail_explicitly(inputs):
    docs, examples, tok = inputs
    with pytest.raises(ValueError, match="insufficient finite structured"):
        M.bounded_runs(docs, examples, tok, {"wikipedia": 5, "code": 5, "structured": 1000}, context_length=8)
    with pytest.raises(ValueError, match="insufficient finite wikipedia"):
        M.bounded_runs(docs, examples, tok, {"wikipedia": 100000, "code": 5, "structured": 1})


def test_capability_keeps_document_beginnings_and_separate_rows(inputs, tmp_path):
    docs, _, tok = inputs
    pops = M.capability_records(docs, tok, count=3, generation_count=2)
    report = M.write_capability(pops, tmp_path)
    for source in ("wikipedia", "code"):
        assert report[source]["teacher_forced_rows"] == 3
        arrays = D.PackedDataset(tmp_path / source / "teacher_forced").arrays
        assert len(arrays["input_ids"]) == 3
        assert (arrays["labels"][:, 0] == -100).all()
        assert not arrays["state_mask"].any() and not arrays["route_mask"].any()
        by_id = {d.identity: d for d in docs}
        for ref in pops[source][2]:
            ids = tok.encode(by_id[ref["example_id"]].text).ids
            assert ref["prompt_ids"] + ref["target_ids"] == ids[:160]
        for line in (tmp_path / source / "prompts.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert set(row) == {"example_id", "input_ids"} and len(row["input_ids"]) == 128


def test_capability_refuses_silent_denominator_reduction(inputs):
    docs, _, tok = inputs
    with pytest.raises(ValueError, match="insufficient"):
        M.capability_records(docs, tok, count=10, generation_count=2)


def test_saved_scoring_retains_missing_short_and_malformed_denominators():
    refs = [{"example_id": str(i), "target_ids": [7] * 32} for i in range(4)]
    rows = [{"example_id": "0", "generated_ids": [7] * 32},
            {"example_id": "1", "generated_ids": [7] * 2},
            {"example_id": "2", "generated_ids": [True] * 32}]
    score = score_continuations(refs, rows)
    assert score["examples"] == 4 and score["exact"] == 1 and score["token_correct"] == 34
    assert score["token_decisions"] == 128 and score["missing"] == 1 and score["malformed"] == 1
    assert not score["complete"]
    for attack in (rows + [rows[0]], [{"example_id": "unknown", "generated_ids": []}]):
        with pytest.raises(ValueError, match="identity"):
            score_continuations(refs, attack)


def test_inference_receives_only_prompt_with_common_bound(monkeypatch):
    from native import generation
    seen = []
    def fake(model, ids, **kwargs):
        seen.append((ids.tolist(), kwargs))
        return torch.cat((ids, torch.full((1, 32), 7)), dim=1)
    monkeypatch.setattr(generation, "greedy_generate", fake)
    model = SimpleNamespace(parameters=lambda: iter([torch.tensor(0.)]))
    row = {"example_id": "one", "input_ids": [3] * 128}
    out = generate_continuations(model, [row])
    assert out == [{"example_id": "one", "generated_ids": [7] * 32}]
    assert seen[0][0] == [[3] * 128] and seen[0][1]["max_new_tokens"] == 32
    with pytest.raises(ValueError, match="prompt-only"):
        generate_continuations(model, [{**row, "target_ids": [7] * 32}])


def test_existing_preparation_is_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        M.prepare(tmp_path, tmp_path, tmp_path, tmp_path)


def test_capability_adapter_checks_bytes_and_counts_before_model_calls(inputs, tmp_path, monkeypatch):
    from native import evaluate_mixed_learning as E, evaluate
    docs, _, tok = inputs
    prepared = tmp_path / "prepared"
    M.write_capability(M.capability_records(docs, tok, count=3, generation_count=2), prepared / "capability")
    artifact_hashes = {str(p.relative_to(prepared)): D.sha256_file(p) for p in prepared.rglob('*') if p.is_file()}
    (prepared / "manifest.json").write_text(json.dumps({"artifacts": artifact_hashes}))
    monkeypatch.setattr(E, "CAPABILITY_DOCUMENTS", 3)
    monkeypatch.setattr(E, "GENERATION_DOCUMENTS", 2)
    calls = []
    def fake_generate(model, prompts):
        calls.append(prompts)
        return [{"example_id": p["example_id"], "generated_ids": [1] * 32} for p in prompts]
    monkeypatch.setattr(E, "generate_continuations", fake_generate)
    monkeypatch.setattr(evaluate, "teacher_forced_evaluate", lambda model, data, **kw: {"rows": len(data)})
    report = E.evaluate_capability(None, prepared, tmp_path / "results")
    assert set(report) == {"code", "wikipedia"} and len(calls) == 2
    assert all(r["teacher_forced"]["rows"] == 3 for r in report.values())
    changed = prepared / "capability/code/prompts.jsonl"
    changed.write_text(changed.read_text() + '\n')
    with pytest.raises(D.HashMismatch):
        E.evaluate_capability(None, prepared, tmp_path / "mutated-result")
    assert len(calls) == 2  # no additional generation before the identity failure


@pytest.mark.parametrize('ce_complete,register_complete', [(True, True), (False, True), (True, False)])
def test_endpoint_keeps_capability_diagnostics_and_completeness_separate(tmp_path, monkeypatch,
                                                                      ce_complete, register_complete):
    from native import evaluate_mixed_learning as E, evaluate, evaluate_initial_state_study, ledger
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    (prepared / 'manifest.json').write_text(json.dumps({'artifacts': {}}))
    checkpoint = tmp_path / 'fixture-checkpoint'; checkpoint.write_bytes(b'not model weights')
    tokenizer = tmp_path / 'fixture-tokenizer'; tokenizer.write_bytes(b'fixture')
    monkeypatch.setitem(D.TOKENIZER_SPEC, 'sha256', D.sha256_file(tokenizer))
    monkeypatch.setattr(ledger, 'Reservation', lambda *a, **kw: pytest.fail('evaluation created a reservation'))
    monkeypatch.setattr(evaluate, 'load_native_model', lambda *a, **kw: (object(), {}))
    capability = {s: {'continuation': {'complete': True}, 'teacher_forced': {'complete': ce_complete}}
                  for s in ['wikipedia', 'code']}
    monkeypatch.setattr(E, 'evaluate_capability', lambda *a: capability)
    monkeypatch.setattr(evaluate_initial_state_study, 'evaluate', lambda *a: {'complete': register_complete})
    report = E.evaluate_endpoint({'path': str(checkpoint), 'sha256': D.sha256_file(checkpoint)},
                                 prepared, tokenizer, tmp_path / 'result')
    assert report['complete'] == (ce_complete and register_complete)
    assert report['capability'] == capability
    assert 'not run' in report['causal_interpretation']
