import json
from pathlib import Path

import numpy as np
import pytest
import torch

from native.config import NativeConfig
from native.data import DataError, Document, TokenRun, TokenizedDocument, pack_runs
from native.model import NativeModel
from native.structured import Instruction, execute, make_example
from native.independent.rescore_generations import ScoringError, execute_saved_program, summarize
from native.prepare_structured_study import prepare
from native.evaluate_state_path import _answer_cross_entropy
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def make_run(doc_id, token_ids, after_state):
    document = Document("structured", f"doc-{doc_id}", f"document {doc_id}")
    length = len(token_ids)
    route_targets = np.zeros((length, 4), dtype=np.int8)
    route_mask = np.zeros((length, 4), dtype=np.bool_)
    state_targets = np.zeros((length, 4, 4), dtype=np.int16)
    state_targets[-1, 2] = after_state
    state_mask = np.zeros((length, 4, 4), dtype=np.bool_)
    state_mask[-1, 2] = True
    tokenized = TokenizedDocument(
        document,
        np.asarray(token_ids, dtype=np.int32),
        route_targets,
        route_mask,
        state_targets,
        state_mask,
    )
    return TokenRun(doc_id, tokenized, length)


def test_boundary_counterexample_has_identical_suffix_and_different_required_states():
    fixture = json.loads(
        (Path(__file__).parents[1] / "research/ti_tnn/native/fixtures/structured_boundary_counterexample.json").read_text()
    )
    runs = [make_run(item["doc_id"], item["input_ids"], item["required_after_state"]) for item in fixture["examples"]]
    legacy, _ = pack_runs(runs, context_length=fixture["context_length"])
    assert np.array_equal(legacy["input_ids"][1], legacy["input_ids"][3])
    assert np.array_equal(legacy["input_ids"][1], fixture["visible_continuation_tokens"])
    assert legacy["labels"][1, 0] == legacy["labels"][3, 0] == -100
    assert legacy["state_targets"][1, 3, 2].tolist() == [1, 2, 3, 4]
    assert legacy["state_targets"][3, 3, 2].tolist() == [5, 6, 7, 8]

    intact, stats = pack_runs(runs, context_length=8, packing="intact")
    assert stats["structured_examples_intact"] == 2
    assert not np.array_equal(intact["input_ids"][0], intact["input_ids"][1])
    assert np.count_nonzero(intact["doc_ids"][0] == 1) == 8
    assert np.count_nonzero(intact["doc_ids"][1] == 2) == 8


def test_intact_builder_moves_fitting_examples_and_audits_long_exclusions():
    runs = [
        make_run(1, [1, 2, 3], [1, 0, 0, 0]),
        make_run(2, [4, 5, 6, 7], [2, 0, 0, 0]),
        make_run(3, [8, 9, 10, 11, 12], [3, 0, 0, 0]),
    ]
    with pytest.raises(DataError, match="longer than context"):
        pack_runs(runs, context_length=4, packing="intact")
    arrays, stats = pack_runs(runs, context_length=4, packing="intact", structured_overflow="exclude")
    assert stats["tokens"] == 7
    assert stats["structured_rows_padded"] == 1
    assert stats["long_examples_excluded"] == 1
    assert stats["long_excluded_tokens"] == 5
    assert stats["long_examples"] == [{"identity": "doc-3", "tokens": 5}]
    assert arrays["doc_ids"][:, 0].tolist() == [1, 2]
    assert np.count_nonzero(arrays["doc_ids"][0] == 1) == 3
    assert np.count_nonzero(arrays["doc_ids"][1] == 2) == 4
    tiny, tiny_stats = pack_runs(
        [make_run(4, [20, 21, 22], [0, 0, 0, 0]), make_run(5, [23, 24, 25], [0, 0, 0, 0]),
         make_run(6, [26, 27, 28], [0, 0, 0, 0])],
        context_length=4, packing="intact",
    )
    assert tiny["input_ids"].shape[0] == 3
    assert tiny_stats["structured_rows_padded"] == 2


def test_after_state_is_placed_before_downstream_read():
    event = execute([Instruction("multiply", 0, src=1)], initial=(2, 3, 4, 5)).events[0]
    assert event.states[0] == event.before
    assert event.states[1:] == (event.after,) * 3


@pytest.mark.parametrize("mode", ["dense", "shared", "untyped", "typed"])
def test_instruction_state_intervention_changes_later_answer_through_cache(mode):
    torch.manual_seed(7)
    model = NativeModel(NativeConfig(
        vocab_size=32, context_length=8, d_model=16, d_slot=8, d_ff=32,
        n_heads=4, mode=mode, n_state_classes=32,
    ))
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    instruction_index, answer_index = 2, 2
    baseline_prefix = model(input_ids[:, :3], use_cache=True)
    baseline_answer = model(input_ids[:, 3:], cache=baseline_prefix.cache).logits
    reader = model.slot_reader if mode != "typed" else model.slot_readers[0]
    calls = [0]

    def mask_instruction_after_state(module, inputs):
        calls[0] += 1
        if calls[0] == (1 if mode == "typed" else 4):
            changed = inputs[0].clone()
            changed[:, instruction_index] = 0
            return (changed,)
        return inputs

    handle = reader.register_forward_pre_hook(mask_instruction_after_state)
    try:
        intervened_prefix = model(input_ids[:, :3], use_cache=True)
    finally:
        handle.remove()
    intervened_answer = model(input_ids[:, 3:], cache=intervened_prefix.cache).logits
    assert calls[0] == (4 if mode == "typed" else 16)
    assert not torch.equal(baseline_answer[:, answer_index], intervened_answer[:, answer_index])


def test_independent_rescore_reports_register_loss_and_complete_prerequisites():
    example = make_example("train", [Instruction("set", 0, value=1)], (0, 0, 0, 0))
    record = example.to_record()
    record.update({
        "completion_text": "r0=0 r1=0 r2=0 r3=0",
        "stopped_on_eos": True,
        "truncated": False,
        "prompt_truncated": False,
    })
    metrics = summarize([record], {example.example_id: example.to_record()})
    assert metrics["exact_count"] == 0
    assert metrics["format_valid_count"] == 1
    assert metrics["answer_value_loss"] == 0.25
    assert metrics["per_register_accuracy"]["0"] == 0.0
    assert metrics["per_register_accuracy"]["1"] == 1.0
    assert metrics["first_errors"][0]["register"] == 0
    assert metrics["prerequisites"]["complete"] == 1
    assert metrics["prerequisites"]["program_source_is_in_prompt"]


def test_independent_scorer_executes_and_rejects_inconsistent_targets():
    example = make_example("train", [Instruction("multiply", 0, src=1)], (2, 3, 4, 5))
    assert execute_saved_program(example.to_record()) == (6, 3, 4, 5)
    invalid = example.to_record()
    invalid["target_text"] = "r0=7 r1=3 r2=4 r3=5"
    with pytest.raises(ScoringError, match="does not execute"):
        execute_saved_program(invalid)
    invalid = example.to_record()
    invalid["program"] = ["add r0 32"]
    with pytest.raises(ScoringError):
        execute_saved_program(invalid)


def test_answer_cross_entropy_is_separate_from_register_error(tmp_path):
    from types import SimpleNamespace
    for split in ("train", "dev", "unseen_composition", "longer_length"):
        (tmp_path / f"{split}.jsonl").write_text(json.dumps({"prompt": "prompt", "target_text": "target"}) + "\n")
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: SimpleNamespace(ids=[1] if text == "prompt" else [2]))
    class FakeModel:
        config = SimpleNamespace(context_length=4)

        def __call__(self, input_ids):
            return SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], 50257))

    model = FakeModel()
    result = _answer_cross_entropy(model, tokenizer, tmp_path)
    for values in result.values():
        assert values["count"] == 2
        assert values["mean"] == pytest.approx(float(torch.log(torch.tensor(50257.0))))


def test_prepare_structured_study_uses_intact_examples_and_audits_exclusions(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "structured").mkdir(parents=True)
    (source / "manifest.json").write_text("{}\n")
    for split in ("train", "dev"):
        examples = [make_example(split, [Instruction("set", 0, value=index) for index in range(8)], (0, 0, 0, 0))]
        examples.extend(make_example(split, [Instruction("set", 0, value=index)], (index, 0, 0, 0))
                        for index in range(10))
        (source / "structured" / f"{split}.jsonl").write_text(
            "".join(json.dumps(example.to_record()) + "\n" for example in examples)
        )
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    texts = [line for path in (source / "structured").glob("*.jsonl") for line in path.read_text().splitlines()]
    tokenizer.train_from_iterator(texts, trainers.WordLevelTrainer(special_tokens=["[UNK]"]))
    monkeypatch.setattr("native.prepare_structured_study.D.load_gpt2_tokenizer", lambda path: tokenizer)
    manifest = prepare(source, tmp_path / "packed", train_tokens=52, dev_tokens=52, context_length=64)
    for split in ("train", "dev"):
        values = manifest["splits"][split]
        assert values["underfill_tokens"] == 1
        assert "no document was split" in values["underfill_reason"]
        assert values["long_examples_excluded"] == 1
        assert values["long_excluded_tokens"] > 0
        doc_ids = np.load(tmp_path / "packed" / split / "doc_ids.npy")
        for row in doc_ids:
            assert len(set(row[row >= 0])) <= 1
