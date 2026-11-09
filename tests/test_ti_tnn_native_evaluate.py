"""Tiny offline checks for native evaluation; none is learning evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
from native import evaluate as E  # noqa: E402
from native.checkpoint import save_checkpoint  # noqa: E402
from native.config import NativeConfig  # noqa: E402
from native.model import NativeCache, NativeModel  # noqa: E402
from native.structured import Instruction, make_example  # noqa: E402


def tiny_model(mode="typed"):
    torch.manual_seed(91)
    return NativeModel(NativeConfig(
        vocab_size=32,
        context_length=32,
        d_model=16,
        n_heads=4,
        d_slot=8,
        d_ff=32,
        mode=mode,
        n_state_classes=5,
    )).eval()


def packed_batch():
    arrays = {
        "input_ids": np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32),
        "labels": np.asarray([[-100, 2, 3, -100], [-100, 6, 7, 8]], dtype=np.int32),
        "source_ids": np.asarray([[0, 0, 1, -1], [2, 2, 2, 2]], dtype=np.int8),
        "positions": np.asarray([[0, 1, 2, 0], [0, 1, 2, 3]], dtype=np.int32),
        "doc_ids": np.asarray([[4, 4, 4, -1], [7, 7, 7, 7]], dtype=np.int32),
        "key_padding_mask": np.asarray([[False, False, False, True], [False] * 4]),
        "route_targets": np.zeros((2, 4, 4), dtype=np.int8),
        "route_mask": np.zeros((2, 4, 4), dtype=np.bool_),
        "state_targets": np.zeros((2, 4, 4, 4), dtype=np.int16),
        "state_mask": np.zeros((2, 4, 4, 4), dtype=np.bool_),
    }
    arrays["route_targets"][1, 1] = np.arange(4)
    arrays["route_mask"][1, 1] = True
    arrays["state_targets"][1, 1] = np.arange(4)[None, :] % 5
    arrays["state_mask"][1, 1] = True
    return arrays


def test_teacher_forcing_separates_target_source_losses_and_collects_routing():
    model = tiny_model()
    data = packed_batch()
    report = E.teacher_forced_evaluate(model, data, batch_size=1)
    assert report["complete"]
    assert report["sources"]["wikipedia"]["loss_tokens"] == 1
    assert report["sources"]["code"]["loss_tokens"] == 1
    assert report["sources"]["structured"]["loss_tokens"] == 3
    assert report["overall"]["loss_tokens"] == 5
    assert report["auxiliary_losses"]["route"]["supervised_values"] == 4
    assert report["auxiliary_losses"]["state"]["supervised_values"] == 16

    with torch.no_grad():
        inputs = {name: torch.from_numpy(data[name]).long() if name != "key_padding_mask"
                  else torch.from_numpy(data[name])
                  for name in E.MODEL_INPUT_ARRAYS}
        output = model(**inputs)
    shifted_logits = output.logits[:, :-1]
    shifted_labels = torch.from_numpy(data["labels"]).long()[:, 1:]
    shifted_sources = torch.from_numpy(data["source_ids"]).long()[:, 1:]
    for name, source_id in E.SOURCE_IDS.items():
        mask = shifted_labels.ne(-100) & shifted_sources.eq(source_id)
        expected = F.cross_entropy(shifted_logits[mask].float(), shifted_labels[mask])
        assert report["sources"][name]["loss"] == pytest.approx(float(expected), abs=1e-7)

    routing = report["routing"]
    assert routing["status"] == "ok"
    assert routing["routing_decisions"] == 7 * 4
    assert routing["starvation"]["pass_expert_cells"] == 16
    for values in routing["per_pass"].values():
        assert values["decisions_per_expert_bank"] == 7
        assert sum(values["selected_counts"].values()) == 7
        assert sum(values["selected_fractions"].values()) == pytest.approx(1.0)
        assert sum(values["mean_probabilities"].values()) == pytest.approx(1.0)
        assert values["mean_entropy_nats"] >= 0
        assert 0 <= values["mean_normalized_entropy"] <= 1
        assert values["load_balance_4sum_fP"] > 0


def test_checkpoint_and_real_packed_npy_teacher_forced_pipeline(tmp_path, capsys):
    packed = tmp_path / "packed"
    packed.mkdir()
    for name, values in packed_batch().items():
        np.save(packed / f"{name}.npy", values, allow_pickle=False)
    model = tiny_model()
    descriptor = save_checkpoint(
        tmp_path / "tiny.pt", model, step=0, config=model.config.to_dict(), code_identity="evaluate-test"
    )
    restored, checkpoint = E.load_native_model(
        descriptor["path"], expected_sha256=descriptor["sha256"]
    )
    assert checkpoint["sha256"] == descriptor["sha256"]
    report = E.teacher_forced_evaluate(restored, E.PackedDataset(packed), batch_size=2)
    assert report["complete"] and report["overall"]["loss_tokens"] == 5
    assert set(report["sources"]) == {"wikipedia", "code", "structured"}

    output = tmp_path / "evaluation.json"
    status = E.main([
        "--checkpoint", descriptor["path"], "--expected-sha256", descriptor["sha256"],
        "--packed-dev", str(packed), "--output", str(output), "--batch-size", "2",
    ])
    assert status == 0
    assert json.loads(capsys.readouterr().out)["teacher_forced"] == "complete"
    cli_report = json.loads(output.read_text())
    assert cli_report["teacher_forced"]["overall"]["loss_tokens"] == 5
    assert cli_report["structured_generation"]["status"] == "not_run"
    assert cli_report["code_generation"]["status"] == "not_run"

    partial_output = tmp_path / "partial-evaluation.json"
    assert E.main([
        "--checkpoint", descriptor["path"], "--packed-dev", str(packed),
        "--output", str(partial_output), "--batch-size", "1", "--max-batches", "1",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["teacher_forced"] == "partial"
    assert not json.loads(partial_output.read_text())["teacher_forced"]["complete"]


def test_routing_starvation_entropy_and_load_have_known_control_values():
    indices = torch.ones(1, 2, 4, dtype=torch.long)
    probabilities = torch.zeros(1, 2, 4, 4)
    probabilities[..., 1] = 1.0
    accumulator = E.RoutingAccumulator()
    accumulator.update(
        SimpleNamespace(route_indices=indices, route_probs=probabilities),
        torch.ones(1, 2, dtype=torch.bool),
    )
    report = accumulator.report()
    assert report["routing_decisions"] == 8
    assert report["starvation"]["zero_selection_cells"] == 12
    assert report["starvation"]["zero_selection_fraction"] == 0.75
    for values in report["per_pass"].values():
        assert values["starved_experts"] == [0, 2, 3]
        assert values["selected_fractions"] == {"0": 0.0, "1": 1.0, "2": 0.0, "3": 0.0}
        assert values["mean_entropy_nats"] == 0.0
        assert values["load_balance_4sum_fP"] == 4.0


def test_teacher_forcing_requires_source_ids_and_marks_dense_routing_not_applicable():
    data = packed_batch()
    del data["source_ids"]
    with pytest.raises(ValueError, match="lack source_ids"):
        E.teacher_forced_evaluate(tiny_model(), data, batch_size=2)
    report = E.teacher_forced_evaluate(tiny_model("dense"), packed_batch(), batch_size=2)
    assert report["routing"]["status"] == "not_applicable"
    assert report["auxiliary_losses"]["route"]["loss"] is None
    assert report["auxiliary_losses"]["state"]["supervised_values"] == 16


def test_independent_strict_register_parser_rejects_partial_or_decorated_text():
    assert E.parse_final_registers("r0=0 r1=2 r2=31 r3=7") == (0, 2, 31, 7)
    invalid = (
        "r0=0 r1=2 r2=3",
        "r0=0 r1=2 r2=3 r3=4 extra",
        " r0=0 r1=2 r2=3 r3=4",
        "r0=0 r1=2 r2=3 r3=4\n",
        "r1=0 r0=2 r2=3 r3=4",
        "r0=00 r1=2 r2=3 r3=4",
        "r0=-1 r1=2 r2=3 r3=4",
        "r0=32 r1=2 r2=3 r3=4",
        "r0=0  r1=2 r2=3 r3=4",
    )
    assert all(E.parse_final_registers(text) is None for text in invalid)


def structured_records():
    return {
        "train": [make_example("train", [Instruction("set", 0, value=1)], (0, 0, 0, 0)).to_record()],
        "dev": [make_example("dev", [Instruction("set", 0, value=2)], (0, 0, 0, 0)).to_record()],
        "unseen_composition": [make_example(
            "unseen_composition",
            [Instruction("multiply", 0, src=1), Instruction("add", 0, src=1)],
            (3, 2, 0, 0),
        ).to_record()],
        "longer_length": [make_example(
            "longer_length", [Instruction("add", 0, src=1)] * 6, (1, 1, 0, 0)
        ).to_record()],
    }


def write_records(directory, records):
    directory.mkdir(parents=True, exist_ok=True)
    for split, split_records in records.items():
        (directory / f"{split}.jsonl").write_text("".join(
            json.dumps(record) + "\n" for record in split_records
        ))


class ScriptedTokenizer:
    """Offline fixture with pre-scripted token chunks, not a trained tokenizer."""

    def __init__(self, records, eos_token_id=3):
        self.eos_token_id = eos_token_id
        self.prompt_ids = {}
        self.pieces = {}
        self.transitions = {}
        next_id = 10
        for record in records:
            prompt_id = next_id
            next_id += 1
            self.prompt_ids[record["prompt"]] = prompt_id
            previous = prompt_id
            target = record["target_text"]
            if target.startswith("r0="):
                fields = target.split(" ")
                chunks = [f"{field} " for field in fields[:-1]] + [fields[-1]]
            else:
                chunks = [target]
            for chunk in chunks:
                token = next_id
                next_id += 1
                self.pieces[token] = chunk
                self.transitions[previous] = token
                previous = token
            self.transitions[previous] = eos_token_id
        self.vocab_size = next_id

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return SimpleNamespace(ids=[self.prompt_ids[text]], offsets=[(0, len(text))])

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces[token] for token in ids)


class ScriptedModel:
    """Prompt-to-suffix fixture; it has no learned parameters or training claim."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.config = SimpleNamespace(context_length=64)
        self.calls = []

    def __call__(self, input_ids, **kwargs):
        return self.forward(input_ids, **kwargs)

    def forward(self, input_ids, *, positions=None, doc_ids=None, key_padding_mask=None,
                cache=None, use_cache=False, **kwargs):
        self.calls.append(dict(kwargs))
        logits = torch.full((*input_ids.shape, self.tokenizer.vocab_size), -10.0)
        for batch in range(input_ids.shape[0]):
            for token in range(input_ids.shape[1]):
                following = self.tokenizer.transitions[int(input_ids[batch, token])]
                logits[batch, token, following] = 10.0
        new_cache = None
        if use_cache:
            if cache is None:
                all_positions = positions
                all_docs = doc_ids
                all_mask = key_padding_mask
            else:
                all_positions = torch.cat((cache.positions, positions), dim=1)
                all_docs = torch.cat((cache.doc_ids, doc_ids), dim=1)
                all_mask = torch.cat((cache.key_padding_mask, key_padding_mask), dim=1)
            length = all_positions.shape[1]
            keys = [torch.zeros(input_ids.shape[0], 1, length, 1) for _ in range(4)]
            new_cache = NativeCache(keys, [key.clone() for key in keys], all_positions, all_docs, all_mask)
        return SimpleNamespace(logits=logits, cache=new_cache)


def scripted_fixture(extra_records=()):
    splits = structured_records()
    flat = [record for split_records in splits.values() for record in split_records]
    flat.extend(extra_records)
    tokenizer = ScriptedTokenizer(flat)
    return ScriptedModel(tokenizer), tokenizer, splits


@pytest.mark.parametrize("use_cache", [False, True])
def test_complete_structured_generation_and_baselines_are_separate(tmp_path, use_cache):
    model, tokenizer, splits = scripted_fixture()
    write_records(tmp_path, splits)
    report = E.evaluate_structured_with_baselines(
        model, tokenizer, tmp_path, max_new_tokens=8, eos_token_id=3, use_cache=use_cache
    )
    for split, values in report["splits"].items():
        assert values["status"] == "complete"
        assert values["examples"] == values["evaluated"] == 1
        assert values["exact_match"] == values["valid_rate"] == 1.0
        assert values["eos_rate"] == 1.0
        assert values["truncated"] == 0
        assert not values["target_length_used_for_decoding"]
        assert values["records"][0]["first_divergence"] is None
    assert report["baselines"]["exact_lookup"]["train"]["exact_match"] == 1.0
    for split in ("dev", "unseen_composition", "longer_length"):
        assert report["baselines"]["exact_lookup"][split]["exact_match"] == 0.0
        assert report["baselines"]["exact_lookup"][split]["missing"] == 1
    for split in E.SPLIT_NAMES:
        assert report["baselines"]["supplied_program_executor"][split]["exact_match"] == 1.0
    assert "not learned composition" in report["baselines"]["boundary"]
    forbidden = {"program", "trace", "route_targets", "state_targets", "target_text", "labels"}
    assert all(forbidden.isdisjoint(call) for call in model.calls)


def test_code_suffix_generation_is_bounded_separate_and_absence_is_not_run(tmp_path):
    code = {"example_id": "code-1", "prompt": "def answer():", "target_text": "return 4\n"}
    model, tokenizer, _ = scripted_fixture([code])
    missing = E.evaluate_code_generation(
        model, tokenizer, tmp_path / "code_generation.jsonl", max_new_tokens=4, eos_token_id=3
    )
    assert missing == {
        "status": "not_run",
        "reason": "code_generation.jsonl is absent",
        "path": str(tmp_path / "code_generation.jsonl"),
        "fabricated_results": False,
    }
    path = tmp_path / "code_generation.jsonl"
    path.write_text(json.dumps(code) + "\n")
    report = E.evaluate_code_generation(model, tokenizer, path, max_new_tokens=4, eos_token_id=3)
    assert report["status"] == "complete"
    assert report["exact_match"] == 1.0
    assert report["max_new_tokens"] == 4
    assert report["separate_from_structured_metrics"]
    assert report["records"][0]["completion_text"] == "return 4\n"


def test_keep_last_prompt_policy_is_explicit_and_structured_default_still_errors():
    model, tokenizer, _ = scripted_fixture()
    prompt = next(iter(tokenizer.prompt_ids))
    prompt_id = tokenizer.prompt_ids[prompt]

    def long_encode(text, add_special_tokens=False):
        return SimpleNamespace(ids=[prompt_id] * 6, offsets=[(0, len(text))] * 6)

    tokenizer.encode = long_encode
    model.config.context_length = 6
    with pytest.raises(ValueError, match="exceeds context_length"):
        E.generate_prompt(model, tokenizer, prompt, max_new_tokens=2, eos_token_id=3)
    generated = E.generate_prompt(
        model, tokenizer, prompt, max_new_tokens=2, eos_token_id=3, prompt_policy="keep_last"
    )
    assert generated["original_prompt_tokens"] == 6
    assert len(generated["prompt_ids"]) == 4
    assert generated["positions"] == [0, 1, 2, 3]
    assert generated["prompt_truncated"]
    assert generated["prompt_policy"] == "keep_last"


class FixedPromptTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[1, 2], offsets=[(0, 1), (1, 2)])

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(token) for token in ids)


def test_zero_shuffle_and_wrong_slot_controls_measure_logits_and_generation():
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    inputs = {
        "input_ids": ids,
        "positions": torch.arange(3).expand(2, 3),
        "doc_ids": torch.zeros_like(ids),
        "key_padding_mask": torch.zeros_like(ids, dtype=torch.bool),
    }
    prompts = [{"example_id": "fixed", "prompt": "prompt only"}]
    specs = [
        E.InterventionSpec(1, 0, "zero"),
        E.InterventionSpec(1, 0, "shuffle"),
        E.InterventionSpec(1, 0, "wrong_slot", source_slot=1),
    ]
    report = E.evaluate_state_interventions(
        model, inputs, prompts, specs, FixedPromptTokenizer(), max_new_tokens=2, eos_token_id=7
    )
    assert [item["kind"] for item in report["interventions"]] == ["zero", "shuffle", "wrong_slot"]
    assert report["interventions"][0]["logit_effect"]["l2"] > 0
    assert report["interventions"][2]["logit_effect"]["l2"] > 0
    for item in report["interventions"]:
        assert item["generation_effect"]["examples"] == 1
        assert item["generation_effect"]["changed_examples"] in (0, 1)
        assert "first_divergence" in item["generation_effect"]["records"][0]
    assert "_read_slots" not in model.__dict__ and "_write_slots" not in model.__dict__
    limits = " ".join(report["limits"]).lower()
    assert "h attention" in limits and "unselected slots" in limits and "off distribution" in limits
    assert "single-row" in limits
