"""Initial-state contrast, leakage attacks, packing and independent saved scoring."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "ti_tnn"))
from native import initial_state_study as study, prepare_initial_state_study as prepare, structured
from native.data import PackedDataset
from native.independent.dependency_diagnostic import analyze_program, composition_contract
from native.independent.score_initial_state_study import score
from native.independent.rescore_generations import ScoringError


@pytest.fixture
def history(tmp_path):
    example = structured.make_example("dev", (structured.Instruction("set", 0, value=9),), (1, 2, 3, 4))
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps(example.to_record()) + "\n")
    return study.read_history([path])


@pytest.fixture
def dataset(history):
    return study.generate(history, seed=23, composed_patterns=64, states_per_pattern=3, evaluation_count=8)


def test_paired_population_is_reproducible_and_has_only_the_declared_contrast(history, dataset):
    second = study.generate(history, seed=23, composed_patterns=64, states_per_pattern=3, evaluation_count=8)
    assert dataset == second
    audit = study.audit(dataset, history)
    assert audit["training_documents_per_condition"] == 384
    assert audit["unique_train_pairs"] == {"repeat_one": 128, "multi_state": 384}
    primitives = {pattern[0] for pattern in dataset["patterns"] if len(pattern) == 1}
    for condition in study.CONDITIONS:
        for example in dataset["training"][condition]:
            assert all(str(i) in primitives for i in example.program)
            assert all(v < 24 for v in example.initial)
            assert structured.RESERVED_COMPOSITION not in structured.composition_pairs(example.program)
        for name in ("program_values_id", "transition_values_id"):
            assert all(v < 24 for e in dataset["evaluations"][condition][name] for v in e.initial)
        for example in dataset["evaluations"][condition]["composition"]:
            assert all(str(i) in primitives for i in example.program)
            last = example.program[-1]
            assert last.op == "add" and example.program[-2].op == "multiply"
            dependency = analyze_program(study.pattern_key(example.program))[last.dst]
            assert len(example.program) - 2 in dependency["relevant_instructions"]
            assert dependency["depends_on_initial"]


def test_no_historical_data_fallback_and_no_missing_exclusion_file(tmp_path):
    with pytest.raises(ValueError, match="required"):
        study.read_history([])
    with pytest.raises(FileNotFoundError):
        study.read_history([tmp_path / "missing.jsonl"])


@pytest.mark.parametrize("program,initial,constant,changed", [
    (("set r0 0", "multiply r0 r1", "add r2 r0"), (2, 3, 4, 5), True, False),
    (("set r1 1", "multiply r0 r1", "add r2 r0"), (2, 3, 4, 5), False, False),
    (("multiply r0 r1", "add r2 r0"), (2, 1, 4, 5), False, False),
    (("multiply r0 r1", "add r2 r0"), (2, 3, 4, 5), False, True),
    (("set r0 2", "multiply r0 r1", "add r2 r0"), (2, 3, 4, 5), False, True),
])
def test_composition_requires_nonconstant_multiply_and_actual_initial_witness(program, initial, constant, changed):
    contract = composition_contract(program, initial)
    assert bool(contract["multiplication_functional_initial_registers"]) != constant
    assert contract["witness"]["designated_target_changes"] == changed
    assert contract["eligible"] == (not constant and changed)
    if changed:
        assert contract["witness"]["ordinary_final"][2] == 10
        assert contract["witness"]["ablated_final"][2] == 6


def test_composition_revision_preserves_training_and_every_other_population(history, dataset):
    old = study.generate(history, seed=23, composed_patterns=64, states_per_pattern=3,
                         evaluation_count=8, composition_version=1)
    assert dataset["training"] == old["training"]
    for name in ("patterns", "slots", "distinct_target_counts"):
        assert dataset[name] == old[name]
    for condition in study.CONDITIONS:
        for name in study.POPULATIONS:
            if name != "composition":
                assert dataset["evaluations"][condition][name] == old["evaluations"][condition][name]
    assert all(w["eligible"] for w in dataset["composition_witnesses"])
    broken = copy.deepcopy(dataset)
    broken["composition_witnesses"][0]["witness"]["ablated_final"][0] = 99
    with pytest.raises(ValueError, match="witness mismatch"):
        study.audit(broken, history)


def test_leakage_and_broken_pairing_are_rejected(history, dataset):
    broken = copy.deepcopy(dataset)
    broken["evaluations"]["multi_state"]["dev"][0] = dataset["training"]["multi_state"][0]
    with pytest.raises(ValueError, match="overlaps training"):
        study.audit(broken, history)
    poisoned = copy.deepcopy(history)
    example = dataset["evaluations"]["multi_state"]["transition_values_id"][0]
    poisoned["pairs"].add(study.pair_key(example.program, example.initial))
    with pytest.raises(ValueError, match="historical example"):
        study.audit(dataset, poisoned)
    broken = copy.deepcopy(dataset)
    broken["training"]["repeat_one"][0] = dataset["training"]["multi_state"][1]
    with pytest.raises(ValueError, match="mismatch"):
        study.audit(broken, history)


class FixtureTokenizer:
    """Small offline offset tokenizer; four tokens per answer register like pinned GPT2.

    Used only for construction tests, never as a study tokenizer. The real command
    enforces the existing tokenizer hash before calling pack().
    """
    def encode(self, text, add_special_tokens=False):
        matches = list(re.finditer(r"[A-Za-z]+|[0-9]+|[^\w\s]", text))
        return SimpleNamespace(ids=[i + 1 for i in range(len(matches))],
                               offsets=[m.span() for m in matches])


def test_intact_packing_positions_model_input_boundary_and_saved_scoring(tmp_path, dataset):
    examples = dataset["evaluations"]["multi_state"]["transition_values_id"]
    stats, arrays = prepare.pack(examples, FixtureTokenizer(), tmp_path / "packed")
    assert stats["examples"] == 8 and stats["answer_tokens_supervised"] == 128
    assert stats["numeric_value_tokens"] == 32 and stats["formatting_eos_tokens"] == 104
    packed = PackedDataset(tmp_path / "packed")
    batch = packed.model_batch([0], as_torch=False)
    assert set(batch) == {"input_ids", "positions", "doc_ids", "key_padding_mask"}
    for document in range(8):
        locations = np.argwhere(arrays["doc_ids"] == document)
        assert len(set(locations[:, 0])) == 1
        row, start = locations[0]
        assert arrays["positions"][row, start] == 0
        assert arrays["labels"][row, start] == -100
    assert all(set(e.inference_record()) == {"prompt", "example_id"} for e in examples)
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(e.to_record()) + "\n" for e in examples))
    predictions = [{"example_id": e.example_id, "completion_text": e.target_text,
                    "stopped_on_eos": True, "truncated": False, "prompt_truncated": False} for e in examples]
    predictions[0]["completion_text"] = "r0=1"
    pred = tmp_path / "pred.jsonl"
    pred.write_text("".join(json.dumps(r) + "\n" for r in predictions))
    result = score(source, pred, tmp_path / "score.json")
    assert result["complete_answers"]["correct"] == 7
    assert result["complete_answers"]["total"] == 8
    assert result["registers"]["total"] == 32 and result["registers"]["malformed"] == 4
    changed = result["by_value_change"]["changed"]
    unchanged = result["by_value_change"]["unchanged"]
    assert changed["total"] + unchanged["total"] == 32
    assert changed["malformed"] + unchanged["malformed"] == 4
    assert changed["correct"] + unchanged["correct"] == result["registers"]["correct"]
    assert result["no_op_baseline"]["by_value_change"]["changed"]["correct"] == 0
    assert result["no_op_baseline"]["by_value_change"]["unchanged"]["correct"] == unchanged["total"]
    with pytest.raises(FileExistsError):
        score(source, pred, tmp_path / "score.json")
    pred.write_text("".join(json.dumps(r) + "\n" for r in predictions[:-1]))
    with pytest.raises(ScoringError, match="incomplete"):
        score(source, pred, tmp_path / "incomplete.json")


def test_projection_counts_all_eight_runs_and_shared_proxy(tmp_path):
    child = {"additional_exposure_tokens": 1000, "train_seconds": 10,
             "checkpoint_seconds": 23, "checkpoint": {"bytes": 100},
             "measured_stage_seconds": {"training_invocation_wall": 40,
                                        "official_evaluation": 830, "answer_diagnostics": 20}}
    (tmp_path / "run_status.json").write_text(json.dumps({"complete": True, "failure": None,
        "children": {f"78-{mode}-full_sequence": child for mode in ("dense", "untyped", "typed")}}))
    result = prepare.resource_projection({"updates": 1000, "full_label_tokens": 1000}, 8, tmp_path, 3)
    assert result["checkpoint_count_per_run"] == 3
    assert result["checkpoint_storage_bytes_all_runs"] == 2400
    arms = result["per_model_per_condition_seconds"]
    assert arms["shared"]["total"] == pytest.approx(arms["dense"]["total"] * 1.25)
    assert result["projected_hours"] * 3600 == pytest.approx(2 * sum(a["total"] for a in arms.values()) + 63)
    assert result["allocation_proposal_hours"] >= result["watchdog_hours"] + 0.05


def test_existing_preparation_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        prepare.prepare(tmp_path, "unused", [], "unused")
