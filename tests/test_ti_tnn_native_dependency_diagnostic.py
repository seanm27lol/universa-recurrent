"""Adversarial reference dependencies and saved-text denominator checks."""
import copy
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "ti_tnn"))
pytest.importorskip("numpy")
from native.independent.dependency_diagnostic import analyze_program, score_population
from native.independent.rescore_generations import ScoringError


def test_final_write_is_per_target_and_dead_arithmetic_is_excluded():
    result = analyze_program(("add r0 r1", "set r0 7", "copy r2 r0"))
    assert [r["final_write"] for r in result] == ["set", "untouched", "copy", "untouched"]
    assert result[0]["relevant_instructions"] == [1]
    assert result[0]["discarded_prior_target_writes"] == [0]
    assert result[0]["dependency_depth"] == 1
    assert result[2]["dependency_depth"] == 2
    assert result[2]["functional_initial_registers"] == []
    assert result[1]["functional_initial_registers"] == [1]


def test_aliases_and_indirectly_retained_prior_writes():
    result = analyze_program(("add r0 r1", "copy r2 r0", "set r0 4", "copy r0 r2"))
    assert result[0]["relevant_instructions"] == [0, 1, 3]
    assert result[0]["discarded_prior_target_writes"] == [2]
    assert result[0]["functional_initial_registers"] == [0, 1]
    assert result[0]["dependency_depth"] == 3
    self_copy = analyze_program(("copy r0 r0",))[0]
    assert self_copy["final_write"] == "copy" and self_copy["dependency_depth"] == 1


def test_modular_cancellation_and_zero_annihilation_are_not_initial_dependence():
    result = analyze_program(tuple(["add r0 r0"] * 5))[0]
    assert result["read_initial_registers"] == [0]
    assert result["functional_initial_registers"] == []  # 32*x == 0 mod 32
    assert result["dependency_depth"] == 5
    result = analyze_program(("set r1 0", "multiply r0 r1"))[0]
    assert result["read_initial_registers"] == [0]
    assert not result["depends_on_initial"]
    # A one-coordinate test at (0,0) would miss both dependencies of x*y.
    assert analyze_program(("multiply r0 r1",))[0]["functional_initial_registers"] == [0, 1]


def test_depth_follows_inputs_before_the_current_write_and_full_four_variable_support():
    result = analyze_program(("add r0 r1", "add r2 r3", "multiply r0 r2"))[0]
    assert result["functional_initial_registers"] == [0, 1, 2, 3]
    assert result["dependency_depth"] == 2
    assert result["discarded_prior_target_writes"] == []


@pytest.mark.parametrize("program", [(), ("set r0 32",), ("copy r0 2",), ("add r4 r0",)])
def test_malformed_program_refused(program):
    with pytest.raises(ScoringError):
        analyze_program(program)


def fixture():
    prompt = ("Initial: r0=1 r1=2 r2=3 r3=4\nProgram:\nset r0 7\n"
              "Answer with the final four registers.\nAnswer:\n")
    target = "r0=7 r1=2 r2=3 r3=4"
    identity = hashlib.sha256((prompt + target).encode()).hexdigest()[:20]
    example = {"example_id": f"structured-dev-{identity}", "split": "dev", "prompt": prompt,
               "target_text": target, "initial": [1, 2, 3, 4], "program": ["set r0 7"], "modulus": 32}
    saved = {"example_id": example["example_id"], "target_text": target, "completion_text": target,
             "stopped_on_eos": True, "truncated": False}
    return {example["example_id"]: example}, saved


@pytest.mark.parametrize("completion", ["r0=7 r1=2", "r0=7 r1=2 r2=3 r3=4 extra",
                                        "r0=07 r1=2 r2=3 r3=4", "r0=32 r1=2 r2=3 r3=4", None])
def test_malformed_outputs_fail_every_register_without_denominator_shrinkage(completion):
    source, saved = fixture()
    saved["completion_text"] = completion
    before = copy.deepcopy((source, saved))
    summary, _ = score_population([saved], source, expected_count=1)
    assert summary["complete_answers"]["total"] == 1
    assert summary["registers"]["total"] == summary["registers"]["malformed"] == 4
    assert summary["registers"]["correct"] == 0
    assert summary["registers"]["accuracy_valid_only"] is None
    assert before == (source, saved)


def test_valid_partial_correct_answer_and_duplicate_record_rejection():
    source, saved = fixture()
    saved["completion_text"] = "r0=0 r1=2 r2=3 r3=4"
    summary, _ = score_population([saved], source)
    assert summary["registers"]["correct"] == 3
    assert summary["complete_answers"]["correct"] == 0
    assert summary["by_final_write"]["untouched"]["correct"] == 3
    with pytest.raises(ScoringError, match="duplicate"):
        score_population([saved, saved], source)
    saved["target_text"] = saved["completion_text"]
    with pytest.raises(ScoringError, match="target"):
        score_population([saved], source)
