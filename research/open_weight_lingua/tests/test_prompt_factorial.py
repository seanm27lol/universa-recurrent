from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_weight_lingua import prompt_factorial as pf
from open_weight_lingua.state_question_first import QUESTION_TEMPLATE


BODY = "x = 9\ny = 8\ny = 3\ny = y + 3\nx = x + 1"


def manifest(split):
    return {"inputs": [
        {"id": f"{split}-0000-{side}-{variable}", "group_id": f"{split}-0000",
         "side": side, "variable": variable, "answer": "10" if variable == "x" else "6",
         "prompt": BODY + "\n" + pf.ORIGINAL.format(variable=variable)}
        for side in ("A", "B") for variable in ("x", "y")
    ]}


def test_exact_factorial_strings_and_separate_original_repeat():
    question = "What is y? Reply with only the integer."
    expanded = "What is y at the end of this program? Reply with only the integer."
    assert pf.render_prompt(BODY, "y", "original_before") == question + "\n" + BODY
    assert pf.render_prompt(BODY, "y", "original_after") == BODY + "\n" + question
    assert pf.render_prompt(BODY, "y", "expanded_before") == expanded + "\n" + BODY
    assert pf.render_prompt(BODY, "y", "expanded_after") == BODY + "\n" + expanded
    assert pf.render_prompt(BODY, "y", "expanded_before") == QUESTION_TEMPLATE.format(variable="y", program=BODY)
    assert pf.render_prompt(BODY, "y", "original_repeat") == question + "\n" + BODY + "\n" + question


def test_builder_keeps_every_pair_label_and_operation_stratum_without_mutating_inputs():
    calibration, pilot = manifest("calibration"), manifest("pilot")
    original = deepcopy((calibration, pilot))
    rows = pf.build_rows(calibration, pilot)
    assert (calibration, pilot) == original
    assert rows == pf.build_rows(calibration, pilot)
    assert len(rows) == 40
    assert Counter(row["condition"] for row in rows) == {condition: 8 for condition in pf.CONDITIONS}
    assert len({row["id"] for row in rows}) == 40
    for row in rows:
        assert row["program"] == BODY
        assert row["primary_factorial"] == (row["condition"] != "original_repeat")
        assert row["last_line_assigns_asked"] == (row["variable"] == "x")
        assert row["last_program_variable"] == "x"
        assert row["asked_last_update_kind"] == "arithmetic"
        assert row["asked_last_update_operation"] == "add"
        assert row["answer"] == ("10" if row["variable"] == "x" else "6")


@pytest.mark.parametrize("problem", ("wrong_answer", "duplicate", "missing_pair", "wording", "reserved_split"))
def test_source_changes_and_incomplete_pairs_are_rejected(problem):
    calibration, pilot = manifest("calibration"), manifest("pilot")
    if problem == "wrong_answer":
        calibration["inputs"][0]["answer"] = "11"
    elif problem == "duplicate":
        calibration["inputs"].append(deepcopy(calibration["inputs"][0]))
    elif problem == "missing_pair":
        calibration["inputs"].pop()
    elif problem == "wording":
        calibration["inputs"][0]["prompt"] += "\n"
    else:
        calibration["inputs"][0]["group_id"] = "validation_a-0000"
    with pytest.raises(ValueError):
        pf.build_rows(calibration, pilot)


def test_tokenization_uses_chat_generation_prefix_and_rejects_bucket_overflow():
    class Tokenizer:
        def __init__(self, n):
            self.n, self.messages = n, []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is True and add_generation_prompt is True
            self.messages.append(messages)
            return list(range(self.n))

    rows = pf.build_rows(manifest("calibration"), manifest("pilot"))
    tokenizer = Tokenizer(pf.TARGET_BUCKET - pf.MAX_NEW_TOKENS)
    pf.tokenize(tokenizer, rows)
    assert tokenizer.messages[0] == [{"role": "user", "content": rows[0]["prompt"]}]
    assert all(row["answer_position"] == 119 and row["attention_mask"] == [1] * 120 for row in rows)
    with pytest.raises(ValueError, match="pinned bucket"):
        pf.tokenize(Tokenizer(121), rows)


def test_preparation_refuses_existing_destination_before_reading_inputs(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("preserve")
    with pytest.raises(FileExistsError):
        pf.prepare(SimpleNamespace(output=output))
    assert output.read_text() == "preserve"


def test_reserved_run_rejected_before_its_manifest_is_read(tmp_path):
    lock = tmp_path / "lock.json"
    lock.write_text("{}")
    args = SimpleNamespace(output=tmp_path / "fresh.json", family="gemma3-12b", lock=lock,
                           calibration_run=Path("unopened-validation-a"), pilot_run=Path("unopened-validation-b"))
    with pytest.raises(ValueError, match="already-opened calibration"):
        pf.prepare(args)
