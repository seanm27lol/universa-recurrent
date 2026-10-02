import pytest

from open_weight_lingua import answer_slot

ONE = (
    'Structured answer format: a mathematical problem with a numeric answer.\n\n'
    'The answer "x + 1 gives the new value of " signals a numeric answer, '
    "confirming the incremented value of the variable '10'.\n\n"
    'Final token "\n" ends a final answer label ("the result is..."), immediately '
    'expecting a numeric answer like "10" or "10" to specify the value. or "10."'
)
MANY = (
    "Structured arithmetic puzzle format.\n\n"
    'Final token "\n" ends a question label ("sum_number"), likely "19" or "12" '
    'to complete the answer, or "17".'
)


def test_single_distinct_candidate_is_eligible_with_every_span():
    slot = answer_slot.parse(ONE)
    assert slot.status == "eligible"
    assert slot.candidates == (10, 10, 10)
    assert slot.lead == 10
    assert all(ONE[a:b] == "10" for a, b in slot.spans)


def test_several_distinct_candidates_are_ambiguous_and_keep_order():
    slot = answer_slot.parse(MANY)
    assert slot.status == "ambiguous"
    assert slot.candidates == (19, 12, 17)
    assert answer_slot.edit_slot(MANY, 5) is None


@pytest.mark.parametrize(
    "text",
    [
        "Structured format with no closing paragraph.",
        'Final token "\n" ends the question with no cue word "10".',
        'Final token "\n" ends a label, expecting a numeric answer soon.',
        'Final token "\n" ends a label, expecting "05" or "-6" or "x = 14".',
    ],
)
def test_absent_slots(text):
    assert answer_slot.parse(text).status == "absent"


def test_candidates_before_the_cue_or_after_the_slot_are_ignored():
    text = (
        'Final token "\n" ("sum = 12") ends a label, expecting "7".\n\n'
        'A later paragraph quoting "3".'
    )
    slot = answer_slot.parse(text)
    assert (slot.status, slot.candidates) == ("eligible", (7,))


def test_two_digit_values_are_not_split_and_leading_zeros_fail():
    slot = answer_slot.parse('Final token "\n" likely "19" or "1."')
    assert slot.candidates == (19, 1)
    assert answer_slot.parse('Final token "\n" likely "019"').status == "absent"


def test_edit_slot_rewrites_only_slot_candidates():
    edited = answer_slot.edit_slot(ONE, 11)
    assert answer_slot.parse(edited).candidates == (11, 11, 11)
    assert "variable '10'" in edited  # outside the slot: untouched


def test_edit_everywhere_rewrites_standalone_occurrences_only():
    text = ONE + ' Also 100 and x10 and "10".'
    edited = answer_slot.edit_everywhere(text, 9)
    assert "variable '9'" in edited
    assert "100" in edited and "x10" in edited
    assert answer_slot.parse(edited).candidates == (9, 9, 9, 9)


def test_edits_reject_out_of_range_values():
    with pytest.raises(ValueError):
        answer_slot.edit_slot(ONE, 20)
    with pytest.raises(ValueError):
        answer_slot.edit_everywhere(ONE, -1)


def test_rule_identity_is_stable_and_versioned():
    assert answer_slot.RULE_VERSION == "AS-1.0.0"
    assert answer_slot.RULE_SHA256 == answer_slot.rule_sha256()
    assert len(answer_slot.RULE_SHA256) == 64


def test_non_string_description_fails_closed():
    with pytest.raises(ValueError):
        answer_slot.parse(None)
