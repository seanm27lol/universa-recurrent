import pytest
from open_weight_lingua.tasks import ProblemGroup, Statement, canonical, interpret
from open_weight_lingua.text_edits import (
    RULE_SHA256,
    RULE_VERSION,
    agreement,
    coverage,
    edit,
    parse,
    rule_sha256,
)

SOURCE = (
    Statement("x", "assign", 3),
    Statement("y", "assign", 8),
    Statement("x", "add", 2),
)


def _group(group_id, affected):
    if affected == "x":
        source = SOURCE
        donor = (Statement("x", "assign", 4), *SOURCE[1:])
        index = 0
    else:
        source = (
            Statement("x", "assign", 3),
            Statement("y", "assign", 8),
            Statement("y", "add", 1),
        )
        donor = (
            Statement("x", "assign", 3),
            Statement("y", "assign", 9),
            Statement("y", "add", 1),
        )
        index = 1
    group = ProblemGroup(group_id, "smoke", source, donor, affected, index)
    group.validate()
    return group


def test_parse_eligible_records_exact_spans():
    description = "After the update, x is now 5 and y is currently 8."
    result = parse(description)
    assert result["x"].status == "eligible" and result["x"].value == 5
    assert result["y"].status == "eligible" and result["y"].value == 8
    start, end = result["x"].value_span
    assert description[start:end] == "5"
    hit = result["x"].hits[0]
    assert hit.rule == "is-now"
    assert description[slice(*hit.statement_span)] == "x is now 5"


@pytest.mark.parametrize(
    "statement,rule",
    [
        ("x is currently 5", "is-currently"),
        ("x is now 5", "is-now"),
        ("the current value of x is 5", "current-value-of"),
    ],
)
def test_each_frozen_form_is_recognized(statement, rule):
    result = parse(statement + ".")
    assert result["x"].status == "eligible" and result["x"].value == 5
    assert result["x"].hits[0].rule == rule
    assert result["y"].status == "absent"


def test_values_are_canonical_integers_in_task_range():
    assert parse("x is now 0")["x"].value == 0
    assert parse("x is now 19")["x"].value == 19
    for text in ("x is now 20", "x is now 99", "x is now 07", "x is now -1"):
        assert parse(text)["x"].status == "absent"


def test_initial_assignments_never_match_even_when_quoting_the_program():
    assert canonical(SOURCE) == "x = 3\ny = 8\nx = x + 2"
    description = (
        "The activation describes the trace 'x = 3; y = 8; x = x + 2': "
        "x starts at 3, and initially y is 8."
    )
    result = parse(description)
    assert result["x"].status == "absent"
    assert result["y"].status == "absent"
    for text in ("x = 3", "x starts at 3", "initially x is 3", "x is initially 3"):
        assert parse(text)["x"].status == "absent"


def test_quoted_program_does_not_block_a_real_current_statement():
    description = (
        f"The trace reads '{canonical(SOURCE)}'. From it, the current value of x is 5."
    )
    result = parse(description)
    assert result["x"].status == "eligible" and result["x"].value == 5


def test_conflicting_current_values_are_ambiguous():
    result = parse("x is now 5, but the current value of x is 6.")
    assert result["x"].status == "ambiguous"
    assert result["x"].detail == "conflicting values"
    assert result["x"].value is None and result["x"].value_span is None
    assert len(result["x"].hits) == 2


def test_repeated_statement_is_ambiguous_not_eligible():
    result = parse("x is now 5; indeed, x is now 5.")
    assert result["x"].status == "ambiguous"
    assert result["x"].detail == "repeated value"


def test_absent_when_nothing_is_stated():
    result = parse("The vector mixes several unrelated features.")
    assert all(result[v].status == "absent" for v in ("x", "y"))
    assert parse("X is now 5")["x"].status == "absent"  # case-sensitive by design


def test_edit_changes_only_the_value_digits():
    description = "Trace: x is now 5; y is currently 8. Done."
    start, end = parse(description)["x"].value_span
    outcome = edit(description, "x", 17)
    assert outcome.status == "edited"
    assert outcome.original_value == 5
    assert outcome.description == "Trace: x is now 17; y is currently 8. Done."
    assert outcome.description == description[:start] + "17" + description[end:]
    assert outcome.description.startswith(description[:start])
    assert outcome.description.endswith(description[end:])


def test_wrong_variable_edit_touches_only_the_other_variable():
    description = "x is now 5; y is currently 8."
    outcome = edit(description, "y", 9)
    assert outcome.status == "edited"
    assert outcome.requested_value == 9 and outcome.original_value == 8
    assert outcome.description == "x is now 5; y is currently 9."
    assert parse(outcome.description)["x"].value == 5
    refused = edit("x is now 5.", "y", 9)
    assert refused.status == "absent" and refused.description is None


def test_edit_refuses_absent_and_ambiguous_without_fabricating():
    for text, status in (
        ("The vector mixes several features.", "absent"),
        ("x is now 5, but x is currently 6.", "ambiguous"),
    ):
        outcome = edit(text, "x", 7)
        assert outcome.status == status
        assert outcome.description is None
        assert outcome.original_value is None and outcome.value_span is None


def test_edit_validates_variable_and_requested_value():
    for bad in (-1, 20, "7", 7.0, True):
        with pytest.raises(ValueError):
            edit("x is now 5.", "x", bad)
    with pytest.raises(ValueError):
        edit("x is now 5.", "z", 7)
    with pytest.raises(ValueError):
        edit(5, "x", 7)


def test_agreement_records_true_and_false_statements_separately():
    assert interpret(SOURCE) == {"x": 5, "y": 8}
    outcome = agreement("x is now 5; y is currently 8.", SOURCE)
    assert outcome["x"]["agrees"] is True and outcome["x"]["parsed_value"] == 5
    assert outcome["y"]["agrees"] is True
    outcome = agreement("x is now 6; y is currently 8.", SOURCE)
    assert outcome["x"]["agrees"] is False
    assert outcome["x"]["parsed_value"] == 6
    assert outcome["x"]["reference_value"] == 5
    outcome = agreement("nothing explicit is stated.", SOURCE)
    assert outcome["x"]["agrees"] is None and outcome["x"]["status"] == "absent"


def test_agreement_never_gates_the_requested_edit():
    description = "x is now 6."
    assert agreement(description, SOURCE)["x"]["agrees"] is False
    outcome = edit(description, "x", 5)
    assert outcome.status == "edited"
    assert outcome.description == "x is now 5."


def test_coverage_counts_all_groups_and_every_exclusion_reason():
    pairs = [
        (_group("g-x-ok", "x"), "x is now 5; y is currently 8."),
        (_group("g-x-absent", "x"), "the trace was computed step by step."),
        (_group("g-y-conflict", "y"), "x is now 3; y is currently 9 and y is now 10."),
        (_group("g-y-ok", "y"), "y is currently 9."),
    ]
    report = coverage(pairs)
    assert report["groups"] == 4
    assert report["eligible_groups"] == 2
    assert report["eligible_fraction"] == 0.5
    assert report["exclusion_reasons"] == {"absent": 1, "ambiguous": 1}
    assert report["per_variable_status"] == {
        "x": {"eligible": 2, "absent": 2, "ambiguous": 0},
        "y": {"eligible": 2, "absent": 1, "ambiguous": 1},
    }
    assert report["rule_sha256"] == RULE_SHA256
    assert coverage([])["groups"] == 0


def test_coverage_rejects_duplicate_groups():
    group = _group("g-dup", "x")
    with pytest.raises(ValueError, match="duplicate"):
        coverage([(group, "x is now 5."), (group, "x is now 5.")])


def test_rule_version_and_hash_are_frozen_and_stable():
    first, second = rule_sha256(), rule_sha256()
    assert first == second == RULE_SHA256
    assert len(first) == 64 and bytes.fromhex(first) is not None
    assert RULE_VERSION
