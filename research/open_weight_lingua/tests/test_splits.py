import json
from dataclasses import replace

import pytest

from open_weight_lingua.splits import (
    DEFAULT_COUNTS,
    SPLIT_ORDER,
    ExclusionInventory,
    build_plan,
    tokenize_split,
    validate_plan,
)
from open_weight_lingua.tasks import canonical, tokenize_groups

TINY_COUNTS = {
    "smoke": 2,
    "calibration": 3,
    "pilot": 4,
    "validation_a": 5,
    "validation_b": 6,
}


class StubTokenizer:
    """Distinct IDs per prompt, constant boundary token per variant pair."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [ord(c) for c in messages[0]["content"]] + [999]

    def decode(self, ids):
        return "<boundary>"


def test_default_plan_matches_brief_table_and_rebuilds_identically():
    first, second = build_plan(), build_plan()
    assert first.counts == DEFAULT_COUNTS
    assert [first.counts[name] for name in SPLIT_ORDER] == [8, 256, 128, 256, 256]
    assert first.counts["validation_a"] + first.counts["validation_b"] == 512
    assert first.plan_hash == second.plan_hash
    assert first.groups == second.groups
    assert first.stats == second.stats
    for name in SPLIT_ORDER:
        assert first.stats[name]["groups"] == first.counts[name]
        assert first.stats[name]["prompt_variants"] == 4 * first.counts[name]


def test_counts_must_cover_exactly_the_five_splits():
    with pytest.raises(ValueError, match="five"):
        build_plan({"smoke": 2})
    with pytest.raises(ValueError, match="five"):
        build_plan({**TINY_COUNTS, "extra": 1})


def test_tiny_plan_disjoint_by_namespace_and_program():
    plan = build_plan(TINY_COUNTS)
    summary = validate_plan(plan, expected_counts=TINY_COUNTS)
    assert summary["groups"] == sum(TINY_COUNTS.values()) == 20
    assert summary["unique_programs"] == 40
    with pytest.raises(ValueError, match="expected split table"):
        validate_plan(plan, expected_counts=DEFAULT_COUNTS)
    programs_by_split = {
        name: {
            canonical(program)
            for group in plan.groups[name]
            for program in (group.source, group.counterfactual)
        }
        for name in SPLIT_ORDER
    }
    for index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[index + 1 :]:
            assert not programs_by_split[left] & programs_by_split[right]
    for name in SPLIT_ORDER:
        assert all(group.split == name for group in plan.groups[name])
        assert all(group.group_id.startswith(name) for group in plan.groups[name])


def test_validator_catches_tampered_and_duplicated_groups():
    plan = build_plan(TINY_COUNTS)
    assert validate_plan(plan)["plan_hash"] == plan.plan_hash
    foreign = replace(
        plan,
        groups={
            **plan.groups,
            "calibration": (plan.groups["smoke"][0],) + plan.groups["calibration"][1:],
        },
    )
    with pytest.raises(ValueError, match="namespace"):
        validate_plan(foreign)
    pilot = plan.groups["pilot"]
    duplicated = replace(
        plan, groups={**plan.groups, "pilot": (pilot[0], pilot[0]) + pilot[2:]}
    )
    with pytest.raises(ValueError, match="duplicated"):
        validate_plan(duplicated)
    short = replace(plan, groups={**plan.groups, "smoke": plan.groups["smoke"][:-1]})
    with pytest.raises(ValueError, match="count"):
        validate_plan(short)
    bad_hash = replace(plan, plan_hash="0" * 64)
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_plan(bad_hash)


def test_exclusion_inventory_serialization_round_trip():
    plan = build_plan(TINY_COUNTS)
    inventory = plan.exclusion_inventory("pilot", tokenizer=StubTokenizer())
    assert len(inventory.canonical_programs) == 2 * (2 + 3)
    assert len(inventory.tokenized_prompts) == 4 * (2 + 3)
    restored = ExclusionInventory.from_dict(json.loads(json.dumps(inventory.to_dict())))
    assert restored == inventory
    full = plan.exclusion_inventory()
    assert len(full.canonical_programs) == 40 and full.tokenized_prompts == ()
    merged = restored.merge(ExclusionInventory(("x = 0\ny = 1",), ((1, 2),)))
    assert "x = 0\ny = 1" in merged.canonical_programs
    assert len(merged.canonical_programs) == len(inventory.canonical_programs) + 1
    assert (1, 2) in merged.tokenized_prompts
    assert len(merged.tokenized_prompts) == len(inventory.tokenized_prompts) + 1
    with pytest.raises(ValueError):
        ExclusionInventory.from_dict({"canonical_programs": [1]})
    with pytest.raises(ValueError):
        ExclusionInventory.from_dict({"canonical_programs": [], "tokenized_prompts": [[True]]})


def test_tokenize_split_excludes_prior_splits_and_reports_counts():
    plan = build_plan(TINY_COUNTS)
    result = tokenize_split(plan, "pilot", StubTokenizer())
    assert result["accepted"] == TINY_COUNTS["pilot"]
    assert result["rejected"] == 0 and result["rejections"] == []
    assert len(result["rows"]) == 4 * TINY_COUNTS["pilot"]
    assert result["prior_excluded_prompts"] == 4 * (2 + 3)
    assert all(row["input_ids"][row["position"]] == 999 for row in result["rows"])
    assert result["tokenized_prompts"] == sorted(
        {tuple(row["input_ids"]) for row in result["rows"]}
    )


def test_tokenize_split_rejects_prompt_already_in_restored_inventory():
    plan = build_plan(TINY_COUNTS)
    stub = StubTokenizer()
    first_rows = tokenize_groups([plan.groups["pilot"][0]], stub)
    poisoned = ExclusionInventory.from_dict(
        {
            "canonical_programs": [],
            "tokenized_prompts": [first_rows[0]["input_ids"]],
        }
    )
    result = tokenize_split(plan, "pilot", stub, inventory=poisoned)
    assert result["accepted"] == TINY_COUNTS["pilot"] - 1
    assert result["rejected"] == 1
    assert result["rejections"][0]["group_id"] == plan.groups["pilot"][0].group_id
    assert "duplicated" in result["rejections"][0]["reason"]


def test_tokenize_split_reports_every_rejection_with_constant_tokenizer(tokenizer):
    # The TinyTokenizer fixture returns constant [3, 6, 7, 8, 4], so every
    # group's variants collide; the helper must report, not replace.
    plan = build_plan(TINY_COUNTS)
    result = tokenize_split(plan, "smoke", tokenizer)
    assert result["accepted"] == 0
    assert result["rejected"] == TINY_COUNTS["smoke"]
    assert {entry["group_id"] for entry in result["rejections"]} == {
        group.group_id for group in plan.groups["smoke"]
    }
    assert all("duplicated" in entry["reason"] for entry in result["rejections"])
