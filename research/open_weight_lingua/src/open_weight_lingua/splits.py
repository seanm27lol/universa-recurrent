"""Deterministic five-split construction with carried exclusion inventories.

Implements the split table of the Phase Two brief (Section 6): fixed order
smoke -> calibration -> pilot -> validation_a -> validation_b, disjoint
namespaces and seeds, no canonical program repeated anywhere, per-split
generation stats, and pair integrity inherited from tasks.generate_groups.
The plan hash covers every group's identity for the locked manifest
(Section 10, "Evidence to save"); it identifies artifacts but does not
authenticate them. Canonical-program disjointness is textual: two distinct
programs with identical behavior are still distinct samples.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

from open_weight_lingua.tasks import (
    ProblemGroup,
    canonical,
    generate_groups,
    tokenize_groups,
)

SPLIT_ORDER = ("smoke", "calibration", "pilot", "validation_a", "validation_b")
DEFAULT_COUNTS = {
    "smoke": 8,
    "calibration": 256,
    "pilot": 128,
    "validation_a": 256,
    "validation_b": 256,
}


@dataclass(frozen=True)
class ExclusionInventory:
    """Sorted, JSON-serializable record of what earlier splits already used."""

    canonical_programs: tuple[str, ...]
    tokenized_prompts: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def from_groups(cls, groups, tokenizer=None) -> "ExclusionInventory":
        programs, prompts = set(), set()
        for group in groups:
            programs.add(canonical(group.source))
            programs.add(canonical(group.counterfactual))
            if tokenizer is not None:
                for row in group.variants():
                    ids = tokenizer.apply_chat_template(
                        [{"role": "user", "content": row["prompt"]}],
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                    if not ids:
                        raise ValueError("empty tokenized prompt; no silent drop")
                    prompts.add(tuple(ids))
        return cls(tuple(sorted(programs)), tuple(sorted(prompts)))

    def merge(self, other: "ExclusionInventory") -> "ExclusionInventory":
        return ExclusionInventory(
            tuple(sorted(set(self.canonical_programs) | set(other.canonical_programs))),
            tuple(sorted(set(self.tokenized_prompts) | set(other.tokenized_prompts))),
        )

    def to_dict(self) -> dict:
        return {
            "canonical_programs": list(self.canonical_programs),
            "tokenized_prompts": [list(ids) for ids in self.tokenized_prompts],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ExclusionInventory":
        try:
            programs = payload["canonical_programs"]
            prompts = payload.get("tokenized_prompts", [])
        except (TypeError, AttributeError, KeyError):
            raise ValueError("malformed exclusion inventory") from None
        if not all(isinstance(text, str) for text in programs):
            raise ValueError("canonical programs must be strings")
        if not all(
            isinstance(ids, (list, tuple))
            and all(type(token) is int for token in ids)
            for ids in prompts
        ):
            raise ValueError("tokenized prompts must be integer sequences")
        return cls(
            tuple(sorted(set(programs))),
            tuple(sorted({tuple(ids) for ids in prompts})),
        )


@dataclass(frozen=True)
class SplitPlan:
    """All five splits, their generation stats, declared counts, and plan hash."""

    groups: dict[str, tuple[ProblemGroup, ...]]
    stats: dict[str, dict]
    counts: dict[str, int]
    plan_hash: str

    def total_groups(self) -> int:
        return sum(len(self.groups[name]) for name in SPLIT_ORDER)

    def exclusion_inventory(self, split: str | None = None, tokenizer=None):
        """Inventory of every split before `split`, or of the whole plan if None."""
        if split is None:
            names = SPLIT_ORDER
        else:
            if split not in self.groups:
                raise ValueError("unknown split")
            names = SPLIT_ORDER[: SPLIT_ORDER.index(split)]
        return ExclusionInventory.from_groups(
            (group for name in names for group in self.groups[name]), tokenizer
        )


def _plan_hash(groups: dict, counts: dict) -> str:
    payload = [
        {
            "split": name,
            "count": counts[name],
            "groups": [asdict(group) for group in groups[name]],
        }
        for name in SPLIT_ORDER
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_plan(counts: dict[str, int] | None = None) -> SplitPlan:
    """Generate all five splits in fixed order, carrying program exclusions."""
    counts = dict(DEFAULT_COUNTS if counts is None else counts)
    if set(counts) != set(SPLIT_ORDER):
        raise ValueError("counts must name exactly the five planned splits")
    excluded: set[str] = set()
    groups, stats = {}, {}
    for name in SPLIT_ORDER:
        split_groups, split_stats = generate_groups(
            name, counts[name], excluded_programs=excluded
        )
        groups[name] = tuple(split_groups)
        stats[name] = split_stats
        for group in split_groups:
            excluded.add(canonical(group.source))
            excluded.add(canonical(group.counterfactual))
    return SplitPlan(groups, stats, counts, _plan_hash(groups, counts))


def validate_plan(plan: SplitPlan, expected_counts: dict[str, int] | None = None) -> dict:
    """Re-check pair integrity, uniqueness, disjointness, counts, and the hash.

    Raises ValueError on the first violation; returns a small summary otherwise.
    """
    if expected_counts is not None and plan.counts != expected_counts:
        raise ValueError("plan counts differ from the expected split table")
    group_ids: set[str] = set()
    programs: set[str] = set()
    for name in SPLIT_ORDER:
        if name not in plan.groups or name not in plan.counts:
            raise ValueError(f"missing split {name!r}")
        if len(plan.groups[name]) != plan.counts[name]:
            raise ValueError(f"{name}: group count differs from the declared count")
        for group in plan.groups[name]:
            group.validate()
            if group.split != name:
                raise ValueError(f"{group.group_id}: split-namespace violation")
            if group.group_id in group_ids:
                raise ValueError(f"{group.group_id}: duplicated group id")
            group_ids.add(group.group_id)
            for program in (group.source, group.counterfactual):
                text = canonical(program)
                if text in programs:
                    raise ValueError(
                        f"{group.group_id}: canonical program repeated in the plan"
                    )
                programs.add(text)
    if _plan_hash(plan.groups, plan.counts) != plan.plan_hash:
        raise ValueError("plan hash mismatch; group identities were altered")
    return {
        "splits": len(SPLIT_ORDER),
        "groups": len(group_ids),
        "unique_programs": len(programs),
        "plan_hash": plan.plan_hash,
    }


def tokenize_split(plan: SplitPlan, split: str, tokenizer, *, inventory=None) -> dict:
    """Tokenize one split while excluding earlier splits' tokenized prompts.

    Per-group failures are counted and reported, never silently replaced
    (Section 6: tokenization rejections and repeated templates must be
    reported). Group-by-group iteration localizes each rejection to one group.
    """
    if split not in plan.groups:
        raise ValueError("unknown split")
    if inventory is None:
        inventory = plan.exclusion_inventory(split, tokenizer=tokenizer)
    seen = {tuple(ids) for ids in inventory.tokenized_prompts}
    prior = len(seen)
    rows, rejections, accepted_prompts = [], [], set()
    for group in plan.groups[split]:
        try:
            accepted = tokenize_groups(
                [group], tokenizer, excluded_tokenized_prompts=seen
            )
        except ValueError as exc:
            rejections.append({"group_id": group.group_id, "reason": str(exc)})
            continue
        for row in accepted:
            seen.add(tuple(row["input_ids"]))
            accepted_prompts.add(tuple(row["input_ids"]))
        rows.extend(accepted)
    return {
        "split": split,
        "groups": len(plan.groups[split]),
        "accepted": len(plan.groups[split]) - len(rejections),
        "rejected": len(rejections),
        "rejections": rejections,
        "prior_excluded_prompts": prior,
        "tokenized_prompts": sorted(accepted_prompts),
        "rows": rows,
    }
