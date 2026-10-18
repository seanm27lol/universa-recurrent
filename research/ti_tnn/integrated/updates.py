"""Offline module replacement: update records, activation and rollback (integration plan section 7, PR 8).

Example: Ardel moves its capital from Ardel Crossing to Lunholm. The update record declares:
- **base and candidate:** the active registry it starts from, and the candidate registry with a new
  `capital_of` module;
- **scope:** the write scope (`capital_of` only);
- **intended change:** `capital_of(Ardel) = Lunholm`, plus every composition through it;
- **versions:** the checker and calibration versions it was judged with.

The gates (`update_checks.py`) judge the candidate against an updated copy of the world. Only then may
the deployment activate it, and only if four things hold:
- the record's base is still the active revision (otherwise the candidate is stale);
- every mandatory gate passed;
- a separate authority allows it. Gate results and confidence cannot substitute for that permission;
- the swap is atomic.

**Rollback** restores the complete previous snapshot. A snapshot whose identity no longer matches its
record is refused, and the active registry is left as it was.

The active registry is never modified in place. Each snapshot is an immutable `Registry`, and a running
program keeps the one it started with.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import content_identity


@dataclass(frozen=True)
class UpdateRecord:
    """What a candidate is, what it may change, and what it is meant to change."""

    base_identity: str
    candidate_identity: str
    scope: tuple[str, ...]  # relations whose modules may change
    intended: tuple[tuple[str, str, str], ...]  # (relation, input, new output) for each intended atomic change
    target_world_identity: str  # the updated world the candidate is judged against
    checker_version: str
    calibration_version: str | None = None
    max_macs_per_application: int | None = None
    proposer: str = "offline-update"

    @property
    def identity(self) -> str:
        return content_identity({k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()})


class Authority:
    """A stand-in for the separately controlled permission to activate (the plan's mock effect broker). It knows
    which proposers may activate and nothing about scores: a confident candidate from a denied proposer is
    still denied."""

    def __init__(self, allowed: set[str]):
        self.allowed = frozenset(allowed)

    def permits(self, record: UpdateRecord) -> bool:
        return record.proposer in self.allowed


MANDATORY = ("identity_and_scope", "types_and_laws", "intended_behavior", "unaffected_behavior",
             "trace_and_continuation", "calibration_and_resources")


@dataclass
class Deployment:
    """The active registry and its history. `active` is only ever replaced whole."""

    active: object
    authority: Authority
    history: list = field(default_factory=list)  # (registry, its identity at activation) pairs
    log: list = field(default_factory=list)

    def activate(self, candidate, record: UpdateRecord, gates: dict) -> dict:
        """Activate `candidate` if the record is current, every mandatory gate passed and authority permits."""
        reasons = []
        if record.base_identity != self.active.identity:
            reasons.append("stale: the record's base is not the active revision")
        if record.candidate_identity != candidate.identity:
            reasons.append("the candidate is not the one the record describes")
        failed = [g for g in MANDATORY if gates.get(g, {}).get("outcome") != "pass"]
        if failed:
            reasons.append(f"mandatory gates not passed: {failed}")
        if not self.authority.permits(record):
            reasons.append(f"authority denies proposer {record.proposer!r}")
        decision = {"activated": not reasons, "reasons": reasons, "record": record.identity}
        if not reasons:
            self.history.append((self.active, self.active.identity))
            self.active = candidate  # one assignment: the swap is atomic
        self.log.append(decision)
        return decision

    def rollback(self) -> dict:
        """Restore the previous snapshot, refusing one that no longer matches its recorded identity."""
        if not self.history:
            decision = {"rolled_back": False, "reasons": ["no earlier snapshot"]}
        else:
            snapshot, identity = self.history[-1]
            if snapshot.identity != identity:
                decision = {"rolled_back": False, "reasons": ["the stored snapshot no longer matches its identity"]}
            else:
                self.history.pop()
                self.active = snapshot
                decision = {"rolled_back": True, "reasons": []}
        self.log.append(decision)
        return decision
