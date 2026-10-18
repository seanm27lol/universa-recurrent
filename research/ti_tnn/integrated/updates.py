"""Offline module replacement: update records, activation and rollback (integration plan section 7, PR 8).

Example: Ardel moves its capital from Ardel Crossing to Lunholm. The update record declares:
- **base and candidate:** the active registry it starts from, and the candidate registry with a new
  `capital_of` module;
- **scope:** the write scope (`capital_of` only);
- **intended change:** `capital_of(Ardel) = Lunholm`, plus every composition through it;
- **versions:** the checker and calibration versions it was judged with.

The gates (`update_checks.py`) judge the candidate against an updated copy of the world. Only then may
the deployment activate it, and only if all of these hold:
- the gate results were issued by the deployment's own checker for this record and this candidate;
- the record's base is still the active revision (otherwise the candidate is stale);
- every mandatory gate passed;
- a separate authority allows it. Gate results and confidence cannot substitute for that permission.

The swap itself is atomic.

**Rollback** restores the complete previous snapshot. A snapshot whose identity no longer matches its
record is refused, and the active registry is left as it was.

The active state is never modified in place. Each snapshot is a private, digested copy
(`snapshots.py`): changing the objects a candidate was built from cannot change a published snapshot, and
changing a snapshot itself is detected before it is used. A running program keeps the snapshot it started
with.

Gate results reach the deployment only as a `CheckResult` sealed by the deployment's own checker
(`update_checks.Checker`). That result is bound to the record, both snapshots, both worlds, the checker
version and the calibration context. Results from one candidate therefore cannot activate another.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import content_identity
from .snapshots import Snapshot, publish
from .update_checks import MANDATORY, PASS


@dataclass(frozen=True)
class UpdateRecord:
    """What a candidate is, what it may change, and what it is meant to change."""

    base_identity: str  # the base snapshot's digest (snapshots.digest_of)
    candidate_identity: str  # the candidate snapshot's digest
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


@dataclass
class Deployment:
    """The active snapshot and its history. `active` is only ever replaced whole, and only by a checked snapshot.

    Activation requires all of:
    - a result issued by this deployment's checker (its seal verifies);
    - the result was made for this record;
    - the record's base is the active snapshot, which still matches its digest (otherwise the candidate is
      stale);
    - the record names the snapshot inside the result, which still matches its digest;
    - the record's target world and checker version are the ones the result was made with;
    - every mandatory gate passed, and the authority permits the proposer.

    The base is published (copied) when the deployment is created, so later changes to the caller's objects
    cannot reach it. Rollback restores the previous snapshot only if it still matches its digest."""

    base: object
    authority: Authority
    checker: object
    history: list = field(default_factory=list)
    log: list = field(default_factory=list)

    def __post_init__(self):
        self.active = self.base if isinstance(self.base, Snapshot) else publish(self.base)
        self.base = None  # the deployment keeps only its own copy

    @property
    def registry(self):
        return self.active.registry

    def activate(self, record: UpdateRecord, result) -> dict:
        reasons = []
        if not self.checker.verify(result):
            reasons.append("the result was not issued by this deployment's checker, or was altered")
        if result.record_identity != record.identity:
            reasons.append("the result was made for a different record")
        if not self.active.verify():
            reasons.append("the active snapshot no longer matches its digest")
        if record.base_identity != self.active.digest or result.base_digest != self.active.digest:
            reasons.append("stale: the record's base is not the active revision")
        if not (record.candidate_identity == result.candidate_digest == result.snapshot.digest):
            reasons.append("the record does not name the checked candidate")
        elif not result.snapshot.verify():
            reasons.append("the checked candidate changed after it was checked")
        if record.target_world_identity != result.new_world_identity or record.checker_version != result.checker_version:
            reasons.append("the result was made in a different context than the record declares")
        failed = [g for g in MANDATORY if result.gates.get(g, {}).get("outcome") != PASS]
        if failed:
            reasons.append(f"mandatory gates not passed: {failed}")
        if not self.authority.permits(record):
            reasons.append(f"authority denies proposer {record.proposer!r}")
        decision = {"activated": not reasons, "reasons": reasons, "record": record.identity}
        if not reasons:
            self.history.append(self.active)
            self.active = result.snapshot  # one assignment: the swap is atomic
        self.log.append(decision)
        return decision

    def rollback(self) -> dict:
        """Restore the previous snapshot, refusing one that no longer matches its digest."""
        if not self.history:
            decision = {"rolled_back": False, "reasons": ["no earlier snapshot"]}
        elif not self.history[-1].verify():
            decision = {"rolled_back": False, "reasons": ["the stored snapshot no longer matches its digest"]}
        else:
            self.active = self.history.pop()
            decision = {"rolled_back": True, "reasons": []}
        self.log.append(decision)
        return decision
