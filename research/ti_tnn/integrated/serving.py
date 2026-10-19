"""The complete serving context, and a deployment that activates and rolls back whole contexts
(protocols/integration_test_v1.md, revision 2: "The serving context").

**Example.** An update makes `operator_3` the operator of `country_40_city_2_airport`. If only the candidate
registry were activated (PR 8's `Deployment` swaps exactly that), the system would serve the new modules
under a certificate computed from the old modules' outputs. That threshold was certified for a
correctness labelling that no longer holds. What is served is therefore one `ServingContext`:

> snapshot (modules) + query certificate + expectation (interpreter, query policy, correctness event,
> sampling distribution, grid) + world identity

- **`ServingDeployment.activate`** installs a complete context. The context must contain the very snapshot
  the checker sealed. Its certificate must be the one sealed with it, judged against this deployment's own
  expectation, and bound to that snapshot.
- **`rollback`** restores the previous complete context.
- **The public view** is the single attribute `active`, replaced in one assignment.

PR 8's `Deployment` still performs every check it did: checker seal, stale base, mandatory gates and
authority. It is held privately and used as the inner mechanism.

A context the deployment no longer holds, such as one a pinned session kept after a rollback, stays
usable. Its snapshot is a private digested copy (`snapshots.py`), and `problems()` re-verifies it before
use.

**Limits.** This is one process, the same as `snapshots.py`: an integrity check, not a security boundary. The
deployment keeps one expectation for its whole life. Changing the interpreter or the query policy is a
different kind of change from a module update, and is out of scope here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .certificates import Expectation, QueryCertificate
from .contracts import content_identity
from .snapshots import Snapshot
from .updates import Deployment


@dataclass(frozen=True)
class ServingContext:
    """Everything that determines what is released for a question."""

    snapshot: Snapshot
    certificate: QueryCertificate
    expectation: Expectation
    world_identity: str

    @property
    def identity(self) -> str:
        return content_identity({"snapshot": self.snapshot.digest, "certificate": self.certificate.identity,
                                 "expectation": self.expectation.identity, "world": self.world_identity})

    def problems(self) -> list[str]:
        """Why this context must not serve: a changed snapshot, a certificate bound to anything else, or a
        certificate that does not re-derive from its own records."""
        out = []
        if not self.snapshot.verify():
            out.append("the snapshot no longer matches its digest")
        out += self.certificate.binding_problems(self.expectation, self.snapshot.digest)
        out += self.certificate.rederivation_problems()
        return out

    def verify(self) -> bool:
        return not self.problems()


def context_for(result, certificate: QueryCertificate, expectation: Expectation, world_identity: str) -> ServingContext:
    """The context a check result would install: the very snapshot the checker sealed."""
    return ServingContext(result.snapshot, certificate, expectation, world_identity)


@dataclass
class ServingDeployment:
    """The active serving context and its history, replaced only whole (module docstring)."""

    initial: ServingContext
    authority: object
    checker: object
    history: list = field(default_factory=list)
    log: list = field(default_factory=list)

    def __post_init__(self):
        problems = self.initial.problems()
        if problems:
            raise ValueError(f"the initial context cannot serve: {problems}")
        self._inner = Deployment(self.initial.snapshot, self.authority, self.checker)
        self.active = self.initial
        self.expectation = self.initial.expectation
        self.initial = None  # the deployment keeps only `active` and `history`

    @property
    def registry(self):
        return self.active.snapshot.registry

    def activate(self, record, result, context: ServingContext) -> dict:
        reasons = []
        if context.snapshot is not result.snapshot:
            reasons.append("the context does not hold the snapshot the checker sealed")
        if context.certificate.identity != result.calibration_identity:
            reasons.append("the check result sealed a different certificate")
        if result.expectation_identity != self.expectation.identity:
            reasons.append("the certificate was not judged against this deployment's expectation")
        if context.expectation != self.expectation:
            reasons.append("the context changes the interpreter, policy, event, distribution or grid")
        if context.world_identity != record.target_world_identity:
            reasons.append("the context names a different world than the record")
        reasons += [f"context: {p}" for p in context.problems()]
        if reasons:
            decision = {"activated": False, "reasons": reasons, "record": record.identity}
        else:
            decision = self._inner.activate(record, result)  # seal, stale base, gates, authority
            if decision["activated"]:
                self.history.append(self.active)
                self.active = context  # one assignment
            decision = {**decision, "context": context.identity if decision["activated"] else None}
        self.log.append(decision)
        return decision

    def rollback(self) -> dict:
        """Restore the previous complete context, refusing one that can no longer serve."""
        if not self.history:
            decision = {"rolled_back": False, "reasons": ["no earlier context"]}
        else:
            previous = self.history[-1]
            problems = previous.problems()
            if problems:
                decision = {"rolled_back": False, "reasons": [f"the stored context cannot serve: {problems}"]}
            else:
                decision = self._inner.rollback()
                if decision["rolled_back"]:
                    if self._inner.active is not previous.snapshot:
                        raise RuntimeError("the inner deployment restored a different snapshot")
                    self.history.pop()
                    self.active = previous
        decision = {**decision, "context": self.active.identity}
        self.log.append(decision)
        return decision
