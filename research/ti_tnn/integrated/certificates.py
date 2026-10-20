"""Versioned query certificates, and the serving expectation they are checked against
(protocols/integration_test_v1.md, revision 2: "The query certificate").

Example: the interpreter's raw scores on 1,000 calibration questions are labelled correct or not under one
snapshot's modules and world. Fixed-sequence testing (`calibration.fixed_sequence_certify`) then certifies
"answer when score >= 0.98" at an accepted-error rate of at most 1%, at level 0.05. A `QueryCertificate`
records that whole result:
- the grid;
- every step's accepted count, error count, p-value and decision;
- the selected threshold and the stop reason.

It also records what the result is valid for:
- **interpreter:** the interpreter whose scores these are;
- **snapshot:** the snapshot, by digest, whose module outputs decided correctness;
- **query policy:** how parses become queries and decisions;
- **correctness event:** what "correct" means;
- **distribution:** the sampling distribution the calibration questions were drawn from.

And it keeps the records themselves (item, raw score, correct), so anyone can re-derive the result.

**Checked against the expectation, never against itself.** The party that will serve states what it serves
in an `Expectation`. `binding_problems` compares the certificate with that statement and with the snapshot
actually being checked; it never compares the certificate with its own fields. This replaces the older
pattern, `Calibration.binding_problems(cal.interpreter_identity, ...)`, in which a calibration vouched for
its own interpreter. That pattern is still used, unchanged, for PR 6's `Calibration`.

**What a certificate does not establish:**
- that its correctness labels are right; that needs the reference world and the parsed queries, which an
  independent verifier must re-derive;
- anything about questions outside the declared distribution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from . import calibration as K
from .contracts import content_identity

VERSION = "ti_tnn_query_certificate_v1"


@dataclass(frozen=True)
class Expectation:
    """What a serving context will actually serve, stated by the party that serves it."""

    interpreter_identity: str
    policy_version: str
    event_version: str
    distribution_identity: str
    grid: tuple[float, ...]
    alpha: float = 0.01
    delta: float = 0.05

    @property
    def identity(self) -> str:
        return content_identity({**asdict(self), "grid": list(self.grid)})


@dataclass(frozen=True)
class QueryCertificate:
    """A complete fixed-sequence result, bound to the context it was computed for. Build it with `certify`."""

    version: str
    interpreter_identity: str
    snapshot_digest: str
    policy_version: str
    event_version: str
    distribution_identity: str
    grid: tuple[float, ...]
    alpha: float
    delta: float
    steps: tuple[tuple[float, int, int, float, bool], ...]  # (threshold, accepted, errors, p-value, rejected)
    threshold: float | None
    certified: bool
    stop_reason: str
    records: tuple[tuple[str, float, bool], ...]  # (item id, raw score, correct), in calibration order

    def as_dict(self) -> dict:
        d = asdict(self)
        d["grid"] = list(self.grid)
        d["steps"] = [list(s) for s in self.steps]
        d["records"] = [list(r) for r in self.records]
        return d

    @property
    def identity(self) -> str:
        return content_identity(self.as_dict())

    def accepts(self, score: float) -> bool:
        return self.certified and score >= self.threshold

    def rederivation_problems(self) -> list[str]:
        """The stored result must be exactly what fixed-sequence testing gives on the stored records."""
        problems = []
        if self.version != VERSION:
            problems.append(f"unknown certificate version {self.version!r}")
        if len({r[0] for r in self.records}) != len(self.records):
            problems.append("calibration records repeat an item")
        try:
            again = K.fixed_sequence_certify([r[1] for r in self.records], [r[2] for r in self.records], self.grid,
                                             self.alpha, self.delta)
        except ValueError as e:
            return problems + [f"the grid is invalid: {e}"]
        steps = tuple((s["threshold"], s["accepted"], s["errors"], s["p_value"], s["rejected"]) for s in again["steps"])
        if steps != self.steps:
            problems.append("the stored steps do not re-derive from the records")
        if (again["threshold"], again["certified"], again["stop_reason"]) != (self.threshold, self.certified,
                                                                            self.stop_reason):
            problems.append("the stored threshold, decision or stop reason does not re-derive from the records")
        return problems

    def binding_problems(self, expected: Expectation, snapshot_digest: str) -> list[str]:
        """Mismatches between this certificate and what is actually being served: `expected` and `snapshot_digest`
        come from the serving side, never from the certificate."""
        out = []
        if self.snapshot_digest != snapshot_digest:
            out.append("certified on a different snapshot's module outputs; recertify before use")
        for name, mine, theirs in (("interpreter", self.interpreter_identity, expected.interpreter_identity),
                                   ("query policy", self.policy_version, expected.policy_version),
                                   ("correctness event", self.event_version, expected.event_version),
                                   ("sampling distribution", self.distribution_identity, expected.distribution_identity),
                                   ("grid", tuple(self.grid), tuple(expected.grid)),
                                   ("risk level", (self.alpha, self.delta), (expected.alpha, expected.delta))):
            if mine != theirs:
                out.append(f"certified for a different {name}")
        return out

    def supporting_step(self, threshold: float) -> tuple | None:
        """The step that certifies `threshold`: tested, with its null rejected. None if there is none."""
        return next((s for s in self.steps if s[0] == threshold and s[4]), None)


def certify(records: list[tuple[str, float, bool]], expected: Expectation, snapshot_digest: str) -> QueryCertificate:
    """Fixed-sequence certification of "answer when score >= t" over `expected.grid` (strictest first), on the
    calibration records (item id, raw score, correct under this snapshot), bound to `expected` and the snapshot."""
    result = K.fixed_sequence_certify([r[1] for r in records], [bool(r[2]) for r in records], expected.grid,
                                      expected.alpha, expected.delta)
    steps = tuple((s["threshold"], s["accepted"], s["errors"], s["p_value"], s["rejected"]) for s in result["steps"])
    return QueryCertificate(VERSION, expected.interpreter_identity, snapshot_digest, expected.policy_version,
                            expected.event_version, expected.distribution_identity, tuple(expected.grid),
                            expected.alpha, expected.delta, steps, result["threshold"], result["certified"],
                            result["stop_reason"], tuple((str(i), float(s), bool(c)) for i, s, c in records))


def effective_release(admitting: QueryCertificate, releasing: QueryCertificate) -> dict:
    """The rule for a question admitted under one certificate and released under another.

    Both gates pass exactly when score >= max(t0, t1). The releasing certificate covers that stricter threshold,
    because it is a grid point no looser than its own selected threshold, and every such point was tested and
    rejected. This holds only when both certificates share the interpreter (hence the raw score), the eligible
    questions (the query policy, the event and the distribution) and the grid. Otherwise there is no effective
    rule, and the result must not be released."""
    problems = []
    for name in ("interpreter_identity", "policy_version", "event_version", "distribution_identity", "grid",
                 "alpha", "delta"):
        if getattr(admitting, name) != getattr(releasing, name):
            problems.append(f"the certificates differ in {name}")
    if not (admitting.certified and releasing.certified):
        problems.append("a certificate certified no rule")
    if problems:
        return {"threshold": None, "supporting_step": None, "problems": problems}
    t = max(admitting.threshold, releasing.threshold)
    step = releasing.supporting_step(t)
    if step is None:
        return {"threshold": None, "supporting_step": None,
                "problems": [f"the releasing certificate has no rejected step at {t}"]}
    return {"threshold": t, "supporting_step": list(step), "problems": []}
