"""Admission and release of interpreted questions across a version change (protocols/integration_test_v1.md:
"Which certificate governs a released result").

**Example.** A question's interpreted query scores 0.95. C0's certificate selected 0.90, so the question is
admitted, and its run starts on C0. The run pauses, and C1, whose certificate selected 0.98, is activated.
- **pin:** the run finishes on C0 and is released under C0's certificate (0.95 ≥ 0.90);
- **restart and migrate:** the result is produced on C1. It is released only if C1's certificate also covers
  it: score ≥ max(t0, t1) = 0.98. That fails, so release is refused.

`sessions.py` executes; it decides nothing about release. This adapter makes the two decisions and records
them with every identity it used:
- **`admit`:** whether a question may start, under the active context's certificate;
- **`release`:** whether a finished run's result is released. The rule depends on the context it finished on:
  - **the admitting context** (no change, or pin): that context's selected threshold;
  - **another context** (restart or migrate): `certificates.effective_release`, the stricter of the two
    thresholds, supported by a rejected step of the releasing certificate. There is no rule, and no release,
    if the two certificates do not share the interpreter, policy, event, distribution and grid.

A refusal before execution (unsupported, invalid, unknown entity), or a run that ends without a result,
releases nothing, whatever the score.
"""
from __future__ import annotations

from . import certificates as Q
from .sessions import outcome


def admit(context, score: float) -> dict:
    cert = context.certificate
    admitted = cert.accepts(score)
    return {"admitted": admitted, "score": score, "context": context.identity, "certificate": cert.identity,
            "threshold": cert.threshold if cert.certified else None,
            "reason": "" if admitted else ("no certificate" if not cert.certified else "insufficient confidence")}


def release(admission: dict, session: dict, contexts: dict) -> dict:
    """The release decision for a finished session. `contexts`: identity -> ServingContext."""
    o = outcome(session)
    final = session["segments"][-1]["context"]
    admitting, releasing = contexts[admission["context"]], contexts[final]
    decision = {"released": False, "answer": None, "result_reason": o["reason"], "score": admission["score"],
                "admitting_context": admission["context"], "admitting_certificate": admitting.certificate.identity,
                "releasing_context": final, "releasing_certificate": releasing.certificate.identity,
                "effective_threshold": None, "supporting_step": None, "reason": ""}
    if not admission["admitted"]:
        return {**decision, "reason": "not admitted"}
    if o["kind"] != "result":
        return {**decision, "reason": f"no result: {o['reason']}"}
    if final == admission["context"]:
        t = admitting.certificate.threshold
        rule = {"threshold": t, "supporting_step": list(admitting.certificate.supporting_step(t) or []) or None,
                "problems": []}
    else:
        rule = Q.effective_release(admitting.certificate, releasing.certificate)
    decision.update(effective_threshold=rule["threshold"], supporting_step=rule["supporting_step"])
    if rule["threshold"] is None:
        return {**decision, "reason": f"no release rule: {rule['problems']}"}
    if admission["score"] < rule["threshold"]:
        return {**decision, "reason": "below the effective threshold"}
    return {**decision, "released": True, "answer": o["answer"], "reason": "released"}
