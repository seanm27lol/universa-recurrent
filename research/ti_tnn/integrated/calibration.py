"""Versioned calibration and a risk-controlled abstention threshold (integration plan section 5.4, PR 6).

**The calibrated event:** "the policy's answer is correct". It is defined on every question where the
policy would answer before thresholding, whatever kind of question it is; answering a question that should
have been refused counts as an error. It is not "the predicted program matches the reference", and not a
routing label.

**The procedure,** all on a calibration split disjoint from training and test:
1. Fit a Platt transform, p = sigmoid(a · logit(score) + b). A Gaussian prior (weight 1) pulls it toward
   the identity, so a split with no errors, or one where the score separates errors perfectly, does not send
   it to infinity.
2. Pick the threshold that answers as many calibration questions as possible while the one-sided
   Clopper–Pearson upper bound on their error rate stays at or below the target risk (default: 1% at 95%
   confidence). If no threshold qualifies, the policy always abstains.

**What the artifact records:** its version, its parameters, the event, and the identities of the
interpreter, registry and policy whose outputs it was fit on. `binding_problems` refuses to apply it to any
other combination. After a module update, it must be refit.

**Sampling assumption** for the bounds reported on test: conditional on a fixed world, trained interpreter
and registry, test questions are independent draws from the test distribution. Different worlds and
trained models are separate units, never pooled into one bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import content_identity

VERSION = "platt_v1"
EVENT = "the policy's answer is correct (on questions the policy would answer before thresholding)"


def _logit(x: float, eps: float = 1e-6) -> float:
    x = min(max(x, eps), 1 - eps)
    return math.log(x / (1 - x))


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def _objective(zs, labels, a, b, prior):
    nll = 0.0
    for z, y in zip(zs, labels):
        u = a * z + b
        nll += (math.log1p(math.exp(-u)) if u > 0 else -u + math.log1p(math.exp(u))) if y else \
               (math.log1p(math.exp(u)) if u < 0 else u + math.log1p(math.exp(-u)))
    return nll + prior / 2 * ((a - 1) ** 2 + b ** 2)


def platt_fit(scores: list[float], labels: list[bool], prior: float = 1.0, iters: int = 100) -> tuple[float, float]:
    """(a, b) by damped Newton steps on the penalized log likelihood (Gaussian prior toward a=1, b=0)."""
    zs = [_logit(s) for s in scores]
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga, gb = prior * (a - 1), prior * b
        haa, hab, hbb = prior, 0.0, prior
        for z, y in zip(zs, labels):
            p = _sigmoid(a * z + b)
            r, w = p - float(y), p * (1 - p)
            ga, gb = ga + r * z, gb + r
            haa, hab, hbb = haa + w * z * z, hab + w * z, hbb + w
        det = haa * hbb - hab * hab
        if det <= 0:
            break
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        before, step = _objective(zs, labels, a, b, prior), 1.0
        while step > 1e-6 and _objective(zs, labels, a - step * da, b - step * db, prior) > before:
            step /= 2
        a, b = a - step * da, b - step * db
        if step * (abs(da) + abs(db)) < 1e-10:
            break
    return a, b


def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in range(k + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def clopper_pearson_upper(errors: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) Clopper–Pearson upper bound on a binomial rate."""
    if n == 0 or errors >= n:
        return 1.0
    lo, hi = errors / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _binom_cdf(errors, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def choose_threshold(probabilities: list[float], correct: list[bool], target: float, alpha: float) -> float:
    """The lowest threshold (most answers) whose accepted set has a Clopper–Pearson upper bound <= target;
    infinity (never answer) if none does. Ties are accepted or refused together."""
    order = sorted(range(len(probabilities)), key=lambda i: -probabilities[i])
    best, errors = math.inf, 0
    for rank, i in enumerate(order):
        errors += not correct[i]
        last_of_tie = rank + 1 == len(order) or probabilities[order[rank + 1]] < probabilities[i]
        if last_of_tie and clopper_pearson_upper(errors, rank + 1, alpha) <= target:
            best = probabilities[i]
    return best


def reliability(probabilities: list[float], correct: list[bool], bins: int = 10) -> dict:
    """Reliability-diagram bins, expected calibration error, Brier score and log loss."""
    n = len(probabilities)
    if n == 0:
        return {"n": 0, "bins": [], "ece": None, "brier": None, "log_loss": None}
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probabilities) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if idx:
            rows.append({"low": lo, "high": hi, "n": len(idx),
                         "mean_probability": sum(probabilities[i] for i in idx) / len(idx),
                         "accuracy": sum(correct[i] for i in idx) / len(idx)})
    ece = sum(r["n"] / n * abs(r["mean_probability"] - r["accuracy"]) for r in rows)
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, correct)) / n
    eps = 1e-12
    log_loss = -sum(math.log(max(p, eps)) if y else math.log(max(1 - p, eps)) for p, y in zip(probabilities, correct)) / n
    return {"n": n, "bins": rows, "ece": ece, "brier": brier, "log_loss": log_loss}


@dataclass(frozen=True)
class Calibration:
    """A fitted calibration, bound to the versions whose outputs it was fit on."""

    version: str
    a: float
    b: float
    threshold: float
    target_risk: float
    alpha: float
    event: str
    interpreter_identity: str
    registry_identity: str
    policy_version: str
    fit_items: int
    fit_identity: str

    def probability(self, score: float) -> float:
        return _sigmoid(self.a * _logit(score) + self.b)

    def accepts(self, score: float) -> bool:
        return self.probability(score) >= self.threshold

    def binding_problems(self, interpreter_identity: str, registry_identity: str, policy_version: str) -> list[str]:
        out = []
        for name, mine, theirs in (("interpreter", self.interpreter_identity, interpreter_identity),
                                   ("registry", self.registry_identity, registry_identity),
                                   ("policy", self.policy_version, policy_version)):
            if mine != theirs:
                out.append(f"calibrated for a different {name}; refit before use")
        return out


def fit(scores: list[float], correct: list[bool], item_ids: list[str], interpreter_identity: str,
        registry_identity: str, policy_version: str, target_risk: float = 0.01, alpha: float = 0.05) -> Calibration:
    """Fit the transform and the threshold on one calibration split."""
    a, b = platt_fit(scores, correct)
    probs = [_sigmoid(a * _logit(s) + b) for s in scores]
    threshold = choose_threshold(probs, correct, target_risk, alpha)
    return Calibration(VERSION, a, b, threshold, target_risk, alpha, EVENT, interpreter_identity, registry_identity,
                       policy_version, len(scores), content_identity(sorted(item_ids)))
