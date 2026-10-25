"""How far a recomputed statistic may differ from a stored one, and nothing else.

**Example.** The factored pilot's unit (world 100, init 100, full exposure) stores A's historical S2 bound:
- 11 errors in 785 answered questions;
- one-sided 95% Clopper–Pearson upper bound 0.02308779216470323.

The producer computed it with CPython 3.11.14. CPython 3.12.3, on the same machine and from the same two counts,
recomputes 0.02308779216470245, which is 7.8e-16 lower. Nothing in the evidence differs:
- the bound is a root, found by bisection on a binomial tail;
- the tail is a floating-point sum of exponentials of `lgamma` differences;
- 3.12's `sum()` rounds that sum differently (compensated summation), and another platform's `log`, `exp` or
  `lgamma` can move it the same way.

Over the pilot's 8 units, CPython 3.12.3 recomputes 723 of the stored bounds and p-values differently, by at most
8.5e-13. Exact equality rejects every one.

**The rule.** Only numbers under a key in `STATISTICAL_KEYS` may differ. These are exact binomial bounds and
p-values, so every one must be a probability:
- **Valid on both sides first.** It must be a number, not a bool, finite, and in [0, 1] (`probability`). This is
  checked before equality or tolerance. An infinity, a NaN, an out-of-range value or a bool is a mismatch, even
  when both sides hold the same one. Otherwise `inf <= inf` would let an infinity pass the tolerance, and equal
  infinities would never reach it.
- **Then within the tolerance:**

      |stored − recomputed| ≤ max(REL · max(|stored|, |recomputed|), ABS),   REL = 1e-9, ABS = 1e-15.

Everything else is compared exactly:
- counts;
- model scores and probabilities read from records;
- thresholds, alpha and delta;
- decisions, including whether each p-value rejected;
- inventories, identities and hashes.

**Why this size.** Against a 50-digit reference (mpmath), every bound and p-value stored in the factored-pilot,
S2-diagnosis and interpreter-robustness bundles errs by at most 5.2e-13 absolute and 6.8e-12 relative. The relative
maximum is a lower interval end near 0. So two correct runtimes differ by at most about 1.4e-11 relative, and REL
is 70 times that. ABS covers a value that is exactly 0 on one runtime and a rounding residue (about 1e-16) on
another. The figures are reported to 4 decimals, so any change visible there is at least 5e-5: 50,000 times the
largest permitted difference, 1e-9.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

STATISTICAL_KEYS = frozenset({"p_value", "error_upper_95", "abstention_interval_95",
                              "S2_error_upper_95", "S3_error_upper_95"})
REL, ABS = 1e-9, 1e-15


def probability(x) -> bool:
    """A statistic this module may compare: a number (not a bool), finite, and in [0, 1]."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= x <= 1.0 and math.isfinite(x)


def close(stored: float, recomputed: float) -> bool:
    """Both values are probabilities (`probability`), within the declared tolerance of each other."""
    if not (probability(stored) and probability(recomputed)):
        return False
    return abs(stored - recomputed) <= max(REL * max(abs(stored), abs(recomputed)), ABS)


@dataclass
class Agreement:
    """The outcome of one comparison, and running totals when one object is reused across comparisons."""
    mismatches: list = field(default_factory=list)
    exact_values: int = 0          # leaves compared exactly
    statistical_values: int = 0    # numbers under a statistical key
    statistical_differing: int = 0  # ... that differed, within the tolerance
    largest_difference: float = 0.0

    def compare(self, stored, recomputed, path: str = "", keys=STATISTICAL_KEYS) -> bool:
        """True when `recomputed` agrees with `stored`: exactly, except values below a key in `keys`. Those must be
        probabilities on both sides (`probability`) and may then differ within `close`. Keys, lengths and every
        other value must be equal. Mismatches are recorded by path; the totals accumulate."""
        before = len(self.mismatches)
        self._walk(stored, recomputed, path, False, keys)
        return len(self.mismatches) == before

    def _walk(self, a, b, path, statistical, keys):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                self.mismatches.append(f"{path or '/'}: keys present on one side only: {sorted(set(a) ^ set(b))[:5]}")
            for k in sorted(set(a) & set(b), key=str):
                self._walk(a[k], b[k], f"{path}/{k}", statistical or k in keys, keys)
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                self.mismatches.append(f"{path}: {len(a)} stored entries, {len(b)} recomputed")
            for i, (x, y) in enumerate(zip(a, b)):
                self._walk(x, y, f"{path}[{i}]", statistical, keys)
        elif statistical:
            self.statistical_values += 1
            invalid = [side for side, v in (("stored", a), ("recomputed", b)) if not probability(v)]
            if invalid:  # before equality: equal infinities, NaNs, bools or out-of-range values do not pass
                self.mismatches.append(f"{path}: stored {a!r}, recomputed {b!r} "
                                       f"({' and '.join(invalid)} not a finite probability in [0, 1])")
                return
            if a != b:
                if not close(a, b):
                    self.mismatches.append(f"{path}: stored {a!r}, recomputed {b!r} (beyond the statistical tolerance)")
                    return
                self.statistical_differing += 1
                self.largest_difference = max(self.largest_difference, abs(a - b))
        else:
            self.exact_values += 1
            if a != b:
                self.mismatches.append(f"{path}: stored {a!r}, recomputed {b!r}")

    def summary(self) -> dict:
        return {"exact_values": self.exact_values, "statistical_values": self.statistical_values,
                "statistical_differing": self.statistical_differing, "largest_difference": self.largest_difference,
                "tolerance": {"rel": REL, "abs": ABS, "keys": sorted(STATISTICAL_KEYS)}}


def require(stored, recomputed, what: str, agreement: Agreement | None = None) -> Agreement:
    """Raise ValueError unless `recomputed` agrees with `stored` (see `Agreement.compare`)."""
    agreement = agreement if agreement is not None else Agreement()
    before = len(agreement.mismatches)
    if not agreement.compare(stored, recomputed):
        new = agreement.mismatches[before:]
        raise ValueError(f"{what} differs from the rebuilt value ({len(new)} differences; first: {new[0]})")
    return agreement
