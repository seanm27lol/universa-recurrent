"""Exact run inventories for frozen evaluators (review of plan PRs 4-8, 2026-10-18).

An evaluator must judge exactly the cases its protocol registered: every expected case once, and nothing
else. A missing case (a crashed or interrupted run), a duplicate (a rerun appended to the same file) or an
unexpected case (another world, init or variant) makes the result incomplete. The predictions are then
not evaluated at all. Without this check, a file holding a fraction of the cases can still be judged
complete.
"""
from __future__ import annotations

from collections import Counter


def problems(keys, expected) -> dict:
    """{} if `keys` is exactly `expected` with no repeats; otherwise what is missing, repeated or unexpected."""
    counts = Counter(keys)
    expected = set(expected)
    missing = sorted(expected - set(counts), key=repr)
    duplicated = sorted((k for k, c in counts.items() if c > 1), key=repr)
    unexpected = sorted(set(counts) - expected, key=repr)
    if not (missing or duplicated or unexpected):
        return {}
    return {"missing": missing[:10], "missing_count": len(missing), "duplicated": duplicated[:10],
            "duplicated_count": len(duplicated), "unexpected": unexpected[:10], "unexpected_count": len(unexpected)}
