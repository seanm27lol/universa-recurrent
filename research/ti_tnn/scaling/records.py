"""Per-request records, their validation, and the run summary with the pre-declared cost unit.

**Example.** On a 16-question run, the summary counts all 16 requests: answers, abstentions, timeouts and failed SQL
alike. Suppose 5 are correct and the 16 requests took 1,040 s end to end in total. Then:
- quality = 5 / 16 = 0.3125;
- serving cost = 1,040 / 5 = 208 s per correctly completed request.

Records are rejected when:
- a requested task is missing or duplicated;
- a numeric field is not a finite, non-negative number;
- an attempt is unaccounted for (the attempt list disagrees with the counted repairs and tool calls);
- an answer has no final SQL, or an abstention has no reason.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import contracts as C


class RecordError(ValueError):
    pass


def _finite(x, name: str, allow_none: bool = False) -> None:
    if x is None and allow_none:
        return
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
        raise RecordError(f"{name}: {x!r} is not a finite non-negative number")


def validate(rec: dict) -> None:
    missing = [k for k in C.RECORD_FIELDS if k not in rec]
    if missing:
        raise RecordError(f"{rec.get('task_id')}: missing fields {missing}")
    f = rec["final"]
    if f.get("kind") not in ("sql", "abstain"):
        raise RecordError(f"{rec['task_id']}: final kind {f.get('kind')!r}")
    if f["kind"] == "sql" and not f.get("sql"):
        raise RecordError(f"{rec['task_id']}: an answer without SQL")
    if f["kind"] == "abstain" and not f.get("reason"):
        raise RecordError(f"{rec['task_id']}: an abstention without a reason")
    _finite(rec["elapsed_seconds"], "elapsed_seconds")
    _finite(rec["peak_memory_bytes"], "peak_memory_bytes", allow_none=True)
    for k, v in (rec.get("tokens") or {}).items():
        _finite(v, f"tokens.{k}")
    if not isinstance(rec["timed_out"], bool):
        raise RecordError(f"{rec['task_id']}: timed_out must be a bool")
    if not isinstance(rec["attempts"], list):
        raise RecordError(f"{rec['task_id']}: attempts must be a list")
    repairs = sum(1 for a in rec["attempts"] if a.get("kind") == "repair")
    if "repairs" in rec and rec["repairs"] != repairs:
        raise RecordError(f"{rec['task_id']}: {rec['repairs']} repairs recorded, {repairs} repair attempts present")
    calls = sum(len(a.get("tool_results", [])) for a in rec["attempts"])
    if "tool_calls" in rec and rec["tool_calls"] > calls:
        raise RecordError(f"{rec['task_id']}: {rec['tool_calls']} tool calls counted, {calls} recorded")
    s = rec["score"]
    if s is not None and s.get("correct") not in (0, 1):
        raise RecordError(f"{rec['task_id']}: score.correct {s.get('correct')!r}")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summarize(records: list[dict], requested: list[str]) -> dict:
    """Every requested task exactly once; quality and cost over all of them."""
    ids = [r["task_id"] for r in records]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise RecordError(f"duplicated requests: {dup[:5]}")
    missing = sorted(set(requested) - set(ids))
    extra = sorted(set(ids) - set(requested))
    if missing or extra:
        raise RecordError(f"requests missing {missing[:5]} / not requested {extra[:5]}")
    for r in records:
        validate(r)
        if r["score"] is None:
            raise RecordError(f"{r['task_id']}: unscored")
    n = len(requested)
    correct = sum(r["score"]["correct"] for r in records)
    elapsed = sum(r["elapsed_seconds"] for r in records)
    reasons: dict[str, int] = {}
    for r in records:
        if r["final"]["kind"] == "abstain":
            reasons[r["final"]["reason"]] = reasons.get(r["final"]["reason"], 0) + 1
    tokens: dict[str, float] = {}
    for r in records:
        for k, v in (r.get("tokens") or {}).items():
            tokens[k] = tokens.get(k, 0) + v
    return {"requested": n, "correct": correct, "quality": correct / n if n else None,
            "answered": sum(r["final"]["kind"] == "sql" for r in records),
            "abstained": sum(r["final"]["kind"] == "abstain" for r in records), "abstention_reasons": reasons,
            "timeouts": sum(r["timed_out"] for r in records),
            "repairs": sum(sum(1 for a in r["attempts"] if a.get("kind") == "repair") for r in records),
            "tool_calls": sum(r.get("tool_calls", 0) for r in records),
            "elapsed_seconds_total": elapsed,
            "cost_seconds_per_correct": (elapsed / correct) if correct else None,  # the primary serving-cost unit
            "tokens": tokens,
            "peak_memory_bytes_max": max((r["peak_memory_bytes"] or 0) for r in records) if records else None}
