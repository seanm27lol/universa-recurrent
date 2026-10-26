"""Per-request records, their validation, and the run summary with the pre-declared cost unit.

**Example.** A 16-question run produces 16 records: answers, abstentions, timeouts and failed SQL alike. If 5 are
correct and the requests took 1,040 s end to end in total:
- quality = 5 / 16 = 0.3125;
- serving cost = 1,040 / 5 = 208 s per correctly completed request.

**Rejected records:**
- **Wrong set of requests:** a requested task is missing, extra or duplicated.
- **Mixed runs:** two arms or two configurations in one summary.
- **Bad numbers:** a number is not finite and non-negative; a token or step count is not an integer; elapsed time is not
  positive.
- **Bad scores:** a score that is not exactly the integer 0 or 1; an abstention scored correct; a score not bound to
  the exact submitted SQL and snapshot.
- **Hidden retries or edits:**
  - a counted repair or tool call not present in the attempts, or the reverse;
  - more repairs than the arm allows;
  - a failed final check not followed by a labelled repair;
  - a final SQL that differs from the last attempt's;
  - token totals that differ from the sum over turns.
- **Inconsistent outcomes:** a timeout flag that disagrees with the final reason; an error on a request that answered.

**Arm-specific attempt shapes:**
- **Q27 and B4:** one attempt per assistant turn (`llm_agent`).
- **L:** one attempt per retrieved template, with its candidates; every tool result is listed in `tool_log`.
- **C4 and T4:** one decode attempt, and no tools.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import contracts as C
from .identity import sha256_bytes

LLM_ARMS, CONTROLLER_ARMS = ("B4", "Q27"), ("C4", "T4", "C4_active", "C4_params")
REPAIR_LIMIT = {"B4": C.Limits().max_repairs_llm, "Q27": C.Limits().max_repairs_llm, "L": 0}


class RecordError(ValueError):
    pass


def _finite(x, name: str, allow_none: bool = False, integer: bool = False, positive: bool = False) -> None:
    if x is None and allow_none:
        return
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
        raise RecordError(f"{name}: {x!r} is not a finite non-negative number")
    if integer and not isinstance(x, int):
        raise RecordError(f"{name}: {x!r} is not an integer")
    if positive and x <= 0:
        raise RecordError(f"{name}: {x!r} is not positive")


def sql_sha256(sql) -> str:
    return sha256_bytes((sql or "").encode())


def _validate_llm(rec: dict, tid: str) -> None:
    attempts = rec["attempts"]
    executed = sum(1 for a in attempts for t in a.get("tool_results", [])
                   if (t.get("result") or {}).get("status") != "rejected")
    if rec["tool_calls"] != executed:
        raise RecordError(f"{tid}: {rec['tool_calls']} tool calls counted, {executed} executed in the attempts")
    repairs = sum(1 for a in attempts if a.get("kind") == "repair")
    if rec["repairs"] != repairs:
        raise RecordError(f"{tid}: {rec['repairs']} repairs counted, {repairs} repair attempts present")
    for i, a in enumerate(attempts[:-1]):
        if a.get("final_status") not in (None, "ok") and attempts[i + 1].get("kind") != "repair":
            raise RecordError(f"{tid}: attempt {i} failed its final check but the next attempt is not a labelled repair")
    if rec["final"]["kind"] == "sql":
        last = attempts[-1] if attempts else {}
        if last.get("final_sql") != rec["final"]["sql"]:
            raise RecordError(f"{tid}: the final SQL differs from the last attempt's")
    sums = {"generated": 0, "prefill": 0, "reasoning": 0, "content": 0}
    for i, a in enumerate(attempts):
        g = a.get("generation") or {}
        for k in ("prompt_tokens", "generated_tokens", "reasoning_tokens", "content_tokens"):
            _finite(g.get(k), f"{tid}: attempt {i} {k}", integer=True)
        for k in ("seconds", "time_to_first_token", "decode_tokens_per_second"):
            _finite(g.get(k), f"{tid}: attempt {i} {k}", allow_none=True)
        sums["generated"] += g["generated_tokens"]
        sums["prefill"] += g["prompt_tokens"]
        sums["reasoning"] += g["reasoning_tokens"]
        sums["content"] += g["content_tokens"]
    t = rec["tokens"]
    for k, v in sums.items():
        if t.get(k) != v:
            raise RecordError(f"{tid}: tokens.{k} = {t.get(k)!r}, the turns sum to {v}")
    if attempts and t.get("initial_prompt") != attempts[0]["generation"]["prompt_tokens"]:
        raise RecordError(f"{tid}: tokens.initial_prompt differs from the first turn's prompt")
    if rec["timed_out"] != (rec["final"].get("reason") == "request timeout"):
        raise RecordError(f"{tid}: timed_out disagrees with the final reason {rec['final'].get('reason')!r}")


def _validate_l(rec: dict, tid: str) -> None:
    log = rec.get("tool_log")
    if not isinstance(log, list) or rec["tool_calls"] != len(log):
        raise RecordError(f"{tid}: L's tool_calls must equal the number of entries in tool_log")
    if rec["final"]["kind"] == "sql":
        ran = [c["sql"] for a in rec["attempts"] for c in a.get("candidates", []) if c.get("status") == "ok"]
        if rec["final"]["sql"] not in ran:
            raise RecordError(f"{tid}: L's final SQL is not a candidate that ran")


def validate(rec: dict) -> None:
    tid = rec.get("task_id")
    if not isinstance(tid, str) or not tid:
        raise RecordError(f"a record without a task_id: {str(rec)[:120]}")
    missing = [k for k in C.RECORD_FIELDS + ("repairs", "tool_calls") if k not in rec]
    if missing:
        raise RecordError(f"{tid}: missing fields {missing}")
    arm = rec["arm"]
    f = rec["final"]
    if f.get("kind") not in ("sql", "abstain"):
        raise RecordError(f"{tid}: final kind {f.get('kind')!r}")
    if f["kind"] == "sql" and not (isinstance(f.get("sql"), str) and f["sql"].strip()):
        raise RecordError(f"{tid}: an answer without SQL")
    if f["kind"] == "abstain" and not (isinstance(f.get("reason"), str) and f["reason"].strip()):
        raise RecordError(f"{tid}: an abstention without a reason")
    if rec["error"] is not None and f["kind"] != "abstain":
        raise RecordError(f"{tid}: a request with an error must abstain")
    _finite(rec["elapsed_seconds"], f"{tid}: elapsed_seconds", positive=True)
    _finite(rec["peak_memory_bytes"], f"{tid}: peak_memory_bytes", allow_none=True, integer=True)
    if not isinstance(rec["tokens"], dict):
        raise RecordError(f"{tid}: tokens must be a dict")
    for k, v in rec["tokens"].items():
        _finite(v, f"{tid}: tokens.{k}", integer=True)
    if not isinstance(rec["timed_out"], bool):
        raise RecordError(f"{tid}: timed_out must be a bool")
    if not isinstance(rec["attempts"], list):
        raise RecordError(f"{tid}: attempts must be a list")
    _finite(rec["repairs"], f"{tid}: repairs", integer=True)
    _finite(rec["tool_calls"], f"{tid}: tool_calls", integer=True)
    limit = REPAIR_LIMIT.get(arm, C.Limits().max_repairs_controller)
    if rec["repairs"] > limit:
        raise RecordError(f"{tid}: {rec['repairs']} repairs exceed {arm}'s limit of {limit}")
    if rec["tool_calls"] > C.Limits().max_tool_calls:
        raise RecordError(f"{tid}: {rec['tool_calls']} tool calls exceed the budget")
    s = rec["score"]
    if s is not None:
        if type(s.get("correct")) is not int or s["correct"] not in (0, 1):
            raise RecordError(f"{tid}: score.correct must be the integer 0 or 1, not {s.get('correct')!r}")
        if f["kind"] == "abstain" and s["correct"] != 0:
            raise RecordError(f"{tid}: an abstention scored correct")
        if s.get("scored_sql_sha256") != sql_sha256(f.get("sql")):
            raise RecordError(f"{tid}: the score is not bound to the submitted SQL")
        if s.get("snapshot_sha256") != rec["snapshot_sha256"]:
            raise RecordError(f"{tid}: the score is not bound to the record's snapshot")
    if arm in LLM_ARMS:
        _validate_llm(rec, tid)
    elif arm == "L":
        _validate_l(rec, tid)
    elif arm in CONTROLLER_ARMS:
        if rec["tool_calls"] != 0:
            raise RecordError(f"{tid}: the controllers use no tools")
        if rec["controller_work"] is None:
            raise RecordError(f"{tid}: a controller record without controller_work")
    else:
        raise RecordError(f"{tid}: unknown arm {arm!r}")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summarize(records: list[dict], requested: list[str]) -> dict:
    """Every requested task exactly once, from one arm and one configuration; quality and cost over all of them."""
    if len(set(requested)) != len(requested):
        raise RecordError("the requested list has duplicates")
    for r in records:
        if not isinstance(r.get("task_id"), str):
            raise RecordError(f"a record without a task_id: {str(r)[:120]}")
    ids = [r["task_id"] for r in records]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise RecordError(f"duplicated requests: {dup[:5]}")
    missing = sorted(set(requested) - set(ids))
    extra = sorted(set(ids) - set(requested))
    if missing or extra:
        raise RecordError(f"requests missing {missing[:5]} / not requested {extra[:5]}")
    arms = {r.get("arm") for r in records}
    configs = {r.get("config_identity") for r in records}
    if len(arms) > 1 or len(configs) > 1:
        raise RecordError(f"one summary must cover one arm and one configuration: arms {sorted(map(str, arms))}, "
                          f"{len(configs)} configurations")
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
    tokens: dict[str, int] = {}
    for r in records:
        for k, v in r["tokens"].items():
            tokens[k] = tokens.get(k, 0) + v
    return {"arm": next(iter(arms)) if records else None, "config_identity": next(iter(configs)) if records else None,
            "requested": n, "correct": correct, "quality": correct / n if n else None,
            "answered": sum(r["final"]["kind"] == "sql" for r in records),
            "abstained": sum(r["final"]["kind"] == "abstain" for r in records), "abstention_reasons": reasons,
            "errors": sum(r["error"] is not None for r in records),
            "timeouts": sum(r["timed_out"] for r in records),
            "repairs": sum(r["repairs"] for r in records), "tool_calls": sum(r["tool_calls"] for r in records),
            "elapsed_seconds_total": elapsed,
            "cost_seconds_per_correct": (elapsed / correct) if correct else None,  # the primary serving-cost unit
            "tokens": tokens,
            "peak_memory_bytes_max": max((r["peak_memory_bytes"] or 0) for r in records) if records else None}
