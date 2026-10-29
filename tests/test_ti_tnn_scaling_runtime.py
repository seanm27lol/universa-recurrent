"""The scaling study's lead-side mechanics, on the CPU: the GPU-hours ledger, request records, identities and the Qwen
output parsing.

**The ledger:**
- **Charging:**
  - it refuses a reservation that would exceed the cap, and NaN or negative projections;
  - it charges failures and crashed runs, using monotonic durations recorded in heartbeats;
  - it nests `extra`, so it cannot overwrite the entry's own keys.
- **Shared location and ownership:**
  - it needs one shared, absolute ledger location;
  - only one owner holds the GPU at a time.
- **Enforcement and failure:**
  - it enforces a run's limit with a watchdog, writing exactly one end entry;
  - it fails closed when the heartbeat cannot be written;
  - it reports a half-written ledger line explicitly.

**Records.** These are rejected (the reviewer's probes R1–R16):
- an abstention scored correct; a correctness value that is not the integer 0 or 1;
- duplicated requests; mixed arms or configurations;
- hidden or excess repairs; tool counts not derived from the attempts;
- a final SQL that differs from the last attempt; a score not bound to the submitted SQL;
- an error on an answered request; a timeout flag that disagrees with its reason;
- zero elapsed time; whitespace SQL; non-finite numbers inside attempts; a missing task id.

**Parsing:**
- Qwen's XML tool calls.
- Reasoning split at `</think>`.
- The final SQL from sql/sqlite/unlabeled fences, never other labels; a bare reply's first statement only.

**Identities** are deterministic and refuse types that would make them depend on a repr or on hash order.

**The agent loop** (a scripted model): a reply without a final query uses the one repair; a model that keeps calling
tools gets one `finalize` turn after the budget, then abstains; records with more turns than the budgets allow, or a
mislabelled `finalize` turn, are rejected.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from scaling import contracts as C  # noqa: E402
from scaling import identity, ledger, records  # noqa: E402
from scaling.llm_agent import extract_final, final_sql, strip_tool_calls  # noqa: E402
from scaling.runtime import SAMPLING, GenerationConfig, parse_tool_calls, split_reasoning  # noqa: E402


def _res(tmp_path, **kw):
    return ledger.Reservation(kw.pop("command", "x"), path=tmp_path / "ledger.jsonl", lock=tmp_path / "gpu.lock", **kw)


# --- the ledger -----------------------------------------------------------------------------------------------------
def test_the_ledger_enforces_the_cap_and_charges_failures(tmp_path):
    with _res(tmp_path, projected_hours=1.0, cap_hours=2.0):
        pass
    with pytest.raises(RuntimeError):
        with _res(tmp_path, command="failed run", projected_hours=0.5, cap_hours=2.0):
            raise RuntimeError("boom")
    entries = ledger.read(tmp_path / "ledger.jsonl")
    assert [e["event"] for e in entries] == ["start", "end", "start", "end"]
    assert entries[3]["status"] == "failed"  # a failed attempt is still a charged reservation
    with pytest.raises(ledger.BudgetExceeded):  # 0 h used, 2.5 h projected, 2 h cap
        with _res(tmp_path, command="too big", projected_hours=2.5, cap_hours=2.0):
            pass
    assert sum(e["event"] == "start" for e in ledger.read(tmp_path / "ledger.jsonl")) == 2  # never started


@pytest.mark.parametrize("projected", [float("nan"), -5.0, 0.0, float("inf"), True])
def test_invalid_projections_are_refused(tmp_path, projected):
    with pytest.raises(ValueError):
        _res(tmp_path, projected_hours=projected)


def test_a_reservation_needs_the_one_shared_absolute_location(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "anchor_file", lambda: tmp_path / "anchor")  # never touch the real repository's anchor
    monkeypatch.delenv("TI_TNN_SCALING_RUNS", raising=False)
    with pytest.raises(RuntimeError, match="TI_TNN_SCALING_RUNS"):
        ledger.Reservation("x", projected_hours=0.1)
    monkeypatch.setenv("TI_TNN_SCALING_RUNS", "relative/runs")
    with pytest.raises(RuntimeError, match="absolute"):
        ledger.Reservation("x", projected_hours=0.1)
    monkeypatch.setenv("TI_TNN_SCALING_RUNS", str(tmp_path))
    with ledger.Reservation("x", projected_hours=0.1) as r:
        assert ledger.active() is r and r.path == tmp_path / "gpu_ledger.jsonl"
    assert ledger.active() is None
    assert ledger.read(tmp_path / "gpu_ledger.jsonl")[0]["ledger"] == str(tmp_path / "gpu_ledger.jsonl")
    # the anchor pins that one location for every worktree: another absolute location is refused
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("TI_TNN_SCALING_RUNS", str(other))
    with pytest.raises(RuntimeError, match="anchored"):
        ledger.Reservation("x", projected_hours=0.1)


def test_a_crashed_reservation_is_charged_from_monotonic_heartbeats(tmp_path):
    path = tmp_path / "ledger.jsonl"
    t = 1_000_000.0
    lines = [{"event": "start", "reservation": "r1", "t": t, "elapsed": 0.0, "command": "x"},
             {"event": "heartbeat", "reservation": "r1", "t": t - 3600, "elapsed": 600.0},  # a wall-clock step back
             {"event": "start", "reservation": "r2", "t": t + 7200, "elapsed": 0.0, "command": "y"},
             {"event": "end", "reservation": "r2", "t": t + 5000, "elapsed": 1800.0, "status": "ok"}]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    u = ledger.usage(path)
    r1 = next(r for r in u["reservations"] if r["reservation"] == "r1")
    assert r1["open"] and math.isclose(r1["hours"], (600 + ledger.HEARTBEAT_SECONDS) / 3600)
    assert math.isclose(u["used_hours"], (600 + ledger.HEARTBEAT_SECONDS + 1800) / 3600)


def test_extra_cannot_overwrite_the_entry(tmp_path):
    with _res(tmp_path, projected_hours=0.1, extra={"event": "note", "reservation": "other"}):
        pass
    entries = ledger.read(tmp_path / "ledger.jsonl")
    assert [e["event"] for e in entries] == ["start", "end"] and entries[0]["extra"]["event"] == "note"


def test_only_one_process_holds_the_gpu(tmp_path):
    with _res(tmp_path, command="owner", projected_hours=0.1):
        with pytest.raises(RuntimeError, match="holds the GPU"):
            with _res(tmp_path, command="second", projected_hours=0.1):
                pass


def test_the_watchdog_enforces_the_limit_with_one_end_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "HEARTBEAT_SECONDS", 0.05)
    with pytest.raises(ledger.BudgetExceeded):
        with _res(tmp_path, projected_hours=0.2 / 3600, cap_hours=1.0, terminate=False) as r:
            assert math.isclose(r.limit_hours, 0.3 / 3600)
            time.sleep(0.6)  # work that never calls check() itself
            r.check()
    entries = ledger.read(tmp_path / "ledger.jsonl")
    ends = [e for e in entries if e["event"] == "end"]
    assert len(ends) == 1 and ends[0]["status"] == "budget_exceeded"


def test_a_dead_heartbeat_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "HEARTBEAT_SECONDS", 0.05)
    with pytest.raises(ledger.BudgetExceeded, match="heartbeat"):
        with _res(tmp_path, projected_hours=1.0, terminate=False) as r:
            real = r._append
            r._append = lambda entry: (_ for _ in ()).throw(OSError("disk full")) if entry["event"] == "heartbeat" \
                else real(entry)
            time.sleep(0.3)
            r.check()


STOP_CHILD = r"""
import sys, time
sys.path.insert(0, {root!r})
from scaling import ledger
ledger.HEARTBEAT_SECONDS = 0.1
ledger.TERMINATION_GRACE_SECONDS = 30.0
with ledger.Reservation("child", projected_hours=0.4 / 3600, path={path!r}, lock={lock!r}):
    end = time.monotonic() + 20
    while time.monotonic() < end:
        try:
            time.sleep(0.02)  # work wrapped in a broad handler, like a request loop
        except Exception:
            pass
print("NOT STOPPED")
"""


def test_a_broad_except_cannot_swallow_the_stop_and_the_charge_runs_to_the_real_stop(tmp_path):
    import subprocess
    path, lock = tmp_path / "ledger.jsonl", tmp_path / "gpu.lock"
    code = STOP_CHILD.format(root=str(ROOT), path=str(path), lock=str(lock))
    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    took = time.monotonic() - t0
    assert "NOT STOPPED" not in proc.stdout and "BudgetExceeded" in proc.stderr and took < 15
    ends = [e for e in ledger.read(path) if e["event"] == "end"]
    assert len(ends) == 1 and ends[0]["status"] == "budget_exceeded"
    assert 0.6 <= ends[0]["elapsed"] < 5  # the limit is 0.6 s (0.4 × 1.5); charged to where it actually stopped


def test_a_half_written_line_is_reported(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps({"event": "start", "reservation": "r", "t": 1.0, "elapsed": 0.0}) + "\n{\"event\": \"hea")
    with pytest.raises(ledger.LedgerCorrupt, match="partial final line"):
        ledger.usage(path)


# --- records --------------------------------------------------------------------------------------------------------
def _gen(prompt=100, generated=20, reasoning=0):
    return {"prompt_tokens": prompt, "generated_tokens": generated, "reasoning_tokens": reasoning,
            "content_tokens": generated - reasoning, "seconds": 1.0, "time_to_first_token": 0.1,
            "decode_tokens_per_second": 20.0, "finish_reason": "eos"}


def _rec(task_id, correct=1, sql="SELECT 1", **kw):
    attempts = [{"kind": "generation", "generation": _gen(), "tool_results": [], "final_sql": sql,
                 "final_status": "ok"}]
    r = {"task_id": task_id, "db_id": "d", "arm": "B4", "config_identity": "c", "input_identity": "i",
         "snapshot_sha256": "s", "attempts": attempts,
         "final": {"kind": "sql", "sql": sql, "reason": "final SQL"},
         "score": {"correct": correct, "scored_sql_sha256": records.sql_sha256(sql), "snapshot_sha256": "s"},
         "tokens": {"initial_prompt": 100, "prefill": 100, "generated": 20, "reasoning": 0, "content": 20,
                    "tool_observation": 0},
         "controller_work": None, "elapsed_seconds": 2.0, "peak_memory_bytes": 1, "timed_out": False, "error": None,
         "repairs": 0, "tool_calls": 0}
    r.update(kw)
    return r


def _abstained(task_id, reason="no final SQL in the reply", **kw):
    r = _rec(task_id, correct=0, final={"kind": "abstain", "sql": None, "reason": reason})
    r["attempts"][0]["final_sql"] = None
    r["score"]["scored_sql_sha256"] = records.sql_sha256(None)
    r.update(kw)
    return r


def test_the_summary_counts_every_request_and_the_cost_unit():
    recs = [_rec("a", 1, elapsed_seconds=10.0), _rec("b", 0, elapsed_seconds=30.0),
            _abstained("c", reason="request timeout", timed_out=True, elapsed_seconds=20.0)]
    s = records.summarize(recs, ["a", "b", "c"])
    assert (s["requested"], s["correct"], s["abstained"], s["timeouts"], s["errors"]) == (3, 1, 1, 1, 0)
    assert s["quality"] == pytest.approx(1 / 3) and s["cost_seconds_per_correct"] == pytest.approx(60.0)


def _repaired(task_id):
    r = _rec(task_id)
    r["attempts"] = [{"kind": "generation", "generation": _gen(), "tool_results": [], "final_sql": "SELEC 1",
                      "final_status": "error"},
                     {"kind": "repair", "generation": _gen(), "tool_results": [], "final_sql": "SELECT 1",
                      "final_status": "ok"}]
    r["tokens"].update(prefill=200, generated=40, content=40)
    r["repairs"] = 1
    return r


def _mutated(fn):
    def make(rs):
        r = copy.deepcopy(rs[0])
        fn(r)
        return [r, rs[1]]
    return make


BAD = {
    "R1 abstention scored correct": (_mutated(lambda r: r.update(final={"kind": "abstain", "sql": None, "reason": "x"},
                                                                 score={**r["score"], "scored_sql_sha256": records.sql_sha256(None)})),
                                     "abstention scored correct"),
    "R2 duplicated requested ids": (None, "duplicates"),
    "R3 repair present but not counted": (lambda rs: [_repaired("a") | {"repairs": 0}, rs[1]], "repair"),
    "R4 more repairs than allowed": (_mutated(lambda r: r.update(repairs=3)), "repair"),
    "R5 retry after a failed final not labelled repair": (
        lambda rs: [(lambda r: (r["attempts"][1].update(kind="generation"), r.update(repairs=0)) and r)(_repaired("a")),
                    rs[1]], "labelled repair"),
    "R6 final SQL differs from the last attempt": (_mutated(lambda r: r["attempts"][0].update(final_sql="SELECT 2")),
                                                   "last attempt"),
    "R7 tokens None": (_mutated(lambda r: r.update(tokens=None)), "tokens"),
    "R8 correct is True": (_mutated(lambda r: r["score"].update(correct=True)), "integer 0 or 1"),
    "R8 correct is 1.0": (_mutated(lambda r: r["score"].update(correct=1.0)), "integer 0 or 1"),
    "R9 executed tool calls not counted": (_mutated(lambda r: r["attempts"][0].update(
        tool_results=[{"call": {}, "result": {"status": "ok"}}] * 3)), "tool calls"),
    "R10 timed_out disagrees with the reason": (lambda rs: [_abstained("a", reason="request timeout"), rs[1]],
                                                "timed_out"),
    "R11 zero elapsed": (_mutated(lambda r: r.update(elapsed_seconds=0.0)), "positive"),
    "R13 error on an answered request": (_mutated(lambda r: r.update(error="boom")), "must abstain"),
    "R14 whitespace SQL": (_mutated(lambda r: r["final"].update(sql="   ")), "without SQL"),
    "R15 NaN inside an attempt": (_mutated(lambda r: r["attempts"][0]["generation"].update(seconds=float("nan"))),
                                  "finite"),
    "R16 missing task_id": (_mutated(lambda r: r.pop("task_id")), "task_id"),
    "score not bound to the SQL": (_mutated(lambda r: r["score"].update(scored_sql_sha256="0" * 64)), "submitted SQL"),
    "score not bound to the snapshot": (_mutated(lambda r: r["score"].update(snapshot_sha256="t")), "snapshot"),
    "token totals differ from the turns": (_mutated(lambda r: r["tokens"].update(generated=21)), "tokens.generated"),
    "negative elapsed": (_mutated(lambda r: r.update(elapsed_seconds=-1.0)), "finite"),
    "inf token count": (_mutated(lambda r: r["tokens"].update(prefill=float("inf"))), "finite"),
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_bad_records_are_rejected(name):
    make, match = BAD[name]
    requested = ["a", "b"]
    if make is None:  # duplicated requested ids
        recs, requested = [_rec("a"), _rec("b")], ["a", "a", "b"]
    else:
        recs = make([_rec("a"), _rec("b")])
    with pytest.raises(records.RecordError, match=match):
        records.summarize(recs, requested)


def test_mixed_arms_or_configurations_are_rejected():
    with pytest.raises(records.RecordError, match="one arm and one configuration"):
        records.summarize([_rec("a"), _rec("b", arm="Q27")], ["a", "b"])
    with pytest.raises(records.RecordError, match="one arm and one configuration"):
        records.summarize([_rec("a"), _rec("b", config_identity="other")], ["a", "b"])
    assert records.summarize([_repaired("a"), _rec("b")], ["a", "b"])["repairs"] == 1  # a labelled repair is fine


# --- parsing --------------------------------------------------------------------------------------------------------
def test_tool_calls_and_reasoning_are_parsed():
    content = ("Let me check.\n<tool_call>\n<function=lookup_values>\n<parameter=table>\nteams\n</parameter>\n"
               "<parameter=column>\nname\n</parameter>\n</function>\n</tool_call>")
    assert parse_tool_calls(content) == [{"name": "lookup_values", "arguments": {"table": "teams", "column": "name"}}]
    assert strip_tool_calls(content) == "Let me check."
    assert split_reasoning("think...\n</think>\n\nanswer", True) == ("think...", "answer")
    assert split_reasoning("still thinking", True) == ("still thinking", "")  # truncated mid-thought
    assert split_reasoning("answer", False) == ("", "answer")


@pytest.mark.parametrize("content, expected", [
    ("draft ```sql\nSELECT 1\n``` then ```sql\nSELECT 2;\n```", "SELECT 2"),
    ("```sqlite\nSELECT 1\n```", "SELECT 1"),
    ("```SQL\nSELECT 1\n```", "SELECT 1"),
    ("```\nSELECT 1\n```", "SELECT 1"),
    ("```sql SELECT 1```", "SELECT 1"),
    ("```sqlx\nSELECT 1\n```", None),
    ("```python\nprint(1)\n```", None),
    ("SELECT count(*) FROM teams\n\nThis counts the teams.", "SELECT count(*) FROM teams"),
    ("SELECT name FROM t; SELECT 2", "SELECT name FROM t"),
    ("SELECT motto FROM t WHERE motto = 'a;b'", "SELECT motto FROM t WHERE motto = 'a;b'"),
    ("WITH a AS (SELECT 1)\n\nSELECT * FROM a", "WITH a AS (SELECT 1)\n\nSELECT * FROM a"),
    ("SELECT motto FROM t WHERE motto = 'a;b'; -- done\n\nThat is the motto.", "SELECT motto FROM t WHERE motto = 'a;b'"),
    ("I cannot answer.", None),
])
def test_the_final_sql_is_taken_exactly(content, expected):
    assert final_sql(content) == expected
    sql, raw = extract_final(content)
    assert (raw is None) == (expected is None)


def test_generation_settings_follow_the_model_cards_and_record_the_template_options():
    lim = C.Limits()
    q = GenerationConfig.official("Q27", True, lim, reasoning_effort="low")
    assert q.max_new_tokens == lim.max_new_tokens_thinking and q.sampling["temperature"] == 1.0
    assert q.template_options()["reasoning_effort"] == "low"
    assert GenerationConfig.official("Q27", True, lim).template_options()["reasoning_effort"] == "xhigh"
    assert GenerationConfig.official("Q27", False, lim).template_options()["reasoning_effort"].startswith("not applied")
    assert GenerationConfig.official("B4", False, lim).sampling["presence_penalty"] == 1.5
    for model, thinking in (("B4", True), ("Q27", False)):
        with pytest.raises(ValueError):
            GenerationConfig.official(model, thinking, lim, reasoning_effort="low")
    assert set(SAMPLING) == {(m, t) for m in ("Q27", "B4") for t in (True, False)}


# --- identities -----------------------------------------------------------------------------------------------------
def test_identities_are_deterministic_and_strict():
    assert identity.content_identity({"a": {3, 1, 2}}) == identity.content_identity({"a": [1, 2, 3]})  # sets sorted
    assert identity.content_identity({"a": [3, 1, 2]}) != identity.content_identity({"a": [1, 2, 3]})  # lists ordered
    assert identity.content_identity({"b": 1, "a": (1, 2)}) == identity.content_identity({"a": [1, 2], "b": 1})
    for bad in ({1: "a"}, {"x": object()}, {"x": float("nan")}):
        with pytest.raises(TypeError):
            identity.content_identity(bad)
    assert identity.content_identity({"limits": C.Limits()}) == identity.content_identity({"limits": C.Limits().as_dict()})


def test_threads_do_not_leak_from_reservations(tmp_path):
    before = threading.active_count()
    with _res(tmp_path, projected_hours=0.1):
        pass
    time.sleep(0.05)
    assert threading.active_count() <= before


# --- the agent loop with a scripted model -----------------------------------------------------------------------------
class _ScriptedRuntime:
    """Replies from a script; the last reply repeats. A tool call is Qwen's XML format; anything else is content."""
    device = "cpu"

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def render(self, messages, tools, cfg):
        return json.dumps(messages)

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def generate_turn(self, messages, tools, cfg, seed, deadline, max_new_tokens):
        from scaling.runtime import Turn
        self.seen.append(copy.deepcopy(messages))
        text = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return Turn(text=text, reasoning="", content=text, tool_calls=parse_tool_calls(text), prompt_tokens=50,
                    generated_tokens=10, reasoning_tokens=0, content_tokens=10, finish_reason="eos", stop_token=None,
                    seconds=0.01, time_to_first_token=0.001, decode_tokens_per_second=1000.0)


_CALL = ("<tool_call>\n<function=execute_select>\n<parameter=sql>\nSELECT name FROM t\n</parameter>\n</function>\n"
         "</tool_call>")


def _agent_record(tmp_path, replies):
    import sqlite3
    from scaling import tools
    from scaling.llm_agent import answer_request
    path = tmp_path / "t.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE t (name TEXT)")
        con.execute("INSERT INTO t VALUES ('a')")
    snap = tools.snapshot_for_file("t", path)
    rt = _ScriptedRuntime(replies)
    out = answer_request(rt, C.Task("x1", "t", "q?", "", "dev"), snap, "t(name)", [],
                         GenerationConfig.official("B4", False, C.Limits()), C.Limits(), seed=0)
    sql = out["final"]["sql"]
    rec = {"task_id": "x1", "db_id": "t", "arm": "B4", "config_identity": "c", "input_identity": "i",
           "snapshot_sha256": snap.sha256, "controller_work": None, **out,
           "score": {"correct": 0, "scored_sql_sha256": records.sql_sha256(sql), "snapshot_sha256": snap.sha256}}
    return rec, rt


def test_a_model_that_keeps_calling_tools_gets_one_finalize_turn_then_abstains(tmp_path):
    rec, rt = _agent_record(tmp_path, [_CALL])
    budget = C.Limits().max_tool_calls
    assert rec["tool_calls"] == budget and len(rec["attempts"]) == budget + 1
    assert [a["kind"] for a in rec["attempts"]][-1] == "finalize"
    assert rec["final"] == {"kind": "abstain", "sql": None, "reason": "tool-call budget exhausted without a final answer"}
    assert rt.seen[-1][-1]["role"] == "user" and "budget is exhausted" in rt.seen[-1][-1]["content"]
    records.validate(rec)


def test_the_finalize_turn_can_answer(tmp_path):
    rec, _ = _agent_record(tmp_path, [_CALL] * C.Limits().max_tool_calls + ["```sql\nSELECT name FROM t\n```"])
    assert rec["final"]["kind"] == "sql" and rec["attempts"][-1]["kind"] == "finalize"
    records.validate(rec)


def test_records_reject_turns_beyond_the_budget(tmp_path):
    rec, _ = _agent_record(tmp_path, [_CALL])
    bad = copy.deepcopy(rec)  # a request that went on after calling tools with the budget spent
    extra = copy.deepcopy(bad["attempts"][-1]) | {"kind": "generation", "turn": len(bad["attempts"])}
    bad["attempts"].append(extra)
    for k in ("prefill", "generated", "content"):
        bad["tokens"][k] += extra["generation"][{"prefill": "prompt_tokens", "generated": "generated_tokens",
                                                 "content": "content_tokens"}[k]]
    with pytest.raises(records.RecordError, match="exceed the tool and repair budgets|after the budget"):
        records.validate(bad)
    relabelled = copy.deepcopy(rec)
    relabelled["attempts"][-1]["kind"] = "generation"
    with pytest.raises(records.RecordError, match="not labelled finalize"):
        records.validate(relabelled)
    early = copy.deepcopy(rec)
    early["attempts"][2]["kind"] = "finalize"
    with pytest.raises(records.RecordError, match="labelled finalize but"):
        records.validate(early)


def test_a_reply_without_a_final_query_uses_the_one_repair(tmp_path):
    (tmp_path / "one").mkdir(), (tmp_path / "two").mkdir()
    rec, rt = _agent_record(tmp_path / "one", ["The answer is a.", "```sql\nSELECT name FROM t\n```"])
    assert [a["kind"] for a in rec["attempts"]] == ["generation", "repair"] and rec["repairs"] == 1
    assert rec["final"]["kind"] == "sql" and "no final query" in rt.seen[-1][-1]["content"]
    records.validate(rec)
    rec, _ = _agent_record(tmp_path / "two", ["The answer is a."])
    assert rec["final"] == {"kind": "abstain", "sql": None, "reason": "no final SQL in the reply"}
    assert len(rec["attempts"]) == 2 and rec["repairs"] == 1
    records.validate(rec)
