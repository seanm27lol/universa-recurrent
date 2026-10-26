"""The scaling study's lead-side mechanics, on the CPU: the GPU-hours ledger, request records, and the Qwen output
parsing (tool calls, reasoning, final SQL).

- **The ledger** refuses a reservation that would exceed the cap. It charges a crashed reservation up to its last
  heartbeat plus one interval, and allows only one owner at a time.
- **Records**:
  - a missing, duplicated or unscored request is rejected;
  - so is a non-finite or negative number;
  - so is an attempt list that disagrees with the counted repairs or tool calls (a hidden retry);
  - the cost unit is total seconds over correct requests.
- **Parsing** reads Qwen's XML tool-call format, splits reasoning from content at `</think>`, and takes the last ```sql
  block as the final query.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from scaling import contracts as C  # noqa: E402
from scaling import ledger, records  # noqa: E402
from scaling.llm_agent import final_sql, strip_tool_calls  # noqa: E402
from scaling.runtime import SAMPLING, GenerationConfig, parse_tool_calls, split_reasoning  # noqa: E402


# --- the ledger -----------------------------------------------------------------------------------------------------
def test_the_ledger_enforces_the_cap_and_charges_failures(tmp_path):
    path, lock = tmp_path / "ledger.jsonl", tmp_path / "gpu.lock"
    with ledger.Reservation("a", projected_hours=1.0, cap_hours=2.0, path=path, lock=lock):
        pass
    with pytest.raises(RuntimeError):
        with ledger.Reservation("failed run", projected_hours=0.5, cap_hours=2.0, path=path, lock=lock):
            raise RuntimeError("boom")
    entries = ledger.read(path)
    assert [e["event"] for e in entries] == ["start", "end", "start", "end"]
    assert entries[3]["status"] == "failed"  # a failed attempt is still a charged reservation
    with pytest.raises(ledger.BudgetExceeded):  # 0 h used, 2.5 h projected, 2 h cap
        with ledger.Reservation("too big", projected_hours=2.5, cap_hours=2.0, path=path, lock=lock):
            pass
    assert sum(e["event"] == "start" for e in ledger.read(path)) == 2  # the refused one never started


def test_a_crashed_reservation_is_charged_to_its_last_heartbeat(tmp_path):
    path = tmp_path / "ledger.jsonl"
    t = 1_000_000.0
    lines = [{"event": "start", "reservation": "r1", "t": t, "command": "x"},
             {"event": "heartbeat", "reservation": "r1", "t": t + 600},
             {"event": "start", "reservation": "r2", "t": t + 7200, "command": "y"},
             {"event": "end", "reservation": "r2", "t": t + 9000, "status": "ok"}]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    u = ledger.usage(path, now=t + 100_000)
    r1 = next(r for r in u["reservations"] if r["reservation"] == "r1")
    assert r1["open"] and math.isclose(r1["hours"], (600 + ledger.HEARTBEAT_SECONDS) / 3600)
    assert math.isclose(u["used_hours"], (600 + ledger.HEARTBEAT_SECONDS + 1800) / 3600)


def test_only_one_process_holds_the_gpu(tmp_path):
    path, lock = tmp_path / "ledger.jsonl", tmp_path / "gpu.lock"
    with ledger.Reservation("owner", projected_hours=0.1, path=path, lock=lock):
        with pytest.raises(RuntimeError, match="holds the GPU"):
            with ledger.Reservation("second", projected_hours=0.1, path=path, lock=lock):
                pass


# --- records --------------------------------------------------------------------------------------------------------
def _rec(task_id, correct=1, **kw):
    r = {"task_id": task_id, "db_id": "d", "arm": "B4", "config_identity": "c", "input_identity": "i",
         "snapshot_sha256": "s", "attempts": [{"kind": "generation", "tool_results": []}],
         "final": {"kind": "sql", "sql": "SELECT 1", "reason": "final SQL"}, "score": {"correct": correct},
         "tokens": {"prompt": 10, "generated": 5}, "controller_work": None, "elapsed_seconds": 2.0,
         "peak_memory_bytes": 1, "timed_out": False, "error": None, "repairs": 0, "tool_calls": 0}
    r.update(kw)
    return r


def test_the_summary_counts_every_request_and_the_cost_unit():
    recs = [_rec("a", 1, elapsed_seconds=10.0), _rec("b", 0, elapsed_seconds=30.0),
            _rec("c", 0, final={"kind": "abstain", "sql": None, "reason": "request timeout"}, timed_out=True,
                 elapsed_seconds=20.0)]
    s = records.summarize(recs, ["a", "b", "c"])
    assert (s["requested"], s["correct"], s["abstained"], s["timeouts"]) == (3, 1, 1, 1)
    assert s["quality"] == pytest.approx(1 / 3) and s["cost_seconds_per_correct"] == pytest.approx(60.0)


@pytest.mark.parametrize("bad, match", [
    (lambda rs: rs[:1], "missing"),
    (lambda rs: rs + [rs[0]], "duplicated"),
    (lambda rs: [_rec("a", elapsed_seconds=float("nan")), rs[1]], "finite"),
    (lambda rs: [_rec("a", elapsed_seconds=-1.0), rs[1]], "finite"),
    (lambda rs: [_rec("a", tokens={"generated": float("inf")}), rs[1]], "finite"),
    (lambda rs: [_rec("a", repairs=1), rs[1]], "repair"),           # a repair counted but not recorded
    (lambda rs: [_rec("a", tool_calls=3), rs[1]], "tool calls"),    # tool calls counted but not recorded
    (lambda rs: [_rec("a", score=None), rs[1]], "unscored"),
    (lambda rs: [_rec("a", final={"kind": "abstain", "sql": None, "reason": ""}), rs[1]], "reason"),
])
def test_bad_records_are_rejected(bad, match):
    with pytest.raises(records.RecordError, match=match):
        records.summarize(bad([_rec("a"), _rec("b")]), ["a", "b"])


# --- parsing --------------------------------------------------------------------------------------------------------
def test_tool_calls_reasoning_and_final_sql_are_parsed():
    content = ("Let me check.\n<tool_call>\n<function=lookup_values>\n<parameter=table>\nteams\n</parameter>\n"
               "<parameter=column>\nname\n</parameter>\n</function>\n</tool_call>")
    assert parse_tool_calls(content) == [{"name": "lookup_values", "arguments": {"table": "teams", "column": "name"}}]
    assert strip_tool_calls(content) == "Let me check."
    assert split_reasoning("think...\n</think>\n\nanswer", True) == ("think...", "answer")
    assert split_reasoning("still thinking", True) == ("still thinking", "")  # truncated mid-thought
    assert split_reasoning("answer", False) == ("", "answer")
    assert final_sql("draft ```sql\nSELECT 1\n``` then ```sql\nSELECT 2;\n```") == "SELECT 2"
    assert final_sql("SELECT name FROM t;") == "SELECT name FROM t" and final_sql("I cannot answer.") is None


def test_generation_settings_follow_the_model_cards_and_limits():
    lim = C.Limits()
    q = GenerationConfig.official("Q27", True, lim, reasoning_effort="low")
    assert q.max_new_tokens == lim.max_new_tokens_thinking and q.sampling["temperature"] == 1.0
    assert GenerationConfig.official("B4", False, lim).sampling["presence_penalty"] == 1.5
    with pytest.raises(ValueError):
        GenerationConfig.official("B4", True, lim, reasoning_effort="low")
    assert set(SAMPLING) == {(m, t) for m in ("Q27", "B4") for t in (True, False)}
