"""The ordinary LLM arms, B4 (Qwen3.5-4B) and Q27 (Qwen3.8-27B): SQL generation with the shared tools, bounded repair
and complete per-request records.

**Example.** For one development request, the model sees:
- the question and BIRD's evidence;
- the serialized schema;
- three retrieved training examples (from other databases, with their SQL);
- three tools: `describe_schema`, `lookup_values` and `execute_select`.

A typical run of turns:
1. The model calls `lookup_values` to check how a team name is spelt.
2. It tests a query with `execute_select` and sees 20 preview rows plus the full row count.
3. It replies with the final SQL in a ```sql``` block.

The harness then executes that final SQL once, as the answer. If it fails (an error, a timeout or a rejected
statement), the model gets one repair turn with the error. Everything is recorded:
- every turn's reasoning and content;
- every tool call and its observation;
- the repair;
- the final SQL or the abstention and its reason;
- token counts, elapsed time, peak memory and timeouts.

The scorer later re-executes the final SQL **in full** against the pinned snapshot. The 20-row preview limits only
what the model sees.

**The same for both models:** the prompt, the tools, the budgets (`contracts.Limits`), the repair policy and the
recording. Only the model, its template options (thinking, reasoning effort) and the official sampling differ.
"""
from __future__ import annotations

import json
import re
import time

from . import contracts as C
from .runtime import GenerationConfig, Runtime

SYSTEM = (
    "You answer questions about a SQLite database by writing one read-only SQLite query.\n"
    "- Use the schema, the evidence and the tools. `execute_select` runs a SELECT statement on the database and shows "
    "the first rows and the total row count; `lookup_values` shows distinct values of a column; `describe_schema` "
    "shows tables in detail.\n"
    "- Return exactly the columns the question asks for, in the order it asks for them.\n"
    "- Only SELECT statements (WITH and set operations included) are allowed; the database is read-only.\n"
    "- When you are done, reply with the final query in a ```sql``` code block as the last thing in your reply."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "describe_schema", "description": "Describe tables of the database: columns, types, keys and descriptions.",
        "parameters": {"type": "object", "properties": {"tables": {"type": "array", "items": {"type": "string"},
                                                                   "description": "table names; omit for all tables"}},
                       "required": []}}},
    {"type": "function", "function": {
        "name": "lookup_values", "description": "Distinct values of one column, optionally filtered by a substring.",
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string"}, "column": {"type": "string"},
            "pattern": {"type": "string", "description": "case-insensitive substring to match (optional)"}},
            "required": ["table", "column"]}}},
    {"type": "function", "function": {
        "name": "execute_select", "description": "Run one read-only SELECT statement and see the first rows.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}},
]

FINAL_SQL = re.compile(r"```sql\s*(.*?)```", re.S | re.I)


def user_message(task: C.Task, schema_text: str, examples: list) -> str:
    parts = [f"Question: {task.question}", f"Evidence: {task.evidence or '(none)'}", "", "Database schema:", schema_text]
    if examples:
        parts += ["", "Examples from other databases (question, evidence and a correct query):"]
        for i, e in enumerate(examples, 1):
            parts += [f"Example {i}. Question: {e.question}", f"Evidence: {e.evidence or '(none)'}", f"SQL: {e.sql}"]
    return "\n".join(parts)


def final_sql(content: str) -> str | None:
    blocks = FINAL_SQL.findall(content)
    if blocks:
        return blocks[-1].strip().rstrip(";").strip() or None
    stripped = content.strip()
    if re.match(r"(?is)^(select|with)\b", stripped):
        return stripped.rstrip(";").strip()
    return None


def strip_tool_calls(content: str) -> str:
    return re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.S).strip()


def run_tool(snapshot: C.Snapshot, call: dict, limits: C.Limits) -> dict:
    from . import tools
    name, args = call["name"], call.get("arguments", {})
    try:
        if name == "describe_schema":
            tables = args.get("tables")
            if isinstance(tables, str):
                try:
                    tables = json.loads(tables)
                except json.JSONDecodeError:
                    tables = [t.strip() for t in tables.split(",") if t.strip()]
            return tools.describe_schema(snapshot, tables=tables)
        if name == "lookup_values":
            return tools.lookup_values(snapshot, args.get("table", ""), args.get("column", ""),
                                       pattern=args.get("pattern") or None, limit=limits.lookup_max_values)
        if name == "execute_select":
            return tools.execute_select(snapshot, args.get("sql", ""), preview_rows=limits.preview_rows)
        return {"status": "error", "error": f"unknown tool {name!r}"}
    except Exception as e:  # a malformed call is an observation for the model, and it is recorded
        return {"status": "error", "error": f"{type(e).__name__}: {e}"[:500]}


def observation(result: dict, limit_chars: int = 6000) -> str:
    text = json.dumps(result, default=str, ensure_ascii=False)
    return text if len(text) <= limit_chars else text[:limit_chars] + " …(truncated)"


def answer_request(rt: Runtime, task: C.Task, snapshot: C.Snapshot, schema_text: str, examples: list,
                   cfg: GenerationConfig, limits: C.Limits, seed: int) -> dict:
    """One request, end to end. Returns the record's `attempts`, `final`, `tokens` and timing fields."""
    import torch
    from . import tools
    timeout = limits.request_timeout_seconds_27b if cfg.model == "Q27" else limits.request_timeout_seconds_4b
    max_total = limits.max_generated_tokens_thinking if cfg.thinking else limits.max_generated_tokens_nonthinking
    t0 = time.time()
    deadline = t0 + timeout
    if rt.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(task, schema_text, examples)}]
    turns, tool_calls_used, repairs, pending_repair = [], 0, 0, False
    tokens = {"prompt": 0, "generated": 0, "reasoning": 0, "content": 0}
    final, stop_reason = None, None
    while True:
        remaining = max_total - tokens["generated"]
        if remaining <= 0:
            stop_reason = "generated-token budget exhausted"
            break
        if time.time() > deadline:
            stop_reason = "request timeout"
            break
        prompt_len = rt.count_tokens(rt.render(messages, TOOLS, cfg))
        if prompt_len + min(remaining, cfg.max_new_tokens) > limits.max_context_tokens:
            stop_reason = "context limit"
            break
        turn = rt.generate_turn(messages, TOOLS, cfg, seed=seed + len(turns), deadline=deadline,
                                max_new_tokens=remaining)
        for k, v in (("prompt", turn.prompt_tokens), ("generated", turn.generated_tokens),
                     ("reasoning", turn.reasoning_tokens), ("content", turn.content_tokens)):
            tokens[k] += v
        entry = {"turn": len(turns), "kind": "repair" if pending_repair else "generation",
                 "generation": turn.as_dict(), "tool_results": [], "final_sql": None, "final_status": None}
        pending_repair = False
        turns.append(entry)
        if turn.finish_reason == "timeout":
            stop_reason = "request timeout"
            break
        if turn.tool_calls:
            assistant = {"role": "assistant", "content": strip_tool_calls(turn.content),
                         "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                        for c in turn.tool_calls]}
            if turn.reasoning:
                assistant["reasoning_content"] = turn.reasoning
            messages.append(assistant)
            for call in turn.tool_calls:
                if tool_calls_used >= limits.max_tool_calls:
                    result = {"status": "rejected", "error": "the tool-call budget is exhausted; give the final SQL now"}
                else:
                    tool_calls_used += 1
                    result = run_tool(snapshot, call, limits)
                entry["tool_results"].append({"call": call, "result": result})
                messages.append({"role": "tool", "content": observation(result)})
            continue
        if turn.finish_reason == "max_new_tokens":
            stop_reason = "turn truncated at the token limit without a final answer"
            break
        sql = final_sql(turn.content)
        entry["final_sql"] = sql
        if sql is None:
            stop_reason = "no final SQL in the reply"
            break
        check = tools.execute_select(snapshot, sql, preview_rows=limits.preview_rows)
        entry["final_status"] = check.get("status")
        entry["final_check"] = {k: check.get(k) for k in ("status", "error", "row_count", "columns", "elapsed_seconds")}
        if check.get("status") == "ok" or repairs >= limits.max_repairs_llm:
            final = sql
            stop_reason = "final SQL" if check.get("status") == "ok" else "final SQL failed after the repair budget"
            break
        repairs += 1
        pending_repair = True
        assistant = {"role": "assistant", "content": turn.content}
        if turn.reasoning:
            assistant["reasoning_content"] = turn.reasoning
        messages += [assistant, {"role": "user", "content": (
            f"Your final query failed: {check.get('status')}: {check.get('error')}. "
            "Reply with a corrected final query in a ```sql``` code block.")}]
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() if rt.device.startswith("cuda") else None
    return {"attempts": turns, "final": {"kind": "sql", "sql": final, "reason": stop_reason} if final is not None
            else {"kind": "abstain", "sql": None, "reason": stop_reason},
            "tokens": tokens, "tool_calls": tool_calls_used, "repairs": repairs, "elapsed_seconds": elapsed,
            "peak_memory_bytes": peak, "timed_out": stop_reason == "request timeout"}
