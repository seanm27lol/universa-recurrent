"""The ordinary LLM arms, B4 (Qwen3.5-4B) and Q27 (Qwen3.8-27B): SQL generation with the shared tools, bounded repair
and complete per-request records.

**Example.** For one development request the model sees:
- the question and BIRD's evidence;
- the serialized schema;
- three retrieved training examples (from other databases, with their SQL);
- three tools: `describe_schema`, `lookup_values` and `execute_select`.

It might call `lookup_values` to see how a team name is spelt, then test a query with `execute_select` (20 preview
rows plus the full row count), then reply with the final query in a ```sql``` block. The harness executes that final
SQL once, as the answer. If it fails (an error, a timeout or a rejected statement), the model gets one repair turn
with the error.

**What is recorded:**
- every turn's reasoning and content;
- every tool call and its observation;
- the repair;
- the final SQL (with the raw block it was taken from) or the abstention and its reason;
- token counts, elapsed time, peak memory and the timeout status.

**What is scored.** The scorer re-executes the final SQL in full against the pinned snapshot. The 20-row preview limits
only what the model sees.

**Limits** (`contracts.Limits`):
- The request deadline is checked before every turn, every tool call and the final check.
- Each SQL tool gets min(30 s, time remaining).
- A prefill or a single decode step cannot be interrupted, so the elapsed time can exceed the timeout by at most that
  step.
- A request that reaches its deadline before a final answer is accepted abstains with "request timeout", even if it had
  proposed SQL.
- An exception at any point returns the attempts made so far, the error, and an abstention ("error: <type>"). The
  caller never re-invokes a request.

**Token counts:**

| Field | Meaning |
|---|---|
| `initial_prompt` | the first turn's prompt |
| `prefill` | prompt tokens processed over all turns; each turn re-reads the conversation |
| `generated` | all turns, stop tokens included |
| `reasoning` | generated tokens before `</think>` |
| `content` | the rest of the generated tokens |
| `tool_observation` | tokens of the tool results fed back to the model |

**The same for both models:** the prompt, the tools, the budgets, the repair policy and the recording. Only the model,
its template options (thinking, reasoning effort) and the official sampling differ.
"""
from __future__ import annotations

import json
import re
import sys
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

# The last fenced block that is unlabeled or labeled sql/sqlite (and nothing longer, e.g. not ```sqlx or ```python).
FENCE = re.compile(r"```(sqlite|sql)?(?![A-Za-z0-9_+-])[ \t]*\n?(.*?)```", re.S | re.I)


def user_message(task: C.Task, schema_text: str, examples: list) -> str:
    parts = [f"Question: {task.question}", f"Evidence: {task.evidence or '(none)'}", "", "Database schema:", schema_text]
    if examples:
        parts += ["", "Examples from other databases (question, evidence and a correct query):"]
        for i, e in enumerate(examples, 1):
            parts += [f"Example {i}. Question: {e.question}", f"Evidence: {e.evidence or '(none)'}", f"SQL: {e.sql}"]
    return "\n".join(parts)


def _first_statement(text: str) -> str:
    """A bare reply's first SQL statement, exactly as written.

    1. The text is cut at the first semicolon outside string literals (sqlglot's tokenizer).
    2. The longest prefix ending at a blank line, or at the end, that parses as exactly one SQLite statement is kept.
       So prose after the query is dropped, and a CTE broken by a blank line is kept whole.
    3. If no prefix parses, the first paragraph is returned; it will fail at execution and may be repaired."""
    import sqlglot
    from sqlglot.tokens import TokenType
    try:
        for tok in sqlglot.Dialect.get_or_raise("sqlite").tokenize(text):
            if tok.token_type == TokenType.SEMICOLON:
                text = text[:tok.start]
                break
    except Exception:
        pass
    cuts = [len(text)] + sorted((m.start() for m in re.finditer(r"\n\s*\n", text)), reverse=True)
    for cut in cuts:
        candidate = text[:cut].strip()
        try:
            if candidate and len([e for e in sqlglot.parse(candidate, read="sqlite") if e is not None]) == 1:
                return candidate
        except Exception:
            continue
    return re.split(r"\n\s*\n", text, maxsplit=1)[0].strip()


def extract_final(content: str) -> tuple[str | None, str | None]:
    """(final SQL, the raw text it was taken from). The SQL is the last sql/sqlite/unlabeled fenced block, or, for a
    bare reply that starts with SELECT/WITH, its first statement (`_first_statement`). Only surrounding whitespace and
    a trailing semicolon are removed from a fenced block."""
    blocks = FENCE.findall(content)
    if blocks:
        raw = blocks[-1][1]
        sql = raw.strip().rstrip(";").strip()
        return (sql or None), raw
    stripped = content.strip()
    if re.match(r"(?is)^(select|with)\b", stripped):
        return (_first_statement(stripped) or None), stripped
    return None, None


def final_sql(content: str) -> str | None:
    return extract_final(content)[0]


def strip_tool_calls(content: str) -> str:
    return re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.S).strip()


def run_tool(snapshot: C.Snapshot, call: dict, limits: C.Limits, timeout: float) -> dict:
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
                                       pattern=args.get("pattern") or None, limit=limits.lookup_max_values,
                                       timeout_seconds=timeout)
        if name == "execute_select":
            return tools.execute_select(snapshot, args.get("sql", ""), preview_rows=limits.preview_rows,
                                        timeout_seconds=timeout)
        return {"status": "error", "error": f"unknown tool {name!r}"}
    except Exception as e:  # a malformed call is an observation for the model, and it is recorded
        return {"status": "error", "error": f"{type(e).__name__}: {e}"[:500]}


def observation(result: dict, limit_chars: int = 6000) -> str:
    text = json.dumps(result, default=str, ensure_ascii=False)
    return text if len(text) <= limit_chars else text[:limit_chars] + " …(truncated)"


def progress(event: dict) -> None:
    """One line per turn and per tool call on stderr, so a long request shows where its time goes (no effect on
    behaviour or records)."""
    print(json.dumps(event, default=str), file=sys.stderr, flush=True)


def answer_request(rt: Runtime, task: C.Task, snapshot: C.Snapshot, schema_text: str, examples: list,
                   cfg: GenerationConfig, limits: C.Limits, seed: int) -> dict:
    """One request after retrieval and schema serialisation (the caller times those too). Returns the record's
    `attempts`, `final`, `tokens`, counters and timing fields. Never raises for a request-level failure."""
    import torch
    from . import tools
    timeout = limits.request_timeout_seconds_27b if cfg.model == "Q27" else limits.request_timeout_seconds_4b
    max_total = limits.max_generated_tokens_thinking if cfg.thinking else limits.max_generated_tokens_nonthinking
    m0 = time.monotonic()
    deadline = m0 + timeout
    cuda = rt.device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(task, schema_text, examples)}]
    turns: list[dict] = []
    tool_calls_used, repairs, pending_repair = 0, 0, False
    tokens = {"initial_prompt": 0, "prefill": 0, "generated": 0, "reasoning": 0, "content": 0, "tool_observation": 0}
    final, stop_reason, error = None, None, None
    remaining_time = lambda: deadline - time.monotonic()  # noqa: E731
    try:
        while True:
            remaining = max_total - tokens["generated"]
            if remaining <= 0:
                stop_reason = "generated-token budget exhausted"
                break
            if remaining_time() <= 0:
                stop_reason = "request timeout"
                break
            prompt_len = rt.count_tokens(rt.render(messages, TOOLS, cfg))
            if prompt_len + min(remaining, cfg.max_new_tokens) > limits.max_context_tokens:
                stop_reason = "context limit"
                break
            turn = rt.generate_turn(messages, TOOLS, cfg, seed=seed + len(turns), deadline=deadline,
                                    max_new_tokens=remaining)
            if not turns:
                tokens["initial_prompt"] = turn.prompt_tokens
            tokens["prefill"] += turn.prompt_tokens
            for k, v in (("generated", turn.generated_tokens), ("reasoning", turn.reasoning_tokens),
                         ("content", turn.content_tokens)):
                tokens[k] += v
            entry = {"turn": len(turns), "kind": "repair" if pending_repair else "generation",
                     "generation": turn.as_dict(), "tool_results": [], "final_sql": None, "final_raw": None,
                     "final_status": None}
            pending_repair = False
            turns.append(entry)
            progress({"task": task.task_id, "turn": entry["turn"], "kind": entry["kind"], "finish": turn.finish_reason,
                      "prompt": turn.prompt_tokens, "generated": turn.generated_tokens,
                      "reasoning": turn.reasoning_tokens, "seconds": round(turn.seconds, 1),
                      "ttft": round(turn.time_to_first_token or 0, 1), "tool_calls": len(turn.tool_calls),
                      "elapsed": round(time.monotonic() - m0, 1)})
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
                    if remaining_time() <= 0:
                        result = {"status": "rejected", "error": "the request deadline has passed"}
                    elif tool_calls_used >= limits.max_tool_calls:
                        result = {"status": "rejected", "error": "the tool-call budget is exhausted; give the final SQL now"}
                    else:
                        tool_calls_used += 1
                        result = run_tool(snapshot, call, limits, timeout=min(limits.sql_timeout_seconds,
                                                                               max(0.01, remaining_time())))
                    obs = observation(result)
                    tokens["tool_observation"] += rt.count_tokens(obs)
                    entry["tool_results"].append({"call": call, "result": result})
                    progress({"task": task.task_id, "tool": call["name"], "status": result.get("status"),
                              "seconds": result.get("elapsed_seconds"), "elapsed": round(time.monotonic() - m0, 1)})
                    messages.append({"role": "tool", "content": obs})
                if remaining_time() <= 0:
                    stop_reason = "request timeout"
                    break
                continue
            if turn.finish_reason in ("max_new_tokens", "budget"):
                stop_reason = "turn truncated at the token limit without a final answer"
                break
            sql, raw = extract_final(turn.content)
            entry["final_sql"], entry["final_raw"] = sql, raw
            if sql is None:
                stop_reason = "no final SQL in the reply"
                break
            if remaining_time() <= 0:
                stop_reason = "request timeout"
                break
            check = tools.execute_select(snapshot, sql, preview_rows=limits.preview_rows,
                                         timeout_seconds=min(limits.sql_timeout_seconds, max(0.01, remaining_time())))
            entry["final_status"] = check.get("status")
            entry["final_check"] = {k: check.get(k) for k in ("status", "error", "row_count", "columns", "elapsed_seconds")}
            if check.get("status") == "ok" or repairs >= limits.max_repairs_llm:
                final = sql
                stop_reason = "final SQL" if check.get("status") == "ok" else "final SQL failed after the repair budget"
                break
            if remaining_time() <= 0:
                stop_reason = "request timeout"
                break
            repairs += 1
            pending_repair = True
            assistant = {"role": "assistant", "content": turn.content}
            if turn.reasoning:
                assistant["reasoning_content"] = turn.reasoning
            messages += [assistant, {"role": "user", "content": (
                f"Your final query failed: {check.get('status')}: {check.get('error')}. "
                "Reply with a corrected final query in a ```sql``` code block.")}]
    except Exception as e:  # keep every attempt made so far; the request abstains with the error
        error = f"{type(e).__name__}: {e}"[:1000]
        final, stop_reason = None, f"error: {type(e).__name__}"
    elapsed = time.monotonic() - m0
    peak = torch.cuda.max_memory_allocated() if cuda else None
    timed_out = stop_reason == "request timeout"
    return {"attempts": turns,
            "final": {"kind": "sql", "sql": final, "reason": stop_reason} if final is not None
            else {"kind": "abstain", "sql": None, "reason": stop_reason},
            "tokens": tokens, "tool_calls": tool_calls_used, "repairs": repairs, "elapsed_seconds": elapsed,
            "peak_memory_bytes": peak, "timed_out": timed_out, "error": error,
            "template_options": cfg.template_options()}
