"""G4 and the direct-generation control: one bounded call to the pinned Qwen3.5-4B (grounding-interface study).

**G4.** The pinned Qwen3.5-4B checkpoint (`contracts.MODELS["B4"]`) answers one task-specific grounding call, which
yields a `gi_record` record, not SQL. The call is configured as follows:
- non-thinking mode;
- the model card's non-thinking sampling (`runtime.SAMPLING[("B4", False)]`), seeded per request;
- at most `MAX_NEW_TOKENS` generated tokens;
- no tools and no repair call.

It sees exactly the B4 harness's user message: the question, the evidence, the serialized schema
(`schema.serialize`, the same budget), and the retrieved training examples from other databases with their
queries. A truncated, malformed or invalid reply falls back to the original policy, and the call is charged in full.

**The direct control.** The same model and configuration, one call with no tools and no repair, asked for SQL
directly. It measures the same extra pretrained-model computation spent on SQL instead of grounding. It is a
short-budget control, not the thinking-mode B4 of the performance study.

Neither call receives gold SQL, a reference answer or evaluator metadata.
"""
from __future__ import annotations

import hashlib
import time

from . import gi_record as GR
from . import llm_agent

PROMPT_VERSION = "gi-g4-prompt-v2"
DIRECT_VERSION = "gi-direct-prompt-v1"
MAX_NEW_TOKENS = 512

# v1 (first development run): the same instructions without the last three sentences and without "always all four".
# v2 was selected on the 128 development questions: v1 left out "status" in 7 replies and quoted names in 3.
SYSTEM_GROUND = (
    "You link a question about a SQLite database to the database schema. Do not write SQL, and do not explain.\n"
    "Reply with one JSON object and nothing else, with exactly these four keys, always all four:\n"
    '- "tables": the names of the tables a query answering the question needs, most important first;\n'
    '- "columns": the columns that query needs in any clause (selected, filtered, joined, grouped or sorted), '
    'most important first, each as {"table": ..., "column": ...};\n'
    '- "literals": the values in the question or evidence that the query compares columns with, each as '
    '{"span": the words of the question or evidence, "value": the value alone as stored in the database, without '
    'an operator, "table": ..., "column": ...};\n'
    '- "status": "ok", or "unresolved" if the question cannot be linked to this schema.\n'
    "Use the exact table and column names of the schema, without quotes or backticks. List only tables and columns "
    "that appear in the schema. Include \"status\" even when it is \"ok\"."
)

SYSTEM_DIRECT = (
    "You answer questions about a SQLite database by writing one read-only SQLite query.\n"
    "- Use the schema and the evidence. No tools are available.\n"
    "- Return exactly the columns the question asks for, in the order it asks for them.\n"
    "- Reply with the final query in a ```sql``` code block."
)


def seed_of(kind: str, task_id: str) -> int:
    return int(hashlib.sha256(f"{kind}:{task_id}".encode()).hexdigest()[:8], 16)


def config(limits):
    """The bounded call's generation configuration: B4, non-thinking, the card's sampling, MAX_NEW_TOKENS."""
    from dataclasses import replace

    from .runtime import GenerationConfig
    return replace(GenerationConfig.official("B4", False, limits), max_new_tokens=MAX_NEW_TOKENS)


def messages(system: str, task, schema_text: str, examples: list) -> list:
    return [{"role": "system", "content": system},
            {"role": "user", "content": llm_agent.user_message(task, schema_text, examples)}]


def _turn_record(turn, seconds: float) -> dict:
    return {"content": turn.content, "reasoning": turn.reasoning, "finish_reason": turn.finish_reason,
            "prompt_tokens": turn.prompt_tokens, "generated_tokens": turn.generated_tokens, "seconds": seconds}


def ground_call(rt, cfg, task, schema_text: str, examples: list, items) -> dict:
    """One G4 call: the raw reply, the parsed and resolved record, and its time (the whole call, generation and
    parsing)."""
    t0 = time.perf_counter()
    turn = rt.generate_turn(messages(SYSTEM_GROUND, task, schema_text, examples), None, cfg,
                            seed_of("G4", task.task_id))
    obj, reason = GR.parse(turn.content)
    truncated = turn.finish_reason != "eos"
    g = GR.Grounding(False, reason) if obj is None and not truncated else \
        GR.resolve(obj, items, task.question, task.evidence, truncated=truncated)
    seconds = time.perf_counter() - t0
    return {"task_id": task.task_id, "call": _turn_record(turn, seconds), "record": obj, "grounding": g.as_dict(),
            "seconds": seconds}


def direct_call(rt, cfg, task, schema_text: str, examples: list) -> dict:
    """One direct SQL call: the final query extracted as the B4 harness does (`llm_agent.extract_final`)."""
    t0 = time.perf_counter()
    turn = rt.generate_turn(messages(SYSTEM_DIRECT, task, schema_text, examples), None, cfg,
                            seed_of("B4-direct", task.task_id))
    sql, _raw = llm_agent.extract_final(turn.content)
    seconds = time.perf_counter() - t0
    return {"task_id": task.task_id, "call": _turn_record(turn, seconds), "sql": sql,
            "final": {"kind": "sql" if sql else "abstain", "sql": sql,
                      "reason": None if sql else ("truncated" if turn.finish_reason != "eos" else "no SQL in reply")},
            "seconds": seconds}
