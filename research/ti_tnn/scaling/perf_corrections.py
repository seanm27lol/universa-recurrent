"""Corrections to the performance study's reporting (2026-11-03 review), computed from the bundled records alone.
The recorded values (in the records and `analysis.json`) are kept as they are; these sit beside them.

**Controller MACs as executed.** Each record's `controller_work` counted, per operator step, the trunk, the semantic
module and *the chosen operator's pointer kind only* (`controller.step_macs`), plus `controller.request_macs` once.
The decoder (`infer_controller.decode_batch`) executes more than that:
- a STOP step runs the trunk (step input, GRU, attention over the memory, state and family classifier), but is not
  charged;
- every operator step calls `pointer_scores` once for all its heads, whatever the operator. That is the table and
  column item scores (2 N d), the span-start scores (T d) and, under v2, the relation terms (2 R d + 2 N R). An
  operator without a pointer is charged 0 for this;
- a span operator then calls `pointer_scores` again for the end: item scores again (2 N d), start (T d), the end
  condition (d²) and the end scores (T d).

Here N is the number of schema items, T the context tokens, d the controller width and R the number of pointer
relation bits. T is the record's `tokens.context`. N follows from the recorded trunk MACs, whose attention term is
2 (T + N) d per operator step. The recorded pointer MACs are then re-derived from (T, N) and must equal the record
exactly, a check on the reconstruction.

Weights only, as before: activations, softmax, masking and embedding lookups are not counted.
"""
from __future__ import annotations

from . import actions as A
from . import controller as K


class MacMismatch(ValueError):
    pass


def executed_macs(record: dict, cfg: K.ControllerConfig) -> dict:
    """{recorded, executed, n_tokens, n_items, stop_steps, op_steps, stop_trunk, pointer_recorded,
    pointer_executed} for one saved controller record (batch size one)."""
    w = record["controller_work"]
    if w.get("batch_size", 1) != 1:
        raise MacMismatch(f"{record['task_id']}: batch size {w['batch_size']}")
    actions = [A.Action.from_json(a) for a in record["attempts"][0]["actions"] or []]
    stops = w["families"].count("STOP")
    ops = w["steps"] - stops
    if ops != len(actions):
        raise MacMismatch(f"{record['task_id']}: {ops} operator steps, {len(actions)} actions")
    d = cfg.d_model
    trunk0 = K.step_macs(cfg, A.STOP, 0, 0)["trunk"]
    T = int(record["tokens"]["context"])
    per_step, rem = divmod(w["macs"]["trunk"], ops) if ops else (trunk0, 0)
    L, rem2 = divmod(per_step - trunk0, 2 * d)
    if rem or rem2:
        raise MacMismatch(f"{record['task_id']}: the recorded trunk MACs give no integral memory length")
    N = L - T
    pointer_recorded = sum(K.step_macs(cfg, 0, T, N, A.POINTER_KIND[a.op])["pointer"] for a in actions)
    if pointer_recorded != w["macs"]["pointer"]:
        raise MacMismatch(f"{record['task_id']}: pointer MACs {w['macs']['pointer']} not re-derived "
                          f"({pointer_recorded}) from T={T}, N={N}")
    R = A.N_POINTER_RELATIONS
    first_call = 2 * N * d + T * d + ((2 * R * d + 2 * N * R) if cfg.grounding == "v2" else 0)
    end_call = 2 * N * d + T * d + d * d + T * d
    pointer_executed = sum(first_call + (end_call if A.POINTER_KIND[a.op] == "span" else 0) for a in actions)
    stop_trunk = stops * K.step_macs(cfg, A.STOP, T, N)["trunk"]
    recorded = w["request_macs"] + sum(w["macs"].values())
    executed = recorded + stop_trunk + pointer_executed - pointer_recorded
    return {"recorded": recorded, "executed": executed, "n_tokens": T, "n_items": N, "stop_steps": stops,
            "op_steps": ops, "stop_trunk": stop_trunk, "pointer_recorded": pointer_recorded,
            "pointer_executed": pointer_executed}


# --- the Qwen timing windows and the machine's load during them -----------------------------------------------------
QWEN_RUNS = ("baseline B4_thinking_p16-64", "baseline Q27_thinking_low_p16-64", "baseline B4_thinking_p64-128",
             "baseline Q27_thinking_low_p64-128")


def qwen_windows(ledger_lines: list) -> dict:
    """{command: (start, end)} in epoch seconds, from the ledger's start and end events (the reservation bounds every
    timed request, its scoring and the model load)."""
    starts, ends = {}, {}
    for e in ledger_lines:
        if e["event"] == "start" and e.get("command") in QWEN_RUNS:
            starts[e["reservation"]] = (e["command"], e["t"])
        elif e["event"] == "end":  # an end event carries only its reservation id
            ends[e["reservation"]] = e["t"]
    return {cmd: (t0, ends[rid]) for rid, (cmd, t0) in starts.items() if rid in ends}


def load_during(log_lines: list, windows: dict) -> dict:
    """The 1-minute load average (all processes on the host, 20 cores) sampled by the co-tenancy loggers, per window:
    samples, mean, median, max and the share of samples above 6."""
    import statistics
    from datetime import datetime
    samples = []
    for line in log_lines:
        p = line.split()
        if len(p) >= 4:
            samples.append((datetime.fromisoformat(p[0].replace("Z", "+00:00")).timestamp(), float(p[1])))
    out = {}
    for cmd, (a, b) in windows.items():
        v = [x for t, x in samples if a <= t <= b]
        out[cmd] = {"samples": len(v), "mean": statistics.mean(v), "median": statistics.median(v), "max": max(v),
                    "share_above_6": sum(x > 6 for x in v) / len(v)} if v else {"samples": 0}
    return out


# --- costs, kept apart ------------------------------------------------------------------------------------------------
def cost_separation(ledger_reservations: list) -> dict:
    """GPU-reservation hours in three kinds that are not added together:
    - setup: what one standalone controller needs before it serves, namely the training pool's backbone states
      (the pilot's completed `pilot-encode train_*` reservations, which every fit reuses) and one fit. The attempt a
      host crash interrupted is research expenditure, reported beside it;
    - research: everything this study and the pilot reserved, failures and smokes included;
    - marginal serving is per request, from the records, not from the ledger."""
    encode = [r for r in ledger_reservations if r["command"].startswith("pilot-encode train") and r["status"] == "ok"]
    interrupted = [r for r in ledger_reservations if r["command"].startswith("pilot-encode train") and r["status"] != "ok"]
    fits = [r for r in ledger_reservations if r["command"].startswith("perf-train")]
    by_label: dict = {}
    for r in fits:
        by_label.setdefault(r["command"].split()[1], []).append(r["hours"])
    v2 = [r for r in ledger_reservations if r.get("authorization") == 2]
    return {"setup": {"training_states_hours": sum(r["hours"] for r in encode), "training_states_reservations": len(encode),
                      "training_states_interrupted_hours": sum(r["hours"] for r in interrupted),
                      "one_fit_hours_mean": {k: sum(v) / len(v) for k, v in sorted(by_label.items())}},
            "research": {"authorization_v2_hours": sum(r["hours"] for r in v2), "authorization_v2_reservations": len(v2),
                         "all_hours": sum(r["hours"] for r in ledger_reservations),
                         "all_reservations": len(ledger_reservations)}}
