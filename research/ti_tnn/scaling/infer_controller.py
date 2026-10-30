"""Greedy legal decoding for C4 and T4: actions -> tirel-1 plan -> SQLite SQL, or an explicit abstention.

**Example.** For a development question the decoder starts from the empty grammar state. At each step it
1. computes the controller state;
2. picks the highest-scoring *legal* family (the grammar's mask);
3. runs the semantic module for that family only (T4: rows of a batch are grouped by family, and each specialist
   runs on its own group);
4. picks the highest-scoring legal operator, then the highest-scoring legal pointer (for a span, the start, then
   the end given that start);
5. applies the action to the grammar machine.

When the stack of unfinished subproblems is empty, STOP is the only legal family. The plan is then validated against
the schema and compiled to SQL.

**No search, no repair.** No top-k, no beam, no sampling, no second attempt (`contracts.Limits.max_repairs_controller
= 0`). Abstentions are explicit and stay in every denominator:

| Reason | When |
|---|---|
| `budget_exhausted` | `max_steps` decisions without completing the plan |
| `invalid_plan` | the completed plan fails `ir.validate` |
| `unsupported_operation` | an action the machine cannot apply (should not happen under the masks; recorded if it does) |
| `dead_end` | the grammar offers no legal family, operator target or span end (should not happen; recorded if it does) |
| `error: <type>` | an exception while encoding or decoding this request; the evaluation goes on with the next |
| `context_too_long` | the request does not fit the backbone's token limit (raised by `encode`) |

**Work accounting** per request: steps, the family chosen at each step, the head executed at each step (T4: that
family's specialist; C4: "shared"; STOP: none), and multiply-accumulates split into trunk, semantic module and
pointer scoring (`controller.step_macs`) plus the once-per-request memory projections (`controller.request_macs`).

**Batch equivalence.** Every operation is row-independent, so decoding a batch gives each row the same decisions as
decoding it alone. The tests check this on CPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from . import actions as A
from . import controller as K
from . import ir

INFER_VERSION = "infer-v1"
DEFAULT_MAX_STEPS = 256


@dataclass
class Decoded:
    task_id: str
    actions: list
    plan: object = None
    sql: "str | None" = None
    abstain_reason: "str | None" = None
    detail: str = ""
    work: dict = field(default_factory=dict)

    def final(self) -> dict:
        if self.sql is not None:
            return {"kind": "sql", "sql": self.sql, "reason": None}
        return {"kind": "abstain", "sql": None, "reason": self.abstain_reason, "detail": self.detail}

    def as_record(self) -> dict:
        return {"task_id": self.task_id, "final": self.final(), "actions": [a.as_json() for a in self.actions],
                "plan": ir.to_json(self.plan) if self.plan is not None else None, "controller_work": self.work}


def _step_tensors(rows: list, device) -> dict:
    """One decoding step's trunk inputs for every row (finished rows get their last inputs; their outputs are
    ignored)."""
    keys = ("prev_op", "prev_family", "prev_kind", "prev_a", "prev_b", "slot", "parent")
    out = {k: torch.tensor([[r[k]] for r in rows], dtype=torch.long, device=device) for k in keys}
    out["features"] = torch.tensor([[r["features"]] for r in rows], dtype=torch.float32, device=device)
    return out


@torch.no_grad()
def decode_batch(model: K.Controller, examples: list, *, max_steps: int = DEFAULT_MAX_STEPS) -> list:
    """Greedy legal decoding of a batch of `controller.Prepared` (no gold is read)."""
    model.eval()
    device = next(model.parameters()).device
    B = len(examples)
    batch = K.collate([K.Prepared(**{**e.__dict__, "steps": None}) for e in examples], device=device)
    t_start = time.perf_counter()
    mem = model.trunk.memory(batch)
    machines = [A.Machine(e.schema, e.spans, max_steps=max_steps) for e in examples]
    results = [Decoded(task_id=e.task_id, actions=[]) for e in examples]
    for r, e in zip(results, examples):
        r.work = {"steps": 0, "families": [], "heads": [], "macs": {"trunk": 0, "semantic": 0, "pointer": 0},
                  "request_macs": K.request_macs(model.cfg, e.n_tokens, e.items.shape[0]),
                  "arm": model.cfg.arm}
    active = [True] * B
    rows = []
    for m in machines:
        f = m.features()
        rows.append({"prev_op": K.BOS_OP, "prev_family": K.BOS_FAMILY, "prev_kind": 0, "prev_a": 0, "prev_b": 0,
                     "slot": f["slot"], "parent": f["parent_op"], "features": f["vector"]})
    hidden = None
    for _ in range(max_steps + 1):
        if not any(active):
            break
        state, hidden = model.trunk.states(mem, _step_tensors(rows, device), hidden)
        state = state[:, 0]
        fam_logits = model.trunk.family_head(state)
        masks = [m.legal_families() for m in machines]
        for b in range(B):
            if active[b] and not any(masks[b]):  # the grammar offers nothing: abstain, never guess
                results[b].abstain_reason, results[b].detail = "dead_end", f"no legal family at step {machines[b].steps}"
                active[b] = False
        legal_fam = torch.tensor(masks, dtype=torch.bool, device=device)
        family = fam_logits.masked_fill(~legal_fam, K.NEG).argmax(-1)
        live = [b for b in range(B) if active[b]]
        op_rows = [b for b in live if int(family[b]) < A.STOP]
        for b in live:
            if int(family[b]) == A.STOP:
                active[b] = False
                results[b].work["families"].append("STOP")
                results[b].work["heads"].append(None)
                results[b].work["steps"] += 1
        if not any(active):
            break
        if not op_rows:
            continue
        index = torch.tensor(op_rows, dtype=torch.long, device=device)
        op_logits, q_rows = model.semantic(state.index_select(0, index), family.index_select(0, index))
        queries = state.new_zeros((B, 1, K.N_POINTERS, model.cfg.d_model))
        queries[index, 0] = q_rows
        chosen = {}
        for k, b in enumerate(op_rows):
            legal = torch.zeros(K.UNION_SIZE, dtype=torch.bool, device=device)
            legal[machines[b].legal_union_ids()] = True
            uid = int(op_logits[k].masked_fill(~legal, K.NEG).argmax())
            fam, op = A.UNION[uid]
            if fam != int(family[b]):
                raise RuntimeError("operator outside the chosen family")
            chosen[b] = op
        starts = torch.zeros((B, 1), dtype=torch.long, device=device)
        items, start, _ = model.trunk.pointer_scores(mem, queries)
        args = {}
        dead = set()
        for b, op in chosen.items():
            kind = A.POINTER_KIND[op]
            if kind is None:
                args[b] = None
                continue
            try:
                _, targets = machines[b].pointer_targets(op)
            except (KeyError, TypeError, ValueError) as error:
                targets, detail = (), f"{op}: {type(error).__name__}: {error}"
            else:
                detail = f"{op}: no legal {kind} target"
            if not targets:
                results[b].abstain_reason, results[b].detail = "dead_end", detail[:300]
                dead.add(b)
                continue
            if kind in ("table", "column"):
                scores = items[b, 0, 0 if kind == "table" else 1]
                mask = torch.zeros_like(scores, dtype=torch.bool)
                mask[list(targets)] = True
                args[b] = int(scores.masked_fill(~mask, K.NEG).argmax())
            else:
                scores = start[b, 0]
                mask = torch.zeros_like(scores, dtype=torch.bool)
                mask[list(targets)] = True
                starts[b, 0] = int(scores.masked_fill(~mask, K.NEG).argmax())
        span_rows = [b for b, op in chosen.items() if A.POINTER_KIND[op] == "span" and b not in dead]
        if span_rows:
            _, _, end = model.trunk.pointer_scores(mem, queries, starts)
            for b in span_rows:
                s = int(starts[b, 0])
                try:
                    ends = machines[b].span_ends(chosen[b], s)
                except (KeyError, TypeError, ValueError, AttributeError) as error:
                    ends, detail = (), f"{chosen[b]}: {type(error).__name__}: {error}"
                else:
                    detail = f"{chosen[b]}: no legal span end after {s}"
                if not ends:
                    results[b].abstain_reason, results[b].detail = "dead_end", detail[:300]
                    dead.add(b)
                    continue
                scores = end[b, 0]
                mask = torch.zeros_like(scores, dtype=torch.bool)
                mask[list(ends)] = True
                args[b] = (s, int(scores.masked_fill(~mask, K.NEG).argmax()))
        heads = model.semantic.executed(family.index_select(0, index).tolist())
        for k, b in enumerate(op_rows):
            fam = int(family[b])
            res = results[b]
            if b in dead:  # the decision was made and its work is counted; the row then abstains
                res.work["steps"] += 1
                res.work["families"].append(A.FAMILIES[fam])
                res.work["heads"].append(A.FAMILIES[heads[k]] if isinstance(heads[k], int) else heads[k])
                active[b] = False
                continue
            action = A.Action(chosen[b], args[b])
            m = machines[b]
            res.work["steps"] += 1
            res.work["families"].append(A.FAMILIES[fam])
            res.work["heads"].append(A.FAMILIES[heads[k]] if isinstance(heads[k], int) else heads[k])
            macs = K.step_macs(model.cfg, fam, examples[b].n_tokens, examples[b].items.shape[0],
                               A.POINTER_KIND[action.op])
            for key in ("trunk", "semantic", "pointer"):
                res.work["macs"][key] += macs[key]
            try:
                m.apply(action)
            except (A.IllegalAction, KeyError, IndexError) as error:
                res.abstain_reason, res.detail = "unsupported_operation", f"{action.op}: {error}"[:300]
                active[b] = False
                continue
            res.actions.append(action)
            if m.steps >= max_steps and not m.finished:
                res.abstain_reason, res.detail = "budget_exhausted", f"{max_steps} steps"
                active[b] = False
                continue
            kind, a0, b0 = K._arg_fields(action)
            f = m.features()
            rows[b] = {"prev_op": A.UNION_INDEX[(fam, action.op)], "prev_family": fam, "prev_kind": kind,
                       "prev_a": a0, "prev_b": b0, "slot": f["slot"], "parent": f["parent_op"],
                       "features": f["vector"]}
    elapsed = time.perf_counter() - t_start
    for b, (res, m) in enumerate(zip(results, machines)):
        res.work["batch_seconds"] = elapsed
        res.work["batch_size"] = B
        if res.abstain_reason is not None:
            continue
        if active[b] or not m.finished:
            res.abstain_reason, res.detail = "budget_exhausted", f"{max_steps} steps"
            continue
        res.plan = m.plan
        errors = ir.validate(m.plan, examples[b].schema.index)
        if errors:
            res.abstain_reason, res.detail = "invalid_plan", "; ".join(errors[:3])
            continue
        res.sql = ir.compile(m.plan)
    return results


def decode_one(model: K.Controller, example, **kw) -> Decoded:
    return decode_batch(model, [example], **kw)[0]


def evaluate(model: K.Controller, task_ids, backbone, *, score: bool = True, max_steps: int = DEFAULT_MAX_STEPS,
             cache_root=None, use_cache: bool = True, log=print) -> list:
    """Decode development tasks at batch size one (the serving-cost unit), each with its own encoding cost, and
    optionally score them with the benchmark scorer. Gold SQL is read only by the scorer. With `use_cache=False` every
    request runs its own backbone forward (timed serving); `states_sha256` (computed after the timing) lets a second
    pass check it decoded from the same states."""
    import hashlib

    from . import data, encode, retrieval, scorer
    from . import schema as schema_mod
    out = []
    for n, tid in enumerate(task_ids):
        task = data.task(tid)
        t0 = time.perf_counter()
        record = {"task_id": tid, "db_id": task.db_id, "arm": model.cfg.arm, "infer_version": INFER_VERSION,
                  "error": None}
        enc = None
        try:
            enc = encode.encode_request(backbone, task, schema_mod.load_schema(task.db_id), retrieval.retrieve(task),
                                        cache_root=cache_root, use_cache=use_cache)
        except encode.ContextTooLong as error:
            record.update(final={"kind": "abstain", "sql": None, "reason": "context_too_long",
                                 "detail": str(error)[:300]}, controller_work=None, encode_cost=None)
        except Exception as error:  # one request's failure is its own abstention, never the whole evaluation's
            record.update(final={"kind": "abstain", "sql": None, "reason": f"error: {type(error).__name__}",
                                 "detail": str(error)[:300]}, controller_work=None, encode_cost=None,
                          error=f"{type(error).__name__}: {error}"[:1000])
        else:
            try:
                dec = decode_one(model, K.prepare(enc), max_steps=max_steps)
            except Exception as error:
                record.update(final={"kind": "abstain", "sql": None, "reason": f"error: {type(error).__name__}",
                                     "detail": str(error)[:300]}, controller_work=None,
                              error=f"{type(error).__name__}: {error}"[:1000])
            else:
                record.update(dec.as_record())
            record["encode_cost"] = enc.cost
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            torch.cuda.synchronize()
        record["elapsed_seconds"] = time.perf_counter() - t0
        record["states_sha256"] = (hashlib.sha256(enc.states.contiguous().view(torch.uint8).numpy().tobytes())
                                   .hexdigest() if enc is not None else None)
        if score:
            snap = data.snapshot(task.db_id)
            record["score"] = scorer.score(snap, record["final"]["sql"], data.gold(tid).sql)
        out.append(record)
        if log:
            log(f"{tid}: {record['final']['kind']} {record['final'].get('reason') or ''} "
                f"correct={record.get('score', {}).get('correct')}")
    return out
