"""CPU-only diagnosis of the pilot's saved records and final checkpoints, before any intervention (performance
study v2). No GPU, no training, and no question outside the pilot's 64 and the training pool.

**Example.** Seed 101's T4 answered `bird23tf-03528` ("the risk category with the most low-risk violations") with

    SELECT T1.risk_category FROM violations AS T1 INNER JOIN violations AS T2 ON T1.business_id = T1.business_id ...

The ON predicate compares one column reference with itself. It resolves to `violations.business_id` of source 0
only, so it is `business_id IS NOT NULL` and leaves source 1 unconstrained: every qualifying row of T1 is paired with
every row of T2, and the query timed out. The diagnosis establishes this from the resolved columns, not from the
text, and asks for each such predicate what the gold program joined instead.

**What it measures**, for the four fits (T4 and C4_params, seeds 100 and 101):
1. **The 64 development questions,** from the saved evaluation records: for each request, the predicted plan against
   the gold plan, component by component (tables, join keys, WHERE conjuncts and literals, aggregates, GROUP BY,
   HAVING, SELECT items, DISTINCT, ORDER BY, LIMIT), the first divergence of the action sequence from the gold
   traversal (classified by clause, decision and family), and every join predicate's semantics (same reference,
   same source, cross source; declared foreign key or not; what the gold program joined at that position).
2. **A fixed training-pool sample:** free (greedy) decoding from the final checkpoint on the cached backbone states,
   scored with the pinned EX rule on the training databases, against teacher-forced loss and teacher-forced
   per-step accuracy on the same examples. The same teacher-forced measures are taken on the 58 representable
   development questions.
3. **Execution:** timeouts and their association with the join predicates; EX-correct answers whose program is
   semantically suspicious.

Decoding here is float32 on the CPU, against TF32 on the GPU in the pilot. The independent re-decode found the two
identical on all 256 development requests; this module re-checks that for the development records it re-decodes.

**Limits.** It reads development gold, as the pilot's analysis did. It must not be used to choose anything on the
new evaluation population (inventory positions 64-127), which it never reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import actions as A
from . import contracts as C
from . import ir

ARMS = ("T4", "c4_params")
SEEDS = (100, 101)
TRAIN_SAMPLE_SALT = "perf-v2-diagnosis-train-sample-v1"
TRAIN_SAMPLE_SIZE = 256


# --- rendering plans with resolved columns ---------------------------------------------------------------------------
class Scope:
    """The FROM lists of the enclosing SELECTs, innermost last: each entry is (table name or None, occurrence)."""

    def __init__(self, frames=()):
        self.frames = tuple(frames)

    def push(self, select: ir.Select) -> "Scope":
        seen: Counter = Counter()
        frame = []
        for item in select.from_:
            if isinstance(item.source, ir.TableSource):
                t = item.source.table.lower()
                frame.append((t, seen[t]))
                seen[t] += 1
            else:
                frame.append((None, seen["<derived>"]))
                seen["<derived>"] += 1
        return Scope(self.frames + (tuple(frame),))

    def resolve(self, col: ir.Column) -> tuple:
        """(table, occurrence, level, source) of a column reference."""
        frame = self.frames[-1 - col.level]
        table, occ = frame[col.source]
        return table, occ, col.level, col.source


def render(e, scope: Scope, occurrences: bool = False) -> str:
    """A canonical text for an expression: columns as table.column (with #occurrence for self-joins when asked),
    symmetric comparisons and AND/OR operands sorted."""
    r = lambda x: render(x, scope, occurrences)  # noqa: E731
    if e is None:
        return "-"
    if isinstance(e, ir.Column):
        t, occ, _lv, _src = scope.resolve(e)
        return f"{t}{'#' + str(occ) if occurrences else ''}.{e.name.lower()}"
    if isinstance(e, ir.DerivedColumn):
        return f"<derived{e.level}.{e.source}>.{e.position}"
    if isinstance(e, ir.Literal):
        return "NULL" if e.kind == "null" else repr(e.value) if e.kind == "text" else str(e.value)
    if isinstance(e, ir.Compare):
        a, b = r(e.left), r(e.right)
        if e.op in ("=", "<>", "!="):
            a, b = sorted((a, b))
        return f"({a} {e.op} {b})"
    if isinstance(e, (ir.And, ir.Or)):
        return "(" + f" {type(e).__name__.upper()} ".join(sorted(r(a) for a in e.args)) + ")"
    if isinstance(e, ir.Binary):
        return f"({r(e.left)} {e.op} {r(e.right)})"
    if isinstance(e, ir.Unary):
        return f"(-{r(e.arg)})"
    if isinstance(e, ir.Not):
        return f"NOT {r(e.arg)}"
    if isinstance(e, ir.Like):
        return f"({r(e.arg)} {'NOT ' if e.negated else ''}LIKE {r(e.pattern)})"
    if isinstance(e, ir.Between):
        return f"({r(e.arg)} {'NOT ' if e.negated else ''}BETWEEN {r(e.low)} AND {r(e.high)})"
    if isinstance(e, ir.InList):
        return f"({r(e.arg)} {'NOT ' if e.negated else ''}IN [{', '.join(sorted(r(i) for i in e.items))}])"
    if isinstance(e, ir.IsNull):
        return f"({r(e.arg)} IS {'NOT ' if e.negated else ''}NULL)"
    if isinstance(e, ir.Agg):
        return f"{e.func}({'DISTINCT ' if e.distinct else ''}{r(e.arg)})"
    if isinstance(e, ir.CountStar):
        return "COUNT(*)"
    if isinstance(e, ir.Func):
        return f"{e.name}({', '.join(r(a) for a in e.args)})"
    if isinstance(e, ir.Cast):
        return f"CAST({r(e.arg)} AS {e.affinity})"
    if isinstance(e, ir.Case):
        whens = " ".join(f"WHEN {r(w.condition)} THEN {r(w.result)}" for w in e.whens)
        return f"CASE {r(e.operand) + ' ' if e.operand is not None else ''}{whens} ELSE {r(e.default)} END"
    if isinstance(e, ir.Star):
        return "*" if e.source is None else f"{scope.frames[-1][e.source][0]}.*"
    if isinstance(e, (ir.InQuery, ir.Exists, ir.ScalarQuery)):
        q = e.query
        inner = render_query(q, scope)
        if isinstance(e, ir.InQuery):
            return f"({r(e.arg)} {'NOT ' if e.negated else ''}IN <{inner}>)"
        if isinstance(e, ir.Exists):
            return f"({'NOT ' if e.negated else ''}EXISTS <{inner}>)"
        return f"<{inner}>"
    return type(e).__name__


def render_query(q, scope: Scope) -> str:
    if isinstance(q, ir.Compound):
        return f"{render_query(q.left, scope)} {q.op} {render_query(q.right, scope)}"
    f = select_features(q, scope)
    return json.dumps({k: f[k] for k in ("tables", "join_keys", "where", "group", "having", "items", "distinct",
                                         "order", "limit")}, sort_keys=True, default=list)


def conjuncts(e) -> list:
    if e is None:
        return []
    if isinstance(e, ir.And):
        out = []
        for a in e.args:
            out.extend(conjuncts(a))
        return out
    return [e]


def _references(e, scope: Scope, level: int = 0) -> set:
    """(level, source) pairs of this SELECT's sources that an expression reads (correlated subqueries included)."""
    out = set()
    if e is None:
        return out
    if isinstance(e, ir.Column):
        if e.level == level:
            out.add(e.source)
        return out
    if isinstance(e, ir.DerivedColumn):
        if e.level == level:
            out.add(e.source)
        return out
    if isinstance(e, ir.Star):
        return set(range(len(scope.frames[-1]))) if e.source is None else {e.source}
    if isinstance(e, (ir.Select, ir.Compound)):
        selects = [e] if isinstance(e, ir.Select) else [n for n in ir.walk(e) if isinstance(n, ir.Select)]
        for s in selects:
            out |= _select_outer_refs(s, level + 1)
        return out
    for child in ir.children(e):
        out |= _references(child, scope, level)
    return out


def _select_outer_refs(s: ir.Select, level: int) -> set:
    out = set()
    parts = [i.on for i in s.from_] + [s.where, s.having] + list(s.group_by) + list(s.items) + \
        [o.expr for o in s.order_by]
    for p in parts:
        if p is None:
            continue
        for n in ir.walk(p):
            if isinstance(n, (ir.Column, ir.DerivedColumn)) and n.level == level:
                out.add(n.source)
    return out


MULTIPLICITY_SENSITIVE = ("COUNT", "SUM", "AVG", "TOTAL")


def select_features(s: ir.Select, outer: Scope) -> dict:
    scope = outer.push(s)
    rr = lambda e: render(e, scope)  # noqa: E731
    tables = sorted(t for t, _ in scope.frames[-1] if t is not None)
    joins, join_keys = [], []
    for k, item in enumerate(s.from_):
        if k == 0:
            continue
        eqs = []
        for c in conjuncts(item.on):
            if isinstance(c, ir.Compare) and c.op == "=" and isinstance(c.left, ir.Column) and \
                    isinstance(c.right, ir.Column):
                a, b = scope.resolve(c.left), scope.resolve(c.right)
                same_ref = (a[2], a[3], c.left.name.lower()) == (b[2], b[3], c.right.name.lower())
                same_src = (a[2], a[3]) == (b[2], b[3])
                kind = "same_reference" if same_ref else "same_source" if same_src else "cross_source"
                key = tuple(sorted([f"{a[0]}.{c.left.name.lower()}", f"{b[0]}.{c.right.name.lower()}"]))
                eqs.append({"kind": kind, "left": f"{a[0]}#{a[1]}.{c.left.name.lower()}",
                            "right": f"{b[0]}#{b[1]}.{c.right.name.lower()}", "key": list(key),
                            "touches_new_source": (a[2] == 0 and a[3] == k) or (b[2] == 0 and b[3] == k),
                            "sources": sorted({(a[2], a[3]), (b[2], b[3])})})
                if kind == "cross_source":
                    join_keys.append(key)
        refs = _references(item.on, scope) if item.on is not None else set()
        joins.append({"position": k, "join": item.join,
                      "table": scope.frames[-1][k][0], "on": rr(item.on), "equalities": eqs,
                      "constrains_new_source": k in refs, "on_references": sorted(refs)})
    aggs = Counter()
    for part in list(s.items) + [s.having] + [o.expr for o in s.order_by]:
        if part is None or isinstance(part, ir.Star):
            continue
        for n in ir.walk(part):
            if isinstance(n, ir.Agg):
                aggs[f"{n.func}{'_DISTINCT' if n.distinct else ''}"] += 1
            elif isinstance(n, ir.CountStar):
                aggs["COUNT_STAR"] += 1
    literals = Counter()
    for part in [s.where, s.having]:
        if part is None:
            continue
        for n in ir.walk(part):
            if isinstance(n, ir.Literal):
                literals[render(n, scope)] += 1
    columns = set()
    for part in [i.on for i in s.from_] + [s.where, s.having] + list(s.group_by) + list(s.items) + \
            [o.expr for o in s.order_by]:
        if part is None or isinstance(part, ir.Star):
            continue
        for n in _walk_same_select(part):
            if isinstance(n, ir.Column) and n.level == 0:
                columns.add(f"{scope.frames[-1][n.source][0]}.{n.name.lower()}")
    return {"tables": tables, "joins": joins, "join_keys": sorted(join_keys),
            "where": sorted(rr(c) for c in conjuncts(s.where)), "where_columns": sorted(_columns_of(s.where, scope)),
            "group": sorted(rr(g) for g in s.group_by), "having": rr(s.having),
            "items": [rr(i) for i in s.items], "distinct": s.distinct,
            "order": [[rr(o.expr), o.descending] for o in s.order_by], "limit": s.limit, "offset": s.offset,
            "aggregates": dict(sorted(aggs.items())), "literals": dict(sorted(literals.items())),
            "columns": sorted(columns), "scope": scope,
            "nested": any(isinstance(n, (ir.InQuery, ir.Exists, ir.ScalarQuery, ir.DerivedSource))
                          for n in ir.walk(s))}


def _walk_same_select(e):
    """Nodes of an expression without descending into subqueries."""
    yield e
    if isinstance(e, (ir.InQuery, ir.Exists, ir.ScalarQuery)):
        if isinstance(e, ir.InQuery):
            yield from _walk_same_select(e.arg)
        return
    for child in ir.children(e):
        yield from _walk_same_select(child)


def _columns_of(e, scope: Scope) -> set:
    out = set()
    if e is None:
        return out
    for n in _walk_same_select(e):
        if isinstance(n, ir.Column) and n.level == 0:
            out.add(f"{scope.frames[-1][n.source][0]}.{n.name.lower()}")
    return out


def top_select(plan):
    q = plan
    while isinstance(q, ir.Compound):
        q = q.left
    return q


# --- schema relations --------------------------------------------------------------------------------------------------
def relations(db_id: str) -> dict:
    """Declared keys of one database: primary-key columns and foreign-key pairs (lower case), both directions."""
    from . import schema as schema_mod
    s = schema_mod.load_schema(db_id)
    pk, fk = set(), set()
    for t in s.tables:
        for c in t.columns:
            if c.primary_key:
                pk.add(f"{t.name.lower()}.{c.name.lower()}")
        for from_col, ref_table, to_col in t.foreign_keys:
            fk.add(tuple(sorted([f"{t.name.lower()}.{str(from_col).lower()}",
                                 f"{str(ref_table).lower()}.{str(to_col).lower()}"])))
    return {"pk": pk, "fk": fk}


# --- the first divergence from the gold traversal -----------------------------------------------------------------------
def first_divergence(pred: list, gold: list, schema, spans) -> dict:
    """Where a predicted action sequence first leaves the gold traversal, and what kind of decision that was."""
    m = A.Machine(schema, spans)
    for i, g in enumerate(gold):
        p = pred[i] if i < len(pred) else None
        if p is None or p != g:
            top = m.top
            ctx = top.ctx
            gold_family = m.family(g.op)
            pred_family = m.family(p.op) if p is not None and p.op in m.options() else None
            if p is None:
                decision = "stopped early"
            elif gold_family != pred_family:
                decision = "family"
            elif p.op != g.op:
                decision = "operator"
            else:
                decision = "pointer"
            operand = getattr(top, "child_index", None)
            return {"index": i, "of": len(gold), "gold": g.as_json(), "pred": p.as_json() if p else None,
                    "clause": ctx.clause, "slot": top.kind, "gold_family": A.FAMILIES[gold_family],
                    "pred_family": A.FAMILIES[pred_family] if pred_family is not None else None,
                    "decision": decision, "parent_op": getattr(top, "parent_op", None), "operand": operand,
                    "category": divergence_category(ctx.clause, top.kind, g, p, decision,
                                                    getattr(top, "parent_op", None), operand)}
        m.apply(g)
    if len(pred) > len(gold):
        return {"index": len(gold), "of": len(gold), "gold": None, "pred": pred[len(gold)].as_json(),
                "clause": "STOP", "slot": "DONE", "decision": "did not stop", "category": "did not stop"}
    return {"index": None, "of": len(gold), "category": "exact"}


def divergence_category(clause, slot, g, p, decision, parent_op, operand) -> str:
    gop = g.op
    if clause == "FROM":
        if gop == "FROM_TABLE":
            return "FROM table" if decision == "pointer" else "FROM structure"
        if gop.startswith("JOIN_") and decision == "pointer":
            return "joined table"
        return "join count or kind"
    if clause == "ON":
        if A.POINTER_KIND.get(gop) == "column" and decision == "pointer":
            return f"ON column (operand {operand})"
        if gop.startswith("COL") and decision == "operator" and p is not None and p.op.startswith("COL"):
            return f"ON occurrence (operand {operand})"
        return "ON structure"
    if clause in ("WHERE", "HAVING"):
        if gop in A.LITERAL_OPS or (p is not None and p.op in A.LITERAL_OPS):
            return f"{clause} literal"
        if A.POINTER_KIND.get(gop) == "column" and decision == "pointer":
            return f"{clause} column"
        return f"{clause} predicate structure"
    if clause == "GROUP":
        return "GROUP BY"
    if clause == "SELECT":
        if slot == "DISTINCT":
            return "DISTINCT"
        if gop in A.AGG_OPS or (p is not None and p.op in A.AGG_OPS):
            return "SELECT aggregate"
        if A.POINTER_KIND.get(gop) == "column" and decision == "pointer":
            return "SELECT column"
        return "SELECT structure"
    if clause == "ORDER":
        return "ORDER BY"
    if clause == "LIMIT":
        return "LIMIT"
    if clause == "QUERY":
        return "query structure"
    return clause


def step_labels(gold: list, schema, spans) -> list:
    """For every step of a gold traversal (STOP included), the decision's category: clause, slot and the role of its
    operator and pointer (e.g. "FROM table", "join: add table", "ON column (operand 1)", "WHERE literal span")."""
    m = A.Machine(schema, spans)
    out = []
    for g in gold:
        top = m.top
        ctx = top.ctx
        clause, slot = ctx.clause, top.kind
        op, kind = g.op, A.POINTER_KIND.get(g.op)
        operand = getattr(top, "child_index", None)
        if clause == "FROM":
            if op == "FROM_TABLE":
                label = "FROM table"
            elif op.startswith("JOIN_"):
                label = "join: add table" if kind == "table" else "join: add derived"
            elif op == "END":
                label = "join: no more tables"
            else:
                label = "FROM structure"
        elif clause == "ON":
            label = f"ON column (operand {operand})" if kind == "column" else "ON operator"
        elif clause in ("WHERE", "HAVING"):
            if op in A.LITERAL_OPS:
                label = f"{clause} literal"
            elif kind == "column":
                label = f"{clause} column"
            else:
                label = f"{clause} operator"
        elif clause == "GROUP":
            label = "GROUP BY column" if kind == "column" else "GROUP BY other"
        elif clause == "SELECT":
            if slot == "DISTINCT":
                label = "DISTINCT"
            elif op in A.AGG_OPS:
                label = "SELECT aggregate"
            elif kind == "column":
                label = "SELECT column"
            else:
                label = "SELECT other"
        elif clause == "ORDER":
            label = "ORDER BY column" if kind == "column" else "ORDER BY other"
        elif clause == "LIMIT":
            label = "LIMIT"
        else:
            label = clause
        out.append(label)
        m.apply(g)
    out.append("STOP")
    return out


# --- one request ------------------------------------------------------------------------------------------------------
def compare_plans(pred, gold) -> dict:
    """Component-by-component comparison of the top SELECTs (and whether either side nests subqueries)."""
    if pred is None or gold is None:
        return {"comparable": False}
    ps, gs = top_select(pred), top_select(gold)
    pf, gf = select_features(ps, Scope()), select_features(gs, Scope())
    out = {"comparable": True, "exact_plan": pred == gold, "compound": isinstance(pred, ir.Compound) or
           isinstance(gold, ir.Compound), "nested": pf["nested"] or gf["nested"]}
    for key in ("tables", "join_keys", "where", "group", "having", "items", "distinct", "order", "limit",
                "aggregates", "literals", "columns", "where_columns"):
        out[f"{key}_equal"] = pf[key] == gf[key]
    out["missing_tables"] = sorted((Counter(gf["tables"]) - Counter(pf["tables"])).elements())
    out["extra_tables"] = sorted((Counter(pf["tables"]) - Counter(gf["tables"])).elements())
    out["missing_columns"] = sorted(set(gf["columns"]) - set(pf["columns"]))
    out["extra_columns"] = sorted(set(pf["columns"]) - set(gf["columns"]))
    out["missing_join_keys"] = [list(k) for k in sorted(set(gf["join_keys"]) - set(pf["join_keys"]))]
    out["extra_join_keys"] = [list(k) for k in sorted(set(pf["join_keys"]) - set(gf["join_keys"]))]
    out["missing_where"] = sorted(set(gf["where"]) - set(pf["where"]))
    out["extra_where"] = sorted(set(pf["where"]) - set(gf["where"]))
    out["pred"] = {k: pf[k] for k in pf if k != "scope"}
    out["gold"] = {k: gf[k] for k in gf if k != "scope"}
    return out


def join_semantics(pred, gold, rel: dict) -> list:
    """Every join equality of the predicted top SELECT, with its semantics and the gold program's join at that
    position. A same-reference equality `X = X` is `X IS NOT NULL`; it constrains no join."""
    if pred is None:
        return []
    ps = top_select(pred)
    pf = select_features(ps, Scope())
    gf = select_features(top_select(gold), Scope()) if gold is not None else None
    gold_keys = set(map(tuple, gf["join_keys"])) if gf else set()
    out = []
    for j in pf["joins"]:
        for eq in j["equalities"]:
            key = tuple(eq["key"])
            entry = {"position": j["position"], "joined_table": j["table"], "kind": eq["kind"],
                     "left": eq["left"], "right": eq["right"], "key": list(key),
                     "declared_fk": key in rel["fk"], "in_gold": key in gold_keys,
                     "join_constrains_new_source": j["constrains_new_source"]}
            if eq["kind"] == "same_reference":
                col = key[0]
                partners = sorted({k[1] if k[0] == col else k[0] for k in gold_keys if col in k})
                fk_partners = sorted({k[1] if k[0] == col else k[0] for k in rel["fk"] if col in k})
                other_sources = [i for i in range(len(ps.from_)) if i not in {s for _lv, s in eq["sources"]}]
                used_elsewhere = _sources_used_outside_on(ps, set(other_sources))
                aggs = [a for a in pf["aggregates"] if a.split("_")[0] in MULTIPLICITY_SENSITIVE
                        and not a.endswith("_DISTINCT")]
                entry.update({
                    "column": col, "gold_partners_of_this_column": partners, "fk_partners_of_this_column": fk_partners,
                    "unconstrained_sources": other_sources if not j["constrains_new_source"] else [],
                    "unconstrained_sources_read_elsewhere": sorted(used_elsewhere),
                    "multiplicity_sensitive_aggregates": aggs,
                    "limit": ps.limit,
                    "result_set_effect": _effect(j["constrains_new_source"], used_elsewhere, aggs, ps.limit)})
            out.append(entry)
    return out


def _sources_used_outside_on(s: ir.Select, sources: set) -> set:
    parts = [s.where, s.having] + list(s.group_by) + [i for i in s.items] + [o.expr for o in s.order_by]
    used = set()
    for p in parts:
        if p is None:
            continue
        if isinstance(p, ir.Star):
            used |= sources if p.source is None else ({p.source} & sources)
            continue
        for n in ir.walk(p):
            if isinstance(n, (ir.Column, ir.DerivedColumn)) and n.level == 0 and n.source in sources:
                used.add(n.source)
    return used


def _effect(constrains_new: bool, used_elsewhere: set, aggs: list, limit) -> str:
    if constrains_new:
        return "the join's new source is constrained by another conjunct"
    if used_elsewhere:
        return "an unconstrained source is read elsewhere: rows are paired arbitrarily (wrong in general)"
    if aggs:
        return "a cross product under a multiplicity-sensitive aggregate (COUNT/SUM/AVG): counts are multiplied"
    if limit is not None:
        return "a cross product under LIMIT: duplicate rows can displace distinct ones"
    return ("result-set equivalent to omitting the join, provided the joined table is non-empty (EX compares sets); "
            "costly, and the X IS NOT NULL filter remains")


# --- the development records ---------------------------------------------------------------------------------------
def _encoding(task_id: str, reader, cache_root, anchors: dict):
    from . import data, encode, retrieval
    from . import schema as schema_mod
    task = data.task(task_id)
    enc = encode.encode_request(reader, task, schema_mod.load_schema(task.db_id), retrieval.retrieve(task),
                                cache_root=cache_root, require_hit=True, require_verified=True)
    a = anchors.get(task_id)
    if a is None or a["states_sha256"] != enc.cost.get("states_sha256"):
        raise RuntimeError(f"{task_id}: the cached states do not match their anchor")
    return enc


def diagnose_request(rec: dict, enc, gold_plan, gold_actions, rel: dict) -> dict:
    """One saved evaluation record against its gold program."""
    attempt = rec["attempts"][0]
    pred_actions = [A.Action.from_json(a) for a in attempt.get("actions") or []]
    pred_plan = ir.from_json(attempt["plan"]) if attempt.get("plan") else None
    if pred_plan is None and pred_actions:
        try:
            pred_plan = A.decode(pred_actions, enc.schema, enc.spans)
        except A.IllegalAction:
            pred_plan = None
    score = rec["score"]
    out = {"task_id": rec["task_id"], "db_id": rec["db_id"], "correct": score["correct"],
           "final_kind": rec["final"]["kind"], "abstain_reason": rec["final"].get("reason"),
           "predicted_status": score["predicted"]["status"],
           "predicted_seconds": score["predicted"]["elapsed_seconds"],
           "predicted_rows": score["predicted"].get("row_count"), "gold_rows": score["gold"].get("row_count"),
           "representable": gold_actions is not None, "steps": len(pred_actions)}
    out["divergence"] = first_divergence(pred_actions, gold_actions, enc.schema, enc.spans) \
        if gold_actions is not None else {"category": "gold not representable"}
    out["plans"] = compare_plans(pred_plan, gold_plan)
    out["joins"] = join_semantics(pred_plan, gold_plan, rel)
    out["categories"] = categories(out)
    return out


def categories(d: dict) -> list:
    """The error categories of one request (several can apply). Empty for an exact plan."""
    p = d["plans"]
    if d["final_kind"] != "sql":
        return ["abstention"]
    cats = []
    if d["predicted_status"] == "timeout":
        cats.append("execution timeout")
    elif d["predicted_status"] not in ("ok",):
        cats.append(f"execution {d['predicted_status']}")
    if not p.get("comparable"):
        return cats + ["no comparable plan"]
    if p["exact_plan"]:
        return cats
    if any(j["kind"] == "same_reference" for j in d["joins"]):
        cats.append("join: same-reference equality")
    if not p["tables_equal"]:
        cats.append("table grounding")
    if p["tables_equal"] and not p["join_keys_equal"]:
        cats.append("join key (same tables)")
    if not p["tables_equal"] and not p["join_keys_equal"]:
        cats.append("join key (different tables)")
    if any(j["kind"] == "cross_source" and not j["declared_fk"] and not j["in_gold"] for j in d["joins"]):
        cats.append("join: undeclared, non-gold key")
    if not p["where_columns_equal"]:
        cats.append("predicate column")
    elif not p["where_equal"]:
        if not p["literals_equal"]:
            cats.append("predicate literal")
        else:
            cats.append("predicate operator or shape")
    if not p["aggregates_equal"]:
        cats.append("aggregation")
    if not p["group_equal"] or not p["having_equal"]:
        cats.append("grouping or HAVING")
    if not p["items_equal"]:
        cats.append("projection")
    if not p["distinct_equal"]:
        cats.append("DISTINCT")
    if not p["order_equal"] or not p["limit_equal"]:
        cats.append("ordering or LIMIT")
    if p["nested"]:
        cats.append("nested query involved")
    return cats


# --- teacher forcing --------------------------------------------------------------------------------------------------
def teacher_forced(model, example) -> dict:
    """Teacher-forced loss and per-step argmax accuracy (family, operator, pointer) on one example."""
    import torch

    from . import controller as K
    with torch.no_grad():
        batch = K.collate([example])
        st = batch.steps
        if hasattr(model.semantic, "calls"):
            model.semantic.calls.clear()
        losses = model.losses(batch)
        mem = model.trunk.memory(batch)
        state, _ = model.trunk.states(mem, st)
        valid = st["valid"][0]
        fam_logits = model.trunk.family_head(state).masked_fill(~st["legal_family"], K.NEG)[0]
        fam_ok = fam_logits.argmax(-1) == st["family"][0]
        rows = valid & (st["family"][0] < 4)
        idx = rows.nonzero(as_tuple=True)[0]
        op_logits, q_rows = model.semantic(state[0, idx], st["family"][0, idx])
        op_logits = op_logits.masked_fill(~st["legal_union"][0, idx], K.NEG)
        op_ok = torch.ones_like(fam_ok)
        op_ok[idx] = op_logits.argmax(-1) == st["union"][0, idx]
        queries = state.new_zeros((1, state.shape[1], K.N_POINTERS, model.cfg.d_model))
        queries[0, idx] = q_rows
        items, start, end = model.trunk.pointer_scores(mem, queries, st["ptr_target"])
        kind = st["ptr_kind"][0]
        ptr_ok = torch.ones_like(fam_ok)
        for k, slot in ((1, 0), (2, 1)):
            sel = kind == k
            if sel.any():
                sc = items[0, :, slot][sel].masked_fill(~st["ptr_items"][0][sel], K.NEG)
                ptr_ok[sel] = sc.argmax(-1) == st["ptr_target"][0][sel]
        sel = kind == 3
        if sel.any():
            s_ok = start[0][sel].masked_fill(~st["span_start"][0][sel], K.NEG).argmax(-1) == st["ptr_target"][0][sel]
            e_ok = end[0][sel].masked_fill(~st["span_end"][0][sel], K.NEG).argmax(-1) == st["ptr_end"][0][sel]
            ptr_ok[sel] = s_ok & e_ok
        step_ok = (fam_ok & op_ok & ptr_ok)[valid]
        first_wrong = int((~step_ok).nonzero()[0]) if (~step_ok).any() else None
    n = int(losses["steps"])
    labels = step_labels([s.action for s in example.steps[:-1]], example.schema, example.spans)
    per_step = [{"label": labels[t], "family_ok": bool(fam_ok[t]), "op_ok": bool(op_ok[t]),
                 "pointer": int(kind[t]) if t < len(kind) else 0, "pointer_ok": bool(ptr_ok[t])}
                for t in range(n)]
    return {"loss": float(losses["total"]) / n, "family": float(losses["family"]) / n,
            "op": float(losses["op"]) / n, "pointer": float(losses["pointer"]) / n, "steps": n,
            "step_accuracy": float(step_ok.float().mean()), "all_steps_correct": bool(step_ok.all()),
            "first_wrong_step": first_wrong, "per_step": per_step}


def by_label(rows: list) -> dict:
    """Teacher-forced error rates by decision category over many examples: steps, family errors, operator errors
    and pointer errors (pointer steps only)."""
    agg = defaultdict(lambda: {"steps": 0, "family_wrong": 0, "op_wrong": 0, "pointer_steps": 0,
                               "pointer_wrong": 0, "any_wrong": 0})
    for r in rows:
        for st in r["per_step"]:
            a = agg[st["label"]]
            a["steps"] += 1
            a["family_wrong"] += not st["family_ok"]
            a["op_wrong"] += not st["op_ok"]
            if st["pointer"]:
                a["pointer_steps"] += 1
                a["pointer_wrong"] += not st["pointer_ok"]
            a["any_wrong"] += not (st["family_ok"] and st["op_ok"] and st["pointer_ok"])
    out = {}
    for label, a in sorted(agg.items(), key=lambda kv: -kv[1]["any_wrong"]):
        out[label] = {**a, "error_rate": a["any_wrong"] / a["steps"]}
    return out


# --- execution in worker processes ----------------------------------------------------------------------------------
def _score_job(job):
    from . import data, scorer
    db_id, pred_sql, gold_sql = job
    s = scorer.score(data.snapshot(db_id), pred_sql, gold_sql)
    return {"correct": s["correct"], "status": s["predicted"]["status"],
            "seconds": s["predicted"]["elapsed_seconds"], "rows": s["predicted"].get("row_count")}


def train_sample(ids: list, n: int = TRAIN_SAMPLE_SIZE) -> list:
    """A fixed sample of training-pool ids: the n smallest sha256(salt:id), in pin order."""
    keyed = sorted(ids, key=lambda t: hashlib.sha256(f"{TRAIN_SAMPLE_SALT}:{t}".encode()).hexdigest())[:n]
    chosen = set(keyed)
    return [t for t in ids if t in chosen]


# --- the command --------------------------------------------------------------------------------------------------------
def summarize(rows: list) -> dict:
    """Counts by category, first-divergence category and join kind, for one fit's requests."""
    cats, div, decisions = Counter(), Counter(), Counter()
    for r in rows:
        cats.update(set(r["categories"]))
        div[r["divergence"]["category"]] += 1
        if r["divergence"].get("decision"):
            decisions[r["divergence"]["decision"]] += 1
    joins = Counter(j["kind"] for r in rows for j in r["joins"])
    comparable = [r for r in rows if r["plans"].get("comparable")]
    right_tables = [r for r in comparable if r["plans"]["tables_equal"]]
    first_in_gold = sum(bool(r["plans"]["pred"]["tables"]) and
                        r["plans"]["pred"]["tables"][0] in r["plans"]["gold"]["tables"] for r in comparable)
    return {"requests": len(rows), "correct": sum(r["correct"] for r in rows),
            "table_set_equal": len(right_tables), "correct_given_table_set_equal": sum(r["correct"]
                                                                                      for r in right_tables),
            "correct_given_table_set_differs": sum(r["correct"] for r in comparable if not
                                                   r["plans"]["tables_equal"]),
            "join_keys_equal_given_tables_equal": sum(r["plans"]["join_keys_equal"] for r in right_tables),
            "every_predicted_table_in_gold": sum(set(r["plans"]["pred"]["tables"]) <= set(r["plans"]["gold"]["tables"])
                                                 for r in comparable),
            "every_gold_table_predicted": sum(set(r["plans"]["gold"]["tables"]) <= set(r["plans"]["pred"]["tables"])
                                              for r in comparable),
            "first_table_in_gold_set": first_in_gold,
            "exact_plan": sum(bool(r["plans"].get("exact_plan")) for r in rows),
            "categories": dict(cats.most_common()), "first_divergence": dict(div.most_common()),
            "divergence_decision": dict(decisions.most_common()), "join_equalities": dict(joins),
            "requests_with_same_reference_join": sum(any(j["kind"] == "same_reference" for j in r["joins"])
                                                     for r in rows),
            "timeouts": sum(r["predicted_status"] == "timeout" for r in rows)}


def run(args) -> dict:
    import torch

    from . import controller as K
    from . import data, encode, infer_controller, pilot, pilot_run, train
    torch.set_num_threads(args.threads)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = pilot_run._tokenizer()
    proto = pilot_run.protocol_identity()
    anchors, ident = pilot_run.anchors(proto)
    reader = encode.Backbone.cache_reader(ident, tok)
    ids = pilot.eval_ids()
    chosen, _all = pilot_run.select_evaluations()
    t0 = time.perf_counter()

    # gold programs and encodings of the 64 (CPU, cached states)
    from . import sql_to_ir
    dev = {}
    for tid in ids:
        enc = _encoding(tid, reader, encode.PILOT_CACHE, anchors)
        gold_plan, err = sql_to_ir.try_convert(data.gold(tid).sql, enc.schema.index)
        steps, _reason = train.gold_steps(data.gold(tid).sql, enc)
        dev[tid] = {"enc": enc, "gold_plan": gold_plan if err is None else None,
                    "gold_actions": [s.action for s in steps[:-1]] if steps else None, "steps": steps,
                    "rel": relations(data.task(tid).db_id)}
    log(f"development encodings and gold: {time.perf_counter() - t0:.0f} s")

    result = {"protocol": proto, "evaluations": {str(s): str(d.name) for s, d in chosen.items()},
              "dev": {}, "dev_teacher_forced": {}, "train": {}, "threads": args.threads}
    per_request = {}
    models = {}
    for seed, d in chosen.items():
        for arm in ARMS:
            recs = [json.loads(x) for x in (d / arm / "records.jsonl").read_text().splitlines() if x.strip()]
            rows = []
            for rec in recs:
                g = dev[rec["task_id"]]
                rows.append(diagnose_request(rec, g["enc"], g["gold_plan"], g["gold_actions"], g["rel"]))
            per_request[f"{arm}/{seed}"] = rows
            result["dev"][f"{arm}/{seed}"] = summarize(rows)
            path, _entry = pilot_run.final_checkpoint(pilot.ROOT / "train" / f"{arm}_s{seed}")
            models[(arm, seed)] = train.model_from_checkpoint(train.load_checkpoint(path), "cpu").eval()
    log(f"development records diagnosed: {time.perf_counter() - t0:.0f} s")

    # teacher forcing on the representable development questions, and a CPU re-decode check of the records
    for (arm, seed), model in models.items():
        tf, same = [], 0
        recs = {r["task_id"]: r for r in per_request[f"{arm}/{seed}"]}
        saved = {json.loads(x)["task_id"]: json.loads(x) for x in
                 (chosen[seed] / arm / "records.jsonl").read_text().splitlines() if x.strip()}
        for tid in ids:
            g = dev[tid]
            ex = K.prepare(g["enc"], g["steps"]) if g["steps"] else K.prepare(g["enc"])
            dec = infer_controller.decode_one(model, ex)
            same += [a.as_json() for a in dec.actions] == saved[tid]["attempts"][0]["actions"]
            if g["steps"]:
                tf.append({"task_id": tid, **teacher_forced(model, ex), "correct": recs[tid]["correct"]})
        result["dev_teacher_forced"][f"{arm}/{seed}"] = {
            "representable": len(tf), "cpu_redecode_identical_actions": same,
            "mean_loss": statistics.mean(x["loss"] for x in tf),
            "mean_step_accuracy": statistics.mean(x["step_accuracy"] for x in tf),
            "all_steps_correct": sum(x["all_steps_correct"] for x in tf), "by_decision": by_label(tf),
            "per_question": [{k: v for k, v in x.items() if k != "per_step"} for x in tf]}
    log(f"development teacher forcing: {time.perf_counter() - t0:.0f} s")

    # the training-pool sample: teacher forcing and free decoding from the final checkpoints
    sample = train_sample(pilot.train_ids(), args.train_sample)
    prepared, _ident, _secs = pilot_run.prepare(sample, tok, proto, log=None)
    by_id = {e.task_id: e for e in prepared}
    jobs, decoded = [], {}
    for (arm, seed), model in models.items():
        rows = []
        for tid in sample:
            ex = by_id[tid]
            tf = teacher_forced(model, ex)
            dec = infer_controller.decode_one(model, ex)
            gold_actions = [s.action for s in ex.steps[:-1]]
            rows.append({"task_id": tid, "db_id": data.task(tid).db_id, **tf,
                         "exact_actions": dec.actions == gold_actions, "sql": dec.sql,
                         "abstain": dec.abstain_reason,
                         "divergence": first_divergence(dec.actions, gold_actions, ex.schema, ex.spans),
                         "plan": dec.plan})
            jobs.append((f"{arm}/{seed}", len(rows) - 1, data.task(tid).db_id, dec.sql, data.gold(tid).sql))
        decoded[f"{arm}/{seed}"] = rows
        log(f"training sample {arm}/{seed}: {time.perf_counter() - t0:.0f} s")
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        scores = list(pool.map(_score_job, [(db, sql, gold) for _k, _i, db, sql, gold in jobs]))
    for (key, i, *_rest), s in zip(jobs, scores):
        decoded[key][i].update(ex_correct=s["correct"], status=s["status"], exec_seconds=s["seconds"])
    for key, rows in decoded.items():
        train_rows = []
        for r in rows:
            gold = sql_to_ir.try_convert(data.gold(r["task_id"]).sql, by_id[r["task_id"]].schema.index)[0]
            js = join_semantics(r["plan"], gold, relations(r["db_id"]))
            r["joins"] = js
            r["plans"] = compare_plans(r["plan"], gold)
            r.pop("plan")
            train_rows.append(r)
        result["train"][key] = {
            "by_decision": by_label(rows),
            "table_set_equal": sum(bool(r["plans"].get("tables_equal")) for r in rows),
            "sample": len(rows), "mean_tf_loss": statistics.mean(r["loss"] for r in rows),
            "mean_tf_step_accuracy": statistics.mean(r["step_accuracy"] for r in rows),
            "tf_all_steps_correct": sum(r["all_steps_correct"] for r in rows),
            "free_exact_actions": sum(r["exact_actions"] for r in rows),
            "free_ex_correct": sum(r["ex_correct"] for r in rows),
            "timeouts": sum(r["status"] == "timeout" for r in rows),
            "abstentions": sum(r["abstain"] is not None for r in rows),
            "same_reference_join_outputs": sum(any(j["kind"] == "same_reference" for j in r["joins"]) for r in rows),
            "first_divergence": dict(Counter(r["divergence"]["category"] for r in rows).most_common()),
        }
    for rows in decoded.values():
        for r in rows:
            r.pop("per_step", None)
    log(f"training sample scored: {time.perf_counter() - t0:.0f} s")
    result["seconds"] = time.perf_counter() - t0
    (out_dir / "diagnosis.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    (out_dir / "dev_requests.json").write_text(json.dumps(per_request, indent=1, sort_keys=True, default=str) + "\n")
    (out_dir / "train_requests.json").write_text(json.dumps(decoded, indent=1, sort_keys=True, default=str) + "\n")
    return result


def log(msg: str) -> None:
    print(msg, flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-sample", type=int, default=TRAIN_SAMPLE_SIZE)
    args = ap.parse_args(argv)
    res = run(args)
    print(json.dumps({k: res[k] for k in ("dev", "train")}, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
