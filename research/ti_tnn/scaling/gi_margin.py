"""A bounded saved-model diagnostic of the grounding interface (CPU; post hoc; evaluator-only interventions).

    python -m scaling.gi_margin            # 32 dev questions x the seed-100 T4-v2 and C4-v2 checkpoints

**The question.** The interface adds ln 2 / r to the table- and column-pointer logits, so a record changes a decision
only when some legal candidate's prior advantage over the selected item exceeds the selected item's logit lead.
How often is that possible? And do evaluator-supplied corrections change the outcome? Nothing here is a policy or an
accuracy claim, and nothing here is trained.

**What it records.** Along each request's unbiased (G0) greedy trajectory, at every table and column pointer step:
- the legal candidates' logits;
- the selected item's margin over the runner-up;
- for the G4 record and the oracle record, the applied prior difference (competitor's prior minus the selected
  item's), and whether the prior could change the selection at that step with the prefix held fixed:
  max_j (s_j + b_j) > s_sel + b_sel over legal j.

A margin above ln 2 cannot be overturned by any record.

**Evaluator-only interventions** (they read the gold program; never deployable):
- **Reference-restricted decoding.** The legal table and column targets are cut to those the gold program uses
  (`gi_record.oracle_record`) wherever the current scope permits any. A step with no reference-supported legal
  candidate keeps its original targets and is counted as undefined there.
- **Forced reference prefix.** The gold traversal is forced through the first decision where the G0 decode departs
  from it, and the remaining suffix decodes freely. It is undefined for a gold program the grammar cannot represent,
  reported as "no divergence" when G0 equals the gold traversal, and reported as undefined if a forced action were
  illegal.

The first FROM table is classified three ways: right, another table of the gold program (reference order), or a
table the gold program does not use (irrelevant).

Nothing of the frozen grounding-interface producer is changed: the interventions are machine subclasses installed
with `gi_constraint.machine_class`, and the logits are read by wrapping the trunk's `pointer_scores` on the instance.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
import time

from . import actions as A
from . import contracts as C

SAMPLE_SALT = "gi-margin-sample-v1"
SAMPLE_SIZE = 32
LN2 = math.log(2.0)
ROOT = C.RUNS / "grounding_interface" / "margin"


def sample(ids: list) -> list:
    """The 32 dev questions with the smallest sha256(salt:task_id), in that order."""
    return sorted(ids, key=lambda t: hashlib.sha256(f"{SAMPLE_SALT}:{t}".encode()).hexdigest())[:SAMPLE_SIZE]


class _Capture:
    """The legal candidates and logits of every table/column pointer decision of one decode."""

    def __init__(self):
        self.items = None
        self.steps = []


@contextlib.contextmanager
def capture_scores(model, cap: _Capture):
    trunk = model.trunk
    original = trunk.pointer_scores

    def wrapped(mem, queries, start_index=None, relations=None):
        items, start, end = original(mem, queries, start_index, relations)
        if start_index is None:      # the first call of an operator step (the span-end call follows it)
            cap.items = items.detach()
        return items, start, end

    trunk.pointer_scores = wrapped
    try:
        yield
    finally:
        del trunk.pointer_scores


def capturing_machine(cap: _Capture):
    class CapturingMachine(A.Machine):
        def pointer_targets(self, op):
            kind, targets = super().pointer_targets(op)
            if kind in ("table", "column") and cap.items is not None:
                slot = 0 if kind == "table" else 1
                scores = cap.items[0, 0, slot]
                cap.steps.append({"step": self.steps, "op": op, "kind": kind, "targets": list(targets),
                                  "scores": [float(scores[t]) for t in targets]})
            return kind, targets
    return CapturingMachine


def restricted_machine(gold_tables: set, gold_columns: set, log: list):
    class ReferenceRestricted(A.Machine):
        def pointer_targets(self, op):
            kind, targets = super().pointer_targets(op)
            if kind not in ("table", "column"):
                return kind, targets
            allowed = gold_tables if kind == "table" else gold_columns
            keep = tuple(t for t in targets if t in allowed)
            log.append({"step": self.steps, "op": op, "kind": kind, "legal": len(targets), "kept": len(keep),
                        "undefined": not keep})
            return (kind, keep) if keep else (kind, targets)
    return ReferenceRestricted


def forced_machine(forced: list, log: list):
    class ForcedPrefix(A.Machine):
        def _forcing(self):
            return self.steps < len(forced)

        def legal_families(self):
            mask = super().legal_families()
            if not self._forcing():
                return mask
            a = forced[self.steps]
            fam = self.family(a.op)
            if a.op not in self.options():
                log.append({"step": self.steps, "illegal": a.op})
                return mask
            return [i == fam for i in range(len(mask))]

        def legal_union_ids(self):
            if not self._forcing() or forced[self.steps].op not in self.options():
                return super().legal_union_ids()
            a = forced[self.steps]
            return [A.UNION_INDEX[(self.family(a.op), a.op)]]

        def pointer_targets(self, op):
            kind, targets = super().pointer_targets(op)
            if not self._forcing() or forced[self.steps].op != op:
                return kind, targets
            arg = forced[self.steps].arg
            return kind, ((arg[0],) if kind == "span" else (arg,))

        def span_ends(self, op, start):
            if self._forcing() and forced[self.steps].op == op:
                return (forced[self.steps].arg[1],)
            return super().span_ends(op, start)
    return ForcedPrefix


def step_analysis(step: dict, biases: "list | None") -> dict:
    """One captured pointer decision against one record's prior."""
    t, s = step["targets"], step["scores"]
    k = max(range(len(t)), key=lambda i: s[i])
    sel, s_sel = t[k], s[k]
    others = [(s[i], t[i]) for i in range(len(t)) if i != k]
    margin = s_sel - max(x for x, _ in others) if others else None
    out = {"selected": sel, "margin": margin, "unchangeable_by_any_record": margin is None or margin > LN2}
    if biases is None:
        return {**out, "prior": None}
    b_sel = biases[sel]
    best = max(((x + biases[j], j) for x, j in others), default=None)
    out["prior"] = {"max_prior_advantage": max((biases[j] - b_sel for _x, j in others), default=0.0),
                    "could_change": best is not None and best[0] > s_sel + b_sel,
                    "flip_to": best[1] if best is not None and best[0] > s_sel + b_sel else None}
    return out


def first_table_class(actions: list, gold_first, gold_tables: set) -> str:
    first = next((a.arg for a in actions if a.op == "FROM_TABLE"), None)
    if first is None:
        return "none"
    return "right" if first == gold_first else "in gold set" if first in gold_tables else "irrelevant"


def run(args) -> dict:
    import torch

    from . import controller as K
    from . import data, gi_run, identity, infer_controller, perf, pilot_run, train
    from . import gi_constraint as GC
    from . import gi_record as GR
    torch.set_num_threads(args.threads)
    ids = sample(perf.eval_ids())
    g4 = gi_run._load_records(gi_run.ROOT / "ground" / args.ground_run)
    fits = {f"{perf.label(a, 'v2')}/100": perf.final_checkpoint(perf.run_dir_of(a, "v2", 100))
            for a in ("T4", "c4_params")}
    run_dir = pilot_run._new_dir(ROOT, "")
    identity.write_manifest(run_dir, config={
        "command": "gi margin", "sample_rule": f"the {SAMPLE_SIZE} smallest sha256('{SAMPLE_SALT}:' + task_id) of the "
        "128 dev questions", "sample": ids, "ground_run": args.ground_run,
        "checkpoints": {k: {"path": identity.run_ref(p), **{f: (r or {}).get(f) for f in ("sha256", "model_digest")}}
                        for k, (p, r) in fits.items()},
        "evaluator_only": ["oracle prior", "reference-restricted decoding", "forced reference prefix"],
        "device": "cpu"}, require_clean=not args.allow_dirty)
    encs = gi_run._encodings(ids)
    scores: dict = {}
    rows = []
    for key, (path, _r) in fits.items():
        model = train.model_from_checkpoint(train.load_checkpoint(path), "cpu").eval()
        for tid in ids:
            task, enc = data.task(tid), encs[tid]
            items, n = enc.schema, len(enc.schema.items)
            prep = K.prepare(enc, grounding="v2")
            gold = gi_run.gold_reference(tid, items)
            gold_tables, gold_cols = set(gold.get("tables", [])), set(gold.get("columns", []))
            oracle = GR.oracle_biases({"tables": sorted(gold_tables), "columns": sorted(gold_cols)}, n) \
                if gold.get("convertible") else None
            rec = g4[tid]
            g = GR.resolve(rec["record"], items, task.question, task.evidence,
                           truncated=rec["call"]["finish_reason"] != "eos") if rec.get("record") is not None else None
            g4_biases = g.biases(n) if g is not None else None
            cap = _Capture()
            with capture_scores(model, cap), GC.machine_class(capturing_machine(cap)):
                g0 = infer_controller.decode_one(model, prep)
            sc0 = gi_run._score(scores, task, g0.sql)
            steps = [{"step": st["step"], "op": st["op"], "kind": st["kind"], "legal": len(st["targets"]),
                      "g4": step_analysis(st, g4_biases), "oracle": step_analysis(st, oracle)} for st in cap.steps]
            row = {"fit": key, "task_id": tid, "db_id": task.db_id, "g0": {"sql": g0.sql, "correct": sc0["correct"],
                                                                              "status": sc0["predicted"]["status"]},
                   "g4_record_valid": bool(g and g.valid), "gold_convertible": bool(gold.get("convertible")),
                   "first_table_g0": first_table_class(g0.actions, gold.get("first_table"), gold_tables)
                   if gold.get("convertible") else "undefined (gold not convertible)", "steps": steps}
            if gold.get("convertible"):
                log: list = []
                with GC.machine_class(restricted_machine(gold_tables, gold_cols, log)):
                    rr = infer_controller.decode_one(model, prep)
                scr = gi_run._score(scores, task, rr.sql)
                row["restricted"] = {"sql": rr.sql, "correct": scr["correct"], "status": scr["predicted"]["status"],
                                     "restricted_steps": sum(1 for x in log if not x["undefined"]),
                                     "undefined_steps": sum(1 for x in log if x["undefined"]),
                                     "first_table": first_table_class(rr.actions, gold["first_table"], gold_tables)}
            else:
                row["restricted"] = {"undefined": "the gold program is not convertible"}
            gold_steps = train.gold_steps(data.gold(tid).sql, enc, "v2")[0]
            if not gold_steps:
                row["forced"] = {"undefined": "the gold program is not representable"}
            else:
                gold_actions = [s.action for s in gold_steps[:-1]]
                pairs = list(zip([a.as_json() for a in g0.actions], [a.as_json() for a in gold_actions]))
                d = next((i for i, (x, y) in enumerate(pairs) if x != y), None)
                if d is None and len(g0.actions) == len(gold_actions):
                    row["forced"] = {"no_divergence": True}
                else:
                    d = d if d is not None else min(len(g0.actions), len(gold_actions))
                    if d >= len(gold_actions):
                        row["forced"] = {"undefined": "G0 continues past the end of the gold traversal"}
                    else:
                        log = []
                        with GC.machine_class(forced_machine(gold_actions[:d + 1], log)):
                            fr = infer_controller.decode_one(model, prep)
                        scf = gi_run._score(scores, task, fr.sql)
                        div_op = gold_actions[d].op
                        row["forced"] = {"forced_steps": d + 1, "divergence_op": div_op,
                                         "divergence_kind": A.POINTER_KIND[div_op],
                                         "illegal_forced": [x for x in log if "illegal" in x],
                                         "sql": fr.sql, "correct": scf["correct"],
                                         "status": scf["predicted"]["status"],
                                         "suffix_reproduces_gold": [a.as_json() for a in fr.actions] ==
                                         [a.as_json() for a in gold_actions],
                                         "first_table": first_table_class(fr.actions, gold.get("first_table"),
                                                                          gold_tables)}
            rows.append(row)
    with open(run_dir / "records.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    out = summarize(rows)
    (run_dir / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1))
    return out


def summarize(rows: list) -> dict:
    out = {}
    for fit in sorted({r["fit"] for r in rows}):
        rs = [r for r in rows if r["fit"] == fit]
        steps = [s for r in rs for s in r["steps"]]
        tab = [s for s in steps if s["kind"] == "table"]
        col = [s for s in steps if s["kind"] == "column"]

        def share(xs, f):
            return {"n": len(xs), "count": sum(1 for x in xs if f(x))}
        res = [r["restricted"] for r in rs if "correct" in r["restricted"]]
        frc = [r["forced"] for r in rs if "correct" in r["forced"]]
        out[fit] = {
            "requests": len(rs), "g0_correct": sum(r["g0"]["correct"] for r in rs),
            "pointer_steps": {"table": len(tab), "column": len(col)},
            "margin_above_ln2": {"table": share(tab, lambda s: s["g4"]["unchangeable_by_any_record"]),
                                 "column": share(col, lambda s: s["g4"]["unchangeable_by_any_record"])},
            "g4_could_change": share([s for s in steps if s["g4"]["prior"]],
                                     lambda s: s["g4"]["prior"]["could_change"]),
            "oracle_could_change": share([s for s in steps if s["oracle"]["prior"]],
                                         lambda s: s["oracle"]["prior"]["could_change"]),
            "g4_record_invalid": sum(not r["g4_record_valid"] for r in rs),
            "first_table_g0": {c: sum(r["first_table_g0"] == c for r in rs)
                               for c in sorted({r["first_table_g0"] for r in rs})},
            "restricted": {"defined": len(res), "undefined": len(rs) - len(res),
                           "correct": sum(x["correct"] for x in res),
                           "g0_correct_on_same": sum(r["g0"]["correct"] for r in rs if "correct" in r["restricted"]),
                           "wrong_to_correct": sum(1 for r in rs if "correct" in r["restricted"]
                                                   and r["restricted"]["correct"] > r["g0"]["correct"]),
                           "correct_to_wrong": sum(1 for r in rs if "correct" in r["restricted"]
                                                   and r["restricted"]["correct"] < r["g0"]["correct"]),
                           "requests_with_undefined_steps": sum(1 for x in res if x["undefined_steps"]),
                           "first_table": {c: sum(x["first_table"] == c for x in res)
                                           for c in sorted({x["first_table"] for x in res})}},
            "forced": {"forced": len(frc), "no_divergence": sum(1 for r in rs if r["forced"].get("no_divergence")),
                       "undefined": {u: sum(1 for r in rs if r["forced"].get("undefined") == u)
                                     for u in sorted({r["forced"]["undefined"] for r in rs
                                                      if "undefined" in r["forced"]})},
                       "correct": sum(x["correct"] for x in frc),
                       "g0_correct_on_same": sum(r["g0"]["correct"] for r in rs if "correct" in r["forced"]),
                       "suffix_reproduces_gold": sum(x["suffix_reproduces_gold"] for x in frc),
                       "illegal_forced_actions": sum(len(x["illegal_forced"]) for x in frc),
                       "by_divergence_kind": {k: {"n": sum(1 for x in frc
                                                           if (x["divergence_kind"] or "operator") == k),
                                                  "correct": sum(x["correct"] for x in frc
                                                                 if (x["divergence_kind"] or "operator") == k)}
                                              for k in sorted({x["divergence_kind"] or "operator" for x in frc})}}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ground-run", default="20261103T191242Z_dev", help="the dev G4 run (prompt v2)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--allow-dirty", action="store_true")
    t0 = time.time()
    run(ap.parse_args(argv))
    print(f"{time.time() - t0:.0f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
