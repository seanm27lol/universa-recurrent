"""S2 diagnosis, part 1: what each interpreter proposed for each S2 question, from the decision records of a
completed interpreter robustness run (protocols/interpreter_robustness_v1.md). Descriptive only: it analyses the
records, and changes no protocol, record or verdict of that study.

    .venv-phase2/bin/python research/ti_tnn/bench/s2_diagnosis.py \
        experiments/results/ti_tnn/interpreter_robustness/20261018T232057 --out <dir>

It first runs the robustness verifier on the run, and stops unless verification passes. It then regenerates each
world's study with the frozen generator and pairs every S2 question with its S2_base counterpart: the same draw
group, so the same program, entity and frame, with one Base phrase replaced by its taught synonym in S2.

**Execution order.** A program is a list in execution order: index 0 is applied first, to the entity. In a prefix
question that is the innermost phrase ("what is [second] [first] X ?"). `taught_position` is the execution-order
index of the taught relation (0: the taught relation is applied first). The study does not store it; it is
recovered by re-rendering. Exactly one choice of Base phrases must reproduce the S2_base text, and exactly one
position must then turn it into the S2 text when its phrase is replaced by the taught one.

**Classes.** Each question gets exactly one class, from its proposed program and the reference program, both in
execution order:
- `exact`: the program and the entity are both right;
- `entity_only`: the program is right but the entity is wrong;
- `refusal`: the interpreter gave no complete program (`<unsupported>`, `<ambiguous>`, malformed output, or input
  the character contract cannot encode), with the reason;
- `dropped`: one reference relation is missing and the rest are in order (records which relation, its position,
  and whether it is the taught one);
- `extra`: one relation is added;
- `substituted`: one position holds another relation (records the expected and the replacement);
- `reordered`: the same relations in another order;
- `ambiguous`: more than one of the descriptions above fits; `other`: none fits.

`ambiguous` and `other` are kept as explicit classes, not folded into the others. Labels refine `dropped` and
`substituted` by whether the affected relation is the taught one. Each question also records whether the runtime
answered, whether the answer was right (a wrong program can still give the right answer), whether the entity was
right, and the refusal reason. A runtime refusal of a complete program (an ill-typed program, an unknown entity)
keeps its program class and is counted as unanswered.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import interpreter_robustness as S  # noqa: E402
import interpreter_robustness_verify as V  # noqa: E402
from integrated import relations as W  # noqa: E402

INTERPRETER_REFUSALS = ("unsupported", "ambiguous", "malformed", "unsupported characters")
CLASSES = ("exact", "entity_only", "dropped", "extra", "substituted", "reordered", "refusal", "ambiguous", "other")
STRATA = ("taught_position", "pair", "pair_position", "taught_relation", "tokens", "world", "init", "frame")
POSITION_NAMES = {0: "taught first", 1: "taught second"}
EXAMPLES_PER_LABEL = 5


# --- pairing and execution order -------------------------------------------------------------------------------
def base_phrases_of(base: dict) -> list[str]:
    """The Base prefix phrases (templates, execution order) that render the S2_base text; exactly one choice must."""
    p, entity, frame = base["program"], base["entity"], base["frame"]
    fits = [list(c) for c in itertools.product(*(S.BASE_PREFIX[r] for r in p))
            if S.render(entity, list(c), frame) == base["text"]]
    if len(fits) != 1:
        raise ValueError(f"{base['id']}: {len(fits)} Base phrasings render {base['text']!r}")
    return fits[0]


def taught_position(s2: dict, base: dict) -> int:
    """The execution-order index of the taught relation in an S2 question, by re-rendering against its pair."""
    if (s2["program"], s2["entity"], s2["frame"], s2["group"]) != (base["program"], base["entity"], base["frame"],
                                                                  base["group"]):
        raise ValueError(f"{s2['id']} and {base['id']} are not a matched pair")
    phrases = base_phrases_of(base)
    p = s2["program"]
    js = [j for j in range(len(p))
          if S.render(s2["entity"], [S.TAUGHT[r] if i == j else phrases[i] for i, r in enumerate(p)], s2["frame"])
          == s2["text"]]
    if len(js) != 1:
        raise ValueError(f"{s2['id']}: {len(js)} taught positions render {s2['text']!r}")
    return js[0]


def pairs_of(items: list[dict]) -> list[tuple[dict, dict, int]]:
    """(S2, S2_base, taught_position) for every S2 question, in study order. Every S2 must have exactly one pair."""
    by_group = defaultdict(dict)
    for x in items:
        if x["category"] in ("S2", "S2_base"):
            if x["category"] in by_group[x["group"]]:
                raise ValueError(f"group {x['group']} has two {x['category']} questions")
            by_group[x["group"]][x["category"]] = x
    out = []
    for x in items:
        if x["category"] == "S2":
            g = by_group[x["group"]]
            if "S2_base" not in g:
                raise ValueError(f"{x['id']} has no S2_base pair")
            out.append((x, g["S2_base"], taught_position(x, g["S2_base"])))
    if len(out) != len([x for x in items if x["category"] == "S2_base"]):
        raise ValueError("unpaired S2_base questions")
    return out


# --- classification --------------------------------------------------------------------------------------------
def compare_programs(expected, proposed) -> tuple:
    """How a proposed program differs from the reference (both in execution order). Returns ("exact",),
    ("dropped", i, relation), ("extra", i, relation), ("substituted", i, expected, replacement), ("reordered",),
    ("ambiguous", [descriptions]) or ("other",). Positions are execution-order indices."""
    e, p = tuple(expected), tuple(proposed)
    if p == e:
        return ("exact",)
    found = []
    if len(p) == len(e) - 1:
        found += [("dropped", i, e[i]) for i in range(len(e)) if e[:i] + e[i + 1:] == p]
    if len(p) == len(e) + 1:
        found += [("extra", i, p[i]) for i in range(len(p)) if p[:i] + p[i + 1:] == e]
    if len(p) == len(e):
        diff = [i for i in range(len(e)) if e[i] != p[i]]
        if len(diff) == 1:
            found.append(("substituted", diff[0], e[diff[0]], p[diff[0]]))
        elif sorted(e) == sorted(p):
            found.append(("reordered",))
    if len(found) == 1:
        return found[0]
    return ("ambiguous", found) if found else ("other",)


def outcome(item: dict, rec: dict, k: int, threshold) -> dict:
    """One question's record, read without any stored flag. `threshold` is the verified certificate (None: none)."""
    reason = S.REASONS[rec["reason"][k]]
    answered = bool(rec["decision"][k])
    program, entity = rec["program"][k], rec["entity"][k]
    return {"proposed": list(program) if program is not None else None, "proposed_entity": entity,
            "reason": reason, "answered": answered, "score": rec["score"][k],
            "correct": answered and rec["answer"][k] == item["answer"],
            "entity_right": entity == item["entity"],
            "exact": program is not None and list(program) == item["program"] and entity == item["entity"],
            "accepted": answered and threshold is not None and rec["score"][k] >= threshold}


def classify(item: dict, o: dict, taught_at: int | None) -> dict:
    """The class and label of one question (module docstring). `taught_at` is the taught relation's
    execution-order index (None for a question without taught wording)."""
    if o["reason"] in INTERPRETER_REFUSALS or o["proposed"] is None:
        return {"class": "refusal", "label": f"refusal:{o['reason']}", "detail": {"partial": o["proposed"]}}
    d = compare_programs(item["program"], o["proposed"])
    kind = d[0]
    if kind == "exact":
        return {"class": "exact" if o["entity_right"] else "entity_only",
                "label": "exact" if o["entity_right"] else "entity_only", "detail": {}}
    which = lambda i: "taught" if taught_at is not None and i == taught_at else "other"
    if kind == "dropped":
        return {"class": "dropped", "label": f"dropped:{which(d[1])}",
                "detail": {"position": d[1], "relation": d[2]}}
    if kind == "substituted":
        return {"class": "substituted", "label": f"substituted:{which(d[1])}",
                "detail": {"position": d[1], "expected": d[2], "replacement": d[3]}}
    if kind == "extra":
        return {"class": "extra", "label": "extra", "detail": {"position": d[1], "relation": d[2]}}
    if kind == "ambiguous":
        return {"class": "ambiguous", "label": "ambiguous", "detail": {"descriptions": [list(x) for x in d[1]]}}
    return {"class": kind, "label": kind, "detail": {}}


def question_rows(items: list[dict], rec: dict, ids: list[str], threshold, world: int, init: int, arm: str) -> list[dict]:
    """One row per S2 question: its stratum keys, its outcome and class, and its S2_base pair's outcome and class."""
    index = {x: k for k, x in enumerate(ids)}
    rows = []
    for s2, base, j in pairs_of(items):
        o, ob = outcome(s2, rec, index[s2["id"]], threshold), outcome(base, rec, index[base["id"]], threshold)
        p = s2["program"]
        rows.append({"world": world, "init": init, "arm": arm, "id": s2["id"], "base_id": base["id"],
                     "program": list(p), "entity": s2["entity"], "pair": " > ".join(p),
                     "taught_position": POSITION_NAMES[j], "pair_position": f"{' > '.join(p)} | {POSITION_NAMES[j]}",
                     "taught_relation": p[j], "taught_phrase": S.TAUGHT[p[j]],
                     "other_relation": p[1 - j], "tokens": len(s2["text"].split()), "frame": s2["frame"],
                     "text": s2["text"], "base_text": base["text"],
                     "s2": {**o, **classify(s2, o, j)}, "base": {**ob, **classify(base, ob, None)}})
    return rows


# --- aggregation -----------------------------------------------------------------------------------------------
def _answer_state(o) -> str:
    return "correct" if o["correct"] else "wrong" if o["answered"] else "unanswered"


def counts_table(rows: list[dict]) -> dict:
    """Label -> {n, correct, wrong, unanswered, entity_wrong}, over every S2 question (a complete partition)."""
    t = defaultdict(Counter)
    for r in rows:
        s = r["s2"]
        t[s["label"]]["n"] += 1
        t[s["label"]][_answer_state(s)] += 1
        t[s["label"]]["entity_wrong"] += not s["entity_right"]
    return {k: {f: v[f] for f in ("n", "correct", "wrong", "unanswered", "entity_wrong")} for k, v in sorted(t.items())}


def details(rows: list[dict]) -> dict:
    """Which relation was dropped, which substitution was made, which reason refused, by taught position."""
    out = {"dropped": Counter(), "substituted": Counter(), "reordered": Counter(), "refusal": Counter(),
           "extra": Counter(), "other": Counter(), "ambiguous": Counter()}
    for r in rows:
        s, pos = r["s2"], r["taught_position"]
        if s["class"] == "dropped":
            out["dropped"][f"{pos} | dropped {s['detail']['relation']} (position {s['detail']['position']}, "
                           f"{s['label'].split(':')[1]}) from {r['pair']}"] += 1
        elif s["class"] == "substituted":
            d = s["detail"]
            out["substituted"][f"{pos} | {d['expected']} -> {d['replacement']} (position {d['position']}, "
                               f"{s['label'].split(':')[1]})"] += 1
        elif s["class"] == "refusal":
            out["refusal"][f"{pos} | {s['detail']['partial'] and ' > '.join(s['detail']['partial']) or '-'} | "
                           f"{s['reason']}"] += 1
        elif s["class"] in ("reordered", "extra", "other", "ambiguous"):
            out[s["class"]][f"{pos} | {r['pair']} -> {' > '.join(s['proposed'])}"] += 1
    return {k: dict(v.most_common()) for k, v in out.items()}


def strata(rows: list[dict]) -> dict:
    """For each stratum key: value -> {n, exact, correct, label counts}."""
    out = {}
    for key in STRATA:
        t = defaultdict(lambda: {"n": 0, "exact": 0, "correct": 0, "labels": Counter()})
        for r in rows:
            cell = t[str(r[key])]
            cell["n"] += 1
            cell["exact"] += r["s2"]["exact"]
            cell["correct"] += r["s2"]["correct"]
            cell["labels"][r["s2"]["label"]] += 1
        out[key] = {v: {**c, "labels": dict(sorted(c["labels"].items()))} for v, c in sorted(t.items())}
    return out


def unit_metrics(rows: list[dict]) -> dict:
    """Per-unit S2 and S2_base metrics, and the paired counts."""
    def side(key):
        n = len(rows)
        o = [r[key] for r in rows]
        return {"n": n, "exact": sum(x["exact"] for x in o) / n, "answer_accuracy": sum(x["correct"] for x in o) / n,
                "coverage": sum(x["answered"] for x in o) / n,
                "certified_answered": sum(x["accepted"] for x in o),
                "certified_errors": sum(x["accepted"] and not x["correct"] for x in o),
                "reasons": dict(sorted(Counter(x["reason"] for x in o).items())),
                "wrong_program_right_answer": sum(x["correct"] and not x["exact"] for x in o)}
    return {"S2": side("s2"), "S2_base": side("base"),
            "base_correct_taught_not": sum(r["base"]["correct"] and not r["s2"]["correct"] for r in rows),
            "base_exact_taught_not": sum(r["base"]["exact"] and not r["s2"]["exact"] for r in rows),
            "taught_correct_base_not": sum(r["s2"]["correct"] and not r["base"]["correct"] for r in rows)}


def examples(rows: list[dict], k: int = EXAMPLES_PER_LABEL) -> dict:
    """A deterministic sample per label: the k rows with the smallest sha256 of (arm, world, init, id)."""
    by = defaultdict(list)
    for r in rows:
        by[r["s2"]["label"]].append(r)
    key = lambda r: hashlib.sha256(f"s2-diagnosis|{r['arm']}|{r['world']}|{r['init']}|{r['id']}".encode()).hexdigest()
    keep = ("world", "init", "id", "text", "base_text", "program", "entity", "taught_position")
    return {label: [{**{f: r[f] for f in keep}, "proposed": r["s2"]["proposed"],
                     "proposed_entity": r["s2"]["proposed_entity"], "reason": r["s2"]["reason"],
                     "answered": r["s2"]["answered"], "correct": r["s2"]["correct"], "score": r["s2"]["score"],
                     "base_proposed": r["base"]["proposed"], "base_correct": r["base"]["correct"]}
                    for r in sorted(rs, key=key)[:k]]
            for label, rs in sorted(by.items())}


def summarize(rows_by_arm: dict[str, list[dict]]) -> dict:
    out = {"counts": {}, "details": {}, "strata": {}, "per_unit": {}, "examples": {}}
    for arm, rows in rows_by_arm.items():
        out["counts"][arm] = counts_table(rows)
        out["details"][arm] = details(rows)
        out["strata"][arm] = strata(rows)
        units = defaultdict(list)
        for r in rows:
            units[f"{r['world']}/{r['init']}"].append(r)
        out["per_unit"][arm] = {u: unit_metrics(rs) for u, rs in sorted(units.items())}
        out["examples"][arm] = examples(rows)
    return out


# --- the historical records ----------------------------------------------------------------------------------
def analyse_run(run_dir: Path, require_confirmation: bool = True) -> tuple[dict, dict]:
    """Verify the run, then classify every S2 question of every unit and arm (and the grammar control).
    Returns (summary, rows_by_arm). The summary's per-unit S2 metrics are checked against the verifier's
    independently derived category metrics."""
    run_dir = Path(run_dir)
    verified = V.verify(run_dir, require_confirmation=require_confirmation)
    result = json.loads((run_dir / "interpreter_robustness.json").read_text())
    cfg = result["config"]
    rows_by_arm = defaultdict(list)
    for w in cfg["worlds"]:
        world = W.generate_world(w, n_countries=cfg["countries"])
        study = S.make_study(world, w, result["sizes"])
        held = [x for x in study["items"] if x["split"] != "train"]
        evals = [x for x in held if x["split"] == "test"]
        for i in cfg["inits"]:
            with gzip.open(run_dir / "records" / f"w{w}_i{i}.json.gz", "rt") as f:
                rec = json.load(f)
            derived = verified["derived"][(w, i)]
            for arm in S.ARMS:
                threshold = derived[arm]["policies"]["certified"]["certificate"]["threshold"]
                rows = question_rows(held, rec["arms"][arm], rec["ids"], threshold, w, i, arm)
                check_against_verifier(rows, derived[arm]["policies"], f"w{w} i{i} {arm}")
                rows_by_arm[arm] += rows
            rows_by_arm["grammar"] += question_rows(evals, rec["grammar"], [x["id"] for x in evals], None, w, i,
                                                    "grammar")
    summary = summarize(rows_by_arm)
    summary["source"] = {"run": run_dir.name, "worlds": cfg["worlds"], "inits": cfg["inits"],
                         "verified": verified["verified"], "verdict_predictions": verified["verdict"]["predictions"]
                         if verified["verdict"] else None}
    return summary, rows_by_arm


def check_against_verifier(rows, policies, where):
    """The analysis reads the records directly; its S2 and S2_base totals must equal the verifier's."""
    for key, cat in (("s2", "S2"), ("base", "S2_base")):
        none, cert = policies["none"]["categories"][cat], policies["certified"]["categories"][cat]
        o = [r[key] for r in rows]
        got = (len(o), sum(x["answered"] for x in o), sum(x["answered"] and not x["correct"] for x in o),
               sum(x["exact"] for x in o) / len(o), sum(x["accepted"] for x in o),
               sum(x["accepted"] and not x["correct"] for x in o))
        want = (none["n"], none["answered"], none["errors"], none["exact_parse"], cert["answered"], cert["errors"])
        if not all(V._close(a, b) for a, b in zip(got, want)):
            raise V.VerificationError(f"{where} {cat}: analysis {got} != verifier {want}")


# --- tables ----------------------------------------------------------------------------------------------------
LABEL_ORDER = ("exact", "entity_only", "dropped:taught", "dropped:other", "substituted:taught", "substituted:other",
               "reordered", "extra", "ambiguous", "other")


def tables(summary: dict) -> str:
    lines = []
    arms = list(summary["counts"])
    labels = [l for l in LABEL_ORDER if any(l in summary["counts"][a] for a in arms)]
    labels += sorted({l for a in arms for l in summary["counts"][a] if l.startswith("refusal:")})
    lines.append("## S2 questions by class (all units; n, of which answered correctly / answered wrongly / unanswered)\n")
    lines.append("| class | " + " | ".join(arms) + " |")
    lines.append("|---|" + "---:|" * len(arms))
    for l in labels:
        cells = []
        for a in arms:
            c = summary["counts"][a].get(l)
            cells.append(f"{c['n']} ({c['correct']}/{c['wrong']}/{c['unanswered']})" if c else "0")
        lines.append(f"| {l} | " + " | ".join(cells) + " |")
    lines.append("| total | " + " | ".join(str(sum(c["n"] for c in summary["counts"][a].values())) for a in arms) + " |")
    for a in arms:
        lines.append(f"\n## {a}: by taught position (execution order)\n")
        st = summary["strata"][a]["taught_position"]
        lines.append("| class | " + " | ".join(st) + " |")
        lines.append("|---|" + "---:|" * len(st))
        for l in labels:
            lines.append(f"| {l} | " + " | ".join(str(st[v]["labels"].get(l, 0)) for v in st) + " |")
        lines.append("| n | " + " | ".join(str(st[v]["n"]) for v in st) + " |")
    for a in arms:
        lines.append(f"\n## {a}: S2 per unit (exact = program and entity; answer = answered correctly)\n")
        lines.append("| unit | S2 exact | S2 answer | S2 coverage | S2_base exact | S2_base answer | base right, S2 not "
                     "| wrong program, right answer | refusal reasons (S2) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for u, m in summary["per_unit"][a].items():
            reasons = ", ".join(f"{k} {v}" for k, v in m["S2"]["reasons"].items() if k != "answered") or "-"
            lines.append(f"| {u} | {m['S2']['exact']:.3f} | {m['S2']['answer_accuracy']:.3f} | {m['S2']['coverage']:.3f} "
                         f"| {m['S2_base']['exact']:.3f} | {m['S2_base']['answer_accuracy']:.3f} "
                         f"| {m['base_correct_taught_not']} | {m['S2']['wrong_program_right_answer']} | {reasons} |")
        for kind in ("dropped", "substituted", "reordered", "extra", "refusal", "ambiguous", "other"):
            entries = summary["details"][a][kind]
            if entries:
                lines.append(f"\n{a}, {kind} (count | taught position | detail):\n")
                for k, v in sorted(entries.items(), key=lambda kv: (-kv[1], kv[0])):
                    lines.append(f"- {v} | {k}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--not-confirmation", action="store_true", help="verify without the frozen-configuration checks")
    args = p.parse_args(argv)
    summary, _ = analyse_run(args.run_dir, require_confirmation=not args.not_confirmation)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "s2_diagnosis.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    text = tables(summary)
    (args.out / "s2_diagnosis_tables.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
