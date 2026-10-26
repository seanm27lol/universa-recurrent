"""A+K's diagnosis and the corrected accounting, from a pilot run's saved records only (no fit, no new decoding).

    .venv-phase2/bin/python research/ti_tnn/bench/factored_pilot_diagnose.py <run dir> --out diagnosis.json

**Example.** In unit (world 100, init 100, full), one S2 question is "name the company operating the airfield of
country_29_city_0 ." A+K's record shows:
- the grammar admitted it with two slots;
- A's entity pointer chose `country_29_city_0`;
- step 0 chose `airport_of` and step 1 chose `operator_of`;
- the score cleared A+K's certified threshold.

So it is a **correct program, released**. Every S2 and non-cancelling S1 question of every unit is put in exactly
one class:

| Class | Meaning |
|---|---|
| grammar refusal | the grammar did not admit the question (A+K abstains with its reason) |
| entity failure | A's entity pointer chose another token than the question's entity (an unknown token, or another entity) |
| decoder refusal | right entity, but the constrained decoder chose `<unsupported>` or `<ambiguous>` |
| wrong program, released / filtered | right entity, a complete program other than the reference; its score is at or above A+K's certified threshold (released) or below it (filtered) |
| correct program, released / filtered | right entity and the reference program; released or filtered as above |

**The policies.** Under the `none` policy every proposed program is released. The threshold is the one A+K's
exact-program certificate selected on calibration. Nothing here chooses a threshold from evaluation results.

**For wrong programs:**
- the first position at which the program differs from the reference;
- the selected and the reference relation there, with their recorded probabilities at that step;
- at that step, **the unmasked top output** (`preference`). Each A+K record keeps the network's full distribution
  before the mask, so the output it ranks first is known: the chosen output, `<end>` where the grammar still
  required a relation, or a relation the mask forbids. The probability it gave `<end>` and the total it gave
  forbidden relations are recorded with it;
- the entity pointer's probability;
- the original probability of the final `<end>`, the step that completes the program.

**What that distribution is.** It is A's network conditioned on **A+K's own selected prefix**: every step feeds back
A+K's previous choices, not A's. Up to the first divergence from the reference, that prefix is the reference
prefix. So the unmasked top output says what the network prefers after A+K's choices; it is **not** A's own
trajectory. A's own trajectory comes from A's records and is analysed separately (`first_divergence`).

**The same source type, by construction.** At the first divergence, A+K's prefix equals the reference prefix and
the entity is right, so the mask admits only relations whose source is the reference relation's source. Every
wrong program's first wrong relation therefore shares its source type with the reference relation: that follows
from the mask, not from the data. In these worlds the only such pairs are `airport_of`/`country_of` (from a City)
and `capital_of`/`largest_city_of` (from a Country). The empirical findings are which of them occurred, where, how
often, and with what probabilities.

**For correct programs,** the unmasked top output at the weakest step: the step whose chosen output had the
lowest probability, the factor that most limits the score.

**Also reported, from the same records:**
- **coverage and accepted-error bounds,** separately, under both policies;
- **A against A+K** on the same questions, under **unthresholded** exact correctness (every proposed program counts).
  Separately, how many questions A had exact had their A+K program filtered by A+K's certified threshold;
- **A's own trajectory**, where only A+K was exact: where A's own program first left the reference
  (`first_divergence`, from A's records);
- **every recorded A+K step whose allowed set tied at its maximum probability.** The pilot recorded at `0e1a3c9`
  chose by probability. Where the maximum is unique, choosing by logit would have made the same choice.
- **the corrected accounting:**
  - A's emitted decisions, against its evaluated decoder row-steps;
  - A+K's emitted decisions, against the row-steps its batched decoder evaluated;
  - A's training MACs at the padded batch width, beside the unpadded figure the run recorded.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import factored_interpreter_verify as FV  # noqa: E402  (the regimes, re-implemented independently of the producer)
import factored_pilot_verify as PV  # noqa: E402  (MAC formulas and the batch replay, independent of the producer)
import interpreter_robustness as IR  # noqa: E402  (the generator)
from integrated import calibration as K  # noqa: E402
from integrated import relations as W  # noqa: E402

CLASSES = ("grammar refusal", "entity failure", "decoder refusal", "wrong program, released",
           "wrong program, filtered", "correct program, released", "correct program, filtered")
CATEGORIES = ("S2", "S1:length3")


def _load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def quartiles(values):
    if not values:
        return None
    v = sorted(values)
    q = statistics.quantiles(v, n=4) if len(v) > 1 else [v[0]] * 3
    return {"n": len(v), "min": v[0], "q1": q[0], "median": q[1], "q3": q[2], "max": v[-1]}


PREFERENCES = ("the chosen output", "<end>", "a relation the constraint forbids", "another allowed output")


def preference(step, outputs):
    """The output the unmasked distribution puts first at a recorded A+K step; the probability it gave `<end>`; and
    the total it gave relations outside the step's allowed set.

    The distribution is A's network conditioned on A+K's own selected prefix (see the module docstring), not A's
    own trajectory. In the recorded runs its top output is either the one chosen, or something the mask did not
    allow there: `<end>` (when the grammar still required a relation) or a relation of the wrong source type."""
    dist, allowed = step["distribution"], set(step["allowed"])
    end = outputs.index("<end>")
    relations = [k for k, o in enumerate(outputs) if not o.startswith("<")]
    top = max(range(len(dist)), key=lambda k: (dist[k], -k))
    if top == step["chosen"]:
        kind = PREFERENCES[0]
    elif top == end and end not in allowed:
        kind = PREFERENCES[1]
    elif top in relations and top not in allowed:
        kind = PREFERENCES[2]
    else:
        kind = PREFERENCES[3]
    return kind, dist[end], sum(dist[k] for k in relations if k not in allowed)


def first_divergence(a_rec, item, world, type_of):
    """Where A's own greedy program first left the reference.

    It stopped early (its program is a proper prefix of the reference), ran long, or chose another relation at some
    position. In that case: was its source type the right one there? Otherwise A produced no program, or chose
    another entity. Judged by A's first divergence, not its length: a program that begins with a wrong-type relation
    and then ends early is a wrong-type choice."""
    if a_rec["outcome"] != "program":
        return f"A {a_rec['outcome']}"
    if a_rec["entity"] != item["entity"]:
        return "A wrong entity"
    prog, ref = a_rec["steps"], item["program"]
    j = next((i for i, (u, v) in enumerate(zip(prog, ref)) if u != v), None)
    if j is None:
        return "A stopped early" if len(prog) < len(ref) else "A ran long"
    at = type_of[item["entity"]] if j == 0 else world.relations[ref[j - 1]].target
    right = world.relations[prog[j]].source == at
    return f"A first chose a relation of the {'right' if right else 'wrong'} source type, at position {j}"


def classify(item, rec, threshold, outputs):
    if rec.get("grammar") != "ok":
        return "grammar refusal", None
    if rec["entity"] != item["entity"]:
        return "entity failure", None
    if rec["outcome"] != "program":
        return "decoder refusal", None
    released = threshold is not None and rec["score"] >= threshold
    kind = "correct program" if rec["program"] == item["program"] else "wrong program"
    detail = None
    if kind == "wrong program":
        j = next((i for i, (a, b) in enumerate(zip(rec["program"], item["program"])) if a != b),
                 min(len(rec["program"]), len(item["program"])))
        if j >= min(len(rec["program"]), len(item["program"])):  # a length difference (not expected under the slot count)
            return f"{kind}, {'released' if released else 'filtered'}", {"position": j, "selected": "<length>",
                                                                       "reference": "<length>", "p_selected": None,
                                                                       "p_reference": None, "reference_allowed": None,
                                                                       "unmasked_top": None, "p_end": None,
                                                                       "p_forbidden": None}
        st = rec["steps"][j]
        sel, ref = rec["program"][j], item["program"][j]
        top, p_end, p_forbidden = preference(st, outputs)
        detail = {"position": j, "selected": sel, "reference": ref,
                  "p_selected": st["distribution"][outputs.index(sel)],
                  "p_reference": st["distribution"][outputs.index(ref)],
                  "reference_allowed": outputs.index(ref) in st["allowed"],
                  "unmasked_top": top, "p_end": p_end, "p_forbidden": p_forbidden}
    return f"{kind}, {'released' if released else 'filtered'}", detail


def diagnose_unit(run_dir: Path, info: dict, countries: int, updates: int, batch: int) -> dict:
    d = run_dir / info["path"]
    unit = json.loads((d / "unit.json").read_text())
    world = W.generate_world(info["world"], n_countries=countries)
    study = IR.make_study(world, info["world"])
    train, aug, _ = FV.regime_items(study, info["regime"], world, info["world"])
    items = [x for x in study["items"] if x["split"] != "train"]
    ak, a = _load(d / "records_AK.json.gz"), _load(d / "records_A.json.gz")
    outputs = sorted(world.relations) + ["<end>", "<unsupported>", "<ambiguous>"]
    type_of = {v: t for t in sorted(world.types) for v in world.types[t].values}
    cert = unit["systems"]["AK"]["certificate"]
    threshold = cert["threshold"] if cert["certified"] else None
    out = {"unit": f"w{info['world']} i{info['init']} {info['regime']}", "ak_certified_threshold": threshold}
    for cat in CATEGORIES:
        sel = [(x, r, ra) for x, r, ra in zip(items, ak, a) if x["split"] == "test"
               and (x["category"] == cat or (cat == "S1:length3" and x["category"] == "S1" and x["family"] == "length3"))]
        counts, wrong_detail, probs, preferred = Counter(), Counter(), defaultdict(list), defaultdict(Counter)
        wrong_by_preference = Counter()
        for x, r, ra in sel:
            cls, detail = classify(x, r, threshold, outputs)
            counts[cls] += 1
            if r.get("grammar") == "ok" and r["outcome"] == "program" and r["entity"] == x["entity"]:
                group = "wrong" if detail else "correct"
                probs[f"{group}: entity probability"].append(r["entity_probability"])
                probs[f"{group}: final <end> probability"].append(r["steps"][-1]["distribution"][outputs.index("<end>")])
                probs[f"{group}: score"].append(r["score"])
                if detail:
                    where = f"position {detail['position']}: {detail['reference']} -> {detail['selected']}"
                    wrong_detail[where] += 1
                    if detail["p_selected"] is not None:
                        probs["wrong: p(selected) at the first difference"].append(detail["p_selected"])
                        probs["wrong: p(reference) at the first difference"].append(detail["p_reference"])
                        probs["wrong: p(<end>) at the first difference"].append(detail["p_end"])
                        probs["wrong: p(forbidden relations) at the first difference"].append(detail["p_forbidden"])
                        preferred["wrong programs, at the first difference"][detail["unmasked_top"]] += 1
                        wrong_by_preference[f"{where}; unmasked top output {detail['unmasked_top']}"] += 1
                        mass = {PREFERENCES[0]: detail["p_selected"], PREFERENCES[1]: detail["p_end"],
                                PREFERENCES[2]: detail["p_forbidden"]}.get(detail["unmasked_top"])
                        if mass is not None:
                            probs[f"wrong, where the unmasked top output was {detail['unmasked_top']}: its probability "
                                  "at the first difference"].append(mass)
                else:
                    weakest = min(r["steps"], key=lambda st: st["distribution"][st["chosen"]])
                    preferred["correct programs, at the weakest step"][preference(weakest, outputs)[0]] += 1
        n = len(sel)
        exact = lambda r, x: (r.get("grammar") == "ok" and r["outcome"] == "program" and r["entity"] == x["entity"]  # noqa: E731
                              and r["program"] == x["program"])
        policies = {}
        for policy in ("none", "certified"):
            rel = [(x, r) for x, r, _ in sel if r.get("grammar") == "ok" and r["outcome"] == "program"
                   and (policy == "none" or (threshold is not None and r["score"] >= threshold))]
            errors = sum(not exact(r, x) for x, r in rel)
            policies[policy] = {"n": n, "released": len(rel), "coverage": len(rel) / n, "errors": errors,
                                "error_upper_95": K.clopper_pearson_upper(errors, len(rel)),
                                "exact_accuracy": sum(exact(r, x) for x, r in rel) / n}
        a_exact = lambda ra, x: (ra["outcome"] == "program" and ra["steps"] == x["program"]  # noqa: E731
                                 and ra["entity"] == x["entity"])
        cross = Counter((("A exact" if a_exact(ra, x) else "A not exact") + " / "
                         + ("A+K exact" if exact(r, x) else "A+K not exact")) for x, r, ra in sel)
        a_kinds = Counter(first_divergence(ra, x, world, type_of) for x, r, ra in sel
                          if not a_exact(ra, x) and exact(r, x))
        # the comparison above counts every proposed program (unthresholded); under A+K's certified threshold, a
        # question A had exact can still lose its (exact) A+K program to the threshold
        filtered = sum(a_exact(ra, x) and exact(r, x) and not (threshold is not None and r["score"] >= threshold)
                       for x, r, ra in sel)
        out[cat] = {"n": n, "classes": {c: counts.get(c, 0) for c in CLASSES}, "policies": policies,
                    "wrong_programs_by_first_difference": dict(wrong_detail.most_common()),
                    "wrong_programs_by_first_difference_and_unmasked_top": dict(wrong_by_preference.most_common()),
                    "probabilities": {k: quartiles(v) for k, v in sorted(probs.items())},
                    "unmasked_top_output_on_the_ak_prefix": {k: dict(v.most_common()) for k, v in sorted(preferred.items())},
                    "a_vs_ak_unthresholded": dict(sorted(cross.items())),
                    "a_exact_with_the_ak_program_filtered_by_its_threshold": filtered,
                    "a_own_trajectory_where_only_ak_was_exact": dict(a_kinds.most_common())}
    # every recorded A+K step: ties at the maximum allowed probability (where logit selection could differ)
    ties = Counter()
    for r in ak:
        for st in r.get("steps", []):
            vals = [st["distribution"][i] for i in st["allowed"]]
            m = max(vals)
            ties["steps"] += 1
            ties["tied maximum among allowed"] += vals.count(m) > 1
            ties["maximum allowed probability is 0.0"] += m == 0.0
            ties["any allowed probability is 0.0"] += min(vals) == 0.0
    out["selection_ties"] = dict(ties)
    # the corrected accounting
    examples = train + aug
    batches = PV.a_batches(len(examples), info["init"], updates, batch)
    lengths = [len(x["text"].split()) for x in examples]
    draws = Counter(i for idx in batches for i in idx)
    costs, ta = unit["evaluation_costs"], unit["training"]["A"]
    out["accounting"] = {  # the recorded pilot (0e1a3c9) used the first field names; corrected runs the second
        "A": {"evaluated_row_steps": len(items) * 4, "emitted_decisions": sum(len(r["step_probabilities"]) for r in a),
              "training_forward_macs_unpadded_recorded": ta.get("forward_macs", ta.get("forward_macs_unpadded")),
              "training_forward_macs_unpadded_recomputed": sum(n * PV.a_macs(lengths[i]) for i, n in draws.items()),
              "training_forward_macs_padded": sum(PV.a_macs_padded([lengths[i] for i in idx]) for idx in batches)},
        "AK": {"emitted_decisions_recorded": costs["AK"].get("decoder_steps", costs["AK"].get("emitted_decisions")),
               "evaluated_row_steps": PV.ak_row_steps(ak)},
        "F": {"training_forward_macs": unit["training"]["F"]["forward_macs"],
              "note": "packed GRU and no attention: padding adds no counted MACs"},
        "F_and_FB_seconds_combined": costs.get("F_and_FB_combined", {}).get("seconds", costs["F"].get("seconds"))}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    run = json.loads((a.run_dir / "run.json").read_text())
    cfg = run["config"]
    units = [diagnose_unit(a.run_dir, info, cfg["countries"], cfg["updates"], cfg["batch"]) for info in run["units"]]
    result = {"run": a.run_dir.name, "commit": run["provenance"]["commit"], "units": units}
    if a.out:
        a.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True)[:3000])


if __name__ == "__main__":
    main()
