"""S2 diagnosis: analysis of the development fits (s2_development.py; protocols/s2_diagnosis_brief.md).

    .venv-phase2/bin/python research/ti_tnn/bench/s2_development_analysis.py <run_dir> [<run_dir> ...] --out <dir>

Nothing stored in a run's summary is trusted. For every fit it re-derives the category metrics, the certificate
and the unit outcomes from the decision records, with the robustness verifier's independent code
(interpreter_robustness_verify.rederive_arm), and stops on any difference from the stored summary. It then reports:
- **Classes**: s2_diagnosis's classification of every S2 question, per arm.
- **Teacher-forced stopping**: at each reference step, with the correct prefix supplied, the probabilities of
  `<end>`, `<unsupported>`, the correct output and the strongest alternative relation. The decisive comparison is
  step 1 (after the first required relation, when a second is still required): P(`<end>`) on S2 against its
  matched S2_base question, by the taught relation's execution-order position. This describes behaviour; it does
  not establish a causal mechanism.
- **Failure kinds** (greedy, program only; the entity is reported separately). With t the first step where the
  greedy output departs from the reference, and E the steps where the teacher-forced argmax is wrong (t is always
  in E, since both decodings share the prefix up to t):
  - `premature_stop`: E = {t} and the greedy output stops (`<end>`) where a relation is required;
  - `relation_confusion`: E = {t}, a wrong relation at t, and the rest of the greedy output is right;
  - `error_propagation`: E = {t}, a wrong relation at t, and the greedy output also departs later, though with the
    correct prefix the model would have continued correctly;
  - `overrun`: E = {t} and a relation where the reference stops;
  - `refusal_token`: E = {t} and `<unsupported>` or `<ambiguous>` at t;
  - `broader`: two or more steps wrong even with the correct prefix.
- **Arms**: the per-unit outcomes and the brief's decision rule.
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
import interpreter_robustness as S  # noqa: E402
import interpreter_robustness_verify as V  # noqa: E402
import s2_diagnosis as D2  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import routing as J  # noqa: E402

TERMINAL = (J.END, J.UNSUPPORTED, J.AMBIGUOUS)
FAILURE_KINDS = ("premature_stop", "relation_confusion", "error_propagation", "overrun", "refusal_token", "broader")
UNIT_FIELDS = ("certified", "sanity", "taught_wording_learned", "joint_novelty", "s2_transfer")


# --- traces ----------------------------------------------------------------------------------------------------
def argmax(dist) -> int:
    return max(range(len(dist)), key=lambda c: dist[c])


def greedy_chain(dists, outputs) -> list[str]:
    """The greedy outputs up to and including the first stop symbol (all steps if none: malformed)."""
    chain = []
    for d in dists:
        chain.append(outputs[argmax(d)])
        if chain[-1] in TERMINAL:
            break
    return chain


def failure_kind(reference: list[str], greedy: list[str], teacher_argmax: list[str]) -> str | None:
    """The greedy failure kind (module docstring), or None when the greedy program is right."""
    if greedy == reference:
        return None
    t = next((s for s in range(min(len(reference), len(greedy))) if greedy[s] != reference[s]),
             min(len(reference), len(greedy)))
    wrong = [s for s in range(len(reference)) if teacher_argmax[s] != reference[s]]
    if t >= len(reference) or t not in wrong or teacher_argmax[t] != greedy[t]:
        raise ValueError(f"inconsistent traces: reference {reference}, greedy {greedy}, teacher {teacher_argmax}")
    if len(wrong) >= 2:
        return "broader"
    g = greedy[t]
    if g in (J.UNSUPPORTED, J.AMBIGUOUS):
        return "refusal_token"
    if g == J.END:
        return "premature_stop"
    if reference[t] == J.END:
        return "overrun"
    return "relation_confusion" if greedy[t + 1:] == reference[t + 1:] else "error_propagation"


def teacher_steps(dists, outputs, reference) -> list[dict]:
    index = {o: k for k, o in enumerate(outputs)}
    relations = [k for k, o in enumerate(outputs) if o not in TERMINAL]
    out = []
    for t, o in enumerate(reference):
        d, c = dists[t], index[o]
        alt = max((r for r in relations if r != c), key=lambda r: d[r])
        out.append({"p_end": d[index[J.END]], "p_unsupported": d[index[J.UNSUPPORTED]],
                    "p_ambiguous": d[index[J.AMBIGUOUS]], "p_correct": d[c], "top_alternative": outputs[alt],
                    "p_top_alternative": d[alt], "argmax": outputs[argmax(d)]})
    return out


def trace_rows(held: list[dict], trace: dict) -> list[dict]:
    """One row per S2 question: teacher-forced steps and greedy chain for it and its S2_base pair."""
    outputs, pos = trace["outputs"], {i: k for k, i in enumerate(trace["ids"])}
    rows = []
    for s2, base, j in D2.pairs_of(held):
        row = {"id": s2["id"], "taught_position": D2.POSITION_NAMES[j], "pair": " > ".join(s2["program"]),
               "taught_relation": s2["program"][j]}
        for key, x in (("s2", s2), ("base", base)):
            k, ref = pos[x["id"]], list(x["program"]) + [J.END]
            steps = teacher_steps(trace["teacher"][k], outputs, ref)
            greedy = greedy_chain(trace["greedy"][k], outputs)
            row[key] = {"teacher": steps, "greedy": greedy,
                        "kind": failure_kind(ref, greedy, [s["argmax"] for s in steps])}
        rows.append(row)
    return rows


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    q = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
    return {"n": len(values), "mean": statistics.fmean(values), "median": q[1], "q25": q[0], "q75": q[2],
            "share_above_half": sum(v > 0.5 for v in values) / len(values)}


def stopping_summary(rows: list[dict]) -> dict:
    """Teacher-forced probabilities by taught position (and overall), S2 against its S2_base pair."""
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[r["taught_position"]].append(r)
        groups["all"].append(r)
    for g, rs in sorted(groups.items()):
        cell = {}
        for step in (0, 1, 2):
            for field in ("p_end", "p_unsupported", "p_correct", "p_top_alternative"):
                s2 = [r["s2"]["teacher"][step][field] for r in rs]
                base = [r["base"]["teacher"][step][field] for r in rs]
                cell[f"step{step}_{field}"] = {"S2": describe(s2), "S2_base": describe(base),
                                               "paired_difference": describe([a - b for a, b in zip(s2, base)]),
                                               "share_S2_higher": sum(a > b for a, b in zip(s2, base)) / len(rs)}
        cell["step1_teacher_argmax"] = {"S2": dict(Counter(r["s2"]["teacher"][1]["argmax"] for r in rs)),
                                        "S2_base": dict(Counter(r["base"]["teacher"][1]["argmax"] for r in rs))}
        cell["step0_top_alternative_when_wrong"] = dict(Counter(
            r["s2"]["teacher"][0]["argmax"] for r in rs if r["s2"]["teacher"][0]["argmax"] != r["pair"].split(" > ")[0]))
        out[g] = cell
    return out


def failure_summary(rows: list[dict]) -> dict:
    out = {}
    for key in ("s2", "base"):
        by = defaultdict(Counter)
        for r in rows:
            by[r["taught_position"]][r[key]["kind"] or "right"] += 1
            by["all"][r[key]["kind"] or "right"] += 1
        out[key] = {g: dict(sorted(c.items())) for g, c in sorted(by.items())}
    return out


# --- loading and re-derivation ---------------------------------------------------------------------------------
def load_runs(run_dirs: list[Path]) -> dict:
    """Every fit of the given runs, re-derived from its records and checked against its stored summary."""
    fits, worlds, configs = {}, {}, []
    studies = {}
    for run_dir in map(Path, run_dirs):
        result = json.loads((run_dir / "s2_development.json").read_text())
        configs.append({"run": run_dir.name, "config": result["config"], "provenance": result["provenance"],
                        "worlds": result["worlds"]})
        cfg = result["config"]
        for u in result["units"]:
            w, i, arm = u["world"], u["init"], u["arm"]
            if (arm, w, i) in fits:
                raise V.VerificationError(f"duplicate fit {(arm, w, i)}")
            if w not in studies:
                world = W.generate_world(w, n_countries=cfg["countries"])
                study = S.make_study(world, w)
                held = [x for x in study["items"] if x["split"] != "train"]
                vocab_words = {t for x in study["items"] if x["split"] == "train" for t in x["text"].split()}
                studies[w] = (world, study, held, vocab_words)
            world, study, held, vocab_words = studies[w]
            tag = f"{arm}_w{w}_i{i}"
            with gzip.open(run_dir / "records" / f"{tag}.json.gz", "rt") as f:
                rec = json.load(f)
            with gzip.open(run_dir / "traces" / f"{tag}.json.gz", "rt") as f:
                trace = json.load(f)
            if rec["ids"] != [x["id"] for x in held]:
                raise V.VerificationError(f"{tag}: record ids are not the declared items")
            d = V.rederive_arm(held, rec["records"], vocab_words, S.GRID, S.DIAGNOSTIC_CUTOFF, ("none", "certified"))
            for policy in ("none", "certified"):
                if not V._same(u["policies"][policy]["categories"], d["policies"][policy]["categories"]):
                    raise V.VerificationError(f"{tag} {policy}: stored metrics differ from the records")
            cert = d["policies"]["certified"]["certificate"]
            if not V._close(u["policies"]["certified"]["certificate"]["threshold"], cert["threshold"]):
                raise V.VerificationError(f"{tag}: certificate differs")
            outcomes = S.unit_outcomes({"policies": {"certified": {"certificate": cert,
                                                                   "categories": d["policies"]["certified"]["categories"]},
                                                     "none": {"categories": d["policies"]["none"]["categories"]}}})
            if outcomes != u["outcomes"]:
                raise V.VerificationError(f"{tag}: stored outcomes differ from the re-derived ones")
            if u["greedy_trace_mismatches"]:
                raise V.VerificationError(f"{tag}: greedy traces do not reproduce the decisions")
            fits[(arm, w, i)] = {"derived": d, "outcomes": outcomes, "threshold": cert["threshold"],
                                 "records": rec["records"], "ids": rec["ids"], "trace": trace, "held": held,
                                 "train_seconds": u["train_seconds"], "train_loss": u["train_loss"],
                                 "training_examples": u["training_examples"]}
    return {"fits": fits, "runs": configs}


# --- arms and the decision rule -------------------------------------------------------------------------------
def unit_table(fit) -> dict:
    n, c = fit["derived"]["policies"]["none"]["categories"], fit["derived"]["policies"]["certified"]["categories"]
    acc = lambda k: (n[k]["answered"] - n[k]["errors"]) / n[k]["n"]
    return {"threshold": fit["threshold"], "S2_exact": n["S2"]["exact_parse"], "S2_answer_accuracy": acc("S2"),
            "S2_certified_coverage": c["S2"]["coverage"], "S2_certified_errors": c["S2"]["errors"],
            "S2_error_upper_95": c["S2"]["error_upper_95"], "S2_base_exact": n["S2_base"]["exact_parse"],
            "S2_base_answer_accuracy": acc("S2_base"), "T1_exact": n["T1"]["exact_parse"], "T1_accuracy": acc("T1"),
            "ID_certified_coverage": c["ID"]["coverage"], "ID_certified_errors": c["ID"]["errors"],
            "R_abstention": c["R"]["abstention"], "S3_certified_coverage": c["S3"]["coverage"],
            "S3_certified_errors": c["S3"]["errors"], "S3_error_upper_95": c["S3"]["error_upper_95"],
            "S4a_primary_abstention": c["S4a_primary"]["abstention"],
            "S4b_primary_abstention": c["S4b_primary"]["abstention"],
            **{f"outcome_{k}": fit["outcomes"][k] for k in UNIT_FIELDS},
            "train_seconds": fit["train_seconds"], "train_loss": fit["train_loss"],
            "training_examples": fit["training_examples"]}


def decision_rule(units: dict, worlds, inits) -> dict:
    """The brief's rule: proceed to a fresh confirmation only if C's S2 exact accuracy (program and entity), averaged
    over the inits of a world, exceeds B's in every development world, and C meets the S3/S4 joint criterion in every
    development unit."""
    mean = lambda arm, w: statistics.fmean(units[f"{arm}/{w}/{i}"]["S2_exact"] for i in inits)
    by_world = {str(w): {arm: mean(arm, w) for arm in "ABCD" if all(f"{arm}/{w}/{i}" in units for i in inits)}
                for w in worlds}
    c_over_b = {str(w): by_world[str(w)]["C"] > by_world[str(w)]["B"] for w in worlds}
    c_joint = {f"{w}/{i}": units[f"C/{w}/{i}"]["outcome_joint_novelty"] for w in worlds for i in inits}
    return {"S2_exact_world_means": by_world, "C_above_B_by_world": c_over_b, "C_joint_by_unit": c_joint,
            "proceed_to_confirmation": all(c_over_b.values()) and all(c_joint.values())}


def analyse(run_dirs: list[Path]) -> dict:
    loaded = load_runs(run_dirs)
    fits = loaded["fits"]
    arms = sorted({a for a, _, _ in fits})
    worlds = sorted({w for _, w, _ in fits})
    inits = sorted({i for _, _, i in fits})
    missing = [(a, w, i) for a in arms for w in worlds for i in inits if (a, w, i) not in fits]
    if missing:
        raise V.VerificationError(f"incomplete inventory: {missing}")
    units, rows_by_arm, traces_by_arm = {}, defaultdict(list), defaultdict(list)
    for (arm, w, i), fit in sorted(fits.items()):
        units[f"{arm}/{w}/{i}"] = unit_table(fit)
        rows_by_arm[arm] += D2.question_rows(fit["held"], fit["records"], fit["ids"], fit["threshold"], w, i, arm)
        traces_by_arm[arm] += [{**r, "world": w, "init": i} for r in trace_rows(fit["held"], fit["trace"])]
    out = {"runs": loaded["runs"], "arms": arms, "worlds": worlds, "inits": inits, "units": units,
           "classes": D2.summarize(rows_by_arm),
           "stopping": {a: stopping_summary(rs) for a, rs in traces_by_arm.items()},
           "failure_kinds": {a: failure_summary(rs) for a, rs in traces_by_arm.items()},
           "failure_kind_by_class": {a: cross_tab(rows_by_arm[a], traces_by_arm[a]) for a in arms}}
    if set("BC") <= set(arms):
        out["decision_rule"] = decision_rule(units, worlds, inits)
    return out


def cross_tab(class_rows, trace_rows_) -> dict:
    """Greedy failure kind against the record-based class, per S2 question (they must describe the same output)."""
    kinds = {(r["world"], r["init"], r["id"]): r["s2"]["kind"] for r in trace_rows_}
    t = defaultdict(Counter)
    for r in class_rows:
        t[r["s2"]["label"]][kinds[(r["world"], r["init"], r["id"])] or "right"] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(t.items())}


# --- tables ----------------------------------------------------------------------------------------------------
def tables(a: dict) -> str:
    lines = ["## Units (S2 exact = program and entity, all 800 S2 questions; certified = the study's policy)\n",
             "| arm/world/init | thr | S2 exact | S2 cert cov/err | S2_base exact | T1 acc | ID cov | R abst | S3 cov/err "
             "| S4a/S4b abst | joint | S2 transfer |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for k, u in a["units"].items():
        lines.append(f"| {k} | {u['threshold']} | {u['S2_exact']:.3f} | {u['S2_certified_coverage']:.3f}/"
                     f"{u['S2_certified_errors']} | {u['S2_base_exact']:.3f} | {u['T1_accuracy']:.3f} | "
                     f"{u['ID_certified_coverage']:.3f} | {u['R_abstention']:.3f} | {u['S3_certified_coverage']:.3f}/"
                     f"{u['S3_certified_errors']} | {u['S4a_primary_abstention']:.4f}/{u['S4b_primary_abstention']:.4f} "
                     f"| {u['outcome_joint_novelty']} | {u['outcome_s2_transfer']} |")
    if "decision_rule" in a:
        dr = a["decision_rule"]
        lines.append("\n## Decision rule\n")
        for w, m in dr["S2_exact_world_means"].items():
            lines.append(f"- world {w}: S2 exact means " + ", ".join(f"{k} {v:.4f}" for k, v in m.items())
                         + f"; C above B: {dr['C_above_B_by_world'][w]}")
        lines.append(f"- C joint criterion by unit: {dr['C_joint_by_unit']}")
        lines.append(f"- **proceed to confirmation: {dr['proceed_to_confirmation']}**")
    lines.append("\n## Teacher-forced P(<end>) at step 1 (after the first required relation; a second is required)\n")
    lines.append("| arm | taught position | n | S2 mean | S2 median | S2 share > 0.5 | S2_base mean | S2_base share > 0.5 "
                 "| mean paired difference | share S2 higher |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm, st in a["stopping"].items():
        for g, cell in st.items():
            e = cell["step1_p_end"]
            lines.append(f"| {arm} | {g} | {e['S2']['n']} | {e['S2']['mean']:.4f} | {e['S2']['median']:.2e} | "
                         f"{e['S2']['share_above_half']:.4f} | {e['S2_base']['mean']:.2e} | "
                         f"{e['S2_base']['share_above_half']:.4f} | {e['paired_difference']['mean']:.4f} | "
                         f"{e['share_S2_higher']:.3f} |")
    lines.append("\n## Teacher-forced P(correct) by step, S2 (S2_base), mean\n")
    lines.append("| arm | taught position | step 0 | step 1 | step 2 (stop) |")
    lines.append("|---|---|---:|---:|---:|")
    for arm, st in a["stopping"].items():
        for g, cell in st.items():
            lines.append(f"| {arm} | {g} | " + " | ".join(
                f"{cell[f'step{s}_p_correct']['S2']['mean']:.4f} ({cell[f'step{s}_p_correct']['S2_base']['mean']:.4f})"
                for s in (0, 1, 2)) + " |")
    lines.append("\n## Greedy failure kinds (S2 programs; S2_base in the JSON)\n")
    kinds = ("right",) + FAILURE_KINDS
    lines.append("| arm | taught position | " + " | ".join(kinds) + " |")
    lines.append("|---|---|" + "---:|" * len(kinds))
    for arm, fk in a["failure_kinds"].items():
        for g, c in fk["s2"].items():
            lines.append(f"| {arm} | {g} | " + " | ".join(str(c.get(k, 0)) for k in kinds) + " |")
    lines.append("\n" + D2.tables(a["classes"]))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    a = analyse(args.run_dirs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "s2_development_analysis.json").write_text(json.dumps(a, indent=1, sort_keys=True))
    text = tables(a)
    (args.out / "s2_development_tables.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
