"""Independently rebuild and check every reported number of an interpreter robustness run
(protocols/interpreter_robustness_v1.md, revision 4).

    .venv-phase2/bin/python research/ti_tnn/bench/interpreter_robustness_verify.py runs/titnn_robustness/<run>

The run's stored summaries and verdict are trusted for nothing. This script takes four inputs:
- the run's configuration;
- each world's study, regenerated from its seed by the frozen generator (and re-audited);
- the per-unit decision records: decision, full-precision score, proposed program, copied entity, produced
  answer;
- the frozen grid, cutoff and bars.

From those it independently re-derives, without calling the harness's `measure`, `certify`, `accepts` or
`evaluate`:
- the inventory: exactly the declared worlds × inits × arms, and exactly the declared items in each record;
- correctness, against the regenerated reference answers, and the novelty filter;
- acceptance under every policy, including the diagnostic cutoff (score ≥ 0.999 and an answer), from
  full-precision scores;
- the fixed-sequence certificate, from the calibration records;
- every category metric;
- the joint-rule verdict.

Every re-derived value is compared with what the run stored. Any discrepancy, missing unit or extra unit
raises `VerificationError`. On success it prints the per-unit tables from the re-derived values.
"""
from __future__ import annotations

import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import interpreter_robustness as S  # noqa: E402  (only the generator, audit and frozen constants are used)
from integrated import calibration as K  # noqa: E402
from integrated import relations as W  # noqa: E402


class VerificationError(AssertionError):
    pass


def _check(ok, message, problems):
    if not ok:
        problems.append(message)


def _close(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def certificate(scores, correct, grid, alpha=0.01, delta=0.05) -> dict:
    """Fixed-sequence testing, written independently of calibration.fixed_sequence_certify."""
    steps, selected = [], None
    for t in grid:
        acc = [c for s, c in zip(scores, correct) if s >= t]
        errors = len(acc) - sum(acc)
        p = K.binomial_p_value(errors, len(acc), alpha)
        steps.append((t, len(acc), errors, p, p <= delta))
        if p > delta:
            break
        selected = t
    return {"threshold": selected, "steps": steps}


def category_metrics(rows) -> dict:
    """rows: (accepted, correct, abstain_reason, exact, right_entity) per item of one category under one policy."""
    n = len(rows)
    answered = sum(r[0] for r in rows)
    errors = sum(r[0] and not r[1] for r in rows)
    reasons = defaultdict(int)
    for r in rows:
        reasons[r[2]] += 1
    return {"n": n, "answered": answered, "errors": errors, "coverage": answered / n, "abstention": 1 - answered / n,
            "error_upper_95": K.clopper_pearson_upper(errors, answered),
            "abstention_interval_95": K.exact_interval(n - answered, n),
            "exact_parse": sum(r[3] for r in rows) / n,
            "right_answer_wrong_parse": sum(r[0] and r[1] and not r[3] for r in rows),
            "reasons": dict(reasons),
            "wrong_entity": sum(r[0] and not r[1] and not r[4] for r in rows),
            "wrong_program": sum(r[0] and not r[1] and r[4] for r in rows)}


def slices(item) -> list[str]:
    keys = [item["category"]]
    if item["category"] == "T1":
        keys.append(f"T1:{item['program'][0]}")
    if item["category"] == "S1":
        keys.append(f"S1:{item['family']}")
    if "stratum" in item:
        keys.append(f"{item['category']}:{item['stratum']}")
    return keys


def rederive_arm(items, rec, vocab_words, grid, cutoff, policies) -> dict:
    """Every policy's certificate and category metrics for one arm, from its records alone."""
    n_items = len(items)
    answered = rec["decision"]
    correct = [bool(answered[i]) and x["expected"] == "answer" and rec["answer"][i] == x["answer"]
               for i, x in enumerate(items)]
    novel = [any(t not in vocab_words and t != rec["entity"][i] for t in x["text"].split()) for i, x in enumerate(items)]
    exact = [rec["program"][i] is not None and x["program"] is not None and list(rec["program"][i]) == x["program"]
             and rec["entity"][i] == x["entity"] for i, x in enumerate(items)]
    right_entity = [rec["entity"][i] == x["entity"] for i, x in enumerate(items)]
    reason_names = S.REASONS
    cal = [i for i, x in enumerate(items) if x["split"] == "calibration"]
    out = {"diagnostic": {}, "policies": {}}
    for policy in policies:
        cert = None
        if policy.endswith("certified"):
            keep = [i for i in cal if answered[i] and not (policy == "novelty_certified" and novel[i])]
            cert = certificate([rec["score"][i] for i in keep], [correct[i] for i in keep], grid)
        threshold = cert["threshold"] if cert else None

        def accept(i):
            if not answered[i]:
                return False, reason_names[rec["reason"][i]]
            if policy in ("novelty", "novelty_certified") and novel[i]:
                return False, "novelty"
            if policy in ("certified", "novelty_certified") and (threshold is None or rec["score"][i] < threshold):
                return False, "threshold"
            return True, "answered"
        groups = {"categories": defaultdict(list), "calibration_categories": defaultdict(list)}
        for i, x in enumerate(items):
            a, why = accept(i)
            target = "calibration_categories" if x["split"] == "calibration" else "categories"
            for k in slices(x):
                groups[target][k].append((a, correct[i], why, exact[i], right_entity[i]))
        out["policies"][policy] = {"certificate": cert, **{g: {k: category_metrics(v) for k, v in rows.items()}
                                                          for g, rows in groups.items()}}
    diag = defaultdict(list)
    for i, x in enumerate(items):
        if x["split"] == "test":
            a = bool(answered[i]) and rec["score"][i] >= cutoff
            for k in slices(x):
                diag[k].append((a, correct[i], "answered" if a else "cutoff", exact[i], right_entity[i]))
    out["diagnostic"] = {k: category_metrics(v) for k, v in diag.items()}
    return out


def compare_arm(stored, derived, where, problems):
    for policy, d in derived["policies"].items():
        sp = stored["policies"][policy]
        if d["certificate"] is not None:
            sc = sp["certificate"]
            _check(sc is not None and _close(sc["threshold"], d["certificate"]["threshold"]),
                   f"{where} {policy}: certificate threshold differs", problems)
            if sc is not None:
                steps = [(s["threshold"], s["accepted"], s["errors"], s["p_value"], s["rejected"]) for s in sc["steps"]]
                _check(len(steps) == len(d["certificate"]["steps"]) and all(
                    all(_close(a, b) for a, b in zip(x, y)) for x, y in zip(steps, d["certificate"]["steps"])),
                    f"{where} {policy}: certificate steps differ", problems)
        for group in ("categories", "calibration_categories"):
            stored_cats, derived_cats = sp[group], d[group]
            _check(set(stored_cats) == set(derived_cats), f"{where} {policy} {group}: category sets differ", problems)
            for k in set(stored_cats) & set(derived_cats):
                for field, v in derived_cats[k].items():
                    sv = stored_cats[k].get(field)
                    same = (all(_close(a, b) for a, b in zip(sv, v)) and len(sv) == len(v)) if isinstance(v, list) \
                        else _close(sv, v)
                    _check(same, f"{where} {policy} {group} {k} {field}: stored {sv!r}, re-derived {v!r}", problems)


def joint_verdict(derived_units, worlds, inits) -> dict:
    """The joint rule, written independently of interpreter_robustness.evaluate."""
    def outcomes(arm_d):
        cert = arm_d["policies"]["certified"]["certificate"]
        c, n = arm_d["policies"]["certified"]["categories"], arm_d["policies"]["none"]["categories"]
        certified = cert["threshold"] is not None
        ans = lambda k, cov: certified and c[k]["coverage"] >= cov and c[k]["error_upper_95"] <= S.ERROR_BOUND
        return {"certified": certified, "sanity": ans("ID", S.ID_COVERAGE) and c["R"]["abstention"] >= S.REFUSAL,
                "taught_wording_learned": (n["T1"]["answered"] - n["T1"]["errors"]) / n["T1"]["n"] >= S.T1_ACCURACY,
                "joint_novelty": ans("S3", S.COVERAGE) and c["S4a_primary"]["abstention"] >= S.REFUSAL
                                 and c["S4b_primary"]["abstention"] >= S.REFUSAL,
                "s2_transfer": ans("S2", S.COVERAGE), "s1_structure": ans("S1:length3", S.COVERAGE)}
    per_unit = {a: {f"{w}/{i}": outcomes(derived_units[(w, i)][a]) for w in worlds for i in inits} for a in S.ARMS}
    holds = lambda a, key: all(sum(per_unit[a][f"{w}/{i}"][key] for i in inits) >= 4 for w in worlds)
    return {"predictions": {"1_sanity_every_arm": all(holds(a, "sanity") for a in S.ARMS),
                            "2_taught_wording_learned_every_arm": all(holds(a, "taught_wording_learned") for a in S.ARMS),
                            "3_word_aug_joint_novelty": holds(S.LEAD, "joint_novelty")},
            "separate_verdicts": {k: {a: holds(a, k) for a in S.ARMS} for k in ("joint_novelty", "s2_transfer", "s1_structure")},
            "per_unit": per_unit}


def verify(run_dir: Path, require_confirmation: bool = False) -> dict:
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "interpreter_robustness.json").read_text())
    cfg, problems = result["config"], []
    worlds, inits = tuple(cfg["worlds"]), tuple(cfg["inits"])
    if require_confirmation:
        _check(worlds == S.CONFIRMATION["worlds"] and inits == S.CONFIRMATION["inits"], "not the confirmation inventory",
               problems)
        _check(result["sizes"] == S.SIZES and tuple(result["grid"]) == S.GRID, "sizes or grid differ from frozen", problems)
    keys = [(u["world"], u["init"]) for u in result["units"]]
    _check(sorted(keys) == sorted((w, i) for w in worlds for i in inits) and len(set(keys)) == len(keys),
           f"unit inventory differs: {keys}", problems)
    derived_units = {}
    for w in worlds:
        world = W.generate_world(w, n_countries=cfg["countries"])
        study = S.make_study(world, w, result["sizes"])
        aug = S.augment(study, w)
        audit = S.audit(world, study, aug)
        _check(all(audit.values()), f"world {w}: regenerated study fails its audit: {audit}", problems)
        _check(audit == result["worlds"][str(w)]["audit"], f"world {w}: stored audit differs", problems)
        held = [x for x in study["items"] if x["split"] != "train"]
        evals = [x for x in held if x["split"] == "test"]
        vocab_words = {t for x in study["items"] if x["split"] == "train" for t in x["text"].split()}
        for i in inits:
            path = run_dir / "records" / f"w{w}_i{i}.json.gz"
            if not path.exists():
                problems.append(f"missing records for world {w} init {i}")
                continue
            with gzip.open(path, "rt") as f:
                rec = json.load(f)
            _check(rec["ids"] == [x["id"] for x in held], f"w{w} i{i}: record ids are not the declared items", problems)
            _check(set(rec["arms"]) == set(S.ARMS), f"w{w} i{i}: arm inventory differs", problems)
            stored_unit = next((u for u in result["units"] if (u["world"], u["init"]) == (w, i)), None)
            derived_units[(w, i)] = {}
            for arm in S.ARMS:
                if arm not in rec["arms"] or stored_unit is None or arm not in stored_unit["arms"]:
                    continue
                a = rec["arms"][arm]
                _check(all(len(v) == len(held) for v in a.values()), f"w{w} i{i} {arm}: record length differs", problems)
                d = rederive_arm(held, a, vocab_words, tuple(result["grid"]), result["diagnostic_cutoff"], S.POLICIES)
                compare_arm(stored_unit["arms"][arm], d, f"w{w} i{i} {arm}", problems)
                derived_units[(w, i)][arm] = d
            if stored_unit is not None and "grammar" in rec:
                g = rederive_arm(evals, rec["grammar"], vocab_words, tuple(result["grid"]), result["diagnostic_cutoff"],
                                 ("none",))
                _check(all(_close(stored_unit["grammar"]["categories"][k][f], v)
                           for k, m in g["policies"]["none"]["categories"].items() for f, v in m.items()
                           if not isinstance(v, (list, dict))), f"w{w} i{i} grammar: metrics differ", problems)
    verdict = None
    if not problems:
        verdict = joint_verdict(derived_units, worlds, inits)
        stored = run_dir / "verdict.json"
        if stored.exists():
            sv = json.loads(stored.read_text())
            _check(sv.get("predictions") == verdict["predictions"], "stored predictions differ from re-derived", problems)
            _check(sv.get("separate_verdicts") == verdict["separate_verdicts"], "stored separate verdicts differ", problems)
            _check(sv.get("per_unit") == verdict["per_unit"], "stored per-unit outcomes differ", problems)
    if problems:
        raise VerificationError(f"{len(problems)} problems; first: {problems[:5]}")
    return {"verified": True, "verdict": verdict, "derived": derived_units}


SHOW = ("ID", "R", "T1", "S2", "S3", "S4a_primary", "S4b_primary", "S4a:answer_ambiguous", "S4a:type_determined",
        "S4b:answer_ambiguous", "S4b:type_determined", "S1:length3", "S1:round_trip")


def tables(derived) -> str:
    lines = []
    for (w, i), arms in sorted(derived.items()):
        lines.append(f"=== world {w} init {i}")
        for arm, d in arms.items():
            cert = d["policies"]["certified"]["certificate"]
            first = cert["steps"][0] if cert["steps"] else None
            power = "/".join(f"{K.certification_power(e, first[1]):.2f}" for e in (0.001, 0.0025, 0.005)) if first else "-"
            lines.append(f"  {arm:8s} certificate {cert['threshold']} (first rule: {first[1] if first else 0} accepted, "
                         f"{first[2] if first else 0} errors; power at true 0.1/0.25/0.5%: {power})")
            for name, cats in (("certified", d["policies"]["certified"]["categories"]), ("@cutoff", d["diagnostic"])):
                cells = []
                for k in SHOW:
                    m = cats.get(k)
                    if m is None:
                        continue
                    if k.split(":")[0] in S.ABSTAIN_CATEGORIES:
                        lo, hi = m["abstention_interval_95"]
                        cells.append(f"{k} abst {m['abstention']:.3f} [{lo:.3f},{hi:.3f}]")
                    else:
                        cells.append(f"{k} cov {m['coverage']:.3f} err {m['errors']} ub {m['error_upper_95']:.4f}")
                lines.append(f"    {name:9s} " + " | ".join(cells))
    return "\n".join(lines)


if __name__ == "__main__":
    out = verify(Path(sys.argv[1]), require_confirmation="--confirmation" in sys.argv)
    print(tables(out["derived"]))
    print(json.dumps(out["verdict"]["predictions"], indent=2) if out["verdict"] else "no verdict")
    print("verified: every stored number was re-derived and matches")
