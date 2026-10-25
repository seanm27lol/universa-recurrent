"""Independent verifier for the factored-interpreter study's no-fit controls (protocols/factored_interpreter_v1.md).

    .venv-phase2/bin/python research/ti_tnn/bench/factored_interpreter_verify.py <run dir>

**Example.** World 100's reduced regime records that L abstained on "name the aerodrome near the biggest city
in country_93 ." because of an unseen segment. To check that record, the verifier:
- re-reads the question with its own implementation of the supplied grammar;
- rebuilds the reduced training set and the training vocabulary;
- re-extracts the training-derived labels and re-induces L's dictionary;
- re-decides the question.

It requires the same segments, the same abstention reason and the same summaries.

**It trusts no stored summary,** and it does not import the producer's grammar (`integrated/segment_grammar.py`),
label extraction, dictionary or decision code (`bench/factored_interpreter.py`). It uses:
- **the study generator** of the robustness study (`interpreter_robustness.make_study` and `augment`), which
  defines what was asked;
- **the world** (`relations.generate_world`);
- **this file's own implementations,** written from the protocol, of:
  - the grammar;
  - the reduced-exposure draw and the training vocabulary;
  - label extraction and the dictionaries;
  - G's reference lexicon;
  - the decisions, with execution by following the facts;
  - the correctness predicate and the summaries.

**What it re-derives, and compares field by field:**
- the regime's dropped taught questions and the vocabulary;
- every training contribution and exclusion, with its reason;
- both dictionaries, conflicts included;
- every evaluated item's decision, reason, program, entity and answer, for L (raw), L (vocab) and G;
- every category summary, the totals, the support report and the ordered alignment audit.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import interpreter_robustness as IR  # noqa: E402  (the generator only: make_study, augment, phrase tables)
import lexical_condition as LC  # noqa: E402  (the lexical condition's generator only: inventory, sampler)
from integrated import calibration as K  # noqa: E402  (numeric binomial tail and bounds only)
from integrated import relations as W  # noqa: E402
from integrated.contracts import content_identity  # noqa: E402  (the canonical hash only)
import statistical_tolerance as ST  # noqa: E402  (the one declared tolerance: recomputed bounds and p-values)

GENERATOR_VERSION = "ti_tnn_factored_interpreter_v1"
REDUCED_PER_PHRASE = 3
UNK, END, UNSUPPORTED, AMBIGUOUS, CONFLICT = "<unk>", "<end>", "<unsupported>", "<ambiguous>", "<conflict>"
REFUSALS = (UNSUPPORTED, AMBIGUOUS)
SYSTEMS = ("L_raw", "L_vocab", "G", "W")
MAX_SEGMENTS = 3
# The development design the no-fit controls must have been run with, and the producer files whose hashes a run
# must record (the producer's SOURCES, relative to the bench directory; listed here, not imported).
DESIGN = {"generator": GENERATOR_VERSION, "countries": 96, "regimes": ["full", "reduced"], "reduced_per_phrase": 3,
          "representations": ["raw", "vocab"], "risk_alpha": 0.01, "risk_delta": 0.05}
DEVELOPMENT_WORLDS = (100, 101)
SOURCES = ("factored_interpreter.py", "lexical_condition.py", "interpreter_robustness.py", "typed_questions.py",
           "../integrated/segment_grammar.py", "../integrated/routing.py", "../integrated/runtime.py",
           "../integrated/registry.py", "../integrated/relations.py", "../integrated/contracts.py",
           "../integrated/reference.py")
# The no-fit controls run recorded with the development pilot is bound to its commit: its producer sources are
# checked against that commit's hashes (from `git show`), never silently replaced by the current files.
BOUND = {
    "0e1a3c94004f1f1d178af6a7c62bb85bfa1ebfe3": {
        "factored_interpreter.py": "43751a21e3b8c14e568575ab5c5800111a1dc54b54d0a088e611f85684cf9435",
        "lexical_condition.py": "10019afdc886c6ba80666504fcb38a941f6e8f28e6b748d9ef555485618e9cac",
        "interpreter_robustness.py": "f1653d63a00bb4164a1b8055e05338bbe6b12390efce2e0a03c3e701789a8d46",
        "typed_questions.py": "b1ecc40692700d09d790ed03820c559e86a5a0fbf34629495b4c487fe9a015c1",
        "../integrated/segment_grammar.py": "0fa572f008118bc0419c065f8ee7d34d82b59c0b7938a62469dab1be816b08a7",
        "../integrated/routing.py": "0997d1798adec59f8c56f931276ed393675927bbc978837c19927fe33f719db7",
        "../integrated/runtime.py": "7e83b8674d3ce1f120c4b8529546cb5b29e3f1e6967f790ad0f0e1c461e6f35a",
        "../integrated/registry.py": "85350d0284bae12a55c5b748c5ca5d61acde8623d91ceeed29108525702674e1",
        "../integrated/relations.py": "742331d263587da665b743584f9afda402286219fec9b947838ca3900d296b5a",
        "../integrated/contracts.py": "7ccf766d3885157f5e53ee419544c1f0f5d49a70f575b22875404b6b452dfdaa",
        "../integrated/reference.py": "1a7889da2b2d67d3f4741101549acd075ce48f05b03473ce1a8509d382907b41",
    },
}


class VerificationError(AssertionError):
    pass


# --- the grammar, from the protocol text ---------------------------------------------------------------------
def read(text: str, entities: set) -> dict:
    """{status, style, entity, segments (execution order)}; status "ok" or the refusal reason."""
    toks = text.split()
    hits = [i for i, w in enumerate(toks) if w in entities]
    if len(hits) != 1:
        return {"status": "unknown entity" if not hits else "malformed: several entities", "style": None,
                "entity": None, "segments": []}
    e = hits[0]
    has_the, has_s = "the" in toks[:e], "'s" in toks[e + 1:]
    base = {"style": None, "entity": toks[e], "segments": []}
    if has_the == has_s:
        return {**base, "status": "malformed: mixed styles" if has_the else "malformed: no segment"}
    if has_the:
        words, segs, cur = toks[toks.index("the"):e], [], []
        for w in words:
            if w == "the" and cur:
                segs.append(cur)
                cur = []
            cur.append(w)
        segs.append(cur)
        segs, style = segs[::-1], "prefix"
    else:
        rest = toks[e + 1:]
        if rest[0] != "'s":
            return {**base, "status": "malformed: words between the entity and its first 's"}
        segs = []
        for w in rest:
            if w in ("?", "."):
                break
            if w == "'s":
                segs.append([w])
            else:
                segs[-1].append(w)
        style = "possessive"
    if len(segs) > MAX_SEGMENTS:
        return {**base, "style": style, "status": f"malformed: more than {MAX_SEGMENTS} segments"}
    return {"status": "ok", "style": style, "entity": toks[e], "segments": [list(s) for s in segs]}


# --- the regime, the vocabulary, the labels, the dictionaries ------------------------------------------------
def regime_items(study, regime, world, seed):
    train = [x for x in study["items"] if x["split"] == "train"]
    dropped = []
    if regime == "reduced":
        rng = random.Random(f"{GENERATOR_VERSION}|reduced|{world.identity}|{seed}")
        ids = defaultdict(list)
        for x in train:
            if x["category"] == "train_taught":
                ids[x["program"][0]].append(x["id"])
        keep = set()
        for r in sorted(ids):
            keep.update(rng.sample(ids[r], min(REDUCED_PER_PHRASE, len(ids[r]))))
        dropped = sorted(x["id"] for x in train if x["category"] == "train_taught" and x["id"] not in keep)
        train = [x for x in train if x["id"] not in set(dropped)]
    return train, IR.augment(study, seed), dropped


def vocabulary(train) -> set:
    return {w for x in train for w in x["text"].split()}


def key(tokens, style, rep, vocab):
    return (style, tuple(tokens) if rep == "raw" else tuple(w if w in vocab else UNK for w in tokens))


def labels(items, entities):
    contributions, excluded = [], []
    for x in items:
        g, out = read(x["text"], entities), x["outputs"]
        source = f"train_refusal:{x['kind']}" if x["category"] == "R" else x["category"]
        if g["status"] != "ok":
            excluded.append({"id": x["id"], "source": source, "reason": f"grammar: {g['status']}"})
            continue
        if out[-1] == END and len(out) - 1 == len(g["segments"]):
            labs = out[:-1]
        elif out[-1] == END:
            excluded.append({"id": x["id"], "source": source, "reason": "segments do not match the target length"})
            continue
        elif len(out) == 1 and out[0] in REFUSALS and len(g["segments"]) == 1:
            labs = out
        elif len(out) == 1 and out[0] in REFUSALS:
            excluded.append({"id": x["id"], "source": source,
                             "reason": "refusal target over several segments: which one carries it is not given"})
            continue
        else:
            excluded.append({"id": x["id"], "source": source, "reason": "unrecognized target"})
            continue
        contributions += [{"id": x["id"], "source": source, "position": k, "style": g["style"], "tokens": seg,
                           "label": lab} for k, (seg, lab) in enumerate(zip(g["segments"], labs))]
    return contributions, excluded


def dictionary(contributions, rep, vocab):
    seen = defaultdict(set)
    for c in contributions:
        seen[key(c["tokens"], c["style"], rep, vocab)].add(c["label"])
    return {k: (min(v) if len(v) == 1 else CONFLICT) for k, v in seen.items()}


def word_sets(contributions):
    sets = defaultdict(set)
    for c in contributions:
        for w in c["tokens"]:
            sets[(c["style"], w)].add(c["label"])
    return sets


def ground_words(seg, style, sets):
    """W, from the protocol: the unique label common to every word's training label set, or a refusal reason."""
    common = None
    for w in seg:
        if (style, w) not in sets:
            return None, "unseen word"
        common = set(sets[(style, w)]) if common is None else common & sets[(style, w)]
    return (next(iter(common)), "") if len(common) == 1 else (None, "no common label" if not common else "several common labels")


def reference():
    lex = {}
    for r in IR.BASE_PREFIX:
        for t in list(IR.BASE_PREFIX[r]) + [IR.TAUGHT[r]]:
            lex[("prefix", tuple(t.split()[:-1]))] = r
        lex[("possessive", tuple(IR.BASE_POSSESSIVE[r].split()[1:]))] = r
    return lex


# --- decisions and correctness --------------------------------------------------------------------------------
def decide(text, world, entities, lookup, rep, vocab):
    g = read(text, entities)
    out = {"decision": "abstain", "reason": "", "program": None, "entity": g["entity"], "answer": None}
    if g["status"] != "ok":
        return {**out, "reason": g["status"]}
    prog = []
    for seg in g["segments"]:
        if rep == "words":
            lab, why = ground_words(seg, g["style"], lookup)
            if lab is None or lab in REFUSALS:
                return {**out, "reason": why or lab}
            prog.append(lab)
            continue
        lab = lookup.get(key(seg, g["style"], rep, vocab))
        if lab is None or lab == CONFLICT or lab in REFUSALS:
            return {**out, "reason": {None: "unseen segment", CONFLICT: "conflicting labels"}.get(lab, lab)}
        prog.append(lab)
    t = next(t for t in sorted(world.types) if g["entity"] in world.types[t].values)
    value = g["entity"]
    for r in prog:
        if world.relations[r].source != t:
            return {**out, "program": prog, "reason": "ill-typed program"}
        value, t = world.facts[r][value], world.relations[r].target
    return {**out, "decision": "answer", "program": prog, "answer": value}


def summary(evals, rows):
    cats = defaultdict(Counter)
    reasons = defaultdict(Counter)
    for x, r in zip(evals, rows):
        ans = r["decision"] == "answer"
        exact = ans and x["expected"] == "answer" and (r["program"], r["entity"], r["answer"]) == (
            x["program"], x["entity"], x["answer"])
        c = cats[x["category"]]
        for k, v in (("n", 1), ("expected_answer", x["expected"] == "answer"), ("answered", ans),
                     ("exact_correct_answers", exact), ("errors", ans and not exact),
                     ("accidental_agreement", ans and not exact and x["expected"] == "answer" and r["answer"] == x["answer"]),
                     ("abstained", not ans), ("wrong_decision", ans != (x["expected"] == "answer"))):
            c[k] += v
        if not ans:
            reasons[x["category"]][r["reason"]] += 1
    return {k: {**dict(v), "abstention_reasons": dict(sorted(reasons[k].items()))} for k, v in sorted(cats.items())}


def support(contributions, excluded, dicts, evals, entities, vocab) -> dict:
    """The whole support report, re-derived."""
    tuples = Counter((c["style"], " ".join(c["tokens"]), c["label"]) for c in contributions)
    mult = defaultdict(dict)
    for (style, text, lab), n in tuples.items():
        mult[lab][f"{style}: {text}"] = n
    coverage = {}
    for rep, dd in dicts.items():
        cov = defaultdict(Counter)
        for x in evals:
            g = read(x["text"], entities)
            for seg in g["segments"] if g["status"] == "ok" else ():
                k = key(seg, g["style"], rep, vocab)
                c = cov[x["category"]]
                c["segments"] += 1
                c["in_dictionary"] += k in dd
                c["refusal_labelled"] += dd.get(k) in REFUSALS
        coverage[rep] = {c: dict(v) for c, v in cov.items()}
    return {"presentations": len(contributions), "by_source": dict(Counter(c["source"] for c in contributions)),
            "unique_tuples": len(tuples), "unique_keys": {rep: len(dd) for rep, dd in dicts.items()},
            "conflicts": {rep: sorted(" ".join(k[1]) for k, v in dd.items() if v == CONFLICT) for rep, dd in dicts.items()},
            "excluded": dict(Counter(f"{e['source']} | {e['reason']}" for e in excluded)),
            "excluded_total": len(excluded), "contributing_questions": len({c["id"] for c in contributions}),
            "label_multiplicities": dict(mult), "evaluation_coverage": coverage}


def risk(evals, rows) -> dict:
    """The single calibration risk test of a deterministic policy, on the calibration split and the primary
    correctness endpoint."""
    acc = [(x, r) for x, r in zip(evals, rows) if x["split"] == "calibration" and r["decision"] == "answer"]
    errors = sum(not (x["expected"] == "answer" and (r["program"], r["entity"], r["answer"]) ==
                      (x["program"], x["entity"], x["answer"])) for x, r in acc)
    p = K.binomial_p_value(errors, len(acc), DESIGN["risk_alpha"]) if acc else 1.0
    return {"calibration_items": sum(x["split"] == "calibration" for x in evals), "accepted": len(acc),
            "errors": errors, "p_value": p, "certified": bool(acc) and p <= DESIGN["risk_delta"],
            "error_upper_95": K.clopper_pearson_upper(errors, len(acc))}


def norm(x):
    return json.loads(json.dumps(x, sort_keys=True))


def audit(items, entities):
    phr = defaultdict(set)
    for r in IR.BASE_PREFIX:
        for t in list(IR.BASE_PREFIX[r]) + [IR.TAUGHT[r], IR.HELD[r], IR.BASE_POSSESSIVE[r]]:
            phr[r].add(tuple(w for w in t.split() if w != "{x}"))
    out = Counter()
    for x in items:
        if not x.get("program") or x.get("kind") == "unknown_entity":
            continue
        g = read(x["text"], entities)
        if g["status"] != "ok" or len(g["segments"]) != len(x["program"]):
            out["unsegmented_or_length_mismatch"] += 1
            continue
        bad = sum(tuple(s) not in phr[r] for s, r in zip(g["segments"], x["program"]))
        out["aug_essential_replaced_position" if x["category"] == "aug_essential" and bad == 1
            else "misaligned" if bad else "aligned"] += 1
    return dict(out)


# --- one unit -------------------------------------------------------------------------------------------------
def verify_unit(run_dir: Path, info: dict, countries: int, problems: list, agreement: ST.Agreement | None = None) -> dict:
    tag = f"w{info['world']} {info['regime']}"
    d = run_dir / info["path"]
    unit = json.loads((d / "unit.json").read_text())
    with gzip.open(d / "labels.json.gz", "rt") as f:
        lab = json.load(f)
    with gzip.open(d / "records.json.gz", "rt") as f:
        recs = json.load(f)
    world = W.generate_world(info["world"], n_countries=countries)
    entities = {v for t in world.types.values() for v in t.values}
    study = IR.make_study(world, info["world"])
    train, aug, dropped = regime_items(study, info["regime"], world, info["world"])
    evals = [x for x in study["items"] if x["split"] != "train"]
    vocab = vocabulary(train)
    contributions, excluded = labels(train + aug, entities)
    dicts = {rep: dictionary(contributions, rep, vocab) for rep in ("raw", "vocab")}

    def fail(msg):
        problems.append(f"{tag}: {msg}")

    if (unit["world"], unit["regime"]) != (info["world"], info["regime"]):
        fail("unit.json is not the unit the run lists")
    if lab["dropped_taught"] != dropped or unit["dropped_taught"] != len(dropped):
        fail("the reduced regime's dropped questions do not re-derive")
    if lab["vocabulary"] != sorted(vocab):
        fail("the training vocabulary does not re-derive")
    if lab["contributions"] != contributions:
        fail("the training-derived labels do not re-derive")
    if lab["excluded"] != excluded:
        fail("the exclusions do not re-derive")
    for rep, dd in dicts.items():
        if lab["dictionaries"].get(rep) != [[k[0], list(k[1]), v] for k, v in sorted(dd.items())]:
            fail(f"L's {rep} dictionary does not re-derive")
    lookups = {"L_raw": (dicts["raw"], "raw"), "L_vocab": (dicts["vocab"], "vocab"), "G": (reference(), "raw"),
               "W": (word_sets(contributions), "words")}
    if set(recs) != set(SYSTEMS):
        fail("the records are not exactly the three no-fit systems")
    summaries = {}
    for name in SYSTEMS:
        lookup, rep = lookups[name]
        rows = [decide(x["text"], world, entities, lookup, rep, vocab) for x in evals]
        want = [{"id": x["id"], **r} for x, r in zip(evals, rows)]
        got = recs.get(name, [])
        if len(got) != len(want) or [g["id"] for g in got] != [w["id"] for w in want]:
            fail(f"{name}: the records are not exactly the evaluated items")
            continue
        bad = [w["id"] for g, w in zip(got, want) if g != w]
        if bad:
            fail(f"{name}: {len(bad)} records do not re-derive (first {bad[0]})")
        summaries[name] = summary(evals, rows)
        if unit["systems"].get(name) != summaries[name]:
            fail(f"{name}: the category summaries do not re-derive")
    totals = {n: {"wrong_decisions": sum(v["wrong_decision"] for v in s.values()), "errors": sum(v["errors"] for v in s.values()),
                  "accidental_agreement": sum(v["accidental_agreement"] for v in s.values())} for n, s in summaries.items()}
    if unit["totals"] != totals:
        fail("the totals do not re-derive")
    if unit["alignment_audit"] != audit(list(study["items"]) + aug, entities):
        fail("the ordered alignment audit does not re-derive")
    if unit["n_evaluated"] != len(evals):
        fail("the evaluated count does not re-derive")
    if unit.get("study_identity") != content_identity([x["id"] + x["text"] for x in study["items"]]):
        fail("the study identity does not re-derive")
    if unit.get("vocabulary_size") != len(vocab):
        fail("the vocabulary size does not re-derive")
    if norm(unit.get("support")) != norm(support(contributions, excluded, dicts, evals, entities, vocab)):
        fail("the support report (exposure, keys, exclusions or coverage) does not re-derive")
    risks = {}
    for name in SYSTEMS:
        lookup, rep = lookups[name]
        rows = [decide(x["text"], world, entities, lookup, rep, vocab) for x in evals]
        risks[name] = risk(evals, rows)
    # the p-value and the bound may differ from recomputation within the declared statistical tolerance; the counts
    # and the certification decision must be equal
    if not (agreement if agreement is not None else ST.Agreement()).compare(norm(unit.get("calibration_risk")), norm(risks)):
        fail("a deterministic policy's calibration risk test does not re-derive")
    # the lexical condition: its items, every no-fit decision on them, and the identifiability audit
    lexical = LC.lexical_items(world, study, info["world"])
    with gzip.open(d / "lexical_records.json.gz", "rt") as f:
        lex_recs = json.load(f)
    lx = unit.get("lexical", {})
    if lx.get("identity") != content_identity([x["id"] + x["text"] for x in lexical]) or lx.get("n") != len(lexical) \
            or lx.get("distinct_forms") != len({x["lg_form"] for x in lexical}) \
            or lx.get("questions_per_form") != dict(Counter(x["lg_form"] for x in lexical if x["category"] in ("LG1", "LG2"))):
        fail("the lexical condition's items do not regenerate")
    lex_summaries = {}
    for name in SYSTEMS:
        lookup, rep = lookups[name]
        rows = [decide(x["text"], world, entities, lookup, rep, vocab) for x in lexical]
        if lex_recs.get(name) != [{"id": x["id"], **r} for x, r in zip(lexical, rows)]:
            fail(f"{name}: the lexical-condition records do not re-derive")
        lex_summaries[name] = summary(lexical, rows)
    if norm(lx.get("systems")) != norm(lex_summaries):
        fail("the lexical-condition summaries do not re-derive")
    ws = lookups["W"][0]
    ident = {}
    for r, templates in LC.inventory().items():
        for t in templates:
            words = LC.phrase_words(t)
            lab, why = ground_words(words, "prefix", ws)
            ident[" ".join(words)] = {"relation": r, "w_label": lab, "reason": why,
                                      "seen_as_segment": ("prefix", tuple(words)) in dicts["raw"],
                                      "words_seen": all(("prefix", w) in ws for w in words)}
    if norm(lx.get("identifiability")) != norm(ident):
        fail("the lexical condition's identifiability audit does not re-derive")
    return {"unit": tag, "totals": totals, "evaluated": len(evals), "calibration_risk": risks,
            "lexical": {n: {c: (v["answered"], v["exact_correct_answers"], v["errors"]) for c, v in res.items()}
                        for n, res in lex_summaries.items()}}


def source_hashes() -> dict:
    here = Path(__file__).resolve().parent
    return {s: hashlib.sha256((here / s).read_bytes()).hexdigest() for s in SOURCES}


def verify(run_dir: Path, agreement: ST.Agreement | None = None) -> dict:
    """`agreement`, when given, accumulates what was compared and the largest tolerated statistical difference."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    problems = []
    if run.get("config") != DESIGN or run.get("countries") != DESIGN["countries"] or run.get("generator") != GENERATOR_VERSION:
        problems.append("the run's configuration is not the declared development design")
    prov = run.get("provenance", {})
    if prov.get("commit") in BOUND:
        if prov.get("sha256") != BOUND[prov["commit"]] or prov.get("research_tree_dirty") is not False:
            problems.append("the run's recorded producer sources are not those of its bound commit")
    elif prov.get("sha256") != source_hashes():
        problems.append("the run's recorded producer sources are not the sources being verified")
    requested = run.get("requested", {})
    want = sorted((w, r) for w in requested.get("worlds", []) for r in requested.get("regimes", []))
    pairs = [(u["world"], u["regime"]) for u in run["units"]]
    if not want or requested.get("regimes") != DESIGN["regimes"] or len(set(pairs)) != len(pairs) or sorted(pairs) != want:
        problems.append("the unit inventory is not exactly the requested worlds x regimes, once each")
    if any(w not in DEVELOPMENT_WORLDS for w in requested.get("worlds", [])) or any(w not in DEVELOPMENT_WORLDS for w, _ in pairs):
        problems.append("a world outside development was run before the protocol was frozen")
    if problems:
        raise VerificationError(f"{len(problems)} problems; first: {problems[:5]}")
    units = [verify_unit(run_dir, info, run["countries"], problems, agreement) for info in run["units"]]
    if problems:  # a unit that does not re-derive has no trustworthy totals to sum
        raise VerificationError(f"{len(problems)} problems; first: {problems[:5]}")
    for info, u in zip(run["units"], units):
        if info.get("totals") != u["totals"]:
            problems.append(f"w{info['world']} {info['regime']}: run.json's unit totals do not re-derive")
    sums = {n: {k: sum(u["totals"][n][k] for u in units) for k in ("wrong_decisions", "errors", "accidental_agreement")}
            for n in SYSTEMS}
    if run.get("totals") != sums:
        problems.append("the run-level totals do not re-derive")
    if problems:
        raise VerificationError(f"{len(problems)} problems; first: {problems[:5]}")
    return {"verified": True, "commit": run["provenance"]["commit"], "units": units}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir", type=Path)
    print(json.dumps(verify(p.parse_args(argv).run_dir), indent=1))


if __name__ == "__main__":
    main()
