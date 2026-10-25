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
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import interpreter_robustness as IR  # noqa: E402  (the generator only: make_study, augment, phrase tables)
from integrated import relations as W  # noqa: E402

GENERATOR_VERSION = "ti_tnn_factored_interpreter_v1"
REDUCED_PER_PHRASE = 3
UNK, END, UNSUPPORTED, AMBIGUOUS, CONFLICT = "<unk>", "<end>", "<unsupported>", "<ambiguous>", "<conflict>"
REFUSALS = (UNSUPPORTED, AMBIGUOUS)
SYSTEMS = ("L_raw", "L_vocab", "G")
MAX_SEGMENTS = 3


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
def verify_unit(run_dir: Path, info: dict, countries: int, problems: list) -> dict:
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
    lookups = {"L_raw": (dicts["raw"], "raw"), "L_vocab": (dicts["vocab"], "vocab"), "G": (reference(), "raw")}
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
    sup = unit["support"]
    if (sup["presentations"], sup["contributing_questions"], sup["excluded_total"]) != (
            len(contributions), len({c["id"] for c in contributions}), len(excluded)) or \
            sup["unique_keys"] != {rep: len(dd) for rep, dd in dicts.items()}:
        fail("the support report does not re-derive")
    return {"unit": tag, "totals": totals, "evaluated": len(evals)}


def verify(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    problems = []
    pairs = [(u["world"], u["regime"]) for u in run["units"]]
    if not pairs or len(set(pairs)) != len(pairs) or sorted(pairs) != sorted(
            (w, r) for w in sorted({u["world"] for u in run["units"]}) for r in ("full", "reduced")):
        problems.append("the unit inventory is not every declared world in both regimes, once each")
    if any(w in (1000, 1001, 1002) for w, _ in pairs):
        problems.append("a confirmation world was run before the protocol was frozen")
    if problems:
        raise VerificationError(f"{len(problems)} problems; first: {problems[:5]}")
    units = [verify_unit(run_dir, info, run["countries"], problems) for info in run["units"]]
    if problems:
        raise VerificationError(f"{len(problems)} problems; first: {problems[:5]}")
    return {"verified": True, "commit": run["provenance"]["commit"], "units": units}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir", type=Path)
    print(json.dumps(verify(p.parse_args(argv).run_dir), indent=1))


if __name__ == "__main__":
    main()
