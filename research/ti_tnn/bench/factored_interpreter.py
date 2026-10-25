"""The factored interpreter study (protocols/factored_interpreter_v1.md): the no-fit controls, L and G.

    .venv-phase2/bin/python research/ti_tnn/bench/factored_interpreter.py --worlds 100 101 --out runs/titnn_factored_dev

**Example.** World 100's training holds 73 one-step questions of the form "what is the airfield of <a city> ?",
each with target (`airport_of`, `<end>`).
- **Segmenting.** The supplied grammar (`integrated/segment_grammar.py`) segments each into one segment,
  ("the", "airfield", "of"), in prefix style.
- **Aligning.** The target aligns that segment to `airport_of`.
- **The dictionary.** **L** records (prefix, "the airfield of") → {`airport_of`}.
- **Use in S2.** At test, S2's "name the company operating the airfield of country_29_city_0 ." segments into
  "the airfield of" and "the company operating". L looks both up, composes `airport_of` then `operator_of`,
  type-checks the program from the entity, and executes it.
- **Refusal.** A segment never seen in training, such as "the aerodrome near", is refused. So is one seen
  with conflicting labels, or one whose label is `<unsupported>` or `<ambiguous>`.

**L and G differ only in where the lexicon comes from.** **L** induces its dictionary from the same aligned
training pairs available to F. **G** receives the reference lexicon. Both use the same grammar, deterministic
entity binding and runtime type checks, and neither is fitted.

**Training-derived labels** come from the same source training pairs (question, target), with the supplied
structural alignment and a declared filtering:
- **A program target** (relations, then `<end>`) aligns the program's relations to the segments, in
  execution order. That includes ill-typed refusals, whose relation phrases are meaningful; only the whole
  question must be refused, by the runtime.
- **A refusal target** (`<unsupported>`, `<ambiguous>`) labels a segment only when the question has exactly one
  segment.
- **Excluded,** each with its reason and exported:
  - questions the grammar cannot segment (unknown-entity refusals: no world entity to bind);
  - refusal targets over two or more segments: the target does not say which segment carries the refusal.

**Two representations of a segment key,** both with the style:
- **raw:** the tokens themselves;
- **vocab:** the training vocabulary of the word interpreters. That is every word of the regime's
  training-split questions, augmentation excluded; any other word becomes `<unk>`.

**Two regimes:**
- **full:** the robustness study's training.
- **reduced exposure:** each taught phrase keeps exactly 3 of its one-step training questions, chosen by a
  seeded draw; everything else is unchanged.

F discards the entity and the frame. So the 3 kept questions present the same distinct segment, the taught
phrase, that the full regime presents about 73 times.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import interpreter_robustness as IR  # noqa: E402  (the study generator, augmentation and phrase tables)
import lexical_condition as LC  # noqa: E402  (the lexical condition's inventory and sampler)
from integrated import calibration as K  # noqa: E402  (the binomial tail and bounds)
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import routing as J  # noqa: E402
from integrated import segment_grammar as SG  # noqa: E402
from integrated.contracts import Budget, Program, content_identity  # noqa: E402
from integrated.runtime import execute  # noqa: E402

GENERATOR_VERSION = "ti_tnn_factored_interpreter_v1"
REGIMES = ("full", "reduced")
REDUCED_PER_PHRASE = 3
REPRESENTATIONS = ("raw", "vocab")
UNK = "<unk>"
REFUSAL_LABELS = (J.UNSUPPORTED, J.AMBIGUOUS)
CONFLICT = "<conflict>"
COUNTRIES = 96
CONFIRMATION_WORLDS = (1000, 1001, 1002)
RISK_ALPHA, RISK_DELTA = 0.01, 0.05  # the certificate's accepted-error level and confidence, shared by every system
DEVELOPMENT_WORLDS = (100, 101)
SOURCES = ("factored_interpreter.py", "lexical_condition.py", "interpreter_robustness.py", "typed_questions.py",
           "../integrated/segment_grammar.py", "../integrated/routing.py", "../integrated/runtime.py",
           "../integrated/registry.py", "../integrated/relations.py", "../integrated/contracts.py",
           "../integrated/reference.py")


# --- the study, the regimes, the training pairs --------------------------------------------------------------
def study_for(world, seed: int) -> dict:
    return IR.make_study(world, seed)


def regime_training(study: dict, regime: str, world, seed: int) -> tuple[list[dict], list[dict], list[str]]:
    """(training-split items, augmentation items, ids of dropped taught questions) for a regime. The reduced regime
    keeps REDUCED_PER_PHRASE one-step taught questions per taught phrase (a seeded draw, in id order); the
    augmentation is built from the Base training questions, which no regime changes."""
    train = [x for x in study["items"] if x["split"] == "train"]
    dropped = []
    if regime == "reduced":
        rng = random.Random(f"{GENERATOR_VERSION}|reduced|{world.identity}|{seed}")
        by_relation = defaultdict(list)
        for x in train:
            if x["category"] == "train_taught":
                by_relation[x["program"][0]].append(x["id"])
        keep = set()
        for r in sorted(by_relation):
            ids = by_relation[r]
            keep |= set(rng.sample(ids, min(REDUCED_PER_PHRASE, len(ids))))
        dropped = sorted(x["id"] for x in train if x["category"] == "train_taught" and x["id"] not in keep)
        train = [x for x in train if x["id"] not in set(dropped)]
    elif regime != "full":
        raise ValueError(regime)
    return train, IR.augment(study, seed), dropped


def training_vocabulary(train: list[dict]) -> set[str]:
    """The word interpreters' vocabulary rule (`routing.TokenVocab` over the training-split texts)."""
    return set(J.TokenVocab([x["text"] for x in train]).words) - {J.PAD, J.UNK}


def key_of(tokens, style: str, representation: str, vocab: set[str] | None) -> tuple:
    words = tuple(tokens) if representation == "raw" else tuple(w if w in vocab else UNK for w in tokens)
    return (style, words)


def entity_names(world) -> set[str]:
    return {v for t in world.types.values() for v in t.values}


def extract_labels(items: list[dict], world) -> dict:
    """Training-derived segment labels from (question, target) pairs: the contributions and the exclusions."""
    entities = entity_names(world)
    contributions, excluded = [], []
    for x in items:
        seg = SG.segment(x["text"], entities)
        outputs = x["outputs"]
        source = x["category"] if x["category"] != "R" else f"train_refusal:{x['kind']}"
        if not seg.ok:
            excluded.append({"id": x["id"], "source": source, "reason": f"grammar: {seg.status}"})
            continue
        if outputs[-1] == J.END:
            program = outputs[:-1]
            if len(program) != len(seg.segments):
                excluded.append({"id": x["id"], "source": source, "reason": "segments do not match the target length"})
                continue
            labels = list(program)
        elif len(outputs) == 1 and outputs[0] in REFUSAL_LABELS:
            if len(seg.segments) != 1:
                excluded.append({"id": x["id"], "source": source,
                                 "reason": "refusal target over several segments: which one carries it is not given"})
                continue
            labels = [outputs[0]]
        else:
            excluded.append({"id": x["id"], "source": source, "reason": "unrecognized target"})
            continue
        for k, (tokens, label) in enumerate(zip(seg.segments, labels)):
            contributions.append({"id": x["id"], "source": source, "position": k, "style": seg.style,
                                  "tokens": list(tokens), "label": label})
    return {"contributions": contributions, "excluded": excluded}


def induce_dictionary(contributions: list[dict], representation: str, vocab: set[str] | None) -> dict:
    """L's dictionary: (style, key) -> the one label observed with it, or CONFLICT when more than one was."""
    seen = defaultdict(set)
    for c in contributions:
        seen[key_of(c["tokens"], c["style"], representation, vocab)].add(c["label"])
    return {k: (next(iter(v)) if len(v) == 1 else CONFLICT) for k, v in seen.items()}


def reference_lexicon() -> dict:
    """G's lexicon: the Base and taught phrasings of the robustness study, as (style, segment tokens) -> relation."""
    lex = {}
    for r, templates in IR.BASE_PREFIX.items():
        for t in list(templates) + [IR.TAUGHT[r]]:
            lex[(SG.PREFIX, tuple(t.replace("{x}", "").split()))] = r
    for r, t in IR.BASE_POSSESSIVE.items():
        lex[(SG.POSSESSIVE, tuple(t.replace("{x}", "").split()))] = r
    return lex


def word_label_sets(contributions: list[dict]) -> dict:
    """W's associations: (style, word) -> the set of labels of the aligned training segments containing that word."""
    sets = defaultdict(set)
    for c in contributions:
        for w in c["tokens"]:
            sets[(c["style"], w)].add(c["label"])
    return dict(sets)


def w_label(tokens, style: str, sets: dict) -> tuple[str | None, str]:
    """W's grounding of one segment: the unique label common to every word's set, or a refusal reason."""
    common = None
    for w in tokens:
        labels = sets.get((style, w))
        if labels is None:
            return None, "unseen word"
        common = set(labels) if common is None else common & labels
    if not common:
        return None, "no common label"
    if len(common) > 1:
        return None, "several common labels"
    return next(iter(common)), ""


# --- the no-fit controls ------------------------------------------------------------------------------------
def decide(text: str, world, registry, lookup, representation: str = "raw", vocab: set[str] | None = None,
           entities: set[str] | None = None) -> dict:
    """One no-fit decision. Segments are looked up in execution order; the first one that is unseen, conflicting
    or refusal-labelled decides the abstention. Otherwise the program is bound and executed by the runtime."""
    seg = SG.segment(text, entities if entities is not None else entity_names(world))
    out = {"decision": "abstain", "reason": "", "program": None, "entity": seg.entity, "answer": None,
           "style": seg.style, "segments": [list(s) for s in seg.segments]}
    if not seg.ok:
        return {**out, "reason": seg.status}
    program = []
    for tokens in seg.segments:
        if representation == "words":  # W: lookup holds the (style, word) label sets
            label, why = w_label(tokens, seg.style, lookup)
            if label is None:
                return {**out, "reason": why}
        else:
            label = lookup.get(key_of(tokens, seg.style, representation, vocab))
            if label is None:
                return {**out, "reason": "unseen segment"}
            if label == CONFLICT:
                return {**out, "reason": "conflicting labels"}
        if label in REFUSAL_LABELS:
            return {**out, "reason": label}
        program.append(label)
    bound = [t for t in sorted(world.types) if world.types[t].contains(seg.entity)]
    ex = execute(Program(tuple(program), world.types[bound[0]].value(seg.entity)), registry, world.types,
                 Budget(len(program)))
    if ex.status != "answered":
        return {**out, "program": program, "reason": ex.decision.reason}
    return {**out, "decision": "answer", "program": program, "answer": ex.answer.value}


def judge(item: dict, row: dict) -> dict:
    """Primary correctness: an answer is correct only when answering is expected and the ordered program, the
    bound entity and the executed answer are all the reference's. An answer where abstention is expected is an
    error, whatever its program or answer. Accidental agreement: a right answer from a wrong program or entity."""
    answered = row["decision"] == "answer"
    exact = (answered and item["expected"] == "answer" and row["program"] == item["program"]
             and row["entity"] == item["entity"] and row["answer"] == item["answer"])
    return {"answered": answered, "correct": exact if answered else item["expected"] == "abstain",
            "error": answered and not exact,
            "accidental": answered and not exact and item["expected"] == "answer" and row["answer"] == item["answer"]}


def summarize(items: list[dict], rows: list[dict]) -> dict:
    cats = defaultdict(Counter)
    reasons = defaultdict(Counter)
    for it, row in zip(items, rows):
        j, c = judge(it, row), cats[it["category"]]
        c["n"] += 1
        c["expected_answer"] += it["expected"] == "answer"
        c["answered"] += j["answered"]
        c["exact_correct_answers"] += j["answered"] and not j["error"]
        c["errors"] += j["error"]
        c["accidental_agreement"] += j["accidental"]
        c["abstained"] += not j["answered"]
        c["wrong_decision"] += j["answered"] != (it["expected"] == "answer")
        if not j["answered"]:
            reasons[it["category"]][row["reason"]] += 1
    return {k: {**dict(v), "abstention_reasons": dict(sorted(reasons[k].items()))} for k, v in sorted(cats.items())}


def calibration_risk(items: list[dict], rows: list[dict]) -> dict:
    """The single calibration risk test of a deterministic policy (L, G, W): on the calibration split, the null
    "accepted error >= RISK_ALPHA" is rejected when P(Binomial(n, RISK_ALPHA) <= e) <= RISK_DELTA, where n is the
    number of calibration items it answers and e the answers that are not exact-program correct. The same items
    and the same correctness endpoint as the neural policies' fixed-sequence certificates."""
    cal = [(x, r) for x, r in zip(items, rows) if x["split"] == "calibration"]
    acc = [(x, r) for x, r in cal if r["decision"] == "answer"]
    errors = sum(judge(x, r)["error"] for x, r in acc)
    p = K.binomial_p_value(errors, len(acc), RISK_ALPHA) if acc else 1.0
    return {"calibration_items": len(cal), "accepted": len(acc), "errors": errors, "p_value": p,
            "certified": bool(acc) and p <= RISK_DELTA, "error_upper_95": K.clopper_pearson_upper(errors, len(acc))}


def alignment_audit(items: list[dict], world) -> dict:
    """The grammar against the generator's construction: for every item with a program, each segment, in execution
    order, must be a declared phrasing of the relation at that position (Base, taught, held-out or possessive), or,
    for an essential augmentation, the replaced position. Counts, not just lengths."""
    phrasings = defaultdict(set)
    for r in IR.BASE_PREFIX:
        for t in list(IR.BASE_PREFIX[r]) + [IR.TAUGHT[r], IR.HELD[r], IR.BASE_POSSESSIVE[r]]:
            phrasings[r].add(tuple(t.replace("{x}", "").split()))
    entities = entity_names(world)
    out = Counter()
    for x in items:
        if not x.get("program") or x.get("kind") == "unknown_entity":
            continue
        seg = SG.segment(x["text"], entities)
        if not seg.ok or len(seg.segments) != len(x["program"]):
            out["unsegmented_or_length_mismatch"] += 1
            continue
        bad = [k for k, (tokens, r) in enumerate(zip(seg.segments, x["program"])) if tuple(tokens) not in phrasings[r]]
        if x["category"] == "aug_essential" and len(bad) == 1:
            out["aug_essential_replaced_position"] += 1
        elif bad:
            out["misaligned"] += 1
        else:
            out["aligned"] += 1
    return dict(out)


# --- exposure and support -----------------------------------------------------------------------------------
def support_report(labels: dict, dictionaries: dict, evals: list[dict], world, vocab: set[str]) -> dict:
    """What the dictionaries were induced from, and how much of each evaluation category's segments they cover."""
    contrib = labels["contributions"]
    tuples = Counter((c["style"], " ".join(c["tokens"]), c["label"]) for c in contrib)
    by_relation = defaultdict(dict)
    for (style, text, label), n in sorted(tuples.items()):
        by_relation[label][f"{style}: {text}"] = n
    entities = entity_names(world)
    coverage = {}
    for rep, d in dictionaries.items():
        cov = defaultdict(Counter)
        for x in evals:
            seg = SG.segment(x["text"], entities)
            for tokens in seg.segments if seg.ok else ():
                k = key_of(tokens, seg.style, rep, vocab)
                cov[x["category"]]["segments"] += 1
                cov[x["category"]]["in_dictionary"] += k in d
                cov[x["category"]]["refusal_labelled"] += d.get(k) in REFUSAL_LABELS
        coverage[rep] = {c: dict(v) for c, v in sorted(cov.items())}
    return {"presentations": len(contrib), "by_source": dict(sorted(Counter(c["source"] for c in contrib).items())),
            "unique_tuples": len(tuples),
            "unique_keys": {rep: len(d) for rep, d in dictionaries.items()},
            "conflicts": {rep: sorted(" ".join(k[1]) for k, v in d.items() if v == CONFLICT) for rep, d in dictionaries.items()},
            "excluded": dict(sorted(Counter(f"{e['source']} | {e['reason']}" for e in labels["excluded"]).items())),
            "excluded_total": len(labels["excluded"]),
            "contributing_questions": len({c["id"] for c in contrib}),
            "label_multiplicities": by_relation, "evaluation_coverage": coverage}


# --- one world and regime -----------------------------------------------------------------------------------
def run_controls(world, seed: int, regime: str, out_dir: Path | None = None) -> dict:
    study = study_for(world, seed)
    train, aug, dropped = regime_training(study, regime, world, seed)
    evals = [x for x in study["items"] if x["split"] != "train"]
    vocab = training_vocabulary(train)
    labels = extract_labels(train + aug, world)
    dictionaries = {rep: induce_dictionary(labels["contributions"], rep, vocab) for rep in REPRESENTATIONS}
    registry, entities = G.table_registry(world), entity_names(world)
    systems = {f"L_{rep}": (dictionaries[rep], rep) for rep in REPRESENTATIONS}
    systems["G"] = (reference_lexicon(), "raw")
    systems["W"] = (word_label_sets(labels["contributions"]), "words")
    results, records, risks = {}, {}, {}
    for name, (lookup, rep) in systems.items():
        rows = [decide(x["text"], world, registry, lookup, rep, vocab, entities) for x in evals]
        results[name] = summarize(evals, rows)
        risks[name] = calibration_risk(evals, rows)
        records[name] = [{"id": x["id"], **{k: r[k] for k in ("decision", "reason", "program", "entity", "answer")}}
                         for x, r in zip(evals, rows)]
    lexical = LC.lexical_items(world, study, seed)
    lexical_results, lexical_records = {}, {}
    for name, (lookup, rep) in systems.items():
        rows = [decide(x["text"], world, registry, lookup, rep, vocab, entities) for x in lexical]
        lexical_results[name] = summarize(lexical, rows)
        lexical_records[name] = [{"id": x["id"], **{k: r[k] for k in ("decision", "reason", "program", "entity", "answer")}}
                                 for x, r in zip(lexical, rows)]
    wsets = systems["W"][0]
    identifiability = {}
    for r, templates in sorted(LC.inventory().items()):
        for t in templates:
            words = LC.phrase_words(t)
            label, why = w_label(words, SG.PREFIX, wsets)
            identifiability[" ".join(words)] = {"relation": r, "w_label": label, "reason": why,
                                                "seen_as_segment": ("prefix", words) in dictionaries["raw"],
                                                "words_seen": all(("prefix", w) in wsets for w in words)}
    unit = {"world": world.seed, "regime": regime, "study_identity": content_identity([x["id"] + x["text"] for x in study["items"]]),
            "n_evaluated": len(evals), "dropped_taught": len(dropped), "vocabulary_size": len(vocab),
            "alignment_audit": alignment_audit([x for x in study["items"]] + aug, world),
            "support": support_report(labels, dictionaries, evals, world, vocab), "systems": results,
            "calibration_risk": risks,
            "lexical": {"identity": content_identity([x["id"] + x["text"] for x in lexical]), "n": len(lexical),
                        "distinct_forms": len({x["lg_form"] for x in lexical}),
                        "questions_per_form": dict(sorted(Counter(x["lg_form"] for x in lexical if x["category"] in
                                                                  ("LG1", "LG2")).items())),
                        "identifiability": identifiability, "systems": lexical_results},
            "totals": {name: {"wrong_decisions": sum(v["wrong_decision"] for v in res.values()),
                              "errors": sum(v["errors"] for v in res.values()),
                              "accidental_agreement": sum(v["accidental_agreement"] for v in res.values())}
                       for name, res in results.items()}}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "unit.json").write_text(json.dumps(unit, indent=1, sort_keys=True))
        with gzip.open(out_dir / "labels.json.gz", "wt") as f:
            json.dump({"dropped_taught": dropped, "vocabulary": sorted(vocab), **labels,
                       "dictionaries": {rep: [[k[0], list(k[1]), v] for k, v in sorted(d.items())]
                                        for rep, d in dictionaries.items()}}, f)
        with gzip.open(out_dir / "records.json.gz", "wt") as f:
            json.dump(records, f)
        with gzip.open(out_dir / "lexical_records.json.gz", "wt") as f:
            json.dump(lexical_records, f)
    return unit


def provenance() -> dict:
    here = Path(__file__).resolve().parent
    sha = {s: hashlib.sha256((here / s).read_bytes()).hexdigest() for s in SOURCES}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", str(here.parent)], cwd=here,
                                             text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "research_tree_dirty": dirty, "sha256": sha, "python": sys.version.split()[0]}


def config(countries: int = COUNTRIES) -> dict:
    """Everything a run's results depend on besides the sources: the verifier requires it unchanged."""
    return {"generator": GENERATOR_VERSION, "countries": countries, "regimes": list(REGIMES),
            "reduced_per_phrase": REDUCED_PER_PHRASE, "representations": list(REPRESENTATIONS),
            "risk_alpha": RISK_ALPHA, "risk_delta": RISK_DELTA}


def run(worlds, out: Path, countries: int = COUNTRIES) -> Path:
    if any(w in CONFIRMATION_WORLDS for w in worlds):
        raise SystemExit("worlds 1000-1002 are reserved for the confirmation; this protocol is not frozen")
    run_dir = out / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    meta = {"generator": GENERATOR_VERSION, "provenance": provenance(), "config": config(countries),
            "requested": {"worlds": list(worlds), "regimes": list(REGIMES)}, "countries": countries, "units": []}
    for w in worlds:
        world = W.generate_world(w, n_countries=countries)
        for regime in REGIMES:
            path = f"units/w{w}_{regime}"
            unit = run_controls(world, w, regime, run_dir / path)
            meta["units"].append({"world": w, "regime": regime, "path": path, "totals": unit["totals"]})
            print(f"w{w} {regime:8s} " + "  ".join(f"{n}: wrong decisions {t['wrong_decisions']}, errors {t['errors']}"
                                                   for n, t in unit["totals"].items()), flush=True)
    meta["totals"] = {name: {k: sum(u["totals"][name][k] for u in meta["units"]) for k in
                             ("wrong_decisions", "errors", "accidental_agreement")} for name in meta["units"][0]["totals"]}
    meta["finished"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1))
    return run_dir


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--worlds", type=int, nargs="+", default=list(DEVELOPMENT_WORLDS))
    p.add_argument("--countries", type=int, default=COUNTRIES)
    p.add_argument("--out", type=Path, default=Path("runs/titnn_factored_dev"))
    a = p.parse_args(argv)
    print(run(a.worlds, a.out, a.countries))


if __name__ == "__main__":
    main()
