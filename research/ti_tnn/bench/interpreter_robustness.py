"""Interpreter robustness study: data, arms, policies, certification and measurements
(protocols/interpreter_robustness_v1.md, DRAFT).

    .venv-phase2/bin/python research/ti_tnn/bench/interpreter_robustness.py --worlds 100 101 --inits 100 101

Everything is generated from a world seed; nothing is read from earlier runs. The generator audits its own
output before any training (`audit`), and refuses to run if any check fails:
- every training target respects the structural limit (at most two relations), including refusals and
  augmentation;
- taught wording appears in training only in one-step questions;
- no held-out or novel word appears in training;
- pseudo-words are disjoint from every lexicon word and entity;
- splits are disjoint by instance, and refusal strings are not repeated across splits;
- every question has one style and encodes under the character alphabet.
"""
from __future__ import annotations

import argparse
import gzip
import json
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import typed_questions as TQ  # noqa: E402  (#48's lexicon: the Base phrasings)
from integrated import calibration as K  # noqa: E402
from integrated import interpreters as I  # noqa: E402
from integrated import reference as R  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import routing as J  # noqa: E402
from integrated.grammar import GrammarInterpreter  # noqa: E402

# --- the controlled language ---------------------------------------------------------------------------------
BASE_PREFIX = {r: list(v[0]) for r, v in TQ.PHRASES.items()}
BASE_POSSESSIVE = {r: v[1] for r, v in TQ.PHRASES.items()}
BASE_FRAMES = ("what is {q} ?", "name {q} .")
TAUGHT = {"capital_of": "the administrative centre of {x}", "largest_city_of": "the most populous city of {x}",
          "country_of": "the nation of {x}", "airport_of": "the airfield of {x}", "operator_of": "the firm running {x}"}
HELD = {"capital_of": "the seat of government of {x}", "largest_city_of": "the metropolis of {x}",
        "country_of": "the state that holds {x}", "airport_of": "the aerodrome near {x}",
        "operator_of": "the outfit managing {x}"}
NOVEL_FRAMES = ("could you tell me {q} ?", "i would like to know {q} .")
NOVEL_FILLERS = ("exactly", "precisely")
UNSUPPORTED_BASE = {t: v[0][0] for t, v in TQ.UNSUPPORTED.items()}
AMBIGUOUS_BASE = TQ.AMBIGUOUS[0][0]
REFUSAL_SHARE = 0.06  # training refusals: a quota per kind (training is fixed, not sampled for a claim)
REFUSALS = ("ill_typed", "unsupported", "unknown_entity", "ambiguous")
AUG_SHARE = 0.25
ARMS = ("word", "word_aug", "char", "char_aug")
POLICIES = ("none", "certified", "novelty", "novelty_certified")
GRID = (0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0)  # provisional (round 2: starts at 0.999)
DIAGNOSTIC_CUTOFF = 0.999  # a fixed raw cutoff for diagnostics only; never a certificate
ANSWER_CATEGORIES = ("CAL", "ID", "T1", "T1_base", "S2", "S2_base", "S3", "S3_base", "S4_base", "S4_primary_base", "S1",
                     "L3_base", "S2_L3", "S3_L3")
ABSTAIN_CATEGORIES = ("CAL_R", "R", "S4a", "S4b", "S4a_primary", "S4b_primary", "S4_L3")


def words_of(templates) -> set[str]:
    return {w for t in templates for w in t.replace("{x}", " ").replace("{q}", " ").split()}


LEXICON = {
    "base": words_of([t for ts in BASE_PREFIX.values() for t in ts] + list(BASE_POSSESSIVE.values()) + list(BASE_FRAMES)
                     + list(UNSUPPORTED_BASE.values()) + [AMBIGUOUS_BASE]),
    "taught": words_of(TAUGHT.values()), "held": words_of(HELD.values()),
    "novel": words_of(NOVEL_FRAMES) | set(NOVEL_FILLERS),
}


# --- rendering -----------------------------------------------------------------------------------------------
def render(entity: str, phrases: list[str], frame: str) -> str:
    """`phrases` are templates in execution order; the first applies to the entity."""
    np_ = entity
    for tpl in phrases:
        np_ = tpl.format(x=np_)
    return frame.format(q=np_)


def with_filler(frame: str, filler: str) -> str:
    head, rest = frame.split("{q}")
    return f"{head}{filler} {{q}}{rest}"


def base_phrases(path, style, rng):
    return [BASE_POSSESSIVE[r] if style == "possessive" else rng.choice(BASE_PREFIX[r]) for r in path]


def pseudo_words(rng, n: int, forbidden: set[str]) -> list[str]:
    """Pronounceable random words (4-9 letters), none of them a lexicon word or an entity."""
    cons, vows, out = "bcdfghjklmnprstvwz", "aeiou", []
    while len(out) < n:
        w = "".join(rng.choice(cons) + rng.choice(vows) for _ in range(rng.randint(2, 4)))[:rng.randint(4, 9)]
        if w not in forbidden and w not in out:
            out.append(w)
    return out


# --- the study's items -------------------------------------------------------------------------------------
def s4_stratum(world, program, entity, j) -> tuple[str, int, int]:
    """How much of the held-out relation the rest of the question determines. Alternatives replace the relation at
    position j (the held-out phrase) while every other relation stays fixed; an alternative is compatible when the
    whole program is well typed from the entity. Returns (stratum, compatible programs, distinct answers):
    - "type_determined": only one complete program satisfies the known type constraints;
    - "program_ambiguous_one_answer": several compatible programs, all giving the same answer for this entity;
    - "answer_ambiguous": several compatible programs giving at least two different answers (the primary test)."""
    start = next(t for t in world.types if world.types[t].contains(entity))
    compatible = []
    for r in sorted(world.relations):
        alt = tuple(program[:j]) + (r,) + tuple(program[j + 1:])
        types = R.path_types(world, alt)
        if types is not None and types[0] == start:
            compatible.append(alt)
    answers = {R.reference_answer(world, alt, entity) for alt in compatible}
    stratum = ("type_determined" if len(compatible) == 1 else
               "program_ambiguous_one_answer" if len(answers) == 1 else "answer_ambiguous")
    return stratum, len(compatible), len(answers)


# --- the frozen sampling design (revision 4) ---------------------------------------------------------------
SIZES = {"calibration": 1000, "id_test": 1000, "T1": 500, "S2": 800, "S3": 1000, "S4_population": 1000,
         "S4_primary": 2000, "S1_length3": 800, "S1_round_trip": 300, "L3_diagnostic": 300}
TRAIN_SHARE = 0.3
P_REFUSAL = 0.24  # an ID-distribution draw is a refusal with this probability, its kind uniform (0.06 each)
VARIANTS = {  # each draw of these designs is a group of matched variants
    "T1": ("T1", "T1_base"), "S2": ("S2", "S2_base"), "S3": ("S3", "S3_base"),
    "S4_population": ("S4a", "S4b", "S4_base"), "S4_primary": ("S4a_primary", "S4b_primary", "S4_primary_base"),
    "L3_diagnostic": ("L3_base", "S2_L3", "S3_L3", "S4_L3")}


def make_study(world: W.World, seed: int, sizes: dict | None = None) -> dict:
    """Every item for one world (revision 4).

    Training is fixed first: 30% of the (program of at most two relations, entity) instances, with their Base
    questions, the taught wording on the one-step ones, quota refusals (deduplicated within training) and, built
    later from these, the augmentation. Everything evaluated is then drawn independently, with replacement, from
    declared distributions over what training did not use:
    - the held-out population: the other 70% of short instances; length-3 instances are all held out;
    - calibration and the ID test: independent draws from one ID distribution (refusal with probability 0.24, its
      kind uniform, drawn from the refusal generator conditioned on its text not being a training question; else a
      held-out instance, uniform, in a uniform style, frame and Base phrasing). Duplicates within or between the two
      samples are allowed, as independent draws imply;
    - each other design draws its base case independently from its declared population and builds matched variants
      from it (see VARIANTS). Variants are separate categories, never pooled as independent observations.
    Every item has a unique draw id."""
    sizes = {**SIZES, **(sizes or {})}
    rng = random.Random(f"robustness-v4|{world.identity}|{seed}")
    short = list(R.well_typed_paths(world, 2))
    long3 = [p for p in R.well_typed_paths(world, 3) if len(p) == 3]
    instances = [(p, s) for p in short for s in world.types[world.relations[p[0]].source].values]
    rng.shuffle(instances)
    n_train = round(TRAIN_SHARE * len(instances))
    train_inst, held = instances[:n_train], instances[n_train:]
    held_len = {n: [(p, s) for p, s in held if len(p) == n] for n in (1, 2)}
    long_inst = [(p, s) for p in long3 for s in world.types[world.relations[p[0]].source].values]
    noncancel = [(p, s) for p, s in long_inst if TQ.path_family(p) == "length3"]
    round_trips = [(p, s) for p, s in long_inst if TQ.path_family(p) == "round_trip"]
    items: list[dict] = []

    def add(**kw):
        kw.setdefault("program", None)
        kw.setdefault("answer", None)
        kw.setdefault("family", None)
        kw.setdefault("style", "prefix")
        kw.setdefault("group", None)
        kw.setdefault("frame", None)
        kw["length"] = len(kw["program"]) if kw["program"] else None
        kw["id"] = f"{world.seed}-{len(items)}"
        kw["entity_position"] = TQ.entity_position(kw["text"], kw["entity"])
        items.append(kw)
        return kw

    def supported(split, category, p, s, style, frame, phrases, group=None, expected="answer", held_at=None, **extra):
        it = add(split=split, category=category, kind="supported", text=render(s, phrases, frame), program=list(p),
                 entity=s, expected=expected, answer=R.reference_answer(world, p, s), style=style,
                 family=TQ.path_family(p), group=group, frame=frame, **extra)
        if held_at is not None:
            it["held_position"] = held_at
            it["stratum"], it["compatible_programs"], it["compatible_answers"] = s4_stratum(world, p, s, held_at)
        return it

    def refusal(kind, split, category):
        frame, style = rng.choice(BASE_FRAMES), rng.choice(("prefix", "possessive"))
        base = {"split": split, "category": category, "kind": kind, "expected": "abstain", "frame": frame}
        if kind in ("ill_typed", "unknown_entity"):
            p = rng.choice(short)  # at most two relations: refusals respect the structural limit too
            source = world.relations[p[0]].source
            if kind == "ill_typed":
                s = rng.choice(world.types[rng.choice([t for t in sorted(world.types) if t != source])].values)
            else:
                n = len(world.types["Country"].values)
                s = {"Country": f"country_{n + rng.randint(0, 50)}", "City": f"country_{n + rng.randint(0, 50)}_city_0",
                     "Airport": f"country_{n + rng.randint(0, 50)}_city_0_airport",
                     "Operator": f"operator_{len(world.types['Operator'].values) + rng.randint(0, 20)}"}[source]
            return {**base, "text": render(s, base_phrases(p, style, rng), frame), "program": list(p), "entity": s,
                    "style": style, "outputs": list(p) + [J.END]}
        inner = rng.choice([()] + [p for p in short if len(p) == 1])
        if kind == "ambiguous":
            inner = rng.choice([()] + [p for p in short if len(p) == 1 and world.relations[p[0]].target == "Country"])
            s = rng.choice(world.types[world.relations[inner[0]].source if inner else "Country"].values)
            np_ = AMBIGUOUS_BASE.format(x=render(s, base_phrases(inner, "prefix", rng), "{q}"))
            return {**base, "text": frame.format(q=np_), "entity": s, "outputs": [J.AMBIGUOUS]}
        start_type = world.relations[inner[0]].source if inner else rng.choice(sorted(world.types))
        s = rng.choice(world.types[start_type].values)
        t = TQ.path_target(world, inner, s)
        np_ = UNSUPPORTED_BASE[t].format(x=render(s, base_phrases(inner, "prefix", rng), "{q}"))
        return {**base, "text": frame.format(q=np_), "entity": s, "outputs": [J.UNSUPPORTED]}

    # --- training (fixed; not part of any statistical claim)
    train_texts = set()
    for p, s in train_inst:
        style = rng.choice(("prefix", "possessive"))
        it = supported("train", "train_base", p, s, style, rng.choice(BASE_FRAMES), base_phrases(p, style, rng))
        it["outputs"] = list(p) + [J.END]
        train_texts.add(it["text"])
        if len(p) == 1:
            it = supported("train", "train_taught", p, s, "prefix", rng.choice(BASE_FRAMES), [TAUGHT[p[0]]])
            it["outputs"] = list(p) + [J.END]
            train_texts.add(it["text"])
    k = max(1, round(REFUSAL_SHARE * len(train_inst)))
    for kind in REFUSALS:
        made = 0
        while made < k:
            it = refusal(kind, "train", "R")
            if it["text"] not in train_texts:
                train_texts.add(it["text"])
                add(**it)
                made += 1

    # --- the ID distribution: calibration and the ID test are independent samples from it
    def id_draw(split, supported_category, refusal_category):
        if rng.random() < P_REFUSAL:
            kind = rng.choice(REFUSALS)
            while True:  # conditioned on not being a training question
                it = refusal(kind, split, refusal_category)
                if it["text"] not in train_texts:
                    return add(**it)
        p, s = rng.choice(held)
        style = rng.choice(("prefix", "possessive"))
        return supported(split, supported_category, p, s, style, rng.choice(BASE_FRAMES), base_phrases(p, style, rng))

    for _ in range(sizes["calibration"]):
        id_draw("calibration", "CAL", "CAL_R")
    for _ in range(sizes["id_test"]):
        id_draw("test", "ID", "R")

    def wrapper(frame):  # the declared novel-wrapper distribution: two novel frames, or the frame with one of two fillers
        return rng.choice([NOVEL_FRAMES[0], NOVEL_FRAMES[1], with_filler(frame, NOVEL_FILLERS[0]),
                           with_filler(frame, NOVEL_FILLERS[1])])

    gid = 0
    def group():
        nonlocal gid
        gid += 1
        return f"g{gid}"

    for _ in range(sizes["T1"]):  # one-step held-out instances, taught phrase against its Base phrasing
        p, s = rng.choice(held_len[1])
        g, frame = group(), rng.choice(BASE_FRAMES)
        supported("test", "T1", p, s, "prefix", frame, [TAUGHT[p[0]]], g)
        supported("test", "T1_base", p, s, "prefix", frame, base_phrases(p, "prefix", rng), g)
    for _ in range(sizes["S2"]):  # two-step: one Base phrase replaced by its taught synonym
        p, s = rng.choice(held_len[2])
        g, frame, phrases, j = group(), rng.choice(BASE_FRAMES), base_phrases(p, "prefix", rng), rng.randrange(2)
        supported("test", "S2", p, s, "prefix", frame, [TAUGHT[r] if i == j else phrases[i] for i, r in enumerate(p)], g)
        supported("test", "S2_base", p, s, "prefix", frame, phrases, g)
    for _ in range(sizes["S3"]):  # the broad inessential-novelty distribution, over all held-out short instances
        p, s = rng.choice(held)
        g, frame, phrases = group(), rng.choice(BASE_FRAMES), base_phrases(p, "prefix", rng)
        supported("test", "S3", p, s, "prefix", wrapper(frame), phrases, g)
        supported("test", "S3_base", p, s, "prefix", frame, phrases, g)

    def s4_case(categories, primary):
        while True:
            p, s = rng.choice(held)
            j = rng.randrange(len(p))
            if not primary or s4_stratum(world, p, s, j)[0] == "answer_ambiguous":
                break  # primary: conditioned on the answer-ambiguous stratum
        g, frame, phrases = group(), rng.choice(BASE_FRAMES), base_phrases(p, "prefix", rng)
        held_phrases = [HELD[r] if i == j else phrases[i] for i, r in enumerate(p)]
        a, b, base_cat = categories
        supported("test", a, p, s, "prefix", frame, held_phrases, g, expected="abstain", held_at=j)
        supported("test", b, p, s, "prefix", wrapper(frame), held_phrases, g, expected="abstain", held_at=j)
        supported("test", base_cat, p, s, "prefix", frame, phrases, g)

    for _ in range(sizes["S4_population"]):  # the full held-out population: every stratum, as in rounds 1-2
        s4_case(VARIANTS["S4_population"], primary=False)
    for _ in range(sizes["S4_primary"]):  # the primary test: answer-ambiguous base cases only
        s4_case(VARIANTS["S4_primary"], primary=True)
    for family, pool, n in (("length3", noncancel, sizes["S1_length3"]), ("round_trip", round_trips, sizes["S1_round_trip"])):
        for _ in range(n):
            p, s = rng.choice(pool)
            style = rng.choice(("prefix", "possessive"))
            supported("test", "S1", p, s, style, rng.choice(BASE_FRAMES), base_phrases(p, style, rng))
    for _ in range(sizes["L3_diagnostic"]):
        p, s = rng.choice(noncancel)
        g, frame, phrases, j = group(), rng.choice(BASE_FRAMES), base_phrases(p, "prefix", rng), rng.randrange(3)
        supported("test", "L3_base", p, s, "prefix", frame, phrases, g)
        supported("test", "S2_L3", p, s, "prefix", frame, [TAUGHT[r] if i == j else phrases[i] for i, r in enumerate(p)], g)
        supported("test", "S3_L3", p, s, "prefix", rng.choice(NOVEL_FRAMES), phrases, g)
        supported("test", "S4_L3", p, s, "prefix", frame, [HELD[r] if i == j else phrases[i] for i, r in enumerate(p)], g,
                  expected="abstain", held_at=j)
    entities = {v for t in world.types.values() for v in t.values}
    forbidden = set().union(*LEXICON.values()) | entities
    return {"items": items, "sizes": sizes, "train_instances": [(list(p), s) for p, s in train_inst],
            "pseudo_pool": pseudo_words(random.Random(f"pseudo|{world.identity}|{seed}"), 400, forbidden)}


def augment(study: dict, seed: int) -> list[dict]:
    """Novelty augmentation (declared extra supervision), built from Base training questions only. One pseudo-word
    pool, and the same count distribution (1-3 words), serve both uses, so neither the words nor their number can
    tell an essential use from an inessential one: only where they sit can.
    - essential: one relation phrase replaced by pseudo-words; target <unsupported>.
    - inessential: a pseudo-word frame, or pseudo-word fillers after a Base frame; target unchanged."""
    rng = random.Random(f"augment|{seed}")
    base = [x for x in study["items"] if x["category"] == "train_base"]
    pool, out = study["pseudo_pool"], []
    for kind in ("essential", "inessential"):
        for x in rng.sample(base, round(AUG_SHARE * len(base))):
            pws = " ".join(rng.sample(pool, rng.choice((1, 2, 3))))
            frame = next(f for f in BASE_FRAMES if x["text"].startswith(f.split("{q}")[0]))
            phrases = base_phrases(x["program"], x["style"], rng)
            if kind == "essential":
                j = rng.randrange(len(phrases))
                phrases[j] = ("{x} 's " + pws) if x["style"] == "possessive" else \
                             ("the " + pws + (" of" if rng.random() < 0.5 else "") + " {x}")
                text, outputs = render(x["entity"], phrases, frame), [J.UNSUPPORTED]
            else:
                wrapper = (pws + " {q} " + rng.choice("?.")) if rng.random() < 0.5 else with_filler(frame, pws)
                text, outputs = render(x["entity"], phrases, wrapper), list(x["program"]) + [J.END]
            out.append({**x, "category": f"aug_{kind}", "text": text, "outputs": outputs,
                        "entity_position": TQ.entity_position(text, x["entity"]), "id": f"{x['id']}-{kind}"})
    return out


def audit(world: W.World, study: dict, aug: list[dict]) -> dict:
    """Checks that must all hold before anything trains (revision 4). Beyond properties of the items present, it
    enforces the declared inventory: a missing category, a wrong count or a broken variant group fails the audit."""
    items, sizes = study["items"], study["sizes"]
    train = [x for x in items if x["split"] == "train"] + aug
    evals = [x for x in items if x["split"] != "train"]
    rel_count = lambda x: sum(o not in (J.END, J.UNSUPPORTED, J.AMBIGUOUS) for o in x["outputs"])
    toks = lambda x: set(x["text"].split())
    checks = {
        "training_targets_at_most_two_relations": all(rel_count(x) <= 2 for x in train),
        "taught_wording_only_in_one_step_training": all(
            not (toks(x) & (LEXICON["taught"] - LEXICON["base"])) or (x["program"] and len(x["program"]) == 1)
            for x in train),
        "no_held_or_novel_words_in_training": all(not toks(x) & ((LEXICON["held"] | LEXICON["novel"]) - LEXICON["base"]
                                                                 - LEXICON["taught"]) for x in train),
        "pseudo_words_disjoint": not set(study["pseudo_pool"]) & set().union(*LEXICON.values()),
        "lexicon_sets_disjoint_apart_from_function_words": not (
            (LEXICON["held"] & (LEXICON["base"] | LEXICON["taught"])) - {"the", "of"}),
        "one_style_per_question": all(x["text"].split().count("'s") in (0, x["length"] or 0) or x["kind"] != "supported"
                                      for x in items + aug),
        "all_questions_encodable": all(_encodable(x["text"]) for x in items + aug),
        "unique_draw_ids": len({x["id"] for x in items}) == len(items),
    }
    train_inst = {(tuple(p), s) for p, s in study["train_instances"]}
    train_texts = {x["text"] for x in items if x["split"] == "train"}
    checks["evaluation_instances_not_in_training"] = all(
        (tuple(x["program"]), x["entity"]) not in train_inst for x in evals if x["kind"] == "supported")
    checks["evaluation_refusals_are_not_training_questions"] = all(
        x["text"] not in train_texts for x in evals if x["kind"] != "supported")
    count = Counter(x["category"] for x in evals)
    families = Counter(x["family"] for x in evals if x["category"] == "S1")
    checks["declared_inventory"] = (
        count["CAL"] + count["CAL_R"] == sizes["calibration"] and count["ID"] + count["R"] == sizes["id_test"]
        and all(count[c] == sizes[design] for design, cats in VARIANTS.items() for c in cats)
        and families["length3"] == sizes["S1_length3"] and families["round_trip"] == sizes["S1_round_trip"]
        and set(count) == {"CAL", "CAL_R", "ID", "R", "S1"} | {c for cats in VARIANTS.values() for c in cats})
    groups = defaultdict(dict)
    for x in evals:
        if x["group"]:
            if x["category"] in groups[x["group"]]:
                groups[x["group"]]["__duplicate__"] = x
            groups[x["group"]][x["category"]] = x
    designs = {frozenset(cats): cats for cats in VARIANTS.values()}

    def matched(g):
        cats = designs.get(frozenset(g))
        if cats is None or len({(tuple(v["program"]), v["entity"]) for v in g.values()}) != 1:
            return False
        frame = {"T1": "T1_base", "S2": "S2_base", "S3": "S3_base", "S4a": "S4_base", "S4b": "S4_base",
                 "S4a_primary": "S4_primary_base", "S4b_primary": "S4_primary_base", "S2_L3": "L3_base", "S4_L3": "L3_base",
                 "S3_L3": "L3_base"}
        for c, ref in frame.items():
            if c in g:
                base = g[ref]["frame"]
                ok_frames = ({base} if c not in ("S3", "S4b", "S4b_primary", "S3_L3") else
                             set(NOVEL_FRAMES) | ({with_filler(base, f) for f in NOVEL_FILLERS} if c != "S3_L3" else set()))
                if g[c]["frame"] not in ok_frames:
                    return False
        return True
    checks["variant_groups_complete_and_matched"] = bool(groups) and all(matched(g) for g in groups.values())
    s4 = [x for x in evals if "held_position" in x]
    checks["s4_strata_recomputed"] = bool(s4) and all(
        x["stratum"] == s4_stratum(world, x["program"], x["entity"], x["held_position"])[0] for x in s4)
    checks["s4_primary_is_answer_ambiguous"] = all(
        x["stratum"] == "answer_ambiguous" for x in evals if x["category"] in ("S4a_primary", "S4b_primary"))
    return checks


def _encodable(text: str) -> bool:
    try:
        I.normalize(text)
        return True
    except I.UnsupportedInput:
        return False


# --- decisions, policies, certification, measurements ------------------------------------------------------
def decide(parses, items, world, registry) -> list[dict]:
    rows = []
    for x, parse in zip(items, parses):
        if parse is None:
            rows.append({"decision": "abstain", "reason": "unsupported characters", "answer": None, "score": 0.0,
                         "exact": False, "entity": None})
            continue
        d = J.propose(parse, world.types, registry)
        exact = parse.outcome == "program" and list(parse.steps) == (x["program"] or []) and parse.entity == x["entity"]
        rows.append({"decision": d["decision"], "reason": d["reason"], "answer": d["answer"], "score": parse.score,
                     "exact": exact, "entity": parse.entity, "outcome": parse.outcome, "steps": list(parse.steps)})
    return rows


def correct(x, row) -> bool:
    return row["decision"] == "answer" and x["expected"] == "answer" and row["answer"] == x["answer"]


def novel(x, row, vocab_words: set[str]) -> bool:
    return any(t not in vocab_words and t != row.get("entity") for t in x["text"].split())


def accepts(policy, x, row, threshold, vocab_words) -> tuple[bool, str]:
    if row["decision"] != "answer":
        return False, row["reason"]
    if policy in ("novelty", "novelty_certified") and novel(x, row, vocab_words):
        return False, "novelty"
    if policy in ("certified", "novelty_certified") and (threshold is None or row["score"] < threshold):
        return False, "threshold"
    return True, "answered"


def certify(cal_items, cal_rows, policy, vocab_words) -> dict:
    """Fixed-sequence certification of "answer when score >= t" on the calibration split (with the novelty
    filter applied first for novelty_certified)."""
    keep = [(x, r) for x, r in zip(cal_items, cal_rows)
            if r["decision"] == "answer" and not (policy == "novelty_certified" and novel(x, r, vocab_words))]
    cert = K.fixed_sequence_certify([r["score"] for _, r in keep], [correct(x, r) for x, r in keep], GRID)
    cert["records"] = [{"id": x["id"], "score": r["score"], "correct": correct(x, r)} for x, r in keep]
    return cert


def measure(items, rows, policy, threshold, vocab_words) -> dict:
    by_cat = defaultdict(list)
    for x, r in zip(items, rows):
        by_cat[x["category"]].append((x, r))
        if x["category"] == "T1":  # per taught phrase (small slices: accuracy only, no per-slice certificate)
            by_cat[f"T1:{x['program'][0]}"].append((x, r))
        if x["category"] == "S1":  # non-cancelling chains apart from round trips
            by_cat[f"S1:{x['family']}"].append((x, r))
        if "stratum" in x:  # S4 by how much the rest of the question determines the held-out relation
            by_cat[f"{x['category']}:{x['stratum']}"].append((x, r))
    out = {}
    for cat, pairs in by_cat.items():
        acc = [(x, r) for x, r in pairs if accepts(policy, x, r, threshold, vocab_words)[0]]
        errors = sum(not correct(x, r) for x, r in acc)
        reasons = Counter(accepts(policy, x, r, threshold, vocab_words)[1] for x, r in pairs)
        wrong = [(x, r) for x, r in acc if not correct(x, r)]
        out[cat] = {"n": len(pairs), "answered": len(acc), "errors": errors,
                    "coverage": len(acc) / len(pairs), "abstention": 1 - len(acc) / len(pairs),
                    "error_upper_95": K.clopper_pearson_upper(errors, len(acc)),
                    "abstention_interval_95": K.exact_interval(len(pairs) - len(acc), len(pairs)),
                    "exact_parse": sum(r["exact"] for _, r in pairs) / len(pairs),
                    "right_answer_wrong_parse": sum(correct(x, r) and not r["exact"] for x, r in acc),
                    "reasons": dict(reasons),
                    "wrong_entity": sum(r.get("entity") != x["entity"] for x, r in wrong),
                    "wrong_program": sum(r.get("entity") == x["entity"] for x, r in wrong)}
    return out


def collisions(study, vocab: J.TokenVocab) -> dict:
    """Input collisions over the test categories, for word encoding (training vocabulary, <unk> otherwise) and
    character encoding (the normalized token strings).
    - Program collisions: the same encoded input and copied entity, with different reference programs or answers.
    - Decision conflicts: the same encoded input and copied entity, requiring an answer in one case and a refusal in
      another. The copied entity is part of the key because the runtime binds it (an unseen real entity and an
      unknown one are both <unk> to a word model, yet only one binds).
    - The encoding-only versions, without the entity, are stricter and are reported too; for S3 against S4 they are
      the relevant ones, since the entity is the same."""
    test = [x for x in study["items"] if x["split"] == "test"]
    out = {}
    for name, enc in (("word", lambda x: tuple(vocab.encode(x["text"].split()))),
                      ("char", lambda x: tuple(x["text"].split()))):
        prog, dec, dec_strict = defaultdict(set), defaultdict(set), defaultdict(set)
        for x in test:
            e = enc(x)
            if x["program"]:
                prog[(e, x["entity"])].add((tuple(x["program"]), x["answer"]))
            dec[(e, x["entity"])].add(x["expected"])
            dec_strict[e].add(x["expected"])
        s34 = defaultdict(set)
        for x in test:
            if x["category"] in ("S3", "S3_base", "S4a", "S4b", "S4_base", "S4a_primary", "S4b_primary", "S4_primary_base"):
                s34[enc(x)].add(x["expected"])
        out[name] = {"program_collision_groups": sum(len(v) > 1 for v in prog.values()),
                     "decision_conflict_groups": sum(len(v) > 1 for v in dec.values()),
                     "decision_conflict_groups_encoding_only": sum(len(v) > 1 for v in dec_strict.values()),
                     "s3_s4_decision_conflicts": sum(len(v) > 1 for v in s34.values())}
    return out


def grammar_control() -> GrammarInterpreter:
    """The conservative grammar rule: Base and taught phrases only (no held-out or novel words)."""
    return GrammarInterpreter({r: BASE_PREFIX[r] + [TAUGHT[r]] for r in BASE_PREFIX},
                              {r: [t] for r, t in BASE_POSSESSIVE.items()})


REASONS = ("answered", "unsupported", "ambiguous", "malformed", "unknown entity", "ill-typed program",
           "module gave no answer", "module produced a value outside its declared type", "unsupported characters",
           "budget exhausted")


def compact_records(items, rows) -> dict:
    """Full-precision decision records, column-wise: everything the verifier needs to rebuild every metric, the
    certificate and the verdict without trusting any stored summary or flag."""
    return {"decision": [r["decision"] == "answer" for r in rows], "reason": [REASONS.index(r["reason"] or "answered")
                                                                          for r in rows],
            "score": [r["score"] for r in rows], "program": [r.get("steps") for r in rows],
            "entity": [r.get("entity") for r in rows], "answer": [r["answer"] for r in rows]}


def run_unit(world, study, aug, init, args) -> tuple[dict, dict]:
    """(summary, records) for one unit. Records cover calibration and test items in study order."""
    items = study["items"]
    table = G.table_registry(world)
    ordinary = [x for x in items if x["split"] == "train"]
    vocab_texts = [x["text"] for x in ordinary]
    vocab_words = {w for t in vocab_texts for w in t.split()}
    held = [x for x in items if x["split"] != "train"]
    cal = [x for x in held if x["split"] == "calibration"]
    evals = [x for x in held if x["split"] == "test"]
    unit = {"world": world.seed, "init": init, "arms": {}}
    records = {"world": world.seed, "init": init, "ids": [x["id"] for x in held], "arms": {}}
    for arm in ARMS:
        train_items = ordinary + (aug if arm.endswith("_aug") else [])
        examples = [J.Example(x["text"], tuple(x["outputs"]), x["entity_position"]) for x in train_items]
        t0 = time.time()
        net, vocab, loss = I.train(arm.split("_")[0], examples, vocab_texts, tuple(world.relations), init, args.updates)
        seconds = time.time() - t0
        rows = decide(I.interpret(net, vocab, [x["text"] for x in held]), held, world, table)
        cal_rows, ev_rows = rows[:len(cal)], rows[len(cal):]
        assert [x["id"] for x in held] == [x["id"] for x in cal + evals]
        res = {"train_seconds": round(seconds, 1), "train_loss": loss, "training_examples": len(examples), "policies": {}}
        for policy in POLICIES:
            cert = certify(cal, cal_rows, policy, vocab_words) if policy.endswith("certified") else None
            threshold = cert["threshold"] if cert else None
            res["policies"][policy] = {"certificate": {k: v for k, v in (cert or {}).items() if k != "records"} or None,
                                       "categories": measure(evals, ev_rows, policy, threshold, vocab_words),
                                       "calibration_categories": measure(cal, cal_rows, policy, threshold, vocab_words)}
        answered = [(x, r) for x, r in zip(cal, cal_rows) if r["decision"] == "answer"]
        a, b = K.platt_fit_increasing([r["score"] for _, r in answered], [correct(x, r) for x, r in answered])
        id_ans = [(x, r) for x, r in zip(evals, ev_rows) if x["category"] == "ID" and r["decision"] == "answer"]
        res["probability_calibration"] = {"platt": [a, b], "id_test": K.reliability(
            [K.platt_probability(r["score"], a, b) for _, r in id_ans], [correct(x, r) for x, r in id_ans])}
        unit["arms"][arm] = res
        records["arms"][arm] = compact_records(held, rows)
        c = res["policies"]["certified"]
        cats = c["categories"]
        print(f"  world {world.seed} init {init} {arm:8s} ({seconds:.0f}s) cert {c['certificate']['threshold']} | "
              + " ".join(f"{k} {cats[k]['coverage']:.2f}/{cats[k]['errors']}" for k in ("ID", "S2", "S3", "S1:length3"))
              + " | " + " ".join(f"{k} abst {cats[k]['abstention']:.3f}" for k in ("R", "S4a_primary", "S4b_primary")),
              flush=True)
    grammar = grammar_control()
    g_rows = decide([grammar.parse(x["text"]) for x in evals], evals, world, table)
    unit["grammar"] = {"categories": measure(evals, g_rows, "none", None, vocab_words)}
    records["grammar"] = compact_records(evals, g_rows)
    return unit, records


def run(args, out: Path) -> dict:
    worlds, units = {}, []
    (out / "records").mkdir(parents=True, exist_ok=True)
    for wseed in args.worlds:
        world = W.generate_world(wseed, n_countries=args.countries)
        study = make_study(world, wseed)
        aug = augment(study, wseed)
        checks = audit(world, study, aug)
        if not all(checks.values()):
            raise SystemExit(f"audit failed for world {wseed}: {[k for k, v in checks.items() if not v]}")
        vocab = J.TokenVocab([x["text"] for x in study["items"] if x["split"] == "train"])
        counts = Counter((x["split"], x["category"]) for x in study["items"])
        worlds[str(wseed)] = {"identity": world.identity, "audit": checks, "collisions": collisions(study, vocab),
                              "counts": {f"{s}/{c}": n for (s, c), n in sorted(counts.items())},
                              "augmentation": dict(Counter(x["category"] for x in aug))}
        print(f"world {wseed}: audit passed", flush=True)
        for init in args.inits:
            unit, records = run_unit(world, study, aug, init, args)
            units.append(unit)
            with gzip.open(out / "records" / f"w{wseed}_i{init}.json.gz", "wt") as f:
                json.dump(records, f)
    return {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "environment": {"torch": torch.__version__, "python": platform.python_version(),
                            "platform": platform.platform()}, "grid": list(GRID), "sizes": SIZES,
            "diagnostic_cutoff": DIAGNOSTIC_CUTOFF, "worlds": worlds, "units": units}


# --- the frozen predictions (revision 4) --------------------------------------------------------------------
CONFIRMATION = {"worlds": (700, 701, 702), "inits": (0, 1, 2, 3, 4)}
LEAD = "word_aug"
COVERAGE, ERROR_BOUND, REFUSAL, ID_COVERAGE, T1_ACCURACY = 0.80, 0.01, 0.95, 0.90, 0.95


def unit_outcomes(arm_res: dict) -> dict:
    """Every per-unit criterion for one arm, from its stored summary (the verifier rebuilds the same from records)."""
    cert = arm_res["policies"]["certified"]["certificate"]
    c = arm_res["policies"]["certified"]["categories"]
    n = arm_res["policies"]["none"]["categories"]
    certified = bool(cert and cert["certified"])
    answers = lambda k, cov: certified and c[k]["coverage"] >= cov and c[k]["error_upper_95"] <= ERROR_BOUND
    return {"certified": certified,
            "sanity": answers("ID", ID_COVERAGE) and c["R"]["abstention"] >= REFUSAL,
            "taught_wording_learned": (n["T1"]["answered"] - n["T1"]["errors"]) / n["T1"]["n"] >= T1_ACCURACY,
            "joint_novelty": answers("S3", COVERAGE) and c["S4a_primary"]["abstention"] >= REFUSAL
                             and c["S4b_primary"]["abstention"] >= REFUSAL,
            "s2_transfer": answers("S2", COVERAGE), "s1_structure": answers("S1:length3", COVERAGE)}


def evaluate(result: dict, worlds=CONFIRMATION["worlds"], inits=CONFIRMATION["inits"]) -> dict:
    """The frozen verdict. Every criterion is judged jointly within a unit; a world holds a criterion when at least
    4 of its 5 inits meet it; a prediction holds when every world does. An uncertified unit is not successful and
    stays in the denominator."""
    from integrated import inventory
    bad = inventory.problems([(u["world"], u["init"], arm) for u in result["units"] for arm in u["arms"]],
                             [(w, i, a) for w in worlds for i in inits for a in ARMS])
    if bad:
        return {"complete": False, **bad}
    by = {(u["world"], u["init"]): u for u in result["units"]}
    per_unit = {arm: {f"{w}/{i}": unit_outcomes(by[(w, i)]["arms"][arm]) for w in worlds for i in inits} for arm in ARMS}
    holds = lambda arm, key: all(sum(per_unit[arm][f"{w}/{i}"][key] for i in inits) >= 4 for w in worlds)
    by_world = lambda arm, key: {w: sum(per_unit[arm][f"{w}/{i}"][key] for i in inits) for w in worlds}
    return {"complete": True,
            "predictions": {"1_sanity_every_arm": all(holds(a, "sanity") for a in ARMS),
                            "2_taught_wording_learned_every_arm": all(holds(a, "taught_wording_learned") for a in ARMS),
                            "3_word_aug_joint_novelty": holds(LEAD, "joint_novelty")},
            "separate_verdicts": {key: {a: holds(a, key) for a in ARMS}
                                  for key in ("joint_novelty", "s2_transfer", "s1_structure")},
            "successes_by_world": {key: {a: by_world(a, key) for a in ARMS}
                                   for key in ("certified", "sanity", "taught_wording_learned", "joint_novelty",
                                               "s2_transfer", "s1_structure")},
            "per_unit": per_unit}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--worlds", nargs="+", type=int, default=list(CONFIRMATION["worlds"]))
    p.add_argument("--inits", nargs="+", type=int, default=list(CONFIRMATION["inits"]))
    p.add_argument("--countries", type=int, default=96)
    p.add_argument("--updates", type=int, default=1500)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("runs/titnn_robustness"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    out = args.out / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    result = run(args, out)
    (out / "interpreter_robustness.json").write_text(json.dumps(result))
    if tuple(args.worlds) == CONFIRMATION["worlds"] and tuple(args.inits) == CONFIRMATION["inits"]:
        verdict = evaluate(result)
        (out / "verdict.json").write_text(json.dumps(verdict, indent=1))
        print(json.dumps(verdict["predictions"], indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
