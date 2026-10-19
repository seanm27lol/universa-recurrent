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
SPLITS = (("train", 0.3), ("calibration", 0.3), ("test", 0.4))
REFUSAL_SHARE = 0.06
REFUSALS = ("ill_typed", "unsupported", "unknown_entity", "ambiguous")
AUG_SHARE = 0.25
ARMS = ("word", "word_aug", "char", "char_aug")
POLICIES = ("none", "certified", "novelty", "novelty_certified")
GRID = (0.9999, 0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0)  # provisional; the pilot sets it
ANSWER_CATEGORIES = ("ID", "T1", "T1_base", "S2", "S2_base", "Q_base", "S3", "S1", "L3_base", "S2_L3", "S3_L3")
ABSTAIN_CATEGORIES = ("R", "S4a", "S4b", "S4_L3")


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
def make_study(world: W.World, seed: int, n_long: int = 600, n_diag: int = 300) -> dict:
    """Every item for one world: {"items": [...], "pseudo_pool": [...]}. Item fields: id, split, category,
    kind, text, program (execution order, or the intended program for S4), entity, entity_position, expected
    ("answer" or "abstain"), answer, length, family, style, group (shared by matched variants), outputs
    (training target, train items only)."""
    rng = random.Random(f"robustness|{world.identity}|{seed}")
    short = [p for p in R.well_typed_paths(world, 2)]
    long3 = [p for p in R.well_typed_paths(world, 3) if len(p) == 3]
    instances = [(p, s) for p in short for s in world.types[world.relations[p[0]].source].values]
    rng.shuffle(instances)
    cut, parts = 0, {}
    for split, share in SPLITS:
        n = round(share * len(instances)) if split != "test" else len(instances) - cut
        parts[split] = instances[cut:cut + n]
        cut += n
    items, refusal_texts = [], set()

    def add(**kw):
        kw.setdefault("program", None)
        kw.setdefault("answer", None)
        kw.setdefault("family", None)
        kw.setdefault("style", "prefix")
        kw.setdefault("group", None)
        kw["length"] = len(kw["program"]) if kw["program"] else None
        kw["id"] = f"{world.seed}-{len(items)}"
        kw["entity_position"] = TQ.entity_position(kw["text"], kw["entity"])
        items.append(kw)
        return kw

    def supported(split, category, p, s, style, frame, phrases, group, expected="answer"):
        return add(split=split, category=category, kind="supported", text=render(s, phrases, frame), program=list(p),
                   entity=s, expected=expected, answer=R.reference_answer(world, p, s), style=style,
                   family=TQ.path_family(p), group=group)

    def refusals(split, n_instances):
        k = max(1, round(REFUSAL_SHARE * n_instances))
        for kind in REFUSALS:
            made = 0
            while made < k:
                it = refusal(kind, rng, split)
                if it["text"] in refusal_texts:
                    continue  # refusal strings are never repeated, within or across splits
                refusal_texts.add(it["text"])
                add(**it)
                made += 1

    def refusal(kind, rng, split):
        base = {"split": split, "category": "R", "kind": kind, "expected": "abstain"}
        frame, style = rng.choice(BASE_FRAMES), rng.choice(("prefix", "possessive"))
        if kind in ("ill_typed", "unknown_entity"):
            p = rng.choice(short)  # never longer than two relations: refusals must respect the structural limit
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

    # training: Base wording up to two relations; taught wording only in one-step questions
    for p, s in parts["train"]:
        style = rng.choice(("prefix", "possessive"))
        it = supported("train", "train_base", p, s, style, rng.choice(BASE_FRAMES), base_phrases(p, style, rng), None)
        it["outputs"] = list(p) + [J.END]
        if len(p) == 1:
            it = supported("train", "train_taught", p, s, "prefix", rng.choice(BASE_FRAMES), [TAUGHT[p[0]]], None)
            it["outputs"] = list(p) + [J.END]
    refusals("train", len(parts["train"]))
    # calibration (certification population): Base wording, one rendering per instance, plus refusals
    for p, s in parts["calibration"]:
        style = rng.choice(("prefix", "possessive"))
        supported("calibration", "CAL", p, s, style, rng.choice(BASE_FRAMES), base_phrases(p, style, rng), None)
    refusals("calibration", len(parts["calibration"]))
    # test categories, matched within each instance
    for gi, (p, s) in enumerate(parts["test"]):
        g = f"t{gi}"
        style = rng.choice(("prefix", "possessive"))
        supported("test", "ID", p, s, style, rng.choice(BASE_FRAMES), base_phrases(p, style, rng), g)
        frame, phrases = rng.choice(BASE_FRAMES), base_phrases(p, "prefix", rng)
        if len(p) == 1:
            supported("test", "T1", p, s, "prefix", frame, [TAUGHT[p[0]]], g)
            supported("test", "T1_base", p, s, "prefix", frame, phrases, g)
        else:
            j = rng.randrange(2)
            supported("test", "S2", p, s, "prefix", frame, [TAUGHT[r] if i == j else phrases[i] for i, r in enumerate(p)], g)
            supported("test", "S2_base", p, s, "prefix", frame, phrases, g)
        wrapper = rng.choice([rng.choice(NOVEL_FRAMES), with_filler(frame, rng.choice(NOVEL_FILLERS))])
        j = rng.randrange(len(p))
        held = [HELD[r] if i == j else phrases[i] for i, r in enumerate(p)]
        supported("test", "Q_base", p, s, "prefix", frame, phrases, g)
        supported("test", "S3", p, s, "prefix", wrapper, phrases, g)
        supported("test", "S4a", p, s, "prefix", frame, held, g, expected="abstain")
        supported("test", "S4b", p, s, "prefix", wrapper, held, g, expected="abstain")
    refusals("test", len(parts["test"]))
    # length three: structure (S1) and its interaction diagnostics
    pool = [(p, s) for p in long3 for s in world.types[world.relations[p[0]].source].values]
    rng.shuffle(pool)
    for gi, (p, s) in enumerate(pool[:n_long]):
        style = rng.choice(("prefix", "possessive"))
        supported("test", "S1", p, s, style, rng.choice(BASE_FRAMES), base_phrases(p, style, rng), f"l{gi}")
    for gi, (p, s) in enumerate(pool[n_long:n_long + n_diag]):
        g, frame, phrases = f"d{gi}", rng.choice(BASE_FRAMES), base_phrases(p, "prefix", rng)
        supported("test", "L3_base", p, s, "prefix", frame, phrases, g)
        j = rng.randrange(3)
        supported("test", "S2_L3", p, s, "prefix", frame, [TAUGHT[r] if i == j else phrases[i] for i, r in enumerate(p)], g)
        supported("test", "S3_L3", p, s, "prefix", rng.choice(NOVEL_FRAMES), phrases, g)
        supported("test", "S4_L3", p, s, "prefix", frame, [HELD[r] if i == j else phrases[i] for i, r in enumerate(p)], g,
                  expected="abstain")
    entities = {v for t in world.types.values() for v in t.values}
    forbidden = set().union(*LEXICON.values()) | entities
    return {"items": items, "pseudo_pool": pseudo_words(random.Random(f"pseudo|{world.identity}|{seed}"), 400, forbidden)}


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


def audit(study: dict, aug: list[dict]) -> dict:
    """Checks that must all hold before anything trains."""
    items = study["items"]
    train = [x for x in items if x["split"] == "train"] + aug
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
    }
    seen = defaultdict(set)
    for x in items:
        if x["kind"] == "supported" and x["split"] != "test" or x["category"] in ("ID",):
            seen[(tuple(x["program"]), x["entity"])].add(x["split"])
    checks["splits_disjoint_by_instance"] = all(len(v) == 1 for v in seen.values())
    texts = Counter(x["text"] for x in items if x["kind"] != "supported")
    checks["refusal_strings_unique"] = all(c == 1 for c in texts.values())
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
    out = {}
    for cat, pairs in by_cat.items():
        acc = [(x, r) for x, r in pairs if accepts(policy, x, r, threshold, vocab_words)[0]]
        errors = sum(not correct(x, r) for x, r in acc)
        reasons = Counter(accepts(policy, x, r, threshold, vocab_words)[1] for x, r in pairs)
        wrong = [(x, r) for x, r in acc if not correct(x, r)]
        out[cat] = {"n": len(pairs), "answered": len(acc), "errors": errors,
                    "coverage": len(acc) / len(pairs), "abstention": 1 - len(acc) / len(pairs),
                    "error_upper_95": K.clopper_pearson_upper(errors, len(acc)),
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
            if x["category"] in ("S3", "S4a", "S4b", "Q_base"):
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


def run_unit(world, study, aug, init, args) -> dict:
    items = study["items"]
    table = G.table_registry(world)
    ordinary = [x for x in items if x["split"] == "train"]
    vocab_texts = [x["text"] for x in ordinary]
    vocab_words = {w for t in vocab_texts for w in t.split()}
    cal = [x for x in items if x["split"] == "calibration"]
    evals = [x for x in items if x["split"] == "test"]
    unit = {"world": world.seed, "init": init, "arms": {}}
    for arm in ARMS:
        train_items = ordinary + (aug if arm.endswith("_aug") else [])
        examples = [J.Example(x["text"], tuple(x["outputs"]), x["entity_position"]) for x in train_items]
        t0 = time.time()
        net, vocab, loss = I.train(arm.split("_")[0], examples, vocab_texts, tuple(world.relations), init, args.updates)
        seconds = time.time() - t0
        cal_rows = decide(I.interpret(net, vocab, [x["text"] for x in cal]), cal, world, table)
        ev_rows = decide(I.interpret(net, vocab, [x["text"] for x in evals]), evals, world, table)
        res = {"train_seconds": round(seconds, 1), "train_loss": loss, "training_examples": len(examples), "policies": {}}
        for policy in POLICIES:
            cert = certify(cal, cal_rows, policy, vocab_words) if policy.endswith("certified") else None
            threshold = cert["threshold"] if cert else None
            res["policies"][policy] = {"certificate": {k: v for k, v in (cert or {}).items() if k != "records"} or None,
                                       "categories": measure(evals, ev_rows, policy, threshold, vocab_words),
                                       "calibration_categories": measure(cal, cal_rows, policy, threshold, vocab_words)}
            if cert:
                res["policies"][policy]["certification_records"] = cert["records"]
        answered = [(x, r) for x, r in zip(cal, cal_rows) if r["decision"] == "answer"]
        a, b = K.platt_fit_increasing([r["score"] for _, r in answered], [correct(x, r) for x, r in answered])
        id_ans = [(x, r) for x, r in zip(evals, ev_rows) if x["category"] == "ID" and r["decision"] == "answer"]
        res["probability_calibration"] = {"platt": [a, b], "id_test": K.reliability(
            [K.platt_probability(r["score"], a, b) for _, r in id_ans], [correct(x, r) for x, r in id_ans])}
        res["records"] = [{"id": x["id"], "category": x["category"], "decision": r["decision"], "reason": r["reason"],
                           "score": round(r["score"], 6), "correct": correct(x, r), "exact": r["exact"]}
                          for x, r in zip(evals, ev_rows)]
        unit["arms"][arm] = res
        c = res["policies"]["certified"]
        cats = c["categories"]
        print(f"  world {world.seed} init {init} {arm:8s} ({seconds:.0f}s) cert {c['certificate']['threshold']} | "
              + " ".join(f"{k} {cats[k]['coverage']:.2f}/{cats[k]['errors']}" for k in ("ID", "T1", "S2", "S3", "S1")
                         if k in cats)
              + " | " + " ".join(f"{k} abst {cats[k]['abstention']:.2f}" for k in ("R", "S4a", "S4b") if k in cats),
              flush=True)
    grammar = grammar_control()
    g_rows = decide([grammar.parse(x["text"]) for x in evals], evals, world, table)
    unit["grammar"] = {"categories": measure(evals, g_rows, "none", None, vocab_words)}
    return unit


def run(args) -> dict:
    worlds, units = {}, []
    for wseed in args.worlds:
        world = W.generate_world(wseed, n_countries=args.countries)
        study = make_study(world, wseed)
        aug = augment(study, wseed)
        checks = audit(study, aug)
        if not all(checks.values()):
            raise SystemExit(f"audit failed for world {wseed}: {[k for k, v in checks.items() if not v]}")
        vocab = J.TokenVocab([x["text"] for x in study["items"] if x["split"] == "train"])
        counts = Counter((x["split"], x["category"]) for x in study["items"])
        worlds[str(wseed)] = {"identity": world.identity, "audit": checks, "collisions": collisions(study, vocab),
                              "counts": {f"{s}/{c}": n for (s, c), n in sorted(counts.items())},
                              "augmentation": dict(Counter(x["category"] for x in aug))}
        print(f"world {wseed}: audit passed; counts {worlds[str(wseed)]['counts']}", flush=True)
        for init in args.inits:
            units.append(run_unit(world, study, aug, init, args))
    return {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "environment": {"torch": torch.__version__, "python": platform.python_version(),
                            "platform": platform.platform()}, "grid": list(GRID), "worlds": worlds, "units": units}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--worlds", nargs="+", type=int, default=[100, 101])
    p.add_argument("--inits", nargs="+", type=int, default=[100, 101])
    p.add_argument("--countries", type=int, default=48)
    p.add_argument("--updates", type=int, default=1500)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("runs/titnn_robustness_pilot"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    result = run(args)
    out = args.out / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    (out / "interpreter_robustness.json").write_text(json.dumps(result))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
