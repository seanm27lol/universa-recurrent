"""Controlled-language questions, a learned interpreter and calibrated abstention, end to end (integration plan
PR 6; protocols/typed_routing_v1.md).

**Questions.** Every well-typed path of length 1–3, from every start value, becomes one question, rendered
with randomly chosen phrasings. For example, "what is the operator of the airport of the largest city of
country_3 ?" or "name country_3 's largest city 's airport 's operator ." These question instances are
split by instance into train (30%), calibration (30%) and test (40%).

Four kinds of question should be refused, about 6% of each split each:
- **ill-typed:** a valid relation applied to the wrong kind of entity ("the capital of <a city>"); the
  runtime must reject it;
- **unsupported:** a relation the system does not have ("the mayor of ...");
- **unknown entity:** an entity that is not in the world;
- **ambiguous:** "the main city of ...", which could mean the capital or the largest city.

**Shift.** Every test instance is rendered again with phrasings and a question frame never used in
training ("the seat of government of ...", "tell me ... ."). Ambiguous and unsupported questions are added
with unseen phrasings too.

**Systems.** All are trained on the same train split.
- `typed_specialists` (proposed): the interpreter, the typed runtime, and learned specialist modules
  (PR 4, trained on atomic facts).
- `typed_shared_dense`: the same interpreter with the shared-network modules.
- `typed_table`: the same interpreter with table modules. It isolates interpreter error, as a diagnostic.
- `answer_only`: the same encoder reading the question and classifying the answer directly (or "abstain").
  It gets the same train questions plus every atomic fact as a one-step question, in every seen phrasing
  and frame, so it has seen every fact the modules saw in every surface form.

Each system's score is calibrated on the calibration split, and its threshold chosen there
(`integrated/calibration.py`). The test and shift splits are then scored once.

    .venv-phase2/bin/python research/ti_tnn/bench/typed_questions.py --worlds 400 401 402 --inits 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrated import calibration as K  # noqa: E402
from integrated import modules as M  # noqa: E402
from integrated import reference as R  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import routing as J  # noqa: E402
from integrated.contracts import Budget, Program  # noqa: E402
from integrated.runtime import execute  # noqa: E402

PHRASES = {  # relation -> (seen prefix phrasings, seen possessive phrasing, held-out prefix phrasing)
    "capital_of": (["the capital of {x}", "the capital city of {x}"], "{x} 's capital", "the seat of government of {x}"),
    "largest_city_of": (["the largest city of {x}", "the biggest city in {x}"], "{x} 's largest city",
                        "the most populous city of {x}"),
    "country_of": (["the country of {x}", "the country containing {x}"], "{x} 's country", "the nation that includes {x}"),
    "airport_of": (["the airport of {x}", "the airport serving {x}"], "{x} 's airport", "the airfield that serves {x}"),
    "operator_of": (["the operator of {x}", "the company operating {x}"], "{x} 's operator", "the firm that runs {x}"),
}
# A question uses one style throughout. Mixing prefix and possessive phrasings would make some strings parse two
# ways ("the country containing X 's capital"), with different programs.
FRAMES = (["what is {q} ?", "name {q} ."], ["tell me {q} ."])
UNSUPPORTED = {  # applies to a noun phrase of this type
    "Country": (["the population of {x}"], ["the currency of {x}"]),
    "City": (["the mayor of {x}"], ["the postcode of {x}"]),
    "Airport": (["the runway of {x}"], ["the elevation of {x}"]),
    "Operator": (["the founder of {x}"], ["the headquarters of {x}"]),
}
AMBIGUOUS = (["the main city of {x}"], ["the principal city of {x}"])  # applies to a Country noun phrase
INVALID = ("ill_typed", "unsupported", "unknown_entity", "ambiguous")
SPLITS = (("train", 0.3), ("calibration", 0.3), ("test", 0.4))
INVALID_SHARE = 0.06
SYSTEMS = ("typed_specialists", "typed_shared_dense", "typed_table", "answer_only")
CANCELLING = {("capital_of", "country_of"), ("largest_city_of", "country_of")}  # a round trip back to the country
FAMILIES = ("length1", "length2", "length3", "round_trip")


def path_family(path) -> str:
    """As in plan PR 4: a path containing a round trip is scored apart from the non-cancelling chains."""
    if any(pair in CANCELLING for pair in zip(path, path[1:])):
        return "round_trip"
    return f"length{len(path)}"
MODULES = {"typed_specialists": "specialists", "typed_shared_dense": "shared_dense"}


def noun_phrase(path, entity, rng, held_out, possessive=None):
    """The noun phrase for a path. Seen questions are all prefix or all possessive (chosen at random unless
    given); held-out questions use the held-out prefix phrasings."""
    if possessive is None:
        possessive = not held_out and rng.random() < 0.5
    np_ = entity
    for r in path:
        prefix, poss, unseen = PHRASES[r]
        np_ = (unseen if held_out else poss if possessive else rng.choice(prefix)).format(x=np_)
    return np_


def frame(np_, rng, held_out):
    return rng.choice(FRAMES[1] if held_out else FRAMES[0]).format(q=np_)


def entity_position(text, entity):
    toks = J.tokenize(text)
    assert toks.count(entity) == 1, (text, entity)
    return toks.index(entity)


def path_target(world, path, start):
    return world.relations[path[-1]].target if path else next(t for t in world.types if world.types[t].contains(start))


def make_items(world: W.World, seed: int) -> list[dict]:
    """Every question for one world, with its split, kind, target outputs and reference answer."""
    rng = random.Random(f"questions|{world.identity}|{seed}")
    paths = R.well_typed_paths(world, 3)
    instances = [(p, s) for p in paths for s in world.types[world.relations[p[0]].source].values]
    rng.shuffle(instances)
    items, cut = [], 0
    for split, share in SPLITS:
        n = round(share * len(instances)) if split != "test" else len(instances) - cut
        chunk = instances[cut:cut + n]
        cut += n
        variants = [(split, False)] + ([("shift", True)] if split == "test" else [])
        for out_split, held_out in variants:
            for p, s in chunk:
                text = frame(noun_phrase(p, s, rng, held_out), rng, held_out)
                items.append({"split": out_split, "kind": "supported", "text": text, "program": list(p), "entity": s,
                              "expected": "answer", "answer": R.reference_answer(world, p, s), "length": len(p),
                              "family": path_family(p)})
            k = max(1, round(INVALID_SHARE * len(chunk)))
            for kind in INVALID:
                for _ in range(k):
                    items.append(invalid_item(world, kind, rng, out_split, held_out, paths))
    for i, item in enumerate(items):
        item["id"] = f"{world.seed}-{i}"
        item["entity_position"] = entity_position(item["text"], item["entity"])
    return items


def invalid_item(world, kind, rng, split, held_out, paths) -> dict:
    base = {"split": split, "kind": kind, "expected": "abstain", "answer": None}
    if kind == "ill_typed":
        p = rng.choice(paths)
        source = world.relations[p[0]].source
        wrong = rng.choice([t for t in sorted(world.types) if t != source])
        s = rng.choice(world.types[wrong].values)
        return {**base, "text": frame(noun_phrase(p, s, rng, held_out), rng, held_out), "program": list(p), "entity": s,
                "length": len(p), "outputs": list(p) + [J.END]}
    if kind == "unknown_entity":
        p = rng.choice(paths)
        source = world.relations[p[0]].source
        n = len(world.types["Country"].values)
        fake = {"Country": f"country_{n + rng.randint(0, 50)}", "City": f"country_{n + rng.randint(0, 50)}_city_0",
                "Airport": f"country_{n + rng.randint(0, 50)}_city_0_airport",
                "Operator": f"operator_{len(world.types['Operator'].values) + rng.randint(0, 20)}"}[source]
        return {**base, "text": frame(noun_phrase(p, fake, rng, held_out), rng, held_out), "program": list(p),
                "entity": fake, "length": len(p), "outputs": list(p) + [J.END]}
    inner = rng.choice([()] + [p for p in paths if len(p) <= 1])
    if kind == "ambiguous":
        inner = rng.choice([()] + [p for p in paths if len(p) == 1 and world.relations[p[0]].target == "Country"])
        start_type = world.relations[inner[0]].source if inner else "Country"
        s = rng.choice(world.types[start_type].values)
        np_ = rng.choice(AMBIGUOUS[1] if held_out else AMBIGUOUS[0]).format(x=noun_phrase(inner, s, rng, held_out, False))
        if rng.random() < 0.5:  # sometimes used inside a longer question
            np_ = (PHRASES["airport_of"][2] if held_out else rng.choice(PHRASES["airport_of"][0])).format(x=np_)
        return {**base, "text": frame(np_, rng, held_out), "program": None, "entity": s, "length": None,
                "outputs": [J.AMBIGUOUS]}
    start_type = world.relations[inner[0]].source if inner else rng.choice(sorted(world.types))
    s = rng.choice(world.types[start_type].values)
    t = path_target(world, inner, s)
    np_ = rng.choice(UNSUPPORTED[t][1 if held_out else 0]).format(x=noun_phrase(inner, s, rng, held_out, False))
    return {**base, "text": frame(np_, rng, held_out), "program": None, "entity": s, "length": None,
            "outputs": [J.UNSUPPORTED]}


def outputs_of(item) -> tuple[str, ...]:
    return tuple(item["outputs"]) if "outputs" in item else tuple(item["program"]) + (J.END,)


class AnswerOnlyQA(nn.Module):
    """The interpreter's encoder, then a classifier over every entity plus "abstain"."""

    def __init__(self, n_words, n_answers, d=48, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(n_words, d, padding_idx=0)
        self.encoder = nn.GRU(d, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.Linear(2 * hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, n_answers))

    def forward(self, ids):
        mask = ids != 0
        packed = nn.utils.rnn.pack_padded_sequence(self.embed(ids), mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        enc, _ = self.encoder(packed)
        enc, _ = nn.utils.rnn.pad_packed_sequence(enc, batch_first=True, total_length=ids.shape[1])
        return self.out((enc * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True))


def atomic_questions(world):
    """Every atomic fact as a one-step question, in every seen phrasing and frame."""
    out = []
    for r in sorted(world.relations):
        prefix, poss, _ = PHRASES[r]
        for s, a in sorted(world.facts[r].items()):
            for phrase in prefix + [poss]:
                for f in FRAMES[0]:
                    out.append({"text": f.format(q=phrase.format(x=s)), "expected": "answer", "answer": a})
    return out


def train_answer_only(world, train_items, seed, epochs, lr, d, hidden):
    """The answer-only baseline, on the train questions plus every atomic fact as a question."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    data = train_items + atomic_questions(world)
    answers = sorted({v for t in world.types.values() for v in t.values}) + ["<abstain>"]
    a_index = {a: i for i, a in enumerate(answers)}
    vocab = J.TokenVocab([x["text"] for x in data])
    net = AnswerOnlyQA(len(vocab.words), len(answers), d, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    y_all = torch.tensor([a_index[x["answer"]] if x["expected"] == "answer" else a_index["<abstain>"] for x in data])
    for _ in range(epochs):
        order = torch.randperm(len(data), generator=gen)
        for s in range(0, len(order), 64):
            idx = order[s:s + 64].tolist()
            ids, *_ = J._batch([J.Example(data[i]["text"], (), 0) for i in idx], vocab, (J.END,), 0.1, gen)
            loss = F.cross_entropy(net(ids), y_all[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return net, vocab, answers


@torch.no_grad()
def answer_only_predict(net, vocab, answers, texts):
    out = []
    for s in range(0, len(texts), 256):
        ids, *_ = J._batch([J.Example(t, (), 0) for t in texts[s:s + 256]], vocab, (J.END,))
        p = F.softmax(net(ids), -1)
        conf, k = p.max(-1)
        out += [(answers[int(i)], float(c)) for i, c in zip(k, conf)]
    return out


def decisions(system, items, world, interp, registry, qa):
    """Pre-threshold decision, answer and score for every item."""
    texts = [x["text"] for x in items]
    if system == "answer_only":
        net, vocab, answers = qa
        rows = []
        for (a, c) in answer_only_predict(net, vocab, answers, texts):
            rows.append({"decision": "abstain" if a == "<abstain>" else "answer", "answer": None if a == "<abstain>" else a,
                         "score": c, "parse": None, "reason": "classified abstain" if a == "<abstain>" else ""})
        return rows
    net, vocab = interp
    rows = []
    for parse in J.interpret(net, vocab, texts):
        d = J.propose(parse, world.types, registry)
        rows.append({"decision": d["decision"], "answer": d["answer"], "score": parse.score, "reason": d["reason"],
                     "parse": {"outcome": parse.outcome, "steps": list(parse.steps), "entity": parse.entity}})
    return rows


def correct_answer(item, row):
    return row["decision"] == "answer" and item["expected"] == "answer" and row["answer"] == item["answer"]


def score_split(items, rows, cal):
    """Thresholded results for one split."""
    accept = [r["decision"] == "answer" and cal.accepts(r["score"]) for r in rows]
    answered = [i for i, a in enumerate(accept) if a]
    wrong = sum(not correct_answer(items[i], rows[i]) for i in answered)
    supported = [i for i, x in enumerate(items) if x["kind"] == "supported"]
    out = {"items": len(items), "supported": len(supported), "answered": len(answered), "accepted_errors": wrong,
           "accepted_error": wrong / len(answered) if answered else None,
           "accepted_error_upper_95": K.clopper_pearson_upper(wrong, len(answered), cal.alpha),
           "coverage_supported": sum(accept[i] for i in supported) / len(supported),
           "accuracy_supported_no_threshold": sum(correct_answer(items[i], rows[i]) for i in supported) / len(supported),
           "abstention_by_kind": {}, "by_family": {}}
    for kind in INVALID:
        idx = [i for i, x in enumerate(items) if x["kind"] == kind]
        out["abstention_by_kind"][kind] = {"n": len(idx), "abstained": sum(not accept[i] for i in idx)}
    for fam in FAMILIES:
        idx = [i for i in supported if items[i]["family"] == fam]
        acc = [i for i in idx if accept[i]]
        out["by_family"][fam] = {"n": len(idx), "answered": len(acc),
                                    "errors": sum(not correct_answer(items[i], rows[i]) for i in acc),
                                    "correct_no_threshold": sum(correct_answer(items[i], rows[i]) for i in idx)}
    pre = [i for i, r in enumerate(rows) if r["decision"] == "answer"]
    out["reliability"] = K.reliability([cal.probability(rows[i]["score"]) for i in pre],
                                       [correct_answer(items[i], rows[i]) for i in pre])
    return out


def interpreter_accuracy(items, rows):
    sup = [(x, r) for x, r in zip(items, rows) if x["kind"] == "supported"]
    return sum(r["parse"]["outcome"] == "program" and r["parse"]["steps"] == x["program"] and r["parse"]["entity"] == x["entity"]
               for x, r in sup) / len(sup)


def supplied_accuracy(world, registry, items):
    sup = [x for x in items if x["kind"] == "supported"]
    ok = 0
    for x in sup:
        p = Program(tuple(x["program"]), world.types[world.relations[x["program"][0]].source].value(x["entity"]))
        ex = execute(p, registry, world.types, Budget(len(x["program"])))
        ok += ex.answer is not None and ex.answer.value == x["answer"]
    return ok / len(sup)


def run_unit(world, items, init, args) -> list[dict]:
    split = lambda s: [x for x in items if x["split"] == s]
    train, cal_items = split("train"), split("calibration")
    examples = [J.Example(x["text"], outputs_of(x), x["entity_position"]) for x in train]
    t0 = time.time()
    net, vocab, loss = J.train_interpreter(examples, tuple(world.relations), init, args.d, args.hidden, args.epochs, args.lr)
    interp_seconds = time.time() - t0
    interp_id = J.interpreter_identity(net, vocab)
    registries = {"typed_table": G.table_registry(world)}
    for system, family in MODULES.items():
        trained = M.train_family(world, family, init, 32, 64, 1500, 1e-2)
        reg = G.Registry(world.relations)
        for rid in sorted(world.relations):
            reg = reg.register(trained["modules"][rid])
        registries[system] = reg
    qa = train_answer_only(world, train, init, args.qa_epochs, args.lr, args.d, args.hidden)
    out = []
    for system in SYSTEMS:
        reg = registries.get(system)
        rows_cal = decisions(system, cal_items, world, (net, vocab), reg, qa)
        pre = [i for i, r in enumerate(rows_cal) if r["decision"] == "answer"]
        cal = K.fit([rows_cal[i]["score"] for i in pre], [correct_answer(cal_items[i], rows_cal[i]) for i in pre],
                    [cal_items[i]["id"] for i in pre], interp_id if system != "answer_only" else "answer_only",
                    reg.identity if reg else "none", J.POLICY_VERSION)
        res = {"system": system, "world": world.seed, "world_identity": world.identity, "init": init,
               "interpreter_identity": interp_id, "interpreter_train_loss": loss,
               "interpreter_train_seconds": round(interp_seconds, 2),
               "registry_identity": reg.identity if reg else None,
               "calibration": {k: (v if not (isinstance(v, float) and math.isinf(v)) else None)
                               for k, v in cal.__dict__.items()},
               "calibration_fit": score_split(cal_items, rows_cal, cal)}
        for s in ("test", "shift"):
            its = split(s)
            rows = decisions(system, its, world, (net, vocab), reg, qa)
            res[s] = score_split(its, rows, cal)
            if system != "answer_only":
                res[s]["interpreter_exact"] = interpreter_accuracy(its, rows)
            if s == "test" and reg is not None:
                res[s]["supplied_program_accuracy"] = supplied_accuracy(world, reg, its)
        out.append(res)
        t = res["test"]
        print(f"  world {world.seed} init {init} {system:18s} test: coverage {t['coverage_supported']:.3f} accepted error "
              f"{t['accepted_errors']}/{t['answered']} (upper {t['accepted_error_upper_95']:.4f}) no-threshold "
              f"{t['accuracy_supported_no_threshold']:.3f} interp {t.get('interpreter_exact', float('nan')):.3f} | shift: "
              f"coverage {res['shift']['coverage_supported']:.3f} error {res['shift']['accepted_errors']}/{res['shift']['answered']}",
              flush=True)
    return out


def run(args) -> dict:
    runs, worlds = [], {}
    for wseed in args.worlds:
        world = W.generate_world(wseed, n_countries=args.countries)
        items = make_items(world, wseed)
        worlds[str(wseed)] = {"identity": world.identity, "items": len(items),
                              "by_split_kind": {f"{s}/{k}": sum(x["split"] == s and x["kind"] == k for x in items)
                                                for s in ("train", "calibration", "test", "shift")
                                                for k in ("supported",) + INVALID}}
        for init in args.inits:
            runs += run_unit(world, items, init, args)
    return {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "environment": {"torch": torch.__version__, "python": platform.python_version(),
                            "platform": platform.platform()}, "worlds": worlds, "runs": runs}


CONFIRMATION = {"worlds": (400, 401, 402), "inits": (0, 1, 2, 3, 4)}


def evaluate(result: dict, worlds=CONFIRMATION["worlds"], inits=CONFIRMATION["inits"]) -> dict:
    """The frozen predictions of protocols/typed_routing_v1.md, from the recorded runs."""
    by = {(r["system"], r["world"], r["init"]): r for r in result["runs"]}
    missing = [(s, w, i) for s in SYSTEMS for w in worlds for i in inits if (s, w, i) not in by]
    if missing:
        return {"complete": False, "missing": missing[:10]}
    reliable = lambda ok: all(sum(ok(w, i) for i in inits) >= 4 for w in worlds)
    P = lambda w, i: by[("typed_specialists", w, i)]
    p = {
        "1_interpreter_competence": reliable(lambda w, i: P(w, i)["test"]["interpreter_exact"] >= 0.99),
        "2_selective_answering": reliable(lambda w, i: P(w, i)["test"]["accepted_error_upper_95"] <= 0.01
                                          and P(w, i)["test"]["coverage_supported"] >= 0.90),
        "3_invalid_questions_refused": all(
            sum(P(w, i)["test"]["abstention_by_kind"][k]["abstained"] for i in inits)
            >= 0.99 * sum(P(w, i)["test"]["abstention_by_kind"][k]["n"] for i in inits) for w in worlds for k in INVALID),
        "4_end_to_end_matches_supplied_program": reliable(
            lambda w, i: P(w, i)["test"]["accuracy_supported_no_threshold"] >= P(w, i)["test"]["supplied_program_accuracy"] - 0.01),
        "5_answer_only_fails_multi_step": all(
            sum(by[("answer_only", w, i)]["test"]["by_family"][f]["correct_no_threshold"] for f in ("length2", "length3"))
            < 0.5 * sum(by[("answer_only", w, i)]["test"]["by_family"][f]["n"] for f in ("length2", "length3"))
            for w in worlds for i in inits),
        "6_shift_accepted_error_bounded": reliable(
            lambda w, i: P(w, i)["shift"]["answered"] == 0 or P(w, i)["shift"]["accepted_error"] <= 0.05),
    }
    summary = {}
    for s in SYSTEMS:
        rs = [by[(s, w, i)] for w in worlds for i in inits]
        summary[s] = {split: {"coverage_min": min(r[split]["coverage_supported"] for r in rs),
                              "coverage_max": max(r[split]["coverage_supported"] for r in rs),
                              "accepted_errors_total": sum(r[split]["accepted_errors"] for r in rs),
                              "answered_total": sum(r[split]["answered"] for r in rs),
                              "accuracy_no_threshold_min": min(r[split]["accuracy_supported_no_threshold"] for r in rs)}
                      for split in ("test", "shift")}
    return {"complete": True, "systems": summary, "predictions": p}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--worlds", nargs="+", type=int, default=list(CONFIRMATION["worlds"]))
    p.add_argument("--inits", nargs="+", type=int, default=list(CONFIRMATION["inits"]))
    p.add_argument("--countries", type=int, default=48)
    p.add_argument("--d", type=int, default=48)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30, help="interpreter training epochs")
    p.add_argument("--qa-epochs", type=int, default=30, help="answer-only baseline training epochs")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("runs/titnn_routing"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    result = run(args)
    out = args.out / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    (out / "typed_questions.json").write_text(json.dumps(result, indent=2))
    if tuple(sorted(args.worlds)) == CONFIRMATION["worlds"] and tuple(sorted(args.inits)) == CONFIRMATION["inits"]:
        verdict = evaluate(result)
        (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
        print(json.dumps(verdict["predictions"], indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
