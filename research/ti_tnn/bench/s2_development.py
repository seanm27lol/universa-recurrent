"""S2 diagnosis, parts 2 and 3: instrumented development fits and the training-data interventions
(protocols/s2_diagnosis_brief.md). Development worlds and inits only; CPU only.

    CUDA_VISIBLE_DEVICES= nice -n 10 .venv-phase2/bin/python research/ti_tnn/bench/s2_development.py \
        --arms A --worlds 100 101 --inits 100 101 102

Each fit is the interpreter robustness study's `word_aug` recipe: word encoder, 1,500 updates of batch 64, the
same architecture, learning rate, entity dropout and novelty augmentation, and the study's own generator, audit,
decisions, certification and measurements (interpreter_robustness.py, imported, unchanged). Only the training set
differs between arms, by an addition of N_w examples per world, where N_w is the number of taught one-step
training questions:
- **A**: the original `word_aug` training set (no addition).
- **B** (exposure): every taught one-step training question once more, verbatim.
- **C** (context beyond exposure): the same taught questions as B, inside a longer inessential wrapper of real
  wrapper words (`WRAPPER_WORDS`, words of the Base frames, all in the training vocabulary). Each gets k filler words,
  where k is drawn from the empirical token-length distribution of the Base prefix phrases in the world's prefix
  training questions, so a C question is as long as the same question with one more Base phrase. The fillers go after
  the frame's opening words or before its closing punctuation (one half each); the noun phrase is untouched. The
  target stays the one relation.
- **D** (additional compositional supervision): N_w two-step training instances, sampled without replacement,
  each rendered in prefix style with one uniformly chosen position's Base phrase replaced by its taught phrase,
  with the full two-relation target. A gain here is not zero-shot S2 transfer: D trains on exactly the kind of
  question S2 tests (on different instances).

Every addition is built from training instances only; every target has at most two relations; no held-out or
novel word enters training; the word vocabulary is built from the ordinary training questions, as in the study, so
every arm has the same vocabulary. `audit_additions` checks all of this before any training, and the run stops if
any check fails.

**Instrumentation.** For S2, S2_base, T1 and T1_base questions it stores the decoder's full output distribution at
every step (full precision), twice: under greedy decoding (the decisions the policy used) and under teacher
forcing with the reference outputs (each step conditioned on the correct prefix; a diagnostic, never a decision).
The greedy traces come from the same batches as the recorded decisions and must reproduce them exactly.
Checkpoints are written under the run directory (git-ignored) and are never committed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import interpreter_robustness as S  # noqa: E402  (generator, audit, decisions, measurements: unchanged)
import s2_diagnosis as D2  # noqa: E402
from integrated import interpreters as I  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import routing as J  # noqa: E402

ARMS = ("A", "B", "C", "D")
WRAPPER_WORDS = ("what", "is", "name")  # the words of the Base frames (punctuation excluded)
TRACED = ("S2", "S2_base", "T1", "T1_base")
DEVELOPMENT = {"worlds": (100, 101), "inits": (100, 101, 102)}
POLICIES = ("none", "certified")  # all that unit_outcomes reads
SOURCES = ("interpreter_robustness.py", "s2_development.py", "s2_diagnosis.py", "../integrated/interpreters.py",
           "../integrated/routing.py")


# --- the additions ---------------------------------------------------------------------------------------------
def phrase_length_distribution(study: dict) -> dict[int, int]:
    """Token counts of the Base prefix phrases in the prefix-style Base training questions (each phrase occurrence
    counted once), recovered by exact re-rendering."""
    lengths = Counter()
    for x in study["items"]:
        if x["category"] == "train_base" and x["style"] == "prefix":
            for tpl in D2.base_phrases_of(x):
                lengths[len(tpl.replace("{x}", "").split())] += 1
    return dict(sorted(lengths.items()))


def additions(world: W.World, study: dict, arm: str) -> list[dict]:
    """The examples arm `arm` adds to the original word_aug training set (module docstring)."""
    if arm == "A":
        return []
    rng = random.Random(f"s2-development|{world.identity}|{arm}")
    taught = [x for x in study["items"] if x["category"] == "train_taught"]
    out = []
    if arm == "B":
        out = [{**x, "category": "add_B", "id": f"{x['id']}-B"} for x in taught]
    elif arm == "C":
        dist = phrase_length_distribution(study)
        ks, weights = list(dist), list(dist.values())
        for x in taught:
            frame = x["frame"]
            assert S.render(x["entity"], [S.TAUGHT[x["program"][0]]], frame) == x["text"]
            k = rng.choices(ks, weights)[0]
            filler = " ".join(rng.choice(WRAPPER_WORDS) for _ in range(k))
            head, tail = frame.split("{q}")
            wrapped = (f"{head}{filler} {{q}}{tail}" if rng.random() < 0.5 else f"{head}{{q}} {filler}{tail}")
            text = S.render(x["entity"], [S.TAUGHT[x["program"][0]]], wrapped)
            out.append({**x, "category": "add_C", "id": f"{x['id']}-C", "text": text, "frame": wrapped,
                        "filler_words": k, "entity_position": S.TQ.entity_position(text, x["entity"])})
    elif arm == "D":
        two = [(p, s) for p, s in study["train_instances"] if len(p) == 2]
        if len(taught) > len(two):
            raise ValueError(f"world {world.seed}: {len(taught)} additions requested, {len(two)} two-step instances")
        for n, (p, s) in enumerate(rng.sample(two, len(taught))):
            frame, phrases, j = rng.choice(S.BASE_FRAMES), S.base_phrases(p, "prefix", rng), rng.randrange(2)
            text = S.render(s, [S.TAUGHT[r] if i == j else phrases[i] for i, r in enumerate(p)], frame)
            out.append({"split": "train", "category": "add_D", "kind": "supported", "text": text, "program": list(p),
                        "entity": s, "expected": "answer", "answer": None, "style": "prefix", "frame": frame,
                        "taught_position": j, "outputs": list(p) + [J.END], "id": f"{world.seed}-D{n}",
                        "entity_position": S.TQ.entity_position(text, s)})
    else:
        raise ValueError(arm)
    return out


def audit_additions(world: W.World, study: dict, aug: list[dict], adds: dict[str, list[dict]]) -> dict:
    """Checks on every arm's additions, before any training."""
    ordinary = [x for x in study["items"] if x["split"] == "train"]
    vocab_words = {t for x in ordinary for t in x["text"].split()}
    train_inst = {(tuple(p), s) for p, s in study["train_instances"]}
    eval_texts = {x["text"] for x in study["items"] if x["split"] != "train"}
    n = len([x for x in ordinary if x["category"] == "train_taught"])
    forbidden = (S.LEXICON["held"] | S.LEXICON["novel"]) - S.LEXICON["base"] - S.LEXICON["taught"]
    rel_count = lambda x: sum(o not in (J.END, J.UNSUPPORTED, J.AMBIGUOUS) for o in x["outputs"])
    everything = [x for xs in adds.values() for x in xs]
    checks = {
        "original_audit_passes": all(S.audit(world, study, aug).values()),
        "matched_counts": all(len(adds[a]) == n for a in ("B", "C", "D")) and not adds["A"],
        "from_training_instances": all((tuple(x["program"]), x["entity"]) in train_inst for x in everything),
        "targets_at_most_two_relations": all(rel_count(x) <= 2 for x in everything),
        "targets_are_the_reference_program": all(x["outputs"] == list(x["program"]) + [J.END] for x in everything),
        "no_held_or_novel_words": all(not set(x["text"].split()) & forbidden for x in everything),
        "every_added_word_in_the_training_vocabulary": all(set(x["text"].split()) <= vocab_words for x in everything),
        "no_added_question_is_an_evaluation_question": all(x["text"] not in eval_texts for x in everything),
        "b_repeats_the_taught_questions": sorted(x["text"] for x in adds["B"]) == sorted(
            x["text"] for x in ordinary if x["category"] == "train_taught"),
        "c_wraps_the_same_questions_with_one_relation_targets": sorted(
            (x["entity"], tuple(x["program"])) for x in adds["C"]) == sorted(
            (x["entity"], tuple(x["program"])) for x in adds["B"]) and all(len(x["program"]) == 1 for x in adds["C"]),
        "c_filler_is_wrapper_words_only": all(
            set(x["text"].split()) - set(S.render(x["entity"], [S.TAUGHT[x["program"][0]]], "{q}").split())
            <= set(WRAPPER_WORDS) | set("?.") for x in adds["C"]),
        "d_is_two_step_with_one_taught_phrase": all(
            len(x["program"]) == 2 and len(set(x["text"].split()) & (S.LEXICON["taught"] - S.LEXICON["base"])) > 0
            for x in adds["D"]),
        "d_instances_distinct": len({(tuple(x["program"]), x["entity"]) for x in adds["D"]}) == len(adds["D"]),
        "all_encodable": all(S._encodable(x["text"]) for x in everything),
    }
    return checks


# --- instrumented decoding -------------------------------------------------------------------------------------
@torch.no_grad()
def traced_interpret(net, vocab, items: list[dict], keep: set[int], batch: int = 256):
    """I.interpret's batches, plus full output distributions for the items at indices `keep`: greedy (as decided)
    and teacher-forced with the reference outputs. Returns (parses, traces)."""
    parses = I.interpret(net, vocab, [x["text"] for x in items], batch)
    index = {o: k for k, o in enumerate(net.outputs)}
    toks = [I.normalize(x["text"]) for x in items]  # every item is encodable (audited)
    traces = {}
    for s in range(0, len(items), batch):
        rows = list(range(s, min(s + batch, len(items))))
        if not keep.intersection(rows):
            continue
        b = I.make_batch(net.mode, [toks[i] for i in rows], vocab)
        entity_logits, greedy_logits = net(b)
        targets = torch.full((len(rows), J.MAX_STEPS + 1), index[J.END], dtype=torch.long)
        for r, i in enumerate(rows):
            for t, o in enumerate(reference_outputs(items[i])):
                targets[r, t] = index[o]
        _, teacher_logits = net(b, targets)
        entity_p, greedy_p, teacher_p = (F.softmax(entity_logits, -1), F.softmax(greedy_logits, -1),
                                         F.softmax(teacher_logits, -1))
        for r, i in enumerate(rows):
            if i in keep:
                x = items[i]
                traces[i] = {"greedy": greedy_p[r].tolist(), "teacher": teacher_p[r].tolist(),
                             "entity_p_reference": float(entity_p[r, x["entity_position"]]),
                             "entity_p_chosen": float(entity_p[r].max())}
    return parses, traces


def reference_outputs(x: dict) -> list[str]:
    if x.get("program"):
        return list(x["program"]) + [J.END]
    return list(x.get("outputs") or [J.UNSUPPORTED])


def check_greedy_trace(parse, trace, outputs) -> bool:
    """The traced greedy distribution reproduces the recorded parse exactly: the same argmax chain to its stop
    symbol and bit-identical chosen probabilities."""
    steps, probs = [], []
    for dist in trace["greedy"]:
        k = max(range(len(dist)), key=lambda c: dist[c])
        probs.append(dist[k])
        o = outputs[k]
        if o in (J.END, J.UNSUPPORTED, J.AMBIGUOUS):
            break
        steps.append(o)
    return tuple(steps) == parse.steps and tuple(probs) == parse.step_probabilities


# --- one fit ---------------------------------------------------------------------------------------------------
def fit_unit(world, study, aug, adds, arm, init, args, out: Path) -> dict:
    items = study["items"]
    table = G.table_registry(world)
    ordinary = [x for x in items if x["split"] == "train"]
    vocab_texts = [x["text"] for x in ordinary]
    vocab_words = {w for t in vocab_texts for w in t.split()}
    held = [x for x in items if x["split"] != "train"]
    cal = [x for x in held if x["split"] == "calibration"]
    evals = [x for x in held if x["split"] == "test"]
    train_items = ordinary + aug + adds
    examples = [J.Example(x["text"], tuple(x["outputs"]), x["entity_position"]) for x in train_items]
    t0 = time.time()
    net, vocab, loss = I.train("word", examples, vocab_texts, tuple(world.relations), init, args.updates)
    seconds = time.time() - t0
    keep = {k for k, x in enumerate(held) if x["category"] in TRACED}
    parses, traces = traced_interpret(net, vocab, held, keep)
    mismatches = [held[k]["id"] for k in keep if not check_greedy_trace(parses[k], traces[k], net.outputs)]
    rows = S.decide(parses, held, world, table)
    cal_rows, ev_rows = rows[:len(cal)], rows[len(cal):]
    res = {"arm": arm, "world": world.seed, "init": init, "train_seconds": round(seconds, 1), "train_loss": loss,
           "training_examples": len(examples), "added_examples": len(adds), "greedy_trace_mismatches": mismatches,
           "policies": {}}
    for policy in POLICIES:
        cert = S.certify(cal, cal_rows, policy, vocab_words) if policy == "certified" else None
        threshold = cert["threshold"] if cert else None
        res["policies"][policy] = {"certificate": {k: v for k, v in (cert or {}).items() if k != "records"} or None,
                                   "categories": S.measure(evals, ev_rows, policy, threshold, vocab_words)}
    res["outcomes"] = S.unit_outcomes(res)
    tag = f"{arm}_w{world.seed}_i{init}"
    (out / "checkpoints").mkdir(exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "vocab_words": vocab.words, "outputs": net.outputs, "mode": "word",
                "arm": arm, "world": world.seed, "init": init}, out / "checkpoints" / f"{tag}.pt")
    with gzip.open(out / "records" / f"{tag}.json.gz", "wt") as f:
        json.dump({"world": world.seed, "init": init, "arm": arm, "ids": [x["id"] for x in held],
                   "records": S.compact_records(held, rows)}, f)
    order = sorted(keep)
    with gzip.open(out / "traces" / f"{tag}.json.gz", "wt") as f:
        json.dump({"world": world.seed, "init": init, "arm": arm, "outputs": list(net.outputs),
                   "ids": [held[k]["id"] for k in order], "greedy": [traces[k]["greedy"] for k in order],
                   "teacher": [traces[k]["teacher"] for k in order],
                   "entity_p_reference": [traces[k]["entity_p_reference"] for k in order],
                   "entity_p_chosen": [traces[k]["entity_p_chosen"] for k in order]}, f)
    c, n = res["policies"]["certified"]["categories"], res["policies"]["none"]["categories"]
    print(f"  {tag} ({seconds:.0f}s, +{len(adds)}) cert {res['policies']['certified']['certificate']['threshold']} | "
          f"S2 exact {n['S2']['exact_parse']:.3f} cov {c['S2']['coverage']:.3f}/{c['S2']['errors']} | "
          f"S2_base exact {n['S2_base']['exact_parse']:.3f} | T1 {n['T1']['exact_parse']:.3f} | "
          f"S3 {c['S3']['coverage']:.3f}/{c['S3']['errors']} | S4a/b abst {c['S4a_primary']['abstention']:.3f}/"
          f"{c['S4b_primary']['abstention']:.3f} | joint {res['outcomes']['joint_novelty']} | "
          f"trace mismatches {len(mismatches)}", flush=True)
    return res


def provenance() -> dict:
    here = Path(__file__).resolve().parent
    files = {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in SOURCES}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here.parent, capture_output=True,
                               text=True).stdout.strip()
    except OSError:
        commit, dirty = None, None
    return {"commit": commit, "research_tree_dirty": bool(dirty), "sha256": files}


def run(args, out: Path) -> dict:
    for sub in ("records", "traces", "checkpoints"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    result = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "provenance": provenance(),
              "environment": {"torch": torch.__version__, "python": platform.python_version(),
                              "platform": platform.platform(), "threads": torch.get_num_threads()},
              "worlds": {}, "units": []}
    for wseed in args.worlds:
        world = W.generate_world(wseed, n_countries=args.countries)
        study = S.make_study(world, wseed)
        aug = S.augment(study, wseed)
        adds = {a: additions(world, study, a) for a in ARMS}
        checks = audit_additions(world, study, aug, adds)
        if not all(checks.values()):
            raise SystemExit(f"world {wseed}: addition audit failed: {[k for k, v in checks.items() if not v]}")
        result["worlds"][str(wseed)] = {
            "identity": world.identity, "audit": checks, "phrase_length_distribution": phrase_length_distribution(study),
            "added": {a: len(adds[a]) for a in ARMS},
            "c_filler_words": dict(sorted(Counter(x["filler_words"] for x in adds["C"]).items())),
            "c_lengths": dict(sorted(Counter(len(x["text"].split()) for x in adds["C"]).items())),
            "d_taught_position": dict(sorted(Counter(x["taught_position"] for x in adds["D"]).items())),
            "addition_sha256": {a: hashlib.sha256(json.dumps([[x["text"], x["outputs"]] for x in adds[a]]).encode())
                                .hexdigest() for a in ARMS}}
        print(f"world {wseed}: audits passed; added {result['worlds'][str(wseed)]['added']}", flush=True)
        for arm in args.arms:
            for init in args.inits:
                result["units"].append(fit_unit(world, study, aug, adds[arm], arm, init, args, out))
                (out / "s2_development.json").write_text(json.dumps(result))  # checkpoint after every fit
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", nargs="+", choices=ARMS, required=True)
    p.add_argument("--worlds", nargs="+", type=int, default=list(DEVELOPMENT["worlds"]))
    p.add_argument("--inits", nargs="+", type=int, default=list(DEVELOPMENT["inits"]))
    p.add_argument("--countries", type=int, default=96)
    p.add_argument("--updates", type=int, default=1500)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("runs/titnn_s2_diagnosis/development"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    out = args.out / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    result = run(args, out)
    (out / "s2_development.json").write_text(json.dumps(result))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
