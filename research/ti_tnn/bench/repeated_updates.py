"""Repeated updates under a bounded replay allowance (protocols/repeated_updates_v1.md).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 10 .venv-phase2/bin/python \\
        research/ti_tnn/bench/repeated_updates.py --worlds 100 101 --inits 100 101 --out runs/titnn_repeated_dev

**Example.** World 100 has 276 facts. Its first scheduled request moves `country_7`'s capital. The arm
`shared/signature/1/8/fixed` fine-tunes the deployed shared network for 300 steps on that fact plus at most
⌊276/8⌋ = 34 unchanged facts, `capital_of` and `largest_city_of` first (both Country → City). The candidate is
screened (every fact of the target world right) and, if it passes, fully checked by PR 8's gates. The first
accepted proposal of up to 8 is activated through a `Deployment`; the next request starts from it. If none is
accepted, the request is refused and the deployed world keeps the old capital.

**What is fixed per world, for every arm and init:** the twelve requests (`schedule`), drawn once against the
original world: six distinct countries' capitals and six distinct airports' operators. Refusals change the
deployed history, never the requests.

**What varies between proposals:** only the replay draw. Training is full-batch Adam from the deployed network on
a canonically ordered training set, with no dropout, so a candidate is a deterministic function of (parent,
training set, stopping rule). A proposal whose training set repeats an earlier one for the same request is a
duplicate: recorded, not trained or checked, and it still uses one of the 8 attempts.

**Export** (weight-free; see the protocol's "Export and independent verification"): `run.json`, and per unit
`unit.json` and `proposals.jsonl.gz`, one line per proposal. `repeated_updates_verify.py` rebuilds everything
from them.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from integrated import modules as M  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import snapshots as S  # noqa: E402
from integrated import update_checks as U  # noqa: E402
from integrated import updates as D  # noqa: E402
from integrated.contracts import ModuleVersion, content_identity  # noqa: E402

GENERATOR_VERSION = "ti_tnn_repeated_updates_v1"
PER_KIND = 6
KINDS = {"capital_of": "capital_moves", "operator_of": "operator_changes"}
N_PROPOSALS = 8
LR, FIXED_STEPS, EARLY_EVERY = 3e-3, 300, 5
BETAS = ("1/16", "1/8", "1/4", "1/2")
POLICIES = ("random", "relation_first", "signature", "check_guided")
STOPPING = ("fixed", "early")
SPECIALIST_BETAS = ("0", "1/8", "1/4", "1/2")
AUDIT_MODULUS = 20  # a screened-out proposal is fully checked when its audit hash is 0 modulo 20 (5%), and the
#                    first screened-out proposal of each update kind in each chain always is
PROPOSER = "replay-study"
CONFIRMATION = {"worlds": (900, 901, 902), "inits": (0, 1, 2, 3, 4)}
DEVELOPMENT_WORLDS = (100, 101)
SOURCES = ("repeated_updates.py", "../integrated/modules.py", "../integrated/update_checks.py",
           "../integrated/costs.py", "../integrated/updates.py", "../integrated/snapshots.py",
           "../integrated/registry.py", "../integrated/relations.py", "../integrated/contracts.py")


def arms() -> list[dict]:
    """Every arm, in run order: 32 shared arms, two shared references, four specialist arms."""
    out = [{"name": f"shared/{p}/{b}/{s}", "family": "shared", "policy": p, "beta": b, "stopping": s}
           for p in POLICIES for b in BETAS for s in STOPPING]
    out.append({"name": "shared/none", "family": "shared", "policy": "none", "beta": "0", "stopping": "fixed"})
    out.append({"name": "shared/full", "family": "shared", "policy": "full", "beta": "1", "stopping": "fixed"})
    out += [{"name": f"specialists/{b}", "family": "specialists", "policy": "own_relation", "beta": b,
             "stopping": "fixed"} for b in SPECIALIST_BETAS]
    return out


# --- what is asked: the schedule, the canonical fact keys, the replay draw ----------------------------------------
def schedule(world) -> list[dict]:
    """The immutable intended requests: six distinct countries' capitals, each to another of its cities, and six
    distinct airports' operators, each to another operator; destinations drawn once against `world`, in one
    shuffled order."""
    rng = random.Random(f"{GENERATOR_VERSION}|schedule|{world.identity}")
    cities = {}
    for city, country in sorted(world.facts["country_of"].items()):
        cities.setdefault(country, []).append(city)
    requests = []
    for x in rng.sample(sorted(c for c in cities if len(cities[c]) > 1), PER_KIND):
        old = world.facts["capital_of"][x]
        requests.append(("capital_of", x, rng.choice([c for c in sorted(cities[x]) if c != old])))
    operators = sorted(world.types["Operator"].values)
    for x in rng.sample(sorted(world.types["Airport"].values), PER_KIND):
        old = world.facts["operator_of"][x]
        requests.append(("operator_of", x, rng.choice([o for o in operators if o != old])))
    rng.shuffle(requests)
    return [{"index": k, "relation": r, "source": x, "target": t, "kind": KINDS[r]}
            for k, (r, x, t) in enumerate(requests)]


def fact_keys(world) -> list[tuple[str, str]]:
    """The canonical fact list: every (relation, source), relations and sources sorted. Indices into it name facts."""
    return [(r, s) for r in sorted(world.relations) for s in sorted(world.facts[r])]


def allowance(beta: str, n: int) -> int | None:
    """⌊β·n⌋ for a fraction β; None for full rehearsal."""
    return None if beta == "1" else int(Fraction(beta) * n)


def replay_draw(arm: dict, keys: list, edit: int, prior_errors: set, rng: random.Random, world) -> list[int]:
    """The replay set (indices of unchanged facts) for one proposal. A priority pool larger than the allowance is
    sampled uniformly; a smaller one is taken whole and filled uniformly from the rest."""
    rel = keys[edit][0]
    if arm["family"] == "specialists":
        pool = [i for i, k in enumerate(keys) if k[0] == rel and i != edit]
        b = allowance(arm["beta"], len(pool) + 1)
        return sorted(rng.sample(pool, min(b, len(pool))))
    pool = [i for i in range(len(keys)) if i != edit]
    if arm["policy"] == "none":
        return []
    if arm["policy"] == "full":
        return pool
    b = allowance(arm["beta"], len(keys))
    sig = (world.relations[rel].source, world.relations[rel].target)
    priority = {"random": [],
                "relation_first": [i for i in pool if keys[i][0] == rel],
                "signature": [i for i in pool if (world.relations[keys[i][0]].source,
                                                  world.relations[keys[i][0]].target) == sig],
                "check_guided": sorted(prior_errors)}[arm["policy"]]
    if len(priority) >= b:
        return sorted(rng.sample(priority, b))
    taken = set(priority)
    return sorted(priority + rng.sample([i for i in pool if i not in taken], b - len(priority)))


def replay_rng(world, init: int, arm: str, update: int, proposal: int) -> random.Random:
    return random.Random(f"{GENERATOR_VERSION}|replay|{world.identity}|{init}|{arm}|{update}|{proposal}")


def audit_selected(world, init: int, arm: str, update: int, proposal: int) -> bool:
    """The frozen 5% rule for fully checking a screened-out proposal; it does not depend on any outcome."""
    h = hashlib.sha256(f"{GENERATOR_VERSION}|audit|{world.identity}|{init}|{arm}|{update}|{proposal}".encode())
    return int(h.hexdigest(), 16) % AUDIT_MODULUS == 0


def with_facts(world, facts) -> W.World:
    return W.World(world.name, world.types, world.relations, facts, world.seed)


# --- training and behaviour ----------------------------------------------------------------------------------------
def _shared_tensors(vocab, world, keys, idx, facts):
    rels = vocab.relations
    x = torch.tensor([vocab.global_index(world.relations[keys[i][0]].source, keys[i][1]) for i in idx])
    rel = torch.tensor([rels.index(keys[i][0]) for i in idx])
    y = torch.tensor([vocab.global_index(world.relations[keys[i][0]].target, facts[keys[i][0]][keys[i][1]])
                      for i in idx])
    return x, rel, y, M.type_mask(vocab, [world.relations[keys[i][0]].target for i in idx])


def table(family: str, nets, vocab, world, keys) -> list[str]:
    """The candidate's answer to every canonical fact key (its atomic table), in key order."""
    out = []
    with torch.no_grad():
        if family == "shared":
            x, rel, _, mask = _shared_tensors(vocab, world, keys, range(len(keys)), world.facts)
            pred = nets(x, rel).masked_fill(~mask, float("-inf")).argmax(-1).tolist()
            return [vocab.entities[p] for p in pred]
        for r in sorted(world.relations):
            spec = world.relations[r]
            src = [s for rr, s in keys if rr == r]
            logits = nets[r](torch.tensor([vocab.local[spec.source][s] for s in src]))
            out += [world.types[spec.target].values[int(i)] for i in logits.argmax(-1)]
    return out


def train(family: str, parent, vocab, world, keys, train_idx: list[int], facts, edit: int, stopping: str):
    """Fine-tune a copy of the parent on the training set (full batch, canonical order). Returns (network, steps)."""
    torch.manual_seed(0)  # nothing here draws randomness; fixed anyway
    rel = keys[edit][0]
    if family == "shared":
        net = copy.deepcopy(parent).train()
        x, r, y, mask = _shared_tensors(vocab, world, keys, train_idx, facts)
        loss_fn = lambda: M.masked_ce(net(x, r), y, mask)  # noqa: E731
        edit_x, edit_r, edit_y, edit_mask = _shared_tensors(vocab, world, keys, [edit], facts)
        learned = lambda: bool(net(edit_x, edit_r).masked_fill(~edit_mask, float("-inf")).argmax(-1)[0] == edit_y[0])  # noqa: E731
    else:
        spec = world.relations[rel]
        net = copy.deepcopy(parent[rel]).train()
        x = torch.tensor([vocab.local[spec.source][keys[i][1]] for i in train_idx])
        y = torch.tensor([vocab.local[spec.target][facts[rel][keys[i][1]]] for i in train_idx])
        loss_fn = lambda: F.cross_entropy(net(x), y)  # noqa: E731
        learned = None
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    steps = 0
    for _ in range(FIXED_STEPS):
        loss = loss_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
        if stopping == "early" and steps % EARLY_EVERY == 0:
            with torch.no_grad():
                if learned():
                    break
    return net.eval(), steps


def shared_registry(world, vocab, net):
    mods = {}
    for ri, r in enumerate(vocab.relations):
        version = ModuleVersion(f"{r}_shared_dense", r, "shared_dense", M.module_identity("shared_dense", r, net), "float32")
        mods[r] = M.LearnedModule(version, M.SharedForward(net, ri), vocab, world.relations[r].source,
                                  world.relations[r].target, False)
    return _registry(world, mods)


def specialist_registry(world, vocab, nets):
    mods = {}
    for r in sorted(world.relations):
        version = ModuleVersion(f"{r}_specialists", r, "specialists", M.module_identity("specialists", r, nets[r]),
                                "float32")
        mods[r] = M.LearnedModule(version, nets[r], vocab, world.relations[r].source, world.relations[r].target, True)
    return _registry(world, mods)


def _registry(world, modules):
    reg = G.Registry(world.relations)
    for r in sorted(world.relations):
        reg = reg.register(modules[r])
    return reg


# --- one chain: one arm, one unit, twelve requests -----------------------------------------------------------------
def chain(arm: dict, world, init: int, base, vocab, keys, requests, checker, out) -> dict:
    family = arm["family"]
    authority = D.Authority({PROPOSER})
    nets = base if family == "shared" else dict(base)
    make = (lambda n: shared_registry(world, vocab, n)) if family == "shared" else \
        (lambda n: specialist_registry(world, vocab, n))
    deployment = D.Deployment(make(nets), authority, checker)
    facts = {r: dict(t) for r, t in world.facts.items()}
    scope = tuple(sorted(world.relations))
    accepted_total, audited_kinds = 0, set()
    for req in requests:
        edit = keys.index((req["relation"], req["source"]))
        deployed = with_facts(world, facts)
        target_facts = {r: dict(t) for r, t in facts.items()}
        target_facts[req["relation"]][req["source"]] = req["target"]
        target = with_facts(world, target_facts)
        seen, prior_errors = {}, set()
        for n in range(N_PROPOSALS):
            t0 = time.time()
            replay = replay_draw(arm, keys, edit, prior_errors, replay_rng(world, init, arm["name"], req["index"], n),
                                 world)
            training = sorted([edit] + replay)
            line = {"arm": arm["name"], "update": req["index"], "proposal": n, "parent": deployment.active.digest,
                    "deployed_world": deployed.identity, "target_world": target.identity, "train": training,
                    "replay_size": len(replay), "duplicate_of": seen.get(tuple(training)), "steps": 0, "errors": None,
                    "table_digest": None, "screen": None, "audited": False, "gates": None, "candidate": None,
                    "accepted": False, "activation": None}
            if line["duplicate_of"] is not None:
                line["seconds"] = round(time.time() - t0, 4)
                out.write(json.dumps(line) + "\n")
                continue
            seen[tuple(training)] = n
            cand_nets, steps = train(family, nets, vocab, world, keys, training, target_facts, edit, arm["stopping"])
            if family == "specialists":
                cand_nets = {**nets, req["relation"]: cand_nets}
            answers = table(family, cand_nets, vocab, world, keys)
            errors = [[i, a] for i, a in enumerate(answers) if a != target_facts[keys[i][0]][keys[i][1]]]
            prior_errors |= {i for i, _ in errors if i != edit}
            screen = not errors
            line.update(steps=steps, errors=errors, table_digest=content_identity(answers), screen=screen)
            due = screen or audit_selected(world, init, arm["name"], req["index"], n) or req["kind"] not in audited_kinds
            if not screen and due:
                audited_kinds.add(req["kind"])
            if due:
                cand = make(cand_nets)
                record = D.UpdateRecord(deployment.active.digest, S.digest_of(cand),
                                        scope if family == "shared" else (req["relation"],),
                                        ((req["relation"], req["source"], req["target"]),), target.identity,
                                        checker.version, None, None, PROPOSER)
                result = checker.check(deployment.active, cand, record, deployed, target)
                line.update(audited=not screen, gates=result.outcomes(), candidate=result.candidate_digest)
                if screen and result.passed():
                    act = deployment.activate(record, result)
                    line["activation"] = {"activated": act["activated"], "reasons": act["reasons"],
                                          "active_after": deployment.active.digest}
                    line["accepted"] = bool(act["activated"])
            line["seconds"] = round(time.time() - t0, 4)
            out.write(json.dumps(line) + "\n")
            if line["accepted"]:
                nets, facts = cand_nets, target_facts
                accepted_total += 1
                break
    return {"accepted": accepted_total}


# --- one unit ------------------------------------------------------------------------------------------------------
def run_unit(world, init: int, args, out_dir: Path, checker) -> dict:
    torch.manual_seed(init)
    keys, requests, vocab = fact_keys(world), schedule(world), M.Vocabulary(world)
    truth = [world.facts[r][s] for r, s in keys]
    bases, unit = {}, {"world": world.seed, "init": init, "keys_identity": content_identity(keys), "n_facts": len(keys),
                       "schedule_identity": content_identity(requests), "bases": {}, "chains": {}}
    for family, trained in (("shared", "shared_dense"), ("specialists", "specialists")):
        b = M.train_family(world, trained, init, args.d, args.hidden, args.steps, args.lr)
        nets = b["net"] if family == "shared" else {r: m.forward for r, m in b["modules"].items()}
        answers = table(family, nets, vocab, world, keys)
        errors = [[i, a] for i, a in enumerate(answers) if a != truth[i]]
        reg = shared_registry(world, vocab, nets) if family == "shared" else specialist_registry(world, vocab, nets)
        unit["bases"][family] = {"exact": not errors, "errors": errors, "table_digest": content_identity(answers),
                                 "snapshot": S.digest_of(reg)}
        bases[family] = nets
    out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_dir / "proposals.jsonl.gz", "wt") as out:
        for arm in arms():
            if not unit["bases"][arm["family"]]["exact"]:
                unit["chains"][arm["name"]] = {"ran": False, "reason": "inexact base"}
                continue
            t0 = time.time()
            res = chain(arm, world, init, bases[arm["family"]], vocab, keys, requests, checker, out)
            unit["chains"][arm["name"]] = {"ran": True, **res, "seconds": round(time.time() - t0, 2)}
            print(f"  w{world.seed} i{init} {arm['name']:34s} accepted {res['accepted']:2d}/12  "
                  f"{unit['chains'][arm['name']]['seconds']:6.1f}s", flush=True)
    (out_dir / "unit.json").write_text(json.dumps(unit, indent=1))
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
    return {"commit": commit, "research_tree_dirty": dirty, "sha256": sha, "torch": torch.__version__,
            "python": sys.version.split()[0], "threads": torch.get_num_threads()}


def run(args, out: Path) -> Path:
    if any(w in CONFIRMATION["worlds"] for w in args.worlds) and not args.confirmation:
        raise SystemExit("worlds 900-902 are the confirmation worlds; they run only from the frozen protocol")
    torch.set_num_threads(args.threads)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = out / stamp
    run_dir.mkdir(parents=True)
    checker = U.Checker()
    config = {k: v for k, v in vars(args).items() if k not in ("out",)}
    meta = {"generator": GENERATOR_VERSION, "config": config, "provenance": provenance(),
            "constants": {"PER_KIND": PER_KIND, "N_PROPOSALS": N_PROPOSALS, "LR": LR, "FIXED_STEPS": FIXED_STEPS,
                          "EARLY_EVERY": EARLY_EVERY, "BETAS": list(BETAS), "POLICIES": list(POLICIES),
                          "STOPPING": list(STOPPING), "SPECIALIST_BETAS": list(SPECIALIST_BETAS),
                          "AUDIT_MODULUS": AUDIT_MODULUS},
            "arms": [a["name"] for a in arms()], "worlds": {}, "units": []}
    for w in args.worlds:
        world = W.generate_world(w, n_countries=args.countries)
        meta["worlds"][str(w)] = {"identity": world.identity, "schedule": schedule(world),
                                  "keys_identity": content_identity(fact_keys(world))}
        for i in args.inits:
            path = f"units/w{w}_i{i}"
            unit = run_unit(world, i, args, run_dir / path, checker)
            meta["units"].append({"world": w, "init": i, "path": path,
                                  "bases_exact": {f: b["exact"] for f, b in unit["bases"].items()}})
    meta["finished"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1))
    return run_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--worlds", type=int, nargs="+", default=list(DEVELOPMENT_WORLDS))
    p.add_argument("--inits", type=int, nargs="+", default=[100, 101])
    p.add_argument("--countries", type=int, default=24)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--confirmation", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path, default=Path("runs/titnn_repeated_dev"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(run(args, args.out))


if __name__ == "__main__":
    main()
