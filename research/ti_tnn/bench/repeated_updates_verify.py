"""Independent verifier for a repeated-updates run (protocols/repeated_updates_v1.md: "Export and independent
verification").

    .venv-phase2/bin/python research/ti_tnn/bench/repeated_updates_verify.py <run dir> [--development] [--reproduce]

**Example.** Proposal 2 of request 5 in chain `shared/check_guided/1/8/fixed` of unit w100 i100 says it
trained on 35 facts and got 3 facts wrong. The verifier:
- **re-derives that training set:** the edit, plus the unchanged facts proposals 0 and 1 got wrong, filled
  uniformly by the declared draw;
- **rebuilds the candidate's table:** the target world's facts with exactly those 3 answers substituted. Its
  digest must equal the exported one;
- **checks the screen:** a candidate with any error fails it;
- **checks the parent:** this proposal must start from the last accepted candidate of the chain.

**It trusts no stored summary.** It does not call the producer's training, selection or gate code. It uses:
- `repeated_updates`'s generator only (`schedule`, `fact_keys`, `arms` and the constants). As in the other
  verifiers, the generator defines what was asked;
- `relations.generate_world`, to rebuild the worlds;
- its own implementation of the replay draws and the audit rule, written from the protocol.

**What it re-derives, for every proposal of every chain:**
- the deployed and target worlds: a refused request never lands;
- the parent: the last accepted candidate;
- the training set, and whether it duplicates an earlier one for the same request;
- every error entry's validity, and the candidate's table digest;
- the screen;
- the step count's consistency with the stopping rule;
- which proposals the full checker had to judge;
- the agreement of the screen with the behavioural gates;
- acceptance: screen, every mandatory gate, activation.

From these it rebuilds every rate, cost, prediction and the PR 9 decision, and the descriptive best observed
ordinary arms, with and without `check_guided`, which do not enter the decision.

**What remains the producer's record:**
- **that an exported table is the trained network's behaviour.** `--reproduce` retrains a frozen,
  outcome-independent sample of complete chains (one per arm and world) and requires identical records. That
  is sampled reproduction with the producer's code, not independent reproduction of every model;
- **the full checker's gate outcomes.** The verifier checks their consistency with the screen and the
  acceptance rule.

**A coherent forgery** (an altered error list with its digest recomputed) re-derives everywhere except in
reproduced chains.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import repeated_updates as RU  # noqa: E402  (the generator: schedule, fact keys, arms, constants)
from integrated import relations as W  # noqa: E402
from integrated.contracts import content_identity  # noqa: E402

MANDATORY = ("record_context", "identity_and_scope", "types_and_laws", "intended_behavior", "unaffected_behavior",
             "trace_and_continuation", "calibration_and_resources")
BEHAVIOURAL = ("intended_behavior", "unaffected_behavior")
# The producer files as hashed into every run (repeated_updates.SOURCES), fixed in the freezing commit of
# protocols/repeated_updates_v1.md revision 3. The run records its commit; a commit cannot name itself.
FROZEN_SOURCES = {
    "repeated_updates.py": "3676fae7958aa8f71ee019e894109a43c0b5d541debd12d2924f8c13f3fdde57",
    "../integrated/modules.py": "fe89f275281579b666f12c5d452d5911c6448b68ae4bf6cc74c19e38647622c7",
    "../integrated/update_checks.py": "1ea1e5fd77ea0662187c8fe192e00668aa37ea9f525b286b6f54cccf951a116b",
    "../integrated/costs.py": "1a7f4aa8649412e94a015cc9e79eb7105281037e976f888e1ca0ca5d58e2e442",
    "../integrated/updates.py": "bda75f302fd09a1066e416a9f085507cd4276dac93b58a8ea776c0e0db50a14e",
    "../integrated/snapshots.py": "30b6eed36651eabded80363cd437d889c6e69b30936b13ccee98f5646cd0323f",
    "../integrated/registry.py": "85350d0284bae12a55c5b748c5ca5d61acde8623d91ceeed29108525702674e1",
    "../integrated/relations.py": "742331d263587da665b743584f9afda402286219fec9b947838ca3900d296b5a",
    "../integrated/contracts.py": "7ccf766d3885157f5e53ee419544c1f0f5d49a70f575b22875404b6b452dfdaa",
}
FROZEN_DESIGN = {"worlds": [900, 901, 902], "inits": [0, 1, 2, 3, 4], "countries": 24, "d": 32, "hidden": 64,
                 "steps": 1500, "lr": 1e-2, "threads": 4, "confirmation": True}
LINE_FIELDS = ("arm", "update", "proposal", "parent", "deployed_world", "target_world", "train", "replay_size",
               "duplicate_of", "steps", "errors", "table_digest", "screen", "audited", "gates", "candidate",
               "accepted", "activation", "seconds")
# decision constants (protocol revision 2, unchanged in revision 3)
REFERENCE_MIN, TARGET, WORLDS_NEEDED = Fraction(95, 100), Fraction(90, 100), 2
H1_BETA = "1/4"  # revision 3 (pilot-informed): where the 47-fact signature pool fits; revision 2 had 1/8


class VerificationError(AssertionError):
    pass


def _lines(path: Path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


# --- the replay draw and the audit rule, from the protocol ----------------------------------------------------------
def allowed(beta: str, n: int) -> int | None:
    return None if beta == "1" else int(Fraction(beta) * n)


def draw(arm: dict, keys: list, edit: int, prior: set, seed: str, world) -> list[int]:
    rng = random.Random(seed)
    rel = keys[edit][0]
    if arm["family"] == "specialists":
        own = [i for i, k in enumerate(keys) if k[0] == rel and i != edit]
        return sorted(rng.sample(own, min(allowed(arm["beta"], len(own) + 1), len(own))))
    others = [i for i in range(len(keys)) if i != edit]
    if arm["policy"] == "none":
        return []
    if arm["policy"] == "full":
        return others
    b = allowed(arm["beta"], len(keys))
    if arm["policy"] == "random":
        first = []
    elif arm["policy"] == "relation_first":
        first = [i for i in others if keys[i][0] == rel]
    elif arm["policy"] == "signature":
        sig = lambda r: (world.relations[r].source, world.relations[r].target)  # noqa: E731
        first = [i for i in others if sig(keys[i][0]) == sig(rel)]
    elif arm["policy"] == "check_guided":
        first = sorted(prior)
    else:
        raise VerificationError(f"unknown policy {arm['policy']!r}")
    if len(first) >= b:
        return sorted(rng.sample(first, b))
    taken = set(first)
    return sorted(first + rng.sample([i for i in others if i not in taken], b - len(first)))


def audit(world_identity: str, init: int, arm: str, update: int, proposal: int) -> bool:
    h = hashlib.sha256(f"{RU.GENERATOR_VERSION}|audit|{world_identity}|{init}|{arm}|{update}|{proposal}".encode())
    return int(h.hexdigest(), 16) % RU.AUDIT_MODULUS == 0


def reproduction_init(world: int, arm: str, inits: list[int]) -> int:
    """The frozen, outcome-independent choice of the chain reproduced for (world, arm)."""
    h = hashlib.sha256(f"{RU.GENERATOR_VERSION}|reproduce|{world}|{arm}".encode())
    return sorted(inits)[int(h.hexdigest(), 16) % len(inits)]


def inventory_problems(run: dict, development: bool) -> list[str]:
    problems, cfg = [], run["config"]
    pairs = [(u["world"], u["init"]) for u in run["units"]]
    if not run["worlds"] or not pairs:
        problems.append("the run has no worlds or no units")
    if len(set(pairs)) != len(pairs):
        problems.append("the run repeats a unit")
    if sorted(pairs) != sorted((int(w), i) for w in run["worlds"] for i in cfg["inits"]):
        problems.append("the unit inventory is not the declared worlds x inits")
    if sorted(int(w) for w in run["worlds"]) != sorted(cfg["worlds"]):
        problems.append("the worlds run are not the declared worlds")
    if run.get("arms") != [a["name"] for a in RU.arms()]:
        problems.append("the arms are not the declared arms")
    constants = {"PER_KIND": RU.PER_KIND, "N_PROPOSALS": RU.N_PROPOSALS, "LR": RU.LR, "FIXED_STEPS": RU.FIXED_STEPS,
                 "EARLY_EVERY": RU.EARLY_EVERY, "BETAS": list(RU.BETAS), "POLICIES": list(RU.POLICIES),
                 "STOPPING": list(RU.STOPPING), "SPECIALIST_BETAS": list(RU.SPECIALIST_BETAS),
                 "AUDIT_MODULUS": RU.AUDIT_MODULUS}
    if run.get("constants") != constants or run.get("generator") != RU.GENERATOR_VERSION:
        problems.append("the run's constants are not the generator's")
    if not development:
        for key, value in FROZEN_DESIGN.items():
            if cfg.get(key) != value:
                problems.append(f"config {key} is {cfg.get(key)!r}, frozen {value!r}")
    return problems


# --- one chain ------------------------------------------------------------------------------------------------------
def verify_chain(arm: dict, lines: list, world, init: int, keys: list, requests: list, base_snapshot: str,
                 fail) -> dict:
    """Re-derive one chain; returns its per-request outcomes and costs."""
    facts = {r: dict(t) for r, t in world.facts.items()}
    parent = base_snapshot
    domain = {r: set(world.types[world.relations[r].target].values) for r in world.relations}
    out, pos, c, audited_kinds = [], 0, Counter(), set()
    for req in requests:
        edit = keys.index((req["relation"], req["source"]))
        deployed = RU.with_facts(world, facts)
        target_facts = {r: dict(t) for r, t in facts.items()}
        target_facts[req["relation"]][req["source"]] = req["target"]
        target = RU.with_facts(world, target_facts)
        truth = [target_facts[r][s] for r, s in keys]
        seen, prior, distinct, accepted_at = {}, set(), set(), None
        for n in range(RU.N_PROPOSALS):
            tag = f"{arm['name']} request {req['index']} proposal {n}"
            if pos >= len(lines):
                fail(f"{tag}: missing")
                return {"requests": out, "costs": c, "broken": True}
            line = lines[pos]
            pos += 1
            if set(line) != set(LINE_FIELDS):
                fail(f"{tag}: its fields are not the declared fields")
            seed = f"{RU.GENERATOR_VERSION}|replay|{world.identity}|{init}|{arm['name']}|{req['index']}|{n}"
            replay = draw(arm, keys, edit, prior, seed, world)
            training = sorted([edit] + replay)
            want = {"arm": arm["name"], "update": req["index"], "proposal": n, "parent": parent,
                    "deployed_world": deployed.identity, "target_world": target.identity, "train": training,
                    "replay_size": len(replay), "duplicate_of": seen.get(tuple(training))}
            got = {k: line.get(k) for k in want}
            if got != want:
                fail(f"{tag}: {[k for k in want if got[k] != want[k]]} do not re-derive")
            distinct |= set(replay)
            c["attempts"] += 1
            if want["duplicate_of"] is not None:
                c["duplicates"] += 1
                if any(line.get(k) not in (None, False, 0) for k in ("steps", "errors", "table_digest", "screen",
                                                                      "audited", "gates", "candidate", "accepted",
                                                                      "activation")):
                    fail(f"{tag}: a duplicate carries a training or check result")
                continue
            seen[tuple(training)] = n
            errors = line.get("errors")
            if not isinstance(errors, list) or any(not (isinstance(e, list) and len(e) == 2) for e in errors):
                fail(f"{tag}: malformed errors")
                return {"requests": out, "costs": c, "broken": True}
            idx = [e[0] for e in errors]
            if len(set(idx)) != len(idx) or any(not (isinstance(i, int) and 0 <= i < len(keys)) for i in idx) or any(
                    a == truth[i] or a not in domain[keys[i][0]] for i, a in errors):
                fail(f"{tag}: an error entry is repeated, out of range, not wrong, or outside its type")
                return {"requests": out, "costs": c, "broken": True}
            table = list(truth)
            for i, a in errors:
                table[i] = a
            if content_identity(table) != line["table_digest"]:
                fail(f"{tag}: its table digest does not match its errors")
            screen = not errors
            if line["screen"] is not screen:
                fail(f"{tag}: the screen does not re-derive")
            steps = line["steps"]
            stopped_early = arm["stopping"] == "early" and steps < RU.FIXED_STEPS
            if arm["stopping"] == "fixed" and steps != RU.FIXED_STEPS or not (
                    isinstance(steps, int) and 1 <= steps <= RU.FIXED_STEPS) or (
                    stopped_early and (steps % RU.EARLY_EVERY or edit in idx)):
                fail(f"{tag}: its step count does not fit the stopping rule")
            prior |= {i for i in idx if i != edit}
            c["trained"] += 1
            c["steps"] += steps
            c["example_steps"] += len(training) * steps
            must_check = screen or audit(world.identity, init, arm["name"], req["index"], n) or \
                req["kind"] not in audited_kinds
            if not screen and must_check:
                audited_kinds.add(req["kind"])
            gates = line["gates"]
            if must_check != (gates is not None) or line["audited"] is not (gates is not None and not screen):
                fail(f"{tag}: the full check was {'missing' if must_check else 'not due'}")
            if gates is not None:
                c["checker_runs"] += 1
                c["audits"] += not screen
                if set(gates) != set(MANDATORY) or any(v not in ("pass", "fail", "insufficient") for v in gates.values()):
                    fail(f"{tag}: its gate outcomes are not the mandatory gates'")
                elif screen != all(gates[g] == "pass" for g in BEHAVIOURAL):
                    c["screen_checker_disagreements"] += 1
            accept = bool(screen and gates is not None and set(gates) == set(MANDATORY)
                          and all(v == "pass" for v in gates.values()))
            act = line["activation"]
            if accept:
                if not (isinstance(act, dict) and act.get("activated") is True and act.get("reasons") == []
                        and act.get("active_after") == line["candidate"] and line["candidate"]):
                    fail(f"{tag}: an acceptable candidate was not activated as itself")
                    accept = False
            elif act is not None:
                fail(f"{tag}: an activation attempt for a candidate that does not qualify")
            if line["accepted"] is not accept:
                fail(f"{tag}: its acceptance does not re-derive")
            if accept:
                accepted_at, parent, facts = n, line["candidate"], target_facts
                break
        out.append({"index": req["index"], "kind": req["kind"], "accepted_at": accepted_at,
                    "distinct_replay": len(distinct)})
        c["distinct_replay"] += len(distinct)
    if pos != len(lines):
        fail(f"{arm['name']}: {len(lines) - pos} proposals beyond the declared chain")
    return {"requests": out, "costs": c, "broken": False}


def verify_unit(run_dir: Path, info: dict, world, run: dict, fail) -> dict:
    d = run_dir / info["path"]
    unit = json.loads((d / "unit.json").read_text())
    keys, requests = RU.fact_keys(world), RU.schedule(world)
    if (unit["world"], unit["init"]) != (info["world"], info["init"]):
        fail("unit.json is not the unit the run lists")
    if unit["keys_identity"] != content_identity(keys) or unit["n_facts"] != len(keys):
        fail("the canonical fact list does not re-derive")
    if unit["schedule_identity"] != content_identity(requests):
        fail("the schedule does not re-derive")
    truth = [world.facts[r][s] for r, s in keys]
    for family, b in unit["bases"].items():
        table = list(truth)
        for i, a in b["errors"]:
            table[i] = a
        if content_identity(table) != b["table_digest"] or b["exact"] is not (not b["errors"]):
            fail(f"{family} base: its exactness does not re-derive")
    if info.get("bases_exact") != {f: b["exact"] for f, b in unit["bases"].items()}:
        fail("run.json's base exactness is not the unit's")
    by_arm = defaultdict(list)
    for line in _lines(d / "proposals.jsonl.gz"):
        by_arm[line.get("arm")].append(line)
    names = [a["name"] for a in RU.arms()]
    if set(by_arm) - set(names):
        fail(f"proposals for undeclared arms: {sorted(set(by_arm) - set(names))}")
    chains = {}
    for arm in RU.arms():
        ran = unit["bases"][arm["family"]]["exact"]
        if unit["chains"].get(arm["name"], {}).get("ran") is not ran:
            fail(f"{arm['name']}: ran although its base is inexact, or did not run although it is exact")
        if not ran:
            if by_arm.get(arm["name"]):
                fail(f"{arm['name']}: proposals for a chain that must not run")
            chains[arm["name"]] = None
            continue
        res = verify_chain(arm, by_arm.get(arm["name"], []), world, info["init"], keys, requests,
                           unit["bases"][arm["family"]]["snapshot"],
                           lambda m, a=arm: fail(f"{a['name']}: {m}" if not m.startswith(a["name"]) else m))
        chains[arm["name"]] = res
        if unit["chains"][arm["name"]].get("accepted") != sum(r["accepted_at"] is not None for r in res["requests"]):
            fail(f"{arm['name']}: unit.json's accepted count does not re-derive")
    return {"unit": f"w{info['world']} i{info['init']}", "bases_exact": {f: b["exact"] for f, b in unit["bases"].items()},
            "chains": chains}


# --- aggregates, predictions and the decision -----------------------------------------------------------------------
def aggregate(units: list, worlds: list, inits: list) -> dict:
    out = {}
    for w in worlds:
        us = [u for u in units if u["unit"].startswith(f"w{w} ")]
        per_arm = {}
        for arm in RU.arms():
            ran = [u["chains"][arm["name"]] for u in us if u["chains"][arm["name"]] is not None]
            reqs = [r for ch in ran for r in ch["requests"]]
            costs = sum((ch["costs"] for ch in ran), Counter())
            per_arm[arm["name"]] = {
                "chains": len(ran), "requests": len(reqs), "accepted": sum(r["accepted_at"] is not None for r in reqs),
                "by_kind": {k: [sum(r["accepted_at"] is not None for r in reqs if r["kind"] == k),
                                sum(r["kind"] == k for r in reqs)] for k in RU.KINDS.values()},
                "accepted_at": [sum(r["accepted_at"] == n for r in reqs) for n in range(RU.N_PROPOSALS)],
                "costs": dict(sorted(costs.items()))}
        out[str(w)] = {"shared_bases_exact": sum(u["bases_exact"]["shared"] for u in us),
                       "specialist_bases_exact": sum(u["bases_exact"]["specialists"] for u in us),
                       "units": len(us), "arms": per_arm}
    return out


def rate(a: dict) -> Fraction | None:
    return Fraction(a["accepted"], a["requests"]) if a["requests"] else None


def decision(agg: dict, inits: list) -> dict:
    """The PR 9 decision per allowance (protocol revision 2): a world qualifies when every shared base is exact and
    full rehearsal accepts >= 95% of its requests. Gap at beta: >= 2 qualifying worlds where the best ordinary arm
    accepts < 90%. No target gap: >= 2 qualifying worlds where it accepts >= 90%. Otherwise inconclusive."""
    ordinary = [a for a in RU.arms() if a["family"] == "shared" and a["policy"] in RU.POLICIES]
    out = {}
    for beta in RU.BETAS:
        per_world = {}
        for w, wa in agg.items():
            full = wa["arms"].get("shared/full", {"requests": 0})
            qualifies = wa["shared_bases_exact"] == len(inits) and full["requests"] > 0 and rate(full) >= REFERENCE_MIN
            rates = [rate(wa["arms"][a["name"]]) for a in ordinary if a["beta"] == beta and a["name"] in wa["arms"]]
            best = max((x for x in rates if x is not None), default=None)
            per_world[w] = {"qualifies": bool(qualifies), "best_ordinary": None if best is None else float(best),
                            "below_target": None if best is None else best < TARGET}
        q = [v for v in per_world.values() if v["qualifies"] and v["below_target"] is not None]
        gap = sum(v["below_target"] for v in q) >= WORLDS_NEEDED
        no_gap = sum(not v["below_target"] for v in q) >= WORLDS_NEEDED
        out[beta] = {"outcome": "gap" if gap else "no target gap" if no_gap else "inconclusive", "worlds": per_world}
    return out


def best_observed(agg: dict, inits: list) -> dict:
    """Descriptive only (protocol revision 3): for each beta and world, the best observed ordinary arm, over all four
    policies and over the three that do not use earlier attempts' error lists to choose replay (excluding
    `check_guided`), with tied winners and each winner's cumulative replay exposure and compute. The same chains,
    attempt limit and denominators as the decision; `qualifies` is the decision's own world rule. It does not
    enter the PR 9 decision, which is `decision`'s, over the complete baseline set."""
    subsets = {"all_ordinary": RU.POLICIES, "excluding_check_guided": tuple(p for p in RU.POLICIES if p != "check_guided")}
    out = {}
    for beta in RU.BETAS:
        per_world = {}
        for w, wa in agg.items():
            full = wa["arms"].get("shared/full", {"requests": 0})
            row = {"qualifies": bool(wa["shared_bases_exact"] == len(inits) and full["requests"] > 0
                                     and rate(full) >= REFERENCE_MIN)}
            for label, policies in subsets.items():
                arms = {a["name"]: wa["arms"][a["name"]] for a in RU.arms()
                        if a["family"] == "shared" and a["policy"] in policies and a["beta"] == beta
                        and a["name"] in wa["arms"] and wa["arms"][a["name"]]["requests"]}
                best = max((rate(a) for a in arms.values()), default=None)
                winners = sorted(n for n, a in arms.items() if rate(a) == best)
                row[label] = {"rate": None if best is None else float(best),
                              "accepted": None if best is None else arms[winners[0]]["accepted"],
                              "requests": None if best is None else arms[winners[0]]["requests"],
                              "winners": {n: {"distinct_replay_per_request": round(
                                  arms[n]["costs"].get("distinct_replay", 0) / arms[n]["requests"], 2),
                                  **{k: arms[n]["costs"].get(k, 0) for k in ("trained", "steps", "example_steps",
                                                                            "checker_runs")}} for n in winners}}
            per_world[w] = row
        out[beta] = per_world
    return out


def predictions(agg: dict) -> dict:
    """R1, R2, H1 and H2 (protocol revision 3), each over every world's per-world aggregate. None when an arm the
    prediction reads has no requests (it is not evaluable). H1 is evaluated at `H1_BETA`."""
    def acc(w, name, kind=None):
        a = agg[w]["arms"].get(name)
        if a is None or not a["requests"]:
            return None
        return Fraction(*a["by_kind"][kind]) if kind and a["by_kind"][kind][1] else \
            None if kind else Fraction(a["accepted"], a["requests"])

    def steps(w, name):
        return agg[w]["arms"][name]["costs"].get("steps", 0)

    def every(f):
        values = [f(w) for w in sorted(agg)]
        return None if not values or any(v is None for v in values) else all(values)

    def diff(x, y, at_least):
        return None if x is None or y is None else x - y >= at_least
    return {
        "R1_full_rehearsal": every(lambda w: None if acc(w, "shared/full") is None
                                   else acc(w, "shared/full") >= Fraction(95, 100)),
        "R2_no_replay": every(lambda w: None if acc(w, "shared/none") is None
                              else acc(w, "shared/none") <= Fraction(1, 2)),
        "H1_signature_over_relation_first_on_capital_moves": every(lambda w: diff(
            acc(w, f"shared/signature/{H1_BETA}/fixed", "capital_moves"),
            acc(w, f"shared/relation_first/{H1_BETA}/fixed", "capital_moves"), Fraction(1, 5))),
        "H2_early_stopping": every(lambda w: None if None in (acc(w, "shared/random/1/16/early"),
                                                               acc(w, "shared/random/1/16/fixed")) else
                                   diff(acc(w, "shared/random/1/16/early"), acc(w, "shared/random/1/16/fixed"),
                                        Fraction(1, 5))
                                   and steps(w, "shared/random/1/16/early") * 10 <= steps(w, "shared/random/1/16/fixed")),
    }


# --- the run --------------------------------------------------------------------------------------------------------
def verify(run_dir: Path, development: bool = False, reproduce: bool = False) -> dict:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    problems = []
    if not development:
        prov = run["provenance"]
        if FROZEN_SOURCES is None:
            problems.append("no frozen sources: this protocol is not frozen yet")
        elif prov["sha256"] != FROZEN_SOURCES or prov["research_tree_dirty"] is not False:
            problems.append("the run is not bound to the frozen source (producer-file hashes or a modified tree)")
    problems += inventory_problems(run, development)
    if problems:
        raise VerificationError(f"{len(problems)} problems; first: {problems[:5]}")
    cfg = run["config"]
    units, worlds = [], {}
    for wseed, winfo in run["worlds"].items():
        world = W.generate_world(int(wseed), n_countries=cfg["countries"])
        if winfo["identity"] != world.identity or winfo["schedule"] != RU.schedule(world) or \
                winfo["keys_identity"] != content_identity(RU.fact_keys(world)):
            problems.append(f"world {wseed}: its identity, schedule or fact list does not regenerate")
        worlds[wseed] = world
        for info in sorted((u for u in run["units"] if str(u["world"]) == wseed), key=lambda u: u["init"]):
            units.append(verify_unit(run_dir, info, world, run,
                                     lambda m, t=f"w{wseed} i{info['init']}": problems.append(f"{t}: {m}")))
    if problems:
        raise VerificationError(f"{len(problems)} problems; first: {problems[:5]}")
    agg = aggregate(units, sorted(int(w) for w in worlds), cfg["inits"])
    construction = {"screen_checker_agreement": all(
        ch is None or ch["costs"].get("screen_checker_disagreements", 0) == 0 for u in units for ch in u["chains"].values())}
    out = {"verified": True, "commit": run["provenance"]["commit"], "worlds": agg, "construction": construction,
           "predictions": predictions(agg), "decision": decision(agg, cfg["inits"]),
           "best_observed": best_observed(agg, cfg["inits"])}
    if reproduce:
        out["reproduction"] = reproduce_sample(run_dir, run, worlds)
    return out


def reproduce_sample(run_dir: Path, run: dict, worlds: dict) -> dict:
    """Sampled reproduction: retrain the frozen sample of complete chains with the producer's code and require
    identical records (every field but wall time)."""
    import io
    import torch
    from integrated import modules as M
    from integrated import update_checks as U
    cfg = run["config"]
    torch.set_num_threads(cfg["threads"])
    results = []
    for wseed, world in worlds.items():
        keys, requests, vocab = RU.fact_keys(world), RU.schedule(world), M.Vocabulary(world)
        for arm in RU.arms():
            init = reproduction_init(int(wseed), arm["name"], cfg["inits"])
            path = next(u["path"] for u in run["units"] if (str(u["world"]), u["init"]) == (wseed, init))
            recorded = [x for x in _lines(run_dir / path / "proposals.jsonl.gz") if x["arm"] == arm["name"]]
            unit = json.loads((run_dir / path / "unit.json").read_text())
            if not unit["bases"][arm["family"]]["exact"]:
                results.append({"world": wseed, "init": init, "arm": arm["name"], "identical": None})
                continue
            torch.manual_seed(init)
            trained = "shared_dense" if arm["family"] == "shared" else "specialists"
            b = M.train_family(world, trained, init, cfg["d"], cfg["hidden"], cfg["steps"], cfg["lr"])
            nets = b["net"] if arm["family"] == "shared" else {r: m.forward for r, m in b["modules"].items()}
            buf = io.StringIO()
            RU.chain(arm, world, init, nets, vocab, keys, requests, U.Checker(), buf)
            again = [json.loads(x) for x in buf.getvalue().splitlines()]
            strip = lambda xs: [{k: v for k, v in x.items() if k != "seconds"} for x in xs]  # noqa: E731
            results.append({"world": wseed, "init": init, "arm": arm["name"], "identical": strip(again) == strip(recorded)})
    return {"chains": len(results), "identical": sum(r["identical"] is True for r in results),
            "all_identical": all(r["identical"] is not False for r in results), "results": results}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--development", action="store_true")
    p.add_argument("--reproduce", action="store_true")
    p.add_argument("--out", type=Path, help="write the complete output here as JSON")
    a = p.parse_args(argv)
    out = verify(a.run_dir, a.development, a.reproduce)
    if a.out:
        a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    keys = ("verified", "commit", "construction", "predictions", "decision", "best_observed")
    print(json.dumps({k: out[k] for k in keys} | ({"reproduction": {k: out["reproduction"][k] for k in (
        "chains", "identical", "all_identical")}} if "reproduction" in out else {}), indent=1))


if __name__ == "__main__":
    main()
