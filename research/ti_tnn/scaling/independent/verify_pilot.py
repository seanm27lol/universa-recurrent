#!/usr/bin/env python3
"""Independent reconstruction of the TI-TNN C4/T4 learning pilot's verdict, written apart from the producers.

    usage: nice -n 19 python3 verify_pilot.py [--bundle DIR] [--repo REPO] [--json OUT]
                                              [--local [--runs-root RUNS] [--representability] [--redecode]]

It has two parts, reported separately, because they establish different things.

**Public arithmetic reconstruction** (always). It reads only the evidence bundle, plus the checkout's pins, protocol
and git history; no runs root, database, checkpoint or cached state. It checks:
- each evaluation run's eligibility, derived from its complete reconstruction inventory: every question once, every
  entry true. A single false entry rejects the run even if the producer's summary still says `passed`; that
  disagreement is itself a failed check;
- the producer's aggregate flags and totals, recomputed from the records: answered, abstained, EX-correct, time to
  SQL submission and generation cost per EX-correct submission;
- the record bindings: configuration, snapshot and scored-SQL hashes, and fresh states equal to the anchors and
  identical across the arms;
- the counts, gates, paired differences, rule, per-question histogram, database table and sign-flip, from each
  record's *recorded* EX score;
- the training records: the recipe, each log's updates, the seeded example stream, and the evaluated checkpoint
  being each run's recorded final;
- the ledger;
- agreement with the producer's `analysis.json`. Runs are matched by their path relative to the runs root, never by
  absolute path, so a moved bundle or another checkout reconstructs the same way.

**Local checks** (`--local`). They need the runs root: the raw release, the SQLite databases, the checkpoint files
and the cached states. They check:
- the 64 are reconstructed from the raw release;
- every final SQL is re-executed with this directory's own EX rule, so the recorded scores the public part counts
  are re-derived;
- each final checkpoint re-hashes to its name and record, with its model and optimizer digests recomputed;
- the bundle's copies equal the run originals.

Own code (no scaling import) throughout, except where flagged in the output:
- `--representability`: scaling.pilot.representability (the grammar's gold-program trace), CPU tokenization;
- `--redecode`: scaling.encode / infer_controller / train re-decode the 64 x 4 requests on CPU from the final
  checkpoints and the anchored cache entries (informational: the GPU decoded at TF32, the CPU at float32).

History: the reviewer wrote this verifier and produced `../evidence/pilot_v1/independent/pilot_verdict*.json` with
the version at commit 0c420ab, against an earlier 84-file bundle snapshot. The public/local split, the
inventory-derived eligibility, the totals, the serving accounting and the run references were added on 2026-11-02 by
the producer-side session at the user's request (see `README.md`).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_baselines as VB  # noqa: E402  (the reviewer's own helpers)

ARMS, SEEDS, UPDATES, CKPT_EVERY, LEAD = ("T4", "c4_params"), (100, 101), 1500, 250, 7
HARNESS = "research/ti_tnn/scaling/"
RUN_KINDS = ("baselines", "pilot", "profiles", "smoke", "timing")  # the runs root's top-level run directories
INVENTORY_KEYS = ("same_actions", "same_final", "fresh_states_equal_anchor")
FEASIBILITY_RUN = "20261030T175558Z"  # the passing feasibility run whose projection the pilot ran under


def tensor_digest(state: dict) -> str:
    import torch
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().to("cpu").contiguous().reshape(-1)
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def optimizer_digest(state: dict) -> str:
    """Re-implemented from its documented definition: param groups as canonical JSON, then each parameter's state
    entries in (index, key) order, tensors by their digest."""
    import torch
    h = hashlib.sha256(json.dumps(state["param_groups"], sort_keys=True, default=str).encode())
    for index in sorted(state["state"]):
        for key in sorted(state["state"][index]):
            v = state["state"][index][key]
            h.update(json.dumps([index, key]).encode())
            h.update(tensor_digest({"v": v}).encode() if torch.is_tensor(v) else json.dumps(v).encode())
    return h.hexdigest()


def stream(n: int, update: int, batch: int, seed: int) -> list:
    import torch
    out, perms = [], {}
    for pos in range(update * batch, (update + 1) * batch):
        e, k = divmod(pos, n)
        if e not in perms:
            g = torch.Generator().manual_seed(int(hashlib.sha256(f"{seed}:epoch:{e}".encode()).hexdigest()[:15], 16))
            perms[e] = torch.randperm(n, generator=g).tolist()
        out.append(perms[e][k])
    return out


def sign_flip(values: list) -> dict:
    observed = sum(values)
    nz = [v for v in values if v]
    total = extreme = 0
    for signs in itertools.product((1, -1), repeat=len(nz)):
        total += 1
        extreme += abs(sum(s * v for s, v in zip(signs, nz))) >= abs(observed)
    return {"observed": observed, "blocks": len(values), "nonzero_blocks": len(nz), "assignments": total,
            "p_two_sided": extreme / total, "smallest_attainable_p": 2 / total if nz else 1.0}


def jl(path: Path) -> list:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def run_ref(path) -> str:
    """A run's path relative to the runs root: a relative path as is; an absolute one cut before its last component
    that names a top-level run kind."""
    p = PurePosixPath(str(path))
    if not p.is_absolute():
        return str(p)
    for i in range(len(p.parts) - 1, -1, -1):
        if p.parts[i] in RUN_KINDS:
            return str(PurePosixPath(*p.parts[i:]))
    raise ValueError(f"{path}: names no run kind")


def inventory_failures(summ: dict, ids: list) -> list:
    """What an evaluation's complete reconstruction inventory shows (empty: eligible), and every place the producer's
    aggregate flags disagree with it."""
    out = []
    arms = summ.get("arms") or {}
    if sorted(arms) != sorted(ARMS):
        out.append(f"arms {sorted(arms)}")
    for arm in ARMS:
        o = arms.get(arm) or {}
        inv = o.get("reconstruction") or []
        if [r.get("task_id") for r in inv] != ids:
            out.append(f"{arm}: inventory does not list the {len(ids)} questions once each, in order")
        bad = [(r.get("task_id"), k) for r in inv for k in INVENTORY_KEYS if r.get(k) is not True]
        if bad:
            out.append(f"{arm}: {len(bad)} false inventory entries, first {bad[:3]}")
        mine = {"reconstruction_all_equal": bool(inv) and all(r.get("same_actions") is True and r.get("same_final") is True for r in inv),
                "fresh_states_all_equal_anchor": bool(inv) and all(r.get("fresh_states_equal_anchor") is True for r in inv)}
        for k, v in mine.items():
            if o.get(k) is not v:
                out.append(f"{arm}: producer {k} = {o.get(k)!r}, inventory gives {v}")
        if o.get("record_errors"):
            out.append(f"{arm}: record errors")
    if summ.get("states_differ_across_arms"):
        out.append("producer: states differ across the arms")
    return out


def totals(recs: list) -> dict:
    """Own totals of one run's records. `elapsed_seconds` is the time to SQL submission; the scorer's execution of
    the submitted query (`score.predicted.elapsed_seconds`) is outside it and reported separately."""
    n = len(recs)
    correct = sum(r["score"]["correct"] for r in recs)
    gen = sum(r["elapsed_seconds"] for r in recs)
    exe = sum(((r["score"].get("predicted") or {}).get("elapsed_seconds") or 0.0) for r in recs)
    return {"requested": n, "answered": sum(r["final"]["kind"] == "sql" for r in recs),
            "abstained": sum(r["final"]["kind"] == "abstain" for r in recs), "correct": correct,
            "elapsed_seconds_total": gen, "cost_seconds_per_correct": gen / correct if correct else None,
            "time_to_submission_mean": gen / n if n else None,
            "submitted_query_execution_seconds": exe,
            "submitted_query_timeouts": sum((r["score"].get("predicted") or {}).get("status") == "timeout" for r in recs),
            "generation_plus_execution_per_correct": (gen + exe) / correct if correct else None}


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def compare_totals(rep, mine: dict, recorded: dict, where: str) -> None:
    for k in ("requested", "answered", "abstained", "correct"):
        rep.check(mine[k] == (recorded or {}).get(k), f"{where}: {k} recomputed = producer summary", f"{mine[k]} vs {(recorded or {}).get(k)}")
    for k in ("elapsed_seconds_total", "cost_seconds_per_correct"):
        rep.check(_close(mine[k], (recorded or {}).get(k)), f"{where}: {k} recomputed = producer summary", f"{mine[k]} vs {(recorded or {}).get(k)}")


# --- the public arithmetic reconstruction ----------------------------------------------------------------------------------
def public_reconstruction(bundle: Path, repo: Path) -> tuple:
    rep = VB.Report("pilot public")
    out = {"reads": "the evidence bundle; the checkout's pins, protocol and git history. No runs root, database, "
                    "checkpoint or cached state.", "producer_trace_used": []}
    sums = [x.split("  ", 1) for x in (bundle / "SHA256SUMS").read_text().splitlines() if x.strip()]
    bad = [p for h, p in sums if not (bundle / p).is_file() or VB.sha256_path(bundle / p) != h]
    listed = {p for _h, p in sums}
    unlisted = sorted(str(x.relative_to(bundle)) for x in bundle.rglob("*") if x.is_file() and x.name != "SHA256SUMS"
                      and str(x.relative_to(bundle)) not in listed)
    rep.check(not bad and not unlisted, "bundle: every file matches SHA256SUMS, and every file is listed", f"{bad[:3]} {unlisted[:3]}")
    out["bundle"] = {"files": len(sums), "sha256sums_sha256": VB.sha256_path(bundle / "SHA256SUMS")}

    # --- the frozen protocol and the code identities ---------------------------------------------------------------
    pin = json.loads((repo / HARNESS / "pins/pilot_protocol_v1.json").read_text())
    proto_here = VB.sha256_path(repo / HARNESS / "docs/PROTOCOL_pilot.md")
    proto_at_freeze = VB.sha256_bytes(VB.git_show(repo, pin["frozen_at"], HARNESS + "docs/PROTOCOL_pilot.md"))
    ev_dirs = sorted((bundle / "eval").glob("s*_*"))
    attempt_mans = sorted((bundle / "train").glob("*/attempts/*/manifest.json"))
    commits = {json.loads((d / "manifest.json").read_text())["git"]["commit"] for d in ev_dirs}
    commits |= {json.loads(m.read_text())["git"]["commit"] for m in attempt_mans}
    rep.check(len(commits) == 1, "every training and evaluation run at one commit", f"{commits}")
    commit = sorted(commits)[0]
    proto_at_run = VB.sha256_bytes(VB.git_show(repo, commit, HARNESS + "docs/PROTOCOL_pilot.md"))
    rep.check(pin["sha256"] == proto_here == proto_at_freeze == proto_at_run,
              "protocol: pin = checkout file = file at frozen_at = file at the run commit")
    out["code"] = {"run_commit": commit, "protocol_sha256": pin["sha256"], "frozen_at": pin["frozen_at"]}
    snaps = json.loads(VB.git_show(repo, commit, HARNESS + "pins/snapshots_v1.json"))["databases"]

    # --- anchors (own parse: clean runs under the frozen protocol) ------------------------------------------------
    anchors = {}
    for entries in sorted((bundle / "encode").glob("*/entries.jsonl")):
        m = json.loads((entries.parent / "manifest.json").read_text())
        if m["git"]["research_tree_dirty"] or (m["config"].get("protocol") or {}).get("sha256") != pin["sha256"]:
            continue
        for x in entries.read_text().split("\n")[:-1]:
            e = json.loads(x)
            if e["task_id"] in anchors and anchors[e["task_id"]]["states_sha256"] != e["states_sha256"]:
                rep.check(False, f"conflicting anchors for {e['task_id']}")
            anchors.setdefault(e["task_id"], e)

    # --- evaluations: eligibility from the complete reconstruction inventory ---------------------------------------
    all_runs, chosen, recorded = [], {}, {}
    for d in ev_dirs:
        man = json.loads((d / "manifest.json").read_text())
        summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        seed, ids = man["config"]["seed"], man["config"]["ids"]
        problems = inventory_failures(summ, ids)
        if not (man["git"]["research_tree_dirty"] is False and man["config"]["protocol"]["sha256"] == pin["sha256"]):
            problems.append("dirty tree or another protocol")
        recs = {arm: jl(d / arm / "records.jsonl") for arm in ARMS}
        for arm in ARMS:
            rr = recs[arm]
            if [r["task_id"] for r in rr] != ids:
                problems.append(f"{arm}: records are not the requested questions once each, in order")
            arm_cid = VB.sha256_bytes(VB.canonical({"run": man["config_identity"], "arm": arm,
                                                     "model_digest": man["config"]["checkpoints"][arm]["model_digest"]}).encode())
            for r in rr:
                pred = r["final"]["sql"] if r["final"]["kind"] == "sql" else None
                if r["score"].get("scored_sql_sha256") != VB.sha256_bytes((pred or "").encode()) or \
                        r["snapshot_sha256"] != snaps[r["db_id"]]["sha256"] or r["config_identity"] != arm_cid:
                    problems.append(f"{arm} {r['task_id']}: score/snapshot/config binding")
                if r.get("states_sha256") != anchors.get(r["task_id"], {}).get("states_sha256"):
                    problems.append(f"{arm} {r['task_id']}: fresh states differ from the anchor")
                if VB.nonfinite(r):
                    problems.append(f"{arm} {r['task_id']}: non-finite number")
            mine = totals(rr)
            compare_totals(rep, mine, ((summ.get("arms") or {}).get(arm) or {}).get("summary"), f"{d.name} {arm}")
            recorded[(d.name, arm)] = (rr, mine)
        a_states = {r["task_id"]: r.get("states_sha256") for r in recs["T4"]}
        cross = [r["task_id"] for r in recs["c4_params"] if a_states.get(r["task_id"]) != r.get("states_sha256")]
        if cross:
            problems.append(f"states differ across the arms: {cross[:3]}")
        eligible = not problems and summ.get("passed") is True
        rep.check((summ.get("passed") is True) == (not problems),
                  f"{d.name}: the producer's passed flag agrees with the inventory and records", f"{problems[:3]}")
        all_runs.append({"run": run_ref(f"pilot/eval/{d.name}"), "seed": seed, "eligible": eligible,
                         "producer_passed": summ.get("passed"), "problems": problems[:10]})
        if eligible:
            if seed in chosen:
                rep.check(False, f"seed {seed}: more than one eligible evaluation")
            chosen[seed] = d
    for s in SEEDS:
        rep.check(s in chosen, f"seed {s}: exactly one eligible evaluation", f"{[r['run'] for r in all_runs if r['seed'] == s]}")
    out["evaluations"] = all_runs
    if len(chosen) != len(SEEDS):
        return out, rep, None
    ids = json.loads((chosen[SEEDS[0]] / "manifest.json").read_text())["config"]["ids"]
    rep.check(all(json.loads((chosen[s] / "manifest.json").read_text())["config"]["ids"] == ids for s in SEEDS),
              "both counted evaluations requested the same questions")
    correct = {(arm, s): {r["task_id"]: r["score"]["correct"] for r in recorded[(chosen[s].name, arm)][0]}
               for arm in ARMS for s in SEEDS}
    db_of = {r["task_id"]: r["db_id"] for r in recorded[(chosen[SEEDS[0]].name, "T4")][0]}
    for (name, arm), (rr, _m) in recorded.items():
        rep.check(all(db_of.get(r["task_id"]) == r["db_id"] for r in rr), f"{name} {arm}: databases as in the counted runs")

    # --- L -----------------------------------------------------------------------------------------------------------
    l_dirs = sorted(p.parent for p in (bundle / "L").glob("*/records.jsonl"))
    rep.check(len(l_dirs) == 1, "exactly one L run in the bundle", f"{l_dirs}")
    l_dir = l_dirs[0]
    l_man = json.loads((l_dir / "manifest.json").read_text())
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", pin["frozen_at"], l_man["git"]["commit"]], cwd=repo).returncode
    rep.check(anc == 0, "the L run's commit contains frozen_at", l_man["git"]["commit"])
    rep.check(l_man["config"]["requested"] == ids and l_man["git"]["research_tree_dirty"] is False, "L: requested = the 64, clean tree")
    l_recs = jl(l_dir / "records.jsonl")
    rep.check(sorted(r["task_id"] for r in l_recs) == sorted(ids), "L: records are exactly the 64, once each")
    rep.check(all(db_of.get(r["task_id"]) == r["db_id"] for r in l_recs), "L: databases as in the counted runs")
    l_correct = {r["task_id"]: r["score"]["correct"] for r in l_recs}
    l_tot = totals(l_recs)
    compare_totals(rep, l_tot, json.loads((l_dir / "summary.json").read_text()).get("summary"), "L")
    out["L"] = {"run": run_ref(f"baselines/{l_dir.name}"), "commit": l_man["git"]["commit"]}

    # --- counts, gate, rule, databases, sign flip (from the recorded EX scores) ------------------------------------
    c = {f"{arm}/{s}": sum(correct[(arm, s)].values()) for arm in ARMS for s in SEEDS}
    cl = sum(l_correct.values())
    cleared = {arm: all(c[f"{arm}/{s}"] > cl for s in SEEDS) for arm in ARMS}
    per_seed = {s: c[f"T4/{s}"] - c[f"c4_params/{s}"] for s in SEEDS}
    pooled = sum(per_seed.values())
    rule = ("A1" if pooled >= LEAD and all(v > 0 for v in per_seed.values()) else
            "A2" if pooled <= -LEAD and all(v < 0 for v in per_seed.values()) else "A3")
    reading = ("both cleared L: the rule outcome is the architectural result" if all(cleared.values()) else
               "exactly one arm cleared L: the finding is that asymmetry; paired results and the rule outcome are descriptive"
               if any(cleared.values()) else "neither cleared L: training or representation needs revision")
    dbt = {}
    for t in ids:
        row = dbt.setdefault(db_of[t], {"questions": 0, "L": 0, **{f"{x}/{s}": 0 for x in ARMS for s in SEEDS}})
        row["questions"] += 1
        row["L"] += l_correct[t]
        for x in ARMS:
            for s in SEEDS:
                row[f"{x}/{s}"] += correct[(x, s)][t]
    for row in dbt.values():
        row["difference"] = sum(row[f"T4/{s}"] - row[f"c4_params/{s}"] for s in SEEDS)
    hist = Counter(sum(correct[("T4", s)][t] - correct[("c4_params", s)][t] for s in SEEDS) for t in ids)
    sf = sign_flip([row["difference"] for _db, row in sorted(dbt.items())])
    out["result"] = {"counts": {**c, "L": cl, "questions": len(ids)}, "cleared_L_in_both_seeds": cleared,
                     "differences": {"per_seed": {str(s): v for s, v in per_seed.items()}, "pooled": pooled, "pooled_out_of": 2 * len(ids)},
                     "rule": rule, "reading": reading, "per_question_difference_histogram": {str(k): hist[k] for k in sorted(hist)},
                     "databases": dict(sorted(dbt.items())), "exploratory_sign_flip": sf}

    # --- agreement with the producer's analysis.json (runs matched by reference) ------------------------------------
    an_files = sorted((bundle / "analysis").glob("*/analysis.json"))
    rep.check(len(an_files) == 1, "exactly one analysis in the bundle")
    an = json.loads(an_files[-1].read_text())
    cmp = {"counts": an["counts"] == out["result"]["counts"],
           "cleared": an["cleared_L_in_both_seeds"] == cleared,
           "differences": an["differences"] == out["result"]["differences"],
           "rule": an["rule"].split(":")[0] == rule,
           "databases": an["databases"] == out["result"]["databases"],
           "histogram": an["per_question_difference_histogram"] == out["result"]["per_question_difference_histogram"],
           "sign_flip": {k: an["exploratory_sign_flip"][k] for k in sf} == sf,
           "evaluations_counted": {s: run_ref(x) for s, x in an["inputs"]["evaluations_counted"].items()}
           == {str(s): run_ref(f"pilot/eval/{d.name}") for s, d in chosen.items()},
           "evaluations_all": sorted((run_ref(r["run"]), r["seed"], r["passed"]) for r in an["inputs"]["evaluations_all"])
           == sorted((r["run"], r["seed"], r["eligible"]) for r in all_runs),
           "L": run_ref(an["inputs"]["L"]) == out["L"]["run"]}
    for k, v in cmp.items():
        rep.check(v, f"analysis.json agrees: {k}")
    out["agreement_with_analysis_json"] = cmp

    # --- representability strata (the per-question trace is the producer's, from analysis.json) ----------------------
    reps = an["gold_representability"]["per_question"]
    out["producer_trace_used"].append("analysis.json gold_representability.per_question (recomputed only with --local --representability)")
    out["representability"] = strata(ids, reps, correct, l_correct)

    # --- serving accounting (own totals; the frozen metric and the descriptive execution accounting) -----------------
    out["serving"] = {**{f"{arm}/{s}": recorded[(chosen[s].name, arm)][1] for arm in ARMS for s in SEEDS}, "L": l_tot}

    # --- training records ----------------------------------------------------------------------------------------------
    train_pin = json.loads(VB.git_show(repo, commit, HARNESS + "pins/pilot_train_v1.json"))["task_ids"]
    rep.check(len(train_pin) == 4741 and not set(train_pin) & set(ids), "training pin: 4,741 ids, none of them evaluated")
    tr = {}
    for s in SEEDS:
        cfgs, logs = {}, {}
        for arm in ARMS:
            rd = bundle / "train" / f"{arm}_s{s}"
            cfg = json.loads((rd / "config.json").read_text())
            cfgs[arm] = cfg
            rep.check(cfg["examples"] == train_pin, f"{arm} s{s}: examples = pilot_train_v1, in order")
            t = cfg["train"]
            rep.check((t["seed"], t["updates"], t["checkpoint_every"], t["matmul_precision"], t["effective_batch"], t["micro_batch"])
                      == (s, UPDATES, CKPT_EVERY, "high", 32, 8), f"{arm} s{s}: recipe")
            log = jl(rd / "train_log.jsonl")
            logs[arm] = log
            rep.check([x["update"] for x in log] == list(range(UPDATES)), f"{arm} s{s}: train log holds updates 0..1499 once")
            bad_u = [x["update"] for x in log if x["examples"] != [train_pin[i] for i in stream(len(train_pin), x["update"], 32, s)]]
            rep.check(not bad_u, f"{arm} s{s}: every update's examples = the documented seeded stream", f"{bad_u[:5]}")
            rep.check(not any(VB.nonfinite(x) for x in log), f"{arm} s{s}: train log finite")
            evm = json.loads((chosen[s] / "manifest.json").read_text())["config"]["checkpoints"][arm]
            finals = [r for r in jl(rd / "checkpoints.jsonl") if r["update"] == UPDATES - 1]
            match = [r for r in finals if (r["sha256"], r["model_digest"]) == (evm["sha256"], evm["model_digest"])]
            rep.check(bool(match) and evm["update"] == UPDATES - 1, f"{arm} s{s}: the evaluated checkpoint is the recorded final (update 1499)")
            tr[f"{arm}/{s}"] = {"final_sha256": evm["sha256"][:16], "final_loss": log[-1]["loss"],
                               "attempts": len(list((rd / "attempts").glob("*")))}
        diff = sorted(k for k in set(cfgs["T4"]) | set(cfgs["c4_params"]) if cfgs["T4"].get(k) != cfgs["c4_params"].get(k))
        tdiff = sorted(k for k in cfgs["T4"]["train"] if cfgs["T4"]["train"][k] != cfgs["c4_params"]["train"].get(k))
        cdiff = sorted(k for k in cfgs["T4"]["controller"] if cfgs["T4"]["controller"][k] != cfgs["c4_params"]["controller"].get(k))
        rep.check(diff == ["controller", "parameters", "train"] and tdiff == ["arm"] and cdiff == ["arm", "c4_params_hidden"],
                  f"s{s}: the arms' configs differ only in arm and the semantic module", f"{diff} {tdiff} {cdiff}")
        rep.check([x["examples"] for x in logs["T4"]] == [x["examples"] for x in logs["c4_params"]], f"s{s}: identical examples at every update")
    out["training"] = tr

    # --- resources (the bundled ledger) --------------------------------------------------------------------------------
    out["resources"] = resources(bundle / "gpu_ledger.jsonl", bundle / "feasibility" / FEASIBILITY_RUN / "summary.json")
    return out, rep, {"ids": ids, "chosen": chosen, "db_of": db_of, "correct": correct, "l_correct": l_correct,
                      "l_dir": l_dir, "commit": commit, "reps": reps}


def strata(ids, reps, correct, l_correct) -> dict:
    out = {}
    for name, members in (("representable", [t for t in ids if reps[t] == "representable"]),
                          ("unrepresentable", [t for t in ids if reps[t] != "representable"])):
        out[name] = {"questions": len(members), "L": sum(l_correct[t] for t in members),
                     **{f"{x}/{s}": sum(correct[(x, s)][t] for t in members) for x in ARMS for s in SEEDS}}
    out["unrepresentable_reasons"] = {t: reps[t] for t in ids if reps[t] != "representable"}
    return out


def resources(ledger: Path, feasibility_summary: Path) -> dict:
    res = {}
    for e in jl(ledger):
        r = res.setdefault(e["reservation"], {"elapsed": 0.0, "ended": False, "status": None, "command": None, "iso": None})
        if e["event"] == "start":
            r.update(command=e.get("command"), iso=e.get("iso"))
        if e["event"] in ("heartbeat", "end"):
            r["elapsed"] = max(r["elapsed"], float(e.get("elapsed", 0.0)))
        if e["event"] == "end":
            r.update(ended=True, status=e.get("status"))
    for r in res.values():
        r["hours"] = (r["elapsed"] + (0 if r["ended"] else 30.0)) / 3600
    pil = [r for r in res.values() if (r["command"] or "").startswith("pilot-")]
    stage = lambda pfx: sum(r["hours"] for r in pil if r["command"].startswith(pfx))  # noqa: E731
    items = json.loads(feasibility_summary.read_text())["projection"]
    pick = lambda pfx: next(v for k, v in items["items_hours"].items() if k.startswith(pfx))  # noqa: E731
    return {"ledger_total_hours": sum(r["hours"] for r in res.values()), "cap_hours": 8.0,
            "pilot_hours": sum(r["hours"] for r in pil),
            "pilot_reservations": [{"started": r["iso"], "command": r["command"], "status": r["status"] if r["ended"] else "NO END (host crash)",
                                    "hours": round(r["hours"], 4)} for r in sorted(pil, key=lambda r: r["iso"])],
            "projected_vs_actual_hours": {
                "encode": {"projected": pick("encode"), "actual": stage("pilot-encode")},
                "train": {"projected": pick("train"), "actual": stage("pilot-train")},
                "evaluate": {"projected": pick("evaluate"), "actual": stage("pilot-eval")},
                "after the projection": {"projected": items["needed_hours"],
                                         "actual": stage("pilot-encode") + stage("pilot-train") + stage("pilot-eval")},
                "feasibility runs": {"actual": stage("pilot-feasibility")}}}


# --- the local checks (runs root: raw release, databases, checkpoints, cached states) --------------------------------------
def local_checks(runs: Path, repo: Path, bundle: Path, pub: dict, representability: bool, redecode: bool) -> tuple:
    rep = VB.Report("pilot local")
    out = {"reads": f"the runs root ({runs}): raw release, databases, checkpoints, cached states", "producer_code_used": []}
    pilot = runs / "pilot"
    ids, commit = pub["ids"], pub["commit"]

    # --- the 64, reconstructed from the raw release ---------------------------------------------------------------
    con = VB.contract_literals(VB.git_show(repo, commit, VB.CONTRACTS))
    raw = (runs / VB.RAW_RELEASE).read_bytes()
    rep.check(VB.sha256_bytes(raw) == con["TRAIN_RELEASE"]["files"][VB.RELEASE_KEY], "raw release sha256")
    lines = raw.decode().split("\n")
    rows = [json.loads(x) for x in (lines[:-1] if lines[-1] == "" else lines)]
    tid = [f"bird23tf-{i:05d}" for i in range(len(rows))]
    by_id = dict(zip(tid, rows))
    sp, inv = con["SPLIT"], con["INVENTORY"]
    dbs = sorted({r["db_id"] for r in rows}, key=lambda d: VB.order_key(sp["salt"], d))
    nd = round(sp["dev_fraction"] * len(dbs))
    dev, pool = dbs[:nd], set(dbs[nd:])
    snaps = json.loads(VB.git_show(repo, commit, HARNESS + "pins/snapshots_v1.json"))["databases"]
    db_path = lambda d: runs / "data/databases" / snaps[d]["file"]  # noqa: E731
    timeout = float(con["OFFICIAL_SCORER"]["meta_time_out_seconds"])
    queues = {d: sorted((t for t in tid if by_id[t]["db_id"] == d), key=lambda t: VB.order_key(inv["salt"], t)) for d in dev}
    ptr, mine = {d: 0 for d in dev}, []
    while len(mine) < 64:
        for d in dev:
            while ptr[d] < len(queues[d]):
                t = queues[d][ptr[d]]
                ptr[d] += 1
                if VB.gold_executes(db_path(d), snaps[d]["sha256"], t, by_id[t]["SQL"], timeout)[0]:
                    mine.append(t)
                    break
            if len(mine) == 64:
                break
    rep.check(mine == ids, "the 64 reconstructed from the raw release = the evaluation manifests' ids")
    rep.check(all(by_id[t]["db_id"] == pub["db_of"][t] for t in ids), "the records' databases = the raw release's")
    train_pin = json.loads(VB.git_show(repo, commit, HARNESS + "pins/pilot_train_v1.json"))["task_ids"]
    rep.check(all(by_id[t]["db_id"] in pool for t in train_pin), "training pin: training-pool databases only")

    # --- every final SQL re-executed with the own EX rule ------------------------------------------------------------
    rex = {}
    for d in sorted((pilot / "eval").glob("s*_*")):
        for arm in ARMS:
            for r in jl(d / arm / "records.jsonl"):
                t = r["task_id"]
                pred = r["final"]["sql"] if r["final"]["kind"] == "sql" else None
                val, status, _s = VB.ex(db_path(by_id[t]["db_id"]), pred, by_id[t]["SQL"], timeout)
                rex[(d.name, arm, t)] = val
                rep.check(val == r["score"]["correct"], f"{d.name} {arm} {t}: recorded EX re-derived", f"recorded {r['score']['correct']}, re-derived {val} ({status})")
    l_dir = runs / "baselines" / pub["l_dir"].name
    for r in jl(l_dir / "records.jsonl"):
        pred = r["final"]["sql"] if r["final"]["kind"] == "sql" else None
        val = VB.ex(db_path(by_id[r["task_id"]]["db_id"]), pred, by_id[r["task_id"]]["SQL"], timeout)[0]
        rep.check(val == r["score"]["correct"], f"L {r['task_id']}: recorded EX re-derived")
    out["re_executed"] = len(rex)

    # --- the final checkpoints --------------------------------------------------------------------------------------
    import torch
    for s in SEEDS:
        for arm in ARMS:
            rd = pilot / "train" / f"{arm}_s{s}"
            cfg = json.loads((rd / "config.json").read_text())
            evm = json.loads((pub["chosen"][s] / "manifest.json").read_text())["config"]["checkpoints"][arm]
            f = [r for r in jl(rd / "checkpoints.jsonl") if r["update"] == UPDATES - 1 and r["sha256"] == evm["sha256"]][-1]
            path = rd / "checkpoints" / Path(f["path"]).name
            h = VB.sha256_path(path)
            rep.check(path.name.split("_")[-1].removesuffix(".pt") == h[:16] and h == f["sha256"], f"{arm} s{s}: final checkpoint re-hashes to its name and record")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            rep.check(tensor_digest(ck["model"]) == f["model_digest"], f"{arm} s{s}: own model digest = checkpoints.jsonl")
            rep.check(optimizer_digest(ck["optimizer"]) == f["optimizer_digest"], f"{arm} s{s}: own optimizer digest = checkpoints.jsonl")
            rep.check(VB.canonical(ck["config"]) == VB.canonical(cfg) and ck["update"] == UPDATES - 1, f"{arm} s{s}: checkpoint config = config.json, update 1499")

    # --- the bundle against the run originals ---------------------------------------------------------------------------
    sums = [x.split("  ", 1) for x in (bundle / "SHA256SUMS").read_text().splitlines() if x.strip()]
    differ, no_original = [], []
    for _h, p in sums:
        q = p.split("/")
        orig = (runs / "gpu_ledger.jsonl" if p == "gpu_ledger.jsonl" else
                runs / "baselines" / "/".join(q[1:]) if q[0] == "L" else pilot / p)
        if not orig.exists():
            no_original.append(p)
        elif VB.sha256_path(orig) != VB.sha256_path(bundle / p):
            differ.append(p)
    rep.check(not differ, "bundle: bundled copies equal the run originals", f"{differ[:5]}")
    rep.check(all(p.startswith("independent/") for p in no_original), "bundle: every file without a run original is a verification output",
              f"{[p for p in no_original if not p.startswith('independent/')][:5]}")
    not_bundled = sorted(str(x.relative_to(pilot)) for x in pilot.rglob("*") if x.is_file() and x.suffix in (".json", ".jsonl")
                         and str(x.relative_to(pilot)) not in {p for _h, p in sums})
    out["bundle"] = {"equal_to_originals": len(sums) - len(differ) - len(no_original), "differ": differ,
                     "no_original": no_original, "pilot_json_not_bundled": not_bundled}

    # --- representability (producer grammar, flagged) -------------------------------------------------------------
    if representability:
        os.environ.setdefault("TI_TNN_SCALING_RUNS", str(runs))
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        sys.path.insert(0, str(repo / "research" / "ti_tnn"))
        from transformers import AutoTokenizer
        from scaling import identity as SI
        from scaling import pilot as SP
        tok = AutoTokenizer.from_pretrained(SI.model_snapshot_dir("B4"), local_files_only=True)
        out["producer_code_used"].append("scaling.pilot.representability (grammar trace) for the representability strata")
        rep.check(SP.representability(ids, tok) == pub["reps"], "representability recomputed (producer grammar) = analysis.json")

    # --- optional CPU re-decode (producer decoder, flagged) --------------------------------------------------------
    if redecode:
        os.environ.setdefault("TI_TNN_SCALING_RUNS", str(runs))
        sys.path.insert(0, str(repo / "research" / "ti_tnn"))
        from transformers import AutoTokenizer
        from scaling import encode as SE
        from scaling import identity as SI
        from scaling import infer_controller as SIC
        from scaling import train as ST
        out["producer_code_used"].append("scaling.encode/infer_controller/train: CPU re-decode from cached states (informational)")
        ident = json.loads((pub["chosen"][SEEDS[0]] / "manifest.json").read_text())["config"]["backbone"]
        reader = SE.Backbone.cache_reader(ident, AutoTokenizer.from_pretrained(SI.model_snapshot_dir("B4"), local_files_only=True))
        red = {}
        for s in SEEDS:
            for arm in ARMS:
                sha = json.loads((pub["chosen"][s] / "manifest.json").read_text())["config"]["checkpoints"][arm]["sha256"]
                rd = pilot / "train" / f"{arm}_s{s}"
                path = rd / "checkpoints" / Path([r for r in jl(rd / "checkpoints.jsonl") if r["sha256"] == sha][-1]["path"]).name
                model = ST.model_from_checkpoint(ST.load_checkpoint(path), "cpu").eval()
                evs = SIC.evaluate(model, ids, reader, score=False, use_cache=True, require_hit=True, require_verified=True,
                                   cache_root=SE.PILOT_CACHE, matmul_precision="highest", log=None)
                recs = {r["task_id"]: r for r in jl(pub["chosen"][s] / arm / "records.jsonl")}
                same = sum(e.get("actions") == recs[e["task_id"]]["attempts"][0].get("actions") and e["final"].get("sql") == recs[e["task_id"]]["final"]["sql"] for e in evs)
                red[f"{arm}/{s}"] = f"{same}/64 identical actions and SQL (CPU float32 vs GPU TF32)"
        out["redecode_cpu"] = red
    return out, rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=HERE.parents[3])  # this checkout
    ap.add_argument("--bundle", type=Path, default=None, help="default: the checkout's evidence/pilot_v1")
    ap.add_argument("--local", action="store_true", help="also run the checks that need the runs root")
    ap.add_argument("--runs-root", type=Path, default=None)
    ap.add_argument("--representability", action="store_true")
    ap.add_argument("--redecode", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    bundle = a.bundle or a.repo / HARNESS / "evidence" / "pilot_v1"
    pub_out, pub_rep, pub = public_reconstruction(bundle, a.repo)
    pub_out["checks"] = {"passed": len(pub_rep.passes), "failed": pub_rep.fails, "warnings": pub_rep.warns}
    git = lambda *args: subprocess.run(["git", *args], cwd=a.repo, capture_output=True, text=True).stdout.strip()  # noqa: E731
    out = {"verifier": {"file": "research/ti_tnn/scaling/independent/verify_pilot.py",
                        "sha256": VB.sha256_path(Path(__file__)), "checkout_commit": git("rev-parse", "HEAD"),
                        # uncommitted changes to what it reads: itself, the bundle, the pins and the protocol
                        "checkout_dirty_paths": git("status", "--porcelain", "--", HARNESS + "independent", str(bundle),
                                                    HARNESS + "pins", HARNESS + "docs/PROTOCOL_pilot.md").splitlines()},
           "public_reconstruction": pub_out, "local_checks": "not run (pass --local, with the runs root)"}
    fails = list(pub_rep.fails)
    if a.local and pub is not None:
        loc_out, loc_rep = local_checks(a.runs_root or a.repo / "runs" / "ti_tnn_scaling", a.repo, bundle, pub,
                                        a.representability, a.redecode)
        loc_out["checks"] = {"passed": len(loc_rep.passes), "failed": loc_rep.fails, "warnings": loc_rep.warns}
        out["local_checks"] = loc_out
        fails += loc_rep.fails
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=str) + "\n")
    r = pub_out.get("result") or {}
    print(json.dumps({"counts": r.get("counts"), "cleared": r.get("cleared_L_in_both_seeds"), "differences": r.get("differences"),
                      "rule": r.get("rule"), "reading": r.get("reading"), "sign_flip": r.get("exploratory_sign_flip"),
                      "agreement": pub_out.get("agreement_with_analysis_json"),
                      "public_checks": {"passed": len(pub_rep.passes), "failed": len(pub_rep.fails)},
                      "local_checks": out["local_checks"]["checks"] if isinstance(out["local_checks"], dict) else out["local_checks"]},
                     indent=1, default=str))
    for x in fails:
        print("FAIL", x)
    return int(bool(fails))


if __name__ == "__main__":
    sys.exit(main())
