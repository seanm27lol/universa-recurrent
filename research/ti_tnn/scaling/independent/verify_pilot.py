#!/usr/bin/env python3
"""Independent reconstruction of the TI-TNN C4/T4 learning pilot's verdict, written apart from the producers.

    usage: nice -n 19 python3 verify_pilot.py [--repo REPO] [--runs-root RUNS] [--l-run DIR] [--bundle DIR]
                                              [--representability] [--redecode] [--json OUT]

Own code (no scaling import): this directory's verify_baselines.py helpers (EX rule, git-show readers, inventory
rules), own JSON/ledger parsing, own tensor and optimizer digests, own sampling-stream recomputation (torch.randperm),
own sign-flip enumeration. Producer code is used only where flagged in the output:
- --representability: scaling.pilot.representability (the grammar's gold-program trace), CPU tokenization;
- --redecode: scaling.encode / infer_controller / train to re-decode the 64 x 4 requests on CPU from the final
  checkpoints and the anchored cache entries (informational: the GPU decoded at TF32, the CPU at float32).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_baselines as VB  # noqa: E402  (the reviewer's own helpers)

ARMS, SEEDS, UPDATES, CKPT_EVERY, LEAD = ("T4", "c4_params"), (100, 101), 1500, 250, 7
HARNESS = "research/ti_tnn/scaling/"


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=HERE.parents[3])  # this checkout
    ap.add_argument("--runs-root", type=Path, default=None)
    ap.add_argument("--l-run", type=Path, default=None)
    ap.add_argument("--bundle", type=Path, default=None)
    ap.add_argument("--representability", action="store_true")
    ap.add_argument("--redecode", action="store_true")
    ap.add_argument("--json", type=Path, default=HERE / "pilot_verdict.json")
    a = ap.parse_args(argv)
    repo = a.repo
    runs = a.runs_root or repo / "runs" / "ti_tnn_scaling"
    pilot = runs / "pilot"
    rep = VB.Report("pilot")
    out = {"producer_code_used": []}

    # --- the frozen protocol and the code identities --------------------------------------------------------------
    ev_dirs = sorted((pilot / "eval").glob("s*_*"))
    commits = {json.loads((d / "manifest.json").read_text())["git"]["commit"] for d in ev_dirs}
    commits |= {json.loads(next(iter(sorted((pilot / "train" / f"{arm}_s{s}" / "attempts").glob("*")))).joinpath("manifest.json").read_text())["git"]["commit"]
                for arm in ARMS for s in SEEDS}
    rep.check(len(commits) == 1, "every training and evaluation run at one commit", f"{commits}")
    commit = sorted(commits)[0]
    pin = json.loads(VB.git_show(repo, commit, HARNESS + "pins/pilot_protocol_v1.json"))
    proto_at_freeze = VB.sha256_bytes(VB.git_show(repo, pin["frozen_at"], HARNESS + "docs/PROTOCOL_pilot.md"))
    proto_at_run = VB.sha256_bytes(VB.git_show(repo, commit, HARNESS + "docs/PROTOCOL_pilot.md"))
    rep.check(pin["sha256"] == proto_at_freeze == proto_at_run, "protocol: pin = file at frozen_at = file at the run commit",
              f"{pin['sha256'][:12]} / {proto_at_freeze[:12]} / {proto_at_run[:12]}")
    out["code"] = {"run_commit": commit, "protocol_sha256": pin["sha256"], "frozen_at": pin["frozen_at"]}

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
    ptr, ids = {d: 0 for d in dev}, []
    while len(ids) < 64:
        for d in dev:
            while ptr[d] < len(queues[d]):
                t = queues[d][ptr[d]]
                ptr[d] += 1
                if VB.gold_executes(db_path(d), snaps[d]["sha256"], t, by_id[t]["SQL"], timeout)[0]:
                    ids.append(t)
                    break
            if len(ids) == 64:
                break
    db_of = {t: by_id[t]["db_id"] for t in ids}
    out["eval_ids_reconstructed"] = ids

    # --- anchors (own parse: clean runs under the frozen protocol) ------------------------------------------------
    anchors = {}
    for entries in sorted((pilot / "encode").glob("*/entries.jsonl")):
        m = json.loads((entries.parent / "manifest.json").read_text())
        if m["git"]["research_tree_dirty"] or (m["config"].get("protocol") or {}).get("sha256") != pin["sha256"]:
            continue
        for x in entries.read_text().split("\n")[:-1]:
            e = json.loads(x)
            if e["task_id"] in anchors and anchors[e["task_id"]]["states_sha256"] != e["states_sha256"]:
                rep.check(False, f"conflicting anchors for {e['task_id']}")
            anchors.setdefault(e["task_id"], e)

    # --- evaluations: re-score every final SQL, own integrity checks, selection ----------------------------------
    correct, all_runs, chosen = {}, [], {}
    for d in ev_dirs:
        man = json.loads((d / "manifest.json").read_text())
        summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        seed = man["config"]["seed"]
        ok = man["git"]["research_tree_dirty"] is False and man["config"]["ids"] == ids and \
            man["config"]["protocol"]["sha256"] == pin["sha256"]
        recs = {arm: jl(d / arm / "records.jsonl") for arm in ARMS}
        run_correct, problems = {}, []
        for arm in ARMS:
            rr = recs[arm]
            if sorted(r["task_id"] for r in rr) != sorted(ids):
                problems.append(f"{arm}: records are not exactly the 64")
            arm_cid = VB.sha256_bytes(VB.canonical({"run": man["config_identity"], "arm": arm,
                                                     "model_digest": man["config"]["checkpoints"][arm]["model_digest"]}).encode())
            c = {}
            for r in rr:
                t = r["task_id"]
                pred = r["final"]["sql"] if r["final"]["kind"] == "sql" else None
                val, status, secs = VB.ex(db_path(db_of[t]), pred, by_id[t]["SQL"], timeout)
                c[t] = val
                if val != r["score"]["correct"]:
                    problems.append(f"{arm} {t}: recorded {r['score']['correct']}, re-derived {val} ({status})")
                if r["score"].get("scored_sql_sha256") != VB.sha256_bytes((pred or "").encode()) or \
                        r["snapshot_sha256"] != snaps[db_of[t]]["sha256"] or r["config_identity"] != arm_cid:
                    problems.append(f"{arm} {t}: score/snapshot/config binding")
                if r.get("states_sha256") != anchors.get(t, {}).get("states_sha256"):
                    problems.append(f"{arm} {t}: fresh states differ from the anchor")
                if VB.nonfinite(r):
                    problems.append(f"{arm} {t}: non-finite number")
            run_correct[arm] = c
        cross = [t for t in ids if {r["task_id"]: r["states_sha256"] for r in recs["T4"]}.get(t)
                 != {r["task_id"]: r["states_sha256"] for r in recs["c4_params"]}.get(t)]
        if cross:
            problems.append(f"states differ across the arms: {cross[:3]}")
        flags = {"summary_passed": summ.get("passed"), "states_differ_across_arms": summ.get("states_differ_across_arms"),
                 **{f"{arm}_{k}": (summ.get("arms", {}).get(arm) or {}).get(k) for arm in ARMS
                    for k in ("reconstruction_all_equal", "fresh_states_all_equal_anchor", "record_errors")}}
        passed = ok and not problems and summ.get("passed") is True
        all_runs.append({"run": d.name, "seed": seed, "passed": passed, "problems": problems[:10], "flags": flags,
                         "commit": man["git"]["commit"]})
        if passed:
            if seed in chosen:
                rep.check(False, f"seed {seed}: more than one passing evaluation")
            chosen[seed] = d
            for arm in ARMS:
                correct[(arm, seed)] = run_correct[arm]
        for p in problems:
            rep.check(False, f"{d.name}: {p}")
    for s in SEEDS:
        rep.check(s in chosen, f"seed {s}: exactly one passing evaluation", f"{[r['run'] for r in all_runs if r['seed'] == s]}")
    out["evaluations"] = all_runs

    # --- L ---------------------------------------------------------------------------------------------------------
    l_runs = sorted((runs / "baselines").glob("*_L_nonthinking_p0-64"))
    rep.check(len(l_runs) == 1, "exactly one L run on the 64", f"{l_runs}")
    l_dir = a.l_run or l_runs[0]
    l_man = json.loads((l_dir / "manifest.json").read_text())
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", pin["frozen_at"], l_man["git"]["commit"]], cwd=repo).returncode
    rep.check(anc == 0, "the L run's commit contains frozen_at", l_man["git"]["commit"])
    rep.check(l_man["config"]["requested"] == ids and l_man["git"]["research_tree_dirty"] is False, "L: requested = the 64, clean tree")
    l_correct = {}
    for r in jl(l_dir / "records.jsonl"):
        pred = r["final"]["sql"] if r["final"]["kind"] == "sql" else None
        l_correct[r["task_id"]] = VB.ex(db_path(db_of[r["task_id"]]), pred, by_id[r["task_id"]]["SQL"], timeout)[0]
        rep.check(l_correct[r["task_id"]] == r["score"]["correct"], f"L {r['task_id']}: correct re-derived")
    rep.check(sorted(l_correct) == sorted(ids), "L: exactly the 64")
    out["L"] = {"run": l_dir.name, "commit": l_man["git"]["commit"]}

    # --- counts, gate, rule, databases, sign flip -------------------------------------------------------------------
    c = {f"{arm}/{s}": sum(correct[(arm, s)].values()) for arm in ARMS for s in SEEDS}
    cl = sum(l_correct.values())
    cleared = {arm: all(c[f"{arm}/{s}"] > cl for s in SEEDS) for arm in ARMS}
    per_seed = {s: c[f"T4/{s}"] - c[f"c4_params/{s}"] for s in SEEDS}
    pooled = sum(per_seed.values())
    rule = ("A1" if pooled >= LEAD and all(v > 0 for v in per_seed.values()) else
            "A2" if pooled <= -LEAD and all(v < 0 for v in per_seed.values()) else "A3")
    both = all(cleared.values())
    reading = ("both cleared L: the rule outcome is the architectural result" if both else
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
                     "differences": {"per_seed": {str(s): v for s, v in per_seed.items()}, "pooled": pooled, "pooled_out_of": 128},
                     "rule": rule, "reading": reading, "per_question_difference_histogram": {str(k): hist[k] for k in sorted(hist)},
                     "databases": dict(sorted(dbt.items())), "exploratory_sign_flip": sf}

    # --- compare with the producer's analysis.json -----------------------------------------------------------------
    an_dirs = sorted((pilot / "analysis").glob("*"))
    an = json.loads((an_dirs[-1] / "analysis.json").read_text())
    cmp = {"counts": an["counts"] == out["result"]["counts"],
           "cleared": an["cleared_L_in_both_seeds"] == cleared,
           "differences": an["differences"] == out["result"]["differences"],
           "rule": an["rule"].split(":")[0] == rule,
           "databases": an["databases"] == out["result"]["databases"],
           "histogram": an["per_question_difference_histogram"] == out["result"]["per_question_difference_histogram"],
           "sign_flip": {k: an["exploratory_sign_flip"][k] for k in sf} == sf,
           "inputs": {str(s): str(d) for s, d in chosen.items()} == an["inputs"]["evaluations_counted"]}
    for k, v in cmp.items():
        rep.check(v, f"analysis.json agrees: {k}")
    out["agreement_with_analysis_json"] = cmp

    # --- representability (producer grammar, flagged) ---------------------------------------------------------------
    reps = an["gold_representability"]["per_question"]
    if a.representability:
        os.environ.setdefault("TI_TNN_SCALING_RUNS", str(runs))
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        sys.path.insert(0, str(repo / "research" / "ti_tnn"))
        from transformers import AutoTokenizer
        from scaling import pilot as SP
        from scaling import identity as SI
        tok = AutoTokenizer.from_pretrained(SI.model_snapshot_dir("B4"), local_files_only=True)
        mine = SP.representability(ids, tok)
        out["producer_code_used"].append("scaling.pilot.representability (grammar trace) for the representability strata")
        rep.check(mine == reps, "representability recomputed (producer grammar) = analysis.json", "")
        reps = mine
    strata = {}
    for name, members in (("representable", [t for t in ids if reps[t] == "representable"]),
                          ("unrepresentable", [t for t in ids if reps[t] != "representable"])):
        strata[name] = {"questions": len(members), "L": sum(l_correct[t] for t in members),
                        **{f"{x}/{s}": sum(correct[(x, s)][t] for t in members) for x in ARMS for s in SEEDS}}
    strata["unrepresentable_reasons"] = {t: reps[t] for t in ids if reps[t] != "representable"}
    out["representability"] = strata

    # --- training integrity ------------------------------------------------------------------------------------------
    import torch
    train_pin = json.loads(VB.git_show(repo, commit, HARNESS + "pins/pilot_train_v1.json"))["task_ids"]
    rep.check(all(by_id[t]["db_id"] in pool for t in train_pin) and len(train_pin) == 4741, "training pin: 4,741 training-pool ids")
    tr = {}
    for s in SEEDS:
        cfgs, logs = {}, {}
        for arm in ARMS:
            rd = pilot / "train" / f"{arm}_s{s}"
            cfg = json.loads((rd / "config.json").read_text())
            cfgs[arm] = cfg
            rep.check(cfg["examples"] == train_pin, f"{arm} s{s}: examples = pilot_train_v1, in order")
            t = cfg["train"]
            rep.check((t["seed"], t["updates"], t["checkpoint_every"], t["matmul_precision"], t["effective_batch"], t["micro_batch"])
                      == (s, UPDATES, CKPT_EVERY, "high", 32, 8), f"{arm} s{s}: recipe")
            log = jl(rd / "train_log.jsonl")
            logs[arm] = log
            rep.check([x["update"] for x in log] == list(range(UPDATES)), f"{arm} s{s}: train log holds updates 0..1499 once")
            bad = [x["update"] for x in log if x["examples"] != [train_pin[i] for i in stream(len(train_pin), x["update"], 32, s)]]
            rep.check(not bad, f"{arm} s{s}: every update's examples = the documented seeded stream", f"{bad[:5]}")
            rep.check(not any(VB.nonfinite(x) for x in log), f"{arm} s{s}: train log finite")
            recs = jl(rd / "checkpoints.jsonl")
            final = [r for r in recs if r["update"] == UPDATES - 1]
            rep.check(len(final) == 1, f"{arm} s{s}: one recorded final checkpoint")
            f = final[-1]
            path = rd / "checkpoints" / Path(f["path"]).name
            h = VB.sha256_path(path)
            rep.check(path.name.split("_")[-1].removesuffix(".pt") == h[:16] and h == f["sha256"], f"{arm} s{s}: final checkpoint re-hashes to its name and record")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            md, od = tensor_digest(ck["model"]), optimizer_digest(ck["optimizer"])
            rep.check(md == f["model_digest"], f"{arm} s{s}: own model digest = checkpoints.jsonl", md[:12])
            rep.check(od == f["optimizer_digest"], f"{arm} s{s}: own optimizer digest = checkpoints.jsonl", od[:12])
            rep.check(VB.canonical(ck["config"]) == VB.canonical(cfg) and ck["update"] == UPDATES - 1, f"{arm} s{s}: checkpoint config = config.json, update 1499")
            evm = json.loads((chosen[s] / "manifest.json").read_text())["config"]["checkpoints"][arm]
            rep.check((evm["sha256"], evm["model_digest"]) == (f["sha256"], f["model_digest"]), f"{arm} s{s}: the evaluated checkpoint is this final one")
            tr[f"{arm}/{s}"] = {"final": path.name, "model_digest": md[:16], "final_loss": log[-1]["loss"], "attempts": len(list((rd / "attempts").glob("*")))}
        diff = sorted(k for k in set(cfgs["T4"]) | set(cfgs["c4_params"]) if cfgs["T4"].get(k) != cfgs["c4_params"].get(k))
        tdiff = sorted(k for k in cfgs["T4"]["train"] if cfgs["T4"]["train"][k] != cfgs["c4_params"]["train"].get(k))
        cdiff = sorted(k for k in cfgs["T4"]["controller"] if cfgs["T4"]["controller"][k] != cfgs["c4_params"]["controller"].get(k))
        rep.check(diff == ["controller", "parameters", "train"] and tdiff == ["arm"] and cdiff == ["arm", "c4_params_hidden"],
                  f"s{s}: the arms' configs differ only in arm and the semantic module", f"{diff} {tdiff} {cdiff}")
        rep.check([x["examples"] for x in logs["T4"]] == [x["examples"] for x in logs["c4_params"]], f"s{s}: identical examples at every update")
    out["training"] = tr

    # --- optional CPU re-decode (producer decoder, flagged) --------------------------------------------------------
    if a.redecode:
        os.environ.setdefault("TI_TNN_SCALING_RUNS", str(runs))
        sys.path.insert(0, str(repo / "research" / "ti_tnn"))
        from transformers import AutoTokenizer
        from scaling import encode as SE, infer_controller as SIC, train as ST, identity as SI
        out["producer_code_used"].append("scaling.encode/infer_controller/train: CPU re-decode from cached states (informational)")
        ident = json.loads((chosen[100] / "manifest.json").read_text())["config"]["backbone"]
        tok = AutoTokenizer.from_pretrained(SI.model_snapshot_dir("B4"), local_files_only=True)
        reader = SE.Backbone.cache_reader(ident, tok)
        red = {}
        for s in SEEDS:
            for arm in ARMS:
                f = [r for r in jl(pilot / "train" / f"{arm}_s{s}" / "checkpoints.jsonl") if r["update"] == UPDATES - 1][-1]
                model = ST.model_from_checkpoint(ST.load_checkpoint(pilot / "train" / f"{arm}_s{s}" / "checkpoints" / Path(f["path"]).name), "cpu").eval()
                evs = SIC.evaluate(model, ids, reader, score=False, use_cache=True, require_hit=True, require_verified=True,
                                   cache_root=SE.PILOT_CACHE, matmul_precision="highest", log=None)
                recs = {r["task_id"]: r for r in jl(chosen[s] / arm / "records.jsonl")}
                same = sum(e.get("actions") == recs[e["task_id"]]["attempts"][0].get("actions") and e["final"].get("sql") == recs[e["task_id"]]["final"]["sql"] for e in evs)
                red[f"{arm}/{s}"] = f"{same}/64 identical actions and SQL (CPU float32 vs GPU TF32)"
        out["redecode_cpu"] = red

    # --- resources ---------------------------------------------------------------------------------------------------
    res = {}
    for e in jl(runs / "gpu_ledger.jsonl"):
        r = res.setdefault(e["reservation"], {"elapsed": 0.0, "ended": False, "status": None, "command": None, "iso": None})
        if e["event"] == "start":
            r.update(command=e.get("command"), iso=e.get("iso"))
        if e["event"] in ("heartbeat", "end"):
            r["elapsed"] = max(r["elapsed"], float(e.get("elapsed", 0.0)))
        if e["event"] == "end":
            r.update(ended=True, status=e.get("status"))
    for r in res.values():
        r["hours"] = (r["elapsed"] + (0 if r["ended"] else 30.0)) / 3600
    total = sum(r["hours"] for r in res.values())
    pil = [r for r in res.values() if (r["command"] or "").startswith("pilot-")]
    stage = lambda pfx: sum(r["hours"] for r in pil if r["command"].startswith(pfx))  # noqa: E731
    fz = json.loads((pilot / "feasibility" / "20261030T175558Z" / "summary.json").read_text())["projection"]
    items = fz["items_hours"]
    first_fz = [r for r in pil if r["command"] == "pilot-feasibility"]
    out["resources"] = {
        "ledger_total_hours": total, "cap_hours": 8.0, "pilot_hours": sum(r["hours"] for r in pil),
        "pilot_reservations": [{"started": r["iso"], "command": r["command"], "status": r["status"] if r["ended"] else "NO END (host crash)",
                                "hours": round(r["hours"], 4)} for r in sorted(pil, key=lambda r: r["iso"])],
        "projected_vs_actual_hours": {
            "encode (dev + 4 train chunks, incl. crash and failed resume)": {"projected": next(v for k, v in items.items() if k.startswith("encode")),
                                                                          "actual": stage("pilot-encode")},
            "train (4 fits)": {"projected": next(v for k, v in items.items() if k.startswith("train")), "actual": stage("pilot-train")},
            "evaluate + reconstruct (2 runs)": {"projected": next(v for k, v in items.items() if k.startswith("evaluate")), "actual": stage("pilot-eval")},
            "projection total (excl. the two feasibility runs)": {"projected": fz["needed_hours"],
                                                                   "actual": stage("pilot-encode") + stage("pilot-train") + stage("pilot-eval")},
            "feasibility runs (before the projection)": {"actual": sum(r["hours"] for r in first_fz)}}}

    # --- the evidence bundle -----------------------------------------------------------------------------------------
    if a.bundle:
        sums = [x.split("  ", 1) for x in (a.bundle / "SHA256SUMS").read_text().splitlines() if x.strip()]
        bad_sum = [p for h, p in sums if VB.sha256_path(a.bundle / p) != h]
        mapped, differ, missing = 0, [], []
        for _h, p in sums:
            q = p.split("/")
            orig = (runs / "gpu_ledger.jsonl" if p == "gpu_ledger.jsonl" else
                    runs / "baselines" / "/".join(q[1:]) if q[0] == "L" else pilot / p)
            if not orig.exists():
                missing.append(p)
                continue
            mapped += 1
            if VB.sha256_path(orig) != VB.sha256_path(a.bundle / p):
                differ.append(p)
        not_bundled = sorted(str(x.relative_to(pilot)) for x in pilot.rglob("*") if x.is_file() and x.suffix in (".json", ".jsonl")
                             and str(x.relative_to(pilot)) not in {p for _h, p in sums})
        rep.check(not bad_sum, "bundle: every file matches SHA256SUMS", f"{bad_sum[:3]}")
        rep.check(not differ, "bundle: bundled copies equal the run originals", f"{differ[:5]}")
        out["bundle"] = {"files": len(sums), "sha256sums_ok": not bad_sum, "equal_to_originals": mapped - len(differ),
                         "differ": differ, "no_original": missing, "pilot_json_not_bundled": not_bundled}

    out["checks"] = {"passed": len(rep.passes), "failed": rep.fails, "warnings": rep.warns}
    a.json.write_text(json.dumps(out, indent=1, default=str))
    r = out["result"]
    print(json.dumps({"counts": r["counts"], "cleared": r["cleared_L_in_both_seeds"], "differences": r["differences"],
                      "rule": r["rule"], "reading": r["reading"], "sign_flip": r["exploratory_sign_flip"],
                      "agreement": cmp, "checks_failed": len(rep.fails)}, indent=1))
    for x in rep.fails:
        print("FAIL", x)
    for x in rep.warns:
        print("WARN", x)
    return int(bool(rep.fails))


if __name__ == "__main__":
    sys.exit(main())
