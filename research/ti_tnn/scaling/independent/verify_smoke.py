#!/usr/bin/env python3
"""Independent check of a TI-TNN scaling smoke run (T4 vs a C4 control), written apart from the producers.

    usage: nice -n 19 python3 verify_smoke.py SMOKE_RUN_DIR [--repo REPO] [--runs-root RUNS] [--no-rehash] [--redecode]

(a)-(c) import nothing from the scaling package. They use this directory's verify_baselines.py helpers, the raw
release, the pins and contracts at the run's commit (`git show`), sqlite3, and torch (to read checkpoints and recompute
the documented sampling stream).
(a) The dev ids are the first 16 of the reconstructed inventory. The training ids equal ir_coverage_v1's smoke_train
    ids, come from the training pool only, and follow the smoke candidate order.
(b) Each checkpoint re-hashes to its name, and its tensors to the recorded model digest (own digest code). Both arms
    share the training config except the arm. Each update's examples equal the documented seeded sampling stream,
    the same in both arms. The train log is complete and finite.
(c) Every record's final SQL is re-executed against gold with this directory's EX implementation. Summaries are
    recomputed. Identities, snapshots and score bindings are checked.
(d) --redecode (uses scaling.controller / infer_controller / encode / train for decoding only): re-decode the 16 dev
    requests on CPU from the final checkpoints and the cached GPU dev states, then compare actions and final SQL with
    the records. The cache key is recomputed from the recorded backbone identity, so the tokenized context must match
    exactly; each cached tensor must hash to the record's states_sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_baselines as VB  # noqa: E402  (this reviewer's own helpers)

PIN_DIR, CONTRACTS = VB.PIN_DIR, VB.CONTRACTS


def tensor_digest(state: dict) -> str:
    """sha256 over tensors in sorted-key order: json([name, dtype, shape]) then the raw bytes (own implementation)."""
    import torch
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().to("cpu").contiguous().reshape(-1)
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def stream(n: int, update: int, batch: int, seed: int) -> list:
    """The documented sampling rule, recomputed: epoch e = torch.randperm(n) with a Generator seeded by
    int(sha256(f'{seed}:epoch:{e}')[:15], 16); update u takes stream positions [u*batch, (u+1)*batch)."""
    import torch
    out, perms = [], {}
    for pos in range(update * batch, (update + 1) * batch):
        e, k = divmod(pos, n)
        if e not in perms:
            g = torch.Generator().manual_seed(int(hashlib.sha256(f"{seed}:epoch:{e}".encode()).hexdigest()[:15], 16))
            perms[e] = torch.randperm(n, generator=g).tolist()
        out.append(perms[e][k])
    return out


def verify(run: Path, repo: Path, runs_root: Path, rehash: bool, redecode: bool) -> VB.Report:
    rep = VB.Report(run.name)
    man = json.loads((run / "manifest.json").read_text())
    summ = json.loads((run / "summary.json").read_text())
    cfg, commit = man["config"], man["git"]["commit"]
    rep.check(man["git"].get("research_tree_dirty") is False, "research tree clean")
    rep.check(VB.sha256_bytes(VB.canonical(cfg).encode()) == man["config_identity"], "config hashes to config_identity")
    contracts = VB.contract_literals(VB.git_show(repo, commit, CONTRACTS))
    pins = {n: VB.git_show(repo, commit, f"{PIN_DIR}/{n}.json") for n in ("inventory_v1", "snapshots_v1", "ir_coverage_v1")}
    rep.check(VB.sha256_bytes(pins["inventory_v1"]) == cfg["inventory_pin_sha256"], "inventory pin hash")
    rep.check(VB.sha256_bytes(pins["snapshots_v1"]) == cfg["snapshots_pin_sha256"], "snapshots pin hash")

    # --- (a) ids ------------------------------------------------------------------------------------------------------
    raw = (runs_root / VB.RAW_RELEASE).read_bytes()
    rep.check(VB.sha256_bytes(raw) == contracts["TRAIN_RELEASE"]["files"][VB.RELEASE_KEY], "raw release sha256")
    lines = raw.decode().split("\n")
    rows = [json.loads(x) for x in (lines[:-1] if lines[-1] == "" else lines)]
    tid = [f"bird23tf-{i:05d}" for i in range(len(rows))]
    by_id = dict(zip(tid, rows))
    sp, inv, sm = contracts["SPLIT"], contracts["INVENTORY"], contracts["SMOKE_TRAIN"]
    dbs = sorted({r["db_id"] for r in rows}, key=lambda d: VB.order_key(sp["salt"], d))
    nd = round(sp["dev_fraction"] * len(dbs))
    dev, pool = dbs[:nd], set(dbs[nd:])
    snaps = json.loads(pins["snapshots_v1"])["databases"]
    db_path = lambda d: runs_root / "data/databases" / snaps[d]["file"]   # noqa: E731
    timeout = float(contracts["OFFICIAL_SCORER"]["meta_time_out_seconds"])
    dev_ids = cfg["dev_task_ids"]
    queues = {d: sorted((t for t in tid if by_id[t]["db_id"] == d), key=lambda t: VB.order_key(inv["salt"], t)) for d in dev}
    ptr, order = {d: 0 for d in dev}, []
    while len(order) < len(dev_ids):
        for d in dev:
            while ptr[d] < len(queues[d]):
                t = queues[d][ptr[d]]
                ptr[d] += 1
                if VB.gold_executes(db_path(d), snaps[d]["sha256"], t, by_id[t]["SQL"], timeout)[0]:
                    order.append(t)
                    break
            if len(order) == len(dev_ids):
                break
    rep.check(order == dev_ids, "dev ids = first 16 of the reconstructed inventory", f"{order} vs {dev_ids}")
    train_ids = cfg["train_task_ids"]
    cov = json.loads(pins["ir_coverage_v1"])["smoke_train"]
    rep.check(train_ids == cov["task_ids"] and len(train_ids) == sm["size"], "training ids = ir_coverage_v1 smoke_train ids (128)")
    rep.check(all(by_id[t]["db_id"] in pool for t in train_ids) and not any(by_id[t]["db_id"] in dev for t in train_ids),
              "training ids all from training-pool databases, none from dev")
    smoke_order = sorted((t for t in tid if by_id[t]["db_id"] in pool), key=lambda t: VB.order_key(sm["salt"], t))
    pos = [smoke_order.index(t) for t in train_ids]
    head = smoke_order[:pos[-1] + 1]
    skipped = [t for t in head if t not in set(train_ids)]
    rep.check(pos == sorted(pos) and set(skipped) == set(cov.get("skipped_before_selection", {})),
              "training ids follow the smoke order; the candidates passed over are exactly the pin's skipped list",
              f"{len(skipped)} passed over vs {len(cov.get('skipped_before_selection', {}))} pinned")

    # --- (b) checkpoints and training ------------------------------------------------------------------------------------
    import torch
    arms = list(cfg["arms"])
    logs, digests = {}, {}
    for arm in arms:
        acfg = json.loads((run / arm / "config.json").read_text())
        rep.check(acfg["examples"] == train_ids and acfg.get("requested_task_ids") == train_ids and not acfg.get("skipped"),
                  f"{arm}: trained on exactly the 128 training ids, none skipped")
        ck = summ["arms"][arm]["checkpoint"]
        files = sorted((run / arm / "checkpoints").glob("*.pt"))
        for f in files:
            h = VB.sha256_path(f)
            rep.check(f.name.split("_")[-1].removesuffix(".pt") == h[:16], f"{arm}: {f.name} re-hashes to its name", h[:16])
        final = run / arm / "checkpoints" / Path(ck["path"]).name
        rep.check(final.exists() and VB.sha256_path(final) == ck["sha256"], f"{arm}: final checkpoint file = recorded sha256")
        state = torch.load(final, map_location="cpu", weights_only=False)
        digests[arm] = tensor_digest(state["model"])
        rep.check(digests[arm] == ck["model_digest"], f"{arm}: own tensor digest = recorded model_digest", digests[arm][:16])
        rep.check(state["update"] == cfg["updates"] - 1 and state["controller"]["arm"] == arm, f"{arm}: final update and arm in the checkpoint")
        log = [json.loads(x) for x in (run / arm / "train_log.jsonl").read_text().splitlines() if x.strip()]
        logs[arm] = log
        rep.check([x["update"] for x in log] == list(range(cfg["updates"])), f"{arm}: train log has updates 0..{cfg['updates'] - 1} once each")
        rep.check(not (run / arm / "train_log.superseded.jsonl").exists(), f"{arm}: no superseded training (no resume)")
        rep.check(not any(VB.nonfinite(x) for x in log), f"{arm}: train log numbers finite")
        tr = cfg["train"][arm]
        bad = [x["update"] for x in log
               if x["examples"] != [train_ids[i] for i in stream(len(train_ids), x["update"], tr["effective_batch"], tr["seed"])]]
        rep.check(not bad, f"{arm}: each update's examples = the documented seeded sampling stream", f"updates {bad[:5]}")
    a, b = arms
    strip = lambda d: {k: v for k, v in d.items() if k != "arm"}   # noqa: E731
    rep.check(strip(cfg["train"][a]) == strip(cfg["train"][b]), "both arms: identical training configuration except the arm")
    rep.check([x["examples"] for x in logs[a]] == [x["examples"] for x in logs[b]], "both arms: identical examples at every update")
    rep.check(digests[a] != digests[b], "the two arms' final weights differ (both trained)")

    # --- (c) records, scores, summaries -----------------------------------------------------------------------------
    hashed = {}
    for arm in arms:
        recs = [json.loads(x) for x in (run / arm / "records.jsonl").read_text().splitlines() if x.strip()]
        ids = [r["task_id"] for r in recs]
        rep.check(sorted(ids) == sorted(dev_ids) and len(set(ids)) == len(ids), f"{arm}: each dev request exactly once")
        arm_cid = VB.sha256_bytes(VB.canonical({"run": man["config_identity"], "arm": arm,
                                                 "model_digest": summ["arms"][arm]["checkpoint"]["model_digest"]}).encode())
        rep.check(all(r["config_identity"] == arm_cid for r in recs) and summ["arms"][arm]["config_identity"] == arm_cid,
                  f"{arm}: config_identity = sha256(run identity, arm, model digest)")
        correct, statuses = 0, []
        for r in recs:
            t, d = r["task_id"], r["db_id"]
            rep.check(by_id[t]["db_id"] == d and r["snapshot_sha256"] == snaps[d]["sha256"] == r["score"]["snapshot_sha256"],
                      f"{arm} {t}: pinned snapshot")
            if rehash and d not in hashed:
                hashed[d] = VB.sha256_path(db_path(d))
                rep.check(hashed[d] == snaps[d]["sha256"], f"{d}: file re-hashes to the pin")
            rep.check(not VB.nonfinite(r), f"{arm} {t}: numbers finite")
            pred = r["final"]["sql"] if r["final"]["kind"] == "sql" else None
            rep.check(r["score"]["scored_sql_sha256"] == VB.sha256_bytes((pred or "").encode()), f"{arm} {t}: score bound to the SQL")
            rep.check(r["attempts"] and r["attempts"][-1].get("final_sql") == pred and r["tool_calls"] == 0 and r["repairs"] == 0,
                      f"{arm} {t}: one decode attempt, no tools, no repair")
            c, st, secs = VB.ex(db_path(d), pred, by_id[t]["SQL"], timeout)
            statuses.append((t, st, round(secs, 1), r["score"]["status"]))
            rep.check(c == r["score"]["correct"], f"{arm} {t}: correct re-derived ({st}, {secs:.1f} s)",
                      f"recorded {r['score']['correct']}, re-derived {c}", warn=(st == "timeout" or r["score"]["status"] == "timeout"))
            correct += c
        s = summ["arms"][arm]["summary"]
        el = sum(r["elapsed_seconds"] for r in recs)
        rep.check(s["requested"] == len(dev_ids) and s["correct"] == correct and math.isclose(s["quality"], correct / len(dev_ids))
                  and math.isclose(s["elapsed_seconds_total"], el, rel_tol=1e-12)
                  and s["cost_seconds_per_correct"] == (el / correct if correct else None)
                  and s["answered"] == sum(r["final"]["kind"] == "sql" for r in recs), f"{arm}: summary recomputed",
                  f"{s['correct']}/{s['requested']} vs {correct}/{len(dev_ids)}")
        timeouts = [x for x in statuses if x[1] == "timeout" or x[3] == "timeout"]
        print(f"   {arm}: re-derived {correct}/{len(dev_ids)} correct; scorer statuses (own, recorded): "
              f"{sorted({(x[1], x[3]) for x in statuses})}; timeouts {timeouts}")

    # --- (d) CPU re-decode -------------------------------------------------------------------------------------------
    if redecode:
        redecode_check(run, rep, summ, arms, dev_ids, runs_root)
    return rep


def redecode_check(run: Path, rep: VB.Report, summ: dict, arms: list, dev_ids: list, runs_root: Path) -> None:
    import os
    os.environ.setdefault("TI_TNN_SCALING_RUNS", str(runs_root))
    os.environ.setdefault("TI_TNN_SCALING_LINEAR_KERNELS", "reference")
    sys.path.insert(0, str(HERE.parents[1]))  # research/ti_tnn, for scaling.* (decoding only)
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from scaling import encode, infer_controller, train
    from scaling import contracts as C
    ident = summ["backbone"]
    tok = AutoTokenizer.from_pretrained(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/" + ident["revision"]), local_files_only=True)
    bb = encode.Backbone.__new__(encode.Backbone)        # no model: the recorded GPU identity and the real tokenizer
    bb.tokenizer, bb.device, bb.hidden_size = tok, "cpu", ident["hidden_size"]
    bb.identity = lambda: dict(ident)

    def no_forward(token_ids):
        raise RuntimeError("cache miss: the CPU re-decode would need a backbone forward")
    bb.states = no_forward
    for arm in arms:
        recs = {r["task_id"]: r for r in (json.loads(x) for x in (run / arm / "records.jsonl").read_text().splitlines() if x.strip())}
        for t in dev_ids:   # the cached tensor must be the one the timed GPU decode used
            key = recs[t]["encode_cost"]["cache_key"]
            st = load_file(str(C.CACHE / "encodings" / key[:2] / f"{key}.safetensors"))["states"]
            h = hashlib.sha256(st.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            rep.check(h == recs[t]["states_sha256"], f"{arm} {t}: cached states = the states the GPU decode used")
        ck = summ["arms"][arm]["checkpoint"]
        model = train.model_from_checkpoint(train.load_checkpoint(run / arm / "checkpoints" / Path(ck["path"]).name), "cpu").eval()
        evals = infer_controller.evaluate(model, dev_ids, bb, score=False, use_cache=True, log=None)
        diff = []
        for ev in evals:
            r = recs[ev["task_id"]]
            keys_ok = (ev.get("encode_cost") or {}).get("cache_key") == r["encode_cost"]["cache_key"]
            same_actions = ev.get("actions") == r["attempts"][0].get("actions")
            same_sql = ev["final"].get("sql") == r["final"]["sql"] and ev["final"].get("kind") == r["final"]["kind"]
            if not (keys_ok and same_actions and same_sql):
                first = next((i for i, (x, y) in enumerate(zip(ev.get("actions") or [], r["attempts"][0].get("actions") or [])) if x != y), None)
                diff.append((ev["task_id"], "key" if not keys_ok else "", f"first differing action {first}", ev["final"].get("reason")))
        rep.check(not diff, f"{arm}: CPU re-decode reproduces every GPU decision (actions and final SQL, 16 requests)", f"{diff}")
        print(f"   {arm}: CPU re-decode: {16 - len(diff)}/16 identical in actions and final SQL")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run", type=Path)
    p.add_argument("--repo", type=Path, default=None)
    p.add_argument("--runs-root", type=Path, default=None)
    p.add_argument("--no-rehash", action="store_true")
    p.add_argument("--redecode", action="store_true")
    a = p.parse_args(argv)
    repo = a.repo or a.run.resolve().parents[3]
    rep = verify(a.run, repo, a.runs_root or a.run.resolve().parents[1], not a.no_rehash, a.redecode)
    print(f"== {rep.run}: {'FAIL' if rep.fails else 'PASS'} ({len(rep.passes)} checks passed, {len(rep.fails)} failed, {len(rep.warns)} warnings)")
    for x in rep.fails:
        print("   FAIL", x)
    for x in rep.warns:
        print("   WARN", x)
    return int(bool(rep.fails))


if __name__ == "__main__":
    sys.exit(main())
