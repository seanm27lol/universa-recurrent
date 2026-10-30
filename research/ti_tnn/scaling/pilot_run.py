"""The pilot's commands (registered in `cli`): encode, feasibility check, train, evaluate, analyze.

**Example of the order** (each GPU step is one ledger reservation under the absolute 8-hour cap):
1. `pilot-feasibility` measures every unit cost on the final code and projects the whole pilot against the remaining
   cap. Every context it encodes is anchored and reused. If the projection does not fit, the pilot stops here.
2. `pilot-encode --what dev`, then `pilot-encode --what train --range a:b` chunk by chunk, write the remaining
   encodings. Each entry records its sha256 in the file and in the run's `entries.jsonl` (the anchor).
3. `pilot-train --seed 100` (then 101) prepares the 4,741 examples on the CPU, from anchored entries only. It then
   trains T4 and c4_params, one reservation each, for 1,500 updates with lazy states and TF32, checkpointing every
   250.
4. `pilot-eval --seed 100` (then 101) evaluates each arm's final checkpoint (update 1,499 only) on the 64. It times a
   fresh backbone forward per request, then reconstructs from a second reload on the anchored cache. The fresh states
   must equal the anchored ones, and be identical across the arms.
5. `baseline --arm L --subset 64` (CPU), then `pilot-analyze`.

Every GPU command refuses to start unless `docs/PROTOCOL_pilot.md` matches its frozen hash in
`pins/pilot_protocol_v1.json`.
"""
from __future__ import annotations

import json
import shutil
import statistics
import time
from pathlib import Path

from . import contracts as C
from . import identity, ledger
from . import pilot as P

PROTOCOL_PIN = C.PINS / "pilot_protocol_v1.json"
DEVICE = "cuda"  # the tests run the whole pipeline on the CPU with a tiny backbone


class PilotStop(RuntimeError):
    """A condition of the frozen protocol failed; the pilot stops and reports."""


def protocol_identity() -> dict:
    """The frozen protocol's identity, or PilotStop if it is not frozen or the file changed since."""
    if not PROTOCOL_PIN.exists():
        raise PilotStop("the pilot protocol is not frozen (no pins/pilot_protocol_v1.json)")
    pin = json.loads(PROTOCOL_PIN.read_text())
    digest = identity.sha256_file(P.PROTOCOL)
    if digest != pin["sha256"]:
        raise PilotStop(f"docs/PROTOCOL_pilot.md is {digest[:12]}, the frozen hash is {pin['sha256'][:12]}")
    name = str(P.PROTOCOL.relative_to(C.REPO)) if P.PROTOCOL.is_relative_to(C.REPO) else str(P.PROTOCOL)
    return {"file": name, "sha256": digest, "frozen_at": pin.get("frozen_at")}


def _cli():
    from . import cli
    return cli


def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(identity.model_snapshot_dir("B4"), local_files_only=True)


def _reader(anchors_identity, tokenizer):
    from . import encode
    if anchors_identity is None:
        raise PilotStop("no encoding anchors yet")
    return encode.Backbone.cache_reader(anchors_identity, tokenizer)


def _load_backbone():
    from . import encode
    from .runtime import Runtime
    rt = Runtime("B4")
    return rt, encode.Backbone.from_pinned(device=DEVICE, model=rt.model, tokenizer=rt.tokenizer)


def _encode(bb, ids: list, run_dir: Path, log=print) -> dict:
    """Encode `ids` into the pilot cache, appending one anchor per entry to run_dir/entries.jsonl."""
    from . import data, encode, retrieval
    from . import schema as schema_mod
    cli = _cli()
    anchors, ident = P.load_anchors()
    if ident is not None and ident != bb.identity():
        raise PilotStop("this backbone's identity differs from the existing anchors'; the cache would not be reused")
    _write_backbone(run_dir, bb)
    seconds, misses = [], 0
    for n, tid in enumerate(ids):
        task = data.task(tid)
        t0 = time.perf_counter()
        enc = encode.encode_request(bb, task, schema_mod.load_schema(task.db_id), retrieval.retrieve(task),
                                    cache_root=encode.PILOT_CACHE, require_verified=True)
        dt = time.perf_counter() - t0
        old = anchors.get(tid)
        if old is not None and old["states_sha256"] != enc.cost["states_sha256"]:
            raise PilotStop(f"{tid}: re-read states differ from their anchor")
        cli._append(run_dir / "entries.jsonl", {"task_id": tid, "cache_key": enc.cost["cache_key"],
                                                "states_sha256": enc.cost["states_sha256"],
                                                "tokens": enc.cost["tokens"], "cache_hit": enc.cost["cache_hit"],
                                                "forward_seconds": enc.cost["forward_seconds"], "seconds": dt})
        if not enc.cost["cache_hit"]:
            misses += 1
            seconds.append(dt)
        if log and (n + 1) % 100 == 0:
            log(f"encoded {n + 1}/{len(ids)}")
    return {"requested": len(ids), "encoded": misses, "already_anchored": len(ids) - misses,
            "seconds_per_new_context": statistics.mean(seconds) if seconds else None,
            "seconds_total_new": sum(seconds)}


def _write_backbone(run_dir: Path, bb) -> None:
    path = run_dir / "backbone.json"
    if not path.exists():
        path.write_text(json.dumps(bb.identity(), indent=1, sort_keys=True) + "\n")


def _context_lengths(ids: list, tokenizer) -> dict:
    from . import data, encode, retrieval
    from . import schema as schema_mod
    out = {}
    for tid in ids:
        task = data.task(tid)
        _ctx, _ser, token_ids, _off = encode.fit_context(task, schema_mod.load_schema(task.db_id),
                                                         retrieval.retrieve(task), tokenizer,
                                                         C.Limits().max_context_tokens)
        out[tid] = len(token_ids)
    return out


def prepare(ids: list, tokenizer, log=print) -> tuple:
    """The examples, from anchored, verified pilot-cache entries only, read on the CPU (no model)."""
    from . import encode, train
    anchors, ident = P.load_anchors()
    missing = [t for t in ids if t not in anchors]
    if missing:
        raise PilotStop(f"{len(missing)} training ids have no encoding anchor (first: {missing[:3]})")
    reader = _reader(ident, tokenizer)
    t0 = time.perf_counter()
    prepared, skipped, costs = train.prepare_examples(ids, reader, cache_root=encode.PILOT_CACHE, lazy=True,
                                                      require_hit=True, require_verified=True, anchors=anchors,
                                                      log=log)
    if skipped:
        raise PilotStop(f"{len(skipped)} pinned training ids could not be prepared: {list(skipped.items())[:3]}")
    return prepared, ident, time.perf_counter() - t0


# --- commands ----------------------------------------------------------------------------------------------------------
def encode_cmd(args) -> None:
    cli = _cli()
    proto = protocol_identity()
    if args.what == "dev":
        ids, label = P.eval_ids(), "dev"
    else:
        lo, hi = (int(x) for x in args.range.split(":"))
        ids, label = P.train_ids()[lo:hi], f"train_{lo}-{hi}"
    run_dir = P.ROOT / "encode" / f"{cli._stamp()}_{label}"
    config = {"command": "pilot-encode", "what": args.what, "range": args.range, "ids": ids,
              "cache_root": "PILOT_CACHE", "protocol": proto}
    identity.write_manifest(run_dir, config=config, require_clean=not args.allow_dirty,
                            extra={"model_identity": identity.model_identity("B4")})
    with ledger.Reservation(f"pilot-encode {label}", projected_hours=args.projected_hours,
                            extra={"run_dir": str(run_dir)}) as res:
        rt, bb = _load_backbone()
        res.check()
        out = _encode(bb, ids, run_dir)
        out["reservation"] = {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}
    cli._write(run_dir / "summary.json", out)
    print(json.dumps(out, indent=1, default=str))


def _fit_and_time(arm, ids, prepared, ident, run_dir, seed, updates, checkpoint_every, log=None) -> dict:
    from . import train
    t0 = time.perf_counter()
    result = train.train_arm(arm, ids, None, run_dir, updates=updates, seed=seed, device=DEVICE, prepared=prepared,
                             matmul_precision=P.MATMUL_PRECISION, checkpoint_every=checkpoint_every,
                             extra_config={"backbone": ident, "pilot_protocol": protocol_identity()}, log=log)
    total = time.perf_counter() - t0
    logs = [json.loads(x) for x in Path(result["log"]).read_text().splitlines()]
    walls = [r["wall_seconds"] for r in logs]
    saves = [c["save_seconds"] for c in result["checkpoints"]]
    return {"result": result, "total_seconds": total, "update_seconds": walls,
            "update_seconds_after_first": statistics.mean(walls[1:]) if len(walls) > 1 else walls[0],
            "checkpoint_save_seconds": saves, "prefetch_wait_mean": statistics.mean(r["prefetch_wait_seconds"]
                                                                                   for r in logs)}


def feasibility_cmd(args) -> None:
    """Measure on the final code: encoding (with integrity records), CPU preparation, both arms' updates on
    representative context lengths with file-backed, integrity-checked reads, checkpoint saves, timed evaluation and
    reconstruction. Then project the complete pilot against the remaining cap."""
    import torch

    from . import encode, infer_controller, train
    cli = _cli()
    proto = protocol_identity()
    tok = _tokenizer()
    pool = P.train_ids()
    t0 = time.perf_counter()
    lengths = _context_lengths(pool, tok)
    length_seconds = time.perf_counter() - t0
    sample = P.feasibility_sample(pool, lengths, args.sample)
    dev = P.eval_ids()[:args.dev]
    run_dir = P.ROOT / "feasibility" / cli._stamp()
    enc_dir = P.ROOT / "encode" / f"{run_dir.name}_feasibility"
    config = {"command": "pilot-feasibility", "sample": sample, "dev": dev, "updates": args.updates,
              "checkpoint_every": args.checkpoint_every, "seed": args.seed, "protocol": proto,
              "pool_tokens": {"mean": statistics.mean(lengths.values()), "max": max(lengths.values()),
                              "sample_mean": statistics.mean(lengths[t] for t in sample)}}
    identity.write_manifest(run_dir, config=config, require_clean=not args.allow_dirty,
                            extra={"model_identity": identity.model_identity("B4")})
    identity.write_manifest(enc_dir, config={"command": "pilot-encode", "what": "feasibility",
                                             "ids": sample + dev, "protocol": proto},
                            require_clean=not args.allow_dirty, extra={"feasibility_run": str(run_dir)})
    out = {"run_dir": str(run_dir), "context_lengths_seconds": length_seconds, "pool": config["pool_tokens"]}
    with ledger.Reservation("pilot-feasibility", projected_hours=args.projected_hours,
                            extra={"run_dir": str(run_dir)}) as res:
        t_load = time.perf_counter()
        rt, bb = _load_backbone()
        out["load_seconds"] = time.perf_counter() - t_load
        res.check()
        out["encode"] = _encode(bb, sample + dev, enc_dir, log=None)
        res.check()
        prepared, ident, prep_seconds = prepare(sample, tok, log=None)
        out["prepare_seconds_per_example"] = prep_seconds / len(sample)
        out["arms"] = {}
        for arm in P.ARMS:
            res.check()
            m = _fit_and_time(arm, sample, prepared, ident, run_dir / arm, args.seed, args.updates,
                              args.checkpoint_every)
            final = m["result"]["checkpoints"][-1]
            model = train.model_from_checkpoint(train.load_checkpoint(final["path"]), DEVICE)
            t1 = time.perf_counter()
            timed = infer_controller.evaluate(model, dev, bb, score=False, use_cache=False,
                                              matmul_precision=P.MATMUL_PRECISION, log=None)
            t2 = time.perf_counter()
            again = infer_controller.evaluate(train.model_from_checkpoint(train.load_checkpoint(final["path"]),
                                                                          DEVICE),
                                              dev, bb, score=False, use_cache=True, require_hit=True,
                                              require_verified=True, cache_root=encode.PILOT_CACHE,
                                              matmul_precision=P.MATMUL_PRECISION, log=None)
            t3 = time.perf_counter()
            out["arms"][arm] = {"update_seconds_after_first": m["update_seconds_after_first"],
                                "update_seconds_max": max(m["update_seconds"]),
                                "checkpoint_save_seconds_mean": statistics.mean(m["checkpoint_save_seconds"]),
                                "fit_total_seconds": m["total_seconds"], "updates": args.updates,
                                "prefetch_wait_mean": m["prefetch_wait_mean"],
                                "eval_seconds_per_request": (t2 - t1) / len(dev),
                                "reconstruction_seconds_per_request": (t3 - t2) / len(dev),
                                "reconstruction_equal": all(a.get("actions") == b.get("actions")
                                                            for a, b in zip(timed, again))}
            for ckpt in (run_dir / arm / "checkpoints").glob("*.pt"):  # measurements only
                ckpt.unlink()
        out["reservation"] = {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}
    out["projection"] = project(out, len(pool), len(P.eval_ids()), len(sample), len(dev))
    cli._write(run_dir / "summary.json", out)
    print(json.dumps({k: out[k] for k in ("encode", "arms", "projection")}, indent=1, default=str))
    if not out["projection"]["fits"]:
        raise PilotStop(f"the pilot does not fit: shortfall {out['projection']['shortfall_hours']:.3f} h")


def project(m: dict, n_pool: int, n_eval: int, n_sample: int, n_dev: int) -> dict:
    """The complete pilot's GPU hours from the measured unit costs, against the remaining cap (the absolute 8 h)."""
    used = ledger.usage()["used_hours"]
    per_ctx = m["encode"]["seconds_per_new_context"]
    load = m["load_seconds"]
    remaining_contexts = (n_pool - n_sample) + (n_eval - n_dev)
    encode_chunks = 4
    items = {
        "encode remaining contexts": remaining_contexts * per_ctx / 3600 + (encode_chunks + 1) * load / 3600,
        "train 2 seeds x 2 arms": sum(
            2 * (P.UPDATES * a["update_seconds_after_first"]
                 + (P.UPDATES // P.CHECKPOINT_EVERY) * a["checkpoint_save_seconds_mean"]) for a in m["arms"].values())
        / 3600,
        "evaluate and reconstruct 4 final checkpoints": (2 * load + sum(
            2 * n_eval * (a["eval_seconds_per_request"] + a["reconstruction_seconds_per_request"])
            for a in m["arms"].values())) / 3600,
    }
    needed = sum(items.values())
    remaining = ledger.CAP_HOURS - used
    return {"items_hours": items, "needed_hours": needed, "used_hours": used, "cap_hours": ledger.CAP_HOURS,
            "remaining_hours": remaining, "fits": needed <= remaining, "slack_hours": remaining - needed,
            "shortfall_hours": max(0.0, needed - remaining),
            "basis": "measured unit costs on the final code; training takes the larger arm's update time per arm; "
                     "preparation runs on the CPU outside reservations"}


def train_cmd(args) -> None:
    cli = _cli()
    proto = protocol_identity()
    if args.seed not in P.SEEDS:
        raise PilotStop(f"seed {args.seed} is not one of the protocol's {P.SEEDS}")
    tok = _tokenizer()
    ids = P.train_ids()
    prepared, ident, prep_seconds = prepare(ids, tok)
    print(json.dumps({"prepared": len(prepared), "prepare_seconds": prep_seconds}), flush=True)
    for arm in args.arms:
        run_dir = P.ROOT / "train" / f"{arm}_s{args.seed}"
        checkpoints = sorted((run_dir / "checkpoints").glob("ckpt_u*.pt")) if run_dir.exists() else []
        if run_dir.exists() and not args.resume:
            raise PilotStop(f"{run_dir} exists; pass --resume to continue it from its last checkpoint")
        resume = str(checkpoints[-1]) if (args.resume and checkpoints) else None
        attempt = run_dir / "attempts" / cli._stamp()
        identity.write_manifest(attempt, config={"command": "pilot-train", "arm": arm, "seed": args.seed,
                                                 "updates": P.UPDATES, "checkpoint_every": P.CHECKPOINT_EVERY,
                                                 "matmul_precision": P.MATMUL_PRECISION,
                                                 "train_pin_sha256": identity.sha256_file(C.PINS / "pilot_train_v1.json"),
                                                 "ids_sha256": identity.content_identity(ids),
                                                 "resume_from": resume, "protocol": proto, "backbone": ident},
                                require_clean=not args.allow_dirty)
        with ledger.Reservation(f"pilot-train {arm} s{args.seed}", projected_hours=args.projected_hours,
                                extra={"run_dir": str(run_dir), "attempt": str(attempt)}) as res:
            res.check()
            from . import train
            t0 = time.perf_counter()
            result = train.train_arm(arm, ids, None, run_dir, updates=P.UPDATES, seed=args.seed, device=DEVICE,
                                     prepared=prepared, resume_from=resume, matmul_precision=P.MATMUL_PRECISION,
                                     checkpoint_every=P.CHECKPOINT_EVERY,
                                     extra_config={"backbone": ident, "pilot_protocol": proto}, log=print)
            out = {"arm": arm, "seed": args.seed, "seconds": time.perf_counter() - t0, "resumed_from": resume,
                   "checkpoints": result["checkpoints"], "log": result["log"],
                   "config_identity": result["config_identity"],
                   "reservation": {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}}
        cli._write(attempt / "summary.json", out)
        print(json.dumps({k: out[k] for k in ("arm", "seed", "seconds", "reservation")}, default=str), flush=True)


def final_checkpoint(run_dir: Path) -> tuple:
    """(path, recorded checkpoint entry) of a completed training run's final checkpoint (update 1,499). Anything else
    (missing or extra updates in the log, no final checkpoint, a digest that differs from the record) stops."""
    from . import train
    log = [json.loads(x) for x in (run_dir / "train_log.jsonl").read_text().splitlines() if x.strip()]
    if [r["update"] for r in log] != list(range(P.UPDATES)):
        raise PilotStop(f"{run_dir}: the training log does not hold updates 0..{P.UPDATES - 1} once each")
    finals = sorted((run_dir / "checkpoints").glob(f"ckpt_u{P.UPDATES - 1:06d}_*.pt"))
    if len(finals) != 1:
        raise PilotStop(f"{run_dir}: {len(finals)} final checkpoints")
    recorded = None
    for summary in sorted((run_dir / "attempts").glob("*/summary.json")):
        for c in json.loads(summary.read_text())["checkpoints"]:
            if Path(c["path"]).name == finals[0].name:
                recorded = c
    if recorded is None:
        raise PilotStop(f"{run_dir}: the final checkpoint is not recorded by any attempt")
    ckpt = train.load_checkpoint(finals[0])
    if ckpt["update"] != P.UPDATES - 1 or train.tensor_digest(ckpt["model"]) != recorded["model_digest"]:
        raise PilotStop(f"{run_dir}: the final checkpoint does not match its record")
    return finals[0], recorded


def eval_cmd(args) -> None:
    import torch

    from . import data, encode, infer_controller, scorer, train
    from . import records as R
    cli = _cli()
    proto = protocol_identity()
    finals = {arm: final_checkpoint(P.ROOT / "train" / f"{arm}_s{args.seed}") for arm in P.ARMS}
    anchors, ident = P.load_anchors()
    ids = P.eval_ids()
    if any(t not in anchors for t in ids):
        raise PilotStop("the evaluation questions are not all encoded and anchored")
    run_dir = P.ROOT / "eval" / f"s{args.seed}_{cli._stamp()}"
    manifest = identity.write_manifest(run_dir, config={
        "command": "pilot-eval", "seed": args.seed, "ids": ids, "protocol": proto, "backbone": ident,
        "checkpoints": {a: {"path": str(p), **{k: c[k] for k in ("sha256", "model_digest", "update")}}
                        for a, (p, c) in finals.items()},
        "matmul_precision": P.MATMUL_PRECISION}, require_clean=not args.allow_dirty,
        extra={"model_identity": identity.model_identity("B4")})
    out = {"run_dir": str(run_dir), "arms": {}}
    evals = {}
    cli._warm_databases([data.task(t).db_id for t in ids])
    with ledger.Reservation(f"pilot-eval s{args.seed}", projected_hours=args.projected_hours,
                            extra={"run_dir": str(run_dir)}) as res:
        rt, bb = _load_backbone()
        if bb.identity() != ident:
            raise PilotStop("the loaded backbone's identity differs from the anchors'; reconstruction would miss")
        for arm, (path, _c) in finals.items():
            res.check()
            model = train.model_from_checkpoint(train.load_checkpoint(path), DEVICE)
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
            timed = infer_controller.evaluate(model, ids, bb, score=False, use_cache=False,
                                              matmul_precision=P.MATMUL_PRECISION, log=None)
            peak = torch.cuda.max_memory_allocated() if DEVICE == "cuda" else None
            again = infer_controller.evaluate(train.model_from_checkpoint(train.load_checkpoint(path), DEVICE), ids,
                                              bb, score=False, use_cache=True, require_hit=True,
                                              require_verified=True, cache_root=encode.PILOT_CACHE,
                                              matmul_precision=P.MATMUL_PRECISION, log=None)
            evals[arm] = (timed, again, peak)
        out["reservation"] = {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}
    states_cross = [t for t, a, b in zip(ids, evals["T4"][0], evals["c4_params"][0])
                    if a["states_sha256"] != b["states_sha256"]]
    for arm, (timed, again, peak) in evals.items():
        recon = [{"task_id": a["task_id"], "same_actions": a.get("actions") == b.get("actions"),
                  "same_final": a["final"] == b["final"],
                  "fresh_states_equal_anchor": a["states_sha256"] == anchors[a["task_id"]]["states_sha256"]}
                 for a, b in zip(timed, again)]
        label = P.LABELS[arm]
        arm_config = identity.content_identity({"run": manifest["config_identity"], "arm": arm,
                                                "model_digest": finals[arm][1]["model_digest"]})
        path = run_dir / arm / "records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        errors = []
        for ev in timed:
            snap = data.snapshot(ev["db_id"])
            score = scorer.score(snap, ev["final"]["sql"], data.gold(ev["task_id"]).sql)
            rec = cli._controller_record(ev, label, arm_config, snap, score)
            try:
                R.validate(rec)
            except R.RecordError as e:
                errors.append(str(e))
            cli._append(path, rec)
        arm_out = {"label": label, "config_identity": arm_config, "eval_peak_memory_bytes": peak,
                   "reconstruction": recon, "record_errors": errors,
                   "reconstruction_all_equal": all(r["same_actions"] and r["same_final"] for r in recon),
                   "fresh_states_all_equal_anchor": all(r["fresh_states_equal_anchor"] for r in recon)}
        try:
            arm_out["summary"] = R.summarize(R.load(path), ids)
        except R.RecordError as e:
            arm_out["summary_error"] = str(e)
        out["arms"][arm] = arm_out
    out["states_differ_across_arms"] = states_cross
    cli._write(run_dir / "summary.json", out)
    print(json.dumps({a: {k: o[k] for k in ("reconstruction_all_equal", "fresh_states_all_equal_anchor")}
                      for a, o in out["arms"].items()} | {"states_differ_across_arms": states_cross}, default=str))
    if states_cross or not all(o["reconstruction_all_equal"] for o in out["arms"].values()):
        raise PilotStop("the evaluation's integrity checks failed; see summary.json")


def analyze_cmd(args) -> None:
    from . import data
    from . import records as R
    cli = _cli()
    proto = protocol_identity()
    ids = P.eval_ids()
    correct = {}
    for d in args.evals:
        d = Path(d)
        seed = json.loads((d / "manifest.json").read_text())["config"]["seed"]
        for arm in P.ARMS:
            recs = R.load(d / arm / "records.jsonl")
            R.summarize(recs, ids)
            correct[(arm, seed)] = {r["task_id"]: r["score"]["correct"] for r in recs}
    l_recs = R.load(Path(args.l) / "records.jsonl")
    R.summarize(l_recs, ids)
    l_correct = {r["task_id"]: r["score"]["correct"] for r in l_recs}
    db_of = {t: data.task(t).db_id for t in ids}
    result = P.analyze(correct, l_correct, db_of, ids)
    rep = P.representability(ids, _tokenizer())
    result["gold_representability"] = {
        "representable": sum(v == "representable" for v in rep.values()), "of": len(ids),
        "reasons": dict(sorted(__import__("collections").Counter(v.split(":")[0] for v in rep.values()
                                                                  if v != "representable").items())),
        "correct_on_unrepresentable": {f"{a}/{s}": sum(correct[(a, s)][t] for t in ids if rep[t] != "representable")
                                       for a in P.ARMS for s in P.SEEDS},
        "per_question": rep}
    pin = json.loads((C.PINS / "pilot_train_v1.json").read_text())
    excluded = pin["training_pool_questions"] - pin["supervisable"]
    result["training_exclusion_rate"] = {"excluded": excluded, "pool": pin["training_pool_questions"],
                                         "rate": excluded / pin["training_pool_questions"],
                                         "note": "training-pool questions without a supervisable gold program; "
                                                 "an exclusion rate for training, not an evaluation ceiling"}
    result["inputs"] = {"evals": [str(d) for d in args.evals], "L": str(args.l), "protocol": proto}
    run_dir = P.ROOT / "analysis" / cli._stamp()
    identity.write_manifest(run_dir, config={"command": "pilot-analyze", **result["inputs"]},
                            require_clean=not args.allow_dirty)
    cli._write(run_dir / "analysis.json", result)
    print(json.dumps({k: result[k] for k in ("counts", "cleared_L_in_both_seeds", "differences", "rule",
                                              "exploratory_sign_flip")}, indent=1))
