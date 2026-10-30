"""The pilot's commands (registered in `cli`): feasibility check, encode, train, evaluate, analyze.

**Example of the order** (each GPU step is one ledger reservation under the absolute 8-hour cap):
1. `pilot-feasibility` measures every unit cost on the final code and projects the whole pilot against the remaining
   cap. Every context it encodes is anchored and reused. If the projection does not fit, the pilot stops here.
2. `pilot-encode --what dev` encodes the 64. It first recomputes, in this new process, the feasibility run's 8
   development contexts and compares them with their anchors: the cross-process determinism the evaluation gate
   relies on, checked before any training. Then `pilot-encode --what train --range a:b` runs over the four chunks.
3. `pilot-train --seed 100` (then 101) prepares the 4,741 examples on the CPU, from anchored entries only. It then
   trains T4 and c4_params, one reservation each, for 1,500 updates with lazy states and TF32, checkpointing every
   250. Each checkpoint is recorded in `checkpoints.jsonl` the moment it is saved.
4. `pilot-eval --seed 100` (then 101) evaluates each arm's final checkpoint (update 1,499 only) on the 64. It times a
   fresh backbone forward per request, then reconstructs from a second reload on the anchored cache. The fresh states
   must equal the anchored ones and be identical across the arms, or the pilot stops.
5. `baseline --arm L --subset 64` (CPU), then `pilot-analyze`. That counts exactly one passing evaluation per seed
   and reports every evaluation run.

Every pilot command refuses to start unless `docs/PROTOCOL_pilot.md` matches its frozen hash in
`pins/pilot_protocol_v1.json`.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

from . import contracts as C
from . import identity, ledger
from . import pilot as P

PROTOCOL_PIN = C.PINS / "pilot_protocol_v1.json"
DEVICE = "cuda"  # the tests run the whole pipeline on the CPU with a tiny backbone
TRAIN_CHUNKS = ("0:1200", "1200:2400", "2400:3600", "3600:4741")  # the protocol's encoding reservations


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


def anchors(proto: dict) -> tuple:
    """The accepted anchors: only from encoding runs on a clean tree under this frozen protocol."""
    try:
        found, ident, ignored = P.load_anchors(protocol_sha256=proto["sha256"])
    except ValueError as e:
        raise PilotStop(str(e)) from e
    for note in ignored:
        print(json.dumps({"anchors_ignored": note}), flush=True)
    return found, ident


def _cli():
    from . import cli
    return cli


def _new_dir(parent: Path, name: str, prefix: str = "") -> Path:
    """parent/<prefix><stamp>_name, made unique if a run of the same second exists (a manifest is never
    overwritten)."""
    stamp, k = prefix + _cli()._stamp(), 0
    while True:
        tag = stamp + (f".{k}" if k else "")
        path = parent / (f"{tag}_{name}" if name else tag)
        if not path.exists():
            return path
        k += 1


def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(identity.model_snapshot_dir("B4"), local_files_only=True)


def _load_backbone():
    from . import encode
    from .runtime import Runtime
    rt = Runtime("B4")
    return rt, encode.Backbone.from_pinned(device=DEVICE, model=rt.model, tokenizer=rt.tokenizer)


def _write_backbone(run_dir: Path, bb) -> None:
    path = run_dir / "backbone.json"
    if not path.exists():
        path.write_text(json.dumps(bb.identity(), indent=1, sort_keys=True) + "\n")


def _encode(bb, ids: list, run_dir: Path, proto: dict, recheck_anchored: bool = False, log=print) -> dict:
    """Encode `ids` into the pilot cache, appending one anchor per entry to run_dir/entries.jsonl (fsynced).
    - A new entry is anchored with the sha256 recorded when it was written.
    - An existing entry that no accepted anchor covers (e.g. a crash between the write and the append) is recomputed
      with a fresh forward, and anchored only if equal.
    - With `recheck_anchored`, already anchored entries are recomputed too and must be equal (cross-process
      determinism); a mismatch stops the pilot."""
    from . import data, encode, retrieval
    from . import schema as schema_mod
    cli = _cli()
    known, ident = anchors(proto)
    if ident is not None and ident != bb.identity():
        raise PilotStop("this backbone's identity differs from the existing anchors'; the cache would not be reused")
    _write_backbone(run_dir, bb)
    new_seconds, rechecked, mismatched = [], [], []
    for n, tid in enumerate(ids):
        task = data.task(tid)
        schema, examples = schema_mod.load_schema(task.db_id), retrieval.retrieve(task)
        t0 = time.perf_counter()
        enc = encode.encode_request(bb, task, schema, examples, cache_root=encode.PILOT_CACHE,
                                    require_verified=True)
        dt = time.perf_counter() - t0
        digest, how = enc.cost["states_sha256"], "written" if not enc.cost["cache_hit"] else "anchored"
        old = known.get(tid)
        if enc.cost["cache_hit"] and (old is None or recheck_anchored):
            fresh = encode.encode_request(bb, task, schema, examples, use_cache=False)
            fresh_digest = encode.states_sha256(fresh.states)
            rechecked.append(tid)
            if fresh_digest != digest:
                mismatched.append(tid)
            how = "recomputed and equal" if fresh_digest == digest else "recomputed and DIFFERENT"
        if old is not None and old["states_sha256"] != digest:
            raise PilotStop(f"{tid}: the entry's recorded sha256 differs from its anchor")
        if mismatched:
            cli._write(run_dir / "summary.json", {"stopped": "recomputed states differ", "mismatched": mismatched,
                                                   "rechecked": rechecked})
            raise PilotStop(f"{tid}: a fresh forward differs from the stored states (cross-process determinism or "
                            f"a corrupt entry); the pilot stops")
        cli._append(run_dir / "entries.jsonl", {"task_id": tid, "cache_key": enc.cost["cache_key"],
                                                "states_sha256": digest, "tokens": enc.cost["tokens"],
                                                "cache_hit": enc.cost["cache_hit"], "how": how,
                                                "forward_seconds": enc.cost["forward_seconds"], "seconds": dt})
        if not enc.cost["cache_hit"]:
            new_seconds.append(dt)
        if log and (n + 1) % 100 == 0:
            log(f"encoded {n + 1}/{len(ids)}")
    return {"requested": len(ids), "encoded": len(new_seconds), "already_present": len(ids) - len(new_seconds),
            "rechecked": len(rechecked), "rechecked_equal": len(rechecked) - len(mismatched),
            "seconds_per_new_context": statistics.mean(new_seconds) if new_seconds else None,
            "seconds_total_new": sum(new_seconds)}


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


def prepare(ids: list, tokenizer, proto: dict, log=print) -> tuple:
    """The examples, from anchored, verified pilot-cache entries only, read on the CPU (no model)."""
    from . import encode, train
    known, ident = anchors(proto)
    missing = [t for t in ids if t not in known]
    if missing:
        raise PilotStop(f"{len(missing)} training ids have no encoding anchor (first: {missing[:3]})")
    if ident is None:
        raise PilotStop("no encoding anchors yet")
    reader = encode.Backbone.cache_reader(ident, tokenizer)
    t0 = time.perf_counter()
    prepared, skipped, _costs = train.prepare_examples(ids, reader, cache_root=encode.PILOT_CACHE, lazy=True,
                                                       require_hit=True, require_verified=True, anchors=known,
                                                       log=log)
    if skipped:
        raise PilotStop(f"{len(skipped)} pinned training ids could not be prepared: {list(skipped.items())[:3]}")
    return prepared, ident, time.perf_counter() - t0


def _drop_from_page_cache(paths) -> None:
    """Ask the kernel to drop these files' cached pages (no root needed), so the next reads come from the disk."""
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


# --- commands ----------------------------------------------------------------------------------------------------------
def encode_cmd(args) -> None:
    cli = _cli()
    proto = protocol_identity()
    if args.what == "dev":
        ids, label = P.eval_ids(), "dev"
    else:
        if args.range not in TRAIN_CHUNKS:
            raise PilotStop(f"--range must be one of the protocol's chunks {TRAIN_CHUNKS}")
        lo, hi = (int(x) for x in args.range.split(":"))
        ids, label = P.train_ids()[lo:hi], f"train_{lo}-{hi}"
    run_dir = _new_dir(P.ROOT / "encode", label)
    config = {"command": "pilot-encode", "what": args.what, "range": args.range, "ids": ids,
              "cache_root": "PILOT_CACHE", "protocol": proto, "recheck_anchored": args.what == "dev"}
    identity.write_manifest(run_dir, config=config, require_clean=not args.allow_dirty,
                            extra={"model_identity": identity.model_identity("B4")})
    with ledger.Reservation(f"pilot-encode {label}", projected_hours=args.projected_hours,
                            extra={"run_dir": str(run_dir)}) as res:
        _rt, bb = _load_backbone()
        res.check()
        out = _encode(bb, ids, run_dir, proto, recheck_anchored=args.what == "dev")
        out["reservation"] = {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}
    cli._write(run_dir / "summary.json", out)
    print(json.dumps(out, indent=1, default=str))


def _fit_and_time(arm, ids, prepared, ident, proto, run_dir, seed, updates, checkpoint_every, log=None) -> dict:
    from . import train
    t0 = time.perf_counter()
    result = train.train_arm(arm, ids, None, run_dir, updates=updates, seed=seed, device=DEVICE, prepared=prepared,
                             matmul_precision=P.MATMUL_PRECISION, checkpoint_every=checkpoint_every,
                             extra_config={"backbone": ident, "pilot_protocol": proto}, log=log)
    total = time.perf_counter() - t0
    logs = [json.loads(x) for x in Path(result["log"]).read_text().splitlines()]
    walls = [r["wall_seconds"] for r in logs]
    saves = [c["save_seconds"] for c in result["checkpoints"]]
    per_epoch = max(1, -(-len(ids) // 32))  # updates per pass over the sample (effective batch 32)
    cold, warm = walls[1:per_epoch + 1], walls[per_epoch + 1:]
    return {"result": result, "total_seconds": total, "update_seconds": walls,
            "first_update_seconds": walls[0],
            "cold_epoch_update_seconds": statistics.mean(cold) if cold else None,
            "warm_update_seconds": statistics.mean(warm) if warm else None,
            # the projection's update time: the larger of the cold first epoch and the warm rest (most pilot reads
            # come from the disk, since its 126 GB of states exceed memory)
            "update_seconds_after_first": max(x for x in (statistics.mean(cold) if cold else None,
                                                          statistics.mean(warm) if warm else None,
                                                          walls[0] if len(walls) == 1 else None) if x is not None),
            "checkpoint_save_seconds": saves,
            "fixed_seconds": total - sum(walls) - sum(saves),  # model build, CUDA set-up, the log and summaries
            "prefetch_wait_mean": statistics.mean(r["prefetch_wait_seconds"] for r in logs)}


def _smoke_states(ids: list) -> dict:
    """The smoke's recorded fresh-forward states_sha256 (2026-10-29, another process), from the dev_v1 bundle."""
    out = {}
    for rec_file in sorted((C.SCALING / "evidence" / "dev_v1" / "smoke").glob("*/T4/records.jsonl")):
        for line in rec_file.read_text().splitlines():
            r = json.loads(line)
            if r["task_id"] in ids:
                out[r["task_id"]] = r.get("states_sha256")
    return out


def feasibility_cmd(args) -> None:
    """Measure on the final code, then project the complete pilot against the remaining cap. The measurements:
    - encoding, with the integrity records;
    - CPU preparation;
    - both arms' updates on representative context lengths, with file-backed, integrity-checked reads, the first
      epoch read from the disk;
    - checkpoint saves;
    - timed evaluation and reconstruction.
    The measurements are written before the projection is made."""
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
    run_dir = _new_dir(P.ROOT / "feasibility", "")
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
    known, _ident = anchors(proto)
    written = [a["seconds"] for a in known.values() if a.get("how") == "written"]
    if all(t in known for t in sample + dev) and not written:
        raise PilotStop("everything to encode is already anchored and no encoding time was recorded; the encoding "
                        "cost cannot be measured")
    out = {"run_dir": str(run_dir), "context_lengths_seconds": length_seconds, "pool": config["pool_tokens"],
           "recorded_seconds_per_new_context": statistics.mean(written) if written else None}
    try:
        with ledger.Reservation("pilot-feasibility", projected_hours=args.projected_hours,
                                extra={"run_dir": str(run_dir)}) as res:
            t_load = time.perf_counter()
            _rt, bb = _load_backbone()
            out["load_seconds"] = time.perf_counter() - t_load
            res.check()
            out["encode"] = _encode(bb, sample + dev, enc_dir, proto, log=None)
            res.check()
            prepared, ident, prep_seconds = prepare(sample, tok, proto, log=None)
            out["prepare_seconds_per_example"] = prep_seconds / len(sample)
            out["arms"] = {}
            for arm in P.ARMS:
                res.check()
                _drop_from_page_cache(e.tokens_file for e in prepared)  # the first epoch reads from the disk
                m = _fit_and_time(arm, sample, prepared, ident, proto, run_dir / arm, args.seed, args.updates,
                                  args.checkpoint_every)
                final = m["result"]["checkpoints"][-1]
                model = train.model_from_checkpoint(train.load_checkpoint(final["path"]), DEVICE)
                t1 = time.perf_counter()
                timed = infer_controller.evaluate(model, dev, bb, score=False, use_cache=False,
                                                  matmul_precision=P.MATMUL_PRECISION, log=None)
                t2 = time.perf_counter()
                again = infer_controller.evaluate(
                    train.model_from_checkpoint(train.load_checkpoint(final["path"]), DEVICE), dev, bb, score=False,
                    use_cache=True, require_hit=True, require_verified=True, cache_root=encode.PILOT_CACHE,
                    matmul_precision=P.MATMUL_PRECISION, log=None)
                t3 = time.perf_counter()
                smoke = _smoke_states(dev)
                out["arms"][arm] = {
                    "update_seconds_after_first": m["update_seconds_after_first"],
                    "cold_epoch_update_seconds": m["cold_epoch_update_seconds"],
                    "warm_update_seconds": m["warm_update_seconds"],
                    "first_update_seconds": m["first_update_seconds"], "update_seconds_max": max(m["update_seconds"]),
                    "checkpoint_save_seconds_mean": statistics.mean(m["checkpoint_save_seconds"]),
                    "fixed_seconds": m["fixed_seconds"], "fit_total_seconds": m["total_seconds"],
                    "updates": args.updates, "prefetch_wait_mean": m["prefetch_wait_mean"],
                    "eval_seconds_per_request": (t2 - t1) / len(dev),
                    "reconstruction_seconds_per_request": (t3 - t2) / len(dev),
                    "reconstruction_equal": all(a.get("actions") == b.get("actions") for a, b in zip(timed, again)),
                    "fresh_states_equal_anchor_same_process": sum(
                        a["states_sha256"] == b["states_sha256"] for a, b in zip(timed, again)),
                    "fresh_states_equal_smoke_other_process": sum(
                        smoke.get(a["task_id"]) == a["states_sha256"] for a in timed),
                    "smoke_states_available": len(smoke)}
                for ckpt in (run_dir / arm / "checkpoints").glob("*.pt"):  # measurements only
                    ckpt.unlink()
            out["reservation"] = {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}
    finally:
        cli._write(run_dir / "measurements.json", out)  # paid for: written whatever happens next
    out["projection"] = project(out, len(pool), len(P.eval_ids()), len(sample), len(dev))
    cli._write(run_dir / "summary.json", out)
    print(json.dumps({k: out.get(k) for k in ("encode", "arms", "projection")}, indent=1, default=str))
    if not out["projection"]["fits"]:
        raise PilotStop(f"the pilot does not fit: shortfall {out['projection']['shortfall_hours']:.3f} h")


def project(m: dict, n_pool: int, n_eval: int, n_sample: int, n_dev: int) -> dict:
    """The complete pilot's GPU hours from the measured unit costs, against the hours left under the 8-hour cap. The
    point projection must fit; the protocol requires no margin, and the slack is reported."""
    per_ctx = m["encode"]["seconds_per_new_context"] or m.get("recorded_seconds_per_new_context")
    if per_ctx is None:
        raise PilotStop("no newly encoded context was measured, here or in any accepted encoding run")
    used = ledger.usage()["used_hours"]
    load = m["load_seconds"]
    remaining_contexts = (n_pool - n_sample) + (n_eval - n_dev)
    encode_reservations = len(TRAIN_CHUNKS) + 1  # four training chunks and the 64
    rechecks = n_dev  # the dev run recomputes the feasibility run's development contexts
    fit = {arm: (P.UPDATES - 1) * a["update_seconds_after_first"] + a["first_update_seconds"]
           + (P.UPDATES // P.CHECKPOINT_EVERY) * a["checkpoint_save_seconds_mean"] + a["fixed_seconds"]
           for arm, a in m["arms"].items()}
    items = {
        "encode the remaining contexts, with their backbone loads and the determinism recheck":
            (remaining_contexts * per_ctx + rechecks * per_ctx + encode_reservations * load) / 3600,
        "train 2 seeds x 2 arms (each arm at its own measured update time)": 2 * sum(fit.values()) / 3600,
        "evaluate and reconstruct 4 final checkpoints, with 2 backbone loads": (2 * load + sum(
            2 * n_eval * (a["eval_seconds_per_request"] + a["reconstruction_seconds_per_request"])
            for a in m["arms"].values())) / 3600,
    }
    needed = sum(items.values())
    remaining = ledger.CAP_HOURS - used
    return {"items_hours": items, "needed_hours": needed, "used_hours": used, "cap_hours": ledger.CAP_HOURS,
            "remaining_hours": remaining, "fits": needed <= remaining, "slack_hours": remaining - needed,
            "shortfall_hours": max(0.0, needed - remaining),
            "basis": "measured on the final code; each arm at its own measured update, save and fixed costs; "
                     "preparation runs on the CPU outside reservations; no margin is required"}


def _train_log_updates(run_dir: Path) -> list:
    path = run_dir / "train_log.jsonl"
    return [json.loads(x)["update"] for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def train_cmd(args) -> None:
    cli = _cli()
    proto = protocol_identity()
    if args.seed not in P.SEEDS:
        raise PilotStop(f"seed {args.seed} is not one of the protocol's {P.SEEDS}")
    for arm in args.arms:
        run_dir = P.ROOT / "train" / f"{arm}_s{args.seed}"
        if run_dir.exists() and not args.resume:
            raise PilotStop(f"{run_dir} exists; pass --resume to continue it from its last checkpoint")
        if _train_log_updates(run_dir) == list(range(P.UPDATES)):
            raise PilotStop(f"{run_dir} is already complete (updates 0..{P.UPDATES - 1}); nothing to resume")
    tok = _tokenizer()
    ids = P.train_ids()
    prepared, ident, prep_seconds = prepare(ids, tok, proto)
    print(json.dumps({"prepared": len(prepared), "prepare_seconds": prep_seconds}), flush=True)
    for arm in args.arms:
        run_dir = P.ROOT / "train" / f"{arm}_s{args.seed}"
        recorded = [json.loads(x) for x in (run_dir / "checkpoints.jsonl").read_text().splitlines()
                    if x.strip()] if (run_dir / "checkpoints.jsonl").exists() else []
        resume = recorded[-1]["path"] if (args.resume and recorded) else None
        attempt = _new_dir(run_dir / "attempts", "")
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
    """(path, recorded entry) of a completed training run's final checkpoint (update 1,499), from the record `fit`
    appends to checkpoints.jsonl at save time. It stops on:
    - a training log that is not updates 0..1,499 exactly once;
    - no final checkpoint, or more than one;
    - a checkpoint that does not match its record.
    A final checkpoint saved but not yet recorded is accepted on its own evidence: its bytes hash to the name, its
    embedded configuration equals the run's, and its digests are recomputed."""
    from . import train
    if _train_log_updates(run_dir) != list(range(P.UPDATES)):
        raise PilotStop(f"{run_dir}: the training log does not hold updates 0..{P.UPDATES - 1} once each")
    finals = sorted((run_dir / "checkpoints").glob(f"ckpt_u{P.UPDATES - 1:06d}_*.pt"))
    if len(finals) != 1:
        raise PilotStop(f"{run_dir}: {len(finals)} final checkpoints")
    records = [json.loads(x) for x in (run_dir / "checkpoints.jsonl").read_text().splitlines() if x.strip()] \
        if (run_dir / "checkpoints.jsonl").exists() else []
    recorded = [c for c in records if Path(c["path"]).name == finals[0].name]
    ckpt = train.load_checkpoint(finals[0])  # re-hashes the file against the hash in its name
    if ckpt["update"] != P.UPDATES - 1:
        raise PilotStop(f"{run_dir}: the final checkpoint holds update {ckpt['update']}")
    digests = {"model_digest": train.tensor_digest(ckpt["model"]),
               "optimizer_digest": train.optimizer_digest(ckpt["optimizer"])}
    if recorded:
        if any(recorded[-1][k] != v for k, v in digests.items()):
            raise PilotStop(f"{run_dir}: the final checkpoint does not match its record")
        return finals[0], recorded[-1]
    # saved but not yet recorded (a stop between the write and the record): accepted on its own evidence
    config = json.loads((run_dir / "config.json").read_text())
    if train._canonical(ckpt["config"]) != train._canonical(config):
        raise PilotStop(f"{run_dir}: the unrecorded final checkpoint was made under another configuration")
    return finals[0], {"path": str(finals[0]), "sha256": ckpt["sha256"], "update": ckpt["update"], **digests,
                       "recovered": "accepted on its own evidence: hash, configuration, recomputed digests"}


def eval_passed(summary: dict) -> bool:
    """Every integrity check of one evaluation run passed."""
    arms = summary.get("arms", {})
    return set(arms) == set(P.ARMS) and not summary.get("states_differ_across_arms") and all(
        o.get("reconstruction_all_equal") and o.get("fresh_states_all_equal_anchor") and not o.get("record_errors")
        and "summary" in o and "summary_error" not in o for o in arms.values())


def eval_cmd(args) -> None:
    import torch

    from . import data, encode, infer_controller, scorer, train
    from . import records as R
    cli = _cli()
    proto = protocol_identity()
    if args.seed not in P.SEEDS:
        raise PilotStop(f"seed {args.seed} is not one of the protocol's {P.SEEDS}")
    finals = {arm: final_checkpoint(P.ROOT / "train" / f"{arm}_s{args.seed}") for arm in P.ARMS}
    known, ident = anchors(proto)
    ids = P.eval_ids()
    if any(t not in known for t in ids):
        raise PilotStop("the evaluation questions are not all encoded and anchored")
    run_dir = _new_dir(P.ROOT / "eval", "", prefix=f"s{args.seed}_")
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
        _rt, bb = _load_backbone()
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
                  "fresh_states_equal_anchor": a["states_sha256"] == known[a["task_id"]]["states_sha256"]}
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
    out["passed"] = eval_passed(out)
    cli._write(run_dir / "summary.json", out)
    print(json.dumps({a: {k: o[k] for k in ("reconstruction_all_equal", "fresh_states_all_equal_anchor")}
                      for a, o in out["arms"].items()} | {"states_differ_across_arms": states_cross,
                                                          "passed": out["passed"]}, default=str))
    if not out["passed"]:
        raise PilotStop("the evaluation's integrity checks failed; the pilot stops (see summary.json)")


def _is_ancestor(ancestor: "str | None", commit: str) -> bool:
    """Whether `commit` contains `ancestor` (the freeze commit) in its history."""
    import subprocess
    if not ancestor:
        return False
    return subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, commit], cwd=C.REPO).returncode == 0


def select_evaluations() -> tuple:
    """({seed: the one passing evaluation run}, [every evaluation run with its seed and pass/fail]). Stops unless
    each seed has exactly one passing run."""
    runs = []
    for d in sorted((P.ROOT / "eval").glob("s*_*")):
        seed = json.loads((d / "manifest.json").read_text())["config"]["seed"]
        summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        runs.append({"run": str(d), "seed": seed, "passed": eval_passed(summary)})
    chosen = {}
    for seed in P.SEEDS:
        passing = [r for r in runs if r["seed"] == seed and r["passed"]]
        if len(passing) != 1:
            raise PilotStop(f"seed {seed}: {len(passing)} passing evaluations; exactly one is required")
        chosen[seed] = Path(passing[0]["run"])
    return chosen, runs


def analyze_cmd(args) -> None:
    from collections import Counter

    from . import data
    from . import records as R
    from . import verify
    cli = _cli()
    proto = protocol_identity()
    ids = P.eval_ids()
    chosen, all_runs = select_evaluations()
    correct = {}
    for seed, d in chosen.items():
        for arm in P.ARMS:
            recs = R.load(d / arm / "records.jsonl")
            R.summarize(recs, ids)
            correct[(arm, seed)] = {r["task_id"]: r["score"]["correct"] for r in recs}
    l_runs = sorted((C.RUNS / "baselines").glob(f"*_L_nonthinking_p0-{P.EVAL_SIZE}"))
    if len(l_runs) != 1:
        raise PilotStop(f"{len(l_runs)} L runs on the {P.EVAL_SIZE}; exactly one is required")
    frozen_at = json.loads(PROTOCOL_PIN.read_text()).get("frozen_at") if PROTOCOL_PIN.exists() else None
    l_commit = json.loads((l_runs[0] / "manifest.json").read_text())["git"]["commit"]
    if not _is_ancestor(frozen_at, l_commit):
        raise PilotStop("the L run on the 64 was not made after the protocol was frozen")
    l_check = verify.verify_run(l_runs[0])
    if l_check["problems"]:
        raise PilotStop(f"the L run fails verification: {l_check['problems'][:3]}")
    l_recs = R.load(l_runs[0] / "records.jsonl")
    R.summarize(l_recs, ids)
    l_correct = {r["task_id"]: r["score"]["correct"] for r in l_recs}
    db_of = {t: data.task(t).db_id for t in ids}
    result = P.analyze(correct, l_correct, db_of, ids)
    rep = P.representability(ids, _tokenizer())
    result["gold_representability"] = {
        "representable": sum(v == "representable" for v in rep.values()), "of": len(ids),
        "reasons": dict(sorted(Counter(v.split(":")[0] for v in rep.values() if v != "representable").items())),
        "correct_on_unrepresentable": {f"{a}/{s}": sum(correct[(a, s)][t] for t in ids if rep[t] != "representable")
                                       for a in P.ARMS for s in P.SEEDS},
        "per_question": rep}
    pin = json.loads((C.PINS / "pilot_train_v1.json").read_text())
    excluded = pin["training_pool_questions"] - pin["supervisable"]
    result["training_exclusion_rate"] = {"excluded": excluded, "pool": pin["training_pool_questions"],
                                         "rate": excluded / pin["training_pool_questions"],
                                         "note": "training-pool questions without a supervisable gold program; "
                                                 "an exclusion rate for training, not an evaluation ceiling"}
    result["inputs"] = {"evaluations_counted": {str(s): str(d) for s, d in chosen.items()},
                        "evaluations_all": all_runs, "L": str(l_runs[0]), "protocol": proto}
    run_dir = _new_dir(P.ROOT / "analysis", "")
    identity.write_manifest(run_dir, config={"command": "pilot-analyze", **result["inputs"]},
                            require_clean=not args.allow_dirty)
    cli._write(run_dir / "analysis.json", result)
    print(json.dumps({k: result[k] for k in ("counts", "cleared_L_in_both_seeds", "differences", "rule",
                                              "exploratory_sign_flip")}, indent=1))
