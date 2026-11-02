"""The performance study v2 (GPU authorization v2): relational grounding against the pilot's controllers, matched over
three seeds, with the Qwen baselines completed (commands registered in `cli`).

**Example of the order** (each GPU step is one ledger reservation under the 32-hour cumulative cap):
1. `perf-pin` (CPU) writes `pins/perf_v2_train.json`, every pilot training question supervisable under both
   groundings (4,740 of the pilot's 4,741), and `pins/perf_v2_eval.json`, the two evaluation populations: the
   pilot's 64 (historical, inventory positions 0-63) and the next 64 (new, positions 64-127).
2. `perf-encode` encodes the new 64 into the pilot cache, anchored, after recomputing 8 historical contexts in this
   new process and comparing them with their pilot anchors.
3. `perf-feasibility` measures the unit costs on the frozen code (each of the four arms' updates, checkpoint saves,
   timed evaluation and reconstruction) and runs the paired smoke: gradients, checkpoint reload, decoding and EX on
   training-pool requests. It projects the remaining plan against the cap.
4. `perf-train --seed S` trains T4-v1, C4-v1, T4-v2 and C4-v2 (C4 is the parameter-matched shared controller,
   `c4_params`) for the protocol's updates, one reservation each, from one prepared list per grounding.
5. `perf-eval --seed S` evaluates the four final checkpoints on the 128 (timed fresh forward per request, then a
   reconstruction on the anchored cache).
6. `baseline --arm L --positions 64:128` (CPU) and, if the projection permits, B4 and Q27 on the same positions.
7. `perf-analyze` (CPU) reads exactly one passing evaluation per seed.

Every command refuses to start unless `docs/PROTOCOL_perf_v2.md` matches its frozen hash in
`pins/perf_v2_protocol.json`, except `perf-pin`, which writes the pins the protocol names.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from . import contracts as C
from . import identity, ledger
from . import pilot as P
from . import pilot_run as PR

STUDY = "ti_tnn_scaling_performance_v2"
ARMS = (("T4", "v1"), ("c4_params", "v1"), ("T4", "v2"), ("c4_params", "v2"))
LABELS = {("T4", "v1"): "T4-v1", ("c4_params", "v1"): "C4-v1", ("T4", "v2"): "T4-v2", ("c4_params", "v2"): "C4-v2"}
RECORD_ARM = {"T4": "T4", "c4_params": "C4_params"}
SEEDS = (100, 101, 102)
UPDATES = 3000
CHECKPOINT_EVERY = 500
MATMUL_PRECISION = "high"
POPULATIONS = {"historical": (0, 64), "new": (64, 128)}
PROTOCOL = C.SCALING / "docs" / "PROTOCOL_perf_v2.md"
PROTOCOL_PIN = C.PINS / "perf_v2_protocol.json"
TRAIN_PIN = C.PINS / "perf_v2_train.json"
EVAL_PIN = C.PINS / "perf_v2_eval.json"
ROOT = C.RUNS / "perf_v2"
DEVICE = "cuda"
RECHECK = 8               # historical contexts recomputed by perf-encode (cross-process determinism)


class PerfStop(RuntimeError):
    """A condition of the frozen protocol failed; the study stops and reports."""


def label(arm: str, grounding: str) -> str:
    return LABELS[(arm, grounding)]


# --- pins --------------------------------------------------------------------------------------------------------------
def train_ids() -> list:
    return list(json.loads(TRAIN_PIN.read_text())["task_ids"])


def eval_ids(population: "str | None" = None) -> list:
    pin = json.loads(EVAL_PIN.read_text())
    if population is None:
        return pin["populations"]["historical"]["task_ids"] + pin["populations"]["new"]["task_ids"]
    return list(pin["populations"][population]["task_ids"])


def pin_cmd(args) -> None:
    """The training and evaluation pins (CPU). The training set is every pilot training question whose gold program
    traces under both groundings; the evaluation populations are fixed inventory positions."""
    from . import data, encode, retrieval, train
    from . import actions as A
    from . import schema as schema_mod
    tok = PR._tokenizer()
    pilot_ids = P.train_ids()
    kept, dropped = [], {}
    for tid in pilot_ids:
        task = data.task(tid)
        full = schema_mod.load_schema(task.db_id)
        ctx, serialized, _ids, offsets = encode.fit_context(task, full, retrieval.retrieve(task), tok,
                                                            C.Limits().max_context_tokens)

        class _Enc:  # the parts of an Encoding that gold_steps reads
            schema = A.SchemaItems(serialized.items, full)
            spans = encode.span_index(ctx, offsets)

        reasons = {g: train.gold_steps(data.gold(tid).sql, _Enc, g)[1] for g in A.GROUNDINGS}
        if any(reasons.values()):
            dropped[tid] = {g: r for g, r in reasons.items() if r}
        else:
            kept.append(tid)
    pilot_pin = C.PINS / "pilot_train_v1.json"
    train_payload = {
        "version": "perf_v2_train", "study": STUDY, "groundings": {g: A.GROUNDING_GRAMMAR[g] for g in A.GROUNDINGS},
        "rule": "every question of pins/pilot_train_v1.json, in its order, whose gold program traces under both "
                "groundings (v1, the pilot's grammar; v2, relational grounding), so that all four arms train on "
                "identical supervision",
        "pilot_train_pin_sha256": identity.sha256_file(pilot_pin), "pilot_questions": len(pilot_ids),
        "kept": len(kept), "task_ids": kept, "dropped": dropped}
    inv = data.inventory_ids(C.INVENTORY["size"])
    pops = {name: {"positions": [lo, hi], "task_ids": inv[lo:hi],
                   "db_ids": [data.task(t).db_id for t in inv[lo:hi]]} for name, (lo, hi) in POPULATIONS.items()}
    eval_payload = {
        "version": "perf_v2_eval", "study": STUDY,
        "rule": "the development inventory (pins/inventory_v1.json) at positions 0-63 (historical: the pilot's 64, "
                "which the diagnosis read) and 64-127 (new: never read by any arm, fixed before revised training)",
        "inventory_pin_sha256": identity.sha256_file(data.PIN_INVENTORY), "populations": pops}
    data.write_pin(TRAIN_PIN, train_payload, rewrite=args.rewrite)
    data.write_pin(EVAL_PIN, eval_payload, rewrite=args.rewrite)
    print(json.dumps({"train": {"kept": len(kept), "dropped": dropped},
                      "eval": {k: len(v["task_ids"]) for k, v in pops.items()}}, indent=1))


# --- protocol and anchors ---------------------------------------------------------------------------------------------
def protocol_identity() -> dict:
    if not PROTOCOL_PIN.exists():
        raise PerfStop("the performance protocol is not frozen (no pins/perf_v2_protocol.json)")
    pin = json.loads(PROTOCOL_PIN.read_text())
    digest = identity.sha256_file(PROTOCOL)
    if digest != pin["sha256"]:
        raise PerfStop(f"docs/PROTOCOL_perf_v2.md is {digest[:12]}, the frozen hash is {pin['sha256'][:12]}")
    for name, path in (("train_pin_sha256", TRAIN_PIN), ("eval_pin_sha256", EVAL_PIN)):
        if pin.get(name) != identity.sha256_file(path):
            raise PerfStop(f"{path.name} differs from the pin the protocol froze")
    return {"file": str(PROTOCOL.relative_to(C.REPO)), "sha256": digest, "frozen_at": pin.get("frozen_at"),
            "updates": pin.get("updates", UPDATES)}


def updates() -> int:
    return int(json.loads(PROTOCOL_PIN.read_text()).get("updates", UPDATES))


def anchors() -> tuple:
    """({task_id: anchor}, backbone identity): the pilot's accepted anchors and this study's, which must agree."""
    pilot_proto = PR.protocol_identity()
    found, ident, ignored = P.load_anchors(protocol_sha256=pilot_proto["sha256"])
    proto = protocol_identity()
    mine, ident2, ignored2 = P.load_anchors(root=ROOT / "encode", protocol_sha256=proto["sha256"])
    if ident is not None and ident2 is not None and ident != ident2:
        raise PerfStop("this study's anchors come from another backbone identity than the pilot's")
    for tid, a in mine.items():
        old = found.get(tid)
        if old is not None and (old["cache_key"], old["states_sha256"]) != (a["cache_key"], a["states_sha256"]):
            raise PerfStop(f"{tid}: this study anchored another entry than the pilot")
        found.setdefault(tid, a)
    for note in ignored + ignored2:
        print(json.dumps({"anchors_ignored": note}), flush=True)
    return found, ident or ident2


# --- encoding the new population --------------------------------------------------------------------------------------
def encode_cmd(args) -> None:
    from . import data, encode, retrieval
    from . import schema as schema_mod
    cli = PR._cli()
    proto = protocol_identity()
    new = eval_ids("new")
    recheck = eval_ids("historical")[:RECHECK]
    run_dir = PR._new_dir(ROOT / "encode", "new")
    identity.write_manifest(run_dir, config={"command": "perf-encode", "ids": new, "recheck": recheck,
                                             "cache_root": "PILOT_CACHE", "protocol": proto},
                            require_clean=not args.allow_dirty,
                            extra={"model_identity": identity.model_identity("B4")})
    known, ident = anchors()
    out = {"requested": len(new), "rechecked": [], "encoded": 0, "already_present": 0}
    seconds = []
    with ledger.Reservation("perf-encode new", projected_hours=args.projected_hours,
                            extra={"run_dir": str(run_dir)}) as res:
        _rt, bb = PR._load_backbone()
        if ident is not None and bb.identity() != ident:
            raise PerfStop("this backbone's identity differs from the anchors'")
        PR._write_backbone(run_dir, bb)
        for tid in recheck:  # cross-process determinism: fresh states equal to the pilot's anchors
            res.check()
            task = data.task(tid)
            fresh = encode.encode_request(bb, task, schema_mod.load_schema(task.db_id), retrieval.retrieve(task),
                                          use_cache=False)
            equal = encode.states_sha256(fresh.states) == known[tid]["states_sha256"]
            out["rechecked"].append({"task_id": tid, "equal": equal})
            if not equal:
                cli._write(run_dir / "summary.json", out)
                raise PerfStop(f"{tid}: a fresh forward differs from its pilot anchor; the study stops")
        for tid in new:
            res.check()
            task = data.task(tid)
            t0 = time.perf_counter()
            enc = encode.encode_request(bb, task, schema_mod.load_schema(task.db_id), retrieval.retrieve(task),
                                        cache_root=encode.PILOT_CACHE, require_verified=True)
            digest, how = enc.cost["states_sha256"], "written"
            if enc.cost["cache_hit"]:  # an entry without an anchor of this study: recomputed, anchored only if equal
                fresh = encode.encode_request(bb, task, schema_mod.load_schema(task.db_id), retrieval.retrieve(task),
                                              use_cache=False)
                if encode.states_sha256(fresh.states) != digest:
                    raise PerfStop(f"{tid}: a fresh forward differs from the cached entry")
                how = "recomputed and equal"
                out["already_present"] += 1
            else:
                out["encoded"] += 1
                seconds.append(time.perf_counter() - t0)
            if tid in known and known[tid]["states_sha256"] != digest:
                raise PerfStop(f"{tid}: the entry differs from its existing anchor")
            cli._append(run_dir / "entries.jsonl", {"task_id": tid, "cache_key": enc.cost["cache_key"],
                                                    "states_sha256": digest, "tokens": enc.cost["tokens"],
                                                    "cache_hit": enc.cost["cache_hit"], "how": how,
                                                    "forward_seconds": enc.cost["forward_seconds"],
                                                    "seconds": time.perf_counter() - t0})
        out["seconds_per_new_context"] = statistics.mean(seconds) if seconds else None
        out["reservation"] = {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}
    cli._write(run_dir / "summary.json", out)
    print(json.dumps(out, indent=1, default=str))


# --- preparation ------------------------------------------------------------------------------------------------------
def prepare(ids: list, tokenizer, log=print) -> tuple:
    """({grounding: [Prepared]}, backbone identity, seconds): every example from its anchored, verified cache entry,
    read once on the CPU and prepared for both groundings (sharing the state tensors)."""
    from . import controller as K
    from . import data, encode, retrieval, train
    from . import schema as schema_mod
    known, ident = anchors()
    missing = [t for t in ids if t not in known]
    if missing:
        raise PerfStop(f"{len(missing)} ids have no encoding anchor (first: {missing[:3]})")
    reader = encode.Backbone.cache_reader(ident, tokenizer)
    t0 = time.perf_counter()
    out = {"v1": [], "v2": []}
    for n, tid in enumerate(ids):
        task = data.task(tid)
        enc = encode.encode_request(reader, task, schema_mod.load_schema(task.db_id), retrieval.retrieve(task),
                                    cache_root=encode.PILOT_CACHE, require_hit=True, require_verified=True)
        a = known[tid]
        if a["cache_key"] != enc.cost["cache_key"] or a["states_sha256"] != enc.cost.get("states_sha256"):
            raise encode.CacheCorrupt(f"{tid}: the cache entry does not match its anchor")
        steps = {}
        for g in ("v1", "v2"):
            steps[g], reason = train.gold_steps(data.gold(tid).sql, enc, g) if task.split == "train" else (None, None)
            if task.split == "train" and steps[g] is None:
                raise PerfStop(f"{tid}: not supervisable under grounding {g} ({reason})")
        tokens_file = str(encode.cache_path(enc.cost["cache_key"], encode.PILOT_CACHE))
        first = K.prepare(enc, steps["v1"], tokens_file=tokens_file, grounding="v1")
        out["v1"].append(first)
        out["v2"].append(K.as_grounding(first, enc, steps["v2"], "v2"))
        if log and (n + 1) % 500 == 0:
            log(f"prepared {n + 1}/{len(ids)}")
    return out, ident, time.perf_counter() - t0


# --- training ---------------------------------------------------------------------------------------------------------
def run_dir_of(arm: str, grounding: str, seed: int) -> Path:
    return ROOT / "train" / f"{label(arm, grounding)}_s{seed}"


def _log_updates(run_dir: Path) -> list:
    path = run_dir / "train_log.jsonl"
    return [json.loads(x)["update"] for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def final_checkpoint(run_dir: Path, n_updates: "int | None" = None) -> tuple:
    """(path, record) of a completed run's final checkpoint (update U - 1), as in the pilot: the log must hold
    0..U-1 once each, there must be exactly one final checkpoint, and it must match its record (or, unrecorded, its own
    configuration and recomputed digests)."""
    from . import train
    U = n_updates or updates()
    if _log_updates(run_dir) != list(range(U)):
        raise PerfStop(f"{run_dir}: the training log does not hold updates 0..{U - 1} once each")
    finals = sorted((run_dir / "checkpoints").glob(f"ckpt_u{U - 1:06d}_*.pt"))
    if len(finals) != 1:
        raise PerfStop(f"{run_dir}: {len(finals)} final checkpoints")
    records = [json.loads(x) for x in (run_dir / "checkpoints.jsonl").read_text().splitlines() if x.strip()] \
        if (run_dir / "checkpoints.jsonl").exists() else []
    recorded = [c for c in records if Path(c["path"]).name == finals[0].name]
    ckpt = train.load_checkpoint(finals[0])
    if ckpt["update"] != U - 1:
        raise PerfStop(f"{run_dir}: the final checkpoint holds update {ckpt['update']}")
    digests = {"model_digest": train.tensor_digest(ckpt["model"]),
               "optimizer_digest": train.optimizer_digest(ckpt["optimizer"])}
    if recorded:
        if any(recorded[-1][k] != v for k, v in digests.items()):
            raise PerfStop(f"{run_dir}: the final checkpoint does not match its record")
        return finals[0], recorded[-1]
    config = json.loads((run_dir / "config.json").read_text())
    if train._canonical(ckpt["config"]) != train._canonical(config):
        raise PerfStop(f"{run_dir}: the unrecorded final checkpoint was made under another configuration")
    return finals[0], {"path": str(finals[0]), "sha256": ckpt["sha256"], "update": ckpt["update"], **digests,
                       "recovered": "accepted on its own evidence: hash, configuration, recomputed digests"}


def completed(run_dir: Path) -> "str | None":
    if _log_updates(run_dir) != list(range(updates())):
        return "the training log is not complete"
    try:
        final_checkpoint(run_dir)
    except Exception as e:
        return f"the log is complete but the final checkpoint is not usable: {type(e).__name__}: {e}"[:300]
    return None


def _resume_point(run_dir: Path) -> tuple:
    """The newest recorded checkpoint that verifies (its bytes hash to its name), and the ones set aside."""
    from . import train
    recorded = [json.loads(x) for x in (run_dir / "checkpoints.jsonl").read_text().splitlines() if x.strip()] \
        if (run_dir / "checkpoints.jsonl").exists() else []
    unusable = []
    for entry in reversed(recorded):
        path = run_dir / "checkpoints" / Path(entry["path"]).name
        try:
            train.load_checkpoint(path)
        except Exception as e:
            aside = None
            if path.exists():
                aside = f"{path}.damaged-{time.time_ns()}"
                path.rename(aside)
            unusable.append({"path": str(path), "error": f"{type(e).__name__}: {e}"[:200], "set_aside_as": aside})
            continue
        return str(path), unusable
    return None, unusable


def train_cmd(args) -> None:
    cli = PR._cli()
    proto = protocol_identity()
    U = updates()
    if args.seed not in SEEDS:
        raise PerfStop(f"seed {args.seed} is not one of the protocol's {SEEDS}")
    arms = [a for a in ARMS if label(*a) in args.arms] if args.arms else list(ARMS)
    for arm, g in arms:
        rd = run_dir_of(arm, g, args.seed)
        if rd.exists() and not args.resume:
            raise PerfStop(f"{rd} exists; pass --resume to continue it from its last checkpoint")
        if rd.exists() and completed(rd) is None:
            raise PerfStop(f"{rd} is already complete (updates 0..{U - 1} and a sound final checkpoint)")
    tok = PR._tokenizer()
    ids = train_ids()
    prepared, ident, prep_seconds = prepare(ids, tok)
    print(json.dumps({"prepared": len(ids), "prepare_seconds": prep_seconds}), flush=True)
    for arm, g in arms:
        rd = run_dir_of(arm, g, args.seed)
        resume, unusable = _resume_point(rd) if args.resume else (None, [])
        attempt = PR._new_dir(rd / "attempts", "")
        identity.write_manifest(attempt, config={
            "command": "perf-train", "arm": arm, "grounding": g, "label": label(arm, g), "seed": args.seed,
            "updates": U, "checkpoint_every": CHECKPOINT_EVERY, "matmul_precision": MATMUL_PRECISION,
            "train_pin_sha256": identity.sha256_file(TRAIN_PIN), "ids_sha256": identity.content_identity(ids),
            "resume_from": resume, "resume_skipped": unusable, "protocol": proto, "backbone": ident},
            require_clean=not args.allow_dirty)
        with ledger.Reservation(f"perf-train {label(arm, g)} s{args.seed}", projected_hours=args.projected_hours,
                                extra={"run_dir": str(rd), "attempt": str(attempt)}) as res:
            res.check()
            from . import train
            t0 = time.perf_counter()
            result = train.train_arm(arm, ids, None, rd, updates=U, seed=args.seed, device=DEVICE,
                                     prepared=prepared[g], resume_from=resume, matmul_precision=MATMUL_PRECISION,
                                     checkpoint_every=CHECKPOINT_EVERY, grounding=g,
                                     extra_config={"backbone": ident, "perf_protocol": proto,
                                                   "label": label(arm, g)}, log=print)
            out = {"arm": arm, "grounding": g, "label": label(arm, g), "seed": args.seed,
                   "seconds": time.perf_counter() - t0, "resumed_from": resume,
                   "checkpoints": result["checkpoints"], "log": result["log"],
                   "config_identity": result["config_identity"],
                   "reservation": {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}}
        if DEVICE == "cuda":
            import torch
            out["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        cli._write(attempt / "summary.json", out)
        print(json.dumps({k: out[k] for k in ("label", "seed", "seconds", "reservation")}, default=str), flush=True)


# --- evaluation -------------------------------------------------------------------------------------------------------
RECONSTRUCTION_KEYS = ("same_actions", "same_final", "fresh_states_equal_anchor")


def eval_failures(summary: dict, ids: list) -> list:
    """Why one evaluation run fails its integrity checks (empty: it passed), from the complete reconstruction
    inventories, as in the pilot."""
    arms = summary.get("arms", {})
    want = {label(*a) for a in ARMS}
    out = [] if set(arms) == want else [f"arms {sorted(arms)}, expected {sorted(want)}"]
    if summary.get("states_differ_across_arms"):
        out.append(f"states differ across the arms: {summary['states_differ_across_arms'][:3]}")
    for name, o in sorted(arms.items()):
        inventory = o.get("reconstruction") or []
        if [r.get("task_id") for r in inventory] != list(ids):
            out.append(f"{name}: the reconstruction inventory does not list the {len(ids)} questions once each")
        false = [(r.get("task_id"), k) for r in inventory for k in RECONSTRUCTION_KEYS if r.get(k) is not True]
        if false:
            out.append(f"{name}: {len(false)} reconstruction entries are not true, first {false[:3]}")
        if o.get("record_errors"):
            out.append(f"{name}: record errors {o['record_errors'][:2]}")
        if "summary" not in o or "summary_error" in o:
            out.append(f"{name}: no summary ({o.get('summary_error')})")
    return out


def eval_cmd(args) -> None:
    import torch

    from . import data, encode, infer_controller, scorer, train
    from . import records as R
    cli = PR._cli()
    proto = protocol_identity()
    if args.seed not in SEEDS:
        raise PerfStop(f"seed {args.seed} is not one of the protocol's {SEEDS}")
    finals = {a: final_checkpoint(run_dir_of(*a, args.seed)) for a in ARMS}
    known, ident = anchors()
    ids = eval_ids()
    if any(t not in known for t in ids):
        raise PerfStop("the evaluation questions are not all encoded and anchored")
    run_dir = PR._new_dir(ROOT / "eval", "", prefix=f"s{args.seed}_")
    manifest = identity.write_manifest(run_dir, config={
        "command": "perf-eval", "seed": args.seed, "ids": ids, "protocol": proto, "backbone": ident,
        "checkpoints": {label(*a): {"path": identity.run_ref(p), **{k: c[k] for k in ("sha256", "model_digest",
                                                                                     "update")}}
                        for a, (p, c) in finals.items()},
        "matmul_precision": MATMUL_PRECISION}, require_clean=not args.allow_dirty,
        extra={"model_identity": identity.model_identity("B4")})
    out = {"run_dir": identity.run_ref(run_dir), "arms": {}}
    evals = {}
    cli._warm_databases([data.task(t).db_id for t in ids])
    with ledger.Reservation(f"perf-eval s{args.seed}", projected_hours=args.projected_hours,
                            extra={"run_dir": str(run_dir)}) as res:
        _rt, bb = PR._load_backbone()
        if bb.identity() != ident:
            raise PerfStop("the loaded backbone's identity differs from the anchors'; reconstruction would miss")
        for a, (path, _c) in finals.items():
            res.check()
            model = train.model_from_checkpoint(train.load_checkpoint(path), DEVICE)
            if DEVICE == "cuda":
                torch.cuda.reset_peak_memory_stats()
            timed = infer_controller.evaluate(model, ids, bb, score=False, use_cache=False,
                                              matmul_precision=MATMUL_PRECISION, log=None)
            peak = torch.cuda.max_memory_allocated() if DEVICE == "cuda" else None
            again = infer_controller.evaluate(train.model_from_checkpoint(train.load_checkpoint(path), DEVICE), ids,
                                              bb, score=False, use_cache=True, require_hit=True,
                                              require_verified=True, cache_root=encode.PILOT_CACHE,
                                              matmul_precision=MATMUL_PRECISION, log=None)
            evals[a] = (timed, again, peak)
        out["reservation"] = {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}
    first = evals[ARMS[0]][0]
    states_cross = sorted({t for a in ARMS[1:] for t, x, y in zip(ids, first, evals[a][0])
                           if x["states_sha256"] != y["states_sha256"]})
    for a, (timed, again, peak) in evals.items():
        arm, g = a
        name = label(arm, g)
        recon = [{"task_id": x["task_id"], "same_actions": x.get("actions") == y.get("actions"),
                  "same_final": x["final"] == y["final"],
                  "fresh_states_equal_anchor": x["states_sha256"] == known[x["task_id"]]["states_sha256"]}
                 for x, y in zip(timed, again)]
        arm_config = identity.content_identity({"run": manifest["config_identity"], "arm": name,
                                                "model_digest": finals[a][1]["model_digest"]})
        path = run_dir / name / "records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        errors = []
        for ev in timed:
            snap = data.snapshot(ev["db_id"])
            score = scorer.score(snap, ev["final"]["sql"], data.gold(ev["task_id"]).sql)
            rec = {**cli._controller_record(ev, RECORD_ARM[arm], arm_config, snap, score), "grounding": g,
                   "label": name}
            try:
                R.validate(rec)
            except R.RecordError as e:
                errors.append(str(e))
            cli._append(path, rec)
        arm_out = {"label": name, "arm": arm, "grounding": g, "config_identity": arm_config,
                   "eval_peak_memory_bytes": peak, "reconstruction": recon, "record_errors": errors,
                   "reconstruction_all_equal": all(r["same_actions"] and r["same_final"] for r in recon),
                   "fresh_states_all_equal_anchor": all(r["fresh_states_equal_anchor"] for r in recon)}
        try:
            arm_out["summary"] = R.summarize(R.load(path), ids)
        except R.RecordError as e:
            arm_out["summary_error"] = str(e)
        out["arms"][name] = arm_out
    out["states_differ_across_arms"] = states_cross
    out["failures"] = eval_failures(out, ids)
    out["passed"] = not out["failures"]
    cli._write(run_dir / "summary.json", out)
    print(json.dumps({n: {"correct": o.get("summary", {}).get("correct"),
                          "reconstruction_all_equal": o["reconstruction_all_equal"]}
                      for n, o in out["arms"].items()} | {"passed": out["passed"]}, default=str))
    if not out["passed"]:
        raise PerfStop(f"the evaluation's integrity checks failed: {out['failures'][:3]}")


def select_evaluations() -> tuple:
    """({seed: the one passing evaluation run}, [every evaluation run]); stops unless each evaluated seed has exactly
    one passing run."""
    runs, dirs = [], {}
    for d in sorted((ROOT / "eval").glob("s*_*")):
        config = json.loads((d / "manifest.json").read_text())["config"]
        summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        ref = identity.run_ref(d)
        dirs[ref] = d
        runs.append({"run": ref, "seed": config["seed"],
                     "passed": summary.get("passed") is True and not eval_failures(summary, config["ids"])})
    chosen = {}
    for seed in SEEDS:
        passing = [r for r in runs if r["seed"] == seed and r["passed"]]
        if len(passing) > 1:
            raise PerfStop(f"seed {seed}: {len(passing)} passing evaluations; at most one is allowed")
        if passing:
            chosen[seed] = dirs[passing[0]["run"]]
    return chosen, runs


# --- feasibility and the paired smoke ---------------------------------------------------------------------------------
def feasibility_cmd(args) -> None:
    """Measure the unit costs on the frozen code and run the paired smoke, then project the remaining plan. The
    smoke's models train on a training-pool sample under measurement seed 999 and decode training-pool requests
    only; they are discarded, and no development question is scored."""
    import torch

    from . import data, encode, infer_controller, scorer, train
    cli = PR._cli()
    proto = protocol_identity()
    tok = PR._tokenizer()
    pool = train_ids()
    lengths = PR._context_lengths(pool, tok)
    sample = P.feasibility_sample(pool, lengths, args.sample)
    decode_ids = sample[:: max(1, len(sample) // args.decode)][:args.decode]
    run_dir = PR._new_dir(ROOT / "feasibility", "")
    identity.write_manifest(run_dir, config={"command": "perf-feasibility", "sample": sample, "decode": decode_ids,
                                             "updates": args.updates, "seed": args.seed, "protocol": proto},
                            require_clean=not args.allow_dirty,
                            extra={"model_identity": identity.model_identity("B4")})
    t0 = time.perf_counter()
    prepared, ident, prep_seconds = prepare(sample, tok, log=None)
    out = {"run_dir": identity.run_ref(run_dir), "prepare_seconds_per_example": prep_seconds / len(sample),
           "sample_tokens_mean": statistics.mean(lengths[t] for t in sample),
           "pool_tokens_mean": statistics.mean(lengths.values()), "arms": {}}
    try:
        with ledger.Reservation("perf-feasibility", projected_hours=args.projected_hours,
                                extra={"run_dir": str(run_dir)}) as res:
            t_load = time.perf_counter()
            _rt, bb = PR._load_backbone()
            out["load_seconds"] = time.perf_counter() - t_load
            for arm, g in ARMS:
                res.check()
                name = label(arm, g)
                PR._drop_from_page_cache(e.tokens_file for e in prepared[g])
                if DEVICE == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                t1 = time.perf_counter()
                result = train.train_arm(arm, sample, None, run_dir / name, updates=args.updates, seed=args.seed,
                                         device=DEVICE, prepared=prepared[g], matmul_precision=MATMUL_PRECISION,
                                         checkpoint_every=args.updates // 2, grounding=g,
                                         extra_config={"backbone": ident, "perf_protocol": proto}, log=None)
                fit_seconds = time.perf_counter() - t1
                peak_train = torch.cuda.max_memory_allocated() if DEVICE == "cuda" else None
                logs = [json.loads(x) for x in Path(result["log"]).read_text().splitlines()]
                walls = [r["wall_seconds"] for r in logs]
                per_epoch = max(1, -(-len(sample) // 32))
                cold, warm = walls[1:per_epoch + 1], walls[per_epoch + 1:]
                final = result["checkpoints"][-1]
                ckpt = train.load_checkpoint(final["path"])
                model = train.model_from_checkpoint(ckpt, DEVICE)
                reload_equal = train.tensor_digest(ckpt["model"]) == final["model_digest"]
                t2 = time.perf_counter()
                timed = infer_controller.evaluate(model, decode_ids, bb, score=False, use_cache=False,
                                                  matmul_precision=MATMUL_PRECISION, log=None)
                t3 = time.perf_counter()
                again = infer_controller.evaluate(
                    train.model_from_checkpoint(train.load_checkpoint(final["path"]), DEVICE), decode_ids, bb,
                    score=False, use_cache=True, require_hit=True, require_verified=True,
                    cache_root=encode.PILOT_CACHE, matmul_precision=MATMUL_PRECISION, log=None)
                t4 = time.perf_counter()
                ex = [scorer.score(data.snapshot(e["db_id"]), e["final"]["sql"], data.gold(e["task_id"]).sql)
                      for e in timed]
                out["arms"][name] = {
                    "first_update_seconds": walls[0], "update_seconds_mean": statistics.mean(walls[1:]),
                    "cold_epoch_update_seconds": statistics.mean(cold) if cold else None,
                    "warm_update_seconds": statistics.mean(warm) if warm else None,
                    "update_seconds_after_first": max(x for x in (statistics.mean(cold) if cold else None,
                                                                  statistics.mean(warm) if warm else None)
                                                      if x is not None),
                    "checkpoint_save_seconds_mean": statistics.mean(c["save_seconds"] for c in result["checkpoints"]),
                    "fixed_seconds": fit_seconds - sum(walls) - sum(c["save_seconds"] for c in result["checkpoints"]),
                    "loss_first_last": [logs[0]["loss"], logs[-1]["loss"]],
                    "grad_norm_finite": all(isinstance(r["grad_norm"], float) and r["grad_norm"] == r["grad_norm"]
                                            and abs(r["grad_norm"]) != float("inf") for r in logs),
                    "train_peak_memory_bytes": peak_train,
                    "checkpoint_reload_digest_equal": reload_equal,
                    "eval_seconds_per_request": (t3 - t2) / len(decode_ids),
                    "reconstruction_seconds_per_request": (t4 - t3) / len(decode_ids),
                    "reconstruction_equal": all(x.get("actions") == y.get("actions") and x["final"] == y["final"]
                                                for x, y in zip(timed, again)),
                    "fresh_states_equal_anchor": sum(
                        x["states_sha256"] == y["states_sha256"] for x, y in zip(timed, again)),
                    "decoded_sql": sum(e["final"]["kind"] == "sql" for e in timed),
                    "training_requests_ex_correct": sum(s["correct"] for s in ex), "decoded": len(timed),
                    "parameters": json.loads((run_dir / name / "config.json").read_text())["parameters"]}
                for path in (run_dir / name / "checkpoints").glob("*.pt"):  # measurements only
                    path.unlink()
            out["reservation"] = {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}
    finally:
        out["wall_seconds"] = time.perf_counter() - t0
        cli._write(run_dir / "measurements.json", out)
    out["projection"] = project(out, len(pool), len(eval_ids()))
    cli._write(run_dir / "summary.json", out)
    print(json.dumps({k: out.get(k) for k in ("arms", "projection")}, indent=1, default=str))
    bad = [n for n, a in out["arms"].items() if not (a["grad_norm_finite"] and a["checkpoint_reload_digest_equal"]
                                                      and a["reconstruction_equal"]
                                                      and a["loss_first_last"][1] < a["loss_first_last"][0])]
    if bad:
        raise PerfStop(f"the paired smoke failed for {bad}")


def project(m: dict, n_pool: int, n_eval: int) -> dict:
    """The remaining controller plan's GPU hours from the measured unit costs (each arm at its own measured costs),
    and the baselines' projections, against the hours left under the cap."""
    U = updates()
    load = m["load_seconds"]
    fit = {n: (U - 1) * a["update_seconds_after_first"] + a["first_update_seconds"]
           + (U // CHECKPOINT_EVERY) * a["checkpoint_save_seconds_mean"] + a["fixed_seconds"]
           for n, a in m["arms"].items()}
    items = {
        f"train {len(SEEDS)} seeds x 4 arms x {U} updates": len(SEEDS) * sum(fit.values()) / 3600,
        f"evaluate and reconstruct 12 final checkpoints on the {n_eval}, with {len(SEEDS)} loads": (
            len(SEEDS) * load + len(SEEDS) * sum(n_eval * (a["eval_seconds_per_request"]
                                                           + a["reconstruction_seconds_per_request"])
                                                 for a in m["arms"].values())) / 3600,
    }
    needed = sum(items.values())
    remaining = ledger.CAP_HOURS - ledger.usage()["used_hours"]
    return {"per_fit_hours": {n: f / 3600 for n, f in fit.items()}, "items_hours": items, "needed_hours": needed,
            "remaining_hours": remaining, "fits": needed <= remaining, "slack_hours": remaining - needed}


# --- the CLI ------------------------------------------------------------------------------------------------------------
def register(sub) -> None:
    s = sub.add_parser("perf-pin")
    s.add_argument("--rewrite", action="store_true")
    s.set_defaults(fn=pin_cmd)
    s = sub.add_parser("perf-encode")
    s.add_argument("--projected-hours", type=float, required=True)
    s.add_argument("--allow-dirty", action="store_true")
    s.set_defaults(fn=encode_cmd)
    s = sub.add_parser("perf-feasibility")
    s.add_argument("--projected-hours", type=float, required=True)
    s.add_argument("--sample", type=int, default=160)
    s.add_argument("--decode", type=int, default=8)
    s.add_argument("--updates", type=int, default=40)
    s.add_argument("--seed", type=int, default=999)
    s.add_argument("--allow-dirty", action="store_true")
    s.set_defaults(fn=feasibility_cmd)
    s = sub.add_parser("perf-train")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--arms", nargs="*", choices=sorted(LABELS.values()))
    s.add_argument("--resume", action="store_true")
    s.add_argument("--projected-hours", type=float, required=True)
    s.add_argument("--allow-dirty", action="store_true")
    s.set_defaults(fn=train_cmd)
    s = sub.add_parser("perf-eval")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--projected-hours", type=float, required=True)
    s.add_argument("--allow-dirty", action="store_true")
    s.set_defaults(fn=eval_cmd)

