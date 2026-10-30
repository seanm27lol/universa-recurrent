"""Command line for the TI-TNN scaling development study (not frozen).

    PY=.venv-scaling/bin/python; export PYTHONPATH=research/ti_tnn
    export TI_TNN_SCALING_RUNS=/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_scaling  # one shared, absolute
    $PY -m scaling.cli preflight                      # hardware, processes, pins, data, git, ledger (no GPU)
    $PY -m scaling.cli inventory                      # the split and the development inventory
    $PY -m scaling.cli ledger                         # GPU-reservation hours used and remaining
    $PY -m scaling.cli project baseline --model Q27   # dry-run resource projection (no GPU)
    $PY -m scaling.cli profile --model B4 --projected-hours 0.4      # load, prefill/decode throughput, memory
    $PY -m scaling.cli baseline --arm B4 --subset 16 --mode nonthinking   # L, B4 or Q27 on the development subset
    $PY -m scaling.cli smoke --projected-hours 0.4    # one paired T4 / c4_params smoke (GPU reservation)
    $PY -m scaling.cli train-timing --projected-hours 0.15   # training throughput, states in memory and from files
    $PY -m scaling.cli verify <run dir>               # the independent verifier

**GPU discipline.**
- Every GPU command runs inside one `ledger.Reservation`, opened before the model loads.
- The ledger and lock live in the one shared `$TI_TNN_SCALING_RUNS` (absolute); without it, no reservation starts.
- The aggregate cap is 8 h. A run is held to min(cap, used + projection × 1.5) and terminated past it.
- GPU commands run from a clean worktree at a committed revision. The manifest records it and refuses a dirty
  research tree.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "scaling"

from scaling import contracts as C  # noqa: E402
from scaling import identity, ledger  # noqa: E402


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")


def _append(path: Path, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


# --- preflight ---------------------------------------------------------------------------------------------------------
def preflight(args) -> dict:
    out = {"time": _stamp(), "git": identity.git_state(), "environment": identity.environment(),
           "ledger": {k: v for k, v in ledger.usage().items() if k != "reservations"},
           "foreign_gpu_processes": ledger.foreign_gpu_processes()}
    try:
        out["shared_runs"] = str(ledger.shared_runs())
    except RuntimeError as e:
        out["shared_runs"] = f"NOT SET: {e}"
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k in ("MemTotal", "MemAvailable", "MemFree", "Cached"):
            mem[k] = v.strip()
    out["memory"] = mem
    out["disk_free_gb"] = round(shutil.disk_usage(C.REPO).free / 1e9, 1)
    models = {}
    for key in C.MODELS:
        try:
            m = identity.model_identity(key, verify=args.verify_models)
            models[key] = {"repo": m["repo"], "revision": m["revision"], "files": len(m["files"]), "matches_pin": True}
        except Exception as e:
            models[key] = {"error": str(e)}
    out["models"] = models
    out["data"] = {"databases_dir": C.DATABASES.is_dir(), "pins": sorted(p.name for p in C.PINS.glob("*.json"))}
    out["load_average"] = Path("/proc/loadavg").read_text().split()[:3]
    print(json.dumps(out, indent=1, default=str))
    return out


# --- projections ---------------------------------------------------------------------------------------------------
DEFAULT_PROJECTIONS_HOURS = {("profile", "B4"): 0.4, ("profile", "Q27"): 1.0}


def measured_profile(model: str) -> dict | None:
    """The latest complete profile: no errors, and every requested prompt length measured."""
    for path in sorted((C.RUNS / "profiles").glob(f"*_{model}/profile.json"), reverse=True):
        p = json.loads(path.read_text())
        want = p.get("prompt_lengths_requested") or []
        if not p.get("errors") and p.get("synthetic") and len(p["synthetic"]) == len(want) and \
                all(s.get("decode_tokens_per_second") and s.get("time_to_first_token") for s in p["synthetic"]):
            return p
    return None


def baseline_projection_hours(model: str, n_requests: int, thinking: bool, limits: C.Limits | None = None,
                              probes: list[float] | None = None) -> dict:
    """Projected reservation hours for an LLM baseline.

    - The worst case per request is the request timeout, plus one uninterruptible prefill and one SQL timeout.
    - With measured probe requests in the same configuration, the estimate per request is min(worst case, 1.5 × the
      largest probe request).
    - Without probes, it is min(worst case, generated-token budget / measured decode rate + 10 prefills of the largest
      measured prompt + 10 SQL timeouts).
    - The total is model load + one warm-up request + n requests.

    The run is then held to its projection × 1.5 by the ledger."""
    limits = limits or C.Limits()
    prof = measured_profile(model)
    if prof is None:
        raise SystemExit(f"no complete measured profile for {model}: run `profile --model {model}` first")
    rate = min(s["decode_tokens_per_second"] for s in prof["synthetic"])
    prefill = max(s["time_to_first_token"] for s in prof["synthetic"])
    load = prof["model"]["load_seconds"]
    timeout = limits.request_timeout_seconds_27b if model == "Q27" else limits.request_timeout_seconds_4b
    worst = timeout + prefill + limits.sql_timeout_seconds
    tokens = limits.max_generated_tokens_thinking if thinking else limits.max_generated_tokens_nonthinking
    if probes:
        per_request = min(worst, 1.5 * max(probes))
        basis = f"1.5 × the largest of {len(probes)} probe requests ({max(probes):.0f} s)"
    else:
        per_request = min(worst, tokens / rate + 10 * prefill + 10 * limits.sql_timeout_seconds)
        basis = "token budget / measured decode rate + 10 prefills + 10 SQL timeouts"
    scoring = 2 * limits.sql_timeout_seconds  # the scorer runs the submitted and the gold SQL inside the reservation
    seconds = load + (n_requests + 1) * per_request + n_requests * scoring
    return {"hours": seconds / 3600, "decode_tokens_per_second": rate, "prefill_seconds_max": prefill,
            "load_seconds": load, "per_request_seconds": per_request, "worst_case_per_request_seconds": worst,
            "scoring_seconds_per_request_worst": scoring, "requests": n_requests, "basis": basis}


def _probe_seconds(model: str, mode: str, effort: str | None) -> list[float]:
    """Per-request elapsed seconds of completed probe runs (inventory positions 16-20, outside the 16) in this
    configuration."""
    out = []
    pattern = f"*_{model}_{mode}{'_' + effort if effort else ''}_p16-20/summary.json"
    for summary in (C.RUNS / "baselines").glob(pattern):
        recs = summary.parent / "records.jsonl"
        if recs.exists():
            out += [json.loads(line)["elapsed_seconds"] for line in recs.read_text().splitlines() if line.strip()]
    return out


def project(args) -> float:
    if args.what == "profile":
        hours, basis = DEFAULT_PROJECTIONS_HOURS[("profile", args.model)], "default (no measured profile yet)"
    else:
        b = baseline_projection_hours(args.model, args.requests, args.mode == "thinking",
                                      probes=_probe_seconds(args.model, args.mode, args.reasoning_effort) or None)
        hours, basis = b["hours"], b
    u = ledger.usage()
    print(json.dumps({"what": args.what, "model": args.model, "projected_hours": hours, "basis": basis,
                      "used_hours": u["used_hours"], "remaining_hours": u["remaining_hours"], "ledger": u["ledger"],
                      "fits": u["used_hours"] + hours <= ledger.CAP_HOURS}, indent=1))
    return hours


# --- profiling -------------------------------------------------------------------------------------------------------
FILLER = ("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT, "
          "total REAL); -- every order placed, with its customer, date, status and total amount\n")


def _synthetic_prompt(rt, tokens: int) -> list[dict]:
    reps = max(1, tokens // max(1, rt.count_tokens(FILLER)))
    return [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": FILLER * reps + "\nList every table above, then count from 1 to 1000."}]


def profile(args) -> None:
    import torch
    from scaling.runtime import GenerationConfig, Runtime, linear_attention_backend
    limits = C.Limits()
    if args.decode_tokens > limits.max_new_tokens_nonthinking:
        raise SystemExit(f"--decode-tokens above the per-turn limit ({limits.max_new_tokens_nonthinking}) would be capped")
    run_dir = C.RUNS / "profiles" / f"{_stamp()}_{args.model}"
    config = {"command": "profile", "model": C.MODELS[args.model], "model_key": args.model, "dtype": "bfloat16",
              "runtime": "transformers generate (batch 1)", "linear_attention": linear_attention_backend(),
              "limits": limits.as_dict(), "prompt_lengths": args.prompt_lengths, "decode_tokens": args.decode_tokens,
              "seed": args.seed, "generation": GenerationConfig.official(args.model, False, limits).as_dict()}
    manifest = identity.write_manifest(run_dir, config=config, require_clean=not args.allow_dirty,
                                       extra={"model_identity": identity.model_identity(args.model)})
    results = {"config_identity": manifest["config_identity"], "prompt_lengths_requested": args.prompt_lengths,
               "model": None, "synthetic": [], "errors": []}
    with ledger.Reservation(f"profile {args.model}", projected_hours=args.projected_hours,
                            extra={"run_dir": str(run_dir)}) as res:
        t_res = time.perf_counter()
        try:
            res.check()
            rt = Runtime(args.model)
            results["model"] = rt.describe()
            res.check()
            cfg = GenerationConfig.official(args.model, False, limits)
            rt.generate_turn(_synthetic_prompt(rt, 256), None, cfg, seed=args.seed, max_new_tokens=16)  # warm-up
            # numerical probe: the last-position logits of one fixed ~1k-token prompt, for comparing kernel paths
            enc = rt.tokenizer(rt.render(_synthetic_prompt(rt, 1024), None, cfg), return_tensors="pt",
                               add_special_tokens=False).to(rt.device)
            with torch.inference_mode():
                logits = rt.model(**enc).logits[0, -1].float().cpu()
            import numpy as np
            np.save(run_dir / "probe_logits.npy", logits.numpy())
            top = torch.topk(logits, 20)
            results["logit_probe"] = {"prompt_tokens": int(enc["input_ids"].shape[1]),
                                      "top20_ids": top.indices.tolist(), "top20_logits": top.values.tolist(),
                                      "file": "probe_logits.npy"}
            for n in args.prompt_lengths:
                res.check()
                torch.cuda.reset_peak_memory_stats()
                turn = rt.generate_turn(_synthetic_prompt(rt, n), None, cfg, seed=args.seed,
                                        max_new_tokens=args.decode_tokens)
                results["synthetic"].append({
                    "prompt_tokens": turn.prompt_tokens, "generated_tokens": turn.generated_tokens,
                    "finish_reason": turn.finish_reason, "seconds": turn.seconds,
                    "time_to_first_token": turn.time_to_first_token,
                    "prefill_tokens_per_second": (turn.prompt_tokens / turn.time_to_first_token)
                    if turn.time_to_first_token else None,
                    "decode_tokens_per_second": turn.decode_tokens_per_second,
                    "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved()})
                print(json.dumps(results["synthetic"][-1]), flush=True)
        except Exception as e:  # recorded; the reservation still ends and is charged
            results["errors"].append(f"{type(e).__name__}: {e}"[:2000])
            raise
        finally:
            results["reservation"] = {"id": res.id, "seconds": time.perf_counter() - t_res}
            _write(run_dir / "profile.json", results)
    print(json.dumps({"profile": str(run_dir / "profile.json"), "ledger": ledger.usage()["used_hours"]}, indent=1))


# --- baselines (L, B4, Q27) -------------------------------------------------------------------------------------------
def _seed(arm: str, task_id: str) -> int:
    return int(identity.sha256_bytes(f"{arm}:{task_id}".encode())[:8], 16)


WARM_UP_TOKENS = 256  # the untimed warm-up request stops after this many generated tokens (prefill and decode warmed)


def _warm_databases(db_ids) -> float:
    """The warm-cache policy, and the run-start check: re-hash every database file the run uses against its pin
    (reading it whole also warms the page cache), so no earlier in-process verification is trusted."""
    from scaling import data, tools
    t0 = time.perf_counter()
    for db in sorted(set(db_ids)):
        tools.verify_snapshot(data.snapshot(db), rehash=True)
    return time.perf_counter() - t0


def _abstain(reason: str, error: str | None = None) -> dict:
    return {"attempts": [], "final": {"kind": "abstain", "sql": None, "reason": reason}, "tokens": {},
            "tool_calls": 0, "repairs": 0, "peak_memory_bytes": None, "timed_out": False, "error": error}


def baseline(args) -> None:
    from scaling import data, retrieval, schema, scorer
    from scaling import records as R
    limits = C.Limits()
    ids = data.inventory_ids(C.INVENTORY["size"])
    lo, hi = (int(x) for x in args.positions.split(":")) if args.positions else (0, args.subset)
    requested = ids[lo:hi]
    arm, thinking = args.arm, args.mode == "thinking"
    tag = f"{arm}_{args.mode}" + (f"_{args.reasoning_effort}" if args.reasoning_effort else "") + f"_p{lo}-{hi}"
    run_dir = C.RUNS / "baselines" / f"{_stamp()}_{tag}"
    gen_cfg, backend = None, None
    if arm in ("B4", "Q27"):
        from scaling.runtime import GenerationConfig, linear_attention_backend
        gen_cfg, backend = GenerationConfig.official(arm, thinking, limits, args.reasoning_effort), linear_attention_backend()
    warm_id = json.loads(Path(data.PIN_SMOKE).read_text())["task_ids"][0]  # a training-pool question, excluded
    config = {"command": "baseline", "arm": arm, "mode": args.mode, "reasoning_effort": args.reasoning_effort,
              "positions": [lo, hi], "requested": requested, "limits": limits.as_dict(),
              "generation": gen_cfg.as_dict() if gen_cfg else None, "model": C.MODELS.get(arm),
              "dtype": "bfloat16" if gen_cfg else None,
              "runtime": "transformers generate (batch 1)" if gen_cfg else "cpu", "linear_attention": backend,
              "retrieval": retrieval.RETRIEVAL_VERSION, "schema_text": schema.SCHEMA_TEXT_VERSION,
              "schema_budget_chars": schema.PROMPT_BUDGET_CHARS, "scorer": scorer.SCORER_VERSION,
              "warm_up_task": warm_id, "warm_up_max_generated_tokens": WARM_UP_TOKENS,
              "seed_rule": "int(sha256(arm:task_id)[:8], 16)",
              "cost_unit": C.CostUnit().__dict__, "inventory_pin_sha256": identity.sha256_file(data.PIN_INVENTORY),
              "snapshots_pin_sha256": identity.sha256_file(data.PIN_SNAPSHOTS)}
    extra = {"model_identity": identity.model_identity(arm)} if arm in C.MODELS else {}
    manifest = identity.write_manifest(run_dir, config=config, require_clean=not args.allow_dirty, extra=extra)
    records_path = run_dir / "records.jsonl"
    index = retrieval.training_pool_index()  # built before timing (warm-cache policy)
    warm_seconds = _warm_databases([data.task(i).db_id for i in requested] + [data.task(warm_id).db_id])
    tasks = [data.task(i) for i in requested]

    def serve(rt, task) -> tuple[dict, float, list, str | None]:
        """One request end to end, timed from retrieval to the final answer (CPU-side work included)."""
        t0 = time.perf_counter()
        examples, text = [], None
        try:
            snap = data.snapshot(task.db_id)
            examples = retrieval.retrieve(task, limits.retrieval_k, index=index)
            if arm == "L":
                from scaling import baseline_l
                out = baseline_l.run(task, snap, examples=examples, max_tool_calls=limits.max_tool_calls)
                result = {"attempts": out["attempts"], "final": out["final"], "tokens": {},
                          "tool_calls": out["tool_call_count"], "tool_log": out["tool_calls"], "repairs": 0,
                          "peak_memory_bytes": None, "timed_out": False, "error": None}
            else:
                from scaling import llm_agent
                text = schema.serialize(schema.load_schema(task.db_id, snap), budget_chars=schema.PROMPT_BUDGET_CHARS).text
                result = llm_agent.answer_request(rt, task, snap, text, examples, gen_cfg, limits,
                                                  seed=_seed(arm, task.task_id))
        except Exception as e:  # a failed request stays in the denominator as an abstention, with its error
            if isinstance(e, ledger.BudgetExceeded):
                raise
            result = _abstain(f"error: {type(e).__name__}", f"{type(e).__name__}: {e}"[:1000])
            if arm == "L":
                result["tool_log"] = []
        return result, time.perf_counter() - t0, examples, text

    def run_all(rt, res=None):
        for task in tasks:
            if res is not None:
                res.check()
            result, elapsed, examples, text = serve(rt, task)
            snap = data.snapshot(task.db_id)
            submitted = result["final"].get("sql")
            score = scorer.score(snap, submitted, data.gold(task.task_id).sql)
            score = {**score, "scored_sql_sha256": R.sql_sha256(submitted), "snapshot_sha256": snap.sha256}
            input_identity = identity.content_identity({
                "task": [task.task_id, task.db_id, task.question, task.evidence],
                "examples": [e.task_id for e in examples],
                "schema_sha256": identity.sha256_bytes((text or "").encode()),
                "config_identity": manifest["config_identity"]})
            rec = {"task_id": task.task_id, "db_id": task.db_id, "arm": arm,
                   "config_identity": manifest["config_identity"], "input_identity": input_identity,
                   "snapshot_sha256": snap.sha256, "retrieved": [e.task_id for e in examples],
                   "attempts": result["attempts"], "final": result["final"], "score": score,
                   "tokens": result["tokens"], "controller_work": None, "elapsed_seconds": elapsed,
                   "peak_memory_bytes": result["peak_memory_bytes"], "timed_out": result["timed_out"],
                   "error": result["error"], "repairs": result["repairs"], "tool_calls": result["tool_calls"],
                   "template_options": result.get("template_options")}
            if arm == "L":
                rec["tool_log"] = result["tool_log"]
            _append(records_path, rec)
            print(json.dumps({"task": task.task_id, "final": rec["final"]["kind"], "reason": rec["final"]["reason"],
                              "correct": score.get("correct"), "elapsed": round(elapsed, 2),
                              "tokens": result["tokens"]}), flush=True)

    out = {"run_dir": str(run_dir), "warm_database_seconds": warm_seconds}
    if arm == "L":
        run_all(None)
    else:
        from scaling import llm_agent
        from scaling.runtime import Runtime
        if args.projected_hours is not None:
            projected = args.projected_hours
        else:
            projected = baseline_projection_hours(arm, len(tasks), thinking, limits,
                                                  probes=_probe_seconds(arm, args.mode, args.reasoning_effort) or None)["hours"]
        with ledger.Reservation(f"baseline {tag}", projected_hours=projected, extra={"run_dir": str(run_dir)}) as res:
            res.check()
            rt = Runtime(arm)
            out["model"] = rt.describe()
            res.check()
            t_w = time.perf_counter()
            wt = data.task(warm_id)
            wsnap = data.snapshot(wt.db_id)
            wtext = schema.serialize(schema.load_schema(wt.db_id, wsnap), budget_chars=schema.PROMPT_BUDGET_CHARS).text
            warm_limits = dataclasses.replace(limits, max_generated_tokens_thinking=WARM_UP_TOKENS,
                                              max_generated_tokens_nonthinking=WARM_UP_TOKENS)
            warm = llm_agent.answer_request(rt, wt, wsnap, wtext, retrieval.retrieve(wt, limits.retrieval_k, index=index),
                                            gen_cfg, warm_limits, seed=_seed(arm, wt.task_id))
            out["warm_up"] = {"task": wt.task_id, "seconds": time.perf_counter() - t_w, "tokens": warm["tokens"],
                              "final": warm["final"]["kind"], "excluded_from_records": True}
            run_all(rt, res)
            out["reservation"] = {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": projected}
    recs = R.load(records_path)
    try:
        out["summary"] = R.summarize(recs, requested)
    except R.RecordError as e:
        out["summary_error"] = str(e)
    _write(run_dir / "summary.json", out)
    print(json.dumps(out.get("summary", out.get("summary_error")), indent=1, default=str))


SMOKE_ARMS = {"T4": "T4", "c4_params": "C4_params"}  # model arm -> record arm (the parameter-matched shared control)


def _controller_record(ev: dict, arm_label: str, config_identity: str, snap, score: dict) -> dict:
    """A per-request record in the common format: one decode attempt, no tools, no repair."""
    from scaling import records as R
    final = {k: ev["final"].get(k) for k in ("kind", "sql", "reason")}
    cost = ev.get("encode_cost") or {}
    return {"task_id": ev["task_id"], "db_id": ev["db_id"], "arm": arm_label, "config_identity": config_identity,
            "input_identity": cost.get("context_identity") or identity.content_identity([ev["task_id"], "no context"]),
            "snapshot_sha256": snap.sha256,
            "attempts": [{"kind": "decode", "actions": ev.get("actions"), "plan": ev.get("plan"),
                          "final_sql": final["sql"], "detail": ev["final"].get("detail")}],
            "final": final, "score": {**score, "scored_sql_sha256": R.sql_sha256(final["sql"]),
                                      "snapshot_sha256": snap.sha256},
            "tokens": {"context": int(cost["tokens"])} if cost.get("tokens") is not None else {},
            "controller_work": ev.get("controller_work") or {"steps": 0, "note": "no decode (encoding failed)"},
            "encode_cost": cost or None, "states_sha256": ev.get("states_sha256"),
            "elapsed_seconds": ev["elapsed_seconds"], "peak_memory_bytes": None, "timed_out": False,
            "error": ev.get("error"), "repairs": 0, "tool_calls": 0}


def smoke(args) -> None:
    """One paired controller smoke: T4 and the parameter-matched shared controller (c4_params), same seed, same 128
    training-pool questions, at most 100 updates each; then each checkpoint is reloaded from disk and evaluated on the
    same 16 development questions at batch size one (timed, every request runs its own backbone forward), and decoded
    again from a second reload on the untimed pass's cached states (the reconstruction)."""
    import torch

    from scaling import data, encode, infer_controller, retrieval, scorer, train
    from scaling import actions as A
    from scaling import records as R
    limits = C.Limits()
    train_ids = json.loads((C.PINS / "ir_coverage_v1.json").read_text())["smoke_train"]["task_ids"]
    if len(train_ids) > C.SMOKE_TRAIN["size"] or not 1 <= args.updates <= C.SMOKE_TRAIN["max_updates"]:
        raise SystemExit(f"the smoke is at most {C.SMOKE_TRAIN['size']} questions and "
                         f"{C.SMOKE_TRAIN['max_updates']} updates per arm")
    dev_ids = data.inventory_ids(C.INVENTORY["size"])[:16]
    tag = f"T4-vs-c4_params_s{args.seed}_u{args.updates}"
    run_dir = C.RUNS / "smoke" / f"{_stamp()}_{tag}"
    tcfg = {arm: train.TrainConfig(arm=arm, seed=args.seed, updates=args.updates, micro_batch=8, device="cuda").as_dict()
            for arm in SMOKE_ARMS}
    config = {"command": "smoke", "arms": SMOKE_ARMS, "seed": args.seed, "updates": args.updates,
              "train_task_ids": train_ids, "dev_task_ids": dev_ids, "train": tcfg, "limits": limits.as_dict(),
              "grammar": A.GRAMMAR_VERSION, "encoder": encode.ENCODER_VERSION, "template": encode.TEMPLATE_VERSION,
              "infer": infer_controller.INFER_VERSION, "max_steps": infer_controller.DEFAULT_MAX_STEPS,
              "backbone": C.MODELS["B4"], "dtype": "bfloat16", "retrieval": retrieval.RETRIEVAL_VERSION,
              "scorer": scorer.SCORER_VERSION, "cost_unit": C.CostUnit().__dict__,
              "timing": "per request: task, retrieval, schema, context, tokenization, a fresh backbone forward "
                        "(no cache), decoding and compilation; scoring is outside the timing",
              "inventory_pin_sha256": identity.sha256_file(data.PIN_INVENTORY),
              "snapshots_pin_sha256": identity.sha256_file(data.PIN_SNAPSHOTS)}
    manifest = identity.write_manifest(run_dir, config=config, require_clean=not args.allow_dirty,
                                       extra={"model_identity": identity.model_identity("B4")})
    out = {"run_dir": str(run_dir), "warm_database_seconds": _warm_databases([data.task(i).db_id for i in dev_ids])}
    retrieval.training_pool_index()  # built before timing
    with ledger.Reservation(f"smoke {tag}", projected_hours=args.projected_hours,
                            extra={"run_dir": str(run_dir)}) as res:
        from scaling.runtime import Runtime
        res.check()
        rt = Runtime("B4")
        bb = encode.Backbone.from_pinned(device="cuda", model=rt.model, tokenizer=rt.tokenizer)
        out["backbone"] = bb.identity()
        res.check()
        t0 = time.perf_counter()
        prepared, skipped, costs = train.prepare_examples(train_ids, bb, log=print)
        out["prepare"] = {"seconds": time.perf_counter() - t0, "examples": len(prepared), "skipped": skipped,
                          "cache_hits": sum(c.get("cache_hit", False) for c in costs.values()),
                          "forward_seconds": sum(c.get("forward_seconds", 0.0) for c in costs.values())}
        res.check()
        t0 = time.perf_counter()  # untimed: the dev states the reconstruction decodes from (and a warm-up)
        encode.encode_tasks(bb, [data.task(i) for i in dev_ids])
        out["dev_cache_pass_seconds"] = time.perf_counter() - t0
        out["arms"] = {}
        for arm, label in SMOKE_ARMS.items():
            res.check()
            t0 = time.perf_counter()
            result = train.train_arm(arm, train_ids, bb, run_dir / arm, updates=args.updates, seed=args.seed,
                                     device="cuda", prepared=prepared, log=print)
            train_seconds = time.perf_counter() - t0
            final = result["checkpoints"][-1]
            res.check()
            model = train.model_from_checkpoint(train.load_checkpoint(final["path"]), "cuda")
            if train.tensor_digest(model.state_dict()) != final["model_digest"]:
                raise RuntimeError(f"{arm}: the reloaded weights differ from the saved ones")
            torch.cuda.reset_peak_memory_stats()
            evals = infer_controller.evaluate(model, dev_ids, bb, score=False, use_cache=False, log=print)
            peak = torch.cuda.max_memory_allocated()
            res.check()
            again_model = train.model_from_checkpoint(train.load_checkpoint(final["path"]), "cuda")
            again = infer_controller.evaluate(again_model, dev_ids, bb, score=False, use_cache=True, log=None)
            recon = [{"task_id": a["task_id"], "same_states": a["states_sha256"] == b["states_sha256"],
                      "same_actions": a.get("actions") == b.get("actions"),
                      "same_final": a["final"] == b["final"]} for a, b in zip(evals, again)]
            arm_config = identity.content_identity({"run": manifest["config_identity"], "arm": arm,
                                                    "model_digest": final["model_digest"]})
            records_path = run_dir / arm / "records.jsonl"
            record_errors = []
            for ev in evals:  # every record is kept; a malformed one is reported, and the summary then fails
                snap = data.snapshot(ev["db_id"])
                score = scorer.score(snap, ev["final"]["sql"], data.gold(ev["task_id"]).sql)
                rec = _controller_record(ev, label, arm_config, snap, score)
                try:
                    R.validate(rec)
                except R.RecordError as e:
                    record_errors.append(str(e))
                _append(records_path, rec)
            recs = R.load(records_path)
            arm_out = {"label": label, "config_identity": arm_config, "train_seconds": train_seconds,
                       "checkpoint": final, "checkpoints": result["checkpoints"], "train_log": result["log"],
                       "examples": result["examples"], "eval_peak_memory_bytes": peak, "reconstruction": recon,
                       "record_errors": record_errors,
                       "reconstruction_all_equal": all(r["same_actions"] and r["same_final"] for r in recon),
                       "log_tail": [json.loads(x) for x in Path(result["log"]).read_text().splitlines()[-1:]]}
            try:
                arm_out["summary"] = R.summarize(recs, dev_ids)
            except R.RecordError as e:
                arm_out["summary_error"] = str(e)
            out["arms"][arm] = arm_out
            print(json.dumps({arm: arm_out.get("summary", arm_out.get("summary_error"))}, default=str), flush=True)
        out["reservation"] = {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}
    _write(run_dir / "summary.json", out)
    print(json.dumps({a: {k: v for k, v in o.items() if k in ("summary", "reconstruction_all_equal", "train_seconds")}
                      for a, o in out["arms"].items()}, indent=1, default=str))


def train_timing(args) -> None:
    """Training throughput on the GPU, after the repair: the smoke's 128 questions and recipe (seed 100), `--updates`
    updates from update 0, twice: states held in memory, then read from the encoding cache files by the prefetcher.
    Each run's losses are compared with the smoke's training log for the same updates. The models are not kept."""
    import statistics

    import torch

    from scaling import encode, train
    train_ids = json.loads((C.PINS / "ir_coverage_v1.json").read_text())["smoke_train"]["task_ids"]
    smoke_runs = sorted(p for p in (C.RUNS / "smoke").iterdir() if (p / "summary.json").exists())
    reference = smoke_runs[-1] / args.arm / "train_log.jsonl"
    ref = [json.loads(x) for x in reference.read_text().splitlines()][:args.updates]
    run_dir = C.RUNS / "timing" / f"{_stamp()}_{args.arm}_u{args.updates}_mb{args.micro_batch}_{args.matmul_precision}"
    config = {"command": "train-timing", "arm": args.arm, "updates": args.updates, "seed": C.SMOKE_TRAIN["seed"],
              "modes": args.modes, "micro_batch": args.micro_batch, "matmul_precision": args.matmul_precision,
              "train_task_ids": train_ids, "reference_log": str(reference),
              "reference_log_sha256": identity.sha256_file(reference), "backbone": C.MODELS["B4"]}
    manifest = identity.write_manifest(run_dir, config=config, require_clean=not args.allow_dirty,
                                       extra={"model_identity": identity.model_identity("B4")})
    out = {"run_dir": str(run_dir), "config_identity": manifest["config_identity"], "modes": {}}
    with ledger.Reservation(f"train-timing {args.arm} u{args.updates}", projected_hours=args.projected_hours,
                            extra={"run_dir": str(run_dir)}) as res:
        from scaling.runtime import Runtime
        rt = Runtime("B4")
        bb = encode.Backbone.from_pinned(device="cuda", model=rt.model, tokenizer=rt.tokenizer)
        for mode, lazy in [m for m in (("memory", False), ("file", True)) if m[0] in args.modes]:
            res.check()
            t0 = time.perf_counter()
            prepared, skipped, costs = train.prepare_examples(train_ids, bb, lazy=lazy, log=None)
            prep = time.perf_counter() - t0
            torch.cuda.reset_peak_memory_stats()
            profiler = None
            if args.profile:
                from torch.profiler import ProfilerActivity, profile, schedule
                profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                   schedule=schedule(wait=2, warmup=1, active=3, repeat=1))
                profiler.start()
                train.PROFILER_STEP = profiler.step
            result = train.train_arm(args.arm, train_ids, bb, run_dir / mode, updates=args.updates,
                                     seed=C.SMOKE_TRAIN["seed"], device="cuda", prepared=prepared,
                                     micro_batch=args.micro_batch, matmul_precision=args.matmul_precision, log=None)
            if profiler is not None:
                profiler.stop()
                train.PROFILER_STEP = None
                for key in ("cpu_time_total", "cuda_time_total"):
                    (run_dir / f"profile_{mode}_{key}.txt").write_text(
                        profiler.key_averages().table(sort_by=key, row_limit=30))
            log = [json.loads(x) for x in Path(result["log"]).read_text().splitlines()]
            walls = [r["wall_seconds"] for r in log]
            rel = [abs(a["loss"] - b["loss"]) / max(abs(b["loss"]), 1e-12) for a, b in zip(log, ref)]
            out["modes"][mode] = {
                "prepare_seconds": prep, "cache_hits": sum(c.get("cache_hit", False) for c in costs.values()),
                "examples": len(prepared), "update_seconds_mean": statistics.mean(walls),
                "update_seconds_median": statistics.median(walls), "update_seconds_after_first": statistics.mean(walls[1:]),
                "prefetch_wait_seconds_mean": statistics.mean(r["prefetch_wait_seconds"] for r in log),
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                "same_examples_as_reference": [r["examples"] for r in log] == [r["examples"] for r in ref],
                "loss_max_relative_difference_from_reference": max(rel), "losses": [r["loss"] for r in log]}
            print(json.dumps({mode: {k: v for k, v in out["modes"][mode].items() if k != "losses"}}, default=str),
                  flush=True)
            del prepared
        out["reference"] = {"log": str(reference), "update_seconds_mean": statistics.mean(r["wall_seconds"] for r in ref)}
        out["reservation"] = {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}
    for mode in args.modes:  # the models are measurements only; their checkpoints are not kept
        for ckpt in (run_dir / mode / "checkpoints").glob("*.pt"):
            ckpt.unlink()
    _write(run_dir / "summary.json", out)
    print(json.dumps({"reference_update_seconds": out["reference"]["update_seconds_mean"],
                      **{m: round(o["update_seconds_after_first"], 3) for m, o in out["modes"].items()}}, indent=1))


def verify_cmd(args) -> None:
    from scaling import verify
    results = [verify.verify_run(d) for d in args.run_dirs]
    print(json.dumps([{k: r[k] for k in ("run_dir", "kind", "verdict", "problems")} for r in results], indent=1))
    if any(r["problems"] for r in results):
        raise SystemExit(1)


def inventory_cmd(args) -> None:
    from collections import Counter
    from scaling import data
    ids = data.inventory_ids(C.INVENTORY["size"])
    out = {"inventory": len(ids)}
    for n in (16, 64, len(ids)):
        out[f"first_{n}_by_database"] = dict(Counter(data.task(i).db_id for i in ids[:n]))
    out["split"] = {k: v for k, v in json.loads(Path(data.PIN_SPLIT).read_text()).items()
                    if k in ("dev", "train_pool", "dev_questions", "train_pool_questions")}
    print(json.dumps(out, indent=1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("preflight")
    s.add_argument("--verify-models", action="store_true", help="re-hash every weight file (slow for the 27B)")
    s.set_defaults(fn=preflight)
    s = sub.add_parser("ledger")
    s.set_defaults(fn=lambda a: print(json.dumps(ledger.usage(), indent=1, default=str)))
    s = sub.add_parser("project")
    s.add_argument("what", choices=["profile", "baseline"])
    s.add_argument("--model", choices=sorted(C.MODELS), default="B4")
    s.add_argument("--requests", type=int, default=16)
    s.add_argument("--mode", choices=["thinking", "nonthinking"], default="nonthinking")
    s.add_argument("--reasoning-effort", choices=["xhigh", "medium", "low"], default=None)
    s.set_defaults(fn=project)
    s = sub.add_parser("profile")
    s.add_argument("--model", choices=sorted(C.MODELS), required=True)
    s.add_argument("--prompt-lengths", type=int, nargs="+", default=[1024, 4096, 12288, 28672])
    s.add_argument("--decode-tokens", type=int, default=256)
    s.add_argument("--seed", type=int, default=100)
    s.add_argument("--projected-hours", type=float, required=True)
    s.add_argument("--allow-dirty", action="store_true", help="development only: skip the clean-tree requirement")
    s.set_defaults(fn=profile)
    s = sub.add_parser("inventory")
    s.set_defaults(fn=inventory_cmd)
    s = sub.add_parser("baseline")
    s.add_argument("--arm", choices=["L", "B4", "Q27"], required=True)
    s.add_argument("--subset", type=int, default=16, choices=[16, 64, 512])
    s.add_argument("--positions", help="inventory positions lo:hi instead of the subset (e.g. 16:20 for the probes)")
    s.add_argument("--mode", choices=["thinking", "nonthinking"], default="nonthinking")
    s.add_argument("--reasoning-effort", choices=["xhigh", "medium", "low"], default=None)
    s.add_argument("--projected-hours", type=float, default=None)
    s.add_argument("--allow-dirty", action="store_true", help="development only: skip the clean-tree requirement")
    s.set_defaults(fn=baseline)
    s = sub.add_parser("train-timing")
    s.add_argument("--arm", choices=["T4", "c4_params", "c4_active"], default="T4")
    s.add_argument("--updates", type=int, default=20)
    s.add_argument("--profile", action="store_true", help="torch.profiler tables for updates 3-5 of each mode")
    s.add_argument("--modes", nargs="+", choices=["memory", "file"], default=["memory", "file"])
    s.add_argument("--micro-batch", type=int, default=8)
    s.add_argument("--matmul-precision", choices=["highest", "high"], default="highest")
    s.add_argument("--projected-hours", type=float, required=True)
    s.add_argument("--allow-dirty", action="store_true", help="development only: skip the clean-tree requirement")
    s.set_defaults(fn=train_timing)
    s = sub.add_parser("verify")
    s.add_argument("run_dirs", nargs="+")
    s.set_defaults(fn=verify_cmd)
    s = sub.add_parser("smoke")
    s.add_argument("--seed", type=int, default=C.SMOKE_TRAIN["seed"])
    s.add_argument("--updates", type=int, default=C.SMOKE_TRAIN["max_updates"])
    s.add_argument("--projected-hours", type=float, required=True)
    s.add_argument("--allow-dirty", action="store_true", help="development only: skip the clean-tree requirement")
    s.set_defaults(fn=smoke)
    args = p.parse_args(argv)
    if getattr(args, "projected_hours", None) is not None and not (math.isfinite(args.projected_hours)
                                                                   and args.projected_hours > 0):
        raise SystemExit("--projected-hours must be finite and positive")
    args.fn(args)


if __name__ == "__main__":
    main()
