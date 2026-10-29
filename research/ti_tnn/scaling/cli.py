"""Command line for the TI-TNN scaling development study (not frozen).

    PY=.venv-scaling/bin/python; export PYTHONPATH=research/ti_tnn
    export TI_TNN_SCALING_RUNS=/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_scaling  # one shared, absolute
    $PY -m scaling.cli preflight                      # hardware, processes, pins, data, git, ledger (no GPU)
    $PY -m scaling.cli inventory                      # the split and the development inventory
    $PY -m scaling.cli ledger                         # GPU-reservation hours used and remaining
    $PY -m scaling.cli project baseline --model Q27   # dry-run resource projection (no GPU)
    $PY -m scaling.cli profile --model B4 --projected-hours 0.4      # load, prefill/decode throughput, memory
    $PY -m scaling.cli baseline --arm B4 --subset 16 --mode nonthinking   # L, B4 or Q27 on the development subset
    $PY -m scaling.cli smoke ...                      # one paired C4/T4 smoke (GPU reservation)
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
              "warm_up_task": warm_id, "seed_rule": "int(sha256(arm:task_id)[:8], 16)",
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
            warm = llm_agent.answer_request(rt, wt, wsnap, wtext, retrieval.retrieve(wt, limits.retrieval_k, index=index),
                                            gen_cfg, limits, seed=_seed(arm, wt.task_id))
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
    args = p.parse_args(argv)
    if getattr(args, "projected_hours", None) is not None and not (math.isfinite(args.projected_hours)
                                                                   and args.projected_hours > 0):
        raise SystemExit("--projected-hours must be finite and positive")
    args.fn(args)


if __name__ == "__main__":
    main()
