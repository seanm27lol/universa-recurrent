"""Command line for the TI-TNN scaling development study (not frozen).

    PY=.venv-scaling/bin/python; export PYTHONPATH=research/ti_tnn
    $PY -m scaling.cli preflight                      # hardware, processes, pins, data, git, ledger (no GPU)
    $PY -m scaling.cli ledger                         # GPU-reservation hours used and remaining
    $PY -m scaling.cli project profile --model Q27    # dry-run resource projection (no GPU)
    $PY -m scaling.cli profile --model B4             # load, prefill/decode throughput, memory (GPU reservation)
    $PY -m scaling.cli baseline --arm B4 --subset 16 --mode nonthinking   # L, B4 or Q27 on the development subset
    $PY -m scaling.cli smoke                          # one paired C4/T4 smoke (GPU reservation)
    $PY -m scaling.cli verify <run dir>               # the independent verifier

**GPU discipline.**
- Every GPU command runs inside one `ledger.Reservation`, opened before the model loads.
- The aggregate cap is 8 h; `--projected-hours` must fit what remains, or the command refuses to start.
- GPU commands run from a clean worktree at a committed revision. The manifest records it and refuses a dirty
  research tree.
- Outputs go to `runs/ti_tnn_scaling/` (or `$TI_TNN_SCALING_RUNS`).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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


# --- preflight ---------------------------------------------------------------------------------------------------------
def preflight(args) -> dict:
    out = {"time": _stamp(), "git": identity.git_state(), "environment": identity.environment(),
           "ledger": {k: v for k, v in ledger.usage().items() if k != "reservations"},
           "foreign_gpu_processes": ledger.foreign_gpu_processes()}
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k in ("MemTotal", "MemAvailable", "MemFree", "Cached"):
            mem[k] = v.strip()
    out["memory"] = mem
    du = shutil.disk_usage(C.REPO)
    out["disk_free_gb"] = round(du.free / 1e9, 1)
    models = {}
    for key in C.MODELS:
        try:
            models[key] = {k: v for k, v in identity.model_identity(key, verify=args.verify_models).items()}
        except FileNotFoundError as e:
            models[key] = {"missing": str(e)}
    out["models"] = models
    out["data"] = {"databases_dir": C.DATABASES.is_dir(), "pins": sorted(p.name for p in C.PINS.glob("*.json"))}
    try:
        out["load_average"] = Path("/proc/loadavg").read_text().split()[:3]
    except OSError:
        pass
    print(json.dumps(out, indent=1, default=str))
    return out


# --- projections ---------------------------------------------------------------------------------------------------
DEFAULT_PROJECTIONS_HOURS = {  # used only until a measured profile exists; deliberately generous
    ("profile", "B4"): 0.4, ("profile", "Q27"): 1.0,
}


def measured_profile(model: str) -> dict | None:
    runs = sorted((C.RUNS / "profiles").glob(f"*_{model}/profile.json"))
    return json.loads(runs[-1].read_text()) if runs else None


def project(args) -> float:
    if args.what == "profile":
        hours = DEFAULT_PROJECTIONS_HOURS[("profile", args.model)]
        basis = "default (no measured profile yet)"
    else:
        raise SystemExit(f"no projection rule for {args.what!r} yet")
    u = ledger.usage()
    print(json.dumps({"what": args.what, "model": getattr(args, "model", None), "projected_hours": hours, "basis": basis,
                      "used_hours": u["used_hours"], "remaining_hours": u["remaining_hours"],
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
    from scaling.runtime import GenerationConfig, Runtime
    limits = C.Limits()
    run_dir = C.RUNS / "profiles" / f"{_stamp()}_{args.model}"
    config = {"command": "profile", "model": C.MODELS[args.model], "model_key": args.model, "dtype": "bfloat16",
              "runtime": "transformers generate (batch 1)", "limits": limits.as_dict(),
              "prompt_lengths": args.prompt_lengths, "decode_tokens": args.decode_tokens, "seed": args.seed,
              "generation": GenerationConfig.official(args.model, False, limits).as_dict()}
    manifest = identity.write_manifest(run_dir, config=config, require_clean=not args.allow_dirty,
                                       extra={"model_identity": identity.model_identity(args.model)})
    results = {"config_identity": manifest["config_identity"], "model": None, "synthetic": [], "errors": []}
    with ledger.Reservation(f"profile {args.model}", projected_hours=args.projected_hours,
                            extra={"run_dir": str(run_dir)}) as res:
        t_res = time.time()
        try:
            rt = Runtime(args.model)
            results["model"] = rt.describe()
            res.check()
            cfg = GenerationConfig.official(args.model, False, limits)
            rt.generate_turn(_synthetic_prompt(rt, 256), None, cfg, seed=args.seed, max_new_tokens=16)  # warm-up
            for n in args.prompt_lengths:
                res.check()
                torch.cuda.reset_peak_memory_stats()
                msgs = _synthetic_prompt(rt, n)
                turn = rt.generate_turn(msgs, None, cfg, seed=args.seed, max_new_tokens=args.decode_tokens)
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
            results["reservation"] = {"id": res.id, "seconds": time.time() - t_res}
            _write(run_dir / "profile.json", results)
    print(json.dumps({"profile": str(run_dir / "profile.json"), "ledger": ledger.usage()["used_hours"]}, indent=1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("preflight")
    s.add_argument("--verify-models", action="store_true", help="re-hash every weight file (slow for the 27B)")
    s.set_defaults(fn=preflight)
    s = sub.add_parser("ledger")
    s.set_defaults(fn=lambda a: print(json.dumps(ledger.usage(), indent=1, default=str)))
    s = sub.add_parser("project")
    s.add_argument("what", choices=["profile"])
    s.add_argument("--model", choices=sorted(C.MODELS), default="B4")
    s.set_defaults(fn=project)
    s = sub.add_parser("profile")
    s.add_argument("--model", choices=sorted(C.MODELS), required=True)
    s.add_argument("--prompt-lengths", type=int, nargs="+", default=[1024, 4096, 12288])
    s.add_argument("--decode-tokens", type=int, default=256)
    s.add_argument("--seed", type=int, default=100)
    s.add_argument("--projected-hours", type=float, required=True)
    s.add_argument("--allow-dirty", action="store_true", help="development only: skip the clean-tree requirement")
    s.set_defaults(fn=profile)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
